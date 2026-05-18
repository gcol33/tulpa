// nested_laplace_multi.cpp
// cpp_nested_laplace_multi: Rcpp entry for multi-latent-block nested Laplace.
//
// Builds a std::vector<tulpa::LatentBlock> from an R-side list of block
// descriptors (one per logical prior block — icar / bym2 / car_proper /
// rw1 / rw2 / ar1 / iid), then dispatches to run_multi_block_nested_laplace
// (declared in nested_laplace.cpp).
//
// Wire-up:
//   blocks_spec    : Rcpp::List of length B. Each element is an Rcpp::List
//                    with fields: type (string) + type-specific params
//                    (spatial_idx / temporal_idx / obs_idx, sizes, adjacency
//                    CSR, scale_factor, cyclic, ...).
//   theta_grid     : NumericMatrix n_cells x sum_axes (per-block axes
//                    concatenated in block-list order).
//   axis_offsets   : IntegerVector length B+1, cumulative axis counts.
//                    Block b's hyperparameters at grid point k are
//                    theta_grid(k, axis_offsets[b] .. axis_offsets[b+1]-1).
//
// Axis layouts per type (must match the R-side prior_from_spec / .NL_REGISTRY):
//   icar       : 1 axis  = (tau,)
//   bym2       : 2 axes  = (sigma, rho)     -> produces 2 LatentBlocks (phi, theta)
//   car_proper : 2 axes  = (tau, rho)
//   rw1        : 1 axis  = (tau,)
//   rw2        : 1 axis  = (tau,)
//   ar1        : 2 axes  = (tau, rho)
//   iid        : 1 axis  = (sigma,)
//
// One block-spec maps to ONE input-list element but may push 1 or 2
// LatentBlocks into the vector (BYM2 is the only 2-block expansion).

#include "laplace_re_priors.h"
#include "laplace_spatial_priors.h"
#include "laplace_temporal_priors.h"
#include "latent_block.h"
#include "nested_laplace_multi.h"
#include "hmc_car_proper.h"
#include <Rcpp.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// Push the LatentBlock(s) for one block-spec entry. Returns the new
// latent_offset after appending this block's sub-vector(s) to the joint
// latent vector layout.
int build_blocks_from_spec(
    const Rcpp::List& bs,
    const Rcpp::NumericMatrix& theta_grid,
    int axis0,                // starting column in theta_grid for this block's axes
    int axis_count,           // number of axes used by this block
    int latent_offset,
    std::vector<tulpa::LatentBlock>& blocks
) {
    std::string type = Rcpp::as<std::string>(bs["type"]);

    auto require_axes = [&](int needed) {
        if (axis_count != needed) {
            Rcpp::stop("Block type '%s' expects %d axes, got %d",
                       type.c_str(), needed, axis_count);
        }
    };

    if (type == "icar") {
        require_axes(1);
        int size = Rcpp::as<int>(bs["n_spatial_units"]);
        Rcpp::IntegerVector spatial_idx = bs["spatial_idx"];
        Rcpp::IntegerVector adj_rp      = bs["adj_row_ptr"];
        Rcpp::IntegerVector adj_ci      = bs["adj_col_idx"];
        Rcpp::IntegerVector n_nbr       = bs["n_neighbors"];
        int start = latent_offset;

        tulpa::LatentBlock block;
        block.start = start;
        block.size  = size;
        block.idx   = [spatial_idx](int i, int /*k_arm*/) { return spatial_idx[i]; };
        block.d_fac = [](int) { return 1.0; };
        block.add_prior = [start, size, axis0, theta_grid, adj_rp, adj_ci, n_nbr](
            tulpa::DenseVec& grad, tulpa::DenseMat& H,
            const Rcpp::NumericVector& x, int k) {
            double tau = theta_grid(k, axis0);
            tulpa::add_icar_prior(grad, H, x, start, size, tau,
                                   adj_rp, adj_ci, n_nbr);
        };
        block.log_prior = [start, size, axis0, theta_grid, adj_rp, adj_ci, n_nbr](
            const Rcpp::NumericVector& x, int k) {
            double tau = theta_grid(k, axis0);
            return tulpa::log_prior_icar(x, start, size, tau,
                                          adj_rp, adj_ci, n_nbr);
        };
        block.center = [start, size](Rcpp::NumericVector& x) -> double {
            return tulpa::center_effects(x, start, size);
        };
        blocks.push_back(block);
        return start + size;
    }

    if (type == "bym2") {
        require_axes(2);
        int size = Rcpp::as<int>(bs["n_spatial_units"]);
        Rcpp::IntegerVector spatial_idx = bs["spatial_idx"];
        Rcpp::IntegerVector adj_rp      = bs["adj_row_ptr"];
        Rcpp::IntegerVector adj_ci      = bs["adj_col_idx"];
        Rcpp::IntegerVector n_nbr       = bs["n_neighbors"];
        double scale_factor = bs.containsElementNamed("scale_factor") ?
            Rcpp::as<double>(bs["scale_factor"]) : 1.0;
        int phi_start   = latent_offset;
        int theta_start = phi_start + size;

        tulpa::LatentBlock phi_block;
        phi_block.start = phi_start;
        phi_block.size  = size;
        phi_block.idx   = [spatial_idx](int i, int /*k_arm*/) { return spatial_idx[i]; };
        phi_block.d_fac = [axis0, theta_grid, scale_factor](int k) {
            double sigma_k = theta_grid(k, axis0);
            double rho_k   = theta_grid(k, axis0 + 1);
            return sigma_k * std::sqrt(rho_k + 1e-10) * scale_factor;
        };
        phi_block.add_prior = [phi_start, size, adj_rp, adj_ci, n_nbr](
            tulpa::DenseVec& grad, tulpa::DenseMat& H,
            const Rcpp::NumericVector& x, int) {
            tulpa::add_icar_prior(grad, H, x, phi_start, size, 1.0,
                                   adj_rp, adj_ci, n_nbr);
        };
        phi_block.log_prior = [phi_start, size, adj_rp, adj_ci, n_nbr](
            const Rcpp::NumericVector& x, int) {
            double quad_form = 0.0;
            for (int s = 0; s < size; s++) {
                double phi_s = x[phi_start + s];
                quad_form += n_nbr[s] * phi_s * phi_s;
                for (int kk = adj_rp[s]; kk < adj_rp[s + 1]; kk++) {
                    int neighbor = adj_ci[kk];
                    if (neighbor > s) {
                        quad_form -= 2.0 * phi_s * x[phi_start + neighbor];
                    }
                }
            }
            return -0.5 * quad_form;
        };
        phi_block.center = [phi_start, size](Rcpp::NumericVector& x) -> double {
            return tulpa::center_effects(x, phi_start, size);
        };
        blocks.push_back(phi_block);

        tulpa::LatentBlock theta_block;
        theta_block.start = theta_start;
        theta_block.size  = size;
        theta_block.idx   = [spatial_idx](int i, int /*k_arm*/) { return spatial_idx[i]; };
        theta_block.d_fac = [axis0, theta_grid](int k) {
            double sigma_k = theta_grid(k, axis0);
            double rho_k   = theta_grid(k, axis0 + 1);
            return sigma_k * std::sqrt(1.0 - rho_k + 1e-10);
        };
        theta_block.add_prior = [theta_start, size](
            tulpa::DenseVec& grad, tulpa::DenseMat& H,
            const Rcpp::NumericVector& x, int) {
            for (int s = 0; s < size; s++) {
                int idx = theta_start + s;
                grad[idx] -= x[idx];
                H[idx][idx] += 1.0;
            }
        };
        theta_block.log_prior = [theta_start, size](
            const Rcpp::NumericVector& x, int) {
            double lp = 0.0;
            for (int s = 0; s < size; s++) {
                lp -= 0.5 * x[theta_start + s] * x[theta_start + s];
            }
            lp -= 0.5 * size * std::log(2.0 * M_PI);
            return lp;
        };
        blocks.push_back(theta_block);
        return theta_start + size;
    }

    if (type == "car_proper") {
        require_axes(2);
        int size = Rcpp::as<int>(bs["n_spatial_units"]);
        Rcpp::IntegerVector spatial_idx = bs["spatial_idx"];
        Rcpp::IntegerVector adj_rp      = bs["adj_row_ptr"];
        Rcpp::IntegerVector adj_ci      = bs["adj_col_idx"];
        Rcpp::IntegerVector n_nbr       = bs["n_neighbors"];
        int start = latent_offset;

        // Cache CSR for the dense log-det helper.
        auto adj_rp_v = std::make_shared<std::vector<int>>(adj_rp.begin(), adj_rp.end());
        auto adj_ci_v = std::make_shared<std::vector<int>>(adj_ci.begin(), adj_ci.end());
        auto n_nbr_v  = std::make_shared<std::vector<int>>(n_nbr.begin(),  n_nbr.end());
        auto log_det_Q_rho = std::make_shared<double>(0.0);

        tulpa::LatentBlock block;
        block.start = start;
        block.size  = size;
        block.idx   = [spatial_idx](int i, int /*k_arm*/) { return spatial_idx[i]; };
        block.d_fac = [](int) { return 1.0; };
        block.prep  = [size, axis0, theta_grid, adj_rp_v, adj_ci_v, n_nbr_v,
                       log_det_Q_rho](int k) -> bool {
            double rho_k = theta_grid(k, axis0 + 1);
            std::vector<double> Qmat = tulpa_car_proper::compute_car_precision(
                size, *adj_rp_v, *adj_ci_v, *n_nbr_v, rho_k);
            *log_det_Q_rho = tulpa_car_proper::car_log_det(size, Qmat);
            return std::isfinite(*log_det_Q_rho);
        };
        block.add_prior = [start, size, axis0, theta_grid, adj_rp, adj_ci, n_nbr](
            tulpa::DenseVec& grad, tulpa::DenseMat& H,
            const Rcpp::NumericVector& x, int k) {
            double tau = theta_grid(k, axis0);
            double rho = theta_grid(k, axis0 + 1);
            tulpa::add_car_proper_prior(grad, H, x, start, size, tau, rho,
                                         adj_rp, adj_ci, n_nbr);
        };
        block.log_prior = [start, size, axis0, theta_grid, adj_rp, adj_ci, n_nbr,
                           log_det_Q_rho](const Rcpp::NumericVector& x, int k) {
            double tau = theta_grid(k, axis0);
            double rho = theta_grid(k, axis0 + 1);
            return tulpa::log_prior_car_proper(x, start, size, tau, rho,
                                                 *log_det_Q_rho,
                                                 adj_rp, adj_ci, n_nbr);
        };
        block.center = [start, size](Rcpp::NumericVector& x) -> double {
            return tulpa::center_effects(x, start, size);
        };
        blocks.push_back(block);
        return start + size;
    }

    if (type == "rw1" || type == "rw2") {
        require_axes(1);
        int size = Rcpp::as<int>(bs["n_times"]);
        Rcpp::IntegerVector temporal_idx = bs["temporal_idx"];
        bool cyclic = (type == "rw1") &&
                      bs.containsElementNamed("cyclic") &&
                      Rcpp::as<bool>(bs["cyclic"]);
        int start = latent_offset;

        tulpa::LatentBlock block;
        block.start = start;
        block.size  = size;
        block.idx   = [temporal_idx](int i, int /*k_arm*/) { return temporal_idx[i]; };
        block.d_fac = [](int) { return 1.0; };
        if (type == "rw1") {
            block.add_prior = [start, size, axis0, theta_grid, cyclic](
                tulpa::DenseVec& grad, tulpa::DenseMat& H,
                const Rcpp::NumericVector& x, int k) {
                double tau = theta_grid(k, axis0);
                tulpa::add_rw1_precision(grad, H, x, start, size, tau, cyclic);
            };
            block.log_prior = [start, size, axis0, theta_grid, cyclic](
                const Rcpp::NumericVector& x, int k) {
                double tau = theta_grid(k, axis0);
                return tulpa::log_prior_rw1(x, start, size, tau, cyclic);
            };
        } else {
            block.add_prior = [start, size, axis0, theta_grid](
                tulpa::DenseVec& grad, tulpa::DenseMat& H,
                const Rcpp::NumericVector& x, int k) {
                double tau = theta_grid(k, axis0);
                tulpa::add_rw2_precision(grad, H, x, start, size, tau, false);
            };
            block.log_prior = [start, size, axis0, theta_grid](
                const Rcpp::NumericVector& x, int k) {
                double tau = theta_grid(k, axis0);
                return tulpa::log_prior_rw2(x, start, size, tau, false);
            };
        }
        block.center = [start, size](Rcpp::NumericVector& x) -> double {
            return tulpa::center_effects(x, start, size);
        };
        blocks.push_back(block);
        return start + size;
    }

    if (type == "ar1") {
        require_axes(2);
        int size = Rcpp::as<int>(bs["n_times"]);
        Rcpp::IntegerVector temporal_idx = bs["temporal_idx"];
        int start = latent_offset;

        tulpa::LatentBlock block;
        block.start = start;
        block.size  = size;
        block.idx   = [temporal_idx](int i, int /*k_arm*/) { return temporal_idx[i]; };
        block.d_fac = [](int) { return 1.0; };
        block.add_prior = [start, size, axis0, theta_grid](
            tulpa::DenseVec& grad, tulpa::DenseMat& H,
            const Rcpp::NumericVector& x, int k) {
            double tau = theta_grid(k, axis0);
            double rho = theta_grid(k, axis0 + 1);
            tulpa::add_ar1_precision(grad, H, x, start, size, tau, rho);
        };
        block.log_prior = [start, size, axis0, theta_grid](
            const Rcpp::NumericVector& x, int k) {
            double tau = theta_grid(k, axis0);
            double rho = theta_grid(k, axis0 + 1);
            return tulpa::log_prior_ar1(x, start, size, tau, rho);
        };
        block.center = [start, size](Rcpp::NumericVector& x) -> double {
            return tulpa::center_effects(x, start, size);
        };
        blocks.push_back(block);
        return start + size;
    }

    if (type == "tgmrf") {
        // User-defined GMRF block. The R side has precomputed Q(theta_k) at
        // every outer-grid row plus log|Q_k| and log p(theta_k); this branch
        // just reads them and assembles add_prior / log_prior callbacks.
        // axis_count = block$theta_dim is allowed to vary; the C++ side
        // never indexes theta_grid for this block (Q_k is precomputed).
        int size = Rcpp::as<int>(bs["n_latent"]);
        Rcpp::IntegerVector obs_idx = bs["obs_idx"];

        Rcpp::List Q_p_list = bs["Q_csc_p_per_grid"];
        Rcpp::List Q_i_list = bs["Q_csc_i_per_grid"];
        Rcpp::List Q_x_list = bs["Q_csc_x_per_grid"];
        Rcpp::NumericVector logdet_Q_v = bs["logdet_Q_per_grid"];
        Rcpp::NumericVector log_pi_v   = bs["log_prior_theta_per_grid"];

        int n_grid_local = Q_p_list.size();
        if (Q_i_list.size() != n_grid_local ||
            Q_x_list.size() != n_grid_local ||
            logdet_Q_v.size() != n_grid_local ||
            log_pi_v.size() != n_grid_local) {
            Rcpp::stop("tgmrf block: per-grid arrays must all have length %d",
                       n_grid_local);
        }

        // Copy the CSC triples into pure C++ vectors. Two reasons: keep the
        // callback closures independent of the R-owned SEXP lifetimes once
        // build_blocks_from_spec returns, and let OpenMP scatter touch them
        // without R locks.
        auto Q_p_vec = std::make_shared<std::vector<std::vector<int>>>(n_grid_local);
        auto Q_i_vec = std::make_shared<std::vector<std::vector<int>>>(n_grid_local);
        auto Q_x_vec = std::make_shared<std::vector<std::vector<double>>>(n_grid_local);
        auto logdet_Q = std::make_shared<std::vector<double>>(n_grid_local);
        auto log_pi_theta = std::make_shared<std::vector<double>>(n_grid_local);
        for (int k = 0; k < n_grid_local; k++) {
            Rcpp::IntegerVector p_k = Q_p_list[k];
            Rcpp::IntegerVector i_k = Q_i_list[k];
            Rcpp::NumericVector x_k = Q_x_list[k];
            if (static_cast<int>(p_k.size()) != size + 1) {
                Rcpp::stop("tgmrf block: Q_csc_p_per_grid[[%d]] has length %d, expected %d",
                           k + 1, p_k.size(), size + 1);
            }
            (*Q_p_vec)[k].assign(p_k.begin(), p_k.end());
            (*Q_i_vec)[k].assign(i_k.begin(), i_k.end());
            (*Q_x_vec)[k].assign(x_k.begin(), x_k.end());
            (*logdet_Q)[k] = logdet_Q_v[k];
            (*log_pi_theta)[k] = log_pi_v[k];
        }

        int start = latent_offset;

        tulpa::LatentBlock block;
        block.start = start;
        block.size  = size;
        block.idx   = [obs_idx](int i, int /*k_arm*/) { return obs_idx[i]; };
        block.d_fac = [](int) { return 1.0; };
        block.add_prior = [start, size, Q_p_vec, Q_i_vec, Q_x_vec](
            tulpa::DenseVec& grad, tulpa::DenseMat& H,
            const Rcpp::NumericVector& x, int k) {
            const auto& p_v = (*Q_p_vec)[k];
            const auto& i_v = (*Q_i_vec)[k];
            const auto& x_v = (*Q_x_vec)[k];
            // Q is stored full (dgCMatrix coerced via generalMatrix). The
            // scatter walks every (i, j) once and adds Q_{ij} to H[i][j]
            // and -Q_{ij} * x_j to grad[i]; symmetric storage is implicit
            // because both (i, j) and (j, i) entries appear in the CSC.
            for (int j = 0; j < size; j++) {
                double xj = x[start + j];
                for (int idx = p_v[j]; idx < p_v[j + 1]; idx++) {
                    int    i_loc = i_v[idx];
                    double q_ij  = x_v[idx];
                    H[start + i_loc][start + j] += q_ij;
                    grad[start + i_loc] -= q_ij * xj;
                }
            }
        };
        block.log_prior = [start, size, Q_p_vec, Q_i_vec, Q_x_vec,
                           logdet_Q, log_pi_theta](
            const Rcpp::NumericVector& x, int k) {
            const auto& p_v = (*Q_p_vec)[k];
            const auto& i_v = (*Q_i_vec)[k];
            const auto& x_v = (*Q_x_vec)[k];
            double quad = 0.0;
            for (int j = 0; j < size; j++) {
                double xj = x[start + j];
                for (int idx = p_v[j]; idx < p_v[j + 1]; idx++) {
                    int    i_loc = i_v[idx];
                    double q_ij  = x_v[idx];
                    quad += x[start + i_loc] * q_ij * xj;
                }
            }
            return 0.5 * (*logdet_Q)[k]
                 - 0.5 * quad
                 - 0.5 * size * std::log(2.0 * M_PI)
                 + (*log_pi_theta)[k];
        };
        // No center: the user owns the parameterisation; centering would
        // require shifting the global intercept by the per-step mean offset
        // which is incompatible with arbitrary Q structures.
        blocks.push_back(block);
        return start + size;
    }

    if (type == "iid") {
        require_axes(1);
        int size = Rcpp::as<int>(bs["n_units"]);
        Rcpp::IntegerVector obs_idx = bs["obs_idx"];
        int start = latent_offset;

        // Standard N(0,1) prior on x_block; the d_fac = sigma_k applied at
        // compute_eta gives effective eta contribution N(0, sigma_k^2).
        // Identifiability anchored by the global intercept — no centering.
        tulpa::LatentBlock block;
        block.start = start;
        block.size  = size;
        block.idx   = [obs_idx](int i, int /*k_arm*/) { return obs_idx[i]; };
        block.d_fac = [axis0, theta_grid](int k) {
            return theta_grid(k, axis0);
        };
        block.add_prior = [start, size](
            tulpa::DenseVec& grad, tulpa::DenseMat& H,
            const Rcpp::NumericVector& x, int) {
            for (int s = 0; s < size; s++) {
                int idx = start + s;
                grad[idx] -= x[idx];
                H[idx][idx] += 1.0;
            }
        };
        block.log_prior = [start, size](const Rcpp::NumericVector& x, int) {
            double lp = 0.0;
            for (int s = 0; s < size; s++) {
                lp -= 0.5 * x[start + s] * x[start + s];
            }
            lp -= 0.5 * size * std::log(2.0 * M_PI);
            return lp;
        };
        // No center: x is anchored by the global intercept already.
        blocks.push_back(block);
        return start + size;
    }

    Rcpp::stop("Unknown block type '%s' in cpp_nested_laplace_multi", type.c_str());
    return latent_offset;
}

} // namespace

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_multi(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::List blocks_spec,
    Rcpp::NumericMatrix theta_grid,
    Rcpp::IntegerVector axis_offsets,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int B = blocks_spec.size();
    if (axis_offsets.size() != B + 1) {
        Rcpp::stop("axis_offsets must have length B+1 (got %d for B=%d)",
                   axis_offsets.size(), B);
    }
    int total_axes = axis_offsets[B];
    if (theta_grid.ncol() != total_axes) {
        Rcpp::stop("theta_grid must have %d columns (sum of axes), got %d",
                   total_axes, theta_grid.ncol());
    }

    int N = y.size();
    int p = X.ncol();
    int n_grid = theta_grid.nrow();
    int latent_offset = p + n_re_groups;

    std::vector<tulpa::LatentBlock> blocks;
    blocks.reserve(B + 2);
    for (int b = 0; b < B; b++) {
        Rcpp::List bs = blocks_spec[b];
        int axis0 = axis_offsets[b];
        int axis_count = axis_offsets[b + 1] - axis0;
        latent_offset = build_blocks_from_spec(
            bs, theta_grid, axis0, axis_count, latent_offset, blocks
        );
    }

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) {
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }

    Rcpp::List out = tulpa::run_multi_block_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        blocks,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, x_init,
        store_Q
    );
    out["theta_grid"]   = theta_grid;
    out["axis_offsets"] = axis_offsets;
    return out;
}
