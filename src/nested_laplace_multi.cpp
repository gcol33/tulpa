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
#include "nl_cell_cache.h"
#include "nested_laplace_multi.h"
#include "spde_block_factory.h"
#include "tgmrf_block_factory.h"
#include "tulpa/nested_likelihood.h"
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

        // Optional per-row design weight (spatially-varying coefficient): when
        // present, obs i's eta contribution is weight[i] * z[idx(i)] rather than
        // z[idx(i)], the areal f(cell, weight, ...) of an SVC field. Absent ->
        // uniform weight 1 (byte-identical to the plain intercept field). Read
        // into an owning std::shared_ptr<vector<double>> so the row_weight
        // closure carries no Rcpp object across the (copied) LatentBlock.
        std::shared_ptr<std::vector<double>> svc_w;
        if (bs.containsElementNamed("svc_weight") &&
            !Rf_isNull(bs["svc_weight"])) {
            Rcpp::NumericVector w = bs["svc_weight"];
            svc_w = std::make_shared<std::vector<double>>(w.begin(), w.end());
        }

        tulpa::LatentBlock block;
        block.start = start;
        block.size  = size;
        block.idx   = [spatial_idx](int i, int /*k_arm*/) { return spatial_idx[i]; };
        block.d_fac = [](int) { return 1.0; };
        if (svc_w) {
            block.row_weight = [svc_w](int i, int /*k_arm*/) {
                return (*svc_w)[i];
            };
        }
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
            // Structured ICAR component (tau = 1); shares the quadratic form and
            // the sum-to-zero penalty with add_icar_prior so the objective stays
            // consistent with the gradient, instead of re-deriving them inline.
            return tulpa::log_prior_icar_structured(x, phi_start, size, /*tau=*/1.0,
                                                    adj_rp, adj_ci, n_nbr);
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

        // Cache CSR for the dense log-det helper. The rho-dependent log|Q| is
        // cell-keyed (NlCellCache) so a parallel outer grid can never read one
        // cell's prep() value into another cell's log_prior() -- matching the
        // single-block CAR_proper path; cell-keyed state costs nothing.
        auto adj_rp_v = std::make_shared<std::vector<int>>(adj_rp.begin(), adj_rp.end());
        auto adj_ci_v = std::make_shared<std::vector<int>>(adj_ci.begin(), adj_ci.end());
        auto n_nbr_v  = std::make_shared<std::vector<int>>(n_nbr.begin(),  n_nbr.end());
        auto log_det_Q_rho = std::make_shared<tulpa::NlCellCache<double>>();

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
            double ld_val = tulpa_car_proper::car_log_det(size, Qmat);
            log_det_Q_rho->claim() = ld_val;
            log_det_Q_rho->publish(k);
            return std::isfinite(ld_val);
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
                                                 log_det_Q_rho->find(k),
                                                 adj_rp, adj_ci, n_nbr);
        };
        block.center = [start, size](Rcpp::NumericVector& x) -> double {
            return tulpa::center_effects(x, start, size);
        };
        blocks.push_back(block);
        return start + size;
    }

    if (type == "spde") {
        require_axes(2);  // (range, sigma)
        int n_mesh = Rcpp::as<int>(bs["n_mesh"]);

        // A is supplied as a single CSC projection (one arm). The shared SPDE
        // factory consumes per-arm lists, so wrap the slots in length-1 lists.
        Rcpp::NumericVector A_x = bs["A_x"];
        Rcpp::IntegerVector A_i = bs["A_i"];
        Rcpp::IntegerVector A_p = bs["A_p"];
        int n_obs = Rcpp::as<int>(bs["n_obs"]);
        Rcpp::List A_x_per_arm = Rcpp::List::create(A_x);
        Rcpp::List A_i_per_arm = Rcpp::List::create(A_i);
        Rcpp::List A_p_per_arm = Rcpp::List::create(A_p);
        Rcpp::IntegerVector n_obs_per_arm = Rcpp::IntegerVector::create(n_obs);

        Rcpp::NumericVector C0_diag = bs["C0_diag"];
        Rcpp::NumericVector G1_x   = bs["G1_x"];
        Rcpp::IntegerVector G1_i   = bs["G1_i"];
        Rcpp::IntegerVector G1_p   = bs["G1_p"];
        double nu = Rcpp::as<double>(bs["nu"]);

        bool use_rational = false;
        std::vector<double> rat_poles, rat_weights;
        if (bs.containsElementNamed("rational_poles") &&
            bs.containsElementNamed("rational_weights") &&
            !Rf_isNull(bs["rational_poles"]) &&
            !Rf_isNull(bs["rational_weights"])) {
            rat_poles   = Rcpp::as<std::vector<double>>(bs["rational_poles"]);
            rat_weights = Rcpp::as<std::vector<double>>(bs["rational_weights"]);
            use_rational = !rat_poles.empty();
        }

        // Same factory and SpdeQBuilder the single-Laplace occupancy SPDE path
        // uses (cpp_nested_laplace_spde); axis_range / axis_sigma index this
        // block's two columns in theta_grid. INDEXED_MULTI: the per-obs eta
        // contribution is (A u)_i, read from obs_indices, not a one-node idx.
        tulpa::LatentBlock block = tulpa::make_spde_block(
            /*start=*/latent_offset, n_mesh,
            A_x_per_arm, A_i_per_arm, A_p_per_arm, n_obs_per_arm,
            /*n_arms=*/1, /*block_index=*/0,
            C0_diag, G1_x, G1_i, G1_p, nu,
            /*axis_range=*/axis0, /*axis_sigma=*/axis0 + 1, theta_grid,
            use_rational, rat_poles, rat_weights
        );
        blocks.push_back(block);
        return latent_offset + n_mesh;
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
        // every outer-grid row plus log|Q_k| and log p(theta_k); the shared
        // factory (also used by the joint multi-arm driver) reads them and
        // assembles the callbacks. The single-block dense driver consumes only
        // block.idx / add_prior / log_prior, so the factory's sparse / pattern
        // / contrib-kind fields are inert here.
        int size = Rcpp::as<int>(bs["n_latent"]);
        Rcpp::IntegerVector obs_idx = bs["obs_idx"];
        int start = latent_offset;

        Rcpp::List Q_p = bs["Q_csc_p_per_grid"];
        Rcpp::List Q_i = bs["Q_csc_i_per_grid"];
        Rcpp::List Q_x = bs["Q_csc_x_per_grid"];
        Rcpp::NumericVector logdet_Q = bs["logdet_Q_per_grid"];
        Rcpp::NumericVector log_pi   = bs["log_prior_theta_per_grid"];

        tulpa::LatentBlock block = tulpa::make_tgmrf_block(
            start, size,
            [obs_idx](int i, int /*k_arm*/) { return obs_idx[i]; },
            Q_p, Q_i, Q_x, logdet_Q, log_pi, /*block_index=*/0);
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
    bool store_Q = false,
    double prune_tol = 0.0,
    SEXP likelihood = R_NilValue,
    bool progress = false, int progress_every = 0,
    double progress_throttle = 0.0,
    std::string progress_file = "",
    std::string checkpoint_path = ""
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

    // Model-supplied likelihood (e.g. tulpaObs occupancy): when present, the
    // inner solve routes its per-observation score / Fisher weight / log-lik
    // through this spec instead of the built-in `family`. The XPtr keeps the
    // spec + its response storage alive for the duration of the call.
    // nullptr => built-in family.
    const tulpa::LikelihoodSpec* ext_spec     = nullptr;
    void*                        ext_response = nullptr;
    if (!Rf_isNull(likelihood)) {
        Rcpp::XPtr<tulpa::NestedLikelihood> lk(likelihood);
        if (lk->spec == nullptr) {
            Rcpp::stop("cpp_nested_laplace_multi: `likelihood` has a null spec.");
        }
        ext_spec     = lk->spec;
        ext_response = lk->response_data;
    }

    // Build the reporter when either channel is wanted: the console line under
    // `progress` (the verbose/TTY channel), or the heartbeat file whenever
    // `progress_file` is set. A detached fit (progress = false) with a
    // progress_file still gets its file ETA -- the channel it exists for

    std::unique_ptr<tulpa_progress::GridProgress> gp;
    if (progress || !progress_file.empty()) {
        gp.reset(new tulpa_progress::GridProgress(
            "nested-laplace", n_grid, progress_every, progress_throttle,
            progress_file, /*emit_console=*/progress));
    }

    // Grid-cell checkpoint/resume. The fingerprint folds the
    // shared observation inputs, the solver settings, the axis layout, and the
    // full block spec (adjacency / coords / type strings, via fold_sexp) so a
    // resume onto a file written for any different input errors. Per-cell keys
    // are the theta_grid row coordinates (column-major, stride 1).
    std::unique_ptr<tulpa::GridCheckpoint> ckpt;
    if (!checkpoint_path.empty()) {
        tulpa::Fingerprint fp;
        fp.fold_str("nl_multi");
        fp.fold_pod(max_iter);
        fp.fold_pod(tol);
        fp.fold_pod(N);
        fp.fold_pod(p);
        fp.fold_pod(n_re_groups);
        fp.fold_pod(sigma_re);
        fp.fold_str(family);
        fp.fold_pod(phi);
        fp.fold_pod(total_axes);
        if (axis_offsets.size()) fp.fold(axis_offsets.begin(),
                                         (std::size_t)axis_offsets.size() * sizeof(int));
        if (y.size())  fp.fold(y.begin(), (std::size_t)y.size() * sizeof(double));
        if (n.size())  fp.fold(n.begin(), (std::size_t)n.size() * sizeof(int));
        if (X.size())  fp.fold(X.begin(), (std::size_t)X.size() * sizeof(double));
        // The per-observation RE group assignment changes every cell's mode and
        // log_marginal, so it must fingerprint alongside n_re_groups (the count).
        if (re_idx.size()) fp.fold(re_idx.begin(),
                                   (std::size_t)re_idx.size() * sizeof(int));
        tulpa::fold_sexp(fp, blocks_spec);
        tulpa::CellKeyBuilder kb(n_grid);
        for (int j = 0; j < total_axes; j++) kb.add_axis(&theta_grid(0, j));
        ckpt.reset(new tulpa::GridCheckpoint(checkpoint_path, fp.value(),
                                             kb.take()));
    }

    Rcpp::List out = tulpa::run_multi_block_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        blocks,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, x_init,
        store_Q,
        /*n_threads_outer=*/1,
        prune_tol,
        ext_spec, ext_response,
        gp.get(), ckpt.get()
    );
    out["theta_grid"]   = theta_grid;
    out["axis_offsets"] = axis_offsets;
    return out;
}
