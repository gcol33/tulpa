// nested_laplace_joint_multi.cpp
// cpp_nested_laplace_joint_multi: Rcpp entry for joint multi-likelihood
// nested Laplace with a list-of-blocks prior. The joint analogue of
// cpp_nested_laplace_multi (single-arm).
//
// At most one block in the list can be designated the "copy block" (first
// ship: must be a spatial block -- icar / bym2 / car_proper). The copy block
// uses INLA `copy=` semantics: donor arms see one amplitude axis, the copy
// arm sees a second. Other blocks are shared identically across arms.
//
// The R-side primary spec is (sigma, alpha) (gcol33/tulpa#22): sigma is the
// donor amplitude axis and alpha is the direct copy coefficient, so the copy
// arm's field amplitude is `alpha * sigma`. The R parser materializes the
// per-arm scaling by passing `sigma` in the donor column and `alpha * sigma`
// in the copy column of theta_grid. This kernel only sees two arm-scale
// columns (donor and copy) and is parameterization-agnostic: any R-side
// reparameterization that yields per-arm amplitudes on the same two columns
// works without changing the kernel. The legacy convention was to grid the
// two arms' sigmas directly; the new convention plus the R-side materialize
// preserves the kernel ABI.
//
// Block-spec input format (R side wraps this into the .NL_REGISTRY shape):
//   blocks_spec[[b]] = list(
//     type             : "icar" | "bym2" | "car_proper" |
//                        "rw1"  | "rw2"  | "ar1"        | "iid"
//     spatial_idx OR temporal_idx OR obs_idx : List of per-arm IntegerVectors
//                                              (length == n_arms)
//     n_spatial_units / n_times / n_units    : structural size
//     adj_row_ptr, adj_col_idx, n_neighbors  : (icar/bym2/car_proper only)
//     scale_factor (bym2 only)
//     cyclic (rw1 only)
//   )
//
// theta_grid columns per block depend on whether the block is the copy block:
//   * Non-copy axes (default):
//       icar      : (tau,)
//       bym2      : (sigma, rho)
//       car_proper: (tau, rho_car)
//       rw1, rw2  : (tau,)
//       ar1       : (tau, rho)
//       iid       : (sigma,)
//   * Copy block (spatial only; first ship restriction):
//       icar      : (sigma_donor, sigma_copy)
//       bym2      : (sigma_donor, sigma_copy, rho)
//       car_proper: (sigma_donor, sigma_copy, rho_car)
// where the R side fills `sigma_donor = sigma`, `sigma_copy = alpha * sigma`
// before calling. The R-side parser is responsible for building theta_grid
// with the correct per-block layout; this kernel just reads the axes by
// offset.

#include "joint_hessian_pattern.h"
#include "laplace_core.h"
#include "laplace_re_priors.h"
#include "laplace_spatial_priors.h"
#include "laplace_temporal_priors.h"
#include "latent_block.h"
#include "nested_laplace_joint_core.h"
#include "nested_laplace_joint_multi.h"
#include "sparse_hessian.h"
#include "hsgp_block_factory.h"
#include "hsgp_mo_block_factory.h"
#include "latent_factor_block_factory.h"
#include "spde_block_factory.h"
#include "tgmrf_block_factory.h"
#include "hmc_car_proper.h"
#include <Rcpp.h>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// Build a per-arm idx closure from an Rcpp::List of per-arm IntegerVectors.
// The closure captures cached IntegerVectors so per-grid-point eta evaluation
// doesn't re-resolve the list each call.
inline std::function<int(int, int)> make_per_arm_idx_fn(
    const Rcpp::List& per_arm_idx_list, int n_arms, const char* field_name,
    int block_index
) {
    if (static_cast<int>(per_arm_idx_list.size()) != n_arms) {
        Rcpp::stop("blocks_spec[[%d]]$%s must be a list of length n_arms (%d), got %d.",
                   block_index + 1, field_name, n_arms,
                   static_cast<int>(per_arm_idx_list.size()));
    }
    std::vector<Rcpp::IntegerVector> cache;
    cache.reserve(n_arms);
    for (int k = 0; k < n_arms; k++) {
        cache.push_back(Rcpp::as<Rcpp::IntegerVector>(per_arm_idx_list[k]));
    }
    return [cache](int i, int k_arm) -> int {
        return cache[k_arm][i];
    };
}

// Per-arm amplitude dispatch for a copy block. axis_donor and axis_copy are
// column indices into theta_grid for this block's donor-arm and copy-arm
// amplitude axes respectively (set up by the R-side parser). Under the
// (sigma, alpha) reparameterization (gcol33/tulpa#22), the R side fills
// theta_grid[, axis_donor] = sigma and theta_grid[, axis_copy] = alpha*sigma
// before calling this kernel.
inline std::function<double(int, int)> make_copy_arm_scale_fn(
    int copy_arm, int axis_donor, int axis_copy,
    const Rcpp::NumericMatrix& theta_grid
) {
    return [copy_arm, axis_donor, axis_copy, theta_grid](
        int k_arm, int k_grid) -> double {
        return (k_arm == copy_arm) ? theta_grid(k_grid, axis_copy)
                                   : theta_grid(k_grid, axis_donor);
    };
}

// Build the LatentBlock(s) for one block-spec entry in the joint multi-block
// driver. Returns the new latent_offset after appending the block's
// sub-vector(s) to the joint latent vector layout.
//
// `is_copy_block` toggles the copy-block parameterization for spatial types:
//   * unit-precision prior (tau = 1) with arm_scale carrying the per-arm
//     amplitudes (donor: sigma, copy: alpha * sigma under the (sigma, alpha)
//     reparam materialized R-side; see file header)
//   * 2 (icar) or 3 (bym2, car_proper) per-block axes
// For non-copy blocks, the standard single-arm parameterization is used
// (tau on the prior, no arm_scale, axes as in the single-arm registry).
int build_joint_blocks_from_spec(
    const Rcpp::List& bs,
    const Rcpp::NumericMatrix& theta_grid,
    int axis0,
    int axis_count,
    int latent_offset,
    int n_arms,
    int block_index,
    bool is_copy_block,
    int copy_arm,
    std::vector<tulpa::LatentBlock>& blocks
) {
    std::string type = Rcpp::as<std::string>(bs["type"]);

    auto require_axes = [&](int needed) {
        if (axis_count != needed) {
            Rcpp::stop("Block %d (type '%s'%s) expects %d axes, got %d",
                       block_index + 1, type.c_str(),
                       is_copy_block ? ", copy" : "",
                       needed, axis_count);
        }
    };

    if (type == "icar") {
        int size = Rcpp::as<int>(bs["n_spatial_units"]);
        Rcpp::List spatial_idx_list = bs["spatial_idx"];
        Rcpp::IntegerVector adj_rp = bs["adj_row_ptr"];
        Rcpp::IntegerVector adj_ci = bs["adj_col_idx"];
        Rcpp::IntegerVector n_nbr  = bs["n_neighbors"];
        int start = latent_offset;

        tulpa::LatentBlock block;
        block.start = start;
        block.size  = size;
        block.idx   = make_per_arm_idx_fn(spatial_idx_list, n_arms,
                                            "spatial_idx", block_index);

        if (is_copy_block) {
            require_axes(2);  // (sigma_donor, sigma_copy)
            block.d_fac = [](int) -> double { return 1.0; };
            block.arm_scale = make_copy_arm_scale_fn(
                copy_arm, axis0, axis0 + 1, theta_grid);
            block.add_prior = [start, size, adj_rp, adj_ci, n_nbr](
                tulpa::DenseVec& grad, tulpa::DenseMat& H,
                const Rcpp::NumericVector& x, int /*k*/) {
                tulpa::add_icar_prior(grad, H, x, start, size, /*tau=*/1.0,
                                       adj_rp, adj_ci, n_nbr);
            };
            block.add_prior_sparse = [start, size, adj_rp, adj_ci, n_nbr](
                tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                const Rcpp::NumericVector& x, int /*k*/) {
                tulpa::add_icar_prior_sparse(grad, H, x, start, size, /*tau=*/1.0,
                                              adj_rp, adj_ci, n_nbr);
            };
            block.log_prior = [start, size, adj_rp, adj_ci, n_nbr](
                const Rcpp::NumericVector& x, int /*k*/) -> double {
                return tulpa::log_prior_icar(x, start, size, /*tau=*/1.0,
                                              adj_rp, adj_ci, n_nbr);
            };
        } else {
            require_axes(1);  // (tau,)
            block.d_fac = [](int) -> double { return 1.0; };
            block.add_prior = [start, size, axis0, theta_grid,
                                adj_rp, adj_ci, n_nbr](
                tulpa::DenseVec& grad, tulpa::DenseMat& H,
                const Rcpp::NumericVector& x, int k) {
                double tau = theta_grid(k, axis0);
                tulpa::add_icar_prior(grad, H, x, start, size, tau,
                                       adj_rp, adj_ci, n_nbr);
            };
            block.add_prior_sparse = [start, size, axis0, theta_grid,
                                       adj_rp, adj_ci, n_nbr](
                tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                const Rcpp::NumericVector& x, int k) {
                double tau = theta_grid(k, axis0);
                tulpa::add_icar_prior_sparse(grad, H, x, start, size, tau,
                                              adj_rp, adj_ci, n_nbr);
            };
            block.log_prior = [start, size, axis0, theta_grid,
                                adj_rp, adj_ci, n_nbr](
                const Rcpp::NumericVector& x, int k) -> double {
                double tau = theta_grid(k, axis0);
                return tulpa::log_prior_icar(x, start, size, tau,
                                              adj_rp, adj_ci, n_nbr);
            };
        }
        block.contrib_kind = tulpa::BlockContribKind::INDEXED_SINGLE;
        block.prior_kind   = tulpa::PriorFillKind::ADJACENCY;
        block.add_prior_pattern = [start, size, adj_rp, adj_ci](
            std::vector<std::pair<int,int>>& out) {
            tulpa::add_car_pattern(out, start, size, adj_rp, adj_ci);
        };
        block.center = [start, size](Rcpp::NumericVector& x) -> double {
            return tulpa::center_effects(x, start, size);
        };
        blocks.push_back(block);
        return start + size;
    }

    if (type == "bym2") {
        int size = Rcpp::as<int>(bs["n_spatial_units"]);
        Rcpp::List spatial_idx_list = bs["spatial_idx"];
        Rcpp::IntegerVector adj_rp = bs["adj_row_ptr"];
        Rcpp::IntegerVector adj_ci = bs["adj_col_idx"];
        Rcpp::IntegerVector n_nbr  = bs["n_neighbors"];
        double scale_factor = bs.containsElementNamed("scale_factor") ?
            Rcpp::as<double>(bs["scale_factor"]) : 1.0;
        int phi_start   = latent_offset;
        int theta_start = phi_start + size;

        auto idx_fn = make_per_arm_idx_fn(spatial_idx_list, n_arms,
                                           "spatial_idx", block_index);

        std::function<double(int, int)> arm_scale_fn;
        std::function<double(int)>      d_fac_phi_fn;
        std::function<double(int)>      d_fac_theta_fn;

        if (is_copy_block) {
            require_axes(3);  // (sigma_donor, sigma_copy, rho)
            arm_scale_fn = make_copy_arm_scale_fn(
                copy_arm, axis0, axis0 + 1, theta_grid);
            int axis_rho = axis0 + 2;
            d_fac_phi_fn = [axis_rho, theta_grid, scale_factor](int k_grid)
                -> double {
                double rho = theta_grid(k_grid, axis_rho);
                return std::sqrt(rho + 1e-10) * scale_factor;
            };
            d_fac_theta_fn = [axis_rho, theta_grid](int k_grid) -> double {
                double rho = theta_grid(k_grid, axis_rho);
                return std::sqrt(1.0 - rho + 1e-10);
            };
        } else {
            require_axes(2);  // (sigma, rho) - single-arm BYM2 conventions
            // Shared block: sigma rolls into d_fac directly (no per-arm
            // scaling). Matches single-arm BYM2 (build_blocks_from_spec).
            int axis_sigma = axis0;
            int axis_rho   = axis0 + 1;
            d_fac_phi_fn = [axis_sigma, axis_rho, theta_grid, scale_factor](
                int k_grid) -> double {
                double sigma = theta_grid(k_grid, axis_sigma);
                double rho   = theta_grid(k_grid, axis_rho);
                return sigma * std::sqrt(rho + 1e-10) * scale_factor;
            };
            d_fac_theta_fn = [axis_sigma, axis_rho, theta_grid](int k_grid)
                -> double {
                double sigma = theta_grid(k_grid, axis_sigma);
                double rho   = theta_grid(k_grid, axis_rho);
                return sigma * std::sqrt(1.0 - rho + 1e-10);
            };
        }

        tulpa::LatentBlock phi_block;
        phi_block.start = phi_start;
        phi_block.size  = size;
        phi_block.idx   = idx_fn;
        phi_block.d_fac = d_fac_phi_fn;
        if (arm_scale_fn) phi_block.arm_scale = arm_scale_fn;
        phi_block.add_prior = [phi_start, size, adj_rp, adj_ci, n_nbr](
            tulpa::DenseVec& grad, tulpa::DenseMat& H,
            const Rcpp::NumericVector& x, int /*k*/) {
            tulpa::add_icar_prior(grad, H, x, phi_start, size, /*tau=*/1.0,
                                   adj_rp, adj_ci, n_nbr);
        };
        phi_block.add_prior_sparse = [phi_start, size, adj_rp, adj_ci, n_nbr](
            tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
            const Rcpp::NumericVector& x, int /*k*/) {
            tulpa::add_icar_prior_sparse(grad, H, x, phi_start, size,
                                          /*tau=*/1.0,
                                          adj_rp, adj_ci, n_nbr);
        };
        phi_block.contrib_kind = tulpa::BlockContribKind::INDEXED_SINGLE;
        phi_block.prior_kind   = tulpa::PriorFillKind::ADJACENCY;
        phi_block.add_prior_pattern = [phi_start, size, adj_rp, adj_ci](
            std::vector<std::pair<int,int>>& out) {
            tulpa::add_car_pattern(out, phi_start, size, adj_rp, adj_ci);
        };
        phi_block.log_prior = [phi_start, size, adj_rp, adj_ci, n_nbr](
            const Rcpp::NumericVector& x, int /*k*/) -> double {
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
        theta_block.idx   = idx_fn;
        theta_block.d_fac = d_fac_theta_fn;
        if (arm_scale_fn) theta_block.arm_scale = arm_scale_fn;
        theta_block.add_prior = [theta_start, size](
            tulpa::DenseVec& grad, tulpa::DenseMat& H,
            const Rcpp::NumericVector& x, int /*k*/) {
            for (int s = 0; s < size; s++) {
                int idx = theta_start + s;
                grad[idx] -= x[idx];
                H[idx][idx] += 1.0;
            }
        };
        theta_block.add_prior_sparse = [theta_start, size](
            tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
            const Rcpp::NumericVector& x, int /*k*/) {
            for (int s = 0; s < size; s++) {
                int idx = theta_start + s;
                grad[idx] -= x[idx];
                H.add(idx, idx, 1.0);
            }
        };
        theta_block.contrib_kind = tulpa::BlockContribKind::INDEXED_SINGLE;
        theta_block.prior_kind   = tulpa::PriorFillKind::NONE;
        // No add_prior_pattern: prior is diagonal, builder adds it
        // unconditionally.
        theta_block.log_prior = [theta_start, size](
            const Rcpp::NumericVector& x, int /*k*/) -> double {
            double lp = 0.0;
            for (int s = 0; s < size; s++) {
                double v = x[theta_start + s];
                lp -= 0.5 * v * v;
            }
            lp -= 0.5 * size * std::log(2.0 * M_PI);
            return lp;
        };
        // theta has no centering: prior is symmetric.
        blocks.push_back(theta_block);
        return theta_start + size;
    }

    if (type == "car_proper") {
        int size = Rcpp::as<int>(bs["n_spatial_units"]);
        Rcpp::List spatial_idx_list = bs["spatial_idx"];
        Rcpp::IntegerVector adj_rp = bs["adj_row_ptr"];
        Rcpp::IntegerVector adj_ci = bs["adj_col_idx"];
        Rcpp::IntegerVector n_nbr  = bs["n_neighbors"];
        int start = latent_offset;

        auto adj_rp_v = std::make_shared<std::vector<int>>(adj_rp.begin(), adj_rp.end());
        auto adj_ci_v = std::make_shared<std::vector<int>>(adj_ci.begin(), adj_ci.end());
        auto n_nbr_v  = std::make_shared<std::vector<int>>(n_nbr.begin(),  n_nbr.end());
        auto log_det_Q_rho = std::make_shared<double>(0.0);

        tulpa::LatentBlock block;
        block.start = start;
        block.size  = size;
        block.idx   = make_per_arm_idx_fn(spatial_idx_list, n_arms,
                                            "spatial_idx", block_index);

        if (is_copy_block) {
            require_axes(3);  // (sigma_donor, sigma_copy, rho_car)
            block.d_fac = [](int) -> double { return 1.0; };
            block.arm_scale = make_copy_arm_scale_fn(
                copy_arm, axis0, axis0 + 1, theta_grid);
            int axis_rho_car = axis0 + 2;
            block.prep = [size, axis_rho_car, theta_grid,
                           adj_rp_v, adj_ci_v, n_nbr_v, log_det_Q_rho](
                int k_grid) -> bool {
                double rho_car = theta_grid(k_grid, axis_rho_car);
                std::vector<double> Qmat = tulpa_car_proper::compute_car_precision(
                    size, *adj_rp_v, *adj_ci_v, *n_nbr_v, rho_car);
                *log_det_Q_rho = tulpa_car_proper::car_log_det(size, Qmat);
                return std::isfinite(*log_det_Q_rho);
            };
            block.add_prior = [start, size, axis_rho_car, theta_grid,
                                adj_rp, adj_ci, n_nbr](
                tulpa::DenseVec& grad, tulpa::DenseMat& H,
                const Rcpp::NumericVector& x, int k_grid) {
                tulpa::add_car_proper_prior(grad, H, x, start, size,
                                             /*tau=*/1.0,
                                             theta_grid(k_grid, axis_rho_car),
                                             adj_rp, adj_ci, n_nbr);
            };
            block.add_prior_sparse = [start, size, axis_rho_car, theta_grid,
                                       adj_rp, adj_ci, n_nbr](
                tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                const Rcpp::NumericVector& x, int k_grid) {
                tulpa::add_car_proper_prior_sparse(
                    grad, H, x, start, size,
                    /*tau=*/1.0,
                    theta_grid(k_grid, axis_rho_car),
                    adj_rp, adj_ci, n_nbr);
            };
            block.log_prior = [start, size, axis_rho_car, theta_grid,
                                adj_rp, adj_ci, n_nbr, log_det_Q_rho](
                const Rcpp::NumericVector& x, int k_grid) -> double {
                return tulpa::log_prior_car_proper(
                    x, start, size, /*tau=*/1.0,
                    theta_grid(k_grid, axis_rho_car), *log_det_Q_rho,
                    adj_rp, adj_ci, n_nbr);
            };
            // Proper CAR has full-rank Q; no centering.
        } else {
            require_axes(2);  // (tau, rho_car) - single-arm conventions
            block.d_fac = [](int) -> double { return 1.0; };
            int axis_tau     = axis0;
            int axis_rho_car = axis0 + 1;
            block.prep = [size, axis_rho_car, theta_grid,
                           adj_rp_v, adj_ci_v, n_nbr_v, log_det_Q_rho](
                int k_grid) -> bool {
                double rho_car = theta_grid(k_grid, axis_rho_car);
                std::vector<double> Qmat = tulpa_car_proper::compute_car_precision(
                    size, *adj_rp_v, *adj_ci_v, *n_nbr_v, rho_car);
                *log_det_Q_rho = tulpa_car_proper::car_log_det(size, Qmat);
                return std::isfinite(*log_det_Q_rho);
            };
            block.add_prior = [start, size, axis_tau, axis_rho_car, theta_grid,
                                adj_rp, adj_ci, n_nbr](
                tulpa::DenseVec& grad, tulpa::DenseMat& H,
                const Rcpp::NumericVector& x, int k_grid) {
                tulpa::add_car_proper_prior(grad, H, x, start, size,
                                             theta_grid(k_grid, axis_tau),
                                             theta_grid(k_grid, axis_rho_car),
                                             adj_rp, adj_ci, n_nbr);
            };
            block.add_prior_sparse = [start, size, axis_tau, axis_rho_car,
                                       theta_grid, adj_rp, adj_ci, n_nbr](
                tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                const Rcpp::NumericVector& x, int k_grid) {
                tulpa::add_car_proper_prior_sparse(
                    grad, H, x, start, size,
                    theta_grid(k_grid, axis_tau),
                    theta_grid(k_grid, axis_rho_car),
                    adj_rp, adj_ci, n_nbr);
            };
            block.log_prior = [start, size, axis_tau, axis_rho_car, theta_grid,
                                adj_rp, adj_ci, n_nbr, log_det_Q_rho](
                const Rcpp::NumericVector& x, int k_grid) -> double {
                return tulpa::log_prior_car_proper(
                    x, start, size, theta_grid(k_grid, axis_tau),
                    theta_grid(k_grid, axis_rho_car), *log_det_Q_rho,
                    adj_rp, adj_ci, n_nbr);
            };
            block.center = [start, size](Rcpp::NumericVector& x) -> double {
                return tulpa::center_effects(x, start, size);
            };
        }
        block.contrib_kind = tulpa::BlockContribKind::INDEXED_SINGLE;
        block.prior_kind   = tulpa::PriorFillKind::ADJACENCY;
        block.add_prior_pattern = [start, size, adj_rp, adj_ci](
            std::vector<std::pair<int,int>>& out) {
            tulpa::add_car_pattern(out, start, size, adj_rp, adj_ci);
        };
        blocks.push_back(block);
        return start + size;
    }

    if (type == "rw1" || type == "rw2") {
        if (is_copy_block) {
            Rcpp::stop("Block %d: copy semantics are only supported on spatial blocks for now (J-B first ship). See dev_notes/plan_multi_block_joint.md.",
                       block_index + 1);
        }
        require_axes(1);
        int size = Rcpp::as<int>(bs["n_times"]);
        Rcpp::List temporal_idx_list = bs["temporal_idx"];
        bool cyclic = (type == "rw1") &&
                      bs.containsElementNamed("cyclic") &&
                      Rcpp::as<bool>(bs["cyclic"]);
        int start = latent_offset;

        tulpa::LatentBlock block;
        block.start = start;
        block.size  = size;
        block.idx   = make_per_arm_idx_fn(temporal_idx_list, n_arms,
                                            "temporal_idx", block_index);
        block.d_fac = [](int) -> double { return 1.0; };
        if (type == "rw1") {
            block.add_prior = [start, size, axis0, theta_grid, cyclic](
                tulpa::DenseVec& grad, tulpa::DenseMat& H,
                const Rcpp::NumericVector& x, int k) {
                double tau = theta_grid(k, axis0);
                tulpa::add_rw1_precision(grad, H, x, start, size, tau, cyclic);
            };
            block.add_prior_sparse = [start, size, axis0, theta_grid, cyclic](
                tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                const Rcpp::NumericVector& x, int k) {
                double tau = theta_grid(k, axis0);
                tulpa::add_rw1_precision_sparse(grad, H, x, start, size, tau, cyclic);
            };
            block.add_prior_pattern = [start, size, cyclic](
                std::vector<std::pair<int,int>>& out) {
                tulpa::add_rw1_pattern(out, start, size, cyclic);
            };
            block.log_prior = [start, size, axis0, theta_grid, cyclic](
                const Rcpp::NumericVector& x, int k) -> double {
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
            block.add_prior_sparse = [start, size, axis0, theta_grid](
                tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                const Rcpp::NumericVector& x, int k) {
                double tau = theta_grid(k, axis0);
                tulpa::add_rw2_precision_sparse(grad, H, x, start, size, tau, false);
            };
            block.add_prior_pattern = [start, size](
                std::vector<std::pair<int,int>>& out) {
                tulpa::add_rw2_pattern(out, start, size, /*cyclic=*/false);
            };
            block.log_prior = [start, size, axis0, theta_grid](
                const Rcpp::NumericVector& x, int k) -> double {
                double tau = theta_grid(k, axis0);
                return tulpa::log_prior_rw2(x, start, size, tau, false);
            };
        }
        block.contrib_kind = tulpa::BlockContribKind::INDEXED_SINGLE;
        block.prior_kind   = tulpa::PriorFillKind::ADJACENCY;
        block.center = [start, size](Rcpp::NumericVector& x) -> double {
            return tulpa::center_effects(x, start, size);
        };
        blocks.push_back(block);
        return start + size;
    }

    if (type == "ar1") {
        if (is_copy_block) {
            Rcpp::stop("Block %d: copy semantics are only supported on spatial blocks for now (J-B first ship). See dev_notes/plan_multi_block_joint.md.",
                       block_index + 1);
        }
        require_axes(2);
        int size = Rcpp::as<int>(bs["n_times"]);
        Rcpp::List temporal_idx_list = bs["temporal_idx"];
        int start = latent_offset;

        tulpa::LatentBlock block;
        block.start = start;
        block.size  = size;
        block.idx   = make_per_arm_idx_fn(temporal_idx_list, n_arms,
                                            "temporal_idx", block_index);
        block.d_fac = [](int) -> double { return 1.0; };
        block.add_prior = [start, size, axis0, theta_grid](
            tulpa::DenseVec& grad, tulpa::DenseMat& H,
            const Rcpp::NumericVector& x, int k) {
            double tau = theta_grid(k, axis0);
            double rho = theta_grid(k, axis0 + 1);
            tulpa::add_ar1_precision(grad, H, x, start, size, tau, rho);
        };
        block.add_prior_sparse = [start, size, axis0, theta_grid](
            tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
            const Rcpp::NumericVector& x, int k) {
            double tau = theta_grid(k, axis0);
            double rho = theta_grid(k, axis0 + 1);
            tulpa::add_ar1_precision_sparse(grad, H, x, start, size, tau, rho);
        };
        block.contrib_kind = tulpa::BlockContribKind::INDEXED_SINGLE;
        block.prior_kind   = tulpa::PriorFillKind::ADJACENCY;
        block.add_prior_pattern = [start, size](
            std::vector<std::pair<int,int>>& out) {
            tulpa::add_ar1_pattern(out, start, size);
        };
        block.log_prior = [start, size, axis0, theta_grid](
            const Rcpp::NumericVector& x, int k) -> double {
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

    if (type == "iid") {
        if (is_copy_block) {
            Rcpp::stop("Block %d: copy semantics are only supported on spatial blocks for now (J-B first ship). See dev_notes/plan_multi_block_joint.md.",
                       block_index + 1);
        }
        require_axes(1);
        int size = Rcpp::as<int>(bs["n_units"]);
        Rcpp::List obs_idx_list = bs["obs_idx"];
        int start = latent_offset;

        tulpa::LatentBlock block;
        block.start = start;
        block.size  = size;
        block.idx   = make_per_arm_idx_fn(obs_idx_list, n_arms,
                                            "obs_idx", block_index);
        block.d_fac = [axis0, theta_grid](int k) -> double {
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
        block.add_prior_sparse = [start, size](
            tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
            const Rcpp::NumericVector& x, int) {
            for (int s = 0; s < size; s++) {
                int idx = start + s;
                grad[idx] -= x[idx];
                H.add(idx, idx, 1.0);
            }
        };
        block.contrib_kind = tulpa::BlockContribKind::INDEXED_SINGLE;
        block.prior_kind   = tulpa::PriorFillKind::NONE;
        // No add_prior_pattern: prior is diagonal, builder adds it
        // unconditionally.
        block.log_prior = [start, size](const Rcpp::NumericVector& x, int)
            -> double {
            double lp = 0.0;
            for (int s = 0; s < size; s++) {
                lp -= 0.5 * x[start + s] * x[start + s];
            }
            lp -= 0.5 * size * std::log(2.0 * M_PI);
            return lp;
        };
        // IID is anchored by the global per-arm intercepts; no centering.
        blocks.push_back(block);
        return start + size;
    }

    if (type == "tgmrf") {
        if (is_copy_block) {
            Rcpp::stop("Block %d: copy semantics for tgmrf blocks are not "
                       "supported (first ship: spatial copy only on icar / "
                       "bym2 / car_proper).",
                       block_index + 1);
        }
        // axis_count is allowed to vary; the C++ side never indexes
        // theta_grid for this block (Q_k is precomputed R-side).

        int size = Rcpp::as<int>(bs["n_latent"]);
        Rcpp::List obs_idx_list = bs["obs_idx"];
        auto obs_idx_fn = make_per_arm_idx_fn(obs_idx_list, n_arms,
                                                "obs_idx", block_index);

        Rcpp::List Q_p_list = bs["Q_csc_p_per_grid"];
        Rcpp::List Q_i_list = bs["Q_csc_i_per_grid"];
        Rcpp::List Q_x_list = bs["Q_csc_x_per_grid"];
        Rcpp::NumericVector logdet_Q_v = bs["logdet_Q_per_grid"];
        Rcpp::NumericVector log_pi_v   = bs["log_prior_theta_per_grid"];

        tulpa::LatentBlock block = tulpa::make_tgmrf_block(
            latent_offset, size, obs_idx_fn,
            Q_p_list, Q_i_list, Q_x_list,
            logdet_Q_v, log_pi_v,
            block_index
        );
        blocks.push_back(block);
        return latent_offset + size;
    }

    if (type == "hsgp") {
        if (is_copy_block) {
            Rcpp::stop("Block %d: copy semantics for HSGP blocks are not "
                       "supported (first ship: spatial copy only on icar / "
                       "bym2 / car_proper).",
                       block_index + 1);
        }
        require_axes(2);  // (log_sigma2, log_lengthscale)

        int m_total = Rcpp::as<int>(bs["m_total"]);
        Rcpp::List phi_per_arm = bs["phi"];
        Rcpp::IntegerVector n_obs_per_arm = bs["n_obs_per_arm"];
        Rcpp::NumericVector eigenvalues = bs["eigenvalues"];

        tulpa::LatentBlock block = tulpa::make_hsgp_block(
            latent_offset, m_total,
            phi_per_arm, n_obs_per_arm, n_arms, block_index,
            eigenvalues,
            /*axis_log_sigma2=*/axis0,
            /*axis_log_ell=*/axis0 + 1,
            theta_grid
        );
        blocks.push_back(block);
        return latent_offset + m_total;
    }

    if (type == "hsgp_mo") {
        // Multi-output (co-regionalization) HSGP block (Stage 1.7).
        // First ship: K == n_arms == 2, with axes
        //   (sigma_1, sigma_2, rho, ell)
        // all raw (no log transform). See src/hsgp_mo_block_factory.h
        // for the full design rationale.
        if (is_copy_block) {
            Rcpp::stop("Block %d: copy semantics are not supported for "
                       "hsgp_mo blocks (the K cross-output fields are the "
                       "shared latent; per-arm scaling lives in Sigma).",
                       block_index + 1);
        }
        require_axes(4);  // (sigma_1, sigma_2, rho, ell)
        if (n_arms != 2) {
            Rcpp::stop("Block %d (type 'hsgp_mo'): first ship requires "
                       "n_arms == 2 (got %d).",
                       block_index + 1, n_arms);
        }

        int m_total = Rcpp::as<int>(bs["m_total"]);
        Rcpp::List phi_per_arm = bs["phi"];
        Rcpp::IntegerVector n_obs_per_arm = bs["n_obs_per_arm"];
        Rcpp::NumericVector eigenvalues = bs["eigenvalues"];

        tulpa::LatentBlock block = tulpa::make_hsgp_mo_block(
            latent_offset, m_total, n_arms,
            phi_per_arm, n_obs_per_arm, block_index,
            eigenvalues,
            /*axis_sigma_1=*/axis0,
            /*axis_sigma_2=*/axis0 + 1,
            /*axis_rho=*/axis0 + 2,
            /*axis_ell=*/axis0 + 3,
            theta_grid
        );
        blocks.push_back(block);
        return latent_offset + n_arms * m_total;
    }

    if (type == "spde") {
        if (is_copy_block) {
            Rcpp::stop("Block %d: copy semantics for SPDE blocks are not "
                       "supported (first ship: spatial copy only on icar / "
                       "bym2 / car_proper).",
                       block_index + 1);
        }
        require_axes(2);  // (range, sigma)

        int n_mesh = Rcpp::as<int>(bs["n_mesh"]);
        Rcpp::List A_x_per_arm = bs["A_x"];
        Rcpp::List A_i_per_arm = bs["A_i"];
        Rcpp::List A_p_per_arm = bs["A_p"];
        Rcpp::IntegerVector n_obs_per_arm = bs["n_obs_per_arm"];
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

        tulpa::LatentBlock block = tulpa::make_spde_block(
            latent_offset, n_mesh,
            A_x_per_arm, A_i_per_arm, A_p_per_arm, n_obs_per_arm,
            n_arms, block_index,
            C0_diag, G1_x, G1_i, G1_p, nu,
            /*axis_range=*/axis0, /*axis_sigma=*/axis0 + 1, theta_grid,
            use_rational, rat_poles, rat_weights
        );
        blocks.push_back(block);
        return latent_offset + n_mesh;
    }

    if (type == "lf") {
        // Latent factor block (Stage 1.6a): u in R^n_latent shared across
        // arms, lambda in R^n_arms per-arm loadings, eta_i += u[obs_idx(i)] *
        // lambda[k_arm]. F = 1 only in this first ship.
        if (is_copy_block) {
            Rcpp::stop("Block %d: copy semantics are not supported for "
                       "latent factor blocks (the factor field IS the "
                       "shared latent; per-arm scaling lives in `lambda`).",
                       block_index + 1);
        }
        // No outer-grid axes for first-ship latent factors. sigma_u,
        // sigma_lambda, anchor_eps are R-side scalar fields on the block
        // spec; data informs (u, lambda) via the joint Newton solve.
        require_axes(0);

        int n_latent = Rcpp::as<int>(bs["n_latent"]);
        Rcpp::List obs_idx_list = bs["obs_idx"];
        auto obs_idx_fn = make_per_arm_idx_fn(obs_idx_list, n_arms,
                                                "obs_idx", block_index);

        double sigma_u      = bs.containsElementNamed("sigma_u")
                                  ? Rcpp::as<double>(bs["sigma_u"])      : 1.0;
        double sigma_lambda = bs.containsElementNamed("sigma_lambda")
                                  ? Rcpp::as<double>(bs["sigma_lambda"]) : 1.0;
        double anchor_eps   = bs.containsElementNamed("anchor_eps")
                                  ? Rcpp::as<double>(bs["anchor_eps"])   : 1e-3;

        tulpa::LatentBlock block = tulpa::make_latent_factor_block(
            latent_offset, n_latent, n_arms, obs_idx_fn,
            sigma_u, sigma_lambda, anchor_eps,
            block_index
        );
        blocks.push_back(block);
        return latent_offset + n_latent + n_arms;
    }

    Rcpp::stop("Unknown block type '%s' in cpp_nested_laplace_joint_multi",
               type.c_str());
    return latent_offset;
}

// Parse the optional phi_grid_per_arm Rcpp::List. Reuses the convention from
// the legacy joint kernels: list of length n_arms; entry k is either NULL
// (no override) or a NumericVector of length n_grid (per outer-grid phi).
std::vector<Rcpp::NumericVector> parse_phi_overrides_multi(
    Rcpp::Nullable<Rcpp::List> phi_grid_per_arm, int n_arms, int n_grid
) {
    std::vector<Rcpp::NumericVector> out(n_arms);
    if (phi_grid_per_arm.isNull()) return out;
    Rcpp::List phi_list(phi_grid_per_arm);
    if (static_cast<int>(phi_list.size()) != n_arms) {
        Rcpp::stop("phi_grid_per_arm must have length n_arms (%d).", n_arms);
    }
    for (int k = 0; k < n_arms; k++) {
        if (Rf_isNull(phi_list[k])) continue;
        Rcpp::NumericVector v = Rcpp::as<Rcpp::NumericVector>(phi_list[k]);
        if (v.size() == 0) continue;
        if (static_cast<int>(v.size()) != n_grid) {
            Rcpp::stop("phi_grid_per_arm[[%d]] must have length 0 or %d "
                       "(matching the flat outer-grid size).",
                       k + 1, n_grid);
        }
        out[k] = v;
    }
    return out;
}

inline void apply_phi_overrides_multi(
    std::vector<tulpa::JointArm>& arms,
    const std::vector<Rcpp::NumericVector>& phi_overrides,
    int k_grid
) {
    for (size_t k = 0; k < arms.size(); k++) {
        if (phi_overrides[k].size() > 0) {
            arms[k].phi = phi_overrides[k][k_grid];
        }
    }
}

} // namespace

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_joint_multi(
    Rcpp::List          arms_list,
    int                 copy_arm,         // -1 if no copy
    int                 copy_block,       // 0-based block index; -1 if no copy
    Rcpp::List          blocks_spec,
    Rcpp::NumericMatrix theta_grid,
    Rcpp::IntegerVector axis_offsets,
    int                 max_iter = 50,
    double              tol = 1e-6,
    int                 n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool                store_Q = false,
    Rcpp::Nullable<Rcpp::List> phi_grid_per_arm = R_NilValue,
    int                 n_threads_outer = 1,
    Rcpp::Nullable<Rcpp::IntegerVector> tile_ids = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> tile_pilot_cells = R_NilValue,
    double              prune_tol = 0.0,
    bool                force_sparse = false
) {
    int n_arms = arms_list.size();
    int B = blocks_spec.size();
    if (axis_offsets.size() != B + 1) {
        Rcpp::stop("axis_offsets must have length B+1 (got %d for B=%d)",
                   static_cast<int>(axis_offsets.size()), B);
    }
    int total_axes = axis_offsets[B];
    if (theta_grid.ncol() != total_axes) {
        Rcpp::stop("theta_grid must have %d columns (sum of axes), got %d",
                   total_axes, static_cast<int>(theta_grid.ncol()));
    }
    int n_grid = theta_grid.nrow();
    bool has_copy = (copy_arm >= 0);
    if (has_copy) {
        if (copy_arm >= n_arms) {
            Rcpp::stop("copy_arm index (%d) out of range for n_arms (%d).",
                       copy_arm, n_arms);
        }
        if (copy_block < 0 || copy_block >= B) {
            Rcpp::stop("copy_block index (%d) out of range for B (%d).",
                       copy_block, B);
        }
    }

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm>  arms;
    int n_x_after_re = tulpa::parse_joint_arms(arms_list, parsed, arms);
    std::vector<Rcpp::NumericVector> phi_overrides =
        parse_phi_overrides_multi(phi_grid_per_arm, n_arms, n_grid);

    std::vector<tulpa::LatentBlock> blocks;
    blocks.reserve(B + 2);
    int latent_offset = n_x_after_re;
    for (int b = 0; b < B; b++) {
        Rcpp::List bs = blocks_spec[b];
        int axis0 = axis_offsets[b];
        int axis_count = axis_offsets[b + 1] - axis0;
        latent_offset = build_joint_blocks_from_spec(
            bs, theta_grid, axis0, axis_count, latent_offset, n_arms, b,
            has_copy && b == copy_block, copy_arm, blocks
        );
    }

    auto prep = [&arms, &phi_overrides](int k_grid) {
        apply_phi_overrides_multi(arms, phi_overrides, k_grid);
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) {
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }

    std::vector<int> tile_ids_vec;
    if (tile_ids.isNotNull()) {
        Rcpp::IntegerVector iv(tile_ids);
        // Empty IntegerVector is interpreted as "no tiles" and falls back
        // to Phase 1 (single-tier) behaviour inside run_nested_laplace_grid.
        if (iv.size() > 0) {
            if (iv.size() != n_grid) {
                Rcpp::stop("tile_ids must have length n_grid (%d), got %d.",
                           n_grid, static_cast<int>(iv.size()));
            }
            tile_ids_vec.assign(iv.begin(), iv.end());
        }
    }
    std::vector<int> tile_pilot_cells_vec;
    if (tile_pilot_cells.isNotNull()) {
        Rcpp::IntegerVector iv(tile_pilot_cells);
        if (iv.size() > 0) {
            tile_pilot_cells_vec.assign(iv.begin(), iv.end());
            for (int k : tile_pilot_cells_vec) {
                if (k < 0 || k >= n_grid) {
                    Rcpp::stop("tile_pilot_cells entry %d out of range [0, %d).",
                               k, n_grid);
                }
            }
        }
    }

    Rcpp::List out = tulpa::run_multi_block_nested_laplace_joint(
        n_grid, arms, parsed, blocks, n_x_after_re,
        max_iter, tol, n_threads,
        /*store_modes=*/true, x_init,
        store_Q,
        prep,
        n_threads_outer,
        tile_ids_vec,
        tile_pilot_cells_vec,
        prune_tol,
        force_sparse
    );
    out["theta_grid"]   = theta_grid;
    out["axis_offsets"] = axis_offsets;
    return out;
}

// ==========================================================================
// Pattern correctness debug entry (1.5a).
// ==========================================================================
// Parses arms_list + blocks_spec the same way cpp_nested_laplace_joint_multi
// does, then stops after build_joint_hessian_pattern and returns the joint
// Hessian sparsity pattern as a CSC triple (i, p, dim). R-side tests build
// a Matrix::sparseMatrix from it and assert exact pattern equality against
// hand-computed references.
//
// No Newton iteration, no scatter, no factorization — pure pattern. Use
// theta_grid with a single dummy row when the block factory needs a
// theta_grid (axis_offsets must still cover the schema for each block
// type's prep callback to validate at parse time).

// [[Rcpp::export]]
Rcpp::List cpp_test_joint_pattern(
    Rcpp::List          arms_list,
    int                 copy_arm,
    int                 copy_block,
    Rcpp::List          blocks_spec,
    Rcpp::NumericMatrix theta_grid,
    Rcpp::IntegerVector axis_offsets
) {
    int n_arms = arms_list.size();
    int B = blocks_spec.size();
    if (axis_offsets.size() != B + 1) {
        Rcpp::stop("axis_offsets must have length B+1 (got %d for B=%d)",
                   static_cast<int>(axis_offsets.size()), B);
    }
    bool has_copy = (copy_arm >= 0);

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm>  arms;
    int n_x_after_re = tulpa::parse_joint_arms(arms_list, parsed, arms);

    std::vector<tulpa::LatentBlock> blocks;
    blocks.reserve(B);
    int latent_offset = n_x_after_re;
    for (int b = 0; b < B; b++) {
        Rcpp::List bs = blocks_spec[b];
        int axis0 = axis_offsets[b];
        int axis_count = axis_offsets[b + 1] - axis0;
        latent_offset = build_joint_blocks_from_spec(
            bs, theta_grid, axis0, axis_count, latent_offset, n_arms, b,
            has_copy && b == copy_block, copy_arm, blocks
        );
    }
    int n_x = latent_offset;

    tulpa::SparseHessianBuilder H_builder;
    tulpa::build_joint_hessian_pattern(parsed, arms, blocks, n_x, H_builder);

    // Return CSC triples. col_ptr has length n+1; row_idx has length nnz.
    // R-side converts to Matrix::sparseMatrix(i = row_idx + 1, p = col_ptr,
    // dims = c(n_x, n_x)) for a lower-triangle pattern matrix.
    Rcpp::IntegerVector col_ptr(H_builder.col_ptr.begin(),
                                 H_builder.col_ptr.end());
    Rcpp::IntegerVector row_idx(H_builder.row_idx.begin(),
                                 H_builder.row_idx.end());
    return Rcpp::List::create(
        Rcpp::Named("n_x")     = n_x,
        Rcpp::Named("nnz")     = H_builder.nnz,
        Rcpp::Named("col_ptr") = col_ptr,
        Rcpp::Named("row_idx") = row_idx
    );
}
