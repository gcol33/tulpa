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

#include "cell_coupling_registry.h"
#include "joint_hessian_pattern.h"
#include "laplace_core.h"
#include "laplace_re_priors.h"
#include "laplace_spatial_priors.h"
#include "laplace_temporal_priors.h"
#include "latent_block.h"
#include "nested_laplace_checkpoint.h"
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

// Build a per-arm per-row weight closure from an Rcpp::List of per-arm
// NumericVectors (areal SVC). Mirrors make_per_arm_idx_fn: caches the
// per-arm vectors so per-grid-point eta evaluation does not re-resolve the
// list. The R side validates lengths; this only checks the list shape.
// Returns an empty std::function when the spec carries no svc_weight, so the
// block's row_weight stays unset (uniform weight 1, no behavior change).
inline std::function<double(int, int)> make_per_arm_row_weight_fn(
    const Rcpp::List& bs, int n_arms, int block_index
) {
    if (!bs.containsElementNamed("svc_weight") ||
        Rf_isNull(bs["svc_weight"])) {
        return std::function<double(int, int)>();
    }
    Rcpp::List w_list = bs["svc_weight"];
    if (static_cast<int>(w_list.size()) != n_arms) {
        Rcpp::stop("blocks_spec[[%d]]$svc_weight must be a list of length "
                   "n_arms (%d), got %d.",
                   block_index + 1, n_arms,
                   static_cast<int>(w_list.size()));
    }
    std::vector<Rcpp::NumericVector> cache;
    cache.reserve(n_arms);
    for (int k = 0; k < n_arms; k++) {
        cache.push_back(Rcpp::as<Rcpp::NumericVector>(w_list[k]));
    }
    return [cache](int i, int k_arm) -> double {
        return cache[k_arm][i];
    };
}

// Per-arm amplitude dispatch for a copy block. axis_donor and axis_copy are
// column indices into theta_grid for this block's donor-arm and copy-arm
// amplitude axes respectively (set up by the R-side parser). Under the
// (sigma, alpha) reparameterization (gcol33/tulpa#22), the R side fills
// theta_grid[, axis_donor] = sigma and theta_grid[, axis_copy] = alpha*sigma
// before calling this kernel.
//
// The returned closure additionally multiplies by each arm's constant
// `field_coef` (parsed onto JointArm from the per-arm `field_coef_const`
// field) so per-arm constant multipliers compose cleanly on top of the
// hyperparam-driven copy axis. `arms_ptr` borrows the kernel's
// `std::vector<JointArm>`; the caller owns its lifetime for the duration of
// the outer-grid pass.
inline std::function<double(int, int)> make_copy_arm_scale_fn(
    int copy_arm, int axis_donor, int axis_copy,
    const Rcpp::NumericMatrix& theta_grid,
    const std::vector<tulpa::JointArm>* arms_ptr
) {
    return [copy_arm, axis_donor, axis_copy, theta_grid, arms_ptr](
        int k_arm, int k_grid) -> double {
        double base = (k_arm == copy_arm) ? theta_grid(k_grid, axis_copy)
                                          : theta_grid(k_grid, axis_donor);
        double fc = (arms_ptr ? (*arms_ptr)[k_arm].field_coef : 1.0);
        return base * fc;
    };
}

// Per-arm constant-multiplier arm_scale for the non-copy path. Used when
// some arm carries a `field_coef != 1` constant and no hyperparam axis is
// declared. The block's `d_fac` continues to carry sigma (or rolls it into
// the prior tau as before); this callback only injects the per-arm multiplier.
inline std::function<double(int, int)> make_field_coef_arm_scale_fn(
    const std::vector<tulpa::JointArm>* arms_ptr
) {
    return [arms_ptr](int k_arm, int /*k_grid*/) -> double {
        return arms_ptr ? (*arms_ptr)[k_arm].field_coef : 1.0;
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
    std::vector<tulpa::LatentBlock>& blocks,
    const std::vector<tulpa::JointArm>* arms_ptr,
    bool any_nontrivial_field_coef
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
        block.row_weight = make_per_arm_row_weight_fn(bs, n_arms, block_index);

        if (is_copy_block) {
            require_axes(2);  // (sigma_donor, sigma_copy)
            block.d_fac = [](int) -> double { return 1.0; };
            block.arm_scale = make_copy_arm_scale_fn(
                copy_arm, axis0, axis0 + 1, theta_grid, arms_ptr);
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
            // Non-copy ICAR: tau on the prior, no d_fac scaling. When any
            // arm carries a `field_coef != 1` constant, inject a per-arm
            // multiplier so sigma_arm = field_coef * sigma is honored. tau
            // already encodes sigma through the prior, so the field
            // amplitude that multiplies x is just field_coef.
            if (any_nontrivial_field_coef) {
                block.arm_scale = make_field_coef_arm_scale_fn(arms_ptr);
            }
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
        auto row_weight_fn = make_per_arm_row_weight_fn(bs, n_arms,
                                                        block_index);

        std::function<double(int, int)> arm_scale_fn;
        std::function<double(int)>      d_fac_phi_fn;
        std::function<double(int)>      d_fac_theta_fn;

        if (is_copy_block) {
            require_axes(3);  // (sigma_donor, sigma_copy, rho)
            arm_scale_fn = make_copy_arm_scale_fn(
                copy_arm, axis0, axis0 + 1, theta_grid, arms_ptr);
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
            // When any arm has a `field_coef != 1` constant, the non-copy
            // BYM2 block still needs a per-arm multiplier on top of d_fac.
            if (any_nontrivial_field_coef) {
                arm_scale_fn = make_field_coef_arm_scale_fn(arms_ptr);
            }
        }

        tulpa::LatentBlock phi_block;
        phi_block.start = phi_start;
        phi_block.size  = size;
        phi_block.idx   = idx_fn;
        phi_block.d_fac = d_fac_phi_fn;
        if (arm_scale_fn)  phi_block.arm_scale  = arm_scale_fn;
        if (row_weight_fn) phi_block.row_weight = row_weight_fn;
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
        theta_block.idx   = idx_fn;
        theta_block.d_fac = d_fac_theta_fn;
        if (arm_scale_fn)  theta_block.arm_scale  = arm_scale_fn;
        if (row_weight_fn) theta_block.row_weight = row_weight_fn;
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
        block.row_weight = make_per_arm_row_weight_fn(bs, n_arms, block_index);

        if (is_copy_block) {
            require_axes(3);  // (sigma_donor, sigma_copy, rho_car)
            block.d_fac = [](int) -> double { return 1.0; };
            block.arm_scale = make_copy_arm_scale_fn(
                copy_arm, axis0, axis0 + 1, theta_grid, arms_ptr);
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
            if (any_nontrivial_field_coef) {
                block.arm_scale = make_field_coef_arm_scale_fn(arms_ptr);
            }
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
    Rcpp::IntegerVector copy_arms,        // 0-based copy arm per copy block
    Rcpp::IntegerVector copy_blocks,      // 0-based copy block index, parallel
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
    bool                force_sparse = false,
    std::string         cell_coupling_name = "separable",
    int                 hessian_pd_mode = 0,
    int                 step_curvature_mode = 0,
    int                 inner_refresh = 1,
    bool                progress = false,
    int                 progress_every = 0,
    double              progress_throttle = 0.0,
    std::string         progress_file = "",
    std::string         checkpoint_path = ""
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

    // Resolve the copy specs into a per-block copy-arm map: entry b is the
    // 0-based copy arm coupled onto block b, or -1 when block b is not a
    // copy block. copy_arms / copy_blocks are parallel vectors; a single
    // `-1` (or empty) means "no copy" (the scalar-shim back-compat shape).
    std::vector<int> copy_arm_of_block(B, -1);
    if (copy_blocks.size() != copy_arms.size()) {
        Rcpp::stop("copy_arms (%d) and copy_blocks (%d) must have equal length.",
                   static_cast<int>(copy_arms.size()),
                   static_cast<int>(copy_blocks.size()));
    }
    for (int c = 0; c < copy_blocks.size(); c++) {
        int cb = copy_blocks[c];
        int ca = copy_arms[c];
        if (cb < 0) continue;  // no-copy sentinel
        if (cb >= B) {
            Rcpp::stop("copy_block index (%d) out of range for B (%d).", cb, B);
        }
        if (ca < 0 || ca >= n_arms) {
            Rcpp::stop("copy_arm index (%d) out of range for n_arms (%d).",
                       ca, n_arms);
        }
        if (copy_arm_of_block[cb] >= 0) {
            Rcpp::stop("block %d is marked as a copy block more than once.",
                       cb + 1);
        }
        copy_arm_of_block[cb] = ca;
    }

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm>  arms;
    int n_x_after_re = tulpa::parse_joint_arms(arms_list, parsed, arms);
    std::vector<Rcpp::NumericVector> phi_overrides =
        parse_phi_overrides_multi(phi_grid_per_arm, n_arms, n_grid);

    // Detect any per-arm constant `field_coef != 1` so the non-copy block
    // factories know whether to install the per-arm multiplier `arm_scale`
    // callback. The copy-block factories always install `arm_scale` (the
    // copy axis lives there); they pick up `arms[k].field_coef` regardless.
    bool any_nontrivial_field_coef = false;
    for (const auto& a : arms) {
        if (std::abs(a.field_coef - 1.0) > 0.0) {
            any_nontrivial_field_coef = true;
            break;
        }
    }

    std::vector<tulpa::LatentBlock> blocks;
    blocks.reserve(B + 2);
    int latent_offset = n_x_after_re;
    for (int b = 0; b < B; b++) {
        Rcpp::List bs = blocks_spec[b];
        int axis0 = axis_offsets[b];
        int axis_count = axis_offsets[b + 1] - axis0;
        bool is_copy_b = (copy_arm_of_block[b] >= 0);
        latent_offset = build_joint_blocks_from_spec(
            bs, theta_grid, axis0, axis_count, latent_offset, n_arms, b,
            is_copy_b, copy_arm_of_block[b], blocks,
            &arms, any_nontrivial_field_coef
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

    // Resolve the cell-coupling spec by name (Layer A registered the
    // separable default under "separable" so empty / default-named
    // calls succeed without consumer registration).
    std::shared_ptr<tulpa::CellCouplingSpec> cell_coupling_spec =
        tulpa::lookup_cell_coupling(cell_coupling_name);
    if (!cell_coupling_spec) {
        Rcpp::stop("cell_coupling = \"%s\" is not registered.",
                   cell_coupling_name.c_str());
    }

    tulpa::JointPDMode pd_mode =
        (hessian_pd_mode == 1) ? tulpa::JointPDMode::PSD : tulpa::JointPDMode::LM;

    // Inner Newton step curvature: Expected = complete-data Fisher (PSD by
    // construction) when control$hessian = "fisher"; otherwise the observed
    // mixture Hessian. The final mode-pass always uses Observed regardless.
    tulpa::CurvatureMode step_curvature =
        (step_curvature_mode == 1) ? tulpa::CurvatureMode::Expected
                                   : tulpa::CurvatureMode::Observed;

    std::unique_ptr<tulpa_progress::GridProgress> gp;
    if (progress) {
        gp.reset(new tulpa_progress::GridProgress(
            "nested-laplace-joint", n_grid, progress_every, progress_throttle,
            progress_file));
    }

    // Grid-cell checkpoint/resume (gcol33/tulpa#50). Built only when the R
    // front door supplied a path. The fingerprint folds in everything that
    // changes a cell's result given its hyperparameter coordinate -- the per-
    // arm responses + designs, the axis layout, and the solver settings -- so
    // a resume onto a checkpoint written for different data or control errors
    // rather than returning a stale result. The per-cell key is the raw bytes
    // of that cell's coordinate (its theta_grid row plus any per-arm phi-grid
    // value), so refinement-pass cells append under their own keys and a
    // resumed run hits every previously completed cell regardless of order.
    std::unique_ptr<tulpa::GridCheckpoint> ckpt;
    if (!checkpoint_path.empty()) {
        tulpa::Fingerprint fp;
        fp.fold_pod(max_iter);
        fp.fold_pod(tol);
        fp.fold_pod(hessian_pd_mode);
        fp.fold_pod(step_curvature_mode);
        fp.fold_pod(inner_refresh);
        fp.fold_pod(total_axes);
        if (axis_offsets.size() > 0)
            fp.fold(&axis_offsets[0], axis_offsets.size() * sizeof(int));
        fp.fold_str(cell_coupling_name);
        for (int k = 0; k < n_arms; k++) {
            const tulpa::ParsedArm& pa = parsed[k];
            const tulpa::JointArm&  a  = arms[k];
            fp.fold_pod(a.N);
            fp.fold_pod(pa.p);
            fp.fold_pod(pa.n_re_groups);
            fp.fold_str(a.family);
            if (a.y.size())        fp.fold(a.y.begin(),
                                           (std::size_t)a.y.size() * sizeof(double));
            if (pa.X.size())       fp.fold(pa.X.begin(),
                                           (std::size_t)pa.X.size() * sizeof(double));
            if (a.n_trials.size()) fp.fold(a.n_trials.begin(),
                                           (std::size_t)a.n_trials.size() * sizeof(int));
        }

        // Per-cell coordinate keys: theta_grid row k ++ per-arm phi at k. Each
        // theta_grid column is contiguous (column-major, nrow == n_grid) so
        // add_axis reads it with stride 1; an arm with no phi override adds
        // nothing.
        tulpa::CellKeyBuilder kb(n_grid);
        for (int j = 0; j < total_axes; j++)
            kb.add_axis(theta_grid.begin() + (std::size_t)j * n_grid);
        for (int a = 0; a < n_arms; a++)
            if (phi_overrides[a].size() > 0)
                kb.add_axis(phi_overrides[a].begin());
        ckpt.reset(new tulpa::GridCheckpoint(checkpoint_path, fp.value(),
                                             kb.take()));
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
        force_sparse,
        cell_coupling_spec,
        pd_mode,
        step_curvature,
        inner_refresh,
        gp.get(),
        ckpt.get()
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
    Rcpp::IntegerVector copy_arms,
    Rcpp::IntegerVector copy_blocks,
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
    if (copy_blocks.size() != copy_arms.size()) {
        Rcpp::stop("copy_arms (%d) and copy_blocks (%d) must have equal length.",
                   static_cast<int>(copy_arms.size()),
                   static_cast<int>(copy_blocks.size()));
    }
    std::vector<int> copy_arm_of_block(B, -1);
    for (int c = 0; c < copy_blocks.size(); c++) {
        int cb = copy_blocks[c];
        int ca = copy_arms[c];
        if (cb < 0) continue;  // no-copy sentinel
        if (cb >= B) {
            Rcpp::stop("copy_block index (%d) out of range for B (%d).", cb, B);
        }
        copy_arm_of_block[cb] = ca;
    }

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm>  arms;
    int n_x_after_re = tulpa::parse_joint_arms(arms_list, parsed, arms);

    bool any_nontrivial_field_coef = false;
    for (const auto& a : arms) {
        if (std::abs(a.field_coef - 1.0) > 0.0) {
            any_nontrivial_field_coef = true;
            break;
        }
    }

    std::vector<tulpa::LatentBlock> blocks;
    blocks.reserve(B);
    int latent_offset = n_x_after_re;
    for (int b = 0; b < B; b++) {
        Rcpp::List bs = blocks_spec[b];
        int axis0 = axis_offsets[b];
        int axis_count = axis_offsets[b + 1] - axis0;
        bool is_copy_b = (copy_arm_of_block[b] >= 0);
        latent_offset = build_joint_blocks_from_spec(
            bs, theta_grid, axis0, axis_count, latent_offset, n_arms, b,
            is_copy_b, copy_arm_of_block[b], blocks,
            &arms, any_nontrivial_field_coef
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

// ==========================================================================
// Joint log-posterior + gradient debug entry (FD gate).
// ==========================================================================
// Builds the same arms + blocks as cpp_nested_laplace_joint_multi but at a
// FIXED single grid point (row k_grid of theta_grid), evaluates the joint
// log-posterior at a supplied latent vector x, and returns its analytic
// gradient. The gradient is the dense per-obs/prior scatter's `grad`, which
// is exactly d/dx [ log_lik + log_prior_joint - 0.5*tau_beta*sum(beta^2) ];
// the returned scalar adds the matching beta-prior term so a central finite
// difference of `logpost` reproduces `grad` to ~1e-4.
//
// Dense path only (small n_x FD fixtures). No cell coupling. The areal SVC
// row_weight is exercised here so a per-row weighted ICAR block's gradient
// wrt the field latent z and wrt a beta can be FD-verified.

// [[Rcpp::export]]
Rcpp::List cpp_test_joint_logpost_grad(
    Rcpp::List          arms_list,
    Rcpp::IntegerVector copy_arms,
    Rcpp::IntegerVector copy_blocks,
    Rcpp::List          blocks_spec,
    Rcpp::NumericMatrix theta_grid,
    Rcpp::IntegerVector axis_offsets,
    Rcpp::NumericVector x,
    int                 k_grid = 0
) {
    int n_arms = arms_list.size();
    int B = blocks_spec.size();
    if (axis_offsets.size() != B + 1) {
        Rcpp::stop("axis_offsets must have length B+1 (got %d for B=%d)",
                   static_cast<int>(axis_offsets.size()), B);
    }
    if (copy_blocks.size() != copy_arms.size()) {
        Rcpp::stop("copy_arms (%d) and copy_blocks (%d) must have equal length.",
                   static_cast<int>(copy_arms.size()),
                   static_cast<int>(copy_blocks.size()));
    }
    if (k_grid < 0 || k_grid >= theta_grid.nrow()) {
        Rcpp::stop("k_grid (%d) out of range for theta_grid rows (%d).",
                   k_grid, static_cast<int>(theta_grid.nrow()));
    }

    std::vector<int> copy_arm_of_block(B, -1);
    for (int c = 0; c < copy_blocks.size(); c++) {
        int cb = copy_blocks[c];
        int ca = copy_arms[c];
        if (cb < 0) continue;
        if (cb >= B) Rcpp::stop("copy_block index (%d) out of range.", cb);
        copy_arm_of_block[cb] = ca;
    }

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm>  arms;
    int n_x_after_re = tulpa::parse_joint_arms(arms_list, parsed, arms);

    bool any_nontrivial_field_coef = false;
    for (const auto& a : arms) {
        if (std::abs(a.field_coef - 1.0) > 0.0) {
            any_nontrivial_field_coef = true;
            break;
        }
    }

    std::vector<tulpa::LatentBlock> blocks;
    blocks.reserve(B);
    int latent_offset = n_x_after_re;
    for (int b = 0; b < B; b++) {
        Rcpp::List bs = blocks_spec[b];
        int axis0 = axis_offsets[b];
        int axis_count = axis_offsets[b + 1] - axis0;
        bool is_copy_b = (copy_arm_of_block[b] >= 0);
        latent_offset = build_joint_blocks_from_spec(
            bs, theta_grid, axis0, axis_count, latent_offset, n_arms, b,
            is_copy_b, copy_arm_of_block[b], blocks,
            &arms, any_nontrivial_field_coef);
    }
    int n_x = latent_offset;
    if (x.size() != n_x) {
        Rcpp::stop("x has length %d but the joint latent dimension is %d.",
                   static_cast<int>(x.size()), n_x);
    }

    for (const auto& b : blocks) {
        if (b.prep && !b.prep(k_grid)) {
            Rcpp::stop("block prep reported infeasible at k_grid=%d.", k_grid);
        }
    }

    tulpa::JointArmSpecs specs = tulpa::build_joint_arm_specs(arms);

    // eta per arm at x.
    std::vector<Rcpp::NumericVector> etas;
    etas.reserve(n_arms);
    for (const auto& a : arms) etas.emplace_back(a.N, 0.0);

    const int Bn = static_cast<int>(blocks.size());
    for (int k_arm = 0; k_arm < n_arms; k_arm++) {
        const tulpa::ParsedArm& pa = parsed[k_arm];
        const int N_k    = arms[k_arm].N;
        const int p_k    = pa.p;
        const int n_re_k = pa.n_re_groups;
        const int bstart = pa.beta_start;
        const int rstart = pa.re_start;
        std::vector<double> d_eff(Bn);
        for (int b = 0; b < Bn; b++) {
            double s = blocks[b].arm_scale ? blocks[b].arm_scale(k_arm, k_grid)
                                           : 1.0;
            d_eff[b] = s * blocks[b].d_fac(k_grid);
        }
        for (int i = 0; i < N_k; i++) {
            double e = 0.0;
            for (int j = 0; j < p_k; j++) e += pa.X(i, j) * x[bstart + j];
            if (n_re_k > 0) {
                int g = static_cast<int>(pa.re_idx[i]) - 1;
                if (g >= 0 && g < n_re_k) e += x[rstart + g];
            }
            for (int b = 0; b < Bn; b++) {
                if (d_eff[b] == 0.0) continue;
                if (!blocks[b].idx) continue;
                int l = blocks[b].idx(i, k_arm);
                if (l > 0 && l <= blocks[b].size) {
                    e += d_eff[b] * tulpa::block_row_weight(blocks[b], i, k_arm)
                                  * x[blocks[b].start + l - 1];
                }
            }
            etas[k_arm][i] = e;
        }
    }

    // Analytic gradient: dense per-arm scatter + block priors + beta/RE priors.
    tulpa::DenseVec grad(n_x, 0.0);
    tulpa::DenseMat H(n_x, tulpa::DenseVec(n_x, 0.0));
    for (int k_arm = 0; k_arm < n_arms; k_arm++) {
        tulpa::scatter_arm_obs_joint_multi(
            x, etas[k_arm], parsed[k_arm], arms[k_arm],
            specs.views[k_arm], k_arm, blocks, k_grid, grad, H);
    }
    for (const auto& b : blocks) {
        if (b.add_prior) b.add_prior(grad, H, x, k_grid);
    }
    tulpa::add_per_arm_beta_re_priors(grad, H, x, parsed);

    // Scalar objective: log_lik + log_prior_joint + beta-prior term. The
    // beta-prior term mirrors add_per_arm_beta_re_priors' weak Gaussian
    // (tau_beta = 1e-4) so the FD of `logpost` matches `grad` on the betas.
    double log_lik = tulpa::compute_total_log_lik_joint(specs.views, etas, 1);
    double log_prior = tulpa::log_prior_per_arm_re(x, parsed);
    for (const auto& b : blocks) {
        if (b.log_prior) log_prior += b.log_prior(x, k_grid);
    }
    constexpr double tau_beta = 1e-4;
    double beta_prior = 0.0;
    for (const auto& pa : parsed) {
        for (int j = 0; j < pa.p; j++) {
            double v = x[pa.beta_start + j];
            beta_prior -= 0.5 * tau_beta * v * v;
        }
    }
    double logpost = log_lik + log_prior + beta_prior;

    Rcpp::NumericVector grad_out(n_x);
    for (int j = 0; j < n_x; j++) grad_out[j] = grad[j];

    return Rcpp::List::create(
        Rcpp::Named("logpost") = logpost,
        Rcpp::Named("grad")    = grad_out,
        Rcpp::Named("n_x")     = n_x
    );
}

// Outer-grid joint drivers (declared in nested_laplace_joint_multi.h).
Rcpp::List tulpa::run_multi_block_nested_laplace_joint(
    int                              n_grid,
    std::vector<JointArm>&           arms,
    const std::vector<ParsedArm>&    parsed,
    const std::vector<LatentBlock>&  blocks,
    int                              n_x_after_re,
    int                              max_iter,
    double                           tol,
    int                              n_threads,
    bool                             store_modes,
    const Rcpp::NumericVector&       x_init,
    bool                             store_Q,
    std::function<void(int)>         prep_at_grid,
    int                              n_threads_outer,
    const std::vector<int>&          tile_ids,
    const std::vector<int>&          tile_pilot_cells,
    double                           prune_tol,
    bool                             force_sparse,
    std::shared_ptr<CellCouplingSpec> cell_coupling_spec,
    JointPDMode                      pd_mode,
    CurvatureMode                    step_curvature,
    int                              hessian_refresh,
    tulpa_progress::GridProgress*    progress,
    GridCheckpoint*                  checkpoint) {
    const int n_arms = static_cast<int>(arms.size());
    if (static_cast<int>(parsed.size()) != n_arms) {
        Rcpp::stop("parsed and arms vectors must have the same length.");
    }

    // Cell-coupling setup (Layer B.1). Resolve the spec's coupled-arm
    // set, validate it agrees with the per-arm `coupled` flags, and
    // invert each coupled arm's `cell_obs_map` into per-cell row lists
    // once for the whole fit.
    std::vector<int> coupled_arms;
    if (cell_coupling_spec) coupled_arms = cell_coupling_spec->arm_ids();
    const bool any_coupling = !coupled_arms.empty();

    std::vector<bool> arm_is_coupled(n_arms, false);
    for (int k : coupled_arms) {
        if (k < 0 || k >= n_arms) {
            Rcpp::stop("cell_coupling: arm_ids() entry %d out of range "
                       "[0, %d).", k, n_arms);
        }
        if (!arms[k].coupled) {
            Rcpp::stop("cell_coupling: spec lists arm %d but the arm "
                       "has coupled = FALSE.", k + 1);
        }
        arm_is_coupled[k] = true;
    }
    for (int k = 0; k < n_arms; k++) {
        if (arms[k].coupled && !arm_is_coupled[k]) {
            Rcpp::stop("cell_coupling: arm %d has coupled = TRUE but the "
                       "registered spec's arm_ids() does not list it. "
                       "Register a spec whose arm_ids() includes this arm "
                       "(default \"separable\" does not couple any arm).",
                       k + 1);
        }
    }

    std::vector<std::vector<std::vector<int>>> cell_rows;
    int n_cells = build_cell_rows_from_arms(arms, coupled_arms, cell_rows);

    int n_x = n_x_after_re;
    for (const auto& b : blocks) {
        n_x = std::max(n_x, b.start + b.size);
    }

    // Sparse-path detection. Triggered by (a) user opt-in via force_sparse,
    // (b) n_x crossing the dense-Newton threshold, or (c) any block whose
    // contrib_kind is not INDEXED_SINGLE — the dense scatter
    // (scatter_arm_obs_joint_multi) only handles INDEXED_SINGLE via the
    // `idx` callback, so SPDE/HSGP/etc. require the sparse path even at
    // small n_x.
    bool needs_sparse = force_sparse || (n_x >= SPARSE_THRESHOLD);
    if (!needs_sparse) {
        for (const auto& b : blocks) {
            if (b.contrib_kind != BlockContribKind::INDEXED_SINGLE) {
                needs_sparse = true;
                break;
            }
        }
    }
    if (needs_sparse) {
        // First-ship: sparse path is serial outer-grid. Each per-thread
        // SparseHessianBuilder would replicate the pattern (a few × 10^7
        // entries at n_sites = 10^6); serializing is the simplest correct
        // shape. Parallel sparse is a follow-up to this stage.
        return run_multi_block_nested_laplace_joint_sparse_impl(
            n_grid, arms, parsed, blocks, n_x,
            max_iter, tol, n_threads,
            store_modes, x_init, store_Q,
            prep_at_grid, tile_ids, tile_pilot_cells, prune_tol,
            cell_coupling_spec, coupled_arms, cell_rows, n_cells, pd_mode,
            step_curvature, hessian_refresh, n_threads_outer, progress,
            checkpoint
        );
    }

    // Per-outer-thread Newton scratch + CHOLMOD solver pool. The solver pool
    // lives inside run_nested_laplace_grid (one per outer thread); scratch
    // pool lives here because it's joint-shaped (per-arm etas).
    int n_outer = std::max(1, n_threads_outer);
    std::vector<NewtonScratchJoint> scratch_pool(n_outer);
    for (auto& s : scratch_pool) s.allocate(n_x, arms);

    // Resolve each arm to a spec view ONCE for the whole grid: built-in family
    // arms materialize a builtin_family_spec + response, model-supplied arms
    // borrow their spec. Read-only after build, so it is shared safely across
    // the parallel outer-grid threads. The scatter and the joint log-lik read
    // every arm only through these views (dev_notes/plans/clean_migration.md Phase L / L4).
    JointArmSpecs specs = build_joint_arm_specs(arms);

    // Force inner OpenMP to single-thread when the outer grid is parallel —
    // see run_multi_block_nested_laplace for the rationale.
    int n_threads_inner_eff = (n_outer > 1) ? 1 : n_threads;

    // Inner implementation: takes max_iter as a parameter so the cheap-pass
    // path can call this with max_iter=1 for a one-Newton-step screen at the
    // pilot mode. The 3-arg `solve_at_theta` wrapper below threads the outer
    // `max_iter` through.
    auto solve_at_theta_impl = [&](int k_grid,
                                   const std::vector<double>& prev_mode,
                                   SparseCholeskySolver* shared_solver,
                                   int max_iter_use,
                                   NewtonScratchJoint* scratch_override
                                   = nullptr) -> LaplaceResult
    {
        if (prep_at_grid) prep_at_grid(k_grid);
        for (const auto& b : blocks) {
            if (b.prep && !b.prep(k_grid)) {
                LaplaceResult bad;
                bad.mode = (static_cast<int>(prev_mode.size()) == n_x)
                           ? prev_mode
                           : std::vector<double>(n_x, 0.0);
                bad.log_marginal = -std::numeric_limits<double>::infinity();
                bad.n_iter = 0;
                bad.converged = false;
                bad.log_det_Q = 0.0;
                return bad;
            }
        }

        // prep_at_grid may have rewritten arm.phi (phi_grid axis); refresh the
        // built-in responses so the spec sees the current dispersion.
        specs.sync_dispersion(arms);

        // Cache per-block (k_arm, k_grid) -> d_eff. Per-block d_fac(k_grid)
        // is evaluated once; per-arm scaling is re-evaluated inside the
        // per-arm loops because compute_eta is called from inside the
        // Newton step many times.
        const int B = static_cast<int>(blocks.size());
        std::vector<double> d_fac_cache(B);
        for (int b = 0; b < B; b++) {
            d_fac_cache[b] = blocks[b].d_fac(k_grid);
        }

        auto compute_eta_joint = [&](const Rcpp::NumericVector& x,
                                     std::vector<Rcpp::NumericVector>& etas) {
            for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                const ParsedArm& pa = parsed[k_arm];
                const int N_k    = arms[k_arm].N;
                const int p_k    = pa.p;
                const int n_re_k = pa.n_re_groups;
                const int bstart = pa.beta_start;
                const int rstart = pa.re_start;

                // Per-arm effective coefficients per block.
                std::vector<double> d_eff(B);
                for (int b = 0; b < B; b++) {
                    double s = blocks[b].arm_scale
                                ? blocks[b].arm_scale(k_arm, k_grid)
                                : 1.0;
                    d_eff[b] = s * d_fac_cache[b];
                }

                #ifdef _OPENMP
                #pragma omp parallel for schedule(static) \
                    num_threads(n_threads_inner_eff > 0 ? n_threads_inner_eff : 1) \
                    if(n_threads_inner_eff > 1)
                #endif
                for (int i = 0; i < N_k; i++) {
                    double e = 0.0;
                    for (int j = 0; j < p_k; j++) e += pa.X(i, j) * x[bstart + j];
                    if (n_re_k > 0) {
                        int g = static_cast<int>(pa.re_idx[i]) - 1;
                        if (g >= 0 && g < n_re_k) e += x[rstart + g];
                    }
                    for (int b = 0; b < B; b++) {
                        // field_coef = 0 arms (and rho = 0 BYM2 components,
                        // etc.) contribute nothing to eta; skip the index
                        // resolution entirely.
                        if (d_eff[b] == 0.0) continue;
                        int l = blocks[b].idx(i, k_arm);
                        if (l > 0 && l <= blocks[b].size) {
                            e += d_eff[b] * block_row_weight(blocks[b], i, k_arm)
                                          * x[blocks[b].start + l - 1];
                        }
                    }
                    etas[k_arm][i] = e;
                }
            }
        };

        // `finalize` selects the coupled-cell curvature: the inner Newton step
        // uses `step_curvature` (Expected = Fisher scoring when
        // control$hessian = "fisher"); the final mode-pass uses the observed
        // Hessian for log_det_Q and the SEs.
        auto scatter_joint = [&](const Rcpp::NumericVector& x,
                                 const std::vector<Rcpp::NumericVector>& etas,
                                 DenseVec& grad, DenseMat& H, bool finalize) {
            for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                // Cell-coupled arms route through the per-cell branch
                // below; skip their per-obs scatter.
                if (arm_is_coupled[k_arm]) continue;
                scatter_arm_obs_joint_multi(
                    x, etas[k_arm], parsed[k_arm], arms[k_arm],
                    specs.views[k_arm], k_arm,
                    blocks, k_grid, grad, H
                );
            }
            if (any_coupling) {
                const CurvatureMode cm =
                    finalize ? CurvatureMode::Observed : step_curvature;
                scatter_cell_coupling_dense_branch(
                    *cell_coupling_spec, coupled_arms, cell_rows, n_cells,
                    arms, parsed, etas, blocks, k_grid, grad, H, cm
                );
            }
            for (const auto& b : blocks) {
                if (b.add_prior) b.add_prior(grad, H, x, k_grid);
            }
            add_per_arm_beta_re_priors(grad, H, x, parsed);
        };

        auto center_joint = [&](Rcpp::NumericVector& x) {
            for (int b = 0; b < B; b++) {
                if (!blocks[b].center) continue;
                double c_b = blocks[b].center(x);
                if (std::abs(c_b) < 1e-15) continue;
                // Per-arm intercept compensation so eta is preserved when a
                // rank-deficient block is re-centered after a Newton step.
                // arm k's first beta column absorbs the constant
                // arm_scale_b(k_arm, k_grid) * d_fac_b(k_grid) * c_b that
                // the centerer removed from x[block]. See the BYM2 / ICAR
                // joint kernel centerers in nested_laplace_joint.cpp for
                // the load-bearing rationale.
                for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                    if (parsed[k_arm].p == 0) continue;
                    double s = blocks[b].arm_scale
                                ? blocks[b].arm_scale(k_arm, k_grid)
                                : 1.0;
                    x[parsed[k_arm].beta_start] += s * d_fac_cache[b] * c_b;
                }
            }
        };

        auto log_prior_joint = [&](const Rcpp::NumericVector& x,
                                    const std::vector<Rcpp::NumericVector>&)
            -> double {
            double lp = log_prior_per_arm_re(x, parsed);
            for (const auto& b : blocks) {
                if (b.log_prior) lp += b.log_prior(x, k_grid);
            }
            return lp;
        };

        int tid;
        #ifdef _OPENMP
        tid = omp_in_parallel() ? omp_get_thread_num() : 0;
        #else
        tid = 0;
        #endif

        NewtonScratchJoint& scratch = scratch_override
                                       ? *scratch_override
                                       : scratch_pool[tid];

        JointSpecLogLik joint_ll{&specs.views, n_threads_inner_eff};
        if (any_coupling) {
            joint_ll.skip_arm = &arm_is_coupled;
            joint_ll.cell_coupling_log_lik_fn =
                [&](const std::vector<Rcpp::NumericVector>& e) {
                    return eval_cell_coupling_log_lik(
                        *cell_coupling_spec, coupled_arms, cell_rows, n_cells,
                        arms, e
                    );
                };
        }
        return laplace_newton_solve_joint_ll(
            n_x,
            max_iter_use, tol,
            compute_eta_joint, scatter_joint, center_joint, log_prior_joint,
            joint_ll, scratch, prev_mode, shared_solver,
            store_Q
        );
    };

    // 3-arg adapter for run_nested_laplace_grid (which calls
    // solve_at_theta(k, prev_mode, solver) without knowing about the
    // max_iter parameter). Threads the outer `max_iter` through.
    auto solve_at_theta = [&](int k_grid,
                              const std::vector<double>& prev_mode,
                              SparseCholeskySolver* shared_solver) -> LaplaceResult
    {
        return solve_at_theta_impl(k_grid, prev_mode, shared_solver,
                                    max_iter, nullptr);
    };

    // Cheap-pass screening: a short inner Newton run warm-started from the
    // neighbour quasi-mode the driver chains across the lattice, returning
    // the quasi-mode and the Laplace log-marginal at it. The driver sweeps
    // cells in flat order and threads the previous screened cell's `.mode`
    // in as `warm`, so each cheap solve only corrects the residual drift
    // between adjacent lattice cells (much cheaper and far more
    // rank-faithful than a one-step screen from a single distant pilot).
    // The cheap pass runs serially after the pilot solve and before any
    // parallel region, so a dedicated thread-local solver + scratch keep it
    // isolated from the parallel fan-out's pool (the pool's entries are
    // reserved for the inner Newton on survivors).
    SparseCholeskySolver cheap_solver;
    NewtonScratchJoint cheap_scratch;
    cheap_scratch.allocate(n_x, arms);
    auto cheap_eval = [&](int k_grid,
                          const std::vector<double>& warm,
                          int n_steps) -> LaplaceResult {
        return solve_at_theta_impl(
            k_grid, warm, &cheap_solver, n_steps, &cheap_scratch);
    };

    return run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, store_modes, n_outer,
        tile_ids, tile_pilot_cells,
        cheap_eval, prune_tol, progress, checkpoint
    );
}

Rcpp::List tulpa::run_multi_block_nested_laplace_joint_sparse_impl(
    int                              n_grid,
    std::vector<JointArm>&           arms,
    const std::vector<ParsedArm>&    parsed,
    const std::vector<LatentBlock>&  blocks,
    int                              n_x,
    int                              max_iter,
    double                           tol,
    int                              n_threads,
    bool                             store_modes,
    const Rcpp::NumericVector&       x_init,
    bool                             store_Q,
    std::function<void(int)>         prep_at_grid,
    const std::vector<int>&          tile_ids,
    const std::vector<int>&          tile_pilot_cells,
    double                           prune_tol,
    std::shared_ptr<CellCouplingSpec> cell_coupling_spec,
    const std::vector<int>&          coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int                              n_cells,
    JointPDMode                      pd_mode,
    CurvatureMode                    step_curvature,
    int                              hessian_refresh,
    int                              n_threads_outer,
    tulpa_progress::GridProgress*    progress,
    GridCheckpoint*                  checkpoint
) {
    const int n_arms = static_cast<int>(arms.size());
    const int B      = static_cast<int>(blocks.size());

    const bool any_coupling = !coupled_arms.empty();
    std::vector<bool> arm_is_coupled(n_arms, false);
    for (int k : coupled_arms) {
        if (k >= 0 && k < n_arms) arm_is_coupled[k] = true;
    }

    // Outer-grid parallelism. Each full per-cell solve owns mutable state that
    // cannot be shared across concurrent cells -- the sparse Hessian builder,
    // the Newton scratch, the per-arm likelihood specs (whose built-in
    // dispersion the phi-grid axis rewrites per cell), the scatter index cache,
    // and the DENSE_BASIS scratch -- so each outer thread gets its own. The
    // cheap-screen sweep runs serially (before any parallel region), so its
    // resources stay a single set. n_outer == 1 reproduces the prior serial
    // path exactly (one pool slot, used by every cell).
    int n_outer = std::max(1, n_threads_outer);

    // Build the shared pattern into slot 0 first, size a memory guard from its
    // nnz, then clamp the thread count so the replicated builders stay bounded.
    // Each builder copies the pattern's row_idx + values + entry_map; at
    // n_sites ~ 10^6 a per-thread copy is large, so a very wide field falls back
    // to fewer outer threads rather than exhausting memory.
    std::vector<SparseHessianBuilder> H_builders(1);
    { TULPA_PROFILE_PHASE(PHASE_PATTERN_BUILD);
      build_joint_hessian_pattern(parsed, arms, blocks, n_x, H_builders[0], coupled_arms); }
    {
        const size_t nnz = H_builders[0].values.size();
        // row_idx(int) + values(double) + a std::map node (~48 B) per nonzero.
        const size_t per_builder = nnz * (sizeof(int) + sizeof(double) + 48) + 1;
        const size_t budget = static_cast<size_t>(2) * 1024 * 1024 * 1024; // 2 GB
        int max_builders =
            static_cast<int>(std::max<size_t>(1, budget / per_builder));
        if (n_outer > max_builders) n_outer = max_builders;
    }
    H_builders.resize(n_outer);
    for (int t = 1; t < n_outer; t++) {
        TULPA_PROFILE_PHASE(PHASE_PATTERN_BUILD);
        build_joint_hessian_pattern(parsed, arms, blocks, n_x,
                                    H_builders[t], coupled_arms);
    }

    std::vector<NewtonScratchJointSparse> scratches(n_outer);
    for (auto& s : scratches) s.allocate(n_x, arms);

    // Per-thread arm specs, built in place: JointArmSpecs is self-referential
    // (its views point at the owner's own storage and members), so it cannot be
    // moved into a pool slot -- see build_joint_arm_specs_into. Read-only in
    // structure; only the built-in dispersion is rewritten per cell, into the
    // owning thread's copy under a short critical (see below).
    std::vector<JointArmSpecs> specs_pool(n_outer);
    for (auto& sp : specs_pool) build_joint_arm_specs_into(arms, sp);

    // Per-thread scatter index cache (validity keyed on the owning builder's H
    // pointer) and DENSE_BASIS scratch.
    std::vector<ScatterIndexCache> idx_caches(n_outer);
    { TULPA_PROFILE_PHASE(PHASE_PATTERN_BUILD);
      for (int t = 0; t < n_outer; t++)
          build_scatter_index_cache(parsed, arms, blocks,
                                    H_builders[t], idx_caches[t]); }
    std::vector<std::vector<DenseBasisScratch>> db_buffers_pool(n_outer);
    for (auto& d : db_buffers_pool)
        d.assign(static_cast<size_t>(n_arms) * B, DenseBasisScratch{});

    // Cheap-pass dedicated scratch / solver / builder / specs / caches. The
    // cheap sweep is serial, so a single set suffices.
    SparseHessianBuilder cheap_builder;
    { TULPA_PROFILE_PHASE(PHASE_PATTERN_BUILD);
      build_joint_hessian_pattern(parsed, arms, blocks, n_x, cheap_builder, coupled_arms); }
    NewtonScratchJointSparse cheap_scratch;
    cheap_scratch.allocate(n_x, arms);
    SparseCholeskySolver cheap_solver;
    JointArmSpecs cheap_specs;
    build_joint_arm_specs_into(arms, cheap_specs);
    ScatterIndexCache idx_cache_cheap;
    { TULPA_PROFILE_PHASE(PHASE_PATTERN_BUILD);
      build_scatter_index_cache(parsed, arms, blocks, cheap_builder, idx_cache_cheap); }
    std::vector<DenseBasisScratch> db_buffers_cheap(
        static_cast<size_t>(n_arms) * B);

    // Inner OpenMP collapses to a single thread when the outer grid is parallel
    // (the dense driver does the same); nested parallelism would oversubscribe.
    const int n_threads_inner_eff = (n_outer > 1) ? 1 : n_threads;

    auto solve_at_theta_impl = [&](int k_grid,
                                   const std::vector<double>& prev_mode,
                                   SparseCholeskySolver* shared_solver,
                                   int max_iter_use,
                                   bool use_cheap_scratch) -> LaplaceResult
    {
        // Pick this outer thread's resource slot. The cheap-screen sweep is
        // serial and uses the single cheap_* set; the full solve uses the
        // per-thread pool indexed by the OpenMP thread id.
        int tid = 0;
        #ifdef _OPENMP
        if (!use_cheap_scratch) tid = omp_get_thread_num();
        #endif
        if (tid < 0 || tid >= n_outer) tid = 0;
        JointArmSpecs& specs_use =
            use_cheap_scratch ? cheap_specs : specs_pool[tid];

        // phi-grid axis: prep_at_grid rewrites the SHARED `arms` dispersion for
        // this cell, then sync copies it into THIS thread's specs. Both run
        // under one short critical so a concurrent cell's prep cannot clobber
        // the dispersion between our write and our snapshot; the expensive
        // Newton solve below then runs lock-free on the thread-local specs.
        // (prep_at_grid touches only arm.phi; the spatial block prep below is
        // independent of it and stays outside the critical, as in the dense
        // parallel driver.)
        {
            TULPA_PROFILE_PHASE(PHASE_PREP);
            #ifdef _OPENMP
            #pragma omp critical(nl_sparse_phi)
            #endif
            {
                if (prep_at_grid) prep_at_grid(k_grid);
                specs_use.sync_dispersion(arms);
            }
        }

        for (const auto& b : blocks) {
            TULPA_PROFILE_PHASE(PHASE_PREP);
            if (b.prep && !b.prep(k_grid)) {
                LaplaceResult bad;
                bad.mode = (static_cast<int>(prev_mode.size()) == n_x)
                           ? prev_mode
                           : std::vector<double>(n_x, 0.0);
                bad.log_marginal = -std::numeric_limits<double>::infinity();
                bad.n_iter = 0;
                bad.converged = false;
                bad.log_det_Q = 0.0;
                return bad;
            }
        }

        // Per-block-per-arm d_eff cache: d_fac(k_grid) * arm_scale(k_arm, k_grid).
        std::vector<std::vector<double>> d_eff(B, std::vector<double>(n_arms, 0.0));
        for (int b = 0; b < B; b++) {
            double dfac = blocks[b].d_fac ? blocks[b].d_fac(k_grid) : 1.0;
            for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                double s = blocks[b].arm_scale
                            ? blocks[b].arm_scale(k_arm, k_grid)
                            : 1.0;
                d_eff[b][k_arm] = s * dfac;
            }
        }

        // Per-call scratch for the inner scatter / eta dispatch.
        std::vector<double>              basis_scratch;
        std::vector<std::pair<int,double>> active_scratch;
        std::vector<std::pair<int,double>> multi_scratch;
        std::vector<DenseBasisActive>     active_db_scratch;
        // db_buffers is a reference into the lifted-out per-thread storage so
        // each (k_arm, b) cache survives across the outer-grid cells this
        // thread solves.
        std::vector<DenseBasisScratch>&   db_buffers =
            use_cheap_scratch ? db_buffers_cheap : db_buffers_pool[tid];

        auto compute_eta_joint = [&](const Rcpp::NumericVector& x,
                                     std::vector<Rcpp::NumericVector>& etas) {
            compute_eta_joint_sparse_dispatch(
                x, etas, arms, parsed, blocks, k_grid,
                d_eff, basis_scratch, multi_scratch);
        };

        SparseHessianBuilder& H_use =
            use_cheap_scratch ? cheap_builder : H_builders[tid];
        const ScatterIndexCache* idx_cache_use =
            use_cheap_scratch ? &idx_cache_cheap : &idx_caches[tid];

        // `finalize` selects the curvature for the coupled-cell scatter: the
        // inner Newton step uses `step_curvature` (Expected = Fisher scoring,
        // PSD by construction, when control$hessian = "fisher"), while the
        // final mode-pass always uses the observed Hessian so log_det_Q and the
        // SEs are the true curvature at the mode.
        auto scatter_joint_sparse = [&](const Rcpp::NumericVector& x,
                                         const std::vector<Rcpp::NumericVector>& etas,
                                         DenseVec& grad,
                                         SparseHessianBuilder& H,
                                         bool finalize,
                                         bool grad_only) {
            for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                if (arm_is_coupled[k_arm]) continue;
                scatter_arm_obs_joint_multi_sparse(
                    x, etas[k_arm], parsed[k_arm], arms[k_arm],
                    specs_use.views[k_arm], k_arm,
                    blocks, k_grid, grad, H,
                    active_scratch, basis_scratch, multi_scratch,
                    active_db_scratch, db_buffers,
                    idx_cache_use
                );
            }
            if (any_coupling) {
                // A grad-only step reuses a cached factor, so the coupled-cell
                // spec may skip its Hessian (digamma/trigamma) work; the
                // curvature mode is irrelevant on such a step.
                const CurvatureMode cm =
                    finalize ? CurvatureMode::Observed : step_curvature;
                scatter_cell_coupling_sparse_branch(
                    *cell_coupling_spec, coupled_arms, cell_rows, n_cells,
                    arms, parsed, etas, blocks, k_grid, grad, H, cm, grad_only
                );
            }
            for (const auto& b : blocks) {
                if (b.add_prior_sparse) b.add_prior_sparse(H, grad, x, k_grid);
            }
            add_per_arm_beta_re_priors_sparse(grad, H, x, parsed);
        };

        auto center_joint = [&](Rcpp::NumericVector& x) {
            for (int b = 0; b < B; b++) {
                if (!blocks[b].center) continue;
                double c_b = blocks[b].center(x);
                if (std::abs(c_b) < 1e-15) continue;
                double dfac = blocks[b].d_fac ? blocks[b].d_fac(k_grid) : 1.0;
                for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                    if (parsed[k_arm].p == 0) continue;
                    double s = blocks[b].arm_scale
                                ? blocks[b].arm_scale(k_arm, k_grid)
                                : 1.0;
                    x[parsed[k_arm].beta_start] += s * dfac * c_b;
                }
            }
        };

        auto log_prior_joint = [&](const Rcpp::NumericVector& x,
                                    const std::vector<Rcpp::NumericVector>&)
            -> double {
            double lp = log_prior_per_arm_re(x, parsed);
            for (const auto& b : blocks) {
                if (b.log_prior) lp += b.log_prior(x, k_grid);
            }
            return lp;
        };

        NewtonScratchJointSparse& sc =
            use_cheap_scratch ? cheap_scratch : scratches[tid];
        JointSpecLogLik joint_ll{&specs_use.views, n_threads_inner_eff};
        if (any_coupling) {
            joint_ll.skip_arm = &arm_is_coupled;
            joint_ll.cell_coupling_log_lik_fn =
                [&](const std::vector<Rcpp::NumericVector>& e) {
                    return eval_cell_coupling_log_lik(
                        *cell_coupling_spec, coupled_arms, cell_rows, n_cells,
                        arms, e
                    );
                };
        }
        // The cheap-screen pass always re-factorizes (refresh = 1) so the
        // pruning ranking stays faithful to a full Newton screen; factor reuse
        // applies only to the full per-cell solve.
        const int refresh_use = use_cheap_scratch ? 1 : hessian_refresh;
        return laplace_newton_solve_joint_sparse_ll(
            n_x,
            max_iter_use, tol,
            compute_eta_joint, scatter_joint_sparse,
            center_joint, log_prior_joint,
            joint_ll, H_use, sc, prev_mode, shared_solver, store_Q, pd_mode,
            refresh_use
        );
    };

    auto solve_at_theta = [&](int k_grid,
                              const std::vector<double>& prev_mode,
                              SparseCholeskySolver* shared_solver) -> LaplaceResult
    {
        return solve_at_theta_impl(k_grid, prev_mode, shared_solver,
                                    max_iter, /*use_cheap_scratch=*/false);
    };

    auto cheap_eval = [&](int k_grid,
                          const std::vector<double>& warm,
                          int n_steps) -> LaplaceResult {
        return solve_at_theta_impl(
            k_grid, warm, &cheap_solver,
            n_steps, /*use_cheap_scratch=*/true);
    };

    return run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, store_modes,
        /*n_outer=*/n_outer,
        tile_ids, tile_pilot_cells,
        cheap_eval, prune_tol, progress, checkpoint
    );
}
