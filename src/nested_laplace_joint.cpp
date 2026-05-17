// nested_laplace_joint.cpp
// Joint multi-likelihood nested-Laplace backends. One [[Rcpp::export]] per
// spatial prior: BYM2, ICAR, CAR_proper.
//
// All three backends share one inner Newton driver
// (run_multi_block_nested_laplace_joint, declared in
// nested_laplace_joint_multi.h). Each backend's body is just a length-1
// (ICAR / CAR_proper) or length-2 (BYM2) std::vector<LatentBlock>
// describing the spatial prior and per-arm sigma scaling:
//
//   backend     blocks  outer-grid axes                       spatial prior
//   ---------   ------  ----------------------------------    ----------------------
//   BYM2        2       (sigma_occ, rho [, sigma_pos])        ICAR(tau=1) phi
//                                                             + iid N(0,1) theta
//   ICAR        1       (sigma_occ       [, sigma_pos])       ICAR(tau=1) phi
//   CAR_proper  1       (sigma_occ, rho_car [, sigma_pos])    D - rho_car * W (tau=1)
//
// Each backend uses a unit-precision latent prior (no tau in the latent
// quad form). The per-arm field amplitude enters via LatentBlock::arm_scale:
// `eta_arm = X beta + arm_scale(k_arm, k_grid) * d_fac(k_grid) * x[idx]`,
// where arm_scale returns sigma_occ for donor arms and sigma_pos for the
// copy arm. This breaks the (sigma, alpha) identifiability ridge — see
// gcol33/tulpa#18 — because each arm's likelihood anchors its own sigma
// directly instead of sharing one sigma scaled by alpha. alpha =
// sigma_pos / sigma_occ is recovered post-hoc on the R side.
//
// Each kernel composes:
//   parse_joint_arms()                       -> (parsed, arms, n_x_after_re)
//   build per-block LatentBlock(s)           -> idx / d_fac / arm_scale /
//                                               add_prior / log_prior /
//                                               prep / center
//   run_multi_block_nested_laplace_joint()   -> outer grid + inner Newton
//
// See dev_notes/joint_nested_laplace.md for the math,
// dev_notes/plan_multi_block_joint.md for the refactor history.

#include "laplace_core.h"
#include "laplace_newton_joint.h"
#include "laplace_re_priors.h"
#include "laplace_spatial_priors.h"
#include "laplace_family_link.h"
#include "latent_block.h"
#include "nested_laplace_grid.h"
#include "nested_laplace_joint_core.h"
#include "nested_laplace_joint_multi.h"
#include "hmc_car_proper.h"  // tulpa_car_proper::compute_car_precision, car_log_det
#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// Shared input validation for all three joint backends. Returns has_copy.
bool validate_joint_inputs(int n_arms, int copy_arm,
                           int n_grid_primary,
                           const Rcpp::NumericVector& sigma_pos_grid,
                           const char* primary_name) {
    bool has_copy = (copy_arm >= 0);
    if (has_copy) {
        if (copy_arm >= n_arms) {
            Rcpp::stop("copy_arm index out of range.");
        }
        if (sigma_pos_grid.size() != n_grid_primary) {
            Rcpp::stop("sigma_pos_grid must have the same length as %s "
                       "when copy_arm >= 0.", primary_name);
        }
    }
    return has_copy;
}

// Parse the optional `phi_grid_per_arm` Rcpp::List into a per-arm vector of
// per-grid-point phi overrides. Empty entry => no override for that arm
// (kernel uses arms_list[k]$phi for every grid point). Non-empty entry must
// have length n_grid: phi_overrides[k][k_grid] is the dispersion for arm k
// at outer-grid index k_grid.
std::vector<Rcpp::NumericVector> parse_phi_overrides(
    Rcpp::Nullable<Rcpp::List> phi_grid_per_arm, int n_arms, int n_grid
) {
    std::vector<Rcpp::NumericVector> out(n_arms);
    if (phi_grid_per_arm.isNull()) return out;
    Rcpp::List phi_list(phi_grid_per_arm);
    if ((int)phi_list.size() != n_arms) {
        Rcpp::stop("phi_grid_per_arm must have length n_arms (%d).", n_arms);
    }
    for (int k = 0; k < n_arms; k++) {
        if (Rf_isNull(phi_list[k])) continue;
        Rcpp::NumericVector v = Rcpp::as<Rcpp::NumericVector>(phi_list[k]);
        if (v.size() == 0) continue;
        if ((int)v.size() != n_grid) {
            Rcpp::stop("phi_grid_per_arm[[%d]] must have length 0 or %d "
                       "(matching the flat outer-grid size).",
                       k + 1, n_grid);
        }
        out[k] = v;
    }
    return out;
}

// Apply per-arm phi overrides for the current outer-grid point. Called at
// the top of each backend's solve_at_theta lambda. Mutates arms[k].phi for
// any arm with an override; arms without overrides keep their parse-time
// value.
inline void apply_phi_overrides(
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

// =====================================================================
// Joint nested Laplace — BYM2 (outer: sigma_occ, rho [, sigma_pos])
// =====================================================================
// Latent layout:
//   x = [ beta_1 | ... | beta_K | re_1 | ... | re_K | phi (n_s) | theta (n_s) ]
//
// Per arm k', obs i:
//   eta_i = X_k' beta_k' + re_k'[g(i)] + sigma_arm_k' * z_s
//   z_s   = sqrt(rho_k) * sf * phi_s + sqrt(1 - rho_k) * theta_s
//   sigma_arm_k' = sigma_pos_k if k' == copy_arm else sigma_occ_k
//
// `copy_arm` is 0-based; pass -1 to disable copy scaling (sigma_pos_grid then
// ignored — all arms use sigma_occ_grid).

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_joint_bym2(
    Rcpp::List arms_list,
    int copy_arm,
    int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double scale_factor,
    Rcpp::NumericVector sigma_occ_grid,
    Rcpp::NumericVector rho_grid,
    Rcpp::NumericVector sigma_pos_grid,
    int max_iter = 50,
    double tol = 1e-6,
    int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    Rcpp::Nullable<Rcpp::List> phi_grid_per_arm = R_NilValue
) {
    int n_arms = arms_list.size();
    int n_grid = sigma_occ_grid.size();
    if (rho_grid.size() != n_grid) {
        Rcpp::stop("sigma_occ_grid and rho_grid must have the same length.");
    }
    bool has_copy = validate_joint_inputs(n_arms, copy_arm, n_grid, sigma_pos_grid,
                                          "sigma_occ_grid");

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm>  arms;
    int n_x_after_re = tulpa::parse_joint_arms(arms_list, parsed, arms);
    std::vector<Rcpp::NumericVector> phi_overrides =
        parse_phi_overrides(phi_grid_per_arm, n_arms, n_grid);

    int phi_start   = n_x_after_re;
    int theta_start = phi_start + n_spatial_units;

    // Length-2 BYM2 block expansion:
    //   phi    : ICAR(tau=1), d_fac = sqrt(rho_k) * scale_factor, centered
    //   theta  : N(0, 1)      d_fac = sqrt(1 - rho_k),            uncentered
    // Both blocks share the same spatial index map and arm_scale. The
    // per-arm sigma is carried by arm_scale; the BYM2 phi/theta mixing
    // is carried by d_fac. When the multi-block scatter loops over pairs
    // (a, b) of active blocks at obs i, it produces the same intra- and
    // inter-block Hessian contributions the legacy n_sub=2 kernel inlined.
    auto arm_scale_fn = [has_copy, copy_arm, sigma_occ_grid, sigma_pos_grid](
        int k_arm, int k_grid) -> double {
        if (has_copy && k_arm == copy_arm) return sigma_pos_grid[k_grid];
        return sigma_occ_grid[k_grid];
    };

    tulpa::LatentBlock phi_block;
    phi_block.start = phi_start;
    phi_block.size  = n_spatial_units;
    phi_block.idx   = [&parsed](int i, int k_arm) -> int {
        return parsed[k_arm].spatial_idx[i];
    };
    phi_block.d_fac = [rho_grid, scale_factor](int k_grid) -> double {
        double rho_k = rho_grid[k_grid];
        return std::sqrt(rho_k + 1e-10) * scale_factor;
    };
    phi_block.arm_scale = arm_scale_fn;
    phi_block.add_prior = [phi_start, n_spatial_units,
                            adj_row_ptr, adj_col_idx, n_neighbors](
        tulpa::DenseVec& grad, tulpa::DenseMat& H,
        const Rcpp::NumericVector& x, int /*k*/) {
        tulpa::add_icar_prior(grad, H, x, phi_start, n_spatial_units,
                               /*tau=*/1.0,
                               adj_row_ptr, adj_col_idx, n_neighbors);
    };
    phi_block.log_prior = [phi_start, n_spatial_units,
                            adj_row_ptr, adj_col_idx, n_neighbors](
        const Rcpp::NumericVector& x, int /*k*/) -> double {
        // ICAR(tau = 1) quad form, same as the legacy BYM2 kernel.
        double quad_form = 0.0;
        for (int s = 0; s < n_spatial_units; s++) {
            double phi_s = x[phi_start + s];
            quad_form += n_neighbors[s] * phi_s * phi_s;
            for (int kk = adj_row_ptr[s]; kk < adj_row_ptr[s + 1]; kk++) {
                int neighbor = adj_col_idx[kk];
                if (neighbor > s) {
                    quad_form -= 2.0 * phi_s * x[phi_start + neighbor];
                }
            }
        }
        return -0.5 * quad_form;
    };
    phi_block.center = [phi_start, n_spatial_units](Rcpp::NumericVector& x)
        -> double {
        return tulpa::center_effects(x, phi_start, n_spatial_units);
    };

    tulpa::LatentBlock theta_block;
    theta_block.start = theta_start;
    theta_block.size  = n_spatial_units;
    theta_block.idx   = [&parsed](int i, int k_arm) -> int {
        return parsed[k_arm].spatial_idx[i];
    };
    theta_block.d_fac = [rho_grid](int k_grid) -> double {
        double rho_k = rho_grid[k_grid];
        return std::sqrt(1.0 - rho_k + 1e-10);
    };
    theta_block.arm_scale = arm_scale_fn;
    theta_block.add_prior = [theta_start, n_spatial_units](
        tulpa::DenseVec& grad, tulpa::DenseMat& H,
        const Rcpp::NumericVector& x, int /*k*/) {
        for (int s = 0; s < n_spatial_units; s++) {
            int idx = theta_start + s;
            grad[idx] -= x[idx];
            H[idx][idx] += 1.0;
        }
    };
    theta_block.log_prior = [theta_start, n_spatial_units](
        const Rcpp::NumericVector& x, int /*k*/) -> double {
        double lp = 0.0;
        for (int s = 0; s < n_spatial_units; s++) {
            double v = x[theta_start + s];
            lp -= 0.5 * v * v;
        }
        lp -= 0.5 * n_spatial_units * std::log(2.0 * M_PI);
        return lp;
    };
    // theta has no centering: prior is symmetric so any mean shift is
    // absorbed by phi's centering + per-arm intercept compensation.

    std::vector<tulpa::LatentBlock> blocks{ phi_block, theta_block };

    auto prep = [&arms, &phi_overrides](int k_grid) {
        apply_phi_overrides(arms, phi_overrides, k_grid);
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) {
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }

    Rcpp::List out = tulpa::run_multi_block_nested_laplace_joint(
        n_grid, arms, parsed, blocks, n_x_after_re,
        max_iter, tol, n_threads,
        /*store_modes=*/true, x_init,
        store_Q,
        prep
    );
    out["sigma_occ_grid"] = sigma_occ_grid;
    out["rho_grid"]       = rho_grid;
    if (has_copy) out["sigma_pos_grid"] = sigma_pos_grid;
    return out;
}

// =====================================================================
// Joint nested Laplace — ICAR (outer: sigma_occ [, sigma_pos])
// =====================================================================
// Latent layout:
//   x = [ beta_1 | ... | beta_K | re_1 | ... | re_K | phi (n_s) ]
//
// Per arm k', obs i:
//   eta_i = X_k' beta_k' + re_k'[g(i)] + sigma_arm_k' * phi_s
//   sigma_arm_k' = sigma_pos_k if k' == copy_arm else sigma_occ_k
//
// Spatial prior: phi ~ N(0, Q_struct^{-1}) with unit precision (tau = 1).
// Field amplitude is carried by the per-arm sigma multiplier on eta. This is
// a relabeling of the old (tau, alpha) parameterization (sigma_occ =
// 1/sqrt(tau), sigma_pos = alpha/sqrt(tau)) that decouples the two arms'
// likelihoods along their own sigma axes — see gcol33/tulpa#18.

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_joint_icar(
    Rcpp::List arms_list,
    int copy_arm,
    int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::NumericVector sigma_occ_grid,
    Rcpp::NumericVector sigma_pos_grid,
    int max_iter = 50,
    double tol = 1e-6,
    int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    Rcpp::Nullable<Rcpp::List> phi_grid_per_arm = R_NilValue
) {
    int n_arms = arms_list.size();
    int n_grid = sigma_occ_grid.size();
    bool has_copy = validate_joint_inputs(n_arms, copy_arm, n_grid, sigma_pos_grid,
                                          "sigma_occ_grid");

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm>  arms;
    int n_x_after_re = tulpa::parse_joint_arms(arms_list, parsed, arms);
    std::vector<Rcpp::NumericVector> phi_overrides =
        parse_phi_overrides(phi_grid_per_arm, n_arms, n_grid);

    int phi_start = n_x_after_re;

    // Length-1 ICAR block at unit precision. arm_scale carries the per-arm
    // sigma multiplier — donor arms see sigma_occ_grid, the copy arm sees
    // sigma_pos_grid. d_fac = 1 because the latent prior is unit-precision;
    // field amplitude is entirely in arm_scale.
    tulpa::LatentBlock spatial_block;
    spatial_block.start = phi_start;
    spatial_block.size  = n_spatial_units;
    spatial_block.idx   = [&parsed](int i, int k_arm) -> int {
        return parsed[k_arm].spatial_idx[i];
    };
    spatial_block.d_fac = [](int) -> double { return 1.0; };
    if (has_copy) {
        spatial_block.arm_scale = [copy_arm, sigma_occ_grid, sigma_pos_grid](
            int k_arm, int k_grid) -> double {
            return (k_arm == copy_arm) ? sigma_pos_grid[k_grid]
                                       : sigma_occ_grid[k_grid];
        };
    } else {
        spatial_block.arm_scale = [sigma_occ_grid](int /*k_arm*/, int k_grid)
            -> double { return sigma_occ_grid[k_grid]; };
    }
    spatial_block.add_prior = [phi_start, n_spatial_units,
                               adj_row_ptr, adj_col_idx, n_neighbors](
        tulpa::DenseVec& grad, tulpa::DenseMat& H,
        const Rcpp::NumericVector& x, int /*k*/) {
        tulpa::add_icar_prior(grad, H, x, phi_start, n_spatial_units,
                               /*tau=*/1.0,
                               adj_row_ptr, adj_col_idx, n_neighbors);
    };
    spatial_block.log_prior = [phi_start, n_spatial_units,
                                adj_row_ptr, adj_col_idx, n_neighbors](
        const Rcpp::NumericVector& x, int /*k*/) -> double {
        return tulpa::log_prior_icar(x, phi_start, n_spatial_units,
                                      /*tau=*/1.0,
                                      adj_row_ptr, adj_col_idx, n_neighbors);
    };
    spatial_block.center = [phi_start, n_spatial_units](Rcpp::NumericVector& x)
        -> double {
        return tulpa::center_effects(x, phi_start, n_spatial_units);
    };

    std::vector<tulpa::LatentBlock> blocks{ spatial_block };

    auto prep = [&arms, &phi_overrides](int k_grid) {
        apply_phi_overrides(arms, phi_overrides, k_grid);
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) {
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }

    Rcpp::List out = tulpa::run_multi_block_nested_laplace_joint(
        n_grid, arms, parsed, blocks, n_x_after_re,
        max_iter, tol, n_threads,
        /*store_modes=*/true, x_init,
        store_Q,
        prep
    );
    out["sigma_occ_grid"] = sigma_occ_grid;
    if (has_copy) out["sigma_pos_grid"] = sigma_pos_grid;
    return out;
}

// =====================================================================
// Joint nested Laplace — proper CAR (outer: sigma_occ, rho_car [, sigma_pos])
// =====================================================================
// Latent layout:
//   x = [ beta_1 | ... | beta_K | re_1 | ... | re_K | phi (n_s) ]
//
// Per arm k', obs i:
//   eta_i = X_k' beta_k' + re_k'[g(i)] + sigma_arm_k' * phi_s
//   sigma_arm_k' = sigma_pos_k if k' == copy_arm else sigma_occ_k
//
// Spatial prior: phi ~ N(0, [D - rho_car_k * W]^{-1}) with unit precision
// scaling (tau = 1). Field amplitude is carried by the per-arm sigma
// multiplier on eta. log|Q(rho_car)| is precomputed once per grid point via
// the dense Cholesky in tulpa_car_proper::car_log_det (n_s typically < 500
// so O(n^3) is fine; swap in a sparse Cholesky here if it becomes the
// bottleneck).

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_joint_car_proper(
    Rcpp::List arms_list,
    int copy_arm,
    int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::NumericVector sigma_occ_grid,
    Rcpp::NumericVector rho_car_grid,
    Rcpp::NumericVector sigma_pos_grid,
    int max_iter = 50,
    double tol = 1e-6,
    int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    Rcpp::Nullable<Rcpp::List> phi_grid_per_arm = R_NilValue
) {
    int n_arms = arms_list.size();
    int n_grid = sigma_occ_grid.size();
    if (rho_car_grid.size() != n_grid) {
        Rcpp::stop("sigma_occ_grid and rho_car_grid must have the same length.");
    }
    bool has_copy = validate_joint_inputs(n_arms, copy_arm, n_grid, sigma_pos_grid,
                                          "sigma_occ_grid");

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm>  arms;
    int n_x_after_re = tulpa::parse_joint_arms(arms_list, parsed, arms);
    std::vector<Rcpp::NumericVector> phi_overrides =
        parse_phi_overrides(phi_grid_per_arm, n_arms, n_grid);

    int phi_start = n_x_after_re;

    // Adjacency CSR copied into std::vector for the dense log|Q(rho)| helper.
    auto adj_rp_v = std::make_shared<std::vector<int>>(adj_row_ptr.begin(), adj_row_ptr.end());
    auto adj_ci_v = std::make_shared<std::vector<int>>(adj_col_idx.begin(), adj_col_idx.end());
    auto n_nbr_v  = std::make_shared<std::vector<int>>(n_neighbors.begin(),  n_neighbors.end());
    // Shared cache for log|Q(rho_car_k)| -- written by block.prep,
    // read by block.log_prior on the same grid point.
    auto log_det_Q_rho = std::make_shared<double>(0.0);

    tulpa::LatentBlock spatial_block;
    spatial_block.start = phi_start;
    spatial_block.size  = n_spatial_units;
    spatial_block.idx   = [&parsed](int i, int k_arm) -> int {
        return parsed[k_arm].spatial_idx[i];
    };
    spatial_block.d_fac = [](int) -> double { return 1.0; };
    if (has_copy) {
        spatial_block.arm_scale = [copy_arm, sigma_occ_grid, sigma_pos_grid](
            int k_arm, int k_grid) -> double {
            return (k_arm == copy_arm) ? sigma_pos_grid[k_grid]
                                       : sigma_occ_grid[k_grid];
        };
    } else {
        spatial_block.arm_scale = [sigma_occ_grid](int /*k_arm*/, int k_grid)
            -> double { return sigma_occ_grid[k_grid]; };
    }
    spatial_block.prep = [n_spatial_units, rho_car_grid,
                          adj_rp_v, adj_ci_v, n_nbr_v, log_det_Q_rho](int k_grid)
        -> bool {
        double rho_k = rho_car_grid[k_grid];
        std::vector<double> Qmat = tulpa_car_proper::compute_car_precision(
            n_spatial_units, *adj_rp_v, *adj_ci_v, *n_nbr_v, rho_k);
        *log_det_Q_rho = tulpa_car_proper::car_log_det(n_spatial_units, Qmat);
        return std::isfinite(*log_det_Q_rho);
    };
    spatial_block.add_prior = [phi_start, n_spatial_units, rho_car_grid,
                                adj_row_ptr, adj_col_idx, n_neighbors](
        tulpa::DenseVec& grad, tulpa::DenseMat& H,
        const Rcpp::NumericVector& x, int k_grid) {
        tulpa::add_car_proper_prior(grad, H, x, phi_start, n_spatial_units,
                                     /*tau=*/1.0, rho_car_grid[k_grid],
                                     adj_row_ptr, adj_col_idx, n_neighbors);
    };
    spatial_block.log_prior = [phi_start, n_spatial_units, rho_car_grid,
                                adj_row_ptr, adj_col_idx, n_neighbors,
                                log_det_Q_rho](
        const Rcpp::NumericVector& x, int k_grid) -> double {
        return tulpa::log_prior_car_proper(x, phi_start, n_spatial_units,
                                            /*tau=*/1.0, rho_car_grid[k_grid],
                                            *log_det_Q_rho,
                                            adj_row_ptr, adj_col_idx,
                                            n_neighbors);
    };
    // Proper CAR has full-rank Q; centering would change the prior quadratic
    // form. No center callback (preserves log_marginal exactly).

    std::vector<tulpa::LatentBlock> blocks{ spatial_block };

    auto prep = [&arms, &phi_overrides](int k_grid) {
        apply_phi_overrides(arms, phi_overrides, k_grid);
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) {
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }

    Rcpp::List out = tulpa::run_multi_block_nested_laplace_joint(
        n_grid, arms, parsed, blocks, n_x_after_re,
        max_iter, tol, n_threads,
        /*store_modes=*/true, x_init,
        store_Q,
        prep
    );
    out["sigma_occ_grid"] = sigma_occ_grid;
    out["rho_car_grid"]   = rho_car_grid;
    if (has_copy) out["sigma_pos_grid"] = sigma_pos_grid;
    return out;
}
