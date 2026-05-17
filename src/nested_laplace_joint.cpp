// nested_laplace_joint.cpp
// Joint multi-likelihood nested-Laplace backends. One [[Rcpp::export]] per
// spatial prior: BYM2, ICAR, CAR_proper.
//
// All three share the same per-arm parsing and per-arm/spatial cross-block
// scatter logic, factored into nested_laplace_joint_core.h. What differs:
//
//   backend     n_sub  outer grid axes                        spatial prior
//   ----------- -----  ------------------------------------  ----------------------
//   BYM2        2      (sigma_occ, rho [, sigma_pos])         ICAR(tau=1) phi + iid theta
//   ICAR        1      (sigma_occ       [, sigma_pos])        ICAR(tau=1) phi
//   CAR_proper  1      (sigma_occ, rho_car [, sigma_pos])     D - rho_car * W (tau=1)
//
// Each backend now uses a unit-precision latent block (no tau in the latent
// prior). The per-arm field amplitude enters via `eta_arm = X beta + sigma_arm
// * z_s`, where sigma_arm = sigma_occ for non-copy arms and sigma_pos for the
// copy arm. This breaks the (sigma, alpha) identifiability ridge — see
// gcol33/tulpa#18 — because each arm's likelihood now anchors its own sigma
// directly instead of sharing one sigma scaled by alpha. alpha = sigma_pos /
// sigma_occ is recovered post-hoc on the R side.
//
// Each kernel composes:
//   parse_joint_arms()                  -> (parsed, arms, n_x_after_re)
//   compute_arm_eta_joint_generic()     -> per-arm eta_k
//   scatter_arm_obs_joint_generic()     -> per-arm beta/RE/spatial Hessian
//   add_per_arm_beta_re_priors()        -> diagonal beta + RE priors
//   add_<spatial>_prior() / log_prior_<spatial>()  -> backend-specific
//   laplace_newton_solve_joint()        -> shared Newton primitive
//   run_nested_laplace_grid()           -> outer grid orchestrator
//
// See dev_notes/joint_nested_laplace.md for the math.

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
#include <array>
#include <cmath>
#include <limits>
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

// Per-arm sigma at the current outer-grid point. Donor arms see sigma_occ;
// the copy arm sees sigma_pos. Used as a multiplier on the unit-precision
// latent field (z_s) when forming each arm's eta.
inline double arm_sigma_at(int k_arm, int k_grid, int copy_arm,
                           bool has_copy,
                           const Rcpp::NumericVector& sigma_occ_grid,
                           const Rcpp::NumericVector& sigma_pos_grid) {
    if (has_copy && k_arm == copy_arm) {
        return sigma_pos_grid[k_grid];
    }
    return sigma_occ_grid[k_grid];
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
    int n_x         = phi_start + 2 * n_spatial_units;
    constexpr int n_sub = 2;
    std::array<int, 2> sub_starts = { phi_start, theta_start };

    tulpa::SparseCholeskySolver shared_solver;

    auto solve_at_theta = [&](int k_grid, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        apply_phi_overrides(arms, phi_overrides, k_grid);
        double rho_k      = rho_grid[k_grid];
        double sqrt_rho   = std::sqrt(rho_k + 1e-10);
        double sqrt_1_rho = std::sqrt(1.0 - rho_k + 1e-10);
        // Unit-precision base contributions: sigma enters per arm below.
        double d_phi_base_unit   = sqrt_rho * scale_factor;
        double d_theta_base_unit = sqrt_1_rho;

        auto compute_eta_joint = [&](const Rcpp::NumericVector& x,
                                     std::vector<Rcpp::NumericVector>& etas) {
            for (int k = 0; k < n_arms; k++) {
                double s_k = arm_sigma_at(k, k_grid, copy_arm, has_copy,
                                           sigma_occ_grid, sigma_pos_grid);
                std::array<double, 2> d_eff = {
                    s_k * d_phi_base_unit,
                    s_k * d_theta_base_unit
                };
                tulpa::compute_arm_eta_joint_generic(
                    x, etas[k], parsed[k], arms[k].N,
                    n_spatial_units, n_sub, sub_starts, d_eff, n_threads
                );
            }
        };

        auto scatter_joint = [&](const Rcpp::NumericVector& x,
                                 const std::vector<Rcpp::NumericVector>& etas,
                                 tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            for (int k = 0; k < n_arms; k++) {
                double s_k = arm_sigma_at(k, k_grid, copy_arm, has_copy,
                                           sigma_occ_grid, sigma_pos_grid);
                std::array<double, 2> d_eff = {
                    s_k * d_phi_base_unit,
                    s_k * d_theta_base_unit
                };
                tulpa::scatter_arm_obs_joint_generic(
                    x, etas[k], parsed[k], arms[k],
                    n_spatial_units, n_sub, sub_starts, d_eff, grad, H
                );
            }

            // BYM2 spatial prior: ICAR(tau=1) on phi + iid N(0,1) on theta.
            // The per-arm sigma multiplier on eta absorbs the field amplitude
            // — the latent prior stays unit-scale.
            tulpa::add_icar_prior(
                grad, H, x, phi_start, n_spatial_units, /*tau_spatial=*/1.0,
                adj_row_ptr, adj_col_idx, n_neighbors
            );
            for (int s = 0; s < n_spatial_units; s++) {
                int idx = theta_start + s;
                grad[idx] -= x[idx];
                H[idx][idx] += 1.0;
            }

            tulpa::add_per_arm_beta_re_priors(grad, H, x, parsed);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            // Center phi block to mean zero AND shift each arm's intercept
            // (first beta column) so eta is preserved. Without the intercept
            // shift, post-hoc centering would corrupt log_lik at the reported
            // mode — see laplace_newton_joint.h note on the ordering of
            // log_marginal vs. centering.
            double c = 0.0;
            for (int s = 0; s < n_spatial_units; s++) c += x[phi_start + s];
            c /= n_spatial_units;
            if (std::abs(c) < 1e-15) return;
            for (int s = 0; s < n_spatial_units; s++) x[phi_start + s] -= c;
            for (int k = 0; k < n_arms; k++) {
                double s_k = arm_sigma_at(k, k_grid, copy_arm, has_copy,
                                           sigma_occ_grid, sigma_pos_grid);
                if (parsed[k].p > 0) {
                    x[parsed[k].beta_start] += s_k * d_phi_base_unit * c;
                }
            }
        };

        auto log_prior_joint = [&](const Rcpp::NumericVector& x,
                                   const std::vector<Rcpp::NumericVector>&) -> double {
            double lp = tulpa::log_prior_per_arm_re(x, parsed);
            // theta iid N(0, 1)
            for (int s = 0; s < n_spatial_units; s++) {
                double v = x[theta_start + s];
                lp -= 0.5 * v * v;
            }
            lp -= 0.5 * n_spatial_units * std::log(2.0 * M_PI);
            // phi ICAR(tau = 1) — uses the same quadratic form as add_icar_prior.
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
            lp += -0.5 * quad_form;
            return lp;
        };

        return tulpa::laplace_newton_solve_joint(
            arms, n_x,
            max_iter, tol, n_threads,
            compute_eta_joint, scatter_joint, center, log_prior_joint,
            prev_mode, &shared_solver,
            store_Q
        );
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) {
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }

    Rcpp::List out = tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, /*store_modes=*/true
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
    int n_x       = phi_start + n_spatial_units;
    constexpr int n_sub = 1;
    std::array<int, 2> sub_starts = { phi_start, -1 };

    // Adjacency CSR copied into std::vector for the dense log|Q(rho)| helper.
    std::vector<int> adj_rp_v(adj_row_ptr.begin(), adj_row_ptr.end());
    std::vector<int> adj_ci_v(adj_col_idx.begin(), adj_col_idx.end());
    std::vector<int> n_nbr_v (n_neighbors.begin(), n_neighbors.end());

    tulpa::SparseCholeskySolver shared_solver;

    auto solve_at_theta = [&](int k_grid, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        apply_phi_overrides(arms, phi_overrides, k_grid);
        double rho_car_k = rho_car_grid[k_grid];
        const double d_phi_base_unit = 1.0;
        const double tau_unit = 1.0;

        // log|Q(rho_car_k)|, used by log_prior_car_proper.
        std::vector<double> Qmat = tulpa_car_proper::compute_car_precision(
            n_spatial_units, adj_rp_v, adj_ci_v, n_nbr_v, rho_car_k);
        double log_det_Q_rho =
            tulpa_car_proper::car_log_det(n_spatial_units, Qmat);
        if (!std::isfinite(log_det_Q_rho)) {
            // Degenerate (D - rho W) at this rho — skip with -inf marginal.
            tulpa::LaplaceResult bad;
            bad.mode = Rcpp::NumericVector(n_x, 0.0);
            bad.converged = false;
            bad.n_iter = 0;
            bad.log_det_Q = 0.0;
            bad.log_marginal = -std::numeric_limits<double>::infinity();
            return bad;
        }

        auto compute_eta_joint = [&](const Rcpp::NumericVector& x,
                                     std::vector<Rcpp::NumericVector>& etas) {
            for (int k = 0; k < n_arms; k++) {
                double s_k = arm_sigma_at(k, k_grid, copy_arm, has_copy,
                                           sigma_occ_grid, sigma_pos_grid);
                std::array<double, 2> d_eff = { s_k * d_phi_base_unit, 0.0 };
                tulpa::compute_arm_eta_joint_generic(
                    x, etas[k], parsed[k], arms[k].N,
                    n_spatial_units, n_sub, sub_starts, d_eff, n_threads
                );
            }
        };

        auto scatter_joint = [&](const Rcpp::NumericVector& x,
                                 const std::vector<Rcpp::NumericVector>& etas,
                                 tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            for (int k = 0; k < n_arms; k++) {
                double s_k = arm_sigma_at(k, k_grid, copy_arm, has_copy,
                                           sigma_occ_grid, sigma_pos_grid);
                std::array<double, 2> d_eff = { s_k * d_phi_base_unit, 0.0 };
                tulpa::scatter_arm_obs_joint_generic(
                    x, etas[k], parsed[k], arms[k],
                    n_spatial_units, n_sub, sub_starts, d_eff, grad, H
                );
            }
            tulpa::add_car_proper_prior(
                grad, H, x, phi_start, n_spatial_units, tau_unit, rho_car_k,
                adj_row_ptr, adj_col_idx, n_neighbors
            );
            tulpa::add_per_arm_beta_re_priors(grad, H, x, parsed);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            // Proper CAR has full-rank Q so shifting phi by a constant changes
            // the prior quadratic form. Skip post-hoc centering here: the
            // reported mode keeps whatever mean(phi) Newton converged to,
            // preserving log_marginal exactly.
            (void)x;
        };

        auto log_prior_joint = [&](const Rcpp::NumericVector& x,
                                   const std::vector<Rcpp::NumericVector>&) -> double {
            double lp = tulpa::log_prior_per_arm_re(x, parsed);
            lp += tulpa::log_prior_car_proper(
                x, phi_start, n_spatial_units, tau_unit, rho_car_k, log_det_Q_rho,
                adj_row_ptr, adj_col_idx, n_neighbors
            );
            return lp;
        };

        return tulpa::laplace_newton_solve_joint(
            arms, n_x,
            max_iter, tol, n_threads,
            compute_eta_joint, scatter_joint, center, log_prior_joint,
            prev_mode, &shared_solver,
            store_Q
        );
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) {
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }

    Rcpp::List out = tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, /*store_modes=*/true
    );
    out["sigma_occ_grid"] = sigma_occ_grid;
    out["rho_car_grid"]   = rho_car_grid;
    if (has_copy) out["sigma_pos_grid"] = sigma_pos_grid;
    return out;
}
