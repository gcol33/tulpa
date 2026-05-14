// nested_laplace_joint.cpp
// Joint multi-likelihood nested-Laplace backends. One [[Rcpp::export]] per
// spatial prior: BYM2, ICAR, CAR_proper.
//
// All three share the same per-arm parsing and per-arm/spatial cross-block
// scatter logic, factored into nested_laplace_joint_core.h. What differs:
//
//   backend     n_sub  outer grid axes                    spatial prior
//   ----------- -----  --------------------------------- ----------------------
//   BYM2        2      (sigma_spatial, rho [, alpha])    ICAR phi + iid theta
//   ICAR        1      (tau_spatial      [, alpha])      ICAR phi (tau-scaled)
//   CAR_proper  1      (tau, rho_car     [, alpha])      D - rho_car * W
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
#include "nested_laplace_grid.h"
#include "nested_laplace_joint_core.h"
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
                           const Rcpp::NumericVector& alpha_grid,
                           const char* primary_name) {
    bool has_copy = (copy_arm >= 0);
    if (has_copy) {
        if (copy_arm >= n_arms) {
            Rcpp::stop("copy_arm index out of range.");
        }
        if (alpha_grid.size() != n_grid_primary) {
            Rcpp::stop("alpha_grid must have the same length as %s "
                       "when copy_arm >= 0.", primary_name);
        }
    }
    return has_copy;
}

} // namespace

// =====================================================================
// Joint nested Laplace — BYM2 (3D outer: sigma_spatial, rho, alpha)
// =====================================================================
// Latent layout:
//   x = [ beta_1 | ... | beta_K | re_1 | ... | re_K | phi (n_s) | theta (n_s) ]
//
// Per arm k', obs i:
//   eta_i = X_k' beta_k' + re_k'[g(i)] + arm_scale_k' * w_s
//   w_s   = sigma_k * (sqrt(rho_k) * sf * phi_s + sqrt(1 - rho_k) * theta_s)
//   arm_scale_k' = alpha_k if k' == copy_arm else 1.0
//
// `copy_arm` is 0-based; pass -1 to disable scaling (alpha_grid then ignored).

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_joint_bym2(
    Rcpp::List arms_list,
    int copy_arm,
    int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double scale_factor,
    Rcpp::NumericVector sigma_spatial_grid,
    Rcpp::NumericVector rho_grid,
    Rcpp::NumericVector alpha_grid,
    int max_iter = 50,
    double tol = 1e-6,
    int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_arms = arms_list.size();
    int n_grid = sigma_spatial_grid.size();
    if (rho_grid.size() != n_grid) {
        Rcpp::stop("sigma_spatial_grid and rho_grid must have the same length.");
    }
    bool has_copy = validate_joint_inputs(n_arms, copy_arm, n_grid, alpha_grid,
                                          "sigma_spatial_grid");

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm>  arms;
    int n_x_after_re = tulpa::parse_joint_arms(arms_list, parsed, arms);

    int phi_start   = n_x_after_re;
    int theta_start = phi_start + n_spatial_units;
    int n_x         = phi_start + 2 * n_spatial_units;
    constexpr int n_sub = 2;
    std::array<int, 2> sub_starts = { phi_start, theta_start };

    tulpa::SparseCholeskySolver shared_solver;

    auto solve_at_theta = [&](int k_grid, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        double sigma_k    = sigma_spatial_grid[k_grid];
        double rho_k      = rho_grid[k_grid];
        double alpha_k    = has_copy ? alpha_grid[k_grid] : 1.0;
        double sqrt_rho   = std::sqrt(rho_k + 1e-10);
        double sqrt_1_rho = std::sqrt(1.0 - rho_k + 1e-10);
        double d_phi_base   = sigma_k * sqrt_rho * scale_factor;
        double d_theta_base = sigma_k * sqrt_1_rho;

        auto compute_eta_joint = [&](const Rcpp::NumericVector& x,
                                     std::vector<Rcpp::NumericVector>& etas) {
            for (int k = 0; k < n_arms; k++) {
                double arm_scale = (has_copy && k == copy_arm) ? alpha_k : 1.0;
                std::array<double, 2> d_eff = {
                    arm_scale * d_phi_base,
                    arm_scale * d_theta_base
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
                double arm_scale = (has_copy && k == copy_arm) ? alpha_k : 1.0;
                std::array<double, 2> d_eff = {
                    arm_scale * d_phi_base,
                    arm_scale * d_theta_base
                };
                tulpa::scatter_arm_obs_joint_generic(
                    x, etas[k], parsed[k], arms[k],
                    n_spatial_units, n_sub, sub_starts, d_eff, grad, H
                );
            }

            // BYM2 spatial prior: ICAR(tau=1) on phi + iid N(0,1) on theta.
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
            tulpa::center_effects(x, phi_start, n_spatial_units);
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
    out["sigma_spatial_grid"] = sigma_spatial_grid;
    out["rho_grid"] = rho_grid;
    if (has_copy) out["alpha_grid"] = alpha_grid;
    return out;
}

// =====================================================================
// Joint nested Laplace — ICAR (2D outer: tau_spatial, alpha)
// =====================================================================
// Latent layout:
//   x = [ beta_1 | ... | beta_K | re_1 | ... | re_K | phi (n_s) ]
//
// Per arm k', obs i:
//   eta_i = X_k' beta_k' + re_k'[g(i)] + arm_scale_k' * phi_s
//   arm_scale_k' = alpha_k if k' == copy_arm else 1.0
//
// Spatial prior: ICAR with precision tau_k * Q_struct.

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_joint_icar(
    Rcpp::List arms_list,
    int copy_arm,
    int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::NumericVector tau_grid,
    Rcpp::NumericVector alpha_grid,
    int max_iter = 50,
    double tol = 1e-6,
    int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_arms = arms_list.size();
    int n_grid = tau_grid.size();
    bool has_copy = validate_joint_inputs(n_arms, copy_arm, n_grid, alpha_grid,
                                          "tau_grid");

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm>  arms;
    int n_x_after_re = tulpa::parse_joint_arms(arms_list, parsed, arms);

    int phi_start = n_x_after_re;
    int n_x       = phi_start + n_spatial_units;
    constexpr int n_sub = 1;
    std::array<int, 2> sub_starts = { phi_start, -1 };

    tulpa::SparseCholeskySolver shared_solver;

    auto solve_at_theta = [&](int k_grid, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        double tau_k   = tau_grid[k_grid];
        double alpha_k = has_copy ? alpha_grid[k_grid] : 1.0;
        // ICAR latent x[s] = phi_s with prior precision tau_k * Q_struct;
        // d eta_s/d phi_s = 1 for the base arm, alpha for the copy arm.
        const double d_phi_base = 1.0;

        auto compute_eta_joint = [&](const Rcpp::NumericVector& x,
                                     std::vector<Rcpp::NumericVector>& etas) {
            for (int k = 0; k < n_arms; k++) {
                double arm_scale = (has_copy && k == copy_arm) ? alpha_k : 1.0;
                std::array<double, 2> d_eff = { arm_scale * d_phi_base, 0.0 };
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
                double arm_scale = (has_copy && k == copy_arm) ? alpha_k : 1.0;
                std::array<double, 2> d_eff = { arm_scale * d_phi_base, 0.0 };
                tulpa::scatter_arm_obs_joint_generic(
                    x, etas[k], parsed[k], arms[k],
                    n_spatial_units, n_sub, sub_starts, d_eff, grad, H
                );
            }
            tulpa::add_icar_prior(
                grad, H, x, phi_start, n_spatial_units, tau_k,
                adj_row_ptr, adj_col_idx, n_neighbors
            );
            tulpa::add_per_arm_beta_re_priors(grad, H, x, parsed);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            tulpa::center_effects(x, phi_start, n_spatial_units);
        };

        auto log_prior_joint = [&](const Rcpp::NumericVector& x,
                                   const std::vector<Rcpp::NumericVector>&) -> double {
            double lp = tulpa::log_prior_per_arm_re(x, parsed);
            lp += tulpa::log_prior_icar(
                x, phi_start, n_spatial_units, tau_k,
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
    out["tau_grid"] = tau_grid;
    if (has_copy) out["alpha_grid"] = alpha_grid;
    return out;
}

// =====================================================================
// Joint nested Laplace — proper CAR (3D outer: tau, rho_car, alpha)
// =====================================================================
// Latent layout:
//   x = [ beta_1 | ... | beta_K | re_1 | ... | re_K | phi (n_s) ]
//
// Per arm k', obs i:
//   eta_i = X_k' beta_k' + re_k'[g(i)] + arm_scale_k' * phi_s
//   arm_scale_k' = alpha_k if k' == copy_arm else 1.0
//
// Spatial prior: phi ~ N(0, [tau_k * (D - rho_car_k * W)]^{-1}).
// log|Q(rho_car)| is precomputed once per grid point via the dense
// Cholesky in tulpa_car_proper::car_log_det (n_s typically < 500 so
// O(n^3) is fine; swap in a sparse Cholesky here if it becomes the
// bottleneck).

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_joint_car_proper(
    Rcpp::List arms_list,
    int copy_arm,
    int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::NumericVector tau_grid,
    Rcpp::NumericVector rho_car_grid,
    Rcpp::NumericVector alpha_grid,
    int max_iter = 50,
    double tol = 1e-6,
    int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_arms = arms_list.size();
    int n_grid = tau_grid.size();
    if (rho_car_grid.size() != n_grid) {
        Rcpp::stop("tau_grid and rho_car_grid must have the same length.");
    }
    bool has_copy = validate_joint_inputs(n_arms, copy_arm, n_grid, alpha_grid,
                                          "tau_grid");

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm>  arms;
    int n_x_after_re = tulpa::parse_joint_arms(arms_list, parsed, arms);

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
        double tau_k     = tau_grid[k_grid];
        double rho_car_k = rho_car_grid[k_grid];
        double alpha_k   = has_copy ? alpha_grid[k_grid] : 1.0;
        const double d_phi_base = 1.0;

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
                double arm_scale = (has_copy && k == copy_arm) ? alpha_k : 1.0;
                std::array<double, 2> d_eff = { arm_scale * d_phi_base, 0.0 };
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
                double arm_scale = (has_copy && k == copy_arm) ? alpha_k : 1.0;
                std::array<double, 2> d_eff = { arm_scale * d_phi_base, 0.0 };
                tulpa::scatter_arm_obs_joint_generic(
                    x, etas[k], parsed[k], arms[k],
                    n_spatial_units, n_sub, sub_starts, d_eff, grad, H
                );
            }
            tulpa::add_car_proper_prior(
                grad, H, x, phi_start, n_spatial_units, tau_k, rho_car_k,
                adj_row_ptr, adj_col_idx, n_neighbors
            );
            tulpa::add_per_arm_beta_re_priors(grad, H, x, parsed);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            tulpa::center_effects(x, phi_start, n_spatial_units);
        };

        auto log_prior_joint = [&](const Rcpp::NumericVector& x,
                                   const std::vector<Rcpp::NumericVector>&) -> double {
            double lp = tulpa::log_prior_per_arm_re(x, parsed);
            lp += tulpa::log_prior_car_proper(
                x, phi_start, n_spatial_units, tau_k, rho_car_k, log_det_Q_rho,
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
    out["tau_grid"]     = tau_grid;
    out["rho_car_grid"] = rho_car_grid;
    if (has_copy) out["alpha_grid"] = alpha_grid;
    return out;
}
