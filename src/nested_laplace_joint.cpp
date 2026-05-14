// nested_laplace_joint.cpp
// Joint multi-likelihood nested-Laplace backends. Currently: BYM2 only.
//
// First backend in the family of joint kernels (Phase 1c). The single-arm
// BYM2 kernel lives in nested_laplace.cpp; this file adds a *joint* variant
// that takes a list of response arms sharing one BYM2 spatial block, with
// optional INLA-style copy-scaling on a designated arm.
//
// See dev_notes/joint_nested_laplace.md for math; the inner Newton primitive
// `laplace_newton_solve_joint` is in laplace_newton_joint.h.

#include "laplace_core.h"
#include "laplace_newton_joint.h"
#include "laplace_re_priors.h"
#include "laplace_spatial_priors.h"
#include "laplace_family_link.h"
#include "nested_laplace_grid.h"
#include <Rcpp.h>
#include <vector>
#include <string>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// One arm parsed out of the R-side `arms_list`. Owns its Rcpp objects so the
// per-grid-point lambdas can capture references safely.
struct ParsedArm {
    Rcpp::NumericMatrix X;            // [N_arm x p_arm], column-major
    Rcpp::NumericVector re_idx;       // [N_arm], 1-based; 0/NA -> no RE
    Rcpp::IntegerVector spatial_idx;  // [N_arm], 1-based
    int                 p;
    int                 n_re_groups;
    double              sigma_re;
    int                 beta_start;   // offset into joint x where this arm's beta begins
    int                 re_start;     // offset for this arm's RE block
    double              tau_re;
};

} // namespace

// =====================================================================
// Joint nested Laplace — BYM2 (3D grid over (sigma_spatial, rho, alpha))
// =====================================================================
// Latent layout:
//   x = [ beta_1 (p_1) | ... | beta_K (p_K)
//       | re_1 (n_re_1) | ... | re_K (n_re_K)
//       | phi (n_s) | theta (n_s) ]
//
// At grid point k:
//   sigma_k, rho_k, alpha_k = grids[k]
//   eta arm k', obs i = X_k' beta_k' + re_k'[g(i)] + arm_scale_k' * w_s(sigma_k, rho_k)
//   where w_s = sigma_k * (sqrt(rho) * sf * phi_s + sqrt(1-rho) * theta_s)
//   and arm_scale_k' = alpha_k if k' == copy_arm else 1.0
//
// `copy_arm` is 0-based; pass -1 to disable copy-scaling (alpha_grid is then
// ignored — caller may pass an empty vector).
//
// arms_list: a List of length n_arms. Each entry is a List with named fields:
//   y           : NumericVector [N_arm]
//   n_trials    : IntegerVector [N_arm] (use 1's for non-binomial)
//   X           : NumericMatrix [N_arm x p_arm]
//   re_idx      : NumericVector [N_arm] (1-based group index, 0 = no group)
//   spatial_idx : IntegerVector [N_arm] (1-based unit index, 0 = no spatial)
//   n_re_groups : int
//   sigma_re    : double
//   family      : character
//   phi         : double
//
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
    if (n_arms < 1) Rcpp::stop("arms_list must contain at least one arm.");

    int n_grid = sigma_spatial_grid.size();
    if (rho_grid.size() != n_grid) {
        Rcpp::stop("sigma_spatial_grid and rho_grid must have the same length.");
    }
    bool has_copy = (copy_arm >= 0);
    if (has_copy) {
        if (copy_arm >= n_arms) {
            Rcpp::stop("copy_arm index out of range.");
        }
        if (alpha_grid.size() != n_grid) {
            Rcpp::stop("alpha_grid must have the same length as sigma_spatial_grid "
                       "when copy_arm >= 0.");
        }
    }

    // ---- Parse arms ------------------------------------------------------
    std::vector<ParsedArm> parsed(n_arms);
    std::vector<tulpa::JointArm> arms(n_arms);

    int n_x_running = 0;
    // First pass: betas
    for (int k = 0; k < n_arms; k++) {
        Rcpp::List a = arms_list[k];
        ParsedArm& pa = parsed[k];
        pa.X           = Rcpp::as<Rcpp::NumericMatrix>(a["X"]);
        pa.re_idx      = Rcpp::as<Rcpp::NumericVector>(a["re_idx"]);
        pa.spatial_idx = Rcpp::as<Rcpp::IntegerVector>(a["spatial_idx"]);
        pa.p           = pa.X.ncol();
        pa.n_re_groups = Rcpp::as<int>(a["n_re_groups"]);
        pa.sigma_re    = Rcpp::as<double>(a["sigma_re"]);
        pa.tau_re      = (pa.n_re_groups > 0)
                         ? 1.0 / (pa.sigma_re * pa.sigma_re + 1e-10)
                         : 0.0;
        pa.beta_start  = n_x_running;
        n_x_running   += pa.p;

        arms[k].y        = Rcpp::as<Rcpp::NumericVector>(a["y"]);
        arms[k].n_trials = Rcpp::as<Rcpp::IntegerVector>(a["n_trials"]);
        arms[k].family   = Rcpp::as<std::string>(a["family"]);
        arms[k].phi      = Rcpp::as<double>(a["phi"]);
        arms[k].N        = (int)arms[k].y.size();

        if ((int)pa.X.nrow() != arms[k].N) {
            Rcpp::stop("Arm %d: nrow(X) (%d) != length(y) (%d).",
                       k + 1, (int)pa.X.nrow(), arms[k].N);
        }
        if ((int)arms[k].n_trials.size() != arms[k].N) {
            Rcpp::stop("Arm %d: length(n_trials) (%d) != length(y) (%d).",
                       k + 1, (int)arms[k].n_trials.size(), arms[k].N);
        }
        if ((int)pa.re_idx.size() != arms[k].N) {
            Rcpp::stop("Arm %d: length(re_idx) (%d) != length(y) (%d).",
                       k + 1, (int)pa.re_idx.size(), arms[k].N);
        }
        if ((int)pa.spatial_idx.size() != arms[k].N) {
            Rcpp::stop("Arm %d: length(spatial_idx) (%d) != length(y) (%d).",
                       k + 1, (int)pa.spatial_idx.size(), arms[k].N);
        }
    }
    // Second pass: REs
    for (int k = 0; k < n_arms; k++) {
        parsed[k].re_start = n_x_running;
        n_x_running += parsed[k].n_re_groups;
    }
    int phi_start   = n_x_running;
    int theta_start = phi_start + n_spatial_units;
    int n_x         = phi_start + 2 * n_spatial_units;

    tulpa::SparseCholeskySolver shared_solver;

    // --------------------------------------------------------------------
    // Per-grid-point inner Laplace
    // --------------------------------------------------------------------
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
                const ParsedArm& pa = parsed[k];
                int N_k = arms[k].N;
                int p_k = pa.p;
                int n_re_k = pa.n_re_groups;
                int bstart = pa.beta_start;
                int rstart = pa.re_start;
                double arm_scale = (has_copy && k == copy_arm) ? alpha_k : 1.0;
                double d_phi   = arm_scale * d_phi_base;
                double d_theta = arm_scale * d_theta_base;

                #ifdef _OPENMP
                #pragma omp parallel for schedule(static) \
                    num_threads(n_threads > 0 ? n_threads : 1)
                #endif
                for (int i = 0; i < N_k; i++) {
                    double e = 0.0;
                    for (int j = 0; j < p_k; j++) e += pa.X(i, j) * x[bstart + j];
                    if (n_re_k > 0) {
                        int g = (int)pa.re_idx[i] - 1;
                        if (g >= 0 && g < n_re_k) e += x[rstart + g];
                    }
                    if (n_spatial_units > 0) {
                        int s = pa.spatial_idx[i] - 1;
                        if (s >= 0 && s < n_spatial_units) {
                            e += d_phi   * x[phi_start   + s] +
                                 d_theta * x[theta_start + s];
                        }
                    }
                    etas[k][i] = e;
                }
            }
        };

        auto scatter_joint = [&](const Rcpp::NumericVector& x,
                                 const std::vector<Rcpp::NumericVector>& etas,
                                 tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            // ---- Per-arm data scatter ----
            for (int k = 0; k < n_arms; k++) {
                const ParsedArm& pa = parsed[k];
                int N_k = arms[k].N;
                int p_k = pa.p;
                int n_re_k = pa.n_re_groups;
                int bstart = pa.beta_start;
                int rstart = pa.re_start;
                const std::string& family = arms[k].family;
                double phi_disp = arms[k].phi;
                double arm_scale = (has_copy && k == copy_arm) ? alpha_k : 1.0;
                double d_phi   = arm_scale * d_phi_base;
                double d_theta = arm_scale * d_theta_base;

                for (int i = 0; i < N_k; i++) {
                    auto gh = tulpa::grad_hess_for_family(
                        arms[k].y[i], arms[k].n_trials[i], etas[k][i],
                        family, phi_disp
                    );

                    int g_re = -1;
                    if (n_re_k > 0) {
                        int gi = (int)pa.re_idx[i] - 1;
                        if (gi >= 0 && gi < n_re_k) g_re = rstart + gi;
                    }
                    int phi_idx   = -1;
                    int theta_idx = -1;
                    if (n_spatial_units > 0) {
                        int s = pa.spatial_idx[i] - 1;
                        if (s >= 0 && s < n_spatial_units) {
                            phi_idx   = phi_start + s;
                            theta_idx = theta_start + s;
                        }
                    }

                    // Beta block: gradient + diagonal-block Hessian
                    for (int j = 0; j < p_k; j++) {
                        double Xij = pa.X(i, j);
                        grad[bstart + j] += gh.grad * Xij;
                        for (int l = 0; l < p_k; l++) {
                            H[bstart + j][bstart + l] +=
                                gh.neg_hess * Xij * pa.X(i, l);
                        }
                        if (g_re >= 0) {
                            H[bstart + j][g_re] += gh.neg_hess * Xij;
                            H[g_re][bstart + j] += gh.neg_hess * Xij;
                        }
                        if (phi_idx >= 0) {
                            H[bstart + j][phi_idx]   += gh.neg_hess * Xij * d_phi;
                            H[phi_idx][bstart + j]   += gh.neg_hess * Xij * d_phi;
                            H[bstart + j][theta_idx] += gh.neg_hess * Xij * d_theta;
                            H[theta_idx][bstart + j] += gh.neg_hess * Xij * d_theta;
                        }
                    }

                    // RE block: diagonal + cross with spatial
                    if (g_re >= 0) {
                        grad[g_re] += gh.grad;
                        H[g_re][g_re] += gh.neg_hess;
                        if (phi_idx >= 0) {
                            H[g_re][phi_idx]   += gh.neg_hess * d_phi;
                            H[phi_idx][g_re]   += gh.neg_hess * d_phi;
                            H[g_re][theta_idx] += gh.neg_hess * d_theta;
                            H[theta_idx][g_re] += gh.neg_hess * d_theta;
                        }
                    }

                    // Spatial block: accumulates from all arms with arm-specific
                    // factors. arm == copy_arm contributes alpha-scaled factors,
                    // so its Hessian-diagonal contribution is alpha^2 * (...).
                    if (phi_idx >= 0) {
                        grad[phi_idx]   += gh.grad * d_phi;
                        grad[theta_idx] += gh.grad * d_theta;
                        H[phi_idx][phi_idx]     += gh.neg_hess * d_phi * d_phi;
                        H[theta_idx][theta_idx] += gh.neg_hess * d_theta * d_theta;
                        H[phi_idx][theta_idx]   += gh.neg_hess * d_phi * d_theta;
                        H[theta_idx][phi_idx]   += gh.neg_hess * d_phi * d_theta;
                    }
                }
            }

            // ---- BYM2 spatial prior on (phi, theta) ----
            tulpa::add_icar_prior(
                grad, H, x, phi_start, n_spatial_units, /*tau_spatial=*/1.0,
                adj_row_ptr, adj_col_idx, n_neighbors
            );
            for (int s = 0; s < n_spatial_units; s++) {
                int idx = theta_start + s;
                grad[idx] -= x[idx];
                H[idx][idx] += 1.0;
            }

            // ---- Per-arm beta + RE Gaussian priors ----
            const double tau_beta = 1e-4;
            for (int k = 0; k < n_arms; k++) {
                const ParsedArm& pa = parsed[k];
                for (int j = 0; j < pa.p; j++) {
                    grad[pa.beta_start + j] -= tau_beta * x[pa.beta_start + j];
                    H[pa.beta_start + j][pa.beta_start + j] += tau_beta;
                }
                for (int g = 0; g < pa.n_re_groups; g++) {
                    grad[pa.re_start + g] -= pa.tau_re * x[pa.re_start + g];
                    H[pa.re_start + g][pa.re_start + g] += pa.tau_re;
                }
            }
        };

        auto center = [&](Rcpp::NumericVector& x) {
            tulpa::center_effects(x, phi_start, n_spatial_units);
        };

        auto log_prior_joint = [&](const Rcpp::NumericVector& x,
                                   const std::vector<Rcpp::NumericVector>&) -> double {
            double lp = 0.0;
            // Per-arm RE prior contribution (matches single-arm convention:
            // weak beta prior is in grad/H but not in log_prior to keep the
            // joint log-marginal comparable to two single-arm fits).
            for (int k = 0; k < n_arms; k++) {
                const ParsedArm& pa = parsed[k];
                for (int g = 0; g < pa.n_re_groups; g++) {
                    double v = x[pa.re_start + g];
                    lp -= 0.5 * pa.tau_re * v * v;
                }
                if (pa.n_re_groups > 0) {
                    lp += 0.5 * pa.n_re_groups * std::log(pa.tau_re / (2.0 * M_PI));
                }
            }
            // BYM2: theta iid N(0, 1)
            for (int s = 0; s < n_spatial_units; s++) {
                double v = x[theta_start + s];
                lp -= 0.5 * v * v;
            }
            lp -= 0.5 * n_spatial_units * std::log(2.0 * M_PI);
            // BYM2: phi ICAR(tau=1)
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
