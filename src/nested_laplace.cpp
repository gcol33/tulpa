// nested_laplace.cpp
// Nested Laplace outer-grid drivers for latent Gaussian backends.
//
// Each backend (ICAR, BYM2, RW1, RW2, AR1) has a thin Rcpp entry that:
//   - builds backend-specific Newton callbacks (compute_eta, scatter,
//     center, log_prior) bound to the current grid theta;
//   - dispatches to tulpa::laplace_newton_solve via a per-grid-point
//     `solve_at_theta` lambda;
//   - hands the lambda to tulpa::run_nested_laplace_grid for the outer
//     loop + warm-starting + result aggregation.
//
// The outer-grid boilerplate lives in nested_laplace_grid.h. SPDE has its
// own pattern (rebuilds Q via SpdeQBuilder) and lives in spde_laplace.cpp.

#include "laplace_core.h"
#include "laplace_helpers.h"
#include "nested_laplace_grid.h"
#include "hmc_car_proper.h"
#include <Rcpp.h>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

// =====================================================================
// Nested Laplace: ICAR (1D grid over tau_spatial)
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_icar(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::NumericVector tau_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int n_x = p + n_re_groups + n_spatial_units;

    tulpa::SparseCholeskySolver shared_solver;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    int spatial_start = p + n_re_groups;

    auto solve_at_theta = [&](int k, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        double tau_k = tau_grid[k];

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
            #endif
            for (int i = 0; i < N; i++) {
                eta[i] = 0.0;
                for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) eta[i] += x[p + g];
                }
                if (n_spatial_units > 0) {
                    int s = spatial_idx[i] - 1;
                    if (s >= 0 && s < n_spatial_units) eta[i] += x[spatial_start + s];
                }
            }
        };

        auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                           tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            std::vector<int> eff_idx(N, -1);
            std::vector<double> d_fac(N, 1.0);
            for (int i = 0; i < N; i++) {
                if (n_spatial_units > 0) {
                    int s = spatial_idx[i] - 1;
                    if (s >= 0 && s < n_spatial_units) eff_idx[i] = spatial_start + s;
                }
            }
            tulpa::scatter_obs_with_latent(y, n, X, re_idx, N, p, n_re_groups,
                                            eta, family, phi, eff_idx, d_fac,
                                            grad, H, n_threads);
            tulpa::add_icar_prior(grad, H, x, spatial_start, n_spatial_units, tau_k,
                                   adj_row_ptr, adj_col_idx, n_neighbors);
            tulpa::add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            tulpa::center_effects(x, spatial_start, n_spatial_units);
        };

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            lp += tulpa::log_prior_icar(x, spatial_start, n_spatial_units, tau_k,
                                         adj_row_ptr, adj_col_idx, n_neighbors);
            return lp;
        };

        return tulpa::laplace_newton_solve(
            y, n, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter, center, log_prior,
            prev_mode, &shared_solver
        );
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    Rcpp::List out = tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, /*store_modes=*/true
    );
    out["tau_grid"] = tau_grid;
    return out;
}

// =====================================================================
// Nested Laplace: BYM2 (2D grid over (sigma_spatial, rho))
// =====================================================================
// BYM2 reparameterization: phi structured (ICAR) + theta IID,
//   eta_s = sigma * (sqrt(rho) * scale_factor * phi + sqrt(1 - rho) * theta)

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_bym2(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double scale_factor,
    Rcpp::NumericVector sigma_spatial_grid,
    Rcpp::NumericVector rho_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue
) {
    int n_grid = sigma_spatial_grid.size();
    int N = y.size();
    int p = X.ncol();
    int n_x = p + n_re_groups + 2 * n_spatial_units;

    tulpa::SparseCholeskySolver shared_solver;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    int phi_start = p + n_re_groups;
    int theta_start = phi_start + n_spatial_units;

    auto solve_at_theta = [&](int k, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        double sigma_k = sigma_spatial_grid[k];
        double rho_k = rho_grid[k];
        double sqrt_rho = std::sqrt(rho_k + 1e-10);
        double sqrt_1_rho = std::sqrt(1.0 - rho_k + 1e-10);

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
            #endif
            for (int i = 0; i < N; i++) {
                eta[i] = 0.0;
                for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) eta[i] += x[p + g];
                }
                if (n_spatial_units > 0) {
                    int s = spatial_idx[i] - 1;
                    if (s >= 0 && s < n_spatial_units) {
                        eta[i] += sigma_k * (
                            sqrt_rho * x[phi_start + s] * scale_factor +
                            sqrt_1_rho * x[theta_start + s]
                        );
                    }
                }
            }
        };

        auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                           tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            tulpa::scatter_obs_grad_hess_base(y, n, X, re_idx, N, p, n_re_groups,
                                               eta, family, phi, grad, H, n_threads);

            double d_phi = sigma_k * sqrt_rho * scale_factor;
            double d_theta = sigma_k * sqrt_1_rho;

            for (int i = 0; i < N; i++) {
                if (n_spatial_units <= 0) continue;
                int s = spatial_idx[i] - 1;
                if (s < 0 || s >= n_spatial_units) continue;

                auto gh = tulpa::grad_hess_for_family(y[i], n[i], eta[i], family, phi);
                int pidx = phi_start + s;
                int tidx = theta_start + s;

                grad[pidx] += gh.grad * d_phi;
                grad[tidx] += gh.grad * d_theta;
                H[pidx][pidx] += gh.neg_hess * d_phi * d_phi;
                H[tidx][tidx] += gh.neg_hess * d_theta * d_theta;
                H[pidx][tidx] += gh.neg_hess * d_phi * d_theta;
                H[tidx][pidx] += gh.neg_hess * d_phi * d_theta;

                for (int j = 0; j < p; j++) {
                    H[j][pidx] += gh.neg_hess * X(i, j) * d_phi;
                    H[pidx][j] += gh.neg_hess * X(i, j) * d_phi;
                    H[j][tidx] += gh.neg_hess * X(i, j) * d_theta;
                    H[tidx][j] += gh.neg_hess * X(i, j) * d_theta;
                }

                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) {
                        H[p + g][pidx] += gh.neg_hess * d_phi;
                        H[pidx][p + g] += gh.neg_hess * d_phi;
                        H[p + g][tidx] += gh.neg_hess * d_theta;
                        H[tidx][p + g] += gh.neg_hess * d_theta;
                    }
                }
            }

            tulpa::add_icar_prior(grad, H, x, phi_start, n_spatial_units, 1.0,
                                   adj_row_ptr, adj_col_idx, n_neighbors);
            for (int s = 0; s < n_spatial_units; s++) {
                int idx = theta_start + s;
                grad[idx] -= x[idx];
                H[idx][idx] += 1.0;
            }
            tulpa::add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            tulpa::center_effects(x, phi_start, n_spatial_units);
        };

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            double lp_theta = 0.0;
            for (int s = 0; s < n_spatial_units; s++) {
                lp_theta -= 0.5 * x[theta_start + s] * x[theta_start + s];
            }
            lp_theta -= 0.5 * n_spatial_units * std::log(2.0 * M_PI);
            lp += lp_theta;
            double quad_form = 0.0;
            for (int s = 0; s < n_spatial_units; s++) {
                double phi_s = x[phi_start + s];
                quad_form += n_neighbors[s] * phi_s * phi_s;
                for (int kk = adj_row_ptr[s]; kk < adj_row_ptr[s + 1]; kk++) {
                    int neighbor = adj_col_idx[kk];
                    if (neighbor > s) quad_form -= 2.0 * phi_s * x[phi_start + neighbor];
                }
            }
            lp += -0.5 * quad_form;
            return lp;
        };

        return tulpa::laplace_newton_solve(
            y, n, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter, center, log_prior,
            prev_mode, &shared_solver
        );
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    Rcpp::List out = tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, /*store_modes=*/false
    );
    out["sigma_spatial_grid"] = sigma_spatial_grid;
    out["rho_grid"] = rho_grid;
    return out;
}

// =====================================================================
// Nested Laplace: proper CAR (2D grid over (tau, rho))
// =====================================================================
// Full-rank Q(rho) = D - rho*W; identical inner shape to ICAR but with
// rho scaling the off-diagonals AND a non-vanishing log|Q(rho)| term in
// the per-grid-point log-marginal. log|Q(rho)| is computed once per grid
// point via the dense Cholesky in tulpa_car_proper::car_log_det.

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_car_proper(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::NumericVector tau_grid, Rcpp::NumericVector rho_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue
) {
    int n_grid = tau_grid.size();
    if (rho_grid.size() != n_grid) {
        Rcpp::stop("tau_grid and rho_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int n_x = p + n_re_groups + n_spatial_units;
    int spatial_start = p + n_re_groups;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

    tulpa::SparseCholeskySolver shared_solver;

    // Adjacency CSR copied into std::vector for the dense log|Q(rho)| helper.
    std::vector<int> adj_rp_v(adj_row_ptr.begin(), adj_row_ptr.end());
    std::vector<int> adj_ci_v(adj_col_idx.begin(), adj_col_idx.end());
    std::vector<int> n_nbr_v(n_neighbors.begin(), n_neighbors.end());

    auto solve_at_theta = [&](int k, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        double tau_k = tau_grid[k];
        double rho_k = rho_grid[k];

        // Dense log|Q(rho_k)| once per grid point. For the spatial graphs
        // typical of areal models (n_spatial_units < ~500) the O(n^3) cost
        // is dwarfed by the inner Newton scatter. Swap in sparse Cholesky
        // if this becomes the bottleneck.
        std::vector<double> Qmat = tulpa_car_proper::compute_car_precision(
            n_spatial_units, adj_rp_v, adj_ci_v, n_nbr_v, rho_k);
        double log_det_Q_rho = tulpa_car_proper::car_log_det(n_spatial_units, Qmat);
        if (!std::isfinite(log_det_Q_rho)) {
            // Q not PD at this rho — emit -inf log-marginal and skip the inner
            // solve. Caller's weight normaliser will handle it gracefully.
            tulpa::LaplaceResult bad;
            bad.mode = prev_mode.size() == n_x ? prev_mode :
                       Rcpp::NumericVector(n_x, 0.0);
            bad.log_marginal = -std::numeric_limits<double>::infinity();
            bad.n_iter = 0;
            bad.converged = false;
            bad.log_det_Q = 0.0;
            return bad;
        }

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
            #endif
            for (int i = 0; i < N; i++) {
                eta[i] = 0.0;
                for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) eta[i] += x[p + g];
                }
                if (n_spatial_units > 0) {
                    int s = spatial_idx[i] - 1;
                    if (s >= 0 && s < n_spatial_units) eta[i] += x[spatial_start + s];
                }
            }
        };

        auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                           tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            std::vector<int> eff_idx(N, -1);
            std::vector<double> d_fac(N, 1.0);
            for (int i = 0; i < N; i++) {
                if (n_spatial_units > 0) {
                    int s = spatial_idx[i] - 1;
                    if (s >= 0 && s < n_spatial_units) eff_idx[i] = spatial_start + s;
                }
            }
            tulpa::scatter_obs_with_latent(y, n, X, re_idx, N, p, n_re_groups,
                                            eta, family, phi, eff_idx, d_fac,
                                            grad, H, n_threads);
            tulpa::add_car_proper_prior(grad, H, x, spatial_start, n_spatial_units,
                                         tau_k, rho_k,
                                         adj_row_ptr, adj_col_idx, n_neighbors);
            tulpa::add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            // Q(rho) is full rank for rho strictly inside the eigenvalue
            // interval, so centering is not required for identifiability.
            // Kept consistent with ICAR/AR1 to stabilise vs the intercept.
            tulpa::center_effects(x, spatial_start, n_spatial_units);
        };

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            lp += tulpa::log_prior_car_proper(x, spatial_start, n_spatial_units,
                                                tau_k, rho_k, log_det_Q_rho,
                                                adj_row_ptr, adj_col_idx, n_neighbors);
            return lp;
        };

        return tulpa::laplace_newton_solve(
            y, n, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter, center, log_prior,
            prev_mode, &shared_solver
        );
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    Rcpp::List out = tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, /*store_modes=*/true
    );
    out["tau_grid"] = tau_grid;
    out["rho_grid"] = rho_grid;
    return out;
}

// =====================================================================
// Nested Laplace: RW1 (1D grid over tau_temporal)
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_rw1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector temporal_idx, int n_times, bool cyclic,
    Rcpp::NumericVector tau_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int n_x = p + n_re_groups + n_times;
    int temporal_start = p + n_re_groups;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

    tulpa::SparseCholeskySolver shared_solver;

    auto solve_at_theta = [&](int k, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        double tau_k = tau_grid[k];

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
            #endif
            for (int i = 0; i < N; i++) {
                eta[i] = 0.0;
                for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) eta[i] += x[p + g];
                }
                int t = temporal_idx[i] - 1;
                if (t >= 0 && t < n_times) eta[i] += x[temporal_start + t];
            }
        };

        auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                           tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            std::vector<int> eff_idx(N, -1);
            std::vector<double> d_fac(N, 1.0);
            for (int i = 0; i < N; i++) {
                int t = temporal_idx[i] - 1;
                if (t >= 0 && t < n_times) eff_idx[i] = temporal_start + t;
            }
            tulpa::scatter_obs_with_latent(y, n, X, re_idx, N, p, n_re_groups,
                                            eta, family, phi, eff_idx, d_fac,
                                            grad, H, n_threads);
            tulpa::add_rw1_precision(grad, H, x, temporal_start, n_times, tau_k, cyclic);
            tulpa::add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            tulpa::center_effects(x, temporal_start, n_times);
        };

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            lp += tulpa::log_prior_rw1(x, temporal_start, n_times, tau_k, cyclic);
            return lp;
        };

        return tulpa::laplace_newton_solve(
            y, n, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter, center, log_prior,
            prev_mode, &shared_solver
        );
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    Rcpp::List out = tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, /*store_modes=*/true
    );
    out["tau_grid"] = tau_grid;
    return out;
}

// =====================================================================
// Nested Laplace: RW2 (1D grid over tau_temporal)
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_rw2(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int n_x = p + n_re_groups + n_times;
    int temporal_start = p + n_re_groups;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

    tulpa::SparseCholeskySolver shared_solver;

    auto solve_at_theta = [&](int k, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        double tau_k = tau_grid[k];

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
            #endif
            for (int i = 0; i < N; i++) {
                eta[i] = 0.0;
                for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) eta[i] += x[p + g];
                }
                int t = temporal_idx[i] - 1;
                if (t >= 0 && t < n_times) eta[i] += x[temporal_start + t];
            }
        };

        auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                           tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            std::vector<int> eff_idx(N, -1);
            std::vector<double> d_fac(N, 1.0);
            for (int i = 0; i < N; i++) {
                int t = temporal_idx[i] - 1;
                if (t >= 0 && t < n_times) eff_idx[i] = temporal_start + t;
            }
            tulpa::scatter_obs_with_latent(y, n, X, re_idx, N, p, n_re_groups,
                                            eta, family, phi, eff_idx, d_fac,
                                            grad, H, n_threads);
            tulpa::add_rw2_precision(grad, H, x, temporal_start, n_times, tau_k, false);
            tulpa::add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            tulpa::center_effects(x, temporal_start, n_times);
        };

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            lp += tulpa::log_prior_rw2(x, temporal_start, n_times, tau_k, false);
            return lp;
        };

        return tulpa::laplace_newton_solve(
            y, n, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter, center, log_prior,
            prev_mode, &shared_solver
        );
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    Rcpp::List out = tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, /*store_modes=*/true
    );
    out["tau_grid"] = tau_grid;
    return out;
}

// =====================================================================
// Nested Laplace: AR1 (2D grid over (tau, rho))
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_ar1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_grid, Rcpp::NumericVector rho_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int n_x = p + n_re_groups + n_times;
    int temporal_start = p + n_re_groups;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

    tulpa::SparseCholeskySolver shared_solver;

    auto solve_at_theta = [&](int k, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        double tau_k = tau_grid[k];
        double rho_k = rho_grid[k];

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
            #endif
            for (int i = 0; i < N; i++) {
                eta[i] = 0.0;
                for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) eta[i] += x[p + g];
                }
                int t = temporal_idx[i] - 1;
                if (t >= 0 && t < n_times) eta[i] += x[temporal_start + t];
            }
        };

        auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                           tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            std::vector<int> eff_idx(N, -1);
            std::vector<double> d_fac(N, 1.0);
            for (int i = 0; i < N; i++) {
                int t = temporal_idx[i] - 1;
                if (t >= 0 && t < n_times) eff_idx[i] = temporal_start + t;
            }
            tulpa::scatter_obs_with_latent(y, n, X, re_idx, N, p, n_re_groups,
                                            eta, family, phi, eff_idx, d_fac,
                                            grad, H, n_threads);
            tulpa::add_ar1_precision(grad, H, x, temporal_start, n_times, tau_k, rho_k);
            tulpa::add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            // AR1 is proper (full rank) — no sum-to-zero constraint required,
            // but centering is harmless and stabilises identifiability against
            // the intercept. Keep consistent with RW1/RW2.
            tulpa::center_effects(x, temporal_start, n_times);
        };

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            lp += tulpa::log_prior_ar1(x, temporal_start, n_times, tau_k, rho_k);
            return lp;
        };

        return tulpa::laplace_newton_solve(
            y, n, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter, center, log_prior,
            prev_mode, &shared_solver
        );
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    Rcpp::List out = tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, /*store_modes=*/true
    );
    out["tau_grid"] = tau_grid;
    out["rho_grid"] = rho_grid;
    return out;
}

