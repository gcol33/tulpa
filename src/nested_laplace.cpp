// nested_laplace.cpp
// Nested Laplace outer-grid drivers for latent Gaussian backends.
//
// Two flavours of backend live here:
//
// 1. Indexed-latent backends (ICAR, CAR_proper, RW1, RW2, AR1): η_i picks one
//    entry x[latent_start + latent_idx[i] - 1] from a single latent block
//    (length n_latent). They share compute_eta / scatter / center, and differ
//    only in (prior_add, log_prior_eval, optional per-grid-point prep). The
//    shared body lives in run_indexed_nested_laplace; each Rcpp entry is a
//    thin wrapper that supplies the three callbacks.
//
// 2. Non-indexed backends (BYM2, NNGP, HSGP): two latent blocks, batch NNGP
//    scatter, or basis-rescaled design — the eta/scatter shape genuinely
//    differs, so each builds its own solve_at_theta lambda directly.
//
// SPDE has its own Q-rebuild pattern and lives in spde_laplace.cpp.
// Outer-grid loop + warm-starting + result aggregation live in
// nested_laplace_grid.h.

#include "laplace_core.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include "laplace_scatter.h"
#include "laplace_spatial_priors.h"
#include "laplace_temporal_priors.h"
#include "nested_laplace_grid.h"
#include "hmc_car_proper.h"
#include "gpu_nngp_laplace.h"
#include "hmc_hsgp_kernels.h"  // Eigen-free spectral density only
#include <Rcpp.h>
#include <functional>
#include <limits>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// Standard linear predictor: X*beta + optional RE + x[latent_start + latent_idx[i]-1].
// Used by ICAR, CAR_proper, RW1, RW2, AR1.
inline void nl_compute_eta(
    const Rcpp::NumericVector& x,
    Rcpp::NumericVector& eta,
    int N, int p, int n_re_groups,
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& re_idx,
    int latent_start, int n_latent,
    const Rcpp::IntegerVector& latent_idx,
    int n_threads
) {
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
        int l = latent_idx[i] - 1;
        if (l >= 0 && l < n_latent) eta[i] += x[latent_start + l];
    }
}

// Scatter (eff_idx build + scatter_obs_with_latent) for simple index-based backends.
// The prior call and add_re_beta_priors remain in the caller.
inline void nl_scatter_obs_indexed(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups,
    const Rcpp::NumericVector& eta,
    const std::string& family, double phi,
    int latent_start, int n_latent,
    const Rcpp::IntegerVector& latent_idx,
    tulpa::DenseVec& grad, tulpa::DenseMat& H,
    int n_threads
) {
    std::vector<int> eff_idx(N, -1);
    std::vector<double> d_fac(N, 1.0);
    for (int i = 0; i < N; i++) {
        int l = latent_idx[i] - 1;
        if (l >= 0 && l < n_latent) eff_idx[i] = latent_start + l;
    }
    tulpa::scatter_obs_with_latent(y, n_trials, X, re_idx, N, p, n_re_groups,
                                    eta, family, phi, eff_idx, d_fac,
                                    grad, H, n_threads);
}

// Generic outer-grid driver for indexed-latent nested Laplace backends.
//
// Shared body for ICAR, CAR_proper, RW1, RW2, AR1: each has a single latent
// block of length n_latent picked by latent_idx (one entry per observation),
// the same X*beta + RE + x[latent_start + latent_idx[i] - 1] eta, and the
// same scatter_obs_with_latent + add_re_beta_priors structure. Backends
// differ only in:
//
//   prep_at_grid(k)      : optional per-grid-point precompute (e.g. log|Q(rho)|
//                          for proper CAR). Returns false to short-circuit the
//                          inner solve at this k with log_marginal = -inf.
//                          Pass a no-op `[](int){ return true; }` if not needed.
//   add_prior_at_k(...)  : adds the latent-block prior contribution to (grad, H)
//                          at grid point k (e.g. add_icar_prior, add_ar1_precision).
//   log_prior_at_k(x, k) : returns the latent-block log-prior at grid point k
//                          (e.g. log_prior_icar). The shared driver adds the
//                          RE/beta log-prior on top.
//
// Caller-provided x_init is passed through as the warm-start for grid point 0.
// Backend-specific output keys (tau_grid, rho_grid, ...) are added by the
// caller to the returned list.
template<typename PrepFn, typename AddPriorFn, typename LogPriorFn>
inline Rcpp::List run_indexed_nested_laplace(
    int n_grid,
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    int latent_start, int n_latent,
    const Rcpp::IntegerVector& latent_idx,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    bool store_modes,
    const Rcpp::NumericVector& x_init,
    PrepFn prep_at_grid,
    AddPriorFn add_prior_at_k,
    LogPriorFn log_prior_at_k,
    bool store_Q = false
) {
    int n_x = p + n_re_groups + n_latent;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    tulpa::SparseCholeskySolver shared_solver;

    auto solve_at_theta = [&](int k, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        if (!prep_at_grid(k)) {
            // Short-circuit: per-grid-point precompute reported a non-PD prior.
            // Emit -inf log-marginal; caller's weight normaliser will skip it.
            tulpa::LaplaceResult bad;
            bad.mode = (prev_mode.size() == n_x) ? prev_mode :
                       Rcpp::NumericVector(n_x, 0.0);
            bad.log_marginal = -std::numeric_limits<double>::infinity();
            bad.n_iter = 0;
            bad.converged = false;
            bad.log_det_Q = 0.0;
            return bad;
        }

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            nl_compute_eta(x, eta, N, p, n_re_groups, X, re_idx,
                           latent_start, n_latent, latent_idx, n_threads);
        };

        auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                           tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            nl_scatter_obs_indexed(y, n_trials, X, re_idx, N, p, n_re_groups,
                                    eta, family, phi,
                                    latent_start, n_latent, latent_idx,
                                    grad, H, n_threads);
            add_prior_at_k(grad, H, x, k);
            tulpa::add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            tulpa::center_effects(x, latent_start, n_latent);
        };

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            lp += log_prior_at_k(x, k);
            return lp;
        };

        return tulpa::laplace_newton_solve(
            y, n_trials, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter, center, log_prior,
            prev_mode, &shared_solver, store_Q
        );
    };

    return tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, store_modes
    );
}

// Convenience: extract optional warm-start mode from an Rcpp Nullable into an
// owning NumericVector (empty vector when null). Used by every indexed shim.
inline Rcpp::NumericVector unwrap_x_init(
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable
) {
    if (x_init_nullable.isNotNull()) {
        return Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }
    return Rcpp::NumericVector();
}

} // namespace

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
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int spatial_start = p + n_re_groups;

    auto add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                         const Rcpp::NumericVector& x, int k) {
        tulpa::add_icar_prior(grad, H, x, spatial_start, n_spatial_units, tau_grid[k],
                               adj_row_ptr, adj_col_idx, n_neighbors);
    };
    auto log_prior = [&](const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_icar(x, spatial_start, n_spatial_units, tau_grid[k],
                                       adj_row_ptr, adj_col_idx, n_neighbors);
    };

    Rcpp::List out = run_indexed_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        spatial_start, n_spatial_units, spatial_idx,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        [](int) { return true; }, add_prior, log_prior,
        store_Q
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
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
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
            prev_mode, &shared_solver,
            store_Q
        );
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    Rcpp::List out = tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, /*store_modes=*/true
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
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_grid.size();
    if (rho_grid.size() != n_grid) {
        Rcpp::stop("tau_grid and rho_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int spatial_start = p + n_re_groups;

    // Adjacency CSR copied into std::vector for the dense log|Q(rho)| helper.
    std::vector<int> adj_rp_v(adj_row_ptr.begin(), adj_row_ptr.end());
    std::vector<int> adj_ci_v(adj_col_idx.begin(), adj_col_idx.end());
    std::vector<int> n_nbr_v(n_neighbors.begin(), n_neighbors.end());

    // Per-grid-point log|Q(rho_k)|, recomputed by `prep` and consumed by
    // `log_prior`. Dense O(n^3) is fine for areal graphs (n_spatial < ~500);
    // swap in sparse Cholesky here if it becomes the bottleneck.
    double log_det_Q_rho = 0.0;
    auto prep = [&](int k) -> bool {
        std::vector<double> Qmat = tulpa_car_proper::compute_car_precision(
            n_spatial_units, adj_rp_v, adj_ci_v, n_nbr_v, rho_grid[k]);
        log_det_Q_rho = tulpa_car_proper::car_log_det(n_spatial_units, Qmat);
        return std::isfinite(log_det_Q_rho);
    };
    auto add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                         const Rcpp::NumericVector& x, int k) {
        tulpa::add_car_proper_prior(grad, H, x, spatial_start, n_spatial_units,
                                     tau_grid[k], rho_grid[k],
                                     adj_row_ptr, adj_col_idx, n_neighbors);
    };
    auto log_prior = [&](const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_car_proper(x, spatial_start, n_spatial_units,
                                             tau_grid[k], rho_grid[k], log_det_Q_rho,
                                             adj_row_ptr, adj_col_idx, n_neighbors);
    };

    Rcpp::List out = run_indexed_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        spatial_start, n_spatial_units, spatial_idx,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        prep, add_prior, log_prior,
        store_Q
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
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int temporal_start = p + n_re_groups;

    auto add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                         const Rcpp::NumericVector& x, int k) {
        tulpa::add_rw1_precision(grad, H, x, temporal_start, n_times, tau_grid[k], cyclic);
    };
    auto log_prior = [&](const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_rw1(x, temporal_start, n_times, tau_grid[k], cyclic);
    };

    Rcpp::List out = run_indexed_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        temporal_start, n_times, temporal_idx,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        [](int) { return true; }, add_prior, log_prior,
        store_Q
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
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int temporal_start = p + n_re_groups;

    auto add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                         const Rcpp::NumericVector& x, int k) {
        tulpa::add_rw2_precision(grad, H, x, temporal_start, n_times, tau_grid[k], false);
    };
    auto log_prior = [&](const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_rw2(x, temporal_start, n_times, tau_grid[k], false);
    };

    Rcpp::List out = run_indexed_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        temporal_start, n_times, temporal_idx,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        [](int) { return true; }, add_prior, log_prior,
        store_Q
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
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int temporal_start = p + n_re_groups;

    // AR1 is proper (full rank) — no sum-to-zero constraint required, but
    // run_indexed_nested_laplace centers the latent block to stabilise
    // identifiability against the intercept (matches RW1/RW2/ICAR).
    auto add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                         const Rcpp::NumericVector& x, int k) {
        tulpa::add_ar1_precision(grad, H, x, temporal_start, n_times,
                                  tau_grid[k], rho_grid[k]);
    };
    auto log_prior = [&](const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_ar1(x, temporal_start, n_times,
                                      tau_grid[k], rho_grid[k]);
    };

    Rcpp::List out = run_indexed_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        temporal_start, n_times, temporal_idx,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        [](int) { return true; }, add_prior, log_prior,
        store_Q
    );
    out["tau_grid"] = tau_grid;
    out["rho_grid"] = rho_grid;
    return out;
}

// =====================================================================
// Nested Laplace: NNGP (2D grid over (sigma2_gp, phi_gp))
// =====================================================================
// Continuous-spatial GP via nearest-neighbor conditional decomposition.
// Inner Newton matches laplace_mode_gp() in laplace_core.cpp; warm-start
// across grid points via run_nested_laplace_grid. cov_type: 0
// exponential, 1 Matern-3/2, 2 Matern-5/2 (matches nngp_cov_gpu).

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_nngp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int n_spatial, int nn,
    Rcpp::NumericVector sigma2_grid, Rcpp::NumericVector phi_gp_grid,
    int cov_type,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue
) {
    int n_grid = sigma2_grid.size();
    if (phi_gp_grid.size() != n_grid)
        Rcpp::stop("sigma2_grid and phi_gp_grid must have the same length");
    int N = y.size();
    int p = X.ncol();
    int n_x = p + n_re_groups + n_spatial;
    int gp_start = p + n_re_groups;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

    tulpa::SparseCholeskySolver shared_solver;

    auto solve_at_theta = [&](int k, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        double sigma2_k = sigma2_grid[k];
        double phi_gp_k = phi_gp_grid[k];

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
                if (i < n_spatial) eta[i] += x[gp_start + i];
            }
        };

        auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                           tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            tulpa::scatter_obs_grad_hess_base(y, n, X, re_idx, N, p, n_re_groups,
                                                eta, family, phi, grad, H, n_threads);
            for (int i = 0; i < N; i++) {
                if (i >= n_spatial) continue;
                auto gh = tulpa::grad_hess_for_family(y[i], n[i], eta[i], family, phi);
                int gp_idx = gp_start + i;
                grad[gp_idx] += gh.grad;
                H[gp_idx][gp_idx] += gh.neg_hess;
                for (int j = 0; j < p; j++) {
                    H[gp_idx][j] += gh.neg_hess * X(i, j);
                    H[j][gp_idx] += gh.neg_hess * X(i, j);
                }
                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) {
                        H[p + g][gp_idx] += gh.neg_hess;
                        H[gp_idx][p + g] += gh.neg_hess;
                    }
                }
            }
            // NNGP prior: diagonal-on-w approximation, matching laplace_mode_gp().
            std::vector<double> w(n_spatial);
            for (int s = 0; s < n_spatial; s++) w[s] = x[gp_start + s];
            std::vector<double> cm, cv;
            bool gpu_used;
            tulpa::batch_nngp_scatter(w, n_spatial, nn, sigma2_k, phi_gp_k, cov_type,
                                        coords, nn_idx, nn_dist, nn_order,
                                        cm, cv, gpu_used);
            for (int s = 0; s < n_spatial; s++) {
                int gp_idx = gp_start + s;
                double tau_cond = 1.0 / cv[s];
                grad[gp_idx] -= tau_cond * (w[s] - cm[s]);
                H[gp_idx][gp_idx] += tau_cond;
            }
            tulpa::add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [](Rcpp::NumericVector&) {};

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            std::vector<double> w(n_spatial);
            for (int s = 0; s < n_spatial; s++) w[s] = x[gp_start + s];
            std::vector<double> cm, cv;
            bool gpu_used;
            tulpa::batch_nngp_scatter(w, n_spatial, nn, sigma2_k, phi_gp_k, cov_type,
                                        coords, nn_idx, nn_dist, nn_order,
                                        cm, cv, gpu_used);
            for (int s = 0; s < n_spatial; s++) {
                double resid = w[s] - cm[s];
                lp += -0.5 * std::log(2.0 * M_PI * cv[s]) -
                       0.5 * resid * resid / cv[s];
            }
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
    out["sigma2_grid"] = sigma2_grid;
    out["phi_gp_grid"] = phi_gp_grid;
    return out;
}

// =====================================================================
// Nested Laplace: HSGP (2D grid over (sigma2, lengthscale))
// =====================================================================
// Hilbert-space GP: f_i = sum_j Phi_ij * sqrt(S(lambda_j; sigma2, ell)) * beta_j
// with beta ~ N(0, I_M). Phi (basis) and lambda_j (eigenvalues) are FIXED
// across the grid; only the spectral diagonal sqrt(S) varies per grid
// point. Inner solve reduces to a fixed-design regression on the
// rescaled basis B = Phi * diag(sqrt(S)). spectral_density_se comes from
// hmc_hsgp_kernels.h (Eigen-free split out of hmc_hsgp.h).

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_hsgp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix phi_basis,
    Rcpp::NumericVector lambda_eig,
    Rcpp::NumericVector sigma2_grid,
    Rcpp::NumericVector lengthscale_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = sigma2_grid.size();
    if (lengthscale_grid.size() != n_grid)
        Rcpp::stop("sigma2_grid and lengthscale_grid must have the same length");
    int N = y.size();
    int p = X.ncol();
    int M = phi_basis.ncol();
    if (phi_basis.nrow() != N)
        Rcpp::stop("phi_basis must have N rows (one per observation)");
    if (lambda_eig.size() != M)
        Rcpp::stop("lambda_eig must have length ncol(phi_basis)");
    int n_x = p + n_re_groups + M;
    int beta_gp_start = p + n_re_groups;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

    tulpa::SparseCholeskySolver shared_solver;

    auto solve_at_theta = [&](int k, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        double sigma2_k = sigma2_grid[k];
        double ell_k = lengthscale_grid[k];

        std::vector<double> sqrt_S(M);
        for (int j = 0; j < M; j++) {
            double S = tulpa_hsgp::spectral_density_se(lambda_eig[j], sigma2_k, ell_k);
            sqrt_S[j] = std::sqrt(std::max(S, 1e-30));
        }

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
            #endif
            for (int i = 0; i < N; i++) {
                double e = 0.0;
                for (int j = 0; j < p; j++) e += X(i, j) * x[j];
                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) e += x[p + g];
                }
                for (int j = 0; j < M; j++) {
                    e += phi_basis(i, j) * sqrt_S[j] * x[beta_gp_start + j];
                }
                eta[i] = e;
            }
        };

        auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                           tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            tulpa::scatter_obs_grad_hess_base(y, n, X, re_idx, N, p, n_re_groups,
                                                eta, family, phi, grad, H, n_threads);
            for (int i = 0; i < N; i++) {
                auto gh = tulpa::grad_hess_for_family(y[i], n[i], eta[i], family, phi);
                for (int j = 0; j < M; j++) {
                    double bij = phi_basis(i, j) * sqrt_S[j];
                    int idx_j = beta_gp_start + j;
                    grad[idx_j] += gh.grad * bij;
                    H[idx_j][idx_j] += gh.neg_hess * bij * bij;
                    for (int kk = j + 1; kk < M; kk++) {
                        double bik = phi_basis(i, kk) * sqrt_S[kk];
                        double cross = gh.neg_hess * bij * bik;
                        H[idx_j][beta_gp_start + kk] += cross;
                        H[beta_gp_start + kk][idx_j] += cross;
                    }
                    for (int jj = 0; jj < p; jj++) {
                        double cross = gh.neg_hess * bij * X(i, jj);
                        H[idx_j][jj] += cross;
                        H[jj][idx_j] += cross;
                    }
                    if (n_re_groups > 0) {
                        int g = (int)re_idx[i] - 1;
                        if (g >= 0 && g < n_re_groups) {
                            double cross = gh.neg_hess * bij;
                            H[idx_j][p + g] += cross;
                            H[p + g][idx_j] += cross;
                        }
                    }
                }
            }
            // beta_gp prior N(0, I): grad -= beta_gp; H_diag += 1.
            for (int j = 0; j < M; j++) {
                int idx = beta_gp_start + j;
                grad[idx] -= x[idx];
                H[idx][idx] += 1.0;
            }
            tulpa::add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [](Rcpp::NumericVector&) {};

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            double q = 0.0;
            for (int j = 0; j < M; j++) {
                double b = x[beta_gp_start + j];
                q += b * b;
            }
            lp += -0.5 * q - 0.5 * M * std::log(2.0 * M_PI);
            return lp;
        };

        return tulpa::laplace_newton_solve(
            y, n, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter, center, log_prior,
            prev_mode, &shared_solver,
            store_Q
        );
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    Rcpp::List out = tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, /*store_modes=*/true
    );
    out["sigma2_grid"] = sigma2_grid;
    out["lengthscale_grid"] = lengthscale_grid;
    return out;
}

// =====================================================================
// Joint (spatial × temporal) nested Laplace — shared helpers
// =====================================================================
// Each observation picks one entry from a spatial latent block at
// s_start + s_idx[i] - 1 AND one entry from a temporal latent block at
// t_start + t_idx[i] - 1. Both are first-class in the joint Hessian: the
// inner Newton operates over [beta] [re] [w_s (n_s)] [w_t (n_t)] and the
// cross-block H[w_s, w_t] is non-zero for every obs that lands on a valid
// (s, t) pair. The off-diagonal cross-block is the reason the two latent
// fields cannot be Laplace-marginalized separately — the math forces a
// joint inner solve at every grid point.
//
// The hyperparameter grid is supplied caller-side as paired vectors of
// length n_grid; entry k is one (θ_s_k, θ_t_k) tuple. The Cartesian
// product is built on the R side (matches the AR1 / CAR_proper /
// BYM2 / HSGP convention).

namespace {

inline void nl_compute_eta_two_indexed(
    const Rcpp::NumericVector& x,
    Rcpp::NumericVector& eta,
    int N, int p, int n_re_groups,
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& re_idx,
    int s_start, int n_s, const Rcpp::IntegerVector& s_idx,
    int t_start, int n_t, const Rcpp::IntegerVector& t_idx,
    int n_threads
) {
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
        int s = s_idx[i] - 1;
        if (s >= 0 && s < n_s) eta[i] += x[s_start + s];
        int t = t_idx[i] - 1;
        if (t >= 0 && t < n_t) eta[i] += x[t_start + t];
    }
}

// Joint-block scatter. Calls scatter_obs_grad_hess_base for the (β, re)
// blocks, then walks observations once adding each latent block AND the
// cross H[w_s, w_t] in the same pass.
inline void nl_scatter_obs_two_indexed(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups,
    const Rcpp::NumericVector& eta,
    const std::string& family, double phi,
    int s_start, int n_s, const Rcpp::IntegerVector& s_idx,
    int t_start, int n_t, const Rcpp::IntegerVector& t_idx,
    tulpa::DenseVec& grad, tulpa::DenseMat& H,
    int n_threads
) {
    tulpa::scatter_obs_grad_hess_base(y, n_trials, X, re_idx, N, p, n_re_groups,
                                       eta, family, phi, grad, H, n_threads);

    for (int i = 0; i < N; i++) {
        int s = s_idx[i] - 1;
        int t = t_idx[i] - 1;
        bool has_s = (s >= 0 && s < n_s);
        bool has_t = (t >= 0 && t < n_t);
        if (!has_s && !has_t) continue;

        auto gh = tulpa::grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi);

        if (has_s) {
            int idx_s = s_start + s;
            grad[idx_s] += gh.grad;
            H[idx_s][idx_s] += gh.neg_hess;
            for (int j = 0; j < p; j++) {
                H[j][idx_s]     += gh.neg_hess * X(i, j);
                H[idx_s][j]     += gh.neg_hess * X(i, j);
            }
            if (n_re_groups > 0) {
                int g = (int)re_idx[i] - 1;
                if (g >= 0 && g < n_re_groups) {
                    H[p + g][idx_s] += gh.neg_hess;
                    H[idx_s][p + g] += gh.neg_hess;
                }
            }
        }
        if (has_t) {
            int idx_t = t_start + t;
            grad[idx_t] += gh.grad;
            H[idx_t][idx_t] += gh.neg_hess;
            for (int j = 0; j < p; j++) {
                H[j][idx_t]     += gh.neg_hess * X(i, j);
                H[idx_t][j]     += gh.neg_hess * X(i, j);
            }
            if (n_re_groups > 0) {
                int g = (int)re_idx[i] - 1;
                if (g >= 0 && g < n_re_groups) {
                    H[p + g][idx_t] += gh.neg_hess;
                    H[idx_t][p + g] += gh.neg_hess;
                }
            }
        }
        if (has_s && has_t) {
            int idx_s = s_start + s;
            int idx_t = t_start + t;
            H[idx_s][idx_t] += gh.neg_hess;
            H[idx_t][idx_s] += gh.neg_hess;
        }
    }
}

// Generic outer-grid driver for joint indexed-spatial × indexed-temporal
// nested Laplace. Mirrors run_indexed_nested_laplace but with two latent
// blocks. Each block has its own (prep_at_k, add_prior_at_k, log_prior_at_k)
// callbacks. n_grid pairs the two hyperparameter trajectories — caller
// builds the Cartesian product of the per-axis grids before calling.
template<typename PrepSFn, typename AddPriorSFn, typename LogPriorSFn,
         typename PrepTFn, typename AddPriorTFn, typename LogPriorTFn>
inline Rcpp::List run_two_indexed_nested_laplace(
    int n_grid,
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    int s_start, int n_s, const Rcpp::IntegerVector& s_idx,
    int t_start, int n_t, const Rcpp::IntegerVector& t_idx,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    bool store_modes,
    const Rcpp::NumericVector& x_init,
    PrepSFn prep_s_at_k,
    AddPriorSFn add_prior_s_at_k,
    LogPriorSFn log_prior_s_at_k,
    PrepTFn prep_t_at_k,
    AddPriorTFn add_prior_t_at_k,
    LogPriorTFn log_prior_t_at_k,
    bool store_Q = false
) {
    int n_x = p + n_re_groups + n_s + n_t;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    tulpa::SparseCholeskySolver shared_solver;

    auto solve_at_theta = [&](int k, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        if (!prep_s_at_k(k) || !prep_t_at_k(k)) {
            tulpa::LaplaceResult bad;
            bad.mode = (prev_mode.size() == n_x) ? prev_mode :
                       Rcpp::NumericVector(n_x, 0.0);
            bad.log_marginal = -std::numeric_limits<double>::infinity();
            bad.n_iter = 0;
            bad.converged = false;
            bad.log_det_Q = 0.0;
            return bad;
        }

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            nl_compute_eta_two_indexed(x, eta, N, p, n_re_groups, X, re_idx,
                                        s_start, n_s, s_idx,
                                        t_start, n_t, t_idx,
                                        n_threads);
        };

        auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                           tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            nl_scatter_obs_two_indexed(y, n_trials, X, re_idx, N, p, n_re_groups,
                                        eta, family, phi,
                                        s_start, n_s, s_idx,
                                        t_start, n_t, t_idx,
                                        grad, H, n_threads);
            add_prior_s_at_k(grad, H, x, k);
            add_prior_t_at_k(grad, H, x, k);
            tulpa::add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            tulpa::center_effects(x, s_start, n_s);
            tulpa::center_effects(x, t_start, n_t);
        };

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            lp += log_prior_s_at_k(x, k);
            lp += log_prior_t_at_k(x, k);
            return lp;
        };

        return tulpa::laplace_newton_solve(
            y, n_trials, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter, center, log_prior,
            prev_mode, &shared_solver, store_Q
        );
    };

    return tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, store_modes
    );
}

// ---- per-kind prior callbacks ---------------------------------------------
// Bundle of (prep_at_k, add_prior_at_k, log_prior_at_k) for one indexed
// latent block. Adding a new (spatial_kind × temporal_kind) ST combination
// is now a matter of picking the matching pair of factories below — the
// shared joint driver (run_two_indexed_nested_laplace) is parameter-free
// in the prior shape.
//
// The lambdas capture the per-kind state (grids, adjacency, basis, ...)
// by reference. The caller (the Rcpp entry function) is responsible for
// keeping those references alive: the Rcpp::Vector / Rcpp::IntegerVector
// arguments live on the entry's stack and outlive the ops bundle.

struct IndexedPriorOps {
    std::function<bool(int)> prep;
    std::function<void(tulpa::DenseVec&, tulpa::DenseMat&,
                       const Rcpp::NumericVector&, int)> add_prior;
    std::function<double(const Rcpp::NumericVector&, int)> log_prior;
};

// ICAR — 1D τ_grid over a spatial adjacency. Stateless prep.
inline IndexedPriorOps make_icar_ops(
    int start, int n_units,
    const Rcpp::NumericVector& tau_grid,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
) {
    IndexedPriorOps ops;
    ops.prep = [](int) { return true; };
    ops.add_prior = [start, n_units, &tau_grid, &adj_row_ptr,
                     &adj_col_idx, &n_neighbors]
                    (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                     const Rcpp::NumericVector& x, int k) {
        tulpa::add_icar_prior(grad, H, x, start, n_units, tau_grid[k],
                               adj_row_ptr, adj_col_idx, n_neighbors);
    };
    ops.log_prior = [start, n_units, &tau_grid, &adj_row_ptr,
                     &adj_col_idx, &n_neighbors]
                    (const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_icar(x, start, n_units, tau_grid[k],
                                       adj_row_ptr, adj_col_idx, n_neighbors);
    };
    return ops;
}

// CAR_proper — 2D (τ, ρ) grid. Prep recomputes log|Q(ρ_k)| per grid point
// and shares it with log_prior via shared_ptr (so the lambdas can be
// returned out of the factory). Caller-side ownership of adj_row_ptr et
// al. via reference capture; ρ-dependent CSR copies are owned by shared
// ptrs.
inline IndexedPriorOps make_car_proper_ops(
    int start, int n_units,
    const Rcpp::NumericVector& tau_grid,
    const Rcpp::NumericVector& rho_grid,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
) {
    auto adj_rp_v = std::make_shared<std::vector<int>>(
        adj_row_ptr.begin(), adj_row_ptr.end());
    auto adj_ci_v = std::make_shared<std::vector<int>>(
        adj_col_idx.begin(), adj_col_idx.end());
    auto n_nbr_v  = std::make_shared<std::vector<int>>(
        n_neighbors.begin(),  n_neighbors.end());
    auto log_det_Q_rho = std::make_shared<double>(0.0);

    IndexedPriorOps ops;
    ops.prep = [n_units, &rho_grid, adj_rp_v, adj_ci_v, n_nbr_v, log_det_Q_rho]
               (int k) -> bool {
        std::vector<double> Qmat = tulpa_car_proper::compute_car_precision(
            n_units, *adj_rp_v, *adj_ci_v, *n_nbr_v, rho_grid[k]);
        *log_det_Q_rho = tulpa_car_proper::car_log_det(n_units, Qmat);
        return std::isfinite(*log_det_Q_rho);
    };
    ops.add_prior = [start, n_units, &tau_grid, &rho_grid,
                     &adj_row_ptr, &adj_col_idx, &n_neighbors]
                    (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                     const Rcpp::NumericVector& x, int k) {
        tulpa::add_car_proper_prior(grad, H, x, start, n_units,
                                     tau_grid[k], rho_grid[k],
                                     adj_row_ptr, adj_col_idx, n_neighbors);
    };
    ops.log_prior = [start, n_units, &tau_grid, &rho_grid,
                     &adj_row_ptr, &adj_col_idx, &n_neighbors, log_det_Q_rho]
                    (const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_car_proper(x, start, n_units,
                                             tau_grid[k], rho_grid[k],
                                             *log_det_Q_rho,
                                             adj_row_ptr, adj_col_idx, n_neighbors);
    };
    return ops;
}

// RW1 — 1D τ grid; cyclic flag closes the chain.
inline IndexedPriorOps make_rw1_ops(
    int start, int n_units,
    const Rcpp::NumericVector& tau_grid,
    bool cyclic
) {
    IndexedPriorOps ops;
    ops.prep = [](int) { return true; };
    ops.add_prior = [start, n_units, &tau_grid, cyclic]
                    (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                     const Rcpp::NumericVector& x, int k) {
        tulpa::add_rw1_precision(grad, H, x, start, n_units, tau_grid[k], cyclic);
    };
    ops.log_prior = [start, n_units, &tau_grid, cyclic]
                    (const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_rw1(x, start, n_units, tau_grid[k], cyclic);
    };
    return ops;
}

// RW2 — 1D τ grid; no cyclic flag (the RW2 implementation ignores it).
inline IndexedPriorOps make_rw2_ops(
    int start, int n_units,
    const Rcpp::NumericVector& tau_grid
) {
    IndexedPriorOps ops;
    ops.prep = [](int) { return true; };
    ops.add_prior = [start, n_units, &tau_grid]
                    (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                     const Rcpp::NumericVector& x, int k) {
        tulpa::add_rw2_precision(grad, H, x, start, n_units, tau_grid[k], false);
    };
    ops.log_prior = [start, n_units, &tau_grid]
                    (const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_rw2(x, start, n_units, tau_grid[k], false);
    };
    return ops;
}

// AR1 — 2D (τ, ρ) grid.
inline IndexedPriorOps make_ar1_ops(
    int start, int n_units,
    const Rcpp::NumericVector& tau_grid,
    const Rcpp::NumericVector& rho_grid
) {
    IndexedPriorOps ops;
    ops.prep = [](int) { return true; };
    ops.add_prior = [start, n_units, &tau_grid, &rho_grid]
                    (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                     const Rcpp::NumericVector& x, int k) {
        tulpa::add_ar1_precision(grad, H, x, start, n_units,
                                  tau_grid[k], rho_grid[k]);
    };
    ops.log_prior = [start, n_units, &tau_grid, &rho_grid]
                    (const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_ar1(x, start, n_units, tau_grid[k], rho_grid[k]);
    };
    return ops;
}

} // namespace

// =====================================================================
// Nested Laplace: ICAR (spatial) × AR1 (temporal)
//   Joint over (τ_s) × (τ_t, ρ_t). Caller passes paired vectors of length
//   n_grid (the Cartesian product is built R-side).
//   Latent: [beta (p)] [re (n_re_groups)] [w_spatial (n_s)] [w_temporal (n_t)].
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_icar_ar1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    Rcpp::NumericVector rho_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_spatial_grid.size();
    if (tau_temporal_grid.size() != n_grid || rho_temporal_grid.size() != n_grid) {
        Rcpp::stop("tau_spatial_grid, tau_temporal_grid, rho_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial_units;

    auto ops_s = make_icar_ops(s_start, n_spatial_units, tau_spatial_grid,
                                adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_ar1_ops(t_start, n_times, tau_temporal_grid, rho_temporal_grid);

    Rcpp::List out = run_two_indexed_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        s_start, n_spatial_units, spatial_idx,
        t_start, n_times,            temporal_idx,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        ops_s.prep, ops_s.add_prior, ops_s.log_prior,
        ops_t.prep, ops_t.add_prior, ops_t.log_prior,
        store_Q
    );
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    out["rho_temporal_grid"] = rho_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: ICAR (spatial) × RW1 (temporal)
//   1D τ_s × 1D τ_t grid. Caller passes paired vectors of length n_grid.
//   Latent: [beta] [re] [w_spatial (n_s)] [w_temporal (n_t)].
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_icar_rw1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times, bool cyclic,
    Rcpp::NumericVector tau_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_spatial_grid.size();
    if (tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("tau_spatial_grid and tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial_units;

    auto ops_s = make_icar_ops(s_start, n_spatial_units, tau_spatial_grid,
                                adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_rw1_ops(t_start, n_times, tau_temporal_grid, cyclic);

    Rcpp::List out = run_two_indexed_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        s_start, n_spatial_units, spatial_idx,
        t_start, n_times,            temporal_idx,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        ops_s.prep, ops_s.add_prior, ops_s.log_prior,
        ops_t.prep, ops_t.add_prior, ops_t.log_prior,
        store_Q
    );
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: ICAR (spatial) × RW2 (temporal)
//   1D τ_s × 1D τ_t grid.
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_icar_rw2(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_spatial_grid.size();
    if (tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("tau_spatial_grid and tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial_units;

    auto ops_s = make_icar_ops(s_start, n_spatial_units, tau_spatial_grid,
                                adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_rw2_ops(t_start, n_times, tau_temporal_grid);

    Rcpp::List out = run_two_indexed_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        s_start, n_spatial_units, spatial_idx,
        t_start, n_times,            temporal_idx,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        ops_s.prep, ops_s.add_prior, ops_s.log_prior,
        ops_t.prep, ops_t.add_prior, ops_t.log_prior,
        store_Q
    );
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: CAR_proper (spatial) × RW1 (temporal)
//   2D (τ_s, ρ_s) × 1D τ_t grid.
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_car_proper_rw1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times, bool cyclic,
    Rcpp::NumericVector tau_spatial_grid,
    Rcpp::NumericVector rho_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_spatial_grid.size();
    if (rho_spatial_grid.size() != n_grid || tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("tau_spatial_grid, rho_spatial_grid, tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial_units;

    auto ops_s = make_car_proper_ops(s_start, n_spatial_units,
                                      tau_spatial_grid, rho_spatial_grid,
                                      adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_rw1_ops(t_start, n_times, tau_temporal_grid, cyclic);

    Rcpp::List out = run_two_indexed_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        s_start, n_spatial_units, spatial_idx,
        t_start, n_times,            temporal_idx,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        ops_s.prep, ops_s.add_prior, ops_s.log_prior,
        ops_t.prep, ops_t.add_prior, ops_t.log_prior,
        store_Q
    );
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["rho_spatial_grid"]  = rho_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: CAR_proper (spatial) × RW2 (temporal)
//   2D (τ_s, ρ_s) × 1D τ_t grid.
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_car_proper_rw2(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_spatial_grid,
    Rcpp::NumericVector rho_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_spatial_grid.size();
    if (rho_spatial_grid.size() != n_grid || tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("tau_spatial_grid, rho_spatial_grid, tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial_units;

    auto ops_s = make_car_proper_ops(s_start, n_spatial_units,
                                      tau_spatial_grid, rho_spatial_grid,
                                      adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_rw2_ops(t_start, n_times, tau_temporal_grid);

    Rcpp::List out = run_two_indexed_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        s_start, n_spatial_units, spatial_idx,
        t_start, n_times,            temporal_idx,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        ops_s.prep, ops_s.add_prior, ops_s.log_prior,
        ops_t.prep, ops_t.add_prior, ops_t.log_prior,
        store_Q
    );
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["rho_spatial_grid"]  = rho_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: CAR_proper (spatial) × AR1 (temporal)
//   2D (τ_s, ρ_s) × 2D (τ_t, ρ_t) grid.
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_car_proper_ar1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_spatial_grid,
    Rcpp::NumericVector rho_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    Rcpp::NumericVector rho_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_spatial_grid.size();
    if (rho_spatial_grid.size() != n_grid ||
        tau_temporal_grid.size() != n_grid ||
        rho_temporal_grid.size() != n_grid) {
        Rcpp::stop("All four paired grids must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial_units;

    auto ops_s = make_car_proper_ops(s_start, n_spatial_units,
                                      tau_spatial_grid, rho_spatial_grid,
                                      adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_ar1_ops(t_start, n_times, tau_temporal_grid, rho_temporal_grid);

    Rcpp::List out = run_two_indexed_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        s_start, n_spatial_units, spatial_idx,
        t_start, n_times,            temporal_idx,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        ops_s.prep, ops_s.add_prior, ops_s.log_prior,
        ops_t.prep, ops_t.add_prior, ops_t.log_prior,
        store_Q
    );
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["rho_spatial_grid"]  = rho_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    out["rho_temporal_grid"] = rho_temporal_grid;
    return out;
}

