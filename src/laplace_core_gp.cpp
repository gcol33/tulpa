// laplace_core_gp.cpp
// GP (NNGP) / Multiscale GP / Multiscale temporal Laplace mode finders + their
// R exports. Split from laplace_core.cpp on 2026-05-02.

#include "laplace_core.h"
#include "laplace_cholesky.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include "laplace_scatter.h"
#include "laplace_temporal_priors.h"
#include "linalg_fast.h"
#include "gpu_nngp_laplace.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace tulpa {

// --- 4. GP (NNGP) ---
// Uses fully sparse Newton when n_spatial >= SPARSE_THRESHOLD (200).
// The H sparsity pattern comes from: beta×beta (dense p×p), spatial diagonal
// (likelihood), and NNGP neighbor pairs (prior). Total nnz ≈ p² + n_spatial × nn.

LaplaceResult laplace_mode_gp(
    const NumericVector& y, const IntegerVector& n,
    const NumericMatrix& X, const NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const NumericMatrix& coords,
    const IntegerMatrix& nn_idx, const NumericMatrix& nn_dist,
    const IntegerVector& nn_order,
    int n_spatial, int nn,
    double sigma2_gp, double phi_gp, int cov_type,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads
) {
    int N = y.size();
    int p = X.ncol();
    int n_x = p + n_re_groups + n_spatial;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    int gp_start = p + n_re_groups;

    auto compute_eta = [&](const NumericVector& x, NumericVector& eta) {
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

    auto log_prior = [&](const NumericVector& x, const NumericVector&) {
        double lp = compute_log_prior_re(x, p, n_re_groups, tau_re);
        std::vector<double> w(n_spatial);
        for (int s = 0; s < n_spatial; s++) w[s] = x[gp_start + s];
        std::vector<double> cm, cv;
        bool gpu;
        batch_nngp_scatter(w, n_spatial, nn, sigma2_gp, phi_gp, cov_type,
                           coords, nn_idx, nn_dist, nn_order, cm, cv, gpu);
        for (int s = 0; s < n_spatial; s++) {
            double resid = w[s] - cm[s];
            lp += -0.5 * std::log(2.0 * M_PI * cv[s]) -
                  0.5 * resid * resid / cv[s];
        }
        return lp;
    };

    auto center = [](NumericVector&) {};

    // --- Sparse Newton path for large n_spatial ---
    if (n_x >= SPARSE_THRESHOLD) {
        // Build sparsity pattern: beta block + RE diagonal + GP diagonal + NNGP neighbors
        std::vector<std::pair<int,int>> pattern;

        // Beta-beta dense block
        for (int j1 = 0; j1 < p; j1++)
            for (int j2 = j1; j2 < p; j2++)
                pattern.push_back({j2, j1});

        // RE diagonal
        for (int g = 0; g < n_re_groups; g++)
            pattern.push_back({p + g, p + g});

        // Beta-GP cross terms (each obs contributes to its GP node × beta)
        for (int i = 0; i < std::min(N, n_spatial); i++) {
            int gp_idx = gp_start + i;
            for (int j = 0; j < p; j++)
                pattern.push_back({gp_idx, j});
        }

        // GP diagonal (likelihood + prior)
        for (int s = 0; s < n_spatial; s++)
            pattern.push_back({gp_start + s, gp_start + s});

        // NNGP neighbor pairs (from the conditional prior — these are the
        // off-diagonal entries that make H sparse rather than diagonal in the GP block)
        // Note: NNGP conditional prior only contributes to the diagonal, not off-diagonal.
        // The Hessian is diagonal in the GP block for NNGP (each w_i's Hessian
        // contribution is independent). So no additional off-diagonal needed.

        SparseHessianBuilder H_builder;
        H_builder.init(n_x, pattern);

        auto scatter_sparse = [&](const NumericVector& x, const NumericVector& eta,
                                   DenseVec& grad, SparseHessianBuilder& H) {
            // Fixed effects
            for (int i = 0; i < N; i++) {
                auto gh = grad_hess_for_family(y[i], n[i], eta[i], family, phi);
                for (int j = 0; j < p; j++) {
                    grad[j] += gh.grad * X(i, j);
                    for (int k = 0; k <= j; k++) {
                        H.add(j, k, gh.neg_hess * X(i, j) * X(i, k));
                    }
                }
                // RE
                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) {
                        grad[p + g] += gh.grad;
                        H.add(p + g, p + g, gh.neg_hess);
                    }
                }
                // GP diagonal
                if (i < n_spatial) {
                    int gp_idx = gp_start + i;
                    grad[gp_idx] += gh.grad;
                    H.add(gp_idx, gp_idx, gh.neg_hess);
                    // Cross with beta
                    for (int j = 0; j < p; j++) {
                        H.add(gp_idx, j, gh.neg_hess * X(i, j));
                    }
                }
            }

            // NNGP prior (diagonal contribution only)
            std::vector<double> w(n_spatial);
            for (int s = 0; s < n_spatial; s++) w[s] = x[gp_start + s];
            std::vector<double> cond_means, cond_vars;
            bool gpu_used;
            batch_nngp_scatter(w, n_spatial, nn, sigma2_gp, phi_gp, cov_type,
                               coords, nn_idx, nn_dist, nn_order,
                               cond_means, cond_vars, gpu_used);
            for (int s = 0; s < n_spatial; s++) {
                int gp_idx = gp_start + s;
                double tau_cond = 1.0 / cond_vars[s];
                grad[gp_idx] -= tau_cond * (w[s] - cond_means[s]);
                H.add(gp_idx, gp_idx, tau_cond);
            }

            // Beta + RE regularization
            double tau_beta = 1e-4;
            for (int j = 0; j < p; j++) { grad[j] -= tau_beta * x[j]; H.add(j, j, tau_beta); }
            for (int g = 0; g < n_re_groups; g++) {
                grad[p + g] -= tau_re * x[p + g]; H.add(p + g, p + g, tau_re);
            }
        };

        return laplace_newton_solve_sparse(
            y, n, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter_sparse, center, log_prior,
            H_builder);
    }

    // --- Dense Newton path for small n_spatial ---
    auto scatter = [&](const NumericVector& x, const NumericVector& eta,
                       DenseVec& grad, DenseMat& H) {
        scatter_obs_grad_hess_base(y, n, X, re_idx, N, p, n_re_groups,
                                    eta, family, phi, grad, H, n_threads);
        for (int i = 0; i < N; i++) {
            if (i >= n_spatial) continue;
            auto gh = grad_hess_for_family(y[i], n[i], eta[i], family, phi);
            int gp_idx = gp_start + i;
            grad[gp_idx] += gh.grad;
            H[gp_idx][gp_idx] += gh.neg_hess;
        }
        std::vector<double> w(n_spatial);
        for (int s = 0; s < n_spatial; s++) w[s] = x[gp_start + s];
        std::vector<double> cond_means, cond_vars;
        bool gpu_used;
        batch_nngp_scatter(w, n_spatial, nn, sigma2_gp, phi_gp, cov_type,
                           coords, nn_idx, nn_dist, nn_order,
                           cond_means, cond_vars, gpu_used);
        for (int s = 0; s < n_spatial; s++) {
            int gp_idx = gp_start + s;
            double tau_cond = 1.0 / cond_vars[s];
            grad[gp_idx] -= tau_cond * (w[s] - cond_means[s]);
            H[gp_idx][gp_idx] += tau_cond;
        }
        add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
    };

    return laplace_newton_solve(y, n, family, phi, N, n_x,
                                 max_iter, tol, n_threads,
                                 compute_eta, scatter, center, log_prior);
}


// --- 5. Multiscale GP (local + regional) ---

LaplaceResult laplace_mode_multiscale_gp(
    const NumericVector& y, const IntegerVector& n,
    const NumericMatrix& X, const NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const NumericMatrix& coords,
    const IntegerMatrix& nn_idx_local, const NumericMatrix& nn_dist_local,
    const IntegerVector& nn_order_local, int nn_local,
    const IntegerMatrix& nn_idx_regional, const NumericMatrix& nn_dist_regional,
    const IntegerVector& nn_order_regional, int nn_regional,
    int n_spatial,
    double sigma2_local, double phi_local,
    double sigma2_regional, double phi_regional,
    int cov_type, const std::string& family, double phi,
    int max_iter, double tol, int n_threads
) {
    int N = y.size();
    int p = X.ncol();
    int local_start = p + n_re_groups;
    int regional_start = local_start + n_spatial;
    int n_x = regional_start + n_spatial;
    double tau_re = (sigma_re > 0) ? 1.0 / (sigma_re * sigma_re) : 0.01;

    auto compute_eta = [&](const NumericVector& x, NumericVector& eta) {
        for (int i = 0; i < N; i++) {
            eta[i] = 0.0;
            for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
            if (n_re_groups > 0) {
                int g = (int)re_idx[i] - 1;
                if (g >= 0 && g < n_re_groups) eta[i] += x[p + g];
            }
            if (i < n_spatial) {
                eta[i] += x[local_start + i];
                eta[i] += x[regional_start + i];
            }
        }
    };

    auto scatter = [&](const NumericVector& x, const NumericVector& eta,
                       DenseVec& grad, DenseMat& H) {
        // Base scatter
        scatter_obs_grad_hess_base(y, n, X, re_idx, N, p, n_re_groups,
                                    eta, family, phi, grad, H, n_threads);

        // Local + regional GP scatter
        for (int i = 0; i < N; i++) {
            if (i >= n_spatial) continue;
            auto gh = grad_hess_for_family(y[i], n[i], eta[i], family, phi);
            int idx_local = local_start + i;
            int idx_regional = regional_start + i;
            grad[idx_local] += gh.grad;
            grad[idx_regional] += gh.grad;
            H[idx_local][idx_local] += gh.neg_hess;
            H[idx_regional][idx_regional] += gh.neg_hess;
            H[idx_local][idx_regional] += gh.neg_hess;
            H[idx_regional][idx_local] += gh.neg_hess;
        }

        // GP priors - local (simplified NNGP diagonal)
        double tau_local = 1.0 / sigma2_local;
        for (int i = 0; i < n_spatial; i++) {
            int idx = local_start + i;
            double cond_mean = 0.0;
            int n_nb = 0;
            for (int k = 0; k < nn_local; k++) {
                int neighbor = nn_idx_local(i, k) - 1;
                if (neighbor >= 0 && neighbor < n_spatial) {
                    double dist = nn_dist_local(i, k);
                    double cov_val = std::exp(-dist / phi_local);
                    cond_mean += cov_val * x[local_start + neighbor];
                    n_nb++;
                }
            }
            if (n_nb > 0) cond_mean *= tau_local;
            grad[idx] -= tau_local * x[idx] - cond_mean;
            H[idx][idx] += tau_local;
        }

        // GP priors - regional
        double tau_regional = 1.0 / sigma2_regional;
        for (int i = 0; i < n_spatial; i++) {
            int idx = regional_start + i;
            double cond_mean = 0.0;
            int n_nb = 0;
            for (int k = 0; k < nn_regional; k++) {
                int neighbor = nn_idx_regional(i, k) - 1;
                if (neighbor >= 0 && neighbor < n_spatial) {
                    double dist = nn_dist_regional(i, k);
                    double cov_val = std::exp(-dist / phi_regional);
                    cond_mean += cov_val * x[regional_start + neighbor];
                    n_nb++;
                }
            }
            if (n_nb > 0) cond_mean *= tau_regional;
            grad[idx] -= tau_regional * x[idx] - cond_mean;
            H[idx][idx] += tau_regional;
        }

        add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
    };

    auto center = [&](NumericVector& x) {
        center_effects(x, local_start, n_spatial);
        center_effects(x, regional_start, n_spatial);
    };

    auto log_prior = [&](const NumericVector& x, const NumericVector&) -> double {
        // Simplified: log_marginal is set to -0.5*log_det + 0.5*n*log(2pi)
        // (original code didn't compute full GP log prior for multiscale)
        return 0.0;
    };

    return laplace_newton_solve(y, n, family, phi, N, n_x,
                                 max_iter, tol, n_threads,
                                 compute_eta, scatter, center, log_prior);
}


// --- 6. Multiscale temporal ---

LaplaceResult laplace_mode_multiscale_temporal(
    const NumericVector& y, const IntegerVector& n,
    const NumericMatrix& X, const NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const IntegerVector& time_idx, int n_times,
    int seasonal_period, int trend_type, int short_type,
    double sigma2_trend, double sigma2_seasonal, double sigma2_short, double rho_short,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads
) {
    int N = y.size();
    int p = X.ncol();
    int n_trend = (trend_type > 0) ? n_times : 0;
    int n_seasonal = (seasonal_period > 0) ? seasonal_period : 0;
    int n_short = (short_type > 0) ? n_times : 0;
    int n_x = p + n_re_groups + n_trend + n_seasonal + n_short;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    double tau_trend = 1.0 / (sigma2_trend + 1e-10);
    double tau_seasonal = 1.0 / (sigma2_seasonal + 1e-10);
    double tau_short = 1.0 / (sigma2_short + 1e-10);
    int trend_start = p + n_re_groups;
    int seasonal_start = trend_start + n_trend;
    int short_start = seasonal_start + n_seasonal;

    auto compute_eta = [&](const NumericVector& x, NumericVector& eta) {
        for (int i = 0; i < N; i++) {
            eta[i] = 0.0;
            for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
            if (n_re_groups > 0) {
                int g = (int)re_idx[i] - 1;
                if (g >= 0 && g < n_re_groups) eta[i] += x[p + g];
            }
            int t = time_idx[i] - 1;
            if (t >= 0 && t < n_times) {
                if (n_trend > 0 && t < n_trend) eta[i] += x[trend_start + t];
                if (n_seasonal > 0) {
                    int s = t % seasonal_period;
                    if (s < n_seasonal) eta[i] += x[seasonal_start + s];
                }
                if (n_short > 0 && t < n_short) eta[i] += x[short_start + t];
            }
        }
    };

    auto scatter = [&](const NumericVector& x, const NumericVector& eta,
                       DenseVec& grad, DenseMat& H) {
        // Base scatter
        scatter_obs_grad_hess_base(y, n, X, re_idx, N, p, n_re_groups,
                                    eta, family, phi, grad, H, n_threads);

        // Temporal effect scatter
        for (int i = 0; i < N; i++) {
            auto gh = grad_hess_for_family(y[i], n[i], eta[i], family, phi);
            int t = time_idx[i] - 1;
            if (t < 0 || t >= n_times) continue;

            if (n_trend > 0 && t < n_trend) {
                int idx = trend_start + t;
                grad[idx] += gh.grad;
                H[idx][idx] += gh.neg_hess;
            }
            if (n_seasonal > 0) {
                int s = t % seasonal_period;
                if (s < n_seasonal) {
                    int idx = seasonal_start + s;
                    grad[idx] += gh.grad;
                    H[idx][idx] += gh.neg_hess;
                }
            }
            if (n_short > 0 && t < n_short) {
                int idx = short_start + t;
                grad[idx] += gh.grad;
                H[idx][idx] += gh.neg_hess;
            }
        }

        // Temporal priors
        if (trend_type == 1) {
            add_rw1_precision(grad, H, x, trend_start, n_trend, tau_trend, false);
        } else if (trend_type == 2) {
            add_rw2_precision(grad, H, x, trend_start, n_trend, tau_trend, false);
        }
        if (n_seasonal > 0) {
            add_rw1_precision(grad, H, x, seasonal_start, n_seasonal, tau_seasonal, true);
        }
        if (short_type == 1) {
            add_ar1_precision(grad, H, x, short_start, n_short, tau_short, rho_short);
        } else if (short_type == 2) {
            for (int t = 0; t < n_short; t++) {
                int idx = short_start + t;
                grad[idx] -= tau_short * x[idx];
                H[idx][idx] += tau_short;
            }
        }

        add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
    };

    auto center = [&](NumericVector& x) {
        if (n_trend > 0) center_effects(x, trend_start, n_trend);
        if (n_seasonal > 0) center_effects(x, seasonal_start, n_seasonal);
    };

    auto log_prior = [&](const NumericVector& x, const NumericVector&) -> double {
        // Original code only computed log_det-based log_marginal for temporal
        return 0.0;
    };

    return laplace_newton_solve(y, n, family, phi, N, n_x,
                                 max_iter, tol, n_threads,
                                 compute_eta, scatter, center, log_prior);
}

} // namespace tulpa

// =====================================================================
// R exports (call into tulpa:: functions defined above)
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_gp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int n_spatial, int nn,
    double sigma2_gp, double phi_gp, int cov_type,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1
) {
    tulpa::LaplaceResult result = tulpa::laplace_mode_gp(
        y, n, X, re_idx, n_re_groups, sigma_re,
        coords, nn_idx, nn_dist, nn_order, n_spatial, nn,
        sigma2_gp, phi_gp, cov_type,
        family, phi, max_iter, tol, n_threads
    );
    return tulpa::laplace_result_to_list(result);
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_multiscale_gp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx_local, Rcpp::NumericMatrix nn_dist_local,
    Rcpp::IntegerVector nn_order_local, int nn_local,
    Rcpp::IntegerMatrix nn_idx_regional, Rcpp::NumericMatrix nn_dist_regional,
    Rcpp::IntegerVector nn_order_regional, int nn_regional,
    int n_spatial,
    double sigma2_local, double phi_local,
    double sigma2_regional, double phi_regional,
    int cov_type, std::string family,
    double phi = 1.0, int max_iter = 100, double tol = 1e-6, int n_threads = 1
) {
    tulpa::LaplaceResult result = tulpa::laplace_mode_multiscale_gp(
        y, n, X, re_idx, n_re_groups, sigma_re,
        coords, nn_idx_local, nn_dist_local, nn_order_local, nn_local,
        nn_idx_regional, nn_dist_regional, nn_order_regional, nn_regional,
        n_spatial, sigma2_local, phi_local, sigma2_regional, phi_regional,
        cov_type, family, phi, max_iter, tol, n_threads
    );
    return tulpa::laplace_result_to_list(result);
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_multiscale_temporal(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector time_idx, int n_times,
    int seasonal_period, int trend_type, int short_type,
    double sigma2_trend, double sigma2_seasonal, double sigma2_short, double rho_short,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1
) {
    tulpa::LaplaceResult result = tulpa::laplace_mode_multiscale_temporal(
        y, n, X, re_idx, n_re_groups, sigma_re,
        time_idx, n_times, seasonal_period, trend_type, short_type,
        sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
        family, phi, max_iter, tol, n_threads
    );
    return tulpa::laplace_result_to_list(result);
}
