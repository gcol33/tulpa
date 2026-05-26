// laplace_core_spatial.cpp
// Spatial / BYM2 Laplace mode finders + their R exports.
// Split from laplace_core.cpp on 2026-05-02.
//
// Each mode finder still uses laplace_newton_solve (in laplace_newton.h) and the
// scatter helpers from laplace_scatter.h — see laplace_core.cpp for the shared
// library context.

#include "laplace_core.h"
#include "laplace_cholesky.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include "laplace_scatter.h"
#include "laplace_spatial_priors.h"
#include "linalg_fast.h"
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


// --- 2. Spatial (ICAR) ---

LaplaceResult laplace_mode_spatial(
    const NumericVector& y, const IntegerVector& n,
    const NumericMatrix& X, const NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const IntegerVector& spatial_idx, int n_spatial_units,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors, double tau_spatial,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    const NumericVector& x_init = NumericVector()
) {
    int N = y.size();
    int p = X.ncol();
    int n_x = p + n_re_groups + n_spatial_units;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    int spatial_start = p + n_re_groups;

    auto compute_eta = [&](const NumericVector& x, NumericVector& eta) {
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

    auto scatter = [&](const NumericVector& x, const NumericVector& eta,
                       DenseVec& grad, DenseMat& H) {
        // Build per-obs effect index for spatial block
        std::vector<int> eff_idx(N, -1);
        std::vector<double> d_fac(N, 1.0);
        for (int i = 0; i < N; i++) {
            if (n_spatial_units > 0) {
                int s = spatial_idx[i] - 1;
                if (s >= 0 && s < n_spatial_units) eff_idx[i] = spatial_start + s;
            }
        }
        scatter_obs_with_latent(y, n, X, re_idx, N, p, n_re_groups,
                                 eta, family, phi, eff_idx, d_fac, grad, H, n_threads);
        add_icar_prior(grad, H, x, spatial_start, n_spatial_units, tau_spatial,
                        adj_row_ptr, adj_col_idx, n_neighbors);
        add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
    };

    auto center = [&](NumericVector& x) {
        center_effects(x, spatial_start, n_spatial_units);
    };

    auto log_prior = [&](const NumericVector& x, const NumericVector&) {
        double lp = compute_log_prior_re(x, p, n_re_groups, tau_re);
        lp += log_prior_icar(x, spatial_start, n_spatial_units, tau_spatial,
                              adj_row_ptr, adj_col_idx, n_neighbors);
        return lp;
    };

    return laplace_newton_solve(y, n, family, phi, N, n_x,
                                 max_iter, tol, n_threads,
                                 compute_eta, scatter, center, log_prior,
                                 x_init);
}

// --- 3. BYM2 ---

LaplaceResult laplace_mode_bym2(
    const NumericVector& y, const IntegerVector& n,
    const NumericMatrix& X, const NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const IntegerVector& spatial_idx, int n_spatial_units,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors,
    double sigma_spatial, double rho, double scale_factor,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads
) {
    int N = y.size();
    int p = X.ncol();
    int n_x = p + n_re_groups + 2 * n_spatial_units;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    int phi_start = p + n_re_groups;
    int theta_start = phi_start + n_spatial_units;
    double sqrt_rho = std::sqrt(rho + 1e-10);
    double sqrt_1_rho = std::sqrt(1.0 - rho + 1e-10);

    auto compute_eta = [&](const NumericVector& x, NumericVector& eta) {
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
                    eta[i] += sigma_spatial * (
                        sqrt_rho * x[phi_start + s] * scale_factor +
                        sqrt_1_rho * x[theta_start + s]
                    );
                }
            }
        }
    };

    auto scatter = [&](const NumericVector& x, const NumericVector& eta,
                       DenseVec& grad, DenseMat& H) {
        // BYM2 has two latent blocks per spatial unit (phi_scaled, theta)
        // with different derivative factors
        scatter_obs_grad_hess_base(y, n, X, re_idx, N, p, n_re_groups,
                                    eta, family, phi, grad, H, n_threads);

        double d_phi = sigma_spatial * sqrt_rho * scale_factor;
        double d_theta = sigma_spatial * sqrt_1_rho;

        for (int i = 0; i < N; i++) {
            if (n_spatial_units <= 0) continue;
            int s = spatial_idx[i] - 1;
            if (s < 0 || s >= n_spatial_units) continue;

            auto gh = grad_hess_for_family(y[i], n[i], eta[i], family, phi);
            int phi_idx = phi_start + s;
            int theta_idx = theta_start + s;

            grad[phi_idx] += gh.grad * d_phi;
            grad[theta_idx] += gh.grad * d_theta;

            H[phi_idx][phi_idx] += gh.neg_hess * d_phi * d_phi;
            H[theta_idx][theta_idx] += gh.neg_hess * d_theta * d_theta;
            H[phi_idx][theta_idx] += gh.neg_hess * d_phi * d_theta;
            H[theta_idx][phi_idx] += gh.neg_hess * d_phi * d_theta;

            for (int j = 0; j < p; j++) {
                H[j][phi_idx] += gh.neg_hess * X(i, j) * d_phi;
                H[phi_idx][j] += gh.neg_hess * X(i, j) * d_phi;
                H[j][theta_idx] += gh.neg_hess * X(i, j) * d_theta;
                H[theta_idx][j] += gh.neg_hess * X(i, j) * d_theta;
            }

            if (n_re_groups > 0) {
                int g = (int)re_idx[i] - 1;
                if (g >= 0 && g < n_re_groups) {
                    H[p + g][phi_idx] += gh.neg_hess * d_phi;
                    H[phi_idx][p + g] += gh.neg_hess * d_phi;
                    H[p + g][theta_idx] += gh.neg_hess * d_theta;
                    H[theta_idx][p + g] += gh.neg_hess * d_theta;
                }
            }
        }

        // ICAR prior on phi_scaled (precision 1)
        add_icar_prior(grad, H, x, phi_start, n_spatial_units, 1.0,
                        adj_row_ptr, adj_col_idx, n_neighbors);

        // IID N(0,1) prior on theta
        for (int s = 0; s < n_spatial_units; s++) {
            int idx = theta_start + s;
            grad[idx] -= x[idx];
            H[idx][idx] += 1.0;
        }

        add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
    };

    auto center = [&](NumericVector& x) {
        center_effects(x, phi_start, n_spatial_units);
    };

    auto log_prior = [&](const NumericVector& x, const NumericVector&) {
        double lp = compute_log_prior_re(x, p, n_re_groups, tau_re);

        // Theta IID prior
        double lp_theta = 0.0;
        for (int s = 0; s < n_spatial_units; s++) {
            lp_theta -= 0.5 * x[theta_start + s] * x[theta_start + s];
        }
        lp_theta -= 0.5 * n_spatial_units * std::log(2.0 * M_PI);
        lp += lp_theta;

        // Phi_scaled ICAR prior
        double quad_form = 0.0;
        for (int s = 0; s < n_spatial_units; s++) {
            double phi_s = x[phi_start + s];
            quad_form += n_neighbors[s] * phi_s * phi_s;
            for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
                int neighbor = adj_col_idx[k];
                if (neighbor > s) {
                    quad_form -= 2.0 * phi_s * x[phi_start + neighbor];
                }
            }
        }
        lp += -0.5 * quad_form;

        return lp;
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
Rcpp::List cpp_laplace_fit_spatial(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors, double tau_spatial,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue
) {
    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) {
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }
    tulpa::LaplaceResult result = tulpa::laplace_mode_spatial(
        y, n, X, re_idx, n_re_groups, sigma_re,
        spatial_idx, n_spatial_units, adj_row_ptr, adj_col_idx, n_neighbors, tau_spatial,
        family, phi, max_iter, tol, n_threads, x_init
    );
    return tulpa::laplace_result_to_list(result);
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_bym2(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double sigma_spatial, double rho, double scale_factor,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1
) {
    tulpa::LaplaceResult result = tulpa::laplace_mode_bym2(
        y, n, X, re_idx, n_re_groups, sigma_re,
        spatial_idx, n_spatial_units, adj_row_ptr, adj_col_idx, n_neighbors,
        sigma_spatial, rho, scale_factor,
        family, phi, max_iter, tol, n_threads
    );
    return tulpa::laplace_result_to_list(result);
}
