// laplace_core.cpp
// Core Laplace approximation engine for tulpa
// Implements Laplace approximation for latent Gaussian models
//
// Feature 9 refactor: all model-specific Laplace functions consolidated
// via laplace_newton_solve() template in laplace_newton.h.

#include "laplace_core.h"
#include "laplace_cholesky.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include "laplace_scatter.h"
#include "laplace_spatial_priors.h"
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

// =====================================================================
// GP / NNGP helpers (moved from inline, unchanged)
// =====================================================================

inline double compute_cov_exp_laplace(double d, double sigma2, double phi) {
    if (d < 1e-10) return sigma2;
    return sigma2 * std::exp(-d / phi);
}

inline double compute_cov_matern15_laplace(double d, double sigma2, double phi) {
    if (d < 1e-10) return sigma2;
    double x = std::sqrt(3.0) * d / phi;
    return sigma2 * (1.0 + x) * std::exp(-x);
}

inline double compute_cov_matern25_laplace(double d, double sigma2, double phi) {
    if (d < 1e-10) return sigma2;
    double x = std::sqrt(5.0) * d / phi;
    return sigma2 * (1.0 + x + x * x / 3.0) * std::exp(-x);
}

inline double compute_cov_laplace(double d, double sigma2, double phi, int cov_type) {
    if (cov_type == 0) return compute_cov_exp_laplace(d, sigma2, phi);
    if (cov_type == 1) return compute_cov_matern15_laplace(d, sigma2, phi);
    return compute_cov_matern25_laplace(d, sigma2, phi);
}

inline void nngp_conditional_laplace(
    int obs_idx, int i,
    const std::vector<double>& w,
    double sigma2, double phi_gp, int cov_type,
    const NumericMatrix& coords,
    const IntegerMatrix& nn_idx, const NumericMatrix& nn_dist,
    const IntegerVector& nn_order, int nn,
    double& cond_mean, double& cond_var
) {
    int n_neighbors = 0;
    for (int j = 0; j < nn; j++) {
        if (nn_idx(i, j) > 0) n_neighbors++;
    }
    if (n_neighbors == 0) {
        cond_mean = 0.0;
        cond_var = sigma2;
        return;
    }

    std::vector<double> c_vec(n_neighbors);
    std::vector<double> C_mat(n_neighbors * n_neighbors);

    for (int j = 0; j < n_neighbors; j++) {
        c_vec[j] = compute_cov_laplace(nn_dist(i, j), sigma2, phi_gp, cov_type);
    }
    for (int j1 = 0; j1 < n_neighbors; j1++) {
        int nn_orig1 = nn_order[nn_idx(i, j1) - 1];
        for (int j2 = 0; j2 < n_neighbors; j2++) {
            int nn_orig2 = nn_order[nn_idx(i, j2) - 1];
            if (j1 == j2) {
                C_mat[j1 * n_neighbors + j2] = sigma2;
            } else {
                double d12 = std::sqrt(
                    std::pow(coords(nn_orig1, 0) - coords(nn_orig2, 0), 2) +
                    std::pow(coords(nn_orig1, 1) - coords(nn_orig2, 1), 2)
                );
                C_mat[j1 * n_neighbors + j2] = compute_cov_laplace(d12, sigma2, phi_gp, cov_type);
            }
        }
    }

    // Small Cholesky for neighbor covariance matrix
    std::vector<double> L(n_neighbors * n_neighbors, 0.0);
    for (int j = 0; j < n_neighbors; j++) {
        for (int k = 0; k <= j; k++) {
            double sum = C_mat[j * n_neighbors + k];
            for (int m = 0; m < k; m++) {
                sum -= L[j * n_neighbors + m] * L[k * n_neighbors + m];
            }
            if (j == k) {
                L[j * n_neighbors + j] = std::sqrt(std::max(1e-10, sum));
            } else {
                L[j * n_neighbors + k] = sum / L[k * n_neighbors + k];
            }
        }
    }

    std::vector<double> y_solve(n_neighbors);
    for (int j = 0; j < n_neighbors; j++) {
        double sum = c_vec[j];
        for (int k = 0; k < j; k++) sum -= L[j * n_neighbors + k] * y_solve[k];
        y_solve[j] = sum / L[j * n_neighbors + j];
    }

    std::vector<double> alpha(n_neighbors);
    for (int j = n_neighbors - 1; j >= 0; j--) {
        double sum = y_solve[j];
        for (int k = j + 1; k < n_neighbors; k++) sum -= L[k * n_neighbors + j] * alpha[k];
        alpha[j] = sum / L[j * n_neighbors + j];
    }

    cond_mean = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
        int nn_orig = nn_order[nn_idx(i, j) - 1];
        cond_mean += alpha[j] * w[nn_orig];
    }

    double c_Cinv_c = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
        c_Cinv_c += c_vec[j] * alpha[j];
    }
    cond_var = std::max(1e-10, sigma2 - c_Cinv_c);
}

// =====================================================================
// Model-specific Laplace mode finders (consolidated via laplace_newton_solve)
// =====================================================================

// --- 1. Dense (no spatial) ---

LaplaceResult laplace_mode_dense(
    const NumericVector& y, const IntegerVector& n,
    const NumericMatrix& X, const NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads
) {
    int N = y.size();
    int p = X.ncol();
    int n_x = p + n_re_groups;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

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
        }
    };

    auto scatter = [&](const NumericVector& x, const NumericVector& eta,
                       DenseVec& grad, DenseMat& H) {
        scatter_obs_grad_hess_base(y, n, X, re_idx, N, p, n_re_groups,
                                    eta, family, phi, grad, H, n_threads);
        add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
    };

    auto center = [](NumericVector&) {};

    auto log_prior = [&](const NumericVector& x, const NumericVector&) {
        return compute_log_prior_re(x, p, n_re_groups, tau_re);
    };

    return laplace_newton_solve(y, n, family, phi, N, n_x,
                                 max_iter, tol, n_threads,
                                 compute_eta, scatter, center, log_prior);
}

// --- 1b. Multi-block RE with slopes (no spatial) ---
// Mode vector: [beta(0..p-1), u_1(p..p+g1*c1-1), u_2(...), ...]
// Term k has n_groups[k] groups, each with n_coefs[k] latent vars.
// For intercept-only: n_coefs=1, Z_k row = [1].
// For (x|g): n_coefs=2, Z_k row = [1, x_i].
// Latent vars for term k, group g: u_k[g*c_k .. g*c_k + c_k - 1].

LaplaceResult laplace_mode_dense_multi_re(
    const NumericVector& y, const IntegerVector& n,
    const NumericMatrix& X,
    const Rcpp::List& re_idx_list,
    const Rcpp::IntegerVector& re_ngroups,
    const Rcpp::List& re_sigma_list,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::List> re_Z_list_ = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> re_ncoefs_ = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> weights_ = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset_ = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_ = R_NilValue
) {
    int N = y.size();
    int p = X.ncol();
    int n_terms = re_ngroups.size();

    // Observation weights (default: all 1.0)
    std::vector<double> w_obs(N, 1.0);
    if (weights_.isNotNull()) {
        Rcpp::NumericVector wv = Rcpp::as<Rcpp::NumericVector>(weights_);
        for (int i = 0; i < N; i++) w_obs[i] = wv[i];
    }

    // Offset (default: all 0.0)
    std::vector<double> off(N, 0.0);
    if (offset_.isNotNull()) {
        Rcpp::NumericVector ov = Rcpp::as<Rcpp::NumericVector>(offset_);
        for (int i = 0; i < N; i++) off[i] = ov[i];
    }

    // Number of coefficients per RE term (default 1 = intercept only)
    std::vector<int> ncoefs_vec(n_terms, 1);
    if (re_ncoefs_.isNotNull()) {
        Rcpp::IntegerVector nc = Rcpp::as<Rcpp::IntegerVector>(re_ncoefs_);
        for (int k = 0; k < n_terms; k++) ncoefs_vec[k] = nc[k];
    }

    // Offsets: term k starts at offsets[k], occupies n_groups[k] * n_coefs[k] entries
    std::vector<int> offsets(n_terms);
    int total_re = 0;
    for (int k = 0; k < n_terms; k++) {
        offsets[k] = p + total_re;
        total_re += re_ngroups[k] * ncoefs_vec[k];
    }
    int n_x = p + total_re;

    // RE precision matrices per term.
    // For uncorrelated (||): diagonal Q_k with tau_k[c] = 1/sigma_k[c]^2
    // For correlated (|): full Q_k = inv(L_k * L_k') where L_k is Cholesky
    // Stored row-major: Q_re[k] has size ncoefs_vec[k]^2
    std::vector<std::vector<double>> Q_re(n_terms);
    // Log-determinant of Q for prior normalization
    std::vector<double> log_det_Q_re(n_terms, 0.0);

    for (int k = 0; k < n_terms; k++) {
        int ck = ncoefs_vec[k];
        Rcpp::NumericVector sig_k = Rcpp::as<Rcpp::NumericVector>(re_sigma_list[k]);
        Q_re[k].assign(ck * ck, 0.0);

        if ((int)sig_k.size() == ck) {
            // Diagonal: uncorrelated (||) or intercept-only
            for (int c = 0; c < ck; c++) {
                double tau = 1.0 / (sig_k[c] * sig_k[c] + 1e-10);
                Q_re[k][c * ck + c] = tau;
                log_det_Q_re[k] += std::log(tau);
            }
        } else if ((int)sig_k.size() == ck * (ck + 1) / 2) {
            // Cholesky factor L (lower-triangular, packed column-major):
            // For ck=2: sig_k = [L11, L21, L22]
            // Sigma = L * L', Q = Sigma^{-1} = L'^{-1} * L^{-1}
            std::vector<double> L(ck * ck, 0.0);
            int idx = 0;
            for (int c = 0; c < ck; c++) {
                for (int r = c; r < ck; r++) {
                    L[r * ck + c] = sig_k[idx++];
                }
            }
            // Invert L (forward substitution)
            std::vector<double> Linv(ck * ck, 0.0);
            for (int c = 0; c < ck; c++) {
                Linv[c * ck + c] = 1.0 / L[c * ck + c];
                for (int r = c + 1; r < ck; r++) {
                    double sum = 0.0;
                    for (int j = c; j < r; j++) sum += L[r * ck + j] * Linv[j * ck + c];
                    Linv[r * ck + c] = -sum / L[r * ck + r];
                }
            }
            // Q = Linv' * Linv
            for (int r = 0; r < ck; r++) {
                for (int c = 0; c < ck; c++) {
                    double sum = 0.0;
                    for (int j = 0; j < ck; j++) sum += Linv[j * ck + r] * Linv[j * ck + c];
                    Q_re[k][r * ck + c] = sum;
                }
            }
            // log|Q| = -2 * sum(log(diag(L)))
            for (int c = 0; c < ck; c++) {
                log_det_Q_re[k] -= 2.0 * std::log(std::abs(L[c * ck + c]) + 1e-10);
            }
        } else {
            // Fallback: use first sigma for all coefs
            double tau = 1.0 / (sig_k[0] * sig_k[0] + 1e-10);
            for (int c = 0; c < ck; c++) {
                Q_re[k][c * ck + c] = tau;
                log_det_Q_re[k] += std::log(tau);
            }
        }
    }

    // Copy RE indices
    std::vector<std::vector<int>> re_idx_plain(n_terms);
    for (int k = 0; k < n_terms; k++) {
        Rcpp::IntegerVector rv = Rcpp::as<Rcpp::IntegerVector>(re_idx_list[k]);
        re_idx_plain[k].assign(rv.begin(), rv.end());
    }

    // Copy Z matrices (RE design matrices per obs)
    // Z_k is N x n_coefs[k]. If NULL, assume intercept-only (Z = column of 1s).
    std::vector<std::vector<double>> Z_data(n_terms);
    std::vector<int> Z_ncol(n_terms, 1);
    if (re_Z_list_.isNotNull()) {
        Rcpp::List zl = Rcpp::as<Rcpp::List>(re_Z_list_);
        for (int k = 0; k < n_terms; k++) {
            if (Rf_isNull(zl[k])) {
                // Intercept only
                Z_data[k].assign(N, 1.0);
                Z_ncol[k] = 1;
            } else {
                Rcpp::NumericMatrix Zk = Rcpp::as<Rcpp::NumericMatrix>(zl[k]);
                Z_ncol[k] = Zk.ncol();
                Z_data[k].resize(N * Z_ncol[k]);
                for (int i = 0; i < N; i++)
                    for (int c = 0; c < Z_ncol[k]; c++)
                        Z_data[k][i * Z_ncol[k] + c] = Zk(i, c);
            }
        }
    } else {
        for (int k = 0; k < n_terms; k++) {
            Z_data[k].assign(N, 1.0);
            Z_ncol[k] = 1;
        }
    }

    std::vector<int> ngroups_vec(n_terms);
    for (int k = 0; k < n_terms; k++) ngroups_vec[k] = re_ngroups[k];

    // Helper: index into mode vector for term k, group g, coef c
    auto re_mode_idx = [&](int k, int g, int c) -> int {
        return offsets[k] + g * ncoefs_vec[k] + c;
    };

    // Helper: Z value for obs i, term k, coef c
    auto Z_val = [&](int k, int i, int c) -> double {
        return Z_data[k][i * Z_ncol[k] + c];
    };

    auto compute_eta = [&](const NumericVector& x, NumericVector& eta) {
        for (int i = 0; i < N; i++) {
            eta[i] = off[i];  // offset
            for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
            for (int k = 0; k < n_terms; k++) {
                int g = re_idx_plain[k][i] - 1;
                if (g < 0 || g >= ngroups_vec[k]) continue;
                for (int c = 0; c < ncoefs_vec[k]; c++) {
                    eta[i] += Z_val(k, i, c) * x[re_mode_idx(k, g, c)];
                }
            }
        }
    };

    auto scatter = [&](const NumericVector& x, const NumericVector& eta,
                       DenseVec& grad, DenseMat& H) {
        for (int i = 0; i < N; i++) {
            auto gh = grad_hess_for_family(y[i], n[i], eta[i], family, phi);
            // Scale by observation weight
            double wi = w_obs[i];
            gh.grad *= wi;
            gh.neg_hess *= wi;

            // Fixed effects
            for (int j = 0; j < p; j++) {
                grad[j] += gh.grad * X(i, j);
                for (int l = 0; l < p; l++) {
                    H[j][l] += gh.neg_hess * X(i, j) * X(i, l);
                }
            }

            // RE blocks
            for (int k = 0; k < n_terms; k++) {
                int g = re_idx_plain[k][i] - 1;
                if (g < 0 || g >= ngroups_vec[k]) continue;
                int ck = ncoefs_vec[k];

                // RE gradient and within-block Hessian
                for (int c1 = 0; c1 < ck; c1++) {
                    int idx1 = re_mode_idx(k, g, c1);
                    double z1 = Z_val(k, i, c1);
                    grad[idx1] += gh.grad * z1;

                    for (int c2 = 0; c2 < ck; c2++) {
                        int idx2 = re_mode_idx(k, g, c2);
                        H[idx1][idx2] += gh.neg_hess * z1 * Z_val(k, i, c2);
                    }

                    // Cross with fixed effects
                    for (int j = 0; j < p; j++) {
                        H[j][idx1] += gh.neg_hess * X(i, j) * z1;
                        H[idx1][j] += gh.neg_hess * z1 * X(i, j);
                    }
                }

                // Cross between RE blocks
                for (int m = k + 1; m < n_terms; m++) {
                    int gm = re_idx_plain[m][i] - 1;
                    if (gm < 0 || gm >= ngroups_vec[m]) continue;
                    int cm = ncoefs_vec[m];
                    for (int c1 = 0; c1 < ck; c1++) {
                        int idx1 = re_mode_idx(k, g, c1);
                        double z1 = Z_val(k, i, c1);
                        for (int c2 = 0; c2 < cm; c2++) {
                            int idx2 = re_mode_idx(m, gm, c2);
                            double cross = gh.neg_hess * z1 * Z_val(m, i, c2);
                            H[idx1][idx2] += cross;
                            H[idx2][idx1] += cross;
                        }
                    }
                }
            }
        }

        // RE priors: Q_re[k] precision matrix per group
        for (int k = 0; k < n_terms; k++) {
            int ck = ncoefs_vec[k];
            for (int g = 0; g < ngroups_vec[k]; g++) {
                for (int c1 = 0; c1 < ck; c1++) {
                    int idx1 = re_mode_idx(k, g, c1);
                    // grad -= Q * u
                    for (int c2 = 0; c2 < ck; c2++) {
                        grad[idx1] -= Q_re[k][c1 * ck + c2] * x[re_mode_idx(k, g, c2)];
                    }
                    // H += Q (off-diagonal and diagonal)
                    for (int c2 = 0; c2 < ck; c2++) {
                        H[idx1][re_mode_idx(k, g, c2)] += Q_re[k][c1 * ck + c2];
                    }
                }
            }
        }

        // Beta prior
        double tau_beta = 1e-4;
        for (int j = 0; j < p; j++) {
            grad[j] -= tau_beta * x[j];
            H[j][j] += tau_beta;
        }
    };

    auto center = [](NumericVector&) {};

    auto log_prior = [&](const NumericVector& x, const NumericVector&) {
        double lp = 0.0;
        double tau_beta = 1e-4;
        for (int j = 0; j < p; j++) lp -= 0.5 * tau_beta * x[j] * x[j];
        for (int k = 0; k < n_terms; k++) {
            int ck = ncoefs_vec[k];
            // -0.5 * u_g' Q_k u_g for each group
            for (int g = 0; g < ngroups_vec[k]; g++) {
                for (int c1 = 0; c1 < ck; c1++) {
                    for (int c2 = 0; c2 < ck; c2++) {
                        lp -= 0.5 * Q_re[k][c1 * ck + c2]
                            * x[re_mode_idx(k, g, c1)]
                            * x[re_mode_idx(k, g, c2)];
                    }
                }
            }
            // Normalization: 0.5 * n_groups * (log|Q_k| - ck * log(2*pi))
            if (ngroups_vec[k] > 0) {
                lp += 0.5 * ngroups_vec[k] * (log_det_Q_re[k] - ck * std::log(2.0 * M_PI));
            }
        }
        return lp;
    };

    Rcpp::NumericVector x_init_vec;
    if (x_init_.isNotNull()) {
        x_init_vec = Rcpp::as<Rcpp::NumericVector>(x_init_);
    }

    return laplace_newton_solve(y, n, family, phi, N, n_x,
                                 max_iter, tol, n_threads,
                                 compute_eta, scatter, center, log_prior,
                                 x_init_vec);
}

} // namespace tulpa

// =====================================================================
// R exports (call into tulpa:: functions defined above)
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1
) {
    tulpa::LaplaceResult result = tulpa::laplace_mode_dense(
        y, n, X, re_idx, n_re_groups, sigma_re, family, phi, max_iter, tol, n_threads
    );
    return tulpa::laplace_result_to_list(result);
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_multi_re(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::List re_idx_list,
    Rcpp::IntegerVector re_ngroups,
    Rcpp::List re_sigma_list,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::List> re_Z_list = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> re_ncoefs = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> weights = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> x_init = R_NilValue
) {
    tulpa::LaplaceResult result = tulpa::laplace_mode_dense_multi_re(
        y, n, X, re_idx_list, re_ngroups, re_sigma_list, family, phi, max_iter, tol, n_threads,
        re_Z_list, re_ncoefs, weights, offset, x_init
    );
    return tulpa::laplace_result_to_list(result);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_laplace_sample(
    Rcpp::NumericVector mode, Rcpp::NumericMatrix H, int n_samples
) {
    int n_x = mode.size();
    Rcpp::NumericMatrix samples(n_samples, n_x);

    // Cholesky of H + ridge*I. The same uniform upstream regularization
    // every Laplace solve uses (see LAPLACE_UNIFORM_RIDGE in
    // laplace_cholesky.h); guarantees PD on rank-deficient priors so the
    // sampler never hits a non-positive pivot.
    for (int j = 0; j < n_x; j++) H(j, j) += tulpa::LAPLACE_UNIFORM_RIDGE;
    Rcpp::NumericMatrix L(n_x, n_x);
    double log_det;
    tulpa::dense_cholesky_factorize(H, n_x, L, log_det);

    // Sample: z ~ N(0, I), x = mode + L^{-T} z
    for (int s = 0; s < n_samples; s++) {
        Rcpp::NumericVector z(n_x);
        for (int j = 0; j < n_x; j++) z[j] = R::rnorm(0.0, 1.0);

        // Solve L' x_centered = z (back substitution)
        Rcpp::NumericVector x_centered(n_x);
        for (int j = n_x - 1; j >= 0; j--) {
            double sum = z[j];
            for (int k = j + 1; k < n_x; k++) sum -= L(k, j) * x_centered[k];
            x_centered[j] = sum / L(j, j);
        }

        for (int j = 0; j < n_x; j++) samples(s, j) = mode[j] + x_centered[j];
    }
    return samples;
}

// Spatial / BYM2 / RSR mode finders + their R exports moved to laplace_core_spatial.cpp.
// GP / Multiscale GP / Multiscale temporal mode finders + their R exports moved to laplace_core_gp.cpp.
// Nested Laplace and SPDE code lives in nested_laplace.cpp and spde_laplace.cpp.
