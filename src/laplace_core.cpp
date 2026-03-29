// laplace_core.cpp
// Core Laplace approximation engine for tulpa
// Implements Laplace approximation for latent Gaussian models
//
// Feature 9 refactor: all model-specific Laplace functions consolidated
// via laplace_newton_solve() template in laplace_helpers.h.

#include "laplace_core.h"
#include "laplace_helpers.h"
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
// Likelihood functions (unchanged — these are the canonical definitions)
// =====================================================================

double log_lik_gaussian(double y, double eta, double phi) {
  double resid = y - eta;
  return -0.5 * std::log(2.0 * M_PI * phi * phi) - resid * resid / (2.0 * phi * phi);
}

double grad_log_lik_gaussian(double y, double eta, double phi) {
  return (y - eta) / (phi * phi);
}

double neg_hess_log_lik_gaussian(double y, double eta, double phi) {
  return 1.0 / (phi * phi);
}

double log_lik_binomial(int y, int n, double eta) {
  double log_p;
  if (eta > 0) {
    log_p = y * eta - n * eta - n * std::log(1.0 + std::exp(-eta));
  } else {
    log_p = y * eta - n * std::log(1.0 + std::exp(eta));
  }
  return log_p;
}

double grad_log_lik_binomial(int y, int n, double eta) {
  double p;
  if (eta > 0) {
    double exp_neg_eta = std::exp(-eta);
    p = 1.0 / (1.0 + exp_neg_eta);
  } else {
    double exp_eta = std::exp(eta);
    p = exp_eta / (1.0 + exp_eta);
  }
  return y - n * p;
}

double neg_hess_log_lik_binomial(int y, int n, double eta) {
  double p;
  if (eta > 0) {
    double exp_neg_eta = std::exp(-eta);
    p = 1.0 / (1.0 + exp_neg_eta);
  } else {
    double exp_eta = std::exp(eta);
    p = exp_eta / (1.0 + exp_eta);
  }
  return n * p * (1.0 - p);
}

double log_lik_negbin(int y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  double log_p = R::lgammafn(y + phi) - R::lgammafn(phi) - R::lgammafn(y + 1.0)
               + phi * std::log(phi / (mu + phi))
               + y * std::log(mu / (mu + phi));
  return log_p;
}

double grad_log_lik_negbin(int y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  double p = mu / (mu + phi);
  return y - (y + phi) * p;
}

double neg_hess_log_lik_negbin(int y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  double denom = mu + phi;
  return (y + phi) * mu * phi / (denom * denom);
}

double log_lik_poisson(int y, double eta) {
  return y * eta - tulpa_linalg::safe_exp(eta) - R::lgammafn(y + 1.0);
}

double grad_log_lik_poisson(int y, double eta) {
  return y - tulpa_linalg::safe_exp(eta);
}

double neg_hess_log_lik_poisson(int y, double eta) {
  return tulpa_linalg::safe_exp(eta);
}

// --- Gamma (log link): phi = shape parameter ---
double log_lik_gamma(double y, double eta, double phi) {
  // mu = exp(eta), phi = shape
  double mu = tulpa_linalg::safe_exp(eta);
  return phi * std::log(phi) - R::lgammafn(phi) + (phi - 1.0) * std::log(y)
         - phi * std::log(mu) - phi * y / mu;
}

double grad_log_lik_gamma(double y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  return phi * (y / mu - 1.0);
}

double neg_hess_log_lik_gamma(double y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  return phi * y / mu;
}

// --- Gamma (inverse link): eta = 1/mu, mu = 1/eta ---
double log_lik_gamma_inverse(double y, double eta, double phi) {
  // phi = shape, eta = 1/mu > 0
  return phi * std::log(phi) + phi * std::log(eta) - R::lgammafn(phi)
         + (phi - 1.0) * std::log(y) - phi * y * eta;
}

double grad_log_lik_gamma_inverse(double y, double eta, double phi) {
  return phi * (1.0 / eta - y);
}

double neg_hess_log_lik_gamma_inverse(double y, double eta, double phi) {
  return phi / (eta * eta);
}

// --- Binomial (probit link): p = Phi(eta) ---
double log_lik_binomial_probit(int y, int n, double eta) {
  return y * R::pnorm(eta, 0.0, 1.0, 1, 1)     // y * log(Phi(eta))
       + (n - y) * R::pnorm(-eta, 0.0, 1.0, 1, 1); // (n-y) * log(Phi(-eta))
}

double grad_log_lik_binomial_probit(int y, int n, double eta) {
  double phi_eta = R::dnorm(eta, 0.0, 1.0, 0);
  double p = R::pnorm(eta, 0.0, 1.0, 1, 0);
  double q = 1.0 - p;
  if (p < 1e-15) p = 1e-15;
  if (q < 1e-15) q = 1e-15;
  return phi_eta * (y / p - (n - y) / q);
}

double neg_hess_log_lik_binomial_probit(int y, int n, double eta) {
  // Fisher scoring: n * phi(eta)^2 / (p * (1-p))
  double phi_eta = R::dnorm(eta, 0.0, 1.0, 0);
  double p = R::pnorm(eta, 0.0, 1.0, 1, 0);
  double q = 1.0 - p;
  if (p < 1e-15) p = 1e-15;
  if (q < 1e-15) q = 1e-15;
  return n * phi_eta * phi_eta / (p * q);
}

// --- Binomial (cloglog link): p = 1 - exp(-exp(eta)) ---
double log_lik_binomial_cloglog(int y, int n, double eta) {
  double u = tulpa_linalg::safe_exp(eta);
  double log_q = -u;                           // log(1-p) = -exp(eta)
  double log_p = std::log1p(-std::exp(log_q)); // log(1 - exp(-u)) = log(p)
  if (log_p < -700.0) log_p = -700.0;
  return y * log_p + (n - y) * log_q;
}

double grad_log_lik_binomial_cloglog(int y, int n, double eta) {
  double u = tulpa_linalg::safe_exp(eta);
  double exp_neg_u = std::exp(-u);
  double p = 1.0 - exp_neg_u;
  if (p < 1e-15) p = 1e-15;
  // dp/deta = u * exp(-u)
  double dp = u * exp_neg_u;
  return y * dp / p - (n - y) * u;
}

double neg_hess_log_lik_binomial_cloglog(int y, int n, double eta) {
  // Fisher scoring: n * (dp/deta)^2 / (p * (1-p))
  double u = tulpa_linalg::safe_exp(eta);
  double exp_neg_u = std::exp(-u);
  double p = 1.0 - exp_neg_u;
  double q = exp_neg_u;
  if (p < 1e-15) p = 1e-15;
  if (q < 1e-15) q = 1e-15;
  double dp = u * exp_neg_u;
  return n * dp * dp / (p * q);
}

// --- Inverse Gaussian (log link): phi = dispersion = 1/lambda ---
double log_lik_inverse_gaussian(double y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  double resid = y - mu;
  return -0.5 * std::log(2.0 * M_PI * phi * y * y * y)
         - resid * resid / (2.0 * phi * mu * mu * y);
}

double grad_log_lik_inverse_gaussian(double y, double eta, double phi) {
  // (y - mu) / (phi * mu^2) where mu = exp(eta)
  double mu = tulpa_linalg::safe_exp(eta);
  return (y - mu) / (phi * mu * mu);
}

double neg_hess_log_lik_inverse_gaussian(double y, double eta, double phi) {
  // Observed: (2y - mu) / (phi * mu^2), can be negative when y < mu/2.
  // Fisher:   1 / (phi * mu), always positive but can underestimate curvature.
  // Use max(observed, fisher) for stable convergence with adequate curvature.
  double mu = tulpa_linalg::safe_exp(eta);
  double fisher = 1.0 / (phi * mu);
  double observed = (2.0 * y - mu) / (phi * mu * mu);
  return (observed > fisher) ? observed : fisher;
}

// =====================================================================
// Sparse matrix helpers (for future CHOLMOD, kept from original)
// =====================================================================

void solve_lower(const std::vector<double>& L_vals,
                 const std::vector<int>& L_row,
                 const std::vector<int>& L_col_ptr,
                 const std::vector<double>& b,
                 std::vector<double>& x) {
  int n = b.size();
  x = b;
  for (int j = 0; j < n; j++) {
    x[j] /= L_vals[L_col_ptr[j]];
    for (int k = L_col_ptr[j] + 1; k < L_col_ptr[j + 1]; k++) {
      x[L_row[k]] -= L_vals[k] * x[j];
    }
  }
}

void solve_upper(const std::vector<double>& L_vals,
                 const std::vector<int>& L_row,
                 const std::vector<int>& L_col_ptr,
                 const std::vector<double>& b,
                 std::vector<double>& x) {
  int n = b.size();
  x = b;
  for (int j = n - 1; j >= 0; j--) {
    for (int k = L_col_ptr[j] + 1; k < L_col_ptr[j + 1]; k++) {
      x[j] -= L_vals[k] * x[L_row[k]];
    }
    x[j] /= L_vals[L_col_ptr[j]];
  }
}

// =====================================================================
// Shared scatter: likelihood grad/hess onto parameter vector
// =====================================================================
//
// This scatters per-observation (g_i, h_i) onto fixed-effect and RE blocks.
// Model-specific blocks (spatial, temporal) are handled by each variant's
// ScatterGradHess callback, which calls this first then adds its own terms.

void scatter_obs_grad_hess_base(
    const NumericVector& y, const IntegerVector& n_trials,
    const NumericMatrix& X, const NumericVector& re_idx,
    int N, int p, int n_re_groups,
    const NumericVector& eta, const std::string& family, double phi,
    DenseVec& grad, DenseMat& H, int n_threads
) {
    #ifdef _OPENMP
    #pragma omp parallel num_threads(n_threads > 0 ? n_threads : 1)
    {
        DenseVec grad_thread(grad.size(), 0.0);
        DenseMat H_thread(H.size(), DenseVec(H.size(), 0.0));

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            auto gh = grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi);

            for (int j = 0; j < p; j++) {
                grad_thread[j] += gh.grad * X(i, j);
                for (int k = 0; k < p; k++) {
                    H_thread[j][k] += gh.neg_hess * X(i, j) * X(i, k);
                }
            }

            if (n_re_groups > 0) {
                int g = (int)re_idx[i] - 1;
                if (g >= 0 && g < n_re_groups) {
                    grad_thread[p + g] += gh.grad;
                    H_thread[p + g][p + g] += gh.neg_hess;
                    for (int j = 0; j < p; j++) {
                        H_thread[j][p + g] += gh.neg_hess * X(i, j);
                        H_thread[p + g][j] += gh.neg_hess * X(i, j);
                    }
                }
            }
        }

        #pragma omp critical
        {
            int n_x = (int)grad.size();
            for (int j = 0; j < n_x; j++) {
                grad[j] += grad_thread[j];
                for (int k = 0; k < n_x; k++) {
                    H[j][k] += H_thread[j][k];
                }
            }
        }
    }
    #else
    for (int i = 0; i < N; i++) {
        auto gh = grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi);

        for (int j = 0; j < p; j++) {
            grad[j] += gh.grad * X(i, j);
            for (int k = 0; k < p; k++) {
                H[j][k] += gh.neg_hess * X(i, j) * X(i, k);
            }
        }

        if (n_re_groups > 0) {
            int g = (int)re_idx[i] - 1;
            if (g >= 0 && g < n_re_groups) {
                grad[p + g] += gh.grad;
                H[p + g][p + g] += gh.neg_hess;
                for (int j = 0; j < p; j++) {
                    H[j][p + g] += gh.neg_hess * X(i, j);
                    H[p + g][j] += gh.neg_hess * X(i, j);
                }
            }
        }
    }
    #endif
}

// Extended scatter that also handles a generic "latent effect" block.
// Each observation maps to a latent index via effect_idx, and contributes
// g_i * d_factor and h_i * d_factor^2 to that index.
// d_factors is per-observation derivative factor (1.0 for additive effects).
// Cross-terms with fixed effects and RE are included.
void scatter_obs_with_latent(
    const NumericVector& y, const IntegerVector& n_trials,
    const NumericMatrix& X, const NumericVector& re_idx,
    int N, int p, int n_re_groups,
    const NumericVector& eta, const std::string& family, double phi,
    const std::vector<int>& effect_idx,   // per-obs: which latent param (absolute index), or -1
    const std::vector<double>& d_factors, // per-obs: derivative factor
    DenseVec& grad, DenseMat& H, int n_threads
) {
    // First scatter base (fixed + RE)
    scatter_obs_grad_hess_base(y, n_trials, X, re_idx, N, p, n_re_groups,
                                eta, family, phi, grad, H, n_threads);

    // Then scatter latent effect contributions
    for (int i = 0; i < N; i++) {
        int idx = effect_idx[i];
        if (idx < 0) continue;

        auto gh = grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi);
        double d = d_factors[i];

        grad[idx] += gh.grad * d;
        H[idx][idx] += gh.neg_hess * d * d;

        // Cross terms with fixed effects
        for (int j = 0; j < p; j++) {
            H[j][idx] += gh.neg_hess * X(i, j) * d;
            H[idx][j] += gh.neg_hess * X(i, j) * d;
        }

        // Cross terms with RE
        if (n_re_groups > 0) {
            int g = (int)re_idx[i] - 1;
            if (g >= 0 && g < n_re_groups) {
                H[p + g][idx] += gh.neg_hess * d;
                H[idx][p + g] += gh.neg_hess * d;
            }
        }
    }
}

// =====================================================================
// ICAR prior helpers
// =====================================================================

void add_icar_prior(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int spatial_start, int n_spatial_units, double tau_spatial,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors
) {
    for (int s = 0; s < n_spatial_units; s++) {
        int sp_idx = spatial_start + s;
        double phi_s = x[sp_idx];

        double neighbor_sum = 0.0;
        for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
            int neighbor = adj_col_idx[k];
            neighbor_sum += x[spatial_start + neighbor];
        }
        grad[sp_idx] -= tau_spatial * (n_neighbors[s] * phi_s - neighbor_sum);
        H[sp_idx][sp_idx] += tau_spatial * n_neighbors[s];

        for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
            int neighbor = adj_col_idx[k];
            H[sp_idx][spatial_start + neighbor] -= tau_spatial;
        }
    }
}

// Log-prior for ICAR (for finalize).
double log_prior_icar(
    const NumericVector& x, int spatial_start, int n_spatial_units,
    double tau_spatial,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors
) {
    double quad_form = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
        double phi_s = x[spatial_start + s];
        quad_form += n_neighbors[s] * phi_s * phi_s;
        for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
            int neighbor = adj_col_idx[k];
            if (neighbor > s) {
                quad_form -= 2.0 * phi_s * x[spatial_start + neighbor];
            }
        }
    }
    double lp = -0.5 * tau_spatial * quad_form;
    lp += 0.5 * (n_spatial_units - 1) * std::log(tau_spatial / (2.0 * M_PI));
    return lp;
}

// =====================================================================
// Temporal prior helpers (moved from inline, unchanged)
// =====================================================================

void add_rw1_precision(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int start_idx, int n_times, double tau, bool cyclic
) {
    for (int t = 1; t < n_times; t++) {
        int idx = start_idx + t;
        double diff = x[idx] - x[idx - 1];
        grad[idx] -= tau * diff;
        grad[idx - 1] += tau * diff;
        H[idx][idx] += tau;
        H[idx - 1][idx - 1] += tau;
        H[idx][idx - 1] -= tau;
        H[idx - 1][idx] -= tau;
    }
    if (cyclic && n_times > 1) {
        int idx_first = start_idx;
        int idx_last = start_idx + n_times - 1;
        double diff = x[idx_first] - x[idx_last];
        grad[idx_first] -= tau * diff;
        grad[idx_last] += tau * diff;
        H[idx_first][idx_first] += tau;
        H[idx_last][idx_last] += tau;
        H[idx_first][idx_last] -= tau;
        H[idx_last][idx_first] -= tau;
    }
}

void add_rw2_precision(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int start_idx, int n_times, double tau, bool cyclic
) {
    if (n_times < 3) return;
    for (int t = 1; t < n_times - 1; t++) {
        int idx = start_idx + t;
        double diff2 = x[idx - 1] - 2.0 * x[idx] + x[idx + 1];
        grad[idx - 1] -= tau * diff2;
        grad[idx] += 2.0 * tau * diff2;
        grad[idx + 1] -= tau * diff2;
        H[idx - 1][idx - 1] += tau;
        H[idx][idx] += 4.0 * tau;
        H[idx + 1][idx + 1] += tau;
        H[idx - 1][idx] -= 2.0 * tau;
        H[idx][idx - 1] -= 2.0 * tau;
        H[idx][idx + 1] -= 2.0 * tau;
        H[idx + 1][idx] -= 2.0 * tau;
        H[idx - 1][idx + 1] += tau;
        H[idx + 1][idx - 1] += tau;
    }
}

void add_ar1_precision(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int start_idx, int n_times, double tau, double rho
) {
    if (n_times < 1) return;
    double tau_marginal = tau * (1.0 - rho * rho);
    int idx0 = start_idx;
    grad[idx0] -= tau_marginal * x[idx0];
    H[idx0][idx0] += tau_marginal;
    for (int t = 1; t < n_times; t++) {
        int idx = start_idx + t;
        int idx_prev = start_idx + t - 1;
        double resid = x[idx] - rho * x[idx_prev];
        grad[idx] -= tau * resid;
        grad[idx_prev] += tau * rho * resid;
        H[idx][idx] += tau;
        H[idx_prev][idx_prev] += tau * rho * rho;
        H[idx][idx_prev] -= tau * rho;
        H[idx_prev][idx] -= tau * rho;
    }
}

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

// --- 7. RSR (Restricted Spatial Regression) ---

LaplaceResult laplace_mode_rsr(
    const NumericVector& y, const IntegerVector& n,
    const NumericMatrix& X, const NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const IntegerVector& spatial_idx, int n_spatial_units,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors, double tau_spatial,
    const NumericVector& rsr_projection, int rsr_n,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads
) {
    int N = y.size();
    int p = X.ncol();
    int n_x = p + n_re_groups + n_spatial_units;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    int spatial_start = p + n_re_groups;

    auto compute_eta = [&](const NumericVector& x, NumericVector& eta) {
        // Compute projected spatial: w_proj = P_perp * w
        NumericVector w_proj(n_spatial_units, 0.0);
        for (int i = 0; i < n_spatial_units; i++) {
            for (int j = 0; j < n_spatial_units; j++) {
                w_proj[i] += rsr_projection[i * rsr_n + j] * x[spatial_start + j];
            }
        }

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
                if (s >= 0 && s < n_spatial_units) eta[i] += w_proj[s];
            }
        }
    };

    auto scatter = [&](const NumericVector& x, const NumericVector& eta,
                       DenseVec& grad, DenseMat& H) {
        // Base scatter for fixed + RE
        scatter_obs_grad_hess_base(y, n, X, re_idx, N, p, n_re_groups,
                                    eta, family, phi, grad, H, n_threads);

        // RSR: gradient w.r.t. projected spatial, then transform through P_perp
        std::vector<double> grad_w_proj(n_spatial_units, 0.0);
        std::vector<double> H_w_proj_diag(n_spatial_units, 0.0);

        for (int i = 0; i < N; i++) {
            if (n_spatial_units <= 0) continue;
            int s = spatial_idx[i] - 1;
            if (s < 0 || s >= n_spatial_units) continue;

            auto gh = grad_hess_for_family(y[i], n[i], eta[i], family, phi);
            grad_w_proj[s] += gh.grad;
            H_w_proj_diag[s] += gh.neg_hess;
        }

        // Transform: grad_w = P_perp' * grad_w_proj
        for (int i = 0; i < n_spatial_units; i++) {
            for (int j = 0; j < n_spatial_units; j++) {
                grad[spatial_start + i] += rsr_projection[i * rsr_n + j] * grad_w_proj[j];
            }
        }

        // Hessian: H_w = P_perp' diag(H_w_proj) P_perp
        for (int i = 0; i < n_spatial_units; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0.0;
                for (int k = 0; k < n_spatial_units; k++) {
                    sum += rsr_projection[k * rsr_n + i] * H_w_proj_diag[k] * rsr_projection[k * rsr_n + j];
                }
                H[spatial_start + i][spatial_start + j] = sum;
                if (i != j) H[spatial_start + j][spatial_start + i] = sum;
            }
        }

        // Cross-terms between fixed effects and projected spatial effects
        for (int i = 0; i < N; i++) {
            if (n_spatial_units <= 0) continue;
            int s = spatial_idx[i] - 1;
            if (s < 0 || s >= n_spatial_units) continue;

            auto gh = grad_hess_for_family(y[i], n[i], eta[i], family, phi);

            for (int j = 0; j < p; j++) {
                for (int k = 0; k < n_spatial_units; k++) {
                    double P_ks = rsr_projection[s * rsr_n + k];
                    H[j][spatial_start + k] += gh.neg_hess * X(i, j) * P_ks;
                    H[spatial_start + k][j] += gh.neg_hess * X(i, j) * P_ks;
                }
            }
            if (n_re_groups > 0) {
                int g = (int)re_idx[i] - 1;
                if (g >= 0 && g < n_re_groups) {
                    for (int k = 0; k < n_spatial_units; k++) {
                        double P_ks = rsr_projection[s * rsr_n + k];
                        H[p + g][spatial_start + k] += gh.neg_hess * P_ks;
                        H[spatial_start + k][p + g] += gh.neg_hess * P_ks;
                    }
                }
            }
        }

        // ICAR prior on unprojected spatial
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
                                 compute_eta, scatter, center, log_prior);
}

} // namespace tulpa

// =====================================================================
// R exports (unchanged signatures, just call into tulpa:: functions)
// =====================================================================

// Helper: convert LaplaceResult to Rcpp::List (single source of truth)
static Rcpp::List laplace_result_to_list(const tulpa::LaplaceResult& result) {
    return Rcpp::List::create(
        Rcpp::Named("mode") = result.mode,
        Rcpp::Named("log_det_Q") = result.log_det_Q,
        Rcpp::Named("log_marginal") = result.log_marginal,
        Rcpp::Named("n_iter") = result.n_iter,
        Rcpp::Named("converged") = result.converged
    );
}

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
    return laplace_result_to_list(result);
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
    return laplace_result_to_list(result);
}

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
    return laplace_result_to_list(result);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_laplace_sample(
    Rcpp::NumericVector mode, Rcpp::NumericMatrix H, int n_samples
) {
    int n_x = mode.size();
    Rcpp::NumericMatrix samples(n_samples, n_x);

    // Cholesky of H
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
    return laplace_result_to_list(result);
}

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
    return laplace_result_to_list(result);
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
    return laplace_result_to_list(result);
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
    return laplace_result_to_list(result);
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_rsr(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors, double tau_spatial,
    Rcpp::NumericVector rsr_projection, int rsr_n,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1
) {
    tulpa::LaplaceResult result = tulpa::laplace_mode_rsr(
        y, n, X, re_idx, n_re_groups, sigma_re,
        spatial_idx, n_spatial_units, adj_row_ptr, adj_col_idx, n_neighbors,
        tau_spatial, rsr_projection, rsr_n,
        family, phi, max_iter, tol, n_threads
    );
    return laplace_result_to_list(result);
}
// Nested Laplace and SPDE code moved to nested_laplace.cpp and spde_laplace.cpp
