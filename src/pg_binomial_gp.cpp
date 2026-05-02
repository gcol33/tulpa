// pg_binomial_gp.cpp
// GP (sequential NNGP) spatial Gibbs sampler for Pólya-Gamma binomial models
// Split from pg_binomial.cpp on 2026-05-02

#include "pg_shared.h"
#include "pg_rng.h"
#include "linalg_fast.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// ---------------------------------------------------------------------
// GP Spatial Gibbs Sampler for Binomial Models
// Uses sequential NNGP updates
// ---------------------------------------------------------------------

// Helper: compute NNGP conditional mean and variance
inline void pg_nngp_conditional(
    int i,
    const std::vector<double>& w,
    double sigma2,
    double phi_gp,
    int cov_type,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::NumericMatrix& nn_dist,
    const Rcpp::IntegerVector& nn_order,
    int nn,
    double& cond_mean,
    double& cond_var
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

  // Covariance function lambda
  auto compute_cov = [sigma2, phi_gp, cov_type](double d) {
    if (d < 1e-10) return sigma2;
    if (cov_type == 0) {  // Exponential
      return sigma2 * std::exp(-d / phi_gp);
    } else if (cov_type == 1) {  // Matern 1.5
      double x = std::sqrt(3.0) * d / phi_gp;
      return sigma2 * (1.0 + x) * std::exp(-x);
    } else {  // Matern 2.5
      double x = std::sqrt(5.0) * d / phi_gp;
      return sigma2 * (1.0 + x + x * x / 3.0) * std::exp(-x);
    }
  };

  std::vector<double> c_vec(n_neighbors);
  std::vector<double> C_mat(n_neighbors * n_neighbors);

  for (int j = 0; j < n_neighbors; j++) {
    c_vec[j] = compute_cov(nn_dist(i, j));
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
        C_mat[j1 * n_neighbors + j2] = compute_cov(d12);
      }
    }
  }

  // Cholesky of C
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

  // Solve L * y_sol = c_vec
  std::vector<double> y_sol(n_neighbors);
  for (int j = 0; j < n_neighbors; j++) {
    double sum = c_vec[j];
    for (int k = 0; k < j; k++) {
      sum -= L[j * n_neighbors + k] * y_sol[k];
    }
    y_sol[j] = sum / L[j * n_neighbors + j];
  }

  // Solve L^T * alpha = y_sol
  std::vector<double> alpha(n_neighbors);
  for (int j = n_neighbors - 1; j >= 0; j--) {
    double sum = y_sol[j];
    for (int k = j + 1; k < n_neighbors; k++) {
      sum -= L[k * n_neighbors + j] * alpha[k];
    }
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

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_gp(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx,
    Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int n_spatial,
    int nn,
    double sigma2_gp_init,
    double phi_gp_init,
    int cov_type,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_sigma_gp_U = 1.0,
    double prior_sigma_gp_alpha = 0.01,
    double prior_phi_lower = 0.01,
    double prior_phi_upper = 10.0,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  // CRITICAL: Must call GetRNGstate/PutRNGstate when using R's RNG from C++
  GetRNGstate();

  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  #ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
  #endif

  // Storage
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix gp_draws(n_save, n_spatial);
  Rcpp::NumericVector sigma2_gp_draws(n_save);
  Rcpp::NumericVector phi_gp_draws(n_save);
  Rcpp::NumericMatrix eta_draws_gp;
  if (store_eta) eta_draws_gp = Rcpp::NumericMatrix(n_save, N);

  // Initialize
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  std::vector<double> w(n_spatial, 0.0);
  double sigma2_gp = sigma2_gp_init;
  double phi_gp = phi_gp_init;

  // Working vectors
  Rcpp::NumericVector omega(N);
  Rcpp::NumericVector kappa(N);
  Rcpp::NumericVector eta(N);
  Rcpp::NumericVector X_beta(N);
  Rcpp::NumericVector re_contrib(N);
  Rcpp::NumericVector gp_contrib(N);
  Rcpp::NumericVector offset(N);

  for (int i = 0; i < N; i++) {
    omega[i] = 0.5;
    kappa[i] = y[i] - 0.5 * n[i];
    X_beta[i] = 0.0;
    re_contrib[i] = 0.0;
    gp_contrib[i] = 0.0;
  }

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    if (verbose && (iter + 1) % 200 == 0) {
      Rcpp::Rcout << "  Iteration " << (iter + 1) << "/" << n_iter << "\n";
    }

    // 1-4. Core Gibbs step (eta, omega, beta, RE) — shared with all variants
    tulpa::pg_gibbs_core_step(
        N, p, beta, re, sigma_re, omega, eta, X_beta, re_contrib,
        gp_contrib, offset, kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale);

    // 5. Update GP effects (sequential NNGP Gibbs)
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i];
    }

    // Aggregate likelihood info per spatial location
    std::vector<double> sum_omega_gp(n_spatial, 0.0);
    std::vector<double> sum_resid_gp(n_spatial, 0.0);
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        sum_omega_gp[i] += omega[i];
        sum_resid_gp[i] += kappa[i] - omega[i] * offset[i];
      }
    }

    // Update each GP effect in NNGP order
    for (int idx = 0; idx < n_spatial; idx++) {
      int obs_i = nn_order[idx];

      double cond_mean, cond_var;
      pg_nngp_conditional(idx, w, sigma2_gp, phi_gp, cov_type,
                          coords, nn_idx, nn_dist, nn_order, nn,
                          cond_mean, cond_var);

      double tau_prior = 1.0 / cond_var;
      double tau_lik = sum_omega_gp[obs_i];
      double tau_post = tau_prior + tau_lik;
      double mean_post = (tau_prior * cond_mean + sum_resid_gp[obs_i]) / tau_post;

      w[obs_i] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
    }

    // Update GP contributions
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        gp_contrib[i] = w[i];
      }
    }

    // 6. Update GP hyperparameters via MH
    double sigma2_prop = tulpa_linalg::safe_exp(std::log(sigma2_gp) + R::rnorm(0, 0.1));
    if (!std::isfinite(sigma2_prop) || sigma2_prop <= 0) sigma2_prop = sigma2_gp;
    double log_prior_curr = -(-std::log(prior_sigma_gp_alpha) / prior_sigma_gp_U) * std::sqrt(sigma2_gp);
    double log_prior_prop = -(-std::log(prior_sigma_gp_alpha) / prior_sigma_gp_U) * std::sqrt(sigma2_prop);

    double log_lik_diff = 0.0;
    for (int i = 0; i < n_spatial; i++) {
      log_lik_diff += -0.5 * w[nn_order[i]] * w[nn_order[i]] / sigma2_prop;
      log_lik_diff -= -0.5 * w[nn_order[i]] * w[nn_order[i]] / sigma2_gp;
    }

    double log_alpha = log_lik_diff + log_prior_prop - log_prior_curr +
                       std::log(sigma2_prop) - std::log(sigma2_gp);
    if (std::log(R::runif(0, 1)) < log_alpha) {
      sigma2_gp = sigma2_prop;
    }

    double phi_prop = tulpa_linalg::safe_exp(std::log(phi_gp) + R::rnorm(0, 0.1));
    if (std::isfinite(phi_prop) && phi_prop >= prior_phi_lower && phi_prop <= prior_phi_upper) {
      double log_alpha_phi = std::log(phi_prop) - std::log(phi_gp);
      if (std::log(R::runif(0, 1)) < log_alpha_phi) {
        phi_gp = phi_prop;
      }
    }

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_re_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_re_draws[save_idx] = sigma_re;
      for (int s = 0; s < n_spatial; s++) {
        gp_draws(save_idx, s) = w[s];
      }
      sigma2_gp_draws[save_idx] = sigma2_gp;
      phi_gp_draws[save_idx] = phi_gp;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws_gp(save_idx, i) = eta[i];
        }
      }
      save_idx++;
    }

    if ((iter + 1) % 100 == 0) Rcpp::checkUserInterrupt();
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = beta_draws,
    Rcpp::Named("re") = re_draws,
    Rcpp::Named("sigma_re") = sigma_re_draws,
    Rcpp::Named("gp") = gp_draws,
    Rcpp::Named("sigma2_gp") = sigma2_gp_draws,
    Rcpp::Named("phi_gp") = phi_gp_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws_gp;
  }

  PutRNGstate();
  return result;
}
