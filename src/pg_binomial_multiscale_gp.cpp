// pg_binomial_multiscale_gp.cpp
// Multiscale GP Gibbs sampler (local + regional components) for Pólya-Gamma
// binomial models. Split from pg_binomial.cpp on 2026-05-02

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

// -----------------------------------------------------------------------------
// Multiscale GP Gibbs sampler (local + regional components)
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_multiscale_gp(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx_local,
    Rcpp::NumericMatrix nn_dist_local,
    Rcpp::IntegerVector nn_order_local,
    int nn_local,
    Rcpp::IntegerMatrix nn_idx_regional,
    Rcpp::NumericMatrix nn_dist_regional,
    Rcpp::IntegerVector nn_order_regional,
    int nn_regional,
    int n_spatial,
    double sigma2_local_init,
    double phi_local_init,
    double sigma2_regional_init,
    double phi_regional_init,
    int cov_type,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_sigma_local_U = 1.0,
    double prior_sigma_local_alpha = 0.01,
    double prior_phi_local_lower = 0.01,
    double prior_phi_local_upper = 5.0,
    double prior_sigma_regional_U = 1.0,
    double prior_sigma_regional_alpha = 0.01,
    double prior_phi_regional_lower = 0.1,
    double prior_phi_regional_upper = 20.0,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  // CRITICAL: Must call GetRNGstate/PutRNGstate when using R's RNG from C++
  GetRNGstate();

  int N = y.size();
  int p = X.ncol();

  if (verbose) {
    Rcpp::Rcout << "PG Binomial Gibbs sampler with multiscale GP spatial\n";
    Rcpp::Rcout << "  N = " << N << ", p = " << p << "\n";
    Rcpp::Rcout << "  n_spatial = " << n_spatial << "\n";
    Rcpp::Rcout << "  nn_local = " << nn_local << ", nn_regional = " << nn_regional << "\n";
  }

  // Initialize parameters
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;

  // Local GP effects
  std::vector<double> w_local(n_spatial, 0.0);
  double sigma2_local = sigma2_local_init;
  double phi_local = phi_local_init;

  // Regional GP effects
  std::vector<double> w_regional(n_spatial, 0.0);
  double sigma2_regional = sigma2_regional_init;
  double phi_regional = phi_regional_init;

  // Working vectors (use Rcpp types for compatibility with update functions)
  Rcpp::NumericVector omega(N, 0.0);
  Rcpp::NumericVector kappa(N, 0.0);
  Rcpp::NumericVector eta_vec(N, 0.0);
  Rcpp::NumericVector X_beta(N, 0.0);
  Rcpp::NumericVector re_contrib(N, 0.0);
  Rcpp::NumericVector local_contrib(N, 0.0);
  Rcpp::NumericVector regional_contrib(N, 0.0);
  Rcpp::NumericVector offset(N, 0.0);

  // Compute kappa
  for (int i = 0; i < N; i++) {
    kappa[i] = (double)y[i] - 0.5 * (double)n[i];
  }

  // Storage for draws
  int n_save = (n_iter - n_warmup) / thin;
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix w_local_draws(n_save, n_spatial);
  Rcpp::NumericMatrix w_regional_draws(n_save, n_spatial);
  Rcpp::NumericVector sigma2_local_draws(n_save);
  Rcpp::NumericVector phi_local_draws(n_save);
  Rcpp::NumericVector sigma2_regional_draws(n_save);
  Rcpp::NumericVector phi_regional_draws(n_save);
  Rcpp::NumericMatrix eta_draws_temp;
  if (store_eta) {
    eta_draws_temp = Rcpp::NumericMatrix(n_save, N);
  }

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    if (verbose && (iter + 1) % 200 == 0) {
      Rcpp::Rcout << "  Iteration " << (iter + 1) << "/" << n_iter << "\n";
    }

    // 1. Compute combined spatial contribution
    Rcpp::NumericVector combined_contrib(N);
    for (int i = 0; i < N; i++) {
      combined_contrib[i] = local_contrib[i] + regional_contrib[i];
    }

    // 2-4. Core Gibbs step (eta, omega, beta, RE) — shared with all variants
    tulpa::pg_gibbs_core_step(
        N, p, beta, re, sigma_re, omega, eta_vec, X_beta, re_contrib,
        combined_contrib, offset, kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale);

    // 5. Update local GP effects
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i] + regional_contrib[i];
    }

    // Aggregate likelihood info per spatial location
    std::vector<double> sum_omega_local(n_spatial, 0.0);
    std::vector<double> sum_resid_local(n_spatial, 0.0);
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        sum_omega_local[i] += omega[i];
        sum_resid_local[i] += kappa[i] - omega[i] * offset[i];
      }
    }

    // Sequential NNGP Gibbs update for local effects
    double tau_local = 1.0 / sigma2_local;
    for (int ii = 0; ii < n_spatial; ii++) {
      int i = nn_order_local[ii];

      double cond_mean = 0.0;
      double cond_prec = tau_local;

      for (int k = 0; k < nn_local; k++) {
        int neighbor = nn_idx_local(i, k) - 1;
        if (neighbor >= 0 && neighbor < n_spatial) {
          double dist = nn_dist_local(i, k);
          double cov_val = std::exp(-dist / phi_local);
          cond_mean += cov_val * w_local[neighbor];
        }
      }
      cond_mean *= tau_local;

      double post_prec = cond_prec + sum_omega_local[i];
      double post_mean = (cond_mean + sum_resid_local[i]) / post_prec;
      w_local[i] = R::rnorm(post_mean, 1.0 / std::sqrt(post_prec));
    }

    // Update local contributions
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        local_contrib[i] = w_local[i];
      }
    }

    // 6. Update regional GP effects
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i] + local_contrib[i];
    }

    std::vector<double> sum_omega_regional(n_spatial, 0.0);
    std::vector<double> sum_resid_regional(n_spatial, 0.0);
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        sum_omega_regional[i] += omega[i];
        sum_resid_regional[i] += kappa[i] - omega[i] * offset[i];
      }
    }

    double tau_regional = 1.0 / sigma2_regional;
    for (int ii = 0; ii < n_spatial; ii++) {
      int i = nn_order_regional[ii];

      double cond_mean = 0.0;
      double cond_prec = tau_regional;

      for (int k = 0; k < nn_regional; k++) {
        int neighbor = nn_idx_regional(i, k) - 1;
        if (neighbor >= 0 && neighbor < n_spatial) {
          double dist = nn_dist_regional(i, k);
          double cov_val = std::exp(-dist / phi_regional);
          cond_mean += cov_val * w_regional[neighbor];
        }
      }
      cond_mean *= tau_regional;

      double post_prec = cond_prec + sum_omega_regional[i];
      double post_mean = (cond_mean + sum_resid_regional[i]) / post_prec;
      w_regional[i] = R::rnorm(post_mean, 1.0 / std::sqrt(post_prec));
    }

    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        regional_contrib[i] = w_regional[i];
      }
    }

    // 7. Update hyperparameters via MH
    // Update sigma2_local (Gibbs from inverse-gamma)
    double ss_local = 0.0;
    for (int i = 0; i < n_spatial; i++) {
      ss_local += w_local[i] * w_local[i];
    }
    double shape_local = 0.5 * n_spatial + 1.0;
    double rate_local = 0.5 * ss_local + prior_sigma_local_U;
    sigma2_local = 1.0 / R::rgamma(shape_local, 1.0 / rate_local);

    // Update sigma2_regional
    double ss_regional = 0.0;
    for (int i = 0; i < n_spatial; i++) {
      ss_regional += w_regional[i] * w_regional[i];
    }
    double shape_regional = 0.5 * n_spatial + 1.0;
    double rate_regional = 0.5 * ss_regional + prior_sigma_regional_U;
    sigma2_regional = 1.0 / R::rgamma(shape_regional, 1.0 / rate_regional);

    // Update phi_local via random walk MH
    double phi_local_prop = phi_local * tulpa_linalg::safe_exp(R::rnorm(0, 0.1));
    if (std::isfinite(phi_local_prop) && phi_local_prop >= prior_phi_local_lower && phi_local_prop <= prior_phi_local_upper) {
      double ll_curr = 0.0, ll_prop = 0.0;
      for (int i = 0; i < n_spatial; i++) {
        double cond_mean_curr = 0.0, cond_mean_prop = 0.0;
        for (int k = 0; k < nn_local; k++) {
          int neighbor = nn_idx_local(i, k) - 1;
          if (neighbor >= 0 && neighbor < n_spatial) {
            double dist = nn_dist_local(i, k);
            cond_mean_curr += std::exp(-dist / phi_local) * w_local[neighbor];
            cond_mean_prop += std::exp(-dist / phi_local_prop) * w_local[neighbor];
          }
        }
        ll_curr += -0.5 * tau_local * (w_local[i] - cond_mean_curr) * (w_local[i] - cond_mean_curr);
        ll_prop += -0.5 * tau_local * (w_local[i] - cond_mean_prop) * (w_local[i] - cond_mean_prop);
      }
      double log_ratio = ll_prop - ll_curr + std::log(phi_local_prop / phi_local);
      if (std::log(R::runif(0, 1)) < log_ratio) {
        phi_local = phi_local_prop;
      }
    }

    // Update phi_regional via MH
    double phi_regional_prop = phi_regional * tulpa_linalg::safe_exp(R::rnorm(0, 0.1));
    if (std::isfinite(phi_regional_prop) && phi_regional_prop >= prior_phi_regional_lower && phi_regional_prop <= prior_phi_regional_upper) {
      double ll_curr = 0.0, ll_prop = 0.0;
      for (int i = 0; i < n_spatial; i++) {
        double cond_mean_curr = 0.0, cond_mean_prop = 0.0;
        for (int k = 0; k < nn_regional; k++) {
          int neighbor = nn_idx_regional(i, k) - 1;
          if (neighbor >= 0 && neighbor < n_spatial) {
            double dist = nn_dist_regional(i, k);
            cond_mean_curr += std::exp(-dist / phi_regional) * w_regional[neighbor];
            cond_mean_prop += std::exp(-dist / phi_regional_prop) * w_regional[neighbor];
          }
        }
        ll_curr += -0.5 * tau_regional * (w_regional[i] - cond_mean_curr) * (w_regional[i] - cond_mean_curr);
        ll_prop += -0.5 * tau_regional * (w_regional[i] - cond_mean_prop) * (w_regional[i] - cond_mean_prop);
      }
      double log_ratio = ll_prop - ll_curr + std::log(phi_regional_prop / phi_regional);
      if (std::log(R::runif(0, 1)) < log_ratio) {
        phi_regional = phi_regional_prop;
      }
    }

    // Store draws after warmup
    if (iter >= n_warmup && (iter - n_warmup + 1) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_re_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_re_draws[save_idx] = sigma_re;

      for (int s = 0; s < n_spatial; s++) {
        w_local_draws(save_idx, s) = w_local[s];
        w_regional_draws(save_idx, s) = w_regional[s];
      }
      sigma2_local_draws[save_idx] = sigma2_local;
      phi_local_draws[save_idx] = phi_local;
      sigma2_regional_draws[save_idx] = sigma2_regional;
      phi_regional_draws[save_idx] = phi_regional;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws_temp(save_idx, i) = eta_vec[i];
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
    Rcpp::Named("w_local") = w_local_draws,
    Rcpp::Named("w_regional") = w_regional_draws,
    Rcpp::Named("sigma2_local") = sigma2_local_draws,
    Rcpp::Named("phi_local") = phi_local_draws,
    Rcpp::Named("sigma2_regional") = sigma2_regional_draws,
    Rcpp::Named("phi_regional") = phi_regional_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws_temp;
  }

  PutRNGstate();
  return result;
}
