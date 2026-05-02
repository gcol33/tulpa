// pg_binomial_temporal.cpp
// Multiscale temporal Gibbs sampler for Pólya-Gamma binomial models.
// Trend (RW1/RW2) + seasonal (cyclic RW1) + short-term (AR1/IID).
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
// Multiscale Temporal Gibbs Sampler for Binomial Models
// Supports trend (RW1/RW2) + seasonal (cyclic RW1) + short-term (AR1/IID)
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_temporal(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector time_idx,
    int n_times,
    int seasonal_period,
    int trend_type,
    int short_type,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_sigma_trend_scale = 1.0,
    double prior_sigma_seasonal_scale = 1.0,
    double prior_sigma_short_scale = 1.0,
    double rho_short_init = 0.5,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  // CRITICAL: Must call GetRNGstate/PutRNGstate when using R's RNG from C++
  GetRNGstate();

  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  int n_trend = (trend_type > 0) ? n_times : 0;
  int n_seasonal = (seasonal_period > 0) ? seasonal_period : 0;
  int n_short = (short_type > 0) ? n_times : 0;

  #ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
  #endif

  // Storage
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix trend_draws(n_save, n_trend);
  Rcpp::NumericMatrix seasonal_draws(n_save, n_seasonal);
  Rcpp::NumericMatrix short_draws(n_save, n_short);
  Rcpp::NumericVector sigma_trend_draws(n_save);
  Rcpp::NumericVector sigma_seasonal_draws(n_save);
  Rcpp::NumericVector sigma_short_draws(n_save);
  Rcpp::NumericVector rho_short_draws(n_save);
  Rcpp::NumericMatrix eta_draws_temp;
  if (store_eta) eta_draws_temp = Rcpp::NumericMatrix(n_save, N);

  // Initialize
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  Rcpp::NumericVector trend(n_trend, 0.0);
  Rcpp::NumericVector seasonal(n_seasonal, 0.0);
  Rcpp::NumericVector short_term(n_short, 0.0);
  double sigma_trend = 1.0;
  double sigma_seasonal = 1.0;
  double sigma_short = 1.0;
  double rho_short = rho_short_init;

  // Working vectors
  Rcpp::NumericVector omega(N);
  Rcpp::NumericVector kappa(N);
  Rcpp::NumericVector eta(N);
  Rcpp::NumericVector X_beta(N);
  Rcpp::NumericVector re_contrib(N);
  Rcpp::NumericVector temp_contrib(N);
  Rcpp::NumericVector offset(N);

  for (int i = 0; i < N; i++) {
    omega[i] = 0.5;
    kappa[i] = y[i] - 0.5 * n[i];
    X_beta[i] = 0.0;
    re_contrib[i] = 0.0;
    temp_contrib[i] = 0.0;
  }

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    if (verbose && (iter + 1) % 200 == 0) {
      Rcpp::Rcout << "  Iteration " << (iter + 1) << "/" << n_iter << "\n";
    }

    // 1. Compute temporal contributions
    for (int i = 0; i < N; i++) {
      int t = time_idx[i] - 1;
      double temp_eff = 0.0;
      if (t >= 0 && t < n_times) {
        if (n_trend > 0 && t < n_trend) temp_eff += trend[t];
        if (n_seasonal > 0) temp_eff += seasonal[t % seasonal_period];
        if (n_short > 0 && t < n_short) temp_eff += short_term[t];
      }
      temp_contrib[i] = temp_eff;
    }

    // 2-4. Core Gibbs step (eta, omega, beta, RE) — shared with all variants
    tulpa::pg_gibbs_core_step(
        N, p, beta, re, sigma_re, omega, eta, X_beta, re_contrib,
        temp_contrib, offset, kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale);

    // 5. Update temporal effects
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i];
    }

    // Aggregate for trend
    std::vector<double> sum_omega_t(n_times, 0.0);
    std::vector<double> sum_resid_t(n_times, 0.0);
    for (int i = 0; i < N; i++) {
      int t = time_idx[i] - 1;
      if (t >= 0 && t < n_times) {
        sum_omega_t[t] += omega[i];
        double other_temp = 0.0;
        if (n_seasonal > 0) other_temp += seasonal[t % seasonal_period];
        if (n_short > 0 && t < n_short) other_temp += short_term[t];
        sum_resid_t[t] += kappa[i] - omega[i] * (offset[i] + other_temp);
      }
    }

    // Update trend (RW1)
    if (trend_type == 1) {
      double tau_trend = 1.0 / (sigma_trend * sigma_trend);
      for (int t = 0; t < n_trend; t++) {
        double tau_prior, mean_prior;
        if (t == 0) {
          tau_prior = tau_trend;
          mean_prior = (n_trend > 1) ? trend[1] : 0.0;
        } else if (t == n_trend - 1) {
          tau_prior = tau_trend;
          mean_prior = trend[t - 1];
        } else {
          tau_prior = 2.0 * tau_trend;
          mean_prior = 0.5 * (trend[t - 1] + trend[t + 1]);
        }

        double tau_post = tau_prior + sum_omega_t[t];
        double mean_post = (tau_prior * mean_prior + sum_resid_t[t]) / tau_post;
        trend[t] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
      }

      double ss = 0.0;
      for (int t = 1; t < n_trend; t++) {
        double diff = trend[t] - trend[t - 1];
        ss += diff * diff;
      }
      double shape = prior_sigma_trend_scale + 0.5 * (n_trend - 1);
      double rate = prior_sigma_trend_scale + 0.5 * ss;
      sigma_trend = 1.0 / std::sqrt(R::rgamma(shape, 1.0 / rate));
    }

    // Update seasonal (cyclic RW1)
    if (n_seasonal > 0) {
      std::vector<double> sum_omega_s(seasonal_period, 0.0);
      std::vector<double> sum_resid_s(seasonal_period, 0.0);
      for (int i = 0; i < N; i++) {
        int t = time_idx[i] - 1;
        if (t >= 0) {
          int s = t % seasonal_period;
          sum_omega_s[s] += omega[i];
          double other_temp = 0.0;
          if (n_trend > 0 && t < n_trend) other_temp += trend[t];
          if (n_short > 0 && t < n_short) other_temp += short_term[t];
          sum_resid_s[s] += kappa[i] - omega[i] * (offset[i] + other_temp);
        }
      }

      double tau_seasonal_val = 1.0 / (sigma_seasonal * sigma_seasonal);
      for (int s = 0; s < n_seasonal; s++) {
        int s_prev = (s == 0) ? n_seasonal - 1 : s - 1;
        int s_next = (s == n_seasonal - 1) ? 0 : s + 1;

        double tau_prior = 2.0 * tau_seasonal_val;
        double mean_prior = 0.5 * (seasonal[s_prev] + seasonal[s_next]);

        double tau_post = tau_prior + sum_omega_s[s];
        double mean_post = (tau_prior * mean_prior + sum_resid_s[s]) / tau_post;
        seasonal[s] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
      }

      double mean_s = 0.0;
      for (int s = 0; s < n_seasonal; s++) mean_s += seasonal[s];
      mean_s /= n_seasonal;
      for (int s = 0; s < n_seasonal; s++) seasonal[s] -= mean_s;

      double ss = 0.0;
      for (int s = 0; s < n_seasonal; s++) {
        int s_next = (s == n_seasonal - 1) ? 0 : s + 1;
        double diff = seasonal[s_next] - seasonal[s];
        ss += diff * diff;
      }
      double shape = prior_sigma_seasonal_scale + 0.5 * n_seasonal;
      double rate = prior_sigma_seasonal_scale + 0.5 * ss;
      sigma_seasonal = 1.0 / std::sqrt(R::rgamma(shape, 1.0 / rate));
    }

    // Update short-term (AR1 or IID)
    if (short_type > 0) {
      std::vector<double> sum_omega_sh(n_short, 0.0);
      std::vector<double> sum_resid_sh(n_short, 0.0);
      for (int i = 0; i < N; i++) {
        int t = time_idx[i] - 1;
        if (t >= 0 && t < n_short) {
          sum_omega_sh[t] += omega[i];
          double other_temp = 0.0;
          if (n_trend > 0 && t < n_trend) other_temp += trend[t];
          if (n_seasonal > 0) other_temp += seasonal[t % seasonal_period];
          sum_resid_sh[t] += kappa[i] - omega[i] * (offset[i] + other_temp);
        }
      }

      double tau_short_val = 1.0 / (sigma_short * sigma_short);

      if (short_type == 1) {  // AR1
        for (int t = 0; t < n_short; t++) {
          double tau_prior, mean_prior;
          if (t == 0) {
            tau_prior = tau_short_val * (1.0 - rho_short * rho_short);
            mean_prior = 0.0;
          } else {
            tau_prior = tau_short_val;
            mean_prior = rho_short * short_term[t - 1];
          }

          double tau_post = tau_prior + sum_omega_sh[t];
          double mean_post = (tau_prior * mean_prior + sum_resid_sh[t]) / tau_post;
          short_term[t] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
        }
      } else {  // IID
        for (int t = 0; t < n_short; t++) {
          double tau_post = tau_short_val + sum_omega_sh[t];
          double mean_post = sum_resid_sh[t] / tau_post;
          short_term[t] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
        }
      }

      double ss = 0.0;
      if (short_type == 1) {
        ss = short_term[0] * short_term[0] * (1.0 - rho_short * rho_short);
        for (int t = 1; t < n_short; t++) {
          double resid = short_term[t] - rho_short * short_term[t - 1];
          ss += resid * resid;
        }
      } else {
        for (int t = 0; t < n_short; t++) {
          ss += short_term[t] * short_term[t];
        }
      }
      double shape = prior_sigma_short_scale + 0.5 * n_short;
      double rate = prior_sigma_short_scale + 0.5 * ss;
      sigma_short = 1.0 / std::sqrt(R::rgamma(shape, 1.0 / rate));
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
      for (int t = 0; t < n_trend; t++) {
        trend_draws(save_idx, t) = trend[t];
      }
      for (int s = 0; s < n_seasonal; s++) {
        seasonal_draws(save_idx, s) = seasonal[s];
      }
      for (int t = 0; t < n_short; t++) {
        short_draws(save_idx, t) = short_term[t];
      }
      sigma_trend_draws[save_idx] = sigma_trend;
      sigma_seasonal_draws[save_idx] = sigma_seasonal;
      sigma_short_draws[save_idx] = sigma_short;
      rho_short_draws[save_idx] = rho_short;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws_temp(save_idx, i) = eta[i];
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
    Rcpp::Named("trend") = trend_draws,
    Rcpp::Named("seasonal") = seasonal_draws,
    Rcpp::Named("short_term") = short_draws,
    Rcpp::Named("sigma_trend") = sigma_trend_draws,
    Rcpp::Named("sigma_seasonal") = sigma_seasonal_draws,
    Rcpp::Named("sigma_short") = sigma_short_draws,
    Rcpp::Named("rho_short") = rho_short_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws_temp;
  }

  PutRNGstate();
  return result;
}
