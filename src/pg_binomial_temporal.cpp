// pg_binomial_temporal.cpp
// Multiscale temporal Gibbs sampler for Pólya-Gamma binomial models.
// Trend (RW1) + seasonal (cyclic RW1) + short-term (AR1/IID).

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
// Supports trend (RW1) + seasonal (cyclic RW1) + short-term (AR1/IID)
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
  if (trend_type < 0 || trend_type > 1) {
    Rcpp::stop("`trend_type` must be 0 (none) or 1 (RW1); got %d. The temporal "
               "Gibbs kernel implements an RW1 trend only.", trend_type);
  }
  if (short_type < 0 || short_type > 2) {
    Rcpp::stop("`short_type` must be 0 (none), 1 (AR1) or 2 (IID); got %d.",
               short_type);
  }
  if (seasonal_period < 0) {
    Rcpp::stop("`seasonal_period` must be >= 0; got %d.", seasonal_period);
  }

  const int n_save = tulpa::pg_n_save(n_iter, n_warmup, thin);
  tulpa::PgGibbsCommon C(y, n, X, re_group, n_re_groups, n_save,
                         prior_sigma_re_scale, n_threads, store_eta);
  const int N = C.N;
  const int p = C.p;
  C.require_intercept("temporal");

  tulpa::pg_check_index(time_idx, N, n_times, "time_idx");

  const int n_trend = (trend_type > 0) ? n_times : 0;
  const int n_seasonal = (seasonal_period > 0) ? seasonal_period : 0;
  const int n_short = (short_type > 0) ? n_times : 0;

  // Per-variant storage
  Rcpp::NumericMatrix trend_draws(n_save, n_trend);
  Rcpp::NumericMatrix seasonal_draws(n_save, n_seasonal);
  Rcpp::NumericMatrix short_draws(n_save, n_short);
  Rcpp::NumericVector sigma_trend_draws(n_save);
  Rcpp::NumericVector sigma_seasonal_draws(n_save);
  Rcpp::NumericVector sigma_short_draws(n_save);
  Rcpp::NumericVector rho_short_draws(n_save);

  // Per-variant state
  Rcpp::NumericVector trend(n_trend, 0.0);
  Rcpp::NumericVector seasonal(n_seasonal, 0.0);
  Rcpp::NumericVector short_term(n_short, 0.0);
  tulpa::PgScaleState sigma_trend = tulpa::pg_scale_state_init(prior_sigma_trend_scale);
  tulpa::PgScaleState sigma_seasonal = tulpa::pg_scale_state_init(prior_sigma_seasonal_scale);
  tulpa::PgScaleState sigma_short = tulpa::pg_scale_state_init(prior_sigma_short_scale);
  double rho_short = rho_short_init;
  Rcpp::NumericVector temp_contrib(N, 0.0);

  // Season of each observation, 1-based (the time index is fixed for the run).
  Rcpp::IntegerVector season_group(n_seasonal > 0 ? N : 0);
  for (int i = 0; i < season_group.size(); i++) {
    season_group[i] = ((time_idx[i] - 1) % seasonal_period) + 1;
  }

  // Per-arm working offset: everything in eta except the arm being updated.
  std::vector<double> arm_offset(N);
  std::vector<double> sum_omega_t(n_times, 0.0), sum_resid_t(n_times, 0.0);
  std::vector<double> sum_omega_s(n_seasonal, 0.0), sum_resid_s(n_seasonal, 0.0);

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    if (verbose && (iter + 1) % 200 == 0) {
      Rcpp::Rcout << "  Iteration " << (iter + 1) << "/" << n_iter << "\n";
    }

    // 1. Compute temporal contributions
    for (int i = 0; i < N; i++) {
      const int t = time_idx[i] - 1;
      double temp_eff = 0.0;
      if (n_trend > 0) temp_eff += trend[t];
      if (n_seasonal > 0) temp_eff += seasonal[t % seasonal_period];
      if (n_short > 0) temp_eff += short_term[t];
      temp_contrib[i] = temp_eff;
    }

    // 2-4. Core Gibbs step (eta, omega, beta, RE) — shared with all variants
    tulpa::pg_gibbs_core_step(
        N, p, C.beta, C.re, C.sigma_re, C.omega, C.eta, C.X_beta, C.re_contrib,
        temp_contrib, C.offset, C.kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale, C.n_threads_team);

    // 5. Update temporal effects
    for (int i = 0; i < N; i++) {
      C.offset[i] = C.X_beta[i] + C.re_contrib[i];
    }

    // Update trend (RW1)
    if (n_trend > 0) {
      for (int i = 0; i < N; i++) {
        const int t = time_idx[i] - 1;
        double other = 0.0;
        if (n_seasonal > 0) other += seasonal[t % seasonal_period];
        if (n_short > 0) other += short_term[t];
        arm_offset[i] = C.offset[i] + other;
      }
      tulpa::pg_accumulate_stats(N, time_idx.begin(), n_times, C.omega.begin(),
                                 C.kappa.begin(), arm_offset.data(),
                                 sum_omega_t.data(), sum_resid_t.data());

      const double tau_trend = 1.0 / (sigma_trend.sigma * sigma_trend.sigma);
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

        const double tau_post = tau_prior + sum_omega_t[t];
        const double mean_post = (tau_prior * mean_prior + sum_resid_t[t]) / tau_post;
        trend[t] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
      }

      // RW1 on n_trend levels has n_trend - 1 independent increments, so that
      // is the rank of the quadratic form the scale conditional reads.
      double ss = 0.0;
      for (int t = 1; t < n_trend; t++) {
        const double diff = trend[t] - trend[t - 1];
        ss += diff * diff;
      }
      tulpa::pg_update_scale_halfcauchy(ss, n_trend - 1,
                                        prior_sigma_trend_scale, sigma_trend);
    }

    // Update seasonal (cyclic RW1)
    if (n_seasonal > 0) {
      for (int i = 0; i < N; i++) {
        const int t = time_idx[i] - 1;
        double other = 0.0;
        if (n_trend > 0) other += trend[t];
        if (n_short > 0) other += short_term[t];
        arm_offset[i] = C.offset[i] + other;
      }
      tulpa::pg_accumulate_stats(N, season_group.begin(), n_seasonal,
                                 C.omega.begin(), C.kappa.begin(),
                                 arm_offset.data(),
                                 sum_omega_s.data(), sum_resid_s.data());

      const double tau_seasonal_val =
          1.0 / (sigma_seasonal.sigma * sigma_seasonal.sigma);
      for (int s = 0; s < n_seasonal; s++) {
        const int s_prev = (s == 0) ? n_seasonal - 1 : s - 1;
        const int s_next = (s == n_seasonal - 1) ? 0 : s + 1;

        const double tau_prior = 2.0 * tau_seasonal_val;
        const double mean_prior = 0.5 * (seasonal[s_prev] + seasonal[s_next]);

        const double tau_post = tau_prior + sum_omega_s[s];
        const double mean_post = (tau_prior * mean_prior + sum_resid_s[s]) / tau_post;
        seasonal[s] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
      }

      double ss = 0.0;
      for (int s = 0; s < n_seasonal; s++) {
        const int s_next = (s == n_seasonal - 1) ? 0 : s + 1;
        const double diff = seasonal[s_next] - seasonal[s];
        ss += diff * diff;
      }
      // Cyclic (ring) RW1 with a sum-to-zero constraint has rank
      // n_seasonal - 1 (one constant null vector).
      tulpa::pg_update_scale_halfcauchy(ss, n_seasonal - 1,
                                        prior_sigma_seasonal_scale,
                                        sigma_seasonal);
    }

    // Update short-term (AR1 or IID)
    if (n_short > 0) {
      for (int i = 0; i < N; i++) {
        const int t = time_idx[i] - 1;
        double other = 0.0;
        if (n_trend > 0) other += trend[t];
        if (n_seasonal > 0) other += seasonal[t % seasonal_period];
        arm_offset[i] = C.offset[i] + other;
      }
      tulpa::pg_accumulate_stats(N, time_idx.begin(), n_short, C.omega.begin(),
                                 C.kappa.begin(), arm_offset.data(),
                                 sum_omega_t.data(), sum_resid_t.data());

      const double tau_short_val = 1.0 / (sigma_short.sigma * sigma_short.sigma);

      if (short_type == 1) {  // AR1
        const double omr2 = 1.0 + rho_short * rho_short;
        for (int t = 0; t < n_short; t++) {
          // The full conditional uses BOTH neighbours: an interior t has prior
          // precision tau*(1+rho^2) and mean rho*(x_{t-1}+x_{t+1})/(1+rho^2);
          // the endpoints keep the single available neighbour at precision tau.
          double tau_prior, mean_prior;
          const bool has_prev = (t > 0), has_next = (t < n_short - 1);
          if (has_prev && has_next) {
            tau_prior  = tau_short_val * omr2;
            mean_prior = rho_short * (short_term[t - 1] + short_term[t + 1]) / omr2;
          } else if (has_next) {          // first
            tau_prior  = tau_short_val;
            mean_prior = rho_short * short_term[t + 1];
          } else {                        // last
            tau_prior  = tau_short_val;
            mean_prior = rho_short * short_term[t - 1];
          }
          const double tau_post = tau_prior + sum_omega_t[t];
          const double mean_post = (tau_prior * mean_prior + sum_resid_t[t]) / tau_post;
          short_term[t] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
        }

        // Sample rho_short: reflected-normal random-walk MH with a
        // Uniform(-1, 1) prior; the AR1 log-density in rho is
        // 0.5*log(1-rho^2) - 0.5*tau*ss(rho).
        auto ar1_ss = [&](double r) {
          double s = short_term[0] * short_term[0] * (1.0 - r * r);
          for (int t = 1; t < n_short; t++) {
            const double d = short_term[t] - r * short_term[t - 1];
            s += d * d;
          }
          return s;
        };
        const double rho_prop = rho_short + R::rnorm(0, 0.08);
        if (rho_prop > -0.999 && rho_prop < 0.999) {
          const double lp_c = 0.5 * std::log(1.0 - rho_short * rho_short)
                              - 0.5 * tau_short_val * ar1_ss(rho_short);
          const double lp_p = 0.5 * std::log(1.0 - rho_prop * rho_prop)
                              - 0.5 * tau_short_val * ar1_ss(rho_prop);
          if (std::log(R::runif(0, 1)) < lp_p - lp_c) rho_short = rho_prop;
        }
      } else {  // IID
        for (int t = 0; t < n_short; t++) {
          const double tau_post = tau_short_val + sum_omega_t[t];
          const double mean_post = sum_resid_t[t] / tau_post;
          short_term[t] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
        }
      }

      double ss = 0.0;
      if (short_type == 1) {
        ss = short_term[0] * short_term[0] * (1.0 - rho_short * rho_short);
        for (int t = 1; t < n_short; t++) {
          const double resid = short_term[t] - rho_short * short_term[t - 1];
          ss += resid * resid;
        }
      } else {
        for (int t = 0; t < n_short; t++) {
          ss += short_term[t] * short_term[t];
        }
      }
      // Both the stationary AR1 and the IID quadratic forms are full rank.
      tulpa::pg_update_scale_halfcauchy(ss, n_short, prior_sigma_short_scale,
                                        sigma_short);
    }

    // Centre the intrinsic arms and absorb the removed levels into the
    // intercept, so eta is unchanged and the move is posterior-invariant.
    // Discarding a removed mean instead lags the intercept behind the field by
    // that amount each sweep and drives the variance up.
    //
    // Both scale updates above read only DIFFERENCES of their arm, which a
    // constant shift leaves untouched, so centring here rather than before them
    // gives the same draws. Placed after every arm's update so none of them
    // reads a C.offset that this has already invalidated.
    {
      double level = 0.0;
      if (n_trend > 0) {
        double m = 0.0;
        for (int t = 0; t < n_trend; t++) m += trend[t];
        m /= n_trend;
        for (int t = 0; t < n_trend; t++) trend[t] -= m;
        level += m;
      }
      if (n_seasonal > 0) {
        double m = 0.0;
        for (int s = 0; s < n_seasonal; s++) m += seasonal[s];
        m /= n_seasonal;
        for (int s = 0; s < n_seasonal; s++) seasonal[s] -= m;
        level += m;
      }
      C.absorb_level(level);
    }

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      C.save(save_idx);
      for (int t = 0; t < n_trend; t++) {
        trend_draws(save_idx, t) = trend[t];
      }
      for (int s = 0; s < n_seasonal; s++) {
        seasonal_draws(save_idx, s) = seasonal[s];
      }
      for (int t = 0; t < n_short; t++) {
        short_draws(save_idx, t) = short_term[t];
      }
      sigma_trend_draws[save_idx] = sigma_trend.sigma;
      sigma_seasonal_draws[save_idx] = sigma_seasonal.sigma;
      sigma_short_draws[save_idx] = sigma_short.sigma;
      rho_short_draws[save_idx] = rho_short;
      save_idx++;
    }

    if ((iter + 1) % 100 == 0) Rcpp::checkUserInterrupt();
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = C.beta_draws,
    Rcpp::Named("re") = C.re_draws,
    Rcpp::Named("sigma_re") = C.sigma_re_draws,
    Rcpp::Named("trend") = trend_draws,
    Rcpp::Named("seasonal") = seasonal_draws,
    Rcpp::Named("short_term") = short_draws,
    Rcpp::Named("sigma_trend") = sigma_trend_draws,
    Rcpp::Named("sigma_seasonal") = sigma_seasonal_draws,
    Rcpp::Named("sigma_short") = sigma_short_draws,
    Rcpp::Named("rho_short") = rho_short_draws
  );

  if (store_eta) {
    result["eta"] = C.eta_draws;
  }

  return result;
}
