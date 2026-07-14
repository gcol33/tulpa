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

// One NNGP scale's Gibbs sweep, shared by the local and regional components
// (principle #5 -- the two scales differ only in their neighbour structure and
// hyperprior bounds). Updates the field `w` by sequential NNGP conditionals
// (tulpa::pg_nngp_conditional -- the same kriging solve the single-scale sampler
// uses, whose weights are bounded, unlike the previous raw covariance-weighted
// sum that could diverge), then (sigma2, phi) by their NNGP-correct full
// conditionals: sigma2 ~ InvGamma on the standardized quadratic form
// sum_i (w_i - m_i)^2 / v0_i (m_i, v0_i evaluated at sigma2 = 1, exploiting that
// the kriging mean is scale-invariant and cond_var scales linearly), and phi by
// a log-random-walk MH on the proper NNGP log-density. `sum_omega` / `sum_resid`
// are the Polya-Gamma likelihood aggregates for this scale (built by the caller
// from the offset that holds the OTHER scale). Indexing follows
// pg_nngp_conditional: `idx` is the ordered position, `nn_order[idx]` the
// original location id.
static inline void update_nngp_scale(
    std::vector<double>& w, double& sigma2, double& phi, int cov_type,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx, const Rcpp::NumericMatrix& nn_dist,
    const Rcpp::IntegerVector& nn_order, int nn, int n_spatial,
    const std::vector<double>& sum_omega, const std::vector<double>& sum_resid,
    double prior_sigma_U, double prior_phi_lower, double prior_phi_upper
) {
  // 1. Field: sequential NNGP Gibbs update.
  for (int idx = 0; idx < n_spatial; idx++) {
    int obs_i = nn_order[idx];
    double cm, cv;
    tulpa::pg_nngp_conditional(idx, w, sigma2, phi, cov_type,
                               coords, nn_idx, nn_dist, nn_order, nn, cm, cv);
    double tau_prior = 1.0 / cv;
    double tau_post  = tau_prior + sum_omega[obs_i];
    double mean_post = (tau_prior * cm + sum_resid[obs_i]) / tau_post;
    w[obs_i] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
  }

  // 2. sigma2 | w, phi ~ InvGamma(n/2 + 1, Q0/2 + prior_sigma_U), where the
  //    phi-only quadratic form Q0 = sum_i (w_i - m_i)^2 / v0_i uses conditionals
  //    at sigma2 = 1.
  double Q0 = 0.0;
  for (int idx = 0; idx < n_spatial; idx++) {
    int obs_i = nn_order[idx];
    double m0, v0;
    tulpa::pg_nngp_conditional(idx, w, 1.0, phi, cov_type,
                               coords, nn_idx, nn_dist, nn_order, nn, m0, v0);
    double r = w[obs_i] - m0;
    Q0 += r * r / v0;
  }
  double shape = 0.5 * n_spatial + 1.0;
  double rate  = 0.5 * Q0 + prior_sigma_U;
  sigma2 = 1.0 / R::rgamma(shape, 1.0 / rate);

  // 3. phi: log-random-walk MH on the proper NNGP log-density.
  double phi_prop = phi * tulpa_linalg::safe_exp(R::rnorm(0, 0.1));
  if (std::isfinite(phi_prop) &&
      phi_prop >= prior_phi_lower && phi_prop <= prior_phi_upper) {
    double ll_curr = 0.0, ll_prop = 0.0;
    for (int idx = 0; idx < n_spatial; idx++) {
      int obs_i = nn_order[idx];
      double cm_c, cv_c, cm_p, cv_p;
      tulpa::pg_nngp_conditional(idx, w, sigma2, phi, cov_type,
                                 coords, nn_idx, nn_dist, nn_order, nn, cm_c, cv_c);
      tulpa::pg_nngp_conditional(idx, w, sigma2, phi_prop, cov_type,
                                 coords, nn_idx, nn_dist, nn_order, nn, cm_p, cv_p);
      double rc = w[obs_i] - cm_c, rp = w[obs_i] - cm_p;
      ll_curr += -0.5 * std::log(cv_c) - 0.5 * rc * rc / cv_c;
      ll_prop += -0.5 * std::log(cv_p) - 0.5 * rp * rp / cv_p;
    }
    double log_ratio = ll_prop - ll_curr + std::log(phi_prop / phi);
    if (std::log(R::runif(0, 1)) < log_ratio) phi = phi_prop;
  }
}

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

  int n_save = (n_iter - n_warmup) / thin;
  tulpa::PgGibbsCommon C(y, n, X.ncol(), n_re_groups, n_save, n_threads, store_eta);
  const int N = C.N;
  const int p = C.p;

  if (verbose) {
    Rcpp::Rcout << "PG Binomial Gibbs sampler with multiscale GP spatial\n";
    Rcpp::Rcout << "  N = " << N << ", p = " << p << "\n";
    Rcpp::Rcout << "  n_spatial = " << n_spatial << "\n";
    Rcpp::Rcout << "  nn_local = " << nn_local << ", nn_regional = " << nn_regional << "\n";
  }

  // Local GP effects
  std::vector<double> w_local(n_spatial, 0.0);
  double sigma2_local = sigma2_local_init;
  double phi_local = phi_local_init;

  // Regional GP effects
  std::vector<double> w_regional(n_spatial, 0.0);
  double sigma2_regional = sigma2_regional_init;
  double phi_regional = phi_regional_init;

  // Per-variant working vectors
  Rcpp::NumericVector local_contrib(N, 0.0);
  Rcpp::NumericVector regional_contrib(N, 0.0);

  // Per-variant draw storage
  Rcpp::NumericMatrix w_local_draws(n_save, n_spatial);
  Rcpp::NumericMatrix w_regional_draws(n_save, n_spatial);
  Rcpp::NumericVector sigma2_local_draws(n_save);
  Rcpp::NumericVector phi_local_draws(n_save);
  Rcpp::NumericVector sigma2_regional_draws(n_save);
  Rcpp::NumericVector phi_regional_draws(n_save);

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
        N, p, C.beta, C.re, C.sigma_re, C.omega, C.eta, C.X_beta, C.re_contrib,
        combined_contrib, C.offset, C.kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale, C.n_threads_team);

    // 5. Update the local scale (field + sigma2 + phi), conditioning on the
    //    regional contribution. sum_resid uses the full current offset.
    for (int i = 0; i < N; i++) {
      C.offset[i] = C.X_beta[i] + C.re_contrib[i] + regional_contrib[i];
    }
    std::vector<double> sum_omega_local(n_spatial, 0.0);
    std::vector<double> sum_resid_local(n_spatial, 0.0);
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        sum_omega_local[i] += C.omega[i];
        sum_resid_local[i] += C.kappa[i] - C.omega[i] * C.offset[i];
      }
    }
    update_nngp_scale(w_local, sigma2_local, phi_local, cov_type,
                      coords, nn_idx_local, nn_dist_local, nn_order_local,
                      nn_local, n_spatial, sum_omega_local, sum_resid_local,
                      prior_sigma_local_U, prior_phi_local_lower,
                      prior_phi_local_upper);
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) local_contrib[i] = w_local[i];
    }

    // 6. Update the regional scale, conditioning on the (just-updated) local
    //    contribution.
    for (int i = 0; i < N; i++) {
      C.offset[i] = C.X_beta[i] + C.re_contrib[i] + local_contrib[i];
    }
    std::vector<double> sum_omega_regional(n_spatial, 0.0);
    std::vector<double> sum_resid_regional(n_spatial, 0.0);
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        sum_omega_regional[i] += C.omega[i];
        sum_resid_regional[i] += C.kappa[i] - C.omega[i] * C.offset[i];
      }
    }
    update_nngp_scale(w_regional, sigma2_regional, phi_regional, cov_type,
                      coords, nn_idx_regional, nn_dist_regional,
                      nn_order_regional, nn_regional, n_spatial,
                      sum_omega_regional, sum_resid_regional,
                      prior_sigma_regional_U, prior_phi_regional_lower,
                      prior_phi_regional_upper);
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) regional_contrib[i] = w_regional[i];
    }

    // Store draws after warmup
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      C.save(save_idx);
      for (int s = 0; s < n_spatial; s++) {
        w_local_draws(save_idx, s) = w_local[s];
        w_regional_draws(save_idx, s) = w_regional[s];
      }
      sigma2_local_draws[save_idx] = sigma2_local;
      phi_local_draws[save_idx] = phi_local;
      sigma2_regional_draws[save_idx] = sigma2_regional;
      phi_regional_draws[save_idx] = phi_regional;
      save_idx++;
    }

    if ((iter + 1) % 100 == 0) Rcpp::checkUserInterrupt();
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = C.beta_draws,
    Rcpp::Named("re") = C.re_draws,
    Rcpp::Named("sigma_re") = C.sigma_re_draws,
    Rcpp::Named("w_local") = w_local_draws,
    Rcpp::Named("w_regional") = w_regional_draws,
    Rcpp::Named("sigma2_local") = sigma2_local_draws,
    Rcpp::Named("phi_local") = phi_local_draws,
    Rcpp::Named("sigma2_regional") = sigma2_regional_draws,
    Rcpp::Named("phi_regional") = phi_regional_draws
  );

  if (store_eta) {
    result["eta"] = C.eta_draws;
  }

  PutRNGstate();
  return result;
}
