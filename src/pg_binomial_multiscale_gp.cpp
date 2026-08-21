// pg_binomial_multiscale_gp.cpp
// Multiscale GP Gibbs sampler (local + regional components) for Pólya-Gamma
// binomial models.

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
//
// Each scale's sweep (field, marginal variance, range) is
// tulpa::pg_nngp_scale_update, shared with the single-scale kernel.
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
  const int n_save = tulpa::pg_n_save(n_iter, n_warmup, thin);
  tulpa::PgGibbsCommon C(y, n, X, re_group, n_re_groups, n_save,
                         prior_sigma_re_scale, n_threads, store_eta);
  const int N = C.N;
  const int p = C.p;
  C.require_intercept("multiscale NNGP spatial");

  if (N != n_spatial) {
    Rcpp::stop("The multiscale NNGP Gibbs kernel maps observation i to "
               "location i, so it needs one observation per location: got %d "
               "observation(s) for %d location(s).", N, n_spatial);
  }
  if (coords.nrow() != n_spatial) {
    Rcpp::stop("`coords` has %d row(s) but `n_spatial` is %d.",
               static_cast<int>(coords.nrow()), n_spatial);
  }
  tulpa::pg_check_pc_prior(prior_sigma_local_U, prior_sigma_local_alpha, "local");
  tulpa::pg_check_pc_prior(prior_sigma_regional_U, prior_sigma_regional_alpha,
                           "regional");

  if (verbose) {
    Rcpp::Rcout << "PG Binomial Gibbs sampler with multiscale GP spatial\n";
    Rcpp::Rcout << "  N = " << N << ", p = " << p << "\n";
    Rcpp::Rcout << "  n_spatial = " << n_spatial << "\n";
    Rcpp::Rcout << "  nn_local = " << nn_local << ", nn_regional = " << nn_regional << "\n";
  }

  // Local GP scale
  tulpa::PgNngpScale local;
  local.w.assign(n_spatial, 0.0);
  local.sigma2 = sigma2_local_init;
  local.phi = phi_local_init;
  local.top = tulpa::pg_nngp_topology(nn_idx_local, nn_dist_local,
                                      nn_order_local, n_spatial, nn_local);

  // Regional GP scale
  tulpa::PgNngpScale regional;
  regional.w.assign(n_spatial, 0.0);
  regional.sigma2 = sigma2_regional_init;
  regional.phi = phi_regional_init;
  regional.top = tulpa::pg_nngp_topology(nn_idx_regional, nn_dist_regional,
                                         nn_order_regional, n_spatial,
                                         nn_regional);

  // Per-variant working vectors
  Rcpp::NumericVector local_contrib(N, 0.0);
  Rcpp::NumericVector regional_contrib(N, 0.0);
  Rcpp::NumericVector combined_contrib(N, 0.0);
  std::vector<double> sum_omega_sc(n_spatial, 0.0), sum_resid_sc(n_spatial, 0.0);

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
    for (int i = 0; i < N; i++) {
      combined_contrib[i] = local_contrib[i] + regional_contrib[i];
    }

    // 2-4. Core Gibbs step (eta, omega, beta, RE) — shared with all variants
    tulpa::pg_gibbs_core_step(
        N, p, C.beta, C.re, C.sigma_re, C.omega, C.eta, C.X_beta, C.re_contrib,
        combined_contrib, C.offset, C.kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale, C.n_threads_team);

    // 5. Update the local scale, conditioning on the regional contribution.
    for (int i = 0; i < N; i++) {
      C.offset[i] = C.X_beta[i] + C.re_contrib[i] + regional_contrib[i];
    }
    tulpa::pg_accumulate_stats(N, nullptr, n_spatial, C.omega.begin(),
                               C.kappa.begin(), C.offset.begin(),
                               sum_omega_sc.data(), sum_resid_sc.data());
    tulpa::pg_nngp_scale_update(
        local, cov_type, coords, nn_dist_local, sum_omega_sc, sum_resid_sc,
        prior_sigma_local_U, prior_sigma_local_alpha,
        prior_phi_local_lower, prior_phi_local_upper);
    // Anchor the local field level into the intercept (both scales share the
    // constant direction with the intercept; leaving it free lets them drift).
    {
      double m = 0.0;
      for (int s = 0; s < n_spatial; s++) m += local.w[s];
      m /= n_spatial;
      for (int s = 0; s < n_spatial; s++) local.w[s] -= m;
      C.absorb_level(m);
    }
    for (int i = 0; i < n_spatial; i++) local_contrib[i] = local.w[i];

    // 6. Update the regional scale, conditioning on the just-updated local
    //    contribution.
    for (int i = 0; i < N; i++) {
      C.offset[i] = C.X_beta[i] + C.re_contrib[i] + local_contrib[i];
    }
    tulpa::pg_accumulate_stats(N, nullptr, n_spatial, C.omega.begin(),
                               C.kappa.begin(), C.offset.begin(),
                               sum_omega_sc.data(), sum_resid_sc.data());
    tulpa::pg_nngp_scale_update(
        regional, cov_type, coords, nn_dist_regional, sum_omega_sc,
        sum_resid_sc, prior_sigma_regional_U, prior_sigma_regional_alpha,
        prior_phi_regional_lower, prior_phi_regional_upper);
    {
      double m = 0.0;
      for (int s = 0; s < n_spatial; s++) m += regional.w[s];
      m /= n_spatial;
      for (int s = 0; s < n_spatial; s++) regional.w[s] -= m;
      C.absorb_level(m);
    }
    for (int i = 0; i < n_spatial; i++) regional_contrib[i] = regional.w[i];

    // Store draws after warmup
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      C.save(save_idx);
      for (int s = 0; s < n_spatial; s++) {
        w_local_draws(save_idx, s) = local.w[s];
        w_regional_draws(save_idx, s) = regional.w[s];
      }
      sigma2_local_draws[save_idx] = local.sigma2;
      phi_local_draws[save_idx] = local.phi;
      sigma2_regional_draws[save_idx] = regional.sigma2;
      phi_regional_draws[save_idx] = regional.phi;
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

  return result;
}
