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
//
// The sequential NNGP conditional N(w_i | cond_mean, cond_var) now lives in
// pg_shared.h (tulpa::pg_nngp_conditional) so the single-scale and multiscale
// samplers share one solve (principle #5).

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

  int n_save = (n_iter - n_warmup) / thin;
  tulpa::PgGibbsCommon C(y, n, X.ncol(), n_re_groups, n_save, n_threads, store_eta);
  const int N = C.N;
  const int p = C.p;

  // Per-variant storage
  Rcpp::NumericMatrix gp_draws(n_save, n_spatial);
  Rcpp::NumericVector sigma2_gp_draws(n_save);
  Rcpp::NumericVector phi_gp_draws(n_save);

  // Per-variant state
  std::vector<double> w(n_spatial, 0.0);
  double sigma2_gp = sigma2_gp_init;
  double phi_gp = phi_gp_init;
  Rcpp::NumericVector gp_contrib(N, 0.0);

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    if (verbose && (iter + 1) % 200 == 0) {
      Rcpp::Rcout << "  Iteration " << (iter + 1) << "/" << n_iter << "\n";
    }

    // 1-4. Core Gibbs step (eta, omega, beta, RE) — shared with all variants
    tulpa::pg_gibbs_core_step(
        N, p, C.beta, C.re, C.sigma_re, C.omega, C.eta, C.X_beta, C.re_contrib,
        gp_contrib, C.offset, C.kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale, C.n_threads_team);

    // 5. Update GP effects (sequential NNGP Gibbs)
    for (int i = 0; i < N; i++) {
      C.offset[i] = C.X_beta[i] + C.re_contrib[i];
    }

    // Aggregate likelihood info per spatial location
    std::vector<double> sum_omega_gp(n_spatial, 0.0);
    std::vector<double> sum_resid_gp(n_spatial, 0.0);
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        sum_omega_gp[i] += C.omega[i];
        sum_resid_gp[i] += C.kappa[i] - C.omega[i] * C.offset[i];
      }
    }

    // Field + (sigma2, phi) via the shared NNGP-correct scale update: the field
    // by sequential NNGP conditionals, sigma2 by the conjugate InvGamma on the
    // standardized quadratic form, phi by an MH on the proper NNGP log-density.
    // The previous inline updates left sigma2 railing (w treated as iid, no NNGP
    // structure and no normalizer) and phi doing a data-free random walk (its MH
    // ratio carried no likelihood or prior). prior_sigma_gp_alpha is no longer
    // used (the conjugate InvGamma uses prior_sigma_gp_U as its rate offset).
    tulpa::update_nngp_scale(
        w, sigma2_gp, phi_gp, cov_type, coords, nn_idx, nn_dist, nn_order,
        nn, n_spatial, sum_omega_gp, sum_resid_gp,
        prior_sigma_gp_U, prior_phi_lower, prior_phi_upper);

    // Anchor the field level: the overall GP mean and the intercept are
    // confounded (both shift eta by a constant), and under the NNGP sequential
    // update that level is only weakly identified, so the pair drifts. Center
    // the field and absorb the removed mean into the intercept -- eta is
    // unchanged and the field/intercept no longer diverge.
    {
      double w_mean = 0.0;
      for (int s = 0; s < n_spatial; s++) w_mean += w[s];
      w_mean /= n_spatial;
      for (int s = 0; s < n_spatial; s++) w[s] -= w_mean;
      C.beta[0] += w_mean;
    }

    // Update GP contributions
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        gp_contrib[i] = w[i];
      }
    }

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      C.save(save_idx);
      for (int s = 0; s < n_spatial; s++) {
        gp_draws(save_idx, s) = w[s];
      }
      sigma2_gp_draws[save_idx] = sigma2_gp;
      phi_gp_draws[save_idx] = phi_gp;
      save_idx++;
    }

    if ((iter + 1) % 100 == 0) Rcpp::checkUserInterrupt();
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = C.beta_draws,
    Rcpp::Named("re") = C.re_draws,
    Rcpp::Named("sigma_re") = C.sigma_re_draws,
    Rcpp::Named("gp") = gp_draws,
    Rcpp::Named("sigma2_gp") = sigma2_gp_draws,
    Rcpp::Named("phi_gp") = phi_gp_draws
  );

  if (store_eta) {
    result["eta"] = C.eta_draws;
  }

  PutRNGstate();
  return result;
}
