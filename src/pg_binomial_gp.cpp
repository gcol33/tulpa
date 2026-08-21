// pg_binomial_gp.cpp
// GP (sequential NNGP) spatial Gibbs sampler for Pólya-Gamma binomial models

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
//
// The field, its marginal variance and its range are updated by
// tulpa::pg_nngp_scale_update, shared with the multiscale sampler.
// ---------------------------------------------------------------------

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
  const int n_save = tulpa::pg_n_save(n_iter, n_warmup, thin);
  tulpa::PgGibbsCommon C(y, n, X, re_group, n_re_groups, n_save,
                         prior_sigma_re_scale, n_threads, store_eta);
  const int N = C.N;
  const int p = C.p;
  C.require_intercept("NNGP spatial");

  if (N != n_spatial) {
    Rcpp::stop("The NNGP Gibbs kernel maps observation i to location i, so it "
               "needs one observation per location: got %d observation(s) for "
               "%d location(s).", N, n_spatial);
  }
  if (coords.nrow() != n_spatial) {
    Rcpp::stop("`coords` has %d row(s) but `n_spatial` is %d.",
               static_cast<int>(coords.nrow()), n_spatial);
  }
  tulpa::pg_check_pc_prior(prior_sigma_gp_U, prior_sigma_gp_alpha, "gp");
  if (!(prior_phi_lower > 0.0) || !(prior_phi_upper > prior_phi_lower)) {
    Rcpp::stop("The range prior needs 0 < prior_phi_lower < prior_phi_upper; "
               "got [%g, %g].", prior_phi_lower, prior_phi_upper);
  }

  // Per-variant storage
  Rcpp::NumericMatrix gp_draws(n_save, n_spatial);
  Rcpp::NumericVector sigma2_gp_draws(n_save);
  Rcpp::NumericVector phi_gp_draws(n_save);

  // Per-variant state
  tulpa::PgNngpScale gp;
  gp.w.assign(n_spatial, 0.0);
  gp.sigma2 = sigma2_gp_init;
  gp.phi = phi_gp_init;
  gp.top = tulpa::pg_nngp_topology(nn_idx, nn_dist, nn_order, n_spatial, nn);
  Rcpp::NumericVector gp_contrib(N, 0.0);
  std::vector<double> sum_omega_gp(n_spatial, 0.0), sum_resid_gp(n_spatial, 0.0);

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

    // 5. Update the GP scale (field, marginal variance, range)
    for (int i = 0; i < N; i++) {
      C.offset[i] = C.X_beta[i] + C.re_contrib[i];
    }
    tulpa::pg_accumulate_stats(N, nullptr, n_spatial, C.omega.begin(),
                               C.kappa.begin(), C.offset.begin(),
                               sum_omega_gp.data(), sum_resid_gp.data());
    tulpa::pg_nngp_scale_update(
        gp, cov_type, coords, nn_dist, sum_omega_gp, sum_resid_gp,
        prior_sigma_gp_U, prior_sigma_gp_alpha,
        prior_phi_lower, prior_phi_upper);

    // Anchor the field level: the overall GP mean and the intercept are
    // confounded (both shift eta by a constant), and under the NNGP sequential
    // update that level is only weakly identified, so the pair drifts. Centre
    // the field and absorb the removed mean into the intercept -- eta is
    // unchanged and the field/intercept no longer diverge.
    {
      double w_mean = 0.0;
      for (int s = 0; s < n_spatial; s++) w_mean += gp.w[s];
      w_mean /= n_spatial;
      for (int s = 0; s < n_spatial; s++) gp.w[s] -= w_mean;
      C.absorb_level(w_mean);
    }

    // Update GP contributions
    for (int i = 0; i < n_spatial; i++) {
      gp_contrib[i] = gp.w[i];
    }

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      C.save(save_idx);
      for (int s = 0; s < n_spatial; s++) {
        gp_draws(save_idx, s) = gp.w[s];
      }
      sigma2_gp_draws[save_idx] = gp.sigma2;
      phi_gp_draws[save_idx] = gp.phi;
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

  return result;
}
