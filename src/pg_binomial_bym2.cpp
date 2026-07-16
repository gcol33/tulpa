// pg_binomial_bym2.cpp
// BYM2 spatial Gibbs sampler for Pólya-Gamma binomial models
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
// BYM2 Spatial Gibbs sampler
// ---------------------------------------------------------------------

// Forward declaration for BYM2 functions
namespace tulpa {
  Rcpp::NumericVector update_spatial_bym2(
      const Rcpp::NumericVector& kappa,
      const Rcpp::NumericVector& omega,
      const Rcpp::NumericVector& offset,
      const Rcpp::IntegerVector& group,
      const Rcpp::List& adj_list,
      const Rcpp::IntegerVector& n_neighbors,
      Rcpp::NumericVector& phi_scaled,
      Rcpp::NumericVector& theta,
      double sigma_spatial,
      double rho,
      double scale_factor
  );

  double update_sigma_spatial(
      const Rcpp::NumericVector& u,
      double scale
  );

  double update_rho_bym2(
      const Rcpp::NumericVector& phi_scaled,
      const Rcpp::NumericVector& theta,
      double sigma_spatial,
      double scale_factor,
      const Rcpp::NumericVector& sum_omega,
      const Rcpp::NumericVector& sum_resid,
      double alpha,
      double beta
  );

  double update_sigma_spatial_bym2(
      const Rcpp::NumericVector& phi_scaled,
      const Rcpp::NumericVector& theta,
      double rho,
      double scale_factor,
      const Rcpp::NumericVector& sum_omega,
      const Rcpp::NumericVector& sum_resid,
      double prior_scale
  );
}

// Binomial Gibbs sampler with random effects AND spatial effects (BYM2)
// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_bym2(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector spatial_group,
    int n_spatial_units,
    Rcpp::List adj_list,
    Rcpp::IntegerVector n_neighbors,
    double scale_factor,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_sigma_spatial_scale = 2.5,
    double prior_rho_alpha = 0.5,
    double prior_rho_beta = 0.5,
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
  Rcpp::NumericMatrix phi_scaled_draws(n_save, n_spatial_units);
  Rcpp::NumericMatrix theta_draws(n_save, n_spatial_units);
  Rcpp::NumericMatrix u_draws(n_save, n_spatial_units);
  Rcpp::NumericVector sigma_spatial_draws(n_save);
  Rcpp::NumericVector rho_draws(n_save);

  // Per-variant state
  Rcpp::NumericVector phi_scaled(n_spatial_units, 0.0);
  Rcpp::NumericVector theta(n_spatial_units, 0.0);
  Rcpp::NumericVector u(n_spatial_units, 0.0);
  double sigma_spatial = 1.0;
  double rho = 0.5;
  Rcpp::NumericVector spatial_contrib(N);

  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {
    // Steps 1-5: shared core (compute eta, sample omega, update beta/RE)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(C.n_threads_team)
    #endif
    for (int i = 0; i < N; i++) {
      spatial_contrib[i] = u[spatial_group[i] - 1];
    }
    tulpa::pg_gibbs_core_step(
        N, p, C.beta, C.re, C.sigma_re, C.omega, C.eta, C.X_beta, C.re_contrib,
        spatial_contrib, C.offset, C.kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale, C.n_threads_team);

    // 6. Update BYM2 spatial effects | omega, beta, re, sigma_spatial, rho
    // Offset for spatial update = X*beta + re (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(C.n_threads_team)
    #endif
    for (int i = 0; i < N; i++) {
      C.offset[i] = C.X_beta[i] + C.re_contrib[i];
    }
    u = tulpa::update_spatial_bym2(C.kappa, C.omega, C.offset, spatial_group, adj_list, n_neighbors,
                                   phi_scaled, theta, sigma_spatial, rho, scale_factor);

    // Polya-Gamma sufficient statistics per spatial unit (offset excludes the
    // spatial field u): the linear/quadratic data terms both the sigma and rho
    // full conditionals flow through.
    Rcpp::NumericVector sum_omega_s(n_spatial_units, 0.0);
    Rcpp::NumericVector sum_resid_s(n_spatial_units, 0.0);
    for (int i = 0; i < N; i++) {
      int s = spatial_group[i] - 1;
      sum_omega_s[s] += C.omega[i];
      sum_resid_s[s] += C.kappa[i] - C.omega[i] * C.offset[i];
    }

    // 7. Update sigma_spatial from its PG full conditional given the current rho
    //    (a Gaussian in sigma via the standardized field), NOT the iid
    //    half-Cauchy on the deterministic convolution u.
    sigma_spatial = tulpa::update_sigma_spatial_bym2(
        phi_scaled, theta, rho, scale_factor,
        sum_omega_s, sum_resid_s, prior_sigma_spatial_scale);

    // 8. Update rho (mixing proportion) at the just-updated sigma.
    rho = tulpa::update_rho_bym2(phi_scaled, theta, sigma_spatial, scale_factor,
                                  sum_omega_s, sum_resid_s, prior_rho_alpha, prior_rho_beta);

    // Recompute the field at the updated (sigma, rho) so the stored draw and the
    // next iteration's offset use the current scale and mixing weight.
    {
      double sr = std::sqrt(rho + 1e-10);
      double s1 = std::sqrt(1.0 - rho + 1e-10);
      for (int s = 0; s < n_spatial_units; s++)
        u[s] = sigma_spatial * (sr * phi_scaled[s] * scale_factor + s1 * theta[s]);
    }

    // Update spatial contributions (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(C.n_threads_team)
    #endif
    for (int i = 0; i < N; i++) {
      spatial_contrib[i] = u[spatial_group[i] - 1];
    }

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      C.save(save_idx);
      for (int s = 0; s < n_spatial_units; s++) {
        phi_scaled_draws(save_idx, s) = phi_scaled[s];
        theta_draws(save_idx, s) = theta[s];
        u_draws(save_idx, s) = u[s];
      }
      sigma_spatial_draws[save_idx] = sigma_spatial;
      rho_draws[save_idx] = rho;
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter << std::endl;
    }

    // Check for user interrupt
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = C.beta_draws,
    Rcpp::Named("re") = C.re_draws,
    Rcpp::Named("sigma_re") = C.sigma_re_draws,
    Rcpp::Named("phi_scaled") = phi_scaled_draws,
    Rcpp::Named("theta") = theta_draws,
    Rcpp::Named("spatial") = u_draws,
    Rcpp::Named("sigma_spatial") = sigma_spatial_draws,
    Rcpp::Named("rho") = rho_draws
  );

  if (store_eta) {
    result["eta"] = C.eta_draws;
  }

  PutRNGstate();
  return result;
}
