// pg_binomial_bym2.cpp
// BYM2 spatial Gibbs sampler for Pólya-Gamma binomial models

#include "pg_shared.h"
#include "pg_spatial.h"
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
  const int n_save = tulpa::pg_n_save(n_iter, n_warmup, thin);
  tulpa::PgGibbsCommon C(y, n, X, re_group, n_re_groups, n_save,
                         prior_sigma_re_scale, n_threads, store_eta);
  const int N = C.N;
  const int p = C.p;
  C.require_intercept("BYM2 spatial");

  tulpa::pg_check_index(spatial_group, N, n_spatial_units, "spatial_group");
  const tulpa::PgAdjacency adj =
      tulpa::pg_build_adjacency(adj_list, n_neighbors, n_spatial_units);

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
  Rcpp::NumericVector sum_omega_s(n_spatial_units);
  Rcpp::NumericVector sum_resid_s(n_spatial_units);

  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {
    // Steps 1-5: shared core (compute eta, sample omega, update beta/RE)
    tulpa_parallel_for(C.n_threads_team, N, [&](int i) {
      spatial_contrib[i] = u[spatial_group[i] - 1];
    });
    tulpa::pg_gibbs_core_step(
        N, p, C.beta, C.re, C.sigma_re, C.omega, C.eta, C.X_beta, C.re_contrib,
        spatial_contrib, C.offset, C.kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale, C.n_threads_team);

    // 6. Update BYM2 spatial effects | omega, beta, re, sigma_spatial, rho
    // Offset for spatial update = X*beta + re
    tulpa_parallel_for(C.n_threads_team, N, [&](int i) {
      C.offset[i] = C.X_beta[i] + C.re_contrib[i];
    });
    double bym2_removed = 0.0;
    tulpa::update_spatial_bym2(C.kappa, C.omega, C.offset, spatial_group, adj,
                               phi_scaled, theta, sigma_spatial, rho,
                               scale_factor, u, bym2_removed);
    // Absorb the field level removed by centering phi into the intercept so eta
    // is unchanged (posterior-invariant), and refresh the cached X_beta /
    // offset that the sigma and rho conditionals below read.
    C.absorb_level(bym2_removed);
    for (int i = 0; i < N; i++) {
      C.offset[i] = C.X_beta[i] + C.re_contrib[i];
    }

    // Polya-Gamma sufficient statistics per spatial unit (offset excludes the
    // spatial field u): the linear/quadratic data terms both the sigma and rho
    // full conditionals flow through.
    tulpa::pg_accumulate_stats(N, spatial_group.begin(), n_spatial_units,
                               C.omega.begin(), C.kappa.begin(),
                               C.offset.begin(),
                               sum_omega_s.begin(), sum_resid_s.begin());

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

    // Update spatial contributions
    tulpa_parallel_for(C.n_threads_team, N, [&](int i) {
      spatial_contrib[i] = u[spatial_group[i] - 1];
    });

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

  return result;
}
