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

  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  // Set number of threads
  #ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  #endif

  // Storage
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix phi_scaled_draws(n_save, n_spatial_units);
  Rcpp::NumericMatrix theta_draws(n_save, n_spatial_units);
  Rcpp::NumericMatrix u_draws(n_save, n_spatial_units);  // Combined spatial effect
  Rcpp::NumericVector sigma_spatial_draws(n_save);
  Rcpp::NumericVector rho_draws(n_save);
  Rcpp::NumericMatrix eta_draws;
  if (store_eta) {
    eta_draws = Rcpp::NumericMatrix(n_save, N);
  }

  // Initialize
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  Rcpp::NumericVector phi_scaled(n_spatial_units, 0.0);  // Structured component
  Rcpp::NumericVector theta(n_spatial_units, 0.0);       // Unstructured component
  Rcpp::NumericVector u(n_spatial_units, 0.0);           // Combined effect
  double sigma_spatial = 1.0;
  double rho = 0.5;  // Start with equal mix
  Rcpp::NumericVector omega(N, 1.0);
  Rcpp::NumericVector kappa(N);
  Rcpp::NumericVector eta(N);
  Rcpp::NumericVector X_beta(N);
  Rcpp::NumericVector re_contrib(N);
  Rcpp::NumericVector spatial_contrib(N);
  Rcpp::NumericVector offset(N);

  // Compute kappa = y - n/2
  for (int i = 0; i < N; i++) {
    kappa[i] = y[i] - n[i] / 2.0;
  }

  // Gibbs iterations
  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {
    // Steps 1-5: shared core (compute eta, sample omega, update beta/RE)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      spatial_contrib[i] = u[spatial_group[i] - 1];
    }
    tulpa::pg_gibbs_core_step(
        N, p, beta, re, sigma_re, omega, eta, X_beta, re_contrib,
        spatial_contrib, offset, kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale);

    // 6. Update BYM2 spatial effects | omega, beta, re, sigma_spatial, rho
    // Offset for spatial update = X*beta + re (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i];
    }
    u = tulpa::update_spatial_bym2(kappa, omega, offset, spatial_group, adj_list, n_neighbors,
                                   phi_scaled, theta, sigma_spatial, rho, scale_factor);

    // 7. Update sigma_spatial
    sigma_spatial = tulpa::update_sigma_spatial(u, prior_sigma_spatial_scale);

    // 8. Update rho (mixing proportion)
    // Need to compute sum_omega and sum_resid for rho update
    Rcpp::NumericVector sum_omega_s(n_spatial_units, 0.0);
    Rcpp::NumericVector sum_resid_s(n_spatial_units, 0.0);
    for (int i = 0; i < N; i++) {
      int s = spatial_group[i] - 1;
      sum_omega_s[s] += omega[i];
      sum_resid_s[s] += kappa[i] - omega[i] * offset[i];
    }
    rho = tulpa::update_rho_bym2(phi_scaled, theta, sigma_spatial, scale_factor,
                                  sum_omega_s, sum_resid_s, prior_rho_alpha, prior_rho_beta);

    // Update spatial contributions (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      spatial_contrib[i] = u[spatial_group[i] - 1];
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
      for (int s = 0; s < n_spatial_units; s++) {
        phi_scaled_draws(save_idx, s) = phi_scaled[s];
        theta_draws(save_idx, s) = theta[s];
        u_draws(save_idx, s) = u[s];
      }
      sigma_spatial_draws[save_idx] = sigma_spatial;
      rho_draws[save_idx] = rho;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws(save_idx, i) = eta[i];
        }
      }
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
    Rcpp::Named("beta") = beta_draws,
    Rcpp::Named("re") = re_draws,
    Rcpp::Named("sigma_re") = sigma_re_draws,
    Rcpp::Named("phi_scaled") = phi_scaled_draws,
    Rcpp::Named("theta") = theta_draws,
    Rcpp::Named("spatial") = u_draws,
    Rcpp::Named("sigma_spatial") = sigma_spatial_draws,
    Rcpp::Named("rho") = rho_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws;
  }

  PutRNGstate();
  return result;
}
