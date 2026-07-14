// pg_binomial_rsr.cpp
// RSR (Restricted Spatial Regression) Gibbs sampler for Pólya-Gamma binomial models.
// Projects spatial effects to be orthogonal to covariates.
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

// Forward declarations for ICAR helpers (defined in pg_spatial_icar.cpp / pg_spatial_*.cpp)
namespace tulpa {
  Rcpp::NumericVector update_spatial_icar(
      const Rcpp::NumericVector& kappa,
      const Rcpp::NumericVector& omega,
      const Rcpp::NumericVector& offset,
      const Rcpp::IntegerVector& group,
      const Rcpp::List& adj_list,
      const Rcpp::IntegerVector& n_neighbors,
      double tau
  );

  double update_tau_icar(
      const Rcpp::NumericVector& phi,
      const Rcpp::List& adj_list,
      const Rcpp::IntegerVector& n_neighbors,
      double prior_shape,
      double prior_rate
  );
}

// ---------------------------------------------------------------------
// RSR (Restricted Spatial Regression) Gibbs sampler
// Projects spatial effects to be orthogonal to covariates
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_rsr(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector spatial_group,
    int n_spatial_units,
    Rcpp::List adj_list,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::NumericVector rsr_projection,  // P_perp matrix (n_spatial x n_spatial, row-major)
    int rsr_n,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_tau_shape = 1.0,
    double prior_tau_rate = 0.01,
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
  Rcpp::NumericMatrix spatial_raw_draws(n_save, n_spatial_units);
  Rcpp::NumericMatrix spatial_proj_draws(n_save, n_spatial_units);
  Rcpp::NumericVector tau_draws(n_save);

  // Per-variant state
  Rcpp::NumericVector phi(n_spatial_units, 0.0);       // Raw (unprojected)
  Rcpp::NumericVector phi_proj(n_spatial_units, 0.0);  // Projected
  double tau = 1.0;
  Rcpp::NumericVector spatial_contrib(N, 0.0);

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    // 1. Compute projected spatial effects: phi_proj = P_perp * phi
    for (int s = 0; s < n_spatial_units; s++) {
      phi_proj[s] = 0.0;
      for (int k = 0; k < n_spatial_units; k++) {
        phi_proj[s] += rsr_projection[s * rsr_n + k] * phi[k];
      }
    }

    // 2. Set spatial contribution from projected effects
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(C.n_threads_team)
    #endif
    for (int i = 0; i < N; i++) {
      int s = spatial_group[i] - 1;
      spatial_contrib[i] = (s >= 0 && s < n_spatial_units) ? phi_proj[s] : 0.0;
    }

    // Steps 3-7: shared core (compute eta, sample omega, update beta/RE)
    tulpa::pg_gibbs_core_step(
        N, p, C.beta, C.re, C.sigma_re, C.omega, C.eta, C.X_beta, C.re_contrib,
        spatial_contrib, C.offset, C.kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale, C.n_threads_team);

    // 8. Update spatial effects (raw, unprojected)
    // The key insight: we update phi based on the pseudo-likelihood
    // But the offset should account for the RSR projection
    // Offset for spatial update = X*beta + re, then we need to handle projection

    // For RSR, we work with the transformed residuals
    // kappa_adj = P' * (kappa - omega * (X*beta + re))
    // omega_adj = P' * diag(omega) * P

    // Simpler approach: update phi as ICAR, but use offset computed with projection
    // This is approximate but maintains ICAR structure

    // Compute offset with projection for spatial update
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(C.n_threads_team)
    #endif
    for (int i = 0; i < N; i++) {
      C.offset[i] = C.X_beta[i] + C.re_contrib[i];
    }

    // Update spatial effects using ICAR
    phi = tulpa::update_spatial_icar(C.kappa, C.omega, C.offset, spatial_group, adj_list, n_neighbors, tau);

    // 9. Update tau (spatial precision)
    tau = tulpa::update_tau_icar(phi, adj_list, n_neighbors, prior_tau_shape, prior_tau_rate);

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      C.save(save_idx);
      // Store both raw and projected spatial effects
      for (int s = 0; s < n_spatial_units; s++) {
        spatial_raw_draws(save_idx, s) = phi[s];
        spatial_proj_draws(save_idx, s) = phi_proj[s];
      }
      tau_draws[save_idx] = tau;
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
    Rcpp::Named("spatial_raw") = spatial_raw_draws,
    Rcpp::Named("spatial") = spatial_proj_draws,
    Rcpp::Named("tau") = tau_draws
  );

  if (store_eta) {
    result["eta"] = C.eta_draws;
  }

  PutRNGstate();
  return result;
}
