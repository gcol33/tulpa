// pg_shared.h
// Shared helpers for Pólya-Gamma Gibbs samplers
// Extracted from pg_binomial.cpp, pg_negbin.cpp, pg_spatial.cpp

#ifndef TULPA_PG_SHARED_H
#define TULPA_PG_SHARED_H

#include <Rcpp.h>
#include <cmath>
#include <algorithm>

namespace tulpa {

// Cholesky decomposition (lower triangular) for small dense matrices
// Used in PG Gibbs samplers for posterior sampling of beta
inline Rcpp::NumericMatrix chol_decomp_pg(const Rcpp::NumericMatrix& A) {
  int n = A.nrow();
  Rcpp::NumericMatrix L(n, n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0.0;
      for (int k = 0; k < j; k++) {
        sum += L(i, k) * L(j, k);
      }
      if (i == j) {
        L(i, j) = std::sqrt(std::max(A(i, i) - sum, 1e-10));
      } else {
        L(i, j) = (A(i, j) - sum) / L(j, j);
      }
    }
  }
  return L;
}

// Update sigma with half-Cauchy prior (inverse-gamma approximation)
// Shared by update_sigma_re, update_sigma_re_negbin, update_sigma_spatial
//
// Half-Cauchy prior: sigma^2 ~ IG(0.5, 0.5 * scale^2)
// Posterior: sigma^2 | effects ~ IG((J+1)/2, ss/2 + 0.5*scale^2)
inline double update_sigma_halfcauchy(
    const Rcpp::NumericVector& effects,
    double scale
) {
  int J = effects.size();

  double ss = 0.0;
  for (int j = 0; j < J; j++) {
    ss += effects[j] * effects[j];
  }

  double shape = (J + 1.0) / 2.0;
  double rate = ss / 2.0 + 0.5 * scale * scale;

  double sigma_sq = 1.0 / R::rgamma(shape, 1.0 / rate);
  return std::sqrt(sigma_sq);
}

// ============================================================================
// Shared per-iteration core for PG Gibbs spatial samplers
// Steps 1-5 are identical across ICAR, BYM2, and RSR variants:
//   1. Compute linear predictor (eta = X*beta + re + spatial)
//   2. Sample omega ~ PG(n, eta)
//   3. Update beta
//   4. Recompute X_beta
//   5. Update RE + sigma_re
//
// The caller must set spatial_contrib BEFORE calling this function.
// After return, X_beta, re_contrib, and offset are updated.
// ============================================================================

// Forward declarations (defined in pg_binomial.cpp, pg_rng.h)
Rcpp::NumericVector update_beta(
    const Rcpp::NumericVector& kappa,
    const Rcpp::NumericVector& omega,
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& offset,
    double prior_sd);

Rcpp::NumericVector update_re(
    const Rcpp::NumericVector& kappa,
    const Rcpp::NumericVector& omega,
    const Rcpp::NumericVector& X_beta,
    const Rcpp::IntegerVector& group,
    int n_groups,
    double sigma_re);

double rpg_int(int n, double z);

inline void pg_gibbs_core_step(
    int N, int p,
    Rcpp::NumericVector& beta,
    Rcpp::NumericVector& re,
    double& sigma_re,
    Rcpp::NumericVector& omega,
    Rcpp::NumericVector& eta,
    Rcpp::NumericVector& X_beta,
    Rcpp::NumericVector& re_contrib,
    const Rcpp::NumericVector& spatial_contrib,
    Rcpp::NumericVector& offset,
    const Rcpp::NumericVector& kappa,
    const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X,
    const Rcpp::IntegerVector& re_group,
    int n_re_groups,
    double prior_beta_sd,
    double prior_sigma_re_scale
) {
    // 1. Compute linear predictor
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
        X_beta[i] = 0.0;
        for (int j = 0; j < p; j++) {
            X_beta[i] += X(i, j) * beta[j];
        }
        re_contrib[i] = (n_re_groups > 0) ? re[re_group[i] - 1] : 0.0;
        eta[i] = X_beta[i] + re_contrib[i] + spatial_contrib[i];
    }

    // 2. Sample omega ~ PG(n, eta) — NOT parallelized (R's RNG not thread-safe)
    for (int i = 0; i < N; i++) {
        omega[i] = rpg_int(n_trials[i], eta[i]);
    }

    // 3. Update beta
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
        offset[i] = re_contrib[i] + spatial_contrib[i];
    }
    beta = update_beta(kappa, omega, X, offset, prior_beta_sd);

    // 4. Recompute X_beta
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
        X_beta[i] = 0.0;
        for (int j = 0; j < p; j++) {
            X_beta[i] += X(i, j) * beta[j];
        }
    }

    // 5. Update random effects
    if (n_re_groups > 0) {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < N; i++) {
            offset[i] = X_beta[i] + spatial_contrib[i];
        }
        re = update_re(kappa, omega, offset, re_group, n_re_groups, sigma_re);
        sigma_re = update_sigma_halfcauchy(re, prior_sigma_re_scale);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < N; i++) {
            re_contrib[i] = re[re_group[i] - 1];
        }
    }
}

} // namespace tulpa

#endif // TULPA_PG_SHARED_H
