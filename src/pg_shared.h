// pg_shared.h
// Shared helpers for Pólya-Gamma Gibbs samplers
// Extracted from pg_binomial.cpp, pg_negbin.cpp, pg_spatial.cpp

#ifndef TULPA_PG_SHARED_H
#define TULPA_PG_SHARED_H

#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

#include "linalg_fast.h"  // shared small-dense Cholesky / NNGP solve core

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// ============================================================================
// Sequential NNGP conditional N(w_i | cond_mean, cond_var)
//
// Conditional of the i-th location (in NNGP ordering) given its already-updated
// neighbours, for an isotropic GP with marginal variance `sigma2`, range `phi`
// and covariance `cov_type` (0 = exponential, 1 = Matern 3/2, 2 = Matern 5/2).
// `cond_mean = c' C^{-1} w_nb` is the kriging predictor (a contraction -- its
// weights are bounded, unlike a raw covariance-weighted sum), `cond_var =
// sigma2 - c' C^{-1} c >= 0`.
//
// Index conventions (shared by the single-scale GP sampler and every scale of
// the multiscale sampler): `i` is the 0-based ordered position; `nn_idx(i, j)`
// is a 1-based neighbour position in ordered space; `nn_order` maps an ordered
// position to its 0-based original location id, so `coords` / `w` are addressed
// in original-location order. Because cond_mean is scale-invariant, calling with
// `sigma2 = 1` yields (m_i, v0_i) for the phi-only quadratic form used by the
// sigma2 conditional.
// ============================================================================
inline void pg_nngp_conditional(
    int i,
    const std::vector<double>& w,
    double sigma2,
    double phi_gp,
    int cov_type,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::NumericMatrix& nn_dist,
    const Rcpp::IntegerVector& nn_order,
    int nn,
    double& cond_mean,
    double& cond_var
) {
  int n_neighbors = 0;
  for (int j = 0; j < nn; j++) {
    if (nn_idx(i, j) > 0) n_neighbors++;
  }

  if (n_neighbors == 0) {
    cond_mean = 0.0;
    cond_var = sigma2;
    return;
  }

  // Covariance function lambda
  auto compute_cov = [sigma2, phi_gp, cov_type](double d) {
    if (d < 1e-10) return sigma2;
    if (cov_type == 0) {  // Exponential
      return sigma2 * std::exp(-d / phi_gp);
    } else if (cov_type == 1) {  // Matern 1.5
      double x = std::sqrt(3.0) * d / phi_gp;
      return sigma2 * (1.0 + x) * std::exp(-x);
    } else {  // Matern 2.5
      double x = std::sqrt(5.0) * d / phi_gp;
      return sigma2 * (1.0 + x + x * x / 3.0) * std::exp(-x);
    }
  };

  std::vector<double> c_vec(n_neighbors);
  std::vector<double> C_mat(n_neighbors * n_neighbors);

  for (int j = 0; j < n_neighbors; j++) {
    c_vec[j] = compute_cov(nn_dist(i, j));
  }

  for (int j1 = 0; j1 < n_neighbors; j1++) {
    int nn_orig1 = nn_order[nn_idx(i, j1) - 1];
    for (int j2 = 0; j2 < n_neighbors; j2++) {
      int nn_orig2 = nn_order[nn_idx(i, j2) - 1];
      if (j1 == j2) {
        C_mat[j1 * n_neighbors + j2] = sigma2;
      } else {
        double d12 = std::sqrt(
          std::pow(coords(nn_orig1, 0) - coords(nn_orig2, 0), 2) +
          std::pow(coords(nn_orig1, 1) - coords(nn_orig2, 1), 2)
        );
        C_mat[j1 * n_neighbors + j2] = compute_cov(d12);
      }
    }
  }

  // Gather neighbor values in c_vec order, then shared factor/solve core
  std::vector<double> w_nb(n_neighbors);
  for (int j = 0; j < n_neighbors; j++) {
    int nn_orig = nn_order[nn_idx(i, j) - 1];
    w_nb[j] = w[nn_orig];
  }
  tulpa_linalg::nngp_conditional_moments(
      C_mat.data(), c_vec.data(), w_nb.data(), n_neighbors, sigma2,
      tulpa_linalg::kCholJitter, tulpa_linalg::kCholJitter,
      cond_mean, cond_var);
}

// Cholesky decomposition (lower triangular) for small dense matrices
// Used in PG Gibbs samplers for posterior sampling of beta
inline Rcpp::NumericMatrix chol_decomp_pg(const Rcpp::NumericMatrix& A) {
  int n = A.nrow();
  // Row-major staging: NumericMatrix is column-major, the shared kernel
  // indexes A[i * n + j]; the matrix is symmetric so the copy is layout-safe.
  std::vector<double> A_flat(static_cast<size_t>(n) * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A_flat[static_cast<size_t>(i) * n + j] = A(i, j);
    }
  }
  std::vector<double> L_flat(static_cast<size_t>(n) * n, 0.0);
  tulpa_linalg::chol_factor_lower(A_flat.data(), L_flat.data(), n, n,
                                  tulpa_linalg::kCholJitter);
  Rcpp::NumericMatrix L(n, n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      L(i, j) = L_flat[static_cast<size_t>(i) * n + j];
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
    double prior_sigma_re_scale,
    int n_threads = 1
) {
    const int team = tulpa_omp_team_size_req(n_threads, N);
    // 1. Compute linear predictor
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(team)
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
    #pragma omp parallel for schedule(static) num_threads(team)
    #endif
    for (int i = 0; i < N; i++) {
        offset[i] = re_contrib[i] + spatial_contrib[i];
    }
    beta = update_beta(kappa, omega, X, offset, prior_beta_sd);

    // 4. Recompute X_beta
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(team)
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
        #pragma omp parallel for schedule(static) num_threads(team)
        #endif
        for (int i = 0; i < N; i++) {
            offset[i] = X_beta[i] + spatial_contrib[i];
        }
        re = update_re(kappa, omega, offset, re_group, n_re_groups, sigma_re);
        sigma_re = update_sigma_halfcauchy(re, prior_sigma_re_scale);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(team)
        #endif
        for (int i = 0; i < N; i++) {
            re_contrib[i] = re[re_group[i] - 1];
        }
    }
}

// ============================================================================
// Common scaffolding for PG binomial Gibbs samplers
//
// Bundles state shared by every variant (no-spatial, ICAR, BYM2, RSR, GP,
// multiscale GP, temporal): working vectors, draw matrices for beta/re/sigma_re
// (+ optional eta), and the per-iteration save of those common fields. Per-
// variant storage (spatial, GP hypers, temporal components, etc.) stays in the
// caller.
// ============================================================================
struct PgGibbsCommon {
  int N, p, n_re_groups, n_save;
  bool store_eta;
  // Clamped team size for the per-obs regions: the caller's n_threads bounded
  // by OMP_THREAD_LIMIT / max threads / N. Passed as a num_threads(...)
  // clause instead of mutating the process-global OpenMP default.
  int n_threads_team = 1;

  // Current chain state
  Rcpp::NumericVector beta, re;
  double sigma_re;

  // Per-observation working vectors (passed to pg_gibbs_core_step)
  Rcpp::NumericVector omega, kappa, eta, X_beta, re_contrib, offset;

  // Draws for the always-present fields
  Rcpp::NumericMatrix beta_draws, re_draws;
  Rcpp::NumericVector sigma_re_draws;
  Rcpp::NumericMatrix eta_draws;

  PgGibbsCommon(const Rcpp::IntegerVector& y,
                const Rcpp::IntegerVector& n_trials,
                int p_,
                int n_re_groups_,
                int n_save_,
                int n_threads,
                bool store_eta_)
    : N(y.size()), p(p_), n_re_groups(n_re_groups_), n_save(n_save_),
      store_eta(store_eta_),
      beta(p_, 0.0),
      re(n_re_groups_, 0.0),
      sigma_re(1.0),
      omega(N, 0.5),
      kappa(N),
      eta(N),
      X_beta(N),
      re_contrib(N),
      offset(N),
      beta_draws(n_save_, p_),
      re_draws(n_save_, n_re_groups_),
      sigma_re_draws(n_save_),
      eta_draws(store_eta_ ? n_save_ : 0, store_eta_ ? N : 0)
  {
    n_threads_team = tulpa_omp_team_size_req(n_threads, N);
    for (int i = 0; i < N; i++) {
      kappa[i] = static_cast<double>(y[i]) - 0.5 * static_cast<double>(n_trials[i]);
    }
  }

  // Save common per-iteration draws. Caller is responsible for invoking this
  // (and per-variant save logic) only when (iter >= n_warmup && (iter - n_warmup) % thin == 0).
  void save(int save_idx) {
    for (int j = 0; j < p; j++) beta_draws(save_idx, j) = beta[j];
    for (int g = 0; g < n_re_groups; g++) re_draws(save_idx, g) = re[g];
    sigma_re_draws[save_idx] = sigma_re;
    if (store_eta) {
      for (int i = 0; i < N; i++) eta_draws(save_idx, i) = eta[i];
    }
  }
};

} // namespace tulpa

#endif // TULPA_PG_SHARED_H
