// pg_binomial.cpp
// Pólya-Gamma Gibbs sampler for binomial models with random effects
// Based on Polson, Scott & Windle (2013) JASA

#include "pg_binomial.h"
#include "pg_shared.h"
#include "pg_spatial.h"
#include "pg_rng.h"
#include "linalg_fast.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

// OpenMP parallelization notes:
// - SAFE to parallelize: matrix-vector products (X*beta), linear predictor computation
// - NOT SAFE: loops calling R's RNG (R::rnorm, rpg_int) or modifying Rcpp objects
// The #pragma omp directives below are applied ONLY to safe arithmetic operations
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace tulpa {

// ---------------------------------------------------------------------
// Update functions
// ---------------------------------------------------------------------

// Update beta (fixed effects). After PG augmentation the conditional is
// N((X'WX + D^-1)^-1 X'(kappa - W offset), (X'WX + D^-1)^-1) with
// W = diag(omega) and D = prior_sd^2 I.
NumericVector update_beta(
    const NumericVector& kappa,
    const NumericVector& omega,
    const NumericMatrix& X,
    const NumericVector& re_contrib,
    double prior_sd
) {
  const int n = X.nrow();
  const int p = X.ncol();
  const double prior_prec = 1.0 / (prior_sd * prior_sd);

  std::vector<double> XWX(static_cast<size_t>(p) * p, 0.0);
  std::vector<double> XWkappa(p, 0.0);

  for (int j = 0; j < p; j++) {
    for (int k = j; k < p; k++) {
      double sum = 0.0;
      for (int i = 0; i < n; i++) {
        sum += X(i, j) * omega[i] * X(i, k);
      }
      XWX[static_cast<size_t>(j) * p + k] = sum;
      if (j != k) XWX[static_cast<size_t>(k) * p + j] = sum;
    }
    XWX[static_cast<size_t>(j) * p + j] += prior_prec;

    // X'(kappa - omega*offset): algebraically omega*(kappa/omega - offset) but
    // finite when omega[i] = 0 (a zero-trial row, where kappa[i] = 0 too and
    // kappa/omega would be 0/0 = NaN).
    double sum_kappa = 0.0;
    for (int i = 0; i < n; i++) {
      sum_kappa += X(i, j) * (kappa[i] - omega[i] * re_contrib[i]);
    }
    XWkappa[j] = sum_kappa;
  }

  NumericVector beta(p);
  pg_draw_gaussian_precision(XWX.data(), p, XWkappa.data(), beta.begin(),
                             "fixed-effect");
  return beta;
}

// Update random effects (blocked by group)
//
// Observations in group g: kappa_g/omega_g = X_beta_g + b_g + N(0, 1/omega_g)
// with prior b_g ~ N(0, sigma_re^2), so
//   b_g | ... ~ N(v_g * sum_resid_g, v_g),
//   v_g = (sum_omega_g + 1/sigma_re^2)^{-1}.
NumericVector update_re(
    const NumericVector& kappa,
    const NumericVector& omega,
    const NumericVector& X_beta,
    const IntegerVector& group,
    int n_groups,
    double sigma_re
) {
  const int n = kappa.size();
  NumericVector re(n_groups);

  const double prior_prec = 1.0 / (sigma_re * sigma_re + 1e-10);

  std::vector<double> sum_omega(n_groups, 0.0), sum_resid(n_groups, 0.0);
  pg_accumulate_stats(n, group.begin(), n_groups, omega.begin(), kappa.begin(),
                      X_beta.begin(), sum_omega.data(), sum_resid.data());

  for (int g = 0; g < n_groups; g++) {
    const double post_var = 1.0 / (sum_omega[g] + prior_prec);
    const double post_mean = post_var * sum_resid[g];
    re[g] = R::rnorm(post_mean, std::sqrt(post_var));
  }

  return re;
}

// ---------------------------------------------------------------------
// Main Gibbs sampler
// ---------------------------------------------------------------------

List pg_binomial_gibbs_impl(
    IntegerVector y,
    IntegerVector n,
    NumericMatrix X,
    IntegerVector group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  const int n_save = pg_n_save(n_iter, n_warmup, thin);
  PgGibbsCommon C(y, n, X, group, n_groups, n_save, prior_sigma_scale,
                  n_threads, store_eta);
  const int N = C.N;
  const int p = C.p;

  // No spatial / temporal prior: zero contribution into pg_gibbs_core_step.
  NumericVector zero_contrib(N, 0.0);

  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {
    pg_gibbs_core_step(
        N, p, C.beta, C.re, C.sigma_re, C.omega, C.eta, C.X_beta, C.re_contrib,
        zero_contrib, C.offset, C.kappa, n, X, group, n_groups,
        prior_beta_sd, prior_sigma_scale, C.n_threads_team);

    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      C.save(save_idx);
      save_idx++;
    }

    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter << std::endl;
    }
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  List result = List::create(
    Named("beta") = C.beta_draws,
    Named("re") = C.re_draws,
    Named("sigma_re") = C.sigma_re_draws
  );
  if (store_eta) {
    result["eta"] = C.eta_draws;
  }
  return result;
}

} // namespace tulpa

// ---------------------------------------------------------------------
// R exports
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector group,
    int n_groups,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_scale = 2.5,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  return tulpa::pg_binomial_gibbs_impl(
    y, n, X, group, n_groups,
    n_iter, n_warmup, thin,
    prior_beta_sd, prior_sigma_scale,
    store_eta, verbose, n_threads
  );
}

// Binomial Gibbs sampler with random effects AND spatial effects (ICAR)
// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_spatial(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector spatial_group,
    int n_spatial_units,
    Rcpp::List adj_list,
    Rcpp::IntegerVector n_neighbors,
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
  const int n_save = tulpa::pg_n_save(n_iter, n_warmup, thin);
  tulpa::PgGibbsCommon C(y, n, X, re_group, n_re_groups, n_save,
                         prior_sigma_re_scale, n_threads, store_eta);
  const int N = C.N;
  const int p = C.p;
  C.require_intercept("ICAR spatial");

  tulpa::pg_check_index(spatial_group, N, n_spatial_units, "spatial_group");
  const tulpa::PgAdjacency adj =
      tulpa::pg_build_adjacency(adj_list, n_neighbors, n_spatial_units);
  tulpa::pg_check_components_observed(adj, spatial_group);

  // Per-variant storage
  Rcpp::NumericMatrix spatial_draws(n_save, n_spatial_units);
  Rcpp::NumericVector tau_draws(n_save);

  // Per-variant state
  Rcpp::NumericVector phi(n_spatial_units, 0.0);
  double tau = 1.0;
  Rcpp::NumericVector spatial_contrib(N);

  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {
    // Steps 1-5: shared core (compute eta, sample omega, update beta/RE)
    tulpa_parallel_for(C.n_threads_team, N, [&](int i) {
      spatial_contrib[i] = phi[spatial_group[i] - 1];
    });
    tulpa::pg_gibbs_core_step(
        N, p, C.beta, C.re, C.sigma_re, C.omega, C.eta, C.X_beta, C.re_contrib,
        spatial_contrib, C.offset, C.kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale, C.n_threads_team);

    // 6. Update spatial effects | omega, beta, re, tau
    // Offset for spatial update = X*beta + re
    tulpa_parallel_for(C.n_threads_team, N, [&](int i) {
      C.offset[i] = C.X_beta[i] + C.re_contrib[i];
    });
    double icar_mean = 0.0;
    tulpa::update_spatial_icar(C.kappa, C.omega, C.offset, spatial_group,
                               adj, tau, phi, icar_mean);
    C.absorb_level(icar_mean);

    // 7. Update tau (spatial precision)
    tau = tulpa::update_tau_icar(phi, adj, prior_tau_shape, prior_tau_rate);

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      C.save(save_idx);
      for (int s = 0; s < n_spatial_units; s++) {
        spatial_draws(save_idx, s) = phi[s];
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
    Rcpp::Named("spatial") = spatial_draws,
    Rcpp::Named("tau") = tau_draws
  );

  if (store_eta) {
    result["eta"] = C.eta_draws;
  }

  return result;
}
