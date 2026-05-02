// pg_binomial.cpp
// Pólya-Gamma Gibbs sampler for binomial models with random effects
// Based on Polson, Scott & Windle (2013) JASA

#include "pg_binomial.h"
#include "pg_shared.h"
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
// Helper: solve linear system with diagonal covariance
// For posterior: (X'WX + D^{-1})^{-1} X'W kappa
// where W = diag(omega), D = prior precision
// ---------------------------------------------------------------------

// Compute X'WX + prior_prec * I and X'W * (kappa - offset)
// Returns list with posterior mean and precision matrix
List compute_posterior_normal(
    const NumericMatrix& X,
    const NumericVector& omega,
    const NumericVector& kappa,
    const NumericVector& offset,
    double prior_prec
) {
  int n = X.nrow();
  int p = X.ncol();

  // X'WX + prior_prec * I
  NumericMatrix XWX(p, p);
  NumericVector XWkappa(p);

  for (int j = 0; j < p; j++) {
    for (int k = j; k < p; k++) {
      double sum = 0.0;
      for (int i = 0; i < n; i++) {
        sum += X(i, j) * omega[i] * X(i, k);
      }
      XWX(j, k) = sum;
      if (j != k) XWX(k, j) = sum;
    }
    // Add prior precision to diagonal
    XWX(j, j) += prior_prec;

    // X'W(kappa - offset)
    double sum_kappa = 0.0;
    for (int i = 0; i < n; i++) {
      sum_kappa += X(i, j) * omega[i] * (kappa[i] / omega[i] - offset[i]);
    }
    XWkappa[j] = sum_kappa;
  }

  return List::create(
    Named("XWX") = XWX,
    Named("XWkappa") = XWkappa
  );
}

// Cholesky solve: solve (L L') x = b
NumericVector chol_solve(const NumericMatrix& L, const NumericVector& b) {
  int p = L.ncol();
  NumericVector y(p), x(p);

  // Forward substitution: L y = b
  for (int i = 0; i < p; i++) {
    double sum = b[i];
    for (int j = 0; j < i; j++) {
      sum -= L(i, j) * y[j];
    }
    y[i] = sum / L(i, i);
  }

  // Back substitution: L' x = y
  for (int i = p - 1; i >= 0; i--) {
    double sum = y[i];
    for (int j = i + 1; j < p; j++) {
      sum -= L(j, i) * x[j];
    }
    x[i] = sum / L(i, i);
  }

  return x;
}

// Cholesky decomposition — delegates to shared helper
NumericMatrix chol_decomp(const NumericMatrix& A) {
  return tulpa::chol_decomp_pg(A);
}

// Sample from multivariate normal using Cholesky
NumericVector rmvnorm_chol(const NumericVector& mean, const NumericMatrix& L) {
  int p = mean.size();
  NumericVector z(p), x(p);

  // Sample standard normal
  for (int i = 0; i < p; i++) {
    z[i] = R::rnorm(0.0, 1.0);
  }

  // x = mean + L * z
  for (int i = 0; i < p; i++) {
    x[i] = mean[i];
    for (int j = 0; j <= i; j++) {
      x[i] += L(i, j) * z[j];
    }
  }

  return x;
}

// ---------------------------------------------------------------------
// Update functions
// ---------------------------------------------------------------------

// Update beta (fixed effects)
NumericVector update_beta(
    const NumericVector& kappa,
    const NumericVector& omega,
    const NumericMatrix& X,
    const NumericVector& re_contrib,
    double prior_sd
) {
  int n = X.nrow();
  int p = X.ncol();
  double prior_prec = 1.0 / (prior_sd * prior_sd);

  // Compute posterior parameters
  List post = compute_posterior_normal(X, omega, kappa, re_contrib, prior_prec);
  NumericMatrix XWX = post["XWX"];
  NumericVector XWkappa = post["XWkappa"];

  // Cholesky decomposition
  NumericMatrix L = chol_decomp(XWX);

  // Posterior mean: solve XWX * mean = XWkappa
  NumericVector post_mean = chol_solve(L, XWkappa);

  // Sample from posterior
  // Need to sample from N(post_mean, XWX^{-1})
  // XWX^{-1} = (L L')^{-1} = L'^{-1} L^{-1}
  // So sample z ~ N(0, I), compute L^{-1} z, add to mean

  // More efficient: solve L z_star = z for z_star, then x = mean + z_star
  NumericVector z(p);
  for (int i = 0; i < p; i++) {
    z[i] = R::rnorm(0.0, 1.0);
  }

  // Solve L z_star = z (forward substitution only)
  NumericVector z_star(p);
  for (int i = 0; i < p; i++) {
    double sum = z[i];
    for (int j = 0; j < i; j++) {
      sum -= L(i, j) * z_star[j];
    }
    z_star[i] = sum / L(i, i);
  }

  // x = mean + z_star
  NumericVector beta(p);
  for (int i = 0; i < p; i++) {
    beta[i] = post_mean[i] + z_star[i];
  }

  return beta;
}

// Update random effects (blocked by group)
NumericVector update_re(
    const NumericVector& kappa,
    const NumericVector& omega,
    const NumericVector& X_beta,
    const IntegerVector& group,
    int n_groups,
    double sigma_re
) {
  int n = kappa.size();
  NumericVector re(n_groups);

  // For each group, compute posterior
  // Observations in group g: Y_g = X_beta_g + b_g + error
  // With PG augmentation: kappa_g/omega_g = X_beta_g + b_g + N(0, 1/omega_g)
  // Prior: b_g ~ N(0, sigma_re^2)
  //
  // Posterior: b_g | ... ~ N(m_g, v_g)
  // v_g = (sum(omega_g) + 1/sigma_re^2)^{-1}
  // m_g = v_g * sum(kappa_g - omega_g * X_beta_g)

  double prior_prec = 1.0 / (sigma_re * sigma_re + 1e-10);

  // Accumulate sufficient statistics by group
  NumericVector sum_omega(n_groups);
  NumericVector sum_resid(n_groups);

  for (int i = 0; i < n; i++) {
    int g = group[i] - 1;  // Convert to 0-based
    sum_omega[g] += omega[i];
    sum_resid[g] += kappa[i] - omega[i] * X_beta[i];
  }

  // Sample from posterior
  for (int g = 0; g < n_groups; g++) {
    double post_var = 1.0 / (sum_omega[g] + prior_prec);
    double post_mean = post_var * sum_resid[g];
    re[g] = R::rnorm(post_mean, std::sqrt(post_var));
  }

  return re;
}

// Update sigma_re with half-Cauchy prior — delegates to shared helper
double update_sigma_re(
    const NumericVector& re,
    double scale
) {
  return tulpa::update_sigma_halfcauchy(re, scale);
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
  NumericMatrix beta_draws(n_save, p);
  NumericMatrix re_draws(n_save, n_groups);
  NumericVector sigma_draws(n_save);
  NumericMatrix eta_draws;
  if (store_eta) {
    eta_draws = NumericMatrix(n_save, N);
  }

  // Initialize
  NumericVector beta(p, 0.0);
  NumericVector re(n_groups, 0.0);
  double sigma_re = 1.0;
  NumericVector omega(N, 1.0);  // PG draws
  NumericVector kappa(N);       // y - n/2
  NumericVector eta(N);         // Linear predictor
  NumericVector X_beta(N);      // X * beta
  NumericVector re_contrib(N);  // Random effects contribution

  // Compute kappa = y - n/2
  for (int i = 0; i < N; i++) {
    kappa[i] = y[i] - n[i] / 2.0;
  }

  // Gibbs iterations
  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {
    // 1. Compute linear predictor (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
      // Only access re if we have random effects
      if (n_groups > 0) {
        re_contrib[i] = re[group[i] - 1];  // group is 1-based
      } else {
        re_contrib[i] = 0.0;
      }
      eta[i] = X_beta[i] + re_contrib[i];
    }

    // 2. Sample omega ~ PG(n, eta)
    // Note: NOT parallelized - R's RNG is not thread-safe
    for (int i = 0; i < N; i++) {
      omega[i] = rpg_int(n[i], eta[i]);
    }

    // 3. Update beta | omega, re, y
    beta = update_beta(kappa, omega, X, re_contrib, prior_beta_sd);

    // 4. Recompute X_beta after beta update (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    }

    // 5. Update random effects | omega, beta, sigma_re
    if (n_groups > 0) {
      re = update_re(kappa, omega, X_beta, group, n_groups, sigma_re);

      // 6. Update sigma_re | re
      sigma_re = update_sigma_re(re, prior_sigma_scale);
    }

    // Save draws (after warmup, respecting thinning)
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_draws[save_idx] = sigma_re;

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

  List result = List::create(
    Named("beta") = beta_draws,
    Named("re") = re_draws,
    Named("sigma_re") = sigma_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws;
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
  // CRITICAL: Must call GetRNGstate/PutRNGstate when using R's RNG from C++
  GetRNGstate();
  Rcpp::List result = tulpa::pg_binomial_gibbs_impl(
    y, n, X, group, n_groups,
    n_iter, n_warmup, thin,
    prior_beta_sd, prior_sigma_scale,
    store_eta, verbose, n_threads
  );
  PutRNGstate();
  return result;
}

// cpp_pg_get_max_threads removed — use cpp_get_max_threads (hmc_sampler.cpp)

// Forward declaration
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
  Rcpp::NumericMatrix spatial_draws(n_save, n_spatial_units);
  Rcpp::NumericVector tau_draws(n_save);
  Rcpp::NumericMatrix eta_draws;
  if (store_eta) {
    eta_draws = Rcpp::NumericMatrix(n_save, N);
  }

  // Initialize
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  Rcpp::NumericVector phi(n_spatial_units, 0.0);  // Spatial effects
  double tau = 1.0;  // Spatial precision
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
      spatial_contrib[i] = phi[spatial_group[i] - 1];
    }
    tulpa::pg_gibbs_core_step(
        N, p, beta, re, sigma_re, omega, eta, X_beta, re_contrib,
        spatial_contrib, offset, kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale);

    // 6. Update spatial effects | omega, beta, re, tau
    // Offset for spatial update = X*beta + re (parallelized)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i];
    }
    phi = tulpa::update_spatial_icar(kappa, omega, offset, spatial_group, adj_list, n_neighbors, tau);

    // 7. Update tau (spatial precision)
    tau = tulpa::update_tau_icar(phi, adj_list, n_neighbors, prior_tau_shape, prior_tau_rate);

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
        spatial_draws(save_idx, s) = phi[s];
      }
      tau_draws[save_idx] = tau;

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
    Rcpp::Named("spatial") = spatial_draws,
    Rcpp::Named("tau") = tau_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws;
  }

  PutRNGstate();
  return result;
}
