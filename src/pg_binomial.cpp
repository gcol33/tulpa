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

// ---------------------------------------------------------------------
// GP Spatial Gibbs Sampler for Binomial Models
// Uses sequential NNGP updates
// ---------------------------------------------------------------------

// Helper: compute NNGP conditional mean and variance
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

  // Cholesky of C
  std::vector<double> L(n_neighbors * n_neighbors, 0.0);
  for (int j = 0; j < n_neighbors; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = C_mat[j * n_neighbors + k];
      for (int m = 0; m < k; m++) {
        sum -= L[j * n_neighbors + m] * L[k * n_neighbors + m];
      }
      if (j == k) {
        L[j * n_neighbors + j] = std::sqrt(std::max(1e-10, sum));
      } else {
        L[j * n_neighbors + k] = sum / L[k * n_neighbors + k];
      }
    }
  }

  // Solve L * y_sol = c_vec
  std::vector<double> y_sol(n_neighbors);
  for (int j = 0; j < n_neighbors; j++) {
    double sum = c_vec[j];
    for (int k = 0; k < j; k++) {
      sum -= L[j * n_neighbors + k] * y_sol[k];
    }
    y_sol[j] = sum / L[j * n_neighbors + j];
  }

  // Solve L^T * alpha = y_sol
  std::vector<double> alpha(n_neighbors);
  for (int j = n_neighbors - 1; j >= 0; j--) {
    double sum = y_sol[j];
    for (int k = j + 1; k < n_neighbors; k++) {
      sum -= L[k * n_neighbors + j] * alpha[k];
    }
    alpha[j] = sum / L[j * n_neighbors + j];
  }

  cond_mean = 0.0;
  for (int j = 0; j < n_neighbors; j++) {
    int nn_orig = nn_order[nn_idx(i, j) - 1];
    cond_mean += alpha[j] * w[nn_orig];
  }

  double c_Cinv_c = 0.0;
  for (int j = 0; j < n_neighbors; j++) {
    c_Cinv_c += c_vec[j] * alpha[j];
  }
  cond_var = std::max(1e-10, sigma2 - c_Cinv_c);
}

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

  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  #ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
  #endif

  // Storage
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix gp_draws(n_save, n_spatial);
  Rcpp::NumericVector sigma2_gp_draws(n_save);
  Rcpp::NumericVector phi_gp_draws(n_save);
  Rcpp::NumericMatrix eta_draws_gp;
  if (store_eta) eta_draws_gp = Rcpp::NumericMatrix(n_save, N);

  // Initialize
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  std::vector<double> w(n_spatial, 0.0);
  double sigma2_gp = sigma2_gp_init;
  double phi_gp = phi_gp_init;

  // Working vectors
  Rcpp::NumericVector omega(N);
  Rcpp::NumericVector kappa(N);
  Rcpp::NumericVector eta(N);
  Rcpp::NumericVector X_beta(N);
  Rcpp::NumericVector re_contrib(N);
  Rcpp::NumericVector gp_contrib(N);
  Rcpp::NumericVector offset(N);

  for (int i = 0; i < N; i++) {
    omega[i] = 0.5;
    kappa[i] = y[i] - 0.5 * n[i];
    X_beta[i] = 0.0;
    re_contrib[i] = 0.0;
    gp_contrib[i] = 0.0;
  }

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    if (verbose && (iter + 1) % 200 == 0) {
      Rcpp::Rcout << "  Iteration " << (iter + 1) << "/" << n_iter << "\n";
    }

    // 1-4. Core Gibbs step (eta, omega, beta, RE) — shared with all variants
    tulpa::pg_gibbs_core_step(
        N, p, beta, re, sigma_re, omega, eta, X_beta, re_contrib,
        gp_contrib, offset, kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale);

    // 5. Update GP effects (sequential NNGP Gibbs)
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i];
    }

    // Aggregate likelihood info per spatial location
    std::vector<double> sum_omega_gp(n_spatial, 0.0);
    std::vector<double> sum_resid_gp(n_spatial, 0.0);
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        sum_omega_gp[i] += omega[i];
        sum_resid_gp[i] += kappa[i] - omega[i] * offset[i];
      }
    }

    // Update each GP effect in NNGP order
    for (int idx = 0; idx < n_spatial; idx++) {
      int obs_i = nn_order[idx];

      double cond_mean, cond_var;
      pg_nngp_conditional(idx, w, sigma2_gp, phi_gp, cov_type,
                          coords, nn_idx, nn_dist, nn_order, nn,
                          cond_mean, cond_var);

      double tau_prior = 1.0 / cond_var;
      double tau_lik = sum_omega_gp[obs_i];
      double tau_post = tau_prior + tau_lik;
      double mean_post = (tau_prior * cond_mean + sum_resid_gp[obs_i]) / tau_post;

      w[obs_i] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
    }

    // Update GP contributions
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        gp_contrib[i] = w[i];
      }
    }

    // 6. Update GP hyperparameters via MH
    double sigma2_prop = tulpa_linalg::safe_exp(std::log(sigma2_gp) + R::rnorm(0, 0.1));
    if (!std::isfinite(sigma2_prop) || sigma2_prop <= 0) sigma2_prop = sigma2_gp;
    double log_prior_curr = -(-std::log(prior_sigma_gp_alpha) / prior_sigma_gp_U) * std::sqrt(sigma2_gp);
    double log_prior_prop = -(-std::log(prior_sigma_gp_alpha) / prior_sigma_gp_U) * std::sqrt(sigma2_prop);

    double log_lik_diff = 0.0;
    for (int i = 0; i < n_spatial; i++) {
      log_lik_diff += -0.5 * w[nn_order[i]] * w[nn_order[i]] / sigma2_prop;
      log_lik_diff -= -0.5 * w[nn_order[i]] * w[nn_order[i]] / sigma2_gp;
    }

    double log_alpha = log_lik_diff + log_prior_prop - log_prior_curr +
                       std::log(sigma2_prop) - std::log(sigma2_gp);
    if (std::log(R::runif(0, 1)) < log_alpha) {
      sigma2_gp = sigma2_prop;
    }

    double phi_prop = tulpa_linalg::safe_exp(std::log(phi_gp) + R::rnorm(0, 0.1));
    if (std::isfinite(phi_prop) && phi_prop >= prior_phi_lower && phi_prop <= prior_phi_upper) {
      double log_alpha_phi = std::log(phi_prop) - std::log(phi_gp);
      if (std::log(R::runif(0, 1)) < log_alpha_phi) {
        phi_gp = phi_prop;
      }
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
      for (int s = 0; s < n_spatial; s++) {
        gp_draws(save_idx, s) = w[s];
      }
      sigma2_gp_draws[save_idx] = sigma2_gp;
      phi_gp_draws[save_idx] = phi_gp;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws_gp(save_idx, i) = eta[i];
        }
      }
      save_idx++;
    }

    if ((iter + 1) % 100 == 0) Rcpp::checkUserInterrupt();
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = beta_draws,
    Rcpp::Named("re") = re_draws,
    Rcpp::Named("sigma_re") = sigma_re_draws,
    Rcpp::Named("gp") = gp_draws,
    Rcpp::Named("sigma2_gp") = sigma2_gp_draws,
    Rcpp::Named("phi_gp") = phi_gp_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws_gp;
  }

  PutRNGstate();
  return result;
}

// ---------------------------------------------------------------------
// Multiscale Temporal Gibbs Sampler for Binomial Models
// Supports trend (RW1/RW2) + seasonal (cyclic RW1) + short-term (AR1/IID)
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_temporal(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector time_idx,
    int n_times,
    int seasonal_period,
    int trend_type,
    int short_type,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_sigma_trend_scale = 1.0,
    double prior_sigma_seasonal_scale = 1.0,
    double prior_sigma_short_scale = 1.0,
    double rho_short_init = 0.5,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  // CRITICAL: Must call GetRNGstate/PutRNGstate when using R's RNG from C++
  GetRNGstate();

  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  int n_trend = (trend_type > 0) ? n_times : 0;
  int n_seasonal = (seasonal_period > 0) ? seasonal_period : 0;
  int n_short = (short_type > 0) ? n_times : 0;

  #ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
  #endif

  // Storage
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix trend_draws(n_save, n_trend);
  Rcpp::NumericMatrix seasonal_draws(n_save, n_seasonal);
  Rcpp::NumericMatrix short_draws(n_save, n_short);
  Rcpp::NumericVector sigma_trend_draws(n_save);
  Rcpp::NumericVector sigma_seasonal_draws(n_save);
  Rcpp::NumericVector sigma_short_draws(n_save);
  Rcpp::NumericVector rho_short_draws(n_save);
  Rcpp::NumericMatrix eta_draws_temp;
  if (store_eta) eta_draws_temp = Rcpp::NumericMatrix(n_save, N);

  // Initialize
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  Rcpp::NumericVector trend(n_trend, 0.0);
  Rcpp::NumericVector seasonal(n_seasonal, 0.0);
  Rcpp::NumericVector short_term(n_short, 0.0);
  double sigma_trend = 1.0;
  double sigma_seasonal = 1.0;
  double sigma_short = 1.0;
  double rho_short = rho_short_init;

  // Working vectors
  Rcpp::NumericVector omega(N);
  Rcpp::NumericVector kappa(N);
  Rcpp::NumericVector eta(N);
  Rcpp::NumericVector X_beta(N);
  Rcpp::NumericVector re_contrib(N);
  Rcpp::NumericVector temp_contrib(N);
  Rcpp::NumericVector offset(N);

  for (int i = 0; i < N; i++) {
    omega[i] = 0.5;
    kappa[i] = y[i] - 0.5 * n[i];
    X_beta[i] = 0.0;
    re_contrib[i] = 0.0;
    temp_contrib[i] = 0.0;
  }

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    if (verbose && (iter + 1) % 200 == 0) {
      Rcpp::Rcout << "  Iteration " << (iter + 1) << "/" << n_iter << "\n";
    }

    // 1. Compute temporal contributions
    for (int i = 0; i < N; i++) {
      int t = time_idx[i] - 1;
      double temp_eff = 0.0;
      if (t >= 0 && t < n_times) {
        if (n_trend > 0 && t < n_trend) temp_eff += trend[t];
        if (n_seasonal > 0) temp_eff += seasonal[t % seasonal_period];
        if (n_short > 0 && t < n_short) temp_eff += short_term[t];
      }
      temp_contrib[i] = temp_eff;
    }

    // 2-4. Core Gibbs step (eta, omega, beta, RE) — shared with all variants
    tulpa::pg_gibbs_core_step(
        N, p, beta, re, sigma_re, omega, eta, X_beta, re_contrib,
        temp_contrib, offset, kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale);

    // 5. Update temporal effects
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i];
    }

    // Aggregate for trend
    std::vector<double> sum_omega_t(n_times, 0.0);
    std::vector<double> sum_resid_t(n_times, 0.0);
    for (int i = 0; i < N; i++) {
      int t = time_idx[i] - 1;
      if (t >= 0 && t < n_times) {
        sum_omega_t[t] += omega[i];
        double other_temp = 0.0;
        if (n_seasonal > 0) other_temp += seasonal[t % seasonal_period];
        if (n_short > 0 && t < n_short) other_temp += short_term[t];
        sum_resid_t[t] += kappa[i] - omega[i] * (offset[i] + other_temp);
      }
    }

    // Update trend (RW1)
    if (trend_type == 1) {
      double tau_trend = 1.0 / (sigma_trend * sigma_trend);
      for (int t = 0; t < n_trend; t++) {
        double tau_prior, mean_prior;
        if (t == 0) {
          tau_prior = tau_trend;
          mean_prior = (n_trend > 1) ? trend[1] : 0.0;
        } else if (t == n_trend - 1) {
          tau_prior = tau_trend;
          mean_prior = trend[t - 1];
        } else {
          tau_prior = 2.0 * tau_trend;
          mean_prior = 0.5 * (trend[t - 1] + trend[t + 1]);
        }

        double tau_post = tau_prior + sum_omega_t[t];
        double mean_post = (tau_prior * mean_prior + sum_resid_t[t]) / tau_post;
        trend[t] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
      }

      double ss = 0.0;
      for (int t = 1; t < n_trend; t++) {
        double diff = trend[t] - trend[t - 1];
        ss += diff * diff;
      }
      double shape = prior_sigma_trend_scale + 0.5 * (n_trend - 1);
      double rate = prior_sigma_trend_scale + 0.5 * ss;
      sigma_trend = 1.0 / std::sqrt(R::rgamma(shape, 1.0 / rate));
    }

    // Update seasonal (cyclic RW1)
    if (n_seasonal > 0) {
      std::vector<double> sum_omega_s(seasonal_period, 0.0);
      std::vector<double> sum_resid_s(seasonal_period, 0.0);
      for (int i = 0; i < N; i++) {
        int t = time_idx[i] - 1;
        if (t >= 0) {
          int s = t % seasonal_period;
          sum_omega_s[s] += omega[i];
          double other_temp = 0.0;
          if (n_trend > 0 && t < n_trend) other_temp += trend[t];
          if (n_short > 0 && t < n_short) other_temp += short_term[t];
          sum_resid_s[s] += kappa[i] - omega[i] * (offset[i] + other_temp);
        }
      }

      double tau_seasonal_val = 1.0 / (sigma_seasonal * sigma_seasonal);
      for (int s = 0; s < n_seasonal; s++) {
        int s_prev = (s == 0) ? n_seasonal - 1 : s - 1;
        int s_next = (s == n_seasonal - 1) ? 0 : s + 1;

        double tau_prior = 2.0 * tau_seasonal_val;
        double mean_prior = 0.5 * (seasonal[s_prev] + seasonal[s_next]);

        double tau_post = tau_prior + sum_omega_s[s];
        double mean_post = (tau_prior * mean_prior + sum_resid_s[s]) / tau_post;
        seasonal[s] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
      }

      double mean_s = 0.0;
      for (int s = 0; s < n_seasonal; s++) mean_s += seasonal[s];
      mean_s /= n_seasonal;
      for (int s = 0; s < n_seasonal; s++) seasonal[s] -= mean_s;

      double ss = 0.0;
      for (int s = 0; s < n_seasonal; s++) {
        int s_next = (s == n_seasonal - 1) ? 0 : s + 1;
        double diff = seasonal[s_next] - seasonal[s];
        ss += diff * diff;
      }
      double shape = prior_sigma_seasonal_scale + 0.5 * n_seasonal;
      double rate = prior_sigma_seasonal_scale + 0.5 * ss;
      sigma_seasonal = 1.0 / std::sqrt(R::rgamma(shape, 1.0 / rate));
    }

    // Update short-term (AR1 or IID)
    if (short_type > 0) {
      std::vector<double> sum_omega_sh(n_short, 0.0);
      std::vector<double> sum_resid_sh(n_short, 0.0);
      for (int i = 0; i < N; i++) {
        int t = time_idx[i] - 1;
        if (t >= 0 && t < n_short) {
          sum_omega_sh[t] += omega[i];
          double other_temp = 0.0;
          if (n_trend > 0 && t < n_trend) other_temp += trend[t];
          if (n_seasonal > 0) other_temp += seasonal[t % seasonal_period];
          sum_resid_sh[t] += kappa[i] - omega[i] * (offset[i] + other_temp);
        }
      }

      double tau_short_val = 1.0 / (sigma_short * sigma_short);

      if (short_type == 1) {  // AR1
        for (int t = 0; t < n_short; t++) {
          double tau_prior, mean_prior;
          if (t == 0) {
            tau_prior = tau_short_val * (1.0 - rho_short * rho_short);
            mean_prior = 0.0;
          } else {
            tau_prior = tau_short_val;
            mean_prior = rho_short * short_term[t - 1];
          }

          double tau_post = tau_prior + sum_omega_sh[t];
          double mean_post = (tau_prior * mean_prior + sum_resid_sh[t]) / tau_post;
          short_term[t] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
        }
      } else {  // IID
        for (int t = 0; t < n_short; t++) {
          double tau_post = tau_short_val + sum_omega_sh[t];
          double mean_post = sum_resid_sh[t] / tau_post;
          short_term[t] = R::rnorm(mean_post, 1.0 / std::sqrt(tau_post));
        }
      }

      double ss = 0.0;
      if (short_type == 1) {
        ss = short_term[0] * short_term[0] * (1.0 - rho_short * rho_short);
        for (int t = 1; t < n_short; t++) {
          double resid = short_term[t] - rho_short * short_term[t - 1];
          ss += resid * resid;
        }
      } else {
        for (int t = 0; t < n_short; t++) {
          ss += short_term[t] * short_term[t];
        }
      }
      double shape = prior_sigma_short_scale + 0.5 * n_short;
      double rate = prior_sigma_short_scale + 0.5 * ss;
      sigma_short = 1.0 / std::sqrt(R::rgamma(shape, 1.0 / rate));
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
      for (int t = 0; t < n_trend; t++) {
        trend_draws(save_idx, t) = trend[t];
      }
      for (int s = 0; s < n_seasonal; s++) {
        seasonal_draws(save_idx, s) = seasonal[s];
      }
      for (int t = 0; t < n_short; t++) {
        short_draws(save_idx, t) = short_term[t];
      }
      sigma_trend_draws[save_idx] = sigma_trend;
      sigma_seasonal_draws[save_idx] = sigma_seasonal;
      sigma_short_draws[save_idx] = sigma_short;
      rho_short_draws[save_idx] = rho_short;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws_temp(save_idx, i) = eta[i];
        }
      }
      save_idx++;
    }

    if ((iter + 1) % 100 == 0) Rcpp::checkUserInterrupt();
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = beta_draws,
    Rcpp::Named("re") = re_draws,
    Rcpp::Named("sigma_re") = sigma_re_draws,
    Rcpp::Named("trend") = trend_draws,
    Rcpp::Named("seasonal") = seasonal_draws,
    Rcpp::Named("short_term") = short_draws,
    Rcpp::Named("sigma_trend") = sigma_trend_draws,
    Rcpp::Named("sigma_seasonal") = sigma_seasonal_draws,
    Rcpp::Named("sigma_short") = sigma_short_draws,
    Rcpp::Named("rho_short") = rho_short_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws_temp;
  }

  PutRNGstate();
  return result;
}


// -----------------------------------------------------------------------------
// Multiscale GP Gibbs sampler (local + regional components)
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_multiscale_gp(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx_local,
    Rcpp::NumericMatrix nn_dist_local,
    Rcpp::IntegerVector nn_order_local,
    int nn_local,
    Rcpp::IntegerMatrix nn_idx_regional,
    Rcpp::NumericMatrix nn_dist_regional,
    Rcpp::IntegerVector nn_order_regional,
    int nn_regional,
    int n_spatial,
    double sigma2_local_init,
    double phi_local_init,
    double sigma2_regional_init,
    double phi_regional_init,
    int cov_type,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_sigma_local_U = 1.0,
    double prior_sigma_local_alpha = 0.01,
    double prior_phi_local_lower = 0.01,
    double prior_phi_local_upper = 5.0,
    double prior_sigma_regional_U = 1.0,
    double prior_sigma_regional_alpha = 0.01,
    double prior_phi_regional_lower = 0.1,
    double prior_phi_regional_upper = 20.0,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  // CRITICAL: Must call GetRNGstate/PutRNGstate when using R's RNG from C++
  GetRNGstate();

  int N = y.size();
  int p = X.ncol();

  if (verbose) {
    Rcpp::Rcout << "PG Binomial Gibbs sampler with multiscale GP spatial\n";
    Rcpp::Rcout << "  N = " << N << ", p = " << p << "\n";
    Rcpp::Rcout << "  n_spatial = " << n_spatial << "\n";
    Rcpp::Rcout << "  nn_local = " << nn_local << ", nn_regional = " << nn_regional << "\n";
  }

  // Initialize parameters
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;

  // Local GP effects
  std::vector<double> w_local(n_spatial, 0.0);
  double sigma2_local = sigma2_local_init;
  double phi_local = phi_local_init;

  // Regional GP effects
  std::vector<double> w_regional(n_spatial, 0.0);
  double sigma2_regional = sigma2_regional_init;
  double phi_regional = phi_regional_init;

  // Working vectors (use Rcpp types for compatibility with update functions)
  Rcpp::NumericVector omega(N, 0.0);
  Rcpp::NumericVector kappa(N, 0.0);
  Rcpp::NumericVector eta_vec(N, 0.0);
  Rcpp::NumericVector X_beta(N, 0.0);
  Rcpp::NumericVector re_contrib(N, 0.0);
  Rcpp::NumericVector local_contrib(N, 0.0);
  Rcpp::NumericVector regional_contrib(N, 0.0);
  Rcpp::NumericVector offset(N, 0.0);

  // Compute kappa
  for (int i = 0; i < N; i++) {
    kappa[i] = (double)y[i] - 0.5 * (double)n[i];
  }

  // Storage for draws
  int n_save = (n_iter - n_warmup) / thin;
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix w_local_draws(n_save, n_spatial);
  Rcpp::NumericMatrix w_regional_draws(n_save, n_spatial);
  Rcpp::NumericVector sigma2_local_draws(n_save);
  Rcpp::NumericVector phi_local_draws(n_save);
  Rcpp::NumericVector sigma2_regional_draws(n_save);
  Rcpp::NumericVector phi_regional_draws(n_save);
  Rcpp::NumericMatrix eta_draws_temp;
  if (store_eta) {
    eta_draws_temp = Rcpp::NumericMatrix(n_save, N);
  }

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    if (verbose && (iter + 1) % 200 == 0) {
      Rcpp::Rcout << "  Iteration " << (iter + 1) << "/" << n_iter << "\n";
    }

    // 1. Compute combined spatial contribution
    Rcpp::NumericVector combined_contrib(N);
    for (int i = 0; i < N; i++) {
      combined_contrib[i] = local_contrib[i] + regional_contrib[i];
    }

    // 2-4. Core Gibbs step (eta, omega, beta, RE) — shared with all variants
    tulpa::pg_gibbs_core_step(
        N, p, beta, re, sigma_re, omega, eta_vec, X_beta, re_contrib,
        combined_contrib, offset, kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale);

    // 5. Update local GP effects
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i] + regional_contrib[i];
    }

    // Aggregate likelihood info per spatial location
    std::vector<double> sum_omega_local(n_spatial, 0.0);
    std::vector<double> sum_resid_local(n_spatial, 0.0);
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        sum_omega_local[i] += omega[i];
        sum_resid_local[i] += kappa[i] - omega[i] * offset[i];
      }
    }

    // Sequential NNGP Gibbs update for local effects
    double tau_local = 1.0 / sigma2_local;
    for (int ii = 0; ii < n_spatial; ii++) {
      int i = nn_order_local[ii];

      double cond_mean = 0.0;
      double cond_prec = tau_local;

      for (int k = 0; k < nn_local; k++) {
        int neighbor = nn_idx_local(i, k) - 1;
        if (neighbor >= 0 && neighbor < n_spatial) {
          double dist = nn_dist_local(i, k);
          double cov_val = std::exp(-dist / phi_local);
          cond_mean += cov_val * w_local[neighbor];
        }
      }
      cond_mean *= tau_local;

      double post_prec = cond_prec + sum_omega_local[i];
      double post_mean = (cond_mean + sum_resid_local[i]) / post_prec;
      w_local[i] = R::rnorm(post_mean, 1.0 / std::sqrt(post_prec));
    }

    // Update local contributions
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        local_contrib[i] = w_local[i];
      }
    }

    // 6. Update regional GP effects
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i] + local_contrib[i];
    }

    std::vector<double> sum_omega_regional(n_spatial, 0.0);
    std::vector<double> sum_resid_regional(n_spatial, 0.0);
    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        sum_omega_regional[i] += omega[i];
        sum_resid_regional[i] += kappa[i] - omega[i] * offset[i];
      }
    }

    double tau_regional = 1.0 / sigma2_regional;
    for (int ii = 0; ii < n_spatial; ii++) {
      int i = nn_order_regional[ii];

      double cond_mean = 0.0;
      double cond_prec = tau_regional;

      for (int k = 0; k < nn_regional; k++) {
        int neighbor = nn_idx_regional(i, k) - 1;
        if (neighbor >= 0 && neighbor < n_spatial) {
          double dist = nn_dist_regional(i, k);
          double cov_val = std::exp(-dist / phi_regional);
          cond_mean += cov_val * w_regional[neighbor];
        }
      }
      cond_mean *= tau_regional;

      double post_prec = cond_prec + sum_omega_regional[i];
      double post_mean = (cond_mean + sum_resid_regional[i]) / post_prec;
      w_regional[i] = R::rnorm(post_mean, 1.0 / std::sqrt(post_prec));
    }

    for (int i = 0; i < N; i++) {
      if (i < n_spatial) {
        regional_contrib[i] = w_regional[i];
      }
    }

    // 7. Update hyperparameters via MH
    // Update sigma2_local (Gibbs from inverse-gamma)
    double ss_local = 0.0;
    for (int i = 0; i < n_spatial; i++) {
      ss_local += w_local[i] * w_local[i];
    }
    double shape_local = 0.5 * n_spatial + 1.0;
    double rate_local = 0.5 * ss_local + prior_sigma_local_U;
    sigma2_local = 1.0 / R::rgamma(shape_local, 1.0 / rate_local);

    // Update sigma2_regional
    double ss_regional = 0.0;
    for (int i = 0; i < n_spatial; i++) {
      ss_regional += w_regional[i] * w_regional[i];
    }
    double shape_regional = 0.5 * n_spatial + 1.0;
    double rate_regional = 0.5 * ss_regional + prior_sigma_regional_U;
    sigma2_regional = 1.0 / R::rgamma(shape_regional, 1.0 / rate_regional);

    // Update phi_local via random walk MH
    double phi_local_prop = phi_local * tulpa_linalg::safe_exp(R::rnorm(0, 0.1));
    if (std::isfinite(phi_local_prop) && phi_local_prop >= prior_phi_local_lower && phi_local_prop <= prior_phi_local_upper) {
      double ll_curr = 0.0, ll_prop = 0.0;
      for (int i = 0; i < n_spatial; i++) {
        double cond_mean_curr = 0.0, cond_mean_prop = 0.0;
        for (int k = 0; k < nn_local; k++) {
          int neighbor = nn_idx_local(i, k) - 1;
          if (neighbor >= 0 && neighbor < n_spatial) {
            double dist = nn_dist_local(i, k);
            cond_mean_curr += std::exp(-dist / phi_local) * w_local[neighbor];
            cond_mean_prop += std::exp(-dist / phi_local_prop) * w_local[neighbor];
          }
        }
        ll_curr += -0.5 * tau_local * (w_local[i] - cond_mean_curr) * (w_local[i] - cond_mean_curr);
        ll_prop += -0.5 * tau_local * (w_local[i] - cond_mean_prop) * (w_local[i] - cond_mean_prop);
      }
      double log_ratio = ll_prop - ll_curr + std::log(phi_local_prop / phi_local);
      if (std::log(R::runif(0, 1)) < log_ratio) {
        phi_local = phi_local_prop;
      }
    }

    // Update phi_regional via MH
    double phi_regional_prop = phi_regional * tulpa_linalg::safe_exp(R::rnorm(0, 0.1));
    if (std::isfinite(phi_regional_prop) && phi_regional_prop >= prior_phi_regional_lower && phi_regional_prop <= prior_phi_regional_upper) {
      double ll_curr = 0.0, ll_prop = 0.0;
      for (int i = 0; i < n_spatial; i++) {
        double cond_mean_curr = 0.0, cond_mean_prop = 0.0;
        for (int k = 0; k < nn_regional; k++) {
          int neighbor = nn_idx_regional(i, k) - 1;
          if (neighbor >= 0 && neighbor < n_spatial) {
            double dist = nn_dist_regional(i, k);
            cond_mean_curr += std::exp(-dist / phi_regional) * w_regional[neighbor];
            cond_mean_prop += std::exp(-dist / phi_regional_prop) * w_regional[neighbor];
          }
        }
        ll_curr += -0.5 * tau_regional * (w_regional[i] - cond_mean_curr) * (w_regional[i] - cond_mean_curr);
        ll_prop += -0.5 * tau_regional * (w_regional[i] - cond_mean_prop) * (w_regional[i] - cond_mean_prop);
      }
      double log_ratio = ll_prop - ll_curr + std::log(phi_regional_prop / phi_regional);
      if (std::log(R::runif(0, 1)) < log_ratio) {
        phi_regional = phi_regional_prop;
      }
    }

    // Store draws after warmup
    if (iter >= n_warmup && (iter - n_warmup + 1) % thin == 0) {
      for (int j = 0; j < p; j++) {
        beta_draws(save_idx, j) = beta[j];
      }
      for (int g = 0; g < n_re_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_re_draws[save_idx] = sigma_re;

      for (int s = 0; s < n_spatial; s++) {
        w_local_draws(save_idx, s) = w_local[s];
        w_regional_draws(save_idx, s) = w_regional[s];
      }
      sigma2_local_draws[save_idx] = sigma2_local;
      phi_local_draws[save_idx] = phi_local;
      sigma2_regional_draws[save_idx] = sigma2_regional;
      phi_regional_draws[save_idx] = phi_regional;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_draws_temp(save_idx, i) = eta_vec[i];
        }
      }
      save_idx++;
    }

    if ((iter + 1) % 100 == 0) Rcpp::checkUserInterrupt();
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = beta_draws,
    Rcpp::Named("re") = re_draws,
    Rcpp::Named("sigma_re") = sigma_re_draws,
    Rcpp::Named("w_local") = w_local_draws,
    Rcpp::Named("w_regional") = w_regional_draws,
    Rcpp::Named("sigma2_local") = sigma2_local_draws,
    Rcpp::Named("phi_local") = phi_local_draws,
    Rcpp::Named("sigma2_regional") = sigma2_regional_draws,
    Rcpp::Named("phi_regional") = phi_regional_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws_temp;
  }

  PutRNGstate();
  return result;
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

  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  #ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  #endif

  // Storage for draws
  Rcpp::NumericMatrix beta_draws(n_save, p);
  Rcpp::NumericMatrix re_draws(n_save, n_re_groups);
  Rcpp::NumericVector sigma_re_draws(n_save);
  Rcpp::NumericMatrix spatial_raw_draws(n_save, n_spatial_units);
  Rcpp::NumericMatrix spatial_proj_draws(n_save, n_spatial_units);
  Rcpp::NumericVector tau_draws(n_save);
  Rcpp::NumericMatrix eta_draws(n_save, N);

  // Current values
  Rcpp::NumericVector beta(p, 0.0);
  Rcpp::NumericVector re(n_re_groups, 0.0);
  double sigma_re = 1.0;
  Rcpp::NumericVector phi(n_spatial_units, 0.0);  // Raw (unprojected) spatial effects
  Rcpp::NumericVector phi_proj(n_spatial_units, 0.0);  // Projected spatial effects
  double tau = 1.0;

  // Compute kappa once
  Rcpp::NumericVector kappa(N);
  for (int i = 0; i < N; i++) {
    kappa[i] = y[i] - n[i] / 2.0;
  }

  // Working vectors
  Rcpp::NumericVector omega(N, 0.1);
  Rcpp::NumericVector offset(N);
  Rcpp::NumericVector eta(N);
  Rcpp::NumericVector X_beta(N);
  Rcpp::NumericVector re_contrib(N, 0.0);
  Rcpp::NumericVector spatial_contrib(N, 0.0);

  // Initialize X_beta
  for (int i = 0; i < N; i++) {
    X_beta[i] = 0.0;
    for (int j = 0; j < p; j++) {
      X_beta[i] += X(i, j) * beta[j];
    }
  }

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
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      int s = spatial_group[i] - 1;
      spatial_contrib[i] = (s >= 0 && s < n_spatial_units) ? phi_proj[s] : 0.0;
    }

    // Steps 3-7: shared core (compute eta, sample omega, update beta/RE)
    tulpa::pg_gibbs_core_step(
        N, p, beta, re, sigma_re, omega, eta, X_beta, re_contrib,
        spatial_contrib, offset, kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale);

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
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      offset[i] = X_beta[i] + re_contrib[i];
    }

    // Update spatial effects using ICAR
    phi = tulpa::update_spatial_icar(kappa, omega, offset, spatial_group, adj_list, n_neighbors, tau);

    // 9. Update tau (spatial precision)
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

      // Store both raw and projected spatial effects
      for (int s = 0; s < n_spatial_units; s++) {
        spatial_raw_draws(save_idx, s) = phi[s];
        spatial_proj_draws(save_idx, s) = phi_proj[s];
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
    Rcpp::Named("spatial_raw") = spatial_raw_draws,
    Rcpp::Named("spatial") = spatial_proj_draws,
    Rcpp::Named("tau") = tau_draws
  );

  if (store_eta) {
    result["eta"] = eta_draws;
  }

  PutRNGstate();
  return result;
}
