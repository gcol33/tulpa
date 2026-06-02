// pg_negbin.cpp
// Pólya-Gamma Gibbs sampler for Negative Binomial models
// Uses PG augmentation for the logit component and CRT for dispersion
// Reference: Zhou et al. (2012) "Lognormal and Gamma Mixed Negative Binomial Regression"

#include "pg_negbin.h"
#include "pg_shared.h"
#include "pg_rng.h"
#include "crt_rng.h"
#include "linalg_fast.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>  // for std::pair

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace tulpa {

// ---------------------------------------------------------------------
// The PG normal-normal beta and random-effect updates are model-agnostic
// (the post-PG conjugate step is identical to the binomial sampler), so
// this file reuses the shared update_beta / update_re kernels declared in
// pg_shared.h. Only the NB-specific dispersion (r) update lives below.
// ---------------------------------------------------------------------

// Update sigma_re with proper half-Cauchy prior using auxiliary variable
// Reference: Gelman (2006) "Prior distributions for variance parameters"
// Half-Cauchy(0, scale): use auxiliary a ~ IG(1/2, 1/scale^2)
//                        and sigma^2 | a ~ IG(1/2, 1/a)
struct SigmaReState {
  double sigma_re;
  double aux;  // auxiliary variable for half-Cauchy
};

SigmaReState update_sigma_re_negbin_hc(
    const NumericVector& re,
    double scale,
    double aux_current
) {
  int J = re.size();
  SigmaReState result;

  double ss = 0.0;
  for (int j = 0; j < J; j++) {
    ss += re[j] * re[j];
  }

  // Update sigma^2 | b, a ~ IG((J + 1)/2, ss/2 + 1/a)
  double shape_sigma = (J + 1.0) / 2.0;
  double rate_sigma = ss / 2.0 + 1.0 / aux_current;

  double sigma_sq = 1.0 / R::rgamma(shape_sigma, 1.0 / rate_sigma);
  result.sigma_re = std::sqrt(sigma_sq);

  // Update a | sigma^2 ~ IG(1, 1/scale^2 + 1/sigma^2)
  double shape_aux = 1.0;
  double rate_aux = 1.0 / (scale * scale) + 1.0 / sigma_sq;
  result.aux = 1.0 / R::rgamma(shape_aux, 1.0 / rate_aux);

  return result;
}

// Simple version for backward compatibility — delegates to shared helper
double update_sigma_re_negbin(
    const NumericVector& re,
    double scale
) {
  return tulpa::update_sigma_halfcauchy(re, scale);
}

// Center random effects (soft sum-to-zero constraint)
// This prevents RE from absorbing intercept and inflating sigma_re
void center_random_effects(NumericVector& re, NumericVector& beta) {
  int J = re.size();
  double re_mean = 0.0;
  for (int j = 0; j < J; j++) {
    re_mean += re[j];
  }
  re_mean /= J;

  // Absorb RE mean into intercept
  beta[0] += re_mean;

  // Center RE
  for (int j = 0; j < J; j++) {
    re[j] -= re_mean;
  }
}

// The PG-NB augmentation samples beta / eta on the Zhou (2012) log-odds scale,
// where the mean is mu = r * exp(eta). tulpa's neg_binomial_2 family is the NB2
// mean scale mu = exp(eta_nb2) (R/family_loglik.R, .mean_log), shared by the
// Laplace / NUTS backends. The two parameterizations differ only by a log(r)
// shift in the intercept: eta_nb2 = eta + log(r), so beta_nb2[0] = beta[0] +
// log(r) (slopes and random effects are identical). Report the user-facing
// draws on the NB2 mean scale so tulpa_gibbs() agrees with the other backends;
// the internal sampler stays on the Zhou scale, where the PG augmentation and
// the joint (r, beta0) update are derived. The transform is applied per draw
// (a derived quantity summarised after sampling), so the posterior is exact.
static inline void store_beta_nb2(Rcpp::NumericMatrix& beta_draws, int row,
                                  const Rcpp::NumericVector& beta, double r) {
  const int p = beta.size();
  beta_draws(row, 0) = beta[0] + std::log(r);   // intercept on the NB2 mean scale
  for (int j = 1; j < p; j++) beta_draws(row, j) = beta[j];
}

// Compute log-likelihood for NB given eta (linear predictor) and r
// In Zhou parameterization: mu = r * exp(eta), so p = exp(eta)/(1+exp(eta))
// This uses the standard NB2 likelihood with R's parameterization
static double negbin_loglik_eta(
    const IntegerVector& y,
    const NumericVector& eta,
    double r
) {
  int n = y.size();
  double ll = 0.0;

  for (int i = 0; i < n; i++) {
    // mu = r * exp(eta) in Zhou's logit parameterization
    double eta_i = std::max(-20.0, std::min(20.0, eta[i]));
    double mu_i = r * std::exp(eta_i);
    mu_i = std::max(1e-10, mu_i);

    // R's prob = r / (r + mu)
    double prob = r / (r + mu_i);
    prob = std::max(1e-10, std::min(1.0 - 1e-10, prob));

    ll += R::lgammafn(y[i] + r) - R::lgammafn(y[i] + 1) - R::lgammafn(r);
    ll += r * std::log(prob);
    ll += y[i] * std::log(1.0 - prob);
  }

  return ll;
}

// Joint MH update for (r, beta_0) to break confounding
// Key insight: mu = r * exp(eta), so when r changes, adjust beta_0 to keep mu stable
// This "ancillary" proposal explores the posterior ridge efficiently
struct JointRBeta0Result {
  double r;
  double beta0;
  bool accepted;
};

JointRBeta0Result update_r_beta0_joint(
    const IntegerVector& y,
    const NumericMatrix& X,
    const NumericVector& beta_current,
    const NumericVector& re_contrib,
    double r_current,
    double prior_r_shape,
    double prior_r_rate,
    double prior_beta_sd
) {
  int n = y.size();
  double beta0_current = beta_current[0];

  // Proposal: log(r) changes, beta_0 adjusts to maintain approximate mean
  double log_r_current = std::log(r_current);
  double proposal_sd_r = 0.2;

  double log_r_prop = R::rnorm(log_r_current, proposal_sd_r);
  double r_prop = std::exp(log_r_prop);

  // Compensating adjustment to beta_0: keep mu ≈ r * exp(beta_0 + ...)
  // If r increases by factor k, decrease beta_0 by log(k)
  double delta_log_r = log_r_prop - log_r_current;
  double beta0_prop = beta0_current - delta_log_r;  // Compensating move

  // Add small perturbation to explore
  beta0_prop += R::rnorm(0.0, 0.05);

  JointRBeta0Result result;
  result.r = r_current;
  result.beta0 = beta0_current;
  result.accepted = false;

  // Bounds check
  if (r_prop < 0.1 || r_prop > 500.0 || std::abs(beta0_prop) > 20.0) {
    return result;
  }

  // Compute eta for current and proposed
  NumericVector eta_current(n), eta_prop(n);
  for (int i = 0; i < n; i++) {
    double x_beta_other = 0.0;
    for (int j = 1; j < X.ncol(); j++) {
      x_beta_other += X(i, j) * beta_current[j];
    }
    eta_current[i] = beta0_current + x_beta_other + re_contrib[i];
    eta_prop[i] = beta0_prop + x_beta_other + re_contrib[i];
  }

  // Log likelihood
  double ll_current = negbin_loglik_eta(y, eta_current, r_current);
  double ll_prop = negbin_loglik_eta(y, eta_prop, r_prop);

  // Log prior for r: Gamma(shape, rate)
  double log_prior_r_current = (prior_r_shape - 1.0) * log_r_current - prior_r_rate * r_current;
  double log_prior_r_prop = (prior_r_shape - 1.0) * log_r_prop - prior_r_rate * r_prop;

  // Log prior for beta_0: N(0, prior_beta_sd^2)
  double log_prior_b0_current = -0.5 * beta0_current * beta0_current / (prior_beta_sd * prior_beta_sd);
  double log_prior_b0_prop = -0.5 * beta0_prop * beta0_prop / (prior_beta_sd * prior_beta_sd);

  // Jacobian for log(r) transform
  double log_jac_current = log_r_current;
  double log_jac_prop = log_r_prop;

  // MH ratio
  double log_alpha = (ll_prop + log_prior_r_prop + log_prior_b0_prop + log_jac_prop)
                   - (ll_current + log_prior_r_current + log_prior_b0_current + log_jac_current);

  if (std::log(R::runif(0.0, 1.0)) < log_alpha) {
    result.r = r_prop;
    result.beta0 = beta0_prop;
    result.accepted = true;
  }

  return result;
}

// Standard MH update for r (used as backup/additional update)
double update_r_negbin(
    const IntegerVector& y,
    const NumericVector& eta,
    double r_current,
    double prior_shape,
    double prior_rate
) {
  double log_r_current = std::log(r_current);
  double proposal_sd = 0.15;

  double log_r_prop = R::rnorm(log_r_current, proposal_sd);
  double r_prop = std::exp(log_r_prop);

  if (r_prop < 0.1 || r_prop > 500.0) {
    return r_current;
  }

  double log_prior_current = (prior_shape - 1.0) * log_r_current - prior_rate * r_current;
  double log_prior_prop = (prior_shape - 1.0) * log_r_prop - prior_rate * r_prop;

  double log_lik_current = negbin_loglik_eta(y, eta, r_current);
  double log_lik_prop = negbin_loglik_eta(y, eta, r_prop);

  double log_alpha = (log_prior_prop + log_lik_prop + log_r_prop)
                   - (log_prior_current + log_lik_current + log_r_current);

  if (std::log(R::runif(0.0, 1.0)) < log_alpha) {
    return r_prop;
  }
  return r_current;
}

// Method-of-moments initialization for NB model
// Returns (r_init, beta0_init) as a pair
static std::pair<double, double> nb_mom_init(const IntegerVector& y) {
  int n = y.size();

  // Compute mean and variance
  double sum_y = 0.0, sum_y2 = 0.0;
  for (int i = 0; i < n; i++) {
    sum_y += y[i];
    sum_y2 += y[i] * y[i];
  }
  double mu_hat = sum_y / n;
  double var_hat = (sum_y2 - sum_y * sum_y / n) / (n - 1);

  // For NB2: var = mu + mu^2/r
  // So: r = mu^2 / (var - mu)
  double r_init = 1.0;  // default
  if (var_hat > mu_hat && mu_hat > 0) {
    r_init = mu_hat * mu_hat / (var_hat - mu_hat);
    r_init = std::max(0.5, std::min(r_init, 100.0));  // Constrain
  }

  // In logit parameterization: mu = r * exp(eta)
  // So eta = log(mu/r), and with intercept-only: beta_0 = log(mu/r)
  double beta0_init = 0.0;
  if (mu_hat > 0 && r_init > 0) {
    beta0_init = std::log(mu_hat / r_init);
    beta0_init = std::max(-10.0, std::min(beta0_init, 10.0));
  }

  return std::make_pair(r_init, beta0_init);
}

// ---------------------------------------------------------------------
// Main Gibbs sampler for single NB process
// ---------------------------------------------------------------------

List pg_negbin_gibbs(
    IntegerVector y,
    NumericMatrix X,
    IntegerVector group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  #ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  #endif

  // Storage
  NumericMatrix beta_draws(n_save, p);
  NumericMatrix re_draws(n_save, n_groups);
  NumericVector sigma_draws(n_save);
  NumericVector r_draws(n_save);
  NumericMatrix eta_draws;
  if (store_eta) {
    eta_draws = NumericMatrix(n_save, N);
  }

  // Smart initialization using method of moments
  std::pair<double, double> mom_init = nb_mom_init(y);
  double r_mom = mom_init.first;
  double beta0_mom = mom_init.second;

  // Use MOM estimates if r_init is default (0 or negative)
  NumericVector beta(p, 0.0);
  NumericVector re(n_groups, 0.0);
  double sigma_re = 1.0;
  double r;

  if (r_init <= 0) {
    // Use method of moments
    r = r_mom;
    beta[0] = beta0_mom;
    if (verbose) {
      Rcpp::Rcout << "Using MOM init: r = " << r << ", beta0 = " << beta[0] << std::endl;
    }
  } else {
    r = r_init;
    // Still use MOM for beta0 to start in a good region
    beta[0] = beta0_mom;
  }
  NumericVector omega(N, 1.0);
  NumericVector kappa(N);
  NumericVector eta(N);
  NumericVector X_beta(N);
  NumericVector re_contrib(N);
  NumericVector p_success(N);  // p_i = logistic(eta_i)

  // Initialize auxiliary variable for half-Cauchy prior on sigma_re
  // a ~ IG(1/2, 1/scale^2) => E[a] = 2*scale^2
  double sigma_aux = 2.0 * prior_sigma_scale * prior_sigma_scale;

  // Gibbs iterations
  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {

    // 1. Compute linear predictor
    // Clamp eta to [-15, 15] to prevent numerical instability
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
      if (n_groups > 0) {
        re_contrib[i] = re[group[i] - 1];
      } else {
        re_contrib[i] = 0.0;
      }
      double eta_raw = X_beta[i] + re_contrib[i];
      eta[i] = std::max(-15.0, std::min(15.0, eta_raw));
    }

    // 2. Compute success probabilities: p_i = logistic(eta_i)
    for (int i = 0; i < N; i++) {
      p_success[i] = 1.0 / (1.0 + std::exp(-eta[i]));
    }

    // 3. Compute kappa = (y - r) / 2
    for (int i = 0; i < N; i++) {
      kappa[i] = (y[i] - r) / 2.0;
    }

    // 4. Sample omega ~ PG(y + r, eta)
    // Note: y + r can be non-integer, but rpg works for integer part
    // For fractional r, we approximate by rounding
    for (int i = 0; i < N; i++) {
      int b = static_cast<int>(std::round(y[i] + r));
      omega[i] = rpg_int(std::max(1, b), eta[i]);
    }

    // 5. Update beta | omega, re, y
    NumericVector beta_new = update_beta(kappa, omega, X, re_contrib, prior_beta_sd);

    // Check for numerical issues and keep previous value if invalid
    bool beta_valid = true;
    for (int j = 0; j < p; j++) {
      if (std::isnan(beta_new[j]) || std::isinf(beta_new[j]) ||
          std::abs(beta_new[j]) > 50.0) {
        beta_valid = false;
        break;
      }
    }
    if (beta_valid) {
      beta = beta_new;
    }

    // 6. Recompute X_beta
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    }

    // 7. Update random effects
    if (n_groups > 0) {
      NumericVector re_new = update_re(kappa, omega, X_beta, group, n_groups, sigma_re);

      // Check for numerical issues
      bool re_valid = true;
      for (int g = 0; g < n_groups; g++) {
        if (std::isnan(re_new[g]) || std::isinf(re_new[g]) ||
            std::abs(re_new[g]) > 50.0) {
          re_valid = false;
          break;
        }
      }
      if (re_valid) {
        re = re_new;
      }

      // Center random effects to prevent absorbing intercept
      // This is critical for correct sigma_re estimation
      center_random_effects(re, beta);

      // Recompute X_beta since beta[0] may have changed
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int i = 0; i < N; i++) {
        X_beta[i] = 0.0;
        for (int j = 0; j < p; j++) {
          X_beta[i] += X(i, j) * beta[j];
        }
      }

      // Update sigma_re with proper half-Cauchy prior using auxiliary variable
      SigmaReState sigma_state = update_sigma_re_negbin_hc(re, prior_sigma_scale, sigma_aux);
      if (!std::isnan(sigma_state.sigma_re) && !std::isinf(sigma_state.sigma_re) &&
          sigma_state.sigma_re < 100.0) {
        sigma_re = sigma_state.sigma_re;
        sigma_aux = sigma_state.aux;
      }
    }

    // 8. Update (r, beta_0) jointly to break confounding
    // This is the key fix for the identification issue
    NumericVector re_contrib_for_joint(N);
    for (int i = 0; i < N; i++) {
      re_contrib_for_joint[i] = (n_groups > 0) ? re[group[i] - 1] : 0.0;
    }

    // Joint update of (r, beta_0)
    JointRBeta0Result joint_result = update_r_beta0_joint(
        y, X, beta, re_contrib_for_joint, r,
        prior_r_shape, prior_r_rate, prior_beta_sd
    );
    if (joint_result.accepted) {
      r = joint_result.r;
      beta[0] = joint_result.beta0;
      // Recompute X_beta with new beta[0]
      for (int i = 0; i < N; i++) {
        X_beta[i] = 0.0;
        for (int j = 0; j < p; j++) {
          X_beta[i] += X(i, j) * beta[j];
        }
      }
    }

    // Also do a standard r update (interweaving for better mixing)
    NumericVector eta_for_r(N);
    for (int i = 0; i < N; i++) {
      eta_for_r[i] = X_beta[i] + re_contrib_for_joint[i];
    }
    r = update_r_negbin(y, eta_for_r, r, prior_r_shape, prior_r_rate);

    // Save draws (reported on the NB2 mean scale; see store_beta_nb2).
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      store_beta_nb2(beta_draws, save_idx, beta, r);
      for (int g = 0; g < n_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_draws[save_idx] = sigma_re;
      r_draws[save_idx] = r;

      if (store_eta) {
        const double log_r = std::log(r);
        for (int i = 0; i < N; i++) {
          eta_draws(save_idx, i) = eta[i] + log_r;  // NB2 log-mean
        }
      }
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter
                  << " (r = " << r << ")" << std::endl;
    }

    // Check for user interrupt
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  return List::create(
    Named("beta") = beta_draws,
    Named("re") = re_draws,
    Named("sigma_re") = sigma_draws,
    Named("r") = r_draws,
    Named("eta") = eta_draws
  );
}

// ---------------------------------------------------------------------
// Gibbs sampler for two-process NB ratio model
// (ratiod_negbin_negbin family)
// ---------------------------------------------------------------------

List pg_negbin_negbin_gibbs(
    IntegerVector y_num,
    IntegerVector y_denom,
    NumericMatrix X_num,
    NumericMatrix X_denom,
    IntegerVector group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    double prior_r_shape,
    double prior_r_rate,
    double r_num_init,
    double r_denom_init,
    bool shared,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  int N = y_num.size();
  int p_num = X_num.ncol();
  int p_denom = X_denom.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  #ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  #endif

  // Storage
  NumericMatrix beta_num_draws(n_save, p_num);
  NumericMatrix beta_denom_draws(n_save, p_denom);
  NumericMatrix re_draws(n_save, n_groups);
  NumericVector sigma_draws(n_save);
  NumericVector r_num_draws(n_save);
  NumericVector r_denom_draws(n_save);
  NumericMatrix eta_num_draws, eta_denom_draws;
  if (store_eta) {
    eta_num_draws = NumericMatrix(n_save, N);
    eta_denom_draws = NumericMatrix(n_save, N);
  }

  // Initialize
  NumericVector beta_num(p_num, 0.0);
  NumericVector beta_denom(p_denom, 0.0);
  NumericVector re(n_groups, 0.0);
  double sigma_re = 1.0;
  double r_num = r_num_init;
  double r_denom = r_denom_init;

  NumericVector omega_num(N, 1.0);
  NumericVector omega_denom(N, 1.0);
  NumericVector kappa_num(N), kappa_denom(N);
  NumericVector eta_num(N), eta_denom(N);
  NumericVector X_beta_num(N), X_beta_denom(N);
  NumericVector re_contrib(N);
  NumericVector eta_r_num(N), eta_r_denom(N);

  // Gibbs iterations
  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {

    // --- Numerator process ---

    // 1a. Compute numerator linear predictor
    for (int i = 0; i < N; i++) {
      X_beta_num[i] = 0.0;
      for (int j = 0; j < p_num; j++) {
        X_beta_num[i] += X_num(i, j) * beta_num[j];
      }
      if (shared && n_groups > 0) {
        re_contrib[i] = re[group[i] - 1];
      } else {
        re_contrib[i] = 0.0;
      }
      eta_num[i] = X_beta_num[i] + re_contrib[i];
      kappa_num[i] = (y_num[i] - r_num) / 2.0;
    }

    // 2a. Sample omega_num ~ PG(y_num + r_num, eta_num)
    for (int i = 0; i < N; i++) {
      int b = static_cast<int>(std::round(y_num[i] + r_num));
      omega_num[i] = rpg_int(std::max(1, b), eta_num[i]);
    }

    // 3a. Update beta_num
    beta_num = update_beta(kappa_num, omega_num, X_num, re_contrib, prior_beta_sd);

    // 4a. Update r_num
    for (int i = 0; i < N; i++) {
      double eta_new = 0.0;
      for (int j = 0; j < p_num; j++) {
        eta_new += X_num(i, j) * beta_num[j];
      }
      if (shared && n_groups > 0) {
        eta_new += re[group[i] - 1];
      }
      eta_r_num[i] = eta_new;
    }
    r_num = update_r_negbin(y_num, eta_r_num, r_num, prior_r_shape, prior_r_rate);

    // --- Denominator process ---

    // 1b. Compute denominator linear predictor
    for (int i = 0; i < N; i++) {
      X_beta_denom[i] = 0.0;
      for (int j = 0; j < p_denom; j++) {
        X_beta_denom[i] += X_denom(i, j) * beta_denom[j];
      }
      if (shared && n_groups > 0) {
        re_contrib[i] = re[group[i] - 1];
      } else {
        re_contrib[i] = 0.0;
      }
      eta_denom[i] = X_beta_denom[i] + re_contrib[i];
      kappa_denom[i] = (y_denom[i] - r_denom) / 2.0;
    }

    // 2b. Sample omega_denom ~ PG(y_denom + r_denom, eta_denom)
    for (int i = 0; i < N; i++) {
      int b = static_cast<int>(std::round(y_denom[i] + r_denom));
      omega_denom[i] = rpg_int(std::max(1, b), eta_denom[i]);
    }

    // 3b. Update beta_denom
    beta_denom = update_beta(kappa_denom, omega_denom, X_denom, re_contrib, prior_beta_sd);

    // 4b. Update r_denom
    for (int i = 0; i < N; i++) {
      double eta_new = 0.0;
      for (int j = 0; j < p_denom; j++) {
        eta_new += X_denom(i, j) * beta_denom[j];
      }
      if (shared && n_groups > 0) {
        eta_new += re[group[i] - 1];
      }
      eta_r_denom[i] = eta_new;
    }
    r_denom = update_r_negbin(y_denom, eta_r_denom, r_denom, prior_r_shape, prior_r_rate);

    // --- Shared random effects ---
    if (shared && n_groups > 0) {
      // Update RE using combined information from both processes
      // Posterior combines evidence from numerator and denominator

      // Recompute X_beta for both processes
      for (int i = 0; i < N; i++) {
        X_beta_num[i] = 0.0;
        for (int j = 0; j < p_num; j++) {
          X_beta_num[i] += X_num(i, j) * beta_num[j];
        }
        X_beta_denom[i] = 0.0;
        for (int j = 0; j < p_denom; j++) {
          X_beta_denom[i] += X_denom(i, j) * beta_denom[j];
        }
      }

      // Combined update for shared RE
      // Posterior: b_g | ... ~ N(m_g, v_g)
      // where information is pooled from both processes
      double prior_prec = 1.0 / (sigma_re * sigma_re + 1e-10);

      NumericVector sum_omega(n_groups);
      NumericVector sum_resid(n_groups);

      // Numerator contribution
      for (int i = 0; i < N; i++) {
        int g = group[i] - 1;
        sum_omega[g] += omega_num[i];
        sum_resid[g] += kappa_num[i] - omega_num[i] * X_beta_num[i];
      }

      // Denominator contribution
      for (int i = 0; i < N; i++) {
        int g = group[i] - 1;
        sum_omega[g] += omega_denom[i];
        sum_resid[g] += kappa_denom[i] - omega_denom[i] * X_beta_denom[i];
      }

      // Sample from posterior
      for (int g = 0; g < n_groups; g++) {
        double post_var = 1.0 / (sum_omega[g] + prior_prec);
        double post_mean = post_var * sum_resid[g];
        re[g] = R::rnorm(post_mean, std::sqrt(post_var));
      }

      // Update sigma_re
      sigma_re = update_sigma_re_negbin(re, prior_sigma_scale);
    }

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      for (int j = 0; j < p_num; j++) {
        beta_num_draws(save_idx, j) = beta_num[j];
      }
      for (int j = 0; j < p_denom; j++) {
        beta_denom_draws(save_idx, j) = beta_denom[j];
      }
      for (int g = 0; g < n_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      sigma_draws[save_idx] = sigma_re;
      r_num_draws[save_idx] = r_num;
      r_denom_draws[save_idx] = r_denom;

      if (store_eta) {
        for (int i = 0; i < N; i++) {
          eta_num_draws(save_idx, i) = eta_num[i];
          eta_denom_draws(save_idx, i) = eta_denom[i];
        }
      }
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter
                  << " (r_num=" << r_num << ", r_denom=" << r_denom << ")"
                  << std::endl;
    }

    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  return List::create(
    Named("beta_num") = beta_num_draws,
    Named("beta_denom") = beta_denom_draws,
    Named("re") = re_draws,
    Named("sigma_re") = sigma_draws,
    Named("r_num") = r_num_draws,
    Named("r_denom") = r_denom_draws,
    Named("eta_num") = eta_num_draws,
    Named("eta_denom") = eta_denom_draws
  );
}

// ---------------------------------------------------------------------
// Gibbs sampler for NB with spatial effects (ICAR)
// ---------------------------------------------------------------------

List pg_negbin_gibbs_spatial(
    IntegerVector y,
    NumericMatrix X,
    IntegerVector re_group,
    int n_re_groups,
    IntegerVector spatial_group,
    int n_spatial_units,
    List adj_list,
    IntegerVector n_neighbors,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_re_scale,
    double prior_tau_shape,
    double prior_tau_rate,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  int N = y.size();
  int p = X.ncol();
  int n_save = (n_iter - n_warmup) / thin;

  #ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(n_threads);
  }
  #endif

  // Storage
  NumericMatrix beta_draws(n_save, p);
  NumericMatrix re_draws(n_save, n_re_groups);
  NumericMatrix spatial_draws(n_save, n_spatial_units);
  NumericVector sigma_re_draws(n_save);
  NumericVector tau_draws(n_save);
  NumericVector r_draws(n_save);
  NumericMatrix eta_draws;
  if (store_eta) {
    eta_draws = NumericMatrix(n_save, N);
  }

  // Initialize
  NumericVector beta(p, 0.0);
  NumericVector re(n_re_groups, 0.0);
  NumericVector spatial(n_spatial_units, 0.0);
  double sigma_re = 1.0;
  double tau = 1.0;  // Spatial precision
  double r = r_init;

  NumericVector omega(N, 1.0);
  NumericVector kappa(N);
  NumericVector eta(N);
  NumericVector X_beta(N);
  NumericVector re_contrib(N);
  NumericVector spatial_contrib(N);

  // Gibbs iterations
  int save_idx = 0;
  for (int iter = 0; iter < n_iter; iter++) {

    // 1. Compute linear predictor
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
      if (n_re_groups > 0) {
        re_contrib[i] = re[re_group[i] - 1];
      } else {
        re_contrib[i] = 0.0;
      }
      spatial_contrib[i] = spatial[spatial_group[i] - 1];
      eta[i] = X_beta[i] + re_contrib[i] + spatial_contrib[i];
      kappa[i] = (y[i] - r) / 2.0;
    }

    // 2. Sample omega ~ PG(y + r, eta)
    for (int i = 0; i < N; i++) {
      int b = static_cast<int>(std::round(y[i] + r));
      omega[i] = rpg_int(std::max(1, b), eta[i]);
    }

    // 3. Update beta
    NumericVector offset(N);
    for (int i = 0; i < N; i++) {
      offset[i] = re_contrib[i] + spatial_contrib[i];
    }
    beta = update_beta(kappa, omega, X, offset, prior_beta_sd);

    // Recompute X_beta
    for (int i = 0; i < N; i++) {
      X_beta[i] = 0.0;
      for (int j = 0; j < p; j++) {
        X_beta[i] += X(i, j) * beta[j];
      }
    }

    // 4. Update random effects
    if (n_re_groups > 0) {
      NumericVector offset_re(N);
      for (int i = 0; i < N; i++) {
        offset_re[i] = X_beta[i] + spatial_contrib[i];
      }

      double prior_prec = 1.0 / (sigma_re * sigma_re + 1e-10);
      NumericVector sum_omega(n_re_groups);
      NumericVector sum_resid(n_re_groups);

      for (int i = 0; i < N; i++) {
        int g = re_group[i] - 1;
        sum_omega[g] += omega[i];
        sum_resid[g] += kappa[i] - omega[i] * offset_re[i];
      }

      for (int g = 0; g < n_re_groups; g++) {
        double post_var = 1.0 / (sum_omega[g] + prior_prec);
        double post_mean = post_var * sum_resid[g];
        re[g] = R::rnorm(post_mean, std::sqrt(post_var));
      }

      sigma_re = update_sigma_re_negbin(re, prior_sigma_re_scale);

      // Update re_contrib
      for (int i = 0; i < N; i++) {
        re_contrib[i] = re[re_group[i] - 1];
      }
    }

    // 5. Update spatial effects (ICAR)
    NumericVector offset_spatial(N);
    for (int i = 0; i < N; i++) {
      offset_spatial[i] = X_beta[i] + re_contrib[i];
    }

    // Aggregate by spatial unit
    NumericVector sum_omega_s(n_spatial_units);
    NumericVector sum_resid_s(n_spatial_units);

    for (int i = 0; i < N; i++) {
      int s = spatial_group[i] - 1;
      sum_omega_s[s] += omega[i];
      sum_resid_s[s] += kappa[i] - omega[i] * offset_spatial[i];
    }

    // Sample spatial effects with ICAR prior
    for (int s = 0; s < n_spatial_units; s++) {
      // ICAR: phi_s | phi_{-s} ~ N(mean of neighbors, 1/(tau * n_neighbors))
      IntegerVector neighbors = adj_list[s];
      int n_neigh = n_neighbors[s];

      double neighbor_mean = 0.0;
      if (n_neigh > 0) {
        for (int j = 0; j < n_neigh; j++) {
          neighbor_mean += spatial[neighbors[j] - 1];
        }
        neighbor_mean /= n_neigh;
      }

      // Combine ICAR prior with data likelihood
      double prior_prec_s = tau * n_neigh;
      double post_prec = sum_omega_s[s] + prior_prec_s;
      double post_mean = (sum_resid_s[s] + prior_prec_s * neighbor_mean) / post_prec;
      double post_sd = 1.0 / std::sqrt(post_prec);

      spatial[s] = R::rnorm(post_mean, post_sd);
    }

    // Center spatial effects (soft sum-to-zero constraint)
    double spatial_mean = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      spatial_mean += spatial[s];
    }
    spatial_mean /= n_spatial_units;
    for (int s = 0; s < n_spatial_units; s++) {
      spatial[s] -= spatial_mean;
    }

    // Update tau (spatial precision)
    // Prior: tau ~ Gamma(shape, rate)
    // Posterior: tau ~ Gamma(shape + (S-1)/2, rate + Q/2)
    // where Q = sum over edges (phi_i - phi_j)^2 / 2
    double Q = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
      IntegerVector neighbors = adj_list[s];
      for (int j = 0; j < n_neighbors[s]; j++) {
        int t = neighbors[j] - 1;
        if (t > s) {  // Count each edge once
          double diff = spatial[s] - spatial[t];
          Q += diff * diff;
        }
      }
    }

    double tau_shape = prior_tau_shape + (n_spatial_units - 1.0) / 2.0;
    double tau_rate = prior_tau_rate + Q / 2.0;
    tau = R::rgamma(tau_shape, 1.0 / tau_rate);

    // Update spatial_contrib
    for (int i = 0; i < N; i++) {
      spatial_contrib[i] = spatial[spatial_group[i] - 1];
    }

    // 6. Update dispersion r
    for (int i = 0; i < N; i++) {
      eta[i] = X_beta[i] + re_contrib[i] + spatial_contrib[i];
    }
    r = update_r_negbin(y, eta, r, prior_r_shape, prior_r_rate);

    // Save draws (reported on the NB2 mean scale; see store_beta_nb2).
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      store_beta_nb2(beta_draws, save_idx, beta, r);
      for (int g = 0; g < n_re_groups; g++) {
        re_draws(save_idx, g) = re[g];
      }
      for (int s = 0; s < n_spatial_units; s++) {
        spatial_draws(save_idx, s) = spatial[s];
      }
      sigma_re_draws[save_idx] = sigma_re;
      tau_draws[save_idx] = tau;
      r_draws[save_idx] = r;

      if (store_eta) {
        const double log_r = std::log(r);
        for (int i = 0; i < N; i++) {
          eta_draws(save_idx, i) = eta[i] + log_r;  // NB2 log-mean
        }
      }
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter
                  << " (r=" << r << ", tau=" << tau << ")" << std::endl;
    }

    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  return List::create(
    Named("beta") = beta_draws,
    Named("re") = re_draws,
    Named("spatial") = spatial_draws,
    Named("sigma_re") = sigma_re_draws,
    Named("tau") = tau_draws,
    Named("r") = r_draws,
    Named("eta") = eta_draws
  );
}

} // namespace tulpa

// ---------------------------------------------------------------------
// R exports
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_negbin_gibbs(
    Rcpp::IntegerVector y,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  return tulpa::pg_negbin_gibbs(
    y, X, group, n_groups, n_iter, n_warmup, thin,
    prior_beta_sd, prior_sigma_scale, prior_r_shape, prior_r_rate, r_init,
    store_eta, verbose, n_threads
  );
}

// [[Rcpp::export]]
Rcpp::List cpp_pg_negbin_negbin_gibbs(
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    Rcpp::IntegerVector group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    double prior_r_shape,
    double prior_r_rate,
    double r_num_init,
    double r_denom_init,
    bool shared,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  return tulpa::pg_negbin_negbin_gibbs(
    y_num, y_denom, X_num, X_denom, group, n_groups,
    n_iter, n_warmup, thin, prior_beta_sd, prior_sigma_scale,
    prior_r_shape, prior_r_rate, r_num_init, r_denom_init, shared,
    store_eta, verbose, n_threads
  );
}

// [[Rcpp::export]]
Rcpp::List cpp_pg_negbin_gibbs_spatial(
    Rcpp::IntegerVector y,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector spatial_group,
    int n_spatial_units,
    Rcpp::List adj_list,
    Rcpp::IntegerVector n_neighbors,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_re_scale,
    double prior_tau_shape,
    double prior_tau_rate,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    bool store_eta,
    bool verbose,
    int n_threads
) {
  return tulpa::pg_negbin_gibbs_spatial(
    y, X, re_group, n_re_groups, spatial_group, n_spatial_units,
    adj_list, n_neighbors, n_iter, n_warmup, thin,
    prior_beta_sd, prior_sigma_re_scale, prior_tau_shape, prior_tau_rate,
    prior_r_shape, prior_r_rate, r_init, store_eta, verbose, n_threads
  );
}
