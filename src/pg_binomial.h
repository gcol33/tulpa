// pg_binomial.h
// Pólya-Gamma Gibbs sampler for binomial models with random effects
// For the ratiod package: hierarchical ratio inference

#ifndef TULPA_PG_BINOMIAL_H
#define TULPA_PG_BINOMIAL_H

#include <Rcpp.h>
#include <vector>

namespace tulpa {

// ---------------------------------------------------------------------
// Core Gibbs sampler for binomial models
// ---------------------------------------------------------------------

// Pure C++ result structure (safe for OpenMP parallel regions)
struct PGResultCpp {
  std::vector<std::vector<double>> beta;       // [n_iter][n_beta]
  std::vector<std::vector<double>> re;         // [n_iter][n_re]
  std::vector<double> sigma_re;                // [n_iter]
  std::vector<std::vector<double>> eta;        // [n_iter][n_obs] (optional)
  int chain_id;
};

// R-compatible result structure (create outside parallel regions)
struct PGGibbsResult {
  Rcpp::NumericMatrix beta;       // Fixed effects [n_iter, n_beta]
  Rcpp::NumericMatrix re;         // Random effects [n_iter, n_re]
  Rcpp::NumericVector sigma_re;   // RE standard deviation [n_iter]
  Rcpp::NumericMatrix eta;        // Linear predictor [n_iter, n_obs] (optional)
  int chain_id;
};

// Convert C++ result to R result (call outside parallel region)
inline PGGibbsResult cpp_to_r_pg_result(const PGResultCpp& cpp_result, int n_beta, int n_re, int n_obs, bool store_eta) {
  int n_save = cpp_result.beta.size();
  PGGibbsResult r_result;
  r_result.beta = Rcpp::NumericMatrix(n_save, n_beta);
  r_result.re = Rcpp::NumericMatrix(n_save, n_re);
  r_result.sigma_re = Rcpp::NumericVector(n_save);
  r_result.chain_id = cpp_result.chain_id;

  for (int i = 0; i < n_save; i++) {
    for (int j = 0; j < n_beta; j++) {
      r_result.beta(i, j) = cpp_result.beta[i][j];
    }
    for (int g = 0; g < n_re; g++) {
      r_result.re(i, g) = cpp_result.re[i][g];
    }
    r_result.sigma_re[i] = cpp_result.sigma_re[i];
  }

  if (store_eta && !cpp_result.eta.empty()) {
    r_result.eta = Rcpp::NumericMatrix(n_save, n_obs);
    for (int i = 0; i < n_save; i++) {
      for (int j = 0; j < n_obs; j++) {
        r_result.eta(i, j) = cpp_result.eta[i][j];
      }
    }
  }

  return r_result;
}

// Main Gibbs sampler for binomial model with random effects
// Model: Y_i ~ Binomial(N_i, p_i)
//        logit(p_i) = X_i * beta + Z_i * b
//        b ~ N(0, sigma_re^2 * I)
//
// Uses Pólya-Gamma data augmentation:
// omega_i ~ PG(N_i, eta_i) where eta_i = logit(p_i)
// Then (Y_i - N_i/2) / omega_i ~ N(eta_i, 1/omega_i)
//
// @param y Integer vector of successes
// @param n Integer vector of trials
// @param X Design matrix for fixed effects
// @param Z Design matrix for random effects (group indicators)
// @param group Integer vector of group indices (1-based)
// @param n_groups Number of groups
// @param n_iter Number of MCMC iterations
// @param n_warmup Number of warmup iterations
// @param thin Thinning interval
// @param prior_beta_mean Prior mean for beta
// @param prior_beta_sd Prior SD for beta (scalar for all)
// @param prior_sigma_shape Shape parameter for sigma_re prior (half-Cauchy scale)
// @param store_eta Whether to store linear predictor draws
// @param verbose Print progress
Rcpp::List pg_binomial_gibbs(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    bool store_eta,
    bool verbose
);

// ---------------------------------------------------------------------
// Helper functions for blocked Gibbs updates
// ---------------------------------------------------------------------

// Update beta (fixed effects) given omega and random effects
// Uses normal-normal conjugacy after PG augmentation
Rcpp::NumericVector update_beta(
    const Rcpp::NumericVector& kappa,     // y - n/2
    const Rcpp::NumericVector& omega,     // PG draws
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& re_contrib, // Z * b
    double prior_sd
);

// Update random effects given omega and beta
// Uses normal-normal conjugacy
Rcpp::NumericVector update_re(
    const Rcpp::NumericVector& kappa,
    const Rcpp::NumericVector& omega,
    const Rcpp::NumericVector& X_beta,    // X * beta
    const Rcpp::IntegerVector& group,
    int n_groups,
    double sigma_re
);

// Update sigma_re using half-Cauchy prior
// p(sigma) ∝ 1/(1 + sigma^2/scale^2)
double update_sigma_re(
    const Rcpp::NumericVector& re,
    double scale
);

} // namespace tulpa

#endif // TULPA_PG_BINOMIAL_H
