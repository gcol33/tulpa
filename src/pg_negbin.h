// pg_negbin.h
// Pólya-Gamma Gibbs sampler for Negative Binomial models
// Uses PG augmentation for the logit component and CRT for dispersion
// Reference: Zhou et al. (2012) "Lognormal and Gamma Mixed Negative Binomial Regression"

#ifndef TULPA_PG_NEGBIN_H
#define TULPA_PG_NEGBIN_H

#include <Rcpp.h>
#include <vector>

namespace tulpa {

// ---------------------------------------------------------------------
// Negative Binomial Model with PG+CRT Augmentation
// ---------------------------------------------------------------------
//
// Model (NB2 parameterization):
//   Y_i ~ NB(r, p_i)
//   E[Y_i] = mu_i = r * p_i / (1 - p_i)
//   Var[Y_i] = mu_i + mu_i^2 / r
//
//   logit(p_i) = eta_i = X_i * beta + Z_i * b
//   b_g ~ N(0, sigma_re^2)
//
// Augmentation scheme:
//   1. omega_i ~ PG(y_i + r, eta_i)  -- Pólya-Gamma for logit
//   2. L_i ~ CRT(y_i, r)             -- Chinese Restaurant Table for dispersion
//
// After augmentation, the model becomes:
//   kappa_i = (y_i - r) / 2
//   kappa_i / omega_i | omega_i ~ N(eta_i, 1/omega_i)
//
// This enables conjugate Gibbs updates for beta, b, and r.
//
// ---------------------------------------------------------------------

// Pure C++ result structure (safe for OpenMP parallel regions)
struct PGNegBinResultCpp {
  std::vector<std::vector<double>> beta;       // [n_iter][n_beta]
  std::vector<std::vector<double>> re;         // [n_iter][n_re]
  std::vector<double> sigma_re;                // [n_iter]
  std::vector<double> r;                       // [n_iter] - dispersion
  std::vector<std::vector<double>> eta;        // [n_iter][n_obs] (optional)
  int chain_id;
};

// R-compatible result structure
struct PGNegBinGibbsResult {
  Rcpp::NumericMatrix beta;       // Fixed effects [n_iter, n_beta]
  Rcpp::NumericMatrix re;         // Random effects [n_iter, n_re]
  Rcpp::NumericVector sigma_re;   // RE standard deviation [n_iter]
  Rcpp::NumericVector r;          // Dispersion parameter [n_iter]
  Rcpp::NumericMatrix eta;        // Linear predictor [n_iter, n_obs]
  int chain_id;
};

// ---------------------------------------------------------------------
// Main Gibbs sampler for single negative binomial process
// ---------------------------------------------------------------------

// PG+CRT Gibbs sampler for single NB response
// @param y Integer vector of counts
// @param X Design matrix for fixed effects
// @param group Integer vector of group indices (1-based)
// @param n_groups Number of groups (0 if no random effects)
// @param n_iter Total MCMC iterations
// @param n_warmup Warmup iterations to discard
// @param thin Thinning interval
// @param prior_beta_sd Prior SD for fixed effects
// @param prior_sigma_scale Half-Cauchy scale for RE SD
// @param prior_r_shape Gamma shape for dispersion prior
// @param prior_r_rate Gamma rate for dispersion prior
// @param r_init Initial value for dispersion
// @param store_eta Store linear predictor draws
// @param verbose Print progress
// @param n_threads Number of threads (for linear algebra)
Rcpp::List pg_negbin_gibbs(
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
);

// ---------------------------------------------------------------------
// Gibbs sampler for ratio model: NB numerator / NB denominator
// (ratiod_negbin_negbin family)
// ---------------------------------------------------------------------

// PG+CRT Gibbs sampler for two-process NB ratio model
// Both numerator and denominator are NB distributed
// Shared random effects enter both linear predictors
//
// @param y_num Integer vector of numerator counts
// @param y_denom Integer vector of denominator counts
// @param X_num Design matrix for numerator
// @param X_denom Design matrix for denominator
// @param group Integer vector of group indices (1-based)
// @param n_groups Number of groups
// @param n_iter Total MCMC iterations
// @param n_warmup Warmup iterations
// @param thin Thinning interval
// @param prior_beta_sd Prior SD for fixed effects
// @param prior_sigma_scale Half-Cauchy scale for RE SD
// @param prior_r_shape Gamma shape for dispersion priors
// @param prior_r_rate Gamma rate for dispersion priors
// @param r_num_init Initial dispersion for numerator
// @param r_denom_init Initial dispersion for denominator
// @param shared Whether to use shared random effects
// @param store_eta Store linear predictor draws
// @param verbose Print progress
// @param n_threads Number of threads
Rcpp::List pg_negbin_negbin_gibbs(
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
);

// ---------------------------------------------------------------------
// Gibbs sampler for NB with spatial effects (ICAR)
// ---------------------------------------------------------------------

Rcpp::List pg_negbin_gibbs_spatial(
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
);

// ---------------------------------------------------------------------
// Helper functions for Gibbs updates
// ---------------------------------------------------------------------

// The beta and random-effect updates reuse the shared PG normal-normal
// kernels update_beta / update_re (declared in pg_shared.h).

// Update dispersion r with a Metropolis-Hastings step on log(r).
// Prior: r ~ Gamma(shape, rate). The likelihood uses the NB-as-logistic
// (Zhou) parameterization mu = r * exp(eta), so the second argument is the
// linear predictor eta, NOT a probability.
double update_r_negbin(
    const Rcpp::IntegerVector& y,
    const Rcpp::NumericVector& eta,  // linear predictor (mu = r * exp(eta))
    double r_current,
    double prior_shape,
    double prior_rate
);

// Update sigma_re with half-Cauchy prior
double update_sigma_re_negbin(
    const Rcpp::NumericVector& re,
    double scale
);

} // namespace tulpa

#endif // TULPA_PG_NEGBIN_H
