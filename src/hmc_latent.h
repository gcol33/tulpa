// hmc_latent.h
// Latent factors for unmeasured confounders
// Provides observation-level random effects shared between numerator and denominator

#ifndef TULPA_HMC_LATENT_H
#define TULPA_HMC_LATENT_H

#include <Rcpp.h>
#include <vector>
#include <cmath>

namespace tulpa_latent {

// =====================================================================
// Data structures
// =====================================================================

enum class LatentConstraint { SUM_TO_ZERO, FIRST_ZERO };

struct LatentFactorData {
  int n_factors;                    // Number of latent factors (K)
  int n_obs;                        // Number of observations (N)
  bool shared;                      // Whether factors enter both num and denom
  bool scale;                       // Whether to standardize factors
  LatentConstraint constraint;      // Identifiability constraint
  double sigma_prior_rate;          // Exponential rate for PC prior on sigma
};

// =====================================================================
// Helper functions
// =====================================================================

// Apply sum-to-zero constraint to factors
// Input: factors[n_obs * n_factors] (flattened, row-major: obs x factors)
// Modifies in place
inline void apply_sum_to_zero(std::vector<double>& factors,
                               int n_obs, int n_factors) {
  for (int k = 0; k < n_factors; ++k) {
    double sum = 0.0;
    for (int i = 0; i < n_obs; ++i) {
      sum += factors[i * n_factors + k];
    }
    double mean = sum / n_obs;
    for (int i = 0; i < n_obs; ++i) {
      factors[i * n_factors + k] -= mean;
    }
  }
}

// Apply first-zero constraint to factors
inline void apply_first_zero(std::vector<double>& factors,
                              int n_obs, int n_factors) {
  for (int k = 0; k < n_factors; ++k) {
    double first_val = factors[k];  // factors[0, k]
    for (int i = 0; i < n_obs; ++i) {
      factors[i * n_factors + k] -= first_val;
    }
  }
}

// Compute variance of a factor column (for scaling)
inline double factor_variance(const std::vector<double>& factors,
                               int n_obs, int n_factors, int k) {
  double mean = 0.0;
  for (int i = 0; i < n_obs; ++i) {
    mean += factors[i * n_factors + k];
  }
  mean /= n_obs;

  double var = 0.0;
  for (int i = 0; i < n_obs; ++i) {
    double diff = factors[i * n_factors + k] - mean;
    var += diff * diff;
  }
  return var / (n_obs - 1);  // Sample variance
}

// =====================================================================
// Log-likelihood contributions
// =====================================================================

// Log prior for factor sigmas (PC prior = exponential on sigma)
// sigma_k ~ Exponential(rate)
// log_sigma_k is the log of sigma_k
// p(log_sigma) = rate * exp(log_sigma) * exp(-rate * exp(log_sigma))
// log p(log_sigma) = log(rate) + log_sigma - rate * exp(log_sigma)
inline double latent_sigma_log_prior(const std::vector<double>& log_sigma,
                                      double rate) {
  double lp = 0.0;
  int n_factors = log_sigma.size();

  for (int k = 0; k < n_factors; ++k) {
    double sigma_k = std::exp(log_sigma[k]);
    // Exponential prior on sigma with Jacobian
    lp += std::log(rate) + log_sigma[k] - rate * sigma_k;
  }
  return lp;
}

// Log prior for factor scores (standard normal, with constraint)
// f_ik ~ N(0, 1) with constraint that removes one degree of freedom per factor
// For sum-to-zero: one observation per factor is determined by others
// For first-zero: first observation is fixed at 0
inline double latent_factor_log_prior(const std::vector<double>& factors,
                                       int n_obs, int n_factors,
                                       LatentConstraint constraint) {
  double lp = 0.0;

  // Standard normal prior on factors
  // For sum-to-zero: n_obs - 1 free parameters per factor
  // For first-zero: n_obs - 1 free parameters per factor (first is 0)

  int n_free = (constraint == LatentConstraint::FIRST_ZERO) ? n_obs - 1 : n_obs;

  for (int k = 0; k < n_factors; ++k) {
    for (int i = 0; i < n_free; ++i) {
      int idx = (constraint == LatentConstraint::FIRST_ZERO) ? (i + 1) : i;
      double f_ik = factors[idx * n_factors + k];
      lp += -0.5 * f_ik * f_ik;  // N(0,1) log density (ignoring constant)
    }
  }

  return lp;
}

// Full log prior for latent factor model
inline double latent_full_log_prior(const std::vector<double>& log_sigma,
                                     const std::vector<double>& factors,
                                     const LatentFactorData& lf_data) {
  double lp = 0.0;

  // Sigma prior
  lp += latent_sigma_log_prior(log_sigma, lf_data.sigma_prior_rate);

  // Factor prior
  lp += latent_factor_log_prior(factors, lf_data.n_obs, lf_data.n_factors,
                                 lf_data.constraint);

  return lp;
}

// =====================================================================
// Linear predictor contribution
// =====================================================================

// Compute the latent factor contribution to linear predictor for observation i
// Returns: sum_k f[i,k] * sigma[k]
inline double latent_contribution(int i,
                                   const std::vector<double>& factors,
                                   const std::vector<double>& sigma,
                                   int n_factors) {
  double contrib = 0.0;
  for (int k = 0; k < n_factors; ++k) {
    contrib += factors[i * n_factors + k] * sigma[k];
  }
  return contrib;
}

// Compute latent contributions for all observations
// Output: contrib[n_obs]
inline void latent_contributions_all(std::vector<double>& contrib,
                                      const std::vector<double>& factors,
                                      const std::vector<double>& sigma,
                                      int n_obs, int n_factors) {
  contrib.resize(n_obs);
  for (int i = 0; i < n_obs; ++i) {
    contrib[i] = latent_contribution(i, factors, sigma, n_factors);
  }
}

// =====================================================================
// Parameter extraction helpers
// =====================================================================

// Extract sigma from log_sigma
inline void extract_sigma(std::vector<double>& sigma,
                           const std::vector<double>& log_sigma) {
  int n = log_sigma.size();
  sigma.resize(n);
  for (int k = 0; k < n; ++k) {
    sigma[k] = std::exp(log_sigma[k]);
  }
}

// Extract and apply constraint to factors from parameter vector
// params layout: [log_sigma_1, ..., log_sigma_K, f_11, f_12, ..., f_NK]
inline void extract_latent_params(std::vector<double>& log_sigma,
                                   std::vector<double>& factors,
                                   const double* params,
                                   int param_start,
                                   const LatentFactorData& lf_data) {
  int n_factors = lf_data.n_factors;
  int n_obs = lf_data.n_obs;

  // Extract log_sigma
  log_sigma.resize(n_factors);
  for (int k = 0; k < n_factors; ++k) {
    log_sigma[k] = params[param_start + k];
  }

  // Extract factors
  int n_factor_params = n_obs * n_factors;
  factors.resize(n_factor_params);
  for (int j = 0; j < n_factor_params; ++j) {
    factors[j] = params[param_start + n_factors + j];
  }

  // Apply constraint
  if (lf_data.constraint == LatentConstraint::SUM_TO_ZERO) {
    apply_sum_to_zero(factors, n_obs, n_factors);
  } else if (lf_data.constraint == LatentConstraint::FIRST_ZERO) {
    apply_first_zero(factors, n_obs, n_factors);
  }
}

// =====================================================================
// Initialization
// =====================================================================

// Initialize latent factor parameters
// Returns vector: [log_sigma_1, ..., log_sigma_K, f_11, ..., f_NK]
inline std::vector<double> initialize_latent_params(
    const LatentFactorData& lf_data,
    std::mt19937& rng) {

  int n_factors = lf_data.n_factors;
  int n_obs = lf_data.n_obs;
  int n_params = n_factors + n_obs * n_factors;

  std::vector<double> params(n_params);
  std::normal_distribution<double> norm(0.0, 0.1);

  // Initialize log_sigma (small positive values)
  for (int k = 0; k < n_factors; ++k) {
    params[k] = -1.0;  // sigma ~ 0.37
  }

  // Initialize factors (small values, constraint will be applied at extraction)
  for (int j = n_factors; j < n_params; ++j) {
    params[j] = norm(rng);
  }

  return params;
}

}  // namespace tulpa_latent

#endif  // TULPA_HMC_LATENT_H
