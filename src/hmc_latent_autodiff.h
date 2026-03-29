// hmc_latent_autodiff.h
// Templated latent factor functions for autodiff (A_t mode)
// Works with both double and ad::Var for automatic differentiation

#ifndef TULPA_HMC_LATENT_AUTODIFF_H
#define TULPA_HMC_LATENT_AUTODIFF_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_latent.h"

namespace tulpa_latent_ad {

using tulpa_latent::LatentConstraint;
using tulpa_latent::LatentFactorData;
using namespace tulpa::math;

// =============================================================================
// Templated constraint application
// =============================================================================

// Apply sum-to-zero constraint to factors
// Input: factors[n_obs * n_factors] (flattened, row-major: obs x factors)
template<typename T>
void apply_sum_to_zero(std::vector<T>& factors, int n_obs, int n_factors) {
    for (int k = 0; k < n_factors; ++k) {
        T sum = T(0.0);
        for (int i = 0; i < n_obs; ++i) {
            sum = sum + factors[i * n_factors + k];
        }
        T mean = sum / T(n_obs);
        for (int i = 0; i < n_obs; ++i) {
            factors[i * n_factors + k] = factors[i * n_factors + k] - mean;
        }
    }
}

// Apply first-zero constraint to factors
template<typename T>
void apply_first_zero(std::vector<T>& factors, int n_obs, int n_factors) {
    for (int k = 0; k < n_factors; ++k) {
        T first_val = factors[k];  // factors[0, k]
        for (int i = 0; i < n_obs; ++i) {
            factors[i * n_factors + k] = factors[i * n_factors + k] - first_val;
        }
    }
}

// =============================================================================
// Templated log-prior functions
// =============================================================================

// Log prior for factor sigmas (PC prior = exponential on sigma)
// sigma_k ~ Exponential(rate)
// log_sigma_k is the log of sigma_k
// p(log_sigma) = rate * exp(log_sigma) * exp(-rate * exp(log_sigma))
// log p(log_sigma) = log(rate) + log_sigma - rate * exp(log_sigma)
template<typename T>
T latent_sigma_log_prior(const std::vector<T>& log_sigma, double rate) {
    T lp = T(0.0);
    int n_factors = log_sigma.size();

    for (int k = 0; k < n_factors; ++k) {
        T sigma_k = safe_exp(log_sigma[k]);
        // Exponential prior on sigma with Jacobian
        lp = lp + T(std::log(rate)) + log_sigma[k] - T(rate) * sigma_k;
    }
    return lp;
}

// Log prior for factor scores (standard normal, with constraint)
// f_ik ~ N(0, 1) with constraint that removes one degree of freedom per factor
template<typename T>
T latent_factor_log_prior(const std::vector<T>& factors,
                          int n_obs, int n_factors,
                          LatentConstraint constraint) {
    T lp = T(0.0);

    // Standard normal prior on factors
    // For sum-to-zero: n_obs - 1 free parameters per factor
    // For first-zero: n_obs - 1 free parameters per factor (first is 0)
    int n_free = (constraint == LatentConstraint::FIRST_ZERO) ? n_obs - 1 : n_obs;

    for (int k = 0; k < n_factors; ++k) {
        for (int i = 0; i < n_free; ++i) {
            int idx = (constraint == LatentConstraint::FIRST_ZERO) ? (i + 1) : i;
            T f_ik = factors[idx * n_factors + k];
            lp = lp - T(0.5) * f_ik * f_ik;  // N(0,1) log density (ignoring constant)
        }
    }

    return lp;
}

// Full log prior for latent factor model
template<typename T>
T latent_full_log_prior(const std::vector<T>& log_sigma,
                        const std::vector<T>& factors,
                        int n_obs, int n_factors,
                        LatentConstraint constraint,
                        double sigma_prior_rate) {
    T lp = T(0.0);

    // Sigma prior
    lp = lp + latent_sigma_log_prior(log_sigma, sigma_prior_rate);

    // Factor prior
    lp = lp + latent_factor_log_prior(factors, n_obs, n_factors, constraint);

    return lp;
}

// =============================================================================
// Templated linear predictor contribution
// =============================================================================

// Compute the latent factor contribution to linear predictor for observation i
// Returns: sum_k f[i,k] * sigma[k]
template<typename T>
T latent_contribution(int i,
                      const std::vector<T>& factors,
                      const std::vector<T>& sigma,
                      int n_factors) {
    T contrib = T(0.0);
    for (int k = 0; k < n_factors; ++k) {
        contrib = contrib + factors[i * n_factors + k] * sigma[k];
    }
    return contrib;
}

// Compute latent contributions for all observations
template<typename T>
void latent_contributions_all(std::vector<T>& contrib,
                              const std::vector<T>& factors,
                              const std::vector<T>& sigma,
                              int n_obs, int n_factors) {
    contrib.resize(n_obs);
    for (int i = 0; i < n_obs; ++i) {
        contrib[i] = latent_contribution(i, factors, sigma, n_factors);
    }
}

// =============================================================================
// Templated parameter extraction
// =============================================================================

// Extract sigma from log_sigma
template<typename T>
void extract_sigma(std::vector<T>& sigma, const std::vector<T>& log_sigma) {
    int n = log_sigma.size();
    sigma.resize(n);
    for (int k = 0; k < n; ++k) {
        sigma[k] = safe_exp(log_sigma[k]);
    }
}

// Extract and apply constraint to factors from parameter vector
template<typename T>
void extract_latent_params(std::vector<T>& log_sigma,
                           std::vector<T>& factors,
                           const std::vector<T>& params,
                           int param_start,
                           int n_factors, int n_obs,
                           int constraint_type) {
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
    if (constraint_type == 0) {  // SUM_TO_ZERO
        apply_sum_to_zero(factors, n_obs, n_factors);
    } else {  // FIRST_ZERO
        apply_first_zero(factors, n_obs, n_factors);
    }
}

}  // namespace tulpa_latent_ad

#endif  // TULPA_HMC_LATENT_AUTODIFF_H
