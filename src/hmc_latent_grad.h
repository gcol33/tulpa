// hmc_latent_grad.h
// Hand-coded gradients for latent factors
// Provides O(N*K) analytical gradients for latent factor models

#ifndef TULPA_HMC_LATENT_GRAD_H
#define TULPA_HMC_LATENT_GRAD_H

#include <vector>
#include <cmath>
#include "hmc_latent.h"

namespace tulpa_latent {

// Structure to hold latent factor gradient results
struct LatentGradients {
    std::vector<double> grad_log_sigma;  // Gradient w.r.t. log(sigma_k), length K
    std::vector<double> grad_factors;     // Gradient w.r.t. factors, length N*K
};

// =============================================================================
// Prior gradients
// =============================================================================

// Gradient of sigma prior w.r.t. log_sigma
// Prior: sigma ~ Exponential(rate)
// log p(log_sigma) = log(rate) + log_sigma - rate * sigma
// d/d(log_sigma) = 1 - rate * sigma
inline void sigma_prior_gradient(
    const std::vector<double>& log_sigma,
    double rate,
    std::vector<double>& grad_log_sigma
) {
    int K = log_sigma.size();
    grad_log_sigma.resize(K);
    for (int k = 0; k < K; k++) {
        double sigma_k = std::exp(log_sigma[k]);
        grad_log_sigma[k] = 1.0 - rate * sigma_k;
    }
}

// Gradient of factor prior w.r.t. factors
// Prior: f_ik ~ N(0, 1)
// log p(f) = -0.5 * f^2 (ignoring constant)
// d/d(f) = -f
// With constraint: skip constrained elements
inline void factor_prior_gradient(
    const std::vector<double>& factors,
    int n_obs, int n_factors,
    LatentConstraint constraint,
    std::vector<double>& grad_factors
) {
    int n_total = n_obs * n_factors;
    grad_factors.assign(n_total, 0.0);

    // For first-zero constraint, skip index 0 for each factor
    // For sum-to-zero, all factors contribute but adjusted by constraint later
    int start_idx = (constraint == LatentConstraint::FIRST_ZERO) ? 1 : 0;

    for (int k = 0; k < n_factors; k++) {
        for (int i = start_idx; i < n_obs; i++) {
            int idx = i * n_factors + k;
            grad_factors[idx] = -factors[idx];
        }
    }
}

// =============================================================================
// Constraint gradient adjustment
// =============================================================================

// For sum-to-zero constraint:
// Constrained: f'[i,k] = f[i,k] - mean(f[*,k])
// df'[i,k]/df[j,k] = 1 if i=j, else -1/N
// Chain rule: d L/d f[j,k] = sum_i d L/d f'[i,k] * df'[i,k]/df[j,k]
//           = sum_i (d L/d f'[i,k] * (delta_ij - 1/N))
//           = d L/d f'[j,k] - (1/N) * sum_i d L/d f'[i,k]
inline void adjust_gradient_sum_to_zero(
    std::vector<double>& grad_factors,
    int n_obs, int n_factors
) {
    for (int k = 0; k < n_factors; k++) {
        // Compute mean gradient for this factor
        double sum_grad = 0.0;
        for (int i = 0; i < n_obs; i++) {
            sum_grad += grad_factors[i * n_factors + k];
        }
        double mean_grad = sum_grad / n_obs;

        // Adjust each gradient
        for (int i = 0; i < n_obs; i++) {
            grad_factors[i * n_factors + k] -= mean_grad;
        }
    }
}

// For first-zero constraint:
// f'[0,k] = 0, f'[i,k] = f[i,k] - f[0,k] for i > 0
// df'[i,k]/df[0,k] = -1 for i > 0, 0 for i = 0
// df'[i,k]/df[j,k] = 1 if i=j and j > 0, else 0
// Chain rule: d L/d f[0,k] = sum_{i>0} d L/d f'[i,k] * (-1) = -sum_{i>0} grad[i,k]
//             d L/d f[j,k] = d L/d f'[j,k] for j > 0
inline void adjust_gradient_first_zero(
    std::vector<double>& grad_factors,
    int n_obs, int n_factors
) {
    for (int k = 0; k < n_factors; k++) {
        // Gradient for f[0,k] = -sum of gradients for i > 0
        double sum_grad = 0.0;
        for (int i = 1; i < n_obs; i++) {
            sum_grad += grad_factors[i * n_factors + k];
        }
        grad_factors[k] = -sum_grad;  // f[0,k]
    }
}

// =============================================================================
// Full prior gradient computation
// =============================================================================

inline void latent_prior_gradients(
    const std::vector<double>& log_sigma,
    const std::vector<double>& factors,
    int n_obs, int n_factors,
    LatentConstraint constraint,
    double sigma_prior_rate,
    LatentGradients& grads
) {
    // Sigma prior gradients
    sigma_prior_gradient(log_sigma, sigma_prior_rate, grads.grad_log_sigma);

    // Factor prior gradients
    factor_prior_gradient(factors, n_obs, n_factors, constraint, grads.grad_factors);

    // Note: constraint adjustment for factors is applied after likelihood gradients
}

// =============================================================================
// Likelihood gradient contribution
// =============================================================================

// Compute gradient contributions from likelihood through eta_latent
// eta_latent[i] = sum_k f[i,k] * sigma[k]
// Given dLL/d(eta_latent[i]), compute gradients w.r.t. sigma and factors
inline void latent_likelihood_gradients(
    const std::vector<double>& dLL_deta,  // Length N (gradient from likelihood)
    const std::vector<double>& factors,    // Length N*K (after constraint)
    const std::vector<double>& sigma,      // Length K
    int n_obs, int n_factors,
    bool shared,                            // If true, effect enters both num and denom
    std::vector<double>& grad_log_sigma,   // Output: add to this
    std::vector<double>& grad_factors      // Output: add to this
) {
    // dLL/d(log_sigma_k) = dLL/d(sigma_k) * d(sigma_k)/d(log_sigma_k)
    //                    = (sum_i dLL_deta[i] * f[i,k]) * sigma_k
    for (int k = 0; k < n_factors; k++) {
        double sum_f_dLL = 0.0;
        for (int i = 0; i < n_obs; i++) {
            sum_f_dLL += dLL_deta[i] * factors[i * n_factors + k];
        }
        grad_log_sigma[k] += sum_f_dLL * sigma[k];
    }

    // dLL/d(f[i,k]) = dLL_deta[i] * sigma[k]
    for (int i = 0; i < n_obs; i++) {
        for (int k = 0; k < n_factors; k++) {
            grad_factors[i * n_factors + k] += dLL_deta[i] * sigma[k];
        }
    }
}

}  // namespace tulpa_latent

#endif  // TULPA_HMC_LATENT_GRAD_H
