// tulpa_priors_latent.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_LATENT_H
#define TULPA_PRIORS_LATENT_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_latent_autodiff.h"

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 10. Latent factors prior
// ============================================================================

template<typename T>
T compute_latent_prior(const std::vector<T>& params, const ModelData& data,
                        const ParamLayout& layout, std::vector<T>& latent_eta)
{
    T log_post = T(0.0);

    if (layout.has_latent && data.latent_n_factors > 0) {
        int K = data.latent_n_factors;
        int N = data.N;

        // Extract log_sigma parameters
        std::vector<T> log_sigma_latent(K);
        for (int k = 0; k < K; k++) {
            log_sigma_latent[k] = params[layout.log_sigma_latent_start + k];
        }

        // Compute sigma from log_sigma
        std::vector<T> latent_sigma(K);
        tulpa_latent_ad::extract_sigma(latent_sigma, log_sigma_latent);

        // Extract factors and apply constraint
        int n_factor_params = N * K;
        std::vector<T> latent_factors(n_factor_params);
        for (int j = 0; j < n_factor_params; j++) {
            latent_factors[j] = params[layout.latent_factor_start + j];
        }

        // Apply identifiability constraint
        if (data.latent_constraint == 0) {  // SUM_TO_ZERO
            tulpa_latent_ad::apply_sum_to_zero(latent_factors, N, K);
        } else {  // FIRST_ZERO
            tulpa_latent_ad::apply_first_zero(latent_factors, N, K);
        }

        // Sigma prior: Exponential on sigma (PC prior)
        log_post = log_post + tulpa_latent_ad::latent_sigma_log_prior(
            log_sigma_latent, data.latent_sigma_prior_rate
        );

        // Factor prior: N(0, 1) on each factor score
        tulpa_latent::LatentConstraint constraint =
            (data.latent_constraint == 0) ? tulpa_latent::LatentConstraint::SUM_TO_ZERO
                                          : tulpa_latent::LatentConstraint::FIRST_ZERO;
        log_post = log_post + tulpa_latent_ad::latent_factor_log_prior(
            latent_factors, N, K, constraint
        );

        // Precompute latent factor contribution to linear predictor
        latent_eta.resize(N, T(0.0));
        tulpa_latent_ad::latent_contributions_all(latent_eta, latent_factors, latent_sigma, N, K);
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_LATENT_H
