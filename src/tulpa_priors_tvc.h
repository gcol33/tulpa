// tulpa_priors_tvc.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_TVC_H
#define TULPA_PRIORS_TVC_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_tvc.h"
#include "pc_prior.h"

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 9. TVC (Temporally-Varying Coefficients) prior
// ============================================================================

template<typename T>
T compute_tvc_prior(const std::vector<T>& params, const ModelData& data,
                     const ParamLayout& layout, std::vector<T>& tvc_eta)
{
    T log_post = T(0.0);

    if (layout.has_tvc && data.tvc_data.n_tvc > 0) {
        int n_tvc = data.tvc_data.n_tvc;
        int n_groups = data.tvc_data.n_groups;
        int n_times = data.tvc_data.n_times;
        int n_obs = data.tvc_data.n_obs;

        // Extract tau (precision) parameters
        std::vector<T> tvc_tau(n_tvc);
        for (int j = 0; j < n_tvc; j++) {
            T log_tau = params[layout.log_tau_tvc_start + j];
            tvc_tau[j] = safe_exp(log_tau);

            // PC prior on sigma = 1/sqrt(tau), on the sampled log-precision
            // scale. P(sigma > 1) = 0.01.
            log_post = log_post + log_prior_log_tau_pc(log_tau, 1.0, 0.01);
        }

        // Extract rho (AR1 correlation) parameters if AR1 structure
        std::vector<T> tvc_rho(n_tvc, T(0.0));
        if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
            for (int j = 0; j < n_tvc; j++) {
                T logit_rho = params[layout.logit_rho_tvc_start + j];
                // Map logit to (-1, 1): rho = 2*invlogit(logit) - 1
                T u = inv_logit(logit_rho);
                tvc_rho[j] = T(2.0) * u - T(1.0);

                // Uniform(-1, 1) prior on rho
                // Jacobian for logit((rho+1)/2) transform
                log_post = log_post + safe_log(u) + safe_log(T(1.0) - u);
            }
        }

        // Extract TVC values
        int n_tvc_params = n_groups * n_tvc * n_times;
        std::vector<T> tvc_w_flat(n_tvc_params);
        for (int k = 0; k < n_tvc_params; k++) {
            tvc_w_flat[k] = params[layout.tvc_w_start + k];
        }

        // TVC temporal prior (RW1, RW2, or AR1)
        log_post = log_post + tulpa_tvc::tvc_log_prior(
            tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho
        );

        // Soft sum-to-zero constraint for identifiability
        log_post = log_post + tulpa_tvc::tvc_sum_to_zero_penalty(
            tvc_w_flat, data.tvc_data
        );

        // Precompute TVC contribution to linear predictor
        tvc_eta.resize(n_obs, T(0.0));
        tulpa_tvc::compute_tvc_eta(tvc_w_flat, data.tvc_data, tvc_eta);
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_TVC_H
