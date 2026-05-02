// tulpa_priors_hsgp.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_HSGP_H
#define TULPA_PRIORS_HSGP_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 5. HSGP spatial prior
// ============================================================================

template<typename T>
T compute_hsgp_spatial_prior(const std::vector<T>& params, const ModelData& data,
                              const ParamLayout& layout, std::vector<T>& hsgp_f)
{
    T log_post = T(0.0);

    if (layout.is_hsgp && data.has_hsgp) {
        T log_sigma2_hsgp = params[layout.log_sigma2_hsgp_idx];
        T log_lengthscale_hsgp = params[layout.log_lengthscale_hsgp_idx];
        T sigma2_hsgp = safe_exp(log_sigma2_hsgp);
        T lengthscale_hsgp = safe_exp(log_lengthscale_hsgp);

        int m_total = data.hsgp_data.m_total;

        // Extract beta coefficients
        std::vector<T> hsgp_beta(m_total);
        for (int j = 0; j < m_total; j++) {
            hsgp_beta[j] = params[layout.hsgp_beta_start + j];
        }

        // PC prior on sigma: P(sigma > 1) = 0.01 -> rate = 4.6
        // log p(sigma) = log(rate) - rate*sigma - log(2*sigma)
        T sigma_hsgp = safe_sqrt(sigma2_hsgp);
        T rate_sigma_hsgp = T(4.6);
        log_post = log_post + safe_log(rate_sigma_hsgp) - rate_sigma_hsgp * sigma_hsgp
                   - safe_log(T(2.0) * sigma_hsgp);
        log_post = log_post + log_sigma2_hsgp * T(0.5);  // Jacobian: d(sigma)/d(log_sigma2)

        // LogNormal(0, 1) prior on lengthscale
        // log p(ell) = -0.5 * log(ell)^2  (Jacobian cancels)
        log_post = log_post - T(0.5) * log_lengthscale_hsgp * log_lengthscale_hsgp;

        // N(0, I) prior on beta
        for (int j = 0; j < m_total; j++) {
            log_post = log_post - T(0.5) * hsgp_beta[j] * hsgp_beta[j];
        }

        // Evaluate HSGP spatial effect: f = Phi * (sqrt(S) .* beta)
        // Phi and eigenvalues are double (precomputed data), but sigma2/lengthscale/beta are T
        hsgp_f.resize(data.N, T(0.0));
        for (int j = 0; j < m_total; j++) {
            T S_j = sigma2_hsgp * T(std::sqrt(2.0 * M_PI)) * lengthscale_hsgp
                    * safe_exp(T(-0.5) * lengthscale_hsgp * lengthscale_hsgp
                               * T(data.hsgp_data.eigenvalues[j]));
            T scaled_beta_j = safe_sqrt(S_j) * hsgp_beta[j];
            for (int i = 0; i < data.N; i++) {
                hsgp_f[i] = hsgp_f[i]
                    + T(data.hsgp_data.phi_flat[i * m_total + j]) * scaled_beta_j;
            }
        }
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_HSGP_H
