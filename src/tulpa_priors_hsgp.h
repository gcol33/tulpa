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
#include "hsgp_spectral.h"
#include "pc_prior.h"
#include "spatial_range_prior.h"

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 5. HSGP spatial prior
// ============================================================================

template<typename T>
void evaluate_hsgp_field(const T* beta, const T& sigma2, const T& lengthscale,
                         const HSGPData& basis, std::vector<T>& out) {
    const int m_total = basis.m_total;
        out.assign(basis.n_obs, T(0.0));
        const HsgpKernel kernel =
            static_cast<HsgpKernel>(basis.kernel);
        for (int j = 0; j < m_total; j++) {
            T S_j = hsgp_spectral_density_2d(
                kernel, sigma2, lengthscale, basis.eigenvalues[j]);
            T scaled_beta_j = safe_sqrt(S_j) * beta[j];
            for (int i = 0; i < basis.n_obs; i++) {
                out[i] = out[i]
                    + T(basis.phi_flat[i * m_total + j]) * scaled_beta_j;
            }
        }
}

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

        // PC prior on sigma, on the sampled log-variance scale, at the
        // spec's own anchors: P(sigma > U) = alpha.
        log_post = log_post + log_prior_log_sigma2_pc(
            log_sigma2_hsgp, data.hsgp_sigma2_prior_U,
            data.hsgp_sigma2_prior_alpha);

        // PC prior on the lengthscale as a range in two dimensions,
        // P(ell < U) = alpha, the prior the nested route puts on the same
        // field's `lengthscale` axis; sampled on log(ell), hence the Jacobian.
        log_post = log_post + spatial_range_log_prior(
            log_lengthscale_hsgp, data.hsgp_range_prior, data.hsgp_lengthscale_prior_U,
            data.hsgp_lengthscale_prior_alpha);

        // N(0, I) prior on beta
        for (int j = 0; j < m_total; j++) {
            log_post = log_post - T(0.5) * hsgp_beta[j] * hsgp_beta[j];
        }

        // Evaluate HSGP spatial effect: f = Phi * (sqrt(S) .* beta)
        // Phi and eigenvalues are double (precomputed data), but sigma2/lengthscale/beta are T
        evaluate_hsgp_field(hsgp_beta.data(), sigma2_hsgp, lengthscale_hsgp,
                            data.hsgp_data, hsgp_f);
    }

    return log_post;
}

// The constant compute_hsgp_spatial_prior leaves out: -0.5 log(2 pi) for each
// N(0, 1) basis coefficient (the PC priors on sigma and on the lengthscale are
// normalized).
inline double hsgp_spatial_prior_log_normalizer(const ModelData& data,
                                                const ParamLayout& layout) {
    if (!(layout.is_hsgp && data.has_hsgp)) return 0.0;
    return data.hsgp_data.m_total * log_prior_normal_log_normalizer(1.0)
        + spatial_range_prior_log_normalizer(data.hsgp_range_prior);
}

} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_HSGP_H
