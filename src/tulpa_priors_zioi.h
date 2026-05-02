// tulpa_priors_zioi.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_ZIOI_H
#define TULPA_PRIORS_ZIOI_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 12. Zero-inflation / One-inflation prior
// ============================================================================

template<typename T>
T compute_zi_oi_prior(const std::vector<T>& params, const ModelData& data,
                       const ParamLayout& layout,
                       std::vector<T>& beta_zi, std::vector<T>& beta_oi)
{
    T log_post = T(0.0);

    if (layout.has_zi && data.p_zi > 0) {
        beta_zi.resize(data.p_zi);
        for (int j = 0; j < data.p_zi; j++) {
            beta_zi[j] = params[layout.beta_zi_start + j];
        }
        // Prior on beta_zi: N(0, zi_prior_sd^2)
        double tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd);
        for (int j = 0; j < data.p_zi; j++) {
            log_post = log_post + log_prior_normal(beta_zi[j], tau_zi);
        }
    }

    if (layout.has_oi && data.p_oi > 0) {
        beta_oi.resize(data.p_oi);
        for (int j = 0; j < data.p_oi; j++) {
            beta_oi[j] = params[layout.beta_oi_start + j];
        }
        // Prior on beta_oi: N(0, oi_prior_sd^2)
        double tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd);
        for (int j = 0; j < data.p_oi; j++) {
            log_post = log_post + log_prior_normal(beta_oi[j], tau_oi);
        }
    }

    return log_post;
}

} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_ZIOI_H
