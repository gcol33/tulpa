// tulpa_priors_gp.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_GP_H
#define TULPA_PRIORS_GP_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_gp_autodiff.h"

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 3. GP spatial prior
// ============================================================================

template<typename T>
T compute_gp_spatial_prior(const std::vector<T>& params, const ModelData& data,
                            const ParamLayout& layout, std::vector<T>& gp_w)
{
    T log_post = T(0.0);

    if (layout.is_gp && data.has_gp) {
        // Extract hyperparameters from log-scale
        T log_sigma2_gp = params[layout.log_sigma2_gp_idx];
        T log_phi_gp = params[layout.log_phi_gp_idx];
        T sigma2_gp = safe_exp(log_sigma2_gp);
        T phi_gp = safe_exp(log_phi_gp);

        // PC prior on sigma2 + Jacobian for log transform
        log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
            sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        log_post = log_post + log_sigma2_gp;  // Jacobian

        // Uniform prior on phi within bounds + Jacobian
        double phi_val = get_value(phi_gp);
        if (phi_val < data.gp_phi_prior_lower || phi_val > data.gp_phi_prior_upper) {
            return T(-INFINITY);
        }
        log_post = log_post + tulpa_gp::log_prior_phi_uniform_t(
            phi_gp, data.gp_phi_prior_lower, data.gp_phi_prior_upper);
        log_post = log_post + log_phi_gp;  // Jacobian

        // Extract GP spatial effects w[0..n_gp-1]
        int n_gp = layout.gp_w_end - layout.gp_w_start;
        gp_w.resize(n_gp);
        for (int k = 0; k < n_gp; k++) {
            gp_w[k] = params[layout.gp_w_start + k];
        }

        // Apply RSR projection if enabled
        if (data.has_rsr && !data.rsr_projection.empty()) {
            std::vector<T> w_projected(data.rsr_n, T(0.0));
            for (int ii = 0; ii < data.rsr_n; ii++) {
                for (int jj = 0; jj < data.rsr_n; jj++) {
                    w_projected[ii] = w_projected[ii]
                        + T(data.rsr_projection[ii * data.rsr_n + jj]) * gp_w[jj];
                }
            }
            gp_w = w_projected;
        }

        // NNGP log-likelihood on spatial effects
        log_post = log_post + tulpa_gp::gp_nngp_log_lik_t(
            gp_w, sigma2_gp, phi_gp, data.gp_data);
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_GP_H
