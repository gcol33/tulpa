// tulpa_priors_gp.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_GP_H
#define TULPA_PRIORS_GP_H

#include <Rcpp.h>
#include <cstddef>
#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "pc_prior.h"
#include "spatial_range_prior.h"
#include "hmc_gp_autodiff.h"
#include <limits>
#include <string>

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

        // PC prior on the range + Jacobian for log transform. phi is sampled
        // unconstrained on the log scale: the PC density is proper on
        // (0, inf) and penalizes short ranges, so it needs no bounding box.
        log_post = log_post + spatial_range_log_prior(
            log_phi_gp, data.gp_range_prior, data.gp_phi_prior_U, data.gp_phi_prior_alpha);

        int n_gp = layout.gp_w_end - layout.gp_w_start;

        if (data.gp_parameterization == 1) {
            // Non-centered: params[gp_w_start ..] are z ~ N(0, I). The field
            // w = f(z, sigma2, phi) and the likelihood at w are reconstructed by
            // apply_gp_nc_transform_* in initialize_generic_state; here we add
            // only the z prior. This avoids Neal's funnel between the field and
            // (sigma2, phi) that collapses the field amplitude under a diagonal
            // mass matrix in the centered parameterization.
            //
            // No z -> w Jacobian: z is a genuine auxiliary with its own N(0, I)
            // prior, and w is a deterministic function of (z, theta), so the
            // (w, theta) posterior is unchanged and the target is
            // -0.5 z'z + loglik(f(z, theta)). Adding log|dw/dz| would inflate
            // the amplitude by sigma^N -- the exact defect this fix removes.
            // The temporal GP path documents the same reasoning.
            for (int k = 0; k < n_gp; k++) {
                T zk = params[layout.gp_w_start + k];
                log_post = log_post - T(0.5) * zk * zk;
            }
            gp_w.clear();  // filled by apply_gp_nc_transform_* on w
        } else {
            // Centered: params[gp_w_start ..] ARE the field w ~ NNGP(0, sigma2,
            // phi), and the NNGP density is placed on them directly.

            // Extract GP spatial effects w[0..n_gp-1]
            gp_w.resize(n_gp);
            for (int k = 0; k < n_gp; k++) {
                gp_w[k] = params[layout.gp_w_start + k];
            }

            // NNGP log-likelihood on spatial effects
            log_post = log_post + tulpa_gp::gp_nngp_log_lik_t(
                gp_w, sigma2_gp, phi_gp, data.gp_data);
        }
    }

    return log_post;
}

// The constant compute_gp_spatial_prior leaves out, or NaN with the reason
// appended to `declined`. The PC priors and the NNGP density on a centered
// field are normalized; a non-centered field's N(0, I) z omits -0.5 log(2 pi)
// per coordinate. A restricted (RSR) field evaluates the NNGP density at the
// projected field, which is not a density on the sampled one.
inline double gp_spatial_prior_log_normalizer(const ModelData& data,
                                              const ParamLayout& layout,
                                              std::vector<std::string>* declined) {
    if (!(layout.is_gp && data.has_gp)) return 0.0;
    const double c = spatial_range_prior_log_normalizer(data.gp_range_prior);
    return c + (data.gp_parameterization == 1
        ? (layout.gp_w_end - layout.gp_w_start) * log_prior_normal_log_normalizer(1.0)
        : 0.0);
}

} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_GP_H
