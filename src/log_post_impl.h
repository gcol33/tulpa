// log_post_impl.h
// Templated compute_log_post that works with both double and ad::Var
// Single implementation for both evaluation and autodiff gradients
//
// NOTE: This header must be included AFTER hmc_sampler.h which defines
// ModelData, ParamLayout, ModelType, TemporalType
//
// Body of compute_log_post_impl<T> is split into per-section fragments
// included directly inside the function body. The fragments are NOT
// standalone-compilable — they reference locals (params, data, layout,
// log_post, T, beta_*, phi_*, re_vals, re_term_offsets, ...) defined
// in the enclosing function scope. Each fragment is included exactly
// once per umbrella inclusion, so they have no header guards.

#ifndef TULPA_LOG_POST_IMPL_H
#define TULPA_LOG_POST_IMPL_H

#include <vector>
#include "autodiff_utils.h"
#include "hmc_gp_autodiff.h"  // Templated GP/NNGP functions
#include "hmc_svc_autodiff.h"  // Templated SVC functions
#include "hmc_tvc_autodiff.h"  // Templated TVC functions
#include "hmc_latent_autodiff.h"  // Templated latent factor functions
#include "hmc_temporal_multiscale_autodiff.h"  // Templated multiscale temporal functions
#include "tulpa_priors.h"  // Shared prior computation helpers
#include "linalg_fast.h"  // tulpa_linalg::matvec for the double fast path
#include "tulpa/likelihood.h"  // LikelihoodSpec for generic dispatch helpers
#include "log_post_car_proper_det.h"  // self-contained: tulpa::car_proper_log_det_t<T>
#include <type_traits>     // std::is_same_v for the constexpr dispatch

// Expects these to be defined by including hmc_sampler.h first:
// - tulpa_hmc::ModelData
// - tulpa_hmc::ParamLayout
// - tulpa_hmc::ModelType
// - tulpa_hmc::TemporalType

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;
using tulpa_hmc::ModelType;
using tulpa_hmc::TemporalType;
using tulpa_zi::ZIType;
using tulpa_spatiotemporal::STType;

namespace tulpa {

using namespace math;

// ============================================================================
// Templated log-posterior computation
// T = double for evaluation, T = ad::Var for autodiff gradients
// ============================================================================

template<typename T>
T compute_log_post_impl(
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout
) {
    // Collapsed ICAR/BYM2 spatial inner state (phi*, theta*) is marginalized
    // via Newton + Laplace inside compute_log_post — not differentiable
    // through autodiff. The layout omits spatial_start/theta_bym2_start
    // (both = -1), so the dereferences below would read params[-1].
    // For T = double, defer to the canonical evaluator; this is the path
    // taken by compute_gradient_numerical_impl during gradient_check_only.
    // For autodiff T, resolve_gradient_fn redirects collapsed requests to
    // numerical/H, so this branch should never execute — return T(0) as
    // a defensive no-op rather than crashing.
    if (data.icar_collapsed || data.bym2_collapsed) {
        if constexpr (std::is_same_v<T, double>) {
            return tulpa_hmc::compute_log_post(params, data, layout);
        } else {
            return T(0.0);
        }
    }

    T log_post = T(0.0);

    // ========================================================================
    // Extract parameters
    // ========================================================================

    // Fixed effects — pointer views into params (no copy, read-only access)
    const T* beta_num = &params[layout.legacy.beta_num_start];
    const T* beta_denom = &params[layout.legacy.beta_denom_start];

    // Random effects: supports simple intercept, crossed RE, random slopes,
    // and correlated random slopes with tanh-Cholesky parameterization.
    // Pre-computed transformed RE values for use in observation loop.
    std::vector<T> re_vals;          // Flat storage: [term_offset + g*n_coefs + c]
    std::vector<int> re_term_offsets; // Start offset for each term in re_vals

    // Overdispersion (phi for Gamma denominator in poisson_gamma)
    T phi_num = T(1.0);
    if (layout.legacy.has_phi_num) {
        T log_phi = params[layout.legacy.log_phi_num_idx];
        phi_num = safe_exp(log_phi);
    }

    T phi_denom = T(1.0);
    if (layout.legacy.has_phi_denom) {
        T log_phi = params[layout.legacy.log_phi_denom_idx];
        phi_denom = safe_exp(log_phi);
    }

    // Spatial effects — pointer views into params (no copy, read-only access)
    const T* phi_spatial = nullptr;
    T tau_spatial = T(1.0);
    T sigma_s_bym2 = T(1.0);
    T sigma_u_bym2 = T(1.0);
    const T* theta_bym2 = nullptr;

    if (layout.has_spatial) {
        phi_spatial = &params[layout.spatial_start];

        if (layout.is_bym2) {
            // Riebler reparameterization: sigma_total, rho -> sigma_s, sigma_u
            T sigma_total_bym2 = safe_exp(params[layout.log_sigma_bym2_idx]);
            T logit_rho_val = params[layout.logit_rho_bym2_idx];
            T rho_bym2 = T(1.0) / (T(1.0) + safe_exp(-logit_rho_val));
            sigma_s_bym2 = sigma_total_bym2 * sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total_bym2 * sqrt(T(1.0) - rho_bym2);

            theta_bym2 = &params[layout.theta_bym2_start];
        } else {
            T log_tau = params[layout.log_tau_spatial_idx];
            tau_spatial = safe_exp(log_tau);
        }
    }

    // Temporal effects
    std::vector<T> phi_temporal;
    T tau_temporal = T(1.0);
    T rho_ar1 = T(0.5);
    T sigma2_temporal_gp = T(1.0);
    T phi_temporal_gp = T(1.0);  // lengthscale

    if (layout.has_temporal) {
        // Extract temporal effects (common to all temporal types)
        int n_temporal = layout.temporal_end - layout.temporal_start;
        phi_temporal.resize(n_temporal);
        for (int t = 0; t < n_temporal; t++) {
            phi_temporal[t] = params[layout.temporal_start + t];
        }

        if (layout.is_temporal_gp) {
            // Temporal GP: sigma2 and phi (lengthscale) parameters
            T log_sigma2 = params[layout.log_sigma2_temporal_gp_idx];
            T logit_phi = params[layout.logit_phi_temporal_gp_idx];
            sigma2_temporal_gp = safe_exp(log_sigma2);

            // Logit-bounded phi: phi = lower + range * sigmoid(logit_phi)
            T sigmoid_phi = inv_logit(logit_phi);
            double phi_lower_lp = data.temporal_gp_phi_prior_lower;
            double phi_range_lp = data.temporal_gp_phi_prior_upper - phi_lower_lp;
            phi_temporal_gp = T(phi_lower_lp) + T(phi_range_lp) * sigmoid_phi;
        } else {
            // RW1/RW2/AR1: tau-based parameterization
            T log_tau = params[layout.log_tau_temporal_idx];
            tau_temporal = safe_exp(log_tau);

            if (layout.is_ar1) {
                T logit_rho = params[layout.logit_rho_ar1_idx];
                rho_ar1 = inv_logit(logit_rho);
            }
        }
    }

    // ========================================================================
    // PRIORS
    // ========================================================================

    // Fixed-effect + random-effect priors.
    #include "log_post_impl_priors_basic_block.h"

    // Overdispersion + spatial (ICAR / BYM2 / CAR-proper) priors.
    #include "log_post_impl_priors_disp_spatial_block.h"

    // GP / multi-scale GP / HSGP spatial priors and precomputed effects.
    #include "log_post_impl_priors_gp_block.h"

    // Temporal + multiscale-temporal priors.
    #include "log_post_impl_priors_temporal_block.h"

    // SVC + TVC + latent-factor priors.
    #include "log_post_impl_priors_svc_tvc_latent_block.h"

    // Spatiotemporal interaction priors + ZI/OI priors.
    #include "log_post_impl_priors_st_zi_block.h"

    // ========================================================================
    // LIKELIHOOD
    // ========================================================================

    // Observation-loop log-likelihood accumulation.
    #include "log_post_impl_likelihood_block.h"

    return log_post;
}

#include "log_post_generic_impl.h"

}  // namespace tulpa

#endif // TULPA_LOG_POST_IMPL_H
