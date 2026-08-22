// tulpa_priors_svc.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_SVC_H
#define TULPA_PRIORS_SVC_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hsgp_spectral.h"
#include "pc_prior.h"
#include "hmc_svc_autodiff.h"

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 8. SVC (Spatially-Varying Coefficients) prior
// ============================================================================

template<typename T>
T compute_svc_prior(const std::vector<T>& params, const ModelData& data,
                     const ParamLayout& layout, std::vector<T>& svc_eta)
{
    T log_post = T(0.0);

    if (layout.has_svc && data.svc_data.n_svc > 0) {
        int n_svc = data.svc_data.n_svc;
        int n_obs = data.svc_data.n_obs;

        if (data.svc_is_hsgp) {
            // HSGP-based SVC: basis function approximation
            int m_total = data.svc_hsgp_data.m_total;

            // Per-term: sigma2, lengthscale, beta[m_total]
            svc_eta.resize(n_obs, T(0.0));
            for (int j = 0; j < n_svc; j++) {
                T log_sigma2 = params[layout.log_sigma2_svc_start + j];
                T sigma2_j = safe_exp(log_sigma2);

                // Half-Cauchy(0, scale) on the marginal SD, carried to the
                // sampled log-variance coordinate by the shared helper. sigma2
                // is term j's marginal spatial variance on either basis -- the
                // basis choice fixes how the field is REPRESENTED, not what its
                // variance means -- so both branches read the one prior and the
                // one user-set scale.
                log_post = log_post + log_prior_log_sigma2_half_cauchy(
                    log_sigma2, data.svc_sigma2_prior_scale);

                T log_ls = params[layout.log_phi_svc_start + j];
                T ls_j = safe_exp(log_ls);

                // LogNormal(0,1) on lengthscale
                log_post = log_post - T(0.5) * log_ls * log_ls;

                // Extract beta_j and compute f_j = Phi * (sqrt(S_j) * beta_j)
                // N(0, I) prior on beta
                for (int k = 0; k < m_total; k++) {
                    T beta_jk = params[layout.svc_w_start + j * m_total + k];
                    log_post = log_post - T(0.5) * beta_jk * beta_jk;

                    // Compute scaled beta: sqrt(S(eigenvalue_k, sigma2_j, ls_j)) * beta_jk
                    double omega_sq = data.svc_hsgp_data.eigenvalues[k];
                    T S_k = hsgp_spectral_density_2d(sigma2_j, ls_j, omega_sq);
                    T sqrt_S_k = safe_sqrt(S_k);

                    // Accumulate f_j[i] = sum_k phi[i,k] * sqrt_S_k * beta_jk
                    for (int i = 0; i < n_obs; i++) {
                        double phi_ik = data.svc_hsgp_data.phi_flat[i * m_total + k];
                        svc_eta[i] = svc_eta[i] + T(phi_ik) * sqrt_S_k * beta_jk *
                                     T(data.svc_data.X_svc[i * n_svc + j]);
                    }
                }
            }
        } else {
            // NNGP-based SVC (original path)

            // Extract sigma2 (spatial variance) parameters
            std::vector<T> svc_sigma2(n_svc);
            for (int j = 0; j < n_svc; j++) {
                T log_sigma2 = params[layout.log_sigma2_svc_start + j];
                svc_sigma2[j] = safe_exp(log_sigma2);

                // Half-Cauchy(0, scale) on the marginal SD, carried to the
                // sampled log-variance coordinate by the shared helper.
                log_post = log_post + log_prior_log_sigma2_half_cauchy(
                    log_sigma2, data.svc_sigma2_prior_scale);
            }

            // Extract phi (spatial range) parameters. Sampled unconstrained on
            // the log scale under a PC prior on the range: the density is
            // proper on (0, inf) and penalizes short ranges, so no bounding
            // box is needed. phi is the range itself (every kernel is
            // exp(-d / phi)), so the d = 2 density applies to it directly.
            std::vector<T> svc_phi(n_svc);
            for (int j = 0; j < n_svc; j++) {
                T log_phi = params[layout.log_phi_svc_start + j];
                svc_phi[j] = safe_exp(log_phi);

                log_post = log_post + log_prior_range_pc_at_log(
                    log_phi, data.svc_phi_prior_U, data.svc_phi_prior_alpha);
                log_post = log_post + log_phi;  // Jacobian
            }

            int n_svc_params = n_svc * n_obs;

            if (data.svc_parameterization == 1) {
                // Non-centered: params[svc_w_start ..] are z_j ~ N(0, I) per
                // term. Each term's field w_j = f(z_j, sigma2_j, phi_j) and
                // its likelihood are reconstructed by apply_svc_nc_transform_*
                // in initialize_generic_state, which also centers each term's
                // reconstructed field and computes svc_eta from it (both need
                // the field, not z). This avoids
                // the field/hyperparameter funnel the centered
                // parameterization collapses under a diagonal mass matrix --
                // the same fix as compute_gp_spatial_prior's gp_parameterization
                // branch.
                //
                // No z -> w Jacobian: z_j is a genuine auxiliary with its own
                // N(0, I) prior and w_j is a deterministic function of
                // (z_j, theta_j), so adding log|dw_j/dz_j| would inflate the
                // term's amplitude instead of leaving the (w, theta) posterior
                // unchanged -- see compute_gp_spatial_prior's identical note.
                for (int k = 0; k < n_svc_params; k++) {
                    T zk = params[layout.svc_w_start + k];
                    log_post = log_post - T(0.5) * zk * zk;
                }
                svc_eta.clear();  // filled by apply_svc_nc_transform_* on w
            } else {
                // Centered: params[svc_w_start ..] ARE the fields w_j, and the
                // NNGP density is placed on them directly.
                std::vector<T> svc_w_flat(n_svc_params);
                for (int k = 0; k < n_svc_params; k++) {
                    svc_w_flat[k] = params[layout.svc_w_start + k];
                }

                // NNGP prior on each SVC term
                for (int j = 0; j < n_svc; j++) {
                    std::vector<T> w_j(n_obs);
                    for (int k = 0; k < n_obs; k++) {
                        w_j[k] = svc_w_flat[j * n_obs + k];
                    }
                    log_post = log_post + tulpa_svc_ad::nngp_log_lik(w_j, svc_sigma2[j], svc_phi[j], data.svc_data);
                }

                // Soft sum-to-zero constraint
                log_post = log_post + tulpa_svc_ad::svc_sum_to_zero_penalty(svc_w_flat, data.svc_data);

                // Precompute SVC contribution to linear predictor
                svc_eta.resize(n_obs, T(0.0));
                tulpa_svc_ad::compute_svc_eta(svc_w_flat, data.svc_data, svc_eta);
            }
        }
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_SVC_H
