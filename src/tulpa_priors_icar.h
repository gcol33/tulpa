// tulpa_priors_icar.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_ICAR_H
#define TULPA_PRIORS_ICAR_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 2. Spatial ICAR/BYM2 prior
// ============================================================================

template<typename T>
T compute_spatial_icar_bym2_prior(const std::vector<T>& params, const ModelData& data,
                                   const ParamLayout& layout,
                                   const T*& phi_spatial_out, T& tau_spatial_out,
                                   T& sigma_s_bym2_out, T& sigma_u_bym2_out,
                                   const T*& theta_bym2_out)
{
    T log_post = T(0.0);

    if (layout.has_spatial) {
        phi_spatial_out = &params[layout.spatial_start];

        if (layout.is_bym2) {
            // Riebler reparameterization: sigma_total, rho -> sigma_s, sigma_u
            T sigma_total_bym2 = safe_exp(params[layout.log_sigma_bym2_idx]);
            T logit_rho_val = params[layout.logit_rho_bym2_idx];
            T rho_bym2 = T(1.0) / (T(1.0) + safe_exp(-logit_rho_val));
            sigma_s_bym2_out = sigma_total_bym2 * sqrt(rho_bym2);
            sigma_u_bym2_out = sigma_total_bym2 * sqrt(T(1.0) - rho_bym2);

            theta_bym2_out = &params[layout.theta_bym2_start];

            // BYM2 Riebler: Half-Cauchy on sigma_total
            T log_sigma = params[layout.log_sigma_bym2_idx];
            log_post = log_post + log_prior_half_cauchy(log_sigma, data.sigma_re_scale);

            // Uniform(0,1) = Beta(1,1) on rho with logit Jacobian:
            // log p(logit_rho) = log(rho) + log(1-rho)
            T rho_bym2_prior = T(1.0) / (T(1.0) + safe_exp(-logit_rho_val));
            log_post = log_post + log(rho_bym2_prior)
                                + log(T(1.0) - rho_bym2_prior);

            // ICAR prior on phi_spatial
            T quad_form = T(0.0);
            for (int i = 0; i < data.n_spatial_units; i++) {
                quad_form = quad_form + T(data.n_neighbors[i]) * phi_spatial_out[i] * phi_spatial_out[i];
                int row_start = data.adj_row_ptr[i];
                int row_end = data.adj_row_ptr[i + 1];
                for (int k = row_start; k < row_end; k++) {
                    int j = data.adj_col_idx[k];
                    if (j > i) {
                        quad_form = quad_form - T(2.0) * phi_spatial_out[i] * phi_spatial_out[j];
                    }
                }
            }
            log_post = log_post - T(0.5) * quad_form;

            // Soft sum-to-zero constraint on ICAR phi
            {
                T phi_sum = T(0.0);
                for (int s = 0; s < data.n_spatial_units; s++) phi_sum = phi_sum + phi_spatial_out[s];
                log_post = log_post - T(0.5) * T(0.01) * phi_sum * phi_sum;
            }

            // N(0, I) prior on theta
            for (int s = 0; s < data.n_spatial_units; s++) {
                log_post = log_post - T(0.5) * theta_bym2_out[s] * theta_bym2_out[s];
            }
        } else {
            T log_tau = params[layout.log_tau_spatial_idx];
            tau_spatial_out = safe_exp(log_tau);

            // ICAR: Gamma prior on tau
            log_post = log_post + log_prior_gamma(log_tau, data.tau_spatial_shape, data.tau_spatial_rate);

            // ICAR prior on phi_spatial
            T quad_form = T(0.0);
            for (int i = 0; i < data.n_spatial_units; i++) {
                quad_form = quad_form + T(data.n_neighbors[i]) * phi_spatial_out[i] * phi_spatial_out[i];
                int row_start = data.adj_row_ptr[i];
                int row_end = data.adj_row_ptr[i + 1];
                for (int k = row_start; k < row_end; k++) {
                    int j = data.adj_col_idx[k];
                    if (j > i) {
                        quad_form = quad_form - T(2.0) * phi_spatial_out[i] * phi_spatial_out[j];
                    }
                }
            }
            int J = data.n_spatial_units;
            T log_tau_sp = params[layout.log_tau_spatial_idx];
            log_post = log_post + T(0.5 * (J - 1)) * log_tau_sp - T(0.5) * tau_spatial_out * quad_form;
        }
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_ICAR_H
