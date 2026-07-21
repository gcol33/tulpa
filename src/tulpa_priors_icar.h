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
#include "icar_kernel.h"                  // for_each_icar_component
#include "tulpa/soft_sum_to_zero.h"       // s2z_precision

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 2. Spatial ICAR/BYM2 prior
// ============================================================================

// Soft sum-to-zero on an intrinsic spatial field: one penalty per connected
// component, each at the precision that pins that component's sum at
// sd = kappa * csize (see tulpa/soft_sum_to_zero.h). Shared by the BYM2 and
// ICAR branches so the two cannot drift apart.
template<typename T>
T icar_sum_to_zero_penalty(const T* phi, int n_spatial_units, int n_components)
{
    T penalty = T(0.0);
    tulpa::for_each_icar_component(0, n_spatial_units, n_components,
        [&](int cstart, int csize) {
            T s = T(0.0);
            for (int i = 0; i < csize; i++) s = s + phi[cstart + i];
            penalty = penalty - T(0.5) * T(tulpa::s2z_precision(csize)) * s * s;
        });
    return penalty;
}

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
            log_post = log_post + icar_sum_to_zero_penalty(
                phi_spatial_out, data.n_spatial_units, data.n_spatial_components);

            // N(0, I) prior on theta
            for (int s = 0; s < data.n_spatial_units; s++) {
                log_post = log_post - T(0.5) * theta_bym2_out[s] * theta_bym2_out[s];
            }
        } else if (layout.is_car_proper) {
            // Proper CAR: Q(rho) = D - rho W is full-rank (PD), so no ICAR
            // sum-to-zero constraint and the field mean is identified.
            //   log p(phi | tau, rho) = 0.5 log|tau Q(rho)| - 0.5 tau phi'Q phi
            // The parameter-dependent part of log|Q(rho)| is sum_i log(1 - rho
            // mu_i) with mu_i the eigenvalues of the symmetric normalized
            // adjacency (data.car_adj_eigenvalues, precomputed); the constant
            // log|D| is dropped (it does not affect the posterior). This closed
            // form is autodiff-differentiable in rho, unlike a per-eval Cholesky.
            T log_tau = params[layout.log_tau_spatial_idx];
            tau_spatial_out = safe_exp(log_tau);
            log_post = log_post + log_prior_gamma(log_tau, data.tau_spatial_shape,
                                                  data.tau_spatial_rate);

            // rho in (car_rho_lower, car_rho_upper) via the logit map, with the
            // Uniform-prior + logit Jacobian log p(logit_rho) = log u + log(1-u).
            T logit_rho = params[layout.logit_rho_car_idx];
            T u = T(1.0) / (T(1.0) + safe_exp(-logit_rho));
            T rho = T(data.car_rho_lower)
                  + (T(data.car_rho_upper) - T(data.car_rho_lower)) * u;
            log_post = log_post + log(u) + log(T(1.0) - u);

            // Quadratic form phi' Q(rho) phi = sum_i d_i phi_i^2
            //   - 2 rho sum_{i~j, j>i} phi_i phi_j.
            T quad_form = T(0.0);
            for (int i = 0; i < data.n_spatial_units; i++) {
                quad_form = quad_form
                    + T(data.n_neighbors[i]) * phi_spatial_out[i] * phi_spatial_out[i];
                int row_start = data.adj_row_ptr[i];
                int row_end = data.adj_row_ptr[i + 1];
                for (int k = row_start; k < row_end; k++) {
                    int j = data.adj_col_idx[k];
                    if (j > i)
                        quad_form = quad_form
                            - T(2.0) * rho * phi_spatial_out[i] * phi_spatial_out[j];
                }
            }

            // 0.5 * (n log tau + sum_i log(1 - rho mu_i)) - 0.5 tau phi'Q phi.
            T log_det = T(data.n_spatial_units) * log_tau;
            for (std::size_t k = 0; k < data.car_adj_eigenvalues.size(); k++)
                log_det = log_det
                    + log(T(1.0) - rho * T(data.car_adj_eigenvalues[k]));
            log_post = log_post + T(0.5) * log_det - T(0.5) * tau_spatial_out * quad_form;
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
            // Soft sum-to-zero constraint on ICAR phi. The precision has a
            // constant null direction (per component) that is otherwise jointly
            // unidentified with the intercept; pin it as the BYM2 branch and the
            // Laplace path do, so exact-NUTS ICAR fits mix. One penalty per
            // component, matching the J - n_components rank normalizer below.
            log_post = log_post + icar_sum_to_zero_penalty(
                phi_spatial_out, data.n_spatial_units, data.n_spatial_components);
            // Rank of the ICAR precision is J - k for k connected components
            // (one constant null direction per component). Using J - 1 on a
            // disconnected graph (spatial(by=) replication) biases tau upward.
            int J = data.n_spatial_units;
            int rank_icar = J - data.n_spatial_components;
            T log_tau_sp = params[layout.log_tau_spatial_idx];
            log_post = log_post + T(0.5 * rank_icar) * log_tau_sp - T(0.5) * tau_spatial_out * quad_form;
        }
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_ICAR_H
