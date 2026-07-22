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
#include "tulpa/sum_to_zero.h"            // s2z_aug_coef / s2z_aug_rank

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 2. Spatial ICAR/BYM2 prior
// ============================================================================

// Augmented intrinsic precision on a spatial field: Q_aug = Q + sum_c 1_c 1_c'
// / J_c (see tulpa/sum_to_zero.h). Returns the addition to the quadratic form,
// sum_c (sum_{i in c} phi_i)^2 / J_c, so the caller scales it by the same tau
// it scales phi'Q phi by. The constant direction of each component then carries
// the field's own precision, and `icar_center_field` removes it on the way into
// eta. Shared by the BYM2 and ICAR branches so the two cannot drift apart.
template<typename T>
T icar_sum_to_zero_augment(const T* phi, const GraphPartition& partition)
{
    T aug = T(0.0);
    tulpa::for_each_icar_component(0, partition,
        [&](int start, const int* idx, int csize) {
            const T s = tulpa::s2z_component_sum(phi, start, idx, csize);
            aug = aug + tulpa::s2z_aug_coef(T(1.0), csize) * s * s;
        });
    return aug;
}

// Centring of the field on its way into eta, into `out`. This is what makes the
// augmentation an identification rather than a weak penalty: the constant
// direction carries precision tau (order 1) instead of the 1/(kappa*J)^2 the
// old soft pin carried, so leaving it in eta would free the level rather than
// fix it.
//
// The WHOLE field is centred, not each component, even though the augmentation
// pins one direction per component. The two counts differ on purpose. Adding a
// constant to every node shifts every eta_i equally and is absorbed by the
// intercept, so it is unidentified and has to go. Adding a constant to ONE
// component shifts eta only for that component's observations, which a single
// intercept cannot absorb -- the data identifies it, and centring it away would
// delete a real level difference between components. The augmentation still
// covers all L directions so the prior is proper; the data covers the L - 1 the
// centring leaves alone. The Laplace path centres the same single direction.
template<typename T>
void icar_center_field(const T* phi, int n_spatial_units,
                       int /*n_components*/, std::vector<T>& out)
{
    out.assign(n_spatial_units, T(0.0));
    const T m = tulpa::s2z_component_mean(phi, 0, n_spatial_units);
    for (int i = 0; i < n_spatial_units; i++) out[i] = phi[i] - m;
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
            // Augmented Q_aug = Q + sum_c 1_c 1_c'/J_c. BYM2's phi is unit-scale
            // (tau = 1; the scale lives in sigma_s_bym2), so the augmentation
            // enters at coefficient 1 and there is no log-tau normalizer to
            // move -- 0.5 * J * log(1) and 0.5 * (J - L) * log(1) are both 0.
            log_post = log_post - T(0.5) * (quad_form + icar_sum_to_zero_augment(
                phi_spatial_out, data.spatial_partition));

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
            // Augmented Q_aug = Q + sum_c 1_c 1_c'/J_c: each component's constant
            // direction carries tau rather than a separate pin precision, and
            // the field is centred per component on its way into eta, so the
            // direction leaves the likelihood entirely.
            //
            // ICAR's null space is exactly the L component constants, and every
            // one of them is pinned, so Q_aug is FULL RANK: (J - L) + L = J. The
            // normalizer takes J log tau, not (J - L) log tau -- keeping the
            // deficient rank alongside the augmentation would bias tau. The
            // freed constants integrate to -0.5 L log tau, cancelling the
            // +0.5 L log tau the full rank adds, so this agrees with the
            // hard-constrained density on the sum-to-zero subspace.
            int J = data.n_spatial_units;
            int L = data.spatial_partition.n_components();
            if (L < 1) L = 1;
            T log_tau_sp = params[layout.log_tau_spatial_idx];
            T quad_aug = quad_form + icar_sum_to_zero_augment(
                phi_spatial_out, data.spatial_partition);
            log_post = log_post
                     + T(0.5 * tulpa::s2z_aug_rank(J - L, L)) * log_tau_sp
                     - T(0.5) * tau_spatial_out * quad_aug;
        }
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_ICAR_H
