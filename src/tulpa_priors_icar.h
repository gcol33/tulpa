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
#include "intrinsic_log_pdet.h"           // icar_log_pdet

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 2. Spatial ICAR/BYM2 prior
// ============================================================================

// phi' Q phi for the adjacency in `data`, with Q = D - rho W: the degree term
// summed over units, minus twice the cross term summed over each edge once
// (the j > i test picks each edge from its lower-indexed endpoint). `rho` null
// is the intrinsic case, rho == 1, where the cross term carries no coefficient.
// `w` null is the unweighted form; otherwise it is phi' diag(w) Q phi with w
// constant within each connected component (BYM2's per-component scaling,
// `ModelData::bym2_node_prec`), so an edge's two endpoints share one weight and
// the product stays symmetric.
template<typename T>
T icar_quad_form(const T* phi, const ModelData& data, const T* rho,
                 const double* w = nullptr)
{
    T quad_form = T(0.0);
    for (int i = 0; i < data.n_spatial_units; i++) {
        // Weighted: node i's own terms, scaled by its component's weight. The
        // unweighted form keeps its single running sum, so its value is the
        // same to the last bit as before the weight existed.
        T qi = w ? T(0.0) : quad_form;
        qi = qi + T(data.n_neighbors[i]) * phi[i] * phi[i];
        int row_start = data.adj_row_ptr[i];
        int row_end = data.adj_row_ptr[i + 1];
        for (int k = row_start; k < row_end; k++) {
            int j = data.adj_col_idx[k];
            if (j <= i) continue;
            qi = rho
                ? qi - T(2.0) * (*rho) * phi[i] * phi[j]
                : qi - T(2.0) * phi[i] * phi[j];
        }
        quad_form = w ? quad_form + T(w[i]) * qi : qi;
    }
    return quad_form;
}

// Augmented intrinsic precision on a spatial field: Q_aug = Q + sum_c 1_c 1_c'
// / J_c (see tulpa/sum_to_zero.h). Returns the addition to the quadratic form,
// sum_c (sum_{i in c} phi_i)^2 / J_c, so the caller scales it by the same tau
// it scales phi'Q phi by. The constant direction of each component then carries
// the field's own precision, and `icar_center_field` removes it on the way into
// eta. Shared by the BYM2 and ICAR branches so the two cannot drift apart.
// Component-centered fields give removed levels unit precision; an island
// keeps its proper precision. Global centering scales the full augmentation
// by each component's node weight.
template<typename T>
T icar_sum_to_zero_augment(const T* phi, const ModelData& data,
                           const double* w = nullptr)
{
    T aug = T(0.0);
    tulpa::for_each_icar_component(0, data.spatial_partition,
        [&](int start, const int* idx, int csize) {
            const T s = tulpa::s2z_component_sum(phi, start, idx, csize);
            const int first = idx ? idx[0] : start;
            const bool weighted = w && (!data.spatial_component_centering || csize == 1);
            aug = weighted ? aug + T(w[first]) * (tulpa::s2z_aug_coef(T(1.0), csize) * s * s)
                    : aug + tulpa::s2z_aug_coef(T(1.0), csize) * s * s;
        });
    return aug;
}

// Centring of the field on its way into eta, into `out`. This is what makes the
// augmentation an identification rather than a weak penalty: the constant
// direction carries precision tau (order 1) instead of the 1/(kappa*J)^2 the
// old soft pin carried, so leaving it in eta would free the level rather than
// fix it.
//
// Global centering removes one overall level and retains differences between
// components. Component centering removes the level of each nontrivial
// component; isolated units retain their proper field values.
template<typename T>
void icar_center_field(const T* phi, const ModelData& data, std::vector<T>& out)
{
    out.assign(phi, phi + data.n_spatial_units);
    if (data.spatial_component_centering) {
        tulpa::for_each_icar_component(0, data.spatial_partition,
            [&](int start, const int* idx, int csize) {
                if (csize <= 1) return;
                const T m = tulpa::s2z_component_sum(phi, start, idx, csize) / T(csize);
                for (int j = 0; j < csize; ++j) {
                    const int i = idx ? idx[j] : start + j;
                    out[i] = phi[i] - m;
                }
            });
    } else {
        const T m = tulpa::s2z_component_mean(phi, 0, data.n_spatial_units);
        for (int i = 0; i < data.n_spatial_units; i++) out[i] = phi[i] - m;
    }
}

template<typename T>
void areal_likelihood_field(const T* phi, const ModelData& data,
                            const ParamLayout& layout, std::vector<T>& out) {
    if (!layout.is_car_proper) {
        icar_center_field(phi, data, out);
    } else {
        out.assign(phi, phi + data.n_spatial_units);
        if (data.car_center)
            tulpa::s2z_centre_component(out.data(), 0, data.n_spatial_units);
    }
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
            T log_sigma = params[layout.log_sigma_bym2_idx];
            T sigma_total_bym2 = safe_exp(log_sigma);
            T logit_rho_val = params[layout.logit_rho_bym2_idx];
            T rho_bym2 = T(1.0) / (T(1.0) + safe_exp(-logit_rho_val));
            sigma_s_bym2_out = sigma_total_bym2 * safe_sqrt(rho_bym2);
            sigma_u_bym2_out = sigma_total_bym2 * safe_sqrt(T(1.0) - rho_bym2);

            theta_bym2_out = &params[layout.theta_bym2_start];

            log_post = log_post + log_prior_log_sigma_pc(
                log_sigma, data.spatial_sigma_prior_U, data.spatial_sigma_prior_alpha);

            // Uniform(0,1) = Beta(1,1) on rho with logit Jacobian:
            // log p(logit_rho) = log(rho) + log(1-rho)
            log_post = log_post + safe_log(rho_bym2)
                                + safe_log(T(1.0) - rho_bym2);

            // ICAR prior on phi_spatial
            // A disconnected graph is scaled per component (#902): the prior
            // weights the intrinsic precision, with node_prec constant within
            // a component. Its log-determinant is constant in every sampled
            // parameter and is supplied by the field normalizer.
            const double* w = data.bym2_node_prec.empty()
                ? nullptr : data.bym2_node_prec.data();
            T quad_form = icar_quad_form<T>(phi_spatial_out, data, nullptr, w);
            // Augmented Q_aug = Q + sum_c 1_c 1_c'/J_c. BYM2's phi is unit-scale
            // (tau = 1; the scale lives in sigma_s_bym2), so the augmentation
            // enters at coefficient 1 and there is no log-tau normalizer to
            // move -- 0.5 * J * log(1) and 0.5 * (J - L) * log(1) are both 0.
            log_post = log_post - T(0.5) * (quad_form + icar_sum_to_zero_augment(
                phi_spatial_out, data, w));

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
            log_post = log_post + log_prior_log_tau_pc(
                log_tau, data.spatial_sigma_prior_U, data.spatial_sigma_prior_alpha);

            // rho in (car_rho_lower, car_rho_upper) via the logit map, with the
            // Uniform-prior + logit Jacobian log p(logit_rho) = log u + log(1-u).
            T logit_rho = params[layout.logit_rho_car_idx];
            T u = T(1.0) / (T(1.0) + safe_exp(-logit_rho));
            T rho = T(data.car_rho_lower)
                  + (T(data.car_rho_upper) - T(data.car_rho_lower)) * u;
            log_post = log_post + T(data.car_rho_prior_a) * safe_log(u)
                + T(data.car_rho_prior_b) * safe_log(T(1.0) - u);

            // Quadratic form phi' Q(rho) phi = sum_i d_i phi_i^2
            //   - 2 rho sum_{i~j, j>i} phi_i phi_j.
            T quad_form = icar_quad_form<T>(phi_spatial_out, data, &rho);

            // 0.5 * (n log tau + sum_i log(1 - rho mu_i)) - 0.5 tau phi'Q phi.
            T log_det = T(data.n_spatial_units) * log_tau;
            for (std::size_t k = 0; k < data.car_adj_eigenvalues.size(); k++)
                log_det = log_det
                    + safe_log(T(1.0) - rho * T(data.car_adj_eigenvalues[k]));
            log_post = log_post + T(0.5) * log_det - T(0.5) * tau_spatial_out * quad_form;
        } else {
            T log_tau = params[layout.log_tau_spatial_idx];
            tau_spatial_out = safe_exp(log_tau);

            log_post = log_post + log_prior_log_tau_pc(
                log_tau, data.spatial_sigma_prior_U, data.spatial_sigma_prior_alpha);

            // ICAR prior on phi_spatial
            T quad_form = icar_quad_form<T>(phi_spatial_out, data, nullptr);
            // Q_aug = Q + sum_c 1_c 1_c'/J_c has rank J: its J-L contrasts
            // and L component constants all carry precision tau. Its Gaussian
            // normalizer therefore uses J log(tau). Component-centered levels
            // integrate out; global centering retains component differences
            // under this proper prior.
            int J = data.n_spatial_units;
            int L = data.spatial_partition.n_components();
            if (L < 1) L = 1;
            T quad_aug = quad_form + icar_sum_to_zero_augment(
                phi_spatial_out, data);
            log_post = log_post
                     + T(0.5 * tulpa::s2z_aug_rank(J - L, L)) * log_tau
                     - T(0.5) * tau_spatial_out * quad_aug;
        }
    }

    return log_post;
}

// The constant compute_spatial_icar_bym2_prior leaves out, so that the two
// together are a normalized density on the block's parameters.
//   ICAR       : of phi ~ N(0, (tau Q_aug)^-1),
//                the 0.5 log|Q_aug| - 0.5 J log(2 pi) its 0.5 J log tau omits.
//   BYM2       : of phi's weighted augmented precision H, the
//                0.5 log|H| - 0.5 J log(2 pi), and of theta ~ N(0, I) the
//                -0.5 J log(2 pi). Removed component levels have unit precision
//                under component centering and precision w_c under global centering.
//   proper CAR : the 0.5 log|D| - 0.5 J
//                log(2 pi) of log|tau (D - rho W)| = J log tau + log|D| +
//                sum_k log(1 - rho mu_k) that the density omits.
inline double spatial_icar_bym2_prior_log_normalizer(const ModelData& data,
                                                     const ParamLayout& layout) {
    if (!layout.has_spatial) return 0.0;
    const int J = data.n_spatial_units;
    const double unit_normal = log_prior_normal_log_normalizer(1.0);
    if (layout.is_car_proper) {
        double log_det_d = 0.0;
        for (int i = 0; i < J; i++) log_det_d += std::log(double(data.n_neighbors[i]));
        return 0.5 * log_det_d + J * unit_normal
            + std::lgamma(data.car_rho_prior_a + data.car_rho_prior_b)
            - std::lgamma(data.car_rho_prior_a) - std::lgamma(data.car_rho_prior_b);
    }
    const double log_pdet = icar_log_pdet(
        J, data.adj_row_ptr.data(), data.adj_col_idx.data(),
        data.n_neighbors.data(), data.spatial_partition);
    if (layout.is_bym2) {
        double sum_log_w = 0.0;
        for (double w : data.bym2_node_prec) sum_log_w += std::log(w);
        if (data.spatial_component_centering && !data.bym2_node_prec.empty()) {
            for (int c = 0; c < data.spatial_partition.n_components(); ++c) {
                if (data.spatial_partition.size(c) <= 1) continue;
                const int* nodes = data.spatial_partition.nodes(c);
                sum_log_w -= std::log(data.bym2_node_prec[nodes ? nodes[0] : 0]);
            }
        }
        return 0.5 * (sum_log_w + log_pdet) + 2.0 * J * unit_normal;
    }
    return 0.5 * log_pdet + J * unit_normal;
}

} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_ICAR_H
