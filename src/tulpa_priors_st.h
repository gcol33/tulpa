// tulpa_priors_st.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_ST_H
#define TULPA_PRIORS_ST_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_temporal.h"  // single-source RW1/RW2 quadratic / cross forms

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 11. Spatiotemporal interaction prior
// ============================================================================

// Kronecker (Q_s (x) Q_t) quadratic form for a Type-IV interaction:
// sum_s n_neigh[s] * q_t(delta_s) - 2 * sum_{s<s2 adjacent} q_t(delta_s, delta_s2)
// where q_t is the RW1/RW2 temporal (cross) quadratic form. Shared by the
// non-centered (tau-free, on z) and centered (on delta) Type-IV paths.
template<typename T>
T st_kronecker_temporal_quad(const std::vector<T>& delta, const ModelData& data,
                             int S, int T_st)
{
    const auto& st = data.spatiotemporal_data;
    T total = T(0.0);
    for (int s = 0; s < S; s++) {
        const T* d_s = delta.data() + s * T_st;
        T quad_s = T(0.0);
        if (st.temporal_type == TemporalType::RW1) {
            quad_s = tulpa_temporal::rw1_quadratic_form(d_s, T_st, false);
        } else if (st.temporal_type == TemporalType::RW2) {
            quad_s = tulpa_temporal::rw2_quadratic_form(d_s, T_st, false);
        }
        int n_neigh = st.n_neighbors.empty() ? 0 : st.n_neighbors[s];
        total = total + T(n_neigh) * quad_s;

        if (!st.adj_row_ptr.empty()) {
            int row_start_s = st.adj_row_ptr[s];
            int row_end_s = st.adj_row_ptr[s + 1];
            for (int jj = row_start_s; jj < row_end_s; jj++) {
                int s2 = st.adj_col_idx[jj] - 1;
                if (s2 > s) {
                    const T* d_s2 = delta.data() + s2 * T_st;
                    T cross = T(0.0);
                    if (st.temporal_type == TemporalType::RW1) {
                        cross = tulpa_temporal::rw1_cross_form(d_s, d_s2, T_st);
                    } else if (st.temporal_type == TemporalType::RW2) {
                        cross = tulpa_temporal::rw2_cross_form(d_s, d_s2, T_st);
                    }
                    total = total - T(2.0) * cross;
                }
            }
        }
    }
    return total;
}

// Soft sum-to-zero on the S x T_st interaction: squared row and column sums.
template<typename T>
T st_sum_to_zero_penalty(const std::vector<T>& delta, int S, int T_st)
{
    T sum_s = T(0.0), sum_t = T(0.0);
    for (int s = 0; s < S; s++) {
        T row_sum = T(0.0);
        for (int t = 0; t < T_st; t++) {
            row_sum = row_sum + delta[s * T_st + t];
        }
        sum_s = sum_s + row_sum * row_sum;
    }
    for (int t = 0; t < T_st; t++) {
        T col_sum = T(0.0);
        for (int s = 0; s < S; s++) {
            col_sum = col_sum + delta[s * T_st + t];
        }
        sum_t = sum_t + col_sum * col_sum;
    }
    return T(-0.5) * T(0.001) * (sum_s + sum_t);
}

template<typename T>
T compute_st_prior(const std::vector<T>& params, const ModelData& data,
                    const ParamLayout& layout, std::vector<T>& st_delta)
{
    T log_post = T(0.0);

    if (layout.has_spatiotemporal && data.spatiotemporal_data.type != STType::NONE) {
        // Extract precision parameter
        T log_tau_st = params[layout.log_tau_st_idx];
        T tau_st = safe_exp(log_tau_st);

        // PC prior on tau (exponential on sigma = 1/sqrt(tau))
        T sigma_st = T(1.0) / safe_sqrt(tau_st);
        T lambda_st = T(-std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U);
        log_post = log_post + safe_log(lambda_st) - lambda_st * sigma_st
                 - safe_log(T(2.0) * sigma_st);
        log_post = log_post + log_tau_st;  // Jacobian for log transform

        // AR1 rho parameter
        T rho_st = T(0.0);
        if (layout.logit_rho_st_idx >= 0) {
            T logit_rho_st = params[layout.logit_rho_st_idx];
            T u_st = inv_logit(logit_rho_st);
            rho_st = T(2.0) * u_st - T(1.0);  // Map to (-1, 1)

            // Uniform(-1, 1) prior on rho
            // Jacobian for logit((rho+1)/2) transform
            log_post = log_post + safe_log(u_st) + safe_log(T(1.0) - u_st);
        }

        // GP range parameters
        T phi_st_space = T(1.0);
        T phi_st_time = T(1.0);
        if (layout.is_st_gp) {
            T log_phi_space = params[layout.log_phi_st_space_idx];
            T log_phi_time = params[layout.log_phi_st_time_idx];
            phi_st_space = safe_exp(log_phi_space);
            phi_st_time = safe_exp(log_phi_time);

            // Uniform prior within bounds
            double phi_space_val = get_value(phi_st_space);
            if (phi_space_val < data.st_phi_space_prior_lower ||
                phi_space_val > data.st_phi_space_prior_upper) {
                return T(-INFINITY);
            }
            double phi_time_val = get_value(phi_st_time);
            if (phi_time_val < data.st_phi_time_prior_lower ||
                phi_time_val > data.st_phi_time_prior_upper) {
                return T(-INFINITY);
            }
            log_post = log_post + log_phi_space + log_phi_time;  // Jacobians
        }

        // Extract delta parameters
        int n_st_params = layout.st_delta_end - layout.st_delta_start;
        st_delta.resize(n_st_params);
        for (int k = 0; k < n_st_params; k++) {
            st_delta[k] = params[layout.st_delta_start + k];
        }

        int S = data.spatiotemporal_data.n_spatial;
        int T_st = data.spatiotemporal_data.n_times;

        // NC reparameterization for Type IV
        const bool st_use_nc = (data.st_parameterization == 1 &&
                                data.spatiotemporal_data.type == STType::TYPE_IV);

        if (st_use_nc) {
            // Forward transform: delta = z / sqrt(tau_st)
            T inv_scale = T(1.0) / safe_sqrt(tau_st);
            std::vector<T> st_delta_nc(n_st_params);
            for (int k = 0; k < n_st_params; k++) {
                st_delta_nc[k] = st_delta[k] * inv_scale;
            }

            // NC prior: -0.5 * z^T (Q_s ⊗ Q_t) z  (tau-free GMRF)
            // Type IV Kronecker quadratic form on z (with tau=1)
            T nc_quad = st_kronecker_temporal_quad(st_delta, data, S, T_st);
            log_post = log_post - T(0.5) * nc_quad;

            // Rank term with actual tau and Jacobian correction
            int rank_space = S - 1;
            int rank_time = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T_st - 1) : (T_st - 2);
            if (data.spatiotemporal_data.temporal_cyclic) rank_time = T_st;
            int total_rank = rank_space * rank_time;
            int ST_total = S * T_st;
            log_post = log_post + T(0.5 * (total_rank - ST_total)) * safe_log(tau_st);

            // Sum-to-zero on reconstructed delta
            log_post = log_post + st_sum_to_zero_penalty(st_delta_nc, S, T_st);

            // Replace st_delta with reconstructed delta for observation loop
            st_delta = std::move(st_delta_nc);

        } else if (data.st_is_hsgp) {
            // HSGP-ST: spectral basis interaction (centered)
            int M = data.st_hsgp_data.m_total;

            // HSGP-ST hyperparameters
            T sigma2_st_hsgp = safe_exp(params[layout.log_sigma2_st_hsgp_idx]);
            T lengthscale_st_hsgp = safe_exp(params[layout.log_lengthscale_st_hsgp_idx]);

            // PC prior on sigma_st_hsgp: rate=4.6
            T sigma_st_h = safe_sqrt(sigma2_st_hsgp);
            log_post = log_post - T(4.6) * sigma_st_h + T(0.5) * params[layout.log_sigma2_st_hsgp_idx];

            // LogNormal(0,1) on lengthscale
            T log_ls_st = params[layout.log_lengthscale_st_hsgp_idx];
            log_post = log_post - T(0.5) * log_ls_st * log_ls_st;

            // Per-basis-function temporal GMRF prior
            int rank_t = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T_st - 1) :
                         (data.spatiotemporal_data.temporal_type == TemporalType::RW2) ? (T_st - 2) : T_st;
            if (data.spatiotemporal_data.temporal_cyclic) rank_t = T_st;

            for (int j = 0; j < M; j++) {
                double omega_sq = data.st_hsgp_data.eigenvalues[j];
                T S_j = sigma2_st_hsgp * T(std::sqrt(2.0 * M_PI)) * lengthscale_st_hsgp
                    * safe_exp(T(-0.5) * lengthscale_st_hsgp * lengthscale_st_hsgp * T(omega_sq));
                T prec_j = tau_st / safe_max(S_j, T(1e-10));

                // GMRF quadratic form: -0.5 * prec_j * delta_j' Q_t delta_j
                T qf = T(0.0);
                if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                    qf = tulpa_temporal::rw1_quadratic_form(
                        st_delta.data() + j * T_st, T_st, false);
                } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                    qf = tulpa_temporal::rw2_quadratic_form(
                        st_delta.data() + j * T_st, T_st, false);
                }
                log_post = log_post + T(0.5 * rank_t) * safe_log(prec_j)
                         - T(0.5) * prec_j * qf;

                // Soft sum-to-zero per basis function
                T sum_j = T(0.0);
                for (int t = 0; t < T_st; t++) sum_j = sum_j + st_delta[j * T_st + t];
                log_post = log_post - T(0.5) * T(0.001) * sum_j * sum_j;
            }

        } else {
            // Centered parameterization (ICAR/BYM2 spatial: Types I-IV)
            if (data.spatiotemporal_data.type == STType::TYPE_I) {
                // IID: delta[s,t] ~ N(0, 1/tau)
                T quad = T(0.0);
                for (int k = 0; k < n_st_params; k++) {
                    quad = quad + st_delta[k] * st_delta[k];
                }
                log_post = log_post + T(0.5 * n_st_params) * safe_log(tau_st)
                         - T(0.5) * tau_st * quad;

            } else if (data.spatiotemporal_data.type == STType::TYPE_II) {
                // Structured time at each location
                bool st_cyclic = data.spatiotemporal_data.temporal_cyclic;
                for (int s = 0; s < S; s++) {
                    if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                        T quad = tulpa_temporal::rw1_quadratic_form(
                            st_delta.data() + s * T_st, T_st, st_cyclic);
                        int rank = st_cyclic ? T_st : T_st - 1;
                        log_post = log_post + T(0.5 * rank) * safe_log(tau_st)
                                 - T(0.5) * tau_st * quad;
                    } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                        T quad = tulpa_temporal::rw2_quadratic_form(
                            st_delta.data() + s * T_st, T_st, st_cyclic);
                        int rank = st_cyclic ? T_st : T_st - 2;
                        log_post = log_post + T(0.5 * rank) * safe_log(tau_st)
                                 - T(0.5) * tau_st * quad;
                    }
                }

            } else if (data.spatiotemporal_data.type == STType::TYPE_III) {
                // Structured space at each time point (ICAR)
                int rank_s = S - 1;
                for (int t = 0; t < T_st; t++) {
                    // Compute ICAR quadratic form for spatial field at time t
                    T quad = T(0.0);
                    for (int s = 0; s < S; s++) {
                        T delta_st = st_delta[s * T_st + t];
                        int n_neigh = data.spatiotemporal_data.n_neighbors.empty() ? 0
                            : data.spatiotemporal_data.n_neighbors[s];
                        quad = quad + T(n_neigh) * delta_st * delta_st;
                        if (!data.spatiotemporal_data.adj_row_ptr.empty()) {
                            int row_start_s = data.spatiotemporal_data.adj_row_ptr[s];
                            int row_end_s = data.spatiotemporal_data.adj_row_ptr[s + 1];
                            for (int jj = row_start_s; jj < row_end_s; jj++) {
                                int s2 = data.spatiotemporal_data.adj_col_idx[jj] - 1;
                                if (s2 > s) {
                                    T delta_s2t = st_delta[s2 * T_st + t];
                                    quad = quad - T(2.0) * delta_st * delta_s2t;
                                }
                            }
                        }
                    }
                    log_post = log_post + T(0.5 * rank_s) * safe_log(tau_st)
                             - T(0.5) * tau_st * quad;
                }

            } else if (data.spatiotemporal_data.type == STType::TYPE_IV) {
                // Kronecker: Q_delta = Q_s ⊗ Q_t
                T kron_quad = st_kronecker_temporal_quad(st_delta, data, S, T_st);
                log_post = log_post - T(0.5) * tau_st * kron_quad;
                // Rank terms
                int rank_space = S - 1;
                int rank_time = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T_st - 1) : (T_st - 2);
                if (data.spatiotemporal_data.temporal_cyclic) rank_time = T_st;
                int total_rank = rank_space * rank_time;
                log_post = log_post + T(0.5 * total_rank) * safe_log(tau_st);
            }

            // Soft sum-to-zero constraint
            log_post = log_post + st_sum_to_zero_penalty(st_delta, S, T_st);
        }
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_ST_H
