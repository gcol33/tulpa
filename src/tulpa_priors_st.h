// tulpa_priors_st.h
// Sliced from tulpa_priors.h. Include via "tulpa_priors.h" or directly.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_ST_H
#define TULPA_PRIORS_ST_H

#include <cstddef>
#include <vector>
#include <tulpa/st_margins.h>
#include <tulpa/st_hsgp_kernel.h>
#include "st_gp_kernel.h"
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_temporal.h"  // single-source RW1/RW2 quadratic / cross forms
#include "icar_kernel.h"   // count_graph_components (spatial rank)
#include "tulpa/soft_sum_to_zero.h"   // s2z_precision
#include "st_null_space.h"            // st_trend_weight / st_needs_trend_pin
#include "pc_prior.h"      // single-source PC prior on every sampled scale

namespace tulpa {
namespace priors {

using namespace math;

// ============================================================================
// 11. Spatiotemporal interaction prior
// ============================================================================

// Rank of the spatiotemporal ICAR spatial precision: S - k for k connected
// components of the (1-based) spatiotemporal adjacency, and S - 1 when no
// adjacency is carried.
//
// This is the ONE place the rank is derived. It multiplies log(tau_st) in the
// GMRF normalizer, so a branch that hardcodes S - 1 puts a different power of
// tau in the target than the branch beside it whenever the graph has more than
// one component -- islands, or any unit its adjacency file leaves with no
// neighbours -- and the centered and non-centered parameterizations of the same
// model then report different tau_st posteriors.
inline int st_spatial_rank(const ModelData& data, int S) {
    const auto& st = data.spatiotemporal_data;
    const int k_s = st.adj_row_ptr.empty()
        ? 1
        : count_graph_components(S, st.adj_row_ptr.data(),
                                 st.adj_col_idx.data(), 1);
    return S - k_s;
}

// The interaction's temporal margin Q_t at unit precision, as the three
// quantities every branch of the density reads: the bilinear form a' Q_t b,
// the rank that multiplies log(precision) in the normalizer, and the part of
// log|Q_t|_+ that moves with a sampled hyperparameter.
//
// RW1 and RW2 are intrinsic: their rank is the operator's own (the cyclic flag
// selects both the operator and its rank), and their pseudo-determinant is a
// constant dropped with the other constants. AR1 is stationary and proper: rank
// T, and log|Q_AR1| = log(1 - rho^2) moves with the correlation, so it stays in
// the target. Every temporal type outside st_time_margin_supported() is refused
// at compute_param_layout, so the fall-through arms are unreachable.
template<typename T>
inline T st_time_cross_form(const T* a, const T* b, int T_st,
                            const SpatiotemporalData& st, const T& rho)
{
    switch (st.temporal_type) {
    case TemporalType::RW1:
        return tulpa_temporal::rw1_cross_form(a, b, T_st, st.temporal_cyclic);
    case TemporalType::RW2:
        return tulpa_temporal::rw2_cross_form(a, b, T_st, st.temporal_cyclic);
    case TemporalType::AR1:
        return tulpa_temporal::ar1_cross_form(a, b, T_st, rho);
    default:
        return T(0.0);
    }
}

inline int st_time_rank(const SpatiotemporalData& st, int T_st) {
    switch (st.temporal_type) {
    case TemporalType::RW1: return tulpa_temporal::rw1_rank(T_st, st.temporal_cyclic);
    case TemporalType::RW2: return tulpa_temporal::rw2_rank(T_st, st.temporal_cyclic);
    case TemporalType::AR1: return T_st;
    default:                return 0;
    }
}

template<typename T>
inline T st_time_log_det(const SpatiotemporalData& st, const T& rho) {
    if (st.temporal_type == TemporalType::AR1) {
        return safe_log(tulpa_temporal::ar1_one_minus_rho2(rho));
    }
    return T(0.0);
}

// Kronecker (Q_s (x) Q_t) quadratic form for a Type-IV interaction:
// sum_s n_neigh[s] * q_t(delta_s) - 2 * sum_{s<s2 adjacent} q_t(delta_s, delta_s2)
// where q_t is the temporal margin's (cross) form. Shared by the non-centered
// (tau-free, on z) and centered (on delta) Type-IV paths.
//
// SpatiotemporalData::temporal_cyclic selects Q_t, in the diagonal and the
// cross term alike: a cyclic Q_t carries the wrap-around edge in both, and the
// rank the normalizer beside this reads (st_time_rank) is the rank of the
// operator the flag selects.
template<typename T>
T st_kronecker_temporal_quad(const std::vector<T>& delta, const ModelData& data,
                             int S, int T_st, const T& rho)
{
    const auto& st = data.spatiotemporal_data;
    T total = T(0.0);
    for (int s = 0; s < S; s++) {
        const T* d_s = delta.data() + s * T_st;
        int n_neigh = st.n_neighbors.empty() ? 0 : st.n_neighbors[s];
        total = total + T(n_neigh) * st_time_cross_form(d_s, d_s, T_st, st, rho);

        if (!st.adj_row_ptr.empty()) {
            int row_start_s = st.adj_row_ptr[s];
            int row_end_s = st.adj_row_ptr[s + 1];
            for (int jj = row_start_s; jj < row_end_s; jj++) {
                int s2 = st.adj_col_idx[jj] - 1;
                if (s2 > s) {
                    const T* d_s2 = delta.data() + s2 * T_st;
                    total = total - T(2.0) *
                        st_time_cross_form(d_s, d_s2, T_st, st, rho);
                }
            }
        }
    }
    return total;
}

// Soft sum-to-zero on the S x T_st interaction: squared row and column sums,
// plus each site's squared linear TREND where the temporal kernel carries one.
// The margins need their own precisions -- a row sums T_st terms and a column
// sums S terms, and s2z_precision pins each sum at sd = kappa * (its own
// length). One shared constant would leave the longer margin under-pinned.
//
// `add_trend` is st_needs_trend_pin()'s answer, threaded in rather than
// re-derived here so the density, the matrix form and the mass override read
// one predicate. Under a non-cyclic RW2 marginal the row and column sums span
// S + T - 1 of the kernel's T + 2S - 2 dimensions, and the third family is the
// S - 1 that would otherwise carry no prior curvature at all
// (st_null_space.h). v is centred, so it is orthogonal to the row sums and
// adds nothing where the kernel is the constants alone.
template<typename T>
T st_sum_to_zero_penalty(const std::vector<T>& delta, int S, int T_st,
                         bool add_trend = false)
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
    T penalty = T(-0.5) * (T(tulpa::s2z_precision(T_st)) * sum_s
                         + T(tulpa::s2z_precision(S)) * sum_t);
    if (add_trend) {
        T sum_v = T(0.0);
        for (int s = 0; s < S; s++) {
            T trend = T(0.0);
            for (int t = 0; t < T_st; t++) {
                trend = trend
                      + T(tulpa_st::st_trend_weight(t, T_st)) * delta[s * T_st + t];
            }
            sum_v = sum_v + trend * trend;
        }
        penalty = penalty
                - T(0.5) * T(tulpa_st::st_trend_precision(T_st)) * sum_v;
    }
    return penalty;
}

template<class T>
T st_unit_quadratic(const std::vector<T>& field, const ModelData& data, const T& rho) {
    const auto& st = data.spatiotemporal_data;
    const int S = st.n_spatial, nt = st.n_times;
    if (st.type == STType::TYPE_IV) return st_kronecker_temporal_quad(field, data, S, nt, rho);
    T quad(0.0);
    if (st.type == STType::TYPE_I) {
        for (const T& value : field) quad = quad + value * value;
    } else if (st.type == STType::TYPE_II) {
        for (int s = 0; s < S; ++s)
            quad = quad + st_time_cross_form(field.data() + s * nt,
                field.data() + s * nt, nt, st, rho);
    } else if (st.type == STType::TYPE_III) {
        for (int s = 0; s < S; ++s) for (int k = st.adj_row_ptr[s]; k < st.adj_row_ptr[s + 1]; ++k) {
            const int other = st.adj_col_idx[k] - 1;
            if (other <= s) continue;
            for (int t = 0; t < nt; ++t) {
                const T diff = field[s * nt + t] - field[other * nt + t];
                quad = quad + diff * diff;
            }
        }
    }
    return quad;
}

inline int st_full_rank(const ModelData& data) {
    const auto& st = data.spatiotemporal_data;
    const int spatial_rank = st_spatial_rank(data, st.n_spatial);
    if (st.type == STType::TYPE_I) return st.n_params;
    if (st.type == STType::TYPE_II) return st.n_spatial * st_time_rank(st, st.n_times);
    if (st.type == STType::TYPE_III) return spatial_rank * st.n_times;
    return spatial_rank * st_time_rank(st, st.n_times);
}

template<class T>
T st_constrained_density(std::vector<T>& field, const ModelData& data,
    const T& tau, const T& rho) {
    const auto& st = data.spatiotemporal_data;
    const int n = static_cast<int>(field.size());
    std::vector<T> projected = field;
    center_st_margins(projected, st.n_spatial, st.n_times);
    T quad = st_unit_quadratic(projected, data, rho);
    for (int i = 0; i < n; ++i) {
        const T difference = field[i] - projected[i];
        quad = quad + difference * difference;
    }
    const bool rw2 = tulpa_st::st_reads_time_margin(st.type) &&
        st.temporal_type == TemporalType::RW2 && !st.temporal_cyclic;
    const int rank = n - (rw2 ? st.n_spatial - 1 : 0);
    const bool nc = data.st_parameterization == 1;
    T lp = T(0.5 * (nc ? rank - n : rank)) * safe_log(tau)
        - T(0.5) * (nc ? T(1.0) : tau) * quad;
    if (tulpa_st::st_reads_time_margin(st.type) && st.temporal_type == TemporalType::AR1)
        lp = lp + T(0.5 * (st.n_spatial - 1)) * safe_log(
            centered_ar1_precision_det(rho, tulpa_temporal::ar1_one_minus_rho2(rho), st.n_times));
    const T scale = nc ? T(1.0) / safe_sqrt(tau) : T(1.0);
    for (int i = 0; i < n; ++i) projected[i] = projected[i] * scale;
    field.swap(projected);
    return lp;
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

        // PC prior on sigma = 1/sqrt(tau), on the sampled log-precision scale.
        log_post = log_post + log_prior_log_tau_pc(
            log_tau_st, data.st_sigma2_prior_U, data.st_sigma2_prior_alpha);

        // AR1 correlation of the temporal margin, laid out only where the
        // density reads one (st_reads_time_margin). Unread by RW1 / RW2.
        T rho_st = T(0.0);
        if (layout.logit_rho_st_idx >= 0) {
            T logit_rho_st = params[layout.logit_rho_st_idx];
            T u_st = inv_logit(logit_rho_st);
            rho_st = T(2.0) * u_st - T(1.0);  // Map to (-1, 1)

            // Uniform(-1, 1) prior on rho
            // Jacobian for logit((rho+1)/2) transform
            log_post = log_post + T(data.st_rho_prior_a) * safe_log(u_st)
                + T(data.st_rho_prior_b) * safe_log(T(1.0) - u_st);
        }

        // Extract delta parameters
        int n_st_params = layout.st_delta_end - layout.st_delta_start;
        st_delta.resize(n_st_params);
        for (int k = 0; k < n_st_params; k++) {
            st_delta[k] = params[layout.st_delta_start + k];
        }

        int S = data.spatiotemporal_data.n_spatial;
        int T_st = data.spatiotemporal_data.n_times;

        if (data.st_is_hsgp) {
            const T log_sigma2 = params[layout.log_sigma2_st_hsgp_idx];
            const T log_range = params[layout.log_lengthscale_st_hsgp_idx];
            log_post = log_post + log_prior_log_sigma2_pc(log_sigma2,
                data.st_hsgp_sigma2_prior_U, data.st_hsgp_sigma2_prior_alpha)
                - T(0.5) * log_range * log_range;
            if (data.st_parameterization == 1) {
                const auto& st = data.spatiotemporal_data;
                const int normal = st.n_times - (st.temporal_type == TemporalType::RW2 && !st.temporal_cyclic ? 1 : 0);
                for (int j = 0; j < data.st_hsgp_data.m_total; ++j)
                    for (int t = 0; t < normal; ++t) {
                        const T z = st_delta[j * st.n_times + t];
                        log_post = log_post - T(0.5) * z * z;
                    }
                std::vector<T> field(n_st_params);
                tulpa_st_hsgp::nc_forward(st_delta.data(), tau_st, rho_st,
                    safe_exp(log_sigma2), safe_exp(log_range), st, data.st_hsgp_data, field.data(), data.st_hsgp_data.kernel);
                st_delta.swap(field);
                return log_post;
            }
            return log_post + tulpa_st_hsgp::log_density(st_delta.data(), tau_st,
                rho_st, safe_exp(log_sigma2), safe_exp(log_range),
                data.spatiotemporal_data, data.st_hsgp_data, data.st_hsgp_data.kernel);
        }
        if (layout.is_st_gp) {
            const T log_space = params[layout.log_phi_st_space_idx];
            const T log_time = params[layout.log_phi_st_time_idx];
            const T space = safe_exp(log_space), time = safe_exp(log_time);
            const auto& st = data.spatiotemporal_data;
            if (data.st_uniform_ranges) {
                if (get_value(space) < st.phi_space_prior_lower || get_value(space) > st.phi_space_prior_upper ||
                    get_value(time) < st.phi_time_prior_lower || get_value(time) > st.phi_time_prior_upper)
                    return T(-std::numeric_limits<double>::infinity());
                log_post = log_post + log_space + log_time;
            } else {
                log_post = log_post + log_prior_range_pc_at_log(log_space,
                    data.st_phi_space_prior_lower, data.st_phi_space_prior_alpha) + log_space
                    + log_prior_range_pc_at_log(log_time,
                    data.st_phi_time_prior_lower, data.st_phi_time_prior_alpha) + log_time;
            }
            if (data.st_parameterization == 1) {
                std::vector<T> field(n_st_params, T(0.0));
                if (!tulpa_st_gp::st_gp_nc_forward(st_delta.data(), T(1.0) / tau_st,
                        space, time, st, field.data()))
                    return T(-std::numeric_limits<double>::infinity());
                for (const T& z : st_delta)
                    log_post = log_post - T(0.5) * z * z - T(0.5 * std::log(2.0 * std::acos(-1.0)));
                st_delta.swap(field);
            } else {
                log_post = log_post + tulpa_st_gp::st_gp_nngp_log_lik(
                    st_delta.data(), T(1.0) / tau_st, space, time, st);
            }
            T normalizer;
            if (!tulpa_st_gp::st_gp_margin_log_normalizer(T(1.0) / tau_st,
                    space, time, st, &normalizer))
                return T(-std::numeric_limits<double>::infinity());
            return log_post + st_sum_to_zero_penalty(st_delta, S, T_st) + normalizer;
        }
        if (data.st_margin_centering)
            return log_post + st_constrained_density(st_delta, data, tau_st, rho_st);

        // NC reparameterization for Type IV
        const bool st_use_nc = tulpa_st::st_non_centered(
            data.spatiotemporal_data.type, data.st_parameterization);

        if (st_use_nc) {
            // Forward transform: delta = z / sqrt(tau_st)
            T inv_scale = T(1.0) / safe_sqrt(tau_st);
            std::vector<T> st_delta_nc(n_st_params);
            for (int k = 0; k < n_st_params; k++) {
                st_delta_nc[k] = st_delta[k] * inv_scale;
            }

            // NC prior: -0.5 * z^T (Q_s (x) Q_t) z  (tau-free GMRF)
            // Type IV Kronecker quadratic form on z (with tau=1)
            T nc_quad = st_kronecker_temporal_quad(st_delta, data, S, T_st, rho_st);
            log_post = log_post - T(0.5) * nc_quad;

            // Rank term with actual tau and Jacobian correction. The
            // hyperparameter-dependent part of log|Q_s (x) Q_t|_+ is
            // rank_space * log|Q_t|, the Kronecker pseudo-determinant's time
            // factor; the delta = z / sqrt(tau) Jacobian does not touch it.
            int rank_space = st_spatial_rank(data, S);
            int total_rank =
                rank_space * st_time_rank(data.spatiotemporal_data, T_st);
            int ST_total = S * T_st;
            log_post = log_post + T(0.5 * (total_rank - ST_total)) * safe_log(tau_st)
                     + T(0.5 * rank_space) *
                       st_time_log_det(data.spatiotemporal_data, rho_st);

            // Sum-to-zero on reconstructed delta
            log_post = log_post + st_sum_to_zero_penalty(
                st_delta_nc, S, T_st,
                tulpa_st::st_needs_trend_pin(
                    data.spatiotemporal_data.type,
                    data.spatiotemporal_data.temporal_type,
                    data.spatiotemporal_data.temporal_cyclic));

            // Replace st_delta with reconstructed delta for observation loop
            st_delta = std::move(st_delta_nc);

        } else {
            const auto& st = data.spatiotemporal_data;
            const int determinant_power = st.type == STType::TYPE_II ? S :
                (st.type == STType::TYPE_IV ? st_spatial_rank(data, S) : 0);
            log_post = log_post + T(0.5 * st_full_rank(data)) * safe_log(tau_st)
                + T(0.5 * determinant_power) * st_time_log_det(st, rho_st)
                - T(0.5) * tau_st * st_unit_quadratic(st_delta, data, rho_st);

            // Soft sum-to-zero constraint
            log_post = log_post + st_sum_to_zero_penalty(
                st_delta, S, T_st,
                tulpa_st::st_needs_trend_pin(
                    data.spatiotemporal_data.type,
                    data.spatiotemporal_data.temporal_type,
                    data.spatiotemporal_data.temporal_cyclic));
        }
    }

    return log_post;
}


} // namespace priors
} // namespace tulpa

#endif // TULPA_PRIORS_ST_H
