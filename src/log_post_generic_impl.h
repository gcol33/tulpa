// log_post_generic_impl.h
// Generic multi-process log-posterior implementation for LikelihoodSpec models.
//
// Included from log_post_impl.h while namespace tulpa is open. The legacy
// log-posterior still owns the shared math/prior definitions this code uses.

#ifndef TULPA_LOG_POST_GENERIC_IMPL_H
#define TULPA_LOG_POST_GENERIC_IMPL_H

#include <Rcpp.h>  // Rcpp::stop for offset-length validation

constexpr int MAX_PROCESSES = 8;

template<typename T>
using LikelihoodFnT = T(*)(
    int i,
    const T* eta,
    const T& logit_zi,
    const T& logit_oi,
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout,
    const void* model_data
);

template<typename T>
struct GenericLogPostState {
    std::vector<const T*> beta;
    std::vector<T> re_vals;
    std::vector<int> re_term_offsets;
    const T* phi_spatial = nullptr;
    T tau_spatial = T(1.0);
    T sigma_s_bym2 = T(1.0);
    T sigma_u_bym2 = T(1.0);
    const T* theta_bym2 = nullptr;
    std::vector<T> gp_w;
    std::vector<T> ms_gp_effect;
    std::vector<T> hsgp_f;
    std::vector<T> spde_w;
    std::vector<T> phi_temporal;
    T tau_temporal = T(1.0);
    T rho_ar1 = T(0.5);
    T sigma2_temporal_gp = T(1.0);
    T phi_temporal_gp = T(1.0);
    std::vector<T> ms_temporal_eta;
    std::vector<T> svc_eta;
    std::vector<T> tvc_eta;
    std::vector<T> latent_eta;
    std::vector<T> st_delta;
    std::vector<T> beta_zi;
    std::vector<T> beta_oi;
    std::vector<std::vector<T>> eta_fixed;
};

template<typename T>
static inline void add_to_shared_processes(
    T* eta,
    const std::vector<bool>& sharing,
    int n_processes,
    const T& effect
) {
    for (int k = 0; k < n_processes; k++) {
        if (sharing[k]) eta[k] = eta[k] + effect;
    }
}

template<typename T>
static T initialize_generic_state(
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout,
    GenericLogPostState<T>& state
) {
    T log_post = T(0.0);
    const int np = data.n_processes;

    state.beta.resize(np);
    const double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int k = 0; k < np; k++) {
        state.beta[k] = &params[layout.process_beta_start[k]];
        for (int j = 0; j < layout.process_beta_count[k]; j++) {
            log_post = log_post + log_prior_normal(
                params[layout.process_beta_start[k] + j], tau_beta);
        }
    }

    if (layout.has_re) {
        log_post = log_post + priors::compute_re_prior(
            params, data, layout, state.re_vals, state.re_term_offsets);
    }

    if (layout.has_spatial) {
        log_post = log_post + priors::compute_spatial_icar_bym2_prior(
            params, data, layout, state.phi_spatial, state.tau_spatial,
            state.sigma_s_bym2, state.sigma_u_bym2, state.theta_bym2);
    }

    if (layout.is_gp && data.has_gp) {
        log_post = log_post + priors::compute_gp_spatial_prior(
            params, data, layout, state.gp_w);
    }

    if (layout.is_multiscale_gp && data.has_multiscale_gp) {
        log_post = log_post + priors::compute_multiscale_gp_prior(
            params, data, layout, state.ms_gp_effect);
    }

    if (layout.is_hsgp && data.has_hsgp) {
        log_post = log_post + priors::compute_hsgp_spatial_prior(
            params, data, layout, state.hsgp_f);
    }

    if (layout.is_spde && data.has_spde) {
        log_post = log_post + priors::compute_spde_prior(
            params, data, layout, state.spde_w);
        // PC prior on (range, sigma), joint-NUTS mode only (stub returns 0
        // when joint_hypers == false). Lives outside compute_spde_prior so
        // the hyper density is computed regardless of whether the latent
        // block contributes a centered or non-centered prior.
        log_post = log_post + priors::compute_spde_hyper_prior<T>(
            params, data, layout);
    }

    if (layout.has_temporal) {
        const int n_temporal = layout.temporal_end - layout.temporal_start;
        state.phi_temporal.resize(n_temporal);
        for (int t = 0; t < n_temporal; t++) {
            state.phi_temporal[t] = params[layout.temporal_start + t];
        }
        log_post = log_post + priors::compute_temporal_prior(
            params, data, layout, state.phi_temporal, state.tau_temporal,
            state.rho_ar1, state.sigma2_temporal_gp, state.phi_temporal_gp);
    }

    if (layout.has_multiscale_temporal && data.has_multiscale_temporal) {
        log_post = log_post + priors::compute_multiscale_temporal_prior(
            params, data, layout, state.ms_temporal_eta);
    }

    if (layout.has_svc && data.has_svc) {
        log_post = log_post + priors::compute_svc_prior(
            params, data, layout, state.svc_eta);
    }

    if (layout.has_tvc && data.has_tvc) {
        log_post = log_post + priors::compute_tvc_prior(
            params, data, layout, state.tvc_eta);
    }

    if (layout.has_latent && data.has_latent) {
        log_post = log_post + priors::compute_latent_prior(
            params, data, layout, state.latent_eta);
    }

    if (layout.has_spatiotemporal && data.has_spatiotemporal) {
        log_post = log_post + priors::compute_st_prior(
            params, data, layout, state.st_delta);
    }

    return log_post + priors::compute_zi_oi_prior(
        params, data, layout, state.beta_zi, state.beta_oi);
}

template<typename T>
static void precompute_generic_fixed_eta(
    const ModelData& data,
    GenericLogPostState<T>& state
) {
    const int np = data.n_processes;
    state.eta_fixed.resize(np);

    for (int k = 0; k < np; k++) {
        const auto& proc = data.processes[k];
        state.eta_fixed[k].assign(data.N, T(0.0));

        if (proc.p > 0) {
            if constexpr (std::is_same_v<T, double>) {
                tulpa_linalg::matvec(proc.X_flat.data(), state.beta[k],
                                     state.eta_fixed[k].data(), data.N, proc.p);
            } else {
                for (int i = 0; i < data.N; i++) {
                    T eta_i = T(0.0);
                    const double* row = &proc.X_flat[i * proc.p];
                    for (int j = 0; j < proc.p; j++) {
                        eta_i = eta_i + T(row[j]) * state.beta[k][j];
                    }
                    state.eta_fixed[k][i] = eta_i;
                }
            }
        }

        // Optional per-process offset on the linear predictor. Empty means
        // "no offset" (treated as zero for every observation). When present
        // it must have length data.N — checked here so a malformed length
        // raises a deterministic error instead of overrunning eta_fixed.
        if (!proc.offset.empty()) {
            if ((int)proc.offset.size() != data.N) {
                Rcpp::stop("ProcessData[%d]: offset length (%d) must equal "
                           "ModelData::N (%d)",
                           k, (int)proc.offset.size(), data.N);
            }
            for (int i = 0; i < data.N; i++) {
                state.eta_fixed[k][i] = state.eta_fixed[k][i] + T(proc.offset[i]);
            }
        }
    }
}

template<typename T>
static T generic_re_effect(
    int i,
    const ModelData& data,
    const ParamLayout& layout,
    const GenericLogPostState<T>& state
) {
    if (!layout.has_re || state.re_vals.empty()) return T(0.0);

    const int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;
    T effect = T(0.0);

    if (layout.has_re_slopes && !data.re_group_multi_flat.empty()) {
        for (int t = 0; t < n_terms; t++) {
            const int flat_idx = i * n_terms + t;
            if (flat_idx >= (int)data.re_group_multi_flat.size()) continue;

            const int group_idx = data.re_group_multi_flat[flat_idx];
            if (group_idx <= 0) continue;

            const int g = group_idx - 1;
            const int n_coefs_t = layout.re_n_coefs_multi[t];
            const int off = state.re_term_offsets[t];
            T term_effect = state.re_vals[off + g * n_coefs_t];
            const int n_slopes = n_coefs_t - 1;

            if (n_slopes > 0 && t < (int)data.re_slope_matrices.size() &&
                !data.re_slope_matrices[t].empty()) {
                for (int s = 0; s < n_slopes; s++) {
                    const double x_slope = data.re_slope_matrices[t][i * n_slopes + s];
                    term_effect = term_effect
                        + state.re_vals[off + g * n_coefs_t + 1 + s] * T(x_slope);
                }
            }
            effect = effect + term_effect;
        }
    } else if (n_terms > 1 && !data.re_group_multi_flat.empty()) {
        for (int t = 0; t < n_terms; t++) {
            const int flat_idx = i * n_terms + t;
            if (flat_idx >= (int)data.re_group_multi_flat.size()) continue;

            const int group_idx = data.re_group_multi_flat[flat_idx];
            if (group_idx > 0) {
                effect = effect + state.re_vals[state.re_term_offsets[t] + group_idx - 1];
            }
        }
    } else if (!data.re_group.empty() && data.re_group[i] > 0) {
        effect = state.re_vals[data.re_group[i] - 1];
    }

    return effect;
}

template<typename T>
static void add_generic_spatial_effect(
    int i,
    T* eta,
    const ModelData& data,
    const ParamLayout& layout,
    const GenericLogPostState<T>& state
) {
    if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
        const int s = data.spatial_group[i] - 1;
        const T effect = layout.is_bym2
            ? state.sigma_s_bym2 * state.phi_spatial[s] * T(data.bym2_scale_factor)
                + state.sigma_u_bym2 * state.theta_bym2[s]
            : state.phi_spatial[s];
        add_to_shared_processes(eta, data.sharing.spatial, data.n_processes, effect);
    }

    if (layout.is_gp && !state.gp_w.empty()) {
        add_to_shared_processes(
            eta, data.sharing.spatial, data.n_processes,
            state.gp_w[data.gp_data.obs_to_loc[i]]);
    }

    if (layout.is_multiscale_gp && !state.ms_gp_effect.empty()) {
        add_to_shared_processes(
            eta, data.sharing.spatial, data.n_processes, state.ms_gp_effect[i]);
    }

    if (layout.is_hsgp && !state.hsgp_f.empty()) {
        add_to_shared_processes(
            eta, data.sharing.spatial, data.n_processes, state.hsgp_f[i]);
    }

    if (layout.is_spde && !state.spde_w.empty()) {
        // eta_i += sum_j A_ij * w_j. The projection A is sparse: each obs
        // is a convex combination of ~3 triangle-vertex weights stored in
        // a_rows[i]. Empty a_rows[i] means the observation falls outside
        // the mesh and contributes no spatial effect.
        const auto& row = data.spde_data.a_rows[i];
        if (!row.empty()) {
            T effect = T(0.0);
            for (const auto& ae : row) {
                effect = effect + T(ae.weight) * state.spde_w[ae.mesh_idx];
            }
            add_to_shared_processes(
                eta, data.sharing.spatial, data.n_processes, effect);
        }
    }
}

template<typename T>
static void add_generic_temporal_effect(
    int i,
    T* eta,
    const ModelData& data,
    const ParamLayout& layout,
    const GenericLogPostState<T>& state
) {
    if (layout.has_temporal && !data.temporal_time_idx.empty() &&
        i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0 &&
        !state.phi_temporal.empty()) {
        const int t = data.temporal_time_idx[i] - 1;
        const int g = (!data.temporal_group_idx.empty() &&
                       i < (int)data.temporal_group_idx.size())
            ? (data.temporal_group_idx[i] - 1) : 0;
        const int idx_t = g * data.n_times + t;
        if (idx_t >= 0 && idx_t < (int)state.phi_temporal.size()) {
            add_to_shared_processes(
                eta, data.sharing.temporal, data.n_processes,
                state.phi_temporal[idx_t]);
        }
    }

    if (layout.has_multiscale_temporal && !state.ms_temporal_eta.empty()) {
        add_to_shared_processes(
            eta, data.sharing.temporal, data.n_processes, state.ms_temporal_eta[i]);
    }
}

template<typename T>
static void add_generic_named_effects(
    int i,
    T* eta,
    const ModelData& data,
    const ParamLayout& layout,
    const GenericLogPostState<T>& state
) {
    const T re = generic_re_effect(i, data, layout, state);
    if (layout.has_re && !state.re_vals.empty()) {
        add_to_shared_processes(eta, data.sharing.re, data.n_processes, re);
    }

    add_generic_spatial_effect(i, eta, data, layout, state);
    add_generic_temporal_effect(i, eta, data, layout, state);

    if (layout.has_svc && !state.svc_eta.empty() && i < (int)state.svc_eta.size()) {
        add_to_shared_processes(eta, data.sharing.svc, data.n_processes, state.svc_eta[i]);
    }

    if (layout.has_tvc && !state.tvc_eta.empty()) {
        add_to_shared_processes(eta, data.sharing.tvc, data.n_processes, state.tvc_eta[i]);
    }

    if (layout.has_latent && !state.latent_eta.empty() &&
        i < (int)state.latent_eta.size()) {
        add_to_shared_processes(
            eta, data.sharing.latent, data.n_processes, state.latent_eta[i]);
    }
}

template<typename T>
static void add_generic_st_effect(
    int i,
    T* eta,
    const ModelData& data,
    const ParamLayout& layout,
    const GenericLogPostState<T>& state
) {
    if (!layout.has_spatiotemporal || state.st_delta.empty()) return;

    T effect = T(0.0);
    if (data.st_is_hsgp) {
        const int t_st = data.spatiotemporal_data.t_idx[i] - 1;
        const int M = data.st_hsgp_data.m_total;
        const int T_st = data.spatiotemporal_data.n_times;
        for (int j = 0; j < M; j++) {
            effect = effect
                + T(data.st_hsgp_data.phi_flat[i * M + j])
                * state.st_delta[j * T_st + t_st];
        }
    } else {
        const int st_idx = data.spatiotemporal_data.st_flat[i];
        if (st_idx > 0) effect = state.st_delta[st_idx - 1];
    }
    add_to_shared_processes(eta, data.sharing.st, data.n_processes, effect);
}

template<typename T>
static void generic_zi_oi_logits(
    int i,
    const ModelData& data,
    const ParamLayout& layout,
    const GenericLogPostState<T>& state,
    T& logit_zi,
    T& logit_oi
) {
    logit_zi = T(0.0);
    logit_oi = T(0.0);

    if (layout.has_zi && data.p_zi > 0) {
        for (int j = 0; j < data.p_zi; j++) {
            logit_zi = logit_zi + T(data.X_zi_flat[i * data.p_zi + j]) * state.beta_zi[j];
        }
    }

    if (layout.has_oi && data.p_oi > 0) {
        for (int j = 0; j < data.p_oi; j++) {
            logit_oi = logit_oi + T(data.X_oi_flat[i * data.p_oi + j]) * state.beta_oi[j];
        }
    }
}

template<typename T>
static T compute_generic_likelihood_sum(
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout,
    LikelihoodFnT<T> likelihood_fn,
    const void* model_response_data,
    const GenericLogPostState<T>& state
) {
    T log_lik = T(0.0);

    for (int i = 0; i < data.N; i++) {
        T eta[MAX_PROCESSES];
        for (int k = 0; k < data.n_processes; k++) {
            eta[k] = state.eta_fixed[k][i];
        }

        add_generic_named_effects(i, eta, data, layout, state);
        add_generic_st_effect(i, eta, data, layout, state);

        T logit_zi;
        T logit_oi;
        generic_zi_oi_logits(i, data, layout, state, logit_zi, logit_oi);

        log_lik = log_lik + likelihood_fn(
            i, eta, logit_zi, logit_oi, params, data, layout, model_response_data);
    }

    return log_lik;
}

template<typename T>
T compute_log_post_generic(
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout,
    LikelihoodFnT<T> likelihood_fn,
    const void* model_response_data,
    bool skip_obs_loop = false
) {
    if (data.n_processes > MAX_PROCESSES) {
        return T(-INFINITY);
    }

    GenericLogPostState<T> state;
    T log_post = initialize_generic_state(params, data, layout, state);
    precompute_generic_fixed_eta(data, state);

    if (!skip_obs_loop) {
        log_post = log_post + compute_generic_likelihood_sum(
            params, data, layout, likelihood_fn, model_response_data, state);
    }

    return log_post;
}

inline double compute_log_post_generic_spec_double(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop = false
) {
    if (data.n_processes <= 0 || data.likelihood_spec == nullptr) {
        return -INFINITY;
    }

    const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);
    if (spec->ll_double == nullptr) {
        return -INFINITY;
    }

    double log_post = compute_log_post_generic<double>(
        params, data, layout, spec->ll_double, data.model_response_data, skip_obs_loop);
    if (spec->extra_prior != nullptr) {
        log_post += spec->extra_prior(params, layout, data.model_response_data);
    }
    return log_post;
}

#endif  // TULPA_LOG_POST_GENERIC_IMPL_H
