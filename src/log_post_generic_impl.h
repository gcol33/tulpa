// log_post_generic_impl.h
// Generic multi-process log-posterior implementation for LikelihoodSpec models.
//
// This header is included from log_post_impl.h after the legacy/shared prior
// helpers have been declared and while namespace tulpa is open.

#ifndef TULPA_LOG_POST_GENERIC_IMPL_H
#define TULPA_LOG_POST_GENERIC_IMPL_H

// Maximum number of processes supported (stack-allocated eta array)
constexpr int MAX_PROCESSES = 8;

template<typename T>
using LikelihoodFnT = T(*)(
    int i,                           // Observation index
    const T* eta,                    // Linear predictor values [n_processes]
    const T& logit_zi,               // ZI linear predictor (0 if no ZI)
    const T& logit_oi,               // OI linear predictor (0 if no OI)
    const std::vector<T>& params,    // Full parameter vector
    const ModelData& data,           // Generic model data
    const ParamLayout& layout,       // Parameter layout
    const void* model_data           // Model-specific response data
);

template<typename T>
T compute_log_post_generic(
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout,
    LikelihoodFnT<T> likelihood_fn,
    const void* model_response_data,
    bool skip_obs_loop = false
) {
    T log_post = T(0.0);
    const int np = data.n_processes;

    if (np > MAX_PROCESSES) {
        return T(-INFINITY);
    }

    std::vector<const T*> beta(np);
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int k = 0; k < np; k++) {
        beta[k] = &params[layout.process_beta_start[k]];
        for (int j = 0; j < layout.process_beta_count[k]; j++) {
            T b = params[layout.process_beta_start[k] + j];
            log_post = log_post + log_prior_normal(b, tau_beta);
        }
    }

    std::vector<T> re_vals;
    std::vector<int> re_term_offsets;
    if (layout.has_re) {
        log_post = log_post + priors::compute_re_prior(params, data, layout,
                                                        re_vals, re_term_offsets);
    }

    const T* phi_spatial = nullptr;
    T tau_spatial = T(1.0);
    T sigma_s_bym2 = T(1.0), sigma_u_bym2 = T(1.0);
    const T* theta_bym2 = nullptr;

    if (layout.has_spatial) {
        log_post = log_post + priors::compute_spatial_icar_bym2_prior(
            params, data, layout,
            phi_spatial, tau_spatial, sigma_s_bym2, sigma_u_bym2, theta_bym2);
    }

    std::vector<T> gp_w;
    if (layout.is_gp && data.has_gp) {
        log_post = log_post + priors::compute_gp_spatial_prior(
            params, data, layout, gp_w);
    }

    std::vector<T> ms_gp_effect;
    if (layout.is_multiscale_gp && data.has_multiscale_gp) {
        log_post = log_post + priors::compute_multiscale_gp_prior(
            params, data, layout, ms_gp_effect);
    }

    std::vector<T> hsgp_f;
    if (layout.is_hsgp && data.has_hsgp) {
        log_post = log_post + priors::compute_hsgp_spatial_prior(
            params, data, layout, hsgp_f);
    }

    std::vector<T> phi_temporal;
    T tau_temporal = T(1.0), rho_ar1 = T(0.5);
    T sigma2_temporal_gp = T(1.0), phi_temporal_gp = T(1.0);

    if (layout.has_temporal) {
        int n_temporal = layout.temporal_end - layout.temporal_start;
        phi_temporal.resize(n_temporal);
        for (int t = 0; t < n_temporal; t++) {
            phi_temporal[t] = params[layout.temporal_start + t];
        }
        log_post = log_post + priors::compute_temporal_prior(
            params, data, layout, phi_temporal,
            tau_temporal, rho_ar1, sigma2_temporal_gp, phi_temporal_gp);
    }

    std::vector<T> ms_temporal_eta;
    if (layout.has_multiscale_temporal && data.has_multiscale_temporal) {
        log_post = log_post + priors::compute_multiscale_temporal_prior(
            params, data, layout, ms_temporal_eta);
    }

    std::vector<T> svc_eta;
    if (layout.has_svc && data.has_svc) {
        log_post = log_post + priors::compute_svc_prior(
            params, data, layout, svc_eta);
    }

    std::vector<T> tvc_eta;
    if (layout.has_tvc && data.has_tvc) {
        log_post = log_post + priors::compute_tvc_prior(
            params, data, layout, tvc_eta);
    }

    std::vector<T> latent_eta;
    if (layout.has_latent && data.has_latent) {
        log_post = log_post + priors::compute_latent_prior(
            params, data, layout, latent_eta);
    }

    std::vector<T> st_delta;
    if (layout.has_spatiotemporal && data.has_spatiotemporal) {
        log_post = log_post + priors::compute_st_prior(
            params, data, layout, st_delta);
    }

    std::vector<T> beta_zi, beta_oi;
    log_post = log_post + priors::compute_zi_oi_prior(
        params, data, layout, beta_zi, beta_oi);

    std::vector<std::vector<T>> eta_fixed(np);
    for (int k = 0; k < np; k++) {
        const auto& proc = data.processes[k];
        eta_fixed[k].assign(data.N, T(0.0));
        if (proc.p == 0) continue;
        if constexpr (std::is_same_v<T, double>) {
            tulpa_linalg::matvec(proc.X_flat.data(), beta[k],
                                 eta_fixed[k].data(), data.N, proc.p);
        } else {
            for (int i = 0; i < data.N; i++) {
                T s = T(0.0);
                const double* row = &proc.X_flat[i * proc.p];
                for (int j = 0; j < proc.p; j++) {
                    s = s + T(row[j]) * beta[k][j];
                }
                eta_fixed[k][i] = s;
            }
        }
    }

    if (skip_obs_loop) {
        return log_post;
    }

    for (int i = 0; i < data.N; i++) {
        T eta[MAX_PROCESSES];
        for (int k = 0; k < np; k++) {
            eta[k] = eta_fixed[k][i];
        }

        if (layout.has_re && !re_vals.empty()) {
            int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;
            T re_contrib_i = T(0.0);

            if (layout.has_re_slopes && !data.re_group_multi_flat.empty()) {
                for (int t = 0; t < n_terms; t++) {
                    int flat_idx = i * n_terms + t;
                    if (flat_idx >= (int)data.re_group_multi_flat.size()) continue;
                    int re_group_idx = data.re_group_multi_flat[flat_idx];
                    if (re_group_idx > 0) {
                        int g = re_group_idx - 1;
                        int n_coefs_t = layout.re_n_coefs_multi[t];
                        int off = re_term_offsets[t];
                        T term_contrib = re_vals[off + g * n_coefs_t];
                        int n_slopes = n_coefs_t - 1;
                        if (n_slopes > 0 && t < (int)data.re_slope_matrices.size() &&
                            !data.re_slope_matrices[t].empty()) {
                            for (int s = 0; s < n_slopes; s++) {
                                double x_slope = data.re_slope_matrices[t][i * n_slopes + s];
                                term_contrib = term_contrib
                                    + re_vals[off + g * n_coefs_t + 1 + s] * T(x_slope);
                            }
                        }
                        re_contrib_i = re_contrib_i + term_contrib;
                    }
                }
            } else if (n_terms > 1 && !data.re_group_multi_flat.empty()) {
                for (int t = 0; t < n_terms; t++) {
                    int flat_idx = i * n_terms + t;
                    if (flat_idx >= (int)data.re_group_multi_flat.size()) continue;
                    int group_idx = data.re_group_multi_flat[flat_idx];
                    if (group_idx > 0) {
                        int g = group_idx - 1;
                        int off = re_term_offsets[t];
                        re_contrib_i = re_contrib_i + re_vals[off + g];
                    }
                }
            } else if (!data.re_group.empty() && data.re_group[i] > 0) {
                int g = data.re_group[i] - 1;
                re_contrib_i = re_vals[g];
            }

            for (int k = 0; k < np; k++) {
                if (data.sharing.re[k]) eta[k] = eta[k] + re_contrib_i;
            }
        }

        if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
            int s = data.spatial_group[i] - 1;
            T spatial_effect;
            if (layout.is_bym2) {
                T scaled_phi = phi_spatial[s] * T(data.bym2_scale_factor);
                spatial_effect = sigma_s_bym2 * scaled_phi + sigma_u_bym2 * theta_bym2[s];
            } else {
                spatial_effect = phi_spatial[s];
            }
            for (int k = 0; k < np; k++) {
                if (data.sharing.spatial[k]) eta[k] = eta[k] + spatial_effect;
            }
        }

        if (layout.is_gp && !gp_w.empty()) {
            int loc_i = data.gp_data.obs_to_loc[i];
            T gp_effect = gp_w[loc_i];
            for (int k = 0; k < np; k++) {
                if (data.sharing.spatial[k]) eta[k] = eta[k] + gp_effect;
            }
        }

        if (layout.is_multiscale_gp && !ms_gp_effect.empty()) {
            T msgp_effect = ms_gp_effect[i];
            for (int k = 0; k < np; k++) {
                if (data.sharing.spatial[k]) eta[k] = eta[k] + msgp_effect;
            }
        }

        if (layout.is_hsgp && !hsgp_f.empty()) {
            T hsgp_effect = hsgp_f[i];
            for (int k = 0; k < np; k++) {
                if (data.sharing.spatial[k]) eta[k] = eta[k] + hsgp_effect;
            }
        }

        if (layout.has_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0 &&
            !phi_temporal.empty()) {
            int t = data.temporal_time_idx[i] - 1;
            int g = (!data.temporal_group_idx.empty() && i < (int)data.temporal_group_idx.size())
                    ? (data.temporal_group_idx[i] - 1) : 0;
            int idx_t = g * data.n_times + t;
            if (idx_t >= 0 && idx_t < (int)phi_temporal.size()) {
                T temporal_effect = phi_temporal[idx_t];
                for (int k = 0; k < np; k++) {
                    if (data.sharing.temporal[k]) eta[k] = eta[k] + temporal_effect;
                }
            }
        }

        if (layout.has_multiscale_temporal && !ms_temporal_eta.empty()) {
            T ms_temp_effect = ms_temporal_eta[i];
            for (int k = 0; k < np; k++) {
                if (data.sharing.temporal[k]) eta[k] = eta[k] + ms_temp_effect;
            }
        }

        if (layout.has_svc && !svc_eta.empty() && i < (int)svc_eta.size()) {
            T svc_effect = svc_eta[i];
            for (int k = 0; k < np; k++) {
                if (data.sharing.svc[k]) eta[k] = eta[k] + svc_effect;
            }
        }

        if (layout.has_tvc && !tvc_eta.empty()) {
            T tvc_effect = tvc_eta[i];
            for (int k = 0; k < np; k++) {
                if (data.sharing.tvc[k]) eta[k] = eta[k] + tvc_effect;
            }
        }

        if (layout.has_latent && !latent_eta.empty() && i < (int)latent_eta.size()) {
            T latent_effect = latent_eta[i];
            for (int k = 0; k < np; k++) {
                if (data.sharing.latent[k]) eta[k] = eta[k] + latent_effect;
            }
        }

        if (layout.has_spatiotemporal && !st_delta.empty()) {
            T st_effect = T(0.0);
            if (data.st_is_hsgp) {
                int t_st = data.spatiotemporal_data.t_idx[i] - 1;
                int M = data.st_hsgp_data.m_total;
                int T_st = data.spatiotemporal_data.n_times;
                for (int j = 0; j < M; j++) {
                    st_effect = st_effect
                        + T(data.st_hsgp_data.phi_flat[i * M + j])
                        * st_delta[j * T_st + t_st];
                }
            } else {
                int st_idx = data.spatiotemporal_data.st_flat[i];
                if (st_idx > 0) st_effect = st_delta[st_idx - 1];
            }
            for (int k = 0; k < np; k++) {
                if (data.sharing.st[k]) eta[k] = eta[k] + st_effect;
            }
        }

        T logit_zi = T(0.0);
        T logit_oi = T(0.0);
        if (layout.has_zi && data.p_zi > 0) {
            for (int j = 0; j < data.p_zi; j++) {
                logit_zi = logit_zi + T(data.X_zi_flat[i * data.p_zi + j]) * beta_zi[j];
            }
        }
        if (layout.has_oi && data.p_oi > 0) {
            for (int j = 0; j < data.p_oi; j++) {
                logit_oi = logit_oi + T(data.X_oi_flat[i * data.p_oi + j]) * beta_oi[j];
            }
        }

        T ll_i = likelihood_fn(i, eta, logit_zi, logit_oi,
                               params, data, layout, model_response_data);
        log_post = log_post + ll_i;
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
