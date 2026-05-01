// =====================================================================
// Multiscale temporal gradient (hand-coded, rows 15, 45, 75)
// Uses analytical gradients from hmc_multiscale_temporal_grad.h
// Supports optional ICAR/BYM2 spatial
// =====================================================================

void compute_gradient_ms_temporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- Spatial parameters (ICAR/BYM2 if present) ---
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0;
    double rho_bym2 = 0.5;
    const double* theta_bym2 = nullptr;
    if (layout.has_spatial) {
        if (!layout.is_bym2) tau_spatial = std::exp(params[layout.log_tau_spatial_idx]);
        spatial_phi = &params[layout.spatial_start];
        if (layout.is_bym2) {
            double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
            double logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
            sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
            theta_bym2 = &params[layout.theta_bym2_start];
        }
    }

    // --- Multiscale temporal parameters ---
    const auto& mst = data.multiscale_temporal_data;
    int n_trend = layout.trend_end - layout.trend_start;
    int n_seasonal = layout.seasonal_end - layout.seasonal_start;
    int n_short = layout.short_term_end - layout.short_term_start;

    const double* trend = (n_trend > 0) ? &params[layout.trend_start] : nullptr;
    const double* seasonal = (n_seasonal > 0) ? &params[layout.seasonal_start] : nullptr;
    const double* short_term = (n_short > 0) ? &params[layout.short_term_start] : nullptr;

    double sigma2_trend = (n_trend > 0) ? std::exp(params[layout.log_sigma2_trend_idx]) : 1.0;
    double sigma2_seasonal = (n_seasonal > 0) ? std::exp(params[layout.log_sigma2_seasonal_idx]) : 1.0;
    double sigma2_short = (n_short > 0) ? std::exp(params[layout.log_sigma2_short_idx]) : 1.0;
    double rho_short = 0.5;
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        double logit_rho = params[layout.logit_rho_short_idx];
        double u = 1.0 / (1.0 + std::exp(-logit_rho));
        rho_short = 2.0 * u - 1.0;
    }

    // --- Shared base priors + feature-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // Spatial prior gradients (ICAR/BYM2)
    if (layout.has_spatial && !layout.is_bym2) {
        grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0) - data.tau_spatial_rate * tau_spatial + 1.0;
    }
    if (layout.is_bym2) {
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        double ratio = sigma_total / data.sigma_re_scale;
        grad[layout.log_sigma_bym2_idx] = -2.0 * ratio * ratio / (1.0 + ratio * ratio) + 1.0;
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }

    // PC priors on multiscale temporal variances
    if (n_trend > 0) {
        grad[layout.log_sigma2_trend_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
            sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha) + 1.0;
    }
    if (n_seasonal > 0) {
        grad[layout.log_sigma2_seasonal_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
            sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha) + 1.0;
    }
    if (n_short > 0) {
        grad[layout.log_sigma2_short_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
            sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha) + 1.0;
    }
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        double u = (rho_short + 1.0) / 2.0;
        grad[layout.logit_rho_short_idx] = 2.0 * (1.0 - 2.0 * u);
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // Pre-assemble temporal effect per time point (avoids per-obs modulo)
    const int n_times = mst.n_times;
    std::vector<double> ms_effect_by_time(n_times, 0.0);
    for (int t = 0; t < n_times; t++) {
        if (trend != nullptr && t < n_trend) ms_effect_by_time[t] += trend[t];
        if (seasonal != nullptr && mst.seasonal_period > 0) {
            ms_effect_by_time[t] += seasonal[t % mst.seasonal_period];
        }
        if (short_term != nullptr && t < n_short) ms_effect_by_time[t] += short_term[t];
    }

    // Feature-specific eta: spatial + multiscale temporal
    std::vector<double> grad_trend_lik(n_trend, 0.0);
    std::vector<double> grad_seasonal_lik(n_seasonal, 0.0);
    std::vector<double> grad_short_lik(n_short, 0.0);
    std::vector<double> grad_spatial_lik;
    if (layout.has_spatial) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_theta_lik;
    if (layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);

    std::vector<int> obs_s_unit(pre.N, -1);
    std::vector<int> obs_t_idx(pre.N, -1);
    for (int i = 0; i < pre.N; i++) {
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            obs_s_unit[i] = s_unit;
            double spatial_eff;
            if (layout.is_bym2) {
                spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit];
            } else {
                spatial_eff = spatial_phi[s_unit];
            }
            vec_grad_ws.eta_num[i] += spatial_eff;
            if (!pre.is_binomial) vec_grad_ws.eta_denom[i] += spatial_eff;
        }
        if (!mst.time_index.empty() && i < (int)mst.time_index.size() && mst.time_index[i] > 0) {
            int t_idx = mst.time_index[i] - 1;
            obs_t_idx[i] = t_idx;
            vec_grad_ws.eta_num[i] += ms_effect_by_time[t_idx];
            if (!pre.is_binomial && mst.shared) vec_grad_ws.eta_denom[i] += ms_effect_by_time[t_idx];
        }
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // Feature-specific residual scatter (spatial + multiscale temporal)
    for (int i = 0; i < pre.N; i++) {
        double dLL_num = vec_grad_ws.resid_num[i];
        double dLL_denom = vec_grad_ws.resid_denom[i];
        double dLL_shared = dLL_num + dLL_denom;
        int s_unit = obs_s_unit[i];
        if (layout.has_spatial && s_unit >= 0) {
            grad_spatial_lik[s_unit] += dLL_shared;
            if (layout.is_bym2) grad_theta_lik[s_unit] += dLL_shared;
        }
        int t_idx = obs_t_idx[i];
        if (t_idx >= 0) {
            double dLL_temporal = mst.shared ? dLL_shared : dLL_num;
            if (trend != nullptr && t_idx < n_trend) grad_trend_lik[t_idx] += dLL_temporal;
            if (seasonal != nullptr && mst.seasonal_period > 0) {
                int s_idx = t_idx % mst.seasonal_period;
                if (s_idx < n_seasonal) grad_seasonal_lik[s_idx] += dLL_temporal;
            }
            if (short_term != nullptr && t_idx < n_short) grad_short_lik[t_idx] += dLL_temporal;
        }
    }

    // Spatial GMRF prior gradients (ICAR/BYM2)
    if (layout.has_spatial) {
        spatial_gmrf_prior_grad(data, layout, spatial_phi, tau_spatial,
                                sigma_s_bym2, sigma_u_bym2, rho_bym2, theta_bym2,
                                grad_spatial_lik.data(), grad_theta_lik.data(), grad.data());
    }

    // Multiscale temporal GMRF prior gradients
    tulpa_temporal_grad::MultiscaleTemporalGradients ms_grads;
    tulpa_temporal_grad::multiscale_temporal_prior_gradients(
        trend, n_trend,
        seasonal, n_seasonal,
        short_term, n_short,
        sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
        mst, ms_grads);

    for (int t = 0; t < n_trend; t++) grad[layout.trend_start + t] = grad_trend_lik[t] + ms_grads.grad_trend[t];
    for (int t = 0; t < n_seasonal; t++) grad[layout.seasonal_start + t] = grad_seasonal_lik[t] + ms_grads.grad_seasonal[t];
    for (int t = 0; t < n_short; t++) grad[layout.short_term_start + t] = grad_short_lik[t] + ms_grads.grad_short_term[t];
    if (n_trend > 0) grad[layout.log_sigma2_trend_idx] += ms_grads.grad_log_sigma2_trend;
    if (n_seasonal > 0) grad[layout.log_sigma2_seasonal_idx] += ms_grads.grad_log_sigma2_seasonal;
    if (n_short > 0) grad[layout.log_sigma2_short_idx] += ms_grads.grad_log_sigma2_short;
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        grad[layout.logit_rho_short_idx] += ms_grads.grad_logit_rho_short;
    }

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// Spatiotemporal interaction gradient (hand-coded, rows 28-29, 58-59, 90-91)
// Supports Knorr-Held Type I-IV with ICAR spatial + RW1/RW2 temporal
// =====================================================================

void compute_gradient_spatiotemporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- Spatial parameters (ICAR/BYM2) ---
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0;
    double rho_bym2 = 0.5;
    const double* theta_bym2 = nullptr;
    if (layout.has_spatial) {
        if (!layout.is_bym2) tau_spatial = std::exp(params[layout.log_tau_spatial_idx]);
        spatial_phi = &params[layout.spatial_start];
        if (layout.is_bym2) {
            double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
            double logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
            sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
            theta_bym2 = &params[layout.theta_bym2_start];
        }
    }

    // --- Temporal parameters ---
    double tau_temporal = 0.0;
    int T_temporal = 0;
    const double* phi_temporal = nullptr;
    double rho_ar1 = 0.5;
    if (layout.has_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_temporal = layout.temporal_end - layout.temporal_start;
        phi_temporal = &params[layout.temporal_start];
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
            rho_ar1 = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx]));
        }
    }

    // --- Spatiotemporal interaction parameters ---
    const auto& st = data.spatiotemporal_data;
    int S = st.n_spatial;
    int T = st.n_times;
    int ST = st.n_params;
    double tau_st = std::exp(params[layout.log_tau_st_idx]);
    double tau_st2 = 1.0;  // Always 1.0: single tau for all ST types

    // NC reparameterization: params store z, reconstruct delta
    const bool st_use_nc = (data.st_parameterization == 1 &&
                            st.type == STType::TYPE_IV);
    const double* z_or_delta = &params[layout.st_delta_start];
    static thread_local std::vector<double> st_delta_buf;
    const double* delta;
    double inv_scale = 1.0;
    if (st_use_nc) {
        inv_scale = 1.0 / std::sqrt(tau_st);
        st_delta_buf.resize(ST);
        for (int k = 0; k < ST; k++) st_delta_buf[k] = z_or_delta[k] * inv_scale;
        delta = st_delta_buf.data();
    } else {
        delta = z_or_delta;
    }

    // --- Shared base priors + feature-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // Spatial prior (ICAR: Gamma on tau)
    if (layout.has_spatial && !layout.is_bym2) {
        grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0) - data.tau_spatial_rate * tau_spatial + 1.0;
    }
    if (layout.is_bym2) {
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        double ratio = sigma_total / data.sigma_re_scale;
        grad[layout.log_sigma_bym2_idx] = -2.0 * ratio * ratio / (1.0 + ratio * ratio) + 1.0;
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }

    // Temporal prior (Gamma on tau, Beta on rho)
    if (layout.has_temporal) {
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
            grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;
        }
    }

    // ST interaction prior on tau_st (PC prior: exponential on sigma_st)
    {
        double sigma_st = 1.0 / std::sqrt(tau_st);
        double lambda = -std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U;
        grad[layout.log_tau_st_idx] = 0.5 * lambda * sigma_st + 0.5 + 1.0;
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // Feature-specific eta: spatial, temporal, ST
    std::vector<double> grad_spatial_lik;
    if (layout.has_spatial) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_theta_lik;
    if (layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_temporal_lik(T_temporal, 0.0);
    std::vector<double> grad_delta_lik(ST, 0.0);

    for (int i = 0; i < pre.N; i++) {
        // Spatial
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            double spatial_eff;
            if (layout.is_bym2) {
                spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit];
            } else {
                spatial_eff = spatial_phi[s_unit];
            }
            vec_grad_ws.eta_num[i] += spatial_eff;
            if (!pre.is_binomial) vec_grad_ws.eta_denom[i] += spatial_eff;
        }
        // Temporal
        if (layout.has_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * data.n_times + t;
            if (t_base >= 0 && t_base < T_temporal) {
                vec_grad_ws.eta_num[i] += phi_temporal[t_base];
                if (!pre.is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += phi_temporal[t_base];
            }
        }
        // Spatiotemporal interaction
        if (st.st_flat[i] > 0) {
            double st_effect = delta[st.st_flat[i] - 1];
            vec_grad_ws.eta_num[i] += st_effect;
            if (!pre.is_binomial && st.shared) vec_grad_ws.eta_denom[i] += st_effect;
        }
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // Feature-specific residual scatter (spatial, temporal, ST)
    for (int i = 0; i < pre.N; i++) {
        double dLL_num = vec_grad_ws.resid_num[i];
        double dLL_denom = vec_grad_ws.resid_denom[i];
        double dLL_shared = dLL_num + dLL_denom;

        // Spatial
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            if (layout.is_bym2) { grad_spatial_lik[s_unit] += dLL_shared; grad_theta_lik[s_unit] += dLL_shared; }
            else { grad_spatial_lik[s_unit] += dLL_shared; }
        }

        // Temporal
        if (layout.has_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * data.n_times + t;
            if (t_base >= 0 && t_base < T_temporal)
                grad_temporal_lik[t_base] += data.temporal_shared ? dLL_shared : dLL_num;
        }

        // Spatiotemporal interaction
        if (st.st_flat[i] > 0) {
            int st_idx = st.st_flat[i] - 1;
            grad_delta_lik[st_idx] += st.shared ? dLL_shared : dLL_num;
        }
    }

    // Spatial GMRF prior gradients (ICAR/BYM2)
    if (layout.has_spatial) {
        spatial_gmrf_prior_grad(data, layout, spatial_phi, tau_spatial,
                                sigma_s_bym2, sigma_u_bym2, rho_bym2, theta_bym2,
                                grad_spatial_lik.data(), grad_theta_lik.data(), grad.data());
    }

    // Temporal GMRF prior gradients
    if (layout.has_temporal && T_temporal > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                                 phi_temporal, T_temporal, grad_temporal_lik.data(), grad.data());
    }

    // =========================================================================
    // Spatiotemporal interaction prior gradients (Type I-IV)
    // =========================================================================
    // delta is stored column-major: delta[s*T + t]
    // Accumulate ST delta log-prior alongside gradient to avoid recomputing
    // the expensive Kronecker quadratic form in compute_log_post.
    double st_lp_accum = 0.0;
    if (st.type == STType::TYPE_I) {
        // IID: log p = 0.5*n*log(tau) - 0.5*tau*sum(delta^2)
        double qf = 0.0;
        for (int k = 0; k < ST; k++) {
            grad[layout.st_delta_start + k] = grad_delta_lik[k] - tau_st * delta[k];
            qf += delta[k] * delta[k];
        }
        grad[layout.log_tau_st_idx] += 0.5 * ST - 0.5 * tau_st * qf;
        st_lp_accum = 0.5 * ST * std::log(tau_st) - 0.5 * tau_st * qf;

    } else if (st.type == STType::TYPE_II) {
        // Structured time per spatial unit: temporal GMRF applied to delta[s,:]
        double total_qf = 0.0;
        for (int s = 0; s < S; s++) {
            // Apply temporal stencil to delta[s*T .. s*T+T-1]
            const double* delta_s = &delta[s * T];
            if (st.temporal_type == TemporalType::RW1) {
                double qf = 0.0;
                for (int t = 0; t < T; t++) {
                    double g = 0.0;
                    if (t > 0) { g += tau_st * (delta_s[t-1] - delta_s[t]); qf += std::pow(delta_s[t] - delta_s[t-1], 2); }
                    if (t < T - 1) g += tau_st * (delta_s[t+1] - delta_s[t]);
                    grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t] + g;
                }
                total_qf += qf;
            } else if (st.temporal_type == TemporalType::RW2) {
                double qf = 0.0;
                for (int t = 0; t < T; t++) {
                    double g = 0.0;
                    if (t >= 2) g -= tau_st * (delta_s[t-2] - 2.0*delta_s[t-1] + delta_s[t]);
                    if (t >= 1 && t < T - 1) g += 2.0 * tau_st * (delta_s[t-1] - 2.0*delta_s[t] + delta_s[t+1]);
                    if (t < T - 2) g -= tau_st * (delta_s[t] - 2.0*delta_s[t+1] + delta_s[t+2]);
                    grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t] + g;
                }
                for (int t = 2; t < T; t++) qf += std::pow(delta_s[t-2] - 2.0*delta_s[t-1] + delta_s[t], 2);
                total_qf += qf;
            } else if (st.temporal_type == TemporalType::AR1) {
                // AR1 with IID fallback for ST interaction (no separate rho)
                double qf = 0.0;
                for (int t = 0; t < T; t++) {
                    grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t] - tau_st * delta_s[t];
                    qf += delta_s[t] * delta_s[t];
                }
                total_qf += qf;
            } else {
                // Fallback: write likelihood gradient only
                for (int t = 0; t < T; t++)
                    grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t];
            }
        }
        int rank_per_unit = (st.temporal_type == TemporalType::RW1) ? (T - 1) :
                            (st.temporal_type == TemporalType::RW2) ? (T - 2) : T;
        grad[layout.log_tau_st_idx] += 0.5 * S * rank_per_unit - 0.5 * tau_st * total_qf;
        st_lp_accum = 0.5 * S * rank_per_unit * std::log(tau_st) - 0.5 * tau_st * total_qf;

    } else if (st.type == STType::TYPE_III) {
        // Structured space per time point: ICAR applied to delta[:,t]
        double total_qf = 0.0;
        for (int t = 0; t < T; t++) {
            // Apply ICAR stencil to delta[0*T+t, 1*T+t, ..., (S-1)*T+t]
            for (int s = 0; s < S; s++) {
                double icar_grad = 0.0;
                for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                    int j = st.adj_col_idx[idx] - 1;
                    icar_grad += tau_st * (delta[j * T + t] - delta[s * T + t]);
                }
                grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t] + icar_grad;
            }
            // Compute ICAR quadratic form for this time slice
            for (int s = 0; s < S; s++) {
                for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                    int j = st.adj_col_idx[idx] - 1;
                    if (j > s) {
                        double diff = delta[s * T + t] - delta[j * T + t];
                        total_qf += diff * diff;
                    }
                }
            }
        }
        int rank_spatial = S - 1;
        grad[layout.log_tau_st_idx] += 0.5 * T * rank_spatial - 0.5 * tau_st * total_qf;
        st_lp_accum = 0.5 * T * rank_spatial * std::log(tau_st) - 0.5 * tau_st * total_qf;

    } else if (st.type == STType::TYPE_IV) {
        // Kronecker: Q_delta = Q_s - Q_t
        // For NC: apply stencil to z (not delta), without tau factor
        const double* stencil_input = st_use_nc ? z_or_delta : delta;

        // Step 1: Apply temporal stencil: v[s,t] = (Q_t * input[s,:])_t
        static thread_local std::vector<double> v;
        v.assign(S * T, 0.0);
        if (st.temporal_type == TemporalType::RW1) {
            for (int s = 0; s < S; s++) {
                for (int t = 0; t < T; t++) {
                    double qt_val = 0.0;
                    int n_t_neigh = 0;
                    if (t > 0) { qt_val -= stencil_input[s * T + t - 1]; n_t_neigh++; }
                    if (t < T - 1) { qt_val -= stencil_input[s * T + t + 1]; n_t_neigh++; }
                    qt_val += n_t_neigh * stencil_input[s * T + t];
                    v[s * T + t] = qt_val;
                }
            }
        } else if (st.temporal_type == TemporalType::RW2) {
            // Unrolled RW2 stencil: Q_t is the second-difference precision matrix.
            // For each t, Q_t[t,:] * x gives a linear combination of second differences.
            // Instead of irregular max/min bounds, handle boundary cases explicitly.
            for (int s = 0; s < S; s++) {
                const double* d_s = &stencil_input[s * T];
                double* v_s = &v[s * T];
                if (T >= 3) {
                    // Precompute second differences d2[k] = d_s[k] - 2*d_s[k+1] + d_s[k+2]
                    // Reused by multiple t values, avoids redundant subtraction
                    const int n_d2 = T - 2;
                    // Use stack for small T (typical: T=20), heap only if huge
                    double d2_stack[64];
                    double* d2 = (n_d2 <= 64) ? d2_stack : new double[n_d2];
                    for (int k = 0; k < n_d2; k++) {
                        d2[k] = d_s[k] - 2.0 * d_s[k + 1] + d_s[k + 2];
                    }

                    // t=0: only k=0 contributes (pos=0, coef=1)
                    v_s[0] = d2[0];
                    // t=1: k=0 (pos=1, coef=-2) + k=1 (pos=0, coef=1) if T>=4
                    v_s[1] = -2.0 * d2[0];
                    if (n_d2 > 1) v_s[1] += d2[1];
                    // Interior: t=2..T-3, all three contributions present
                    for (int t = 2; t < T - 2; t++) {
                        v_s[t] = d2[t - 2] - 2.0 * d2[t - 1] + d2[t];
                    }
                    // t=T-2: k=T-4 (pos=2, coef=1) + k=T-3 (pos=1, coef=-2)
                    if (T >= 4) {
                        v_s[T - 2] = d2[n_d2 - 2] - 2.0 * d2[n_d2 - 1];
                    } else {
                        // T==3: t=1 already handled, t=T-2=1 is same slot
                        // Just set directly
                        v_s[T - 2] = -2.0 * d2[0];
                    }
                    // t=T-1: k=T-3 (pos=2, coef=1)
                    v_s[T - 1] = d2[n_d2 - 1];

                    if (n_d2 > 64) delete[] d2;
                } else {
                    // T < 3: no second differences possible, v stays zero
                }
            }
        }

        // Step 2: Apply spatial ICAR stencil to v: (Q_s - Q_t) * input
        double total_qf = 0.0;
        for (int s = 0; s < S; s++) {
            for (int t = 0; t < T; t++) {
                double qs_v = 0.0;
                for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                    int j = st.adj_col_idx[idx] - 1;
                    qs_v -= v[j * T + t];
                }
                int n_neigh = st.adj_row_ptr[s + 1] - st.adj_row_ptr[s];
                qs_v += n_neigh * v[s * T + t];

                if (st_use_nc) {
                    // NC: grad_z = -(Q z)_k + dL/d(delta_k) / sqrt(tau)
                    grad[layout.st_delta_start + s * T + t] =
                        grad_delta_lik[s * T + t] * inv_scale - qs_v;
                } else {
                    // Centered: grad_delta = dL/d(delta_k) - tau * (Q delta)_k
                    grad[layout.st_delta_start + s * T + t] =
                        grad_delta_lik[s * T + t] - tau_st * qs_v;
                }
                total_qf += stencil_input[s * T + t] * qs_v;
            }
        }

        int rank_space = S - 1;
        int rank_time = (st.temporal_type == TemporalType::RW1) ? (T - 1) :
                        (st.temporal_type == TemporalType::RW2) ? (T - 2) : T;
        if (st.temporal_cyclic) rank_time = T;
        int total_rank = rank_space * rank_time;

        if (st_use_nc) {
            // NC tau gradient: 0.5*(rank - ST) from combined normalization+Jacobian
            // plus likelihood chain rule: -0.5 * dot(grad_delta_lik, delta)
            double lik_tau_grad = 0.0;
            for (int k = 0; k < ST; k++) {
                lik_tau_grad += grad_delta_lik[k] * delta[k];
            }
            grad[layout.log_tau_st_idx] += 0.5 * (total_rank - ST) - 0.5 * lik_tau_grad;
            // NC log-prior: -0.5*qf + 0.5*(rank-ST)*log(tau)
            st_lp_accum = -0.5 * total_qf + 0.5 * (total_rank - ST) * std::log(tau_st);
        } else {
            // Centered tau gradient: 0.5*rank - 0.5*tau*qf
            grad[layout.log_tau_st_idx] += 0.5 * total_rank - 0.5 * tau_st * total_qf;
            st_lp_accum = 0.5 * total_rank * std::log(tau_st) - 0.5 * tau_st * total_qf;
        }
    }

    // Sum-to-zero penalty gradients (on reconstructed delta)
    // For NC: chain rule d/dz = inv_scale * d/d(delta), and penalty also contributes to tau
    {
        double lambda_stz = 0.001;
        // For NC, the penalty -0.5*lambda*sum(delta)^2 where delta=z*inv_scale
        // d/dz = -lambda * sum(delta) * inv_scale
        // d/d(log_tau) = -lambda * sum(delta) * d(delta)/d(log_tau) summed
        //              = -lambda * sum(delta) * (-0.5 * delta[k]) for each k in sum
        //              = but simpler: penalty = -0.5*lambda*(inv_scale * sum(z))^2
        //              = -0.5*lambda*inv_scale^2 * sum(z)^2
        // d/dz_k = -lambda*inv_scale^2 * sum(z)
        // This equals -lambda*inv_scale * sum(delta) = centered penalty * inv_scale
        double stz_scale = st_use_nc ? inv_scale : 1.0;

        // Pre-compute row sums (over space) and col sums (over time) in a single pass
        // This replaces 4 separate double-loops with one pass + two apply loops
        static thread_local std::vector<double> row_sums_buf, col_sums_buf;
        row_sums_buf.assign(T, 0.0);
        col_sums_buf.assign(S, 0.0);
        for (int s = 0; s < S; s++) {
            for (int t = 0; t < T; t++) {
                double d = delta[s * T + t];
                row_sums_buf[t] += d;
                col_sums_buf[s] += d;
            }
        }

        // Apply delta gradients + accumulate NC tau gradient in one pass
        double tau_stz_grad = 0.0;
        for (int s = 0; s < S; s++) {
            for (int t = 0; t < T; t++) {
                grad[layout.st_delta_start + s * T + t] -=
                    lambda_stz * stz_scale * (row_sums_buf[t] + col_sums_buf[s]);
            }
        }
        if (st_use_nc) {
            for (int t = 0; t < T; t++) tau_stz_grad += row_sums_buf[t] * row_sums_buf[t];
            for (int s = 0; s < S; s++) tau_stz_grad += col_sums_buf[s] * col_sums_buf[s];
            tau_stz_grad *= 0.5 * lambda_stz;
            grad[layout.log_tau_st_idx] += tau_stz_grad;
        }

        // Accumulate sum-to-zero penalty into ST log-prior
        for (int t = 0; t < T; t++) st_lp_accum -= 0.5 * lambda_stz * row_sums_buf[t] * row_sums_buf[t];
        for (int s = 0; s < S; s++) st_lp_accum -= 0.5 * lambda_stz * col_sums_buf[s] * col_sums_buf[s];
    }

    // --- Custom epilogue (spatiotemporal has fused ST prior accumulation) ---
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), pre.cp.sigma_re);

    if (pre.fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true, &st_lp_accum) + pre.obs_log_lik;
}
