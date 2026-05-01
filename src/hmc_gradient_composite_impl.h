// =====================================================================
// Composite hand-coded gradient: handles ANY combination of features.
// This is the catch-all H-mode function for exotic multi-feature combos
// that no specialized gradient function covers (e.g., HSGP+TVC, SVC+RW1,
// latent+spatial, etc.). Slower than specialized functions but much faster
// than A_r/N fallback.
//
// Architecture: single observation loop with conditional feature blocks.
// Each feature contributes additively to eta; gradient scattering is
// independent per feature. Structural/prior gradients computed after the
// observation loop.
// =====================================================================

void compute_gradient_composite(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    const int N = data.N;
    const bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                              data.legacy.model_type == ModelType::BETA_BINOMIAL);

    // =========================================================================
    // Phase 1: Extract all parameters
    // =========================================================================
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // --- ZI/OI parameters ---
    const double* beta_zi = nullptr;
    const double* beta_oi = nullptr;
    if (layout.has_zi && data.p_zi > 0)
        beta_zi = &params[layout.beta_zi_start];
    if (layout.has_oi && data.p_oi > 0)
        beta_oi = &params[layout.beta_oi_start];

    // --- Spatial (ICAR/BYM2/pCAR) ---
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0, rho_bym2 = 0.5;
    const double* theta_bym2 = nullptr;
    const bool has_icar_bym2 = layout.has_spatial && !layout.is_hsgp && !layout.is_gp &&
                                !layout.is_multiscale_gp && !layout.has_svc;
    if (has_icar_bym2) {
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

    // --- GP (NNGP) spatial ---
    const bool has_gp_spatial = layout.is_gp && data.has_gp && !data.gp_collapsed;
    double sigma2_gp_c = 1.0, phi_gp_c = 1.0;
    int N_gp_c = 0;
    bool use_nc_gp = false;
    static thread_local tulpa_gp::NNGPNCWorkspace nc_ws_gp_c;
    static thread_local std::vector<double> gp_w_c;
    if (has_gp_spatial) {
        N_gp_c = data.gp_data.n_obs;
        sigma2_gp_c = std::exp(params[layout.log_sigma2_gp_idx]);
        phi_gp_c = std::exp(params[layout.log_phi_gp_idx]);
        use_nc_gp = (data.gp_parameterization == 1);
        gp_w_c.resize(N_gp_c);
        if (use_nc_gp) {
            const double* z_gp = &params[layout.gp_w_start];
            tulpa_gp::nngp_nc_forward(z_gp, sigma2_gp_c, phi_gp_c, data.gp_data, nc_ws_gp_c);
            std::memcpy(gp_w_c.data(), nc_ws_gp_c.w.data(), N_gp_c * sizeof(double));
        } else {
            for (int i = 0; i < N_gp_c; i++) gp_w_c[i] = params[layout.gp_w_start + i];
        }
    }

    // --- HSGP spatial ---
    static thread_local tulpa_hsgp::HSGPWorkspace hsgp_ws;
    double hsgp_sigma2 = 0.0, hsgp_lengthscale = 0.0;
    const double* hsgp_beta_ptr = nullptr;
    if (layout.is_hsgp && data.has_hsgp) {
        hsgp_sigma2 = std::exp(params[layout.log_sigma2_hsgp_idx]);
        hsgp_lengthscale = std::exp(params[layout.log_lengthscale_hsgp_idx]);
        hsgp_beta_ptr = &params[layout.hsgp_beta_start];
        hsgp_ws.init(data.hsgp_data.n_obs, data.hsgp_data.m_total);
        tulpa_hsgp::hsgp_evaluate_ws(hsgp_beta_ptr, hsgp_sigma2, hsgp_lengthscale,
                                       data.hsgp_data, hsgp_ws);
    }

    // --- Temporal (RW1/RW2/AR1 GMRF) ---
    double tau_temporal = 0.0;
    int T_temporal = 0;
    const double* phi_temporal = nullptr;
    double rho_ar1 = 0.5;
    const bool has_gmrf_temporal = layout.has_temporal && !layout.is_temporal_gp &&
                                   !layout.has_multiscale_temporal && !layout.has_tvc;
    if (has_gmrf_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_temporal = layout.temporal_end - layout.temporal_start;
        phi_temporal = &params[layout.temporal_start];
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
            rho_ar1 = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx]));
        }
    }

    // --- Temporal GP ---
    const bool has_temporal_gp = layout.is_temporal_gp && layout.has_temporal;
    int T_gp = 0;
    const double* z_temporal_gp = nullptr;
    static thread_local tulpa_temporal_gp::TemporalGPNCWorkspace nc_ws_composite;
    static thread_local std::vector<double> temporal_gp_f;
    double sigma2_tgp_comp = 0.0, phi_tgp_comp = 0.0;
    double phi_lower_tgp = 0.0, phi_upper_tgp = 0.0;
    bool use_nc_tgp = false;
    int n_groups_gp = 0;
    if (has_temporal_gp) {
        T_gp = data.n_times;
        n_groups_gp = data.n_temporal_groups;
        z_temporal_gp = &params[layout.temporal_start];
        int total_gp = n_groups_gp * T_gp;
        temporal_gp_f.resize(total_gp);

        sigma2_tgp_comp = std::exp(params[layout.log_sigma2_temporal_gp_idx]);
        double phi_gp_raw = params[layout.logit_phi_temporal_gp_idx];
        phi_lower_tgp = data.temporal_gp_phi_prior_lower;
        phi_upper_tgp = data.temporal_gp_phi_prior_upper;
        double phi_range = phi_upper_tgp - phi_lower_tgp;
        phi_tgp_comp = phi_lower_tgp + phi_range / (1.0 + std::exp(-phi_gp_raw));

        use_nc_tgp = (data.temporal_gp_parameterization == 1);
        if (use_nc_tgp) {
            nc_ws_composite.init(T_gp, n_groups_gp);
            tulpa_temporal_gp::temporal_gp_nc_forward(
                z_temporal_gp, T_gp, n_groups_gp, sigma2_tgp_comp, phi_tgp_comp,
                data.temporal_gp_data.time_values, nc_ws_composite);
            for (int k = 0; k < total_gp; k++) temporal_gp_f[k] = nc_ws_composite.f[k];
        } else {
            for (int k = 0; k < total_gp; k++) temporal_gp_f[k] = z_temporal_gp[k];
        }
    }

    // --- TVC ---
    static thread_local std::vector<double> tvc_eta_precomp;
    int n_tvc = 0, n_tvc_times = 0, n_tvc_groups = 1, n_w = 0;
    static thread_local std::vector<double> tvc_tau_buf, tvc_rho_buf, tvc_w_flat_buf;
    if (layout.has_tvc && data.has_tvc) {
        n_tvc = data.tvc_data.n_tvc;
        n_tvc_times = data.tvc_data.n_times;
        n_tvc_groups = data.tvc_data.n_groups;
        n_w = n_tvc_groups * n_tvc * n_tvc_times;

        tvc_tau_buf.resize(n_tvc);
        tvc_rho_buf.resize(n_tvc);
        tvc_w_flat_buf.resize(n_w);

        for (int j = 0; j < n_tvc; j++) {
            tvc_tau_buf[j] = std::exp(params[layout.log_tau_tvc_start + j]);
            if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
                double logit_rho = params[layout.logit_rho_tvc_start + j];
                double u = 1.0 / (1.0 + std::exp(-logit_rho));
                tvc_rho_buf[j] = 2.0 * u - 1.0;
            } else {
                tvc_rho_buf[j] = 0.0;
            }
        }
        for (int k = 0; k < n_w; k++) tvc_w_flat_buf[k] = params[layout.tvc_w_start + k];

        // Precompute TVC eta contribution
        tvc_eta_precomp.assign(N, 0.0);
        for (int i = 0; i < N; i++) {
            int t = data.tvc_data.time_index[i] - 1;
            int g = data.tvc_data.group_index[i] - 1;
            for (int j = 0; j < n_tvc; j++) {
                int w_idx = (g * n_tvc + j) * n_tvc_times + t;
                tvc_eta_precomp[i] += data.tvc_data.X_tvc[i * n_tvc + j] * tvc_w_flat_buf[w_idx];
            }
        }
    }

    // --- MSGP-HSGP (Multi-scale HSGP) spatial ---
    static thread_local tulpa_hsgp::HSGPWorkspace msgp_ws_local_c, msgp_ws_regional_c;
    double msgp_sigma2_local = 0, msgp_sigma2_regional = 0;
    double msgp_ls_local = 0, msgp_ls_regional = 0;
    const double* msgp_beta_local = nullptr;
    const double* msgp_beta_regional = nullptr;
    int msgp_m_total = 0;
    const bool has_msgp_hsgp = layout.is_multiscale_gp && data.has_multiscale_gp && data.msgp_is_hsgp;
    if (has_msgp_hsgp) {
        msgp_sigma2_local = std::exp(params[layout.log_sigma2_gp_local_idx]);
        msgp_ls_local = std::exp(params[layout.log_phi_gp_local_idx]);
        msgp_sigma2_regional = std::exp(params[layout.log_sigma2_gp_regional_idx]);
        msgp_ls_regional = std::exp(params[layout.log_phi_gp_regional_idx]);
        msgp_m_total = data.msgp_hsgp_data.m_total;
        msgp_beta_local = &params[layout.gp_local_start];
        msgp_beta_regional = &params[layout.gp_regional_start];
        msgp_ws_local_c.init(data.N, msgp_m_total);
        msgp_ws_regional_c.init(data.N, msgp_m_total);
        tulpa_hsgp::hsgp_evaluate_ws(msgp_beta_local, msgp_sigma2_local, msgp_ls_local,
                                       data.msgp_hsgp_data, msgp_ws_local_c);
        tulpa_hsgp::hsgp_evaluate_ws(msgp_beta_regional, msgp_sigma2_regional, msgp_ls_regional,
                                       data.msgp_hsgp_data, msgp_ws_regional_c);
    }

    // --- SVC (HSGP) ---
    static thread_local tulpa_hsgp::HSGPWorkspace svc_hsgp_ws;
    int n_svc = 0, svc_m_total = 0;
    static thread_local std::vector<double> svc_eta_precomp, svc_f_all;
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        n_svc = data.svc_data.n_svc;
        svc_m_total = data.svc_hsgp_data.m_total;
        svc_hsgp_ws.init(data.svc_hsgp_data.n_obs, svc_m_total);
        svc_eta_precomp.assign(N, 0.0);
        svc_f_all.resize(n_svc * N);

        for (int j = 0; j < n_svc; j++) {
            double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
            double lengthscale_j = std::exp(params[layout.log_phi_svc_start + j]);
            const double* beta_j = &params[layout.svc_w_start + j * svc_m_total];

            tulpa_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                           data.svc_hsgp_data, svc_hsgp_ws);

            for (int i = 0; i < N; i++) {
                double f_ji = svc_hsgp_ws.hsgp_f[i];
                svc_f_all[j * N + i] = f_ji;
                double x_ij = data.svc_data.X_svc[i * n_svc + j];
                svc_eta_precomp[i] += x_ij * f_ji;
            }
        }
    }

    // --- Latent factors ---
    int K_latent = 0;
    static thread_local std::vector<double> latent_eta_precomp;
    static thread_local std::vector<double> factors_constrained, sigma_latent_vec;
    if (layout.has_latent && data.latent_n_factors > 0) {
        K_latent = data.latent_n_factors;
        sigma_latent_vec.resize(K_latent);
        for (int k = 0; k < K_latent; k++)
            sigma_latent_vec[k] = std::exp(params[layout.log_sigma_latent_start + k]);

        int n_factor_params = N * K_latent;
        factors_constrained.resize(n_factor_params);
        for (int j = 0; j < n_factor_params; j++)
            factors_constrained[j] = params[layout.latent_factor_start + j];

        // Apply sum-to-zero constraint
        if (data.latent_constraint == 0) {
            for (int k = 0; k < K_latent; k++) {
                double sum = 0.0;
                for (int i = 0; i < N; i++) sum += factors_constrained[i * K_latent + k];
                double mean = sum / N;
                for (int i = 0; i < N; i++) factors_constrained[i * K_latent + k] -= mean;
            }
        }

        latent_eta_precomp.assign(N, 0.0);
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K_latent; k++)
                latent_eta_precomp[i] += factors_constrained[i * K_latent + k] * sigma_latent_vec[k];
    }

    // --- Spatiotemporal ---
    const bool has_st = layout.has_spatiotemporal && !layout.is_st_gp &&
                        data.spatiotemporal_data.type != STType::NONE &&
                        layout.st_delta_start >= 0 && layout.log_tau_st_idx >= 0;
    double tau_st = 0.0;
    const double* st_delta = nullptr;
    static thread_local std::vector<double> st_delta_buf;
    bool st_use_nc = false;
    double inv_scale_st = 1.0;
    const double* z_or_delta_st = nullptr;
    int ST_n = 0, S_st = 0, T_st = 0;
    if (has_st) {
        const auto& st = data.spatiotemporal_data;
        S_st = st.n_spatial;
        T_st = st.n_times;
        ST_n = st.n_params;
        tau_st = std::exp(params[layout.log_tau_st_idx]);
        st_use_nc = (data.st_parameterization == 1 && st.type == STType::TYPE_IV);
        z_or_delta_st = &params[layout.st_delta_start];
        if (st_use_nc) {
            inv_scale_st = 1.0 / std::sqrt(tau_st);
            st_delta_buf.resize(ST_n);
            for (int k = 0; k < ST_n; k++) st_delta_buf[k] = z_or_delta_st[k] * inv_scale_st;
            st_delta = st_delta_buf.data();
        } else {
            st_delta = z_or_delta_st;
        }
    }

    // --- Random slopes ---
    const bool has_slopes = layout.has_re_slopes;
    bool slopes_nc = has_slopes && (data.re_parameterization == 1);
    int n_re_terms_slopes = 0;
    static thread_local std::vector<std::vector<double>> grad_re_slopes_lik;
    static thread_local std::vector<std::vector<double>> nc_L_flats, nc_sigmas_vec;
    static thread_local std::vector<double> re_nc_flat_c;
    // Slopes priors and eta contribution are handled inline below

    // --- Multiscale temporal ---
    const bool has_ms_temporal = layout.has_multiscale_temporal;
    int n_trend = 0, n_seasonal = 0, n_short = 0;
    const double* trend = nullptr;
    const double* seasonal = nullptr;
    const double* short_term = nullptr;
    double sigma2_trend = 1.0, sigma2_seasonal = 1.0, sigma2_short = 1.0;
    double rho_short = 0.5;
    static thread_local std::vector<double> ms_effect_by_time_c;
    static thread_local std::vector<double> grad_trend_lik_c, grad_seasonal_lik_c, grad_short_lik_c;
    static thread_local std::vector<int> obs_t_idx_ms_c;
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        n_trend = layout.trend_end - layout.trend_start;
        n_seasonal = layout.seasonal_end - layout.seasonal_start;
        n_short = layout.short_term_end - layout.short_term_start;
        if (n_trend > 0) { trend = &params[layout.trend_start]; sigma2_trend = std::exp(params[layout.log_sigma2_trend_idx]); }
        if (n_seasonal > 0) { seasonal = &params[layout.seasonal_start]; sigma2_seasonal = std::exp(params[layout.log_sigma2_seasonal_idx]); }
        if (n_short > 0) { short_term = &params[layout.short_term_start]; sigma2_short = std::exp(params[layout.log_sigma2_short_idx]); }
        if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
            double u = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_short_idx]));
            rho_short = 2.0 * u - 1.0;
        }
        // Pre-assemble temporal effect per time point
        ms_effect_by_time_c.assign(mst.n_times, 0.0);
        for (int t = 0; t < mst.n_times; t++) {
            if (trend != nullptr && t < n_trend) ms_effect_by_time_c[t] += trend[t];
            if (seasonal != nullptr && mst.seasonal_period > 0) ms_effect_by_time_c[t] += seasonal[t % mst.seasonal_period];
            if (short_term != nullptr && t < n_short) ms_effect_by_time_c[t] += short_term[t];
        }
        grad_trend_lik_c.assign(n_trend, 0.0);
        grad_seasonal_lik_c.assign(n_seasonal, 0.0);
        grad_short_lik_c.assign(n_short, 0.0);
        obs_t_idx_ms_c.assign(N, -1);
    }

    // =========================================================================
    // Phase 2: Prior gradients (independent of data, computed first)
    // =========================================================================
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    if (!has_slopes) re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // ZI/OI beta priors: N(0, zi_prior_sd^2) / N(0, oi_prior_sd^2)
    if (layout.has_zi && data.p_zi > 0) {
        double tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd + 1e-10);
        for (int j = 0; j < data.p_zi; j++)
            grad[layout.beta_zi_start + j] = -tau_zi * beta_zi[j];
    }
    if (layout.has_oi && data.p_oi > 0) {
        double tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd + 1e-10);
        for (int j = 0; j < data.p_oi; j++)
            grad[layout.beta_oi_start + j] = -tau_oi * beta_oi[j];
    }

    // ICAR/BYM2 spatial priors
    if (has_icar_bym2 && !layout.is_bym2) {
        grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0) - data.tau_spatial_rate * tau_spatial + 1.0;
    }
    if (has_icar_bym2 && layout.is_bym2) {
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        double ratio = sigma_total / data.sigma_re_scale;
        grad[layout.log_sigma_bym2_idx] = -2.0 * ratio * ratio / (1.0 + ratio * ratio) + 1.0;
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }

    // GP (NNGP) priors
    if (has_gp_spatial) {
        // PC prior on sigma2_gp
        grad[layout.log_sigma2_gp_idx] = gp_pc_prior_grad_log_sigma2(
            sigma2_gp_c, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        // Uniform prior on phi: just Jacobian for log transform
        grad[layout.log_phi_gp_idx] = 1.0;
        if (!use_nc_gp) {
            // Centered: NNGP prior gradients on w
            tulpa_gp::NNGPGradients nngp_grads;
            tulpa_gp::gp_nngp_gradients(gp_w_c, sigma2_gp_c, phi_gp_c, data.gp_data, nngp_grads);
            for (int i = 0; i < N_gp_c; i++)
                grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
            grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
            grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;
        } else {
            // NC: N(0,1) prior on z
            for (int i = 0; i < N_gp_c; i++)
                grad[layout.gp_w_start + i] = -params[layout.gp_w_start + i];
        }
    }

    // HSGP priors (must match hardcoded rate=4.6 in compute_log_post)
    if (layout.is_hsgp && data.has_hsgp) {
        double sigma = std::sqrt(hsgp_sigma2);
        double rate = 4.6;  // Matches compute_log_post line 1457
        grad[layout.log_sigma2_hsgp_idx] = -0.5 * rate * sigma + 0.5 - 0.5;
        double log_ls = params[layout.log_lengthscale_hsgp_idx];
        grad[layout.log_lengthscale_hsgp_idx] = -log_ls;  // LogNormal(0,1) prior
        // N(0,I) prior on beta
        int M = data.hsgp_data.m_total;
        for (int j = 0; j < M; j++)
            grad[layout.hsgp_beta_start + j] = -hsgp_beta_ptr[j];
    }

    // MSGP-HSGP priors
    if (has_msgp_hsgp) {
        double sigma_local = std::sqrt(msgp_sigma2_local);
        double rate_local = -std::log(data.ms_sigma2_local_prior_alpha) / data.ms_sigma2_local_prior_U;
        grad[layout.log_sigma2_gp_local_idx] = -0.5 * rate_local * sigma_local + 0.5 - 0.5;

        double sigma_regional = std::sqrt(msgp_sigma2_regional);
        double rate_regional = -std::log(data.ms_sigma2_regional_prior_alpha) / data.ms_sigma2_regional_prior_U;
        grad[layout.log_sigma2_gp_regional_idx] = -0.5 * rate_regional * sigma_regional + 0.5 - 0.5;

        double log_ls_local_v = params[layout.log_phi_gp_local_idx];
        double z_local_v = (log_ls_local_v - data.ms_log_ls_local_mean) / data.ms_log_ls_local_sd;
        grad[layout.log_phi_gp_local_idx] = -z_local_v / data.ms_log_ls_local_sd;

        double log_ls_regional_v = params[layout.log_phi_gp_regional_idx];
        double z_regional_v = (log_ls_regional_v - data.ms_log_ls_regional_mean) / data.ms_log_ls_regional_sd;
        grad[layout.log_phi_gp_regional_idx] = -z_regional_v / data.ms_log_ls_regional_sd;

        for (int j = 0; j < msgp_m_total; j++) {
            grad[layout.gp_local_start + j] = -msgp_beta_local[j];
            grad[layout.gp_regional_start + j] = -msgp_beta_regional[j];
        }
    }

    // Temporal GMRF priors
    if (has_gmrf_temporal) {
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
            grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;
    }

    // Temporal GP priors
    if (has_temporal_gp) {
        double sigma_gp = std::sqrt(sigma2_tgp_comp);
        double rate_gp = -std::log(data.temporal_gp_sigma2_prior_alpha) / data.temporal_gp_sigma2_prior_U;
        grad[layout.log_sigma2_temporal_gp_idx] = -0.5 * rate_gp * sigma_gp + 0.5;
        // Phi: logit-bounded Jacobian
        double phi_range_tgp = phi_upper_tgp - phi_lower_tgp;
        grad[layout.logit_phi_temporal_gp_idx] = (phi_upper_tgp + phi_lower_tgp - 2.0 * phi_tgp_comp) / phi_range_tgp;

        if (use_nc_tgp) {
            // NC Jacobian: d/d(log_sigma2) of log|det(df/dz)| = T/2 per group
            grad[layout.log_sigma2_temporal_gp_idx] += 0.5 * T_gp * n_groups_gp;

            // NC Jacobian: d/d(log_phi) = -sum rho^2*(dt/phi) / (1-rho^2) per group
            double chi_tgp_prior = (phi_tgp_comp - phi_lower_tgp) * (phi_upper_tgp - phi_tgp_comp) /
                                   (phi_tgp_comp * phi_range_tgp);
            double jac_phi_log = 0.0;
            for (int t = 1; t < T_gp; t++) {
                double rho_t = nc_ws_composite.rho[t - 1];
                double rho2 = rho_t * rho_t;
                double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
                double one_minus_rho2 = 1.0 - rho2;
                if (one_minus_rho2 < 1e-10) one_minus_rho2 = 1e-10;
                jac_phi_log -= rho2 * (dt / phi_tgp_comp) / one_minus_rho2;
            }
            grad[layout.logit_phi_temporal_gp_idx] += jac_phi_log * n_groups_gp * chi_tgp_prior;
        }
    }

    // TVC priors (PC prior: P(sigma > 1) = 0.01)
    if (layout.has_tvc && data.has_tvc) {
        double tvc_pc_rate = -std::log(0.01) / 1.0;
        for (int j = 0; j < n_tvc; j++) {
            double sigma_j = 1.0 / std::sqrt(tvc_tau_buf[j]);
            grad[layout.log_tau_tvc_start + j] = 0.5 * tvc_pc_rate * sigma_j + 1.5;
            if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
                double u = (tvc_rho_buf[j] + 1.0) / 2.0;
                grad[layout.logit_rho_tvc_start + j] = 1.0 - 2.0 * u;
            }
        }
    }

    // SVC-HSGP priors: PC prior on sigma, LogNormal(0,1) on lengthscale, N(0,I) on beta
    // Must match compute_log_post: -rate*sigma + 0.5*log_sigma2 (Jacobian)
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        double rate_svc = 4.6;  // Matches compute_log_post line 1781
        for (int j = 0; j < n_svc; j++) {
            double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
            double sigma_j = std::sqrt(sigma2_j);
            // d/d(log_sigma2) [-rate*sigma + 0.5*log_sigma2] = -0.5*rate*sigma + 0.5
            grad[layout.log_sigma2_svc_start + j] = -0.5 * rate_svc * sigma_j + 0.5;
            // LogNormal(0,1) on lengthscale: d/d(log_ls) [-0.5*log_ls^2] = -log_ls
            double log_ls_j = params[layout.log_phi_svc_start + j];
            grad[layout.log_phi_svc_start + j] = -log_ls_j;
            // N(0,I) prior on SVC beta
            for (int m = 0; m < svc_m_total; m++)
                grad[layout.svc_w_start + j * svc_m_total + m] = -params[layout.svc_w_start + j * svc_m_total + m];
        }
    }

    // Latent priors
    if (K_latent > 0) {
        double latent_rate = data.latent_sigma_prior_rate;
        for (int k = 0; k < K_latent; k++)
            grad[layout.log_sigma_latent_start + k] = 1.0 - latent_rate * sigma_latent_vec[k];
    }

    // Multiscale temporal priors (PC priors on variances + AR1 rho prior)
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        if (n_trend > 0)
            grad[layout.log_sigma2_trend_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
                sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha) + 1.0;
        if (n_seasonal > 0)
            grad[layout.log_sigma2_seasonal_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
                sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha) + 1.0;
        if (n_short > 0)
            grad[layout.log_sigma2_short_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
                sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha) + 1.0;
        if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
            double u = (rho_short + 1.0) / 2.0;
            grad[layout.logit_rho_short_idx] = 2.0 * (1.0 - 2.0 * u);
        }
    }

    // ST interaction prior on tau_st
    if (has_st) {
        double sigma_st = 1.0 / std::sqrt(tau_st);
        double lambda = -std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U;
        grad[layout.log_tau_st_idx] = 0.5 * lambda * sigma_st + 0.5 + 1.0;
    }

    // Slopes priors - mirrors compute_gradient_analytical (correlated + uncorrelated)
    nc_L_flats.clear();
    nc_sigmas_vec.clear();
    re_nc_flat_c.clear();
    if (has_slopes) {
        n_re_terms_slopes = data.n_re_terms;
        grad_re_slopes_lik.resize(n_re_terms_slopes);
        nc_L_flats.resize(n_re_terms_slopes);
        nc_sigmas_vec.resize(n_re_terms_slopes);

        for (int t = 0; t < n_re_terms_slopes; t++) {
            int n_groups = data.re_n_groups_multi[t];
            int n_coefs = layout.re_n_coefs_multi[t];
            int re_start_t = layout.re_start_multi[t];
            bool is_correlated = layout.has_re_correlated_slopes &&
                                 t < (int)layout.re_correlated_multi.size() &&
                                 layout.re_correlated_multi[t];
            grad_re_slopes_lik[t].assign(n_groups * n_coefs, 0.0);

            // Extract sigmas and compute Half-Cauchy prior on each
            std::vector<double> sigmas(n_coefs);
            for (int c = 0; c < n_coefs; c++) {
                int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
                sigmas[c] = std::exp(params[log_sigma_idx]);
                double ratio_c = sigmas[c] / data.sigma_re_scale;
                double ratio_c_sq = ratio_c * ratio_c;
                grad[log_sigma_idx] = -2.0 * ratio_c_sq / (1.0 + ratio_c_sq) + 1.0;
            }

            if (is_correlated && n_coefs > 1) {
                // Tanh-parameterized L + LKJ(eta=2) prior gradient (raw-space,
                // includes tanh + L->R Jacobians) via lkj_chol_helpers.
                int chol_start = layout.chol_re_start_multi[t];
                std::vector<double> L_flat(n_coefs * n_coefs, 0.0);
                if (!tulpa::build_L_from_raw(&params[chol_start], n_coefs, L_flat.data())) {
                    return;
                }
                tulpa::lkj_log_prior_grad_add(&params[chol_start], L_flat.data(), n_coefs,
                                              /*eta=*/2.0, &grad[chol_start]);

                // Pre-compute re = diag(sigma) * L * z for observation loop
                if (re_nc_flat_c.empty()) re_nc_flat_c.assign(params.size(), 0.0);
                tulpa::compute_u_eff(L_flat.data(), n_coefs, sigmas.data(),
                                      &params[re_start_t], n_groups,
                                      &re_nc_flat_c[re_start_t]);

                // z prior: N(0,I) -> grad = -z
                for (int g = 0; g < n_groups; g++)
                    for (int c = 0; c < n_coefs; c++)
                        grad[re_start_t + g * n_coefs + c] = -params[re_start_t + g * n_coefs + c];

                nc_L_flats[t] = std::move(L_flat);
                nc_sigmas_vec[t] = std::move(sigmas);
            } else {
                // Uncorrelated slopes
                nc_L_flats[t].clear();
                nc_sigmas_vec[t] = sigmas;
                if (slopes_nc) {
                    for (int g = 0; g < n_groups; g++)
                        for (int c = 0; c < n_coefs; c++)
                            grad[re_start_t + g * n_coefs + c] = -params[re_start_t + g * n_coefs + c];
                } else {
                    for (int c = 0; c < n_coefs; c++) {
                        double tau_c = 1.0 / (sigmas[c] * sigmas[c] + 1e-10);
                        double sigma_grad_c = 0.0;
                        for (int g = 0; g < n_groups; g++) {
                            double re_gc = params[re_start_t + g * n_coefs + c];
                            grad[re_start_t + g * n_coefs + c] = -tau_c * re_gc;
                            sigma_grad_c += tau_c * re_gc * re_gc - 1.0;
                        }
                        grad[layout.log_sigma_re_slopes[t][c]] += sigma_grad_c;
                    }
                }
            }
        }
    }

    // =========================================================================
    // Phase 3: Vectorized 3-pass observation loop
    // =========================================================================
    // Pass 1: Assemble eta vectors (Eigen matvec + feature additions)
    // Pass 2: Compute residuals + phi gradients (tight family-dispatched loop)
    // Pass 3: Scatter gradients (Eigen X^T*resid + feature-specific scatter)
    // =========================================================================
    static thread_local std::vector<double> grad_spatial_lik, grad_theta_lik;
    static thread_local std::vector<double> grad_temporal_lik;
    static thread_local std::vector<double> grad_hsgp_f;
    static thread_local std::vector<double> grad_delta_lik;
    static thread_local std::vector<double> grad_tvc_w;
    static thread_local std::vector<double> grad_factors_c;
    static thread_local std::vector<double> grad_svc_f;
    static thread_local std::vector<double> grad_msgp_f;

    static thread_local std::vector<double> grad_gp_w_lik;
    if (has_gp_spatial) grad_gp_w_lik.assign(N_gp_c, 0.0);
    if (has_msgp_hsgp) grad_msgp_f.assign(N, 0.0);
    if (has_icar_bym2) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    if (has_icar_bym2 && layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);
    if (has_gmrf_temporal) grad_temporal_lik.assign(T_temporal, 0.0);
    if (has_temporal_gp) grad_temporal_lik.assign(data.temporal_gp_data.n_groups * T_gp, 0.0);
    if (layout.is_hsgp && data.has_hsgp) grad_hsgp_f.assign(N, 0.0);
    if (has_st) grad_delta_lik.assign(ST_n, 0.0);
    if (layout.has_tvc && data.has_tvc) grad_tvc_w.assign(n_w, 0.0);
    if (K_latent > 0) grad_factors_c.assign(N * K_latent, 0.0);
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) grad_svc_f.assign(n_svc * N, 0.0);

    // --- Pass 1: Assemble eta_num and eta_denom vectors ---
    static thread_local std::vector<double> eta_num_v, eta_denom_v;
    eta_num_v.resize(N);
    eta_denom_v.resize(N);

    // X * beta via Eigen matvec
    {
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_num_mat(data.legacy.X_num_flat.data(), N, data.legacy.p_num);
        Eigen::Map<const Eigen::VectorXd> beta_num_vec(beta_num, data.legacy.p_num);
        Eigen::Map<Eigen::VectorXd>(eta_num_v.data(), N) = X_num_mat * beta_num_vec;
    }
    if (!is_binomial) {
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_denom_mat(data.legacy.X_denom_flat.data(), N, data.legacy.p_denom);
        Eigen::Map<const Eigen::VectorXd> beta_denom_vec(beta_denom, data.legacy.p_denom);
        Eigen::Map<Eigen::VectorXd>(eta_denom_v.data(), N) = X_denom_mat * beta_denom_vec;
    } else {
        std::memset(eta_denom_v.data(), 0, N * sizeof(double));
    }

    // RE (simple intercept)
    if (layout.has_re && !has_slopes) {
        for (int i = 0; i < N; i++) if (data.re_group[i] > 0) {
            double re_eff = re_value_for_eta(re, data.re_group[i] - 1, sigma_re, data.re_parameterization);
            eta_num_v[i] += re_eff;
            if (!is_binomial) eta_denom_v[i] += re_eff;
        }
    }

    // RE (slopes - multi-term)
    // For correlated NC: use pre-computed re_nc_flat_c (= diag(sigma) * L * z)
    // For uncorrelated NC: use sigma * z inline
    // For centered: use raw params
    if (has_slopes) {
        for (int i = 0; i < N; i++) {
            for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
                int n_coefs = layout.re_n_coefs_multi[t_re];
                int re_start_t = layout.re_start_multi[t_re];
                bool is_corr_nc = !nc_L_flats.empty() && t_re < (int)nc_L_flats.size() && !nc_L_flats[t_re].empty();
                int g = -1;
                if (!data.re_group_multi_flat.empty())
                    g = data.re_group_multi_flat[i * data.n_re_terms + t_re] - 1;
                else if (data.re_group[i] > 0)
                    g = data.re_group[i] - 1;
                if (g < 0) continue;

                // Intercept (coef 0)
                double re_val_0;
                if (is_corr_nc) {
                    re_val_0 = re_nc_flat_c[re_start_t + g * n_coefs + 0];
                } else {
                    re_val_0 = params[re_start_t + g * n_coefs + 0];
                    if (slopes_nc) re_val_0 *= std::exp(params[layout.log_sigma_re_slopes[t_re][0]]);
                }
                eta_num_v[i] += re_val_0;
                if (!is_binomial) eta_denom_v[i] += re_val_0;

                // Slopes (coef 1+)
                int n_slopes = n_coefs - 1;
                if (n_slopes > 0 && t_re < (int)data.re_slope_matrices.size() && !data.re_slope_matrices[t_re].empty()) {
                    for (int s = 0; s < n_slopes; s++) {
                        double re_val_s;
                        if (is_corr_nc) {
                            re_val_s = re_nc_flat_c[re_start_t + g * n_coefs + 1 + s];
                        } else {
                            re_val_s = params[re_start_t + g * n_coefs + 1 + s];
                            if (slopes_nc) re_val_s *= std::exp(params[layout.log_sigma_re_slopes[t_re][1 + s]]);
                        }
                        double eff = re_val_s * data.re_slope_matrices[t_re][i * n_slopes + s];
                        eta_num_v[i] += eff;
                        if (!is_binomial) eta_denom_v[i] += eff;
                    }
                }
            }
        }
    }

    // ICAR/BYM2 spatial
    if (has_icar_bym2) {
        for (int i = 0; i < N; i++) if (data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            double spatial_eff = layout.is_bym2
                ? sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit]
                : spatial_phi[s_unit];
            eta_num_v[i] += spatial_eff;
            if (!is_binomial) eta_denom_v[i] += spatial_eff;
        }
    }

    // GP (NNGP) spatial
    if (has_gp_spatial) {
        for (int i = 0; i < N; i++) {
            double gp_eff = gp_w_c[data.gp_data.obs_to_loc[i]];
            eta_num_v[i] += gp_eff;
            if (!is_binomial && data.gp_data.shared) eta_denom_v[i] += gp_eff;
        }
    }

    // HSGP spatial (vectorized Eigen add)
    if (layout.is_hsgp && data.has_hsgp) {
        Eigen::Map<Eigen::VectorXd>(eta_num_v.data(), N) +=
            Eigen::Map<const Eigen::VectorXd>(hsgp_ws.hsgp_f.data(), N);
        if (!is_binomial && data.hsgp_data.shared)
            Eigen::Map<Eigen::VectorXd>(eta_denom_v.data(), N) +=
                Eigen::Map<const Eigen::VectorXd>(hsgp_ws.hsgp_f.data(), N);
    }

    // MSGP-HSGP spatial (vectorized)
    if (has_msgp_hsgp) {
        for (int i = 0; i < N; i++) {
            double ms_spatial = msgp_ws_local_c.hsgp_f[i] + msgp_ws_regional_c.hsgp_f[i];
            eta_num_v[i] += ms_spatial;
            if (!is_binomial && data.multiscale_gp_data.shared) eta_denom_v[i] += ms_spatial;
        }
    }

    // Temporal GMRF
    if (has_gmrf_temporal && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) if (data.temporal_time_idx[i] > 0) {
            int t_base = (data.temporal_group_idx[i] - 1) * data.n_times + data.temporal_time_idx[i] - 1;
            if (t_base >= 0 && t_base < T_temporal) {
                eta_num_v[i] += phi_temporal[t_base];
                if (!is_binomial && data.temporal_shared) eta_denom_v[i] += phi_temporal[t_base];
            }
        }
    }

    // Temporal GP
    if (has_temporal_gp && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) if (data.temporal_time_idx[i] > 0) {
            int t_base = (data.temporal_group_idx[i] - 1) * T_gp + data.temporal_time_idx[i] - 1;
            if (t_base >= 0 && t_base < (int)temporal_gp_f.size()) {
                eta_num_v[i] += temporal_gp_f[t_base];
                if (!is_binomial && data.temporal_shared) eta_denom_v[i] += temporal_gp_f[t_base];
            }
        }
    }

    // TVC (already precomputed)
    if (layout.has_tvc && data.has_tvc) {
        for (int i = 0; i < N; i++) {
            eta_num_v[i] += tvc_eta_precomp[i];
            if (!is_binomial) eta_denom_v[i] += tvc_eta_precomp[i];
        }
    }

    // SVC (already precomputed)
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        for (int i = 0; i < N; i++) {
            eta_num_v[i] += svc_eta_precomp[i];
            if (!is_binomial) eta_denom_v[i] += svc_eta_precomp[i];
        }
    }

    // Latent factors (already precomputed)
    if (K_latent > 0) {
        for (int i = 0; i < N; i++) {
            eta_num_v[i] += latent_eta_precomp[i];
            if (data.latent_shared) eta_denom_v[i] += latent_eta_precomp[i];
        }
    }

    // Multiscale temporal
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        for (int i = 0; i < N; i++) if (!mst.time_index.empty() && i < (int)mst.time_index.size() && mst.time_index[i] > 0) {
            int t_idx = mst.time_index[i] - 1;
            obs_t_idx_ms_c[i] = t_idx;
            eta_num_v[i] += ms_effect_by_time_c[t_idx];
            if (!is_binomial && mst.shared) eta_denom_v[i] += ms_effect_by_time_c[t_idx];
        }
    }

    // Spatiotemporal
    if (has_st) {
        for (int i = 0; i < N; i++) {
            double st_eff = 0.0;
            if (data.st_is_hsgp) {
                int t = data.spatiotemporal_data.t_idx[i] - 1;
                int M_st = data.st_hsgp_data.m_total;
                int T_st_c = data.spatiotemporal_data.n_times;
                for (int j = 0; j < M_st; j++)
                    st_eff += data.st_hsgp_data.phi_flat[i * M_st + j] * st_delta[j * T_st_c + t];
            } else if (data.spatiotemporal_data.st_flat[i] > 0) {
                st_eff = st_delta[data.spatiotemporal_data.st_flat[i] - 1];
            }
            eta_num_v[i] += st_eff;
            if (!is_binomial && data.spatiotemporal_data.shared) eta_denom_v[i] += st_eff;
        }
    }

    // --- Pass 2: Compute residuals + phi gradients (tight family-dispatched loop) ---
    static thread_local std::vector<double> dLL_num_v, dLL_denom_v;
    dLL_num_v.resize(N);
    dLL_denom_v.resize(N);

    // ZI/OI-aware accumulators for per-observation logit gradients
    static thread_local std::vector<double> grad_logit_zi_v, grad_logit_oi_v;
    const bool has_zi_oi = layout.has_zi || layout.has_oi;
    if (has_zi_oi) {
        grad_logit_zi_v.assign(N, 0.0);
        grad_logit_oi_v.assign(N, 0.0);
    }

    if (has_zi_oi) {
        // ZI/OI path: use compute_obs_residuals_zi for all families
        double grad_phi_num_acc = 0.0, grad_phi_denom_acc = 0.0;
        for (int i = 0; i < N; i++) {
            double gp_num_i = 0.0, gp_denom_i = 0.0;
            double gz_i = 0.0, go_i = 0.0;
            compute_obs_residuals_zi(data, layout, i,
                eta_num_v[i], eta_denom_v[i], phi_num, phi_denom,
                beta_zi, beta_oi,
                dLL_num_v[i], dLL_denom_v[i],
                gp_num_i, gp_denom_i, gz_i, go_i);
            grad_phi_num_acc += gp_num_i;
            grad_phi_denom_acc += gp_denom_i;
            grad_logit_zi_v[i] = gz_i;
            grad_logit_oi_v[i] = go_i;
        }
        // Phi gradients are on d(LL)/d(phi) scale; convert to d(LL)/d(log_phi) = phi * d(LL)/d(phi)
        if (layout.legacy.has_phi_num) grad[layout.legacy.log_phi_num_idx] += grad_phi_num_acc * phi_num;
        if (layout.legacy.has_phi_denom) grad[layout.legacy.log_phi_denom_idx] += grad_phi_denom_acc * phi_denom;
    } else if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
        double grad_phi_denom_acc = 0.0;
        for (int i = 0; i < N; i++) {
            double mu_num = std::exp(eta_num_v[i]);
            double mu_denom = std::exp(eta_denom_v[i]);
            dLL_num_v[i] = data.legacy.y_num[i] - mu_num;
            dLL_denom_v[i] = (data.legacy.y_denom_cont[i] > 0.0) ? phi_denom * (data.legacy.y_denom_cont[i] / mu_denom - 1.0) : 0.0;
            if (layout.legacy.has_phi_denom && data.legacy.y_denom_cont[i] > 0.0) {
                grad_phi_denom_acc += (std::log(phi_denom / mu_denom) + 1.0 + std::log(data.legacy.y_denom_cont[i])
                    - tulpa::math::portable_digamma(phi_denom) - data.legacy.y_denom_cont[i] / mu_denom) * phi_denom;
            }
        }
        if (layout.legacy.has_phi_denom) grad[layout.legacy.log_phi_denom_idx] += grad_phi_denom_acc;
    } else if (data.legacy.model_type == ModelType::BINOMIAL) {
        for (int i = 0; i < N; i++) {
            double p = 1.0 / (1.0 + std::exp(-eta_num_v[i]));
            dLL_num_v[i] = data.legacy.y_num[i] - data.legacy.y_denom[i] * p;
            dLL_denom_v[i] = 0.0;
        }
    } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
        double grad_phi_num_acc = 0.0, grad_phi_denom_acc = 0.0;
        for (int i = 0; i < N; i++) {
            double mu_num = std::exp(eta_num_v[i]);
            double mu_denom = std::exp(eta_denom_v[i]);
            dLL_num_v[i] = data.legacy.y_num[i] - mu_num * (data.legacy.y_num[i] + phi_num) / (mu_num + phi_num);
            dLL_denom_v[i] = data.legacy.y_denom[i] - mu_denom * (data.legacy.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
            if (layout.legacy.has_phi_num) {
                double y = data.legacy.y_num[i];
                grad_phi_num_acc += (tulpa::math::portable_digamma(y + phi_num) - tulpa::math::portable_digamma(phi_num)
                    + std::log(phi_num / (mu_num + phi_num)) + 1.0 - (y + phi_num) / (mu_num + phi_num)) * phi_num;
            }
            if (layout.legacy.has_phi_denom) {
                double y = data.legacy.y_denom[i];
                grad_phi_denom_acc += (tulpa::math::portable_digamma(y + phi_denom) - tulpa::math::portable_digamma(phi_denom)
                    + std::log(phi_denom / (mu_denom + phi_denom)) + 1.0 - (y + phi_denom) / (mu_denom + phi_denom)) * phi_denom;
            }
        }
        if (layout.legacy.has_phi_num) grad[layout.legacy.log_phi_num_idx] += grad_phi_num_acc;
        if (layout.legacy.has_phi_denom) grad[layout.legacy.log_phi_denom_idx] += grad_phi_denom_acc;
    } else if (data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
        double grad_phi_num_acc = 0.0, grad_phi_denom_acc = 0.0;
        for (int i = 0; i < N; i++) {
            double mu_num = std::exp(eta_num_v[i]);
            double mu_denom = std::exp(eta_denom_v[i]);
            double denom_nb = mu_num + phi_num;
            dLL_num_v[i] = data.legacy.y_num[i] - mu_num * (data.legacy.y_num[i] + phi_num) / denom_nb;
            dLL_denom_v[i] = (data.legacy.y_denom_cont[i] > 0.0) ? phi_denom * (data.legacy.y_denom_cont[i] / mu_denom - 1.0) : 0.0;
            if (layout.legacy.has_phi_num) {
                double y = data.legacy.y_num[i];
                grad_phi_num_acc += (tulpa::math::portable_digamma(y + phi_num) - tulpa::math::portable_digamma(phi_num)
                    + std::log(phi_num / denom_nb) + 1.0 - (y + phi_num) / denom_nb) * phi_num;
            }
            if (layout.legacy.has_phi_denom && data.legacy.y_denom_cont[i] > 0.0) {
                grad_phi_denom_acc += (std::log(phi_denom / mu_denom) + 1.0 + std::log(data.legacy.y_denom_cont[i])
                    - tulpa::math::portable_digamma(phi_denom) - data.legacy.y_denom_cont[i] / mu_denom) * phi_denom;
            }
        }
        if (layout.legacy.has_phi_num) grad[layout.legacy.log_phi_num_idx] += grad_phi_num_acc;
        if (layout.legacy.has_phi_denom) grad[layout.legacy.log_phi_denom_idx] += grad_phi_denom_acc;
    } else {
        // Fallback: GAMMA_GAMMA, LOGNORMAL, BETA_BINOMIAL - use per-obs helpers
        for (int i = 0; i < N; i++) {
            compute_obs_residuals(data, i, eta_num_v[i], eta_denom_v[i], phi_num, phi_denom,
                                  dLL_num_v[i], dLL_denom_v[i]);
            accumulate_phi_likelihood_grad(data, layout, i, eta_num_v[i], eta_denom_v[i],
                                            phi_num, phi_denom, grad.data());
        }
    }

    // --- Pass 3: Scatter gradients ---

    // Beta gradients via Eigen X^T * resid
    {
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_num_mat(data.legacy.X_num_flat.data(), N, data.legacy.p_num);
        Eigen::Map<Eigen::VectorXd> grad_beta_num(&grad[layout.legacy.beta_num_start], data.legacy.p_num);
        grad_beta_num += X_num_mat.transpose() * Eigen::Map<const Eigen::VectorXd>(dLL_num_v.data(), N);
    }
    if (!is_binomial) {
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_denom_mat(data.legacy.X_denom_flat.data(), N, data.legacy.p_denom);
        Eigen::Map<Eigen::VectorXd> grad_beta_denom(&grad[layout.legacy.beta_denom_start], data.legacy.p_denom);
        grad_beta_denom += X_denom_mat.transpose() * Eigen::Map<const Eigen::VectorXd>(dLL_denom_v.data(), N);
    }

    // ZI/OI beta scatter: grad[beta_zi] += X_zi^T * grad_logit_zi, similarly for OI
    if (has_zi_oi) {
        if (layout.has_zi && data.p_zi > 0) {
            for (int i = 0; i < N; i++)
                for (int j = 0; j < data.p_zi; j++)
                    grad[layout.beta_zi_start + j] += data.X_zi_flat[i * data.p_zi + j] * grad_logit_zi_v[i];
        }
        if (layout.has_oi && data.p_oi > 0) {
            for (int i = 0; i < N; i++)
                for (int j = 0; j < data.p_oi; j++)
                    grad[layout.beta_oi_start + j] += data.X_oi_flat[i * data.p_oi + j] * grad_logit_oi_v[i];
        }
    }

    // RE (simple)
    if (layout.has_re && !has_slopes) {
        for (int i = 0; i < N; i++) if (data.re_group[i] > 0)
            grad[layout.re_start + data.re_group[i] - 1] += dLL_num_v[i] + dLL_denom_v[i];
    }

    // RE (slopes)
    if (has_slopes) {
        for (int i = 0; i < N; i++) {
            double dLL_shared_i = dLL_num_v[i] + dLL_denom_v[i];
            for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
                int n_coefs = layout.re_n_coefs_multi[t_re];
                int g = -1;
                if (!data.re_group_multi_flat.empty())
                    g = data.re_group_multi_flat[i * data.n_re_terms + t_re] - 1;
                else if (data.re_group[i] > 0)
                    g = data.re_group[i] - 1;
                if (g < 0) continue;
                grad_re_slopes_lik[t_re][g * n_coefs + 0] += dLL_shared_i;
                int n_slopes_sc = n_coefs - 1;
                if (n_slopes_sc > 0 && t_re < (int)data.re_slope_matrices.size() && !data.re_slope_matrices[t_re].empty())
                    for (int s = 0; s < n_slopes_sc; s++)
                        grad_re_slopes_lik[t_re][g * n_coefs + 1 + s] += dLL_shared_i * data.re_slope_matrices[t_re][i * n_slopes_sc + s];
            }
        }
    }

    // ICAR/BYM2 spatial
    if (has_icar_bym2) {
        for (int i = 0; i < N; i++) if (data.spatial_group[i] > 0) {
            double dLL_s = dLL_num_v[i] + dLL_denom_v[i];
            int s_unit = data.spatial_group[i] - 1;
            grad_spatial_lik[s_unit] += dLL_s;
            if (layout.is_bym2) grad_theta_lik[s_unit] += dLL_s;
        }
    }

    // GP (NNGP) spatial
    if (has_gp_spatial) {
        for (int i = 0; i < N; i++) {
            double dLL_gp = data.gp_data.shared ? (dLL_num_v[i] + dLL_denom_v[i]) : dLL_num_v[i];
            grad_gp_w_lik[data.gp_data.obs_to_loc[i]] += dLL_gp;
        }
    }

    // HSGP spatial
    if (layout.is_hsgp && data.has_hsgp) {
        if (data.hsgp_data.shared)
            for (int i = 0; i < N; i++) grad_hsgp_f[i] = dLL_num_v[i] + dLL_denom_v[i];
        else
            std::memcpy(grad_hsgp_f.data(), dLL_num_v.data(), N * sizeof(double));
    }

    // MSGP-HSGP spatial
    if (has_msgp_hsgp) {
        if (data.multiscale_gp_data.shared)
            for (int i = 0; i < N; i++) grad_msgp_f[i] = dLL_num_v[i] + dLL_denom_v[i];
        else
            std::memcpy(grad_msgp_f.data(), dLL_num_v.data(), N * sizeof(double));
    }

    // Temporal GMRF
    if (has_gmrf_temporal && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) if (data.temporal_time_idx[i] > 0) {
            int t_base = (data.temporal_group_idx[i] - 1) * data.n_times + data.temporal_time_idx[i] - 1;
            if (t_base >= 0 && t_base < T_temporal)
                grad_temporal_lik[t_base] += data.temporal_shared ? (dLL_num_v[i] + dLL_denom_v[i]) : dLL_num_v[i];
        }
    }

    // Temporal GP
    if (has_temporal_gp && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) if (data.temporal_time_idx[i] > 0) {
            int t_base = (data.temporal_group_idx[i] - 1) * T_gp + data.temporal_time_idx[i] - 1;
            if (t_base >= 0 && t_base < (int)temporal_gp_f.size())
                grad_temporal_lik[t_base] += data.temporal_shared ? (dLL_num_v[i] + dLL_denom_v[i]) : dLL_num_v[i];
        }
    }

    // TVC
    if (layout.has_tvc && data.has_tvc) {
        for (int i = 0; i < N; i++) {
            double dLL_s = dLL_num_v[i] + dLL_denom_v[i];
            int t = data.tvc_data.time_index[i] - 1;
            int g = data.tvc_data.group_index[i] - 1;
            for (int j = 0; j < n_tvc; j++)
                grad_tvc_w[(g * n_tvc + j) * n_tvc_times + t] += dLL_s * data.tvc_data.X_tvc[i * n_tvc + j];
        }
    }

    // SVC
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        for (int i = 0; i < N; i++) {
            double dLL_s = dLL_num_v[i] + dLL_denom_v[i];
            for (int j = 0; j < n_svc; j++)
                grad_svc_f[j * N + i] = dLL_s * data.svc_data.X_svc[i * n_svc + j];
        }
    }

    // Latent factors
    if (K_latent > 0) {
        for (int i = 0; i < N; i++) {
            double dLL_latent = data.latent_shared ? (dLL_num_v[i] + dLL_denom_v[i]) : dLL_num_v[i];
            for (int k = 0; k < K_latent; k++) {
                grad_factors_c[i * K_latent + k] += dLL_latent * sigma_latent_vec[k];
                grad[layout.log_sigma_latent_start + k] += dLL_latent * factors_constrained[i * K_latent + k] * sigma_latent_vec[k];
            }
        }
    }

    // Multiscale temporal
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        for (int i = 0; i < N; i++) if (obs_t_idx_ms_c[i] >= 0) {
            int t_idx = obs_t_idx_ms_c[i];
            double dLL_temporal = mst.shared ? (dLL_num_v[i] + dLL_denom_v[i]) : dLL_num_v[i];
            if (trend != nullptr && t_idx < n_trend) grad_trend_lik_c[t_idx] += dLL_temporal;
            if (seasonal != nullptr && mst.seasonal_period > 0) {
                int s_idx = t_idx % mst.seasonal_period;
                if (s_idx < n_seasonal) grad_seasonal_lik_c[s_idx] += dLL_temporal;
            }
            if (short_term != nullptr && t_idx < n_short) grad_short_lik_c[t_idx] += dLL_temporal;
        }
    }

    // ST interaction
    if (has_st) {
        for (int i = 0; i < N; i++) {
            double dLL_st = data.spatiotemporal_data.shared ? (dLL_num_v[i] + dLL_denom_v[i]) : dLL_num_v[i];
            if (data.st_is_hsgp) {
                int t = data.spatiotemporal_data.t_idx[i] - 1;
                int M_st = data.st_hsgp_data.m_total;
                int T_st_c = data.spatiotemporal_data.n_times;
                for (int j = 0; j < M_st; j++)
                    grad_delta_lik[j * T_st_c + t] += data.st_hsgp_data.phi_flat[i * M_st + j] * dLL_st;
            } else if (data.spatiotemporal_data.st_flat[i] > 0) {
                grad_delta_lik[data.spatiotemporal_data.st_flat[i] - 1] += dLL_st;
            }
        }
    }

    // Phi likelihood gradients already accumulated in Pass 2

    // =========================================================================
    // Phase 4: Structural/GMRF prior gradients (post-observation loop)
    // =========================================================================

    // ICAR/BYM2 spatial GMRF
    if (has_icar_bym2) {
        spatial_gmrf_prior_grad(data, layout, spatial_phi, tau_spatial,
                                sigma_s_bym2, sigma_u_bym2, rho_bym2, theta_bym2,
                                grad_spatial_lik.data(),
                                layout.is_bym2 ? grad_theta_lik.data() : nullptr,
                                grad.data());
    }

    // GP (NNGP) backward pass
    if (has_gp_spatial) {
        if (use_nc_gp) {
            // NC: dL/dw -> grad_z, grad_log_sigma2, grad_log_phi via backward pass
            std::vector<double> grad_z_gp(N_gp_c, 0.0);
            double grad_log_sigma2_gp_bw = 0.0, grad_log_phi_gp_bw = 0.0, grad_log_phi_gp_jac = 0.0;
            tulpa_gp::nngp_nc_backward(
                &params[layout.gp_w_start], sigma2_gp_c, phi_gp_c,
                data.gp_data, nc_ws_gp_c, grad_gp_w_lik.data(),
                grad_z_gp.data(), grad_log_sigma2_gp_bw, grad_log_phi_gp_bw, grad_log_phi_gp_jac);
            for (int i = 0; i < N_gp_c; i++)
                grad[layout.gp_w_start + i] += grad_z_gp[i];
            grad[layout.log_sigma2_gp_idx] += grad_log_sigma2_gp_bw;
            grad[layout.log_phi_gp_idx] += grad_log_phi_gp_bw + grad_log_phi_gp_jac;
        } else {
            // Centered: just add likelihood gradients to w
            for (int i = 0; i < N_gp_c; i++)
                grad[layout.gp_w_start + i] += grad_gp_w_lik[i];
        }
    }

    // HSGP spectral density gradients
    if (layout.is_hsgp && data.has_hsgp) {
        int M = data.hsgp_data.m_total;
        std::copy(grad_hsgp_f.begin(), grad_hsgp_f.end(), hsgp_ws.grad_f.begin());
        double grad_log_sigma2 = 0.0, grad_log_lengthscale = 0.0;
        tulpa_hsgp::hsgp_compute_gradients_ws(
            hsgp_beta_ptr, hsgp_sigma2, hsgp_lengthscale,
            data.hsgp_data, hsgp_ws, grad_log_sigma2, grad_log_lengthscale);
        grad[layout.log_sigma2_hsgp_idx] += grad_log_sigma2;
        grad[layout.log_lengthscale_hsgp_idx] += grad_log_lengthscale;
        for (int j = 0; j < M; j++)
            grad[layout.hsgp_beta_start + j] += hsgp_ws.grad_beta_out[j];
    }

    // MSGP-HSGP spectral density gradients
    if (has_msgp_hsgp) {
        std::memcpy(msgp_ws_local_c.grad_f.data(), grad_msgp_f.data(), N * sizeof(double));
        std::memcpy(msgp_ws_regional_c.grad_f.data(), grad_msgp_f.data(), N * sizeof(double));

        double grad_log_sigma2_local_c = 0.0, grad_log_ls_local_c = 0.0;
        tulpa_hsgp::hsgp_compute_gradients_ws(
            msgp_beta_local, msgp_sigma2_local, msgp_ls_local,
            data.msgp_hsgp_data, msgp_ws_local_c,
            grad_log_sigma2_local_c, grad_log_ls_local_c);

        double grad_log_sigma2_regional_c = 0.0, grad_log_ls_regional_c = 0.0;
        tulpa_hsgp::hsgp_compute_gradients_ws(
            msgp_beta_regional, msgp_sigma2_regional, msgp_ls_regional,
            data.msgp_hsgp_data, msgp_ws_regional_c,
            grad_log_sigma2_regional_c, grad_log_ls_regional_c);

        for (int j = 0; j < msgp_m_total; j++) {
            grad[layout.gp_local_start + j] += msgp_ws_local_c.grad_beta_out[j];
            grad[layout.gp_regional_start + j] += msgp_ws_regional_c.grad_beta_out[j];
        }
        grad[layout.log_sigma2_gp_local_idx] += grad_log_sigma2_local_c;
        grad[layout.log_phi_gp_local_idx] += grad_log_ls_local_c;
        grad[layout.log_sigma2_gp_regional_idx] += grad_log_sigma2_regional_c;
        grad[layout.log_phi_gp_regional_idx] += grad_log_ls_regional_c;
    }

    // Temporal GMRF prior
    if (has_gmrf_temporal && T_temporal > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                                 phi_temporal, T_temporal, grad_temporal_lik.data(), grad.data());
    }

    // Temporal GP backward pass
    if (has_temporal_gp) {
        int T_len_gp = layout.temporal_end - layout.temporal_start;

        if (use_nc_tgp) {
            // Copy likelihood gradients to nc workspace
            for (int k = 0; k < n_groups_gp * T_gp; k++)
                nc_ws_composite.dL_df[k] = grad_temporal_lik[k];

            double grad_log_sigma2_gp = 0.0, grad_log_phi_gp = 0.0;
            tulpa_temporal_gp::temporal_gp_nc_backward(
                z_temporal_gp, T_gp, n_groups_gp, sigma2_tgp_comp, phi_tgp_comp,
                data.temporal_gp_data.time_values, nc_ws_composite,
                &grad[layout.temporal_start],
                grad_log_sigma2_gp, grad_log_phi_gp);
            grad[layout.log_sigma2_temporal_gp_idx] += grad_log_sigma2_gp;
            // Convert log_phi gradient to logit_phi gradient
            double chi_tgp = (phi_tgp_comp - phi_lower_tgp) * (phi_upper_tgp - phi_tgp_comp) /
                             (phi_tgp_comp * (phi_upper_tgp - phi_lower_tgp));
            grad[layout.logit_phi_temporal_gp_idx] += grad_log_phi_gp * chi_tgp;
        } else {
            // Centered: just add likelihood gradients
            for (int k = 0; k < T_len_gp; k++)
                grad[layout.temporal_start + k] = grad_temporal_lik[k];
        }
    }

    // Multiscale temporal GMRF prior gradients
    if (has_ms_temporal) {
        tulpa_temporal_grad::MultiscaleTemporalGradients ms_grads;
        tulpa_temporal_grad::multiscale_temporal_prior_gradients(
            trend, n_trend, seasonal, n_seasonal, short_term, n_short,
            sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
            data.multiscale_temporal_data, ms_grads);
        for (int t = 0; t < n_trend; t++) grad[layout.trend_start + t] = grad_trend_lik_c[t] + ms_grads.grad_trend[t];
        for (int t = 0; t < n_seasonal; t++) grad[layout.seasonal_start + t] = grad_seasonal_lik_c[t] + ms_grads.grad_seasonal[t];
        for (int t = 0; t < n_short; t++) grad[layout.short_term_start + t] = grad_short_lik_c[t] + ms_grads.grad_short_term[t];
        if (n_trend > 0) grad[layout.log_sigma2_trend_idx] += ms_grads.grad_log_sigma2_trend;
        if (n_seasonal > 0) grad[layout.log_sigma2_seasonal_idx] += ms_grads.grad_log_sigma2_seasonal;
        if (n_short > 0) grad[layout.log_sigma2_short_idx] += ms_grads.grad_log_sigma2_short;
        if (data.multiscale_temporal_data.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0)
            grad[layout.logit_rho_short_idx] += ms_grads.grad_logit_rho_short;
    }

    // TVC structural prior gradients
    if (layout.has_tvc && data.has_tvc) {
        // Initialize TVC gradient workspace
        static thread_local tulpa_tvc::TVCGradientWS tvc_grad_ws;
        static thread_local std::vector<double> tvc_grad_w_buf, tvc_grad_log_tau_buf;
        static thread_local std::vector<double> tvc_grad_logit_rho_buf, tvc_grad_w_jg_buf, tvc_d_buf;
        tvc_grad_w_buf.assign(n_w, 0.0);
        tvc_grad_log_tau_buf.assign(n_tvc, 0.0);
        tvc_grad_logit_rho_buf.assign(n_tvc, 0.0);
        tvc_grad_w_jg_buf.resize(n_tvc_times);
        tvc_d_buf.resize(n_tvc_times);

        tvc_grad_ws.grad_w = tvc_grad_w_buf.data();
        tvc_grad_ws.grad_log_tau = tvc_grad_log_tau_buf.data();
        tvc_grad_ws.grad_logit_rho = tvc_grad_logit_rho_buf.data();
        tvc_grad_ws.grad_w_jg = tvc_grad_w_jg_buf.data();
        tvc_grad_ws.d_buf = tvc_d_buf.data();
        tvc_grad_ws.n_w = n_w;
        tvc_grad_ws.n_tvc = n_tvc;

        tulpa_tvc::tvc_prior_gradients_ws(
            tvc_w_flat_buf.data(), data.tvc_data,
            tvc_tau_buf.data(), tvc_rho_buf.data(), tvc_grad_ws);

        // Add likelihood + prior to main gradient
        for (int k = 0; k < n_w; k++)
            grad[layout.tvc_w_start + k] += grad_tvc_w[k] + tvc_grad_w_buf[k];
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.log_tau_tvc_start + j] += tvc_grad_log_tau_buf[j];
            if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1)
                grad[layout.logit_rho_tvc_start + j] += tvc_grad_logit_rho_buf[j];
        }
    }

    // SVC spectral density gradients
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        for (int j = 0; j < n_svc; j++) {
            double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
            double lengthscale_j = std::exp(params[layout.log_phi_svc_start + j]);
            const double* beta_j = &params[layout.svc_w_start + j * svc_m_total];

            // Re-evaluate to cache sqrt_S
            tulpa_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                           data.svc_hsgp_data, svc_hsgp_ws);

            // Copy per-SVC-term grad_f
            for (int i = 0; i < N; i++) svc_hsgp_ws.grad_f[i] = grad_svc_f[j * N + i];

            double gls2 = 0.0, gll = 0.0;
            tulpa_hsgp::hsgp_compute_gradients_ws(
                beta_j, sigma2_j, lengthscale_j,
                data.svc_hsgp_data, svc_hsgp_ws, gls2, gll);

            grad[layout.log_sigma2_svc_start + j] += gls2;
            grad[layout.log_phi_svc_start + j] += gll;
            for (int m = 0; m < svc_m_total; m++)
                grad[layout.svc_w_start + j * svc_m_total + m] += svc_hsgp_ws.grad_beta_out[m];
        }
    }

    // Latent factor constraint chain rule
    if (K_latent > 0) {
        // N(0,1) prior on constrained factors
        for (int k = 0; k < K_latent; k++)
            for (int i = 0; i < N; i++)
                grad_factors_c[i * K_latent + k] -= factors_constrained[i * K_latent + k];

        // Chain rule: constrained -> raw
        if (data.latent_constraint == 0) {
            // Sum-to-zero: d/d(raw[i,k]) = d/d(constrained[i,k]) - mean(d/d(constrained[:,k]))
            for (int k = 0; k < K_latent; k++) {
                double sum_gc = 0.0;
                for (int i = 0; i < N; i++) sum_gc += grad_factors_c[i * K_latent + k];
                double mean_gc = sum_gc / N;
                for (int i = 0; i < N; i++)
                    grad[layout.latent_factor_start + i * K_latent + k] += grad_factors_c[i * K_latent + k] - mean_gc;
            }
        } else {
            for (int j = 0; j < N * K_latent; j++)
                grad[layout.latent_factor_start + j] += grad_factors_c[j];
        }
    }

    // ST interaction prior gradients
    if (has_st && data.st_is_hsgp) {
        // HSGP-ST: per-basis-function temporal GMRF with spectral precision scaling
        int M_st = data.st_hsgp_data.m_total;
        int T_st_h = data.spatiotemporal_data.n_times;
        double sigma2_st_h = std::exp(params[layout.log_sigma2_st_hsgp_idx]);
        double ls_st_h = std::exp(params[layout.log_lengthscale_st_hsgp_idx]);

        int rank_t = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T_st_h - 1) :
                     (data.spatiotemporal_data.temporal_type == TemporalType::RW2) ? (T_st_h - 2) : T_st_h;
        if (data.spatiotemporal_data.temporal_cyclic) rank_t = T_st_h;

        double grad_log_sigma2_st = 0.0;
        double grad_log_ls_st = 0.0;

        for (int j = 0; j < M_st; j++) {
            double omega_sq = data.st_hsgp_data.eigenvalues[j];
            double S_j = tulpa_hsgp::spectral_density_se(omega_sq, sigma2_st_h, ls_st_h);
            double S_j_safe = std::max(S_j, 1e-10);
            double prec_j = tau_st / S_j_safe;

            // Temporal GMRF stencil for basis function j
            const double* dj = &st_delta[j * T_st_h];
            double qf = 0.0;
            if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                for (int t = 0; t < T_st_h; t++) {
                    double g = 0.0;
                    if (t > 0) { g += prec_j * (dj[t-1] - dj[t]); qf += (dj[t] - dj[t-1]) * (dj[t] - dj[t-1]); }
                    if (t < T_st_h - 1) g += prec_j * (dj[t+1] - dj[t]);
                    grad[layout.st_delta_start + j * T_st_h + t] = grad_delta_lik[j * T_st_h + t] + g;
                }
            } else {
                // RW2 or other - use generic stencil
                for (int t = 0; t < T_st_h; t++)
                    grad[layout.st_delta_start + j * T_st_h + t] = grad_delta_lik[j * T_st_h + t];
            }

            // Sum-to-zero per basis function
            double sum_j = 0.0;
            for (int t = 0; t < T_st_h; t++) sum_j += dj[t];
            for (int t = 0; t < T_st_h; t++)
                grad[layout.st_delta_start + j * T_st_h + t] -= 0.001 * sum_j;

            // Tau gradient contribution from this basis function
            // d/d(log_tau) [0.5*rank*log(prec_j) - 0.5*prec_j*qf]
            //   = 0.5*rank - 0.5*prec_j*qf  (since d(prec_j)/d(log_tau) = prec_j)
            grad[layout.log_tau_st_idx] += 0.5 * rank_t - 0.5 * prec_j * qf;

            // Sigma2/lengthscale gradient through S_j
            // d/d(log_sigma2) S_j = S_j (SE kernel)
            // d/d(log_sigma2) [0.5*rank*log(tau/S_j) - 0.5*tau/S_j*qf]
            //   = -0.5*rank + 0.5*prec_j*qf  (times sigma2, cancels with chain rule)
            double dS_dsigma2 = S_j_safe / sigma2_st_h;
            double dLogPrior_dS = -0.5 * rank_t / S_j_safe + 0.5 * tau_st * qf / (S_j_safe * S_j_safe);
            grad_log_sigma2_st += dLogPrior_dS * dS_dsigma2 * sigma2_st_h;

            // d/d(log_ell) S_j = S_j * (1/ell - ell*omega_sq) * ell
            double dS_dl = S_j_safe * (1.0 / ls_st_h - ls_st_h * omega_sq);
            grad_log_ls_st += dLogPrior_dS * dS_dl * ls_st_h;
        }

        // HSGP-ST hyperparameter priors + likelihood chain rule
        // PC prior on sigma: d/d(log_sigma2) [-4.6*sigma + 0.5*log_sigma2] = -0.5*4.6*sigma + 0.5
        grad[layout.log_sigma2_st_hsgp_idx] = -0.5 * 4.6 * std::sqrt(sigma2_st_h) + 0.5 + grad_log_sigma2_st;
        // LogNormal(0,1) on lengthscale
        grad[layout.log_lengthscale_st_hsgp_idx] = -params[layout.log_lengthscale_st_hsgp_idx] + grad_log_ls_st;

    } else if (has_st) {
        const auto& st = data.spatiotemporal_data;
        if (st.type == STType::TYPE_I) {
            double qf = 0.0;
            for (int k = 0; k < ST_n; k++) {
                grad[layout.st_delta_start + k] = grad_delta_lik[k] - tau_st * st_delta[k];
                qf += st_delta[k] * st_delta[k];
            }
            grad[layout.log_tau_st_idx] += 0.5 * ST_n - 0.5 * tau_st * qf;
        } else if (st.type == STType::TYPE_II) {
            double total_qf = 0.0;
            for (int s = 0; s < S_st; s++) {
                const double* delta_s = &st_delta[s * T_st];
                if (st.temporal_type == TemporalType::RW1) {
                    double qf = 0.0;
                    for (int t = 0; t < T_st; t++) {
                        double g = 0.0;
                        if (t > 0) { g += tau_st * (delta_s[t-1] - delta_s[t]); qf += std::pow(delta_s[t] - delta_s[t-1], 2); }
                        if (t < T_st - 1) g += tau_st * (delta_s[t+1] - delta_s[t]);
                        grad[layout.st_delta_start + s * T_st + t] = grad_delta_lik[s * T_st + t] + g;
                    }
                    total_qf += qf;
                } else if (st.temporal_type == TemporalType::AR1) {
                    // AR1: precision Q = tau * tridiag(-rho, 1+rho^2, -rho), first/last diagonal = 1
                    // ST interaction doesn't have its own rho - use IID (rho=0) as fallback
                    // This is equivalent to tau * I for the ST delta prior
                    double qf = 0.0;
                    for (int t = 0; t < T_st; t++) {
                        grad[layout.st_delta_start + s * T_st + t] = grad_delta_lik[s * T_st + t] - tau_st * delta_s[t];
                        qf += delta_s[t] * delta_s[t];
                    }
                    total_qf += qf;
                } else {
                    // Fallback: at minimum write likelihood gradient
                    for (int t = 0; t < T_st; t++)
                        grad[layout.st_delta_start + s * T_st + t] = grad_delta_lik[s * T_st + t];
                }
            }
            int rank_per_unit = (st.temporal_type == TemporalType::RW1) ? (T_st - 1) :
                                (st.temporal_type == TemporalType::RW2) ? (T_st - 2) : T_st;
            grad[layout.log_tau_st_idx] += 0.5 * S_st * rank_per_unit - 0.5 * tau_st * total_qf;
        } else if (st.type == STType::TYPE_III) {
            double total_qf = 0.0;
            for (int t = 0; t < T_st; t++) {
                for (int s = 0; s < S_st; s++) {
                    double icar_grad = 0.0;
                    for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                        int j = st.adj_col_idx[idx] - 1;
                        icar_grad += tau_st * (st_delta[j * T_st + t] - st_delta[s * T_st + t]);
                    }
                    grad[layout.st_delta_start + s * T_st + t] = grad_delta_lik[s * T_st + t] + icar_grad;
                }
                for (int s = 0; s < S_st; s++) {
                    for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                        int j = st.adj_col_idx[idx] - 1;
                        if (j > s) total_qf += std::pow(st_delta[s * T_st + t] - st_delta[j * T_st + t], 2);
                    }
                }
            }
            grad[layout.log_tau_st_idx] += 0.5 * T_st * (S_st - 1) - 0.5 * tau_st * total_qf;
        } else if (st.type == STType::TYPE_IV) {
            // Kronecker: Q_delta = Q_s - Q_t (analytical gradient, same as specialized ST function)
            const double* stencil_input = st_use_nc ? z_or_delta_st : st_delta;
            double inv_scale_nc = st_use_nc ? (1.0 / std::sqrt(tau_st)) : 1.0;

            // Step 1: Apply temporal stencil: v[s,t] = (Q_t * input[s,:])_t
            static thread_local std::vector<double> v_kron;
            v_kron.assign(S_st * T_st, 0.0);
            if (st.temporal_type == TemporalType::RW1) {
                for (int s = 0; s < S_st; s++) {
                    for (int t = 0; t < T_st; t++) {
                        double qt_val = 0.0;
                        int n_t_neigh = 0;
                        if (t > 0) { qt_val -= stencil_input[s * T_st + t - 1]; n_t_neigh++; }
                        if (t < T_st - 1) { qt_val -= stencil_input[s * T_st + t + 1]; n_t_neigh++; }
                        qt_val += n_t_neigh * stencil_input[s * T_st + t];
                        v_kron[s * T_st + t] = qt_val;
                    }
                }
            } else if (st.temporal_type == TemporalType::RW2) {
                for (int s = 0; s < S_st; s++) {
                    const double* d_s = &stencil_input[s * T_st];
                    double* v_s = &v_kron[s * T_st];
                    if (T_st >= 3) {
                        const int n_d2 = T_st - 2;
                        double d2_stack[64];
                        double* d2 = (n_d2 <= 64) ? d2_stack : new double[n_d2];
                        for (int k = 0; k < n_d2; k++) d2[k] = d_s[k] - 2.0 * d_s[k + 1] + d_s[k + 2];
                        v_s[0] = d2[0];
                        v_s[1] = -2.0 * d2[0];
                        if (n_d2 > 1) v_s[1] += d2[1];
                        for (int t = 2; t < T_st - 2; t++) v_s[t] = d2[t - 2] - 2.0 * d2[t - 1] + d2[t];
                        if (T_st >= 4) v_s[T_st - 2] = d2[n_d2 - 2] - 2.0 * d2[n_d2 - 1];
                        else v_s[T_st - 2] = -2.0 * d2[0];
                        v_s[T_st - 1] = d2[n_d2 - 1];
                        if (n_d2 > 64) delete[] d2;
                    }
                }
            }

            // Step 2: Apply spatial ICAR stencil to v: (Q_s - Q_t) * input
            double total_qf = 0.0;
            for (int s = 0; s < S_st; s++) {
                for (int t = 0; t < T_st; t++) {
                    double qs_v = 0.0;
                    for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                        int j = st.adj_col_idx[idx] - 1;
                        qs_v -= v_kron[j * T_st + t];
                    }
                    int n_neigh = st.adj_row_ptr[s + 1] - st.adj_row_ptr[s];
                    qs_v += n_neigh * v_kron[s * T_st + t];

                    if (st_use_nc) {
                        grad[layout.st_delta_start + s * T_st + t] =
                            grad_delta_lik[s * T_st + t] * inv_scale_nc - qs_v;
                    } else {
                        grad[layout.st_delta_start + s * T_st + t] =
                            grad_delta_lik[s * T_st + t] - tau_st * qs_v;
                    }
                    total_qf += stencil_input[s * T_st + t] * qs_v;
                }
            }

            int rank_space = S_st - 1;
            int rank_time = (st.temporal_type == TemporalType::RW1) ? (T_st - 1) :
                            (st.temporal_type == TemporalType::RW2) ? (T_st - 2) : T_st;
            if (st.temporal_cyclic) rank_time = T_st;
            int total_rank = rank_space * rank_time;

            if (st_use_nc) {
                double lik_tau_grad = 0.0;
                for (int k = 0; k < ST_n; k++) lik_tau_grad += grad_delta_lik[k] * st_delta[k];
                grad[layout.log_tau_st_idx] += 0.5 * (total_rank - ST_n) - 0.5 * lik_tau_grad;
            } else {
                grad[layout.log_tau_st_idx] += 0.5 * total_rank - 0.5 * tau_st * total_qf;
            }
        }

        // Sum-to-zero penalty on ST delta
        double lambda_stz = 0.001;
        for (int t = 0; t < T_st; t++) {
            double row_sum = 0.0;
            for (int s = 0; s < S_st; s++) row_sum += st_delta[s * T_st + t];
            for (int s = 0; s < S_st; s++)
                grad[layout.st_delta_start + s * T_st + t] -= lambda_stz * row_sum;
        }
        for (int s = 0; s < S_st; s++) {
            double col_sum = 0.0;
            for (int t = 0; t < T_st; t++) col_sum += st_delta[s * T_st + t];
            for (int t = 0; t < T_st; t++)
                grad[layout.st_delta_start + s * T_st + t] -= lambda_stz * col_sum;
        }
    }

    // NC RE chain rule (simple intercepts)
    if (!has_slopes) re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // NC slopes chain rule - mirrors compute_gradient_analytical write-back
    if (has_slopes) {
        for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
            int n_groups = data.re_n_groups_multi[t_re];
            int n_coefs = layout.re_n_coefs_multi[t_re];
            int re_start_t = layout.re_start_multi[t_re];
            bool is_corr_nc = !nc_L_flats.empty() && t_re < (int)nc_L_flats.size() && !nc_L_flats[t_re].empty();

            if (is_corr_nc) {
                // Correlated NC: chain rule from dLL/d(re_nc) back to
                // (z, log_sigma, raw_chol). grad_z and grad_raw slots are
                // contiguous in grad; log_sigma is scattered, so use temp.
                const auto& L_flat = nc_L_flats[t_re];
                const auto& sigmas = nc_sigmas_vec[t_re];
                int chol_start = layout.chol_re_start_multi[t_re];
                std::vector<double> g_log_sigma(n_coefs, 0.0);

                tulpa::chol_nc_chain_rule_add(
                    L_flat.data(), n_coefs, sigmas.data(),
                    &params[re_start_t], &params[chol_start],
                    &re_nc_flat_c[re_start_t], n_groups,
                    grad_re_slopes_lik[t_re].data(),
                    &grad[re_start_t],
                    g_log_sigma.data(),
                    &grad[chol_start]);

                for (int c = 0; c < n_coefs; c++) {
                    grad[layout.log_sigma_re_slopes[t_re][c]] += g_log_sigma[c];
                }
            } else if (slopes_nc) {
                // Uncorrelated NC: chain rule re = sigma * z
                for (int g = 0; g < n_groups; g++) {
                    for (int c = 0; c < n_coefs; c++) {
                        int idx = re_start_t + g * n_coefs + c;
                        double z_gc = params[idx];
                        double sigma_c = std::exp(params[layout.log_sigma_re_slopes[t_re][c]]);
                        double lik_grad = grad_re_slopes_lik[t_re][g * n_coefs + c];
                        grad[idx] += sigma_c * lik_grad;
                        grad[layout.log_sigma_re_slopes[t_re][c]] += z_gc * lik_grad * sigma_c;
                    }
                }
            } else {
                // Centered: direct
                for (int g = 0; g < n_groups; g++)
                    for (int c = 0; c < n_coefs; c++)
                        grad[re_start_t + g * n_coefs + c] += grad_re_slopes_lik[t_re][g * n_coefs + c];
            }
        }
    }

    // Fused log-posterior
    if (fuse_lp && !layout.has_zi) {
        *log_post_out = compute_log_post(params, data, layout);
    }
}
