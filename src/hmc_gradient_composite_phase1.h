// hmc_gradient_composite_phase1.h
// Function-body fragment of compute_gradient_composite: Phase 1 —
// extract all parameters into local handles. NOT standalone-compilable.
// Included exactly once by the umbrella; relies on locals declared in
// compute_gradient_composite (params, data, layout, grad, beta_num,
// beta_denom, sigma_re, re, phi_num, phi_denom, ...).

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

