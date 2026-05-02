// =====================================================================
// GP + Temporal gradient (hand-coded)
// Combines GP spatial with temporal RW1/RW2/AR1
// =====================================================================

void compute_gradient_gp_plus_temporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- GP-specific parameters ---
    int N_gp = data.gp_data.n_obs;
    double sigma2_gp = std::exp(params[layout.log_sigma2_gp_idx]);
    double phi_gp = std::exp(params[layout.log_phi_gp_idx]);
    std::vector<double> gp_w(N_gp);
    for (int i = 0; i < N_gp; i++) gp_w[i] = params[layout.gp_w_start + i];

    // --- Temporal parameters ---
    double tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
    int T_len = layout.temporal_end - layout.temporal_start;
    const double* phi_temporal = &params[layout.temporal_start];
    double rho_ar1 = (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        ? 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx])) : 0.5;

    // --- Shared base priors + feature-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    grad[layout.log_sigma2_gp_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
    grad[layout.log_phi_gp_idx] = 1.0;
    tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;

    // NNGP prior gradients
    tulpa_gp::NNGPGradients nngp_grads;
    tulpa_gp::gp_nngp_gradients(gp_w, sigma2_gp, phi_gp, data.gp_data, nngp_grads);
    for (int i = 0; i < N_gp; i++) grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
    grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
    grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // GP-specific + temporal eta contribution
    std::vector<double> grad_temporal_lik(T_len, 0.0);
    std::vector<int> obs_t_idx(pre.N, -1);
    for (int i = 0; i < pre.N; i++) {
        int loc_i = data.gp_data.obs_to_loc[i];
        double gp_effect = gp_w[loc_i];
        vec_grad_ws.eta_num[i] += gp_effect;
        if (!pre.is_binomial && data.gp_data.shared) vec_grad_ws.eta_denom[i] += gp_effect;

        if (!data.temporal_time_idx.empty() && i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_idx = g * data.n_times + t;
            if (t_idx >= 0 && t_idx < T_len) {
                obs_t_idx[i] = t_idx;
                vec_grad_ws.eta_num[i] += phi_temporal[t_idx];
                if (!pre.is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += phi_temporal[t_idx];
            }
        }
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // GP + temporal residual scatter
    for (int i = 0; i < pre.N; i++) {
        int loc_i = data.gp_data.obs_to_loc[i];
        grad[layout.gp_w_start + loc_i] += data.gp_data.shared
            ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
            : vec_grad_ws.resid_num[i];
        if (obs_t_idx[i] >= 0) grad_temporal_lik[obs_t_idx[i]] += data.temporal_shared
            ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
            : vec_grad_ws.resid_num[i];
    }

    // Temporal GMRF gradients
    temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                             phi_temporal, T_len, grad_temporal_lik.data(), grad.data());

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// Temporal GP (standalone) hand-coded gradients
// Temporal GP with exponential covariance uses state-space AR(1) form
// =====================================================================

void compute_gradient_temporal_gp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);
    double tgp_lp_accum = 0.0;  // Accumulates temporal GP prior terms

    // --- Temporal GP hyperparameters ---
    double sigma2_tgp = std::exp(params[layout.log_sigma2_temporal_gp_idx]);
    double logit_phi_val = params[layout.logit_phi_temporal_gp_idx];

    // Logit-bounded phi: phi = lower + range * sigmoid(logit_phi)
    double phi_lower = data.temporal_gp_phi_prior_lower;
    double phi_upper = data.temporal_gp_phi_prior_upper;
    double phi_range = phi_upper - phi_lower;
    double sigmoid_val = 1.0 / (1.0 + std::exp(-logit_phi_val));
    double phi_tgp = phi_lower + phi_range * sigmoid_val;

    // Conversion factor: grad_logit = grad_log * chi
    double chi_tgp = (phi_tgp - phi_lower) * (phi_upper - phi_tgp) / (phi_tgp * phi_range);

    // Temporal effects: n_temporal_groups * n_times parameters
    int T_times = data.n_times;
    int n_groups = data.n_temporal_groups;
    const double* phi_temporal = &params[layout.temporal_start];
    int T_len = layout.temporal_end - layout.temporal_start;

    // Non-centered parameterization: params store z ~ N(0,1), reconstruct f
    const bool use_nc = (data.temporal_gp_parameterization == 1);
    static thread_local tulpa_temporal_gp::TemporalGPNCWorkspace nc_ws;
    const double* f_temporal = phi_temporal;  // Default: centered, f stored directly

    if (use_nc) {
        nc_ws.init(T_times, n_groups);
        tulpa_temporal_gp::temporal_gp_nc_forward(
            phi_temporal, T_times, n_groups,
            sigma2_tgp, phi_tgp,
            data.temporal_gp_data.time_values, nc_ws);
        f_temporal = nc_ws.f.data();  // Use reconstructed f for eta
        std::memset(nc_ws.dL_df.data(), 0, T_len * sizeof(double));
    }

    // --- Shared base priors + temporal GP-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // sigma2: PC prior
    double sigma_tgp = std::sqrt(sigma2_tgp);
    double rate_tgp = -std::log(data.temporal_gp_sigma2_prior_alpha) / data.temporal_gp_sigma2_prior_U;
    grad[layout.log_sigma2_temporal_gp_idx] = -0.5 * rate_tgp * sigma_tgp + 0.5;

    // phi: Logit-bounded Jacobian gradient
    grad[layout.logit_phi_temporal_gp_idx] = (phi_upper + phi_lower - 2.0 * phi_tgp) / phi_range;

    if (use_nc) {
        // NC prior: z ~ N(0, I), Jacobian gradients
        grad[layout.log_sigma2_temporal_gp_idx] += 0.5 * T_times * n_groups;

        double jac_phi_log = 0.0;
        for (int t = 1; t < T_times; t++) {
            double rho_t = nc_ws.rho[t - 1];
            double rho2 = rho_t * rho_t;
            double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
            double dt_over_phi = dt / phi_tgp;
            double one_minus_rho2 = 1.0 - rho2;
            if (one_minus_rho2 < 1e-10) one_minus_rho2 = 1e-10;
            jac_phi_log -= rho2 * dt_over_phi / one_minus_rho2;
        }
        grad[layout.logit_phi_temporal_gp_idx] += jac_phi_log * n_groups * chi_tgp;

        for (int t = 0; t < T_len; t++) {
            grad[layout.temporal_start + t] = -phi_temporal[t];
        }

        // Fuse temporal GP prior log-prob
        if (pre.fuse_lp) {
            double log_sigma2 = params[layout.log_sigma2_temporal_gp_idx];
            tgp_lp_accum += std::log(rate_tgp) - rate_tgp * sigma_tgp
                          - std::log(2.0 * sigma_tgp) + log_sigma2;
            tgp_lp_accum += std::log(phi_tgp - phi_lower)
                          + std::log(phi_upper - phi_tgp)
                          - std::log(phi_range);
            double nc_jac = T_times * std::log(sigma_tgp);
            for (int t = 1; t < T_times; t++) {
                double one_m_rho2 = 1.0 - nc_ws.rho[t-1] * nc_ws.rho[t-1];
                if (one_m_rho2 < 1e-10) one_m_rho2 = 1e-10;
                nc_jac += 0.5 * std::log(one_m_rho2);
            }
            tgp_lp_accum += nc_jac * n_groups;
            for (int t = 0; t < T_len; t++) {
                tgp_lp_accum += -0.5 * phi_temporal[t] * phi_temporal[t];
            }
        }
    } else {
        // Centered: temporal GP prior gradients (state-space exponential form)
        for (int g = 0; g < n_groups; g++) {
            int offset = g * T_times;

            double f0 = phi_temporal[offset];
            grad[layout.temporal_start + offset] += -f0 / sigma2_tgp;

            double grad_log_sigma2_prior = -0.5 + 0.5 * f0 * f0 / sigma2_tgp;
            double grad_log_phi_prior = 0.0;

            for (int t = 1; t < T_times; t++) {
                double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
                double rho = std::exp(-dt / phi_tgp);
                double rho2 = rho * rho;
                double cv = sigma2_tgp * (1.0 - rho2);
                if (cv < 1e-10) cv = 1e-10;

                double f_prev = phi_temporal[offset + t - 1];
                double f_curr = phi_temporal[offset + t];
                double r = f_curr - rho * f_prev;

                grad[layout.temporal_start + offset + t] += -r / cv;
                grad[layout.temporal_start + offset + t - 1] += rho * r / cv;

                grad_log_sigma2_prior += -0.5 + 0.5 * r * r / cv;

                double dt_over_phi = dt / phi_tgp;
                grad_log_phi_prior += dt_over_phi * (
                    sigma2_tgp * rho2 / cv
                    + rho * r * f_prev / cv
                    + sigma2_tgp * rho2 * r * r / (cv * cv)
                );
            }

            grad[layout.log_sigma2_temporal_gp_idx] += grad_log_sigma2_prior;
            grad[layout.logit_phi_temporal_gp_idx] += grad_log_phi_prior * chi_tgp;
        }
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // Temporal GP-specific eta contribution
    static thread_local std::vector<double> grad_temporal_lik;
    grad_temporal_lik.assign(T_len, 0.0);

    for (int i = 0; i < pre.N; i++) {
        if (!data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = (i < (int)data.temporal_group_idx.size() && data.temporal_group_idx[i] > 0)
                    ? data.temporal_group_idx[i] - 1 : 0;
            int flat_idx = g * T_times + t;
            if (flat_idx >= 0 && flat_idx < T_len) {
                vec_grad_ws.eta_num[i] += f_temporal[flat_idx];
                if (!pre.is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += f_temporal[flat_idx];
            }
        }
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // Temporal GP-specific residual scatter
    for (int i = 0; i < pre.N; i++) {
        if (!data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = (i < (int)data.temporal_group_idx.size() && data.temporal_group_idx[i] > 0)
                    ? data.temporal_group_idx[i] - 1 : 0;
            int flat_idx = g * T_times + t;
            if (flat_idx >= 0 && flat_idx < T_len)
                grad_temporal_lik[flat_idx] += data.temporal_shared
                    ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
                    : vec_grad_ws.resid_num[i];
        }
    }

    if (use_nc) {
        // NC: backward pass converts dL/df -> dL/dz and accumulates sigma2/phi grads
        std::memcpy(nc_ws.dL_df.data(), grad_temporal_lik.data(), T_len * sizeof(double));

        double grad_log_sigma2_lik = 0.0, grad_log_phi_lik_tgp = 0.0;
        tulpa_temporal_gp::temporal_gp_nc_backward(
            phi_temporal, T_times, n_groups,
            sigma2_tgp, phi_tgp,
            data.temporal_gp_data.time_values,
            nc_ws, &grad[layout.temporal_start],
            grad_log_sigma2_lik, grad_log_phi_lik_tgp);
        grad[layout.log_sigma2_temporal_gp_idx] += grad_log_sigma2_lik;
        grad[layout.logit_phi_temporal_gp_idx] += grad_log_phi_lik_tgp * chi_tgp;
    } else {
        // Centered: add likelihood contribution to temporal effects directly
        for (int t = 0; t < T_len; t++) {
            grad[layout.temporal_start + t] += grad_temporal_lik[t];
        }
    }

    // --- Custom epilogue (temporal GP has fused prior accumulation) ---
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), pre.cp.sigma_re);

    if (pre.fuse_lp) {
        if (use_nc && tgp_lp_accum != 0.0) {
            *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true,
                                             nullptr, &tgp_lp_accum) + pre.obs_log_lik;
        } else {
            *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + pre.obs_log_lik;
        }
    }
}

// =====================================================================
// Multi-scale GP + Temporal hand-coded gradients
// Combines MSGP spatial gradients with temporal GMRF gradients
// =====================================================================

void compute_gradient_msgp_plus_temporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- Multi-scale GP parameters ---
    int N_gp = data.multiscale_gp_data.n_obs;
    double sigma2_local = std::exp(params[layout.log_sigma2_gp_local_idx]);
    double phi_local = std::exp(params[layout.log_phi_gp_local_idx]);
    double sigma2_regional = std::exp(params[layout.log_sigma2_gp_regional_idx]);
    double phi_regional = std::exp(params[layout.log_phi_gp_regional_idx]);

    std::vector<double> w_local(N_gp), w_regional(N_gp);
    for (int i = 0; i < N_gp; i++) {
        w_local[i] = params[layout.gp_local_start + i];
        w_regional[i] = params[layout.gp_regional_start + i];
    }

    // --- Temporal parameters ---
    double tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
    int T_len = layout.temporal_end - layout.temporal_start;
    const double* phi_temporal = &params[layout.temporal_start];
    double rho_ar1 = (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        ? 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx])) : 0.5;

    // Bounds check for phi
    if (phi_local < data.multiscale_gp_data.range_local_lower ||
        phi_local > data.multiscale_gp_data.range_local_upper ||
        phi_regional < data.multiscale_gp_data.range_regional_lower ||
        phi_regional > data.multiscale_gp_data.range_regional_upper) {
        return;
    }

    // --- Shared base priors + feature-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // PC priors on MSGP variances
    grad[layout.log_sigma2_gp_local_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_local, data.ms_sigma2_local_prior_U, data.ms_sigma2_local_prior_alpha);
    grad[layout.log_sigma2_gp_regional_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_regional, data.ms_sigma2_regional_prior_U, data.ms_sigma2_regional_prior_alpha);
    grad[layout.log_phi_gp_local_idx] = 1.0;
    grad[layout.log_phi_gp_regional_idx] = 1.0;

    // Temporal prior
    tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;

    // NNGP prior gradients for multi-scale GP
    auto [gp_local, gp_regional] = make_msgp_gp_views(data.multiscale_gp_data);

    tulpa_gp::NNGPGradients nngp_grads_local, nngp_grads_regional;
    tulpa_gp::gp_nngp_gradients(w_local, sigma2_local, phi_local, gp_local, nngp_grads_local);
    tulpa_gp::gp_nngp_gradients(w_regional, sigma2_regional, phi_regional, gp_regional, nngp_grads_regional);

    for (int i = 0; i < N_gp; i++) {
        grad[layout.gp_local_start + i] += nngp_grads_local.grad_w[i];
        grad[layout.gp_regional_start + i] += nngp_grads_regional.grad_w[i];
    }
    grad[layout.log_sigma2_gp_local_idx] += nngp_grads_local.grad_log_sigma2;
    grad[layout.log_phi_gp_local_idx] += nngp_grads_local.grad_log_phi;
    grad[layout.log_sigma2_gp_regional_idx] += nngp_grads_regional.grad_log_sigma2;
    grad[layout.log_phi_gp_regional_idx] += nngp_grads_regional.grad_log_phi;

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // MSGP + temporal eta contribution
    std::vector<double> grad_temporal_lik(T_len, 0.0);
    std::vector<int> obs_t_idx(pre.N, -1);
    for (int i = 0; i < pre.N; i++) {
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        double ms_spatial = w_local[loc_i] + w_regional[loc_i];
        vec_grad_ws.eta_num[i] += ms_spatial;
        if (!pre.is_binomial && data.multiscale_gp_data.shared) vec_grad_ws.eta_denom[i] += ms_spatial;

        if (!data.temporal_time_idx.empty() && i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_idx = g * data.n_times + t;
            if (t_idx >= 0 && t_idx < T_len) {
                obs_t_idx[i] = t_idx;
                vec_grad_ws.eta_num[i] += phi_temporal[t_idx];
                if (!pre.is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += phi_temporal[t_idx];
            }
        }
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // MSGP + temporal residual scatter
    for (int i = 0; i < pre.N; i++) {
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        double dLL_dspatial = data.multiscale_gp_data.shared
            ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
            : vec_grad_ws.resid_num[i];
        grad[layout.gp_local_start + loc_i] += dLL_dspatial;
        grad[layout.gp_regional_start + loc_i] += dLL_dspatial;
        if (obs_t_idx[i] >= 0) grad_temporal_lik[obs_t_idx[i]] += data.temporal_shared
            ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
            : vec_grad_ws.resid_num[i];
    }

    // Temporal GMRF gradients
    temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                             phi_temporal, T_len, grad_temporal_lik.data(), grad.data());

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}
