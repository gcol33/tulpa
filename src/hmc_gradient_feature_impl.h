// =====================================================================
// SVC gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from svc_nngp_gradients for NNGP prior
// =====================================================================

void compute_gradient_svc_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- SVC-specific parameters ---
    int n_svc = data.svc_data.n_svc;
    int N_obs = data.svc_data.n_obs;

    double* svc_sigma2 = data.svc_data.sigma2_ws.data();
    double* svc_phi = data.svc_data.phi_ws.data();
    for (int j = 0; j < n_svc; j++) {
        svc_sigma2[j] = std::exp(params[layout.log_sigma2_svc_start + j]);
        svc_phi[j] = std::exp(params[layout.log_phi_svc_start + j]);
        if (svc_phi[j] < data.svc_phi_prior_lower || svc_phi[j] > data.svc_phi_prior_upper) {
            return;
        }
    }

    double* svc_w_flat = data.svc_data.w_flat_ws.data();
    for (int k = 0; k < N_obs * n_svc; k++) {
        svc_w_flat[k] = params[layout.svc_w_start + k];
    }

    // --- Shared base priors + SVC-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    for (int j = 0; j < n_svc; j++) {
        double sigma = std::sqrt(svc_sigma2[j]);
        double ratio = sigma / data.svc_sigma2_prior_scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma2_svc_start + j] = -ratio_sq / (1.0 + ratio_sq) + 1.0;
        grad[layout.log_phi_svc_start + j] = 1.0;
    }

    // NNGP gradients for SVC effects
    double* w_j_ptr = data.svc_data.w_j_ws.data();
    for (int j = 0; j < n_svc; j++) {
        for (int i = 0; i < N_obs; i++) {
            w_j_ptr[i] = svc_w_flat[j * N_obs + i];
        }
        std::vector<double> w_j_vec(w_j_ptr, w_j_ptr + N_obs);
        tulpa_svc::SVCGradients svc_grads;
        tulpa_svc::svc_nngp_gradients(w_j_vec, svc_sigma2[j], svc_phi[j], data.svc_data, svc_grads);

        for (int i = 0; i < N_obs; i++) {
            grad[layout.svc_w_start + j * N_obs + i] += svc_grads.grad_w[i];
        }
        grad[layout.log_sigma2_svc_start + j] += svc_grads.grad_log_sigma2;
        grad[layout.log_phi_svc_start + j] += svc_grads.grad_log_phi;

        double sum_w = 0.0;
        for (int i = 0; i < N_obs; i++) sum_w += w_j_ptr[i];
        double stz_grad = -sum_w / N_obs;
        for (int i = 0; i < N_obs; i++) {
            grad[layout.svc_w_start + j * N_obs + i] += stz_grad;
        }
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // SVC-specific eta contribution
    for (int i = 0; i < pre.N; i++) {
        double svc_effect = 0.0;
        for (int j = 0; j < n_svc; j++) {
            svc_effect += data.svc_data.X_svc[i * n_svc + j] * svc_w_flat[j * N_obs + i];
        }
        vec_grad_ws.eta_num[i] += svc_effect;
        if (!pre.is_binomial && data.svc_data.shared) vec_grad_ws.eta_denom[i] += svc_effect;
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // SVC-specific residual scatter
    for (int i = 0; i < pre.N; i++) {
        double dLL_dsvc = data.svc_data.shared
            ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
            : vec_grad_ws.resid_num[i];
        for (int j = 0; j < n_svc; j++) {
            grad[layout.svc_w_start + j * N_obs + i] += dLL_dsvc * data.svc_data.X_svc[i * n_svc + j];
        }
    }

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// HSGP-SVC gradient (hand-coded)
// Uses HSGP basis function approximation for spatially-varying coefficients.
// Each SVC term k has its own GP with sigma2_k, lengthscale_k, beta_k[m^2].
// All terms share the same basis matrix Phi (same coordinates/eigenvalues).
// =====================================================================

void compute_gradient_svc_hsgp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Thread-local HSGP workspace (one per SVC term, reused)
    static thread_local tulpa_hsgp::HSGPWorkspace hsgp_ws;
    hsgp_ws.init(data.N, data.svc_hsgp_data.m_total);

    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- SVC HSGP-specific parameters ---
    const int n_svc = data.svc_data.n_svc;
    const int m_total = data.svc_hsgp_data.m_total;

    // Evaluate f_k(s_i) for each SVC term and accumulate SVC contribution to eta
    std::vector<double> svc_eta(pre.N, 0.0);
    std::vector<double> svc_f_all(n_svc * pre.N);

    for (int j = 0; j < n_svc; j++) {
        double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
        double lengthscale_j = std::exp(params[layout.log_phi_svc_start + j]);
        const double* beta_j = &params[layout.svc_w_start + j * m_total];

        tulpa_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                       data.svc_hsgp_data, hsgp_ws);

        std::memcpy(&svc_f_all[j * pre.N], hsgp_ws.hsgp_f.data(), pre.N * sizeof(double));
        for (int i = 0; i < pre.N; i++) {
            svc_eta[i] += data.svc_data.X_svc[i * n_svc + j] * hsgp_ws.hsgp_f[i];
        }
    }

    // --- Temporal parameters (GMRF: RW1/AR1) ---
    double tau_temporal = 0.0;
    int T_len_svc = 0;
    const double* phi_temporal_svc = nullptr;
    double rho_ar1_svc = 0.5;
    if (layout.has_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_len_svc = layout.temporal_end - layout.temporal_start;
        phi_temporal_svc = &params[layout.temporal_start];
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
            rho_ar1_svc = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx]));
        }
    }
    std::vector<double> grad_temporal_lik_svc(T_len_svc, 0.0);

    // --- Shared base priors + feature-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // Temporal prior
    if (layout.has_temporal) {
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
            grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1_svc;
    }

    // Per-term HSGP hyperparameter priors
    for (int j = 0; j < n_svc; j++) {
        double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
        double log_ls_j = params[layout.log_phi_svc_start + j];

        double sigma_j = std::sqrt(sigma2_j);
        double rate_sigma = 4.6;
        grad[layout.log_sigma2_svc_start + j] = -0.5 * rate_sigma * sigma_j + 0.5;

        grad[layout.log_phi_svc_start + j] = -log_ls_j;

        for (int k = 0; k < m_total; k++) {
            grad[layout.svc_w_start + j * m_total + k] = -params[layout.svc_w_start + j * m_total + k];
        }
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // SVC HSGP-specific eta contribution
    for (int i = 0; i < pre.N; i++) {
        vec_grad_ws.eta_num[i] += svc_eta[i];
        if (!pre.is_binomial && data.svc_data.shared) vec_grad_ws.eta_denom[i] += svc_eta[i];
    }

    // Temporal eta contribution
    if (layout.has_temporal && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < pre.N; i++) {
            if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t = data.temporal_time_idx[i] - 1;
                int g = data.temporal_group_idx[i] - 1;
                int t_idx = g * data.n_times + t;
                if (t_idx >= 0 && t_idx < T_len_svc) {
                    vec_grad_ws.eta_num[i] += phi_temporal_svc[t_idx];
                    if (!pre.is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += phi_temporal_svc[t_idx];
                }
            }
        }
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // --- HSGP gradient backprop per SVC term ---
    for (int j = 0; j < n_svc; j++) {
        double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
        double lengthscale_j = std::exp(params[layout.log_phi_svc_start + j]);
        const double* beta_j = &params[layout.svc_w_start + j * m_total];

        double* grad_f_ptr = hsgp_ws.grad_f.data();
        for (int i = 0; i < pre.N; i++) {
            double dLL = data.svc_data.shared
                ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
                : vec_grad_ws.resid_num[i];
            grad_f_ptr[i] = dLL * data.svc_data.X_svc[i * n_svc + j];
        }

        tulpa_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                       data.svc_hsgp_data, hsgp_ws);

        double grad_log_sigma2_j, grad_log_lengthscale_j;
        tulpa_hsgp::hsgp_compute_gradients_ws(beta_j, sigma2_j, lengthscale_j,
                                                data.svc_hsgp_data, hsgp_ws,
                                                grad_log_sigma2_j, grad_log_lengthscale_j);

        for (int k = 0; k < m_total; k++) {
            grad[layout.svc_w_start + j * m_total + k] += hsgp_ws.grad_beta_out[k];
        }
        grad[layout.log_sigma2_svc_start + j] += grad_log_sigma2_j;
        grad[layout.log_phi_svc_start + j] += grad_log_lengthscale_j;
    }

    // Temporal likelihood gradient scatter
    if (layout.has_temporal && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < pre.N; i++) {
            if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t = data.temporal_time_idx[i] - 1;
                int g = data.temporal_group_idx[i] - 1;
                int t_idx = g * data.n_times + t;
                if (t_idx >= 0 && t_idx < T_len_svc) {
                    double lik_grad = data.temporal_shared ?
                        (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i]) :
                        vec_grad_ws.resid_num[i];
                    grad_temporal_lik_svc[t_idx] += lik_grad;
                }
            }
        }
    }

    // Temporal GMRF prior gradients
    if (layout.has_temporal && T_len_svc > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1_svc,
                                 phi_temporal_svc, T_len_svc, grad_temporal_lik_svc.data(), grad.data());
    }

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// TVC gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from hmc_tvc_grad.h for RW1/RW2/AR1 priors
// =====================================================================

void compute_gradient_tvc_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- TVC-specific parameters ---
    int n_tvc = data.tvc_data.n_tvc;
    int n_times = data.tvc_data.n_times;
    int n_groups = data.tvc_data.n_groups;
    int n_w = n_groups * n_tvc * n_times;

    double* tvc_tau = data.tvc_data.tau_ws.data();
    double* tvc_rho = data.tvc_data.rho_ws.data();
    for (int j = 0; j < n_tvc; j++) {
        tvc_tau[j] = std::exp(params[layout.log_tau_tvc_start + j]);
        if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
            double logit_rho = params[layout.logit_rho_tvc_start + j];
            double u = 1.0 / (1.0 + std::exp(-logit_rho));
            tvc_rho[j] = 2.0 * u - 1.0;
        } else {
            tvc_rho[j] = 0.0;
        }
    }

    double* tvc_w_flat = data.tvc_data.w_flat_ws.data();
    for (int k = 0; k < n_w; k++) {
        tvc_w_flat[k] = params[layout.tvc_w_start + k];
    }

    // --- Shared base priors + TVC-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    double tvc_pc_rate = -std::log(0.01) / 1.0;
    for (int j = 0; j < n_tvc; j++) {
        double sigma_j = 1.0 / std::sqrt(tvc_tau[j]);
        grad[layout.log_tau_tvc_start + j] = 0.5 * tvc_pc_rate * sigma_j + 1.5;
    }
    if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
        for (int j = 0; j < n_tvc; j++) {
            double u = (tvc_rho[j] + 1.0) / 2.0;
            grad[layout.logit_rho_tvc_start + j] = 1.0 - 2.0 * u;
        }
    }

    // TVC prior gradients
    tulpa_tvc::TVCGradientWS tvc_ws;
    tvc_ws.grad_w = data.tvc_data.grad_w_ws.data();
    tvc_ws.grad_log_tau = data.tvc_data.grad_log_tau_ws.data();
    tvc_ws.grad_logit_rho = data.tvc_data.grad_logit_rho_ws.data();
    tvc_ws.grad_w_jg = data.tvc_data.grad_w_jg_ws.data();
    tvc_ws.d_buf = data.tvc_data.d_ws.data();
    tvc_ws.n_w = n_w;
    tvc_ws.n_tvc = n_tvc;
    tulpa_tvc::tvc_prior_gradients_ws(tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho, tvc_ws);

    for (int k = 0; k < n_w; k++) {
        grad[layout.tvc_w_start + k] += tvc_ws.grad_w[k];
    }
    for (int j = 0; j < n_tvc; j++) {
        grad[layout.log_tau_tvc_start + j] += tvc_ws.grad_log_tau[j];
    }
    if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.logit_rho_tvc_start + j] += tvc_ws.grad_logit_rho[j];
        }
    }

    // Precompute TVC eta contribution
    double* tvc_eta = data.tvc_data.eta_ws.data();
    std::fill(tvc_eta, tvc_eta + data.N, 0.0);
    for (int i = 0; i < data.N; i++) {
        int t = data.tvc_data.time_index[i] - 1;
        int g = data.tvc_data.group_index[i] - 1;
        for (int j = 0; j < n_tvc; j++) {
            tvc_eta[i] += data.tvc_data.X_tvc[i * n_tvc + j] * tvc_w_flat[(g * n_tvc + j) * n_times + t];
        }
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // TVC-specific eta contribution
    for (int i = 0; i < pre.N; i++) {
        vec_grad_ws.eta_num[i] += tvc_eta[i];
        if (!pre.is_binomial && data.tvc_data.shared) vec_grad_ws.eta_denom[i] += tvc_eta[i];
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // TVC-specific residual scatter
    for (int i = 0; i < pre.N; i++) {
        double dLL_dtvc = data.tvc_data.shared
            ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
            : vec_grad_ws.resid_num[i];
        int t = data.tvc_data.time_index[i] - 1;
        int g = data.tvc_data.group_index[i] - 1;
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.tvc_w_start + (g * n_tvc + j) * n_times + t] +=
                dLL_dtvc * data.tvc_data.X_tvc[i * n_tvc + j];
        }
    }

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// Latent factor gradient (hand-coded, O(N*K))
// Uses analytical gradients for latent factor models
// =====================================================================

void compute_gradient_latent_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- Latent factor parameters ---
    int K = data.latent_n_factors;

    // Extract log_sigma for latent factors
    std::vector<double> log_sigma_latent(K);
    std::vector<double> sigma_latent(K);
    for (int k = 0; k < K; k++) {
        log_sigma_latent[k] = params[layout.log_sigma_latent_start + k];
        sigma_latent[k] = std::exp(log_sigma_latent[k]);
    }

    // Extract factors (unconstrained)
    int n_factor_params = pre.N * K;
    std::vector<double> factors_raw(n_factor_params);
    for (int j = 0; j < n_factor_params; j++) {
        factors_raw[j] = params[layout.latent_factor_start + j];
    }

    // Apply constraint to get constrained factors
    std::vector<double> factors_constrained = factors_raw;
    if (data.latent_constraint == 0) {  // SUM_TO_ZERO
        for (int k = 0; k < K; k++) {
            double sum = 0.0;
            for (int i = 0; i < pre.N; i++) {
                sum += factors_constrained[i * K + k];
            }
            double mean = sum / pre.N;
            for (int i = 0; i < pre.N; i++) {
                factors_constrained[i * K + k] -= mean;
            }
        }
    } else {  // FIRST_ZERO
        for (int k = 0; k < K; k++) {
            double first_val = factors_constrained[k];  // factors[0, k]
            for (int i = 0; i < pre.N; i++) {
                factors_constrained[i * K + k] -= first_val;
            }
        }
    }

    // Precompute latent contribution to eta
    std::vector<double> latent_eta(pre.N, 0.0);
    for (int i = 0; i < pre.N; i++) {
        for (int k = 0; k < K; k++) {
            latent_eta[i] += factors_constrained[i * K + k] * sigma_latent[k];
        }
    }

    // --- Shared base priors + latent-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // Latent sigma prior: Exponential(rate) on sigma, with Jacobian for log transform
    double latent_rate = data.latent_sigma_prior_rate;
    for (int k = 0; k < K; k++) {
        grad[layout.log_sigma_latent_start + k] = 1.0 - latent_rate * sigma_latent[k];
    }

    // =========================================================================
    // Likelihood loop - compute dLL/deta and chain-rule to all parameters
    // (Latent uses scalar per-obs loop - latent factor chain rule requires it)
    // =========================================================================
    std::vector<double> grad_factors_constrained(n_factor_params, 0.0);

    for (int i = 0; i < pre.N; i++) {
        // Linear predictors
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.legacy.p_num; j++) {
            eta_num += data.legacy.X_num_flat[i * data.legacy.p_num + j] * pre.cp.beta_num[j];
        }
        for (int j = 0; j < data.legacy.p_denom; j++) {
            eta_denom += data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * pre.cp.beta_denom[j];
        }

        // Random effects (handles NC parameterization)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            double re_eff = re_value_for_eta(pre.cp.re, g, pre.cp.sigma_re, data.re_parameterization);
            eta_num += re_eff;
            eta_denom += re_eff;
        }

        // Latent effect
        double latent_effect = latent_eta[i];
        if (data.latent_shared) {
            eta_num += latent_effect;
            eta_denom += latent_effect;
        } else {
            eta_num += latent_effect;
        }

        if (pre.fuse_lp) pre.obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, pre.cp.phi_num, pre.cp.phi_denom);

        double dLL_deta_num = 0.0, dLL_deta_denom = 0.0;
        compute_obs_residuals(data, i, eta_num, eta_denom, pre.cp.phi_num, pre.cp.phi_denom, dLL_deta_num, dLL_deta_denom);

        // Total gradient through latent effect
        double dLL_dlatent = data.latent_shared ?
                             (dLL_deta_num + dLL_deta_denom) : dLL_deta_num;

        scatter_beta_gradients(data, layout, i, dLL_deta_num, dLL_deta_denom, grad.data());
        scatter_re_gradient(data, layout, i, dLL_deta_num, dLL_deta_denom, grad.data());

        // Gradients for latent factors (on constrained space)
        for (int k = 0; k < K; k++) {
            grad_factors_constrained[i * K + k] = dLL_dlatent * sigma_latent[k];
            grad[layout.log_sigma_latent_start + k] += dLL_dlatent * factors_constrained[i * K + k] * sigma_latent[k];
        }

        accumulate_phi_likelihood_grad(data, layout, i, eta_num, eta_denom, pre.cp.phi_num, pre.cp.phi_denom, grad.data());
    }

    // =========================================================================
    // Add prior gradient to grad_factors_constrained
    // =========================================================================
    int prior_start = (data.latent_constraint == 0) ? 0 : 1;
    for (int k = 0; k < K; k++) {
        for (int i = prior_start; i < pre.N; i++) {
            grad_factors_constrained[i * K + k] += -factors_constrained[i * K + k];
        }
    }

    // =========================================================================
    // Apply constraint chain-rule to get gradients on raw (unconstrained) factors
    // =========================================================================
    if (data.latent_constraint == 0) {  // SUM_TO_ZERO
        for (int k = 0; k < K; k++) {
            double sum_grad = 0.0;
            for (int i = 0; i < pre.N; i++) {
                sum_grad += grad_factors_constrained[i * K + k];
            }
            double mean_grad = sum_grad / pre.N;

            for (int i = 0; i < pre.N; i++) {
                grad[layout.latent_factor_start + i * K + k] +=
                    grad_factors_constrained[i * K + k] - mean_grad;
            }
        }
    } else {  // FIRST_ZERO
        for (int k = 0; k < K; k++) {
            double sum_grad = 0.0;
            for (int i = 1; i < pre.N; i++) {
                sum_grad += grad_factors_constrained[i * K + k];
                grad[layout.latent_factor_start + i * K + k] += grad_factors_constrained[i * K + k];
            }
            grad[layout.latent_factor_start + k] += -sum_grad;  // factor_raw[0,k]
        }
    }

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}
