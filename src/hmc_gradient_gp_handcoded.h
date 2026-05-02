// =====================================================================
// GP gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from gp_nngp_gradients for NNGP prior
// =====================================================================

void compute_gradient_gp_handcoded(
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

    if (phi_gp < data.gp_phi_prior_lower || phi_gp > data.gp_phi_prior_upper) {
        return;
    }

    const bool use_nc = (data.gp_parameterization == 1);
    static thread_local tulpa_gp::NNGPNCWorkspace nc_ws;

    // Get spatial effects: either w directly (centered) or reconstruct from z (NC)
    std::vector<double> gp_w(N_gp);
    if (use_nc) {
        const double* z_params = &params[layout.gp_w_start];
        tulpa_gp::nngp_nc_forward(z_params, sigma2_gp, phi_gp, data.gp_data, nc_ws);
        std::memcpy(gp_w.data(), nc_ws.w.data(), N_gp * sizeof(double));
    } else {
        for (int i = 0; i < N_gp; i++) {
            gp_w[i] = params[layout.gp_w_start + i];
        }
    }

    // --- Shared base priors + GP-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // PC prior on GP variance
    grad[layout.log_sigma2_gp_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
    // Uniform prior on phi - just Jacobian for log transform
    grad[layout.log_phi_gp_idx] = 1.0;

    if (!use_nc) {
        tulpa_gp::NNGPGradients nngp_grads;
        tulpa_gp::gp_nngp_gradients(gp_w, sigma2_gp, phi_gp, data.gp_data, nngp_grads);
        for (int i = 0; i < N_gp; i++) {
            grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
        }
        grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
        grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // GP-specific eta contribution
    for (int i = 0; i < pre.N; i++) {
        int loc_i = data.gp_data.obs_to_loc[i];
        double gp_effect = gp_w[loc_i];
        vec_grad_ws.eta_num[i] += gp_effect;
        if (!pre.is_binomial && data.gp_data.shared) vec_grad_ws.eta_denom[i] += gp_effect;
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // --- GP-specific residual scatter ---
    if (use_nc) {
        std::vector<double> dL_dw(N_gp, 0.0);
        for (int i = 0; i < pre.N; i++) {
            int loc_i = data.gp_data.obs_to_loc[i];
            double dLL = data.gp_data.shared
                ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
                : vec_grad_ws.resid_num[i];
            dL_dw[loc_i] += dLL;
        }

        std::vector<double> grad_z(N_gp, 0.0);
        double grad_log_sigma2_lik = 0.0, grad_log_phi_lik = 0.0, grad_log_phi_jac = 0.0;
        const double* z_params = &params[layout.gp_w_start];
        tulpa_gp::nngp_nc_backward(
            z_params, sigma2_gp, phi_gp, data.gp_data, nc_ws,
            dL_dw.data(), grad_z.data(),
            grad_log_sigma2_lik, grad_log_phi_lik, grad_log_phi_jac);

        for (int i = 0; i < N_gp; i++) {
            grad[layout.gp_w_start + i] += grad_z[i] - z_params[i];
        }
        grad[layout.log_sigma2_gp_idx] += grad_log_sigma2_lik;
        grad[layout.log_phi_gp_idx] += grad_log_phi_lik;
    } else {
        for (int i = 0; i < pre.N; i++) {
            int loc_i = data.gp_data.obs_to_loc[i];
            double dLL_dspatial = data.gp_data.shared
                ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
                : vec_grad_ws.resid_num[i];
            grad[layout.gp_w_start + loc_i] += dLL_dspatial;
        }
    }

    // --- Shared epilogue ---
    if (use_nc && pre.fuse_lp) {
        // NC: use full compute_log_post to ensure perfect consistency
        re_gradient_nc_transform(data, layout, params.data(), grad.data(), pre.cp.sigma_re);
        *log_post_out = compute_log_post(params, data, layout);
    } else {
        gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
    }
}
