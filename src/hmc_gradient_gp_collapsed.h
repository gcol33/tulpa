// =====================================================================
// Collapsed GP gradient (hand-coded)
// GP effects marginalized via inner Laplace - only hyperparams in HMC
// =====================================================================

// collapsed_gp_ws declared earlier (shared with compute_log_post)

void compute_gradient_gp_collapsed(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // GP hyperparameters
    double sigma2_gp = std::exp(params[layout.log_sigma2_gp_idx]);
    double phi_gp = std::exp(params[layout.log_phi_gp_idx]);

    if (phi_gp < data.gp_phi_prior_lower || phi_gp > data.gp_phi_prior_upper) {
        if (log_post_out) *log_post_out = -INFINITY;
        return;
    }

    int N_gp = data.gp_data.n_obs;
    int N = data.N;
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    // ---- Inner Laplace: find w* ----
    double collapsed_lp = collapsed_gp_find_mode(
        beta_num, beta_denom, sigma2_gp, phi_gp,
        phi_num, phi_denom, data, collapsed_gp_ws);

    // ---- Prior gradients (outer params only) ----
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // GP hyperparameter priors
    grad[layout.log_sigma2_gp_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
    grad[layout.log_phi_gp_idx] = 1.0;  // Uniform prior Jacobian

    // ---- Data likelihood gradient at w* ----
    // Compute residuals at the mode w*
    std::vector<double> resid_num(N), resid_denom(N);
    collapsed_gp_compute_residuals(
        collapsed_gp_ws.w_star.data(), beta_num, beta_denom,
        phi_num, phi_denom, data,
        resid_num.data(), resid_denom.data());

    // Scatter to beta gradients + phi likelihood gradient
    for (int i = 0; i < N; i++) {
        for (int p = 0; p < data.legacy.p_num; p++) {
            grad[layout.legacy.beta_num_start + p] += resid_num[i] * data.legacy.X_num_flat[i * data.legacy.p_num + p];
        }
        if (!is_binomial) {
            for (int p = 0; p < data.legacy.p_denom; p++) {
                grad[layout.legacy.beta_denom_start + p] += resid_denom[i] * data.legacy.X_denom_flat[i * data.legacy.p_denom + p];
            }
        }
        // Scatter to RE gradients
        if (layout.has_re && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += resid_num[i] + resid_denom[i];
        }

        // Phi (dispersion) likelihood gradient
        if (layout.legacy.has_phi_num || layout.legacy.has_phi_denom) {
            int loc_i = data.gp_data.obs_to_loc[i];
            double eta_num_i = 0.0, eta_denom_i = 0.0;
            for (int p = 0; p < data.legacy.p_num; p++)
                eta_num_i += data.legacy.X_num_flat[i * data.legacy.p_num + p] * beta_num[p];
            if (!is_binomial) {
                for (int p = 0; p < data.legacy.p_denom; p++)
                    eta_denom_i += data.legacy.X_denom_flat[i * data.legacy.p_denom + p] * beta_denom[p];
            }
            eta_num_i += collapsed_gp_ws.w_star[loc_i];
            if (!is_binomial && data.gp_data.shared) eta_denom_i += collapsed_gp_ws.w_star[loc_i];
            if (layout.has_re && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                double re_val = (data.re_parameterization == 1) ?
                    sigma_re * re[data.re_group[i] - 1] : re[data.re_group[i] - 1];
                eta_num_i += re_val;
                if (!is_binomial) eta_denom_i += re_val;
            }
            accumulate_phi_likelihood_grad(data, layout, i, eta_num_i, eta_denom_i,
                                            phi_num, phi_denom, grad.data());
        }
    }

    // ---- GP hyperparameter gradients from NNGP prior at w* ----
    // d/d(log sigma2) and d/d(log phi) of log p_NNGP(w*|sigma2,phi)
    // Use the existing NNGP gradient function
    tulpa_gp::NNGPGradients nngp_grads;
    std::vector<double> w_star_vec(collapsed_gp_ws.w_star.begin(),
                                    collapsed_gp_ws.w_star.end());
    tulpa_gp::gp_nngp_gradients(w_star_vec, sigma2_gp, phi_gp,
                                  data.gp_data, nngp_grads);
    grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
    grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;

    // ---- Laplace correction gradient via numerical differentiation ----
    // The Laplace log-det depends on w* which depends on ALL params implicitly.
    // We compute d/drho [-0.5 log det(W+Q)] numerically for each outer param,
    // using warm-started Newton solves (1-2 iters each from current w*).
    // For non-GP params (beta, phi, RE), we reuse the NNGP structure from
    // collapsed_gp_ws (Q doesn't change), skipping NNGP rebuild.
    {
        const double eps = 1e-5;
        std::vector<double> params_pert = params;
        const int gp_idx1 = layout.log_sigma2_gp_idx;
        const int gp_idx2 = layout.log_phi_gp_idx;

        for (int j = 0; j < n_params; j++) {
            double orig = params_pert[j];
            bool is_gp_hyperparam = (j == gp_idx1 || j == gp_idx2);

            // Forward perturbation
            params_pert[j] = orig + eps;
            double sigma2_p = std::exp(params_pert[gp_idx1]);
            double phi_p = std::exp(params_pert[gp_idx2]);
            const double* beta_num_p = &params_pert[layout.legacy.beta_num_start];
            const double* beta_denom_p = is_binomial ? beta_num_p : &params_pert[layout.legacy.beta_denom_start];
            double phi_num_p = layout.legacy.has_phi_num ? std::exp(params_pert[layout.legacy.log_phi_num_idx]) : phi_num;
            double phi_denom_p = layout.legacy.has_phi_denom ? std::exp(params_pert[layout.legacy.log_phi_denom_idx]) : phi_denom;
            double ld_plus = laplace_log_det_full(
                beta_num_p, beta_denom_p, sigma2_p, phi_p,
                phi_num_p, phi_denom_p, data, collapsed_gp_ws.w_star,
                is_gp_hyperparam ? nullptr : &collapsed_gp_ws,
                is_gp_hyperparam);

            // Backward perturbation
            params_pert[j] = orig - eps;
            sigma2_p = std::exp(params_pert[gp_idx1]);
            phi_p = std::exp(params_pert[gp_idx2]);
            beta_num_p = &params_pert[layout.legacy.beta_num_start];
            beta_denom_p = is_binomial ? beta_num_p : &params_pert[layout.legacy.beta_denom_start];
            phi_num_p = layout.legacy.has_phi_num ? std::exp(params_pert[layout.legacy.log_phi_num_idx]) : phi_num;
            phi_denom_p = layout.legacy.has_phi_denom ? std::exp(params_pert[layout.legacy.log_phi_denom_idx]) : phi_denom;
            double ld_minus = laplace_log_det_full(
                beta_num_p, beta_denom_p, sigma2_p, phi_p,
                phi_num_p, phi_denom_p, data, collapsed_gp_ws.w_star,
                is_gp_hyperparam ? nullptr : &collapsed_gp_ws,
                is_gp_hyperparam);

            grad[j] += (ld_plus - ld_minus) / (2.0 * eps);
            params_pert[j] = orig;
        }
    }

    // ---- NC transform for RE ----
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // ---- Log-posterior ----
    if (log_post_out) {
        // Compute full log-posterior including priors on outer params
        // collapsed_lp already has data_ll + nngp_prior
        double lp = collapsed_lp;

        // Laplace correction: -0.5 * log det(W + Q) via sparse Cholesky
        lp += collapsed_gp_ws.laplace_log_det;

        // Beta priors
        for (int p = 0; p < data.legacy.p_num; p++)
            lp += -0.5 * beta_num[p] * beta_num[p] / (data.sigma_beta * data.sigma_beta);
        for (int p = 0; p < data.legacy.p_denom; p++)
            lp += -0.5 * beta_denom[p] * beta_denom[p] / (data.sigma_beta * data.sigma_beta);

        // RE priors
        if (layout.has_re) {
            int n_re = layout.re_end - layout.re_start;
            double sigma_re2 = sigma_re * sigma_re;
            for (int g = 0; g < n_re; g++) {
                if (data.re_parameterization == 1) {
                    // NC: z ~ N(0,1)
                    lp += -0.5 * re[g] * re[g];
                } else {
                    lp += -0.5 * re[g] * re[g] / sigma_re2;
                }
            }
            // sigma_re half-Cauchy prior (log-scale)
            double ratio = sigma_re / data.sigma_re_scale;
            lp += -std::log(1.0 + ratio * ratio)
                  + params[layout.log_sigma_re_idx];
        }

        // GP hyperparameter priors
        double sigma_gp = std::sqrt(sigma2_gp);
        double rate = -std::log(data.gp_sigma2_prior_alpha) / data.gp_sigma2_prior_U;
        lp += std::log(rate) - rate * sigma_gp - std::log(2.0 * sigma_gp)
              + params[layout.log_sigma2_gp_idx];
        // phi uniform + Jacobian
        lp += params[layout.log_phi_gp_idx]
              - std::log(data.gp_phi_prior_upper - data.gp_phi_prior_lower);

        // Phi (dispersion) priors
        if (layout.legacy.has_phi_num) {
            double log_phi = params[layout.legacy.log_phi_num_idx];
            lp += log_phi;  // Jacobian for log transform (exponential prior)
        }
        if (layout.legacy.has_phi_denom) {
            double log_phi = params[layout.legacy.log_phi_denom_idx];
            lp += log_phi;
        }

        *log_post_out = lp;
    }
}
