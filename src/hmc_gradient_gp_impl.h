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

// =====================================================================
// Collapsed ICAR/BYM2 gradient
// ICAR: H-mode - analytical envelope + analytical Laplace via implicit fn
//       thm (Woodbury for sum-to-zero rank-1, sparse Cholesky on A = W+tau*Q).
// BYM2: H-mode - same structure with 2S inner state [phi; theta]; direct
//       sigma/rho traces via dense 2S Hinv plus indirect IFT corrections
//       via cross-Hessians (compute_laplace_gradient_bym2_H).
// =====================================================================

void compute_gradient_icar_collapsed(
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

    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);
    int N = data.N;
    int S = data.n_spatial_units;

    // Pre-compute actual RE values (NC - actual)
    std::vector<double> re_vals;
    if (layout.has_re) {
        int n_re = layout.re_end - layout.re_start;
        re_vals.resize(n_re);
        for (int g = 0; g < n_re; g++) {
            re_vals[g] = (data.re_parameterization == 1) ? sigma_re * re[g] : re[g];
        }
    }

    // Spatial hyperparameters
    bool is_bym2 = layout.is_bym2_collapsed;
    double tau = 0.0, sigma_total = 0.0, rho = 0.0;
    double a = 0.0, c_bym2 = 0.0;

    if (is_bym2) {
        sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
        double logit_rho = params[layout.logit_rho_bym2_idx];
        rho = 1.0 / (1.0 + std::exp(-logit_rho));
        a = sigma_total * std::sqrt(rho) * data.bym2_scale_factor;
        c_bym2 = sigma_total * std::sqrt(1.0 - rho);
    } else {
        tau = std::exp(params[layout.log_tau_spatial_idx]);
    }

    // ---- Inner Laplace: find phi* (and theta* for BYM2) ----
    double collapsed_lp;
    if (is_bym2) {
        collapsed_lp = collapsed_bym2_find_mode(
            beta_num, beta_denom, sigma_total, rho, data.bym2_scale_factor,
            phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            data, collapsed_icar_ws);
    } else {
        collapsed_lp = collapsed_icar_find_mode(
            beta_num, beta_denom, tau, phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            data, collapsed_icar_ws);
    }

    // ---- Outer priors (simple analytical, don't depend on phi*) ----
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // ---- H-mode: analytical envelope + analytical Laplace ----
    // ICAR: 1 spatial hyperparameter (log_tau).
    // BYM2: 2 spatial hyperparameters (log_sigma, logit_rho); same structure
    //       with 2S inner state and IFT cross-Hessians for both sigma and rho.
    if (!is_bym2) {
        // === Part A: Envelope theorem gradient ===
        // At mode, df/dinner = 0, so d/drho[f(mode*,rho)] = df/dinner|_{mode*}

        // A1: Data LL gradient via residual scattering
        std::vector<double> resid_num(N), resid_denom(N);
        collapsed_icar_compute_residuals(
            collapsed_icar_ws, beta_num, beta_denom,
            phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            0.0, 0.0,  // not BYM2
            data, resid_num.data(), resid_denom.data());

        // Scatter residuals to beta_num: X_num' * resid_num
        for (int k = 0; k < data.legacy.p_num; k++) {
            double sum = 0.0;
            for (int i = 0; i < N; i++)
                sum += resid_num[i] * data.legacy.X_num_flat[i * data.legacy.p_num + k];
            grad[layout.legacy.beta_num_start + k] += sum;
        }
        // Scatter to beta_denom: X_denom' * resid_denom
        if (!is_binomial) {
            for (int k = 0; k < data.legacy.p_denom; k++) {
                double sum = 0.0;
                for (int i = 0; i < N; i++)
                    sum += resid_denom[i] * data.legacy.X_denom_flat[i * data.legacy.p_denom + k];
                grad[layout.legacy.beta_denom_start + k] += sum;
            }
        }
        // Scatter to RE (w.r.t. centered values)
        if (layout.has_re) {
            int n_re = layout.re_end - layout.re_start;
            for (int i = 0; i < N; i++) {
                if (data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                    int g = data.re_group[i] - 1;
                    if (g < n_re) {
                        grad[layout.re_start + g] += resid_num[i] + resid_denom[i];
                    }
                }
            }
        }

        // A2: Dispersion parameter gradients from data LL at mode
        // (envelope theorem: dLL/dlog_phi evaluated at mode*)
        if (layout.legacy.has_phi_num || layout.legacy.has_phi_denom) {
            for (int i = 0; i < N; i++) {
                int s = data.spatial_group[i] - 1;
                double eta_num_i = 0.0, eta_denom_i = 0.0;
                for (int p = 0; p < data.legacy.p_num; p++)
                    eta_num_i += data.legacy.X_num_flat[i * data.legacy.p_num + p] * beta_num[p];
                if (!is_binomial) {
                    for (int p = 0; p < data.legacy.p_denom; p++)
                        eta_denom_i += data.legacy.X_denom_flat[i * data.legacy.p_denom + p] * beta_denom[p];
                }
                eta_num_i += collapsed_icar_ws.phi_star[s];
                if (!is_binomial) eta_denom_i += collapsed_icar_ws.phi_star[s];
                if (re_vals.data() && data.re_group.size() > (size_t)i && data.re_group[i] > 0)  {
                    eta_num_i += re_vals[data.re_group[i] - 1];
                    if (!is_binomial) eta_denom_i += re_vals[data.re_group[i] - 1];
                }

                double mu_num = std::exp(std::min(eta_num_i, 20.0));

                // Per-family dispersion gradients
                switch (data.legacy.model_type) {
                    case ModelType::POISSON_GAMMA: {
                        if (layout.legacy.has_phi_denom) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = data.legacy.y_denom_cont[i];
                            double alpha = phi_denom;
                            // d/d(log_alpha)[LL_denom] = alpha * dLL/dalpha
                            // dLL/dalpha = log(alpha) + 1 - digamma(alpha) + log(y/mu) - y/mu
                            grad[layout.legacy.log_phi_denom_idx] += alpha * (
                                std::log(alpha) + 1.0 - R::digamma(alpha)
                                + std::log(std::max(y_d, 1e-10)) - std::log(mu_d)
                                - y_d / mu_d);
                        }
                        break;
                    }
                    case ModelType::NEGBIN_NEGBIN: {
                        if (layout.legacy.has_phi_num) {
                            double r = phi_num;
                            double y = data.legacy.y_num[i];
                            grad[layout.legacy.log_phi_num_idx] += r * (
                                R::digamma(y + r) - R::digamma(r)
                                + std::log(r) + 1.0
                                - std::log(mu_num + r) - (y + r) / (mu_num + r));
                        }
                        if (layout.legacy.has_phi_denom && !is_binomial) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = (double)data.legacy.y_denom[i];
                            double r_d = phi_denom;
                            grad[layout.legacy.log_phi_denom_idx] += r_d * (
                                R::digamma(y_d + r_d) - R::digamma(r_d)
                                + std::log(r_d) + 1.0
                                - std::log(mu_d + r_d) - (y_d + r_d) / (mu_d + r_d));
                        }
                        break;
                    }
                    case ModelType::NEGBIN_GAMMA: {
                        if (layout.legacy.has_phi_num) {
                            double r = phi_num;
                            double y = data.legacy.y_num[i];
                            grad[layout.legacy.log_phi_num_idx] += r * (
                                R::digamma(y + r) - R::digamma(r)
                                + std::log(r) + 1.0
                                - std::log(mu_num + r) - (y + r) / (mu_num + r));
                        }
                        if (layout.legacy.has_phi_denom) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = data.legacy.y_denom_cont[i];
                            double alpha = phi_denom;
                            grad[layout.legacy.log_phi_denom_idx] += alpha * (
                                std::log(alpha) + 1.0 - R::digamma(alpha)
                                + std::log(std::max(y_d, 1e-10)) - std::log(mu_d)
                                - y_d / mu_d);
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
        }

        // A3: ICAR prior gradient w.r.t. log_tau (envelope: holding mode* fixed)
        // d/d(log_tau)[-0.5*tau*phi*'Q*phi* + 0.5*(S-1)*log(tau)]
        //   = -0.5*tau*phi*'Q*phi* + 0.5*(S-1)
        {
            std::vector<double> Qphi(S);
            icar_precision_matvec(collapsed_icar_ws.phi_star.data(), Qphi.data(), S,
                                  data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
            double phiQphi = 0.0;
            for (int s = 0; s < S; s++) phiQphi += collapsed_icar_ws.phi_star[s] * Qphi[s];
            grad[layout.log_tau_spatial_idx] += -0.5 * tau * phiQphi + 0.5 * (S - 1);
        }

        // === Part B: H-mode Laplace gradient ===
        auto laplace_result = compute_laplace_gradient_icar_H(
            collapsed_icar_ws, beta_num, beta_denom,
            tau, phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            data, layout, n_params);

        if (laplace_result.success) {
            for (int j = 0; j < n_params; j++) {
                grad[j] += laplace_result.laplace_grad[j];
            }
        } else {
            // Fallback: numerical Laplace gradient for all params
            // (recomputes mode_lp + Laplace via central differences)
            auto collapsed_log_post = [&](const std::vector<double>& p) -> double {
                auto cp_l = extract_common_params(p, layout);
                double tau_l = std::exp(p[layout.log_tau_spatial_idx]);
                std::vector<double> re_vals_l;
                if (layout.has_re) {
                    int n_re = layout.re_end - layout.re_start;
                    re_vals_l.resize(n_re);
                    for (int g = 0; g < n_re; g++) {
                        re_vals_l[g] = (data.re_parameterization == 1)
                            ? cp_l.sigma_re * cp_l.re[g] : cp_l.re[g];
                    }
                }
                CollapsedICARWorkspace temp_ws;
                temp_ws.init(S, false);
                temp_ws.phi_star = collapsed_icar_ws.phi_star;
                temp_ws.mode_found = true;
                double mode_lp = collapsed_icar_find_mode(
                    cp_l.beta_num, cp_l.beta_denom, tau_l,
                    cp_l.phi_num, cp_l.phi_denom,
                    re_vals_l.empty() ? nullptr : re_vals_l.data(),
                    data, temp_ws);
                double lp = mode_lp + temp_ws.laplace_log_det;
                lp += (data.tau_spatial_shape - 1.0) * std::log(tau_l) - data.tau_spatial_rate * tau_l
                      + p[layout.log_tau_spatial_idx];
                return lp;
            };
            const double eps = 1e-5;
            std::vector<double> params_pert = params;
            for (int j = 0; j < n_params; j++) {
                double orig = params_pert[j];
                params_pert[j] = orig + eps;
                double lp_plus = collapsed_log_post(params_pert);
                params_pert[j] = orig - eps;
                double lp_minus = collapsed_log_post(params_pert);
                grad[j] += (lp_plus - lp_minus) / (2.0 * eps);
                params_pert[j] = orig;
            }
        }

        // === Part C: Spatial hyperparameter prior gradient (analytical) ===
        // Gamma(shape, rate) prior on tau, on log scale:
        // d/d(log_tau)[(shape-1)*log(tau) - rate*tau + log_tau]
        //   = (shape-1) - rate*tau + 1
        grad[layout.log_tau_spatial_idx] += (data.tau_spatial_shape - 1.0)
                                           - data.tau_spatial_rate * tau + 1.0;

    } else {
        // ---- BYM2: H-mode analytical gradient ----
        // Same structure as ICAR: envelope + analytical Laplace + outer priors

        // === Part A: Envelope theorem gradient ===
        // A1: Data LL gradient via residual scattering
        std::vector<double> resid_num(N), resid_denom(N);
        collapsed_icar_compute_residuals(
            collapsed_icar_ws, beta_num, beta_denom,
            phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            a, c_bym2,  // BYM2 scaling
            data, resid_num.data(), resid_denom.data());

        // Scatter residuals to beta_num
        for (int k = 0; k < data.legacy.p_num; k++) {
            double sum = 0.0;
            for (int i = 0; i < N; i++)
                sum += resid_num[i] * data.legacy.X_num_flat[i * data.legacy.p_num + k];
            grad[layout.legacy.beta_num_start + k] += sum;
        }
        // Scatter to beta_denom
        if (!is_binomial) {
            for (int k = 0; k < data.legacy.p_denom; k++) {
                double sum = 0.0;
                for (int i = 0; i < N; i++)
                    sum += resid_denom[i] * data.legacy.X_denom_flat[i * data.legacy.p_denom + k];
                grad[layout.legacy.beta_denom_start + k] += sum;
            }
        }
        // Scatter to RE
        if (layout.has_re) {
            int n_re = layout.re_end - layout.re_start;
            for (int i = 0; i < N; i++) {
                if (data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                    int g = data.re_group[i] - 1;
                    if (g < n_re) {
                        grad[layout.re_start + g] += resid_num[i] + resid_denom[i];
                    }
                }
            }
        }

        // A2: Dispersion parameter gradients from data LL at mode (envelope)
        if (layout.legacy.has_phi_num || layout.legacy.has_phi_denom) {
            for (int i = 0; i < N; i++) {
                int s = data.spatial_group[i] - 1;
                double b_s = a * collapsed_icar_ws.phi_star[s]
                           + c_bym2 * collapsed_icar_ws.theta_star[s];
                double eta_num_i = 0.0, eta_denom_i = 0.0;
                for (int p = 0; p < data.legacy.p_num; p++)
                    eta_num_i += data.legacy.X_num_flat[i * data.legacy.p_num + p] * beta_num[p];
                if (!is_binomial) {
                    for (int p = 0; p < data.legacy.p_denom; p++)
                        eta_denom_i += data.legacy.X_denom_flat[i * data.legacy.p_denom + p] * beta_denom[p];
                }
                eta_num_i += b_s;
                if (!is_binomial) eta_denom_i += b_s;
                if (re_vals.data() && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                    eta_num_i += re_vals[data.re_group[i] - 1];
                    if (!is_binomial) eta_denom_i += re_vals[data.re_group[i] - 1];
                }
                double mu_num = std::exp(std::min(eta_num_i, 20.0));

                switch (data.legacy.model_type) {
                    case ModelType::POISSON_GAMMA: {
                        if (layout.legacy.has_phi_denom) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = data.legacy.y_denom_cont[i];
                            double alpha = phi_denom;
                            grad[layout.legacy.log_phi_denom_idx] += alpha * (
                                std::log(alpha) + 1.0 - R::digamma(alpha)
                                + std::log(std::max(y_d, 1e-10)) - std::log(mu_d)
                                - y_d / mu_d);
                        }
                        break;
                    }
                    case ModelType::NEGBIN_NEGBIN: {
                        if (layout.legacy.has_phi_num) {
                            double r = phi_num;
                            double y = data.legacy.y_num[i];
                            grad[layout.legacy.log_phi_num_idx] += r * (
                                R::digamma(y + r) - R::digamma(r)
                                + std::log(r) + 1.0
                                - std::log(mu_num + r) - (y + r) / (mu_num + r));
                        }
                        if (layout.legacy.has_phi_denom && !is_binomial) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = (double)data.legacy.y_denom[i];
                            double r_d = phi_denom;
                            grad[layout.legacy.log_phi_denom_idx] += r_d * (
                                R::digamma(y_d + r_d) - R::digamma(r_d)
                                + std::log(r_d) + 1.0
                                - std::log(mu_d + r_d) - (y_d + r_d) / (mu_d + r_d));
                        }
                        break;
                    }
                    case ModelType::NEGBIN_GAMMA: {
                        if (layout.legacy.has_phi_num) {
                            double r = phi_num;
                            double y = data.legacy.y_num[i];
                            grad[layout.legacy.log_phi_num_idx] += r * (
                                R::digamma(y + r) - R::digamma(r)
                                + std::log(r) + 1.0
                                - std::log(mu_num + r) - (y + r) / (mu_num + r));
                        }
                        if (layout.legacy.has_phi_denom) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = data.legacy.y_denom_cont[i];
                            double alpha = phi_denom;
                            grad[layout.legacy.log_phi_denom_idx] += alpha * (
                                std::log(alpha) + 1.0 - R::digamma(alpha)
                                + std::log(std::max(y_d, 1e-10)) - std::log(mu_d)
                                - y_d / mu_d);
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
        }

        // A3: BYM2 prior envelope gradients for log_sigma and logit_rho
        // (envelope: holding phi*, theta* fixed, differentiate data LL through b_s)
        {
            double grad_sigma_env = 0.0;
            double grad_rho_env = 0.0;
            double da_drho = a * (1.0 - rho) / 2.0;
            double dc_drho = -c_bym2 * rho / 2.0;
            for (int s = 0; s < S; s++) {
                // Per-site residual sum (from residuals already computed)
                double r_sum = 0.0;
                for (int i = 0; i < N; i++) {
                    if (data.spatial_group[i] - 1 == s)
                        r_sum += resid_num[i] + resid_denom[i];
                }
                double b_s = a * collapsed_icar_ws.phi_star[s]
                           + c_bym2 * collapsed_icar_ws.theta_star[s];
                grad_sigma_env += r_sum * b_s;  // d/d(log_sigma) = b_s
                double d_rho_s = da_drho * collapsed_icar_ws.phi_star[s]
                               + dc_drho * collapsed_icar_ws.theta_star[s];
                grad_rho_env += r_sum * d_rho_s;
            }
            grad[layout.log_sigma_bym2_idx] += grad_sigma_env;
            grad[layout.logit_rho_bym2_idx] += grad_rho_env;
        }

        // === Part B: H-mode Laplace gradient ===
        auto laplace_result = compute_laplace_gradient_bym2_H(
            collapsed_icar_ws, beta_num, beta_denom,
            a, c_bym2, rho,
            phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            data, layout, n_params);

        if (laplace_result.success) {
            for (int j = 0; j < n_params; j++) {
                grad[j] += laplace_result.laplace_grad[j];
            }
        } else {
            // Numerical fallback for Laplace gradient only
            auto laplace_only = [&](const std::vector<double>& p) -> double {
                auto cp_l = extract_common_params(p, layout);
                double sigma_l = std::exp(p[layout.log_sigma_bym2_idx]);
                double logit_rho_l = p[layout.logit_rho_bym2_idx];
                double rho_l = 1.0 / (1.0 + std::exp(-logit_rho_l));
                double a_l = sigma_l * std::sqrt(rho_l) * data.bym2_scale_factor;
                double c_l = sigma_l * std::sqrt(1.0 - rho_l);
                std::vector<double> re_vals_l;
                if (layout.has_re) {
                    int n_re = layout.re_end - layout.re_start;
                    re_vals_l.resize(n_re);
                    for (int g = 0; g < n_re; g++) {
                        re_vals_l[g] = (data.re_parameterization == 1)
                            ? cp_l.sigma_re * cp_l.re[g] : cp_l.re[g];
                    }
                }
                CollapsedICARWorkspace temp_ws;
                temp_ws.init(S, true);
                temp_ws.phi_star = collapsed_icar_ws.phi_star;
                temp_ws.theta_star = collapsed_icar_ws.theta_star;
                temp_ws.mode_found = true;
                collapsed_bym2_find_mode(
                    cp_l.beta_num, cp_l.beta_denom, sigma_l, rho_l, data.bym2_scale_factor,
                    cp_l.phi_num, cp_l.phi_denom,
                    re_vals_l.empty() ? nullptr : re_vals_l.data(),
                    data, temp_ws);
                return temp_ws.laplace_log_det;
            };
            const double eps = 1e-5;
            std::vector<double> params_pert = params;
            for (int j = 0; j < n_params; j++) {
                double orig = params_pert[j];
                params_pert[j] = orig + eps;
                double ld_plus = laplace_only(params_pert);
                params_pert[j] = orig - eps;
                double ld_minus = laplace_only(params_pert);
                grad[j] += (ld_plus - ld_minus) / (2.0 * eps);
                params_pert[j] = orig;
            }
        }

        // === Part C: BYM2 hyperparameter prior gradients ===
        // Sigma prior: half-Cauchy via d/d(log_sigma)[-log(1 + (sigma/s)^2) + log_sigma]
        {
            double ratio_sc = sigma_total / data.sigma_re_scale;
            double r2 = ratio_sc * ratio_sc;
            grad[layout.log_sigma_bym2_idx] += -2.0 * r2 / (1.0 + r2) + 1.0;
        }
        // Rho prior: Uniform(0,1) Jacobian: log(rho) + log(1-rho)
        // d/d(logit_rho) = d(log(rho))/d(logit_rho) + d(log(1-rho))/d(logit_rho)
        //                = (1-rho) + (-rho) = 1 - 2*rho
        grad[layout.logit_rho_bym2_idx] += 1.0 - 2.0 * rho;
    }

    // ---- NC transform for RE ----
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // ---- Log-posterior ----
    if (log_post_out) {
        double lp = collapsed_lp + collapsed_icar_ws.laplace_log_det;

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
                    lp += -0.5 * re[g] * re[g];
                } else {
                    lp += -0.5 * re[g] * re[g] / sigma_re2;
                }
            }
            double ratio_hc = sigma_re / data.sigma_re_scale;
            lp += -std::log(1.0 + ratio_hc * ratio_hc) + params[layout.log_sigma_re_idx];
        }

        // Spatial hyperparameter priors
        if (is_bym2) {
            double ratio_sc = sigma_total / data.sigma_re_scale;
            lp += -std::log(1.0 + ratio_sc * ratio_sc) + params[layout.log_sigma_bym2_idx];
            lp += std::log(rho) + std::log(1.0 - rho);
        } else {
            lp += (data.tau_spatial_shape - 1.0) * std::log(tau) - data.tau_spatial_rate * tau
                  + params[layout.log_tau_spatial_idx];
        }

        // Phi (dispersion) priors
        if (layout.legacy.has_phi_num) lp += params[layout.legacy.log_phi_num_idx];
        if (layout.legacy.has_phi_denom) lp += params[layout.legacy.log_phi_denom_idx];

        *log_post_out = lp;
    }
}

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
