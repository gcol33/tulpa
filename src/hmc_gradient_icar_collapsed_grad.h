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
