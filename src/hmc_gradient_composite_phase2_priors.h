// hmc_gradient_composite_phase2_priors.h
// Function-body fragment of compute_gradient_composite: Phase 2 —
// prior gradients computed before the observation loop. NOT
// standalone-compilable; relies on the surrounding scope.

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

