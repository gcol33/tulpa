// log_post_impl_likelihood_block.h
// Function-body fragment of compute_log_post_impl<T> in log_post_impl.h.
// Included via #include directly inside the function body — relies on the
// surrounding lexical scope (params, data, layout, log_post, T, beta_*,
// phi_*, re_vals, re_term_offsets, etc.). NOT standalone-compilable.
// No header guard: each fragment is included exactly once per umbrella.
// Observation-loop log-likelihood accumulation.


    // Note: No OpenMP here - autodiff tape is not thread-safe
    // For double, this is slightly slower but correct

    for (int i = 0; i < data.N; i++) {
        // Compute linear predictors
        T eta_num = T(0.0);
        T eta_denom = T(0.0);

        for (int j = 0; j < data.legacy.p_num; j++) {
            eta_num = eta_num + data.legacy.X_num_flat[i * data.legacy.p_num + j] * beta_num[j];
        }
        for (int j = 0; j < data.legacy.p_denom; j++) {
            eta_denom = eta_denom + data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * beta_denom[j];
        }

        // Add random effects (shared between num and denom)
        if (layout.has_re) {
            int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;

            if (layout.has_re_slopes) {
                // Random slopes: iterate over ALL terms (some may have slopes, others intercept-only)
                // Matches compute_log_post() in hmc_sampler.cpp (lines 1939-1966)
                for (int t = 0; t < n_terms; t++) {
                    int re_group_idx = data.re_group_multi_flat[i * n_terms + t];
                    if (re_group_idx > 0) {
                        int g = re_group_idx - 1;
                        int n_coefs_t = layout.re_n_coefs_multi[t];
                        int off = re_term_offsets[t];
                        bool is_corr_t = !layout.re_correlated_multi.empty() &&
                                         layout.re_correlated_multi[t] && n_coefs_t > 1;

                        // For correlated or non-centered: use pre-computed re_vals
                        // For centered uncorrelated: re_vals also pre-computed above
                        T re_contrib = re_vals[off + g * n_coefs_t];

                        // Slope contributions (only if this term has slopes)
                        int n_slopes = n_coefs_t - 1;
                        if (n_slopes > 0 && t < (int)data.re_slope_matrices.size() &&
                            !data.re_slope_matrices[t].empty()) {
                            for (int s = 0; s < n_slopes; s++) {
                                double x_slope = data.re_slope_matrices[t][i * n_slopes + s];
                                re_contrib = re_contrib
                                    + re_vals[off + g * n_coefs_t + 1 + s] * T(x_slope);
                            }
                        }

                        eta_num = eta_num + re_contrib;
                        eta_denom = eta_denom + re_contrib;
                    }
                }
            } else if (n_terms > 1) {
                // Crossed RE: multiple intercept-only terms
                for (int t = 0; t < n_terms; t++) {
                    int group_idx = data.re_group_multi_flat[i * n_terms + t];
                    if (group_idx > 0) {
                        int g = group_idx - 1;
                        int off = re_term_offsets[t];
                        eta_num = eta_num + re_vals[off + g];
                        eta_denom = eta_denom + re_vals[off + g];
                    }
                }
            } else if (data.re_group[i] > 0) {
                // Simple single-term intercept RE
                int g = data.re_group[i] - 1;
                eta_num = eta_num + re_vals[g];
                eta_denom = eta_denom + re_vals[g];
            }
        }

        // Add spatial effects
        if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
            int s = data.spatial_group[i] - 1;
            T spatial_effect;

            if (layout.is_bym2) {
                T scaled_phi = phi_spatial[s] * T(data.bym2_scale_factor);
                spatial_effect = sigma_s_bym2 * scaled_phi + sigma_u_bym2 * theta_bym2[s];
            } else {
                spatial_effect = phi_spatial[s];
            }

            eta_num = eta_num + spatial_effect;
            eta_denom = eta_denom + spatial_effect;
        }

        // Add GP spatial effect (map observation to unique location)
        if (layout.is_gp && !gp_w.empty()) {
            int loc_i = data.gp_data.obs_to_loc[i];
            T gp_effect = gp_w[loc_i];
            if (data.gp_data.shared) {
                eta_num = eta_num + gp_effect;
                eta_denom = eta_denom + gp_effect;
            } else {
                eta_num = eta_num + gp_effect;
            }
        }

        // Add multi-scale GP spatial effect
        if (layout.is_multiscale_gp && !ms_gp_effect.empty()) {
            T msgp_effect = ms_gp_effect[i];
            if (data.multiscale_gp_data.shared) {
                eta_num = eta_num + msgp_effect;
                eta_denom = eta_denom + msgp_effect;
            } else {
                eta_num = eta_num + msgp_effect;
            }
        }

        // Add HSGP spatial effect
        if (layout.is_hsgp && !hsgp_f_impl.empty()) {
            T hsgp_effect = hsgp_f_impl[i];
            if (data.hsgp_data.shared) {
                eta_num = eta_num + hsgp_effect;
                eta_denom = eta_denom + hsgp_effect;
            } else {
                eta_num = eta_num + hsgp_effect;
            }
        }

        // Add temporal effects
        if (layout.has_temporal && !data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            T temporal_effect = phi_temporal[g * data.n_times + t];

            if (data.temporal_shared) {
                eta_num = eta_num + temporal_effect;
                eta_denom = eta_denom + temporal_effect;
            } else {
                eta_num = eta_num + temporal_effect;
            }
        }

        // Add multiscale temporal effect
        if (layout.has_multiscale_temporal && !ms_temporal_eta.empty()) {
            T ms_temp_effect = ms_temporal_eta[i];
            if (data.multiscale_temporal_data.shared) {
                eta_num = eta_num + ms_temp_effect;
                eta_denom = eta_denom + ms_temp_effect;
            } else {
                eta_num = eta_num + ms_temp_effect;
            }
        }

        // Add SVC (Spatially-Varying Coefficients) effect
        if (layout.has_svc && !svc_eta.empty()) {
            T svc_effect = svc_eta[i];
            if (data.svc_data.shared) {
                eta_num = eta_num + svc_effect;
                eta_denom = eta_denom + svc_effect;
            } else {
                eta_num = eta_num + svc_effect;
            }
        }

        // Add TVC (Temporally-Varying Coefficients) effect
        if (layout.has_tvc && !tvc_eta.empty()) {
            T tvc_effect = tvc_eta[i];
            if (data.tvc_data.shared) {
                eta_num = eta_num + tvc_effect;
                eta_denom = eta_denom + tvc_effect;
            } else {
                eta_num = eta_num + tvc_effect;
            }
        }

        // Add latent factor effect
        if (layout.has_latent && !latent_eta.empty()) {
            T latent_effect = latent_eta[i];
            if (data.latent_shared) {
                eta_num = eta_num + latent_effect;
                eta_denom = eta_denom + latent_effect;
            } else {
                eta_num = eta_num + latent_effect;
            }
        }

        // Add spatiotemporal interaction effect
        if (layout.has_spatiotemporal && !st_delta_impl.empty()) {
            T st_effect = T(0.0);
            if (data.st_is_hsgp) {
                // HSGP-ST: sum_j Phi[i,j] * delta_st[j * T + t - 1]
                int t_st = data.spatiotemporal_data.t_idx[i] - 1;  // 0-based
                int M = data.st_hsgp_data.m_total;
                int T_st = data.spatiotemporal_data.n_times;
                for (int j = 0; j < M; j++) {
                    st_effect = st_effect
                        + T(data.st_hsgp_data.phi_flat[i * M + j])
                        * st_delta_impl[j * T_st + t_st];
                }
            } else {
                // ICAR-ST: direct index lookup
                int st_idx = data.spatiotemporal_data.st_flat[i];
                if (st_idx > 0) st_effect = st_delta_impl[st_idx - 1];
            }
            if (data.spatiotemporal_data.shared) {
                eta_num = eta_num + st_effect;
                eta_denom = eta_denom + st_effect;
            } else {
                eta_num = eta_num + st_effect;
            }
        }

        // Compute ZI/OI linear predictors if needed
        T logit_zi = T(0.0);
        T logit_oi = T(0.0);

        if (layout.has_zi && data.p_zi > 0) {
            for (int j = 0; j < data.p_zi; j++) {
                logit_zi = logit_zi + data.X_zi_flat[i * data.p_zi + j] * beta_zi[j];
            }
        }

        if (layout.has_oi && data.p_oi > 0) {
            for (int j = 0; j < data.p_oi; j++) {
                logit_oi = logit_oi + data.X_oi_flat[i * data.p_oi + j] * beta_oi[j];
            }
        }

        // Compute likelihood based on model type
        T ll_i = T(0.0);

        if (data.legacy.model_type == ModelType::BINOMIAL) {
            T p = inv_logit(eta_num);
            int y = data.legacy.y_num[i];
            int n = data.legacy.y_denom[i];

            // Handle different ZI types for binomial
            if (data.zi_type == ZIType::ZI_BINOMIAL) {
                ll_i = log_lik_zi_binomial(y, n, p, logit_zi);
            } else if (data.zi_type == ZIType::OI_BINOMIAL) {
                ll_i = log_lik_oi_binomial(y, n, p, logit_oi);
            } else if (data.zi_type == ZIType::ZOIB) {
                ll_i = log_lik_zoib(y, n, p, logit_zi, logit_oi);
            } else if (data.zi_type == ZIType::HURDLE_BINOMIAL) {
                ll_i = log_lik_hurdle_binomial(y, n, p, logit_zi);
            } else {
                ll_i = log_lik_binomial(y, n, p);
            }

        } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
            T mu_num = safe_exp(eta_num);
            T mu_denom = safe_exp(eta_denom);

            // Check for zero-inflation on numerator
            if (layout.has_zi) {
                if (data.zi_type == ZIType::ZI_NEGBIN) {
                    ll_i = log_lik_zi_negbin(data.legacy.y_num[i], mu_num, phi_num, logit_zi);
                } else if (data.zi_type == ZIType::HURDLE_NEGBIN) {
                    ll_i = log_lik_hurdle_negbin(data.legacy.y_num[i], mu_num, phi_num, logit_zi);
                } else if (data.zi_type == ZIType::ZI_POISSON) {
                    ll_i = log_lik_zi_negbin(data.legacy.y_num[i], mu_num, phi_num, logit_zi);
                } else if (data.zi_type == ZIType::HURDLE_POISSON) {
                    ll_i = log_lik_hurdle_negbin(data.legacy.y_num[i], mu_num, phi_num, logit_zi);
                } else {
                    ll_i = log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num);
                }
            } else {
                ll_i = log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num);
            }
            // Denominator is always standard (not zero-inflated)
            ll_i = ll_i + log_lik_negbin(data.legacy.y_denom[i], mu_denom, phi_denom);

        } else if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
            T mu_num = safe_exp(eta_num);
            T mu_denom = safe_exp(eta_denom);

            // Check for zero-inflation on numerator
            if (layout.has_zi) {
                if (data.zi_type == ZIType::ZI_POISSON) {
                    ll_i = log_lik_zi_poisson(data.legacy.y_num[i], mu_num, logit_zi);
                } else if (data.zi_type == ZIType::HURDLE_POISSON) {
                    ll_i = log_lik_hurdle_poisson(data.legacy.y_num[i], mu_num, logit_zi);
                } else if (data.zi_type == ZIType::ZI_NEGBIN) {
                    ll_i = log_lik_zi_poisson(data.legacy.y_num[i], mu_num, logit_zi);
                } else if (data.zi_type == ZIType::HURDLE_NEGBIN) {
                    ll_i = log_lik_hurdle_poisson(data.legacy.y_num[i], mu_num, logit_zi);
                } else {
                    ll_i = log_lik_poisson(data.legacy.y_num[i], mu_num);
                }
            } else {
                ll_i = log_lik_poisson(data.legacy.y_num[i], mu_num);
            }
            // Denominator is gamma (continuous)
            ll_i = ll_i + log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);

        } else if (data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
            T mu_num = safe_exp(eta_num);
            T mu_denom = safe_exp(eta_denom);

            // Check for zero-inflation on numerator
            if (layout.has_zi) {
                if (data.zi_type == ZIType::ZI_NEGBIN) {
                    ll_i = log_lik_zi_negbin(data.legacy.y_num[i], mu_num, phi_num, logit_zi);
                } else if (data.zi_type == ZIType::HURDLE_NEGBIN) {
                    ll_i = log_lik_hurdle_negbin(data.legacy.y_num[i], mu_num, phi_num, logit_zi);
                } else if (data.zi_type == ZIType::ZI_POISSON) {
                    ll_i = log_lik_zi_negbin(data.legacy.y_num[i], mu_num, phi_num, logit_zi);
                } else if (data.zi_type == ZIType::HURDLE_POISSON) {
                    ll_i = log_lik_hurdle_negbin(data.legacy.y_num[i], mu_num, phi_num, logit_zi);
                } else {
                    ll_i = log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num);
                }
            } else {
                ll_i = log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num);
            }
            // Denominator is gamma (continuous)
            ll_i = ll_i + log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);

        } else if (data.legacy.model_type == ModelType::GAMMA_GAMMA) {
            // Gamma-Gamma: both responses are continuous Gamma
            // phi_num = shape_num, phi_denom = shape_denom
            T mu_num = safe_exp(eta_num);
            T mu_denom = safe_exp(eta_denom);
            ll_i = log_lik_gamma_gamma(data.legacy.y_num_cont[i], data.legacy.y_denom_cont[i],
                                       mu_num, mu_denom, phi_num, phi_denom);

        } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
            // Lognormal-Lognormal: both responses are continuous Lognormal
            // eta = mean on log scale, phi = sigma (std dev on log scale)
            ll_i = log_lik_lognormal_lognormal(data.legacy.y_num_cont[i], data.legacy.y_denom_cont[i],
                                               eta_num, eta_denom, phi_num, phi_denom);

        } else if (data.legacy.model_type == ModelType::BETA_BINOMIAL) {
            // Beta-Binomial: overdispersed binomial
            // phi_num = precision parameter (alpha + beta)
            T p = inv_logit(eta_num);
            int y = data.legacy.y_num[i];
            int n = data.legacy.y_denom[i];
            ll_i = log_lik_beta_binomial(y, n, p, phi_num);
        }

        log_post = log_post + ll_i;
    }

