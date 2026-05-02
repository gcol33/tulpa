// hmc_gradient_composite_phase3_loop.h
// Function-body fragment of compute_gradient_composite: Phase 3 —
// vectorized 3-pass observation loop. NOT standalone-compilable;
// relies on the surrounding scope.

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

