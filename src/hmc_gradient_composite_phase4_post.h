// hmc_gradient_composite_phase4_post.h
// Function-body fragment of compute_gradient_composite: Phase 4 —
// structural / GMRF prior gradients accumulated after the observation
// loop, plus fused log-posterior output. NOT standalone-compilable;
// relies on the surrounding scope.

    // =========================================================================
    // Phase 4: Structural/GMRF prior gradients (post-observation loop)
    // =========================================================================

    // ICAR/BYM2 spatial GMRF
    if (has_icar_bym2) {
        spatial_gmrf_prior_grad(data, layout, spatial_phi, tau_spatial,
                                sigma_s_bym2, sigma_u_bym2, rho_bym2, theta_bym2,
                                grad_spatial_lik.data(),
                                layout.is_bym2 ? grad_theta_lik.data() : nullptr,
                                grad.data());
    }

    // GP (NNGP) backward pass
    if (has_gp_spatial) {
        if (use_nc_gp) {
            // NC: dL/dw -> grad_z, grad_log_sigma2, grad_log_phi via backward pass
            std::vector<double> grad_z_gp(N_gp_c, 0.0);
            double grad_log_sigma2_gp_bw = 0.0, grad_log_phi_gp_bw = 0.0, grad_log_phi_gp_jac = 0.0;
            tulpa_gp::nngp_nc_backward(
                &params[layout.gp_w_start], sigma2_gp_c, phi_gp_c,
                data.gp_data, nc_ws_gp_c, grad_gp_w_lik.data(),
                grad_z_gp.data(), grad_log_sigma2_gp_bw, grad_log_phi_gp_bw, grad_log_phi_gp_jac);
            for (int i = 0; i < N_gp_c; i++)
                grad[layout.gp_w_start + i] += grad_z_gp[i];
            grad[layout.log_sigma2_gp_idx] += grad_log_sigma2_gp_bw;
            grad[layout.log_phi_gp_idx] += grad_log_phi_gp_bw + grad_log_phi_gp_jac;
        } else {
            // Centered: just add likelihood gradients to w
            for (int i = 0; i < N_gp_c; i++)
                grad[layout.gp_w_start + i] += grad_gp_w_lik[i];
        }
    }

    // HSGP spectral density gradients
    if (layout.is_hsgp && data.has_hsgp) {
        int M = data.hsgp_data.m_total;
        std::copy(grad_hsgp_f.begin(), grad_hsgp_f.end(), hsgp_ws.grad_f.begin());
        double grad_log_sigma2 = 0.0, grad_log_lengthscale = 0.0;
        tulpa_hsgp::hsgp_compute_gradients_ws(
            hsgp_beta_ptr, hsgp_sigma2, hsgp_lengthscale,
            data.hsgp_data, hsgp_ws, grad_log_sigma2, grad_log_lengthscale);
        grad[layout.log_sigma2_hsgp_idx] += grad_log_sigma2;
        grad[layout.log_lengthscale_hsgp_idx] += grad_log_lengthscale;
        for (int j = 0; j < M; j++)
            grad[layout.hsgp_beta_start + j] += hsgp_ws.grad_beta_out[j];
    }

    // MSGP-HSGP spectral density gradients
    if (has_msgp_hsgp) {
        std::memcpy(msgp_ws_local_c.grad_f.data(), grad_msgp_f.data(), N * sizeof(double));
        std::memcpy(msgp_ws_regional_c.grad_f.data(), grad_msgp_f.data(), N * sizeof(double));

        double grad_log_sigma2_local_c = 0.0, grad_log_ls_local_c = 0.0;
        tulpa_hsgp::hsgp_compute_gradients_ws(
            msgp_beta_local, msgp_sigma2_local, msgp_ls_local,
            data.msgp_hsgp_data, msgp_ws_local_c,
            grad_log_sigma2_local_c, grad_log_ls_local_c);

        double grad_log_sigma2_regional_c = 0.0, grad_log_ls_regional_c = 0.0;
        tulpa_hsgp::hsgp_compute_gradients_ws(
            msgp_beta_regional, msgp_sigma2_regional, msgp_ls_regional,
            data.msgp_hsgp_data, msgp_ws_regional_c,
            grad_log_sigma2_regional_c, grad_log_ls_regional_c);

        for (int j = 0; j < msgp_m_total; j++) {
            grad[layout.gp_local_start + j] += msgp_ws_local_c.grad_beta_out[j];
            grad[layout.gp_regional_start + j] += msgp_ws_regional_c.grad_beta_out[j];
        }
        grad[layout.log_sigma2_gp_local_idx] += grad_log_sigma2_local_c;
        grad[layout.log_phi_gp_local_idx] += grad_log_ls_local_c;
        grad[layout.log_sigma2_gp_regional_idx] += grad_log_sigma2_regional_c;
        grad[layout.log_phi_gp_regional_idx] += grad_log_ls_regional_c;
    }

    // Temporal GMRF prior
    if (has_gmrf_temporal && T_temporal > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                                 phi_temporal, T_temporal, grad_temporal_lik.data(), grad.data());
    }

    // Temporal GP backward pass
    if (has_temporal_gp) {
        int T_len_gp = layout.temporal_end - layout.temporal_start;

        if (use_nc_tgp) {
            // Copy likelihood gradients to nc workspace
            for (int k = 0; k < n_groups_gp * T_gp; k++)
                nc_ws_composite.dL_df[k] = grad_temporal_lik[k];

            double grad_log_sigma2_gp = 0.0, grad_log_phi_gp = 0.0;
            tulpa_temporal_gp::temporal_gp_nc_backward(
                z_temporal_gp, T_gp, n_groups_gp, sigma2_tgp_comp, phi_tgp_comp,
                data.temporal_gp_data.time_values, nc_ws_composite,
                &grad[layout.temporal_start],
                grad_log_sigma2_gp, grad_log_phi_gp);
            grad[layout.log_sigma2_temporal_gp_idx] += grad_log_sigma2_gp;
            // Convert log_phi gradient to logit_phi gradient
            double chi_tgp = (phi_tgp_comp - phi_lower_tgp) * (phi_upper_tgp - phi_tgp_comp) /
                             (phi_tgp_comp * (phi_upper_tgp - phi_lower_tgp));
            grad[layout.logit_phi_temporal_gp_idx] += grad_log_phi_gp * chi_tgp;
        } else {
            // Centered: just add likelihood gradients
            for (int k = 0; k < T_len_gp; k++)
                grad[layout.temporal_start + k] = grad_temporal_lik[k];
        }
    }

    // Multiscale temporal GMRF prior gradients
    if (has_ms_temporal) {
        tulpa_temporal_grad::MultiscaleTemporalGradients ms_grads;
        tulpa_temporal_grad::multiscale_temporal_prior_gradients(
            trend, n_trend, seasonal, n_seasonal, short_term, n_short,
            sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
            data.multiscale_temporal_data, ms_grads);
        for (int t = 0; t < n_trend; t++) grad[layout.trend_start + t] = grad_trend_lik_c[t] + ms_grads.grad_trend[t];
        for (int t = 0; t < n_seasonal; t++) grad[layout.seasonal_start + t] = grad_seasonal_lik_c[t] + ms_grads.grad_seasonal[t];
        for (int t = 0; t < n_short; t++) grad[layout.short_term_start + t] = grad_short_lik_c[t] + ms_grads.grad_short_term[t];
        if (n_trend > 0) grad[layout.log_sigma2_trend_idx] += ms_grads.grad_log_sigma2_trend;
        if (n_seasonal > 0) grad[layout.log_sigma2_seasonal_idx] += ms_grads.grad_log_sigma2_seasonal;
        if (n_short > 0) grad[layout.log_sigma2_short_idx] += ms_grads.grad_log_sigma2_short;
        if (data.multiscale_temporal_data.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0)
            grad[layout.logit_rho_short_idx] += ms_grads.grad_logit_rho_short;
    }

    // TVC structural prior gradients
    if (layout.has_tvc && data.has_tvc) {
        // Initialize TVC gradient workspace
        static thread_local tulpa_tvc::TVCGradientWS tvc_grad_ws;
        static thread_local std::vector<double> tvc_grad_w_buf, tvc_grad_log_tau_buf;
        static thread_local std::vector<double> tvc_grad_logit_rho_buf, tvc_grad_w_jg_buf, tvc_d_buf;
        tvc_grad_w_buf.assign(n_w, 0.0);
        tvc_grad_log_tau_buf.assign(n_tvc, 0.0);
        tvc_grad_logit_rho_buf.assign(n_tvc, 0.0);
        tvc_grad_w_jg_buf.resize(n_tvc_times);
        tvc_d_buf.resize(n_tvc_times);

        tvc_grad_ws.grad_w = tvc_grad_w_buf.data();
        tvc_grad_ws.grad_log_tau = tvc_grad_log_tau_buf.data();
        tvc_grad_ws.grad_logit_rho = tvc_grad_logit_rho_buf.data();
        tvc_grad_ws.grad_w_jg = tvc_grad_w_jg_buf.data();
        tvc_grad_ws.d_buf = tvc_d_buf.data();
        tvc_grad_ws.n_w = n_w;
        tvc_grad_ws.n_tvc = n_tvc;

        tulpa_tvc::tvc_prior_gradients_ws(
            tvc_w_flat_buf.data(), data.tvc_data,
            tvc_tau_buf.data(), tvc_rho_buf.data(), tvc_grad_ws);

        // Add likelihood + prior to main gradient
        for (int k = 0; k < n_w; k++)
            grad[layout.tvc_w_start + k] += grad_tvc_w[k] + tvc_grad_w_buf[k];
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.log_tau_tvc_start + j] += tvc_grad_log_tau_buf[j];
            if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1)
                grad[layout.logit_rho_tvc_start + j] += tvc_grad_logit_rho_buf[j];
        }
    }

    // SVC spectral density gradients
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        for (int j = 0; j < n_svc; j++) {
            double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
            double lengthscale_j = std::exp(params[layout.log_phi_svc_start + j]);
            const double* beta_j = &params[layout.svc_w_start + j * svc_m_total];

            // Re-evaluate to cache sqrt_S
            tulpa_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                           data.svc_hsgp_data, svc_hsgp_ws);

            // Copy per-SVC-term grad_f
            for (int i = 0; i < N; i++) svc_hsgp_ws.grad_f[i] = grad_svc_f[j * N + i];

            double gls2 = 0.0, gll = 0.0;
            tulpa_hsgp::hsgp_compute_gradients_ws(
                beta_j, sigma2_j, lengthscale_j,
                data.svc_hsgp_data, svc_hsgp_ws, gls2, gll);

            grad[layout.log_sigma2_svc_start + j] += gls2;
            grad[layout.log_phi_svc_start + j] += gll;
            for (int m = 0; m < svc_m_total; m++)
                grad[layout.svc_w_start + j * svc_m_total + m] += svc_hsgp_ws.grad_beta_out[m];
        }
    }

    // Latent factor constraint chain rule
    if (K_latent > 0) {
        // N(0,1) prior on constrained factors
        for (int k = 0; k < K_latent; k++)
            for (int i = 0; i < N; i++)
                grad_factors_c[i * K_latent + k] -= factors_constrained[i * K_latent + k];

        // Chain rule: constrained -> raw
        if (data.latent_constraint == 0) {
            // Sum-to-zero: d/d(raw[i,k]) = d/d(constrained[i,k]) - mean(d/d(constrained[:,k]))
            for (int k = 0; k < K_latent; k++) {
                double sum_gc = 0.0;
                for (int i = 0; i < N; i++) sum_gc += grad_factors_c[i * K_latent + k];
                double mean_gc = sum_gc / N;
                for (int i = 0; i < N; i++)
                    grad[layout.latent_factor_start + i * K_latent + k] += grad_factors_c[i * K_latent + k] - mean_gc;
            }
        } else {
            for (int j = 0; j < N * K_latent; j++)
                grad[layout.latent_factor_start + j] += grad_factors_c[j];
        }
    }

    // ST interaction prior gradients
    if (has_st && data.st_is_hsgp) {
        // HSGP-ST: per-basis-function temporal GMRF with spectral precision scaling
        int M_st = data.st_hsgp_data.m_total;
        int T_st_h = data.spatiotemporal_data.n_times;
        double sigma2_st_h = std::exp(params[layout.log_sigma2_st_hsgp_idx]);
        double ls_st_h = std::exp(params[layout.log_lengthscale_st_hsgp_idx]);

        int rank_t = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T_st_h - 1) :
                     (data.spatiotemporal_data.temporal_type == TemporalType::RW2) ? (T_st_h - 2) : T_st_h;
        if (data.spatiotemporal_data.temporal_cyclic) rank_t = T_st_h;

        double grad_log_sigma2_st = 0.0;
        double grad_log_ls_st = 0.0;

        for (int j = 0; j < M_st; j++) {
            double omega_sq = data.st_hsgp_data.eigenvalues[j];
            double S_j = tulpa_hsgp::spectral_density_se(omega_sq, sigma2_st_h, ls_st_h);
            double S_j_safe = std::max(S_j, 1e-10);
            double prec_j = tau_st / S_j_safe;

            // Temporal GMRF stencil for basis function j
            const double* dj = &st_delta[j * T_st_h];
            double qf = 0.0;
            if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                for (int t = 0; t < T_st_h; t++) {
                    double g = 0.0;
                    if (t > 0) { g += prec_j * (dj[t-1] - dj[t]); qf += (dj[t] - dj[t-1]) * (dj[t] - dj[t-1]); }
                    if (t < T_st_h - 1) g += prec_j * (dj[t+1] - dj[t]);
                    grad[layout.st_delta_start + j * T_st_h + t] = grad_delta_lik[j * T_st_h + t] + g;
                }
            } else {
                // RW2 or other - use generic stencil
                for (int t = 0; t < T_st_h; t++)
                    grad[layout.st_delta_start + j * T_st_h + t] = grad_delta_lik[j * T_st_h + t];
            }

            // Sum-to-zero per basis function
            double sum_j = 0.0;
            for (int t = 0; t < T_st_h; t++) sum_j += dj[t];
            for (int t = 0; t < T_st_h; t++)
                grad[layout.st_delta_start + j * T_st_h + t] -= 0.001 * sum_j;

            // Tau gradient contribution from this basis function
            // d/d(log_tau) [0.5*rank*log(prec_j) - 0.5*prec_j*qf]
            //   = 0.5*rank - 0.5*prec_j*qf  (since d(prec_j)/d(log_tau) = prec_j)
            grad[layout.log_tau_st_idx] += 0.5 * rank_t - 0.5 * prec_j * qf;

            // Sigma2/lengthscale gradient through S_j
            // d/d(log_sigma2) S_j = S_j (SE kernel)
            // d/d(log_sigma2) [0.5*rank*log(tau/S_j) - 0.5*tau/S_j*qf]
            //   = -0.5*rank + 0.5*prec_j*qf  (times sigma2, cancels with chain rule)
            double dS_dsigma2 = S_j_safe / sigma2_st_h;
            double dLogPrior_dS = -0.5 * rank_t / S_j_safe + 0.5 * tau_st * qf / (S_j_safe * S_j_safe);
            grad_log_sigma2_st += dLogPrior_dS * dS_dsigma2 * sigma2_st_h;

            // d/d(log_ell) S_j = S_j * (1/ell - ell*omega_sq) * ell
            double dS_dl = S_j_safe * (1.0 / ls_st_h - ls_st_h * omega_sq);
            grad_log_ls_st += dLogPrior_dS * dS_dl * ls_st_h;
        }

        // HSGP-ST hyperparameter priors + likelihood chain rule
        // PC prior on sigma: d/d(log_sigma2) [-4.6*sigma + 0.5*log_sigma2] = -0.5*4.6*sigma + 0.5
        grad[layout.log_sigma2_st_hsgp_idx] = -0.5 * 4.6 * std::sqrt(sigma2_st_h) + 0.5 + grad_log_sigma2_st;
        // LogNormal(0,1) on lengthscale
        grad[layout.log_lengthscale_st_hsgp_idx] = -params[layout.log_lengthscale_st_hsgp_idx] + grad_log_ls_st;

    } else if (has_st) {
        const auto& st = data.spatiotemporal_data;
        if (st.type == STType::TYPE_I) {
            double qf = 0.0;
            for (int k = 0; k < ST_n; k++) {
                grad[layout.st_delta_start + k] = grad_delta_lik[k] - tau_st * st_delta[k];
                qf += st_delta[k] * st_delta[k];
            }
            grad[layout.log_tau_st_idx] += 0.5 * ST_n - 0.5 * tau_st * qf;
        } else if (st.type == STType::TYPE_II) {
            double total_qf = 0.0;
            for (int s = 0; s < S_st; s++) {
                const double* delta_s = &st_delta[s * T_st];
                if (st.temporal_type == TemporalType::RW1) {
                    double qf = 0.0;
                    for (int t = 0; t < T_st; t++) {
                        double g = 0.0;
                        if (t > 0) { g += tau_st * (delta_s[t-1] - delta_s[t]); qf += std::pow(delta_s[t] - delta_s[t-1], 2); }
                        if (t < T_st - 1) g += tau_st * (delta_s[t+1] - delta_s[t]);
                        grad[layout.st_delta_start + s * T_st + t] = grad_delta_lik[s * T_st + t] + g;
                    }
                    total_qf += qf;
                } else if (st.temporal_type == TemporalType::AR1) {
                    // AR1: precision Q = tau * tridiag(-rho, 1+rho^2, -rho), first/last diagonal = 1
                    // ST interaction doesn't have its own rho - use IID (rho=0) as fallback
                    // This is equivalent to tau * I for the ST delta prior
                    double qf = 0.0;
                    for (int t = 0; t < T_st; t++) {
                        grad[layout.st_delta_start + s * T_st + t] = grad_delta_lik[s * T_st + t] - tau_st * delta_s[t];
                        qf += delta_s[t] * delta_s[t];
                    }
                    total_qf += qf;
                } else {
                    // Fallback: at minimum write likelihood gradient
                    for (int t = 0; t < T_st; t++)
                        grad[layout.st_delta_start + s * T_st + t] = grad_delta_lik[s * T_st + t];
                }
            }
            int rank_per_unit = (st.temporal_type == TemporalType::RW1) ? (T_st - 1) :
                                (st.temporal_type == TemporalType::RW2) ? (T_st - 2) : T_st;
            grad[layout.log_tau_st_idx] += 0.5 * S_st * rank_per_unit - 0.5 * tau_st * total_qf;
        } else if (st.type == STType::TYPE_III) {
            double total_qf = 0.0;
            for (int t = 0; t < T_st; t++) {
                for (int s = 0; s < S_st; s++) {
                    double icar_grad = 0.0;
                    for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                        int j = st.adj_col_idx[idx] - 1;
                        icar_grad += tau_st * (st_delta[j * T_st + t] - st_delta[s * T_st + t]);
                    }
                    grad[layout.st_delta_start + s * T_st + t] = grad_delta_lik[s * T_st + t] + icar_grad;
                }
                for (int s = 0; s < S_st; s++) {
                    for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                        int j = st.adj_col_idx[idx] - 1;
                        if (j > s) total_qf += std::pow(st_delta[s * T_st + t] - st_delta[j * T_st + t], 2);
                    }
                }
            }
            grad[layout.log_tau_st_idx] += 0.5 * T_st * (S_st - 1) - 0.5 * tau_st * total_qf;
        } else if (st.type == STType::TYPE_IV) {
            // Kronecker: Q_delta = Q_s - Q_t (analytical gradient, same as specialized ST function)
            const double* stencil_input = st_use_nc ? z_or_delta_st : st_delta;
            double inv_scale_nc = st_use_nc ? (1.0 / std::sqrt(tau_st)) : 1.0;

            // Step 1: Apply temporal stencil: v[s,t] = (Q_t * input[s,:])_t
            static thread_local std::vector<double> v_kron;
            v_kron.assign(S_st * T_st, 0.0);
            if (st.temporal_type == TemporalType::RW1) {
                for (int s = 0; s < S_st; s++) {
                    for (int t = 0; t < T_st; t++) {
                        double qt_val = 0.0;
                        int n_t_neigh = 0;
                        if (t > 0) { qt_val -= stencil_input[s * T_st + t - 1]; n_t_neigh++; }
                        if (t < T_st - 1) { qt_val -= stencil_input[s * T_st + t + 1]; n_t_neigh++; }
                        qt_val += n_t_neigh * stencil_input[s * T_st + t];
                        v_kron[s * T_st + t] = qt_val;
                    }
                }
            } else if (st.temporal_type == TemporalType::RW2) {
                for (int s = 0; s < S_st; s++) {
                    const double* d_s = &stencil_input[s * T_st];
                    double* v_s = &v_kron[s * T_st];
                    if (T_st >= 3) {
                        const int n_d2 = T_st - 2;
                        double d2_stack[64];
                        double* d2 = (n_d2 <= 64) ? d2_stack : new double[n_d2];
                        for (int k = 0; k < n_d2; k++) d2[k] = d_s[k] - 2.0 * d_s[k + 1] + d_s[k + 2];
                        v_s[0] = d2[0];
                        v_s[1] = -2.0 * d2[0];
                        if (n_d2 > 1) v_s[1] += d2[1];
                        for (int t = 2; t < T_st - 2; t++) v_s[t] = d2[t - 2] - 2.0 * d2[t - 1] + d2[t];
                        if (T_st >= 4) v_s[T_st - 2] = d2[n_d2 - 2] - 2.0 * d2[n_d2 - 1];
                        else v_s[T_st - 2] = -2.0 * d2[0];
                        v_s[T_st - 1] = d2[n_d2 - 1];
                        if (n_d2 > 64) delete[] d2;
                    }
                }
            }

            // Step 2: Apply spatial ICAR stencil to v: (Q_s - Q_t) * input
            double total_qf = 0.0;
            for (int s = 0; s < S_st; s++) {
                for (int t = 0; t < T_st; t++) {
                    double qs_v = 0.0;
                    for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                        int j = st.adj_col_idx[idx] - 1;
                        qs_v -= v_kron[j * T_st + t];
                    }
                    int n_neigh = st.adj_row_ptr[s + 1] - st.adj_row_ptr[s];
                    qs_v += n_neigh * v_kron[s * T_st + t];

                    if (st_use_nc) {
                        grad[layout.st_delta_start + s * T_st + t] =
                            grad_delta_lik[s * T_st + t] * inv_scale_nc - qs_v;
                    } else {
                        grad[layout.st_delta_start + s * T_st + t] =
                            grad_delta_lik[s * T_st + t] - tau_st * qs_v;
                    }
                    total_qf += stencil_input[s * T_st + t] * qs_v;
                }
            }

            int rank_space = S_st - 1;
            int rank_time = (st.temporal_type == TemporalType::RW1) ? (T_st - 1) :
                            (st.temporal_type == TemporalType::RW2) ? (T_st - 2) : T_st;
            if (st.temporal_cyclic) rank_time = T_st;
            int total_rank = rank_space * rank_time;

            if (st_use_nc) {
                double lik_tau_grad = 0.0;
                for (int k = 0; k < ST_n; k++) lik_tau_grad += grad_delta_lik[k] * st_delta[k];
                grad[layout.log_tau_st_idx] += 0.5 * (total_rank - ST_n) - 0.5 * lik_tau_grad;
            } else {
                grad[layout.log_tau_st_idx] += 0.5 * total_rank - 0.5 * tau_st * total_qf;
            }
        }

        // Sum-to-zero penalty on ST delta
        double lambda_stz = 0.001;
        for (int t = 0; t < T_st; t++) {
            double row_sum = 0.0;
            for (int s = 0; s < S_st; s++) row_sum += st_delta[s * T_st + t];
            for (int s = 0; s < S_st; s++)
                grad[layout.st_delta_start + s * T_st + t] -= lambda_stz * row_sum;
        }
        for (int s = 0; s < S_st; s++) {
            double col_sum = 0.0;
            for (int t = 0; t < T_st; t++) col_sum += st_delta[s * T_st + t];
            for (int t = 0; t < T_st; t++)
                grad[layout.st_delta_start + s * T_st + t] -= lambda_stz * col_sum;
        }
    }

    // NC RE chain rule (simple intercepts)
    if (!has_slopes) re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // NC slopes chain rule - mirrors compute_gradient_analytical write-back
    if (has_slopes) {
        for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
            int n_groups = data.re_n_groups_multi[t_re];
            int n_coefs = layout.re_n_coefs_multi[t_re];
            int re_start_t = layout.re_start_multi[t_re];
            bool is_corr_nc = !nc_L_flats.empty() && t_re < (int)nc_L_flats.size() && !nc_L_flats[t_re].empty();

            if (is_corr_nc) {
                // Correlated NC: chain rule from dLL/d(re_nc) back to
                // (z, log_sigma, raw_chol). grad_z and grad_raw slots are
                // contiguous in grad; log_sigma is scattered, so use temp.
                const auto& L_flat = nc_L_flats[t_re];
                const auto& sigmas = nc_sigmas_vec[t_re];
                int chol_start = layout.chol_re_start_multi[t_re];
                std::vector<double> g_log_sigma(n_coefs, 0.0);

                tulpa::chol_nc_chain_rule_add(
                    L_flat.data(), n_coefs, sigmas.data(),
                    &params[re_start_t], &params[chol_start],
                    &re_nc_flat_c[re_start_t], n_groups,
                    grad_re_slopes_lik[t_re].data(),
                    &grad[re_start_t],
                    g_log_sigma.data(),
                    &grad[chol_start]);

                for (int c = 0; c < n_coefs; c++) {
                    grad[layout.log_sigma_re_slopes[t_re][c]] += g_log_sigma[c];
                }
            } else if (slopes_nc) {
                // Uncorrelated NC: chain rule re = sigma * z
                for (int g = 0; g < n_groups; g++) {
                    for (int c = 0; c < n_coefs; c++) {
                        int idx = re_start_t + g * n_coefs + c;
                        double z_gc = params[idx];
                        double sigma_c = std::exp(params[layout.log_sigma_re_slopes[t_re][c]]);
                        double lik_grad = grad_re_slopes_lik[t_re][g * n_coefs + c];
                        grad[idx] += sigma_c * lik_grad;
                        grad[layout.log_sigma_re_slopes[t_re][c]] += z_gc * lik_grad * sigma_c;
                    }
                }
            } else {
                // Centered: direct
                for (int g = 0; g < n_groups; g++)
                    for (int c = 0; c < n_coefs; c++)
                        grad[re_start_t + g * n_coefs + c] += grad_re_slopes_lik[t_re][g * n_coefs + c];
            }
        }
    }

    // Fused log-posterior
    if (fuse_lp && !layout.has_zi) {
        *log_post_out = compute_log_post(params, data, layout);
    }
