// log_post_impl_priors_st_zi_block.h
// Function-body fragment of compute_log_post_impl<T> in log_post_impl.h.
// Included via #include directly inside the function body — relies on the
// surrounding lexical scope (params, data, layout, log_post, T, beta_*,
// phi_*, re_vals, re_term_offsets, etc.). NOT standalone-compilable.
// No header guard: each fragment is included exactly once per umbrella.
// Spatiotemporal interaction priors + zero-inflation / one-inflation priors.

    // Spatiotemporal interaction effects
    std::vector<T> st_delta_impl;

    if (layout.has_spatiotemporal && data.spatiotemporal_data.type != STType::NONE) {
        // Extract precision parameter
        T log_tau_st = params[layout.log_tau_st_idx];
        T tau_st = safe_exp(log_tau_st);

        // PC prior on tau (exponential on sigma = 1/sqrt(tau))
        T sigma_st = T(1.0) / safe_sqrt(tau_st);
        T lambda_st = T(-std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U);
        log_post = log_post + safe_log(lambda_st) - lambda_st * sigma_st
                 - safe_log(T(2.0) * sigma_st);
        log_post = log_post + log_tau_st;  // Jacobian for log transform

        // AR1 rho parameter
        T rho_st = T(0.0);
        if (layout.logit_rho_st_idx >= 0) {
            T logit_rho_st = params[layout.logit_rho_st_idx];
            T u_st = inv_logit(logit_rho_st);
            rho_st = T(2.0) * u_st - T(1.0);  // Map to (-1, 1)

            // Uniform(-1, 1) prior on rho
            // Jacobian for logit((rho+1)/2) transform
            log_post = log_post + safe_log(u_st) + safe_log(T(1.0) - u_st);
        }

        // GP range parameters
        T phi_st_space = T(1.0);
        T phi_st_time = T(1.0);
        if (layout.is_st_gp) {
            T log_phi_space = params[layout.log_phi_st_space_idx];
            T log_phi_time = params[layout.log_phi_st_time_idx];
            phi_st_space = safe_exp(log_phi_space);
            phi_st_time = safe_exp(log_phi_time);

            // Uniform prior within bounds
            double phi_space_val = get_value(phi_st_space);
            if (phi_space_val < data.st_phi_space_prior_lower ||
                phi_space_val > data.st_phi_space_prior_upper) {
                return T(-INFINITY);
            }
            double phi_time_val = get_value(phi_st_time);
            if (phi_time_val < data.st_phi_time_prior_lower ||
                phi_time_val > data.st_phi_time_prior_upper) {
                return T(-INFINITY);
            }
            log_post = log_post + log_phi_space + log_phi_time;  // Jacobians
        }

        // Extract delta parameters
        int n_st_params = layout.st_delta_end - layout.st_delta_start;
        st_delta_impl.resize(n_st_params);
        for (int k = 0; k < n_st_params; k++) {
            st_delta_impl[k] = params[layout.st_delta_start + k];
        }

        int S = data.spatiotemporal_data.n_spatial;
        int T_st = data.spatiotemporal_data.n_times;

        // NC reparameterization for Type IV
        const bool st_use_nc = (data.st_parameterization == 1 &&
                                data.spatiotemporal_data.type == STType::TYPE_IV);

        if (st_use_nc) {
            // Forward transform: delta = z / sqrt(tau_st)
            T inv_scale = T(1.0) / safe_sqrt(tau_st);
            std::vector<T> st_delta_nc(n_st_params);
            for (int k = 0; k < n_st_params; k++) {
                st_delta_nc[k] = st_delta_impl[k] * inv_scale;
            }

            // NC prior: -0.5 * z^T (Q_s ⊗ Q_t) z  (tau-free GMRF)
            // Type IV Kronecker quadratic form on z (with tau=1)
            // Compute per-spatial-unit temporal GMRF quadratic form
            T nc_quad = T(0.0);
            for (int s = 0; s < S; s++) {
                // Extract temporal series for this spatial unit
                // Apply spatial neighbor structure
                T spatial_quad_s = T(0.0);
                if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                    for (int t = 1; t < T_st; t++) {
                        T diff = st_delta_impl[s * T_st + t] - st_delta_impl[s * T_st + t - 1];
                        spatial_quad_s = spatial_quad_s + diff * diff;
                    }
                } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                    for (int t = 2; t < T_st; t++) {
                        T d2 = st_delta_impl[s * T_st + t]
                             - T(2.0) * st_delta_impl[s * T_st + t - 1]
                             + st_delta_impl[s * T_st + t - 2];
                        spatial_quad_s = spatial_quad_s + d2 * d2;
                    }
                }
                // Now multiply by spatial structure (neighbor weights)
                int n_neigh = data.spatiotemporal_data.n_neighbors.empty() ? 0
                    : data.spatiotemporal_data.n_neighbors[s];
                nc_quad = nc_quad + T(n_neigh) * spatial_quad_s;
                // Off-diagonal: -2 * (neighbor pairs)
                if (!data.spatiotemporal_data.adj_row_ptr.empty()) {
                    int row_start_s = data.spatiotemporal_data.adj_row_ptr[s];
                    int row_end_s = data.spatiotemporal_data.adj_row_ptr[s + 1];
                    for (int jj = row_start_s; jj < row_end_s; jj++) {
                        int s2 = data.spatiotemporal_data.adj_col_idx[jj] - 1;
                        if (s2 > s) {
                            // Temporal quadratic form between units s and s2
                            T cross = T(0.0);
                            if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                                for (int t = 1; t < T_st; t++) {
                                    T diff_s = st_delta_impl[s * T_st + t] - st_delta_impl[s * T_st + t - 1];
                                    T diff_s2 = st_delta_impl[s2 * T_st + t] - st_delta_impl[s2 * T_st + t - 1];
                                    cross = cross + diff_s * diff_s2;
                                }
                            } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                                for (int t = 2; t < T_st; t++) {
                                    T d2_s = st_delta_impl[s * T_st + t]
                                           - T(2.0) * st_delta_impl[s * T_st + t - 1]
                                           + st_delta_impl[s * T_st + t - 2];
                                    T d2_s2 = st_delta_impl[s2 * T_st + t]
                                            - T(2.0) * st_delta_impl[s2 * T_st + t - 1]
                                            + st_delta_impl[s2 * T_st + t - 2];
                                    cross = cross + d2_s * d2_s2;
                                }
                            }
                            nc_quad = nc_quad - T(2.0) * cross;
                        }
                    }
                }
            }
            log_post = log_post - T(0.5) * nc_quad;

            // Rank term with actual tau and Jacobian correction
            int rank_space = S - 1;
            int rank_time = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T_st - 1) : (T_st - 2);
            if (data.spatiotemporal_data.temporal_cyclic) rank_time = T_st;
            int total_rank = rank_space * rank_time;
            int ST_total = S * T_st;
            log_post = log_post + T(0.5 * (total_rank - ST_total)) * safe_log(tau_st);

            // Sum-to-zero on reconstructed delta
            T sum_s = T(0.0), sum_t = T(0.0);
            for (int s = 0; s < S; s++) {
                T row_sum = T(0.0);
                for (int t = 0; t < T_st; t++) {
                    row_sum = row_sum + st_delta_nc[s * T_st + t];
                }
                sum_s = sum_s + row_sum * row_sum;
            }
            for (int t = 0; t < T_st; t++) {
                T col_sum = T(0.0);
                for (int s = 0; s < S; s++) {
                    col_sum = col_sum + st_delta_nc[s * T_st + t];
                }
                sum_t = sum_t + col_sum * col_sum;
            }
            log_post = log_post - T(0.5) * T(0.001) * (sum_s + sum_t);

            // Replace st_delta_impl with reconstructed delta for observation loop
            st_delta_impl = std::move(st_delta_nc);

        } else if (data.st_is_hsgp) {
            // HSGP-ST: spectral basis interaction (centered)
            int M = data.st_hsgp_data.m_total;

            // HSGP-ST hyperparameters
            T sigma2_st_hsgp = safe_exp(params[layout.log_sigma2_st_hsgp_idx]);
            T lengthscale_st_hsgp = safe_exp(params[layout.log_lengthscale_st_hsgp_idx]);

            // PC prior on sigma_st_hsgp: rate=4.6
            T sigma_st_h = safe_sqrt(sigma2_st_hsgp);
            log_post = log_post - T(4.6) * sigma_st_h + T(0.5) * params[layout.log_sigma2_st_hsgp_idx];

            // LogNormal(0,1) on lengthscale
            T log_ls_st = params[layout.log_lengthscale_st_hsgp_idx];
            log_post = log_post - T(0.5) * log_ls_st * log_ls_st;

            // Per-basis-function temporal GMRF prior
            int rank_t = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T_st - 1) :
                         (data.spatiotemporal_data.temporal_type == TemporalType::RW2) ? (T_st - 2) : T_st;
            if (data.spatiotemporal_data.temporal_cyclic) rank_t = T_st;

            for (int j = 0; j < M; j++) {
                double omega_sq = data.st_hsgp_data.eigenvalues[j];
                T S_j = sigma2_st_hsgp * T(std::sqrt(2.0 * M_PI)) * lengthscale_st_hsgp
                    * safe_exp(T(-0.5) * lengthscale_st_hsgp * lengthscale_st_hsgp * T(omega_sq));
                T prec_j = tau_st / safe_max(S_j, T(1e-10));

                // GMRF quadratic form: -0.5 * prec_j * delta_j' Q_t delta_j
                T qf = T(0.0);
                if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                    for (int t = 1; t < T_st; t++) {
                        T diff = st_delta_impl[j * T_st + t] - st_delta_impl[j * T_st + t - 1];
                        qf = qf + diff * diff;
                    }
                } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                    for (int t = 2; t < T_st; t++) {
                        T d2 = st_delta_impl[j * T_st + t]
                             - T(2.0) * st_delta_impl[j * T_st + t - 1]
                             + st_delta_impl[j * T_st + t - 2];
                        qf = qf + d2 * d2;
                    }
                }
                log_post = log_post + T(0.5 * rank_t) * safe_log(prec_j)
                         - T(0.5) * prec_j * qf;

                // Soft sum-to-zero per basis function
                T sum_j = T(0.0);
                for (int t = 0; t < T_st; t++) sum_j = sum_j + st_delta_impl[j * T_st + t];
                log_post = log_post - T(0.5) * T(0.001) * sum_j * sum_j;
            }

        } else {
            // Centered parameterization (ICAR/BYM2 spatial: Types I-IV)
            if (data.spatiotemporal_data.type == STType::TYPE_I) {
                // IID: delta[s,t] ~ N(0, 1/tau)
                T quad = T(0.0);
                for (int k = 0; k < n_st_params; k++) {
                    quad = quad + st_delta_impl[k] * st_delta_impl[k];
                }
                log_post = log_post + T(0.5 * n_st_params) * safe_log(tau_st)
                         - T(0.5) * tau_st * quad;

            } else if (data.spatiotemporal_data.type == STType::TYPE_II) {
                // Structured time at each location
                for (int s = 0; s < S; s++) {
                    T quad = T(0.0);
                    if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                        for (int t = 1; t < T_st; t++) {
                            T diff = st_delta_impl[s * T_st + t] - st_delta_impl[s * T_st + t - 1];
                            quad = quad + diff * diff;
                        }
                        if (data.spatiotemporal_data.temporal_cyclic) {
                            T diff_c = st_delta_impl[s * T_st] - st_delta_impl[s * T_st + T_st - 1];
                            quad = quad + diff_c * diff_c;
                        }
                        int rank = data.spatiotemporal_data.temporal_cyclic ? T_st : T_st - 1;
                        log_post = log_post + T(0.5 * rank) * safe_log(tau_st)
                                 - T(0.5) * tau_st * quad;
                    } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                        for (int t = 2; t < T_st; t++) {
                            T d2 = st_delta_impl[s * T_st + t]
                                 - T(2.0) * st_delta_impl[s * T_st + t - 1]
                                 + st_delta_impl[s * T_st + t - 2];
                            quad = quad + d2 * d2;
                        }
                        if (data.spatiotemporal_data.temporal_cyclic && T_st >= 3) {
                            T d2_a = st_delta_impl[s * T_st + T_st - 2]
                                   - T(2.0) * st_delta_impl[s * T_st + T_st - 1]
                                   + st_delta_impl[s * T_st];
                            T d2_b = st_delta_impl[s * T_st + T_st - 1]
                                   - T(2.0) * st_delta_impl[s * T_st]
                                   + st_delta_impl[s * T_st + 1];
                            quad = quad + d2_a * d2_a + d2_b * d2_b;
                        }
                        int rank = data.spatiotemporal_data.temporal_cyclic ? T_st : T_st - 2;
                        log_post = log_post + T(0.5 * rank) * safe_log(tau_st)
                                 - T(0.5) * tau_st * quad;
                    }
                }

            } else if (data.spatiotemporal_data.type == STType::TYPE_III) {
                // Structured space at each time point (ICAR)
                int rank_s = S - 1;
                for (int t = 0; t < T_st; t++) {
                    // Compute ICAR quadratic form for spatial field at time t
                    T quad = T(0.0);
                    for (int s = 0; s < S; s++) {
                        T delta_st = st_delta_impl[s * T_st + t];
                        int n_neigh = data.spatiotemporal_data.n_neighbors.empty() ? 0
                            : data.spatiotemporal_data.n_neighbors[s];
                        quad = quad + T(n_neigh) * delta_st * delta_st;
                        if (!data.spatiotemporal_data.adj_row_ptr.empty()) {
                            int row_start_s = data.spatiotemporal_data.adj_row_ptr[s];
                            int row_end_s = data.spatiotemporal_data.adj_row_ptr[s + 1];
                            for (int jj = row_start_s; jj < row_end_s; jj++) {
                                int s2 = data.spatiotemporal_data.adj_col_idx[jj] - 1;
                                if (s2 > s) {
                                    T delta_s2t = st_delta_impl[s2 * T_st + t];
                                    quad = quad - T(2.0) * delta_st * delta_s2t;
                                }
                            }
                        }
                    }
                    log_post = log_post + T(0.5 * rank_s) * safe_log(tau_st)
                             - T(0.5) * tau_st * quad;
                }

            } else if (data.spatiotemporal_data.type == STType::TYPE_IV) {
                // Kronecker: Q_delta = Q_s ⊗ Q_t
                // Compute using per-spatial-unit temporal GMRF
                for (int s = 0; s < S; s++) {
                    T spatial_quad_s = T(0.0);
                    if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                        for (int t = 1; t < T_st; t++) {
                            T diff = st_delta_impl[s * T_st + t] - st_delta_impl[s * T_st + t - 1];
                            spatial_quad_s = spatial_quad_s + diff * diff;
                        }
                    } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                        for (int t = 2; t < T_st; t++) {
                            T d2 = st_delta_impl[s * T_st + t]
                                 - T(2.0) * st_delta_impl[s * T_st + t - 1]
                                 + st_delta_impl[s * T_st + t - 2];
                            spatial_quad_s = spatial_quad_s + d2 * d2;
                        }
                    }
                    int n_neigh = data.spatiotemporal_data.n_neighbors.empty() ? 0
                        : data.spatiotemporal_data.n_neighbors[s];
                    T weighted_quad = T(n_neigh) * spatial_quad_s;

                    if (!data.spatiotemporal_data.adj_row_ptr.empty()) {
                        int row_start_s = data.spatiotemporal_data.adj_row_ptr[s];
                        int row_end_s = data.spatiotemporal_data.adj_row_ptr[s + 1];
                        for (int jj = row_start_s; jj < row_end_s; jj++) {
                            int s2 = data.spatiotemporal_data.adj_col_idx[jj] - 1;
                            if (s2 > s) {
                                T cross = T(0.0);
                                if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                                    for (int t = 1; t < T_st; t++) {
                                        T diff_s_t = st_delta_impl[s * T_st + t] - st_delta_impl[s * T_st + t - 1];
                                        T diff_s2_t = st_delta_impl[s2 * T_st + t] - st_delta_impl[s2 * T_st + t - 1];
                                        cross = cross + diff_s_t * diff_s2_t;
                                    }
                                } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
                                    for (int t = 2; t < T_st; t++) {
                                        T d2_s = st_delta_impl[s * T_st + t]
                                               - T(2.0) * st_delta_impl[s * T_st + t - 1]
                                               + st_delta_impl[s * T_st + t - 2];
                                        T d2_s2 = st_delta_impl[s2 * T_st + t]
                                                - T(2.0) * st_delta_impl[s2 * T_st + t - 1]
                                                + st_delta_impl[s2 * T_st + t - 2];
                                        cross = cross + d2_s * d2_s2;
                                    }
                                }
                                weighted_quad = weighted_quad - T(2.0) * cross;
                            }
                        }
                    }
                    log_post = log_post - T(0.5) * tau_st * weighted_quad;
                }
                // Rank terms
                int rank_space = S - 1;
                int rank_time = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T_st - 1) : (T_st - 2);
                if (data.spatiotemporal_data.temporal_cyclic) rank_time = T_st;
                int total_rank = rank_space * rank_time;
                log_post = log_post + T(0.5 * total_rank) * safe_log(tau_st);
            }

            // Soft sum-to-zero constraint
            T sum_s = T(0.0), sum_t = T(0.0);
            for (int s = 0; s < S; s++) {
                T row_sum = T(0.0);
                for (int t = 0; t < T_st; t++) {
                    row_sum = row_sum + st_delta_impl[s * T_st + t];
                }
                sum_s = sum_s + row_sum * row_sum;
            }
            for (int t = 0; t < T_st; t++) {
                T col_sum = T(0.0);
                for (int s = 0; s < S; s++) {
                    col_sum = col_sum + st_delta_impl[s * T_st + t];
                }
                sum_t = sum_t + col_sum * col_sum;
            }
            log_post = log_post - T(0.5) * T(0.001) * (sum_s + sum_t);
        }
    }

    // Zero-inflation / One-inflation parameters
    std::vector<T> beta_zi;
    std::vector<T> beta_oi;

    if (layout.has_zi && data.p_zi > 0) {
        beta_zi.resize(data.p_zi);
        for (int j = 0; j < data.p_zi; j++) {
            beta_zi[j] = params[layout.beta_zi_start + j];
        }
        // Prior on beta_zi: N(0, zi_prior_sd^2)
        double tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd);
        for (int j = 0; j < data.p_zi; j++) {
            log_post = log_post + log_prior_normal(beta_zi[j], tau_zi);
        }
    }

    if (layout.has_oi && data.p_oi > 0) {
        beta_oi.resize(data.p_oi);
        for (int j = 0; j < data.p_oi; j++) {
            beta_oi[j] = params[layout.beta_oi_start + j];
        }
        // Prior on beta_oi: N(0, oi_prior_sd^2)
        double tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd);
        for (int j = 0; j < data.p_oi; j++) {
            log_post = log_post + log_prior_normal(beta_oi[j], tau_oi);
        }
    }


