// log_post_impl_priors_temporal_block.h
// Function-body fragment of compute_log_post_impl<T> in log_post_impl.h.
// Included via #include directly inside the function body — relies on the
// surrounding lexical scope (params, data, layout, log_post, T, beta_*,
// phi_*, re_vals, re_term_offsets, etc.). NOT standalone-compilable.
// No header guard: each fragment is included exactly once per umbrella.
// Temporal priors (RW1, RW2, AR1, GP) + multiscale temporal.

    // Temporal priors
    if (layout.has_temporal) {
        int T_times = data.n_times;

        if (layout.is_temporal_gp) {
            // Temporal GP: PC prior on sigma2, logit-bounded phi (lengthscale)
            T log_sigma2 = params[layout.log_sigma2_temporal_gp_idx];

            // PC prior on sigma2 (favor smaller variance)
            double rate = -std::log(data.temporal_gp_sigma2_prior_alpha) / data.temporal_gp_sigma2_prior_U;
            T sigma_gp = safe_sqrt(sigma2_temporal_gp);
            log_post = log_post + T(std::log(rate)) - T(rate) * sigma_gp - safe_log(T(2.0) * sigma_gp);
            log_post = log_post + log_sigma2;  // Jacobian for log transform

            // Uniform prior on phi: logit-bounded parameterization guarantees bounds
            // Jacobian: log(phi - lower) + log(upper - phi) - log(range)
            double phi_lower_pr = data.temporal_gp_phi_prior_lower;
            double phi_upper_pr = data.temporal_gp_phi_prior_upper;
            double phi_range_pr = phi_upper_pr - phi_lower_pr;
            log_post = log_post + safe_log(phi_temporal_gp - T(phi_lower_pr))
                     + safe_log(T(phi_upper_pr) - phi_temporal_gp)
                     - T(std::log(phi_range_pr));

            const bool use_nc = (data.temporal_gp_parameterization == 1);

            // Precompute shared rho[t] and derived quantities once (same dt for all groups)
                std::vector<T> rho_shared(T_times > 1 ? T_times - 1 : 0);
                std::vector<T> log_one_minus_rho2_shared(T_times > 1 ? T_times - 1 : 0);
                std::vector<T> a_shared(T_times > 1 ? T_times - 1 : 0);
                T sigma_t = safe_sqrt(sigma2_temporal_gp);
                for (int t = 1; t < T_times; t++) {
                    double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
                    rho_shared[t - 1] = safe_exp(T(-dt) / phi_temporal_gp);
                    T one_minus_rho2 = T(1.0) - rho_shared[t - 1] * rho_shared[t - 1];
                    T one_minus_rho2_safe = safe_max(one_minus_rho2, T(1e-10));
                    log_one_minus_rho2_shared[t - 1] = safe_log(one_minus_rho2_safe);
                    a_shared[t - 1] = sigma_t * safe_sqrt(one_minus_rho2_safe);
                }

            if (use_nc) {
                // Non-centered: params store z ~ N(0,1)
                // Prior: z ~ N(0, I) for each temporal effect
                int n_temporal = layout.temporal_end - layout.temporal_start;
                for (int t = 0; t < n_temporal; t++) {
                    log_post = log_post - T(0.5) * phi_temporal[t] * phi_temporal[t];
                }

                // Jacobian of transform f = g(z, sigma2, phi):
                // log|det(df/dz)| = T*log(sigma) + 0.5*sum_{t>=1} log(1 - rho_t^2) per group
                T log_jac_per_group = T(T_times) * safe_log(sigma_t);
                for (int t = 1; t < T_times; t++) {
                    log_jac_per_group = log_jac_per_group + T(0.5) * log_one_minus_rho2_shared[t - 1];
                }
                log_post = log_post + T(data.n_temporal_groups) * log_jac_per_group;

                // Forward transform z -> f: overwrite phi_temporal for use in obs loop
                // f[0] = sigma * z[0]
                // f[t] = rho_t * f[t-1] + a_t * z[t]
                std::vector<T> f_reconstructed(n_temporal);
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    int off = g * T_times;
                    f_reconstructed[off] = sigma_t * phi_temporal[off];
                    for (int t = 1; t < T_times; t++) {
                        f_reconstructed[off + t] = rho_shared[t - 1] * f_reconstructed[off + t - 1] + a_shared[t - 1] * phi_temporal[off + t];
                    }
                }
                // Replace phi_temporal with reconstructed f for observation loop
                phi_temporal = std::move(f_reconstructed);
            } else {
                // Centered: GP log-likelihood using state-space representation
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    T f0 = phi_temporal[g * T_times];
                    log_post = log_post - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2_temporal_gp);
                    log_post = log_post - T(0.5) * f0 * f0 / sigma2_temporal_gp;

                    for (int t = 1; t < T_times; t++) {
                        T f_prev = phi_temporal[g * T_times + t - 1];
                        T f_curr = phi_temporal[g * T_times + t];

                        T cond_var = sigma2_temporal_gp * (T(1.0) - rho_shared[t - 1] * rho_shared[t - 1]);
                        T cond_var_safe = safe_max(cond_var, T(1e-10));
                        T cond_mean = rho_shared[t - 1] * f_prev;
                        T resid = f_curr - cond_mean;

                        log_post = log_post - T(0.5) * safe_log(T(2.0 * M_PI) * cond_var_safe);
                        log_post = log_post - T(0.5) * resid * resid / cond_var_safe;
                    }
                }
            }
        } else {
            // RW1/RW2/AR1: tau-based parameterization
            T log_tau = params[layout.log_tau_temporal_idx];
            log_post = log_post + log_prior_gamma(log_tau, data.tau_temporal_shape, data.tau_temporal_rate);

            if (data.temporal_type == TemporalType::RW1) {
                // RW1: sum of (phi[t] - phi[t-1])^2
                T quad_form = T(0.0);
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    for (int t = 1; t < T_times; t++) {
                        T diff = phi_temporal[g * T_times + t] - phi_temporal[g * T_times + t - 1];
                        quad_form = quad_form + diff * diff;
                    }
                    // Cyclic: add wrap-around edge (phi[0] - phi[T-1])
                    if (data.temporal_cyclic) {
                        T diff_cyclic = phi_temporal[g * T_times] - phi_temporal[g * T_times + T_times - 1];
                        quad_form = quad_form + diff_cyclic * diff_cyclic;
                    }
                }
                // Rank: T for cyclic, T-1 for non-cyclic
                int rank_rw1 = data.temporal_cyclic ? T_times : T_times - 1;
                log_post = log_post + T(0.5 * rank_rw1 * data.n_temporal_groups) * log_tau;
                log_post = log_post - T(0.5) * tau_temporal * quad_form;

            } else if (data.temporal_type == TemporalType::RW2) {
                // RW2: sum of (phi[t] - 2*phi[t-1] + phi[t-2])^2
                T quad_form = T(0.0);
                for (int g = 0; g < data.n_temporal_groups; g++) {
                    for (int t = 2; t < T_times; t++) {
                        T diff = phi_temporal[g * T_times + t]
                               - T(2.0) * phi_temporal[g * T_times + t - 1]
                               + phi_temporal[g * T_times + t - 2];
                        quad_form = quad_form + diff * diff;
                    }
                    // Cyclic: add wrap-around second-order differences
                    if (data.temporal_cyclic && T_times >= 3) {
                        T d2_a = phi_temporal[g * T_times + T_times - 2]
                               - T(2.0) * phi_temporal[g * T_times + T_times - 1]
                               + phi_temporal[g * T_times];
                        T d2_b = phi_temporal[g * T_times + T_times - 1]
                               - T(2.0) * phi_temporal[g * T_times]
                               + phi_temporal[g * T_times + 1];
                        quad_form = quad_form + d2_a * d2_a + d2_b * d2_b;
                    }
                }
                // Rank: T for cyclic, T-2 for non-cyclic
                int rank_rw2 = data.temporal_cyclic ? T_times : T_times - 2;
                log_post = log_post + T(0.5 * rank_rw2 * data.n_temporal_groups) * log_tau;
                log_post = log_post - T(0.5) * tau_temporal * quad_form;

            } else if (data.temporal_type == TemporalType::AR1) {
                // AR1: phi[t] | phi[t-1] ~ N(rho * phi[t-1], 1/tau)
                // Uniform(0,1) prior on rho with logit Jacobian: log(rho) + log(1-rho)
                log_post = log_post + safe_log(rho_ar1) + safe_log(T(1.0) - rho_ar1);

                T sigma2_ar1 = T(1.0) / tau_temporal;
                T one_minus_rho2 = T(1.0) - rho_ar1 * rho_ar1;

                for (int g = 0; g < data.n_temporal_groups; g++) {
                    // First time point: phi[0] ~ N(0, sigma^2/(1-rho^2))
                    T var_stationary = sigma2_ar1 / one_minus_rho2;
                    log_post = log_post - T(0.5) * phi_temporal[g * T_times] * phi_temporal[g * T_times] / var_stationary;
                    // Normalization: -0.5 * log(2*pi*var_stationary)
                    log_post = log_post - T(0.5) * safe_log(T(2.0 * M_PI) * var_stationary);

                    // Subsequent: phi[t] | phi[t-1] ~ N(rho*phi[t-1], sigma^2)
                    T log_norm_cond = T(-0.5) * safe_log(T(2.0 * M_PI) * sigma2_ar1);
                    for (int t = 1; t < T_times; t++) {
                        T resid = phi_temporal[g * T_times + t] - rho_ar1 * phi_temporal[g * T_times + t - 1];
                        log_post = log_post - T(0.5) * tau_temporal * resid * resid;
                        log_post = log_post + log_norm_cond;
                    }
                }
            }
        }
    }

    // Multiscale temporal parameters and priors
    std::vector<T> ms_trend;
    std::vector<T> ms_seasonal;
    std::vector<T> ms_short_term;
    std::vector<T> ms_temporal_eta;
    T ms_sigma2_trend = T(1.0);
    T ms_sigma2_seasonal = T(1.0);
    T ms_sigma2_short = T(1.0);
    T ms_rho_short = T(0.5);

    if (layout.has_multiscale_temporal) {
        const auto& ms_data = data.multiscale_temporal_data;

        // Trend component
        if (layout.log_sigma2_trend_idx >= 0) {
            T log_sigma2_trend = params[layout.log_sigma2_trend_idx];
            ms_sigma2_trend = safe_exp(log_sigma2_trend);

            int n_trend = layout.trend_end - layout.trend_start;
            ms_trend.resize(n_trend);
            for (int t = 0; t < n_trend; t++) {
                ms_trend[t] = params[layout.trend_start + t];
            }

            // PC prior on sigma2_trend + Jacobian for log transform
            log_post = log_post + tulpa_multiscale_ad::log_prior_sigma2_temporal_pc(
                ms_sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha);
            log_post = log_post + log_sigma2_trend;  // Jacobian
        }

        // Seasonal component
        if (layout.log_sigma2_seasonal_idx >= 0) {
            T log_sigma2_seasonal = params[layout.log_sigma2_seasonal_idx];
            ms_sigma2_seasonal = safe_exp(log_sigma2_seasonal);

            int n_seasonal = layout.seasonal_end - layout.seasonal_start;
            ms_seasonal.resize(n_seasonal);
            for (int t = 0; t < n_seasonal; t++) {
                ms_seasonal[t] = params[layout.seasonal_start + t];
            }

            // PC prior on sigma2_seasonal + Jacobian
            log_post = log_post + tulpa_multiscale_ad::log_prior_sigma2_temporal_pc(
                ms_sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha);
            log_post = log_post + log_sigma2_seasonal;  // Jacobian
        }

        // Short-term component
        if (layout.log_sigma2_short_idx >= 0) {
            T log_sigma2_short = params[layout.log_sigma2_short_idx];
            ms_sigma2_short = safe_exp(log_sigma2_short);

            int n_short = layout.short_term_end - layout.short_term_start;
            ms_short_term.resize(n_short);
            for (int t = 0; t < n_short; t++) {
                ms_short_term[t] = params[layout.short_term_start + t];
            }

            // PC prior on sigma2_short + Jacobian
            log_post = log_post + tulpa_multiscale_ad::log_prior_sigma2_temporal_pc(
                ms_sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha);
            log_post = log_post + log_sigma2_short;  // Jacobian

            // AR1 rho parameter
            if (layout.logit_rho_short_idx >= 0) {
                T logit_rho_short = params[layout.logit_rho_short_idx];
                // Map logit to (-1, 1): rho = 2*invlogit(logit) - 1
                T u = inv_logit(logit_rho_short);
                ms_rho_short = T(2.0) * u - T(1.0);

                // Beta(2,2) prior on u + Jacobian for logit transform
                log_post = log_post + safe_log(u) + safe_log(T(1.0) - u);  // Beta(2,2)
                log_post = log_post + safe_log(u) + safe_log(T(1.0) - u);  // Jacobian
            }
        }

        // GMRF log-likelihood for all components
        log_post = log_post + tulpa_multiscale_ad::multiscale_temporal_log_lik(
            ms_trend, ms_seasonal, ms_short_term,
            ms_sigma2_trend, ms_sigma2_seasonal, ms_sigma2_short, ms_rho_short,
            ms_data);

        // Precompute multiscale temporal contribution to linear predictor
        tulpa_multiscale_ad::compute_temporal_eta(
            ms_trend, ms_seasonal, ms_short_term, ms_data, ms_temporal_eta);
    }


