// log_post_impl_priors_gp_block.h
// Function-body fragment of compute_log_post_impl<T> in log_post_impl.h.
// Included via #include directly inside the function body — relies on the
// surrounding lexical scope (params, data, layout, log_post, T, beta_*,
// phi_*, re_vals, re_term_offsets, etc.). NOT standalone-compilable.
// No header guard: each fragment is included exactly once per umbrella.
// GP / multiscale-GP / HSGP spatial parameters + priors + precomputed effects.

    // GP spatial parameters and priors
    std::vector<T> gp_w;

    if (layout.is_gp && data.has_gp) {
        // Extract hyperparameters from log-scale
        T log_sigma2_gp = params[layout.log_sigma2_gp_idx];
        T log_phi_gp = params[layout.log_phi_gp_idx];
        T sigma2_gp = safe_exp(log_sigma2_gp);
        T phi_gp = safe_exp(log_phi_gp);

        // PC prior on sigma2 + Jacobian for log transform
        log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
            sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        log_post = log_post + log_sigma2_gp;  // Jacobian

        // Uniform prior on phi within bounds + Jacobian
        double phi_val = get_value(phi_gp);
        if (phi_val < data.gp_phi_prior_lower || phi_val > data.gp_phi_prior_upper) {
            return T(-INFINITY);
        }
        log_post = log_post + tulpa_gp::log_prior_phi_uniform_t(
            phi_gp, data.gp_phi_prior_lower, data.gp_phi_prior_upper);
        log_post = log_post + log_phi_gp;  // Jacobian

        // Extract GP spatial effects w[0..n_gp-1]
        int n_gp = layout.gp_w_end - layout.gp_w_start;
        gp_w.resize(n_gp);
        for (int k = 0; k < n_gp; k++) {
            gp_w[k] = params[layout.gp_w_start + k];
        }

        // Apply RSR projection if enabled
        if (data.has_rsr && !data.rsr_projection.empty()) {
            std::vector<T> w_projected(data.rsr_n, T(0.0));
            for (int ii = 0; ii < data.rsr_n; ii++) {
                for (int jj = 0; jj < data.rsr_n; jj++) {
                    w_projected[ii] = w_projected[ii]
                        + T(data.rsr_projection[ii * data.rsr_n + jj]) * gp_w[jj];
                }
            }
            gp_w = w_projected;
        }

        // NNGP log-likelihood on spatial effects
        log_post = log_post + tulpa_gp::gp_nngp_log_lik_t(
            gp_w, sigma2_gp, phi_gp, data.gp_data);
    }

    // Multi-scale GP spatial parameters and priors
    std::vector<T> ms_gp_w_local;
    std::vector<T> ms_gp_w_regional;
    std::vector<T> ms_gp_effect;

    if (layout.is_multiscale_gp && data.has_multiscale_gp) {
        if (data.msgp_is_hsgp) {
            // --- HSGP-MSGP: two independent HSGP evaluations with shared basis ---
            T log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
            T log_ls_local = params[layout.log_phi_gp_local_idx];  // log_lengthscale
            T sigma2_local_h = safe_exp(log_sigma2_local);
            T ls_local = safe_exp(log_ls_local);

            T log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
            T log_ls_regional = params[layout.log_phi_gp_regional_idx];
            T sigma2_regional_h = safe_exp(log_sigma2_regional);
            T ls_regional = safe_exp(log_ls_regional);

            int m_total = data.msgp_hsgp_data.m_total;

            // PC priors on sigma for both scales
            T sigma_local = safe_sqrt(sigma2_local_h);
            double rate_local = -std::log(data.ms_sigma2_local_prior_alpha) / data.ms_sigma2_local_prior_U;
            log_post = log_post + T(std::log(rate_local)) - T(rate_local) * sigma_local
                     - safe_log(T(2.0) * sigma_local);
            log_post = log_post + log_sigma2_local * T(0.5);  // Jacobian

            T sigma_regional = safe_sqrt(sigma2_regional_h);
            double rate_regional = -std::log(data.ms_sigma2_regional_prior_alpha) / data.ms_sigma2_regional_prior_U;
            log_post = log_post + T(std::log(rate_regional)) - T(rate_regional) * sigma_regional
                     - safe_log(T(2.0) * sigma_regional);
            log_post = log_post + log_sigma2_regional * T(0.5);  // Jacobian

            // LogNormal priors on lengthscales (centered at scale-appropriate ranges)
            T z_local = (log_ls_local - T(data.ms_log_ls_local_mean)) / T(data.ms_log_ls_local_sd);
            log_post = log_post - T(0.5) * z_local * z_local - T(std::log(data.ms_log_ls_local_sd));

            T z_regional = (log_ls_regional - T(data.ms_log_ls_regional_mean)) / T(data.ms_log_ls_regional_sd);
            log_post = log_post - T(0.5) * z_regional * z_regional - T(std::log(data.ms_log_ls_regional_sd));

            // N(0, I) priors on beta coefficients
            for (int j = 0; j < m_total; j++) {
                T beta_local_j = params[layout.gp_local_start + j];
                T beta_regional_j = params[layout.gp_regional_start + j];
                log_post = log_post - T(0.5) * beta_local_j * beta_local_j;
                log_post = log_post - T(0.5) * beta_regional_j * beta_regional_j;
            }

            // Evaluate HSGP spatial effects for both scales separately
            // (matching compute_log_post's separate evaluation order for float precision)
            std::vector<T> f_local(data.N, T(0.0));
            std::vector<T> f_regional(data.N, T(0.0));
            for (int j = 0; j < m_total; j++) {
                double omega_sq = data.msgp_hsgp_data.eigenvalues[j];
                T beta_local_j = params[layout.gp_local_start + j];
                T beta_regional_j = params[layout.gp_regional_start + j];

                // Spectral density for local scale
                T S_local_j = sigma2_local_h * T(std::sqrt(2.0 * M_PI)) * ls_local
                    * safe_exp(T(-0.5) * ls_local * ls_local * T(omega_sq));
                T scaled_local_j = safe_sqrt(S_local_j) * beta_local_j;

                // Spectral density for regional scale
                T S_regional_j = sigma2_regional_h * T(std::sqrt(2.0 * M_PI)) * ls_regional
                    * safe_exp(T(-0.5) * ls_regional * ls_regional * T(omega_sq));
                T scaled_regional_j = safe_sqrt(S_regional_j) * beta_regional_j;

                for (int ii = 0; ii < data.N; ii++) {
                    double phi_ij = data.msgp_hsgp_data.phi_flat[ii * m_total + j];
                    f_local[ii] = f_local[ii] + T(phi_ij) * scaled_local_j;
                    f_regional[ii] = f_regional[ii] + T(phi_ij) * scaled_regional_j;
                }
            }
            ms_gp_effect.resize(data.N, T(0.0));
            for (int ii = 0; ii < data.N; ii++) {
                ms_gp_effect[ii] = f_local[ii] + f_regional[ii];
            }
        } else {
            // --- NNGP-MSGP: standard implementation ---
            T log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
            T log_phi_local = params[layout.log_phi_gp_local_idx];
            T log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
            T log_phi_regional = params[layout.log_phi_gp_regional_idx];

            T sigma2_local_n = safe_exp(log_sigma2_local);
            T phi_local = safe_exp(log_phi_local);
            T sigma2_regional_n = safe_exp(log_sigma2_regional);
            T phi_regional = safe_exp(log_phi_regional);

            // PC priors on sigma2 + Jacobians
            log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
                sigma2_local_n, data.ms_sigma2_local_prior_U, data.ms_sigma2_local_prior_alpha);
            log_post = log_post + log_sigma2_local;  // Jacobian

            log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
                sigma2_regional_n, data.ms_sigma2_regional_prior_U, data.ms_sigma2_regional_prior_alpha);
            log_post = log_post + log_sigma2_regional;  // Jacobian

            // Uniform priors on phi (range) within bounds + Jacobians
            double phi_local_val = get_value(phi_local);
            if (phi_local_val < data.multiscale_gp_data.range_local_lower ||
                phi_local_val > data.multiscale_gp_data.range_local_upper) {
                return T(-INFINITY);
            }
            log_post = log_post + log_phi_local;  // Jacobian

            double phi_regional_val = get_value(phi_regional);
            if (phi_regional_val < data.multiscale_gp_data.range_regional_lower ||
                phi_regional_val > data.multiscale_gp_data.range_regional_upper) {
                return T(-INFINITY);
            }
            log_post = log_post + log_phi_regional;  // Jacobian

            // Extract local GP effects
            int n_gp_local = layout.gp_local_end - layout.gp_local_start;
            ms_gp_w_local.resize(n_gp_local);
            for (int k = 0; k < n_gp_local; k++) {
                ms_gp_w_local[k] = params[layout.gp_local_start + k];
            }

            // Extract regional GP effects
            int n_gp_regional = layout.gp_regional_end - layout.gp_regional_start;
            ms_gp_w_regional.resize(n_gp_regional);
            for (int k = 0; k < n_gp_regional; k++) {
                ms_gp_w_regional[k] = params[layout.gp_regional_start + k];
            }

            // Apply RSR projection if enabled
            if (data.has_rsr && !data.rsr_projection.empty()) {
                std::vector<T> local_proj(data.rsr_n, T(0.0));
                std::vector<T> regional_proj(data.rsr_n, T(0.0));
                for (int ii = 0; ii < data.rsr_n; ii++) {
                    for (int jj = 0; jj < data.rsr_n; jj++) {
                        local_proj[ii] = local_proj[ii]
                            + T(data.rsr_projection[ii * data.rsr_n + jj]) * ms_gp_w_local[jj];
                        regional_proj[ii] = regional_proj[ii]
                            + T(data.rsr_projection[ii * data.rsr_n + jj]) * ms_gp_w_regional[jj];
                    }
                }
                ms_gp_w_local = local_proj;
                ms_gp_w_regional = regional_proj;
            }

            // Multiscale NNGP log-likelihood for both scales
            log_post = log_post + tulpa_gp::multiscale_gp_log_lik_t(
                ms_gp_w_local, ms_gp_w_regional,
                sigma2_local_n, phi_local, sigma2_regional_n, phi_regional,
                data.multiscale_gp_data);

            // Precompute combined effect at observation level
            ms_gp_effect.resize(data.N, T(0.0));
            for (int ii = 0; ii < data.N; ii++) {
                int loc = data.multiscale_gp_data.obs_to_loc[ii];
                ms_gp_effect[ii] = ms_gp_w_local[loc] + ms_gp_w_regional[loc];
            }
        }
    }

    // HSGP spatial parameters, priors, and precomputed effects
    std::vector<T> hsgp_f_impl;

    if (layout.is_hsgp && data.has_hsgp) {
        T log_sigma2_hsgp = params[layout.log_sigma2_hsgp_idx];
        T log_lengthscale_hsgp = params[layout.log_lengthscale_hsgp_idx];
        T sigma2_hsgp = safe_exp(log_sigma2_hsgp);
        T lengthscale_hsgp = safe_exp(log_lengthscale_hsgp);

        int m_total = data.hsgp_data.m_total;

        // Extract beta coefficients
        std::vector<T> hsgp_beta(m_total);
        for (int j = 0; j < m_total; j++) {
            hsgp_beta[j] = params[layout.hsgp_beta_start + j];
        }

        // PC prior on sigma: P(sigma > 1) = 0.01 -> rate = 4.6
        // log p(sigma) = log(rate) - rate*sigma - log(2*sigma)
        T sigma_hsgp = safe_sqrt(sigma2_hsgp);
        T rate_sigma_hsgp = T(4.6);
        log_post = log_post + safe_log(rate_sigma_hsgp) - rate_sigma_hsgp * sigma_hsgp
                   - safe_log(T(2.0) * sigma_hsgp);
        log_post = log_post + log_sigma2_hsgp * T(0.5);  // Jacobian: d(sigma)/d(log_sigma2)

        // LogNormal(0, 1) prior on lengthscale
        // log p(ell) = -0.5 * log(ell)^2  (Jacobian cancels)
        log_post = log_post - T(0.5) * log_lengthscale_hsgp * log_lengthscale_hsgp;

        // N(0, I) prior on beta
        for (int j = 0; j < m_total; j++) {
            log_post = log_post - T(0.5) * hsgp_beta[j] * hsgp_beta[j];
        }

        // Evaluate HSGP spatial effect: f = Phi * (sqrt(S) .* beta)
        // Phi and eigenvalues are double (precomputed data), but sigma2/lengthscale/beta are T
        hsgp_f_impl.resize(data.N, T(0.0));
        for (int j = 0; j < m_total; j++) {
            T S_j = sigma2_hsgp * T(std::sqrt(2.0 * M_PI)) * lengthscale_hsgp
                    * safe_exp(T(-0.5) * lengthscale_hsgp * lengthscale_hsgp
                               * T(data.hsgp_data.eigenvalues[j]));
            T scaled_beta_j = safe_sqrt(S_j) * hsgp_beta[j];
            for (int i = 0; i < data.N; i++) {
                hsgp_f_impl[i] = hsgp_f_impl[i]
                    + T(data.hsgp_data.phi_flat[i * m_total + j]) * scaled_beta_j;
            }
        }
    }


