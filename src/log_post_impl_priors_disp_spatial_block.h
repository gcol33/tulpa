// log_post_impl_priors_disp_spatial_block.h
// Function-body fragment of compute_log_post_impl<T> in log_post_impl.h.
// Included via #include directly inside the function body — relies on the
// surrounding lexical scope (params, data, layout, log_post, T, beta_*,
// phi_*, re_vals, re_term_offsets, etc.). NOT standalone-compilable.
// No header guard: each fragment is included exactly once per umbrella.
// Overdispersion (phi) priors + ICAR / BYM2 / CAR-proper spatial priors.

    // Overdispersion: Gamma prior
    if (layout.legacy.has_phi_num) {
        T log_phi = params[layout.legacy.log_phi_num_idx];
        log_post = log_post + log_prior_gamma(log_phi, data.phi_prior_shape, data.phi_prior_rate);
    }
    if (layout.legacy.has_phi_denom) {
        T log_phi = params[layout.legacy.log_phi_denom_idx];
        log_post = log_post + log_prior_gamma(log_phi, data.phi_prior_shape, data.phi_prior_rate);
    }

    // Spatial priors
    if (layout.has_spatial) {
        if (layout.is_bym2) {
            // BYM2 Riebler: Half-Cauchy on sigma_total
            T log_sigma = params[layout.log_sigma_bym2_idx];
            log_post = log_post + log_prior_half_cauchy(log_sigma, data.sigma_re_scale);

            // Uniform(0,1) = Beta(1,1) on rho with logit Jacobian:
            // log p(logit_rho) = log(rho) + log(1-rho)
            T logit_rho_val = params[layout.logit_rho_bym2_idx];
            T rho_bym2_prior = T(1.0) / (T(1.0) + safe_exp(-logit_rho_val));
            log_post = log_post + log(rho_bym2_prior)
                                + log(T(1.0) - rho_bym2_prior);

            // ICAR prior on phi_spatial
            T quad_form = T(0.0);
            for (int i = 0; i < data.n_spatial_units; i++) {
                quad_form = quad_form + T(data.n_neighbors[i]) * phi_spatial[i] * phi_spatial[i];
                int row_start = data.adj_row_ptr[i];
                int row_end = data.adj_row_ptr[i + 1];
                for (int k = row_start; k < row_end; k++) {
                    int j = data.adj_col_idx[k];
                    if (j > i) {
                        quad_form = quad_form - T(2.0) * phi_spatial[i] * phi_spatial[j];
                    }
                }
            }
            log_post = log_post - T(0.5) * quad_form;

            // Soft sum-to-zero constraint on ICAR phi
            {
                T phi_sum = T(0.0);
                for (int s = 0; s < data.n_spatial_units; s++) phi_sum = phi_sum + phi_spatial[s];
                log_post = log_post - T(0.5) * T(0.01) * phi_sum * phi_sum;
            }

            // N(0, I) prior on theta
            for (int s = 0; s < data.n_spatial_units; s++) {
                log_post = log_post - T(0.5) * theta_bym2[s] * theta_bym2[s];
            }
        } else if (layout.is_car_proper) {
            // Proper CAR: Q(rho) = D - rho*W is full-rank inside rho bounds.
            T log_tau = params[layout.log_tau_spatial_idx];
            log_post = log_post + log_prior_gamma(
                log_tau, data.tau_spatial_shape, data.tau_spatial_rate);

            T logit_rho_val = params[layout.logit_rho_car_idx];
            T u = T(1.0) / (T(1.0) + safe_exp(-logit_rho_val));
            T rho_car = T(data.car_rho_lower) +
                T(data.car_rho_upper - data.car_rho_lower) * u;

            // Uniform prior on rho bounds plus logit Jacobian.
            log_post = log_post + safe_log(u) + safe_log(T(1.0) - u);

            T quad_form = T(0.0);
            for (int i = 0; i < data.n_spatial_units; i++) {
                quad_form = quad_form + T(data.n_neighbors[i]) *
                    phi_spatial[i] * phi_spatial[i];
                int row_start = data.adj_row_ptr[i];
                int row_end = data.adj_row_ptr[i + 1];
                for (int k = row_start; k < row_end; k++) {
                    int j = data.adj_col_idx[k];
                    if (j > i) {
                        quad_form = quad_form - T(2.0) * rho_car *
                            phi_spatial[i] * phi_spatial[j];
                    }
                }
            }

            T log_det_Q = car_proper_log_det_t(
                data.n_spatial_units, data.adj_row_ptr, data.adj_col_idx,
                data.n_neighbors, rho_car);
            if (std::isinf(get_value(log_det_Q)) && get_value(log_det_Q) < 0.0) {
                return T(-INFINITY);
            }

            int J = data.n_spatial_units;
            log_post = log_post + T(0.5) * log_det_Q
                     + T(0.5 * J) * log_tau
                     - T(0.5) * tau_spatial * quad_form;
        } else {
            // ICAR: Gamma prior on tau
            T log_tau = params[layout.log_tau_spatial_idx];
            log_post = log_post + log_prior_gamma(log_tau, data.tau_spatial_shape, data.tau_spatial_rate);

            // ICAR prior on phi_spatial
            T quad_form = T(0.0);
            for (int i = 0; i < data.n_spatial_units; i++) {
                quad_form = quad_form + T(data.n_neighbors[i]) * phi_spatial[i] * phi_spatial[i];
                int row_start = data.adj_row_ptr[i];
                int row_end = data.adj_row_ptr[i + 1];
                for (int k = row_start; k < row_end; k++) {
                    int j = data.adj_col_idx[k];
                    if (j > i) {
                        quad_form = quad_form - T(2.0) * phi_spatial[i] * phi_spatial[j];
                    }
                }
            }
            int J = data.n_spatial_units;
            T log_tau_sp = params[layout.log_tau_spatial_idx];
            log_post = log_post + T(0.5 * (J - 1)) * log_tau_sp - T(0.5) * tau_spatial * quad_form;
        }
    }


