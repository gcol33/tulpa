  // ============ PRIORS ============

  // Fixed effects: N(0, sigma_beta^2)
  double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
  for (int j = 0; j < data.legacy.p_num; j++) {
    log_post -= 0.5 * tau_beta * beta_num[j] * beta_num[j];
  }
  for (int j = 0; j < data.legacy.p_denom; j++) {
    log_post -= 0.5 * tau_beta * beta_denom[j] * beta_denom[j];
  }

  // Random effects priors (supports multiple crossed RE terms with slopes)
  // Non-centered parameterization: params store z ~ N(0,1), re = sigma * (L * z)
  // Pre-compute actual RE values from z for use in the likelihood loop.
  // Only allocate when the observation loop will run (skip_obs_loop=true avoids
  // this heap allocation on every gradient call ? saves ~100ns per call).
  // re_nc_flat lives in state (aliased above).
  if (!skip_obs_loop) {
    re_nc_flat.assign(params.size(), 0.0);
  }
  if (layout.has_re) {
    int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;

    for (int t = 0; t < n_terms; t++) {
      int n_groups = (n_terms > 1 || data.n_re_terms > 0) ? data.re_n_groups_multi[t] : data.n_re_groups;
      int n_coefs = layout.has_re_slopes ? layout.re_n_coefs_multi[t] : 1;
      bool is_correlated = layout.has_re_slopes && layout.re_correlated_multi[t];

      // Extract sigma parameters for this term
      std::vector<double> sigmas(n_coefs);
      for (int c = 0; c < n_coefs; c++) {
        int log_sigma_idx;
        if (layout.has_re_slopes) {
          log_sigma_idx = layout.log_sigma_re_slopes[t][c];
        } else if (n_terms > 1) {
          log_sigma_idx = layout.log_sigma_re_multi[t];
        } else {
          log_sigma_idx = layout.log_sigma_re_idx;
        }
        double log_sigma = params[log_sigma_idx];
        sigmas[c] = std::exp(log_sigma);

        // Half-Cauchy(0, scale) prior for each sigma
        double ratio = sigmas[c] / data.sigma_re_scale;
        log_post -= std::log(1.0 + ratio * ratio);
        log_post += log_sigma;  // Jacobian
      }

      // For correlated slopes: LKJ prior on correlation matrix via Cholesky.
      // Parameterization: Sigma = diag(sigma) * L * L' * diag(sigma).
      // Tanh-parameterized L plus LKJ(eta=2) prior ? see lkj_chol_helpers.h.
      std::vector<double> L_flat;
      if (is_correlated && n_coefs > 1) {
        int chol_start = layout.chol_re_start_multi[t];
        L_flat.resize(n_coefs * n_coefs, 0.0);

        double log_jac_tanh = 0.0;
        if (!tulpa::build_L_from_raw(&params[chol_start], n_coefs,
                                     L_flat.data(), &log_jac_tanh)) {
          return -std::numeric_limits<double>::infinity();
        }
        log_post += log_jac_tanh;
        log_post += tulpa::lkj_log_prior_density(L_flat.data(), n_coefs, /*eta=*/2.0);
      }

      // Get RE parameters for this term
      int re_start = (n_terms > 1 || layout.has_re_slopes) ? layout.re_start_multi[t] : layout.re_start;

      // Non-centered parameterization for correlated slopes:
      // Params store z ~ N(0,1), compute re = diag(sigma) * L * z
      // This eliminates the funnel geometry that plagues centered parameterization.
      // For uncorrelated slopes, params store z ~ N(0,1), re = sigma * z
      if (is_correlated && n_coefs > 1) {
        // Pre-compute re from z for all groups: re[g] = diag(sigma) * L * z[g]
        // Store in re_nc_flat for use in likelihood loop only
        if (!skip_obs_loop) {
          tulpa::compute_u_eff(L_flat.data(), n_coefs, sigmas.data(),
                                &params[re_start], n_groups,
                                &re_nc_flat[re_start]);
        }

        // N(0, I) prior on z (trivial in non-centered)
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            double z_gc = params[re_start + g * n_coefs + c];
            log_post -= 0.5 * z_gc * z_gc;
          }
        }
        // No log-determinant term: in non-centered parameterization,
        // the |det(diag(sigma)*L)| from the change of variables cancels exactly
        // with the |Sigma|^{-1/2} normalization of the MVN density.

      } else if (data.re_parameterization == 1) {
        // Uncorrelated non-centered: params store z ~ N(0,1), re = sigma * z
        // Pre-compute re = sigma * z for observation loop
        if (!skip_obs_loop) {
          for (int g = 0; g < n_groups; g++) {
            for (int c = 0; c < n_coefs; c++) {
              double z_gc = params[re_start + g * n_coefs + c];
              re_nc_flat[re_start + g * n_coefs + c] = sigmas[c] * z_gc;
            }
          }
        }
        // N(0, I) prior on z
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            double z_gc = params[re_start + g * n_coefs + c];
            log_post -= 0.5 * z_gc * z_gc;
          }
        }
      } else {
        // Uncorrelated centered: params store actual re values, prior is N(0, sigma_c^2)
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            double re_val = params[re_start + g * n_coefs + c];
            double tau_re = 1.0 / (sigmas[c] * sigmas[c] + 1e-10);
            log_post -= 0.5 * tau_re * re_val * re_val;
            log_post += 0.5 * std::log(tau_re);
          }
        }
        log_post -= 0.5 * n_groups * n_coefs * std::log(2.0 * M_PI);
      }
    }
  }

  // Overdispersion: Gamma prior
  if (layout.legacy.has_phi_num) {
    log_post += (data.phi_prior_shape - 1.0) * log_phi_num
              - data.phi_prior_rate * phi_num + log_phi_num;
  }
  if (layout.legacy.has_phi_denom) {
    log_post += (data.phi_prior_shape - 1.0) * log_phi_denom
              - data.phi_prior_rate * phi_denom + log_phi_denom;
  }

  // Spatial priors
  if (layout.has_spatial) {
    int J = data.n_spatial_units;

    if (layout.is_bym2) {
      // BYM2 Riebler: Half-Cauchy on sigma_total (always needed, even collapsed)
      double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);  // recover sigma_total
      double ratio = sigma_total / data.sigma_re_scale;
      log_post -= std::log(1.0 + ratio * ratio);
      log_post += params[layout.log_sigma_bym2_idx];  // Jacobian for log transform

      // Uniform(0,1) = Beta(1,1) on rho with logit Jacobian:
      // log p(logit_rho) = log(rho) + log(1-rho)
      log_post += std::log(rho_bym2) + std::log(1.0 - rho_bym2);

      if (!data.bym2_collapsed) {
        // Standard: phi/theta are explicit params
        // phi_scaled ~ N(0, Q^{-1}) with soft sum-to-zero
        std::vector<double> phi_vec(phi_spatial, phi_spatial + J);
        double quad = icar_quadratic_form(phi_vec, data);
        log_post -= 0.5 * quad;

        {
          double phi_sum = 0.0;
          for (int j = 0; j < J; j++) phi_sum += phi_spatial[j];
          log_post -= 0.5 * 0.01 * phi_sum * phi_sum;
        }

        // theta ~ N(0, I)
        for (int j = 0; j < J; j++) {
          log_post -= 0.5 * theta_bym2[j] * theta_bym2[j];
        }
      }
      // Collapsed: phi/theta priors already included in collapsed_lp above
    } else if (layout.is_car_proper) {
      // Proper CAR: Q(rho) = D - rho*W is full-rank for rho ? (rho_lower, rho_upper).
      // Prior: phi | tau, rho ~ N(0, (tau*Q)^{-1})
      //   log p = 0.5*log|tau*Q| - 0.5*tau*phi'Q*phi
      //         = 0.5*J*log(tau) + 0.5*log|Q(rho)| - 0.5*tau*phi'Q(rho)*phi
      // tau ~ Gamma(shape, rate) (with log-Jacobian).
      // rho via logit transform with uniform prior on (rho_lower, rho_upper):
      //   p(logit_rho) ? u*(1-u) where u = (rho-lower)/(upper-lower).

      // tau prior (Gamma + log-Jacobian)
      log_post += (data.tau_spatial_shape - 1.0) * log_tau_spatial
                - data.tau_spatial_rate * tau_spatial + log_tau_spatial;

      // Logit-rho Jacobian: log(u) + log(1-u), where u ? (0,1)
      double u_logit = (rho_car - data.car_rho_lower) /
                       (data.car_rho_upper - data.car_rho_lower);
      // Guard against numerical edges
      double u_clip = std::min(std::max(u_logit, 1e-12), 1.0 - 1e-12);
      log_post += std::log(u_clip) + std::log(1.0 - u_clip);

      // phi quadratic form: phi' Q(rho) phi  (shared CAR/ICAR kernel)
      double quad = tulpa::car_quad_form(
          phi_spatial, J,
          data.adj_row_ptr.data(), data.adj_col_idx.data(), data.n_neighbors.data(),
          rho_car);

      // Log-determinant of Q(rho) ? recompute each call (rho changes Q).
      // Dense O(J^3) Cholesky is fine for small J; switch to sparse for large J.
      std::vector<double> Q = tulpa_car_proper::compute_car_precision(
          J, data.adj_row_ptr, data.adj_col_idx, data.n_neighbors, rho_car);
      double log_det_Q = tulpa_car_proper::car_log_det(J, Q);
      if (std::isinf(log_det_Q)) {
        return -std::numeric_limits<double>::infinity();
      }

      log_post += 0.5 * log_det_Q + 0.5 * J * log_tau_spatial - 0.5 * tau_spatial * quad;
    } else {
      // ICAR
      // tau ~ Gamma(shape, rate) (always needed, even collapsed)
      log_post += (data.tau_spatial_shape - 1.0) * log_tau_spatial
                - data.tau_spatial_rate * tau_spatial + log_tau_spatial;

      if (!data.icar_collapsed) {
        // Standard: phi is explicit param
        // phi ~ ICAR(tau): p(phi|tau) propto tau^{(J-1)/2} exp(-0.5 * tau * phi'Qphi)
        std::vector<double> phi_vec(phi_spatial, phi_spatial + J);
        double quad = icar_quadratic_form(phi_vec, data);
        log_post += 0.5 * (J - 1) * log_tau_spatial - 0.5 * tau_spatial * quad;
      }
      // Collapsed: ICAR prior on phi* + 0.5*(J-1)*log(tau) already in collapsed_lp above
    }
  }

  // ZI coefficient priors
  const double* beta_zi = nullptr;
  if (layout.has_zi) {
    beta_zi = &params[layout.beta_zi_start];
    // N(0, zi_prior_sd^2) prior on ZI coefficients
    double tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd + 1e-10);
    for (int j = 0; j < data.p_zi; j++) {
      log_post -= 0.5 * tau_zi * beta_zi[j] * beta_zi[j];
    }
  }

  // OI coefficient priors (for OI_BINOMIAL and ZOIB)
  const double* beta_oi = nullptr;
  if (layout.has_oi && data.p_oi > 0) {
    beta_oi = &params[layout.beta_oi_start];
    // N(0, oi_prior_sd^2) prior on OI coefficients
    double tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd + 1e-10);
    for (int j = 0; j < data.p_oi; j++) {
      log_post -= 0.5 * tau_oi * beta_oi[j] * beta_oi[j];
    }
  }
