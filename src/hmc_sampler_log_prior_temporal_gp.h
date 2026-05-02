  // Temporal priors
  double tau_temporal = 1.0, log_tau_temporal = 0.0;
  double rho_ar1 = 0.5;
  const double* phi_temporal = nullptr;
  double sigma2_temporal_gp = 1.0, phi_temporal_gp = 1.0;
  // temporal_f_nc lives in state (aliased above).

  if (layout.has_temporal) {
    phi_temporal = &params[layout.temporal_start];

    if (layout.is_temporal_gp) {
      if (precomputed_tgp_log_prior) {
        // Fused path: all temporal GP prior terms already computed by gradient function
        log_post += *precomputed_tgp_log_prior;
      } else {
        // Temporal GP: sigma2 and phi (lengthscale) parameters
        double log_sigma2 = params[layout.log_sigma2_temporal_gp_idx];
        double logit_phi = params[layout.logit_phi_temporal_gp_idx];
        sigma2_temporal_gp = std::exp(log_sigma2);

        // Logit-bounded phi: phi = lower + (upper - lower) * sigmoid(logit_phi)
        double phi_lower = data.temporal_gp_phi_prior_lower;
        double phi_upper = data.temporal_gp_phi_prior_upper;
        double phi_range = phi_upper - phi_lower;
        double sigmoid_val = 1.0 / (1.0 + std::exp(-logit_phi));
        phi_temporal_gp = phi_lower + phi_range * sigmoid_val;

        // PC prior on sigma2 (favor smaller variance)
        log_post += tulpa_temporal_gp::log_prior_sigma2_temporal_pc(
            sigma2_temporal_gp, data.temporal_gp_sigma2_prior_U,
            data.temporal_gp_sigma2_prior_alpha);
        log_post += log_sigma2;  // Jacobian for log transform

        // Uniform prior on phi within bounds (always satisfied by construction)
        // Jacobian: log(phi - lower) + log(upper - phi) - log(upper - lower)
        log_post += std::log(phi_temporal_gp - phi_lower)
                  + std::log(phi_upper - phi_temporal_gp)
                  - std::log(phi_range);

        // GP log-likelihood for temporal effects
        int T = data.n_times;
        int n_temporal = data.n_temporal_groups * T;

        if (data.temporal_gp_parameterization == 1) {
          // Non-centered: params store z ~ N(0,I), reconstruct f for obs loop
          // z ~ N(0, I) prior
          for (int t = 0; t < n_temporal; t++) {
            log_post += -0.5 * phi_temporal[t] * phi_temporal[t];
          }

          // Jacobian of NC transform: log|det(df/dz)| per group
          double sigma = std::sqrt(sigma2_temporal_gp);
          for (int g = 0; g < data.n_temporal_groups; g++) {
            log_post += T * std::log(sigma);
            for (int t = 1; t < T; t++) {
              double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
              double rho = std::exp(-dt / phi_temporal_gp);
              double one_minus_rho2 = 1.0 - rho * rho;
              if (one_minus_rho2 < 1e-10) one_minus_rho2 = 1e-10;
              log_post += 0.5 * std::log(one_minus_rho2);
            }
          }

          // Forward transform z -> f
          temporal_f_nc.resize(n_temporal);
          for (int g = 0; g < data.n_temporal_groups; g++) {
            int off = g * T;
            temporal_f_nc[off] = sigma * phi_temporal[off];
            for (int t = 1; t < T; t++) {
              double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
              double rho = std::exp(-dt / phi_temporal_gp);
              double one_minus_rho2 = 1.0 - rho * rho;
              if (one_minus_rho2 < 1e-10) one_minus_rho2 = 1e-10;
              double a_t = sigma * std::sqrt(one_minus_rho2);
              temporal_f_nc[off + t] = rho * temporal_f_nc[off + t - 1] + a_t * phi_temporal[off + t];
            }
          }
          // Override phi_temporal pointer to use reconstructed f
          phi_temporal = temporal_f_nc.data();

        } else {
          // Centered: GP log-likelihood for temporal effects
          for (int g = 0; g < data.n_temporal_groups; g++) {
            std::vector<double> phi_g_vec(phi_temporal + g * T, phi_temporal + (g + 1) * T);
            log_post += tulpa_temporal_gp::temporal_gp_log_lik(
                phi_g_vec, data.temporal_gp_data, sigma2_temporal_gp, phi_temporal_gp);
          }
        }
      }

    } else {
      // RW1/RW2/AR1: tau-based parameterization
      log_tau_temporal = params[layout.log_tau_temporal_idx];
      tau_temporal = std::exp(log_tau_temporal);

      // tau ~ Gamma(shape, rate) with Jacobian
      log_post += (data.tau_temporal_shape - 1.0) * log_tau_temporal
                - data.tau_temporal_rate * tau_temporal + log_tau_temporal;

      // AR1: also estimate rho
      if (layout.is_ar1) {
        double logit_rho = params[layout.logit_rho_ar1_idx];
        rho_ar1 = 1.0 / (1.0 + std::exp(-logit_rho));

        // rho ~ Uniform(0,1) prior with logit Jacobian
        log_post += std::log(rho_ar1) + std::log(1.0 - rho_ar1);
      }

      // Temporal effects prior (per group)
      int T = data.n_times;
      for (int g = 0; g < data.n_temporal_groups; g++) {
        const double* phi_g = phi_temporal + g * T;

        if (data.temporal_type == TemporalType::RW1) {
          double quad = tulpa_temporal::rw1_quadratic_form(phi_g, T, data.temporal_cyclic);
          int rank = data.temporal_cyclic ? T : T - 1;
          log_post += 0.5 * rank * log_tau_temporal - 0.5 * tau_temporal * quad;
          // Soft sum-to-zero constraint
          log_post += tulpa_temporal::sum_to_zero_penalty(phi_g, T, 0.001);

        } else if (data.temporal_type == TemporalType::RW2) {
          double quad = tulpa_temporal::rw2_quadratic_form(phi_g, T, data.temporal_cyclic);
          int rank = data.temporal_cyclic ? T : T - 2;
          log_post += 0.5 * rank * log_tau_temporal - 0.5 * tau_temporal * quad;
          // Soft sum-to-zero constraint
          log_post += tulpa_temporal::sum_to_zero_penalty(phi_g, T, 0.001);

        } else if (data.temporal_type == TemporalType::AR1) {
          log_post += tulpa_temporal::ar1_log_density(phi_g, T, rho_ar1, tau_temporal);
        }
      }
    }
  }

  // GP spatial priors
  double sigma2_gp = 1.0, phi_gp = 1.0;
  const double* gp_w = nullptr;
  // gp_w_nc_buf lives in state (aliased above).

  if (layout.is_gp && data.has_gp) {
    double log_sigma2_gp = params[layout.log_sigma2_gp_idx];
    double log_phi_gp = params[layout.log_phi_gp_idx];
    sigma2_gp = std::exp(log_sigma2_gp);
    phi_gp = std::exp(log_phi_gp);

    // PC prior on sigma2 (favor smaller variance)
    log_post += tulpa_gp::log_prior_sigma2_pc(sigma2_gp, data.gp_sigma2_prior_U,
                                                data.gp_sigma2_prior_alpha);
    log_post += log_sigma2_gp;  // Jacobian for log transform

    // Uniform prior on phi within bounds
    log_post += tulpa_gp::log_prior_phi_uniform(phi_gp, data.gp_phi_prior_lower,
                                                  data.gp_phi_prior_upper);
    log_post += log_phi_gp;  // Jacobian

    if (data.gp_collapsed) {
      // Collapsed GP: find w* via inner Laplace, add NNGP prior + Laplace correction
      double phi_num_val = 1.0, phi_denom_val = 1.0;
      if (layout.legacy.log_phi_num_idx >= 0) phi_num_val = std::exp(params[layout.legacy.log_phi_num_idx]);
      if (layout.legacy.log_phi_denom_idx >= 0) phi_denom_val = std::exp(params[layout.legacy.log_phi_denom_idx]);

      auto gp_res = collapsed_gp_log_post_contribution(
          &params[layout.legacy.beta_num_start], &params[layout.legacy.beta_denom_start],
          sigma2_gp, phi_gp, phi_num_val, phi_denom_val, data, collapsed_gp_ws);

      // Point gp_w at w* for the observation loop
      gp_w_nc_buf.resize(data.gp_data.n_obs);
      std::memcpy(gp_w_nc_buf.data(), collapsed_gp_ws.w_star.data(),
                  data.gp_data.n_obs * sizeof(double));
      gp_w = gp_w_nc_buf.data();
      log_post += gp_res.log_post_contribution;
    } else if (layout.gp_w_start < 0 || layout.gp_w_start + data.gp_data.n_obs > (int)params.size()) {
      return -INFINITY;  // Invalid parameter layout
    } else {

    int N_gp = data.gp_data.n_obs;

    if (data.gp_parameterization == 1) {
      // Non-centered: params hold z ~ N(0,1), transform to w
      // The NNGP prior on w + Jacobian |dw/dz| = N(0,I) on z (exact cancellation)
      // So prior is just -0.5*sum(z^2), no explicit Jacobian needed
      const double* z_params = &params[layout.gp_w_start];

      // N(0,1) prior on z (includes implicit Jacobian cancellation)
      double z_sq_sum = 0.0;
      for (int i = 0; i < N_gp; i++) z_sq_sum += z_params[i] * z_params[i];
      log_post += -0.5 * z_sq_sum;

      // Forward pass z -> w for observation loop
      static thread_local tulpa_gp::NNGPNCWorkspace nc_ws_lp;
      tulpa_gp::nngp_nc_forward(z_params, sigma2_gp, phi_gp, data.gp_data, nc_ws_lp);

      // Point gp_w to reconstructed w for observation loop
      gp_w_nc_buf.resize(N_gp);
      std::memcpy(gp_w_nc_buf.data(), nc_ws_lp.w.data(), N_gp * sizeof(double));
      gp_w = gp_w_nc_buf.data();
    } else {
      // Centered: NNGP prior on w directly
      gp_w = &params[layout.gp_w_start];
      std::vector<double> w_vec(gp_w, gp_w + N_gp);

      // Apply RSR projection if enabled
      if (data.has_rsr && !data.rsr_projection.empty()) {
        std::vector<double> w_projected(data.rsr_n, 0.0);
        for (int i = 0; i < data.rsr_n; i++) {
          for (int j = 0; j < data.rsr_n; j++) {
            w_projected[i] += data.rsr_projection[i * data.rsr_n + j] * w_vec[j];
          }
        }
        w_vec = w_projected;
      }

      double gp_ll = tulpa_gp::gp_nngp_log_lik(w_vec, sigma2_gp, phi_gp, data.gp_data);
      log_post += gp_ll;
    }
    } // end else (non-collapsed)
  }

  // Multi-scale GP spatial priors
  double sigma2_local = 1.0, phi_local = 1.0;
  double sigma2_regional = 1.0, phi_regional = 1.0;
  const double* gp_local = nullptr;
  const double* gp_regional = nullptr;
  // msgp_hsgp_f_local / msgp_hsgp_f_regional live in state (aliased above).

  if (layout.is_multiscale_gp && data.has_multiscale_gp) {
    if (data.msgp_is_hsgp) {
      // --- HSGP-MSGP: two independent HSGP evaluations with shared basis ---
      double log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
      double log_ls_local = params[layout.log_phi_gp_local_idx];  // log_lengthscale
      sigma2_local = std::exp(log_sigma2_local);
      double ls_local = std::exp(log_ls_local);

      double log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
      double log_ls_regional = params[layout.log_phi_gp_regional_idx];
      sigma2_regional = std::exp(log_sigma2_regional);
      double ls_regional = std::exp(log_ls_regional);

      int m_total = data.msgp_hsgp_data.m_total;
      const double* beta_local = &params[layout.gp_local_start];
      const double* beta_regional = &params[layout.gp_regional_start];

      // PC priors on sigma for both scales
      double sigma_local = std::sqrt(sigma2_local);
      double rate_local = -std::log(data.ms_sigma2_local_prior_alpha) / data.ms_sigma2_local_prior_U;
      log_post += std::log(rate_local) - rate_local * sigma_local - std::log(2.0 * sigma_local);
      log_post += log_sigma2_local * 0.5;  // Jacobian

      double sigma_regional = std::sqrt(sigma2_regional);
      double rate_regional = -std::log(data.ms_sigma2_regional_prior_alpha) / data.ms_sigma2_regional_prior_U;
      log_post += std::log(rate_regional) - rate_regional * sigma_regional - std::log(2.0 * sigma_regional);
      log_post += log_sigma2_regional * 0.5;  // Jacobian

      // LogNormal priors on lengthscales (centered at scale-appropriate ranges)
      double z_local = (log_ls_local - data.ms_log_ls_local_mean) / data.ms_log_ls_local_sd;
      log_post += -0.5 * z_local * z_local - std::log(data.ms_log_ls_local_sd);
      // No Jacobian needed: parameterized on log scale, prior on log scale

      double z_regional = (log_ls_regional - data.ms_log_ls_regional_mean) / data.ms_log_ls_regional_sd;
      log_post += -0.5 * z_regional * z_regional - std::log(data.ms_log_ls_regional_sd);

      // N(0, I) priors on beta coefficients
      for (int j = 0; j < m_total; j++) {
        log_post += -0.5 * beta_local[j] * beta_local[j];
        log_post += -0.5 * beta_regional[j] * beta_regional[j];
      }

      // Evaluate HSGP spatial effects
      std::vector<double> beta_local_vec(beta_local, beta_local + m_total);
      std::vector<double> beta_regional_vec(beta_regional, beta_regional + m_total);
      tulpa_hsgp::hsgp_evaluate(beta_local_vec, sigma2_local, ls_local,
                                  data.msgp_hsgp_data, msgp_hsgp_f_local);
      tulpa_hsgp::hsgp_evaluate(beta_regional_vec, sigma2_regional, ls_regional,
                                  data.msgp_hsgp_data, msgp_hsgp_f_regional);

    } else {
      // --- NNGP-MSGP: standard implementation ---
      // Local scale parameters
      double log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
      double log_phi_local = params[layout.log_phi_gp_local_idx];
      sigma2_local = std::exp(log_sigma2_local);
      phi_local = std::exp(log_phi_local);
      gp_local = &params[layout.gp_local_start];

      // Regional scale parameters
      double log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
      double log_phi_regional = params[layout.log_phi_gp_regional_idx];
      sigma2_regional = std::exp(log_sigma2_regional);
      phi_regional = std::exp(log_phi_regional);
      gp_regional = &params[layout.gp_regional_start];

      // PC priors on variances
      log_post += tulpa_gp::log_prior_sigma2_pc(sigma2_local, data.ms_sigma2_local_prior_U,
                                                  data.ms_sigma2_local_prior_alpha);
      log_post += log_sigma2_local;

      log_post += tulpa_gp::log_prior_sigma2_pc(sigma2_regional, data.ms_sigma2_regional_prior_U,
                                                  data.ms_sigma2_regional_prior_alpha);
      log_post += log_sigma2_regional;

      // Range priors (uniform within bounds)
      if (phi_local < data.multiscale_gp_data.range_local_lower ||
          phi_local > data.multiscale_gp_data.range_local_upper) {
        return -std::numeric_limits<double>::infinity();
      }
      log_post += log_phi_local;

      if (phi_regional < data.multiscale_gp_data.range_regional_lower ||
          phi_regional > data.multiscale_gp_data.range_regional_upper) {
        return -std::numeric_limits<double>::infinity();
      }
      log_post += log_phi_regional;

      // NNGP likelihood for each scale
      std::vector<double> w_local_vec(gp_local, gp_local + data.multiscale_gp_data.n_obs);
      std::vector<double> w_regional_vec(gp_regional, gp_regional + data.multiscale_gp_data.n_obs);

      // Apply RSR projection if enabled
      if (data.has_rsr && !data.rsr_projection.empty()) {
        std::vector<double> local_proj(data.rsr_n, 0.0);
        std::vector<double> regional_proj(data.rsr_n, 0.0);
        for (int i = 0; i < data.rsr_n; i++) {
          for (int j = 0; j < data.rsr_n; j++) {
            local_proj[i] += data.rsr_projection[i * data.rsr_n + j] * w_local_vec[j];
            regional_proj[i] += data.rsr_projection[i * data.rsr_n + j] * w_regional_vec[j];
          }
        }
        w_local_vec = local_proj;
        w_regional_vec = regional_proj;
      }

      log_post += tulpa_gp::multiscale_gp_log_lik(w_local_vec, w_regional_vec,
                                                    sigma2_local, phi_local,
                                                    sigma2_regional, phi_regional,
                                                    data.multiscale_gp_data);
    }
  }

  // HSGP (Hilbert Space GP) priors
  double sigma2_hsgp = 1.0, lengthscale_hsgp = 1.0;
  std::vector<double> hsgp_beta;
  // hsgp_f lives in state (aliased above).

  if (layout.is_hsgp && data.has_hsgp) {
    double log_sigma2 = params[layout.log_sigma2_hsgp_idx];
    double log_lengthscale = params[layout.log_lengthscale_hsgp_idx];
    sigma2_hsgp = std::exp(log_sigma2);
    lengthscale_hsgp = std::exp(log_lengthscale);

    // Extract beta coefficients
    int m_total = data.hsgp_data.m_total;
    hsgp_beta.resize(m_total);
    for (int j = 0; j < m_total; j++) {
      hsgp_beta[j] = params[layout.hsgp_beta_start + j];
    }

    // PC prior on sigma: P(sigma > 1) = 0.01 -> rate = 4.6
    // log p(sigma) = log(rate) - rate*sigma - log(2*sigma)
    // d/d(log_sigma2) includes Jacobian
    double sigma = std::sqrt(sigma2_hsgp);
    double rate_sigma = 4.6;
    log_post += std::log(rate_sigma) - rate_sigma * sigma - std::log(2.0 * sigma);
    log_post += log_sigma2 * 0.5;  // Jacobian: d(sigma)/d(log_sigma2) = 0.5*sigma

    // LogNormal(0, 1) prior on lengthscale
    // log p(ell) = -0.5 * log(ell)^2 - log(ell)
    log_post += -0.5 * log_lengthscale * log_lengthscale - log_lengthscale;
    log_post += log_lengthscale;  // Jacobian for log transform

    // N(0, I) prior on beta
    log_post += tulpa_hsgp::hsgp_log_prior_beta(hsgp_beta);

    // Evaluate HSGP spatial effect: f = Phi * sqrt(S) * beta
    tulpa_hsgp::hsgp_evaluate(hsgp_beta, sigma2_hsgp, lengthscale_hsgp,
                                data.hsgp_data, hsgp_f);
  }

  // Multi-scale temporal priors
  double sigma2_trend = 1.0, sigma2_seasonal = 1.0, sigma2_short = 1.0;
  double rho_short = 0.5;
  const double* trend = nullptr;
  const double* seasonal = nullptr;
  const double* short_term = nullptr;

  if (layout.has_multiscale_temporal) {
    std::vector<double> trend_vec, seasonal_vec, short_term_vec;

    // Trend component
    if (layout.log_sigma2_trend_idx >= 0) {
      double log_sigma2_trend = params[layout.log_sigma2_trend_idx];
      sigma2_trend = std::exp(log_sigma2_trend);
      trend = &params[layout.trend_start];
      trend_vec.assign(trend, trend + data.multiscale_temporal_data.n_times);

      // PC prior
      log_post += tulpa_temporal::log_prior_sigma2_temporal_pc(
        sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha);
      log_post += log_sigma2_trend;
    }

    // Seasonal component
    if (layout.log_sigma2_seasonal_idx >= 0) {
      double log_sigma2_seasonal = params[layout.log_sigma2_seasonal_idx];
      sigma2_seasonal = std::exp(log_sigma2_seasonal);
      seasonal = &params[layout.seasonal_start];
      seasonal_vec.assign(seasonal, seasonal + data.multiscale_temporal_data.seasonal_period);

      // PC prior
      log_post += tulpa_temporal::log_prior_sigma2_temporal_pc(
        sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha);
      log_post += log_sigma2_seasonal;
    }

    // Short-term component
    if (layout.log_sigma2_short_idx >= 0) {
      double log_sigma2_short = params[layout.log_sigma2_short_idx];
      sigma2_short = std::exp(log_sigma2_short);
      short_term = &params[layout.short_term_start];
      short_term_vec.assign(short_term, short_term + data.multiscale_temporal_data.n_times);

      // PC prior
      log_post += tulpa_temporal::log_prior_sigma2_temporal_pc(
        sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha);
      log_post += log_sigma2_short;

      // AR1 rho parameter
      if (layout.logit_rho_short_idx >= 0) {
        double logit_rho_short = params[layout.logit_rho_short_idx];
        rho_short = 2.0 / (1.0 + std::exp(-logit_rho_short)) - 1.0;  // Map to (-1, 1)

        // Prior on rho (Beta(2,2) on transformed scale)
        log_post += tulpa_temporal::log_prior_rho(rho_short, 2.0, 2.0);
        // Jacobian for logit transform
        double x = (rho_short + 1.0) / 2.0;
        log_post += std::log(x) + std::log(1.0 - x);
      }
    }

    // Multi-scale temporal log-likelihood
    log_post += tulpa_temporal::multiscale_temporal_log_lik(
      trend_vec, seasonal_vec, short_term_vec,
      sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
      data.multiscale_temporal_data);
  }
