  // Skip flag preserves the original API: when true (gradient-only callers),
  // we don't populate the heap-backed obs-context buffers.
  const bool skip_obs_loop = !populate_obs_state;

  // Alias state-owned buffers to preserve the original local-variable names
  // throughout the body. Pointers stored into `ctx` at the end of this
  // function reference these buffers, so they must outlive the call —
  // hence they live in `state`, not on this stack frame.
  std::vector<double>& re_nc_flat               = state.re_nc_flat;
  std::vector<double>& temporal_f_nc            = state.temporal_f_nc;
  std::vector<double>& gp_w_nc_buf              = state.gp_w_nc_buf;
  std::vector<double>& msgp_hsgp_f_local        = state.msgp_hsgp_f_local;
  std::vector<double>& msgp_hsgp_f_regional     = state.msgp_hsgp_f_regional;
  std::vector<double>& hsgp_f                   = state.hsgp_f;
  std::vector<double>& latent_sigma             = state.latent_sigma;
  std::vector<double>& latent_factors_vec       = state.latent_factors_vec;
  std::vector<double>& tvc_eta                  = state.tvc_eta;

  // Extract parameters
  const double* beta_num = &params[layout.legacy.beta_num_start];
  const double* beta_denom = &params[layout.legacy.beta_denom_start];

  double log_sigma_re = 0.0, sigma_re = 1.0;
  const double* re = nullptr;
  if (layout.has_re) {
    log_sigma_re = params[layout.log_sigma_re_idx];
    sigma_re = std::exp(log_sigma_re);
    re = &params[layout.re_start];
  }

  double phi_num = 1.0, phi_denom = 1.0;
  double log_phi_num = 0.0, log_phi_denom = 0.0;
  if (layout.legacy.has_phi_num) {
    log_phi_num = params[layout.legacy.log_phi_num_idx];
    phi_num = std::exp(log_phi_num);
  }
  if (layout.legacy.has_phi_denom) {
    log_phi_denom = params[layout.legacy.log_phi_denom_idx];
    phi_denom = std::exp(log_phi_denom);
  }

  // Spatial parameters
  double tau_spatial = 1.0, log_tau_spatial = 0.0;
  double sigma_s_bym2 = 1.0, sigma_u_bym2 = 1.0;
  double rho_bym2 = 0.5;  // Riebler mixing parameter
  double rho_car = 0.5;   // Proper-CAR spatial autocorrelation parameter
  const double* phi_spatial = nullptr;
  const double* theta_bym2 = nullptr;

  if (layout.has_spatial) {
    if (layout.is_bym2) {
      // Riebler reparameterization: sigma_total, rho -> sigma_s, sigma_u
      double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
      double logit_rho = params[layout.logit_rho_bym2_idx];
      rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
      sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
      sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
      if (!data.bym2_collapsed) {
        phi_spatial = &params[layout.spatial_start];
        theta_bym2 = &params[layout.theta_bym2_start];
      }
      // For collapsed BYM2: phi_spatial remains nullptr, obs loop won't add spatial
    } else if (layout.is_car_proper) {
      log_tau_spatial = params[layout.log_tau_spatial_idx];
      tau_spatial = std::exp(log_tau_spatial);
      // logit-transform rho so HMC sees an unconstrained parameter:
      // rho = lower + (upper - lower) / (1 + exp(-logit_rho))
      double logit_rho = params[layout.logit_rho_car_idx];
      double u = 1.0 / (1.0 + std::exp(-logit_rho));
      rho_car = data.car_rho_lower + (data.car_rho_upper - data.car_rho_lower) * u;
      phi_spatial = &params[layout.spatial_start];
    } else {
      log_tau_spatial = params[layout.log_tau_spatial_idx];
      tau_spatial = std::exp(log_tau_spatial);
      if (!data.icar_collapsed) {
        phi_spatial = &params[layout.spatial_start];
      }
      // For collapsed ICAR: phi_spatial remains nullptr initially
    }
  }

  double log_post = 0.0;

  // Collapsed ICAR/BYM2: find phi* via inner Laplace, point phi_spatial at mode
  if (data.icar_collapsed || data.bym2_collapsed) {
    // Pre-compute actual RE values for mode finder (NC -> actual)
    std::vector<double> re_vals_collapsed;
    if (layout.has_re) {
      int n_re = layout.re_end - layout.re_start;
      re_vals_collapsed.resize(n_re);
      const double* re_ptr = &params[layout.re_start];
      for (int g = 0; g < n_re; g++) {
        re_vals_collapsed[g] = (data.re_parameterization == 1) ? sigma_re * re_ptr[g] : re_ptr[g];
      }
    }

    auto cres = collapsed_icar_log_post_contribution(
        data.bym2_collapsed, tau_spatial,
        data.bym2_collapsed ? std::exp(params[layout.log_sigma_bym2_idx]) : 0.0,
        data.bym2_collapsed ? params[layout.logit_rho_bym2_idx] : 0.0,
        data.bym2_scale_factor, phi_num, phi_denom,
        &params[layout.legacy.beta_num_start], &params[layout.legacy.beta_denom_start],
        re_vals_collapsed.empty() ? nullptr : re_vals_collapsed.data(),
        data, collapsed_icar_ws);

    phi_spatial = cres.phi_spatial;
    theta_bym2 = cres.theta_bym2;
    log_post += cres.log_post_contribution;
  }

