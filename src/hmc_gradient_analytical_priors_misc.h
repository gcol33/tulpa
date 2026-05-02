// hmc_gradient_analytical_priors_misc.h
// Function-body fragment of compute_gradient_analytical: temporal,
// spatial, ZI, and OI prior gradients. NOT standalone-compilable;
// relies on the surrounding compute_gradient_analytical scope.

  // ============ Temporal prior gradients ============
  double log_tau_temporal = 0.0, tau_temporal = 1.0;
  double logit_rho_ar1 = 0.0, rho_ar1 = 0.5;
  int T_len = 0;
  const double* phi_temporal = nullptr;
  std::vector<double> grad_temporal_lik;  // Likelihood contribution

  if (layout.has_temporal) {
    log_tau_temporal = params[layout.log_tau_temporal_idx];
    tau_temporal = std::exp(log_tau_temporal);
    T_len = layout.temporal_end - layout.temporal_start;
    phi_temporal = &params[layout.temporal_start];
    grad_temporal_lik.assign(T_len, 0.0);

    // tau prior: Gamma(shape, rate) via log transform
    tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());

    // AR1: extract rho and add prior
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
      logit_rho_ar1 = params[layout.logit_rho_ar1_idx];
      rho_ar1 = 1.0 / (1.0 + std::exp(-logit_rho_ar1));
      // Uniform(0,1) prior on rho with logit Jacobian: grad = 1 - 2*rho
      grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;
    }
  }

  // ============ Spatial prior gradients (ICAR / BYM2 / CAR_PROPER) ============
  double log_tau_spatial = 0.0, tau_spatial = 1.0;
  double sigma_s_bym2 = 1.0, sigma_u_bym2 = 1.0;
  double rho_bym2 = 0.5;  // Riebler mixing parameter
  double rho_car = 0.5;   // Proper-CAR spatial autocorrelation
  int n_spatial = 0;
  const double* phi_spatial = nullptr;
  const double* theta_bym2 = nullptr;
  std::vector<double> grad_spatial_lik;  // Likelihood contribution

  if (layout.has_spatial) {
    n_spatial = data.n_spatial_units;
    phi_spatial = &params[layout.spatial_start];
    grad_spatial_lik.assign(n_spatial, 0.0);

    if (data.spatial_type == SpatialType::BYM2) {
      // BYM2 Riebler: derive sigma_s, sigma_u from sigma_total, rho
      double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
      double logit_rho = params[layout.logit_rho_bym2_idx];
      rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
      sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
      sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
      theta_bym2 = &params[layout.theta_bym2_start];

      // Half-Cauchy prior on sigma_total
      double ratio = sigma_total / data.sigma_re_scale;
      double ratio_sq = ratio * ratio;
      grad[layout.log_sigma_bym2_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;

      // Uniform(0,1) = Beta(1,1) on rho with logit Jacobian:
      // d/d(logit_rho) [log(rho) + log(1-rho)] = (1-rho) - rho = 1 - 2*rho
      grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;

      // Initialize theta gradients (N(0,1) prior: d/d(theta) = -theta)
      for (int s = 0; s < n_spatial; s++) {
        grad[layout.theta_bym2_start + s] = -theta_bym2[s];
      }
    } else if (data.spatial_type == SpatialType::CAR_PROPER) {
      // Proper CAR: extract tau and rho
      log_tau_spatial = params[layout.log_tau_spatial_idx];
      tau_spatial = std::exp(log_tau_spatial);
      double logit_rho = params[layout.logit_rho_car_idx];
      double u_inv = 1.0 / (1.0 + std::exp(-logit_rho));
      rho_car = data.car_rho_lower + (data.car_rho_upper - data.car_rho_lower) * u_inv;

      // Gamma prior on tau via log transform (same as ICAR)
      grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0)
                                         - data.tau_spatial_rate * tau_spatial + 1.0;

      // Logit-rho Jacobian gradient: d/d(logit_rho) [log(u) + log(1-u)] = 1 - 2u
      // (uniform Beta(1,1) prior on u in (0,1)).
      grad[layout.logit_rho_car_idx] = 1.0 - 2.0 * u_inv;
    } else {
      // ICAR: extract tau
      log_tau_spatial = params[layout.log_tau_spatial_idx];
      tau_spatial = std::exp(log_tau_spatial);

      // Gamma prior on tau via log transform
      grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0)
                                         - data.tau_spatial_rate * tau_spatial + 1.0;
    }
  }

  // ============ Zero-inflation prior gradients ============
  const double* beta_zi = nullptr;
  std::vector<double> grad_beta_zi;
  double tau_zi = 1.0;

  if (layout.has_zi && data.p_zi > 0) {
    beta_zi = &params[layout.beta_zi_start];
    tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd + 1e-10);
    grad_beta_zi.assign(data.p_zi, 0.0);

    // N(0, zi_prior_sd^2) prior on ZI coefficients
    for (int j = 0; j < data.p_zi; j++) {
      grad[layout.beta_zi_start + j] = -tau_zi * beta_zi[j];
    }
  }

  // ============ One-inflation (OI) prior gradients ============
  const double* beta_oi = nullptr;
  std::vector<double> grad_beta_oi;
  double tau_oi = 1.0;

  if (layout.has_oi && data.p_oi > 0) {
    beta_oi = &params[layout.beta_oi_start];
    tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd + 1e-10);
    grad_beta_oi.assign(data.p_oi, 0.0);

    // N(0, oi_prior_sd^2) prior on OI coefficients
    for (int j = 0; j < data.p_oi; j++) {
      grad[layout.beta_oi_start + j] = -tau_oi * beta_oi[j];
    }
  }

