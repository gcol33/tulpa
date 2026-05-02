// hmc_gradient_analytical_post.h
// Function-body fragment of compute_gradient_analytical: non-centered
// RE post-processing + temporal/spatial GMRF prior gradients +
// fused log-posterior output. NOT standalone-compilable; relies on
// the surrounding scope.

  // ============ Non-centered RE post-processing ============
  // At this point, grad[re+g] = prior_grad + centered_lik_grad
  // For non-centered: prior_grad = -z_g, so centered_lik = grad[re+g] + z_g
  // Transform: grad[z_g] = -z_g + sigma * centered_lik (chain rule through re = sigma*z)
  //            grad[log_sigma] += sigma * sum(z_g * centered_lik)
  if (layout.has_re && !layout.has_re_slopes && data.re_parameterization == 1) {
    int n_terms = (data.n_re_terms > 1) ? data.n_re_terms : 1;
    for (int t = 0; t < n_terms; t++) {
      int re_start_t = (n_terms > 1) ? layout.re_start_multi[t] : layout.re_start;
      int n_groups_t = (n_terms > 1) ? data.re_n_groups_multi[t] : data.n_re_groups;
      int log_sigma_idx = (n_terms > 1) ? layout.log_sigma_re_multi[t] : layout.log_sigma_re_idx;
      double sigma_t = std::exp(params[log_sigma_idx]);

      double sigma_lik_grad = 0.0;
      for (int g = 0; g < n_groups_t; g++) {
        double z_g = params[re_start_t + g];
        // Extract centered lik grad: total - prior = (grad[re+g]) - (-z_g) = grad[re+g] + z_g
        double centered_lik = grad[re_start_t + g] + z_g;
        // z gradient = prior + chain rule through sigma*z
        grad[re_start_t + g] = -z_g + sigma_t * centered_lik;
        // sigma gradient from likelihood: z_g * d_ll/d_re_g
        sigma_lik_grad += z_g * centered_lik;
      }
      // d_ll/d_log_sigma = sigma * sum(z_g * d_ll/d_re_g)
      grad[log_sigma_idx] += sigma_t * sigma_lik_grad;
    }
  }

  // ============ Temporal GMRF prior gradients ============
  if (layout.has_temporal && T_len > 0) {
    temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                             phi_temporal, T_len, grad_temporal_lik.data(), grad.data());
  }

  // ============ Spatial GMRF prior gradients (ICAR / BYM2 / CAR_PROPER) ============
  if (layout.has_spatial && n_spatial > 0) {
    // Add likelihood contribution to phi_spatial gradients
    for (int s = 0; s < n_spatial; s++) {
      grad[layout.spatial_start + s] = grad_spatial_lik[s];
    }

    // Generic prior:  -0.5 * tau * phi' Q(rho) phi  (rho=1 for ICAR/BYM2)
    // d/d(phi[i]) = -tau * (Q*phi)[i] = -tau * (D[i]*phi[i] - rho*sum_{j~i} phi[j])
    double icar_quad = 0.0;       // ICAR-style quadratic form (rho=1)
    double car_quad = 0.0;        // CAR_PROPER quadratic form: phi' Q(rho) phi
    double car_phi_W_phi = 0.0;   // phi' W phi = sum_{i~j} phi[i]*phi[j] (over directed edges)
    double current_rho = (data.spatial_type == SpatialType::CAR_PROPER) ? rho_car : 1.0;
    // BYM2 soft sum-to-zero: -0.01 * sum(phi)
    double bym2_phi_sum = 0.0;
    if (data.spatial_type == SpatialType::BYM2) {
      for (int i = 0; i < n_spatial; i++) bym2_phi_sum += phi_spatial[i];
    }
    for (int i = 0; i < n_spatial; i++) {
      double Di_phi = data.n_neighbors[i] * phi_spatial[i];
      double Wphi_i = 0.0;        // sum_{j~i} phi[j]
      int row_start = data.adj_row_ptr[i];
      int row_end = data.adj_row_ptr[i + 1];
      for (int k = row_start; k < row_end; k++) {
        int j = data.adj_col_idx[k];
        Wphi_i += phi_spatial[j];
        if (j > i) {
          double diff = phi_spatial[i] - phi_spatial[j];
          icar_quad += diff * diff;
        }
      }
      double Qphi_i = Di_phi - current_rho * Wphi_i;
      // CAR_PROPER quadratic form: phi' Q phi = sum_i phi_i * (Q*phi)_i
      car_quad += phi_spatial[i] * Qphi_i;
      // rho' W - contribution: phi[i] * sum_{j~i} phi[j]
      car_phi_W_phi += phi_spatial[i] * Wphi_i;

      if (data.spatial_type == SpatialType::BYM2) {
        // For BYM2, ICAR prior has no tau scaling (absorbed into sigma/rho).
        // BYM2 always has rho=1 in the structured part. Use Di_phi - Wphi_i.
        grad[layout.spatial_start + i] += -(Di_phi - Wphi_i) - 0.01 * bym2_phi_sum;
      } else {
        // ICAR (rho=1) or CAR_PROPER: same form, just different rho.
        grad[layout.spatial_start + i] += -tau_spatial * Qphi_i;
      }
    }

    if (data.spatial_type == SpatialType::BYM2) {
      // BYM2 Riebler: transform (grad_sigma_s, grad_sigma_u) to (grad_log_sigma, grad_logit_rho)
      // grad_sigma_s_lik = d(LL)/d(sigma_s) * sigma_s  (chain rule for log)
      // grad_sigma_u_lik = d(LL)/d(sigma_u) * sigma_u
      double grad_sigma_s_lik = 0.0;
      double grad_sigma_u_lik = 0.0;

      for (int s = 0; s < n_spatial; s++) {
        double scaled_phi = phi_spatial[s] * data.bym2_scale_factor;
        double d_LL_d_spatial = grad_spatial_lik[s] / (sigma_s_bym2 * data.bym2_scale_factor);
        grad_sigma_s_lik += d_LL_d_spatial * sigma_s_bym2 * scaled_phi;
        grad_sigma_u_lik += d_LL_d_spatial * sigma_u_bym2 * theta_bym2[s];
      }

      // grad[log_sigma] = grad_sigma_s_lik + grad_sigma_u_lik
      grad[layout.log_sigma_bym2_idx] += grad_sigma_s_lik + grad_sigma_u_lik;
      // grad[logit_rho] = 0.5 * ((1-rho)*grad_sigma_s_lik - rho*grad_sigma_u_lik)
      grad[layout.logit_rho_bym2_idx] += 0.5 * ((1.0 - rho_bym2) * grad_sigma_s_lik
                                                  - rho_bym2 * grad_sigma_u_lik);

    } else if (data.spatial_type == SpatialType::CAR_PROPER) {
      // Proper CAR tau gradient:
      //   log_post += 0.5 * J * log(tau) + 0.5 * log|Q(rho)| - 0.5 * tau * phi'Q*phi
      //   d/d(log_tau) = 0.5 * J - 0.5 * tau * phi'Q*phi
      grad[layout.log_tau_spatial_idx] += 0.5 * n_spatial - 0.5 * tau_spatial * car_quad;

      // Proper CAR rho gradient:
      //   d/drho [0.5 log|Q(rho)| - 0.5 - rho'Q(rho)rho]
      //     = -0.5 * tr(Q^{-1} W)  +  0.5 * - * rho'Wrho
      // Chain rule from rho = lower + (upper-lower)*u, u = 1/(1+exp(-logit_rho)):
      //   drho/d(logit_rho) = (upper-lower) * u * (1-u)
      double log_det_unused;
      double trace_QinvW;
      bool ok = tulpa_car_proper::car_proper_log_det_and_grad_rho(
          n_spatial, data.adj_row_ptr, data.adj_col_idx, data.n_neighbors,
          rho_car, &log_det_unused, &trace_QinvW);
      if (ok) {
        double d_logp_d_rho = -0.5 * trace_QinvW + 0.5 * tau_spatial * car_phi_W_phi;
        double rho_span = data.car_rho_upper - data.car_rho_lower;
        // Recover u from rho_car
        double u = (rho_car - data.car_rho_lower) / rho_span;
        double drho_dlogit = rho_span * u * (1.0 - u);
        grad[layout.logit_rho_car_idx] += d_logp_d_rho * drho_dlogit;
      }
    } else {
      // Plain ICAR: tau gradient
      // log_post = 0.5*(n-1)*log(tau) - 0.5*tau*quad + const
      // d/d(log_tau) = 0.5*(n-1) - 0.5*tau*quad
      grad[layout.log_tau_spatial_idx] += 0.5 * (n_spatial - 1) - 0.5 * tau_spatial * icar_quad;
    }
  }

  // Fused log-posterior output: combine prior/structural terms with observation log-lik.
  // Prior/structural terms are computed via compute_log_post with skip_obs_loop=true (O(p+S+T)).
  // Observation log-lik was accumulated inline during the gradient computation (O(N)).
  // Total: one O(N) pass instead of two.
  if (log_post_out) {
    *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
  }

