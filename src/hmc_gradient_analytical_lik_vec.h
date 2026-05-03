// hmc_gradient_analytical_lik_vec.h
// Function-body fragment of compute_gradient_analytical: likelihood
// gradient dispatch (fused / vectorized) plus the hybrid slopes path.
// NOT standalone-compilable; relies on the surrounding scope.

  // ============ Likelihood gradients (O(n)) ============

  // Try fused single-pass gradient first (best for small p <= 4).
  // Then fall back to 3-pass vectorized (better for larger p with Eigen).
  // Finally fall back to scalar loop for complex models (ZI, slopes, etc.).
  bool used_vectorized = vectorized::dispatch_fused_gradient(
      params, data, layout, grad, obs_log_lik,
      compute_lp, grad_temporal_lik, grad_spatial_lik, vec_grad_ws);

  // Fall back to 3-pass vectorized for p > 4 (Eigen matvec is beneficial)
  if (!used_vectorized) {
    used_vectorized = vectorized::dispatch_vectorized_gradient(
        params, data, layout, grad, obs_log_lik,
        compute_lp, grad_temporal_lik, grad_spatial_lik, vec_grad_ws);
  }

  // Hybrid path for slopes models (no ZI/OI): vectorize X*beta + residuals,
  // scalar loop for slopes RE expansion + gradient scatter
  if (!used_vectorized && (layout.has_re_slopes || layout.has_re_correlated_slopes) &&
      !layout.has_zi && !layout.has_oi &&
      !data.has_svc && !data.has_tvc && !data.has_latent &&
      !data.has_spatiotemporal && !data.has_temporal_gp && !data.has_multiscale_temporal &&
      data.spatial_type != SpatialType::GP &&
      data.spatial_type != SpatialType::MULTISCALE_GP &&
      data.spatial_type != SpatialType::HSGP) {

    const int N = data.N;
    const bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                              data.legacy.model_type == ModelType::BETA_BINOMIAL);

    // --- Pass 1: Vectorized eta base (Eigen matvec) ---
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;
    vec_grad_ws.init(N);

    Eigen::Map<const RowMajorMatrix> X_num(data.legacy.X_num_flat.data(), N, data.legacy.p_num);
    Eigen::Map<const VectorXd> b_num(beta_num, data.legacy.p_num);
    Eigen::Map<VectorXd> eta_n(vec_grad_ws.eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;

    if (!is_binomial && data.legacy.p_denom > 0) {
      Eigen::Map<const RowMajorMatrix> X_denom(data.legacy.X_denom_flat.data(), N, data.legacy.p_denom);
      Eigen::Map<const VectorXd> b_denom(beta_denom, data.legacy.p_denom);
      Eigen::Map<VectorXd> eta_d(vec_grad_ws.eta_denom.data(), N);
      eta_d.noalias() = X_denom * b_denom;
    } else if (!is_binomial) {
      std::memset(vec_grad_ws.eta_denom.data(), 0, N * sizeof(double));
    }

    // Pre-compute sigma values for NC slopes (avoid N * n_slopes exp() calls)
    std::vector<std::vector<double>> precomp_sigma(n_re_terms_slopes);
    if (slopes_nc) {
      for (int t = 0; t < n_re_terms_slopes; t++) {
        int n_coefs = layout.re_n_coefs_multi[t];
        precomp_sigma[t].resize(n_coefs);
        for (int c = 0; c < n_coefs; c++) {
          precomp_sigma[t][c] = std::exp(params[layout.log_sigma_re_slopes[t][c]]);
        }
      }
    }

    // Pre-compute per-term correlated-NC flags (constant across i).
    std::vector<char> term_is_corr_buf;
    const char* term_is_corr_ptr = nullptr;
    if (n_re_terms_slopes > 0 && !re_nc_flat.empty()) {
      term_is_corr_buf.assign(n_re_terms_slopes, 0);
      for (int t = 0; t < n_re_terms_slopes; t++) {
        term_is_corr_buf[t] = (t < (int)layout.re_correlated_multi.size() &&
                               layout.re_correlated_multi[t] &&
                               layout.re_n_coefs_multi[t] > 1) ? 1 : 0;
      }
      term_is_corr_ptr = term_is_corr_buf.data();
    }

    // Scalar loop: add slopes RE + spatial + temporal to eta
    // Track per-obs indices for scatter pass
    std::vector<int> obs_s_idx(N, -1);       // spatial group index
    std::vector<int> obs_t_idx(N, -1);       // temporal flat index
    for (int i = 0; i < N; i++) {
      // Slopes RE contribution (all terms - supports crossed+slopes).
      // Per-(i, t_re) assembly shared with scalar and composite phase-3 paths.
      if (layout.has_re_slopes && n_re_terms_slopes > 0) {
        for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
          double re_contrib = re_slopes_term_contribution(
              i, t_re, data, layout, params.data(),
              re_nc_flat.empty() ? nullptr : re_nc_flat.data(),
              term_is_corr_ptr,
              slopes_nc ? &precomp_sigma : nullptr,
              slopes_nc);
          vec_grad_ws.eta_num[i] += re_contrib;
          if (!is_binomial) vec_grad_ws.eta_denom[i] += re_contrib;
        }
      }

      // Spatial effect (ICAR or BYM2 only - GP/HSGP/MSGP excluded above)
      if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
        int s = data.spatial_group[i] - 1;
        obs_s_idx[i] = s;
        double spatial_eff;
        if (data.spatial_type == SpatialType::BYM2) {
          spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * phi_spatial[s] + sigma_u_bym2 * theta_bym2[s];
        } else {
          spatial_eff = phi_spatial[s];
        }
        vec_grad_ws.eta_num[i] += spatial_eff;
        if (!is_binomial) vec_grad_ws.eta_denom[i] += spatial_eff;
      }

      // Temporal effect
      if (layout.has_temporal && !data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
        int t = data.temporal_time_idx[i] - 1;
        int g = data.temporal_group_idx[i] - 1;
        int t_flat = g * data.n_times + t;
        obs_t_idx[i] = t_flat;
        vec_grad_ws.eta_num[i] += phi_temporal[t_flat];
        if (!is_binomial) vec_grad_ws.eta_denom[i] += phi_temporal[t_flat];
      }
    }

    // --- Pass 2+3: Vectorized residuals + beta grads (template-dispatched) ---
    {
      double grad_phi_num_lik_v = 0.0, grad_phi_denom_lik_v = 0.0;
      vectorized::dispatch_residuals_and_beta_grads(
          data, layout,
          vec_grad_ws.eta_num.data(), vec_grad_ws.eta_denom.data(),
          vec_grad_ws.resid_num.data(), vec_grad_ws.resid_denom.data(),
          grad.data(), grad_phi_num_lik_v, grad_phi_denom_lik_v,
          obs_log_lik, compute_lp, phi_num, phi_denom, vec_grad_ws);
    }

    // Scatter residuals to slopes RE, spatial, temporal gradient buffers
    for (int i = 0; i < N; i++) {
      double dLL_num = vec_grad_ws.resid_num[i];
      double dLL_denom = vec_grad_ws.resid_denom[i];
      double dLL_shared = dLL_num + dLL_denom;

      // Slopes RE gradient scatter (all terms - supports crossed+slopes).
      // Per-(i, t_re) scatter shared with scalar and composite phase-3 paths.
      if (layout.has_re_slopes && n_re_terms_slopes > 0) {
        for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
          re_slopes_term_scatter_impl(
              i, t_re, dLL_shared, data, layout, grad_re_slopes_lik,
              [](double* arr, int idx, double val) { arr[idx] += val; });
        }
      }

      // Spatial gradient scatter
      if (obs_s_idx[i] >= 0) {
        int s = obs_s_idx[i];
        if (data.spatial_type == SpatialType::BYM2) {
          grad_spatial_lik[s] += dLL_shared * sigma_s_bym2 * data.bym2_scale_factor;
          grad[layout.theta_bym2_start + s] += dLL_shared * sigma_u_bym2;
        } else {
          grad_spatial_lik[s] += dLL_shared;
        }
      }

      // Temporal gradient scatter
      if (obs_t_idx[i] >= 0) {
        grad_temporal_lik[obs_t_idx[i]] += data.temporal_shared ? dLL_shared : dLL_num;
      }
    }

    used_vectorized = true;
  }
