// =====================================================================
// HSGP gradient (O(N*M^2) - analytical, ~50x faster than numerical)
// =====================================================================

void compute_gradient_hsgp(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Thread-local workspace eliminates 9+ heap allocations per call
    static thread_local tulpa_hsgp::HSGPWorkspace hsgp_ws;
    hsgp_ws.init(data.N, data.hsgp_data.m_total);

    // Fused log-posterior: accumulate obs log-lik during gradient loop,
    // then add prior/structural terms via skip_obs_loop=true (avoids 2nd O(N) pass)
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
  int n_params = params.size();
  grad.assign(n_params, 0.0);

  // Extract common parameters
  auto cp = extract_common_params(params, layout);
  const double* beta_num = cp.beta_num;
  const double* beta_denom = cp.beta_denom;
  double sigma_re = cp.sigma_re;
  const double* re = cp.re;
  double phi_num = cp.phi_num;
  double phi_denom = cp.phi_denom;

  // HSGP parameters
  double log_sigma2 = params[layout.log_sigma2_hsgp_idx];
  double log_lengthscale = params[layout.log_lengthscale_hsgp_idx];
  double sigma2_hsgp = std::exp(log_sigma2);
  double lengthscale_hsgp = std::exp(log_lengthscale);

  int m_total = data.hsgp_data.m_total;
  const double* hsgp_beta_ptr = &params[layout.hsgp_beta_start];

  // Evaluate HSGP spatial effect (uses workspace, zero allocation)
  tulpa_hsgp::hsgp_evaluate_ws(hsgp_beta_ptr, sigma2_hsgp, lengthscale_hsgp,
                                 data.hsgp_data, hsgp_ws);

  // Temporal parameters (for HSGP + temporal combinations)
  double tau_temporal = 0.0;
  int T_len = 0;
  const double* phi_temporal = nullptr;
  double rho_ar1 = 0.5;
  const bool has_gmrf_temporal = layout.has_temporal && !layout.is_temporal_gp &&
                                  !layout.has_multiscale_temporal && !layout.has_tvc;
  if (has_gmrf_temporal) {
    tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
    T_len = layout.temporal_end - layout.temporal_start;
    phi_temporal = &params[layout.temporal_start];
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
      rho_ar1 = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx]));
    }
  }

  // Temporal GP parameters (for HSGP + temporal GP combinations)
  const bool has_temporal_gp = layout.is_temporal_gp && layout.has_temporal;
  static thread_local tulpa_temporal_gp::TemporalGPNCWorkspace nc_ws_hsgp;
  static thread_local std::vector<double> temporal_gp_f_hsgp;
  double sigma2_tgp = 0.0, phi_tgp = 0.0, phi_lower_tgp = 0.0, phi_upper_tgp = 0.0;
  bool use_nc_tgp = false;
  int T_gp = 0, n_groups_gp = 0;
  const double* z_temporal_gp = nullptr;
  if (has_temporal_gp) {
    T_gp = data.n_times;
    n_groups_gp = data.n_temporal_groups;
    z_temporal_gp = &params[layout.temporal_start];
    T_len = layout.temporal_end - layout.temporal_start;
    int total_gp = n_groups_gp * T_gp;
    temporal_gp_f_hsgp.resize(total_gp);
    sigma2_tgp = std::exp(params[layout.log_sigma2_temporal_gp_idx]);
    double phi_gp_raw = params[layout.logit_phi_temporal_gp_idx];
    phi_lower_tgp = data.temporal_gp_phi_prior_lower;
    phi_upper_tgp = data.temporal_gp_phi_prior_upper;
    double phi_range = phi_upper_tgp - phi_lower_tgp;
    phi_tgp = phi_lower_tgp + phi_range / (1.0 + std::exp(-phi_gp_raw));
    use_nc_tgp = (data.temporal_gp_parameterization == 1);
    if (use_nc_tgp) {
      nc_ws_hsgp.init(T_gp, n_groups_gp);
      tulpa_temporal_gp::temporal_gp_nc_forward(
        z_temporal_gp, T_gp, n_groups_gp, sigma2_tgp, phi_tgp,
        data.temporal_gp_data.time_values, nc_ws_hsgp);
      for (int k = 0; k < total_gp; k++) temporal_gp_f_hsgp[k] = nc_ws_hsgp.f[k];
    } else {
      for (int k = 0; k < total_gp; k++) temporal_gp_f_hsgp[k] = z_temporal_gp[k];
    }
  }

  // TVC parameters (for HSGP + TVC combinations)
  static thread_local std::vector<double> tvc_eta_hsgp;
  int n_tvc = 0, n_tvc_times = 0, n_tvc_groups = 1, n_w = 0;
  static thread_local std::vector<double> tvc_tau_buf_h, tvc_rho_buf_h, tvc_w_flat_h;
  if (layout.has_tvc && data.has_tvc) {
    n_tvc = data.tvc_data.n_tvc;
    n_tvc_times = data.tvc_data.n_times;
    n_tvc_groups = data.tvc_data.n_groups;
    n_w = n_tvc_groups * n_tvc * n_tvc_times;
    tvc_tau_buf_h.resize(n_tvc);
    tvc_rho_buf_h.resize(n_tvc);
    tvc_w_flat_h.resize(n_w);
    for (int j = 0; j < n_tvc; j++) {
      tvc_tau_buf_h[j] = std::exp(params[layout.log_tau_tvc_start + j]);
      if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
        double logit_rho = params[layout.logit_rho_tvc_start + j];
        double u = 1.0 / (1.0 + std::exp(-logit_rho));
        tvc_rho_buf_h[j] = 2.0 * u - 1.0;
      } else {
        tvc_rho_buf_h[j] = 0.0;
      }
    }
    for (int k = 0; k < n_w; k++) tvc_w_flat_h[k] = params[layout.tvc_w_start + k];
    tvc_eta_hsgp.assign(data.N, 0.0);
    for (int i = 0; i < data.N; i++) {
      int t = data.tvc_data.time_index[i] - 1;
      int g = data.tvc_data.group_index[i] - 1;
      for (int j = 0; j < n_tvc; j++) {
        int w_idx = (g * n_tvc + j) * n_tvc_times + t;
        tvc_eta_hsgp[i] += data.tvc_data.X_tvc[i * n_tvc + j] * tvc_w_flat_h[w_idx];
      }
    }
  }

  // Zero the grad_f buffer (workspace, no allocation)
  std::memset(hsgp_ws.grad_f.data(), 0, data.N * sizeof(double));
  int grad_temporal_len = T_len;
  if (has_temporal_gp) grad_temporal_len = n_groups_gp * T_gp;
  std::vector<double> grad_temporal_lik(grad_temporal_len, 0.0);
  static thread_local std::vector<double> grad_tvc_w_h;
  if (layout.has_tvc && data.has_tvc) grad_tvc_w_h.assign(n_w, 0.0);

  // --- Prior gradients ---

  beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
  re_gradient_prior(data, layout, re, grad.data(), sigma_re);
  phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

  // HSGP prior gradients (will be added to by hsgp_compute_gradients_ws)
  double sigma = std::sqrt(sigma2_hsgp);
  double rate_sigma = 4.6;
  grad[layout.log_sigma2_hsgp_idx] = -0.5 * rate_sigma * sigma + 0.5 - 0.5;

  // LogNormal(0,1) on lengthscale
  grad[layout.log_lengthscale_hsgp_idx] = -log_lengthscale;

  // N(0, I) prior on beta: d/d(beta_j) = -beta_j
  for (int j = 0; j < m_total; j++) {
    grad[layout.hsgp_beta_start + j] = -hsgp_beta_ptr[j];
  }

  // Temporal prior on tau (Gamma) and rho (Beta) - GMRF only
  if (has_gmrf_temporal) {
    tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
      grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;
    }
  }

  // Temporal GP priors
  if (has_temporal_gp) {
    double sigma_gp = std::sqrt(sigma2_tgp);
    double rate_gp = -std::log(data.temporal_gp_sigma2_prior_alpha) / data.temporal_gp_sigma2_prior_U;
    grad[layout.log_sigma2_temporal_gp_idx] = -0.5 * rate_gp * sigma_gp + 0.5;
    double phi_range_tgp = phi_upper_tgp - phi_lower_tgp;
    grad[layout.logit_phi_temporal_gp_idx] = (phi_upper_tgp + phi_lower_tgp - 2.0 * phi_tgp) / phi_range_tgp;
    if (use_nc_tgp) {
      grad[layout.log_sigma2_temporal_gp_idx] += 0.5 * T_gp * n_groups_gp;
      double chi_tgp_prior = (phi_tgp - phi_lower_tgp) * (phi_upper_tgp - phi_tgp) /
                              (phi_tgp * phi_range_tgp);
      double jac_phi_log = 0.0;
      for (int t = 1; t < T_gp; t++) {
        double rho_t = nc_ws_hsgp.rho[t - 1];
        double rho2 = rho_t * rho_t;
        double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
        double one_minus_rho2 = std::max(1.0 - rho2, 1e-10);
        jac_phi_log -= rho2 * (dt / phi_tgp) / one_minus_rho2;
      }
      grad[layout.logit_phi_temporal_gp_idx] += jac_phi_log * n_groups_gp * chi_tgp_prior;
    }
  }

  // TVC priors
  if (layout.has_tvc && data.has_tvc) {
    double tvc_pc_rate = -std::log(0.01) / 1.0;
    for (int j = 0; j < n_tvc; j++) {
      double sigma_j = 1.0 / std::sqrt(tvc_tau_buf_h[j]);
      grad[layout.log_tau_tvc_start + j] = 0.5 * tvc_pc_rate * sigma_j + 1.5;
      if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
        double u = (tvc_rho_buf_h[j] + 1.0) / 2.0;
        grad[layout.logit_rho_tvc_start + j] = 1.0 - 2.0 * u;
      }
    }
  }

  // --- Vectorized observation loop (3-pass: Eigen matvec, scalar residuals, Eigen scatter) ---
  const double* hsgp_f = hsgp_ws.hsgp_f.data();
  double* grad_f_ptr = hsgp_ws.grad_f.data();

  const int N = data.N;
  const bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                            data.legacy.model_type == ModelType::BETA_BINOMIAL);

  // Use thread-local workspace for eta/resid buffers (zero allocation)
  vec_grad_ws.init(N);

  // --- Pass 1: Vectorized linear predictor computation ---
  using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using VectorXd = Eigen::VectorXd;

  Eigen::Map<const RowMajorMatrix> X_num(data.legacy.X_num_flat.data(), N, data.legacy.p_num);
  Eigen::Map<const VectorXd> b_num(beta_num, data.legacy.p_num);
  Eigen::Map<VectorXd> eta_n(vec_grad_ws.eta_num.data(), N);
  eta_n.noalias() = X_num * b_num;

  if (!is_binomial) {
    Eigen::Map<const RowMajorMatrix> X_denom(data.legacy.X_denom_flat.data(), N, data.legacy.p_denom);
    Eigen::Map<const VectorXd> b_denom(beta_denom, data.legacy.p_denom);
    Eigen::Map<VectorXd> eta_d(vec_grad_ws.eta_denom.data(), N);
    eta_d.noalias() = X_denom * b_denom;
  }

  // Add RE (expand to dense and vectorized add)
  if (layout.has_re) {
    for (int i = 0; i < N; i++) {
      if (data.re_group[i] > 0) {
        int g = data.re_group[i] - 1;
        double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
        vec_grad_ws.eta_num[i] += re_eff;
        if (!is_binomial) vec_grad_ws.eta_denom[i] += re_eff;
      }
    }
  }

  // Add GMRF temporal (expand to observation level)
  if (has_gmrf_temporal && !data.temporal_time_idx.empty()) {
    for (int i = 0; i < N; i++) {
      if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
        int t = data.temporal_time_idx[i] - 1;
        int g = data.temporal_group_idx[i] - 1;
        int t_idx = g * data.n_times + t;
        if (t_idx >= 0 && t_idx < T_len) {
          vec_grad_ws.eta_num[i] += phi_temporal[t_idx];
          if (!is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += phi_temporal[t_idx];
        }
      }
    }
  }

  // Add temporal GP (expand to observation level)
  if (has_temporal_gp && !data.temporal_time_idx.empty()) {
    for (int i = 0; i < N; i++) {
      if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
        int t = data.temporal_time_idx[i] - 1;
        int g = data.temporal_group_idx[i] - 1;
        int t_base = g * T_gp + t;
        if (t_base >= 0 && t_base < (int)temporal_gp_f_hsgp.size()) {
          vec_grad_ws.eta_num[i] += temporal_gp_f_hsgp[t_base];
          if (!is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += temporal_gp_f_hsgp[t_base];
        }
      }
    }
  }

  // Add TVC (expand to observation level)
  if (layout.has_tvc && data.has_tvc) {
    for (int i = 0; i < N; i++) {
      vec_grad_ws.eta_num[i] += tvc_eta_hsgp[i];
      if (!is_binomial) vec_grad_ws.eta_denom[i] += tvc_eta_hsgp[i];
    }
  }

  // Add HSGP spatial effect (vectorized)
  Eigen::Map<const VectorXd> hsgp_fv(hsgp_f, N);
  eta_n += hsgp_fv;
  if (data.hsgp_data.shared && !is_binomial) {
    Eigen::Map<VectorXd>(vec_grad_ws.eta_denom.data(), N) += hsgp_fv;
  }

  // --- Pass 2+3: Vectorized residuals + beta grads (template-dispatched) ---
  {
    double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
    vectorized::dispatch_residuals_and_beta_grads(
        data, layout,
        vec_grad_ws.eta_num.data(), vec_grad_ws.eta_denom.data(),
        vec_grad_ws.resid_num.data(), vec_grad_ws.resid_denom.data(),
        grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
        obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws);
  }

  // Accumulate grad_f for HSGP from residuals
  for (int i = 0; i < N; i++) {
    grad_f_ptr[i] = data.hsgp_data.shared
        ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
        : vec_grad_ws.resid_num[i];
  }

  // RE gradients (scatter from residuals to group-level)
  if (layout.has_re) {
    for (int i = 0; i < N; i++) {
      scatter_re_gradient(data, layout, i, vec_grad_ws.resid_num[i],
                          vec_grad_ws.resid_denom[i], grad.data());
    }
  }

  // Temporal likelihood gradients (scatter to temporal buffer - GMRF or GP)
  if ((has_gmrf_temporal || has_temporal_gp) && !data.temporal_time_idx.empty()) {
    for (int i = 0; i < N; i++) {
      if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
        int t = data.temporal_time_idx[i] - 1;
        int g = data.temporal_group_idx[i] - 1;
        int t_base = has_temporal_gp ? (g * T_gp + t) : (g * data.n_times + t);
        if (t_base >= 0 && t_base < grad_temporal_len) {
          double lik_grad = data.temporal_shared ?
            (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i]) :
            vec_grad_ws.resid_num[i];
          grad_temporal_lik[t_base] += lik_grad;
        }
      }
    }
  }

  // TVC likelihood gradients (scatter to TVC buffer)
  if (layout.has_tvc && data.has_tvc) {
    for (int i = 0; i < N; i++) {
      double dLL_shared = vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i];
      int t = data.tvc_data.time_index[i] - 1;
      int g = data.tvc_data.group_index[i] - 1;
      for (int j = 0; j < n_tvc; j++) {
        int w_idx = (g * n_tvc + j) * n_tvc_times + t;
        grad_tvc_w_h[w_idx] += dLL_shared * data.tvc_data.X_tvc[i * n_tvc + j];
      }
    }
  }

  // Compute HSGP parameter gradients using workspace (zero allocation)
  double hsgp_grad_log_sigma2, hsgp_grad_log_lengthscale;
  tulpa_hsgp::hsgp_compute_gradients_ws(hsgp_beta_ptr, sigma2_hsgp, lengthscale_hsgp,
                                          data.hsgp_data, hsgp_ws,
                                          hsgp_grad_log_sigma2, hsgp_grad_log_lengthscale);

  // Add likelihood contribution to HSGP gradients
  for (int j = 0; j < m_total; j++) {
    grad[layout.hsgp_beta_start + j] += hsgp_ws.grad_beta_out[j];
  }
  grad[layout.log_sigma2_hsgp_idx] += hsgp_grad_log_sigma2;
  grad[layout.log_lengthscale_hsgp_idx] += hsgp_grad_log_lengthscale;

  // Temporal GMRF gradients
  if (has_gmrf_temporal && T_len > 0) {
    temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                             phi_temporal, T_len, grad_temporal_lik.data(), grad.data());
  }

  // Temporal GP backward pass
  if (has_temporal_gp) {
    int T_len_gp = layout.temporal_end - layout.temporal_start;
    if (use_nc_tgp) {
      for (int k = 0; k < n_groups_gp * T_gp; k++)
        nc_ws_hsgp.dL_df[k] = grad_temporal_lik[k];
      double grad_log_sigma2_gp = 0.0, grad_log_phi_gp = 0.0;
      tulpa_temporal_gp::temporal_gp_nc_backward(
        z_temporal_gp, T_gp, n_groups_gp, sigma2_tgp, phi_tgp,
        data.temporal_gp_data.time_values, nc_ws_hsgp,
        &grad[layout.temporal_start], grad_log_sigma2_gp, grad_log_phi_gp);
      grad[layout.log_sigma2_temporal_gp_idx] += grad_log_sigma2_gp;
      double chi_tgp = (phi_tgp - phi_lower_tgp) * (phi_upper_tgp - phi_tgp) /
                        (phi_tgp * (phi_upper_tgp - phi_lower_tgp));
      grad[layout.logit_phi_temporal_gp_idx] += grad_log_phi_gp * chi_tgp;
    } else {
      for (int k = 0; k < T_len_gp; k++)
        grad[layout.temporal_start + k] = grad_temporal_lik[k];
    }
  }

  // TVC structural prior gradients
  if (layout.has_tvc && data.has_tvc) {
    static thread_local tulpa_tvc::TVCGradientWS tvc_grad_ws_h;
    static thread_local std::vector<double> tvc_grad_w_buf_h, tvc_grad_log_tau_buf_h;
    static thread_local std::vector<double> tvc_grad_logit_rho_buf_h, tvc_grad_w_jg_buf_h, tvc_d_buf_h;
    tvc_grad_w_buf_h.assign(n_w, 0.0);
    tvc_grad_log_tau_buf_h.assign(n_tvc, 0.0);
    tvc_grad_logit_rho_buf_h.assign(n_tvc, 0.0);
    tvc_grad_w_jg_buf_h.resize(n_tvc_times);
    tvc_d_buf_h.resize(n_tvc_times);
    tvc_grad_ws_h.grad_w = tvc_grad_w_buf_h.data();
    tvc_grad_ws_h.grad_log_tau = tvc_grad_log_tau_buf_h.data();
    tvc_grad_ws_h.grad_logit_rho = tvc_grad_logit_rho_buf_h.data();
    tvc_grad_ws_h.grad_w_jg = tvc_grad_w_jg_buf_h.data();
    tvc_grad_ws_h.d_buf = tvc_d_buf_h.data();
    tvc_grad_ws_h.n_w = n_w;
    tvc_grad_ws_h.n_tvc = n_tvc;
    tulpa_tvc::tvc_prior_gradients_ws(
      tvc_w_flat_h.data(), data.tvc_data,
      tvc_tau_buf_h.data(), tvc_rho_buf_h.data(), tvc_grad_ws_h);
    for (int k = 0; k < n_w; k++)
      grad[layout.tvc_w_start + k] += grad_tvc_w_h[k] + tvc_grad_w_buf_h[k];
    for (int j = 0; j < n_tvc; j++) {
      grad[layout.log_tau_tvc_start + j] += tvc_grad_log_tau_buf_h[j];
      if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1)
        grad[layout.logit_rho_tvc_start + j] += tvc_grad_logit_rho_buf_h[j];
    }
  }

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}

// =====================================================================
// HSGP-MSGP gradient (hand-coded)
// Two independent HSGP evaluations with shared basis matrix
// =====================================================================

void compute_gradient_msgp_hsgp(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Two thread-local HSGP workspaces (one per scale)
    static thread_local tulpa_hsgp::HSGPWorkspace ws_local, ws_regional;
    ws_local.init(data.N, data.msgp_hsgp_data.m_total);
    ws_regional.init(data.N, data.msgp_hsgp_data.m_total);

    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    // Extract common parameters
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // HSGP-MSGP parameters
    double log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
    double log_ls_local = params[layout.log_phi_gp_local_idx];
    double sigma2_local = std::exp(log_sigma2_local);
    double ls_local = std::exp(log_ls_local);

    double log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
    double log_ls_regional = params[layout.log_phi_gp_regional_idx];
    double sigma2_regional = std::exp(log_sigma2_regional);
    double ls_regional = std::exp(log_ls_regional);

    int m_total = data.msgp_hsgp_data.m_total;
    const double* beta_local = &params[layout.gp_local_start];
    const double* beta_regional = &params[layout.gp_regional_start];

    // Evaluate HSGP spatial effects for both scales
    tulpa_hsgp::hsgp_evaluate_ws(beta_local, sigma2_local, ls_local,
                                   data.msgp_hsgp_data, ws_local);
    tulpa_hsgp::hsgp_evaluate_ws(beta_regional, sigma2_regional, ls_regional,
                                   data.msgp_hsgp_data, ws_regional);

    // Classify temporal type (mutually exclusive)
    const bool has_gmrf_temporal = layout.has_temporal && !layout.is_temporal_gp &&
                                   !layout.has_multiscale_temporal && !layout.has_tvc;
    const bool has_temporal_gp = layout.is_temporal_gp && layout.has_temporal;
    const bool has_tvc = layout.has_tvc && data.has_tvc;
    const bool has_ms_temporal = layout.has_multiscale_temporal;

    // --- GMRF temporal (RW1/RW2/AR1) ---
    double tau_temporal = 0.0;
    int T_len = 0;
    const double* phi_temporal = nullptr;
    double rho_ar1 = 0.5;
    if (has_gmrf_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_len = layout.temporal_end - layout.temporal_start;
        phi_temporal = &params[layout.temporal_start];
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
            rho_ar1 = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx]));
        }
    }

    // --- Temporal GP (NC parameterization) ---
    static thread_local tulpa_temporal_gp::TemporalGPNCWorkspace nc_ws_msgp;
    static thread_local std::vector<double> temporal_gp_f_msgp;
    int T_gp = 0, n_groups_gp = 0;
    const double* z_temporal_gp = nullptr;
    double sigma2_tgp = 0.0, phi_tgp = 0.0;
    double phi_lower_tgp = 0.0, phi_upper_tgp = 0.0;
    bool use_nc_tgp = false;
    if (has_temporal_gp) {
        T_gp = data.n_times;
        n_groups_gp = data.n_temporal_groups;
        z_temporal_gp = &params[layout.temporal_start];
        int total_gp = n_groups_gp * T_gp;
        temporal_gp_f_msgp.resize(total_gp);

        sigma2_tgp = std::exp(params[layout.log_sigma2_temporal_gp_idx]);
        double phi_gp_raw = params[layout.logit_phi_temporal_gp_idx];
        phi_lower_tgp = data.temporal_gp_phi_prior_lower;
        phi_upper_tgp = data.temporal_gp_phi_prior_upper;
        double phi_range = phi_upper_tgp - phi_lower_tgp;
        phi_tgp = phi_lower_tgp + phi_range / (1.0 + std::exp(-phi_gp_raw));

        use_nc_tgp = (data.temporal_gp_parameterization == 1);
        if (use_nc_tgp) {
            nc_ws_msgp.init(T_gp, n_groups_gp);
            tulpa_temporal_gp::temporal_gp_nc_forward(
                z_temporal_gp, T_gp, n_groups_gp, sigma2_tgp, phi_tgp,
                data.temporal_gp_data.time_values, nc_ws_msgp);
            for (int k = 0; k < total_gp; k++) temporal_gp_f_msgp[k] = nc_ws_msgp.f[k];
        } else {
            for (int k = 0; k < total_gp; k++) temporal_gp_f_msgp[k] = z_temporal_gp[k];
        }
    }

    // --- TVC ---
    static thread_local std::vector<double> tvc_eta_precomp_msgp;
    int n_tvc = 0, n_tvc_times = 0, n_tvc_groups = 1, n_w_tvc = 0;
    static thread_local std::vector<double> tvc_tau_buf_m, tvc_rho_buf_m, tvc_w_flat_buf_m;
    if (has_tvc) {
        n_tvc = data.tvc_data.n_tvc;
        n_tvc_times = data.tvc_data.n_times;
        n_tvc_groups = data.tvc_data.n_groups;
        n_w_tvc = n_tvc_groups * n_tvc * n_tvc_times;
        tvc_tau_buf_m.resize(n_tvc);
        tvc_rho_buf_m.resize(n_tvc);
        tvc_w_flat_buf_m.resize(n_w_tvc);
        for (int j = 0; j < n_tvc; j++) {
            tvc_tau_buf_m[j] = std::exp(params[layout.log_tau_tvc_start + j]);
            if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
                double u = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_tvc_start + j]));
                tvc_rho_buf_m[j] = 2.0 * u - 1.0;
            } else {
                tvc_rho_buf_m[j] = 0.0;
            }
        }
        for (int k = 0; k < n_w_tvc; k++) tvc_w_flat_buf_m[k] = params[layout.tvc_w_start + k];
        tvc_eta_precomp_msgp.assign(data.N, 0.0);
        for (int i = 0; i < data.N; i++) {
            int t = data.tvc_data.time_index[i] - 1;
            int g = data.tvc_data.group_index[i] - 1;
            for (int j = 0; j < n_tvc; j++) {
                int w_idx = (g * n_tvc + j) * n_tvc_times + t;
                tvc_eta_precomp_msgp[i] += data.tvc_data.X_tvc[i * n_tvc + j] * tvc_w_flat_buf_m[w_idx];
            }
        }
    }

    // --- Multiscale temporal ---
    static thread_local std::vector<double> ms_effect_by_time_m;
    static thread_local std::vector<double> grad_trend_lik_m, grad_seasonal_lik_m, grad_short_lik_m;
    static thread_local std::vector<int> obs_t_idx_ms_m;
    int n_trend = 0, n_seasonal = 0, n_short = 0;
    const double* trend_m = nullptr;
    const double* seasonal_m = nullptr;
    const double* short_term_m = nullptr;
    double sigma2_trend = 1.0, sigma2_seasonal = 1.0, sigma2_short = 1.0;
    double rho_short = 0.5;
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        n_trend = layout.trend_end - layout.trend_start;
        n_seasonal = layout.seasonal_end - layout.seasonal_start;
        n_short = layout.short_term_end - layout.short_term_start;
        if (n_trend > 0) { trend_m = &params[layout.trend_start]; sigma2_trend = std::exp(params[layout.log_sigma2_trend_idx]); }
        if (n_seasonal > 0) { seasonal_m = &params[layout.seasonal_start]; sigma2_seasonal = std::exp(params[layout.log_sigma2_seasonal_idx]); }
        if (n_short > 0) { short_term_m = &params[layout.short_term_start]; sigma2_short = std::exp(params[layout.log_sigma2_short_idx]); }
        if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
            double u = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_short_idx]));
            rho_short = 2.0 * u - 1.0;
        }
        ms_effect_by_time_m.assign(mst.n_times, 0.0);
        for (int t = 0; t < mst.n_times; t++) {
            if (trend_m && t < n_trend) ms_effect_by_time_m[t] += trend_m[t];
            if (seasonal_m && mst.seasonal_period > 0) ms_effect_by_time_m[t] += seasonal_m[t % mst.seasonal_period];
            if (short_term_m && t < n_short) ms_effect_by_time_m[t] += short_term_m[t];
        }
        grad_trend_lik_m.assign(n_trend, 0.0);
        grad_seasonal_lik_m.assign(n_seasonal, 0.0);
        grad_short_lik_m.assign(n_short, 0.0);
        obs_t_idx_ms_m.assign(data.N, -1);
    }

    // Zero grad_f buffers
    std::memset(ws_local.grad_f.data(), 0, data.N * sizeof(double));
    int T_len_grad = has_gmrf_temporal ? T_len :
                     has_temporal_gp ? (n_groups_gp * T_gp) : 0;
    static thread_local std::vector<double> grad_temporal_lik;
    grad_temporal_lik.assign(T_len_grad, 0.0);
    static thread_local std::vector<double> grad_tvc_w_m;
    if (has_tvc) grad_tvc_w_m.assign(n_w_tvc, 0.0);

    // --- Prior gradients ---
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // PC priors on sigma for both scales
    double sigma_local = std::sqrt(sigma2_local);
    double rate_local = -std::log(data.ms_sigma2_local_prior_alpha) / data.ms_sigma2_local_prior_U;
    grad[layout.log_sigma2_gp_local_idx] = -0.5 * rate_local * sigma_local + 0.5 - 0.5;

    double sigma_regional = std::sqrt(sigma2_regional);
    double rate_regional = -std::log(data.ms_sigma2_regional_prior_alpha) / data.ms_sigma2_regional_prior_U;
    grad[layout.log_sigma2_gp_regional_idx] = -0.5 * rate_regional * sigma_regional + 0.5 - 0.5;

    // LogNormal priors on lengthscales
    double z_local = (log_ls_local - data.ms_log_ls_local_mean) / data.ms_log_ls_local_sd;
    grad[layout.log_phi_gp_local_idx] = -z_local / data.ms_log_ls_local_sd;

    double z_regional = (log_ls_regional - data.ms_log_ls_regional_mean) / data.ms_log_ls_regional_sd;
    grad[layout.log_phi_gp_regional_idx] = -z_regional / data.ms_log_ls_regional_sd;

    // N(0, I) prior on beta: d/d(beta_j) = -beta_j
    for (int j = 0; j < m_total; j++) {
        grad[layout.gp_local_start + j] = -beta_local[j];
        grad[layout.gp_regional_start + j] = -beta_regional[j];
    }

    // Temporal priors (type-specific)
    if (has_gmrf_temporal) {
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
            grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;
    }
    if (has_temporal_gp) {
        double sigma_gp = std::sqrt(sigma2_tgp);
        double rate_gp = -std::log(data.temporal_gp_sigma2_prior_alpha) / data.temporal_gp_sigma2_prior_U;
        grad[layout.log_sigma2_temporal_gp_idx] = -0.5 * rate_gp * sigma_gp + 0.5;
        double phi_range_tgp = phi_upper_tgp - phi_lower_tgp;
        grad[layout.logit_phi_temporal_gp_idx] = (phi_upper_tgp + phi_lower_tgp - 2.0 * phi_tgp) / phi_range_tgp;
        if (use_nc_tgp) {
            grad[layout.log_sigma2_temporal_gp_idx] += 0.5 * T_gp * n_groups_gp;
            double chi_tgp = (phi_tgp - phi_lower_tgp) * (phi_upper_tgp - phi_tgp) /
                             (phi_tgp * phi_range_tgp);
            double jac_phi_log = 0.0;
            for (int t = 1; t < T_gp; t++) {
                double rho_t = nc_ws_msgp.rho[t - 1];
                double rho2 = rho_t * rho_t;
                double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
                double omr2 = std::max(1.0 - rho2, 1e-10);
                jac_phi_log -= rho2 * (dt / phi_tgp) / omr2;
            }
            grad[layout.logit_phi_temporal_gp_idx] += jac_phi_log * n_groups_gp * chi_tgp;
        }
    }
    if (has_tvc) {
        double tvc_pc_rate = -std::log(0.01) / 1.0;
        for (int j = 0; j < n_tvc; j++) {
            double sigma_j = 1.0 / std::sqrt(tvc_tau_buf_m[j]);
            grad[layout.log_tau_tvc_start + j] = 0.5 * tvc_pc_rate * sigma_j + 1.5;
            if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
                double u = (tvc_rho_buf_m[j] + 1.0) / 2.0;
                grad[layout.logit_rho_tvc_start + j] = 1.0 - 2.0 * u;
            }
        }
    }
    if (has_ms_temporal) {
        if (n_trend > 0)
            grad[layout.log_sigma2_trend_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
                sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha) + 1.0;
        if (n_seasonal > 0)
            grad[layout.log_sigma2_seasonal_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
                sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha) + 1.0;
        if (n_short > 0)
            grad[layout.log_sigma2_short_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
                sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha) + 1.0;
        if (data.multiscale_temporal_data.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
            double u = (rho_short + 1.0) / 2.0;
            grad[layout.logit_rho_short_idx] = 2.0 * (1.0 - 2.0 * u);
        }
    }

    // --- Vectorized observation loop ---
    const int N = data.N;
    const bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                              data.legacy.model_type == ModelType::BETA_BINOMIAL);
    const bool shared = data.multiscale_gp_data.shared;

    vec_grad_ws.init(N);

    // Pass 1: vectorized linear predictor
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    Eigen::Map<const RowMajorMatrix> X_num(data.legacy.X_num_flat.data(), N, data.legacy.p_num);
    Eigen::Map<const VectorXd> b_num(beta_num, data.legacy.p_num);
    Eigen::Map<VectorXd> eta_n(vec_grad_ws.eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;

    if (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.legacy.X_denom_flat.data(), N, data.legacy.p_denom);
        Eigen::Map<const VectorXd> b_denom(beta_denom, data.legacy.p_denom);
        Eigen::Map<VectorXd> eta_d(vec_grad_ws.eta_denom.data(), N);
        eta_d.noalias() = X_denom * b_denom;
    }

    // Add RE
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            if (data.re_group[i] > 0) {
                int g = data.re_group[i] - 1;
                double re_eff = re_value_for_eta(re, g, sigma_re, data.re_parameterization);
                vec_grad_ws.eta_num[i] += re_eff;
                if (!is_binomial) vec_grad_ws.eta_denom[i] += re_eff;
            }
        }
    }

    // Add temporal (type-specific)
    if (has_gmrf_temporal && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) {
            if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t = data.temporal_time_idx[i] - 1;
                int g = data.temporal_group_idx[i] - 1;
                int t_idx = g * data.n_times + t;
                if (t_idx >= 0 && t_idx < T_len) {
                    vec_grad_ws.eta_num[i] += phi_temporal[t_idx];
                    if (!is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += phi_temporal[t_idx];
                }
            }
        }
    }
    if (has_temporal_gp && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) {
            if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t = data.temporal_time_idx[i] - 1;
                int g = data.temporal_group_idx[i] - 1;
                int t_base = g * T_gp + t;
                if (t_base >= 0 && t_base < (int)temporal_gp_f_msgp.size()) {
                    vec_grad_ws.eta_num[i] += temporal_gp_f_msgp[t_base];
                    if (!is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += temporal_gp_f_msgp[t_base];
                }
            }
        }
    }
    if (has_tvc) {
        for (int i = 0; i < N; i++) {
            vec_grad_ws.eta_num[i] += tvc_eta_precomp_msgp[i];
            if (!is_binomial) vec_grad_ws.eta_denom[i] += tvc_eta_precomp_msgp[i];
        }
    }
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        for (int i = 0; i < N; i++) {
            if (!mst.time_index.empty() && i < (int)mst.time_index.size() && mst.time_index[i] > 0) {
                int t_idx = mst.time_index[i] - 1;
                obs_t_idx_ms_m[i] = t_idx;
                vec_grad_ws.eta_num[i] += ms_effect_by_time_m[t_idx];
                if (!is_binomial && mst.shared) vec_grad_ws.eta_denom[i] += ms_effect_by_time_m[t_idx];
            }
        }
    }

    // Add combined HSGP-MSGP spatial effect: f_local + f_regional (vectorized)
    Eigen::Map<const VectorXd> f_local_v(ws_local.hsgp_f.data(), N);
    Eigen::Map<const VectorXd> f_regional_v(ws_regional.hsgp_f.data(), N);
    eta_n += f_local_v + f_regional_v;
    if (shared && !is_binomial) {
        Eigen::Map<VectorXd>(vec_grad_ws.eta_denom.data(), N) += f_local_v + f_regional_v;
    }

    // Pass 2+3: vectorized residuals + beta grads
    {
        double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
        vectorized::dispatch_residuals_and_beta_grads(
            data, layout,
            vec_grad_ws.eta_num.data(), vec_grad_ws.eta_denom.data(),
            vec_grad_ws.resid_num.data(), vec_grad_ws.resid_denom.data(),
            grad.data(), grad_phi_num_lik, grad_phi_denom_lik,
            obs_log_lik, fuse_lp, phi_num, phi_denom, vec_grad_ws);
    }

    // Accumulate grad_f for both HSGP scales from residuals
    // Both scales share the same grad_f (additive model)
    for (int i = 0; i < N; i++) {
        double gf = shared ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
                           : vec_grad_ws.resid_num[i];
        ws_local.grad_f[i] = gf;
    }
    // Copy to regional workspace (same values)
    std::memcpy(ws_regional.grad_f.data(), ws_local.grad_f.data(), N * sizeof(double));

    // RE gradients
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            scatter_re_gradient(data, layout, i, vec_grad_ws.resid_num[i],
                                vec_grad_ws.resid_denom[i], grad.data());
        }
    }

    // Temporal likelihood gradients (type-specific scatter)
    if ((has_gmrf_temporal || has_temporal_gp) && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) {
            if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t = data.temporal_time_idx[i] - 1;
                int g = data.temporal_group_idx[i] - 1;
                int t_idx = has_gmrf_temporal ? (g * data.n_times + t) : (g * T_gp + t);
                if (t_idx >= 0 && t_idx < T_len_grad) {
                    double lik_grad = data.temporal_shared ?
                        (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i]) :
                        vec_grad_ws.resid_num[i];
                    grad_temporal_lik[t_idx] += lik_grad;
                }
            }
        }
    }
    if (has_tvc) {
        for (int i = 0; i < N; i++) {
            double dLL = vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i];
            int t = data.tvc_data.time_index[i] - 1;
            int g = data.tvc_data.group_index[i] - 1;
            for (int j = 0; j < n_tvc; j++) {
                int w_idx = (g * n_tvc + j) * n_tvc_times + t;
                grad_tvc_w_m[w_idx] += dLL * data.tvc_data.X_tvc[i * n_tvc + j];
            }
        }
    }
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        for (int i = 0; i < N; i++) {
            if (obs_t_idx_ms_m[i] >= 0) {
                int t_idx = obs_t_idx_ms_m[i];
                double dLL = mst.shared ?
                    (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i]) :
                    vec_grad_ws.resid_num[i];
                if (trend_m && t_idx < n_trend) grad_trend_lik_m[t_idx] += dLL;
                if (seasonal_m && mst.seasonal_period > 0) {
                    int s_idx = t_idx % mst.seasonal_period;
                    if (s_idx < n_seasonal) grad_seasonal_lik_m[s_idx] += dLL;
                }
                if (short_term_m && t_idx < n_short) grad_short_lik_m[t_idx] += dLL;
            }
        }
    }

    // Compute HSGP parameter gradients for both scales
    double grad_log_sigma2_local, grad_log_ls_local;
    tulpa_hsgp::hsgp_compute_gradients_ws(beta_local, sigma2_local, ls_local,
                                            data.msgp_hsgp_data, ws_local,
                                            grad_log_sigma2_local, grad_log_ls_local);

    double grad_log_sigma2_regional, grad_log_ls_regional;
    tulpa_hsgp::hsgp_compute_gradients_ws(beta_regional, sigma2_regional, ls_regional,
                                            data.msgp_hsgp_data, ws_regional,
                                            grad_log_sigma2_regional, grad_log_ls_regional);

    // Add likelihood contribution to HSGP gradients
    for (int j = 0; j < m_total; j++) {
        grad[layout.gp_local_start + j] += ws_local.grad_beta_out[j];
        grad[layout.gp_regional_start + j] += ws_regional.grad_beta_out[j];
    }
    grad[layout.log_sigma2_gp_local_idx] += grad_log_sigma2_local;
    grad[layout.log_phi_gp_local_idx] += grad_log_ls_local;
    grad[layout.log_sigma2_gp_regional_idx] += grad_log_sigma2_regional;
    grad[layout.log_phi_gp_regional_idx] += grad_log_ls_regional;

    // Post-loop temporal gradients (type-specific)
    if (has_gmrf_temporal && T_len > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                                 phi_temporal, T_len, grad_temporal_lik.data(), grad.data());
    }
    if (has_temporal_gp) {
        if (use_nc_tgp) {
            for (int k = 0; k < n_groups_gp * T_gp; k++)
                nc_ws_msgp.dL_df[k] = grad_temporal_lik[k];
            double grad_log_sigma2_gp = 0.0, grad_log_phi_gp = 0.0;
            tulpa_temporal_gp::temporal_gp_nc_backward(
                z_temporal_gp, T_gp, n_groups_gp, sigma2_tgp, phi_tgp,
                data.temporal_gp_data.time_values, nc_ws_msgp,
                &grad[layout.temporal_start],
                grad_log_sigma2_gp, grad_log_phi_gp);
            grad[layout.log_sigma2_temporal_gp_idx] += grad_log_sigma2_gp;
            double chi_tgp = (phi_tgp - phi_lower_tgp) * (phi_upper_tgp - phi_tgp) /
                             (phi_tgp * (phi_upper_tgp - phi_lower_tgp));
            grad[layout.logit_phi_temporal_gp_idx] += grad_log_phi_gp * chi_tgp;
        } else {
            int T_len_gp = layout.temporal_end - layout.temporal_start;
            for (int k = 0; k < T_len_gp; k++)
                grad[layout.temporal_start + k] = grad_temporal_lik[k];
        }
    }
    if (has_tvc) {
        static thread_local tulpa_tvc::TVCGradientWS tvc_grad_ws_m;
        static thread_local std::vector<double> tvc_gw, tvc_glt, tvc_glr, tvc_gwjg, tvc_db;
        tvc_gw.assign(n_w_tvc, 0.0);
        tvc_glt.assign(n_tvc, 0.0);
        tvc_glr.assign(n_tvc, 0.0);
        tvc_gwjg.resize(n_tvc_times);
        tvc_db.resize(n_tvc_times);
        tvc_grad_ws_m.grad_w = tvc_gw.data();
        tvc_grad_ws_m.grad_log_tau = tvc_glt.data();
        tvc_grad_ws_m.grad_logit_rho = tvc_glr.data();
        tvc_grad_ws_m.grad_w_jg = tvc_gwjg.data();
        tvc_grad_ws_m.d_buf = tvc_db.data();
        tvc_grad_ws_m.n_w = n_w_tvc;
        tvc_grad_ws_m.n_tvc = n_tvc;
        tulpa_tvc::tvc_prior_gradients_ws(
            tvc_w_flat_buf_m.data(), data.tvc_data,
            tvc_tau_buf_m.data(), tvc_rho_buf_m.data(), tvc_grad_ws_m);
        for (int k = 0; k < n_w_tvc; k++)
            grad[layout.tvc_w_start + k] += grad_tvc_w_m[k] + tvc_gw[k];
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.log_tau_tvc_start + j] += tvc_glt[j];
            if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1)
                grad[layout.logit_rho_tvc_start + j] += tvc_glr[j];
        }
    }
    if (has_ms_temporal) {
        tulpa_temporal_grad::MultiscaleTemporalGradients ms_grads;
        tulpa_temporal_grad::multiscale_temporal_prior_gradients(
            trend_m, n_trend, seasonal_m, n_seasonal, short_term_m, n_short,
            sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
            data.multiscale_temporal_data, ms_grads);
        for (int t = 0; t < n_trend; t++) grad[layout.trend_start + t] = grad_trend_lik_m[t] + ms_grads.grad_trend[t];
        for (int t = 0; t < n_seasonal; t++) grad[layout.seasonal_start + t] = grad_seasonal_lik_m[t] + ms_grads.grad_seasonal[t];
        for (int t = 0; t < n_short; t++) grad[layout.short_term_start + t] = grad_short_lik_m[t] + ms_grads.grad_short_term[t];
        if (n_trend > 0) grad[layout.log_sigma2_trend_idx] += ms_grads.grad_log_sigma2_trend;
        if (n_seasonal > 0) grad[layout.log_sigma2_seasonal_idx] += ms_grads.grad_log_sigma2_seasonal;
        if (n_short > 0) grad[layout.log_sigma2_short_idx] += ms_grads.grad_log_sigma2_short;
        if (data.multiscale_temporal_data.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0)
            grad[layout.logit_rho_short_idx] += ms_grads.grad_logit_rho_short;
    }

    // Non-centered RE chain rule transformation
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    if (fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
}
