// hmc_gradient_vectorized_main.h
// Fragment of hmc_gradient_vectorized.h.
// Included from the hmc_gradient_vectorized.h umbrella header inside
// namespace tulpa_hmc { namespace vectorized { ... } } in hmc_gradients.cpp.
// Do NOT wrap contents in any namespace — already inside namespace vectorized.
// can_use_vectorized check + main vectorized gradient + top-level dispatcher.
#ifndef TULPA_HMC_GRADIENT_VECTORIZED_MAIN_H
#define TULPA_HMC_GRADIENT_VECTORIZED_MAIN_H

// ============================================================================
// Check whether a model configuration can use the vectorized path
// ============================================================================

inline bool can_use_vectorized(const ModelData& data, const ParamLayout& layout) {
  if (layout.has_zi || layout.has_oi) return false;
  if (layout.has_re_slopes || layout.has_re_correlated_slopes) return false;
  if (data.has_re_slopes) return false;
  if (data.spatial_type == SpatialType::GP ||
      data.spatial_type == SpatialType::MULTISCALE_GP ||
      data.spatial_type == SpatialType::HSGP) return false;
  if (data.has_svc || data.has_tvc || data.has_latent ||
      data.has_spatiotemporal || data.has_temporal_gp ||
      data.has_multiscale_temporal) return false;
  return true;
}

// ============================================================================
// Main vectorized gradient function (templated on ModelType)
// Computes observation-loop gradients for beta, RE, spatial, temporal, phi.
// Prior gradients and post-loop structural gradients (ICAR, BYM2, RW, AR1)
// are NOT computed here — they remain in compute_gradient_analytical().
//
// Temporal likelihood grads are written into grad_temporal_lik_out (caller
// handles the GMRF prior combination). Spatial likelihood grads are written
// into grad[layout.spatial_start + s] (ICAR) or both spatial and theta_bym2
// positions (BYM2).
// ============================================================================

template<ModelType MT>
void compute_obs_gradients_vectorized(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double& obs_ll,
    bool compute_lp,
    std::vector<double>& grad_temporal_lik_out,
    std::vector<double>& grad_spatial_lik_out,
    VecGradWorkspace& ws
) {
  const int N = data.N;
  ws.init(N);
  ws.precompute(data);  // Precompute lgamma(y+1), lchoose — no-op if already done

  const double* beta_num = &params[layout.legacy.beta_num_start];
  const double* beta_denom = &params[layout.legacy.beta_denom_start];
  double phi_num = layout.legacy.has_phi_num ? std::exp(params[layout.legacy.log_phi_num_idx]) : 1.0;
  double phi_denom = layout.legacy.has_phi_denom ? std::exp(params[layout.legacy.log_phi_denom_idx]) : 1.0;

  constexpr bool is_binomial = (MT == ModelType::BINOMIAL) || (MT == ModelType::BETA_BINOMIAL);

  // === Pass 1: Vectorized linear predictor computation (Eigen matvec) ===
  using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using VectorXd = Eigen::VectorXd;

  Eigen::Map<const RowMajorMatrix> X_num(data.legacy.X_num_flat.data(), N, data.legacy.p_num);
  Eigen::Map<const VectorXd> b_num(beta_num, data.legacy.p_num);
  Eigen::Map<VectorXd> eta_n(ws.eta_num.data(), N);
  eta_n.noalias() = X_num * b_num;

  if constexpr (!is_binomial) {
    Eigen::Map<const RowMajorMatrix> X_denom(data.legacy.X_denom_flat.data(), N, data.legacy.p_denom);
    Eigen::Map<const VectorXd> b_denom(beta_denom, data.legacy.p_denom);
    Eigen::Map<VectorXd> eta_d(ws.eta_denom.data(), N);
    eta_d.noalias() = X_denom * b_denom;
  }

  // Add RE (expand to dense, then vectorized add)
  // Non-centered: multiply z by sigma to get actual RE values
  double sigma_re_nc = 0.0;  // >0 when non-centered
  std::vector<double> sigma_re_terms_nc;  // per-term sigmas for crossed NC
  if (layout.has_re && data.re_parameterization == 1 && !layout.has_re_slopes) {
    if (data.n_re_terms > 1) {
      sigma_re_terms_nc.resize(data.n_re_terms);
      for (int t = 0; t < data.n_re_terms; t++) {
        sigma_re_terms_nc[t] = std::exp(params[layout.log_sigma_re_multi[t]]);
      }
    } else {
      sigma_re_nc = std::exp(params[layout.log_sigma_re_idx]);
    }
  }
  if (layout.has_re) {
    if (data.n_re_terms > 1) {
      expand_re_crossed(data, layout, params.data(), ws.effect_dense.data(), N,
                        sigma_re_terms_nc.empty() ? nullptr : sigma_re_terms_nc.data());
    } else {
      expand_re_single(data, layout, params.data(), ws.effect_dense.data(), N, sigma_re_nc);
    }
    Eigen::Map<VectorXd>(ws.eta_num.data(), N) +=
        Eigen::Map<const VectorXd>(ws.effect_dense.data(), N);
    if constexpr (!is_binomial) {
      Eigen::Map<VectorXd>(ws.eta_denom.data(), N) +=
          Eigen::Map<const VectorXd>(ws.effect_dense.data(), N);
    }
  }

  // Add temporal
  if (layout.has_temporal && !data.temporal_time_idx.empty()) {
    const double* phi_temporal = &params[layout.temporal_start];
    expand_temporal(data, phi_temporal, ws.effect_dense.data(), N);
    Eigen::Map<VectorXd>(ws.eta_num.data(), N) +=
        Eigen::Map<const VectorXd>(ws.effect_dense.data(), N);
    if constexpr (!is_binomial) {
      Eigen::Map<VectorXd>(ws.eta_denom.data(), N) +=
          Eigen::Map<const VectorXd>(ws.effect_dense.data(), N);
    }
  }

  // Add spatial
  double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0;
  if (layout.has_spatial && !data.spatial_group.empty()) {
    const double* phi_spatial = &params[layout.spatial_start];
    if (data.spatial_type == SpatialType::BYM2) {
      double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
      double logit_rho = params[layout.logit_rho_bym2_idx];
      double rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
      sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
      sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
      const double* theta_bym2 = &params[layout.theta_bym2_start];
      expand_spatial_bym2(data, phi_spatial, theta_bym2,
                          sigma_s_bym2, sigma_u_bym2,
                          ws.effect_dense.data(), N);
    } else {
      expand_spatial_icar(data, phi_spatial, ws.effect_dense.data(), N);
    }
    Eigen::Map<VectorXd>(ws.eta_num.data(), N) +=
        Eigen::Map<const VectorXd>(ws.effect_dense.data(), N);
    if constexpr (!is_binomial) {
      Eigen::Map<VectorXd>(ws.eta_denom.data(), N) +=
          Eigen::Map<const VectorXd>(ws.effect_dense.data(), N);
    }
  }

  // Build digamma/lgamma lookup tables for NB (same optimization as fused path)
  if constexpr (MT == ModelType::NEGBIN_NEGBIN) {
    ws.build_digamma_tables(phi_num, phi_denom, true);
  } else if constexpr (MT == ModelType::NEGBIN_GAMMA) {
    ws.build_digamma_tables(phi_num, phi_denom, false);
  }

  // === Pass 2: Model-specific residuals (scalar, template-specialized) ===
  double grad_phi_num_lik = 0.0;
  double grad_phi_denom_lik = 0.0;

  compute_residuals<MT>(
    N, ws.eta_num.data(), ws.eta_denom.data(), data,
    phi_num, phi_denom,
    ws.resid_num.data(), ws.resid_denom.data(),
    grad_phi_num_lik, grad_phi_denom_lik,
    obs_ll, compute_lp, ws
  );

  // === Pass 3: Vectorized gradient accumulation ===

  // Beta gradients: grad_beta += X^T * resid
  {
    Eigen::Map<const VectorXd> rn(ws.resid_num.data(), N);
    Eigen::Map<VectorXd> gb_num(&grad[layout.legacy.beta_num_start], data.legacy.p_num);
    gb_num.noalias() += X_num.transpose() * rn;
  }

  if constexpr (!is_binomial) {
    Eigen::Map<const RowMajorMatrix> X_denom(data.legacy.X_denom_flat.data(), N, data.legacy.p_denom);
    Eigen::Map<const VectorXd> rd(ws.resid_denom.data(), N);
    Eigen::Map<VectorXd> gb_denom(&grad[layout.legacy.beta_denom_start], data.legacy.p_denom);
    gb_denom.noalias() += X_denom.transpose() * rd;
  }

  // Phi gradients (with log-transform Jacobian)
  if (layout.legacy.has_phi_num) {
    grad[layout.legacy.log_phi_num_idx] += phi_num * grad_phi_num_lik;
  }
  if (layout.legacy.has_phi_denom) {
    grad[layout.legacy.log_phi_denom_idx] += phi_denom * grad_phi_denom_lik;
  }

  // RE gradients (scatter from dense residuals to grouped params)
  // accumulate_re_* writes centered likelihood gradient to grad[re+g]
  // Non-centered post-processing (chain rule) happens in compute_gradient_analytical
  if (layout.has_re) {
    if (data.n_re_terms > 1) {
      accumulate_re_gradient_crossed(data, layout,
        ws.resid_num.data(), ws.resid_denom.data(),
        is_binomial, grad.data(), N);
    } else {
      accumulate_re_gradient_single(data, layout,
        ws.resid_num.data(), ws.resid_denom.data(),
        is_binomial, grad.data(), N);
    }
  }

  // Temporal likelihood gradients → separate buffer for GMRF combination
  if (layout.has_temporal && !data.temporal_time_idx.empty()) {
    accumulate_temporal_gradient(data,
      ws.resid_num.data(), ws.resid_denom.data(),
      is_binomial, grad_temporal_lik_out.data(), N);
  }

  // Spatial likelihood gradients
  if (layout.has_spatial && !data.spatial_group.empty()) {
    int n_spatial = data.n_spatial_units;
    if (data.spatial_type == SpatialType::BYM2) {
      accumulate_spatial_gradient_bym2(data,
        sigma_s_bym2, sigma_u_bym2,
        ws.resid_num.data(), ws.resid_denom.data(),
        is_binomial,
        grad_spatial_lik_out.data(),
        &grad[layout.theta_bym2_start],
        N);
    } else {
      // ICAR
      const auto& group = data.spatial_group;
      for (int i = 0; i < N; i++) {
        if (group[i] > 0) {
          int s = group[i] - 1;
          double lik_grad = ws.resid_num[i];
          if (!is_binomial) lik_grad += ws.resid_denom[i];
          grad_spatial_lik_out[s] += lik_grad;
        }
      }
    }
  }
}

// ============================================================================
// Top-level dispatcher: switches on ModelType, calls template instantiation
// ============================================================================

inline bool dispatch_vectorized_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double& obs_ll,
    bool compute_lp,
    std::vector<double>& grad_temporal_lik_out,
    std::vector<double>& grad_spatial_lik_out,
    VecGradWorkspace& ws
) {
  if (!can_use_vectorized(data, layout)) return false;

  switch (data.legacy.model_type) {
    case ModelType::BINOMIAL:
      compute_obs_gradients_vectorized<ModelType::BINOMIAL>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::NEGBIN_NEGBIN:
      compute_obs_gradients_vectorized<ModelType::NEGBIN_NEGBIN>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::POISSON_GAMMA:
      compute_obs_gradients_vectorized<ModelType::POISSON_GAMMA>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::NEGBIN_GAMMA:
      compute_obs_gradients_vectorized<ModelType::NEGBIN_GAMMA>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::GAMMA_GAMMA:
      compute_obs_gradients_vectorized<ModelType::GAMMA_GAMMA>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::LOGNORMAL:
      compute_obs_gradients_vectorized<ModelType::LOGNORMAL>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::BETA_BINOMIAL:
      compute_obs_gradients_vectorized<ModelType::BETA_BINOMIAL>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
  }
  return false;
}

#endif  // TULPA_HMC_GRADIENT_VECTORIZED_MAIN_H
