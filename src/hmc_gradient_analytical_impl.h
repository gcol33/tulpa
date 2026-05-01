// Analytical gradient for simple Poisson-Gamma models
// O(n) instead of O(n*p) - huge speedup for typical models
// =====================================================================

bool can_use_analytical_gradient(const ModelData& data, const ParamLayout& layout) {
  // Hand-coded gradients for basic models without complex structure
  bool is_basic_family = (data.legacy.model_type == ModelType::POISSON_GAMMA ||
                          data.legacy.model_type == ModelType::NEGBIN_NEGBIN ||
                          data.legacy.model_type == ModelType::NEGBIN_GAMMA ||
                          data.legacy.model_type == ModelType::BINOMIAL ||
                          data.legacy.model_type == ModelType::GAMMA_GAMMA ||
                          data.legacy.model_type == ModelType::BETA_BINOMIAL ||
                          data.legacy.model_type == ModelType::LOGNORMAL);

  // Check if spatial type is one we have hand-coded gradients for.
  // CAR_PROPER uses the same handcoded path as ICAR (with rho-aware Q),
  // so it qualifies. The CAR_PROPER-specific log|Q(rho)|/tr(Q^{-1}W) work
  // happens in the CAR_PROPER branch of the spatial-prior gradient block.
  bool spatial_is_icar_bym2 = (data.spatial_type == SpatialType::ICAR ||
                               data.spatial_type == SpatialType::BYM2 ||
                               data.spatial_type == SpatialType::CAR_PROPER);

  // Temporal is OK alone or combined with ICAR/BYM2 spatial (no spatiotemporal interaction)
  // Note: Temporal GP is excluded - use autodiff for that
  bool temporal_ok = !layout.has_temporal ||
                     (layout.has_temporal && !layout.is_temporal_gp &&
                      !layout.has_spatiotemporal &&
                      (!layout.has_spatial || spatial_is_icar_bym2));

  // Spatial is OK for ICAR/BYM2 (alone or combined with temporal)
  bool spatial_ok = !layout.has_spatial ||
                    (layout.has_spatial && spatial_is_icar_bym2 && !layout.has_spatiotemporal);

  // ZI is OK for basic models, including with ICAR/BYM2 spatial and temporal
  // Components are additive in eta; ZI modifies residuals independently
  bool zi_ok = !layout.has_zi ||
               (layout.has_zi &&
                (!layout.has_spatial || spatial_is_icar_bym2));

  // Random slopes: both correlated (|) and uncorrelated (||) are supported
  // Can combine with ICAR/BYM2 spatial, temporal, and ZI
  // Components are additive in eta; gradient scatter is independent
  bool slopes_ok = !layout.has_re_slopes ||
                   (layout.has_re_slopes &&
                    (!layout.has_spatial || spatial_is_icar_bym2));

  return (is_basic_family &&
          !layout.is_gp && !layout.is_multiscale_gp && !layout.is_hsgp &&
          !layout.is_icar_collapsed && !layout.is_bym2_collapsed &&
          temporal_ok && spatial_ok && zi_ok && slopes_ok &&
          !layout.has_latent && !layout.has_spatiotemporal &&
          !layout.has_multiscale_temporal && !layout.has_tvc &&
          !layout.has_svc &&  // SVC has its own gradient function
          data.n_re_terms >= 0);  // All RE combos: single, crossed, slopes, crossed+slopes
}

// Forward declarations of shared gradient helpers (defined after main gradient function)
struct CommonGradParams;
static inline CommonGradParams extract_common_params(
    const std::vector<double>& params, const ParamLayout& layout);
static inline void beta_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    const double* beta_num, const double* beta_denom, double* grad);
static inline void phi_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    double phi_num, double phi_denom, double* grad);
static inline void compute_obs_residuals(
    const ModelData& data, int i,
    double eta_num, double eta_denom,
    double phi_num, double phi_denom,
    double& dLL_deta_num, double& dLL_deta_denom);
static inline void scatter_beta_gradients(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom, double* grad);
static inline void scatter_re_gradient(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom, double* grad);
static inline void accumulate_phi_likelihood_grad(
    const ModelData& data, const ParamLayout& layout,
    int i, double eta_num, double eta_denom,
    double phi_num, double phi_denom, double* grad);
static inline void tau_temporal_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double* grad);
static inline void temporal_sum_to_zero_grad(
    const double* phi, int T, int base_idx, double lambda, double* grad);
static inline void temporal_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double rho_ar1,
    const double* phi_temporal, int T_len,
    const double* grad_temporal_lik, double* grad);
static inline double gp_pc_prior_grad_log_sigma2(
    double sigma2, double U, double alpha);
static inline std::pair<GPData, GPData> make_msgp_gp_views(
    const MultiscaleGPData& msgp);
static inline void spatial_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    const double* spatial_phi, double tau_spatial,
    double sigma_s_bym2, double sigma_u_bym2, double rho_bym2,
    const double* theta_bym2,
    const double* grad_spatial_lik, const double* grad_theta_lik,
    double* grad);

void compute_gradient_analytical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
  int n_params = params.size();
  grad.assign(n_params, 0.0);

  // Fused log-posterior computation: accumulate observation log-likelihood
  // alongside gradients to avoid a separate O(N) pass.
  const bool compute_lp = (log_post_out != nullptr);
  double obs_log_lik = 0.0;

  // Extract parameters
  const double* beta_num = &params[layout.legacy.beta_num_start];
  const double* beta_denom = &params[layout.legacy.beta_denom_start];

  double log_sigma_re = 0.0, sigma_re = 1.0, tau_re = 1.0;
  const double* re = nullptr;
  if (layout.has_re) {
    log_sigma_re = params[layout.log_sigma_re_idx];
    sigma_re = std::exp(log_sigma_re);
    tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    re = &params[layout.re_start];
  }

  double phi_num = 1.0, log_phi_num = 0.0;
  double phi_denom = 1.0, log_phi_denom = 0.0;
  if (layout.legacy.has_phi_num) {
    log_phi_num = params[layout.legacy.log_phi_num_idx];
    phi_num = std::exp(log_phi_num);
  }
  if (layout.legacy.has_phi_denom) {
    log_phi_denom = params[layout.legacy.log_phi_denom_idx];
    phi_denom = std::exp(log_phi_denom);
  }

  // ============ Prior gradients (cheap) ============

  // Beta priors: N(0, sigma_beta^2)
  beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());

  // sigma_re: Half-Cauchy prior with scale = data.sigma_re_scale (via log transform)
  // log_post = -log(1 + (sigma/scale)^2) + log(sigma) (Jacobian)
  // d/d(log_sigma) = -2*(sigma/scale)^2/(1+(sigma/scale)^2) + 1
  if (layout.has_re) {
    double ratio = sigma_re / data.sigma_re_scale;
    double ratio_sq = ratio * ratio;
    grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;
  }

  // phi priors: Gamma(shape, rate) via log transform
  phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

  // RE prior: N(0, sigma_re^2)
  // d/d(re[g]) = -tau_re * re[g]
  // Also accumulates contribution to sigma_re gradient
  double re_prior_grad_sigma = 0.0;
  std::vector<std::vector<double>> grad_re_slopes_lik;  // [term][g*n_coefs+c] likelihood contributions
  int n_re_terms_slopes = 0;
  bool slopes_nc = (layout.has_re_slopes && data.re_parameterization == 1);

  // Non-centered slopes: pre-computed RE values for observation loop
  std::vector<double> re_nc_flat;
  // Per-term storage for non-centered chain rule in write-back
  std::vector<std::vector<double>> nc_L_flats;   // [term] -> L_flat
  std::vector<std::vector<double>> nc_sigmas_vec; // [term] -> sigmas

  if (layout.has_re && layout.has_re_slopes && !layout.has_re_correlated_slopes) {
    // ============ Uncorrelated random slopes prior gradients ============
    n_re_terms_slopes = data.n_re_terms;
    grad_re_slopes_lik.resize(n_re_terms_slopes);

    for (int t = 0; t < n_re_terms_slopes; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = layout.re_n_coefs_multi[t];
      int re_start_t = layout.re_start_multi[t];
      grad_re_slopes_lik[t].assign(n_groups * n_coefs, 0.0);

      // Extract sigma parameters and compute priors
      for (int c = 0; c < n_coefs; c++) {
        int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
        double log_sigma_c = params[log_sigma_idx];
        double sigma_c = std::exp(log_sigma_c);

        // Half-Cauchy prior on sigma_c
        double ratio_c = sigma_c / data.sigma_re_scale;
        double ratio_c_sq = ratio_c * ratio_c;
        grad[log_sigma_idx] = -2.0 * ratio_c_sq / (1.0 + ratio_c_sq) + 1.0;

        if (slopes_nc) {
          // Non-centered: params store z ~ N(0,1). Prior: -0.5*z^2
          // No sigma contribution from z prior (chain rule applied after obs loop)
          for (int g = 0; g < n_groups; g++) {
            double z_gc = params[re_start_t + g * n_coefs + c];
            grad[re_start_t + g * n_coefs + c] = -z_gc;
          }
        } else {
          // Centered: params store re ~ N(0, sigma_c^2)
          double tau_c = 1.0 / (sigma_c * sigma_c + 1e-10);
          double sigma_grad_c = 0.0;
          for (int g = 0; g < n_groups; g++) {
            double re_gc = params[re_start_t + g * n_coefs + c];
            grad[re_start_t + g * n_coefs + c] = -tau_c * re_gc;
            sigma_grad_c += tau_c * re_gc * re_gc - 1.0;
          }
          grad[log_sigma_idx] += sigma_grad_c;
        }
      }
    }
  } else if (layout.has_re && layout.has_re_slopes && layout.has_re_correlated_slopes) {
    // ============ Correlated random slopes prior gradients ============
    // Multivariate normal with Sigma = diag(sigma) * L * L' * diag(sigma)
    // where L is lower-triangular Cholesky factor with L[i,i] = sqrt(1 - sum_{j<i} L[i,j]^2)
    // LKJ(eta=2) prior on correlation matrix
    n_re_terms_slopes = data.n_re_terms;
    grad_re_slopes_lik.resize(n_re_terms_slopes);

    for (int t = 0; t < n_re_terms_slopes; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = layout.re_n_coefs_multi[t];
      int re_start_t = layout.re_start_multi[t];
      bool is_correlated = layout.re_correlated_multi[t];
      grad_re_slopes_lik[t].assign(n_groups * n_coefs, 0.0);

      // Extract sigma parameters
      std::vector<double> sigmas(n_coefs);
      for (int c = 0; c < n_coefs; c++) {
        int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
        sigmas[c] = std::exp(params[log_sigma_idx]);

        // Half-Cauchy prior on sigma_c: d/d(log_sigma) = -2*(sigma/scale)^2/(1+(sigma/scale)^2) + 1
        double ratio_c = sigmas[c] / data.sigma_re_scale;
        double ratio_c_sq = ratio_c * ratio_c;
        grad[log_sigma_idx] = -2.0 * ratio_c_sq / (1.0 + ratio_c_sq) + 1.0;
      }

      if (is_correlated && n_coefs > 1) {
        // Build Cholesky factor L with tanh parameterization (must match
        // compute_log_post). LKJ(eta=2) prior + L -> R + tanh-Jacobian
        // gradient is written directly into grad[chol_start..] - slots are
        // zero-initialised at function entry.
        int chol_start = layout.chol_re_start_multi[t];
        std::vector<double> L_flat(n_coefs * n_coefs, 0.0);
        if (!tulpa::build_L_from_raw(&params[chol_start], n_coefs, L_flat.data())) {
          return;  // Safety guard (shouldn't trigger with tanh)
        }
        tulpa::lkj_log_prior_grad_add(&params[chol_start], L_flat.data(), n_coefs,
                                      /*eta=*/2.0, &grad[chol_start]);

        // ---- Non-centered parameterization ----
        // Params store z ~ N(0,1). Compute re = diag(sigma) * L * z.
        if (re_nc_flat.empty()) {
          re_nc_flat.assign(params.size(), 0.0);
        }
        tulpa::compute_u_eff(L_flat.data(), n_coefs, sigmas.data(),
                              &params[re_start_t], n_groups,
                              &re_nc_flat[re_start_t]);

        // Save term data for write-back chain rule
        nc_L_flats.resize(n_re_terms_slopes);
        nc_sigmas_vec.resize(n_re_terms_slopes);
        nc_L_flats[t] = L_flat;
        nc_sigmas_vec[t] = sigmas;

        // Prior on z: N(0, I) -> grad[z_idx] = -z[g,c]
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            grad[re_start_t + g * n_coefs + c] = -params[re_start_t + g * n_coefs + c];
          }
        }
        // Sigma: Half-Cauchy prior already written above. No centered contribution.
        // In non-centered, the log-det of Jacobian (re = diag(sigma)*L*z)
        // cancels with the |Sigma|^{-1/2} normalization, so no -n_groups term.
      } else {
        // Uncorrelated term within a mixed model (e.g., intercept-only term
        // alongside correlated slopes terms in crossed+slopes)
        if (data.re_parameterization == 1) {
          // Non-centered: z ~ N(0,1), prior grad = -z
          for (int g = 0; g < n_groups; g++) {
            for (int c = 0; c < n_coefs; c++) {
              grad[re_start_t + g * n_coefs + c] = -params[re_start_t + g * n_coefs + c];
            }
          }
        } else {
          // Centered: re ~ N(0, sigma^2), prior grad = -tau*re
          for (int c = 0; c < n_coefs; c++) {
            double tau_c = 1.0 / (sigmas[c] * sigmas[c] + 1e-10);
            double sigma_grad_c = 0.0;
            for (int g = 0; g < n_groups; g++) {
              double re_gc = params[re_start_t + g * n_coefs + c];
              grad[re_start_t + g * n_coefs + c] = -tau_c * re_gc;
              sigma_grad_c += tau_c * re_gc * re_gc - 1.0;
            }
            int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
            grad[log_sigma_idx] += sigma_grad_c;
          }
        }
      }
    }
  } else if (layout.has_re && !layout.has_re_slopes) {
    // Intercept-only RE (single or crossed terms)
    int n_terms = (data.n_re_terms > 1) ? data.n_re_terms : 1;

    for (int t = 0; t < n_terms; t++) {
      // Get term-specific parameters
      int log_sigma_idx = (n_terms > 1) ? layout.log_sigma_re_multi[t] : layout.log_sigma_re_idx;
      int re_start_t = (n_terms > 1) ? layout.re_start_multi[t] : layout.re_start;
      int n_groups_t = (n_terms > 1) ? data.re_n_groups_multi[t] : data.n_re_groups;

      double log_sigma_t = params[log_sigma_idx];
      double sigma_t = std::exp(log_sigma_t);

      // Half-Cauchy prior on sigma_t (same for both parameterizations)
      double ratio_t = sigma_t / data.sigma_re_scale;
      double ratio_t_sq = ratio_t * ratio_t;
      grad[log_sigma_idx] = -2.0 * ratio_t_sq / (1.0 + ratio_t_sq) + 1.0;

      if (data.re_parameterization == 1) {
        // Non-centered: z ~ N(0, 1), prior grad = -z
        // No sigma contribution from z prior (sigma gradient comes from likelihood chain rule)
        for (int g = 0; g < n_groups_t; g++) {
          double z_g = params[re_start_t + g];
          grad[re_start_t + g] = -z_g;
        }
      } else {
        // Centered: re ~ N(0, sigma_t^2)
        double tau_t = 1.0 / (sigma_t * sigma_t + 1e-10);
        double sigma_grad_t = 0.0;
        for (int g = 0; g < n_groups_t; g++) {
          double re_g = params[re_start_t + g];
          grad[re_start_t + g] = -tau_t * re_g;
          sigma_grad_t += tau_t * re_g * re_g - 1.0;
        }
        grad[log_sigma_idx] += sigma_grad_t;
      }
    }
  }

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

    // Scalar loop: add slopes RE + spatial + temporal to eta
    // Track per-obs indices for scatter pass
    std::vector<int> obs_s_idx(N, -1);       // spatial group index
    std::vector<int> obs_t_idx(N, -1);       // temporal flat index
    for (int i = 0; i < N; i++) {
      // Slopes RE contribution (all terms - supports crossed+slopes)
      if (layout.has_re_slopes && n_re_terms_slopes > 0) {
        for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
          int re_group_idx_i = data.re_group_multi_flat[i * data.n_re_terms + t_re];
          if (re_group_idx_i <= 0) continue;
          int g = re_group_idx_i - 1;
          int n_coefs = layout.re_n_coefs_multi[t_re];
          int re_base = layout.re_start_multi[t_re] + g * n_coefs;

          bool is_corr_t = !re_nc_flat.empty() &&
                           t_re < (int)layout.re_correlated_multi.size() &&
                           layout.re_correlated_multi[t_re] && n_coefs > 1;
          bool is_uncorr_nc = !is_corr_t && slopes_nc;

          // Intercept
          double re_contrib;
          if (is_corr_t) {
            re_contrib = re_nc_flat[re_base];
          } else if (is_uncorr_nc) {
            re_contrib = precomp_sigma[t_re][0] * params[re_base];
          } else {
            re_contrib = params[re_base];
          }

          // Slope contributions (only for terms with slopes)
          int n_slopes = n_coefs - 1;
          if (n_slopes > 0 && t_re < (int)data.re_slope_matrices.size() &&
              !data.re_slope_matrices[t_re].empty()) {
            for (int s = 0; s < n_slopes; s++) {
              double x_slope = data.re_slope_matrices[t_re][i * n_slopes + s];
              double re_slope;
              if (is_corr_t) {
                re_slope = re_nc_flat[re_base + 1 + s];
              } else if (is_uncorr_nc) {
                re_slope = precomp_sigma[t_re][1 + s] * params[re_base + 1 + s];
              } else {
                re_slope = params[re_base + 1 + s];
              }
              re_contrib += re_slope * x_slope;
            }
          }

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

      // Slopes RE gradient scatter (all terms - supports crossed+slopes)
      if (layout.has_re_slopes && n_re_terms_slopes > 0) {
        for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
          int re_group_idx_i = data.re_group_multi_flat[i * data.n_re_terms + t_re];
          if (re_group_idx_i <= 0) continue;
          int g = re_group_idx_i - 1;
          int n_coefs = layout.re_n_coefs_multi[t_re];

          // Intercept gradient
          grad_re_slopes_lik[t_re][g * n_coefs] += dLL_shared;

          // Slope gradients (chain rule: d(LL)/d(re_slope) = d(LL)/d(eta) * x_slope)
          int n_slopes = n_coefs - 1;
          if (n_slopes > 0 && t_re < (int)data.re_slope_matrices.size() &&
              !data.re_slope_matrices[t_re].empty()) {
            for (int s = 0; s < n_slopes; s++) {
              double x_slope = data.re_slope_matrices[t_re][i * n_slopes + s];
              grad_re_slopes_lik[t_re][g * n_coefs + 1 + s] += dLL_shared * x_slope;
            }
          }
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

  if (!used_vectorized) {

  // Accumulators for beta gradients (will be added via X' * residual)
  std::vector<double> grad_beta_num(data.legacy.p_num, 0.0);
  std::vector<double> grad_beta_denom(data.legacy.p_denom, 0.0);
  double grad_phi_num_lik = 0.0;
  double grad_phi_denom_lik = 0.0;

  #ifdef _OPENMP
  #pragma omp parallel
  {
    std::vector<double> local_grad_beta_num(data.legacy.p_num, 0.0);
    std::vector<double> local_grad_beta_denom(data.legacy.p_denom, 0.0);
    double local_grad_phi_num = 0.0;
    double local_grad_phi_denom = 0.0;
    std::vector<double> local_grad_re(layout.has_re ? data.n_re_groups : 0, 0.0);
    // Thread-local buffer for crossed RE gradients (all terms combined)
    std::vector<double> local_grad_re_crossed(
        (layout.has_re && data.n_re_terms > 1) ? data.total_re_groups : 0, 0.0);
    double local_obs_ll = 0.0;  // Fused log-likelihood accumulator
    // Pre-allocated per-obs group index buffer for crossed RE (reused across iterations)
    std::vector<int> re_idx_multi_buf(
        (layout.has_re && data.n_re_terms > 1) ? data.n_re_terms : 0, -1);

    #pragma omp for schedule(static)
    for (int i = 0; i < data.N; i++) {
  #else
    // Pre-allocated per-obs group index buffer for crossed RE
    std::vector<int> re_idx_multi_buf(
        (layout.has_re && data.n_re_terms > 1) ? data.n_re_terms : 0, -1);
    for (int i = 0; i < data.N; i++) {
  #endif
      // Compute linear predictors
      double eta_num = tulpa_linalg::dot_product(
          &data.legacy.X_num_flat[i * data.legacy.p_num], beta_num, data.legacy.p_num);
      double eta_denom = (data.legacy.p_denom > 0) ?
          tulpa_linalg::dot_product(
              &data.legacy.X_denom_flat[i * data.legacy.p_denom], beta_denom, data.legacy.p_denom) :
          0.0;

      // Add RE if present
      int re_idx = -1;
      int n_crossed_terms = 0;
      if (layout.has_re) {
        if (layout.has_re_slopes && n_re_terms_slopes > 0) {
          // Random slopes case: loop over ALL terms (supports crossed+slopes)
          // Each term can have intercept-only (n_coefs=1) or intercept+slopes (n_coefs>1)
          for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
            int group_idx = data.re_group_multi_flat[i * data.n_re_terms + t_re];
            if (group_idx <= 0) continue;
            int g = group_idx - 1;
            int n_coefs = layout.re_n_coefs_multi[t_re];
            int re_base = layout.re_start_multi[t_re] + g * n_coefs;

            bool is_corr_t = !re_nc_flat.empty() &&
                             t_re < (int)layout.re_correlated_multi.size() &&
                             layout.re_correlated_multi[t_re] && n_coefs > 1;
            bool is_uncorr_nc = !is_corr_t && slopes_nc;

            // Intercept contribution
            double re_contrib;
            if (is_corr_t) {
              re_contrib = re_nc_flat[re_base];
            } else if (is_uncorr_nc) {
              double sigma_int = std::exp(params[layout.log_sigma_re_slopes[t_re][0]]);
              re_contrib = sigma_int * params[re_base];
            } else {
              re_contrib = params[re_base];
            }

            // Slope contributions (only for terms with slopes)
            int n_slopes = n_coefs - 1;
            if (n_slopes > 0 && t_re < (int)data.re_slope_matrices.size() &&
                !data.re_slope_matrices[t_re].empty()) {
              for (int s = 0; s < n_slopes; s++) {
                double x_slope = data.re_slope_matrices[t_re][i * n_slopes + s];
                double re_slope;
                if (is_corr_t) {
                  re_slope = re_nc_flat[re_base + 1 + s];
                } else if (is_uncorr_nc) {
                  double sigma_s = std::exp(params[layout.log_sigma_re_slopes[t_re][1 + s]]);
                  re_slope = sigma_s * params[re_base + 1 + s];
                } else {
                  re_slope = params[re_base + 1 + s];
                }
                re_contrib += re_slope * x_slope;
              }
            }

            eta_num += re_contrib;
            if (data.legacy.model_type != ModelType::BINOMIAL) {
              eta_denom += re_contrib;
            }
          }
        } else if (data.n_re_terms > 1) {
          // Crossed RE (multiple intercept-only terms)
          // Non-centered: re_val = sigma * z; centered: re_val = params directly
          n_crossed_terms = data.n_re_terms;
          for (int t = 0; t < n_crossed_terms; t++) {
            int group_idx = data.re_group_multi_flat[i * n_crossed_terms + t];
            if (group_idx > 0) {
              int g = group_idx - 1;
              re_idx_multi_buf[t] = g;
              double z_or_re = params[layout.re_start_multi[t] + g];
              double re_val = z_or_re;
              if (data.re_parameterization == 1) {
                double sigma_t = std::exp(params[layout.log_sigma_re_multi[t]]);
                re_val = sigma_t * z_or_re;
              }
              eta_num += re_val;
              if (data.legacy.model_type != ModelType::BINOMIAL) {
                eta_denom += re_val;
              }
            } else {
              re_idx_multi_buf[t] = -1;
            }
          }
        } else if (data.re_group[i] > 0) {
          // Simple intercept-only RE (single term)
          // Non-centered: re = sigma * z; centered: re = params directly
          re_idx = data.re_group[i] - 1;
          double re_val = re[re_idx];
          if (data.re_parameterization == 1) {
            re_val = sigma_re * re_val;  // re_val was z, now sigma*z
          }
          eta_num += re_val;
          if (data.legacy.model_type != ModelType::BINOMIAL) {
            eta_denom += re_val;
          }
        }
      }
      // Add temporal effect if present
      int t_idx = -1;
      if (layout.has_temporal && !data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
        int t = data.temporal_time_idx[i] - 1;
        int g = data.temporal_group_idx[i] - 1;
        t_idx = g * data.n_times + t;  // Panel temporal: flat index
        double temporal_effect = phi_temporal[t_idx];
        eta_num += temporal_effect;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          eta_denom += temporal_effect;
        }
      }

      // Add spatial effect if present
      int s_idx = -1;
      double d_spatial_d_phi = 0.0;  // Derivative of spatial_effect wrt phi_spatial
      double d_spatial_d_theta = 0.0;  // Derivative of spatial_effect wrt theta_bym2
      if (layout.has_spatial && !data.spatial_group.empty() && data.spatial_group[i] > 0) {
        s_idx = data.spatial_group[i] - 1;
        double spatial_effect;
        if (data.spatial_type == SpatialType::BYM2) {
          // BYM2: spatial_effect = sigma_s * scale * phi + sigma_u * theta
          double scaled_phi = phi_spatial[s_idx] * data.bym2_scale_factor;
          spatial_effect = sigma_s_bym2 * scaled_phi + sigma_u_bym2 * theta_bym2[s_idx];
          d_spatial_d_phi = sigma_s_bym2 * data.bym2_scale_factor;
          d_spatial_d_theta = sigma_u_bym2;
        } else {
          // ICAR: spatial_effect = phi_spatial
          spatial_effect = phi_spatial[s_idx];
          d_spatial_d_phi = 1.0;
        }
        eta_num += spatial_effect;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          eta_denom += spatial_effect;
        }
      }

      double resid_num = 0.0;
      double resid_denom = 0.0;
      double grad_phi_num_i = 0.0;
      double grad_phi_denom_i = 0.0;
      double grad_logit_zi_i = 0.0;
      double grad_logit_oi_i = 0.0;

      // Compute ZI linear predictor if applicable
      double logit_zi = 0.0;
      double zi_prob = 0.0;
      if (layout.has_zi && data.p_zi > 0) {
        logit_zi = tulpa_linalg::dot_product(
            &data.X_zi_flat[i * data.p_zi], beta_zi, data.p_zi);
        zi_prob = 1.0 / (1.0 + std::exp(-logit_zi));
      }

      // Compute OI linear predictor if applicable
      double logit_oi = 0.0;
      double oi_prob = 0.0;
      if (layout.has_oi && data.p_oi > 0) {
        logit_oi = tulpa_linalg::dot_product(
            &data.X_oi_flat[i * data.p_oi], beta_oi, data.p_oi);
        oi_prob = 1.0 / (1.0 + std::exp(-logit_oi));
      }

      if (data.legacy.model_type == ModelType::BINOMIAL) {
        // ---- BINOMIAL ----
        // p = inv_logit(eta_num), LL = y*log(p) + (n-y)*log(1-p)
        // d(LL)/d(eta) = y - n*p
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int n_trials = data.legacy.y_denom[i];
        int y_num_i = data.legacy.y_num[i];

        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_BINOMIAL) {
          // ZI-Binomial
          if (y_num_i == 0) {
            // P(Y=0) = zi + (1-zi)*(1-p)^n
            double p0_binom = std::pow(1.0 - p, n_trials);  // (1-p)^n
            double p0 = zi_prob + (1.0 - zi_prob) * p0_binom;
            // d(LL)/d(eta) = (1-zi) * d((1-p)^n)/d(eta) / p0
            // d((1-p)^n)/d(eta) = n*(1-p)^(n-1) * (-p*(1-p)) = -n*p*(1-p)^n
            resid_num = -(1.0 - zi_prob) * n_trials * p * p0_binom / p0;
            // Gradient w.r.t. logit_zi
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_binom) / p0;
          } else {
            // P(Y=y) = (1-zi) * Binomial(y|n,p)
            resid_num = y_num_i - n_trials * p;
            grad_logit_zi_i = -zi_prob;  // d/d(logit_zi) log(1-zi) = -zi
          }
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_BINOMIAL) {
          // Hurdle-Binomial
          if (y_num_i == 0) {
            // P(Y=0) = 1 - theta
            resid_num = 0.0;  // No p contribution when y=0
            grad_logit_zi_i = -zi_prob;  // d/d(logit_theta) log(1-theta) = -theta
          } else {
            // P(Y=y|Y>0) * theta = theta * TruncBinomial(y|n,p)
            double p0_binom = std::pow(1.0 - p, n_trials);
            double normalizer = 1.0 - p0_binom;
            if (normalizer < 1e-12) normalizer = 1e-12;
            // Gradient from truncated binomial
            // d(log(1-(1-p)^n))/d(eta) = n*p*(1-p)^n / (1-(1-p)^n)
            double grad_normalizer = n_trials * p * p0_binom / normalizer;
            resid_num = (y_num_i - n_trials * p) - grad_normalizer;
            grad_logit_zi_i = 1.0 - zi_prob;  // d/d(logit_theta) log(theta) = 1-theta
          }
        } else if (layout.has_oi && data.zi_type == tulpa_zi::ZIType::OI_BINOMIAL) {
          // OI-Binomial (One-inflation only)
          // P(Y=n) = oi + (1-oi) * p^n
          // P(Y=y, y<n) = (1-oi) * Binomial(y|n,p)
          if (y_num_i == n_trials) {
            // y = n (structural one or binomial one)
            double pn = std::pow(p, n_trials);  // p^n
            double P_yn = oi_prob + (1.0 - oi_prob) * pn;
            if (P_yn < 1e-12) P_yn = 1e-12;
            // d(log P)/d(eta) = (1-oi) * n * p^(n-1) * p*(1-p) / P
            //                 = (1-oi) * n * p^n * (1-p) / P
            resid_num = (1.0 - oi_prob) * n_trials * pn * (1.0 - p) / P_yn;
            // d(log P)/d(logit_oi) = oi*(1-oi)*(1 - p^n) / P
            grad_logit_oi_i = oi_prob * (1.0 - oi_prob) * (1.0 - pn) / P_yn;
          } else {
            // y < n: P(Y=y) = (1-oi) * Binomial(y|n,p)
            resid_num = y_num_i - n_trials * p;  // Standard binomial residual
            grad_logit_oi_i = -oi_prob;  // d/d(logit_oi) log(1-oi) = -oi
          }
        } else if (layout.has_oi && data.zi_type == tulpa_zi::ZIType::ZOIB) {
          // ZOIB (Zero-One Inflated Binomial) - MIXTURE MODEL
          // P(Y=0) = zi + (1-zi)*(1-oi)*(1-p)^n
          // P(Y=n) = (1-zi)*(oi + (1-oi)*p^n)
          // P(Y=y, 0<y<n) = (1-zi)*(1-oi)*Binomial(y|n,p)
          // Note: zi_prob = zi, oi_prob = oi
          if (y_num_i == 0) {
            // y = 0: P = zi + (1-zi)*(1-oi)*(1-p)^n = A + B
            double binom_zero = std::pow(1.0 - p, n_trials);  // (1-p)^n
            double A = zi_prob;  // structural zero component
            double B = (1.0 - zi_prob) * (1.0 - oi_prob) * binom_zero;  // binomial zero
            double P = A + B;
            if (P < 1e-12) P = 1e-12;

            // d(log P)/d(eta) = (1-zi)*(1-oi) * d((1-p)^n)/d(eta) / P
            // d((1-p)^n)/d(eta) = n*(1-p)^(n-1) * (-p*(1-p)) = -n*(1-p)^n * p
            double d_binom_d_eta = -n_trials * binom_zero * p;
            resid_num = (1.0 - zi_prob) * (1.0 - oi_prob) * d_binom_d_eta / P;

            // d(log P)/d(logit_zi) = [dA - B] * zi*(1-zi) / P
            // dA/d(logit_zi) = zi*(1-zi), dB/d(logit_zi) = -(1-oi)*binom_zero * zi*(1-zi)
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - (1.0 - oi_prob) * binom_zero) / P;

            // d(log P)/d(logit_oi) = dB/d(logit_oi) / P
            // dB/d(logit_oi) = -(1-zi)*binom_zero * oi*(1-oi)
            grad_logit_oi_i = -(1.0 - zi_prob) * binom_zero * oi_prob * (1.0 - oi_prob) / P;

          } else if (y_num_i == n_trials) {
            // y = n: P = (1-zi)*(oi + (1-oi)*p^n) = (1-zi)*C
            double pn = std::pow(p, n_trials);  // p^n
            double C = oi_prob + (1.0 - oi_prob) * pn;  // oi + (1-oi)*p^n
            double P = (1.0 - zi_prob) * C;
            if (P < 1e-12) P = 1e-12;

            // d(log P)/d(eta) = (1-zi)*(1-oi) * d(p^n)/d(eta) / P
            // d(p^n)/d(eta) = n*p^(n-1) * p*(1-p) = n*p^n*(1-p)
            double d_pn_d_eta = n_trials * pn * (1.0 - p);
            resid_num = (1.0 - zi_prob) * (1.0 - oi_prob) * d_pn_d_eta / P;

            // d(log P)/d(logit_zi) = d(log(1-zi))/d(logit_zi) = -zi
            grad_logit_zi_i = -zi_prob;

            // d(log P)/d(logit_oi) = dC/d(logit_oi) / C
            // dC/d(logit_oi) = oi*(1-oi) - p^n * oi*(1-oi) = oi*(1-oi)*(1 - p^n)
            grad_logit_oi_i = oi_prob * (1.0 - oi_prob) * (1.0 - pn) / C;

          } else {
            // 0 < y < n: P = (1-zi)*(1-oi)*Binomial
            // log P = log(1-zi) + log(1-oi) + log_binom
            resid_num = y_num_i - n_trials * p;  // Standard binomial residual
            grad_logit_zi_i = -zi_prob;  // d/d(logit_zi) log(1-zi) = -zi
            grad_logit_oi_i = -oi_prob;  // d/d(logit_oi) log(1-oi) = -oi
          }
        } else {
          // Standard binomial (no ZI)
          resid_num = y_num_i - n_trials * p;
        }
        // No denominator contribution for binomial

      } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
        // ---- NEGBIN_NEGBIN ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.legacy.y_num[i];
        int y_denom_i = data.legacy.y_denom[i];

        // Denominator NegBin gradient (always standard, not ZI)
        double denom_d = mu_denom + phi_denom;
        resid_denom = y_denom_i - mu_denom * (y_denom_i + phi_denom) / denom_d;
        grad_phi_denom_i = tulpa::math::portable_digamma(y_denom_i + phi_denom) - tulpa::math::portable_digamma(phi_denom)
                           + std::log(phi_denom / denom_d)
                           + (mu_denom - y_denom_i) / denom_d;

        // Numerator with ZI handling
        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_NEGBIN) {
          // ZI-NegBin numerator
          double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);

          if (y_num_i == 0) {
            // P(Y=0) = zi + (1-zi)*p0_nb
            double p0 = zi_prob + (1.0 - zi_prob) * p0_nb;
            // d(LL)/d(mu) = (1-zi) * d(p0_nb)/d(mu) / p0
            // d(p0_nb)/d(mu) = phi * (phi/(phi+mu))^phi * (-1/(phi+mu)) = -phi * p0_nb / (phi+mu)
            double d_p0_nb_d_mu = -phi_num * p0_nb / (phi_num + mu_num);
            resid_num = (1.0 - zi_prob) * d_p0_nb_d_mu * mu_num / p0;
            // Gradient w.r.t. logit_zi
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_nb) / p0;
            // phi gradient for ZI-NegBin at y=0 (complex, using approximation)
            grad_phi_num_i = (1.0 - zi_prob) * p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / p0;
          } else {
            // P(Y=y) = (1-zi) * NB(y|mu,phi)
            double denom_num = mu_num + phi_num;
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
            grad_logit_zi_i = -zi_prob;  // d/d(logit_zi) log(1-zi) = -zi
            grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
          }
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_NEGBIN) {
          // Hurdle-NegBin numerator
          if (y_num_i == 0) {
            // P(Y=0) = 1 - theta, where theta = sigmoid(logit_zi) here represents P(Y>0)
            // Note: for hurdle, logit_zi parameterizes theta = P(Y>0), so zi_prob IS theta
            resid_num = 0.0;  // No mu contribution when y=0
            grad_logit_zi_i = -zi_prob;  // d/d(logit_theta) log(1-theta) = -theta
            grad_phi_num_i = 0.0;
          } else {
            // P(Y=y|Y>0) * theta = theta * TruncNB(y|mu,phi)
            double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
            double log_normalizer = std::log(1.0 - p0_nb);
            double denom_num = mu_num + phi_num;
            // Gradient from truncated NB: NB residual MINUS normalizer correction
            // LL = log NB(y) - log(1-p0), so d(LL)/d(eta) = NB_resid - d(log(1-p0))/d(mu)*mu
            // d(log(1-p0))/d(mu) = phi*p0 / ((phi+mu)*(1-p0))
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num
                        - phi_num * p0_nb * mu_num / ((phi_num + mu_num) * (1.0 - p0_nb));
            grad_logit_zi_i = 1.0 - zi_prob;  // d/d(logit_theta) log(theta) = 1-theta
            grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
            // Truncation correction for phi gradient
            grad_phi_num_i += p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / (1.0 - p0_nb);
          }
        } else {
          // Standard NegBin (no ZI)
          double denom_num = mu_num + phi_num;
          resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
          grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                           + std::log(phi_num / denom_num)
                           + (mu_num - y_num_i) / denom_num;
        }

      } else if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
        // ---- POISSON_GAMMA ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.legacy.y_num[i];

        // Denominator: Gamma (always standard) - skip if y <= 0
        double y_denom_i = data.legacy.y_denom_cont[i];
        double grad_phi_gamma = 0.0;
        if (y_denom_i > 0.0) {
          resid_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
          double rate = phi_denom / mu_denom;
          grad_phi_gamma = std::log(rate) + 1.0 + std::log(y_denom_i)
                                  - tulpa::math::portable_digamma(phi_denom) - rate * y_denom_i / phi_denom;
        }

        // Numerator with ZI handling
        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_POISSON) {
          // ZI-Poisson numerator
          double exp_neg_mu = std::exp(-mu_num);

          if (y_num_i == 0) {
            // P(Y=0) = zi + (1-zi)*exp(-mu)
            double p0 = zi_prob + (1.0 - zi_prob) * exp_neg_mu;
            // d(LL)/d(eta) = d(LL)/d(mu) * mu = -(1-zi)*exp(-mu)*mu / p0
            resid_num = -(1.0 - zi_prob) * exp_neg_mu * mu_num / p0;
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - exp_neg_mu) / p0;
            grad_phi_denom_i = grad_phi_gamma;  // Only gamma part
          } else {
            // P(Y=y) = (1-zi) * Poisson(y|mu)
            resid_num = y_num_i - mu_num;
            grad_logit_zi_i = -zi_prob;
            grad_phi_denom_i = grad_phi_gamma;
          }
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_POISSON) {
          // Hurdle-Poisson numerator
          if (y_num_i == 0) {
            resid_num = 0.0;
            grad_logit_zi_i = -zi_prob;  // zi_prob is theta here
            grad_phi_denom_i = grad_phi_gamma;
          } else {
            // Truncated Poisson: d(LL)/d(eta) = y - mu - mu*exp(-mu)/(1-exp(-mu))
            // LL = log Poi(y) - log(1-exp(-mu)), correction is SUBTRACTED
            double exp_neg_mu = std::exp(-mu_num);
            resid_num = y_num_i - mu_num - mu_num * exp_neg_mu / (1.0 - exp_neg_mu);
            grad_logit_zi_i = 1.0 - zi_prob;
            grad_phi_denom_i = grad_phi_gamma;
          }
        } else {
          // Standard Poisson (no ZI)
          resid_num = y_num_i - mu_num;
          grad_phi_denom_i = grad_phi_gamma;
        }

      } else if (data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
        // ---- NEGBIN_GAMMA ----
        // NegBin numerator + Gamma denominator
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.legacy.y_num[i];

        // Denominator: Gamma (always standard) - skip if y <= 0
        double y_denom_i = data.legacy.y_denom_cont[i];
        double grad_phi_gamma = 0.0;
        if (y_denom_i > 0.0) {
          resid_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
          double rate = phi_denom / mu_denom;
          grad_phi_gamma = std::log(rate) + 1.0 + std::log(y_denom_i)
                                  - tulpa::math::portable_digamma(phi_denom) - rate * y_denom_i / phi_denom;
        }

        // Numerator with ZI handling (same as NEGBIN_NEGBIN)
        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_NEGBIN) {
          // ZI-NegBin numerator (same logic as NEGBIN_NEGBIN ZI case)
          double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);

          if (y_num_i == 0) {
            double p0 = zi_prob + (1.0 - zi_prob) * p0_nb;
            // d(p0_nb)/d(mu) = -phi * p0_nb / (phi+mu)
            double d_p0_nb_d_mu = -phi_num * p0_nb / (phi_num + mu_num);
            resid_num = (1.0 - zi_prob) * d_p0_nb_d_mu * mu_num / p0;
            grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_nb) / p0;
            grad_phi_num_i = (1.0 - zi_prob) * p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / p0;
          } else {
            double denom_num = mu_num + phi_num;
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
            grad_logit_zi_i = -zi_prob;
            grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
          }
          grad_phi_denom_i = grad_phi_gamma;
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_NEGBIN) {
          // Hurdle-NegBin numerator (same as NEGBIN_NEGBIN hurdle)
          if (y_num_i == 0) {
            resid_num = 0.0;
            grad_logit_zi_i = -zi_prob;  // d/d(logit_theta) log(1-theta)
            grad_phi_num_i = 0.0;
          } else {
            double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
            double denom_num = mu_num + phi_num;
            resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num
                        - phi_num * p0_nb * mu_num / ((phi_num + mu_num) * (1.0 - p0_nb));
            grad_logit_zi_i = 1.0 - zi_prob;  // d/d(logit_theta) log(theta)
            grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
            // Truncation correction for phi gradient
            grad_phi_num_i += p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / (1.0 - p0_nb);
          }
          grad_phi_denom_i = grad_phi_gamma;
        } else {
          // Standard NegBin numerator (no ZI)
          double denom_num = mu_num + phi_num;
          resid_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
          grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                           + std::log(phi_num / denom_num)
                           + (mu_num - y_num_i) / denom_num;
          grad_phi_denom_i = grad_phi_gamma;
        }

      } else if (data.legacy.model_type == ModelType::GAMMA_GAMMA) {
        // ---- GAMMA_GAMMA ----
        // Gamma requires y > 0; skip contributions for y <= 0 (matches log_lik_gamma)
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        double y_num_i = data.legacy.y_num_cont[i];
        double y_denom_i = data.legacy.y_denom_cont[i];

        if (y_num_i > 0.0) {
          resid_num = phi_num * (y_num_i / mu_num - 1.0);
          double rate_num = phi_num / mu_num;
          grad_phi_num_i = std::log(rate_num) + 1.0 + std::log(y_num_i)
                           - tulpa::math::portable_digamma(phi_num) - y_num_i / mu_num;
        }
        if (y_denom_i > 0.0) {
          resid_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
          double rate_denom = phi_denom / mu_denom;
          grad_phi_denom_i = std::log(rate_denom) + 1.0 + std::log(y_denom_i)
                             - tulpa::math::portable_digamma(phi_denom) - y_denom_i / mu_denom;
        }

      } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
        // ---- LOGNORMAL ----
        // Both numerator and denominator are Lognormal distributed
        // log(y) ~ Normal(mu, sigma^2), so y ~ Lognormal(mu, sigma^2)
        // LL = -log(y) - log(sigma) - 0.5*((log(y) - mu)/sigma)^2
        // d(LL)/d(eta) = d(LL)/d(mu) = (log(y) - mu) / sigma^2
        // (Note: eta IS mu for lognormal, so no chain rule needed)
        double mu_num = eta_num;  // mu is directly the linear predictor
        double mu_denom = eta_denom;
        double y_num_i = data.legacy.y_num_cont[i];
        double y_denom_i = data.legacy.y_denom_cont[i];
        double log_y_num = std::log(y_num_i);
        double log_y_denom = std::log(y_denom_i);

        // phi_num, phi_denom are sigma (std dev on log scale)
        double sigma_num = phi_num;
        double sigma_denom = phi_denom;
        double sigma_num_sq = sigma_num * sigma_num;
        double sigma_denom_sq = sigma_denom * sigma_denom;

        // Residuals (gradient w.r.t. mu)
        resid_num = (log_y_num - mu_num) / sigma_num_sq;
        resid_denom = (log_y_denom - mu_denom) / sigma_denom_sq;

        // Sigma gradients: d(LL)/d(sigma), NOT d(LL)/d(log_sigma)
        // Because the accumulation code multiplies by phi to get d(LL)/d(log_phi)
        // d(LL)/d(sigma) = -1/sigma + z^2/sigma = (-1 + z^2) / sigma
        double z_num = (log_y_num - mu_num) / sigma_num;
        double z_denom = (log_y_denom - mu_denom) / sigma_denom;
        grad_phi_num_i = (-1.0 + z_num * z_num) / sigma_num;
        grad_phi_denom_i = (-1.0 + z_denom * z_denom) / sigma_denom;

      } else if (data.legacy.model_type == ModelType::BETA_BINOMIAL) {
        // ---- BETA_BINOMIAL ----
        // Overdispersed binomial: y ~ BetaBinom(n, alpha, beta)
        // where p = alpha/(alpha+beta), phi = alpha + beta (concentration)
        // We parameterize: logit(p) = eta, phi = overdispersion
        // alpha = p * phi, beta_param = (1-p) * phi
        // LL = lgamma(y+alpha) + lgamma(n-y+beta) - lgamma(n+phi)
        //      - lgamma(alpha) - lgamma(beta) + lgamma(phi) + lchoose(n,y)
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int y_i = data.legacy.y_num[i];
        int n_i = data.legacy.y_denom[i];
        double alpha = p * phi_num;
        double beta_param = (1.0 - p) * phi_num;

        // d(LL)/d(eta) = d(LL)/d(p) * d(p)/d(eta) where d(p)/d(eta) = p*(1-p)
        // d(LL)/d(p) = phi * (digamma(y+alpha) - digamma(n-y+beta) - digamma(alpha) + digamma(beta))
        double psi_y_alpha = tulpa::math::portable_digamma(y_i + alpha);
        double psi_nmy_beta = tulpa::math::portable_digamma(n_i - y_i + beta_param);
        double psi_alpha = tulpa::math::portable_digamma(alpha);
        double psi_beta = tulpa::math::portable_digamma(beta_param);
        double dLL_dp = phi_num * (psi_y_alpha - psi_nmy_beta - psi_alpha + psi_beta);
        resid_num = dLL_dp * p * (1.0 - p);

        // d(LL)/d(phi) where phi = alpha + beta
        // d(LL)/d(phi) = p*digamma(y+alpha) + (1-p)*digamma(n-y+beta) - digamma(n+phi)
        //               - p*digamma(alpha) - (1-p)*digamma(beta) + digamma(phi)
        double psi_n_phi = tulpa::math::portable_digamma(n_i + phi_num);
        double psi_phi = tulpa::math::portable_digamma(phi_num);
        grad_phi_num_i = p * psi_y_alpha + (1.0 - p) * psi_nmy_beta - psi_n_phi
                         - p * psi_alpha - (1.0 - p) * psi_beta + psi_phi;

        // No denominator contribution for beta-binomial (like binomial)
      }

      // Accumulate ZI coefficient gradients
      if (layout.has_zi && data.p_zi > 0) {
        for (int j = 0; j < data.p_zi; j++) {
          #ifdef _OPENMP
          #pragma omp atomic
          grad_beta_zi[j] += data.X_zi_flat[i * data.p_zi + j] * grad_logit_zi_i;
          #else
          grad_beta_zi[j] += data.X_zi_flat[i * data.p_zi + j] * grad_logit_zi_i;
          #endif
        }
      }

      // Accumulate OI coefficient gradients
      if (layout.has_oi && data.p_oi > 0) {
        for (int j = 0; j < data.p_oi; j++) {
          #ifdef _OPENMP
          #pragma omp atomic
          grad_beta_oi[j] += data.X_oi_flat[i * data.p_oi + j] * grad_logit_oi_i;
          #else
          grad_beta_oi[j] += data.X_oi_flat[i * data.p_oi + j] * grad_logit_oi_i;
          #endif
        }
      }

      // Accumulate beta gradients: grad += X[i,:] * resid
      for (int j = 0; j < data.legacy.p_num; j++) {
        #ifdef _OPENMP
        local_grad_beta_num[j] += data.legacy.X_num_flat[i * data.legacy.p_num + j] * resid_num;
        #else
        grad_beta_num[j] += data.legacy.X_num_flat[i * data.legacy.p_num + j] * resid_num;
        #endif
      }
      // For BINOMIAL, beta_denom doesn't affect likelihood
      if (data.legacy.model_type != ModelType::BINOMIAL) {
        for (int j = 0; j < data.legacy.p_denom; j++) {
          #ifdef _OPENMP
          local_grad_beta_denom[j] += data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * resid_denom;
          #else
          grad_beta_denom[j] += data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * resid_denom;
          #endif
        }
      }

      // Accumulate RE gradient
      if (layout.has_re_slopes && n_re_terms_slopes > 0) {
        // Random slopes case: scatter to ALL terms (supports crossed+slopes)
        double re_grad_base = resid_num;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          re_grad_base += resid_denom;
        }
        for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
          int group_idx = data.re_group_multi_flat[i * data.n_re_terms + t_re];
          if (group_idx <= 0) continue;
          int g = group_idx - 1;
          int n_coefs = layout.re_n_coefs_multi[t_re];

          // Intercept gradient
          #ifdef _OPENMP
          #pragma omp atomic
          grad_re_slopes_lik[t_re][g * n_coefs] += re_grad_base;
          #else
          grad_re_slopes_lik[t_re][g * n_coefs] += re_grad_base;
          #endif

          // Slope gradients: multiply by slope design value
          int n_slopes = n_coefs - 1;
          if (n_slopes > 0 && t_re < (int)data.re_slope_matrices.size() &&
              !data.re_slope_matrices[t_re].empty()) {
            for (int s = 0; s < n_slopes; s++) {
              double x_slope = data.re_slope_matrices[t_re][i * n_slopes + s];
              double slope_grad = re_grad_base * x_slope;
              #ifdef _OPENMP
              #pragma omp atomic
              grad_re_slopes_lik[t_re][g * n_coefs + 1 + s] += slope_grad;
              #else
              grad_re_slopes_lik[t_re][g * n_coefs + 1 + s] += slope_grad;
              #endif
            }
          }
        }
      } else if (n_crossed_terms > 0) {
        // Crossed RE (multiple intercept-only terms)
        double re_grad_i = resid_num;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          re_grad_i += resid_denom;
        }
        for (int t = 0; t < n_crossed_terms; t++) {
          if (re_idx_multi_buf[t] >= 0) {
            #ifdef _OPENMP
            // Thread-local accumulation (reduced at end of parallel block)
            local_grad_re_crossed[data.re_offsets[t] + re_idx_multi_buf[t]] += re_grad_i;
            #else
            grad[layout.re_start_multi[t] + re_idx_multi_buf[t]] += re_grad_i;
            #endif
          }
        }
      } else if (re_idx >= 0) {
        // Simple intercept-only RE (single term)
        double re_grad_i = resid_num;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          re_grad_i += resid_denom;  // Shared RE affects both processes
        }
        #ifdef _OPENMP
        local_grad_re[re_idx] += re_grad_i;
        #else
        grad[layout.re_start + re_idx] += re_grad_i;
        #endif
      }

      // Accumulate temporal gradient (from likelihood)
      if (t_idx >= 0) {
        double temp_grad_i = resid_num;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          temp_grad_i += resid_denom;
        }
        #ifdef _OPENMP
        #pragma omp atomic
        grad_temporal_lik[t_idx] += temp_grad_i;
        #else
        grad_temporal_lik[t_idx] += temp_grad_i;
        #endif
      }

      // Accumulate spatial gradient (from likelihood)
      if (s_idx >= 0) {
        double lik_grad = resid_num;
        if (data.legacy.model_type != ModelType::BINOMIAL) {
          lik_grad += resid_denom;
        }
        // For ICAR: grad_spatial[s] += lik_grad (d_spatial_d_phi = 1)
        // For BYM2: grad_phi[s] += lik_grad * d_spatial_d_phi, grad_theta[s] += lik_grad * d_spatial_d_theta
        #ifdef _OPENMP
        #pragma omp atomic
        grad_spatial_lik[s_idx] += lik_grad * d_spatial_d_phi;
        #else
        grad_spatial_lik[s_idx] += lik_grad * d_spatial_d_phi;
        #endif

        if (data.spatial_type == SpatialType::BYM2) {
          #ifdef _OPENMP
          #pragma omp atomic
          grad[layout.theta_bym2_start + s_idx] += lik_grad * d_spatial_d_theta;
          #else
          grad[layout.theta_bym2_start + s_idx] += lik_grad * d_spatial_d_theta;
          #endif
        }
      }

      // Accumulate phi gradients
      #ifdef _OPENMP
      local_grad_phi_num += grad_phi_num_i;
      local_grad_phi_denom += grad_phi_denom_i;
      #else
      grad_phi_num_lik += grad_phi_num_i;
      grad_phi_denom_lik += grad_phi_denom_i;
      #endif

      // Fused log-likelihood: compute per-observation log_lik using already-computed
      // intermediates. This avoids a separate O(N) pass through compute_log_post.
      if (compute_lp) {
        double ll_i = 0.0;
        if (data.legacy.model_type == ModelType::BINOMIAL) {
          double p_i = 1.0 / (1.0 + std::exp(-eta_num));
          int n_trials = data.legacy.y_denom[i];
          int y_i = data.legacy.y_num[i];
          if (data.zi_type == tulpa_zi::ZIType::ZI_BINOMIAL) {
            ll_i = tulpa_zi::zi_binomial_lpmf_logit(y_i, n_trials, p_i, logit_zi);
          } else if (data.zi_type == tulpa_zi::ZIType::HURDLE_BINOMIAL) {
            ll_i = tulpa_zi::hurdle_binomial_lpmf_logit(y_i, n_trials, p_i, logit_zi);
          } else if (data.zi_type == tulpa_zi::ZIType::OI_BINOMIAL) {
            ll_i = tulpa_zi::oi_binomial_lpmf_logit(y_i, n_trials, p_i, logit_oi);
          } else if (data.zi_type == tulpa_zi::ZIType::ZOIB) {
            ll_i = tulpa_zi::zoib_lpmf_logit(y_i, n_trials, p_i, logit_zi, logit_oi);
          } else {
            ll_i = log_lik_binomial(y_i, n_trials, eta_num);
          }
        } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          if (layout.has_zi) {
            ll_i = tulpa_zi::zi_log_likelihood(data.legacy.y_num[i], mu_num_i, phi_num,
                                               logit_zi, data.zi_type);
          } else {
            ll_i = log_lik_negbin(data.legacy.y_num[i], mu_num_i, phi_num);
          }
          ll_i += log_lik_negbin(data.legacy.y_denom[i], mu_denom_i, phi_denom);
        } else if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          if (layout.has_zi) {
            ll_i = tulpa_zi::zi_log_likelihood(data.legacy.y_num[i], mu_num_i, phi_num,
                                               logit_zi, data.zi_type);
          } else {
            ll_i = log_lik_poisson(data.legacy.y_num[i], mu_num_i);
          }
          ll_i += log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom_i);
        } else if (data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          if (layout.has_zi) {
            ll_i = tulpa_zi::zi_log_likelihood(data.legacy.y_num[i], mu_num_i, phi_num,
                                               logit_zi, data.zi_type);
          } else {
            ll_i = log_lik_negbin(data.legacy.y_num[i], mu_num_i, phi_num);
          }
          ll_i += log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom_i);
        } else if (data.legacy.model_type == ModelType::GAMMA_GAMMA) {
          double mu_num_i = std::exp(eta_num);
          double mu_denom_i = std::exp(eta_denom);
          ll_i = log_lik_gamma(data.legacy.y_num_cont[i], phi_num, mu_num_i);
          ll_i += log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom_i);
        } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
          double log_y_num_i = std::log(data.legacy.y_num_cont[i]);
          double log_y_denom_i = std::log(data.legacy.y_denom_cont[i]);
          double z_num_i = (log_y_num_i - eta_num) / phi_num;
          double z_denom_i = (log_y_denom_i - eta_denom) / phi_denom;
          ll_i = -log_y_num_i - std::log(phi_num) - 0.5 * z_num_i * z_num_i;
          ll_i += -log_y_denom_i - std::log(phi_denom) - 0.5 * z_denom_i * z_denom_i;
        } else if (data.legacy.model_type == ModelType::BETA_BINOMIAL) {
          double p_i = 1.0 / (1.0 + std::exp(-eta_num));
          int y_i = data.legacy.y_num[i];
          int n_i = data.legacy.y_denom[i];
          double alpha_i = p_i * phi_num;
          double beta_i = (1.0 - p_i) * phi_num;
          ll_i = std::lgamma(y_i + alpha_i) + std::lgamma(n_i - y_i + beta_i) - std::lgamma(n_i + phi_num);
          ll_i += -std::lgamma(alpha_i) - std::lgamma(beta_i) + std::lgamma(phi_num);
          ll_i += tulpa::math::portable_lchoose(n_i, y_i);
        }
        #ifdef _OPENMP
        local_obs_ll += ll_i;
        #else
        obs_log_lik += ll_i;
        #endif
      }
    }

  #ifdef _OPENMP
    // Reduce local accumulators
    #pragma omp critical
    {
      for (int j = 0; j < data.legacy.p_num; j++) {
        grad_beta_num[j] += local_grad_beta_num[j];
      }
      for (int j = 0; j < data.legacy.p_denom; j++) {
        grad_beta_denom[j] += local_grad_beta_denom[j];
      }
      grad_phi_num_lik += local_grad_phi_num;
      grad_phi_denom_lik += local_grad_phi_denom;
      for (int g = 0; g < (int)local_grad_re.size(); g++) {
        grad[layout.re_start + g] += local_grad_re[g];
      }
      // Reduce crossed RE thread-local buffers to scattered grad positions
      // Only for crossed intercept-only RE (n_re_terms > 1, not slopes)
      if (!local_grad_re_crossed.empty()) {
        for (int t = 0; t < (int)data.re_n_groups_multi.size(); t++) {
          int re_start_t = layout.re_start_multi[t];
          int offset_t = data.re_offsets[t];
          for (int g = 0; g < data.re_n_groups_multi[t]; g++) {
            grad[re_start_t + g] += local_grad_re_crossed[offset_t + g];
          }
        }
      }
      obs_log_lik += local_obs_ll;
    }
  }  // end parallel
  #endif

  // Add likelihood gradients to total
  for (int j = 0; j < data.legacy.p_num; j++) {
    grad[layout.legacy.beta_num_start + j] += grad_beta_num[j];
  }
  for (int j = 0; j < data.legacy.p_denom; j++) {
    grad[layout.legacy.beta_denom_start + j] += grad_beta_denom[j];
  }

  // Phi gradients (with Jacobian for log transform)
  if (layout.legacy.has_phi_num) {
    grad[layout.legacy.log_phi_num_idx] += phi_num * grad_phi_num_lik;
  }
  if (layout.legacy.has_phi_denom) {
    grad[layout.legacy.log_phi_denom_idx] += phi_denom * grad_phi_denom_lik;
  }

  // ZI coefficient gradients (likelihood contribution)
  if (layout.has_zi && data.p_zi > 0) {
    for (int j = 0; j < data.p_zi; j++) {
      grad[layout.beta_zi_start + j] += grad_beta_zi[j];
    }
  }

  // OI coefficient gradients (likelihood contribution)
  if (layout.has_oi && data.p_oi > 0) {
    for (int j = 0; j < data.p_oi; j++) {
      grad[layout.beta_oi_start + j] += grad_beta_oi[j];
    }
  }

  } // end if (!used_vectorized)

  // Random slopes likelihood gradients (MUST be outside used_vectorized check:
  // the hybrid slopes path sets used_vectorized=true but populates grad_re_slopes_lik
  // via its own scatter pass - the write-back chain rule runs for all paths)
  if (layout.has_re_slopes && n_re_terms_slopes > 0) {
    for (int t = 0; t < n_re_terms_slopes; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = layout.re_n_coefs_multi[t];
      int re_start_t = layout.re_start_multi[t];
      bool is_nc = (t < (int)nc_L_flats.size() && !nc_L_flats[t].empty());

      if (is_nc) {
        // Non-centered correlated slopes: chain rule from dLL/d(re_nc) back
        // to (z, log_sigma, raw_chol). Single helper covers all three pieces;
        // grad_z and grad_raw slots are contiguous, log_sigma is scattered.
        const auto& L_flat = nc_L_flats[t];
        const auto& sigmas = nc_sigmas_vec[t];
        int chol_start = layout.chol_re_start_multi[t];
        std::vector<double> g_log_sigma(n_coefs, 0.0);

        tulpa::chol_nc_chain_rule_add(
            L_flat.data(), n_coefs, sigmas.data(),
            &params[re_start_t], &params[chol_start],
            &re_nc_flat[re_start_t], n_groups,
            grad_re_slopes_lik[t].data(),
            &grad[re_start_t],         // grad_z (contiguous)
            g_log_sigma.data(),         // grad_log_sigma (scattered into temp)
            &grad[chol_start]);         // grad_raw (contiguous)

        for (int c = 0; c < n_coefs; c++) {
          grad[layout.log_sigma_re_slopes[t][c]] += g_log_sigma[c];
        }

      } else if (data.re_parameterization == 1) {
        // Uncorrelated non-centered: apply chain rule re = sigma * z
        // grad_re_slopes_lik[t][g*nc+c] = dLL/d(re[g,c])
        // grad[z_gc] += dLL/d(re_gc) * sigma_c
        // grad[log_sigma_c] += dLL/d(re_gc) * z_gc * sigma_c
        for (int c = 0; c < n_coefs; c++) {
          double sigma_c = std::exp(params[layout.log_sigma_re_slopes[t][c]]);
          double sigma_lik_grad = 0.0;
          for (int g = 0; g < n_groups; g++) {
            double lik_gc = grad_re_slopes_lik[t][g * n_coefs + c];
            double z_gc = params[re_start_t + g * n_coefs + c];
            grad[re_start_t + g * n_coefs + c] += lik_gc * sigma_c;
            sigma_lik_grad += lik_gc * z_gc * sigma_c;
          }
          grad[layout.log_sigma_re_slopes[t][c]] += sigma_lik_grad;
        }
      } else {
        // Uncorrelated centered: add grad_re_slopes_lik directly
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            grad[re_start_t + g * n_coefs + c] += grad_re_slopes_lik[t][g * n_coefs + c];
          }
        }
      }
    }
  }

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

}
