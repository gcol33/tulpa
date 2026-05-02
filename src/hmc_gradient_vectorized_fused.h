// hmc_gradient_vectorized_fused.h
// Fragment of hmc_gradient_vectorized.h.
// Included from the hmc_gradient_vectorized.h umbrella header inside
// namespace tulpa_hmc { namespace vectorized { ... } } in hmc_gradients.cpp.
// Do NOT wrap contents in any namespace — already inside namespace vectorized.
// Fused single-pass gradient: eta + residuals + accumulation in one obs loop.
#ifndef TULPA_HMC_GRADIENT_VECTORIZED_FUSED_H
#define TULPA_HMC_GRADIENT_VECTORIZED_FUSED_H

// ============================================================================
// FUSED SINGLE-PASS GRADIENT
// Computes eta + residuals + gradient accumulation in one observation loop,
// eliminating intermediate N-vectors (eta, resid, effect_dense) and Eigen
// overhead for small p. For p_num,p_denom <= 4, manual dot products keep
// beta accumulators in registers.
//
// Supports: base, +RE (single/crossed), +ICAR, +BYM2, +temporal
// Requires: same exclusions as vectorized (no ZI, slopes, GP, etc.)
// ============================================================================

inline bool can_use_fused(const ModelData& data, const ParamLayout& layout) {
  if (layout.has_zi || layout.has_oi) return false;
  if (layout.has_re_slopes || layout.has_re_correlated_slopes) return false;
  if (data.has_re_slopes) return false;
  if (data.spatial_type == SpatialType::GP ||
      data.spatial_type == SpatialType::MULTISCALE_GP ||
      data.spatial_type == SpatialType::HSGP) return false;
  if (data.has_svc || data.has_tvc || data.has_latent ||
      data.has_spatiotemporal || data.has_temporal_gp ||
      data.has_multiscale_temporal) return false;
  // Only fuse when Eigen overhead exceeds benefit (small p)
  if (data.legacy.p_num > 4 || data.legacy.p_denom > 4) return false;
  return true;
}

template<ModelType MT>
void compute_gradient_fused(
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
  const int p_num = data.legacy.p_num;
  const int p_denom = data.legacy.p_denom;
  const double* X_num = data.legacy.X_num_flat.data();
  const double* X_denom = data.legacy.X_denom_flat.data();
  const double* beta_num = &params[layout.legacy.beta_num_start];
  const double* beta_denom = &params[layout.legacy.beta_denom_start];

  constexpr bool is_binomial = (MT == ModelType::BINOMIAL) || (MT == ModelType::BETA_BINOMIAL);

  ws.init(N);
  ws.precompute(data);  // lgamma(y+1), lchoose — no-op if already done

  double phi_num = layout.legacy.has_phi_num ? std::exp(params[layout.legacy.log_phi_num_idx]) : 1.0;
  double phi_denom = layout.legacy.has_phi_denom ? std::exp(params[layout.legacy.log_phi_denom_idx]) : 1.0;

  // Build digamma/lgamma lookup tables for NB (indexed by integer y, rebuilt each call since phi changes)
  if constexpr (MT == ModelType::NEGBIN_NEGBIN) {
    ws.build_digamma_tables(phi_num, phi_denom, true);
  } else if constexpr (MT == ModelType::NEGBIN_GAMMA) {
    ws.build_digamma_tables(phi_num, phi_denom, false);  // numerator only (denom is Gamma, not NB)
  } else if constexpr (MT == ModelType::BETA_BINOMIAL) {
    // Beta-binomial needs digamma(y+alpha), digamma(n-y+beta) — not integer-indexed by y alone
    // (alpha = p*phi, beta = (1-p)*phi depend on eta which varies per obs). Skip table.
  }

  // Stack-allocated beta gradient accumulators (p <= 4, stay in registers)
  double gb_num[4] = {};
  double gb_denom[4] = {};
  double grad_phi_num_lik = 0.0;
  double grad_phi_denom_lik = 0.0;
  double ll = 0.0;

  // --- Hoisted phi-dependent constants (model-specific, computed once) ---
  double h_digamma_phi_n = 0.0, h_digamma_phi_d = 0.0;
  double h_log_phi_n = 0.0, h_log_phi_d = 0.0;
  double h_lgamma_phi_n = 0.0, h_lgamma_phi_d = 0.0;

  if constexpr (MT == ModelType::NEGBIN_NEGBIN) {
    h_digamma_phi_n = tulpa::math::portable_digamma(phi_num);
    h_digamma_phi_d = tulpa::math::portable_digamma(phi_denom);
    h_log_phi_n = std::log(phi_num);
    h_log_phi_d = std::log(phi_denom);
    h_lgamma_phi_n = compute_lp ? std::lgamma(phi_num) : 0.0;
    h_lgamma_phi_d = compute_lp ? std::lgamma(phi_denom) : 0.0;
  }
  if constexpr (MT == ModelType::POISSON_GAMMA) {
    h_digamma_phi_d = tulpa::math::portable_digamma(phi_denom);
    h_log_phi_d = std::log(phi_denom);
    h_lgamma_phi_d = compute_lp ? std::lgamma(phi_denom) : 0.0;
  }
  if constexpr (MT == ModelType::NEGBIN_GAMMA) {
    h_digamma_phi_n = tulpa::math::portable_digamma(phi_num);
    h_digamma_phi_d = tulpa::math::portable_digamma(phi_denom);
    h_log_phi_n = std::log(phi_num);
    h_log_phi_d = std::log(phi_denom);
    h_lgamma_phi_n = compute_lp ? std::lgamma(phi_num) : 0.0;
    h_lgamma_phi_d = compute_lp ? std::lgamma(phi_denom) : 0.0;
  }
  if constexpr (MT == ModelType::GAMMA_GAMMA) {
    h_digamma_phi_n = tulpa::math::portable_digamma(phi_num);
    h_digamma_phi_d = tulpa::math::portable_digamma(phi_denom);
    h_log_phi_n = std::log(phi_num);
    h_log_phi_d = std::log(phi_denom);
    h_lgamma_phi_n = compute_lp ? std::lgamma(phi_num) : 0.0;
    h_lgamma_phi_d = compute_lp ? std::lgamma(phi_denom) : 0.0;
  }
  double h_sigma_num_sq = 0.0, h_sigma_denom_sq = 0.0;
  double h_log_sigma_n = 0.0, h_log_sigma_d = 0.0;
  if constexpr (MT == ModelType::LOGNORMAL) {
    h_sigma_num_sq = phi_num * phi_num;
    h_sigma_denom_sq = phi_denom * phi_denom;
    h_log_sigma_n = compute_lp ? std::log(phi_num) : 0.0;
    h_log_sigma_d = compute_lp ? std::log(phi_denom) : 0.0;
  }
  double h_psi_phi_bb = 0.0, h_lgamma_phi_bb = 0.0;
  if constexpr (MT == ModelType::BETA_BINOMIAL) {
    h_psi_phi_bb = tulpa::math::portable_digamma(phi_num);
    h_lgamma_phi_bb = compute_lp ? std::lgamma(phi_num) : 0.0;
  }

  // BYM2 spatial parameters
  double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0, bym2_scale = 0.0;
  const double* phi_spatial = nullptr;
  const double* theta_bym2 = nullptr;
  if (layout.has_spatial) {
    phi_spatial = &params[layout.spatial_start];
    if (data.spatial_type == SpatialType::BYM2) {
      double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
      double logit_rho = params[layout.logit_rho_bym2_idx];
      double rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
      sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
      sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
      theta_bym2 = &params[layout.theta_bym2_start];
      bym2_scale = data.bym2_scale_factor;
    }
  }
  const double* phi_temporal = nullptr;
  if (layout.has_temporal) {
    phi_temporal = &params[layout.temporal_start];
  }

  // Families using exp(eta) benefit from Eigen-vectorized exp (SIMD 2-4x faster).
  // Split into 3 passes: eta computation → vectorized exp → scalar kernel.
  // NB/GG: also enables vectorized log pass for log(mu+phi).
  // PG: SIMD exp + eliminates redundant scalar exp(log_rate) per obs.
  // Binomial/LN/BB stay single-pass (simpler kernels, overhead not worthwhile).
  constexpr bool use_multipass = (MT == ModelType::NEGBIN_NEGBIN) ||
                                 (MT == ModelType::NEGBIN_GAMMA) ||
                                 (MT == ModelType::GAMMA_GAMMA) ||
                                 (MT == ModelType::POISSON_GAMMA);

  // Non-centered RE: extract sigma for computing re = sigma * z
  double fused_sigma_re = 0.0;  // >0 when non-centered single term
  std::vector<double> fused_sigma_re_terms;  // per-term for crossed NC
  if (layout.has_re && data.re_parameterization == 1 && !layout.has_re_slopes) {
    if (data.n_re_terms > 1) {
      fused_sigma_re_terms.resize(data.n_re_terms);
      for (int t = 0; t < data.n_re_terms; t++) {
        fused_sigma_re_terms[t] = std::exp(params[layout.log_sigma_re_multi[t]]);
      }
    } else {
      fused_sigma_re = std::exp(params[layout.log_sigma_re_idx]);
    }
  }

  // --- Shared lambda: compute eta for observation i ---
  // (avoids code duplication between needs_exp and !needs_exp paths)
  auto compute_eta_i = [&](int i, double& eta_n, double& eta_d) {
    const double* xi_num = &X_num[i * p_num];
    eta_n = 0.0;
    for (int j = 0; j < p_num; j++)
      eta_n += xi_num[j] * beta_num[j];
    eta_d = 0.0;
    if constexpr (!is_binomial) {
      const double* xi_d = &X_denom[i * p_denom];
      for (int j = 0; j < p_denom; j++)
        eta_d += xi_d[j] * beta_denom[j];
    }
    if (layout.has_re) {
      double re_val;
      if (data.n_re_terms > 1) {
        re_val = 0.0;
        for (int t = 0; t < data.n_re_terms; t++) {
          int g = data.re_group_multi_flat[i * data.n_re_terms + t];
          if (g > 0) {
            double z_or_re = params[layout.re_start_multi[t] + g - 1];
            re_val += fused_sigma_re_terms.empty() ? z_or_re : fused_sigma_re_terms[t] * z_or_re;
          }
        }
      } else {
        int g = data.re_group[i];
        if (g > 0) {
          double z_or_re = params[layout.re_start + g - 1];
          re_val = (fused_sigma_re > 0.0) ? fused_sigma_re * z_or_re : z_or_re;
        } else {
          re_val = 0.0;
        }
      }
      eta_n += re_val;
      if constexpr (!is_binomial) eta_d += re_val;
    }
    if (layout.has_spatial) {
      int s = data.spatial_group[i];
      if (s > 0) {
        s -= 1;
        double sv;
        if (data.spatial_type == SpatialType::BYM2) {
          sv = sigma_s_bym2 * phi_spatial[s] * bym2_scale + sigma_u_bym2 * theta_bym2[s];
        } else {
          sv = phi_spatial[s];
        }
        eta_n += sv;
        if constexpr (!is_binomial) eta_d += sv;
      }
    }
    if (layout.has_temporal && !data.temporal_time_idx.empty()) {
      int t = data.temporal_time_idx[i];
      if (t > 0) {
        int g = data.temporal_group_idx[i] - 1;
        int flat = g * data.n_times + (t - 1);  // Panel temporal: flat index
        eta_n += phi_temporal[flat];
        if constexpr (!is_binomial) eta_d += phi_temporal[flat];
      }
    }
  };

  // --- Shared lambda: scatter grouped effect gradients for obs i ---
  auto scatter_gradients_i = [&](int i, double resid_n, double resid_d) {
    const double* xi_num = &X_num[i * p_num];
    for (int j = 0; j < p_num; j++)
      gb_num[j] += xi_num[j] * resid_n;
    if constexpr (!is_binomial) {
      const double* xi_d = &X_denom[i * p_denom];
      for (int j = 0; j < p_denom; j++)
        gb_denom[j] += xi_d[j] * resid_d;
    }
    double total_lik_grad = resid_n;
    if constexpr (!is_binomial) total_lik_grad += resid_d;
    if (layout.has_re) {
      if (data.n_re_terms > 1) {
        for (int t = 0; t < data.n_re_terms; t++) {
          int g = data.re_group_multi_flat[i * data.n_re_terms + t];
          if (g > 0) grad[layout.re_start_multi[t] + g - 1] += total_lik_grad;
        }
      } else {
        int g = data.re_group[i];
        if (g > 0) grad[layout.re_start + g - 1] += total_lik_grad;
      }
    }
    if (layout.has_spatial) {
      int s = data.spatial_group[i];
      if (s > 0) {
        s -= 1;
        if (data.spatial_type == SpatialType::BYM2) {
          grad_spatial_lik_out[s] += total_lik_grad * sigma_s_bym2 * bym2_scale;
          grad[layout.theta_bym2_start + s] += total_lik_grad * sigma_u_bym2;
        } else {
          grad_spatial_lik_out[s] += total_lik_grad;
        }
      }
    }
    if (layout.has_temporal && !data.temporal_time_idx.empty()) {
      int t = data.temporal_time_idx[i];
      if (t > 0) {
        int g = data.temporal_group_idx[i] - 1;
        int flat = g * data.n_times + (t - 1);  // Panel temporal: flat index
        grad_temporal_lik_out[flat] += total_lik_grad;
      }
    }
  };

  if constexpr (use_multipass) {
    // === 2.5-PASS: vectorized exp for NB/PG/GG ===
    using ArrayXd = Eigen::Array<double, Eigen::Dynamic, 1>;

    // Pass 1: compute etas into workspace
    for (int i = 0; i < N; i++) {
      compute_eta_i(i, ws.eta_num[i], ws.eta_denom[i]);
    }

    // Pass 1.5: Eigen-vectorized exp (uses SIMD: SSE2=2x, AVX=4x)
    Eigen::Map<ArrayXd>(ws.mu_num.data(), N) =
        Eigen::Map<const ArrayXd>(ws.eta_num.data(), N).exp();
    if constexpr (!is_binomial) {
      Eigen::Map<ArrayXd>(ws.mu_denom.data(), N) =
          Eigen::Map<const ArrayXd>(ws.eta_denom.data(), N).exp();
    }

    // Pass 1.75: Eigen-vectorized log for NB (log(mu+phi) needed per-obs)
    if constexpr (MT == ModelType::NEGBIN_NEGBIN) {
      Eigen::Map<ArrayXd>(ws.log_dn.data(), N) =
          (Eigen::Map<const ArrayXd>(ws.mu_num.data(), N) + phi_num).log();
      Eigen::Map<ArrayXd>(ws.log_dd.data(), N) =
          (Eigen::Map<const ArrayXd>(ws.mu_denom.data(), N) + phi_denom).log();
    }
    if constexpr (MT == ModelType::NEGBIN_GAMMA) {
      Eigen::Map<ArrayXd>(ws.log_dn.data(), N) =
          (Eigen::Map<const ArrayXd>(ws.mu_num.data(), N) + phi_num).log();
    }

    // Pass 2: scalar residual kernel using precomputed mu + scatter
    for (int i = 0; i < N; i++) {
      const double mu_n = ws.mu_num[i];
      const double mu_d = ws.mu_denom[i];
      const double eta_n = ws.eta_num[i];
      const double eta_d = ws.eta_denom[i];
      double resid_n = 0.0, resid_d = 0.0;

      if constexpr (MT == ModelType::NEGBIN_NEGBIN) {
        const int y_n = data.legacy.y_num[i];
        const int y_d = data.legacy.y_denom[i];
        const double inv_dn = 1.0 / (mu_n + phi_num);   // reciprocal: 1 div replaces 2
        const double inv_dd = 1.0 / (mu_d + phi_denom);
        resid_n = y_n - mu_n * (y_n + phi_num) * inv_dn;
        resid_d = y_d - mu_d * (y_d + phi_denom) * inv_dd;
        // Table lookup: digamma/lgamma indexed by integer y value
        const double digamma_yn_phi = ws.dig_table_num[y_n];
        const double lgamma_yn_phi = ws.lg_table_num[y_n];
        const double digamma_yd_phi = ws.dig_table_denom[y_d];
        const double lgamma_yd_phi = ws.lg_table_denom[y_d];
        const double log_dn = ws.log_dn[i];   // pre-computed via Eigen SIMD (pass 1.75)
        const double log_dd = ws.log_dd[i];   // pre-computed via Eigen SIMD (pass 1.75)
        const double log_phi_dn = h_log_phi_n - log_dn;
        const double log_phi_dd = h_log_phi_d - log_dd;
        grad_phi_num_lik += digamma_yn_phi - h_digamma_phi_n + log_phi_dn + (mu_n - y_n) * inv_dn;
        grad_phi_denom_lik += digamma_yd_phi - h_digamma_phi_d + log_phi_dd + (mu_d - y_d) * inv_dd;
        if (compute_lp) {
          ll += lgamma_yn_phi - ws.lgamma_y_num_p1[i] - h_lgamma_phi_n
              + phi_num * log_phi_dn + y_n * (eta_n - log_dn);
          ll += lgamma_yd_phi - ws.lgamma_y_denom_p1[i] - h_lgamma_phi_d
              + phi_denom * log_phi_dd + y_d * (eta_d - log_dd);
        }
      }
      else if constexpr (MT == ModelType::POISSON_GAMMA) {
        const int y_n = data.legacy.y_num[i];
        const double y_d = data.legacy.y_denom_cont[i];
        resid_n = y_n - mu_n;
        // Gamma requires y > 0; skip denom contribution for invalid obs
        // (matches log_lik_gamma returning -1e10 for y <= 0)
        if (y_d > 0.0) {
          const double inv_mu_d = 1.0 / mu_d;
          const double yd_over_mud = y_d * inv_mu_d;
          resid_d = phi_denom * (yd_over_mud - 1.0);
          const double log_rate = h_log_phi_d - eta_d;
          grad_phi_denom_lik += log_rate + 1.0 + ws.log_y_denom_cont[i] - h_digamma_phi_d
                            - yd_over_mud;
          if (compute_lp) {
            ll += y_n * eta_n - mu_n - ws.lgamma_y_num_p1[i];
            ll += (phi_denom - 1.0) * ws.log_y_denom_cont[i] + phi_denom * log_rate
                - phi_denom * yd_over_mud - h_lgamma_phi_d;
          }
        } else {
          resid_d = 0.0;
          if (compute_lp) {
            ll += y_n * eta_n - mu_n - ws.lgamma_y_num_p1[i];
            ll += -1e10;  // Match log_lik_gamma penalty
          }
        }
      }
      else if constexpr (MT == ModelType::NEGBIN_GAMMA) {
        const int y_n = data.legacy.y_num[i];
        const double y_d = data.legacy.y_denom_cont[i];
        // NB numerator (same as NEGBIN_NEGBIN num)
        const double inv_dn = 1.0 / (mu_n + phi_num);
        resid_n = y_n - mu_n * (y_n + phi_num) * inv_dn;
        const double digamma_yn_phi = ws.dig_table_num[y_n];
        const double lgamma_yn_phi = ws.lg_table_num[y_n];
        const double log_dn = ws.log_dn[i];
        const double log_phi_dn = h_log_phi_n - log_dn;
        grad_phi_num_lik += digamma_yn_phi - h_digamma_phi_n + log_phi_dn + (mu_n - y_n) * inv_dn;
        // Gamma denominator (same as PG denom) — skip if y <= 0
        if (y_d > 0.0) {
          const double inv_mu_d = 1.0 / mu_d;
          const double yd_over_mud = y_d * inv_mu_d;
          resid_d = phi_denom * (yd_over_mud - 1.0);
          const double log_rate = h_log_phi_d - eta_d;
          grad_phi_denom_lik += log_rate + 1.0 + ws.log_y_denom_cont[i] - h_digamma_phi_d
                            - yd_over_mud;
          if (compute_lp) {
            ll += lgamma_yn_phi - ws.lgamma_y_num_p1[i] - h_lgamma_phi_n
                + phi_num * log_phi_dn + y_n * (eta_n - log_dn);
            ll += (phi_denom - 1.0) * ws.log_y_denom_cont[i] + phi_denom * log_rate
                - phi_denom * yd_over_mud - h_lgamma_phi_d;
          }
        } else {
          resid_d = 0.0;
          if (compute_lp) {
            ll += lgamma_yn_phi - ws.lgamma_y_num_p1[i] - h_lgamma_phi_n
                + phi_num * log_phi_dn + y_n * (eta_n - log_dn);
            ll += -1e10;
          }
        }
      }
      else if constexpr (MT == ModelType::GAMMA_GAMMA) {
        const double y_n = data.legacy.y_num_cont[i];
        const double y_d = data.legacy.y_denom_cont[i];
        // Gamma requires y > 0 for both num and denom
        if (y_n > 0.0) {
          const double inv_mu_n = 1.0 / mu_n;
          const double yn_over_mun = y_n * inv_mu_n;
          resid_n = phi_num * (yn_over_mun - 1.0);
          const double log_rate_n = h_log_phi_n - eta_n;
          grad_phi_num_lik += log_rate_n + 1.0 + ws.log_y_num_cont[i] - h_digamma_phi_n - yn_over_mun;
          if (compute_lp)
            ll += (phi_num - 1.0) * ws.log_y_num_cont[i] + phi_num * log_rate_n
                - phi_num * yn_over_mun - h_lgamma_phi_n;
        } else {
          resid_n = 0.0;
          if (compute_lp) ll += -1e10;
        }
        if (y_d > 0.0) {
          const double inv_mu_d = 1.0 / mu_d;
          const double yd_over_mud = y_d * inv_mu_d;
          resid_d = phi_denom * (yd_over_mud - 1.0);
          const double log_rate_d = h_log_phi_d - eta_d;
          grad_phi_denom_lik += log_rate_d + 1.0 + ws.log_y_denom_cont[i] - h_digamma_phi_d - yd_over_mud;
          if (compute_lp)
            ll += (phi_denom - 1.0) * ws.log_y_denom_cont[i] + phi_denom * log_rate_d
                - phi_denom * yd_over_mud - h_lgamma_phi_d;
        } else {
          resid_d = 0.0;
          if (compute_lp) ll += -1e10;
        }
      }

      scatter_gradients_i(i, resid_n, resid_d);
    } // end pass 2
  } else {
    // === SINGLE-PASS for Binomial / Lognormal / Beta-Binomial ===
    // (PG now uses multipass above)
    for (int i = 0; i < N; i++) {
      double eta_n = 0.0, eta_d = 0.0;
      compute_eta_i(i, eta_n, eta_d);

      double resid_n = 0.0, resid_d = 0.0;

      if constexpr (MT == ModelType::POISSON_GAMMA) {
        // Dead code path (PG now uses multipass), kept for completeness
        const int y_n = data.legacy.y_num[i];
        const double y_d = data.legacy.y_denom_cont[i];
        const double mu_n = std::exp(eta_n);
        const double mu_d = std::exp(eta_d);
        resid_n = y_n - mu_n;
        resid_d = phi_denom * (y_d / mu_d - 1.0);
        const double log_rate = h_log_phi_d - eta_d;
        grad_phi_denom_lik += log_rate + 1.0 + ws.log_y_denom_cont[i] - h_digamma_phi_d
                          - y_d / mu_d;
        if (compute_lp) {
          ll += y_n * eta_n - mu_n - ws.lgamma_y_num_p1[i];
          ll += (phi_denom - 1.0) * ws.log_y_denom_cont[i] + phi_denom * log_rate
              - (phi_denom / mu_d) * y_d - h_lgamma_phi_d;
        }
      }
      else if constexpr (MT == ModelType::BINOMIAL) {
        const int n_trials = data.legacy.y_denom[i];
        const int y = data.legacy.y_num[i];
        // Numerically stable logistic: 1 exp + 1 log (was 3 exp + 2 log)
        double p, log_p, log_1mp;
        if (eta_n >= 0.0) {
          const double exp_neg = std::exp(-eta_n);
          const double denom = 1.0 + exp_neg;
          p = 1.0 / denom;
          log_p = -std::log(denom);
          log_1mp = -eta_n + log_p;
        } else {
          const double exp_pos = std::exp(eta_n);
          const double denom = 1.0 + exp_pos;
          p = exp_pos / denom;
          log_1mp = -std::log(denom);
          log_p = eta_n + log_1mp;
        }
        resid_n = y - n_trials * p;
        if (compute_lp) {
          ll += y * log_p + (n_trials - y) * log_1mp + ws.lchoose_cache[i];
        }
      }
      else if constexpr (MT == ModelType::LOGNORMAL) {
        const double log_yn = ws.log_y_num_cont[i];
        const double log_yd = ws.log_y_denom_cont[i];
        resid_n = (log_yn - eta_n) / h_sigma_num_sq;
        resid_d = (log_yd - eta_d) / h_sigma_denom_sq;
        const double z_n = (log_yn - eta_n) / phi_num;
        const double z_d = (log_yd - eta_d) / phi_denom;
        grad_phi_num_lik += (-1.0 + z_n * z_n) / phi_num;
        grad_phi_denom_lik += (-1.0 + z_d * z_d) / phi_denom;
        if (compute_lp) {
          ll += -log_yn - h_log_sigma_n - 0.5 * z_n * z_n;
          ll += -log_yd - h_log_sigma_d - 0.5 * z_d * z_d;
        }
      }
      else if constexpr (MT == ModelType::BETA_BINOMIAL) {
        // Numerically stable sigmoid (avoids exp overflow for |eta| > 700)
        const double p = (eta_n >= 0.0)
            ? 1.0 / (1.0 + std::exp(-eta_n))
            : std::exp(eta_n) / (1.0 + std::exp(eta_n));
        const int y_i = data.legacy.y_num[i];
        const int n_i = data.legacy.y_denom[i];
        const double alpha = p * phi_num;
        const double beta_param = (1.0 - p) * phi_num;
        auto [psi_y_alpha, lg_y_alpha] = tulpa::math::portable_digamma_lgamma(y_i + alpha);
        auto [psi_nmy_beta, lg_nmy_beta] = tulpa::math::portable_digamma_lgamma(n_i - y_i + beta_param);
        auto [psi_alpha, lg_alpha] = tulpa::math::portable_digamma_lgamma(alpha);
        auto [psi_beta, lg_beta] = tulpa::math::portable_digamma_lgamma(beta_param);
        auto [psi_n_phi, lg_n_phi] = tulpa::math::portable_digamma_lgamma(n_i + phi_num);
        const double dLL_dp = phi_num * (psi_y_alpha - psi_nmy_beta - psi_alpha + psi_beta);
        resid_n = dLL_dp * p * (1.0 - p);
        grad_phi_num_lik += p * psi_y_alpha + (1.0 - p) * psi_nmy_beta - psi_n_phi
                          - p * psi_alpha - (1.0 - p) * psi_beta + h_psi_phi_bb;
        if (compute_lp) {
          ll += lg_y_alpha + lg_nmy_beta - lg_n_phi;
          ll += -lg_alpha - lg_beta + h_lgamma_phi_bb;
          ll += ws.lchoose_cache[i];
        }
      }

      scatter_gradients_i(i, resid_n, resid_d);
    } // end single-pass
  }

  // Write back beta gradients from stack accumulators
  for (int j = 0; j < p_num; j++)
    grad[layout.legacy.beta_num_start + j] += gb_num[j];
  if constexpr (!is_binomial) {
    for (int j = 0; j < p_denom; j++)
      grad[layout.legacy.beta_denom_start + j] += gb_denom[j];
  }

  // Phi gradients (with log-transform Jacobian)
  if (layout.legacy.has_phi_num) {
    grad[layout.legacy.log_phi_num_idx] += phi_num * grad_phi_num_lik;
  }
  if (layout.legacy.has_phi_denom) {
    grad[layout.legacy.log_phi_denom_idx] += phi_denom * grad_phi_denom_lik;
  }

  obs_ll += ll;
}

// Fused dispatcher: switches on ModelType
inline bool dispatch_fused_gradient(
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
  if (!can_use_fused(data, layout)) return false;

  switch (data.legacy.model_type) {
    case ModelType::BINOMIAL:
      compute_gradient_fused<ModelType::BINOMIAL>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::NEGBIN_NEGBIN:
      compute_gradient_fused<ModelType::NEGBIN_NEGBIN>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::POISSON_GAMMA:
      compute_gradient_fused<ModelType::POISSON_GAMMA>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::NEGBIN_GAMMA:
      compute_gradient_fused<ModelType::NEGBIN_GAMMA>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::GAMMA_GAMMA:
      compute_gradient_fused<ModelType::GAMMA_GAMMA>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::LOGNORMAL:
      compute_gradient_fused<ModelType::LOGNORMAL>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
    case ModelType::BETA_BINOMIAL:
      compute_gradient_fused<ModelType::BETA_BINOMIAL>(
        params, data, layout, grad, obs_ll, compute_lp,
        grad_temporal_lik_out, grad_spatial_lik_out, ws);
      return true;
  }
  return false;
}

#endif  // TULPA_HMC_GRADIENT_VECTORIZED_FUSED_H
