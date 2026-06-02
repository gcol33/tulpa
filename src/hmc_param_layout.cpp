// hmc_param_layout.cpp
// Parameter layout and size accounting for HMC/NUTS model data.

#include "hmc_sampler.h"
#include "tulpa/likelihood.h"

namespace tulpa_hmc {

ParamLayout compute_param_layout(const ModelData& data) {
  ParamLayout layout;
  int idx = 0;

  // ================================================================
  // GENERIC MULTI-PROCESS LAYOUT
  // ================================================================
  // Phase D (gcol33/tulpa#15): the legacy ratio (n_processes == 0)
  // branch was removed along with the cpp_hmc_fit entry point.
  if (data.n_processes == 0) {
    Rcpp::stop("tulpa: compute_param_layout requires n_processes > 0. "
               "The legacy ratio path was removed in Phase D "
               "(gcol33/tulpa#15).");
  }
  layout.process_beta_start.resize(data.n_processes);
  layout.process_beta_count.resize(data.n_processes);
  for (int k = 0; k < data.n_processes; k++) {
    layout.process_beta_start[k] = idx;
    layout.process_beta_count[k] = data.processes[k].p;
    idx += data.processes[k].p;
  }

  // Random effects (supports multiple crossed RE terms with slopes)
  layout.has_re = (data.n_re_groups > 0 || data.total_re_groups > 0);
  layout.has_re_slopes = data.has_re_slopes;
  layout.has_re_correlated_slopes = data.has_re_correlated_slopes;

  if (data.has_re_slopes && data.n_re_terms > 0) {
    // Random slopes case: need sigma per coefficient type + Cholesky params + RE effects
    int n_terms = data.n_re_terms;

    layout.log_sigma_re_multi.resize(n_terms);
    layout.log_sigma_re_slopes.resize(n_terms);
    layout.re_start_multi.resize(n_terms);
    layout.re_end_multi.resize(n_terms);
    layout.re_n_coefs_multi.resize(n_terms);
    layout.re_correlated_multi.resize(n_terms);
    layout.chol_re_start_multi.resize(n_terms);
    layout.chol_re_end_multi.resize(n_terms);

    // First pass: allocate sigma parameters for each term
    for (int t = 0; t < n_terms; t++) {
      int n_coefs = data.re_n_coefs[t];
      layout.re_n_coefs_multi[t] = n_coefs;
      layout.re_correlated_multi[t] = data.re_correlated[t];

      // Allocate log_sigma for each coefficient type (intercept, slopes)
      layout.log_sigma_re_slopes[t].resize(n_coefs);
      for (int c = 0; c < n_coefs; c++) {
        layout.log_sigma_re_slopes[t][c] = idx++;
      }
      // Legacy: point to first sigma for backwards compat
      layout.log_sigma_re_multi[t] = layout.log_sigma_re_slopes[t][0];
    }

    // Second pass: allocate Cholesky parameters for correlated terms
    for (int t = 0; t < n_terms; t++) {
      int n_chol = data.re_n_chol[t];  // k*(k-1)/2 for correlated, 0 otherwise
      if (n_chol > 0) {
        layout.chol_re_start_multi[t] = idx;
        idx += n_chol;
        layout.chol_re_end_multi[t] = idx;
      } else {
        layout.chol_re_start_multi[t] = -1;
        layout.chol_re_end_multi[t] = -1;
      }
    }

    // Third pass: allocate RE effects for each term
    for (int t = 0; t < n_terms; t++) {
      int n_groups = data.re_n_groups_multi[t];
      int n_coefs = data.re_n_coefs[t];

      layout.re_start_multi[t] = idx;
      idx += n_groups * n_coefs;  // Each group has n_coefs parameters
      layout.re_end_multi[t] = idx;
    }

    // Legacy fields: point to first term
    layout.log_sigma_re_idx = layout.log_sigma_re_multi[0];
    layout.re_start = layout.re_start_multi[0];
    layout.re_end = layout.re_end_multi[0];

  } else if (data.n_re_terms > 1) {
    // Multiple RE terms (intercept only): allocate sigma and RE for each term
    layout.log_sigma_re_multi.resize(data.n_re_terms);
    layout.re_start_multi.resize(data.n_re_terms);
    layout.re_end_multi.resize(data.n_re_terms);
    layout.re_n_coefs_multi.resize(data.n_re_terms, 1);  // All intercept-only
    layout.re_correlated_multi.resize(data.n_re_terms, false);
    layout.chol_re_start_multi.resize(data.n_re_terms, -1);
    layout.chol_re_end_multi.resize(data.n_re_terms, -1);

    for (int t = 0; t < data.n_re_terms; t++) {
      layout.log_sigma_re_multi[t] = idx++;
    }
    for (int t = 0; t < data.n_re_terms; t++) {
      layout.re_start_multi[t] = idx;
      idx += data.re_n_groups_multi[t];
      layout.re_end_multi[t] = idx;
    }

    // Set legacy fields to first term for backwards compatibility
    layout.log_sigma_re_idx = layout.log_sigma_re_multi[0];
    layout.re_start = layout.re_start_multi[0];
    layout.re_end = layout.re_end_multi[0];
  } else if (layout.has_re) {
    // Single RE term (intercept only)
    layout.log_sigma_re_idx = idx++;
    layout.re_start = idx;
    idx += data.n_re_groups;
    layout.re_end = idx;

    // Also set multi arrays for consistency
    layout.log_sigma_re_multi.resize(1);
    layout.log_sigma_re_multi[0] = layout.log_sigma_re_idx;
    layout.re_start_multi.resize(1);
    layout.re_start_multi[0] = layout.re_start;
    layout.re_end_multi.resize(1);
    layout.re_end_multi[0] = layout.re_end;
    layout.re_n_coefs_multi.resize(1, 1);
    layout.re_correlated_multi.resize(1, false);
    layout.chol_re_start_multi.resize(1, -1);
    layout.chol_re_end_multi.resize(1, -1);
  } else {
    layout.log_sigma_re_idx = -1;
    layout.re_start = layout.re_end = -1;
  }

  // Model-specific overdispersion / shape / sigma scalars are owned by the
  // model package's LikelihoodSpec: the package packs them into a contiguous
  // block at layout.extra_offset (length spec->n_extra_params), allocated at
  // the bottom of this function.

  // Spatial effects (ICAR/BYM2/CAR_PROPER only - GP handled separately below)
  layout.has_spatial = (data.spatial_type == SpatialType::ICAR ||
                        data.spatial_type == SpatialType::BYM2 ||
                        data.spatial_type == SpatialType::CAR_PROPER);
  layout.is_bym2 = (data.spatial_type == SpatialType::BYM2);
  layout.is_car_proper = (data.spatial_type == SpatialType::CAR_PROPER);
  layout.is_icar_collapsed = (data.spatial_type == SpatialType::ICAR && data.icar_collapsed);
  layout.is_bym2_collapsed = (data.spatial_type == SpatialType::BYM2 && data.bym2_collapsed);

  if (layout.has_spatial) {
    if (layout.is_bym2) {
      // BYM2 Riebler: log_sigma_total, logit_rho, [phi_scaled, theta if not collapsed]
      layout.log_sigma_bym2_idx = idx++;
      layout.logit_rho_bym2_idx = idx++;
      if (data.bym2_collapsed) {
        // Collapsed: phi and theta marginalized out, not in param vector
        layout.spatial_start = layout.spatial_end = -1;
        layout.theta_bym2_start = layout.theta_bym2_end = -1;
      } else {
        layout.spatial_start = idx;
        idx += data.n_spatial_units;  // phi_scaled (structured)
        layout.spatial_end = idx;
        layout.theta_bym2_start = idx;
        idx += data.n_spatial_units;  // theta (unstructured)
        layout.theta_bym2_end = idx;
      }
      layout.log_tau_spatial_idx = -1;
      layout.logit_rho_car_idx = -1;
    } else if (layout.is_car_proper) {
      // Proper CAR: log_tau, logit_rho_car, phi
      // Q = D - rho*W (proper precision matrix; rho estimated from data)
      layout.log_tau_spatial_idx = idx++;
      layout.logit_rho_car_idx = idx++;
      layout.spatial_start = idx;
      idx += data.n_spatial_units;
      layout.spatial_end = idx;
      layout.log_sigma_bym2_idx = -1;
      layout.logit_rho_bym2_idx = -1;
      layout.theta_bym2_start = layout.theta_bym2_end = -1;
    } else {
      // ICAR: log_tau, [phi if not collapsed]
      layout.log_tau_spatial_idx = idx++;
      if (data.icar_collapsed) {
        // Collapsed: phi marginalized out, not in param vector
        layout.spatial_start = layout.spatial_end = -1;
      } else {
        layout.spatial_start = idx;
        idx += data.n_spatial_units;
        layout.spatial_end = idx;
      }
      layout.log_sigma_bym2_idx = -1;
      layout.logit_rho_bym2_idx = -1;
      layout.theta_bym2_start = layout.theta_bym2_end = -1;
      layout.logit_rho_car_idx = -1;
    }
  } else {
    layout.log_tau_spatial_idx = -1;
    layout.spatial_start = layout.spatial_end = -1;
    layout.log_sigma_bym2_idx = -1;
    layout.logit_rho_bym2_idx = -1;
    layout.theta_bym2_start = layout.theta_bym2_end = -1;
    layout.logit_rho_car_idx = -1;
  }

  // Temporal effects
  layout.has_temporal = (data.temporal_type != TemporalType::NONE);
  layout.is_ar1 = (data.temporal_type == TemporalType::AR1);
  layout.is_temporal_gp = (data.temporal_type == TemporalType::GP);

  if (layout.has_temporal) {
    if (layout.is_temporal_gp) {
      // Temporal GP: log_sigma2 + log_phi + effects
      layout.log_sigma2_temporal_gp_idx = idx++;
      layout.logit_phi_temporal_gp_idx = idx++;
      layout.log_tau_temporal_idx = -1;  // Not used for GP
      layout.logit_rho_ar1_idx = -1;
    } else {
      // RW1/RW2/AR1: log_tau + effects (+ rho for AR1)
      layout.log_tau_temporal_idx = idx++;
      layout.log_sigma2_temporal_gp_idx = -1;
      layout.logit_phi_temporal_gp_idx = -1;

      // AR1 also has rho parameter
      if (layout.is_ar1) {
        layout.logit_rho_ar1_idx = idx++;
      } else {
        layout.logit_rho_ar1_idx = -1;
      }
    }

    // Temporal effects: n_times * n_groups parameters
    layout.temporal_start = idx;
    idx += data.n_temporal_params;
    layout.temporal_end = idx;
  } else {
    layout.log_tau_temporal_idx = -1;
    layout.logit_rho_ar1_idx = -1;
    layout.log_sigma2_temporal_gp_idx = -1;
    layout.logit_phi_temporal_gp_idx = -1;
    layout.temporal_start = layout.temporal_end = -1;
  }

  // Zero-inflation parameters
  layout.has_zi = (data.zi_type != ZIType::NONE);

  if (layout.has_zi) {
    layout.beta_zi_start = idx;
    idx += data.p_zi;
    layout.beta_zi_end = idx;
  } else {
    layout.beta_zi_start = layout.beta_zi_end = -1;
  }

  // One-inflation parameters (for OI-binomial and ZOIB)
  layout.has_oi = (data.zi_type == ZIType::OI_BINOMIAL || data.zi_type == ZIType::ZOIB);

  if (layout.has_oi && data.p_oi > 0) {
    layout.beta_oi_start = idx;
    idx += data.p_oi;
    layout.beta_oi_end = idx;
  } else {
    layout.beta_oi_start = layout.beta_oi_end = -1;
  }

  // GP spatial parameters
  layout.is_gp = (data.spatial_type == SpatialType::GP);
  layout.is_multiscale_gp = (data.spatial_type == SpatialType::MULTISCALE_GP);

  layout.is_gp_collapsed = layout.is_gp && data.has_gp && data.gp_collapsed;

  if (layout.is_gp && data.has_gp) {
    layout.log_sigma2_gp_idx = idx++;
    layout.log_phi_gp_idx = idx++;
    if (!data.gp_collapsed) {
      // Standard: allocate slots for GP effects
      layout.gp_w_start = idx;
      idx += data.gp_data.n_obs;
      layout.gp_w_end = idx;
    } else {
      // Collapsed: GP effects marginalized out, not in param vector
      layout.gp_w_start = layout.gp_w_end = -1;
    }
  } else {
    layout.log_sigma2_gp_idx = -1;
    layout.log_phi_gp_idx = -1;
    layout.gp_w_start = layout.gp_w_end = -1;
  }

  // Multi-scale GP parameters
  if (layout.is_multiscale_gp && data.has_multiscale_gp) {
    // Number of spatial effects per scale: m^2 for HSGP, n_obs for NNGP
    int n_per_scale = data.msgp_is_hsgp ? data.msgp_hsgp_data.m_total
                                        : data.multiscale_gp_data.n_obs;
    // Local scale
    layout.log_sigma2_gp_local_idx = idx++;
    layout.log_phi_gp_local_idx = idx++;  // log_lengthscale for HSGP
    layout.gp_local_start = idx;
    idx += n_per_scale;
    layout.gp_local_end = idx;

    // Regional scale
    layout.log_sigma2_gp_regional_idx = idx++;
    layout.log_phi_gp_regional_idx = idx++;
    layout.gp_regional_start = idx;
    idx += n_per_scale;
    layout.gp_regional_end = idx;
  } else {
    layout.log_sigma2_gp_local_idx = -1;
    layout.log_phi_gp_local_idx = -1;
    layout.gp_local_start = layout.gp_local_end = -1;
    layout.log_sigma2_gp_regional_idx = -1;
    layout.log_phi_gp_regional_idx = -1;
    layout.gp_regional_start = layout.gp_regional_end = -1;
  }

  // Multi-scale temporal parameters
  layout.has_multiscale_temporal = data.has_multiscale_temporal;

  if (layout.has_multiscale_temporal) {
    // Trend component
    if (data.multiscale_temporal_data.trend_type != tulpa_temporal::TemporalType::NONE) {
      layout.log_sigma2_trend_idx = idx++;
      layout.trend_start = idx;
      idx += data.multiscale_temporal_data.n_times;
      layout.trend_end = idx;
    } else {
      layout.log_sigma2_trend_idx = -1;
      layout.trend_start = layout.trend_end = -1;
    }

    // Seasonal component
    if (data.multiscale_temporal_data.seasonal_period > 0) {
      layout.log_sigma2_seasonal_idx = idx++;
      layout.seasonal_start = idx;
      idx += data.multiscale_temporal_data.seasonal_period;
      layout.seasonal_end = idx;
    } else {
      layout.log_sigma2_seasonal_idx = -1;
      layout.seasonal_start = layout.seasonal_end = -1;
    }

    // Short-term component
    if (data.multiscale_temporal_data.short_term_type != tulpa_temporal::TemporalType::NONE) {
      layout.log_sigma2_short_idx = idx++;
      if (data.multiscale_temporal_data.short_term_type == tulpa_temporal::TemporalType::AR1) {
        layout.logit_rho_short_idx = idx++;
      } else {
        layout.logit_rho_short_idx = -1;
      }
      layout.short_term_start = idx;
      idx += data.multiscale_temporal_data.n_times;
      layout.short_term_end = idx;
    } else {
      layout.log_sigma2_short_idx = -1;
      layout.logit_rho_short_idx = -1;
      layout.short_term_start = layout.short_term_end = -1;
    }
  } else {
    layout.log_sigma2_trend_idx = -1;
    layout.trend_start = layout.trend_end = -1;
    layout.log_sigma2_seasonal_idx = -1;
    layout.seasonal_start = layout.seasonal_end = -1;
    layout.log_sigma2_short_idx = -1;
    layout.logit_rho_short_idx = -1;
    layout.short_term_start = layout.short_term_end = -1;
  }

  // SVC (Spatially-Varying Coefficients) parameters
  layout.has_svc = data.has_svc;
  if (layout.has_svc && data.svc_data.n_svc > 0) {
    // Log sigma2 per SVC term (spatial variance)
    layout.log_sigma2_svc_start = idx;
    idx += data.svc_data.n_svc;
    layout.log_sigma2_svc_end = idx;

    // Log phi/lengthscale per SVC term (spatial range)
    layout.log_phi_svc_start = idx;
    idx += data.svc_data.n_svc;
    layout.log_phi_svc_end = idx;

    // SVC spatial parameters:
    //   NNGP: w_flat[j * n_obs + i] for j in 0..n_svc-1, i in 0..n_obs-1
    //   HSGP: beta[j * m_total + k] for j in 0..n_svc-1, k in 0..m^2-1
    layout.svc_w_start = idx;
    if (data.svc_is_hsgp) {
      idx += data.svc_data.n_svc * data.svc_hsgp_data.m_total;
    } else {
      idx += data.svc_data.n_svc * data.svc_data.n_obs;
    }
    layout.svc_w_end = idx;
  } else {
    layout.log_sigma2_svc_start = layout.log_sigma2_svc_end = -1;
    layout.log_phi_svc_start = layout.log_phi_svc_end = -1;
    layout.svc_w_start = layout.svc_w_end = -1;
  }

  // Latent factors for unmeasured confounders
  layout.has_latent = data.has_latent;
  if (layout.has_latent && data.latent_n_factors > 0) {
    // Log sigma for each factor
    layout.log_sigma_latent_start = idx;
    idx += data.latent_n_factors;
    layout.log_sigma_latent_end = idx;

    // Factor scores (N x K)
    layout.latent_factor_start = idx;
    idx += data.N * data.latent_n_factors;
    layout.latent_factor_end = idx;
  } else {
    layout.log_sigma_latent_start = layout.log_sigma_latent_end = -1;
    layout.latent_factor_start = layout.latent_factor_end = -1;
  }

  // Spatiotemporal interaction
  layout.has_spatiotemporal = data.has_spatiotemporal;
  layout.is_st_gp = (data.has_spatiotemporal &&
                     (data.spatiotemporal_data.type == STType::SEPARABLE ||
                      data.spatiotemporal_data.type == STType::NONSEP_GP));

  if (layout.has_spatiotemporal && data.spatiotemporal_data.type != STType::NONE) {
    // log_tau for interaction precision
    layout.log_tau_st_idx = idx++;

    // Second precision removed for Type IV (single tau suffices)
    layout.log_tau_st2_idx = -1;

    // AR1 rho if temporal uses AR1
    if (data.spatiotemporal_data.temporal_type == TemporalType::AR1) {
      layout.logit_rho_st_idx = idx++;
    } else {
      layout.logit_rho_st_idx = -1;
    }

    // GP range parameters (for separable/non-separable GP)
    if (layout.is_st_gp) {
      layout.log_phi_st_space_idx = idx++;
      layout.log_phi_st_time_idx = idx++;
    } else {
      layout.log_phi_st_space_idx = -1;
      layout.log_phi_st_time_idx = -1;
    }

    // HSGP-ST: separate sigma2 and lengthscale for spectral basis interaction
    layout.is_st_hsgp = data.st_is_hsgp;
    if (data.st_is_hsgp) {
      layout.log_sigma2_st_hsgp_idx = idx++;
      layout.log_lengthscale_st_hsgp_idx = idx++;
    } else {
      layout.log_sigma2_st_hsgp_idx = -1;
      layout.log_lengthscale_st_hsgp_idx = -1;
    }

    // Spatiotemporal interaction effects
    layout.st_delta_start = idx;
    idx += data.spatiotemporal_data.n_params;
    layout.st_delta_end = idx;
  } else {
    layout.log_tau_st_idx = -1;
    layout.log_tau_st2_idx = -1;
    layout.logit_rho_st_idx = -1;
    layout.log_phi_st_space_idx = -1;
    layout.log_phi_st_time_idx = -1;
    layout.log_sigma2_st_hsgp_idx = -1;
    layout.log_lengthscale_st_hsgp_idx = -1;
    layout.is_st_hsgp = false;
    layout.st_delta_start = layout.st_delta_end = -1;
  }

  // HSGP (Hilbert Space GP) parameters
  layout.is_hsgp = (data.spatial_type == SpatialType::HSGP);
  if (layout.is_hsgp && data.has_hsgp) {
    layout.log_sigma2_hsgp_idx = idx++;
    layout.log_lengthscale_hsgp_idx = idx++;
    layout.hsgp_beta_start = idx;
    idx += data.hsgp_data.m_total;  // m^2 basis coefficients
    layout.hsgp_beta_end = idx;
  } else {
    layout.log_sigma2_hsgp_idx = -1;
    layout.log_lengthscale_hsgp_idx = -1;
    layout.hsgp_beta_start = layout.hsgp_beta_end = -1;
  }

  // SPDE (continuous Matern via FEM). Joint NUTS layout: the latent block
  // params[spde_w_start..spde_w_end) holds z (non-centered draws), and the
  // two hyper slots log_kappa_spde_idx / log_tau_spde_idx come immediately
  // after. The non-centered transform w = L^{-T}(theta) z is applied
  // downstream in initialize_generic_state. (kappa, tau_spde) on
  // ModelData::spde_data are kept as fallback fixed-hyper values for the
  // legacy nested-Laplace outer loop in cpp_nested_laplace_spde; the
  // joint-NUTS path overrides them per draw via params[log_*_spde_idx].
  layout.is_spde = (data.spatial_type == SpatialType::SPDE);
  if (layout.is_spde && data.has_spde) {
    layout.spde_w_start = idx;
    idx += data.spde_data.n_mesh;
    layout.spde_w_end = idx;
    if (data.spde_data.joint_hypers) {
      layout.log_kappa_spde_idx = idx++;
      layout.log_tau_spde_idx   = idx++;
    } else {
      layout.log_kappa_spde_idx = -1;
      layout.log_tau_spde_idx   = -1;
    }
  } else {
    layout.spde_w_start = layout.spde_w_end = -1;
    layout.log_kappa_spde_idx = -1;
    layout.log_tau_spde_idx   = -1;
  }

  // TVC (Temporally-Varying Coefficients) parameters
  layout.has_tvc = data.has_tvc;
  if (layout.has_tvc && data.tvc_data.n_tvc > 0) {
    // Log precision per TVC term
    layout.log_tau_tvc_start = idx;
    idx += data.tvc_data.n_tvc;
    layout.log_tau_tvc_end = idx;

    // AR1 rho parameters (only if structure is AR1)
    if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
      layout.logit_rho_tvc_start = idx;
      idx += data.tvc_data.n_tvc;
      layout.logit_rho_tvc_end = idx;
    } else {
      layout.logit_rho_tvc_start = layout.logit_rho_tvc_end = -1;
    }

    // TVC values: w[g, j, t] for g in groups, j in tvc terms, t in times
    // Layout: w_flat[g * n_tvc * n_times + j * n_times + t]
    layout.tvc_w_start = idx;
    idx += data.tvc_data.n_groups * data.tvc_data.n_tvc * data.tvc_data.n_times;
    layout.tvc_w_end = idx;
  } else {
    layout.log_tau_tvc_start = layout.log_tau_tvc_end = -1;
    layout.logit_rho_tvc_start = layout.logit_rho_tvc_end = -1;
    layout.tvc_w_start = layout.tvc_w_end = -1;
  }

  // Generic multi-process: extra parameters from LikelihoodSpec
  if (data.n_processes > 0 && data.likelihood_spec != nullptr) {
    const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);
    if (spec->n_extra_params > 0) {
      layout.extra_offset = idx;
      layout.n_extra_params = spec->n_extra_params;
      idx += spec->n_extra_params;
    }
    if (spec->extend_layout) {
      spec->extend_layout(data, layout, data.model_response_data);
    }
  }

  layout.total_params = idx;
  return layout;
}

int get_n_params(const ModelData& data) {
  ParamLayout layout = compute_param_layout(data);
  return layout.total_params;
}


}
