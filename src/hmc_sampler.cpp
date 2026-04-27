// hmc_sampler.cpp
// Full HMC/NUTS backend with spatial, temporal, and ZI support
// Provides Stan-free Bayesian inference for all ratiod models

#include "hmc_sampler.h"
#include "linalg_fast.h"
#include <RcppEigen.h>
#include "hmc_progress.h"
#include "autodiff.h"
#include "autodiff_utils.h"
#include "hmc_gp_autodiff.h"
#include "hmc_gp_collapsed.h"
#include "hmc_icar_collapsed.h"
#include "hmc_temporal_autodiff.h"
#include "hmc_tvc_grad.h"
#include "hmc_multiscale_temporal_grad.h"
#include "lkj_chol_helpers.h"
#include <Rcpp.h>

// Include log_post_impl.h AFTER hmc_sampler.h so types are defined
#include "log_post_impl.h"
#include "tulpa/likelihood.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace tulpa_hmc {

// =====================================================================
// Dense mass matrix: Cholesky decomposition via Eigen
// =====================================================================

bool DenseMassMatrix::update_from_covariance(const double* cov, int n_samples) {
  // Map the covariance data into an Eigen matrix (column-major)
  Eigen::Map<const Eigen::MatrixXd> C(cov, n, n);

  // Eigendecomposition for condition number control.
  // Without conditioning, ill-conditioned mass matrices force epsilon to be
  // tiny (driven by the stiffest direction), making sampling extremely slow.
  // E.g., HSGP+RW1 gets epsilon=3.2e-5 unconditioned vs ~0.01 conditioned.
  //
  // Clip eigenvalue ratio to MAX_COND so the step size ratio between the
  // loosest and stiffest directions is at most sqrt(MAX_COND) ≈ 100:1.
  constexpr double MAX_COND = 1e4;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C);
  if (eig.info() != Eigen::Success) {
    // Eigendecomposition failed — degrade to diagonal
    type = MassMatrixType::DIAG;
    adapted = true;
    for (int i = 0; i < n; i++) {
      double var_i = cov[static_cast<size_t>(i) * n + i];
      inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
      sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
    }
    return false;
  }

  Eigen::VectorXd evals = eig.eigenvalues();
  double lambda_max = evals.maxCoeff();
  double lambda_floor = std::max(lambda_max / MAX_COND, 1e-8);
  bool clipped = false;
  for (int i = 0; i < n; i++) {
    if (evals[i] < lambda_floor) {
      evals[i] = lambda_floor;
      clipped = true;
    }
  }

  // Reconstruct conditioned covariance: V * diag(λ_clipped) * V^T
  Eigen::MatrixXd C_cond;
  if (clipped) {
    const Eigen::MatrixXd& V = eig.eigenvectors();
    C_cond = V * evals.asDiagonal() * V.transpose();
  } else {
    C_cond = C;
  }

  // Cholesky of (conditioned) covariance — guaranteed to succeed after clipping
  Eigen::LLT<Eigen::MatrixXd> llt(C_cond);
  if (llt.info() != Eigen::Success) {
    // Should not happen after eigenvalue clipping, but handle gracefully
    type = MassMatrixType::DIAG;
    adapted = true;
    for (int i = 0; i < n; i++) {
      double var_i = cov[static_cast<size_t>(i) * n + i];
      inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
      sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
    }
    return false;
  }

  // Store conditioned covariance as inv_mass_dense
  std::memcpy(inv_mass_dense.data(), C_cond.data(),
              static_cast<size_t>(n) * n * sizeof(double));

  // Store Cholesky factor L
  Eigen::MatrixXd L_mat = llt.matrixL();
  std::memcpy(L_inv_mass.data(), L_mat.data(),
              static_cast<size_t>(n) * n * sizeof(double));

  // Also update diagonal for fallback and find_reasonable_epsilon compatibility
  for (int i = 0; i < n; i++) {
    double var_i = C_cond(i, i);
    inv_mass_diag[i] = std::max(1e-3, std::min(var_i, 1e3));
    sqrt_mass_diag[i] = 1.0 / std::sqrt(inv_mass_diag[i]);
  }

  adapted = true;
  return true;
}

// =====================================================================
// Parameter layout computation
// =====================================================================

ParamLayout compute_param_layout(const ModelData& data) {
  ParamLayout layout;
  int idx = 0;

  // ================================================================
  // GENERIC MULTI-PROCESS LAYOUT
  // ================================================================
  if (data.n_processes > 0) {
    layout.process_beta_start.resize(data.n_processes);
    layout.process_beta_count.resize(data.n_processes);
    for (int k = 0; k < data.n_processes; k++) {
      layout.process_beta_start[k] = idx;
      layout.process_beta_count[k] = data.processes[k].p;
      idx += data.processes[k].p;
    }
    // Legacy fields unused but set to safe values
    layout.legacy.beta_num_start = layout.legacy.beta_num_end = 0;
    layout.legacy.beta_denom_start = layout.legacy.beta_denom_end = 0;
  } else {
    // Legacy ratio layout
    layout.legacy.beta_num_start = idx;
    idx += data.legacy.p_num;
    layout.legacy.beta_num_end = idx;

    layout.legacy.beta_denom_start = idx;
    idx += data.legacy.p_denom;
    layout.legacy.beta_denom_end = idx;
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

  // Overdispersion / shape / sigma parameters
  // NEGBIN_NEGBIN: phi_num (overdispersion for num), phi_denom (overdispersion for denom)
  // POISSON_GAMMA: phi_denom (shape for gamma denom)
  // GAMMA_GAMMA: phi_num (shape for num), phi_denom (shape for denom)
  // LOGNORMAL: phi_num (sigma for num), phi_denom (sigma for denom)
  // BETA_BINOMIAL: phi_num (precision parameter)
  layout.legacy.has_phi_num = (data.legacy.model_type == ModelType::NEGBIN_NEGBIN ||
                        data.legacy.model_type == ModelType::NEGBIN_GAMMA ||
                        data.legacy.model_type == ModelType::GAMMA_GAMMA ||
                        data.legacy.model_type == ModelType::LOGNORMAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);
  layout.legacy.has_phi_denom = (data.legacy.model_type == ModelType::NEGBIN_NEGBIN ||
                          data.legacy.model_type == ModelType::POISSON_GAMMA ||
                          data.legacy.model_type == ModelType::NEGBIN_GAMMA ||
                          data.legacy.model_type == ModelType::GAMMA_GAMMA ||
                          data.legacy.model_type == ModelType::LOGNORMAL);

  if (layout.legacy.has_phi_num) {
    layout.legacy.log_phi_num_idx = idx++;
  } else {
    layout.legacy.log_phi_num_idx = -1;
  }
  if (layout.legacy.has_phi_denom) {
    layout.legacy.log_phi_denom_idx = idx++;
  } else {
    layout.legacy.log_phi_denom_idx = -1;
  }

  // Spatial effects (ICAR/BYM2 only - GP handled separately below)
  layout.has_spatial = (data.spatial_type == SpatialType::ICAR ||
                        data.spatial_type == SpatialType::BYM2);
  layout.is_bym2 = (data.spatial_type == SpatialType::BYM2);
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
    }
  } else {
    layout.log_tau_spatial_idx = -1;
    layout.spatial_start = layout.spatial_end = -1;
    layout.log_sigma_bym2_idx = -1;
    layout.logit_rho_bym2_idx = -1;
    layout.theta_bym2_start = layout.theta_bym2_end = -1;
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

// =====================================================================
// Likelihood functions
// =====================================================================

inline double log_lik_binomial(int y, int n, double eta) {
  // Numerically stable binomial log-likelihood
  if (eta > 0) {
    return y * eta - n * eta - n * std::log(1.0 + std::exp(-eta));
  } else {
    return y * eta - n * std::log(1.0 + std::exp(eta));
  }
}

inline double log_lik_negbin(int y, double mu, double phi) {
  if (mu <= 0 || phi <= 0) return -1e10;
  return std::lgamma(y + phi) - std::lgamma(phi) - std::lgamma(y + 1.0)
       + phi * std::log(phi / (mu + phi))
       + y * std::log(mu / (mu + phi));
}

inline double log_lik_poisson(int y, double mu) {
  if (mu <= 0) return -1e10;
  return y * std::log(mu) - mu - std::lgamma(y + 1.0);
}

inline double log_lik_gamma(double y, double shape, double mu) {
  if (y <= 0 || shape <= 0 || mu <= 0) return -1e10;
  double rate = shape / mu;
  return shape * std::log(rate) + (shape - 1.0) * std::log(y)
       - rate * y - std::lgamma(shape);
}

// Include vectorized gradient header AFTER log_lik_* functions and hmc_sampler.h
// so all types and helpers are defined.
#include "hmc_gradient_vectorized.h"

// Thread-local vectorized gradient workspace (avoids per-call allocation)
static thread_local vectorized::VecGradWorkspace vec_grad_ws;

// =====================================================================
// Observation log-likelihood helper (fused with gradient computation)
// Matches compute_log_post observation loop exactly. Used by specialized
// H gradient functions to avoid a separate O(N) pass.
// =====================================================================

inline double compute_obs_ll(
    const ModelData& data, int i,
    double eta_num, double eta_denom,
    double phi_num, double phi_denom
) {
  if (data.legacy.model_type == ModelType::BINOMIAL) {
    return log_lik_binomial(data.legacy.y_num[i], data.legacy.y_denom[i], eta_num);
  } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
    double mu_num = std::exp(eta_num);
    double mu_denom = std::exp(eta_denom);
    return log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num)
         + log_lik_negbin(data.legacy.y_denom[i], mu_denom, phi_denom);
  } else if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
    double mu_num = std::exp(eta_num);
    double mu_denom = std::exp(eta_denom);
    return log_lik_poisson(data.legacy.y_num[i], mu_num)
         + log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);
  } else if (data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
    double mu_num = std::exp(eta_num);
    double mu_denom = std::exp(eta_denom);
    return log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num)
         + log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);
  } else if (data.legacy.model_type == ModelType::GAMMA_GAMMA) {
    double mu_num = std::exp(eta_num);
    double mu_denom = std::exp(eta_denom);
    return log_lik_gamma(data.legacy.y_num_cont[i], phi_num, mu_num)
         + log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);
  } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
    double log_y_num = std::log(data.legacy.y_num_cont[i]);
    double log_y_denom = std::log(data.legacy.y_denom_cont[i]);
    double z_num = (log_y_num - eta_num) / phi_num;
    double z_denom = (log_y_denom - eta_denom) / phi_denom;
    return -log_y_num - std::log(phi_num) - 0.5 * z_num * z_num
           -log_y_denom - std::log(phi_denom) - 0.5 * z_denom * z_denom;
  } else if (data.legacy.model_type == ModelType::BETA_BINOMIAL) {
    double p = 1.0 / (1.0 + std::exp(-eta_num));
    int y = data.legacy.y_num[i];
    int n = data.legacy.y_denom[i];
    double alpha = p * phi_num;
    double beta_param = (1.0 - p) * phi_num;
    return std::lgamma(y + alpha) + std::lgamma(n - y + beta_param) - std::lgamma(n + phi_num)
         - std::lgamma(alpha) - std::lgamma(beta_param) + std::lgamma(phi_num)
         + tulpa::math::portable_lchoose(n, y);
  }
  return 0.0;
}

// =====================================================================
// ICAR quadratic form: phi' Q phi
// =====================================================================

// Pointer-based version (zero allocation, used in hot gradient path)
static inline double icar_quadratic_form_ptr(
    const double* phi, int J,
    const ModelData& data
) {
  double quad_form = 0.0;
  for (int i = 0; i < J; i++) {
    quad_form += data.n_neighbors[i] * phi[i] * phi[i];
    int row_start = data.adj_row_ptr[i];
    int row_end = data.adj_row_ptr[i + 1];
    for (int k = row_start; k < row_end; k++) {
      int j = data.adj_col_idx[k];
      if (j > i) {
        quad_form -= 2.0 * phi[i] * phi[j];
      }
    }
  }
  return quad_form;
}

double icar_quadratic_form(
    const std::vector<double>& phi,
    const ModelData& data
) {
  return icar_quadratic_form_ptr(phi.data(), data.n_spatial_units, data);
}

// Shared collapsed GP workspace (used by both compute_log_post and compute_gradient_gp_collapsed)
static thread_local CollapsedGPWorkspace collapsed_gp_ws;

// Shared collapsed ICAR/BYM2 workspace
static thread_local CollapsedICARWorkspace collapsed_icar_ws;

// =====================================================================
// Log-posterior computation with OpenMP parallelization
// =====================================================================

double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop,
    const double* precomputed_st_log_prior,
    const double* precomputed_tgp_log_prior
) {
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

  // ============ PRIORS ============

  // Fixed effects: N(0, sigma_beta^2)
  double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
  for (int j = 0; j < data.legacy.p_num; j++) {
    log_post -= 0.5 * tau_beta * beta_num[j] * beta_num[j];
  }
  for (int j = 0; j < data.legacy.p_denom; j++) {
    log_post -= 0.5 * tau_beta * beta_denom[j] * beta_denom[j];
  }

  // Random effects priors (supports multiple crossed RE terms with slopes)
  // Non-centered parameterization: params store z ~ N(0,1), re = sigma * (L * z)
  // Pre-compute actual RE values from z for use in the likelihood loop.
  // Only allocate when the observation loop will run (skip_obs_loop=true avoids
  // this heap allocation on every gradient call — saves ~100ns per call).
  std::vector<double> re_nc_flat;
  if (!skip_obs_loop) {
    re_nc_flat.assign(params.size(), 0.0);
  }
  if (layout.has_re) {
    int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;

    for (int t = 0; t < n_terms; t++) {
      int n_groups = (n_terms > 1 || data.n_re_terms > 0) ? data.re_n_groups_multi[t] : data.n_re_groups;
      int n_coefs = layout.has_re_slopes ? layout.re_n_coefs_multi[t] : 1;
      bool is_correlated = layout.has_re_slopes && layout.re_correlated_multi[t];

      // Extract sigma parameters for this term
      std::vector<double> sigmas(n_coefs);
      for (int c = 0; c < n_coefs; c++) {
        int log_sigma_idx;
        if (layout.has_re_slopes) {
          log_sigma_idx = layout.log_sigma_re_slopes[t][c];
        } else if (n_terms > 1) {
          log_sigma_idx = layout.log_sigma_re_multi[t];
        } else {
          log_sigma_idx = layout.log_sigma_re_idx;
        }
        double log_sigma = params[log_sigma_idx];
        sigmas[c] = std::exp(log_sigma);

        // Half-Cauchy(0, scale) prior for each sigma
        double ratio = sigmas[c] / data.sigma_re_scale;
        log_post -= std::log(1.0 + ratio * ratio);
        log_post += log_sigma;  // Jacobian
      }

      // For correlated slopes: LKJ prior on correlation matrix via Cholesky.
      // Parameterization: Sigma = diag(sigma) * L * L' * diag(sigma).
      // Tanh-parameterized L plus LKJ(eta=2) prior — see lkj_chol_helpers.h.
      std::vector<double> L_flat;
      if (is_correlated && n_coefs > 1) {
        int chol_start = layout.chol_re_start_multi[t];
        L_flat.resize(n_coefs * n_coefs, 0.0);

        double log_jac_tanh = 0.0;
        if (!tulpa::build_L_from_raw(&params[chol_start], n_coefs,
                                     L_flat.data(), &log_jac_tanh)) {
          return -std::numeric_limits<double>::infinity();
        }
        log_post += log_jac_tanh;
        log_post += tulpa::lkj_log_prior_density(L_flat.data(), n_coefs, /*eta=*/2.0);
      }

      // Get RE parameters for this term
      int re_start = (n_terms > 1 || layout.has_re_slopes) ? layout.re_start_multi[t] : layout.re_start;

      // Non-centered parameterization for correlated slopes:
      // Params store z ~ N(0,1), compute re = diag(sigma) * L * z
      // This eliminates the funnel geometry that plagues centered parameterization.
      // For uncorrelated slopes, params store z ~ N(0,1), re = sigma * z
      if (is_correlated && n_coefs > 1) {
        // Pre-compute re from z for all groups: re[g] = diag(sigma) * L * z[g]
        // Store in re_nc_flat for use in likelihood loop only
        if (!skip_obs_loop) {
          tulpa::compute_u_eff(L_flat.data(), n_coefs, sigmas.data(),
                                &params[re_start], n_groups,
                                &re_nc_flat[re_start]);
        }

        // N(0, I) prior on z (trivial in non-centered)
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            double z_gc = params[re_start + g * n_coefs + c];
            log_post -= 0.5 * z_gc * z_gc;
          }
        }
        // No log-determinant term: in non-centered parameterization,
        // the |det(diag(sigma)*L)| from the change of variables cancels exactly
        // with the |Sigma|^{-1/2} normalization of the MVN density.

      } else if (data.re_parameterization == 1) {
        // Uncorrelated non-centered: params store z ~ N(0,1), re = sigma * z
        // Pre-compute re = sigma * z for observation loop
        if (!skip_obs_loop) {
          for (int g = 0; g < n_groups; g++) {
            for (int c = 0; c < n_coefs; c++) {
              double z_gc = params[re_start + g * n_coefs + c];
              re_nc_flat[re_start + g * n_coefs + c] = sigmas[c] * z_gc;
            }
          }
        }
        // N(0, I) prior on z
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            double z_gc = params[re_start + g * n_coefs + c];
            log_post -= 0.5 * z_gc * z_gc;
          }
        }
      } else {
        // Uncorrelated centered: params store actual re values, prior is N(0, sigma_c^2)
        for (int g = 0; g < n_groups; g++) {
          for (int c = 0; c < n_coefs; c++) {
            double re_val = params[re_start + g * n_coefs + c];
            double tau_re = 1.0 / (sigmas[c] * sigmas[c] + 1e-10);
            log_post -= 0.5 * tau_re * re_val * re_val;
            log_post += 0.5 * std::log(tau_re);
          }
        }
        log_post -= 0.5 * n_groups * n_coefs * std::log(2.0 * M_PI);
      }
    }
  }

  // Overdispersion: Gamma prior
  if (layout.legacy.has_phi_num) {
    log_post += (data.phi_prior_shape - 1.0) * log_phi_num
              - data.phi_prior_rate * phi_num + log_phi_num;
  }
  if (layout.legacy.has_phi_denom) {
    log_post += (data.phi_prior_shape - 1.0) * log_phi_denom
              - data.phi_prior_rate * phi_denom + log_phi_denom;
  }

  // Spatial priors
  if (layout.has_spatial) {
    int J = data.n_spatial_units;

    if (layout.is_bym2) {
      // BYM2 Riebler: Half-Cauchy on sigma_total (always needed, even collapsed)
      double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);  // recover sigma_total
      double ratio = sigma_total / data.sigma_re_scale;
      log_post -= std::log(1.0 + ratio * ratio);
      log_post += params[layout.log_sigma_bym2_idx];  // Jacobian for log transform

      // Uniform(0,1) = Beta(1,1) on rho with logit Jacobian:
      // log p(logit_rho) = log(rho) + log(1-rho)
      log_post += std::log(rho_bym2) + std::log(1.0 - rho_bym2);

      if (!data.bym2_collapsed) {
        // Standard: phi/theta are explicit params
        // phi_scaled ~ N(0, Q^{-1}) with soft sum-to-zero
        std::vector<double> phi_vec(phi_spatial, phi_spatial + J);
        double quad = icar_quadratic_form(phi_vec, data);
        log_post -= 0.5 * quad;

        {
          double phi_sum = 0.0;
          for (int j = 0; j < J; j++) phi_sum += phi_spatial[j];
          log_post -= 0.5 * 0.01 * phi_sum * phi_sum;
        }

        // theta ~ N(0, I)
        for (int j = 0; j < J; j++) {
          log_post -= 0.5 * theta_bym2[j] * theta_bym2[j];
        }
      }
      // Collapsed: phi/theta priors already included in collapsed_lp above
    } else {
      // ICAR
      // tau ~ Gamma(shape, rate) (always needed, even collapsed)
      log_post += (data.tau_spatial_shape - 1.0) * log_tau_spatial
                - data.tau_spatial_rate * tau_spatial + log_tau_spatial;

      if (!data.icar_collapsed) {
        // Standard: phi is explicit param
        // phi ~ ICAR(tau): p(phi|tau) propto tau^{(J-1)/2} exp(-0.5 * tau * phi'Qphi)
        std::vector<double> phi_vec(phi_spatial, phi_spatial + J);
        double quad = icar_quadratic_form(phi_vec, data);
        log_post += 0.5 * (J - 1) * log_tau_spatial - 0.5 * tau_spatial * quad;
      }
      // Collapsed: ICAR prior on phi* + 0.5*(J-1)*log(tau) already in collapsed_lp above
    }
  }

  // ZI coefficient priors
  const double* beta_zi = nullptr;
  if (layout.has_zi) {
    beta_zi = &params[layout.beta_zi_start];
    // N(0, zi_prior_sd^2) prior on ZI coefficients
    double tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd + 1e-10);
    for (int j = 0; j < data.p_zi; j++) {
      log_post -= 0.5 * tau_zi * beta_zi[j] * beta_zi[j];
    }
  }

  // OI coefficient priors (for OI_BINOMIAL and ZOIB)
  const double* beta_oi = nullptr;
  if (layout.has_oi && data.p_oi > 0) {
    beta_oi = &params[layout.beta_oi_start];
    // N(0, oi_prior_sd^2) prior on OI coefficients
    double tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd + 1e-10);
    for (int j = 0; j < data.p_oi; j++) {
      log_post -= 0.5 * tau_oi * beta_oi[j] * beta_oi[j];
    }
  }

  // Temporal priors
  double tau_temporal = 1.0, log_tau_temporal = 0.0;
  double rho_ar1 = 0.5;
  const double* phi_temporal = nullptr;
  double sigma2_temporal_gp = 1.0, phi_temporal_gp = 1.0;
  std::vector<double> temporal_f_nc;  // Reconstructed f for NC temporal GP

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
        log_post += tulpa_temporal_gp::log_prior_temporal_sigma2_pc(
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
  std::vector<double> gp_w_nc_buf;  // Buffer for NC-reconstructed w

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
  std::vector<double> msgp_hsgp_f_local, msgp_hsgp_f_regional;  // HSGP-MSGP fields

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
  std::vector<double> hsgp_f;

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

  // Latent factor priors
  std::vector<double> latent_sigma;
  std::vector<double> latent_factors_vec;
  if (layout.has_latent && data.latent_n_factors > 0) {
    int K = data.latent_n_factors;
    int N = data.N;

    // Extract log_sigma parameters
    std::vector<double> log_sigma_latent(K);
    for (int k = 0; k < K; k++) {
      log_sigma_latent[k] = params[layout.log_sigma_latent_start + k];
    }

    // Extract sigma (exponentiated)
    latent_sigma.resize(K);
    for (int k = 0; k < K; k++) {
      latent_sigma[k] = std::exp(log_sigma_latent[k]);
    }

    // Extract factor scores (N x K, row-major)
    int n_factor_params = N * K;
    latent_factors_vec.resize(n_factor_params);
    for (int j = 0; j < n_factor_params; j++) {
      latent_factors_vec[j] = params[layout.latent_factor_start + j];
    }

    // Apply constraint (sum-to-zero or first-zero)
    if (data.latent_constraint == 0) {  // sum_to_zero
      tulpa_latent::apply_sum_to_zero(latent_factors_vec, N, K);
    } else {  // first_zero
      tulpa_latent::apply_first_zero(latent_factors_vec, N, K);
    }

    // PC prior on sigma (exponential prior with Jacobian)
    log_post += tulpa_latent::latent_sigma_log_prior(log_sigma_latent,
                                                       data.latent_sigma_prior_rate);

    // Standard normal prior on factor scores
    tulpa_latent::LatentConstraint constraint =
      (data.latent_constraint == 0) ? tulpa_latent::LatentConstraint::SUM_TO_ZERO
                                    : tulpa_latent::LatentConstraint::FIRST_ZERO;
    log_post += tulpa_latent::latent_factor_log_prior(latent_factors_vec, N, K, constraint);
  }

  // Spatiotemporal interaction priors
  double tau_st = 1.0, tau_st2 = 1.0, rho_st = 0.0;
  double phi_st_space = 1.0, phi_st_time = 1.0;
  const double* st_delta = nullptr;

  if (layout.has_spatiotemporal && data.spatiotemporal_data.type != STType::NONE) {
    // Extract precision parameter
    double log_tau_st = params[layout.log_tau_st_idx];
    tau_st = std::exp(log_tau_st);

    // PC prior on tau (exponential on sigma = 1/sqrt(tau))
    double sigma_st = 1.0 / std::sqrt(tau_st);
    double lambda = -std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U;
    log_post += std::log(lambda) - lambda * sigma_st - std::log(2.0 * sigma_st);
    log_post += log_tau_st;  // Jacobian for log transform

    // Single precision for all ST types (including Type IV)
    // tau_st2 always 1.0 — removed non-identifiable second precision

    // AR1 rho parameter
    if (layout.logit_rho_st_idx >= 0) {
      double logit_rho_st = params[layout.logit_rho_st_idx];
      rho_st = 2.0 / (1.0 + std::exp(-logit_rho_st)) - 1.0;  // Map to (-1, 1)

      // Uniform(-1, 1) prior on rho
      // Jacobian for logit((rho+1)/2) transform
      double x = (rho_st + 1.0) / 2.0;
      log_post += std::log(x) + std::log(1.0 - x);
    }

    // GP range parameters
    if (layout.is_st_gp) {
      double log_phi_space = params[layout.log_phi_st_space_idx];
      double log_phi_time = params[layout.log_phi_st_time_idx];
      phi_st_space = std::exp(log_phi_space);
      phi_st_time = std::exp(log_phi_time);

      // Uniform prior within bounds
      if (phi_st_space < data.st_phi_space_prior_lower ||
          phi_st_space > data.st_phi_space_prior_upper) {
        return -std::numeric_limits<double>::infinity();
      }
      if (phi_st_time < data.st_phi_time_prior_lower ||
          phi_st_time > data.st_phi_time_prior_upper) {
        return -std::numeric_limits<double>::infinity();
      }
      log_post += log_phi_space + log_phi_time;  // Jacobians
    }

    // Spatiotemporal interaction effects
    // When precomputed_st_log_prior is provided (from gradient function), use it
    // to avoid recomputing the expensive Kronecker quadratic form (O(S*T^2) with
    // heap allocations). The precomputed value includes:
    //   - GMRF quadratic form (spatiotemporal_log_prior)
    //   - Rank/Jacobian terms
    //   - Sum-to-zero penalty
    if (precomputed_st_log_prior) {
      log_post += *precomputed_st_log_prior;
    } else {
      const double* z_or_delta = &params[layout.st_delta_start];
      int S = data.spatiotemporal_data.n_spatial;
      int T = data.spatiotemporal_data.n_times;
      int ST = S * T;

      // NC reparameterization for Type IV: store z, reconstruct delta
      const bool st_use_nc = (data.st_parameterization == 1 &&
                              data.spatiotemporal_data.type == STType::TYPE_IV);

      static thread_local std::vector<double> st_delta_nc;
      if (st_use_nc) {
        // Forward transform: delta = z / sqrt(tau_st)
        double inv_scale = 1.0 / std::sqrt(tau_st);
        st_delta_nc.resize(ST);
        for (int k = 0; k < ST; k++) {
          st_delta_nc[k] = z_or_delta[k] * inv_scale;
        }
        st_delta = st_delta_nc.data();

        // NC prior: -0.5 * z^T (Q_s ⊗ Q_t) z  (tau-free GMRF)
        log_post += tulpa_spatiotemporal::spatiotemporal_log_prior(
          z_or_delta, 1.0, 1.0, rho_st, phi_st_space, phi_st_time,
          data.spatiotemporal_data
        );

        // Add the rank term with actual tau and Jacobian correction
        int rank_space = S - 1;
        int rank_time = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T - 1) : (T - 2);
        if (data.spatiotemporal_data.temporal_cyclic) rank_time = T;
        int total_rank = rank_space * rank_time;
        // GMRF normalization: 0.5 * rank * log(tau)
        // NC Jacobian: -ST/2 * log(tau)
        // Combined: 0.5 * (rank - ST) * log(tau)
        log_post += 0.5 * (total_rank - ST) * std::log(tau_st);

        // Sum-to-zero on reconstructed delta
        log_post += tulpa_spatiotemporal::st_sum_to_zero_penalty(
          st_delta, S, T, 0.001, true, true
        );
      } else if (data.st_is_hsgp) {
        // HSGP-ST: spectral basis interaction (centered)
        // Each basis function j gets independent temporal GMRF with precision tau_st / S(lambda_j)
        st_delta = z_or_delta;
        int M = data.st_hsgp_data.m_total;

        // HSGP-ST hyperparameters
        double sigma2_st_hsgp = std::exp(params[layout.log_sigma2_st_hsgp_idx]);
        double lengthscale_st_hsgp = std::exp(params[layout.log_lengthscale_st_hsgp_idx]);

        // PC prior on sigma_st_hsgp: rate=4.6
        double sigma_st = std::sqrt(sigma2_st_hsgp);
        log_post += -4.6 * sigma_st + 0.5 * params[layout.log_sigma2_st_hsgp_idx];

        // LogNormal(0,1) on lengthscale
        double log_ls_st = params[layout.log_lengthscale_st_hsgp_idx];
        log_post += -0.5 * log_ls_st * log_ls_st;

        // Per-basis-function temporal GMRF prior
        int rank_t = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T - 1) :
                     (data.spatiotemporal_data.temporal_type == TemporalType::RW2) ? (T - 2) : T;
        if (data.spatiotemporal_data.temporal_cyclic) rank_t = T;

        for (int j = 0; j < M; j++) {
          double S_j = tulpa_hsgp::spectral_density_se(
            data.st_hsgp_data.eigenvalues[j], sigma2_st_hsgp, lengthscale_st_hsgp);
          double prec_j = tau_st / std::max(S_j, 1e-10);

          // GMRF quadratic form: -0.5 * prec_j * delta_j' Q_t delta_j
          const double* dj = &st_delta[j * T];
          double qf = 0.0;
          if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
            for (int t = 1; t < T; t++) qf += (dj[t] - dj[t-1]) * (dj[t] - dj[t-1]);
          } else if (data.spatiotemporal_data.temporal_type == TemporalType::RW2) {
            for (int t = 2; t < T; t++) {
              double d2 = dj[t] - 2.0*dj[t-1] + dj[t-2];
              qf += d2 * d2;
            }
          }
          log_post += 0.5 * rank_t * std::log(prec_j) - 0.5 * prec_j * qf;

          // Soft sum-to-zero per basis function
          double sum_j = 0.0;
          for (int t = 0; t < T; t++) sum_j += dj[t];
          log_post += -0.5 * 0.001 * sum_j * sum_j;
        }
      } else {
        // Centered parameterization (ICAR/BYM2 spatial)
        st_delta = z_or_delta;

        // Apply spatiotemporal log-prior from hmc_spatiotemporal.h
        log_post += tulpa_spatiotemporal::spatiotemporal_log_prior(
          st_delta, tau_st, tau_st2, rho_st, phi_st_space, phi_st_time,
          data.spatiotemporal_data
        );

        // Soft sum-to-zero constraint for identifiability
        log_post += tulpa_spatiotemporal::st_sum_to_zero_penalty(
          st_delta, S, T, 0.001, true, true
        );
      }
    }
  }

  // TVC (Temporally-Varying Coefficients) priors
  std::vector<double> tvc_tau;
  std::vector<double> tvc_rho;
  std::vector<double> tvc_w_flat;
  std::vector<double> tvc_eta;

  if (layout.has_tvc && data.tvc_data.n_tvc > 0) {
    int n_tvc = data.tvc_data.n_tvc;
    int n_times = data.tvc_data.n_times;
    int n_groups = data.tvc_data.n_groups;

    // Extract tau (precision) parameters
    tvc_tau.resize(n_tvc);
    for (int j = 0; j < n_tvc; j++) {
      double log_tau = params[layout.log_tau_tvc_start + j];
      tvc_tau[j] = std::exp(log_tau);

      // PC prior on tau (exponential prior on sigma = 1/sqrt(tau))
      // P(sigma > U) = alpha  =>  rate = -log(alpha) / U
      double sigma = 1.0 / std::sqrt(tvc_tau[j]);
      double rate = -std::log(0.01) / 1.0;  // P(sigma > 1) = 0.01
      log_post += std::log(rate) - rate * sigma - std::log(2.0 * sigma);
      log_post += log_tau;  // Jacobian for log transform
    }

    // Extract rho parameters for AR1 structure
    tvc_rho.resize(n_tvc, 0.0);
    if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1 &&
        layout.logit_rho_tvc_start >= 0) {
      for (int j = 0; j < n_tvc; j++) {
        double logit_rho = params[layout.logit_rho_tvc_start + j];
        tvc_rho[j] = 2.0 / (1.0 + std::exp(-logit_rho)) - 1.0;  // Map to (-1, 1)

        // Uniform(-1, 1) prior on rho
        // Jacobian for logit((rho+1)/2) transform
        double x = (tvc_rho[j] + 1.0) / 2.0;
        log_post += std::log(x) + std::log(1.0 - x);
      }
    }

    // Extract TVC values
    int n_tvc_params = n_groups * n_tvc * n_times;
    tvc_w_flat.resize(n_tvc_params);
    for (int k = 0; k < n_tvc_params; k++) {
      tvc_w_flat[k] = params[layout.tvc_w_start + k];
    }

    // TVC temporal prior
    log_post += tulpa_tvc::tvc_log_prior(tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho);

    // Soft sum-to-zero constraint for identifiability
    log_post += tulpa_tvc::tvc_sum_to_zero_penalty(tvc_w_flat, data.tvc_data, 0.001);

    // Precompute TVC contribution to linear predictor
    tvc_eta.resize(data.N, 0.0);
    tulpa_tvc::compute_tvc_eta(tvc_w_flat, data.tvc_data, tvc_eta);
  }

  // ============ SVC (Spatially-Varying Coefficients) ============
  double* svc_eta_ptr = nullptr;

  if (layout.has_svc && data.svc_data.n_svc > 0) {
    int n_svc = data.svc_data.n_svc;
    int n_obs = data.svc_data.n_obs;

    if (data.svc_is_hsgp) {
      // HSGP-based SVC: basis function approximation
      int m_total = data.svc_hsgp_data.m_total;

      svc_eta_ptr = data.svc_data.eta_ws.data();
      std::fill(svc_eta_ptr, svc_eta_ptr + data.N, 0.0);

      static thread_local tulpa_hsgp::HSGPWorkspace hsgp_ws_lp;
      hsgp_ws_lp.init(data.N, m_total);

      for (int j = 0; j < n_svc; j++) {
        double log_sigma2 = params[layout.log_sigma2_svc_start + j];
        double sigma2_j = std::exp(log_sigma2);
        double sigma_j = std::sqrt(sigma2_j);
        double rate_sigma = 4.6;
        log_post += -rate_sigma * sigma_j + 0.5 * log_sigma2;

        double log_ls = params[layout.log_phi_svc_start + j];
        log_post += -0.5 * log_ls * log_ls;  // LogNormal(0,1)

        double ls_j = std::exp(log_ls);
        const double* beta_j = &params[layout.svc_w_start + j * m_total];

        // N(0,I) prior on beta
        for (int k = 0; k < m_total; k++) {
          log_post += -0.5 * beta_j[k] * beta_j[k];
        }

        // Evaluate f_j = Phi * (sqrt(S_j) * beta_j)
        tulpa_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, ls_j, data.svc_hsgp_data, hsgp_ws_lp);

        // Accumulate into svc_eta
        for (int i = 0; i < n_obs; i++) {
          svc_eta_ptr[i] += data.svc_data.X_svc[i * n_svc + j] * hsgp_ws_lp.hsgp_f[i];
        }
      }
    } else {
      // NNGP-based SVC (original path)
      double* svc_sigma2 = data.svc_data.sigma2_ws.data();
      double* svc_phi = data.svc_data.phi_ws.data();
      double* svc_w_flat = data.svc_data.w_flat_ws.data();

      for (int j = 0; j < n_svc; j++) {
        double log_sigma2 = params[layout.log_sigma2_svc_start + j];
        svc_sigma2[j] = std::exp(log_sigma2);
        double sigma = std::sqrt(svc_sigma2[j]);
        double scale = data.svc_sigma2_prior_scale;
        log_post += -std::log(1.0 + (sigma * sigma) / (scale * scale));
        log_post += log_sigma2;
      }

      for (int j = 0; j < n_svc; j++) {
        double log_phi = params[layout.log_phi_svc_start + j];
        svc_phi[j] = std::exp(log_phi);
        if (svc_phi[j] < data.svc_phi_prior_lower || svc_phi[j] > data.svc_phi_prior_upper) {
          return -INFINITY;
        }
        log_post += log_phi;
      }

      int n_svc_params = n_svc * n_obs;
      for (int k = 0; k < n_svc_params; k++) {
        svc_w_flat[k] = params[layout.svc_w_start + k];
      }

      double* w_j = data.svc_data.w_j_ws.data();
      for (int j = 0; j < n_svc; j++) {
        for (int i = 0; i < n_obs; i++) {
          w_j[i] = svc_w_flat[j * n_obs + i];
        }
        std::vector<double> w_j_vec(w_j, w_j + n_obs);
        log_post += tulpa_svc::nngp_log_lik(w_j_vec, svc_sigma2[j], svc_phi[j], data.svc_data);
      }

      std::vector<double> svc_w_flat_vec(svc_w_flat, svc_w_flat + n_svc_params);
      log_post += tulpa_svc::svc_sum_to_zero_penalty(svc_w_flat_vec, data.svc_data, 1.0);

      svc_eta_ptr = data.svc_data.eta_ws.data();
      std::fill(svc_eta_ptr, svc_eta_ptr + data.N, 0.0);
      tulpa_svc::compute_svc_eta(svc_w_flat_vec, data.svc_data, data.svc_data.eta_ws);
    }
  }

  // ============ LIKELIHOOD (parallelized) ============
  // When skip_obs_loop is true, we skip this section entirely.
  // This is used by the fused gradient+log_post computation where
  // the observation log-likelihood is accumulated during the gradient pass.

  double log_lik = 0.0;

  if (!skip_obs_loop) {

  // NOTE: Disable OpenMP for GP models to avoid race conditions
  // The GP NNGP likelihood accesses shared data structures that may not be thread-safe
  #ifdef _OPENMP
  int use_threads = (layout.is_gp || layout.is_multiscale_gp) ? 1 : data.n_threads;
  #pragma omp parallel for reduction(+:log_lik) schedule(static) \
          num_threads(use_threads)
  #endif
  for (int i = 0; i < data.N; i++) {
    // Linear predictor for numerator (using optimized dot product)
    double eta_num = tulpa_linalg::dot_product(
        &data.legacy.X_num_flat[i * data.legacy.p_num], beta_num, data.legacy.p_num);

    // Linear predictor for denominator (using optimized dot product)
    double eta_denom = tulpa_linalg::dot_product(
        &data.legacy.X_denom_flat[i * data.legacy.p_denom], beta_denom, data.legacy.p_denom);

    // Add random effects (shared between num and denom)
    // Supports multiple crossed RE terms with slopes
    if (layout.has_re) {
      int n_terms = (data.n_re_terms > 0) ? data.n_re_terms : 1;

      if (layout.has_re_slopes) {
        // Random slopes case
        for (int t = 0; t < n_terms; t++) {
          int group_idx = data.re_group_multi_flat[i * n_terms + t];
          if (group_idx > 0) {
            int g = group_idx - 1;
            int n_coefs = layout.re_n_coefs_multi[t];
            int re_base = layout.re_start_multi[t] + g * n_coefs;
            bool is_corr_t = layout.re_correlated_multi[t] && n_coefs > 1;

            // For correlated or non-centered: use pre-computed re_nc_flat
            // For centered uncorrelated: use params directly
            bool use_nc = is_corr_t || (data.re_parameterization == 1);
            double re_contrib = use_nc ? re_nc_flat[re_base] : params[re_base];

            int n_slopes = n_coefs - 1;
            if (n_slopes > 0 && !data.re_slope_matrices[t].empty()) {
              for (int s = 0; s < n_slopes; s++) {
                double x_slope = data.re_slope_matrices[t][i * n_slopes + s];
                double re_slope = use_nc ? re_nc_flat[re_base + 1 + s] : params[re_base + 1 + s];
                re_contrib += re_slope * x_slope;
              }
            }

            eta_num += re_contrib;
            eta_denom += re_contrib;
          }
        }
      } else if (n_terms > 1) {
        // Multiple RE terms (intercept only)
        // Non-centered: use pre-computed re_nc_flat; centered: use params directly
        for (int t = 0; t < n_terms; t++) {
          int group_idx = data.re_group_multi_flat[i * n_terms + t];
          if (group_idx > 0) {
            int g = group_idx - 1;
            int re_idx = layout.re_start_multi[t] + g;
            double re_val = (data.re_parameterization == 1) ? re_nc_flat[re_idx] : params[re_idx];
            eta_num += re_val;
            eta_denom += re_val;
          }
        }
      } else {
        // Single RE term (legacy path)
        // Non-centered: use pre-computed re_nc_flat; centered: use params directly
        if (data.re_group[i] > 0) {
          int g = data.re_group[i] - 1;
          double re_val = (data.re_parameterization == 1) ? re_nc_flat[layout.re_start + g] : re[g];
          eta_num += re_val;
          eta_denom += re_val;
        }
      }
    }

    // Add spatial effect (ICAR/BYM2, not GP which is handled separately)
    if (layout.has_spatial && phi_spatial != nullptr &&
        !data.spatial_group.empty() && data.spatial_group[i] > 0) {
      int s = data.spatial_group[i] - 1;
      double spatial_effect;

      if (layout.is_bym2) {
        // Standard BYM2: u = sigma_s * scale * phi + sigma_u * theta
        double scaled_phi = phi_spatial[s] * data.bym2_scale_factor;
        spatial_effect = sigma_s_bym2 * scaled_phi + sigma_u_bym2 * theta_bym2[s];
      } else {
        // Standard ICAR
        spatial_effect = phi_spatial[s];
      }

      eta_num += spatial_effect;
      eta_denom += spatial_effect;
    }

    // Add temporal effect (shared between num and denom by default)
    if (layout.has_temporal && !data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
      int t = data.temporal_time_idx[i] - 1;  // Time index (0-based)
      int g = data.temporal_group_idx[i] - 1;  // Group index (0-based)
      int T = data.n_times;

      // Get temporal effect for this observation
      double temporal_effect = phi_temporal[g * T + t];

      // Add to both linear predictors (shared structure)
      if (data.temporal_shared) {
        eta_num += temporal_effect;
        eta_denom += temporal_effect;
      } else {
        // If not shared, only add to numerator (or we'd need separate params)
        eta_num += temporal_effect;
      }
    }

    // Add GP spatial effect (map observation to unique location)
    if (layout.is_gp && data.has_gp && gp_w != nullptr) {
      int loc_i = data.gp_data.obs_to_loc[i];
      double gp_effect = gp_w[loc_i];
      if (data.gp_data.shared) {
        eta_num += gp_effect;
        eta_denom += gp_effect;
      } else {
        eta_num += gp_effect;
      }
    }

    // Add multi-scale GP spatial effects
    if (layout.is_multiscale_gp && data.has_multiscale_gp) {
      double ms_spatial_effect;
      if (data.msgp_is_hsgp) {
        // HSGP-MSGP: observation-level effects from precomputed fields
        ms_spatial_effect = msgp_hsgp_f_local[i] + msgp_hsgp_f_regional[i];
      } else {
        // NNGP-MSGP: location-level via obs_to_loc mapping
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        ms_spatial_effect = gp_local[loc_i] + gp_regional[loc_i];
      }

      if (data.multiscale_gp_data.shared) {
        eta_num += ms_spatial_effect;
        eta_denom += ms_spatial_effect;
      } else {
        eta_num += ms_spatial_effect;
      }
    }

    // Add HSGP spatial effect (observation-level)
    if (layout.is_hsgp && data.has_hsgp && !hsgp_f.empty()) {
      double hsgp_effect = hsgp_f[i];
      if (data.hsgp_data.shared) {
        eta_num += hsgp_effect;
        eta_denom += hsgp_effect;
      } else {
        eta_num += hsgp_effect;
      }
    }

    // Add multi-scale temporal effect
    if (layout.has_multiscale_temporal) {
      double ms_temporal_effect = 0.0;
      int t_idx = data.multiscale_temporal_data.time_index[i] - 1;  // 0-based

      // Trend component
      if (trend != nullptr && t_idx >= 0 &&
          t_idx < static_cast<int>(data.multiscale_temporal_data.n_times)) {
        ms_temporal_effect += trend[t_idx];
      }

      // Seasonal component
      if (seasonal != nullptr && data.multiscale_temporal_data.seasonal_period > 0) {
        int s_idx = t_idx % data.multiscale_temporal_data.seasonal_period;
        ms_temporal_effect += seasonal[s_idx];
      }

      // Short-term component
      if (short_term != nullptr && t_idx >= 0 &&
          t_idx < static_cast<int>(data.multiscale_temporal_data.n_times)) {
        ms_temporal_effect += short_term[t_idx];
      }

      if (data.multiscale_temporal_data.shared) {
        eta_num += ms_temporal_effect;
        eta_denom += ms_temporal_effect;
      } else {
        eta_num += ms_temporal_effect;
      }
    }

    // Add latent factor effect
    if (layout.has_latent && data.latent_n_factors > 0 && !latent_factors_vec.empty()) {
      int K = data.latent_n_factors;
      double latent_effect = 0.0;
      for (int k = 0; k < K; k++) {
        latent_effect += latent_factors_vec[i * K + k] * latent_sigma[k];
      }
      if (data.latent_shared) {
        eta_num += latent_effect;
        eta_denom += latent_effect;
      } else {
        eta_num += latent_effect;
      }
    }

    // Add spatiotemporal interaction effect
    if (layout.has_spatiotemporal && st_delta != nullptr) {
      double st_effect = 0.0;
      if (data.st_is_hsgp) {
        // HSGP-ST: sum_j Phi[i,j] * delta_st[j * T + t - 1]
        int t = data.spatiotemporal_data.t_idx[i] - 1;  // 0-based
        int M = data.st_hsgp_data.m_total;
        int T_st = data.spatiotemporal_data.n_times;
        for (int j = 0; j < M; j++) {
          st_effect += data.st_hsgp_data.phi_flat[i * M + j] * st_delta[j * T_st + t];
        }
      } else {
        // ICAR-ST: direct index lookup
        int st_idx = data.spatiotemporal_data.st_flat[i];
        if (st_idx > 0) st_effect = st_delta[st_idx - 1];
      }
      if (data.spatiotemporal_data.shared) {
        eta_num += st_effect;
        eta_denom += st_effect;
      } else {
        eta_num += st_effect;
      }
    }

    // Add TVC (Temporally-Varying Coefficients) effect
    if (layout.has_tvc && !tvc_eta.empty()) {
      double tvc_effect = tvc_eta[i];
      if (data.tvc_data.shared) {
        eta_num += tvc_effect;
        eta_denom += tvc_effect;
      } else {
        eta_num += tvc_effect;
      }
    }

    // Add SVC (Spatially-Varying Coefficients) effect
    if (layout.has_svc && svc_eta_ptr != nullptr) {
      double svc_effect = svc_eta_ptr[i];
      if (data.svc_data.shared) {
        eta_num += svc_effect;
        eta_denom += svc_effect;
      } else {
        eta_num += svc_effect;
      }
    }

    // Compute ZI linear predictor if applicable (using optimized dot product)
    double logit_zi = 0.0;
    if (layout.has_zi) {
      logit_zi = tulpa_linalg::dot_product(
          &data.X_zi_flat[i * data.p_zi], beta_zi, data.p_zi);
    }

    // Compute OI linear predictor if applicable (for OI_BINOMIAL and ZOIB)
    double logit_oi = 0.0;
    if (layout.has_oi && data.p_oi > 0) {
      logit_oi = tulpa_linalg::dot_product(
          &data.X_oi_flat[i * data.p_oi], beta_oi, data.p_oi);
    }

    // Likelihood contribution
    double ll_i = 0.0;
    if (data.legacy.model_type == ModelType::BINOMIAL) {
      // Handle all binomial ZI/OI variants
      double p = 1.0 / (1.0 + std::exp(-eta_num));
      int n_trials = data.legacy.y_denom[i];
      int y = data.legacy.y_num[i];

      if (data.zi_type == tulpa_zi::ZIType::ZI_BINOMIAL) {
        ll_i = tulpa_zi::zi_binomial_lpmf_logit(y, n_trials, p, logit_zi);
      } else if (data.zi_type == tulpa_zi::ZIType::HURDLE_BINOMIAL) {
        ll_i = tulpa_zi::hurdle_binomial_lpmf_logit(y, n_trials, p, logit_zi);
      } else if (data.zi_type == tulpa_zi::ZIType::OI_BINOMIAL) {
        ll_i = tulpa_zi::oi_binomial_lpmf_logit(y, n_trials, p, logit_oi);
      } else if (data.zi_type == tulpa_zi::ZIType::ZOIB) {
        ll_i = tulpa_zi::zoib_lpmf_logit(y, n_trials, p, logit_zi, logit_oi);
      } else {
        // Plain binomial (no inflation)
        ll_i = log_lik_binomial(data.legacy.y_num[i], data.legacy.y_denom[i], eta_num);
      }
    } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
      double mu_num = std::exp(eta_num);
      double mu_denom = std::exp(eta_denom);

      // Check for zero-inflation on numerator
      if (layout.has_zi) {
        ll_i = tulpa_zi::zi_log_likelihood(data.legacy.y_num[i], mu_num, phi_num,
                                           logit_zi, data.zi_type);
      } else {
        ll_i = log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num);
      }
      // Denominator is always standard (not zero-inflated)
      ll_i += log_lik_negbin(data.legacy.y_denom[i], mu_denom, phi_denom);

    } else if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
      double mu_num = std::exp(eta_num);
      double mu_denom = std::exp(eta_denom);

      // Check for zero-inflation on numerator
      if (layout.has_zi) {
        ll_i = tulpa_zi::zi_log_likelihood(data.legacy.y_num[i], mu_num, phi_num,
                                           logit_zi, data.zi_type);
      } else {
        ll_i = log_lik_poisson(data.legacy.y_num[i], mu_num);
      }
      // Denominator is gamma (continuous)
      ll_i += log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);

    } else if (data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
      double mu_num = std::exp(eta_num);
      double mu_denom = std::exp(eta_denom);

      // Check for zero-inflation on numerator
      if (layout.has_zi) {
        ll_i = tulpa_zi::zi_log_likelihood(data.legacy.y_num[i], mu_num, phi_num,
                                           logit_zi, data.zi_type);
      } else {
        ll_i = log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num);
      }
      // Denominator is gamma (continuous)
      ll_i += log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);

    } else if (data.legacy.model_type == ModelType::GAMMA_GAMMA) {
      // Gamma-Gamma: both numerator and denominator are Gamma distributed
      double mu_num = std::exp(eta_num);
      double mu_denom = std::exp(eta_denom);
      // phi_num = shape_num, phi_denom = shape_denom
      ll_i = log_lik_gamma(data.legacy.y_num_cont[i], phi_num, mu_num);
      ll_i += log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);

    } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
      // Lognormal-Lognormal: both responses are Lognormal
      // eta = mean on log scale, phi = sigma (std dev on log scale)
      double log_y_num = std::log(data.legacy.y_num_cont[i]);
      double log_y_denom = std::log(data.legacy.y_denom_cont[i]);
      // Log-lik: -log(y) - log(sigma) - 0.5*((log(y)-mu)/sigma)^2
      double z_num = (log_y_num - eta_num) / phi_num;
      double z_denom = (log_y_denom - eta_denom) / phi_denom;
      ll_i = -log_y_num - std::log(phi_num) - 0.5 * z_num * z_num;
      ll_i += -log_y_denom - std::log(phi_denom) - 0.5 * z_denom * z_denom;

    } else if (data.legacy.model_type == ModelType::BETA_BINOMIAL) {
      // Beta-binomial: overdispersed binomial
      double p = 1.0 / (1.0 + std::exp(-eta_num));
      int y = data.legacy.y_num[i];
      int n = data.legacy.y_denom[i];
      // phi_num = precision parameter (alpha + beta)
      double alpha = p * phi_num;
      double beta_param = (1.0 - p) * phi_num;
      // Beta-binomial log-likelihood
      ll_i = std::lgamma(y + alpha) + std::lgamma(n - y + beta_param) - std::lgamma(n + phi_num);
      ll_i += -std::lgamma(alpha) - std::lgamma(beta_param) + std::lgamma(phi_num);
      ll_i += tulpa::math::portable_lchoose(n, y);
    }

    log_lik += ll_i;
  }

  } // end if (!skip_obs_loop)

  log_post += log_lik;
  return log_post;
}

// =====================================================================
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

  // Check if spatial type is one we have hand-coded gradients for
  bool spatial_is_icar_bym2 = (data.spatial_type == SpatialType::ICAR ||
                               data.spatial_type == SpatialType::BYM2);

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
        // gradient is written directly into grad[chol_start..] — slots are
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

  // ============ Spatial prior gradients (ICAR and BYM2) ============
  double log_tau_spatial = 0.0, tau_spatial = 1.0;
  double sigma_s_bym2 = 1.0, sigma_u_bym2 = 1.0;
  double rho_bym2 = 0.5;  // Riebler mixing parameter
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
      // Slopes RE contribution (all terms — supports crossed+slopes)
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

      // Spatial effect (ICAR or BYM2 only — GP/HSGP/MSGP excluded above)
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

      // Slopes RE gradient scatter (all terms — supports crossed+slopes)
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

        // Denominator: Gamma (always standard) — skip if y <= 0
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

        // Denominator: Gamma (always standard) — skip if y <= 0
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
  // via its own scatter pass — the write-back chain rule runs for all paths)
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
            g_log_sigma.data(),         // grad_log_sigma (scattered — temp)
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

  // ============ Spatial GMRF prior gradients (ICAR and BYM2) ============
  if (layout.has_spatial && n_spatial > 0) {
    // Add likelihood contribution to phi_spatial gradients
    for (int s = 0; s < n_spatial; s++) {
      grad[layout.spatial_start + s] = grad_spatial_lik[s];
    }

    // ICAR prior: -0.5 * tau * phi' * Q * phi where Q_ij = n_neighbors[i] if i=j, -1 if i~j
    // d/d(phi[i]) = -tau * (n_neighbors[i]*phi[i] - sum_{j~i} phi[j])
    double icar_quad = 0.0;
    // BYM2 soft sum-to-zero: -0.01 * sum(phi)
    double bym2_phi_sum = 0.0;
    if (data.spatial_type == SpatialType::BYM2) {
      for (int i = 0; i < n_spatial; i++) bym2_phi_sum += phi_spatial[i];
    }
    for (int i = 0; i < n_spatial; i++) {
      double Qphi_i = data.n_neighbors[i] * phi_spatial[i];
      int row_start = data.adj_row_ptr[i];
      int row_end = data.adj_row_ptr[i + 1];
      for (int k = row_start; k < row_end; k++) {
        int j = data.adj_col_idx[k];
        Qphi_i -= phi_spatial[j];
        if (j > i) {
          double diff = phi_spatial[i] - phi_spatial[j];
          icar_quad += diff * diff;
        }
      }

      if (data.spatial_type == SpatialType::BYM2) {
        // For BYM2, ICAR prior has no tau scaling (it's absorbed into sigma/rho)
        // + soft sum-to-zero gradient
        grad[layout.spatial_start + i] += -Qphi_i - 0.01 * bym2_phi_sum;
      } else {
        // For plain ICAR
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

// Gradient mode: controls which gradient function is used.
// Moved here (before verify_gradient_runtime) so mode-aware functions
// can reference it. Previously defined later in the file.
static GradientMode g_gradient_mode = GradientMode::AUTO;

// Forward declarations for autodiff gradient functions (needed by verify_gradient_runtime)
void compute_gradient_arena(const std::vector<double>&, const ModelData&, const ParamLayout&,
                            std::vector<double>&, double*);
void compute_gradient_forward(const std::vector<double>&, const ModelData&, const ParamLayout&,
                              std::vector<double>&, double*);
void compute_gradient_autodiff(const std::vector<double>&, const ModelData&, const ParamLayout&,
                               std::vector<double>&, double*);

// =====================================================================
// Numerical gradient (fallback for complex models)
// =====================================================================

void compute_gradient_numerical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
  int n = params.size();
  grad.resize(n);

  // Compute log_post at central point if requested (cheap: one extra eval)
  if (log_post_out) {
    *log_post_out = compute_log_post(params, data, layout);
  }

  double h = 1e-5;

  for (int i = 0; i < n; i++) {
    std::vector<double> params_plus = params;
    std::vector<double> params_minus = params;

    params_plus[i] = params[i] + h;
    params_minus[i] = params[i] - h;

    double f_plus = compute_log_post(params_plus, data, layout);
    double f_minus = compute_log_post(params_minus, data, layout);

    grad[i] = (f_plus - f_minus) / (2.0 * h);
  }
}

// =====================================================================
// Numerical gradient using compute_log_post_impl<double>
// Used for verifying A_r/A modes which use compute_log_post_impl<T>.
// This ensures the numerical reference matches the same function that
// the autodiff mode differentiates (important when parameterization
// differs from H-mode, e.g. non-centered RE).
// =====================================================================

void compute_gradient_numerical_impl(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
  int n = params.size();
  grad.resize(n);

  if (log_post_out) {
    *log_post_out = tulpa::compute_log_post_impl(params, data, layout);
  }

  double h = 1e-5;

  for (int i = 0; i < n; i++) {
    std::vector<double> params_plus = params;
    std::vector<double> params_minus = params;

    params_plus[i] = params[i] + h;
    params_minus[i] = params[i] - h;

    double f_plus = tulpa::compute_log_post_impl(params_plus, data, layout);
    double f_minus = tulpa::compute_log_post_impl(params_minus, data, layout);

    grad[i] = (f_plus - f_minus) / (2.0 * h);
  }
}

// =====================================================================
// Unified gradient interface
// =====================================================================

// Debug: compare analytical vs numerical gradients
bool verify_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    double tol = 1e-4
) {
  std::vector<double> grad_analytical, grad_numerical;
  compute_gradient_analytical(params, data, layout, grad_analytical);
  compute_gradient_numerical(params, data, layout, grad_numerical);

  double max_diff = 0.0;
  int worst_idx = -1;
  for (size_t i = 0; i < grad_analytical.size(); i++) {
    double diff = std::abs(grad_analytical[i] - grad_numerical[i]);
    double scale = std::max(1.0, std::max(std::abs(grad_analytical[i]), std::abs(grad_numerical[i])));
    double rel_diff = diff / scale;
    if (rel_diff > max_diff) {
      max_diff = rel_diff;
      worst_idx = i;
    }
  }

  if (max_diff > tol) {
    Rcpp::Rcerr << "Gradient mismatch! Max rel diff: " << max_diff
                << " at param " << worst_idx
                << " (analytical: " << grad_analytical[worst_idx]
                << ", numerical: " << grad_numerical[worst_idx] << ")\n";
    return false;
  }
  return true;
}

// Runtime gradient check: compare compute_gradient() dispatcher output
// against numerical gradients at the first warmup iteration.
// Catches log-post/gradient mismatches in ALL specialized gradient functions
// (GP, HSGP, SVC, TVC, MSGP, spatiotemporal, etc.), not just the main
// compute_gradient_analytical().
bool verify_gradient_runtime(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    double tol
) {
  std::vector<double> grad_active, grad_numerical;
  // Resolve the actual gradient function that will be used
  GradientFn active_fn = resolve_gradient_fn(g_gradient_mode, data, layout);
  active_fn(params, data, layout, grad_active, nullptr);

  // Choose the correct numerical reference based on which function is active.
  // Autodiff functions differentiate compute_log_post_impl<T>, so the reference
  // must finite-diff the same template. H-mode functions use compute_log_post,
  // so the reference must finite-diff that.
  // This matters when AUTO resolves to autodiff for models like HSGP+ZI.
  bool active_is_autodiff = (active_fn == &compute_gradient_arena ||
                             active_fn == &compute_gradient_forward ||
                             active_fn == &compute_gradient_autodiff);
  if (active_is_autodiff) {
    compute_gradient_numerical_impl(params, data, layout, grad_numerical);
  } else {
    compute_gradient_numerical(params, data, layout, grad_numerical);
  }

  double max_diff = 0.0;
  int worst_idx = -1;
  for (size_t i = 0; i < grad_active.size(); i++) {
    double diff = std::abs(grad_active[i] - grad_numerical[i]);
    double scale = std::max(1.0, std::max(std::abs(grad_active[i]),
                                           std::abs(grad_numerical[i])));
    double rel_diff = diff / scale;
    if (rel_diff > max_diff) {
      max_diff = rel_diff;
      worst_idx = i;
    }
  }

  if (max_diff > tol) {
    // Use REprintf for immediate output, then Rcpp::warning for R-level notice
    REprintf("[numdenom] WARNING: gradient mismatch detected at param %d!\n"
             "  max |active - numerical| / scale = %.6e (tol = %.1e)\n"
             "  active[%d] = %.8e, numerical[%d] = %.8e\n"
             "  This indicates a bug in the specialized gradient function.\n"
             "  Falling back to numerical gradients for safety.\n",
             worst_idx, max_diff, tol,
             worst_idx, grad_active[worst_idx],
             worst_idx, grad_numerical[worst_idx]);
    return false;
  }
  return true;
}

// =====================================================================
// Autodiff gradient (O(n) - works for ALL models)
// =====================================================================

void compute_gradient_autodiff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    using namespace tulpa::ad;

    // Thread-safe: each call gets its own tape via RAII
    TapeScope tape_scope;
    Tape* tape = tape_scope.tape;

    // Create autodiff variables from parameters
    std::vector<Var> params_ad = make_vars(tape, params);

    // Compute log posterior using templated implementation
    Var log_post = tulpa::compute_log_post_impl(params_ad, data, layout);

    // Extract log_post value before backward pass (free: already computed)
    if (log_post_out) *log_post_out = log_post.val();

    // Backward pass to compute gradients
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ad);

    // TapeScope destructor handles cleanup
}

// =====================================================================
// Arena-based reverse-mode autodiff gradient (O(N) - fast, all models)
// Uses contiguous SoA memory layout with pre-computed partials.
// ~10-30x faster than tape autodiff, within 50% of hand-coded speed.
// =====================================================================

void compute_gradient_arena(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    using namespace tulpa::arena;

    int n_nodes_used = 0;
    {
        // Thread-safe: each call gets its own arena via RAII
        ArenaScope scope;
        Arena* arena = scope.arena();

        // Create autodiff variables from parameters
        std::vector<Var> params_ar = make_vars(arena, params);

        // Compute log posterior using templated implementation
        Var log_post = tulpa::compute_log_post_impl(params_ar, data, layout);

        // Extract log_post value before backward pass
        if (log_post_out) *log_post_out = log_post.val();

        // Backward pass to compute gradients
        log_post.backward();

        // Extract gradients
        grad = get_adjoints(params_ar);

        n_nodes_used = arena->size();
        // ArenaScope destructor handles cleanup
    }

    (void)n_nodes_used;  // suppress unused warning
}

// =====================================================================
// Forward-mode autodiff gradient (O(n×p) - but ~10x faster than tape)
// Uses dual numbers for efficient gradient computation without heap allocation
// =====================================================================

void compute_gradient_forward(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    int n_params = static_cast<int>(params.size());
    grad.assign(n_params, 0.0);

    // Forward-mode: compute one gradient component per forward pass
    // Seed each parameter in turn and evaluate
    std::vector<fwd::Dual> params_dual(n_params);

    for (int i = 0; i < n_params; i++) {
        // Seed parameter i: value=params[i], gradient=1.0
        // All others: value=params[j], gradient=0.0
        for (int j = 0; j < n_params; j++) {
            params_dual[j].val = params[j];
            params_dual[j].grad = (j == i) ? 1.0 : 0.0;
        }

        // Compute log posterior with dual numbers
        fwd::Dual log_post = tulpa::compute_log_post_impl(params_dual, data, layout);

        // Extract gradient component
        grad[i] = log_post.grad;

        // Extract log_post value on first pass (free: already computed)
        if (i == 0 && log_post_out) *log_post_out = log_post.val;
    }
}

// =====================================================================
// RE gradient helpers for specialized gradient functions
// Handles both centered and non-centered parameterizations correctly
// =====================================================================

// Initialize RE gradient with prior contribution
static inline void re_gradient_prior(
    const ModelData& data,
    const ParamLayout& layout,
    const double* re,   // re[g] = params[re_start + g]
    double* grad,
    double sigma_re
) {
    if (!layout.has_re || data.n_re_groups <= 0) return;

    // Half-Cauchy prior on sigma_re (log-scale)
    double ratio = sigma_re / data.sigma_re_scale;
    double ratio_sq = ratio * ratio;
    grad[layout.log_sigma_re_idx] = -2.0 * ratio_sq / (1.0 + ratio_sq) + 1.0;

    if (data.re_parameterization == 1) {
        // Non-centered: z ~ N(0,1), prior grad = -z
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -re[g];
        }
    } else {
        // Centered: re ~ N(0, sigma^2)
        double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
        for (int g = 0; g < data.n_re_groups; g++) {
            grad[layout.re_start + g] = -tau_re * re[g];
            grad[layout.log_sigma_re_idx] += tau_re * re[g] * re[g] - 1.0;
        }
    }
}

// Get RE value for observation (handles NC -> sigma*z transformation)
static inline double re_value_for_eta(
    const double* re,
    int g,
    double sigma_re,
    int re_parameterization
) {
    double val = re[g];
    if (re_parameterization == 1) val *= sigma_re;
    return val;
}

// Apply NC chain rule transformation after observation loop
// Must be called AFTER observation loop has accumulated likelihood gradients in grad[re+g]
static inline void re_gradient_nc_transform(
    const ModelData& data,
    const ParamLayout& layout,
    const double* params,
    double* grad,
    double sigma_re
) {
    if (!layout.has_re || data.n_re_groups <= 0 || data.re_parameterization != 1) return;

    double sigma_lik_grad = 0.0;
    for (int g = 0; g < data.n_re_groups; g++) {
        double z_g = params[layout.re_start + g];
        // Extract centered lik grad: total - prior = grad[re+g] - (-z_g) = grad[re+g] + z_g
        double centered_lik = grad[layout.re_start + g] + z_g;
        // z gradient = prior + chain rule through sigma*z
        grad[layout.re_start + g] = -z_g + sigma_re * centered_lik;
        // sigma gradient from likelihood: z_g * d_ll/d_re_g
        sigma_lik_grad += z_g * centered_lik;
    }
    // d_ll/d_log_sigma = sigma * sum(z_g * d_ll/d_re_g)
    grad[layout.log_sigma_re_idx] += sigma_re * sigma_lik_grad;
}

// =====================================================================
// Shared gradient building blocks for specialized H-mode functions
// These helpers extract the duplicated code from 11 specialized gradient
// functions into single-source-of-truth implementations.
// All are static inline — zero overhead, compiler inlines them.
// =====================================================================

// Common parameters extracted from the parameter vector
struct CommonGradParams {
    const double* beta_num;
    const double* beta_denom;
    double sigma_re;
    const double* re;
    double phi_num;
    double phi_denom;
};

// Extract common parameters from the HMC parameter vector
static inline CommonGradParams extract_common_params(
    const std::vector<double>& params,
    const ParamLayout& layout
) {
    CommonGradParams cp;
    cp.beta_num = &params[layout.legacy.beta_num_start];
    cp.beta_denom = &params[layout.legacy.beta_denom_start];
    cp.sigma_re = layout.has_re ? std::exp(params[layout.log_sigma_re_idx]) : 1.0;
    cp.re = layout.has_re ? &params[layout.re_start] : nullptr;
    cp.phi_num = layout.legacy.has_phi_num ? std::exp(params[layout.legacy.log_phi_num_idx]) : 1.0;
    cp.phi_denom = layout.legacy.has_phi_denom ? std::exp(params[layout.legacy.log_phi_denom_idx]) : 1.0;
    return cp;
}

// Beta N(0, sigma_beta^2) prior gradient
// d/d(beta) = -tau_beta * beta where tau_beta = 1/sigma_beta^2
static inline void beta_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    const double* beta_num, const double* beta_denom,
    double* grad
) {
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.legacy.p_num; j++) {
        grad[layout.legacy.beta_num_start + j] = -tau_beta * beta_num[j];
    }
    for (int j = 0; j < data.legacy.p_denom; j++) {
        grad[layout.legacy.beta_denom_start + j] = -tau_beta * beta_denom[j];
    }
}

// Phi Gamma(shape, rate) prior gradient on log-scale
// d/d(log_phi) = shape - rate*phi
// (equivalently: (shape-1) - rate*phi + 1 with Jacobian expanded)
static inline void phi_gradient_prior(
    const ModelData& data, const ParamLayout& layout,
    double phi_num, double phi_denom,
    double* grad
) {
    if (layout.legacy.has_phi_num) {
        grad[layout.legacy.log_phi_num_idx] = data.phi_prior_shape
                                       - data.phi_prior_rate * phi_num;
    }
    if (layout.legacy.has_phi_denom) {
        grad[layout.legacy.log_phi_denom_idx] = data.phi_prior_shape
                                         - data.phi_prior_rate * phi_denom;
    }
}

// Per-observation residual computation (dLL/deta for each family)
// Handles all model types: BINOMIAL, NEGBIN_NEGBIN, POISSON_GAMMA,
// NEGBIN_GAMMA, and catch-all (GAMMA_GAMMA, LOGNORMAL, BETA_BINOMIAL)
static inline void compute_obs_residuals(
    const ModelData& data, int i,
    double eta_num, double eta_denom,
    double phi_num, double phi_denom,
    double& dLL_deta_num, double& dLL_deta_denom
) {
    dLL_deta_num = 0.0;
    dLL_deta_denom = 0.0;

    if (data.legacy.model_type == ModelType::BINOMIAL) {
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        dLL_deta_num = data.legacy.y_num[i] - data.legacy.y_denom[i] * p;
    } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        dLL_deta_num = data.legacy.y_num[i] - mu_num * (data.legacy.y_num[i] + phi_num) / (mu_num + phi_num);
        dLL_deta_denom = data.legacy.y_denom[i] - mu_denom * (data.legacy.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
    } else if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        dLL_deta_num = data.legacy.y_num[i] - mu_num;
        // Gamma requires y > 0; skip if y_denom_cont <= 0 (matches log_lik_gamma)
        dLL_deta_denom = (data.legacy.y_denom_cont[i] > 0.0)
            ? phi_denom * (data.legacy.y_denom_cont[i] / mu_denom - 1.0) : 0.0;
    } else if (data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        double denom_nb = mu_num + phi_num;
        dLL_deta_num = data.legacy.y_num[i] - mu_num * (data.legacy.y_num[i] + phi_num) / denom_nb;
        dLL_deta_denom = (data.legacy.y_denom_cont[i] > 0.0)
            ? phi_denom * (data.legacy.y_denom_cont[i] / mu_denom - 1.0) : 0.0;
    } else if (data.legacy.model_type == ModelType::GAMMA_GAMMA) {
        // Gamma: dLL/d(eta) = shape * (y/mu - 1) where mu = exp(eta)
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        dLL_deta_num = (data.legacy.y_num_cont[i] > 0.0)
            ? phi_num * (data.legacy.y_num_cont[i] / mu_num - 1.0) : 0.0;
        dLL_deta_denom = (data.legacy.y_denom_cont[i] > 0.0)
            ? phi_denom * (data.legacy.y_denom_cont[i] / mu_denom - 1.0) : 0.0;
    } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
        // Lognormal: dLL/d(eta) = (log(y) - eta) / sigma^2
        double sigma_num = phi_num, sigma_denom = phi_denom;
        dLL_deta_num = (data.legacy.y_num_cont[i] > 0.0)
            ? (std::log(data.legacy.y_num_cont[i]) - eta_num) / (sigma_num * sigma_num) : 0.0;
        dLL_deta_denom = (data.legacy.y_denom_cont[i] > 0.0)
            ? (std::log(data.legacy.y_denom_cont[i]) - eta_denom) / (sigma_denom * sigma_denom) : 0.0;
    } else if (data.legacy.model_type == ModelType::BETA_BINOMIAL) {
        // Beta-binomial: mu = logit^{-1}(eta), phi = precision
        // dLL/d(eta) = d/d(eta) [lbeta(y + mu*phi, n-y + (1-mu)*phi) - lbeta(mu*phi, (1-mu)*phi)]
        // = mu*(1-mu)*phi * [digamma(y + mu*phi) - digamma(n-y + (1-mu)*phi)
        //                    - digamma(mu*phi) + digamma((1-mu)*phi)]
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        double a = p * phi_num;
        double b = (1.0 - p) * phi_num;
        double y = data.legacy.y_num[i];
        double n = data.legacy.y_denom[i];
        double dp_deta = p * (1.0 - p);  // sigmoid derivative
        double da_deta = dp_deta * phi_num;
        double db_deta = -dp_deta * phi_num;
        dLL_deta_num = da_deta * (tulpa::math::portable_digamma(y + a) - tulpa::math::portable_digamma(a))
                     + db_deta * (tulpa::math::portable_digamma(n - y + b) - tulpa::math::portable_digamma(b));
    }
}

// Per-observation ZI-aware residual computation
// Produces dLL/deta_num, dLL/deta_denom, plus phi and ZI/OI gradients per observation.
// When no ZI/OI is active, delegates to compute_obs_residuals and zeros ZI outputs.
static inline void compute_obs_residuals_zi(
    const ModelData& data, const ParamLayout& layout,
    int i, double eta_num, double eta_denom,
    double phi_num, double phi_denom,
    const double* beta_zi, const double* beta_oi,
    double& dLL_deta_num, double& dLL_deta_denom,
    double& grad_phi_num_i, double& grad_phi_denom_i,
    double& grad_logit_zi_i, double& grad_logit_oi_i
) {
    dLL_deta_num = 0.0;
    dLL_deta_denom = 0.0;
    grad_phi_num_i = 0.0;
    grad_phi_denom_i = 0.0;
    grad_logit_zi_i = 0.0;
    grad_logit_oi_i = 0.0;

    // Non-ZI fast path: delegate to compute_obs_residuals
    if (!layout.has_zi && !layout.has_oi) {
        compute_obs_residuals(data, i, eta_num, eta_denom, phi_num, phi_denom,
                              dLL_deta_num, dLL_deta_denom);
        return;
    }

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
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int n_trials = data.legacy.y_denom[i];
        int y_num_i = data.legacy.y_num[i];

        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_BINOMIAL) {
            if (y_num_i == 0) {
                double p0_binom = std::pow(1.0 - p, n_trials);
                double p0 = zi_prob + (1.0 - zi_prob) * p0_binom;
                dLL_deta_num = -(1.0 - zi_prob) * n_trials * p * p0_binom / p0;
                grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_binom) / p0;
            } else {
                dLL_deta_num = y_num_i - n_trials * p;
                grad_logit_zi_i = -zi_prob;
            }
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_BINOMIAL) {
            if (y_num_i == 0) {
                dLL_deta_num = 0.0;
                grad_logit_zi_i = -zi_prob;
            } else {
                double p0_binom = std::pow(1.0 - p, n_trials);
                double normalizer = 1.0 - p0_binom;
                if (normalizer < 1e-12) normalizer = 1e-12;
                double grad_normalizer = n_trials * p * p0_binom / normalizer;
                dLL_deta_num = (y_num_i - n_trials * p) - grad_normalizer;
                grad_logit_zi_i = 1.0 - zi_prob;
            }
        } else if (layout.has_oi && data.zi_type == tulpa_zi::ZIType::OI_BINOMIAL) {
            if (y_num_i == n_trials) {
                double pn = std::pow(p, n_trials);
                double P_yn = oi_prob + (1.0 - oi_prob) * pn;
                if (P_yn < 1e-12) P_yn = 1e-12;
                dLL_deta_num = (1.0 - oi_prob) * n_trials * pn * (1.0 - p) / P_yn;
                grad_logit_oi_i = oi_prob * (1.0 - oi_prob) * (1.0 - pn) / P_yn;
            } else {
                dLL_deta_num = y_num_i - n_trials * p;
                grad_logit_oi_i = -oi_prob;
            }
        } else if (layout.has_oi && data.zi_type == tulpa_zi::ZIType::ZOIB) {
            if (y_num_i == 0) {
                double binom_zero = std::pow(1.0 - p, n_trials);
                double A = zi_prob;
                double B = (1.0 - zi_prob) * (1.0 - oi_prob) * binom_zero;
                double P = A + B;
                if (P < 1e-12) P = 1e-12;
                double d_binom_d_eta = -n_trials * binom_zero * p;
                dLL_deta_num = (1.0 - zi_prob) * (1.0 - oi_prob) * d_binom_d_eta / P;
                grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - (1.0 - oi_prob) * binom_zero) / P;
                grad_logit_oi_i = -(1.0 - zi_prob) * binom_zero * oi_prob * (1.0 - oi_prob) / P;
            } else if (y_num_i == n_trials) {
                double pn = std::pow(p, n_trials);
                double C = oi_prob + (1.0 - oi_prob) * pn;
                double P = (1.0 - zi_prob) * C;
                if (P < 1e-12) P = 1e-12;
                double d_pn_d_eta = n_trials * pn * (1.0 - p);
                dLL_deta_num = (1.0 - zi_prob) * (1.0 - oi_prob) * d_pn_d_eta / P;
                grad_logit_zi_i = -zi_prob;
                grad_logit_oi_i = oi_prob * (1.0 - oi_prob) * (1.0 - pn) / C;
            } else {
                dLL_deta_num = y_num_i - n_trials * p;
                grad_logit_zi_i = -zi_prob;
                grad_logit_oi_i = -oi_prob;
            }
        } else {
            dLL_deta_num = y_num_i - n_trials * p;
        }

    } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
        // ---- NEGBIN_NEGBIN ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.legacy.y_num[i];
        int y_denom_i = data.legacy.y_denom[i];

        // Denominator NegBin gradient (always standard, not ZI)
        double denom_d = mu_denom + phi_denom;
        dLL_deta_denom = y_denom_i - mu_denom * (y_denom_i + phi_denom) / denom_d;
        grad_phi_denom_i = tulpa::math::portable_digamma(y_denom_i + phi_denom) - tulpa::math::portable_digamma(phi_denom)
                           + std::log(phi_denom / denom_d)
                           + (mu_denom - y_denom_i) / denom_d;

        // Numerator with ZI handling
        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_NEGBIN) {
            double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
            if (y_num_i == 0) {
                double p0 = zi_prob + (1.0 - zi_prob) * p0_nb;
                double d_p0_nb_d_mu = -phi_num * p0_nb / (phi_num + mu_num);
                dLL_deta_num = (1.0 - zi_prob) * d_p0_nb_d_mu * mu_num / p0;
                grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_nb) / p0;
                grad_phi_num_i = (1.0 - zi_prob) * p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / p0;
            } else {
                double denom_num = mu_num + phi_num;
                dLL_deta_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
                grad_logit_zi_i = -zi_prob;
                grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                                 + std::log(phi_num / denom_num)
                                 + (mu_num - y_num_i) / denom_num;
            }
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_NEGBIN) {
            if (y_num_i == 0) {
                dLL_deta_num = 0.0;
                grad_logit_zi_i = -zi_prob;
                grad_phi_num_i = 0.0;
            } else {
                double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
                double denom_num = mu_num + phi_num;
                dLL_deta_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num
                            - phi_num * p0_nb * mu_num / ((phi_num + mu_num) * (1.0 - p0_nb));
                grad_logit_zi_i = 1.0 - zi_prob;
                grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                                 + std::log(phi_num / denom_num)
                                 + (mu_num - y_num_i) / denom_num;
                grad_phi_num_i += p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / (1.0 - p0_nb);
            }
        } else {
            double denom_num = mu_num + phi_num;
            dLL_deta_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
            grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
        }

    } else if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
        // ---- POISSON_GAMMA ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.legacy.y_num[i];

        // Denominator: Gamma (always standard) — skip if y <= 0
        double y_denom_i = data.legacy.y_denom_cont[i];
        double grad_phi_gamma = 0.0;
        if (y_denom_i > 0.0) {
            dLL_deta_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
            double rate = phi_denom / mu_denom;
            grad_phi_gamma = std::log(rate) + 1.0 + std::log(y_denom_i)
                                    - tulpa::math::portable_digamma(phi_denom) - rate * y_denom_i / phi_denom;
        }

        // Numerator with ZI handling
        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_POISSON) {
            double exp_neg_mu = std::exp(-mu_num);
            if (y_num_i == 0) {
                double p0 = zi_prob + (1.0 - zi_prob) * exp_neg_mu;
                dLL_deta_num = -(1.0 - zi_prob) * exp_neg_mu * mu_num / p0;
                grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - exp_neg_mu) / p0;
                grad_phi_denom_i = grad_phi_gamma;
            } else {
                dLL_deta_num = y_num_i - mu_num;
                grad_logit_zi_i = -zi_prob;
                grad_phi_denom_i = grad_phi_gamma;
            }
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_POISSON) {
            if (y_num_i == 0) {
                dLL_deta_num = 0.0;
                grad_logit_zi_i = -zi_prob;
                grad_phi_denom_i = grad_phi_gamma;
            } else {
                double exp_neg_mu = std::exp(-mu_num);
                dLL_deta_num = y_num_i - mu_num - mu_num * exp_neg_mu / (1.0 - exp_neg_mu);
                grad_logit_zi_i = 1.0 - zi_prob;
                grad_phi_denom_i = grad_phi_gamma;
            }
        } else {
            dLL_deta_num = y_num_i - mu_num;
            grad_phi_denom_i = grad_phi_gamma;
        }

    } else if (data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
        // ---- NEGBIN_GAMMA ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        int y_num_i = data.legacy.y_num[i];

        // Denominator: Gamma (always standard) — skip if y <= 0
        double y_denom_i = data.legacy.y_denom_cont[i];
        double grad_phi_gamma = 0.0;
        if (y_denom_i > 0.0) {
            dLL_deta_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
            double rate = phi_denom / mu_denom;
            grad_phi_gamma = std::log(rate) + 1.0 + std::log(y_denom_i)
                                    - tulpa::math::portable_digamma(phi_denom) - rate * y_denom_i / phi_denom;
        }

        // Numerator with ZI handling (same as NEGBIN_NEGBIN)
        if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::ZI_NEGBIN) {
            double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
            if (y_num_i == 0) {
                double p0 = zi_prob + (1.0 - zi_prob) * p0_nb;
                double d_p0_nb_d_mu = -phi_num * p0_nb / (phi_num + mu_num);
                dLL_deta_num = (1.0 - zi_prob) * d_p0_nb_d_mu * mu_num / p0;
                grad_logit_zi_i = zi_prob * (1.0 - zi_prob) * (1.0 - p0_nb) / p0;
                grad_phi_num_i = (1.0 - zi_prob) * p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / p0;
            } else {
                double denom_num = mu_num + phi_num;
                dLL_deta_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
                grad_logit_zi_i = -zi_prob;
                grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                                 + std::log(phi_num / denom_num)
                                 + (mu_num - y_num_i) / denom_num;
            }
            grad_phi_denom_i = grad_phi_gamma;
        } else if (layout.has_zi && data.zi_type == tulpa_zi::ZIType::HURDLE_NEGBIN) {
            if (y_num_i == 0) {
                dLL_deta_num = 0.0;
                grad_logit_zi_i = -zi_prob;
                grad_phi_num_i = 0.0;
            } else {
                double p0_nb = std::pow(phi_num / (phi_num + mu_num), phi_num);
                double denom_num = mu_num + phi_num;
                dLL_deta_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num
                            - phi_num * p0_nb * mu_num / ((phi_num + mu_num) * (1.0 - p0_nb));
                grad_logit_zi_i = 1.0 - zi_prob;
                grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                                 + std::log(phi_num / denom_num)
                                 + (mu_num - y_num_i) / denom_num;
                grad_phi_num_i += p0_nb * (std::log(phi_num / (phi_num + mu_num)) + mu_num / (phi_num + mu_num)) / (1.0 - p0_nb);
            }
            grad_phi_denom_i = grad_phi_gamma;
        } else {
            double denom_num = mu_num + phi_num;
            dLL_deta_num = y_num_i - mu_num * (y_num_i + phi_num) / denom_num;
            grad_phi_num_i = tulpa::math::portable_digamma(y_num_i + phi_num) - tulpa::math::portable_digamma(phi_num)
                             + std::log(phi_num / denom_num)
                             + (mu_num - y_num_i) / denom_num;
            grad_phi_denom_i = grad_phi_gamma;
        }

    } else if (data.legacy.model_type == ModelType::GAMMA_GAMMA) {
        // ---- GAMMA_GAMMA ----
        double mu_num = std::exp(eta_num);
        double mu_denom = std::exp(eta_denom);
        double y_num_i = data.legacy.y_num_cont[i];
        double y_denom_i = data.legacy.y_denom_cont[i];

        if (y_num_i > 0.0) {
            dLL_deta_num = phi_num * (y_num_i / mu_num - 1.0);
            double rate_num = phi_num / mu_num;
            grad_phi_num_i = std::log(rate_num) + 1.0 + std::log(y_num_i)
                             - tulpa::math::portable_digamma(phi_num) - y_num_i / mu_num;
        }
        if (y_denom_i > 0.0) {
            dLL_deta_denom = phi_denom * (y_denom_i / mu_denom - 1.0);
            double rate_denom = phi_denom / mu_denom;
            grad_phi_denom_i = std::log(rate_denom) + 1.0 + std::log(y_denom_i)
                               - tulpa::math::portable_digamma(phi_denom) - y_denom_i / mu_denom;
        }

    } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
        // ---- LOGNORMAL ----
        double mu_num = eta_num;
        double mu_denom = eta_denom;
        double y_num_i = data.legacy.y_num_cont[i];
        double y_denom_i = data.legacy.y_denom_cont[i];
        double log_y_num = std::log(y_num_i);
        double log_y_denom = std::log(y_denom_i);
        double sigma_num = phi_num;
        double sigma_denom = phi_denom;
        double sigma_num_sq = sigma_num * sigma_num;
        double sigma_denom_sq = sigma_denom * sigma_denom;

        dLL_deta_num = (log_y_num - mu_num) / sigma_num_sq;
        dLL_deta_denom = (log_y_denom - mu_denom) / sigma_denom_sq;

        double z_num = (log_y_num - mu_num) / sigma_num;
        double z_denom = (log_y_denom - mu_denom) / sigma_denom;
        grad_phi_num_i = (-1.0 + z_num * z_num) / sigma_num;
        grad_phi_denom_i = (-1.0 + z_denom * z_denom) / sigma_denom;

    } else if (data.legacy.model_type == ModelType::BETA_BINOMIAL) {
        // ---- BETA_BINOMIAL ----
        double p = 1.0 / (1.0 + std::exp(-eta_num));
        int y_i = data.legacy.y_num[i];
        int n_i = data.legacy.y_denom[i];
        double alpha = p * phi_num;
        double beta_param = (1.0 - p) * phi_num;

        double psi_y_alpha = tulpa::math::portable_digamma(y_i + alpha);
        double psi_nmy_beta = tulpa::math::portable_digamma(n_i - y_i + beta_param);
        double psi_alpha = tulpa::math::portable_digamma(alpha);
        double psi_beta = tulpa::math::portable_digamma(beta_param);
        double dLL_dp = phi_num * (psi_y_alpha - psi_nmy_beta - psi_alpha + psi_beta);
        dLL_deta_num = dLL_dp * p * (1.0 - p);

        double psi_n_phi = tulpa::math::portable_digamma(n_i + phi_num);
        double psi_phi = tulpa::math::portable_digamma(phi_num);
        grad_phi_num_i = p * psi_y_alpha + (1.0 - p) * psi_nmy_beta - psi_n_phi
                         - p * psi_alpha - (1.0 - p) * psi_beta + psi_phi;
    }
}

// Scatter residuals to beta gradient slots
static inline void scatter_beta_gradients(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom,
    double* grad
) {
    for (int j = 0; j < data.legacy.p_num; j++) {
        grad[layout.legacy.beta_num_start + j] += dLL_deta_num * data.legacy.X_num_flat[i * data.legacy.p_num + j];
    }
    for (int j = 0; j < data.legacy.p_denom; j++) {
        grad[layout.legacy.beta_denom_start + j] += dLL_deta_denom * data.legacy.X_denom_flat[i * data.legacy.p_denom + j];
    }
}

// Scatter residuals to RE gradient slot
static inline void scatter_re_gradient(
    const ModelData& data, const ParamLayout& layout,
    int i, double dLL_deta_num, double dLL_deta_denom,
    double* grad
) {
    if (layout.has_re && data.re_group[i] > 0) {
        int g = data.re_group[i] - 1;
        grad[layout.re_start + g] += dLL_deta_num + dLL_deta_denom;
    }
}

// Per-observation phi likelihood gradient accumulation
// Handles NB phi_num, NB phi_denom, and Gamma phi_denom
static inline void accumulate_phi_likelihood_grad(
    const ModelData& data, const ParamLayout& layout,
    int i, double eta_num, double eta_denom,
    double phi_num, double phi_denom,
    double* grad
) {
    // phi_num gradient
    if (layout.legacy.has_phi_num) {
        if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN ||
            data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
            double mu_num = std::exp(eta_num);
            double y = data.legacy.y_num[i];
            double dLL_dphi = tulpa::math::portable_digamma(y + phi_num) - tulpa::math::portable_digamma(phi_num)
                             + std::log(phi_num / (mu_num + phi_num)) + 1.0
                             - (y + phi_num) / (mu_num + phi_num);
            grad[layout.legacy.log_phi_num_idx] += dLL_dphi * phi_num;
        } else if (data.legacy.model_type == ModelType::GAMMA_GAMMA) {
            // Gamma shape_num: dLL/d(shape) = log(shape/mu) + 1 + log(y) - digamma(shape) - y/mu
            double y = data.legacy.y_num_cont[i];
            if (y > 0.0) {
                double mu_num = std::exp(eta_num);
                double dLL_dphi = std::log(phi_num / mu_num) + 1.0 + std::log(y)
                                 - tulpa::math::portable_digamma(phi_num) - y / mu_num;
                grad[layout.legacy.log_phi_num_idx] += dLL_dphi * phi_num;
            }
        } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
            // Lognormal sigma_num: dLL/d(sigma) = -1/sigma + z^2/sigma where z = (log(y)-eta)/sigma
            double y = data.legacy.y_num_cont[i];
            if (y > 0.0) {
                double z = (std::log(y) - eta_num) / phi_num;
                double dLL_dphi = (-1.0 + z * z) / phi_num;
                grad[layout.legacy.log_phi_num_idx] += dLL_dphi * phi_num;  // chain rule: d/d(log_phi) = dphi * phi
            }
        } else if (data.legacy.model_type == ModelType::BETA_BINOMIAL) {
            // Beta-binomial precision: phi = alpha + beta, a = p*phi, b = (1-p)*phi
            // dLL/d(phi) = p*[digamma(y+a) - digamma(a)] + (1-p)*[digamma(n-y+b) - digamma(b)]
            //              - [digamma(n+phi) - digamma(phi)]
            double p = 1.0 / (1.0 + std::exp(-eta_num));
            double a = p * phi_num;
            double b = (1.0 - p) * phi_num;
            double y = data.legacy.y_num[i];
            double n = data.legacy.y_denom[i];
            double dLL_dphi = p * (tulpa::math::portable_digamma(y + a) - tulpa::math::portable_digamma(a))
                            + (1.0 - p) * (tulpa::math::portable_digamma(n - y + b) - tulpa::math::portable_digamma(b))
                            - (tulpa::math::portable_digamma(n + phi_num) - tulpa::math::portable_digamma(phi_num));
            grad[layout.legacy.log_phi_num_idx] += dLL_dphi * phi_num;
        }
    }

    // phi_denom gradient
    if (layout.legacy.has_phi_denom) {
        if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
            double mu_denom = std::exp(eta_denom);
            double y = data.legacy.y_denom[i];
            double dLL_dphi = tulpa::math::portable_digamma(y + phi_denom) - tulpa::math::portable_digamma(phi_denom)
                             + std::log(phi_denom / (mu_denom + phi_denom)) + 1.0
                             - (y + phi_denom) / (mu_denom + phi_denom);
            grad[layout.legacy.log_phi_denom_idx] += dLL_dphi * phi_denom;
        } else if (data.legacy.model_type == ModelType::POISSON_GAMMA ||
                   data.legacy.model_type == ModelType::NEGBIN_GAMMA ||
                   data.legacy.model_type == ModelType::GAMMA_GAMMA) {
            // Gamma shape_denom: same formula for PG, NG, and GG
            double y = data.legacy.y_denom_cont[i];
            if (y > 0.0) {
                double mu_denom = std::exp(eta_denom);
                double rate = phi_denom / mu_denom;
                double dLL_dphi = std::log(rate) + 1.0 + std::log(y)
                                 - tulpa::math::portable_digamma(phi_denom) - y / mu_denom;
                grad[layout.legacy.log_phi_denom_idx] += dLL_dphi * phi_denom;
            }
        } else if (data.legacy.model_type == ModelType::LOGNORMAL) {
            // Lognormal sigma_denom
            double y = data.legacy.y_denom_cont[i];
            if (y > 0.0) {
                double z = (std::log(y) - eta_denom) / phi_denom;
                double dLL_dphi = (-1.0 + z * z) / phi_denom;
                grad[layout.legacy.log_phi_denom_idx] += dLL_dphi * phi_denom;
            }
        }
    }
}

// Temporal tau prior gradient on log-scale
// d/d(log_tau) = shape - rate*tau
// (equivalently: (shape-1) - rate*tau + 1 with Jacobian expanded)
static inline void tau_temporal_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double* grad
) {
    grad[layout.log_tau_temporal_idx] = data.tau_temporal_shape
                                        - data.tau_temporal_rate * tau_temporal;
}

// Temporal sum-to-zero penalty gradient for RW1/RW2
// d/d(phi[t]) [-0.5 * lambda * (sum phi)^2] = -lambda * sum(phi)
static inline void temporal_sum_to_zero_grad(
    const double* phi, int T, int base_idx, double lambda,
    double* grad
) {
    double sp = 0.0;
    for (int t = 0; t < T; t++) sp += phi[t];
    for (int t = 0; t < T; t++) grad[base_idx + t] -= lambda * sp;
}

// =====================================================================
// Temporal GMRF prior gradient helper (RW1/RW2/AR1)
// Shared by all gradient functions that include temporal effects.
// Writes phi gradients, tau gradient, and rho gradient (for AR1).
// Expects grad_temporal_lik[0..T_len-1] to hold likelihood contributions.
// =====================================================================
static inline void temporal_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    double tau_temporal, double rho_ar1,
    const double* phi_temporal, int T_len,
    const double* grad_temporal_lik,
    double* grad
) {
    int T = data.n_times;
    int n_groups = data.n_temporal_groups;

    // Initialize temporal gradients with likelihood contribution
    for (int t = 0; t < T_len; t++) {
        grad[layout.temporal_start + t] = grad_temporal_lik[t];
    }

    if (data.temporal_type == TemporalType::RW1) {
        double total_qf = 0.0;
        int total_rank = 0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            double qf = 0.0;
            for (int t = 0; t < T; t++) {
                double g = 0.0;
                if (t > 0) {
                    g += tau_temporal * (phi_g[t - 1] - phi_g[t]);
                    qf += (phi_g[t] - phi_g[t - 1]) * (phi_g[t] - phi_g[t - 1]);
                }
                if (t < T - 1) g += tau_temporal * (phi_g[t + 1] - phi_g[t]);
                grad[base + t] += g;
            }
            if (data.temporal_cyclic) {
                double dc = phi_g[0] - phi_g[T - 1];
                qf += dc * dc;
                grad[base + 0] -= tau_temporal * dc;
                grad[base + T - 1] += tau_temporal * dc;
            }
            total_qf += qf;
            total_rank += data.temporal_cyclic ? T : T - 1;
            temporal_sum_to_zero_grad(phi_g, T, base, 0.001, grad);
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * total_rank - 0.5 * tau_temporal * total_qf;

    } else if (data.temporal_type == TemporalType::RW2) {
        double total_qf = 0.0;
        int total_rank = 0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            double qf = 0.0;
            for (int t = 0; t < T; t++) {
                double g = 0.0;
                if (t >= 2) g -= tau_temporal * (phi_g[t - 2] - 2.0 * phi_g[t - 1] + phi_g[t]);
                if (t >= 1 && t < T - 1) g += 2.0 * tau_temporal * (phi_g[t - 1] - 2.0 * phi_g[t] + phi_g[t + 1]);
                if (t < T - 2) g -= tau_temporal * (phi_g[t] - 2.0 * phi_g[t + 1] + phi_g[t + 2]);
                grad[base + t] += g;
            }
            for (int t = 2; t < T; t++) {
                double d2 = phi_g[t - 2] - 2.0 * phi_g[t - 1] + phi_g[t];
                qf += d2 * d2;
            }
            if (data.temporal_cyclic && T >= 3) {
                double d2_a = phi_g[T - 2] - 2.0 * phi_g[T - 1] + phi_g[0];
                double d2_b = phi_g[T - 1] - 2.0 * phi_g[0] + phi_g[1];
                qf += d2_a * d2_a + d2_b * d2_b;
                grad[base + T - 2] -= tau_temporal * d2_a;
                grad[base + T - 1] += 2.0 * tau_temporal * d2_a;
                grad[base + 0] -= tau_temporal * d2_a;
                grad[base + T - 1] -= tau_temporal * d2_b;
                grad[base + 0] += 2.0 * tau_temporal * d2_b;
                grad[base + 1] -= tau_temporal * d2_b;
            }
            total_qf += qf;
            total_rank += data.temporal_cyclic ? T : T - 2;
            temporal_sum_to_zero_grad(phi_g, T, base, 0.001, grad);
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * total_rank - 0.5 * tau_temporal * total_qf;

    } else if (data.temporal_type == TemporalType::AR1) {
        double omr2 = 1.0 - rho_ar1 * rho_ar1;
        double total_qf = 0.0, total_gr = 0.0;
        for (int gg = 0; gg < n_groups; gg++) {
            const double* phi_g = phi_temporal + gg * T;
            int base = layout.temporal_start + gg * T;
            grad[base] += -tau_temporal * omr2 * phi_g[0];
            if (T > 1) grad[base] += tau_temporal * rho_ar1 * (phi_g[1] - rho_ar1 * phi_g[0]);
            double qf = omr2 * phi_g[0] * phi_g[0];
            for (int t = 1; t < T; t++) {
                double r = phi_g[t] - rho_ar1 * phi_g[t - 1];
                qf += r * r;
                double g = -tau_temporal * r;
                if (t < T - 1) g += tau_temporal * rho_ar1 * (phi_g[t + 1] - rho_ar1 * phi_g[t]);
                grad[base + t] += g;
            }
            total_qf += qf;
            total_gr += tau_temporal * rho_ar1 * phi_g[0] * phi_g[0];
            for (int t = 1; t < T; t++) {
                total_gr += tau_temporal * (phi_g[t] - rho_ar1 * phi_g[t - 1]) * phi_g[t - 1];
            }
        }
        grad[layout.log_tau_temporal_idx] += 0.5 * T_len - 0.5 * tau_temporal * total_qf;
        if (layout.logit_rho_ar1_idx >= 0) {
            double gr = -n_groups * rho_ar1 / omr2 + total_gr;
            grad[layout.logit_rho_ar1_idx] += gr * rho_ar1 * (1.0 - rho_ar1);
        }
    }
}

// =====================================================================
// PC prior gradient on log(sigma2) for GP variances
// Returns d log_prior / d log_sigma2 INCLUDING Jacobian for exp transform.
// Formula: -0.5 * rate * sigma + 0.5  where rate = -log(alpha) / U
// =====================================================================
static inline double gp_pc_prior_grad_log_sigma2(
    double sigma2, double U, double alpha
) {
    double sigma = std::sqrt(sigma2 + 1e-10);
    double rate = -std::log(alpha + 1e-10) / (U + 1e-10);
    return -0.5 * rate * sigma + 0.5;
}

// =====================================================================
// Build GPData views from MultiscaleGPData for local and regional scales
// =====================================================================
static inline std::pair<GPData, GPData> make_msgp_gp_views(
    const MultiscaleGPData& msgp
) {
    GPData gp_local;
    gp_local.n_obs = msgp.n_obs;
    gp_local.nn = msgp.nn_local;
    gp_local.coords = msgp.coords;
    gp_local.nn_idx = msgp.nn_idx_local;
    gp_local.nn_dist = msgp.nn_dist_local;
    gp_local.nn_order = msgp.nn_order_local;
    gp_local.nn_order_inv = msgp.nn_order_inv_local;
    gp_local.cov_type = msgp.cov_type;

    GPData gp_regional;
    gp_regional.n_obs = msgp.n_obs;
    gp_regional.nn = msgp.nn_regional;
    gp_regional.coords = msgp.coords;
    gp_regional.nn_idx = msgp.nn_idx_regional;
    gp_regional.nn_dist = msgp.nn_dist_regional;
    gp_regional.nn_order = msgp.nn_order_regional;
    gp_regional.nn_order_inv = msgp.nn_order_inv_regional;
    gp_regional.cov_type = msgp.cov_type;

    return {gp_local, gp_regional};
}

// =====================================================================
// Spatial ICAR/BYM2 GMRF prior gradient helper
// Used by ms_temporal and spatiotemporal gradient functions.
// Writes phi_spatial gradients, theta_bym2 gradients (BYM2), tau gradient (ICAR).
// grad_spatial_lik and grad_theta_lik must hold accumulated likelihood contributions.
// =====================================================================
static inline void spatial_gmrf_prior_grad(
    const ModelData& data, const ParamLayout& layout,
    const double* spatial_phi,
    double tau_spatial,
    double sigma_s_bym2, double sigma_u_bym2,
    double rho_bym2,
    const double* theta_bym2,
    const double* grad_spatial_lik,
    const double* grad_theta_lik,
    double* grad
) {
    int S = data.n_spatial_units;
    if (layout.is_bym2) {
        // Soft sum-to-zero gradient: -0.01 * sum(phi) for each phi[s]
        double phi_sum = 0.0;
        for (int s = 0; s < S; s++) phi_sum += spatial_phi[s];
        for (int s = 0; s < S; s++) {
            double icar_grad = 0.0;
            for (int idx = data.adj_row_ptr[s]; idx < data.adj_row_ptr[s + 1]; idx++) {
                int j = data.adj_col_idx[idx];
                icar_grad += (spatial_phi[j] - spatial_phi[s]);
            }
            grad[layout.spatial_start + s] = grad_spatial_lik[s] * sigma_s_bym2 * data.bym2_scale_factor + icar_grad - 0.01 * phi_sum;
            grad[layout.theta_bym2_start + s] = grad_theta_lik[s] * sigma_u_bym2 - theta_bym2[s];
        }
        double grad_sigma_s_lik = 0.0, grad_sigma_u_lik = 0.0;
        for (int s = 0; s < S; s++) {
            grad_sigma_s_lik += grad_spatial_lik[s] * sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s];
            grad_sigma_u_lik += grad_theta_lik[s] * sigma_u_bym2 * theta_bym2[s];
        }
        grad[layout.log_sigma_bym2_idx] += grad_sigma_s_lik + grad_sigma_u_lik;
        grad[layout.logit_rho_bym2_idx] += 0.5 * ((1.0 - rho_bym2) * grad_sigma_s_lik
                                                    - rho_bym2 * grad_sigma_u_lik);
    } else {
        for (int s = 0; s < S; s++) {
            double icar_grad = 0.0;
            for (int idx = data.adj_row_ptr[s]; idx < data.adj_row_ptr[s + 1]; idx++) {
                int j = data.adj_col_idx[idx];
                icar_grad += tau_spatial * (spatial_phi[j] - spatial_phi[s]);
            }
            grad[layout.spatial_start + s] = grad_spatial_lik[s] + icar_grad;
        }
        double icar_qf = icar_quadratic_form_ptr(spatial_phi, S, data);
        grad[layout.log_tau_spatial_idx] += 0.5 * (S - 1) - 0.5 * tau_spatial * icar_qf;
    }
}

// =====================================================================
// Shared gradient building blocks (preamble, vectorized eta, dispatch,
// RE scatter, epilogue). Used by all handcoded gradient functions below.
// =====================================================================
#include "hmc_gradient_shared.h"

// =====================================================================
// GP gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from gp_nngp_gradients for NNGP prior
// =====================================================================

void compute_gradient_gp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- GP-specific parameters ---
    int N_gp = data.gp_data.n_obs;
    double sigma2_gp = std::exp(params[layout.log_sigma2_gp_idx]);
    double phi_gp = std::exp(params[layout.log_phi_gp_idx]);

    if (phi_gp < data.gp_phi_prior_lower || phi_gp > data.gp_phi_prior_upper) {
        return;
    }

    const bool use_nc = (data.gp_parameterization == 1);
    static thread_local tulpa_gp::NNGPNCWorkspace nc_ws;

    // Get spatial effects: either w directly (centered) or reconstruct from z (NC)
    std::vector<double> gp_w(N_gp);
    if (use_nc) {
        const double* z_params = &params[layout.gp_w_start];
        tulpa_gp::nngp_nc_forward(z_params, sigma2_gp, phi_gp, data.gp_data, nc_ws);
        std::memcpy(gp_w.data(), nc_ws.w.data(), N_gp * sizeof(double));
    } else {
        for (int i = 0; i < N_gp; i++) {
            gp_w[i] = params[layout.gp_w_start + i];
        }
    }

    // --- Shared base priors + GP-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // PC prior on GP variance
    grad[layout.log_sigma2_gp_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
    // Uniform prior on phi - just Jacobian for log transform
    grad[layout.log_phi_gp_idx] = 1.0;

    if (!use_nc) {
        tulpa_gp::NNGPGradients nngp_grads;
        tulpa_gp::gp_nngp_gradients(gp_w, sigma2_gp, phi_gp, data.gp_data, nngp_grads);
        for (int i = 0; i < N_gp; i++) {
            grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
        }
        grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
        grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // GP-specific eta contribution
    for (int i = 0; i < pre.N; i++) {
        int loc_i = data.gp_data.obs_to_loc[i];
        double gp_effect = gp_w[loc_i];
        vec_grad_ws.eta_num[i] += gp_effect;
        if (!pre.is_binomial && data.gp_data.shared) vec_grad_ws.eta_denom[i] += gp_effect;
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // --- GP-specific residual scatter ---
    if (use_nc) {
        std::vector<double> dL_dw(N_gp, 0.0);
        for (int i = 0; i < pre.N; i++) {
            int loc_i = data.gp_data.obs_to_loc[i];
            double dLL = data.gp_data.shared
                ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
                : vec_grad_ws.resid_num[i];
            dL_dw[loc_i] += dLL;
        }

        std::vector<double> grad_z(N_gp, 0.0);
        double grad_log_sigma2_lik = 0.0, grad_log_phi_lik = 0.0, grad_log_phi_jac = 0.0;
        const double* z_params = &params[layout.gp_w_start];
        tulpa_gp::nngp_nc_backward(
            z_params, sigma2_gp, phi_gp, data.gp_data, nc_ws,
            dL_dw.data(), grad_z.data(),
            grad_log_sigma2_lik, grad_log_phi_lik, grad_log_phi_jac);

        for (int i = 0; i < N_gp; i++) {
            grad[layout.gp_w_start + i] += grad_z[i] - z_params[i];
        }
        grad[layout.log_sigma2_gp_idx] += grad_log_sigma2_lik;
        grad[layout.log_phi_gp_idx] += grad_log_phi_lik;
    } else {
        for (int i = 0; i < pre.N; i++) {
            int loc_i = data.gp_data.obs_to_loc[i];
            double dLL_dspatial = data.gp_data.shared
                ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
                : vec_grad_ws.resid_num[i];
            grad[layout.gp_w_start + loc_i] += dLL_dspatial;
        }
    }

    // --- Shared epilogue ---
    if (use_nc && pre.fuse_lp) {
        // NC: use full compute_log_post to ensure perfect consistency
        re_gradient_nc_transform(data, layout, params.data(), grad.data(), pre.cp.sigma_re);
        *log_post_out = compute_log_post(params, data, layout);
    } else {
        gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
    }
}

// =====================================================================
// Collapsed GP gradient (hand-coded)
// GP effects marginalized via inner Laplace — only hyperparams in HMC
// =====================================================================

// collapsed_gp_ws declared earlier (shared with compute_log_post)

void compute_gradient_gp_collapsed(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
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

    // GP hyperparameters
    double sigma2_gp = std::exp(params[layout.log_sigma2_gp_idx]);
    double phi_gp = std::exp(params[layout.log_phi_gp_idx]);

    if (phi_gp < data.gp_phi_prior_lower || phi_gp > data.gp_phi_prior_upper) {
        if (log_post_out) *log_post_out = -INFINITY;
        return;
    }

    int N_gp = data.gp_data.n_obs;
    int N = data.N;
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    // ---- Inner Laplace: find w* ----
    double collapsed_lp = collapsed_gp_find_mode(
        beta_num, beta_denom, sigma2_gp, phi_gp,
        phi_num, phi_denom, data, collapsed_gp_ws);

    // ---- Prior gradients (outer params only) ----
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // GP hyperparameter priors
    grad[layout.log_sigma2_gp_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
    grad[layout.log_phi_gp_idx] = 1.0;  // Uniform prior Jacobian

    // ---- Data likelihood gradient at w* ----
    // Compute residuals at the mode w*
    std::vector<double> resid_num(N), resid_denom(N);
    collapsed_gp_compute_residuals(
        collapsed_gp_ws.w_star.data(), beta_num, beta_denom,
        phi_num, phi_denom, data,
        resid_num.data(), resid_denom.data());

    // Scatter to beta gradients + phi likelihood gradient
    for (int i = 0; i < N; i++) {
        for (int p = 0; p < data.legacy.p_num; p++) {
            grad[layout.legacy.beta_num_start + p] += resid_num[i] * data.legacy.X_num_flat[i * data.legacy.p_num + p];
        }
        if (!is_binomial) {
            for (int p = 0; p < data.legacy.p_denom; p++) {
                grad[layout.legacy.beta_denom_start + p] += resid_denom[i] * data.legacy.X_denom_flat[i * data.legacy.p_denom + p];
            }
        }
        // Scatter to RE gradients
        if (layout.has_re && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] += resid_num[i] + resid_denom[i];
        }

        // Phi (dispersion) likelihood gradient
        if (layout.legacy.has_phi_num || layout.legacy.has_phi_denom) {
            int loc_i = data.gp_data.obs_to_loc[i];
            double eta_num_i = 0.0, eta_denom_i = 0.0;
            for (int p = 0; p < data.legacy.p_num; p++)
                eta_num_i += data.legacy.X_num_flat[i * data.legacy.p_num + p] * beta_num[p];
            if (!is_binomial) {
                for (int p = 0; p < data.legacy.p_denom; p++)
                    eta_denom_i += data.legacy.X_denom_flat[i * data.legacy.p_denom + p] * beta_denom[p];
            }
            eta_num_i += collapsed_gp_ws.w_star[loc_i];
            if (!is_binomial && data.gp_data.shared) eta_denom_i += collapsed_gp_ws.w_star[loc_i];
            if (layout.has_re && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                double re_val = (data.re_parameterization == 1) ?
                    sigma_re * re[data.re_group[i] - 1] : re[data.re_group[i] - 1];
                eta_num_i += re_val;
                if (!is_binomial) eta_denom_i += re_val;
            }
            accumulate_phi_likelihood_grad(data, layout, i, eta_num_i, eta_denom_i,
                                            phi_num, phi_denom, grad.data());
        }
    }

    // ---- GP hyperparameter gradients from NNGP prior at w* ----
    // d/d(log sigma2) and d/d(log phi) of log p_NNGP(w*|sigma2,phi)
    // Use the existing NNGP gradient function
    tulpa_gp::NNGPGradients nngp_grads;
    std::vector<double> w_star_vec(collapsed_gp_ws.w_star.begin(),
                                    collapsed_gp_ws.w_star.end());
    tulpa_gp::gp_nngp_gradients(w_star_vec, sigma2_gp, phi_gp,
                                  data.gp_data, nngp_grads);
    grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
    grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;

    // ---- Laplace correction gradient via numerical differentiation ----
    // The Laplace log-det depends on w* which depends on ALL params implicitly.
    // We compute d/dθ [-0.5 log det(W+Q)] numerically for each outer param,
    // using warm-started Newton solves (1-2 iters each from current w*).
    // For non-GP params (beta, phi, RE), we reuse the NNGP structure from
    // collapsed_gp_ws (Q doesn't change), skipping NNGP rebuild.
    {
        const double eps = 1e-5;
        std::vector<double> params_pert = params;
        const int gp_idx1 = layout.log_sigma2_gp_idx;
        const int gp_idx2 = layout.log_phi_gp_idx;

        for (int j = 0; j < n_params; j++) {
            double orig = params_pert[j];
            bool is_gp_hyperparam = (j == gp_idx1 || j == gp_idx2);

            // Forward perturbation
            params_pert[j] = orig + eps;
            double sigma2_p = std::exp(params_pert[gp_idx1]);
            double phi_p = std::exp(params_pert[gp_idx2]);
            const double* beta_num_p = &params_pert[layout.legacy.beta_num_start];
            const double* beta_denom_p = is_binomial ? beta_num_p : &params_pert[layout.legacy.beta_denom_start];
            double phi_num_p = layout.legacy.has_phi_num ? std::exp(params_pert[layout.legacy.log_phi_num_idx]) : phi_num;
            double phi_denom_p = layout.legacy.has_phi_denom ? std::exp(params_pert[layout.legacy.log_phi_denom_idx]) : phi_denom;
            double ld_plus = laplace_log_det_full(
                beta_num_p, beta_denom_p, sigma2_p, phi_p,
                phi_num_p, phi_denom_p, data, collapsed_gp_ws.w_star,
                is_gp_hyperparam ? nullptr : &collapsed_gp_ws,
                is_gp_hyperparam);

            // Backward perturbation
            params_pert[j] = orig - eps;
            sigma2_p = std::exp(params_pert[gp_idx1]);
            phi_p = std::exp(params_pert[gp_idx2]);
            beta_num_p = &params_pert[layout.legacy.beta_num_start];
            beta_denom_p = is_binomial ? beta_num_p : &params_pert[layout.legacy.beta_denom_start];
            phi_num_p = layout.legacy.has_phi_num ? std::exp(params_pert[layout.legacy.log_phi_num_idx]) : phi_num;
            phi_denom_p = layout.legacy.has_phi_denom ? std::exp(params_pert[layout.legacy.log_phi_denom_idx]) : phi_denom;
            double ld_minus = laplace_log_det_full(
                beta_num_p, beta_denom_p, sigma2_p, phi_p,
                phi_num_p, phi_denom_p, data, collapsed_gp_ws.w_star,
                is_gp_hyperparam ? nullptr : &collapsed_gp_ws,
                is_gp_hyperparam);

            grad[j] += (ld_plus - ld_minus) / (2.0 * eps);
            params_pert[j] = orig;
        }
    }

    // ---- NC transform for RE ----
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // ---- Log-posterior ----
    if (log_post_out) {
        // Compute full log-posterior including priors on outer params
        // collapsed_lp already has data_ll + nngp_prior
        double lp = collapsed_lp;

        // Laplace correction: -0.5 * log det(W + Q) via sparse Cholesky
        lp += collapsed_gp_ws.laplace_log_det;

        // Beta priors
        for (int p = 0; p < data.legacy.p_num; p++)
            lp += -0.5 * beta_num[p] * beta_num[p] / (data.sigma_beta * data.sigma_beta);
        for (int p = 0; p < data.legacy.p_denom; p++)
            lp += -0.5 * beta_denom[p] * beta_denom[p] / (data.sigma_beta * data.sigma_beta);

        // RE priors
        if (layout.has_re) {
            int n_re = layout.re_end - layout.re_start;
            double sigma_re2 = sigma_re * sigma_re;
            for (int g = 0; g < n_re; g++) {
                if (data.re_parameterization == 1) {
                    // NC: z ~ N(0,1)
                    lp += -0.5 * re[g] * re[g];
                } else {
                    lp += -0.5 * re[g] * re[g] / sigma_re2;
                }
            }
            // sigma_re half-Cauchy prior (log-scale)
            double ratio = sigma_re / data.sigma_re_scale;
            lp += -std::log(1.0 + ratio * ratio)
                  + params[layout.log_sigma_re_idx];
        }

        // GP hyperparameter priors
        double sigma_gp = std::sqrt(sigma2_gp);
        double rate = -std::log(data.gp_sigma2_prior_alpha) / data.gp_sigma2_prior_U;
        lp += std::log(rate) - rate * sigma_gp - std::log(2.0 * sigma_gp)
              + params[layout.log_sigma2_gp_idx];
        // phi uniform + Jacobian
        lp += params[layout.log_phi_gp_idx]
              - std::log(data.gp_phi_prior_upper - data.gp_phi_prior_lower);

        // Phi (dispersion) priors
        if (layout.legacy.has_phi_num) {
            double log_phi = params[layout.legacy.log_phi_num_idx];
            lp += log_phi;  // Jacobian for log transform (exponential prior)
        }
        if (layout.legacy.has_phi_denom) {
            double log_phi = params[layout.legacy.log_phi_denom_idx];
            lp += log_phi;
        }

        *log_post_out = lp;
    }
}

// =====================================================================
// Collapsed ICAR/BYM2 gradient
// ICAR: H-mode (analytical envelope + analytical Laplace via implicit fn thm)
// BYM2: numerical fallback (H-mode planned)
// =====================================================================

void compute_gradient_icar_collapsed(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
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

    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);
    int N = data.N;
    int S = data.n_spatial_units;

    // Pre-compute actual RE values (NC → actual)
    std::vector<double> re_vals;
    if (layout.has_re) {
        int n_re = layout.re_end - layout.re_start;
        re_vals.resize(n_re);
        for (int g = 0; g < n_re; g++) {
            re_vals[g] = (data.re_parameterization == 1) ? sigma_re * re[g] : re[g];
        }
    }

    // Spatial hyperparameters
    bool is_bym2 = layout.is_bym2_collapsed;
    double tau = 0.0, sigma_total = 0.0, rho = 0.0;
    double a = 0.0, c_bym2 = 0.0;

    if (is_bym2) {
        sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
        double logit_rho = params[layout.logit_rho_bym2_idx];
        rho = 1.0 / (1.0 + std::exp(-logit_rho));
        a = sigma_total * std::sqrt(rho) * data.bym2_scale_factor;
        c_bym2 = sigma_total * std::sqrt(1.0 - rho);
    } else {
        tau = std::exp(params[layout.log_tau_spatial_idx]);
    }

    // ---- Inner Laplace: find phi* (and theta* for BYM2) ----
    double collapsed_lp;
    if (is_bym2) {
        collapsed_lp = collapsed_bym2_find_mode(
            beta_num, beta_denom, sigma_total, rho, data.bym2_scale_factor,
            phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            data, collapsed_icar_ws);
    } else {
        collapsed_lp = collapsed_icar_find_mode(
            beta_num, beta_denom, tau, phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            data, collapsed_icar_ws);
    }

    // ---- Outer priors (simple analytical, don't depend on phi*) ----
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // ---- ICAR H-mode: analytical envelope + analytical Laplace ----
    // BYM2: numerical fallback (TODO: implement BYM2 H-mode)
    if (!is_bym2) {
        // === Part A: Envelope theorem gradient ===
        // At mode, ∂f/∂φ = 0, so d/dθ[f(φ*,θ)] = ∂f/∂θ|_{φ*}

        // A1: Data LL gradient via residual scattering
        std::vector<double> resid_num(N), resid_denom(N);
        collapsed_icar_compute_residuals(
            collapsed_icar_ws, beta_num, beta_denom,
            phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            0.0, 0.0,  // not BYM2
            data, resid_num.data(), resid_denom.data());

        // Scatter residuals to beta_num: X_num' * resid_num
        for (int k = 0; k < data.legacy.p_num; k++) {
            double sum = 0.0;
            for (int i = 0; i < N; i++)
                sum += resid_num[i] * data.legacy.X_num_flat[i * data.legacy.p_num + k];
            grad[layout.legacy.beta_num_start + k] += sum;
        }
        // Scatter to beta_denom: X_denom' * resid_denom
        if (!is_binomial) {
            for (int k = 0; k < data.legacy.p_denom; k++) {
                double sum = 0.0;
                for (int i = 0; i < N; i++)
                    sum += resid_denom[i] * data.legacy.X_denom_flat[i * data.legacy.p_denom + k];
                grad[layout.legacy.beta_denom_start + k] += sum;
            }
        }
        // Scatter to RE (w.r.t. centered values)
        if (layout.has_re) {
            int n_re = layout.re_end - layout.re_start;
            for (int i = 0; i < N; i++) {
                if (data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                    int g = data.re_group[i] - 1;
                    if (g < n_re) {
                        grad[layout.re_start + g] += resid_num[i] + resid_denom[i];
                    }
                }
            }
        }

        // A2: Dispersion parameter gradients from data LL at mode
        // (envelope theorem: ∂LL/∂log_phi evaluated at φ*)
        if (layout.legacy.has_phi_num || layout.legacy.has_phi_denom) {
            for (int i = 0; i < N; i++) {
                int s = data.spatial_group[i] - 1;
                double eta_num_i = 0.0, eta_denom_i = 0.0;
                for (int p = 0; p < data.legacy.p_num; p++)
                    eta_num_i += data.legacy.X_num_flat[i * data.legacy.p_num + p] * beta_num[p];
                if (!is_binomial) {
                    for (int p = 0; p < data.legacy.p_denom; p++)
                        eta_denom_i += data.legacy.X_denom_flat[i * data.legacy.p_denom + p] * beta_denom[p];
                }
                eta_num_i += collapsed_icar_ws.phi_star[s];
                if (!is_binomial) eta_denom_i += collapsed_icar_ws.phi_star[s];
                if (re_vals.data() && data.re_group.size() > (size_t)i && data.re_group[i] > 0)  {
                    eta_num_i += re_vals[data.re_group[i] - 1];
                    if (!is_binomial) eta_denom_i += re_vals[data.re_group[i] - 1];
                }

                double mu_num = std::exp(std::min(eta_num_i, 20.0));

                // Per-family dispersion gradients
                switch (data.legacy.model_type) {
                    case ModelType::POISSON_GAMMA: {
                        if (layout.legacy.has_phi_denom) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = data.legacy.y_denom_cont[i];
                            double alpha = phi_denom;
                            // d/d(log_alpha)[LL_denom] = alpha * dLL/dalpha
                            // dLL/dalpha = log(alpha) + 1 - digamma(alpha) + log(y/mu) - y/mu
                            grad[layout.legacy.log_phi_denom_idx] += alpha * (
                                std::log(alpha) + 1.0 - R::digamma(alpha)
                                + std::log(std::max(y_d, 1e-10)) - std::log(mu_d)
                                - y_d / mu_d);
                        }
                        break;
                    }
                    case ModelType::NEGBIN_NEGBIN: {
                        if (layout.legacy.has_phi_num) {
                            double r = phi_num;
                            double y = data.legacy.y_num[i];
                            grad[layout.legacy.log_phi_num_idx] += r * (
                                R::digamma(y + r) - R::digamma(r)
                                + std::log(r) + 1.0
                                - std::log(mu_num + r) - (y + r) / (mu_num + r));
                        }
                        if (layout.legacy.has_phi_denom && !is_binomial) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = (double)data.legacy.y_denom[i];
                            double r_d = phi_denom;
                            grad[layout.legacy.log_phi_denom_idx] += r_d * (
                                R::digamma(y_d + r_d) - R::digamma(r_d)
                                + std::log(r_d) + 1.0
                                - std::log(mu_d + r_d) - (y_d + r_d) / (mu_d + r_d));
                        }
                        break;
                    }
                    case ModelType::NEGBIN_GAMMA: {
                        if (layout.legacy.has_phi_num) {
                            double r = phi_num;
                            double y = data.legacy.y_num[i];
                            grad[layout.legacy.log_phi_num_idx] += r * (
                                R::digamma(y + r) - R::digamma(r)
                                + std::log(r) + 1.0
                                - std::log(mu_num + r) - (y + r) / (mu_num + r));
                        }
                        if (layout.legacy.has_phi_denom) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = data.legacy.y_denom_cont[i];
                            double alpha = phi_denom;
                            grad[layout.legacy.log_phi_denom_idx] += alpha * (
                                std::log(alpha) + 1.0 - R::digamma(alpha)
                                + std::log(std::max(y_d, 1e-10)) - std::log(mu_d)
                                - y_d / mu_d);
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
        }

        // A3: ICAR prior gradient w.r.t. log_tau (envelope: holding φ* fixed)
        // d/d(log_tau)[-0.5*tau*phi*'Q*phi* + 0.5*(S-1)*log(tau)]
        //   = -0.5*tau*phi*'Q*phi* + 0.5*(S-1)
        {
            std::vector<double> Qphi(S);
            icar_precision_matvec(collapsed_icar_ws.phi_star.data(), Qphi.data(), S,
                                  data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
            double phiQphi = 0.0;
            for (int s = 0; s < S; s++) phiQphi += collapsed_icar_ws.phi_star[s] * Qphi[s];
            grad[layout.log_tau_spatial_idx] += -0.5 * tau * phiQphi + 0.5 * (S - 1);
        }

        // === Part B: H-mode Laplace gradient ===
        auto laplace_result = compute_laplace_gradient_icar_H(
            collapsed_icar_ws, beta_num, beta_denom,
            tau, phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            data, layout, n_params);

        if (laplace_result.success) {
            for (int j = 0; j < n_params; j++) {
                grad[j] += laplace_result.laplace_grad[j];
            }
        } else {
            // Fallback: numerical Laplace gradient for all params
            // (recomputes mode_lp + Laplace via central differences)
            auto collapsed_log_post = [&](const std::vector<double>& p) -> double {
                auto cp_l = extract_common_params(p, layout);
                double tau_l = std::exp(p[layout.log_tau_spatial_idx]);
                std::vector<double> re_vals_l;
                if (layout.has_re) {
                    int n_re = layout.re_end - layout.re_start;
                    re_vals_l.resize(n_re);
                    for (int g = 0; g < n_re; g++) {
                        re_vals_l[g] = (data.re_parameterization == 1)
                            ? cp_l.sigma_re * cp_l.re[g] : cp_l.re[g];
                    }
                }
                CollapsedICARWorkspace temp_ws;
                temp_ws.init(S, false);
                temp_ws.phi_star = collapsed_icar_ws.phi_star;
                temp_ws.mode_found = true;
                double mode_lp = collapsed_icar_find_mode(
                    cp_l.beta_num, cp_l.beta_denom, tau_l,
                    cp_l.phi_num, cp_l.phi_denom,
                    re_vals_l.empty() ? nullptr : re_vals_l.data(),
                    data, temp_ws);
                double lp = mode_lp + temp_ws.laplace_log_det;
                lp += (data.tau_spatial_shape - 1.0) * std::log(tau_l) - data.tau_spatial_rate * tau_l
                      + p[layout.log_tau_spatial_idx];
                return lp;
            };
            const double eps = 1e-5;
            std::vector<double> params_pert = params;
            for (int j = 0; j < n_params; j++) {
                double orig = params_pert[j];
                params_pert[j] = orig + eps;
                double lp_plus = collapsed_log_post(params_pert);
                params_pert[j] = orig - eps;
                double lp_minus = collapsed_log_post(params_pert);
                grad[j] += (lp_plus - lp_minus) / (2.0 * eps);
                params_pert[j] = orig;
            }
        }

        // === Part C: Spatial hyperparameter prior gradient (analytical) ===
        // Gamma(shape, rate) prior on tau, on log scale:
        // d/d(log_tau)[(shape-1)*log(tau) - rate*tau + log_tau]
        //   = (shape-1) - rate*tau + 1
        grad[layout.log_tau_spatial_idx] += (data.tau_spatial_shape - 1.0)
                                           - data.tau_spatial_rate * tau + 1.0;

    } else {
        // ---- BYM2: H-mode analytical gradient ----
        // Same structure as ICAR: envelope + analytical Laplace + outer priors

        // === Part A: Envelope theorem gradient ===
        // A1: Data LL gradient via residual scattering
        std::vector<double> resid_num(N), resid_denom(N);
        collapsed_icar_compute_residuals(
            collapsed_icar_ws, beta_num, beta_denom,
            phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            a, c_bym2,  // BYM2 scaling
            data, resid_num.data(), resid_denom.data());

        // Scatter residuals to beta_num
        for (int k = 0; k < data.legacy.p_num; k++) {
            double sum = 0.0;
            for (int i = 0; i < N; i++)
                sum += resid_num[i] * data.legacy.X_num_flat[i * data.legacy.p_num + k];
            grad[layout.legacy.beta_num_start + k] += sum;
        }
        // Scatter to beta_denom
        if (!is_binomial) {
            for (int k = 0; k < data.legacy.p_denom; k++) {
                double sum = 0.0;
                for (int i = 0; i < N; i++)
                    sum += resid_denom[i] * data.legacy.X_denom_flat[i * data.legacy.p_denom + k];
                grad[layout.legacy.beta_denom_start + k] += sum;
            }
        }
        // Scatter to RE
        if (layout.has_re) {
            int n_re = layout.re_end - layout.re_start;
            for (int i = 0; i < N; i++) {
                if (data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                    int g = data.re_group[i] - 1;
                    if (g < n_re) {
                        grad[layout.re_start + g] += resid_num[i] + resid_denom[i];
                    }
                }
            }
        }

        // A2: Dispersion parameter gradients from data LL at mode (envelope)
        if (layout.legacy.has_phi_num || layout.legacy.has_phi_denom) {
            for (int i = 0; i < N; i++) {
                int s = data.spatial_group[i] - 1;
                double b_s = a * collapsed_icar_ws.phi_star[s]
                           + c_bym2 * collapsed_icar_ws.theta_star[s];
                double eta_num_i = 0.0, eta_denom_i = 0.0;
                for (int p = 0; p < data.legacy.p_num; p++)
                    eta_num_i += data.legacy.X_num_flat[i * data.legacy.p_num + p] * beta_num[p];
                if (!is_binomial) {
                    for (int p = 0; p < data.legacy.p_denom; p++)
                        eta_denom_i += data.legacy.X_denom_flat[i * data.legacy.p_denom + p] * beta_denom[p];
                }
                eta_num_i += b_s;
                if (!is_binomial) eta_denom_i += b_s;
                if (re_vals.data() && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                    eta_num_i += re_vals[data.re_group[i] - 1];
                    if (!is_binomial) eta_denom_i += re_vals[data.re_group[i] - 1];
                }
                double mu_num = std::exp(std::min(eta_num_i, 20.0));

                switch (data.legacy.model_type) {
                    case ModelType::POISSON_GAMMA: {
                        if (layout.legacy.has_phi_denom) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = data.legacy.y_denom_cont[i];
                            double alpha = phi_denom;
                            grad[layout.legacy.log_phi_denom_idx] += alpha * (
                                std::log(alpha) + 1.0 - R::digamma(alpha)
                                + std::log(std::max(y_d, 1e-10)) - std::log(mu_d)
                                - y_d / mu_d);
                        }
                        break;
                    }
                    case ModelType::NEGBIN_NEGBIN: {
                        if (layout.legacy.has_phi_num) {
                            double r = phi_num;
                            double y = data.legacy.y_num[i];
                            grad[layout.legacy.log_phi_num_idx] += r * (
                                R::digamma(y + r) - R::digamma(r)
                                + std::log(r) + 1.0
                                - std::log(mu_num + r) - (y + r) / (mu_num + r));
                        }
                        if (layout.legacy.has_phi_denom && !is_binomial) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = (double)data.legacy.y_denom[i];
                            double r_d = phi_denom;
                            grad[layout.legacy.log_phi_denom_idx] += r_d * (
                                R::digamma(y_d + r_d) - R::digamma(r_d)
                                + std::log(r_d) + 1.0
                                - std::log(mu_d + r_d) - (y_d + r_d) / (mu_d + r_d));
                        }
                        break;
                    }
                    case ModelType::NEGBIN_GAMMA: {
                        if (layout.legacy.has_phi_num) {
                            double r = phi_num;
                            double y = data.legacy.y_num[i];
                            grad[layout.legacy.log_phi_num_idx] += r * (
                                R::digamma(y + r) - R::digamma(r)
                                + std::log(r) + 1.0
                                - std::log(mu_num + r) - (y + r) / (mu_num + r));
                        }
                        if (layout.legacy.has_phi_denom) {
                            double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                            double y_d = data.legacy.y_denom_cont[i];
                            double alpha = phi_denom;
                            grad[layout.legacy.log_phi_denom_idx] += alpha * (
                                std::log(alpha) + 1.0 - R::digamma(alpha)
                                + std::log(std::max(y_d, 1e-10)) - std::log(mu_d)
                                - y_d / mu_d);
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
        }

        // A3: BYM2 prior envelope gradients for log_sigma and logit_rho
        // (envelope: holding phi*, theta* fixed, differentiate data LL through b_s)
        {
            double grad_sigma_env = 0.0;
            double grad_rho_env = 0.0;
            double da_drho = a * (1.0 - rho) / 2.0;
            double dc_drho = -c_bym2 * rho / 2.0;
            for (int s = 0; s < S; s++) {
                // Per-site residual sum (from residuals already computed)
                double r_sum = 0.0;
                for (int i = 0; i < N; i++) {
                    if (data.spatial_group[i] - 1 == s)
                        r_sum += resid_num[i] + resid_denom[i];
                }
                double b_s = a * collapsed_icar_ws.phi_star[s]
                           + c_bym2 * collapsed_icar_ws.theta_star[s];
                grad_sigma_env += r_sum * b_s;  // d/d(log_sigma) = b_s
                double d_rho_s = da_drho * collapsed_icar_ws.phi_star[s]
                               + dc_drho * collapsed_icar_ws.theta_star[s];
                grad_rho_env += r_sum * d_rho_s;
            }
            grad[layout.log_sigma_bym2_idx] += grad_sigma_env;
            grad[layout.logit_rho_bym2_idx] += grad_rho_env;
        }

        // === Part B: H-mode Laplace gradient ===
        auto laplace_result = compute_laplace_gradient_bym2_H(
            collapsed_icar_ws, beta_num, beta_denom,
            a, c_bym2, rho,
            phi_num, phi_denom,
            re_vals.empty() ? nullptr : re_vals.data(),
            data, layout, n_params);

        if (laplace_result.success) {
            for (int j = 0; j < n_params; j++) {
                grad[j] += laplace_result.laplace_grad[j];
            }
        } else {
            // Numerical fallback for Laplace gradient only
            auto laplace_only = [&](const std::vector<double>& p) -> double {
                auto cp_l = extract_common_params(p, layout);
                double sigma_l = std::exp(p[layout.log_sigma_bym2_idx]);
                double logit_rho_l = p[layout.logit_rho_bym2_idx];
                double rho_l = 1.0 / (1.0 + std::exp(-logit_rho_l));
                double a_l = sigma_l * std::sqrt(rho_l) * data.bym2_scale_factor;
                double c_l = sigma_l * std::sqrt(1.0 - rho_l);
                std::vector<double> re_vals_l;
                if (layout.has_re) {
                    int n_re = layout.re_end - layout.re_start;
                    re_vals_l.resize(n_re);
                    for (int g = 0; g < n_re; g++) {
                        re_vals_l[g] = (data.re_parameterization == 1)
                            ? cp_l.sigma_re * cp_l.re[g] : cp_l.re[g];
                    }
                }
                CollapsedICARWorkspace temp_ws;
                temp_ws.init(S, true);
                temp_ws.phi_star = collapsed_icar_ws.phi_star;
                temp_ws.theta_star = collapsed_icar_ws.theta_star;
                temp_ws.mode_found = true;
                collapsed_bym2_find_mode(
                    cp_l.beta_num, cp_l.beta_denom, sigma_l, rho_l, data.bym2_scale_factor,
                    cp_l.phi_num, cp_l.phi_denom,
                    re_vals_l.empty() ? nullptr : re_vals_l.data(),
                    data, temp_ws);
                return temp_ws.laplace_log_det;
            };
            const double eps = 1e-5;
            std::vector<double> params_pert = params;
            for (int j = 0; j < n_params; j++) {
                double orig = params_pert[j];
                params_pert[j] = orig + eps;
                double ld_plus = laplace_only(params_pert);
                params_pert[j] = orig - eps;
                double ld_minus = laplace_only(params_pert);
                grad[j] += (ld_plus - ld_minus) / (2.0 * eps);
                params_pert[j] = orig;
            }
        }

        // === Part C: BYM2 hyperparameter prior gradients ===
        // Sigma prior: half-Cauchy via d/d(log_sigma)[-log(1 + (sigma/s)^2) + log_sigma]
        {
            double ratio_sc = sigma_total / data.sigma_re_scale;
            double r2 = ratio_sc * ratio_sc;
            grad[layout.log_sigma_bym2_idx] += -2.0 * r2 / (1.0 + r2) + 1.0;
        }
        // Rho prior: Uniform(0,1) Jacobian: log(rho) + log(1-rho)
        // d/d(logit_rho) = d(log(rho))/d(logit_rho) + d(log(1-rho))/d(logit_rho)
        //                = (1-rho) + (-rho) = 1 - 2*rho
        grad[layout.logit_rho_bym2_idx] += 1.0 - 2.0 * rho;
    }

    // ---- NC transform for RE ----
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // ---- Log-posterior ----
    if (log_post_out) {
        double lp = collapsed_lp + collapsed_icar_ws.laplace_log_det;

        // Beta priors
        for (int p = 0; p < data.legacy.p_num; p++)
            lp += -0.5 * beta_num[p] * beta_num[p] / (data.sigma_beta * data.sigma_beta);
        for (int p = 0; p < data.legacy.p_denom; p++)
            lp += -0.5 * beta_denom[p] * beta_denom[p] / (data.sigma_beta * data.sigma_beta);

        // RE priors
        if (layout.has_re) {
            int n_re = layout.re_end - layout.re_start;
            double sigma_re2 = sigma_re * sigma_re;
            for (int g = 0; g < n_re; g++) {
                if (data.re_parameterization == 1) {
                    lp += -0.5 * re[g] * re[g];
                } else {
                    lp += -0.5 * re[g] * re[g] / sigma_re2;
                }
            }
            double ratio_hc = sigma_re / data.sigma_re_scale;
            lp += -std::log(1.0 + ratio_hc * ratio_hc) + params[layout.log_sigma_re_idx];
        }

        // Spatial hyperparameter priors
        if (is_bym2) {
            double ratio_sc = sigma_total / data.sigma_re_scale;
            lp += -std::log(1.0 + ratio_sc * ratio_sc) + params[layout.log_sigma_bym2_idx];
            lp += std::log(rho) + std::log(1.0 - rho);
        } else {
            lp += (data.tau_spatial_shape - 1.0) * std::log(tau) - data.tau_spatial_rate * tau
                  + params[layout.log_tau_spatial_idx];
        }

        // Phi (dispersion) priors
        if (layout.legacy.has_phi_num) lp += params[layout.legacy.log_phi_num_idx];
        if (layout.legacy.has_phi_denom) lp += params[layout.legacy.log_phi_denom_idx];

        *log_post_out = lp;
    }
}

// =====================================================================
// GP + Temporal gradient (hand-coded)
// Combines GP spatial with temporal RW1/RW2/AR1
// =====================================================================

void compute_gradient_gp_plus_temporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- GP-specific parameters ---
    int N_gp = data.gp_data.n_obs;
    double sigma2_gp = std::exp(params[layout.log_sigma2_gp_idx]);
    double phi_gp = std::exp(params[layout.log_phi_gp_idx]);
    std::vector<double> gp_w(N_gp);
    for (int i = 0; i < N_gp; i++) gp_w[i] = params[layout.gp_w_start + i];

    // --- Temporal parameters ---
    double tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
    int T_len = layout.temporal_end - layout.temporal_start;
    const double* phi_temporal = &params[layout.temporal_start];
    double rho_ar1 = (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        ? 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx])) : 0.5;

    // --- Shared base priors + feature-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    grad[layout.log_sigma2_gp_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
    grad[layout.log_phi_gp_idx] = 1.0;
    tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;

    // NNGP prior gradients
    tulpa_gp::NNGPGradients nngp_grads;
    tulpa_gp::gp_nngp_gradients(gp_w, sigma2_gp, phi_gp, data.gp_data, nngp_grads);
    for (int i = 0; i < N_gp; i++) grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
    grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
    grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // GP-specific + temporal eta contribution
    std::vector<double> grad_temporal_lik(T_len, 0.0);
    std::vector<int> obs_t_idx(pre.N, -1);
    for (int i = 0; i < pre.N; i++) {
        int loc_i = data.gp_data.obs_to_loc[i];
        double gp_effect = gp_w[loc_i];
        vec_grad_ws.eta_num[i] += gp_effect;
        if (!pre.is_binomial && data.gp_data.shared) vec_grad_ws.eta_denom[i] += gp_effect;

        if (!data.temporal_time_idx.empty() && i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_idx = g * data.n_times + t;
            if (t_idx >= 0 && t_idx < T_len) {
                obs_t_idx[i] = t_idx;
                vec_grad_ws.eta_num[i] += phi_temporal[t_idx];
                if (!pre.is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += phi_temporal[t_idx];
            }
        }
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // GP + temporal residual scatter
    for (int i = 0; i < pre.N; i++) {
        int loc_i = data.gp_data.obs_to_loc[i];
        grad[layout.gp_w_start + loc_i] += data.gp_data.shared
            ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
            : vec_grad_ws.resid_num[i];
        if (obs_t_idx[i] >= 0) grad_temporal_lik[obs_t_idx[i]] += data.temporal_shared
            ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
            : vec_grad_ws.resid_num[i];
    }

    // Temporal GMRF gradients
    temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                             phi_temporal, T_len, grad_temporal_lik.data(), grad.data());

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// Temporal GP (standalone) hand-coded gradients
// Temporal GP with exponential covariance uses state-space AR(1) form
// =====================================================================

void compute_gradient_temporal_gp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);
    double tgp_lp_accum = 0.0;  // Accumulates temporal GP prior terms

    // --- Temporal GP hyperparameters ---
    double sigma2_tgp = std::exp(params[layout.log_sigma2_temporal_gp_idx]);
    double logit_phi_val = params[layout.logit_phi_temporal_gp_idx];

    // Logit-bounded phi: phi = lower + range * sigmoid(logit_phi)
    double phi_lower = data.temporal_gp_phi_prior_lower;
    double phi_upper = data.temporal_gp_phi_prior_upper;
    double phi_range = phi_upper - phi_lower;
    double sigmoid_val = 1.0 / (1.0 + std::exp(-logit_phi_val));
    double phi_tgp = phi_lower + phi_range * sigmoid_val;

    // Conversion factor: grad_logit = grad_log * chi
    double chi_tgp = (phi_tgp - phi_lower) * (phi_upper - phi_tgp) / (phi_tgp * phi_range);

    // Temporal effects: n_temporal_groups * n_times parameters
    int T_times = data.n_times;
    int n_groups = data.n_temporal_groups;
    const double* phi_temporal = &params[layout.temporal_start];
    int T_len = layout.temporal_end - layout.temporal_start;

    // Non-centered parameterization: params store z ~ N(0,1), reconstruct f
    const bool use_nc = (data.temporal_gp_parameterization == 1);
    static thread_local tulpa_temporal_gp::TemporalGPNCWorkspace nc_ws;
    const double* f_temporal = phi_temporal;  // Default: centered, f stored directly

    if (use_nc) {
        nc_ws.init(T_times, n_groups);
        tulpa_temporal_gp::temporal_gp_nc_forward(
            phi_temporal, T_times, n_groups,
            sigma2_tgp, phi_tgp,
            data.temporal_gp_data.time_values, nc_ws);
        f_temporal = nc_ws.f.data();  // Use reconstructed f for eta
        std::memset(nc_ws.dL_df.data(), 0, T_len * sizeof(double));
    }

    // --- Shared base priors + temporal GP-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // sigma2: PC prior
    double sigma_tgp = std::sqrt(sigma2_tgp);
    double rate_tgp = -std::log(data.temporal_gp_sigma2_prior_alpha) / data.temporal_gp_sigma2_prior_U;
    grad[layout.log_sigma2_temporal_gp_idx] = -0.5 * rate_tgp * sigma_tgp + 0.5;

    // phi: Logit-bounded Jacobian gradient
    grad[layout.logit_phi_temporal_gp_idx] = (phi_upper + phi_lower - 2.0 * phi_tgp) / phi_range;

    if (use_nc) {
        // NC prior: z ~ N(0, I), Jacobian gradients
        grad[layout.log_sigma2_temporal_gp_idx] += 0.5 * T_times * n_groups;

        double jac_phi_log = 0.0;
        for (int t = 1; t < T_times; t++) {
            double rho_t = nc_ws.rho[t - 1];
            double rho2 = rho_t * rho_t;
            double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
            double dt_over_phi = dt / phi_tgp;
            double one_minus_rho2 = 1.0 - rho2;
            if (one_minus_rho2 < 1e-10) one_minus_rho2 = 1e-10;
            jac_phi_log -= rho2 * dt_over_phi / one_minus_rho2;
        }
        grad[layout.logit_phi_temporal_gp_idx] += jac_phi_log * n_groups * chi_tgp;

        for (int t = 0; t < T_len; t++) {
            grad[layout.temporal_start + t] = -phi_temporal[t];
        }

        // Fuse temporal GP prior log-prob
        if (pre.fuse_lp) {
            double log_sigma2 = params[layout.log_sigma2_temporal_gp_idx];
            tgp_lp_accum += std::log(rate_tgp) - rate_tgp * sigma_tgp
                          - std::log(2.0 * sigma_tgp) + log_sigma2;
            tgp_lp_accum += std::log(phi_tgp - phi_lower)
                          + std::log(phi_upper - phi_tgp)
                          - std::log(phi_range);
            double nc_jac = T_times * std::log(sigma_tgp);
            for (int t = 1; t < T_times; t++) {
                double one_m_rho2 = 1.0 - nc_ws.rho[t-1] * nc_ws.rho[t-1];
                if (one_m_rho2 < 1e-10) one_m_rho2 = 1e-10;
                nc_jac += 0.5 * std::log(one_m_rho2);
            }
            tgp_lp_accum += nc_jac * n_groups;
            for (int t = 0; t < T_len; t++) {
                tgp_lp_accum += -0.5 * phi_temporal[t] * phi_temporal[t];
            }
        }
    } else {
        // Centered: temporal GP prior gradients (state-space exponential form)
        for (int g = 0; g < n_groups; g++) {
            int offset = g * T_times;

            double f0 = phi_temporal[offset];
            grad[layout.temporal_start + offset] += -f0 / sigma2_tgp;

            double grad_log_sigma2_prior = -0.5 + 0.5 * f0 * f0 / sigma2_tgp;
            double grad_log_phi_prior = 0.0;

            for (int t = 1; t < T_times; t++) {
                double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
                double rho = std::exp(-dt / phi_tgp);
                double rho2 = rho * rho;
                double cv = sigma2_tgp * (1.0 - rho2);
                if (cv < 1e-10) cv = 1e-10;

                double f_prev = phi_temporal[offset + t - 1];
                double f_curr = phi_temporal[offset + t];
                double r = f_curr - rho * f_prev;

                grad[layout.temporal_start + offset + t] += -r / cv;
                grad[layout.temporal_start + offset + t - 1] += rho * r / cv;

                grad_log_sigma2_prior += -0.5 + 0.5 * r * r / cv;

                double dt_over_phi = dt / phi_tgp;
                grad_log_phi_prior += dt_over_phi * (
                    sigma2_tgp * rho2 / cv
                    + rho * r * f_prev / cv
                    + sigma2_tgp * rho2 * r * r / (cv * cv)
                );
            }

            grad[layout.log_sigma2_temporal_gp_idx] += grad_log_sigma2_prior;
            grad[layout.logit_phi_temporal_gp_idx] += grad_log_phi_prior * chi_tgp;
        }
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // Temporal GP-specific eta contribution
    static thread_local std::vector<double> grad_temporal_lik;
    grad_temporal_lik.assign(T_len, 0.0);

    for (int i = 0; i < pre.N; i++) {
        if (!data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = (i < (int)data.temporal_group_idx.size() && data.temporal_group_idx[i] > 0)
                    ? data.temporal_group_idx[i] - 1 : 0;
            int flat_idx = g * T_times + t;
            if (flat_idx >= 0 && flat_idx < T_len) {
                vec_grad_ws.eta_num[i] += f_temporal[flat_idx];
                if (!pre.is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += f_temporal[flat_idx];
            }
        }
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // Temporal GP-specific residual scatter
    for (int i = 0; i < pre.N; i++) {
        if (!data.temporal_time_idx.empty() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = (i < (int)data.temporal_group_idx.size() && data.temporal_group_idx[i] > 0)
                    ? data.temporal_group_idx[i] - 1 : 0;
            int flat_idx = g * T_times + t;
            if (flat_idx >= 0 && flat_idx < T_len)
                grad_temporal_lik[flat_idx] += data.temporal_shared
                    ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
                    : vec_grad_ws.resid_num[i];
        }
    }

    if (use_nc) {
        // NC: backward pass converts dL/df -> dL/dz and accumulates sigma2/phi grads
        std::memcpy(nc_ws.dL_df.data(), grad_temporal_lik.data(), T_len * sizeof(double));

        double grad_log_sigma2_lik = 0.0, grad_log_phi_lik_tgp = 0.0;
        tulpa_temporal_gp::temporal_gp_nc_backward(
            phi_temporal, T_times, n_groups,
            sigma2_tgp, phi_tgp,
            data.temporal_gp_data.time_values,
            nc_ws, &grad[layout.temporal_start],
            grad_log_sigma2_lik, grad_log_phi_lik_tgp);
        grad[layout.log_sigma2_temporal_gp_idx] += grad_log_sigma2_lik;
        grad[layout.logit_phi_temporal_gp_idx] += grad_log_phi_lik_tgp * chi_tgp;
    } else {
        // Centered: add likelihood contribution to temporal effects directly
        for (int t = 0; t < T_len; t++) {
            grad[layout.temporal_start + t] += grad_temporal_lik[t];
        }
    }

    // --- Custom epilogue (temporal GP has fused prior accumulation) ---
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), pre.cp.sigma_re);

    if (pre.fuse_lp) {
        if (use_nc && tgp_lp_accum != 0.0) {
            *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true,
                                             nullptr, &tgp_lp_accum) + pre.obs_log_lik;
        } else {
            *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + pre.obs_log_lik;
        }
    }
}

// =====================================================================
// Multi-scale GP + Temporal hand-coded gradients
// Combines MSGP spatial gradients with temporal GMRF gradients
// =====================================================================

void compute_gradient_msgp_plus_temporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- Multi-scale GP parameters ---
    int N_gp = data.multiscale_gp_data.n_obs;
    double sigma2_local = std::exp(params[layout.log_sigma2_gp_local_idx]);
    double phi_local = std::exp(params[layout.log_phi_gp_local_idx]);
    double sigma2_regional = std::exp(params[layout.log_sigma2_gp_regional_idx]);
    double phi_regional = std::exp(params[layout.log_phi_gp_regional_idx]);

    std::vector<double> w_local(N_gp), w_regional(N_gp);
    for (int i = 0; i < N_gp; i++) {
        w_local[i] = params[layout.gp_local_start + i];
        w_regional[i] = params[layout.gp_regional_start + i];
    }

    // --- Temporal parameters ---
    double tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
    int T_len = layout.temporal_end - layout.temporal_start;
    const double* phi_temporal = &params[layout.temporal_start];
    double rho_ar1 = (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        ? 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx])) : 0.5;

    // Bounds check for phi
    if (phi_local < data.multiscale_gp_data.range_local_lower ||
        phi_local > data.multiscale_gp_data.range_local_upper ||
        phi_regional < data.multiscale_gp_data.range_regional_lower ||
        phi_regional > data.multiscale_gp_data.range_regional_upper) {
        return;
    }

    // --- Shared base priors + feature-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // PC priors on MSGP variances
    grad[layout.log_sigma2_gp_local_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_local, data.ms_sigma2_local_prior_U, data.ms_sigma2_local_prior_alpha);
    grad[layout.log_sigma2_gp_regional_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_regional, data.ms_sigma2_regional_prior_U, data.ms_sigma2_regional_prior_alpha);
    grad[layout.log_phi_gp_local_idx] = 1.0;
    grad[layout.log_phi_gp_regional_idx] = 1.0;

    // Temporal prior
    tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
    if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
        grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;

    // NNGP prior gradients for multi-scale GP
    auto [gp_local, gp_regional] = make_msgp_gp_views(data.multiscale_gp_data);

    tulpa_gp::NNGPGradients nngp_grads_local, nngp_grads_regional;
    tulpa_gp::gp_nngp_gradients(w_local, sigma2_local, phi_local, gp_local, nngp_grads_local);
    tulpa_gp::gp_nngp_gradients(w_regional, sigma2_regional, phi_regional, gp_regional, nngp_grads_regional);

    for (int i = 0; i < N_gp; i++) {
        grad[layout.gp_local_start + i] += nngp_grads_local.grad_w[i];
        grad[layout.gp_regional_start + i] += nngp_grads_regional.grad_w[i];
    }
    grad[layout.log_sigma2_gp_local_idx] += nngp_grads_local.grad_log_sigma2;
    grad[layout.log_phi_gp_local_idx] += nngp_grads_local.grad_log_phi;
    grad[layout.log_sigma2_gp_regional_idx] += nngp_grads_regional.grad_log_sigma2;
    grad[layout.log_phi_gp_regional_idx] += nngp_grads_regional.grad_log_phi;

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // MSGP + temporal eta contribution
    std::vector<double> grad_temporal_lik(T_len, 0.0);
    std::vector<int> obs_t_idx(pre.N, -1);
    for (int i = 0; i < pre.N; i++) {
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        double ms_spatial = w_local[loc_i] + w_regional[loc_i];
        vec_grad_ws.eta_num[i] += ms_spatial;
        if (!pre.is_binomial && data.multiscale_gp_data.shared) vec_grad_ws.eta_denom[i] += ms_spatial;

        if (!data.temporal_time_idx.empty() && i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_idx = g * data.n_times + t;
            if (t_idx >= 0 && t_idx < T_len) {
                obs_t_idx[i] = t_idx;
                vec_grad_ws.eta_num[i] += phi_temporal[t_idx];
                if (!pre.is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += phi_temporal[t_idx];
            }
        }
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // MSGP + temporal residual scatter
    for (int i = 0; i < pre.N; i++) {
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        double dLL_dspatial = data.multiscale_gp_data.shared
            ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
            : vec_grad_ws.resid_num[i];
        grad[layout.gp_local_start + loc_i] += dLL_dspatial;
        grad[layout.gp_regional_start + loc_i] += dLL_dspatial;
        if (obs_t_idx[i] >= 0) grad_temporal_lik[obs_t_idx[i]] += data.temporal_shared
            ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
            : vec_grad_ws.resid_num[i];
    }

    // Temporal GMRF gradients
    temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                             phi_temporal, T_len, grad_temporal_lik.data(), grad.data());

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// SVC gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from svc_nngp_gradients for NNGP prior
// =====================================================================

void compute_gradient_svc_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- SVC-specific parameters ---
    int n_svc = data.svc_data.n_svc;
    int N_obs = data.svc_data.n_obs;

    double* svc_sigma2 = data.svc_data.sigma2_ws.data();
    double* svc_phi = data.svc_data.phi_ws.data();
    for (int j = 0; j < n_svc; j++) {
        svc_sigma2[j] = std::exp(params[layout.log_sigma2_svc_start + j]);
        svc_phi[j] = std::exp(params[layout.log_phi_svc_start + j]);
        if (svc_phi[j] < data.svc_phi_prior_lower || svc_phi[j] > data.svc_phi_prior_upper) {
            return;
        }
    }

    double* svc_w_flat = data.svc_data.w_flat_ws.data();
    for (int k = 0; k < N_obs * n_svc; k++) {
        svc_w_flat[k] = params[layout.svc_w_start + k];
    }

    // --- Shared base priors + SVC-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    for (int j = 0; j < n_svc; j++) {
        double sigma = std::sqrt(svc_sigma2[j]);
        double ratio = sigma / data.svc_sigma2_prior_scale;
        double ratio_sq = ratio * ratio;
        grad[layout.log_sigma2_svc_start + j] = -ratio_sq / (1.0 + ratio_sq) + 1.0;
        grad[layout.log_phi_svc_start + j] = 1.0;
    }

    // NNGP gradients for SVC effects
    double* w_j_ptr = data.svc_data.w_j_ws.data();
    for (int j = 0; j < n_svc; j++) {
        for (int i = 0; i < N_obs; i++) {
            w_j_ptr[i] = svc_w_flat[j * N_obs + i];
        }
        std::vector<double> w_j_vec(w_j_ptr, w_j_ptr + N_obs);
        tulpa_svc::SVCGradients svc_grads;
        tulpa_svc::svc_nngp_gradients(w_j_vec, svc_sigma2[j], svc_phi[j], data.svc_data, svc_grads);

        for (int i = 0; i < N_obs; i++) {
            grad[layout.svc_w_start + j * N_obs + i] += svc_grads.grad_w[i];
        }
        grad[layout.log_sigma2_svc_start + j] += svc_grads.grad_log_sigma2;
        grad[layout.log_phi_svc_start + j] += svc_grads.grad_log_phi;

        double sum_w = 0.0;
        for (int i = 0; i < N_obs; i++) sum_w += w_j_ptr[i];
        double stz_grad = -sum_w / N_obs;
        for (int i = 0; i < N_obs; i++) {
            grad[layout.svc_w_start + j * N_obs + i] += stz_grad;
        }
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // SVC-specific eta contribution
    for (int i = 0; i < pre.N; i++) {
        double svc_effect = 0.0;
        for (int j = 0; j < n_svc; j++) {
            svc_effect += data.svc_data.X_svc[i * n_svc + j] * svc_w_flat[j * N_obs + i];
        }
        vec_grad_ws.eta_num[i] += svc_effect;
        if (!pre.is_binomial && data.svc_data.shared) vec_grad_ws.eta_denom[i] += svc_effect;
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // SVC-specific residual scatter
    for (int i = 0; i < pre.N; i++) {
        double dLL_dsvc = data.svc_data.shared
            ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
            : vec_grad_ws.resid_num[i];
        for (int j = 0; j < n_svc; j++) {
            grad[layout.svc_w_start + j * N_obs + i] += dLL_dsvc * data.svc_data.X_svc[i * n_svc + j];
        }
    }

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// HSGP-SVC gradient (hand-coded)
// Uses HSGP basis function approximation for spatially-varying coefficients.
// Each SVC term k has its own GP with sigma2_k, lengthscale_k, beta_k[m^2].
// All terms share the same basis matrix Phi (same coordinates/eigenvalues).
// =====================================================================

void compute_gradient_svc_hsgp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // Thread-local HSGP workspace (one per SVC term, reused)
    static thread_local tulpa_hsgp::HSGPWorkspace hsgp_ws;
    hsgp_ws.init(data.N, data.svc_hsgp_data.m_total);

    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- SVC HSGP-specific parameters ---
    const int n_svc = data.svc_data.n_svc;
    const int m_total = data.svc_hsgp_data.m_total;

    // Evaluate f_k(s_i) for each SVC term and accumulate SVC contribution to eta
    std::vector<double> svc_eta(pre.N, 0.0);
    std::vector<double> svc_f_all(n_svc * pre.N);

    for (int j = 0; j < n_svc; j++) {
        double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
        double lengthscale_j = std::exp(params[layout.log_phi_svc_start + j]);
        const double* beta_j = &params[layout.svc_w_start + j * m_total];

        tulpa_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                       data.svc_hsgp_data, hsgp_ws);

        std::memcpy(&svc_f_all[j * pre.N], hsgp_ws.hsgp_f.data(), pre.N * sizeof(double));
        for (int i = 0; i < pre.N; i++) {
            svc_eta[i] += data.svc_data.X_svc[i * n_svc + j] * hsgp_ws.hsgp_f[i];
        }
    }

    // --- Temporal parameters (GMRF: RW1/AR1) ---
    double tau_temporal = 0.0;
    int T_len_svc = 0;
    const double* phi_temporal_svc = nullptr;
    double rho_ar1_svc = 0.5;
    if (layout.has_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_len_svc = layout.temporal_end - layout.temporal_start;
        phi_temporal_svc = &params[layout.temporal_start];
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
            rho_ar1_svc = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx]));
        }
    }
    std::vector<double> grad_temporal_lik_svc(T_len_svc, 0.0);

    // --- Shared base priors + feature-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // Temporal prior
    if (layout.has_temporal) {
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
            grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1_svc;
    }

    // Per-term HSGP hyperparameter priors
    for (int j = 0; j < n_svc; j++) {
        double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
        double log_ls_j = params[layout.log_phi_svc_start + j];

        double sigma_j = std::sqrt(sigma2_j);
        double rate_sigma = 4.6;
        grad[layout.log_sigma2_svc_start + j] = -0.5 * rate_sigma * sigma_j + 0.5;

        grad[layout.log_phi_svc_start + j] = -log_ls_j;

        for (int k = 0; k < m_total; k++) {
            grad[layout.svc_w_start + j * m_total + k] = -params[layout.svc_w_start + j * m_total + k];
        }
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // SVC HSGP-specific eta contribution
    for (int i = 0; i < pre.N; i++) {
        vec_grad_ws.eta_num[i] += svc_eta[i];
        if (!pre.is_binomial && data.svc_data.shared) vec_grad_ws.eta_denom[i] += svc_eta[i];
    }

    // Temporal eta contribution
    if (layout.has_temporal && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < pre.N; i++) {
            if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t = data.temporal_time_idx[i] - 1;
                int g = data.temporal_group_idx[i] - 1;
                int t_idx = g * data.n_times + t;
                if (t_idx >= 0 && t_idx < T_len_svc) {
                    vec_grad_ws.eta_num[i] += phi_temporal_svc[t_idx];
                    if (!pre.is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += phi_temporal_svc[t_idx];
                }
            }
        }
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // --- HSGP gradient backprop per SVC term ---
    for (int j = 0; j < n_svc; j++) {
        double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
        double lengthscale_j = std::exp(params[layout.log_phi_svc_start + j]);
        const double* beta_j = &params[layout.svc_w_start + j * m_total];

        double* grad_f_ptr = hsgp_ws.grad_f.data();
        for (int i = 0; i < pre.N; i++) {
            double dLL = data.svc_data.shared
                ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
                : vec_grad_ws.resid_num[i];
            grad_f_ptr[i] = dLL * data.svc_data.X_svc[i * n_svc + j];
        }

        tulpa_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                       data.svc_hsgp_data, hsgp_ws);

        double grad_log_sigma2_j, grad_log_lengthscale_j;
        tulpa_hsgp::hsgp_compute_gradients_ws(beta_j, sigma2_j, lengthscale_j,
                                                data.svc_hsgp_data, hsgp_ws,
                                                grad_log_sigma2_j, grad_log_lengthscale_j);

        for (int k = 0; k < m_total; k++) {
            grad[layout.svc_w_start + j * m_total + k] += hsgp_ws.grad_beta_out[k];
        }
        grad[layout.log_sigma2_svc_start + j] += grad_log_sigma2_j;
        grad[layout.log_phi_svc_start + j] += grad_log_lengthscale_j;
    }

    // Temporal likelihood gradient scatter
    if (layout.has_temporal && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < pre.N; i++) {
            if (i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t = data.temporal_time_idx[i] - 1;
                int g = data.temporal_group_idx[i] - 1;
                int t_idx = g * data.n_times + t;
                if (t_idx >= 0 && t_idx < T_len_svc) {
                    double lik_grad = data.temporal_shared ?
                        (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i]) :
                        vec_grad_ws.resid_num[i];
                    grad_temporal_lik_svc[t_idx] += lik_grad;
                }
            }
        }
    }

    // Temporal GMRF prior gradients
    if (layout.has_temporal && T_len_svc > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1_svc,
                                 phi_temporal_svc, T_len_svc, grad_temporal_lik_svc.data(), grad.data());
    }

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// TVC gradient (hand-coded, ~3x faster than autodiff)
// Uses analytical gradients from hmc_tvc_grad.h for RW1/RW2/AR1 priors
// =====================================================================

void compute_gradient_tvc_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- TVC-specific parameters ---
    int n_tvc = data.tvc_data.n_tvc;
    int n_times = data.tvc_data.n_times;
    int n_groups = data.tvc_data.n_groups;
    int n_w = n_groups * n_tvc * n_times;

    double* tvc_tau = data.tvc_data.tau_ws.data();
    double* tvc_rho = data.tvc_data.rho_ws.data();
    for (int j = 0; j < n_tvc; j++) {
        tvc_tau[j] = std::exp(params[layout.log_tau_tvc_start + j]);
        if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
            double logit_rho = params[layout.logit_rho_tvc_start + j];
            double u = 1.0 / (1.0 + std::exp(-logit_rho));
            tvc_rho[j] = 2.0 * u - 1.0;
        } else {
            tvc_rho[j] = 0.0;
        }
    }

    double* tvc_w_flat = data.tvc_data.w_flat_ws.data();
    for (int k = 0; k < n_w; k++) {
        tvc_w_flat[k] = params[layout.tvc_w_start + k];
    }

    // --- Shared base priors + TVC-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    double tvc_pc_rate = -std::log(0.01) / 1.0;
    for (int j = 0; j < n_tvc; j++) {
        double sigma_j = 1.0 / std::sqrt(tvc_tau[j]);
        grad[layout.log_tau_tvc_start + j] = 0.5 * tvc_pc_rate * sigma_j + 1.5;
    }
    if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
        for (int j = 0; j < n_tvc; j++) {
            double u = (tvc_rho[j] + 1.0) / 2.0;
            grad[layout.logit_rho_tvc_start + j] = 1.0 - 2.0 * u;
        }
    }

    // TVC prior gradients
    tulpa_tvc::TVCGradientWS tvc_ws;
    tvc_ws.grad_w = data.tvc_data.grad_w_ws.data();
    tvc_ws.grad_log_tau = data.tvc_data.grad_log_tau_ws.data();
    tvc_ws.grad_logit_rho = data.tvc_data.grad_logit_rho_ws.data();
    tvc_ws.grad_w_jg = data.tvc_data.grad_w_jg_ws.data();
    tvc_ws.d_buf = data.tvc_data.d_ws.data();
    tvc_ws.n_w = n_w;
    tvc_ws.n_tvc = n_tvc;
    tulpa_tvc::tvc_prior_gradients_ws(tvc_w_flat, data.tvc_data, tvc_tau, tvc_rho, tvc_ws);

    for (int k = 0; k < n_w; k++) {
        grad[layout.tvc_w_start + k] += tvc_ws.grad_w[k];
    }
    for (int j = 0; j < n_tvc; j++) {
        grad[layout.log_tau_tvc_start + j] += tvc_ws.grad_log_tau[j];
    }
    if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.logit_rho_tvc_start + j] += tvc_ws.grad_logit_rho[j];
        }
    }

    // Precompute TVC eta contribution
    double* tvc_eta = data.tvc_data.eta_ws.data();
    std::fill(tvc_eta, tvc_eta + data.N, 0.0);
    for (int i = 0; i < data.N; i++) {
        int t = data.tvc_data.time_index[i] - 1;
        int g = data.tvc_data.group_index[i] - 1;
        for (int j = 0; j < n_tvc; j++) {
            tvc_eta[i] += data.tvc_data.X_tvc[i * n_tvc + j] * tvc_w_flat[(g * n_tvc + j) * n_times + t];
        }
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // TVC-specific eta contribution
    for (int i = 0; i < pre.N; i++) {
        vec_grad_ws.eta_num[i] += tvc_eta[i];
        if (!pre.is_binomial && data.tvc_data.shared) vec_grad_ws.eta_denom[i] += tvc_eta[i];
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // TVC-specific residual scatter
    for (int i = 0; i < pre.N; i++) {
        double dLL_dtvc = data.tvc_data.shared
            ? (vec_grad_ws.resid_num[i] + vec_grad_ws.resid_denom[i])
            : vec_grad_ws.resid_num[i];
        int t = data.tvc_data.time_index[i] - 1;
        int g = data.tvc_data.group_index[i] - 1;
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.tvc_w_start + (g * n_tvc + j) * n_times + t] +=
                dLL_dtvc * data.tvc_data.X_tvc[i * n_tvc + j];
        }
    }

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// Latent factor gradient (hand-coded, O(N*K))
// Uses analytical gradients for latent factor models
// =====================================================================

void compute_gradient_latent_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- Latent factor parameters ---
    int K = data.latent_n_factors;

    // Extract log_sigma for latent factors
    std::vector<double> log_sigma_latent(K);
    std::vector<double> sigma_latent(K);
    for (int k = 0; k < K; k++) {
        log_sigma_latent[k] = params[layout.log_sigma_latent_start + k];
        sigma_latent[k] = std::exp(log_sigma_latent[k]);
    }

    // Extract factors (unconstrained)
    int n_factor_params = pre.N * K;
    std::vector<double> factors_raw(n_factor_params);
    for (int j = 0; j < n_factor_params; j++) {
        factors_raw[j] = params[layout.latent_factor_start + j];
    }

    // Apply constraint to get constrained factors
    std::vector<double> factors_constrained = factors_raw;
    if (data.latent_constraint == 0) {  // SUM_TO_ZERO
        for (int k = 0; k < K; k++) {
            double sum = 0.0;
            for (int i = 0; i < pre.N; i++) {
                sum += factors_constrained[i * K + k];
            }
            double mean = sum / pre.N;
            for (int i = 0; i < pre.N; i++) {
                factors_constrained[i * K + k] -= mean;
            }
        }
    } else {  // FIRST_ZERO
        for (int k = 0; k < K; k++) {
            double first_val = factors_constrained[k];  // factors[0, k]
            for (int i = 0; i < pre.N; i++) {
                factors_constrained[i * K + k] -= first_val;
            }
        }
    }

    // Precompute latent contribution to eta
    std::vector<double> latent_eta(pre.N, 0.0);
    for (int i = 0; i < pre.N; i++) {
        for (int k = 0; k < K; k++) {
            latent_eta[i] += factors_constrained[i * K + k] * sigma_latent[k];
        }
    }

    // --- Shared base priors + latent-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // Latent sigma prior: Exponential(rate) on sigma, with Jacobian for log transform
    double latent_rate = data.latent_sigma_prior_rate;
    for (int k = 0; k < K; k++) {
        grad[layout.log_sigma_latent_start + k] = 1.0 - latent_rate * sigma_latent[k];
    }

    // =========================================================================
    // Likelihood loop - compute dLL/deta and chain-rule to all parameters
    // (Latent uses scalar per-obs loop — latent factor chain rule requires it)
    // =========================================================================
    std::vector<double> grad_factors_constrained(n_factor_params, 0.0);

    for (int i = 0; i < pre.N; i++) {
        // Linear predictors
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.legacy.p_num; j++) {
            eta_num += data.legacy.X_num_flat[i * data.legacy.p_num + j] * pre.cp.beta_num[j];
        }
        for (int j = 0; j < data.legacy.p_denom; j++) {
            eta_denom += data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * pre.cp.beta_denom[j];
        }

        // Random effects (handles NC parameterization)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            double re_eff = re_value_for_eta(pre.cp.re, g, pre.cp.sigma_re, data.re_parameterization);
            eta_num += re_eff;
            eta_denom += re_eff;
        }

        // Latent effect
        double latent_effect = latent_eta[i];
        if (data.latent_shared) {
            eta_num += latent_effect;
            eta_denom += latent_effect;
        } else {
            eta_num += latent_effect;
        }

        if (pre.fuse_lp) pre.obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, pre.cp.phi_num, pre.cp.phi_denom);

        double dLL_deta_num = 0.0, dLL_deta_denom = 0.0;
        compute_obs_residuals(data, i, eta_num, eta_denom, pre.cp.phi_num, pre.cp.phi_denom, dLL_deta_num, dLL_deta_denom);

        // Total gradient through latent effect
        double dLL_dlatent = data.latent_shared ?
                             (dLL_deta_num + dLL_deta_denom) : dLL_deta_num;

        scatter_beta_gradients(data, layout, i, dLL_deta_num, dLL_deta_denom, grad.data());
        scatter_re_gradient(data, layout, i, dLL_deta_num, dLL_deta_denom, grad.data());

        // Gradients for latent factors (on constrained space)
        for (int k = 0; k < K; k++) {
            grad_factors_constrained[i * K + k] = dLL_dlatent * sigma_latent[k];
            grad[layout.log_sigma_latent_start + k] += dLL_dlatent * factors_constrained[i * K + k] * sigma_latent[k];
        }

        accumulate_phi_likelihood_grad(data, layout, i, eta_num, eta_denom, pre.cp.phi_num, pre.cp.phi_denom, grad.data());
    }

    // =========================================================================
    // Add prior gradient to grad_factors_constrained
    // =========================================================================
    int prior_start = (data.latent_constraint == 0) ? 0 : 1;
    for (int k = 0; k < K; k++) {
        for (int i = prior_start; i < pre.N; i++) {
            grad_factors_constrained[i * K + k] += -factors_constrained[i * K + k];
        }
    }

    // =========================================================================
    // Apply constraint chain-rule to get gradients on raw (unconstrained) factors
    // =========================================================================
    if (data.latent_constraint == 0) {  // SUM_TO_ZERO
        for (int k = 0; k < K; k++) {
            double sum_grad = 0.0;
            for (int i = 0; i < pre.N; i++) {
                sum_grad += grad_factors_constrained[i * K + k];
            }
            double mean_grad = sum_grad / pre.N;

            for (int i = 0; i < pre.N; i++) {
                grad[layout.latent_factor_start + i * K + k] +=
                    grad_factors_constrained[i * K + k] - mean_grad;
            }
        }
    } else {  // FIRST_ZERO
        for (int k = 0; k < K; k++) {
            double sum_grad = 0.0;
            for (int i = 1; i < pre.N; i++) {
                sum_grad += grad_factors_constrained[i * K + k];
                grad[layout.latent_factor_start + i * K + k] += grad_factors_constrained[i * K + k];
            }
            grad[layout.latent_factor_start + k] += -sum_grad;  // factor_raw[0,k]
        }
    }

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// GP gradient via autodiff (O(N*nn^3) - much faster than numerical O(N^2))
// Uses templated NNGP likelihood from hmc_gp_autodiff.h
// =====================================================================

// =====================================================================
// Common autodiff prior setup (beta, RE, phi priors)
// Shared by gp_autodiff, msgp_autodiff, gp_temporal_autodiff.
// Returns log_post, sigma_re, phi_num, phi_denom via output parameters.
// =====================================================================
struct AutodiffCommonResult {
    tulpa::ad::Var log_post;
    tulpa::ad::Var sigma_re;
    tulpa::ad::Var phi_num;
    tulpa::ad::Var phi_denom;
};

static inline AutodiffCommonResult add_common_priors_ad(
    tulpa::ad::Tape* tape,
    const std::vector<tulpa::ad::Var>& params_ad,
    const ModelData& data,
    const ParamLayout& layout
) {
    using namespace tulpa::ad;
    using namespace tulpa::math;

    Var log_post(tape, 0.0);

    // Fixed effects priors: N(0, sigma_beta^2)
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    for (int j = 0; j < data.legacy.p_num; j++) {
        Var beta = params_ad[layout.legacy.beta_num_start + j];
        log_post = log_post - (0.5 * tau_beta) * beta * beta;
    }
    for (int j = 0; j < data.legacy.p_denom; j++) {
        Var beta = params_ad[layout.legacy.beta_denom_start + j];
        log_post = log_post - (0.5 * tau_beta) * beta * beta;
    }

    // Random effects priors (if present)
    Var sigma_re(tape, 1.0);
    if (layout.has_re && data.n_re_groups > 0) {
        Var log_sigma_re = params_ad[layout.log_sigma_re_idx];
        sigma_re = safe_exp(log_sigma_re);

        Var ratio = sigma_re / data.sigma_re_scale;
        log_post = log_post - safe_log(1.0 + ratio * ratio);
        log_post = log_post + log_sigma_re;  // Jacobian

        if (data.re_parameterization == 1) {
            for (int g = 0; g < data.n_re_groups; g++) {
                Var re_g = params_ad[layout.re_start + g];
                log_post = log_post - 0.5 * re_g * re_g;
            }
        } else {
            Var tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
            for (int g = 0; g < data.n_re_groups; g++) {
                Var re_g = params_ad[layout.re_start + g];
                log_post = log_post - 0.5 * tau_re * re_g * re_g;
                log_post = log_post + 0.5 * safe_log(tau_re);
            }
        }
    }

    // Overdispersion priors (Gamma)
    Var phi_num(tape, 1.0);
    Var phi_denom(tape, 1.0);
    if (layout.legacy.has_phi_num) {
        Var log_phi = params_ad[layout.legacy.log_phi_num_idx];
        phi_num = safe_exp(log_phi);
        log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi
                            - data.phi_prior_rate * phi_num + log_phi;
    }
    if (layout.legacy.has_phi_denom) {
        Var log_phi = params_ad[layout.legacy.log_phi_denom_idx];
        phi_denom = safe_exp(log_phi);
        log_post = log_post + (data.phi_prior_shape - 1.0) * log_phi
                            - data.phi_prior_rate * phi_denom + log_phi;
    }

    return {log_post, sigma_re, phi_num, phi_denom};
}

void compute_gradient_gp_autodiff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
    using namespace tulpa::ad;
    using namespace tulpa::math;

    TapeScope tape_scope;
    Tape* tape = tape_scope.tape;

    int n_params = params.size();
    grad.assign(n_params, 0.0);

    std::vector<Var> params_ad = make_vars(tape, params);

    auto [log_post, sigma_re, phi_num, phi_denom] = add_common_priors_ad(tape, params_ad, data, layout);

    // =========================================================================
    // GP priors and NNGP likelihood
    // =========================================================================
    std::vector<Var> gp_w_ad;
    Var sigma2_gp(tape, 1.0);
    Var phi_gp(tape, 0.1);

    if (layout.is_gp && data.has_gp) {
        Var log_sigma2_gp = params_ad[layout.log_sigma2_gp_idx];
        Var log_phi_gp = params_ad[layout.log_phi_gp_idx];
        sigma2_gp = safe_exp(log_sigma2_gp);
        phi_gp = safe_exp(log_phi_gp);

        // PC prior on sigma2 (penalizes large variance)
        log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
            sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        log_post = log_post + log_sigma2_gp;  // Jacobian

        // Uniform prior on phi within bounds
        log_post = log_post + tulpa_gp::log_prior_phi_uniform_t(
            phi_gp, data.gp_phi_prior_lower, data.gp_phi_prior_upper);
        log_post = log_post + log_phi_gp;  // Jacobian

        // Extract GP spatial effects
        int N_gp = data.gp_data.n_obs;
        gp_w_ad.resize(N_gp);
        for (int i = 0; i < N_gp; i++) {
            gp_w_ad[i] = params_ad[layout.gp_w_start + i];
        }

        // NNGP log-likelihood using templated function
        Var gp_ll = tulpa_gp::gp_nngp_log_lik_t(gp_w_ad, sigma2_gp, phi_gp, data.gp_data);
        log_post = log_post + gp_ll;
    }

    // =========================================================================
    // Data likelihood
    // =========================================================================
    std::vector<Var> beta_num_ad(data.legacy.p_num);
    std::vector<Var> beta_denom_ad(data.legacy.p_denom);
    for (int j = 0; j < data.legacy.p_num; j++) {
        beta_num_ad[j] = params_ad[layout.legacy.beta_num_start + j];
    }
    for (int j = 0; j < data.legacy.p_denom; j++) {
        beta_denom_ad[j] = params_ad[layout.legacy.beta_denom_start + j];
    }

    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        Var eta_num(tape, 0.0);
        Var eta_denom(tape, 0.0);

        for (int j = 0; j < data.legacy.p_num; j++) {
            eta_num = eta_num + data.legacy.X_num_flat[i * data.legacy.p_num + j] * beta_num_ad[j];
        }
        for (int j = 0; j < data.legacy.p_denom; j++) {
            eta_denom = eta_denom + data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * beta_denom_ad[j];
        }

        // Add random effects (shared)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            Var re_g = params_ad[layout.re_start + g];
            Var re_eff = (data.re_parameterization == 1) ? sigma_re * re_g : re_g;
            eta_num = eta_num + re_eff;
            eta_denom = eta_denom + re_eff;
        }

        // Add GP spatial effect (map observation to unique location)
        if (layout.is_gp && data.has_gp && !gp_w_ad.empty()) {
            int loc_i = data.gp_data.obs_to_loc[i];
            Var gp_effect = gp_w_ad[loc_i];
            if (data.gp_data.shared) {
                eta_num = eta_num + gp_effect;
                eta_denom = eta_denom + gp_effect;
            } else {
                eta_num = eta_num + gp_effect;
            }
        }

        // Compute likelihood based on model type
        Var ll_i(tape, 0.0);

        if (data.legacy.model_type == ModelType::BINOMIAL) {
            Var p = inv_logit(eta_num);
            ll_i = tulpa::math::log_lik_binomial(data.legacy.y_num[i], data.legacy.y_denom[i], p);
        } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = tulpa::math::log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num) +
                   tulpa::math::log_lik_negbin(data.legacy.y_denom[i], mu_denom, phi_denom);
        } else {  // POISSON_GAMMA
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = tulpa::math::log_lik_poisson(data.legacy.y_num[i], mu_num) +
                   tulpa::math::log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);
        }

        log_post = log_post + ll_i;
    }

    // Backward pass
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ad);

    // TapeScope destructor handles cleanup
}

// =====================================================================
// Multi-scale GP gradient (hand-coded, ~2-3x faster than autodiff)
// Uses analytical gradients for w and numerical for sigma2/phi
// =====================================================================

void compute_gradient_msgp_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- Multi-scale GP parameters ---
    int N_gp = data.multiscale_gp_data.n_obs;

    double log_sigma2_local = params[layout.log_sigma2_gp_local_idx];
    double log_phi_local = params[layout.log_phi_gp_local_idx];
    double sigma2_local = std::exp(log_sigma2_local);
    double phi_local = std::exp(log_phi_local);

    double log_sigma2_regional = params[layout.log_sigma2_gp_regional_idx];
    double log_phi_regional = params[layout.log_phi_gp_regional_idx];
    double sigma2_regional = std::exp(log_sigma2_regional);
    double phi_regional = std::exp(log_phi_regional);

    // Extract spatial effects
    std::vector<double> w_local(N_gp), w_regional(N_gp);
    for (int i = 0; i < N_gp; i++) {
        w_local[i] = params[layout.gp_local_start + i];
        w_regional[i] = params[layout.gp_regional_start + i];
    }

    // Bounds check for phi
    if (phi_local < data.multiscale_gp_data.range_local_lower ||
        phi_local > data.multiscale_gp_data.range_local_upper ||
        phi_regional < data.multiscale_gp_data.range_regional_lower ||
        phi_regional > data.multiscale_gp_data.range_regional_upper) {
        return; // Out of bounds - return zero gradient
    }

    // --- Shared base priors + MSGP-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // PC priors on GP variances
    grad[layout.log_sigma2_gp_local_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_local, data.ms_sigma2_local_prior_U, data.ms_sigma2_local_prior_alpha);
    grad[layout.log_sigma2_gp_regional_idx] = gp_pc_prior_grad_log_sigma2(
        sigma2_regional, data.ms_sigma2_regional_prior_U, data.ms_sigma2_regional_prior_alpha);

    // Jacobians for log-transforms
    grad[layout.log_phi_gp_local_idx] = 1.0;
    grad[layout.log_phi_gp_regional_idx] = 1.0;

    // Compute NNGP gradients w.r.t. spatial effects (analytical)
    auto [gp_local, gp_regional] = make_msgp_gp_views(data.multiscale_gp_data);

    tulpa_gp::NNGPGradients nngp_grads_local, nngp_grads_regional;
    tulpa_gp::gp_nngp_gradients(w_local, sigma2_local, phi_local, gp_local, nngp_grads_local);
    tulpa_gp::gp_nngp_gradients(w_regional, sigma2_regional, phi_regional, gp_regional, nngp_grads_regional);

    for (int i = 0; i < N_gp; i++) {
        grad[layout.gp_local_start + i] += nngp_grads_local.grad_w[i];
        grad[layout.gp_regional_start + i] += nngp_grads_regional.grad_w[i];
    }

    grad[layout.log_sigma2_gp_local_idx] += nngp_grads_local.grad_log_sigma2;
    grad[layout.log_phi_gp_local_idx] += nngp_grads_local.grad_log_phi;
    grad[layout.log_sigma2_gp_regional_idx] += nngp_grads_regional.grad_log_sigma2;
    grad[layout.log_phi_gp_regional_idx] += nngp_grads_regional.grad_log_phi;

    // =========================================================================
    // Data likelihood loop (scalar per-obs — MSGP scatter requires it)
    // =========================================================================
    for (int i = 0; i < pre.N; i++) {
        // Linear predictors
        double eta_num = 0.0, eta_denom = 0.0;
        for (int j = 0; j < data.legacy.p_num; j++) {
            eta_num += data.legacy.X_num_flat[i * data.legacy.p_num + j] * pre.cp.beta_num[j];
        }
        for (int j = 0; j < data.legacy.p_denom; j++) {
            eta_denom += data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * pre.cp.beta_denom[j];
        }

        // Random effects (handles NC parameterization)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            double re_eff = re_value_for_eta(pre.cp.re, g, pre.cp.sigma_re, data.re_parameterization);
            eta_num += re_eff;
            eta_denom += re_eff;
        }

        // Multi-scale GP spatial effect
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        double ms_spatial = w_local[loc_i] + w_regional[loc_i];
        if (data.multiscale_gp_data.shared) {
            eta_num += ms_spatial;
            eta_denom += ms_spatial;
        } else {
            eta_num += ms_spatial;
        }

        if (pre.fuse_lp) pre.obs_log_lik += compute_obs_ll(data, i, eta_num, eta_denom, pre.cp.phi_num, pre.cp.phi_denom);

        double dLL_deta_num = 0.0, dLL_deta_denom = 0.0;
        compute_obs_residuals(data, i, eta_num, eta_denom, pre.cp.phi_num, pre.cp.phi_denom, dLL_deta_num, dLL_deta_denom);

        scatter_beta_gradients(data, layout, i, dLL_deta_num, dLL_deta_denom, grad.data());
        scatter_re_gradient(data, layout, i, dLL_deta_num, dLL_deta_denom, grad.data());

        // Gradients for GP spatial effects
        double dLL_dspatial = data.multiscale_gp_data.shared ?
                              (dLL_deta_num + dLL_deta_denom) : dLL_deta_num;
        grad[layout.gp_local_start + loc_i] += dLL_dspatial;
        grad[layout.gp_regional_start + loc_i] += dLL_dspatial;

        accumulate_phi_likelihood_grad(data, layout, i, eta_num, eta_denom, pre.cp.phi_num, pre.cp.phi_denom, grad.data());
    }

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// Multi-scale GP gradient (autodiff, ~3x faster than numerical)
// =====================================================================

void compute_gradient_msgp_autodiff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
    using namespace tulpa::ad;
    using namespace tulpa::math;

    TapeScope tape_scope;
    Tape* tape = tape_scope.tape;

    int n_params = params.size();
    grad.assign(n_params, 0.0);

    std::vector<Var> params_ad = make_vars(tape, params);

    auto [log_post, sigma_re, phi_num, phi_denom] = add_common_priors_ad(tape, params_ad, data, layout);

    // =========================================================================
    // Multi-scale GP priors and NNGP likelihoods
    // =========================================================================
    int N_gp = data.multiscale_gp_data.n_obs;

    // Local scale parameters
    Var log_sigma2_local = params_ad[layout.log_sigma2_gp_local_idx];
    Var log_phi_local = params_ad[layout.log_phi_gp_local_idx];
    Var sigma2_local = safe_exp(log_sigma2_local);
    Var phi_local = safe_exp(log_phi_local);

    // Regional scale parameters
    Var log_sigma2_regional = params_ad[layout.log_sigma2_gp_regional_idx];
    Var log_phi_regional = params_ad[layout.log_phi_gp_regional_idx];
    Var sigma2_regional = safe_exp(log_sigma2_regional);
    Var phi_regional = safe_exp(log_phi_regional);

    // PC priors on variances
    log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
        sigma2_local, data.ms_sigma2_local_prior_U, data.ms_sigma2_local_prior_alpha);
    log_post = log_post + log_sigma2_local;  // Jacobian

    log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
        sigma2_regional, data.ms_sigma2_regional_prior_U, data.ms_sigma2_regional_prior_alpha);
    log_post = log_post + log_sigma2_regional;  // Jacobian

    // Range priors (uniform within bounds) - check bounds, return -inf if violated
    double phi_local_val = get_value(phi_local);
    double phi_regional_val = get_value(phi_regional);
    if (phi_local_val < data.multiscale_gp_data.range_local_lower ||
        phi_local_val > data.multiscale_gp_data.range_local_upper ||
        phi_regional_val < data.multiscale_gp_data.range_regional_lower ||
        phi_regional_val > data.multiscale_gp_data.range_regional_upper) {
        // Out of bounds - return zero gradients (log_post = -inf)
        // TapeScope destructor handles cleanup
        return;
    }
    log_post = log_post + log_phi_local;    // Jacobian
    log_post = log_post + log_phi_regional; // Jacobian

    // Extract GP spatial effects
    std::vector<Var> w_local_ad(N_gp);
    std::vector<Var> w_regional_ad(N_gp);
    for (int i = 0; i < N_gp; i++) {
        w_local_ad[i] = params_ad[layout.gp_local_start + i];
        w_regional_ad[i] = params_ad[layout.gp_regional_start + i];
    }

    // NNGP log-likelihood for each scale using templated function
    Var msgp_ll = tulpa_gp::multiscale_gp_log_lik_t(
        w_local_ad, w_regional_ad,
        sigma2_local, phi_local,
        sigma2_regional, phi_regional,
        data.multiscale_gp_data);
    log_post = log_post + msgp_ll;

    // =========================================================================
    // Data likelihood
    // =========================================================================
    std::vector<Var> beta_num_ad(data.legacy.p_num);
    std::vector<Var> beta_denom_ad(data.legacy.p_denom);
    for (int j = 0; j < data.legacy.p_num; j++) {
        beta_num_ad[j] = params_ad[layout.legacy.beta_num_start + j];
    }
    for (int j = 0; j < data.legacy.p_denom; j++) {
        beta_denom_ad[j] = params_ad[layout.legacy.beta_denom_start + j];
    }

    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        Var eta_num(tape, 0.0);
        Var eta_denom(tape, 0.0);

        for (int j = 0; j < data.legacy.p_num; j++) {
            eta_num = eta_num + data.legacy.X_num_flat[i * data.legacy.p_num + j] * beta_num_ad[j];
        }
        for (int j = 0; j < data.legacy.p_denom; j++) {
            eta_denom = eta_denom + data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * beta_denom_ad[j];
        }

        // Add random effects (shared)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            Var re_g = params_ad[layout.re_start + g];
            Var re_eff = (data.re_parameterization == 1) ? sigma_re * re_g : re_g;
            eta_num = eta_num + re_eff;
            eta_denom = eta_denom + re_eff;
        }

        // Add multi-scale GP spatial effect (map observation to unique location)
        int loc_i = data.multiscale_gp_data.obs_to_loc[i];
        Var ms_spatial = w_local_ad[loc_i] + w_regional_ad[loc_i];
        if (data.multiscale_gp_data.shared) {
            eta_num = eta_num + ms_spatial;
            eta_denom = eta_denom + ms_spatial;
        } else {
            eta_num = eta_num + ms_spatial;
        }

        // Compute likelihood based on model type
        Var ll_i(tape, 0.0);

        if (data.legacy.model_type == ModelType::BINOMIAL) {
            Var p = inv_logit(eta_num);
            ll_i = tulpa::math::log_lik_binomial(data.legacy.y_num[i], data.legacy.y_denom[i], p);
        } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = tulpa::math::log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num) +
                   tulpa::math::log_lik_negbin(data.legacy.y_denom[i], mu_denom, phi_denom);
        } else {  // POISSON_GAMMA
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = tulpa::math::log_lik_poisson(data.legacy.y_num[i], mu_num) +
                   tulpa::math::log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);
        }

        log_post = log_post + ll_i;
    }

    // Backward pass
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ad);

    // TapeScope destructor handles cleanup
}

// =====================================================================
// GP + Temporal gradient (autodiff, combines GP and temporal effects)
// =====================================================================

void compute_gradient_gp_plus_temporal_autodiff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad
) {
    using namespace tulpa::ad;
    using namespace tulpa::math;

    TapeScope tape_scope;
    Tape* tape = tape_scope.tape;

    int n_params = params.size();
    grad.assign(n_params, 0.0);

    std::vector<Var> params_ad = make_vars(tape, params);

    auto [log_post, sigma_re, phi_num, phi_denom] = add_common_priors_ad(tape, params_ad, data, layout);

    // =========================================================================
    // Temporal priors
    // =========================================================================
    Var tau_temporal(tape, 1.0);
    Var rho_ar1(tape, 0.0);
    std::vector<std::vector<Var>> phi_temporal_ad;  // [group][time]

    if (layout.has_temporal) {
        Var log_tau_temporal = params_ad[layout.log_tau_temporal_idx];
        tau_temporal = safe_exp(log_tau_temporal);

        // tau ~ Gamma(shape, rate) with Jacobian
        log_post = log_post + (data.tau_temporal_shape - 1.0) * log_tau_temporal
                            - data.tau_temporal_rate * tau_temporal
                            + log_tau_temporal;

        // AR1: also estimate rho
        if (layout.is_ar1) {
            Var logit_rho = params_ad[layout.logit_rho_ar1_idx];
            rho_ar1 = 1.0 / (1.0 + safe_exp(-logit_rho));

            // rho ~ Uniform(0,1) prior with logit Jacobian
            log_post = log_post + safe_log(rho_ar1) + safe_log(1.0 - rho_ar1);
        }

        // Extract temporal effects for each group
        int T = data.n_times;
        phi_temporal_ad.resize(data.n_temporal_groups);
        for (int g = 0; g < data.n_temporal_groups; g++) {
            phi_temporal_ad[g].resize(T);
            for (int t = 0; t < T; t++) {
                phi_temporal_ad[g][t] = params_ad[layout.temporal_start + g * T + t];
            }

            // Temporal prior
            Var temporal_prior = tulpa_temporal::temporal_log_prior_t(
                phi_temporal_ad[g], data.temporal_type, tau_temporal, rho_ar1, data.temporal_cyclic);
            log_post = log_post + temporal_prior;
        }
    }

    // =========================================================================
    // GP priors and NNGP likelihood
    // =========================================================================
    std::vector<Var> gp_w_ad;
    Var sigma2_gp(tape, 1.0);
    Var phi_gp(tape, 0.1);

    if (layout.is_gp && data.has_gp) {
        Var log_sigma2_gp = params_ad[layout.log_sigma2_gp_idx];
        Var log_phi_gp = params_ad[layout.log_phi_gp_idx];
        sigma2_gp = safe_exp(log_sigma2_gp);
        phi_gp = safe_exp(log_phi_gp);

        // PC prior on sigma2 (penalizes large variance)
        log_post = log_post + tulpa_gp::log_prior_sigma2_pc_t(
            sigma2_gp, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        log_post = log_post + log_sigma2_gp;  // Jacobian

        // Uniform prior on phi within bounds
        log_post = log_post + tulpa_gp::log_prior_phi_uniform_t(
            phi_gp, data.gp_phi_prior_lower, data.gp_phi_prior_upper);
        log_post = log_post + log_phi_gp;  // Jacobian

        // Extract GP spatial effects
        int N_gp = data.gp_data.n_obs;
        gp_w_ad.resize(N_gp);
        for (int i = 0; i < N_gp; i++) {
            gp_w_ad[i] = params_ad[layout.gp_w_start + i];
        }

        // NNGP log-likelihood using templated function
        Var gp_ll = tulpa_gp::gp_nngp_log_lik_t(gp_w_ad, sigma2_gp, phi_gp, data.gp_data);
        log_post = log_post + gp_ll;
    }

    // =========================================================================
    // Data likelihood
    // =========================================================================
    std::vector<Var> beta_num_ad(data.legacy.p_num);
    std::vector<Var> beta_denom_ad(data.legacy.p_denom);
    for (int j = 0; j < data.legacy.p_num; j++) {
        beta_num_ad[j] = params_ad[layout.legacy.beta_num_start + j];
    }
    for (int j = 0; j < data.legacy.p_denom; j++) {
        beta_denom_ad[j] = params_ad[layout.legacy.beta_denom_start + j];
    }

    int T = data.n_times;

    for (int i = 0; i < data.N; i++) {
        // Linear predictors
        Var eta_num(tape, 0.0);
        Var eta_denom(tape, 0.0);

        for (int j = 0; j < data.legacy.p_num; j++) {
            eta_num = eta_num + data.legacy.X_num_flat[i * data.legacy.p_num + j] * beta_num_ad[j];
        }
        for (int j = 0; j < data.legacy.p_denom; j++) {
            eta_denom = eta_denom + data.legacy.X_denom_flat[i * data.legacy.p_denom + j] * beta_denom_ad[j];
        }

        // Add random effects (shared)
        if (layout.has_re && data.re_group[i] > 0) {
            int g = data.re_group[i] - 1;
            Var re_g = params_ad[layout.re_start + g];
            Var re_eff = (data.re_parameterization == 1) ? sigma_re * re_g : re_g;
            eta_num = eta_num + re_eff;
            eta_denom = eta_denom + re_eff;
        }

        // Add temporal effect
        if (layout.has_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;  // 0-based
            int g = data.temporal_group_idx[i] - 1; // 0-based
            if (g >= 0 && g < (int)phi_temporal_ad.size() &&
                t >= 0 && t < (int)phi_temporal_ad[g].size()) {
                Var temporal_effect = phi_temporal_ad[g][t];
                if (data.temporal_shared) {
                    eta_num = eta_num + temporal_effect;
                    eta_denom = eta_denom + temporal_effect;
                } else {
                    eta_num = eta_num + temporal_effect;
                }
            }
        }

        // Add GP spatial effect (map observation to unique location)
        if (layout.is_gp && data.has_gp && !gp_w_ad.empty()) {
            int loc_i = data.gp_data.obs_to_loc[i];
            Var gp_effect = gp_w_ad[loc_i];
            if (data.gp_data.shared) {
                eta_num = eta_num + gp_effect;
                eta_denom = eta_denom + gp_effect;
            } else {
                eta_num = eta_num + gp_effect;
            }
        }

        // Compute likelihood based on model type
        Var ll_i(tape, 0.0);

        if (data.legacy.model_type == ModelType::BINOMIAL) {
            Var p = inv_logit(eta_num);
            ll_i = tulpa::math::log_lik_binomial(data.legacy.y_num[i], data.legacy.y_denom[i], p);
        } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = tulpa::math::log_lik_negbin(data.legacy.y_num[i], mu_num, phi_num) +
                   tulpa::math::log_lik_negbin(data.legacy.y_denom[i], mu_denom, phi_denom);
        } else {  // POISSON_GAMMA
            Var mu_num = safe_exp(eta_num);
            Var mu_denom = safe_exp(eta_denom);
            ll_i = tulpa::math::log_lik_poisson(data.legacy.y_num[i], mu_num) +
                   tulpa::math::log_lik_gamma(data.legacy.y_denom_cont[i], phi_denom, mu_denom);
        }

        log_post = log_post + ll_i;
    }

    // Backward pass
    log_post.backward();

    // Extract gradients
    grad = get_adjoints(params_ad);

    // TapeScope destructor handles cleanup
}

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

  // Temporal prior on tau (Gamma) and rho (Beta) — GMRF only
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

  // Temporal likelihood gradients (scatter to temporal buffer — GMRF or GP)
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

// =====================================================================
// Multiscale temporal gradient (hand-coded, rows 15, 45, 75)
// Uses analytical gradients from hmc_multiscale_temporal_grad.h
// Supports optional ICAR/BYM2 spatial
// =====================================================================

void compute_gradient_ms_temporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- Spatial parameters (ICAR/BYM2 if present) ---
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0;
    double rho_bym2 = 0.5;
    const double* theta_bym2 = nullptr;
    if (layout.has_spatial) {
        if (!layout.is_bym2) tau_spatial = std::exp(params[layout.log_tau_spatial_idx]);
        spatial_phi = &params[layout.spatial_start];
        if (layout.is_bym2) {
            double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
            double logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
            sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
            theta_bym2 = &params[layout.theta_bym2_start];
        }
    }

    // --- Multiscale temporal parameters ---
    const auto& mst = data.multiscale_temporal_data;
    int n_trend = layout.trend_end - layout.trend_start;
    int n_seasonal = layout.seasonal_end - layout.seasonal_start;
    int n_short = layout.short_term_end - layout.short_term_start;

    const double* trend = (n_trend > 0) ? &params[layout.trend_start] : nullptr;
    const double* seasonal = (n_seasonal > 0) ? &params[layout.seasonal_start] : nullptr;
    const double* short_term = (n_short > 0) ? &params[layout.short_term_start] : nullptr;

    double sigma2_trend = (n_trend > 0) ? std::exp(params[layout.log_sigma2_trend_idx]) : 1.0;
    double sigma2_seasonal = (n_seasonal > 0) ? std::exp(params[layout.log_sigma2_seasonal_idx]) : 1.0;
    double sigma2_short = (n_short > 0) ? std::exp(params[layout.log_sigma2_short_idx]) : 1.0;
    double rho_short = 0.5;
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        double logit_rho = params[layout.logit_rho_short_idx];
        double u = 1.0 / (1.0 + std::exp(-logit_rho));
        rho_short = 2.0 * u - 1.0;
    }

    // --- Shared base priors + feature-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // Spatial prior gradients (ICAR/BYM2)
    if (layout.has_spatial && !layout.is_bym2) {
        grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0) - data.tau_spatial_rate * tau_spatial + 1.0;
    }
    if (layout.is_bym2) {
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        double ratio = sigma_total / data.sigma_re_scale;
        grad[layout.log_sigma_bym2_idx] = -2.0 * ratio * ratio / (1.0 + ratio * ratio) + 1.0;
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }

    // PC priors on multiscale temporal variances
    if (n_trend > 0) {
        grad[layout.log_sigma2_trend_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
            sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha) + 1.0;
    }
    if (n_seasonal > 0) {
        grad[layout.log_sigma2_seasonal_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
            sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha) + 1.0;
    }
    if (n_short > 0) {
        grad[layout.log_sigma2_short_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
            sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha) + 1.0;
    }
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        double u = (rho_short + 1.0) / 2.0;
        grad[layout.logit_rho_short_idx] = 2.0 * (1.0 - 2.0 * u);
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // Pre-assemble temporal effect per time point (avoids per-obs modulo)
    const int n_times = mst.n_times;
    std::vector<double> ms_effect_by_time(n_times, 0.0);
    for (int t = 0; t < n_times; t++) {
        if (trend != nullptr && t < n_trend) ms_effect_by_time[t] += trend[t];
        if (seasonal != nullptr && mst.seasonal_period > 0) {
            ms_effect_by_time[t] += seasonal[t % mst.seasonal_period];
        }
        if (short_term != nullptr && t < n_short) ms_effect_by_time[t] += short_term[t];
    }

    // Feature-specific eta: spatial + multiscale temporal
    std::vector<double> grad_trend_lik(n_trend, 0.0);
    std::vector<double> grad_seasonal_lik(n_seasonal, 0.0);
    std::vector<double> grad_short_lik(n_short, 0.0);
    std::vector<double> grad_spatial_lik;
    if (layout.has_spatial) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_theta_lik;
    if (layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);

    std::vector<int> obs_s_unit(pre.N, -1);
    std::vector<int> obs_t_idx(pre.N, -1);
    for (int i = 0; i < pre.N; i++) {
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            obs_s_unit[i] = s_unit;
            double spatial_eff;
            if (layout.is_bym2) {
                spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit];
            } else {
                spatial_eff = spatial_phi[s_unit];
            }
            vec_grad_ws.eta_num[i] += spatial_eff;
            if (!pre.is_binomial) vec_grad_ws.eta_denom[i] += spatial_eff;
        }
        if (!mst.time_index.empty() && i < (int)mst.time_index.size() && mst.time_index[i] > 0) {
            int t_idx = mst.time_index[i] - 1;
            obs_t_idx[i] = t_idx;
            vec_grad_ws.eta_num[i] += ms_effect_by_time[t_idx];
            if (!pre.is_binomial && mst.shared) vec_grad_ws.eta_denom[i] += ms_effect_by_time[t_idx];
        }
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // Feature-specific residual scatter (spatial + multiscale temporal)
    for (int i = 0; i < pre.N; i++) {
        double dLL_num = vec_grad_ws.resid_num[i];
        double dLL_denom = vec_grad_ws.resid_denom[i];
        double dLL_shared = dLL_num + dLL_denom;
        int s_unit = obs_s_unit[i];
        if (layout.has_spatial && s_unit >= 0) {
            grad_spatial_lik[s_unit] += dLL_shared;
            if (layout.is_bym2) grad_theta_lik[s_unit] += dLL_shared;
        }
        int t_idx = obs_t_idx[i];
        if (t_idx >= 0) {
            double dLL_temporal = mst.shared ? dLL_shared : dLL_num;
            if (trend != nullptr && t_idx < n_trend) grad_trend_lik[t_idx] += dLL_temporal;
            if (seasonal != nullptr && mst.seasonal_period > 0) {
                int s_idx = t_idx % mst.seasonal_period;
                if (s_idx < n_seasonal) grad_seasonal_lik[s_idx] += dLL_temporal;
            }
            if (short_term != nullptr && t_idx < n_short) grad_short_lik[t_idx] += dLL_temporal;
        }
    }

    // Spatial GMRF prior gradients (ICAR/BYM2)
    if (layout.has_spatial) {
        spatial_gmrf_prior_grad(data, layout, spatial_phi, tau_spatial,
                                sigma_s_bym2, sigma_u_bym2, rho_bym2, theta_bym2,
                                grad_spatial_lik.data(), grad_theta_lik.data(), grad.data());
    }

    // Multiscale temporal GMRF prior gradients
    tulpa_temporal_grad::MultiscaleTemporalGradients ms_grads;
    tulpa_temporal_grad::multiscale_temporal_prior_gradients(
        trend, n_trend,
        seasonal, n_seasonal,
        short_term, n_short,
        sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
        mst, ms_grads);

    for (int t = 0; t < n_trend; t++) grad[layout.trend_start + t] = grad_trend_lik[t] + ms_grads.grad_trend[t];
    for (int t = 0; t < n_seasonal; t++) grad[layout.seasonal_start + t] = grad_seasonal_lik[t] + ms_grads.grad_seasonal[t];
    for (int t = 0; t < n_short; t++) grad[layout.short_term_start + t] = grad_short_lik[t] + ms_grads.grad_short_term[t];
    if (n_trend > 0) grad[layout.log_sigma2_trend_idx] += ms_grads.grad_log_sigma2_trend;
    if (n_seasonal > 0) grad[layout.log_sigma2_seasonal_idx] += ms_grads.grad_log_sigma2_seasonal;
    if (n_short > 0) grad[layout.log_sigma2_short_idx] += ms_grads.grad_log_sigma2_short;
    if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
        grad[layout.logit_rho_short_idx] += ms_grads.grad_logit_rho_short;
    }

    // --- Shared epilogue ---
    gradient_epilogue(params, data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, grad, log_post_out);
}

// =====================================================================
// Spatiotemporal interaction gradient (hand-coded, rows 28-29, 58-59, 90-91)
// Supports Knorr-Held Type I-IV with ICAR spatial + RW1/RW2 temporal
// =====================================================================

void compute_gradient_spatiotemporal_handcoded(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    // --- Shared preamble ---
    auto pre = gradient_preamble(params, data, layout, grad, log_post_out);

    // --- Spatial parameters (ICAR/BYM2) ---
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0;
    double rho_bym2 = 0.5;
    const double* theta_bym2 = nullptr;
    if (layout.has_spatial) {
        if (!layout.is_bym2) tau_spatial = std::exp(params[layout.log_tau_spatial_idx]);
        spatial_phi = &params[layout.spatial_start];
        if (layout.is_bym2) {
            double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
            double logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
            sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
            theta_bym2 = &params[layout.theta_bym2_start];
        }
    }

    // --- Temporal parameters ---
    double tau_temporal = 0.0;
    int T_temporal = 0;
    const double* phi_temporal = nullptr;
    double rho_ar1 = 0.5;
    if (layout.has_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_temporal = layout.temporal_end - layout.temporal_start;
        phi_temporal = &params[layout.temporal_start];
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
            rho_ar1 = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx]));
        }
    }

    // --- Spatiotemporal interaction parameters ---
    const auto& st = data.spatiotemporal_data;
    int S = st.n_spatial;
    int T = st.n_times;
    int ST = st.n_params;
    double tau_st = std::exp(params[layout.log_tau_st_idx]);
    double tau_st2 = 1.0;  // Always 1.0 — single tau for all ST types

    // NC reparameterization: params store z, reconstruct delta
    const bool st_use_nc = (data.st_parameterization == 1 &&
                            st.type == STType::TYPE_IV);
    const double* z_or_delta = &params[layout.st_delta_start];
    static thread_local std::vector<double> st_delta_buf;
    const double* delta;
    double inv_scale = 1.0;
    if (st_use_nc) {
        inv_scale = 1.0 / std::sqrt(tau_st);
        st_delta_buf.resize(ST);
        for (int k = 0; k < ST; k++) st_delta_buf[k] = z_or_delta[k] * inv_scale;
        delta = st_delta_buf.data();
    } else {
        delta = z_or_delta;
    }

    // --- Shared base priors + feature-specific priors ---
    gradient_base_priors(data, layout, pre.cp, grad.data());

    // Spatial prior (ICAR: Gamma on tau)
    if (layout.has_spatial && !layout.is_bym2) {
        grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0) - data.tau_spatial_rate * tau_spatial + 1.0;
    }
    if (layout.is_bym2) {
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        double ratio = sigma_total / data.sigma_re_scale;
        grad[layout.log_sigma_bym2_idx] = -2.0 * ratio * ratio / (1.0 + ratio * ratio) + 1.0;
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }

    // Temporal prior (Gamma on tau, Beta on rho)
    if (layout.has_temporal) {
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
            grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;
        }
    }

    // ST interaction prior on tau_st (PC prior: exponential on sigma_st)
    {
        double sigma_st = 1.0 / std::sqrt(tau_st);
        double lambda = -std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U;
        grad[layout.log_tau_st_idx] = 0.5 * lambda * sigma_st + 0.5 + 1.0;
    }

    // --- Shared vectorized eta (X*beta + RE) ---
    compute_base_eta(data, layout, pre.cp, pre.is_binomial, vec_grad_ws);

    // Feature-specific eta: spatial, temporal, ST
    std::vector<double> grad_spatial_lik;
    if (layout.has_spatial) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_theta_lik;
    if (layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);
    std::vector<double> grad_temporal_lik(T_temporal, 0.0);
    std::vector<double> grad_delta_lik(ST, 0.0);

    for (int i = 0; i < pre.N; i++) {
        // Spatial
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            double spatial_eff;
            if (layout.is_bym2) {
                spatial_eff = sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit];
            } else {
                spatial_eff = spatial_phi[s_unit];
            }
            vec_grad_ws.eta_num[i] += spatial_eff;
            if (!pre.is_binomial) vec_grad_ws.eta_denom[i] += spatial_eff;
        }
        // Temporal
        if (layout.has_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * data.n_times + t;
            if (t_base >= 0 && t_base < T_temporal) {
                vec_grad_ws.eta_num[i] += phi_temporal[t_base];
                if (!pre.is_binomial && data.temporal_shared) vec_grad_ws.eta_denom[i] += phi_temporal[t_base];
            }
        }
        // Spatiotemporal interaction
        if (st.st_flat[i] > 0) {
            double st_effect = delta[st.st_flat[i] - 1];
            vec_grad_ws.eta_num[i] += st_effect;
            if (!pre.is_binomial && st.shared) vec_grad_ws.eta_denom[i] += st_effect;
        }
    }

    // --- Shared dispatch residuals + RE scatter ---
    dispatch_residuals(data, layout, pre.cp, pre.fuse_lp, pre.obs_log_lik, vec_grad_ws, grad.data());
    scatter_re_residuals(data, layout, vec_grad_ws, grad.data());

    // Feature-specific residual scatter (spatial, temporal, ST)
    for (int i = 0; i < pre.N; i++) {
        double dLL_num = vec_grad_ws.resid_num[i];
        double dLL_denom = vec_grad_ws.resid_denom[i];
        double dLL_shared = dLL_num + dLL_denom;

        // Spatial
        if (layout.has_spatial && data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            if (layout.is_bym2) { grad_spatial_lik[s_unit] += dLL_shared; grad_theta_lik[s_unit] += dLL_shared; }
            else { grad_spatial_lik[s_unit] += dLL_shared; }
        }

        // Temporal
        if (layout.has_temporal && !data.temporal_time_idx.empty() &&
            i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
            int t = data.temporal_time_idx[i] - 1;
            int g = data.temporal_group_idx[i] - 1;
            int t_base = g * data.n_times + t;
            if (t_base >= 0 && t_base < T_temporal)
                grad_temporal_lik[t_base] += data.temporal_shared ? dLL_shared : dLL_num;
        }

        // Spatiotemporal interaction
        if (st.st_flat[i] > 0) {
            int st_idx = st.st_flat[i] - 1;
            grad_delta_lik[st_idx] += st.shared ? dLL_shared : dLL_num;
        }
    }

    // Spatial GMRF prior gradients (ICAR/BYM2)
    if (layout.has_spatial) {
        spatial_gmrf_prior_grad(data, layout, spatial_phi, tau_spatial,
                                sigma_s_bym2, sigma_u_bym2, rho_bym2, theta_bym2,
                                grad_spatial_lik.data(), grad_theta_lik.data(), grad.data());
    }

    // Temporal GMRF prior gradients
    if (layout.has_temporal && T_temporal > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                                 phi_temporal, T_temporal, grad_temporal_lik.data(), grad.data());
    }

    // =========================================================================
    // Spatiotemporal interaction prior gradients (Type I-IV)
    // =========================================================================
    // delta is stored column-major: delta[s*T + t]
    // Accumulate ST delta log-prior alongside gradient to avoid recomputing
    // the expensive Kronecker quadratic form in compute_log_post.
    double st_lp_accum = 0.0;
    if (st.type == STType::TYPE_I) {
        // IID: log p = 0.5*n*log(tau) - 0.5*tau*sum(delta^2)
        double qf = 0.0;
        for (int k = 0; k < ST; k++) {
            grad[layout.st_delta_start + k] = grad_delta_lik[k] - tau_st * delta[k];
            qf += delta[k] * delta[k];
        }
        grad[layout.log_tau_st_idx] += 0.5 * ST - 0.5 * tau_st * qf;
        st_lp_accum = 0.5 * ST * std::log(tau_st) - 0.5 * tau_st * qf;

    } else if (st.type == STType::TYPE_II) {
        // Structured time per spatial unit: temporal GMRF applied to delta[s,:]
        double total_qf = 0.0;
        for (int s = 0; s < S; s++) {
            // Apply temporal stencil to delta[s*T .. s*T+T-1]
            const double* delta_s = &delta[s * T];
            if (st.temporal_type == TemporalType::RW1) {
                double qf = 0.0;
                for (int t = 0; t < T; t++) {
                    double g = 0.0;
                    if (t > 0) { g += tau_st * (delta_s[t-1] - delta_s[t]); qf += std::pow(delta_s[t] - delta_s[t-1], 2); }
                    if (t < T - 1) g += tau_st * (delta_s[t+1] - delta_s[t]);
                    grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t] + g;
                }
                total_qf += qf;
            } else if (st.temporal_type == TemporalType::RW2) {
                double qf = 0.0;
                for (int t = 0; t < T; t++) {
                    double g = 0.0;
                    if (t >= 2) g -= tau_st * (delta_s[t-2] - 2.0*delta_s[t-1] + delta_s[t]);
                    if (t >= 1 && t < T - 1) g += 2.0 * tau_st * (delta_s[t-1] - 2.0*delta_s[t] + delta_s[t+1]);
                    if (t < T - 2) g -= tau_st * (delta_s[t] - 2.0*delta_s[t+1] + delta_s[t+2]);
                    grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t] + g;
                }
                for (int t = 2; t < T; t++) qf += std::pow(delta_s[t-2] - 2.0*delta_s[t-1] + delta_s[t], 2);
                total_qf += qf;
            } else if (st.temporal_type == TemporalType::AR1) {
                // AR1 with IID fallback for ST interaction (no separate rho)
                double qf = 0.0;
                for (int t = 0; t < T; t++) {
                    grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t] - tau_st * delta_s[t];
                    qf += delta_s[t] * delta_s[t];
                }
                total_qf += qf;
            } else {
                // Fallback: write likelihood gradient only
                for (int t = 0; t < T; t++)
                    grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t];
            }
        }
        int rank_per_unit = (st.temporal_type == TemporalType::RW1) ? (T - 1) :
                            (st.temporal_type == TemporalType::RW2) ? (T - 2) : T;
        grad[layout.log_tau_st_idx] += 0.5 * S * rank_per_unit - 0.5 * tau_st * total_qf;
        st_lp_accum = 0.5 * S * rank_per_unit * std::log(tau_st) - 0.5 * tau_st * total_qf;

    } else if (st.type == STType::TYPE_III) {
        // Structured space per time point: ICAR applied to delta[:,t]
        double total_qf = 0.0;
        for (int t = 0; t < T; t++) {
            // Apply ICAR stencil to delta[0*T+t, 1*T+t, ..., (S-1)*T+t]
            for (int s = 0; s < S; s++) {
                double icar_grad = 0.0;
                for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                    int j = st.adj_col_idx[idx] - 1;
                    icar_grad += tau_st * (delta[j * T + t] - delta[s * T + t]);
                }
                grad[layout.st_delta_start + s * T + t] = grad_delta_lik[s * T + t] + icar_grad;
            }
            // Compute ICAR quadratic form for this time slice
            for (int s = 0; s < S; s++) {
                for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                    int j = st.adj_col_idx[idx] - 1;
                    if (j > s) {
                        double diff = delta[s * T + t] - delta[j * T + t];
                        total_qf += diff * diff;
                    }
                }
            }
        }
        int rank_spatial = S - 1;
        grad[layout.log_tau_st_idx] += 0.5 * T * rank_spatial - 0.5 * tau_st * total_qf;
        st_lp_accum = 0.5 * T * rank_spatial * std::log(tau_st) - 0.5 * tau_st * total_qf;

    } else if (st.type == STType::TYPE_IV) {
        // Kronecker: Q_delta = Q_s ⊗ Q_t
        // For NC: apply stencil to z (not delta), without tau factor
        const double* stencil_input = st_use_nc ? z_or_delta : delta;

        // Step 1: Apply temporal stencil: v[s,t] = (Q_t * input[s,:])_t
        static thread_local std::vector<double> v;
        v.assign(S * T, 0.0);
        if (st.temporal_type == TemporalType::RW1) {
            for (int s = 0; s < S; s++) {
                for (int t = 0; t < T; t++) {
                    double qt_val = 0.0;
                    int n_t_neigh = 0;
                    if (t > 0) { qt_val -= stencil_input[s * T + t - 1]; n_t_neigh++; }
                    if (t < T - 1) { qt_val -= stencil_input[s * T + t + 1]; n_t_neigh++; }
                    qt_val += n_t_neigh * stencil_input[s * T + t];
                    v[s * T + t] = qt_val;
                }
            }
        } else if (st.temporal_type == TemporalType::RW2) {
            // Unrolled RW2 stencil: Q_t is the second-difference precision matrix.
            // For each t, Q_t[t,:] * x gives a linear combination of second differences.
            // Instead of irregular max/min bounds, handle boundary cases explicitly.
            for (int s = 0; s < S; s++) {
                const double* d_s = &stencil_input[s * T];
                double* v_s = &v[s * T];
                if (T >= 3) {
                    // Precompute second differences d2[k] = d_s[k] - 2*d_s[k+1] + d_s[k+2]
                    // Reused by multiple t values, avoids redundant subtraction
                    const int n_d2 = T - 2;
                    // Use stack for small T (typical: T=20), heap only if huge
                    double d2_stack[64];
                    double* d2 = (n_d2 <= 64) ? d2_stack : new double[n_d2];
                    for (int k = 0; k < n_d2; k++) {
                        d2[k] = d_s[k] - 2.0 * d_s[k + 1] + d_s[k + 2];
                    }

                    // t=0: only k=0 contributes (pos=0, coef=1)
                    v_s[0] = d2[0];
                    // t=1: k=0 (pos=1, coef=-2) + k=1 (pos=0, coef=1) if T>=4
                    v_s[1] = -2.0 * d2[0];
                    if (n_d2 > 1) v_s[1] += d2[1];
                    // Interior: t=2..T-3, all three contributions present
                    for (int t = 2; t < T - 2; t++) {
                        v_s[t] = d2[t - 2] - 2.0 * d2[t - 1] + d2[t];
                    }
                    // t=T-2: k=T-4 (pos=2, coef=1) + k=T-3 (pos=1, coef=-2)
                    if (T >= 4) {
                        v_s[T - 2] = d2[n_d2 - 2] - 2.0 * d2[n_d2 - 1];
                    } else {
                        // T==3: t=1 already handled, t=T-2=1 is same slot
                        // Just set directly
                        v_s[T - 2] = -2.0 * d2[0];
                    }
                    // t=T-1: k=T-3 (pos=2, coef=1)
                    v_s[T - 1] = d2[n_d2 - 1];

                    if (n_d2 > 64) delete[] d2;
                } else {
                    // T < 3: no second differences possible, v stays zero
                }
            }
        }

        // Step 2: Apply spatial ICAR stencil to v: (Q_s ⊗ Q_t) * input
        double total_qf = 0.0;
        for (int s = 0; s < S; s++) {
            for (int t = 0; t < T; t++) {
                double qs_v = 0.0;
                for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                    int j = st.adj_col_idx[idx] - 1;
                    qs_v -= v[j * T + t];
                }
                int n_neigh = st.adj_row_ptr[s + 1] - st.adj_row_ptr[s];
                qs_v += n_neigh * v[s * T + t];

                if (st_use_nc) {
                    // NC: grad_z = -(Q z)_k + dL/d(delta_k) / sqrt(tau)
                    grad[layout.st_delta_start + s * T + t] =
                        grad_delta_lik[s * T + t] * inv_scale - qs_v;
                } else {
                    // Centered: grad_delta = dL/d(delta_k) - tau * (Q delta)_k
                    grad[layout.st_delta_start + s * T + t] =
                        grad_delta_lik[s * T + t] - tau_st * qs_v;
                }
                total_qf += stencil_input[s * T + t] * qs_v;
            }
        }

        int rank_space = S - 1;
        int rank_time = (st.temporal_type == TemporalType::RW1) ? (T - 1) :
                        (st.temporal_type == TemporalType::RW2) ? (T - 2) : T;
        if (st.temporal_cyclic) rank_time = T;
        int total_rank = rank_space * rank_time;

        if (st_use_nc) {
            // NC tau gradient: 0.5*(rank - ST) from combined normalization+Jacobian
            // plus likelihood chain rule: -0.5 * dot(grad_delta_lik, delta)
            double lik_tau_grad = 0.0;
            for (int k = 0; k < ST; k++) {
                lik_tau_grad += grad_delta_lik[k] * delta[k];
            }
            grad[layout.log_tau_st_idx] += 0.5 * (total_rank - ST) - 0.5 * lik_tau_grad;
            // NC log-prior: -0.5*qf + 0.5*(rank-ST)*log(tau)
            st_lp_accum = -0.5 * total_qf + 0.5 * (total_rank - ST) * std::log(tau_st);
        } else {
            // Centered tau gradient: 0.5*rank - 0.5*tau*qf
            grad[layout.log_tau_st_idx] += 0.5 * total_rank - 0.5 * tau_st * total_qf;
            st_lp_accum = 0.5 * total_rank * std::log(tau_st) - 0.5 * tau_st * total_qf;
        }
    }

    // Sum-to-zero penalty gradients (on reconstructed delta)
    // For NC: chain rule d/dz = inv_scale * d/d(delta), and penalty also contributes to tau
    {
        double lambda_stz = 0.001;
        // For NC, the penalty -0.5*lambda*sum(delta)^2 where delta=z*inv_scale
        // d/dz = -lambda * sum(delta) * inv_scale
        // d/d(log_tau) = -lambda * sum(delta) * d(delta)/d(log_tau) summed
        //              = -lambda * sum(delta) * (-0.5 * delta[k]) for each k in sum
        //              = but simpler: penalty = -0.5*lambda*(inv_scale * sum(z))^2
        //              = -0.5*lambda*inv_scale^2 * sum(z)^2
        // d/dz_k = -lambda*inv_scale^2 * sum(z)
        // This equals -lambda*inv_scale * sum(delta) = centered penalty * inv_scale
        double stz_scale = st_use_nc ? inv_scale : 1.0;

        // Pre-compute row sums (over space) and col sums (over time) in a single pass
        // This replaces 4 separate double-loops with one pass + two apply loops
        static thread_local std::vector<double> row_sums_buf, col_sums_buf;
        row_sums_buf.assign(T, 0.0);
        col_sums_buf.assign(S, 0.0);
        for (int s = 0; s < S; s++) {
            for (int t = 0; t < T; t++) {
                double d = delta[s * T + t];
                row_sums_buf[t] += d;
                col_sums_buf[s] += d;
            }
        }

        // Apply delta gradients + accumulate NC tau gradient in one pass
        double tau_stz_grad = 0.0;
        for (int s = 0; s < S; s++) {
            for (int t = 0; t < T; t++) {
                grad[layout.st_delta_start + s * T + t] -=
                    lambda_stz * stz_scale * (row_sums_buf[t] + col_sums_buf[s]);
            }
        }
        if (st_use_nc) {
            for (int t = 0; t < T; t++) tau_stz_grad += row_sums_buf[t] * row_sums_buf[t];
            for (int s = 0; s < S; s++) tau_stz_grad += col_sums_buf[s] * col_sums_buf[s];
            tau_stz_grad *= 0.5 * lambda_stz;
            grad[layout.log_tau_st_idx] += tau_stz_grad;
        }

        // Accumulate sum-to-zero penalty into ST log-prior
        for (int t = 0; t < T; t++) st_lp_accum -= 0.5 * lambda_stz * row_sums_buf[t] * row_sums_buf[t];
        for (int s = 0; s < S; s++) st_lp_accum -= 0.5 * lambda_stz * col_sums_buf[s] * col_sums_buf[s];
    }

    // --- Custom epilogue (spatiotemporal has fused ST prior accumulation) ---
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), pre.cp.sigma_re);

    if (pre.fuse_lp) *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true, &st_lp_accum) + pre.obs_log_lik;
}

// =====================================================================
// Composite hand-coded gradient: handles ANY combination of features.
// This is the catch-all H-mode function for exotic multi-feature combos
// that no specialized gradient function covers (e.g., HSGP+TVC, SVC+RW1,
// latent+spatial, etc.). Slower than specialized functions but much faster
// than A_r/N fallback.
//
// Architecture: single observation loop with conditional feature blocks.
// Each feature contributes additively to eta; gradient scattering is
// independent per feature. Structural/prior gradients computed after the
// observation loop.
// =====================================================================

void compute_gradient_composite(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
) {
    const bool fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) *log_post_out = compute_log_post(params, data, layout);
    double obs_log_lik = 0.0;
    int n_params = params.size();
    grad.assign(n_params, 0.0);

    const int N = data.N;
    const bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                              data.legacy.model_type == ModelType::BETA_BINOMIAL);

    // =========================================================================
    // Phase 1: Extract all parameters
    // =========================================================================
    auto cp = extract_common_params(params, layout);
    const double* beta_num = cp.beta_num;
    const double* beta_denom = cp.beta_denom;
    double sigma_re = cp.sigma_re;
    const double* re = cp.re;
    double phi_num = cp.phi_num;
    double phi_denom = cp.phi_denom;

    // --- ZI/OI parameters ---
    const double* beta_zi = nullptr;
    const double* beta_oi = nullptr;
    if (layout.has_zi && data.p_zi > 0)
        beta_zi = &params[layout.beta_zi_start];
    if (layout.has_oi && data.p_oi > 0)
        beta_oi = &params[layout.beta_oi_start];

    // --- Spatial (ICAR/BYM2/pCAR) ---
    double tau_spatial = 0.0;
    const double* spatial_phi = nullptr;
    double sigma_s_bym2 = 0.0, sigma_u_bym2 = 0.0, rho_bym2 = 0.5;
    const double* theta_bym2 = nullptr;
    const bool has_icar_bym2 = layout.has_spatial && !layout.is_hsgp && !layout.is_gp &&
                                !layout.is_multiscale_gp && !layout.has_svc;
    if (has_icar_bym2) {
        if (!layout.is_bym2) tau_spatial = std::exp(params[layout.log_tau_spatial_idx]);
        spatial_phi = &params[layout.spatial_start];
        if (layout.is_bym2) {
            double sigma_total = std::exp(params[layout.log_sigma_bym2_idx]);
            double logit_rho = params[layout.logit_rho_bym2_idx];
            rho_bym2 = 1.0 / (1.0 + std::exp(-logit_rho));
            sigma_s_bym2 = sigma_total * std::sqrt(rho_bym2);
            sigma_u_bym2 = sigma_total * std::sqrt(1.0 - rho_bym2);
            theta_bym2 = &params[layout.theta_bym2_start];
        }
    }

    // --- GP (NNGP) spatial ---
    const bool has_gp_spatial = layout.is_gp && data.has_gp && !data.gp_collapsed;
    double sigma2_gp_c = 1.0, phi_gp_c = 1.0;
    int N_gp_c = 0;
    bool use_nc_gp = false;
    static thread_local tulpa_gp::NNGPNCWorkspace nc_ws_gp_c;
    static thread_local std::vector<double> gp_w_c;
    if (has_gp_spatial) {
        N_gp_c = data.gp_data.n_obs;
        sigma2_gp_c = std::exp(params[layout.log_sigma2_gp_idx]);
        phi_gp_c = std::exp(params[layout.log_phi_gp_idx]);
        use_nc_gp = (data.gp_parameterization == 1);
        gp_w_c.resize(N_gp_c);
        if (use_nc_gp) {
            const double* z_gp = &params[layout.gp_w_start];
            tulpa_gp::nngp_nc_forward(z_gp, sigma2_gp_c, phi_gp_c, data.gp_data, nc_ws_gp_c);
            std::memcpy(gp_w_c.data(), nc_ws_gp_c.w.data(), N_gp_c * sizeof(double));
        } else {
            for (int i = 0; i < N_gp_c; i++) gp_w_c[i] = params[layout.gp_w_start + i];
        }
    }

    // --- HSGP spatial ---
    static thread_local tulpa_hsgp::HSGPWorkspace hsgp_ws;
    double hsgp_sigma2 = 0.0, hsgp_lengthscale = 0.0;
    const double* hsgp_beta_ptr = nullptr;
    if (layout.is_hsgp && data.has_hsgp) {
        hsgp_sigma2 = std::exp(params[layout.log_sigma2_hsgp_idx]);
        hsgp_lengthscale = std::exp(params[layout.log_lengthscale_hsgp_idx]);
        hsgp_beta_ptr = &params[layout.hsgp_beta_start];
        hsgp_ws.init(data.hsgp_data.n_obs, data.hsgp_data.m_total);
        tulpa_hsgp::hsgp_evaluate_ws(hsgp_beta_ptr, hsgp_sigma2, hsgp_lengthscale,
                                       data.hsgp_data, hsgp_ws);
    }

    // --- Temporal (RW1/RW2/AR1 GMRF) ---
    double tau_temporal = 0.0;
    int T_temporal = 0;
    const double* phi_temporal = nullptr;
    double rho_ar1 = 0.5;
    const bool has_gmrf_temporal = layout.has_temporal && !layout.is_temporal_gp &&
                                   !layout.has_multiscale_temporal && !layout.has_tvc;
    if (has_gmrf_temporal) {
        tau_temporal = std::exp(params[layout.log_tau_temporal_idx]);
        T_temporal = layout.temporal_end - layout.temporal_start;
        phi_temporal = &params[layout.temporal_start];
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0) {
            rho_ar1 = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_ar1_idx]));
        }
    }

    // --- Temporal GP ---
    const bool has_temporal_gp = layout.is_temporal_gp && layout.has_temporal;
    int T_gp = 0;
    const double* z_temporal_gp = nullptr;
    static thread_local tulpa_temporal_gp::TemporalGPNCWorkspace nc_ws_composite;
    static thread_local std::vector<double> temporal_gp_f;
    double sigma2_tgp_comp = 0.0, phi_tgp_comp = 0.0;
    double phi_lower_tgp = 0.0, phi_upper_tgp = 0.0;
    bool use_nc_tgp = false;
    int n_groups_gp = 0;
    if (has_temporal_gp) {
        T_gp = data.n_times;
        n_groups_gp = data.n_temporal_groups;
        z_temporal_gp = &params[layout.temporal_start];
        int total_gp = n_groups_gp * T_gp;
        temporal_gp_f.resize(total_gp);

        sigma2_tgp_comp = std::exp(params[layout.log_sigma2_temporal_gp_idx]);
        double phi_gp_raw = params[layout.logit_phi_temporal_gp_idx];
        phi_lower_tgp = data.temporal_gp_phi_prior_lower;
        phi_upper_tgp = data.temporal_gp_phi_prior_upper;
        double phi_range = phi_upper_tgp - phi_lower_tgp;
        phi_tgp_comp = phi_lower_tgp + phi_range / (1.0 + std::exp(-phi_gp_raw));

        use_nc_tgp = (data.temporal_gp_parameterization == 1);
        if (use_nc_tgp) {
            nc_ws_composite.init(T_gp, n_groups_gp);
            tulpa_temporal_gp::temporal_gp_nc_forward(
                z_temporal_gp, T_gp, n_groups_gp, sigma2_tgp_comp, phi_tgp_comp,
                data.temporal_gp_data.time_values, nc_ws_composite);
            for (int k = 0; k < total_gp; k++) temporal_gp_f[k] = nc_ws_composite.f[k];
        } else {
            for (int k = 0; k < total_gp; k++) temporal_gp_f[k] = z_temporal_gp[k];
        }
    }

    // --- TVC ---
    static thread_local std::vector<double> tvc_eta_precomp;
    int n_tvc = 0, n_tvc_times = 0, n_tvc_groups = 1, n_w = 0;
    static thread_local std::vector<double> tvc_tau_buf, tvc_rho_buf, tvc_w_flat_buf;
    if (layout.has_tvc && data.has_tvc) {
        n_tvc = data.tvc_data.n_tvc;
        n_tvc_times = data.tvc_data.n_times;
        n_tvc_groups = data.tvc_data.n_groups;
        n_w = n_tvc_groups * n_tvc * n_tvc_times;

        tvc_tau_buf.resize(n_tvc);
        tvc_rho_buf.resize(n_tvc);
        tvc_w_flat_buf.resize(n_w);

        for (int j = 0; j < n_tvc; j++) {
            tvc_tau_buf[j] = std::exp(params[layout.log_tau_tvc_start + j]);
            if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
                double logit_rho = params[layout.logit_rho_tvc_start + j];
                double u = 1.0 / (1.0 + std::exp(-logit_rho));
                tvc_rho_buf[j] = 2.0 * u - 1.0;
            } else {
                tvc_rho_buf[j] = 0.0;
            }
        }
        for (int k = 0; k < n_w; k++) tvc_w_flat_buf[k] = params[layout.tvc_w_start + k];

        // Precompute TVC eta contribution
        tvc_eta_precomp.assign(N, 0.0);
        for (int i = 0; i < N; i++) {
            int t = data.tvc_data.time_index[i] - 1;
            int g = data.tvc_data.group_index[i] - 1;
            for (int j = 0; j < n_tvc; j++) {
                int w_idx = (g * n_tvc + j) * n_tvc_times + t;
                tvc_eta_precomp[i] += data.tvc_data.X_tvc[i * n_tvc + j] * tvc_w_flat_buf[w_idx];
            }
        }
    }

    // --- MSGP-HSGP (Multi-scale HSGP) spatial ---
    static thread_local tulpa_hsgp::HSGPWorkspace msgp_ws_local_c, msgp_ws_regional_c;
    double msgp_sigma2_local = 0, msgp_sigma2_regional = 0;
    double msgp_ls_local = 0, msgp_ls_regional = 0;
    const double* msgp_beta_local = nullptr;
    const double* msgp_beta_regional = nullptr;
    int msgp_m_total = 0;
    const bool has_msgp_hsgp = layout.is_multiscale_gp && data.has_multiscale_gp && data.msgp_is_hsgp;
    if (has_msgp_hsgp) {
        msgp_sigma2_local = std::exp(params[layout.log_sigma2_gp_local_idx]);
        msgp_ls_local = std::exp(params[layout.log_phi_gp_local_idx]);
        msgp_sigma2_regional = std::exp(params[layout.log_sigma2_gp_regional_idx]);
        msgp_ls_regional = std::exp(params[layout.log_phi_gp_regional_idx]);
        msgp_m_total = data.msgp_hsgp_data.m_total;
        msgp_beta_local = &params[layout.gp_local_start];
        msgp_beta_regional = &params[layout.gp_regional_start];
        msgp_ws_local_c.init(data.N, msgp_m_total);
        msgp_ws_regional_c.init(data.N, msgp_m_total);
        tulpa_hsgp::hsgp_evaluate_ws(msgp_beta_local, msgp_sigma2_local, msgp_ls_local,
                                       data.msgp_hsgp_data, msgp_ws_local_c);
        tulpa_hsgp::hsgp_evaluate_ws(msgp_beta_regional, msgp_sigma2_regional, msgp_ls_regional,
                                       data.msgp_hsgp_data, msgp_ws_regional_c);
    }

    // --- SVC (HSGP) ---
    static thread_local tulpa_hsgp::HSGPWorkspace svc_hsgp_ws;
    int n_svc = 0, svc_m_total = 0;
    static thread_local std::vector<double> svc_eta_precomp, svc_f_all;
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        n_svc = data.svc_data.n_svc;
        svc_m_total = data.svc_hsgp_data.m_total;
        svc_hsgp_ws.init(data.svc_hsgp_data.n_obs, svc_m_total);
        svc_eta_precomp.assign(N, 0.0);
        svc_f_all.resize(n_svc * N);

        for (int j = 0; j < n_svc; j++) {
            double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
            double lengthscale_j = std::exp(params[layout.log_phi_svc_start + j]);
            const double* beta_j = &params[layout.svc_w_start + j * svc_m_total];

            tulpa_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                           data.svc_hsgp_data, svc_hsgp_ws);

            for (int i = 0; i < N; i++) {
                double f_ji = svc_hsgp_ws.hsgp_f[i];
                svc_f_all[j * N + i] = f_ji;
                double x_ij = data.svc_data.X_svc[i * n_svc + j];
                svc_eta_precomp[i] += x_ij * f_ji;
            }
        }
    }

    // --- Latent factors ---
    int K_latent = 0;
    static thread_local std::vector<double> latent_eta_precomp;
    static thread_local std::vector<double> factors_constrained, sigma_latent_vec;
    if (layout.has_latent && data.latent_n_factors > 0) {
        K_latent = data.latent_n_factors;
        sigma_latent_vec.resize(K_latent);
        for (int k = 0; k < K_latent; k++)
            sigma_latent_vec[k] = std::exp(params[layout.log_sigma_latent_start + k]);

        int n_factor_params = N * K_latent;
        factors_constrained.resize(n_factor_params);
        for (int j = 0; j < n_factor_params; j++)
            factors_constrained[j] = params[layout.latent_factor_start + j];

        // Apply sum-to-zero constraint
        if (data.latent_constraint == 0) {
            for (int k = 0; k < K_latent; k++) {
                double sum = 0.0;
                for (int i = 0; i < N; i++) sum += factors_constrained[i * K_latent + k];
                double mean = sum / N;
                for (int i = 0; i < N; i++) factors_constrained[i * K_latent + k] -= mean;
            }
        }

        latent_eta_precomp.assign(N, 0.0);
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K_latent; k++)
                latent_eta_precomp[i] += factors_constrained[i * K_latent + k] * sigma_latent_vec[k];
    }

    // --- Spatiotemporal ---
    const bool has_st = layout.has_spatiotemporal && !layout.is_st_gp &&
                        data.spatiotemporal_data.type != STType::NONE &&
                        layout.st_delta_start >= 0 && layout.log_tau_st_idx >= 0;
    double tau_st = 0.0;
    const double* st_delta = nullptr;
    static thread_local std::vector<double> st_delta_buf;
    bool st_use_nc = false;
    double inv_scale_st = 1.0;
    const double* z_or_delta_st = nullptr;
    int ST_n = 0, S_st = 0, T_st = 0;
    if (has_st) {
        const auto& st = data.spatiotemporal_data;
        S_st = st.n_spatial;
        T_st = st.n_times;
        ST_n = st.n_params;
        tau_st = std::exp(params[layout.log_tau_st_idx]);
        st_use_nc = (data.st_parameterization == 1 && st.type == STType::TYPE_IV);
        z_or_delta_st = &params[layout.st_delta_start];
        if (st_use_nc) {
            inv_scale_st = 1.0 / std::sqrt(tau_st);
            st_delta_buf.resize(ST_n);
            for (int k = 0; k < ST_n; k++) st_delta_buf[k] = z_or_delta_st[k] * inv_scale_st;
            st_delta = st_delta_buf.data();
        } else {
            st_delta = z_or_delta_st;
        }
    }

    // --- Random slopes ---
    const bool has_slopes = layout.has_re_slopes;
    bool slopes_nc = has_slopes && (data.re_parameterization == 1);
    int n_re_terms_slopes = 0;
    static thread_local std::vector<std::vector<double>> grad_re_slopes_lik;
    static thread_local std::vector<std::vector<double>> nc_L_flats, nc_sigmas_vec;
    static thread_local std::vector<double> re_nc_flat_c;
    // Slopes priors and eta contribution are handled inline below

    // --- Multiscale temporal ---
    const bool has_ms_temporal = layout.has_multiscale_temporal;
    int n_trend = 0, n_seasonal = 0, n_short = 0;
    const double* trend = nullptr;
    const double* seasonal = nullptr;
    const double* short_term = nullptr;
    double sigma2_trend = 1.0, sigma2_seasonal = 1.0, sigma2_short = 1.0;
    double rho_short = 0.5;
    static thread_local std::vector<double> ms_effect_by_time_c;
    static thread_local std::vector<double> grad_trend_lik_c, grad_seasonal_lik_c, grad_short_lik_c;
    static thread_local std::vector<int> obs_t_idx_ms_c;
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        n_trend = layout.trend_end - layout.trend_start;
        n_seasonal = layout.seasonal_end - layout.seasonal_start;
        n_short = layout.short_term_end - layout.short_term_start;
        if (n_trend > 0) { trend = &params[layout.trend_start]; sigma2_trend = std::exp(params[layout.log_sigma2_trend_idx]); }
        if (n_seasonal > 0) { seasonal = &params[layout.seasonal_start]; sigma2_seasonal = std::exp(params[layout.log_sigma2_seasonal_idx]); }
        if (n_short > 0) { short_term = &params[layout.short_term_start]; sigma2_short = std::exp(params[layout.log_sigma2_short_idx]); }
        if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
            double u = 1.0 / (1.0 + std::exp(-params[layout.logit_rho_short_idx]));
            rho_short = 2.0 * u - 1.0;
        }
        // Pre-assemble temporal effect per time point
        ms_effect_by_time_c.assign(mst.n_times, 0.0);
        for (int t = 0; t < mst.n_times; t++) {
            if (trend != nullptr && t < n_trend) ms_effect_by_time_c[t] += trend[t];
            if (seasonal != nullptr && mst.seasonal_period > 0) ms_effect_by_time_c[t] += seasonal[t % mst.seasonal_period];
            if (short_term != nullptr && t < n_short) ms_effect_by_time_c[t] += short_term[t];
        }
        grad_trend_lik_c.assign(n_trend, 0.0);
        grad_seasonal_lik_c.assign(n_seasonal, 0.0);
        grad_short_lik_c.assign(n_short, 0.0);
        obs_t_idx_ms_c.assign(N, -1);
    }

    // =========================================================================
    // Phase 2: Prior gradients (independent of data, computed first)
    // =========================================================================
    beta_gradient_prior(data, layout, beta_num, beta_denom, grad.data());
    if (!has_slopes) re_gradient_prior(data, layout, re, grad.data(), sigma_re);
    phi_gradient_prior(data, layout, phi_num, phi_denom, grad.data());

    // ZI/OI beta priors: N(0, zi_prior_sd^2) / N(0, oi_prior_sd^2)
    if (layout.has_zi && data.p_zi > 0) {
        double tau_zi = 1.0 / (data.zi_prior_sd * data.zi_prior_sd + 1e-10);
        for (int j = 0; j < data.p_zi; j++)
            grad[layout.beta_zi_start + j] = -tau_zi * beta_zi[j];
    }
    if (layout.has_oi && data.p_oi > 0) {
        double tau_oi = 1.0 / (data.oi_prior_sd * data.oi_prior_sd + 1e-10);
        for (int j = 0; j < data.p_oi; j++)
            grad[layout.beta_oi_start + j] = -tau_oi * beta_oi[j];
    }

    // ICAR/BYM2 spatial priors
    if (has_icar_bym2 && !layout.is_bym2) {
        grad[layout.log_tau_spatial_idx] = (data.tau_spatial_shape - 1.0) - data.tau_spatial_rate * tau_spatial + 1.0;
    }
    if (has_icar_bym2 && layout.is_bym2) {
        double sigma_total = sigma_s_bym2 / std::sqrt(rho_bym2);
        double ratio = sigma_total / data.sigma_re_scale;
        grad[layout.log_sigma_bym2_idx] = -2.0 * ratio * ratio / (1.0 + ratio * ratio) + 1.0;
        grad[layout.logit_rho_bym2_idx] = 1.0 - 2.0 * rho_bym2;
    }

    // GP (NNGP) priors
    if (has_gp_spatial) {
        // PC prior on sigma2_gp
        grad[layout.log_sigma2_gp_idx] = gp_pc_prior_grad_log_sigma2(
            sigma2_gp_c, data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha);
        // Uniform prior on phi: just Jacobian for log transform
        grad[layout.log_phi_gp_idx] = 1.0;
        if (!use_nc_gp) {
            // Centered: NNGP prior gradients on w
            tulpa_gp::NNGPGradients nngp_grads;
            tulpa_gp::gp_nngp_gradients(gp_w_c, sigma2_gp_c, phi_gp_c, data.gp_data, nngp_grads);
            for (int i = 0; i < N_gp_c; i++)
                grad[layout.gp_w_start + i] += nngp_grads.grad_w[i];
            grad[layout.log_sigma2_gp_idx] += nngp_grads.grad_log_sigma2;
            grad[layout.log_phi_gp_idx] += nngp_grads.grad_log_phi;
        } else {
            // NC: N(0,1) prior on z
            for (int i = 0; i < N_gp_c; i++)
                grad[layout.gp_w_start + i] = -params[layout.gp_w_start + i];
        }
    }

    // HSGP priors (must match hardcoded rate=4.6 in compute_log_post)
    if (layout.is_hsgp && data.has_hsgp) {
        double sigma = std::sqrt(hsgp_sigma2);
        double rate = 4.6;  // Matches compute_log_post line 1457
        grad[layout.log_sigma2_hsgp_idx] = -0.5 * rate * sigma + 0.5 - 0.5;
        double log_ls = params[layout.log_lengthscale_hsgp_idx];
        grad[layout.log_lengthscale_hsgp_idx] = -log_ls;  // LogNormal(0,1) prior
        // N(0,I) prior on beta
        int M = data.hsgp_data.m_total;
        for (int j = 0; j < M; j++)
            grad[layout.hsgp_beta_start + j] = -hsgp_beta_ptr[j];
    }

    // MSGP-HSGP priors
    if (has_msgp_hsgp) {
        double sigma_local = std::sqrt(msgp_sigma2_local);
        double rate_local = -std::log(data.ms_sigma2_local_prior_alpha) / data.ms_sigma2_local_prior_U;
        grad[layout.log_sigma2_gp_local_idx] = -0.5 * rate_local * sigma_local + 0.5 - 0.5;

        double sigma_regional = std::sqrt(msgp_sigma2_regional);
        double rate_regional = -std::log(data.ms_sigma2_regional_prior_alpha) / data.ms_sigma2_regional_prior_U;
        grad[layout.log_sigma2_gp_regional_idx] = -0.5 * rate_regional * sigma_regional + 0.5 - 0.5;

        double log_ls_local_v = params[layout.log_phi_gp_local_idx];
        double z_local_v = (log_ls_local_v - data.ms_log_ls_local_mean) / data.ms_log_ls_local_sd;
        grad[layout.log_phi_gp_local_idx] = -z_local_v / data.ms_log_ls_local_sd;

        double log_ls_regional_v = params[layout.log_phi_gp_regional_idx];
        double z_regional_v = (log_ls_regional_v - data.ms_log_ls_regional_mean) / data.ms_log_ls_regional_sd;
        grad[layout.log_phi_gp_regional_idx] = -z_regional_v / data.ms_log_ls_regional_sd;

        for (int j = 0; j < msgp_m_total; j++) {
            grad[layout.gp_local_start + j] = -msgp_beta_local[j];
            grad[layout.gp_regional_start + j] = -msgp_beta_regional[j];
        }
    }

    // Temporal GMRF priors
    if (has_gmrf_temporal) {
        tau_temporal_prior_grad(data, layout, tau_temporal, grad.data());
        if (data.temporal_type == TemporalType::AR1 && layout.logit_rho_ar1_idx >= 0)
            grad[layout.logit_rho_ar1_idx] = 1.0 - 2.0 * rho_ar1;
    }

    // Temporal GP priors
    if (has_temporal_gp) {
        double sigma_gp = std::sqrt(sigma2_tgp_comp);
        double rate_gp = -std::log(data.temporal_gp_sigma2_prior_alpha) / data.temporal_gp_sigma2_prior_U;
        grad[layout.log_sigma2_temporal_gp_idx] = -0.5 * rate_gp * sigma_gp + 0.5;
        // Phi: logit-bounded Jacobian
        double phi_range_tgp = phi_upper_tgp - phi_lower_tgp;
        grad[layout.logit_phi_temporal_gp_idx] = (phi_upper_tgp + phi_lower_tgp - 2.0 * phi_tgp_comp) / phi_range_tgp;

        if (use_nc_tgp) {
            // NC Jacobian: d/d(log_sigma2) of log|det(df/dz)| = T/2 per group
            grad[layout.log_sigma2_temporal_gp_idx] += 0.5 * T_gp * n_groups_gp;

            // NC Jacobian: d/d(log_phi) = -sum rho^2*(dt/phi) / (1-rho^2) per group
            double chi_tgp_prior = (phi_tgp_comp - phi_lower_tgp) * (phi_upper_tgp - phi_tgp_comp) /
                                   (phi_tgp_comp * phi_range_tgp);
            double jac_phi_log = 0.0;
            for (int t = 1; t < T_gp; t++) {
                double rho_t = nc_ws_composite.rho[t - 1];
                double rho2 = rho_t * rho_t;
                double dt = data.temporal_gp_data.time_values[t] - data.temporal_gp_data.time_values[t - 1];
                double one_minus_rho2 = 1.0 - rho2;
                if (one_minus_rho2 < 1e-10) one_minus_rho2 = 1e-10;
                jac_phi_log -= rho2 * (dt / phi_tgp_comp) / one_minus_rho2;
            }
            grad[layout.logit_phi_temporal_gp_idx] += jac_phi_log * n_groups_gp * chi_tgp_prior;
        }
    }

    // TVC priors (PC prior: P(sigma > 1) = 0.01)
    if (layout.has_tvc && data.has_tvc) {
        double tvc_pc_rate = -std::log(0.01) / 1.0;
        for (int j = 0; j < n_tvc; j++) {
            double sigma_j = 1.0 / std::sqrt(tvc_tau_buf[j]);
            grad[layout.log_tau_tvc_start + j] = 0.5 * tvc_pc_rate * sigma_j + 1.5;
            if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1) {
                double u = (tvc_rho_buf[j] + 1.0) / 2.0;
                grad[layout.logit_rho_tvc_start + j] = 1.0 - 2.0 * u;
            }
        }
    }

    // SVC-HSGP priors: PC prior on sigma, LogNormal(0,1) on lengthscale, N(0,I) on beta
    // Must match compute_log_post: -rate*sigma + 0.5*log_sigma2 (Jacobian)
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        double rate_svc = 4.6;  // Matches compute_log_post line 1781
        for (int j = 0; j < n_svc; j++) {
            double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
            double sigma_j = std::sqrt(sigma2_j);
            // d/d(log_sigma2) [-rate*sigma + 0.5*log_sigma2] = -0.5*rate*sigma + 0.5
            grad[layout.log_sigma2_svc_start + j] = -0.5 * rate_svc * sigma_j + 0.5;
            // LogNormal(0,1) on lengthscale: d/d(log_ls) [-0.5*log_ls^2] = -log_ls
            double log_ls_j = params[layout.log_phi_svc_start + j];
            grad[layout.log_phi_svc_start + j] = -log_ls_j;
            // N(0,I) prior on SVC beta
            for (int m = 0; m < svc_m_total; m++)
                grad[layout.svc_w_start + j * svc_m_total + m] = -params[layout.svc_w_start + j * svc_m_total + m];
        }
    }

    // Latent priors
    if (K_latent > 0) {
        double latent_rate = data.latent_sigma_prior_rate;
        for (int k = 0; k < K_latent; k++)
            grad[layout.log_sigma_latent_start + k] = 1.0 - latent_rate * sigma_latent_vec[k];
    }

    // Multiscale temporal priors (PC priors on variances + AR1 rho prior)
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        if (n_trend > 0)
            grad[layout.log_sigma2_trend_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
                sigma2_trend, data.ms_sigma2_trend_prior_U, data.ms_sigma2_trend_prior_alpha) + 1.0;
        if (n_seasonal > 0)
            grad[layout.log_sigma2_seasonal_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
                sigma2_seasonal, data.ms_sigma2_seasonal_prior_U, data.ms_sigma2_seasonal_prior_alpha) + 1.0;
        if (n_short > 0)
            grad[layout.log_sigma2_short_idx] = tulpa_temporal_grad::pc_prior_grad_log_sigma2(
                sigma2_short, data.ms_sigma2_short_prior_U, data.ms_sigma2_short_prior_alpha) + 1.0;
        if (mst.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0) {
            double u = (rho_short + 1.0) / 2.0;
            grad[layout.logit_rho_short_idx] = 2.0 * (1.0 - 2.0 * u);
        }
    }

    // ST interaction prior on tau_st
    if (has_st) {
        double sigma_st = 1.0 / std::sqrt(tau_st);
        double lambda = -std::log(data.st_sigma2_prior_alpha) / data.st_sigma2_prior_U;
        grad[layout.log_tau_st_idx] = 0.5 * lambda * sigma_st + 0.5 + 1.0;
    }

    // Slopes priors — mirrors compute_gradient_analytical (correlated + uncorrelated)
    nc_L_flats.clear();
    nc_sigmas_vec.clear();
    re_nc_flat_c.clear();
    if (has_slopes) {
        n_re_terms_slopes = data.n_re_terms;
        grad_re_slopes_lik.resize(n_re_terms_slopes);
        nc_L_flats.resize(n_re_terms_slopes);
        nc_sigmas_vec.resize(n_re_terms_slopes);

        for (int t = 0; t < n_re_terms_slopes; t++) {
            int n_groups = data.re_n_groups_multi[t];
            int n_coefs = layout.re_n_coefs_multi[t];
            int re_start_t = layout.re_start_multi[t];
            bool is_correlated = layout.has_re_correlated_slopes &&
                                 t < (int)layout.re_correlated_multi.size() &&
                                 layout.re_correlated_multi[t];
            grad_re_slopes_lik[t].assign(n_groups * n_coefs, 0.0);

            // Extract sigmas and compute Half-Cauchy prior on each
            std::vector<double> sigmas(n_coefs);
            for (int c = 0; c < n_coefs; c++) {
                int log_sigma_idx = layout.log_sigma_re_slopes[t][c];
                sigmas[c] = std::exp(params[log_sigma_idx]);
                double ratio_c = sigmas[c] / data.sigma_re_scale;
                double ratio_c_sq = ratio_c * ratio_c;
                grad[log_sigma_idx] = -2.0 * ratio_c_sq / (1.0 + ratio_c_sq) + 1.0;
            }

            if (is_correlated && n_coefs > 1) {
                // Tanh-parameterized L + LKJ(eta=2) prior gradient (raw-space,
                // includes tanh + L->R Jacobians) via lkj_chol_helpers.
                int chol_start = layout.chol_re_start_multi[t];
                std::vector<double> L_flat(n_coefs * n_coefs, 0.0);
                if (!tulpa::build_L_from_raw(&params[chol_start], n_coefs, L_flat.data())) {
                    return;
                }
                tulpa::lkj_log_prior_grad_add(&params[chol_start], L_flat.data(), n_coefs,
                                              /*eta=*/2.0, &grad[chol_start]);

                // Pre-compute re = diag(sigma) * L * z for observation loop
                if (re_nc_flat_c.empty()) re_nc_flat_c.assign(params.size(), 0.0);
                tulpa::compute_u_eff(L_flat.data(), n_coefs, sigmas.data(),
                                      &params[re_start_t], n_groups,
                                      &re_nc_flat_c[re_start_t]);

                // z prior: N(0,I) -> grad = -z
                for (int g = 0; g < n_groups; g++)
                    for (int c = 0; c < n_coefs; c++)
                        grad[re_start_t + g * n_coefs + c] = -params[re_start_t + g * n_coefs + c];

                nc_L_flats[t] = std::move(L_flat);
                nc_sigmas_vec[t] = std::move(sigmas);
            } else {
                // Uncorrelated slopes
                nc_L_flats[t].clear();
                nc_sigmas_vec[t] = sigmas;
                if (slopes_nc) {
                    for (int g = 0; g < n_groups; g++)
                        for (int c = 0; c < n_coefs; c++)
                            grad[re_start_t + g * n_coefs + c] = -params[re_start_t + g * n_coefs + c];
                } else {
                    for (int c = 0; c < n_coefs; c++) {
                        double tau_c = 1.0 / (sigmas[c] * sigmas[c] + 1e-10);
                        double sigma_grad_c = 0.0;
                        for (int g = 0; g < n_groups; g++) {
                            double re_gc = params[re_start_t + g * n_coefs + c];
                            grad[re_start_t + g * n_coefs + c] = -tau_c * re_gc;
                            sigma_grad_c += tau_c * re_gc * re_gc - 1.0;
                        }
                        grad[layout.log_sigma_re_slopes[t][c]] += sigma_grad_c;
                    }
                }
            }
        }
    }

    // =========================================================================
    // Phase 3: Vectorized 3-pass observation loop
    // =========================================================================
    // Pass 1: Assemble eta vectors (Eigen matvec + feature additions)
    // Pass 2: Compute residuals + phi gradients (tight family-dispatched loop)
    // Pass 3: Scatter gradients (Eigen X^T*resid + feature-specific scatter)
    // =========================================================================
    static thread_local std::vector<double> grad_spatial_lik, grad_theta_lik;
    static thread_local std::vector<double> grad_temporal_lik;
    static thread_local std::vector<double> grad_hsgp_f;
    static thread_local std::vector<double> grad_delta_lik;
    static thread_local std::vector<double> grad_tvc_w;
    static thread_local std::vector<double> grad_factors_c;
    static thread_local std::vector<double> grad_svc_f;
    static thread_local std::vector<double> grad_msgp_f;

    static thread_local std::vector<double> grad_gp_w_lik;
    if (has_gp_spatial) grad_gp_w_lik.assign(N_gp_c, 0.0);
    if (has_msgp_hsgp) grad_msgp_f.assign(N, 0.0);
    if (has_icar_bym2) grad_spatial_lik.assign(data.n_spatial_units, 0.0);
    if (has_icar_bym2 && layout.is_bym2) grad_theta_lik.assign(data.n_spatial_units, 0.0);
    if (has_gmrf_temporal) grad_temporal_lik.assign(T_temporal, 0.0);
    if (has_temporal_gp) grad_temporal_lik.assign(data.temporal_gp_data.n_groups * T_gp, 0.0);
    if (layout.is_hsgp && data.has_hsgp) grad_hsgp_f.assign(N, 0.0);
    if (has_st) grad_delta_lik.assign(ST_n, 0.0);
    if (layout.has_tvc && data.has_tvc) grad_tvc_w.assign(n_w, 0.0);
    if (K_latent > 0) grad_factors_c.assign(N * K_latent, 0.0);
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) grad_svc_f.assign(n_svc * N, 0.0);

    // --- Pass 1: Assemble eta_num and eta_denom vectors ---
    static thread_local std::vector<double> eta_num_v, eta_denom_v;
    eta_num_v.resize(N);
    eta_denom_v.resize(N);

    // X * beta via Eigen matvec
    {
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_num_mat(data.legacy.X_num_flat.data(), N, data.legacy.p_num);
        Eigen::Map<const Eigen::VectorXd> beta_num_vec(beta_num, data.legacy.p_num);
        Eigen::Map<Eigen::VectorXd>(eta_num_v.data(), N) = X_num_mat * beta_num_vec;
    }
    if (!is_binomial) {
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_denom_mat(data.legacy.X_denom_flat.data(), N, data.legacy.p_denom);
        Eigen::Map<const Eigen::VectorXd> beta_denom_vec(beta_denom, data.legacy.p_denom);
        Eigen::Map<Eigen::VectorXd>(eta_denom_v.data(), N) = X_denom_mat * beta_denom_vec;
    } else {
        std::memset(eta_denom_v.data(), 0, N * sizeof(double));
    }

    // RE (simple intercept)
    if (layout.has_re && !has_slopes) {
        for (int i = 0; i < N; i++) if (data.re_group[i] > 0) {
            double re_eff = re_value_for_eta(re, data.re_group[i] - 1, sigma_re, data.re_parameterization);
            eta_num_v[i] += re_eff;
            if (!is_binomial) eta_denom_v[i] += re_eff;
        }
    }

    // RE (slopes — multi-term)
    // For correlated NC: use pre-computed re_nc_flat_c (= diag(sigma) * L * z)
    // For uncorrelated NC: use sigma * z inline
    // For centered: use raw params
    if (has_slopes) {
        for (int i = 0; i < N; i++) {
            for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
                int n_coefs = layout.re_n_coefs_multi[t_re];
                int re_start_t = layout.re_start_multi[t_re];
                bool is_corr_nc = !nc_L_flats.empty() && t_re < (int)nc_L_flats.size() && !nc_L_flats[t_re].empty();
                int g = -1;
                if (!data.re_group_multi_flat.empty())
                    g = data.re_group_multi_flat[i * data.n_re_terms + t_re] - 1;
                else if (data.re_group[i] > 0)
                    g = data.re_group[i] - 1;
                if (g < 0) continue;

                // Intercept (coef 0)
                double re_val_0;
                if (is_corr_nc) {
                    re_val_0 = re_nc_flat_c[re_start_t + g * n_coefs + 0];
                } else {
                    re_val_0 = params[re_start_t + g * n_coefs + 0];
                    if (slopes_nc) re_val_0 *= std::exp(params[layout.log_sigma_re_slopes[t_re][0]]);
                }
                eta_num_v[i] += re_val_0;
                if (!is_binomial) eta_denom_v[i] += re_val_0;

                // Slopes (coef 1+)
                int n_slopes = n_coefs - 1;
                if (n_slopes > 0 && t_re < (int)data.re_slope_matrices.size() && !data.re_slope_matrices[t_re].empty()) {
                    for (int s = 0; s < n_slopes; s++) {
                        double re_val_s;
                        if (is_corr_nc) {
                            re_val_s = re_nc_flat_c[re_start_t + g * n_coefs + 1 + s];
                        } else {
                            re_val_s = params[re_start_t + g * n_coefs + 1 + s];
                            if (slopes_nc) re_val_s *= std::exp(params[layout.log_sigma_re_slopes[t_re][1 + s]]);
                        }
                        double eff = re_val_s * data.re_slope_matrices[t_re][i * n_slopes + s];
                        eta_num_v[i] += eff;
                        if (!is_binomial) eta_denom_v[i] += eff;
                    }
                }
            }
        }
    }

    // ICAR/BYM2 spatial
    if (has_icar_bym2) {
        for (int i = 0; i < N; i++) if (data.spatial_group[i] > 0) {
            int s_unit = data.spatial_group[i] - 1;
            double spatial_eff = layout.is_bym2
                ? sigma_s_bym2 * data.bym2_scale_factor * spatial_phi[s_unit] + sigma_u_bym2 * theta_bym2[s_unit]
                : spatial_phi[s_unit];
            eta_num_v[i] += spatial_eff;
            if (!is_binomial) eta_denom_v[i] += spatial_eff;
        }
    }

    // GP (NNGP) spatial
    if (has_gp_spatial) {
        for (int i = 0; i < N; i++) {
            double gp_eff = gp_w_c[data.gp_data.obs_to_loc[i]];
            eta_num_v[i] += gp_eff;
            if (!is_binomial && data.gp_data.shared) eta_denom_v[i] += gp_eff;
        }
    }

    // HSGP spatial (vectorized Eigen add)
    if (layout.is_hsgp && data.has_hsgp) {
        Eigen::Map<Eigen::VectorXd>(eta_num_v.data(), N) +=
            Eigen::Map<const Eigen::VectorXd>(hsgp_ws.hsgp_f.data(), N);
        if (!is_binomial && data.hsgp_data.shared)
            Eigen::Map<Eigen::VectorXd>(eta_denom_v.data(), N) +=
                Eigen::Map<const Eigen::VectorXd>(hsgp_ws.hsgp_f.data(), N);
    }

    // MSGP-HSGP spatial (vectorized)
    if (has_msgp_hsgp) {
        for (int i = 0; i < N; i++) {
            double ms_spatial = msgp_ws_local_c.hsgp_f[i] + msgp_ws_regional_c.hsgp_f[i];
            eta_num_v[i] += ms_spatial;
            if (!is_binomial && data.multiscale_gp_data.shared) eta_denom_v[i] += ms_spatial;
        }
    }

    // Temporal GMRF
    if (has_gmrf_temporal && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) if (data.temporal_time_idx[i] > 0) {
            int t_base = (data.temporal_group_idx[i] - 1) * data.n_times + data.temporal_time_idx[i] - 1;
            if (t_base >= 0 && t_base < T_temporal) {
                eta_num_v[i] += phi_temporal[t_base];
                if (!is_binomial && data.temporal_shared) eta_denom_v[i] += phi_temporal[t_base];
            }
        }
    }

    // Temporal GP
    if (has_temporal_gp && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) if (data.temporal_time_idx[i] > 0) {
            int t_base = (data.temporal_group_idx[i] - 1) * T_gp + data.temporal_time_idx[i] - 1;
            if (t_base >= 0 && t_base < (int)temporal_gp_f.size()) {
                eta_num_v[i] += temporal_gp_f[t_base];
                if (!is_binomial && data.temporal_shared) eta_denom_v[i] += temporal_gp_f[t_base];
            }
        }
    }

    // TVC (already precomputed)
    if (layout.has_tvc && data.has_tvc) {
        for (int i = 0; i < N; i++) {
            eta_num_v[i] += tvc_eta_precomp[i];
            if (!is_binomial) eta_denom_v[i] += tvc_eta_precomp[i];
        }
    }

    // SVC (already precomputed)
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        for (int i = 0; i < N; i++) {
            eta_num_v[i] += svc_eta_precomp[i];
            if (!is_binomial) eta_denom_v[i] += svc_eta_precomp[i];
        }
    }

    // Latent factors (already precomputed)
    if (K_latent > 0) {
        for (int i = 0; i < N; i++) {
            eta_num_v[i] += latent_eta_precomp[i];
            if (data.latent_shared) eta_denom_v[i] += latent_eta_precomp[i];
        }
    }

    // Multiscale temporal
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        for (int i = 0; i < N; i++) if (!mst.time_index.empty() && i < (int)mst.time_index.size() && mst.time_index[i] > 0) {
            int t_idx = mst.time_index[i] - 1;
            obs_t_idx_ms_c[i] = t_idx;
            eta_num_v[i] += ms_effect_by_time_c[t_idx];
            if (!is_binomial && mst.shared) eta_denom_v[i] += ms_effect_by_time_c[t_idx];
        }
    }

    // Spatiotemporal
    if (has_st) {
        for (int i = 0; i < N; i++) {
            double st_eff = 0.0;
            if (data.st_is_hsgp) {
                int t = data.spatiotemporal_data.t_idx[i] - 1;
                int M_st = data.st_hsgp_data.m_total;
                int T_st_c = data.spatiotemporal_data.n_times;
                for (int j = 0; j < M_st; j++)
                    st_eff += data.st_hsgp_data.phi_flat[i * M_st + j] * st_delta[j * T_st_c + t];
            } else if (data.spatiotemporal_data.st_flat[i] > 0) {
                st_eff = st_delta[data.spatiotemporal_data.st_flat[i] - 1];
            }
            eta_num_v[i] += st_eff;
            if (!is_binomial && data.spatiotemporal_data.shared) eta_denom_v[i] += st_eff;
        }
    }

    // --- Pass 2: Compute residuals + phi gradients (tight family-dispatched loop) ---
    static thread_local std::vector<double> dLL_num_v, dLL_denom_v;
    dLL_num_v.resize(N);
    dLL_denom_v.resize(N);

    // ZI/OI-aware accumulators for per-observation logit gradients
    static thread_local std::vector<double> grad_logit_zi_v, grad_logit_oi_v;
    const bool has_zi_oi = layout.has_zi || layout.has_oi;
    if (has_zi_oi) {
        grad_logit_zi_v.assign(N, 0.0);
        grad_logit_oi_v.assign(N, 0.0);
    }

    if (has_zi_oi) {
        // ZI/OI path: use compute_obs_residuals_zi for all families
        double grad_phi_num_acc = 0.0, grad_phi_denom_acc = 0.0;
        for (int i = 0; i < N; i++) {
            double gp_num_i = 0.0, gp_denom_i = 0.0;
            double gz_i = 0.0, go_i = 0.0;
            compute_obs_residuals_zi(data, layout, i,
                eta_num_v[i], eta_denom_v[i], phi_num, phi_denom,
                beta_zi, beta_oi,
                dLL_num_v[i], dLL_denom_v[i],
                gp_num_i, gp_denom_i, gz_i, go_i);
            grad_phi_num_acc += gp_num_i;
            grad_phi_denom_acc += gp_denom_i;
            grad_logit_zi_v[i] = gz_i;
            grad_logit_oi_v[i] = go_i;
        }
        // Phi gradients are on d(LL)/d(phi) scale; convert to d(LL)/d(log_phi) = phi * d(LL)/d(phi)
        if (layout.legacy.has_phi_num) grad[layout.legacy.log_phi_num_idx] += grad_phi_num_acc * phi_num;
        if (layout.legacy.has_phi_denom) grad[layout.legacy.log_phi_denom_idx] += grad_phi_denom_acc * phi_denom;
    } else if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
        double grad_phi_denom_acc = 0.0;
        for (int i = 0; i < N; i++) {
            double mu_num = std::exp(eta_num_v[i]);
            double mu_denom = std::exp(eta_denom_v[i]);
            dLL_num_v[i] = data.legacy.y_num[i] - mu_num;
            dLL_denom_v[i] = (data.legacy.y_denom_cont[i] > 0.0) ? phi_denom * (data.legacy.y_denom_cont[i] / mu_denom - 1.0) : 0.0;
            if (layout.legacy.has_phi_denom && data.legacy.y_denom_cont[i] > 0.0) {
                grad_phi_denom_acc += (std::log(phi_denom / mu_denom) + 1.0 + std::log(data.legacy.y_denom_cont[i])
                    - tulpa::math::portable_digamma(phi_denom) - data.legacy.y_denom_cont[i] / mu_denom) * phi_denom;
            }
        }
        if (layout.legacy.has_phi_denom) grad[layout.legacy.log_phi_denom_idx] += grad_phi_denom_acc;
    } else if (data.legacy.model_type == ModelType::BINOMIAL) {
        for (int i = 0; i < N; i++) {
            double p = 1.0 / (1.0 + std::exp(-eta_num_v[i]));
            dLL_num_v[i] = data.legacy.y_num[i] - data.legacy.y_denom[i] * p;
            dLL_denom_v[i] = 0.0;
        }
    } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN) {
        double grad_phi_num_acc = 0.0, grad_phi_denom_acc = 0.0;
        for (int i = 0; i < N; i++) {
            double mu_num = std::exp(eta_num_v[i]);
            double mu_denom = std::exp(eta_denom_v[i]);
            dLL_num_v[i] = data.legacy.y_num[i] - mu_num * (data.legacy.y_num[i] + phi_num) / (mu_num + phi_num);
            dLL_denom_v[i] = data.legacy.y_denom[i] - mu_denom * (data.legacy.y_denom[i] + phi_denom) / (mu_denom + phi_denom);
            if (layout.legacy.has_phi_num) {
                double y = data.legacy.y_num[i];
                grad_phi_num_acc += (tulpa::math::portable_digamma(y + phi_num) - tulpa::math::portable_digamma(phi_num)
                    + std::log(phi_num / (mu_num + phi_num)) + 1.0 - (y + phi_num) / (mu_num + phi_num)) * phi_num;
            }
            if (layout.legacy.has_phi_denom) {
                double y = data.legacy.y_denom[i];
                grad_phi_denom_acc += (tulpa::math::portable_digamma(y + phi_denom) - tulpa::math::portable_digamma(phi_denom)
                    + std::log(phi_denom / (mu_denom + phi_denom)) + 1.0 - (y + phi_denom) / (mu_denom + phi_denom)) * phi_denom;
            }
        }
        if (layout.legacy.has_phi_num) grad[layout.legacy.log_phi_num_idx] += grad_phi_num_acc;
        if (layout.legacy.has_phi_denom) grad[layout.legacy.log_phi_denom_idx] += grad_phi_denom_acc;
    } else if (data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
        double grad_phi_num_acc = 0.0, grad_phi_denom_acc = 0.0;
        for (int i = 0; i < N; i++) {
            double mu_num = std::exp(eta_num_v[i]);
            double mu_denom = std::exp(eta_denom_v[i]);
            double denom_nb = mu_num + phi_num;
            dLL_num_v[i] = data.legacy.y_num[i] - mu_num * (data.legacy.y_num[i] + phi_num) / denom_nb;
            dLL_denom_v[i] = (data.legacy.y_denom_cont[i] > 0.0) ? phi_denom * (data.legacy.y_denom_cont[i] / mu_denom - 1.0) : 0.0;
            if (layout.legacy.has_phi_num) {
                double y = data.legacy.y_num[i];
                grad_phi_num_acc += (tulpa::math::portable_digamma(y + phi_num) - tulpa::math::portable_digamma(phi_num)
                    + std::log(phi_num / denom_nb) + 1.0 - (y + phi_num) / denom_nb) * phi_num;
            }
            if (layout.legacy.has_phi_denom && data.legacy.y_denom_cont[i] > 0.0) {
                grad_phi_denom_acc += (std::log(phi_denom / mu_denom) + 1.0 + std::log(data.legacy.y_denom_cont[i])
                    - tulpa::math::portable_digamma(phi_denom) - data.legacy.y_denom_cont[i] / mu_denom) * phi_denom;
            }
        }
        if (layout.legacy.has_phi_num) grad[layout.legacy.log_phi_num_idx] += grad_phi_num_acc;
        if (layout.legacy.has_phi_denom) grad[layout.legacy.log_phi_denom_idx] += grad_phi_denom_acc;
    } else {
        // Fallback: GAMMA_GAMMA, LOGNORMAL, BETA_BINOMIAL — use per-obs helpers
        for (int i = 0; i < N; i++) {
            compute_obs_residuals(data, i, eta_num_v[i], eta_denom_v[i], phi_num, phi_denom,
                                  dLL_num_v[i], dLL_denom_v[i]);
            accumulate_phi_likelihood_grad(data, layout, i, eta_num_v[i], eta_denom_v[i],
                                            phi_num, phi_denom, grad.data());
        }
    }

    // --- Pass 3: Scatter gradients ---

    // Beta gradients via Eigen X^T * resid
    {
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_num_mat(data.legacy.X_num_flat.data(), N, data.legacy.p_num);
        Eigen::Map<Eigen::VectorXd> grad_beta_num(&grad[layout.legacy.beta_num_start], data.legacy.p_num);
        grad_beta_num += X_num_mat.transpose() * Eigen::Map<const Eigen::VectorXd>(dLL_num_v.data(), N);
    }
    if (!is_binomial) {
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X_denom_mat(data.legacy.X_denom_flat.data(), N, data.legacy.p_denom);
        Eigen::Map<Eigen::VectorXd> grad_beta_denom(&grad[layout.legacy.beta_denom_start], data.legacy.p_denom);
        grad_beta_denom += X_denom_mat.transpose() * Eigen::Map<const Eigen::VectorXd>(dLL_denom_v.data(), N);
    }

    // ZI/OI beta scatter: grad[beta_zi] += X_zi^T * grad_logit_zi, similarly for OI
    if (has_zi_oi) {
        if (layout.has_zi && data.p_zi > 0) {
            for (int i = 0; i < N; i++)
                for (int j = 0; j < data.p_zi; j++)
                    grad[layout.beta_zi_start + j] += data.X_zi_flat[i * data.p_zi + j] * grad_logit_zi_v[i];
        }
        if (layout.has_oi && data.p_oi > 0) {
            for (int i = 0; i < N; i++)
                for (int j = 0; j < data.p_oi; j++)
                    grad[layout.beta_oi_start + j] += data.X_oi_flat[i * data.p_oi + j] * grad_logit_oi_v[i];
        }
    }

    // RE (simple)
    if (layout.has_re && !has_slopes) {
        for (int i = 0; i < N; i++) if (data.re_group[i] > 0)
            grad[layout.re_start + data.re_group[i] - 1] += dLL_num_v[i] + dLL_denom_v[i];
    }

    // RE (slopes)
    if (has_slopes) {
        for (int i = 0; i < N; i++) {
            double dLL_shared_i = dLL_num_v[i] + dLL_denom_v[i];
            for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
                int n_coefs = layout.re_n_coefs_multi[t_re];
                int g = -1;
                if (!data.re_group_multi_flat.empty())
                    g = data.re_group_multi_flat[i * data.n_re_terms + t_re] - 1;
                else if (data.re_group[i] > 0)
                    g = data.re_group[i] - 1;
                if (g < 0) continue;
                grad_re_slopes_lik[t_re][g * n_coefs + 0] += dLL_shared_i;
                int n_slopes_sc = n_coefs - 1;
                if (n_slopes_sc > 0 && t_re < (int)data.re_slope_matrices.size() && !data.re_slope_matrices[t_re].empty())
                    for (int s = 0; s < n_slopes_sc; s++)
                        grad_re_slopes_lik[t_re][g * n_coefs + 1 + s] += dLL_shared_i * data.re_slope_matrices[t_re][i * n_slopes_sc + s];
            }
        }
    }

    // ICAR/BYM2 spatial
    if (has_icar_bym2) {
        for (int i = 0; i < N; i++) if (data.spatial_group[i] > 0) {
            double dLL_s = dLL_num_v[i] + dLL_denom_v[i];
            int s_unit = data.spatial_group[i] - 1;
            grad_spatial_lik[s_unit] += dLL_s;
            if (layout.is_bym2) grad_theta_lik[s_unit] += dLL_s;
        }
    }

    // GP (NNGP) spatial
    if (has_gp_spatial) {
        for (int i = 0; i < N; i++) {
            double dLL_gp = data.gp_data.shared ? (dLL_num_v[i] + dLL_denom_v[i]) : dLL_num_v[i];
            grad_gp_w_lik[data.gp_data.obs_to_loc[i]] += dLL_gp;
        }
    }

    // HSGP spatial
    if (layout.is_hsgp && data.has_hsgp) {
        if (data.hsgp_data.shared)
            for (int i = 0; i < N; i++) grad_hsgp_f[i] = dLL_num_v[i] + dLL_denom_v[i];
        else
            std::memcpy(grad_hsgp_f.data(), dLL_num_v.data(), N * sizeof(double));
    }

    // MSGP-HSGP spatial
    if (has_msgp_hsgp) {
        if (data.multiscale_gp_data.shared)
            for (int i = 0; i < N; i++) grad_msgp_f[i] = dLL_num_v[i] + dLL_denom_v[i];
        else
            std::memcpy(grad_msgp_f.data(), dLL_num_v.data(), N * sizeof(double));
    }

    // Temporal GMRF
    if (has_gmrf_temporal && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) if (data.temporal_time_idx[i] > 0) {
            int t_base = (data.temporal_group_idx[i] - 1) * data.n_times + data.temporal_time_idx[i] - 1;
            if (t_base >= 0 && t_base < T_temporal)
                grad_temporal_lik[t_base] += data.temporal_shared ? (dLL_num_v[i] + dLL_denom_v[i]) : dLL_num_v[i];
        }
    }

    // Temporal GP
    if (has_temporal_gp && !data.temporal_time_idx.empty()) {
        for (int i = 0; i < N; i++) if (data.temporal_time_idx[i] > 0) {
            int t_base = (data.temporal_group_idx[i] - 1) * T_gp + data.temporal_time_idx[i] - 1;
            if (t_base >= 0 && t_base < (int)temporal_gp_f.size())
                grad_temporal_lik[t_base] += data.temporal_shared ? (dLL_num_v[i] + dLL_denom_v[i]) : dLL_num_v[i];
        }
    }

    // TVC
    if (layout.has_tvc && data.has_tvc) {
        for (int i = 0; i < N; i++) {
            double dLL_s = dLL_num_v[i] + dLL_denom_v[i];
            int t = data.tvc_data.time_index[i] - 1;
            int g = data.tvc_data.group_index[i] - 1;
            for (int j = 0; j < n_tvc; j++)
                grad_tvc_w[(g * n_tvc + j) * n_tvc_times + t] += dLL_s * data.tvc_data.X_tvc[i * n_tvc + j];
        }
    }

    // SVC
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        for (int i = 0; i < N; i++) {
            double dLL_s = dLL_num_v[i] + dLL_denom_v[i];
            for (int j = 0; j < n_svc; j++)
                grad_svc_f[j * N + i] = dLL_s * data.svc_data.X_svc[i * n_svc + j];
        }
    }

    // Latent factors
    if (K_latent > 0) {
        for (int i = 0; i < N; i++) {
            double dLL_latent = data.latent_shared ? (dLL_num_v[i] + dLL_denom_v[i]) : dLL_num_v[i];
            for (int k = 0; k < K_latent; k++) {
                grad_factors_c[i * K_latent + k] += dLL_latent * sigma_latent_vec[k];
                grad[layout.log_sigma_latent_start + k] += dLL_latent * factors_constrained[i * K_latent + k] * sigma_latent_vec[k];
            }
        }
    }

    // Multiscale temporal
    if (has_ms_temporal) {
        const auto& mst = data.multiscale_temporal_data;
        for (int i = 0; i < N; i++) if (obs_t_idx_ms_c[i] >= 0) {
            int t_idx = obs_t_idx_ms_c[i];
            double dLL_temporal = mst.shared ? (dLL_num_v[i] + dLL_denom_v[i]) : dLL_num_v[i];
            if (trend != nullptr && t_idx < n_trend) grad_trend_lik_c[t_idx] += dLL_temporal;
            if (seasonal != nullptr && mst.seasonal_period > 0) {
                int s_idx = t_idx % mst.seasonal_period;
                if (s_idx < n_seasonal) grad_seasonal_lik_c[s_idx] += dLL_temporal;
            }
            if (short_term != nullptr && t_idx < n_short) grad_short_lik_c[t_idx] += dLL_temporal;
        }
    }

    // ST interaction
    if (has_st) {
        for (int i = 0; i < N; i++) {
            double dLL_st = data.spatiotemporal_data.shared ? (dLL_num_v[i] + dLL_denom_v[i]) : dLL_num_v[i];
            if (data.st_is_hsgp) {
                int t = data.spatiotemporal_data.t_idx[i] - 1;
                int M_st = data.st_hsgp_data.m_total;
                int T_st_c = data.spatiotemporal_data.n_times;
                for (int j = 0; j < M_st; j++)
                    grad_delta_lik[j * T_st_c + t] += data.st_hsgp_data.phi_flat[i * M_st + j] * dLL_st;
            } else if (data.spatiotemporal_data.st_flat[i] > 0) {
                grad_delta_lik[data.spatiotemporal_data.st_flat[i] - 1] += dLL_st;
            }
        }
    }

    // Phi likelihood gradients already accumulated in Pass 2

    // =========================================================================
    // Phase 4: Structural/GMRF prior gradients (post-observation loop)
    // =========================================================================

    // ICAR/BYM2 spatial GMRF
    if (has_icar_bym2) {
        spatial_gmrf_prior_grad(data, layout, spatial_phi, tau_spatial,
                                sigma_s_bym2, sigma_u_bym2, rho_bym2, theta_bym2,
                                grad_spatial_lik.data(),
                                layout.is_bym2 ? grad_theta_lik.data() : nullptr,
                                grad.data());
    }

    // GP (NNGP) backward pass
    if (has_gp_spatial) {
        if (use_nc_gp) {
            // NC: dL/dw -> grad_z, grad_log_sigma2, grad_log_phi via backward pass
            std::vector<double> grad_z_gp(N_gp_c, 0.0);
            double grad_log_sigma2_gp_bw = 0.0, grad_log_phi_gp_bw = 0.0, grad_log_phi_gp_jac = 0.0;
            tulpa_gp::nngp_nc_backward(
                &params[layout.gp_w_start], sigma2_gp_c, phi_gp_c,
                data.gp_data, nc_ws_gp_c, grad_gp_w_lik.data(),
                grad_z_gp.data(), grad_log_sigma2_gp_bw, grad_log_phi_gp_bw, grad_log_phi_gp_jac);
            for (int i = 0; i < N_gp_c; i++)
                grad[layout.gp_w_start + i] += grad_z_gp[i];
            grad[layout.log_sigma2_gp_idx] += grad_log_sigma2_gp_bw;
            grad[layout.log_phi_gp_idx] += grad_log_phi_gp_bw + grad_log_phi_gp_jac;
        } else {
            // Centered: just add likelihood gradients to w
            for (int i = 0; i < N_gp_c; i++)
                grad[layout.gp_w_start + i] += grad_gp_w_lik[i];
        }
    }

    // HSGP spectral density gradients
    if (layout.is_hsgp && data.has_hsgp) {
        int M = data.hsgp_data.m_total;
        std::copy(grad_hsgp_f.begin(), grad_hsgp_f.end(), hsgp_ws.grad_f.begin());
        double grad_log_sigma2 = 0.0, grad_log_lengthscale = 0.0;
        tulpa_hsgp::hsgp_compute_gradients_ws(
            hsgp_beta_ptr, hsgp_sigma2, hsgp_lengthscale,
            data.hsgp_data, hsgp_ws, grad_log_sigma2, grad_log_lengthscale);
        grad[layout.log_sigma2_hsgp_idx] += grad_log_sigma2;
        grad[layout.log_lengthscale_hsgp_idx] += grad_log_lengthscale;
        for (int j = 0; j < M; j++)
            grad[layout.hsgp_beta_start + j] += hsgp_ws.grad_beta_out[j];
    }

    // MSGP-HSGP spectral density gradients
    if (has_msgp_hsgp) {
        std::memcpy(msgp_ws_local_c.grad_f.data(), grad_msgp_f.data(), N * sizeof(double));
        std::memcpy(msgp_ws_regional_c.grad_f.data(), grad_msgp_f.data(), N * sizeof(double));

        double grad_log_sigma2_local_c = 0.0, grad_log_ls_local_c = 0.0;
        tulpa_hsgp::hsgp_compute_gradients_ws(
            msgp_beta_local, msgp_sigma2_local, msgp_ls_local,
            data.msgp_hsgp_data, msgp_ws_local_c,
            grad_log_sigma2_local_c, grad_log_ls_local_c);

        double grad_log_sigma2_regional_c = 0.0, grad_log_ls_regional_c = 0.0;
        tulpa_hsgp::hsgp_compute_gradients_ws(
            msgp_beta_regional, msgp_sigma2_regional, msgp_ls_regional,
            data.msgp_hsgp_data, msgp_ws_regional_c,
            grad_log_sigma2_regional_c, grad_log_ls_regional_c);

        for (int j = 0; j < msgp_m_total; j++) {
            grad[layout.gp_local_start + j] += msgp_ws_local_c.grad_beta_out[j];
            grad[layout.gp_regional_start + j] += msgp_ws_regional_c.grad_beta_out[j];
        }
        grad[layout.log_sigma2_gp_local_idx] += grad_log_sigma2_local_c;
        grad[layout.log_phi_gp_local_idx] += grad_log_ls_local_c;
        grad[layout.log_sigma2_gp_regional_idx] += grad_log_sigma2_regional_c;
        grad[layout.log_phi_gp_regional_idx] += grad_log_ls_regional_c;
    }

    // Temporal GMRF prior
    if (has_gmrf_temporal && T_temporal > 0) {
        temporal_gmrf_prior_grad(data, layout, tau_temporal, rho_ar1,
                                 phi_temporal, T_temporal, grad_temporal_lik.data(), grad.data());
    }

    // Temporal GP backward pass
    if (has_temporal_gp) {
        int T_len_gp = layout.temporal_end - layout.temporal_start;

        if (use_nc_tgp) {
            // Copy likelihood gradients to nc workspace
            for (int k = 0; k < n_groups_gp * T_gp; k++)
                nc_ws_composite.dL_df[k] = grad_temporal_lik[k];

            double grad_log_sigma2_gp = 0.0, grad_log_phi_gp = 0.0;
            tulpa_temporal_gp::temporal_gp_nc_backward(
                z_temporal_gp, T_gp, n_groups_gp, sigma2_tgp_comp, phi_tgp_comp,
                data.temporal_gp_data.time_values, nc_ws_composite,
                &grad[layout.temporal_start],
                grad_log_sigma2_gp, grad_log_phi_gp);
            grad[layout.log_sigma2_temporal_gp_idx] += grad_log_sigma2_gp;
            // Convert log_phi gradient to logit_phi gradient
            double chi_tgp = (phi_tgp_comp - phi_lower_tgp) * (phi_upper_tgp - phi_tgp_comp) /
                             (phi_tgp_comp * (phi_upper_tgp - phi_lower_tgp));
            grad[layout.logit_phi_temporal_gp_idx] += grad_log_phi_gp * chi_tgp;
        } else {
            // Centered: just add likelihood gradients
            for (int k = 0; k < T_len_gp; k++)
                grad[layout.temporal_start + k] = grad_temporal_lik[k];
        }
    }

    // Multiscale temporal GMRF prior gradients
    if (has_ms_temporal) {
        tulpa_temporal_grad::MultiscaleTemporalGradients ms_grads;
        tulpa_temporal_grad::multiscale_temporal_prior_gradients(
            trend, n_trend, seasonal, n_seasonal, short_term, n_short,
            sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
            data.multiscale_temporal_data, ms_grads);
        for (int t = 0; t < n_trend; t++) grad[layout.trend_start + t] = grad_trend_lik_c[t] + ms_grads.grad_trend[t];
        for (int t = 0; t < n_seasonal; t++) grad[layout.seasonal_start + t] = grad_seasonal_lik_c[t] + ms_grads.grad_seasonal[t];
        for (int t = 0; t < n_short; t++) grad[layout.short_term_start + t] = grad_short_lik_c[t] + ms_grads.grad_short_term[t];
        if (n_trend > 0) grad[layout.log_sigma2_trend_idx] += ms_grads.grad_log_sigma2_trend;
        if (n_seasonal > 0) grad[layout.log_sigma2_seasonal_idx] += ms_grads.grad_log_sigma2_seasonal;
        if (n_short > 0) grad[layout.log_sigma2_short_idx] += ms_grads.grad_log_sigma2_short;
        if (data.multiscale_temporal_data.short_term_type == TemporalType::AR1 && layout.logit_rho_short_idx >= 0)
            grad[layout.logit_rho_short_idx] += ms_grads.grad_logit_rho_short;
    }

    // TVC structural prior gradients
    if (layout.has_tvc && data.has_tvc) {
        // Initialize TVC gradient workspace
        static thread_local tulpa_tvc::TVCGradientWS tvc_grad_ws;
        static thread_local std::vector<double> tvc_grad_w_buf, tvc_grad_log_tau_buf;
        static thread_local std::vector<double> tvc_grad_logit_rho_buf, tvc_grad_w_jg_buf, tvc_d_buf;
        tvc_grad_w_buf.assign(n_w, 0.0);
        tvc_grad_log_tau_buf.assign(n_tvc, 0.0);
        tvc_grad_logit_rho_buf.assign(n_tvc, 0.0);
        tvc_grad_w_jg_buf.resize(n_tvc_times);
        tvc_d_buf.resize(n_tvc_times);

        tvc_grad_ws.grad_w = tvc_grad_w_buf.data();
        tvc_grad_ws.grad_log_tau = tvc_grad_log_tau_buf.data();
        tvc_grad_ws.grad_logit_rho = tvc_grad_logit_rho_buf.data();
        tvc_grad_ws.grad_w_jg = tvc_grad_w_jg_buf.data();
        tvc_grad_ws.d_buf = tvc_d_buf.data();
        tvc_grad_ws.n_w = n_w;
        tvc_grad_ws.n_tvc = n_tvc;

        tulpa_tvc::tvc_prior_gradients_ws(
            tvc_w_flat_buf.data(), data.tvc_data,
            tvc_tau_buf.data(), tvc_rho_buf.data(), tvc_grad_ws);

        // Add likelihood + prior to main gradient
        for (int k = 0; k < n_w; k++)
            grad[layout.tvc_w_start + k] += grad_tvc_w[k] + tvc_grad_w_buf[k];
        for (int j = 0; j < n_tvc; j++) {
            grad[layout.log_tau_tvc_start + j] += tvc_grad_log_tau_buf[j];
            if (data.tvc_data.structure == tulpa_temporal::TemporalType::AR1)
                grad[layout.logit_rho_tvc_start + j] += tvc_grad_logit_rho_buf[j];
        }
    }

    // SVC spectral density gradients
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp) {
        for (int j = 0; j < n_svc; j++) {
            double sigma2_j = std::exp(params[layout.log_sigma2_svc_start + j]);
            double lengthscale_j = std::exp(params[layout.log_phi_svc_start + j]);
            const double* beta_j = &params[layout.svc_w_start + j * svc_m_total];

            // Re-evaluate to cache sqrt_S
            tulpa_hsgp::hsgp_evaluate_ws(beta_j, sigma2_j, lengthscale_j,
                                           data.svc_hsgp_data, svc_hsgp_ws);

            // Copy per-SVC-term grad_f
            for (int i = 0; i < N; i++) svc_hsgp_ws.grad_f[i] = grad_svc_f[j * N + i];

            double gls2 = 0.0, gll = 0.0;
            tulpa_hsgp::hsgp_compute_gradients_ws(
                beta_j, sigma2_j, lengthscale_j,
                data.svc_hsgp_data, svc_hsgp_ws, gls2, gll);

            grad[layout.log_sigma2_svc_start + j] += gls2;
            grad[layout.log_phi_svc_start + j] += gll;
            for (int m = 0; m < svc_m_total; m++)
                grad[layout.svc_w_start + j * svc_m_total + m] += svc_hsgp_ws.grad_beta_out[m];
        }
    }

    // Latent factor constraint chain rule
    if (K_latent > 0) {
        // N(0,1) prior on constrained factors
        for (int k = 0; k < K_latent; k++)
            for (int i = 0; i < N; i++)
                grad_factors_c[i * K_latent + k] -= factors_constrained[i * K_latent + k];

        // Chain rule: constrained -> raw
        if (data.latent_constraint == 0) {
            // Sum-to-zero: d/d(raw[i,k]) = d/d(constrained[i,k]) - mean(d/d(constrained[:,k]))
            for (int k = 0; k < K_latent; k++) {
                double sum_gc = 0.0;
                for (int i = 0; i < N; i++) sum_gc += grad_factors_c[i * K_latent + k];
                double mean_gc = sum_gc / N;
                for (int i = 0; i < N; i++)
                    grad[layout.latent_factor_start + i * K_latent + k] += grad_factors_c[i * K_latent + k] - mean_gc;
            }
        } else {
            for (int j = 0; j < N * K_latent; j++)
                grad[layout.latent_factor_start + j] += grad_factors_c[j];
        }
    }

    // ST interaction prior gradients
    if (has_st && data.st_is_hsgp) {
        // HSGP-ST: per-basis-function temporal GMRF with spectral precision scaling
        int M_st = data.st_hsgp_data.m_total;
        int T_st_h = data.spatiotemporal_data.n_times;
        double sigma2_st_h = std::exp(params[layout.log_sigma2_st_hsgp_idx]);
        double ls_st_h = std::exp(params[layout.log_lengthscale_st_hsgp_idx]);

        int rank_t = (data.spatiotemporal_data.temporal_type == TemporalType::RW1) ? (T_st_h - 1) :
                     (data.spatiotemporal_data.temporal_type == TemporalType::RW2) ? (T_st_h - 2) : T_st_h;
        if (data.spatiotemporal_data.temporal_cyclic) rank_t = T_st_h;

        double grad_log_sigma2_st = 0.0;
        double grad_log_ls_st = 0.0;

        for (int j = 0; j < M_st; j++) {
            double omega_sq = data.st_hsgp_data.eigenvalues[j];
            double S_j = tulpa_hsgp::spectral_density_se(omega_sq, sigma2_st_h, ls_st_h);
            double S_j_safe = std::max(S_j, 1e-10);
            double prec_j = tau_st / S_j_safe;

            // Temporal GMRF stencil for basis function j
            const double* dj = &st_delta[j * T_st_h];
            double qf = 0.0;
            if (data.spatiotemporal_data.temporal_type == TemporalType::RW1) {
                for (int t = 0; t < T_st_h; t++) {
                    double g = 0.0;
                    if (t > 0) { g += prec_j * (dj[t-1] - dj[t]); qf += (dj[t] - dj[t-1]) * (dj[t] - dj[t-1]); }
                    if (t < T_st_h - 1) g += prec_j * (dj[t+1] - dj[t]);
                    grad[layout.st_delta_start + j * T_st_h + t] = grad_delta_lik[j * T_st_h + t] + g;
                }
            } else {
                // RW2 or other — use generic stencil
                for (int t = 0; t < T_st_h; t++)
                    grad[layout.st_delta_start + j * T_st_h + t] = grad_delta_lik[j * T_st_h + t];
            }

            // Sum-to-zero per basis function
            double sum_j = 0.0;
            for (int t = 0; t < T_st_h; t++) sum_j += dj[t];
            for (int t = 0; t < T_st_h; t++)
                grad[layout.st_delta_start + j * T_st_h + t] -= 0.001 * sum_j;

            // Tau gradient contribution from this basis function
            // d/d(log_tau) [0.5*rank*log(prec_j) - 0.5*prec_j*qf]
            //   = 0.5*rank - 0.5*prec_j*qf  (since d(prec_j)/d(log_tau) = prec_j)
            grad[layout.log_tau_st_idx] += 0.5 * rank_t - 0.5 * prec_j * qf;

            // Sigma2/lengthscale gradient through S_j
            // d/d(log_sigma2) S_j = S_j (SE kernel)
            // d/d(log_sigma2) [0.5*rank*log(tau/S_j) - 0.5*tau/S_j*qf]
            //   = -0.5*rank + 0.5*prec_j*qf  (times sigma2, cancels with chain rule)
            double dS_dsigma2 = S_j_safe / sigma2_st_h;
            double dLogPrior_dS = -0.5 * rank_t / S_j_safe + 0.5 * tau_st * qf / (S_j_safe * S_j_safe);
            grad_log_sigma2_st += dLogPrior_dS * dS_dsigma2 * sigma2_st_h;

            // d/d(log_ell) S_j = S_j * (1/ell - ell*omega_sq) * ell
            double dS_dl = S_j_safe * (1.0 / ls_st_h - ls_st_h * omega_sq);
            grad_log_ls_st += dLogPrior_dS * dS_dl * ls_st_h;
        }

        // HSGP-ST hyperparameter priors + likelihood chain rule
        // PC prior on sigma: d/d(log_sigma2) [-4.6*sigma + 0.5*log_sigma2] = -0.5*4.6*sigma + 0.5
        grad[layout.log_sigma2_st_hsgp_idx] = -0.5 * 4.6 * std::sqrt(sigma2_st_h) + 0.5 + grad_log_sigma2_st;
        // LogNormal(0,1) on lengthscale
        grad[layout.log_lengthscale_st_hsgp_idx] = -params[layout.log_lengthscale_st_hsgp_idx] + grad_log_ls_st;

    } else if (has_st) {
        const auto& st = data.spatiotemporal_data;
        if (st.type == STType::TYPE_I) {
            double qf = 0.0;
            for (int k = 0; k < ST_n; k++) {
                grad[layout.st_delta_start + k] = grad_delta_lik[k] - tau_st * st_delta[k];
                qf += st_delta[k] * st_delta[k];
            }
            grad[layout.log_tau_st_idx] += 0.5 * ST_n - 0.5 * tau_st * qf;
        } else if (st.type == STType::TYPE_II) {
            double total_qf = 0.0;
            for (int s = 0; s < S_st; s++) {
                const double* delta_s = &st_delta[s * T_st];
                if (st.temporal_type == TemporalType::RW1) {
                    double qf = 0.0;
                    for (int t = 0; t < T_st; t++) {
                        double g = 0.0;
                        if (t > 0) { g += tau_st * (delta_s[t-1] - delta_s[t]); qf += std::pow(delta_s[t] - delta_s[t-1], 2); }
                        if (t < T_st - 1) g += tau_st * (delta_s[t+1] - delta_s[t]);
                        grad[layout.st_delta_start + s * T_st + t] = grad_delta_lik[s * T_st + t] + g;
                    }
                    total_qf += qf;
                } else if (st.temporal_type == TemporalType::AR1) {
                    // AR1: precision Q = tau * tridiag(-rho, 1+rho^2, -rho), first/last diagonal = 1
                    // ST interaction doesn't have its own rho — use IID (rho=0) as fallback
                    // This is equivalent to tau * I for the ST delta prior
                    double qf = 0.0;
                    for (int t = 0; t < T_st; t++) {
                        grad[layout.st_delta_start + s * T_st + t] = grad_delta_lik[s * T_st + t] - tau_st * delta_s[t];
                        qf += delta_s[t] * delta_s[t];
                    }
                    total_qf += qf;
                } else {
                    // Fallback: at minimum write likelihood gradient
                    for (int t = 0; t < T_st; t++)
                        grad[layout.st_delta_start + s * T_st + t] = grad_delta_lik[s * T_st + t];
                }
            }
            int rank_per_unit = (st.temporal_type == TemporalType::RW1) ? (T_st - 1) :
                                (st.temporal_type == TemporalType::RW2) ? (T_st - 2) : T_st;
            grad[layout.log_tau_st_idx] += 0.5 * S_st * rank_per_unit - 0.5 * tau_st * total_qf;
        } else if (st.type == STType::TYPE_III) {
            double total_qf = 0.0;
            for (int t = 0; t < T_st; t++) {
                for (int s = 0; s < S_st; s++) {
                    double icar_grad = 0.0;
                    for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                        int j = st.adj_col_idx[idx] - 1;
                        icar_grad += tau_st * (st_delta[j * T_st + t] - st_delta[s * T_st + t]);
                    }
                    grad[layout.st_delta_start + s * T_st + t] = grad_delta_lik[s * T_st + t] + icar_grad;
                }
                for (int s = 0; s < S_st; s++) {
                    for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                        int j = st.adj_col_idx[idx] - 1;
                        if (j > s) total_qf += std::pow(st_delta[s * T_st + t] - st_delta[j * T_st + t], 2);
                    }
                }
            }
            grad[layout.log_tau_st_idx] += 0.5 * T_st * (S_st - 1) - 0.5 * tau_st * total_qf;
        } else if (st.type == STType::TYPE_IV) {
            // Kronecker: Q_delta = Q_s ⊗ Q_t (analytical gradient, same as specialized ST function)
            const double* stencil_input = st_use_nc ? z_or_delta_st : st_delta;
            double inv_scale_nc = st_use_nc ? (1.0 / std::sqrt(tau_st)) : 1.0;

            // Step 1: Apply temporal stencil: v[s,t] = (Q_t * input[s,:])_t
            static thread_local std::vector<double> v_kron;
            v_kron.assign(S_st * T_st, 0.0);
            if (st.temporal_type == TemporalType::RW1) {
                for (int s = 0; s < S_st; s++) {
                    for (int t = 0; t < T_st; t++) {
                        double qt_val = 0.0;
                        int n_t_neigh = 0;
                        if (t > 0) { qt_val -= stencil_input[s * T_st + t - 1]; n_t_neigh++; }
                        if (t < T_st - 1) { qt_val -= stencil_input[s * T_st + t + 1]; n_t_neigh++; }
                        qt_val += n_t_neigh * stencil_input[s * T_st + t];
                        v_kron[s * T_st + t] = qt_val;
                    }
                }
            } else if (st.temporal_type == TemporalType::RW2) {
                for (int s = 0; s < S_st; s++) {
                    const double* d_s = &stencil_input[s * T_st];
                    double* v_s = &v_kron[s * T_st];
                    if (T_st >= 3) {
                        const int n_d2 = T_st - 2;
                        double d2_stack[64];
                        double* d2 = (n_d2 <= 64) ? d2_stack : new double[n_d2];
                        for (int k = 0; k < n_d2; k++) d2[k] = d_s[k] - 2.0 * d_s[k + 1] + d_s[k + 2];
                        v_s[0] = d2[0];
                        v_s[1] = -2.0 * d2[0];
                        if (n_d2 > 1) v_s[1] += d2[1];
                        for (int t = 2; t < T_st - 2; t++) v_s[t] = d2[t - 2] - 2.0 * d2[t - 1] + d2[t];
                        if (T_st >= 4) v_s[T_st - 2] = d2[n_d2 - 2] - 2.0 * d2[n_d2 - 1];
                        else v_s[T_st - 2] = -2.0 * d2[0];
                        v_s[T_st - 1] = d2[n_d2 - 1];
                        if (n_d2 > 64) delete[] d2;
                    }
                }
            }

            // Step 2: Apply spatial ICAR stencil to v: (Q_s ⊗ Q_t) * input
            double total_qf = 0.0;
            for (int s = 0; s < S_st; s++) {
                for (int t = 0; t < T_st; t++) {
                    double qs_v = 0.0;
                    for (int idx = st.adj_row_ptr[s]; idx < st.adj_row_ptr[s + 1]; idx++) {
                        int j = st.adj_col_idx[idx] - 1;
                        qs_v -= v_kron[j * T_st + t];
                    }
                    int n_neigh = st.adj_row_ptr[s + 1] - st.adj_row_ptr[s];
                    qs_v += n_neigh * v_kron[s * T_st + t];

                    if (st_use_nc) {
                        grad[layout.st_delta_start + s * T_st + t] =
                            grad_delta_lik[s * T_st + t] * inv_scale_nc - qs_v;
                    } else {
                        grad[layout.st_delta_start + s * T_st + t] =
                            grad_delta_lik[s * T_st + t] - tau_st * qs_v;
                    }
                    total_qf += stencil_input[s * T_st + t] * qs_v;
                }
            }

            int rank_space = S_st - 1;
            int rank_time = (st.temporal_type == TemporalType::RW1) ? (T_st - 1) :
                            (st.temporal_type == TemporalType::RW2) ? (T_st - 2) : T_st;
            if (st.temporal_cyclic) rank_time = T_st;
            int total_rank = rank_space * rank_time;

            if (st_use_nc) {
                double lik_tau_grad = 0.0;
                for (int k = 0; k < ST_n; k++) lik_tau_grad += grad_delta_lik[k] * st_delta[k];
                grad[layout.log_tau_st_idx] += 0.5 * (total_rank - ST_n) - 0.5 * lik_tau_grad;
            } else {
                grad[layout.log_tau_st_idx] += 0.5 * total_rank - 0.5 * tau_st * total_qf;
            }
        }

        // Sum-to-zero penalty on ST delta
        double lambda_stz = 0.001;
        for (int t = 0; t < T_st; t++) {
            double row_sum = 0.0;
            for (int s = 0; s < S_st; s++) row_sum += st_delta[s * T_st + t];
            for (int s = 0; s < S_st; s++)
                grad[layout.st_delta_start + s * T_st + t] -= lambda_stz * row_sum;
        }
        for (int s = 0; s < S_st; s++) {
            double col_sum = 0.0;
            for (int t = 0; t < T_st; t++) col_sum += st_delta[s * T_st + t];
            for (int t = 0; t < T_st; t++)
                grad[layout.st_delta_start + s * T_st + t] -= lambda_stz * col_sum;
        }
    }

    // NC RE chain rule (simple intercepts)
    if (!has_slopes) re_gradient_nc_transform(data, layout, params.data(), grad.data(), sigma_re);

    // NC slopes chain rule — mirrors compute_gradient_analytical write-back
    if (has_slopes) {
        for (int t_re = 0; t_re < n_re_terms_slopes; t_re++) {
            int n_groups = data.re_n_groups_multi[t_re];
            int n_coefs = layout.re_n_coefs_multi[t_re];
            int re_start_t = layout.re_start_multi[t_re];
            bool is_corr_nc = !nc_L_flats.empty() && t_re < (int)nc_L_flats.size() && !nc_L_flats[t_re].empty();

            if (is_corr_nc) {
                // Correlated NC: chain rule from dLL/d(re_nc) back to
                // (z, log_sigma, raw_chol). grad_z and grad_raw slots are
                // contiguous in grad; log_sigma is scattered, so use temp.
                const auto& L_flat = nc_L_flats[t_re];
                const auto& sigmas = nc_sigmas_vec[t_re];
                int chol_start = layout.chol_re_start_multi[t_re];
                std::vector<double> g_log_sigma(n_coefs, 0.0);

                tulpa::chol_nc_chain_rule_add(
                    L_flat.data(), n_coefs, sigmas.data(),
                    &params[re_start_t], &params[chol_start],
                    &re_nc_flat_c[re_start_t], n_groups,
                    grad_re_slopes_lik[t_re].data(),
                    &grad[re_start_t],
                    g_log_sigma.data(),
                    &grad[chol_start]);

                for (int c = 0; c < n_coefs; c++) {
                    grad[layout.log_sigma_re_slopes[t_re][c]] += g_log_sigma[c];
                }
            } else if (slopes_nc) {
                // Uncorrelated NC: chain rule re = sigma * z
                for (int g = 0; g < n_groups; g++) {
                    for (int c = 0; c < n_coefs; c++) {
                        int idx = re_start_t + g * n_coefs + c;
                        double z_gc = params[idx];
                        double sigma_c = std::exp(params[layout.log_sigma_re_slopes[t_re][c]]);
                        double lik_grad = grad_re_slopes_lik[t_re][g * n_coefs + c];
                        grad[idx] += sigma_c * lik_grad;
                        grad[layout.log_sigma_re_slopes[t_re][c]] += z_gc * lik_grad * sigma_c;
                    }
                }
            } else {
                // Centered: direct
                for (int g = 0; g < n_groups; g++)
                    for (int c = 0; c < n_coefs; c++)
                        grad[re_start_t + g * n_coefs + c] += grad_re_slopes_lik[t_re][g * n_coefs + c];
            }
        }
    }

    // Fused log-posterior
    if (fuse_lp && !layout.has_zi) {
        *log_post_out = compute_log_post(params, data, layout);
    }
}

// g_gradient_mode defined earlier in file (before verify_gradient_runtime)

// Set global gradient mode (called at start of sampling)
void set_gradient_mode(GradientMode mode) {
    g_gradient_mode = mode;
}

GradientMode get_gradient_mode() {
    return g_gradient_mode;
}

void reset_grad_workspace_cache() {
    vec_grad_ws.cached_data_id = 0;
}

void compute_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    // Delegate to resolve_gradient_fn() — single source of truth for dispatch logic.
    // For hot paths (NUTS leapfrog), callers should resolve once via resolve_gradient_fn()
    // and reuse the pointer. This convenience wrapper is for cold-path callers.
    GradientFn fn = resolve_gradient_fn(g_gradient_mode, data, layout);
    fn(params, data, layout, grad, log_post_out);
}

// =====================================================================
// Generic multi-process gradient (central differences)
// Used by model packages (tulpaOcc, etc.) that plug in via LikelihoodSpec.
// =====================================================================

static void compute_gradient_generic_numerical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);
    auto ll_fn = spec->ll_double;
    const void* resp = data.model_response_data;

    auto log_post_fn = [&](const std::vector<double>& p) -> double {
        return tulpa::compute_log_post_generic<double>(p, data, layout, ll_fn, resp);
    };

    double f0 = log_post_fn(params);
    if (log_post_out) *log_post_out = f0;

    const double eps = 1e-6;
    const int n = static_cast<int>(params.size());
    std::vector<double> pw = params;

    for (int j = 0; j < n; j++) {
        pw[j] = params[j] + eps;
        double fp = log_post_fn(pw);
        pw[j] = params[j] - eps;
        double fm = log_post_fn(pw);
        pw[j] = params[j];
        grad[j] = (fp - fm) / (2.0 * eps);
    }
}

// =====================================================================
// Generic multi-process gradient via arena reverse-mode AD
// =====================================================================

static void compute_gradient_generic_arena(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    using namespace tulpa::arena;
    const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);

    ArenaScope scope;
    Arena* ar = scope.arena();
    std::vector<Var> params_ar = make_vars(ar, params);

    Var log_post = tulpa::compute_log_post_generic<Var>(
        params_ar, data, layout,
        spec->ll_arena, data.model_response_data);

    if (log_post_out) *log_post_out = log_post.val();
    log_post.backward();
    grad = get_adjoints(params_ar);
}

// =====================================================================
// Resolve gradient function pointer — SINGLE SOURCE OF TRUTH for dispatch.
// Called once at sampling start, returns a function pointer to eliminate
// per-call branching during leapfrog steps. compute_gradient() delegates here.
// =====================================================================

GradientFn resolve_gradient_fn(GradientMode mode, const ModelData& data, const ParamLayout& layout) {
    // Generic multi-process models: route through generic gradient
    if (data.n_processes > 0 && data.likelihood_spec != nullptr) {
        const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);
        // Use arena AD if model provides it, otherwise fall back to numerical
        if (spec->ll_arena != nullptr) {
            return &compute_gradient_generic_arena;
        }
        return &compute_gradient_generic_numerical;
    }

    // Explicit mode overrides
    if (mode == GradientMode::NUMERICAL)
        return &compute_gradient_numerical;
    if (mode == GradientMode::AUTODIFF_TAPE)
        return &compute_gradient_autodiff;
    if (mode == GradientMode::AUTODIFF_ARENA)
        return &compute_gradient_arena;
    if (mode == GradientMode::AUTODIFF_FWD)
        return &compute_gradient_forward;

    // AUTO or HANDCODED: use fastest available (H > A_r > A > N)
    if (can_use_analytical_gradient(data, layout)) {
        return &compute_gradient_analytical;
    }

    // ZI/OI supported in analytical (all non-exotic configs) and composite (exotic combos).
    // Specialized H-mode functions do NOT handle ZI — skip them entirely for ZI/OI models.
    // Simple ZI/OI already went to analytical above; exotic combos go to composite.
    // ZOIB has a pre-existing gradient bug in the H-mode ZOIB residual code (param 0
    // mismatch even in analytical). Route to A_r until fixed.
    if (data.zi_type == tulpa_zi::ZIType::ZOIB)
        return &compute_gradient_arena;
    if (layout.has_zi || layout.has_oi)
        return &compute_gradient_composite;

    // Early bail: crossed RE (without slopes) + exotic — composite doesn't handle
    // crossed intercept-only RE (multiple sigma_re terms with separate param blocks).
    // Slopes ARE handled (correlated + uncorrelated, NC + centered).
    bool has_exotic = layout.is_temporal_gp || layout.has_tvc ||
        layout.has_multiscale_temporal || layout.is_gp || layout.is_multiscale_gp ||
        layout.has_svc || layout.has_latent;
    if (has_exotic && !layout.has_re_slopes && data.n_re_terms > 1)
        return &compute_gradient_arena;

    // Early bail: TVC + latent — neither specialized function handles both.
    if (layout.has_tvc && layout.has_latent)
        return &compute_gradient_arena;

    // Early bail: ST interaction + AR1 temporal type — H and composite both use IID
    // fallback (tau*I) for AR1 but compute_log_post/log_post_impl return 0 prior
    // (type_ii_log_prior has no AR1 branch). Route to A_r for correctness.
    if (layout.has_spatiotemporal &&
        data.spatiotemporal_data.temporal_type == TemporalType::AR1)
        return &compute_gradient_arena;

    // Specialized H-mode functions: only dispatch if model has NO features
    // beyond what the function can handle. Otherwise fall through to composite.
    if (layout.is_hsgp && data.has_hsgp &&
        !layout.has_spatiotemporal &&
        !layout.has_latent && !layout.has_svc && !layout.has_re_slopes &&
        !layout.has_multiscale_temporal &&
        !layout.is_temporal_gp && !layout.has_tvc)
        return &compute_gradient_hsgp;
    // Collapsed GP: analytical gradient + numerical Laplace correction
    if (layout.is_gp_collapsed && data.has_gp && data.gp_collapsed)
        return &compute_gradient_gp_collapsed;
    // Collapsed ICAR/BYM2: analytical gradient + numerical Laplace correction
    if (layout.is_icar_collapsed || layout.is_bym2_collapsed)
        return &compute_gradient_icar_collapsed;
    if (layout.is_gp && data.has_gp && !layout.has_temporal && !layout.has_re_slopes
        && !layout.is_temporal_gp && !layout.has_tvc && !layout.has_multiscale_temporal
        && !layout.has_latent && !layout.has_svc && !layout.has_spatiotemporal)
        return &compute_gradient_gp_handcoded;
    if (layout.is_multiscale_gp && data.has_multiscale_gp && data.msgp_is_hsgp && !layout.has_re_slopes
        && !layout.has_latent && !layout.has_spatiotemporal && !layout.has_svc
        && !layout.is_temporal_gp && !layout.has_tvc && !layout.has_multiscale_temporal
        && !layout.has_temporal)
        return &compute_gradient_msgp_hsgp;
    if (layout.is_multiscale_gp && data.has_multiscale_gp && layout.has_temporal && !layout.has_re_slopes
        && !layout.is_temporal_gp && !layout.has_multiscale_temporal && !layout.has_tvc)
        return &compute_gradient_msgp_plus_temporal_handcoded;
    if (layout.is_multiscale_gp && data.has_multiscale_gp && !layout.has_re_slopes
        && !layout.is_temporal_gp && !layout.has_multiscale_temporal && !layout.has_tvc)
        return &compute_gradient_msgp_handcoded;
    if (layout.is_gp && layout.has_temporal && !layout.is_temporal_gp && !layout.has_re_slopes
        && !layout.has_tvc && !layout.has_multiscale_temporal
        && !layout.has_latent && !layout.has_svc && !layout.has_spatiotemporal)
        return &compute_gradient_gp_plus_temporal_handcoded;
    if (layout.has_svc && data.has_svc && data.svc_is_hsgp &&
        !layout.has_spatiotemporal &&
        !layout.has_latent && !layout.has_tvc &&
        !layout.is_temporal_gp && !layout.has_multiscale_temporal &&
        !layout.has_temporal && !layout.has_re_slopes)
        return &compute_gradient_svc_hsgp_handcoded;
    if (layout.has_svc && data.has_svc &&
        !layout.has_temporal && !layout.has_spatiotemporal &&
        !layout.has_latent && !layout.has_tvc && !layout.has_re_slopes &&
        !layout.is_temporal_gp && !layout.has_multiscale_temporal)
        return &compute_gradient_svc_handcoded;
    if (layout.has_tvc && data.has_tvc &&
        !layout.has_spatial && !layout.has_latent &&
        !layout.is_hsgp && !layout.is_gp && !layout.is_multiscale_gp &&
        !layout.has_svc && !layout.has_re_slopes)
        return &compute_gradient_tvc_handcoded;
    // ST interaction: route AR1 temporal type to A_r (H uses IID fallback which is wrong;
    // type_ii_log_prior and log_post_impl both return 0 prior for AR1, H adds -tau*delta)
    if (layout.has_spatiotemporal && !layout.is_st_gp &&
        !layout.is_gp && !layout.is_multiscale_gp && !layout.is_hsgp &&
        !layout.has_latent &&
        data.spatiotemporal_data.type != STType::NONE &&
        layout.st_delta_start >= 0 && layout.log_tau_st_idx >= 0 &&
        data.spatiotemporal_data.temporal_type != TemporalType::AR1)
        return &compute_gradient_spatiotemporal_handcoded;
    if (layout.is_temporal_gp && layout.has_temporal &&
        !layout.is_gp && !layout.is_multiscale_gp && !layout.is_hsgp &&
        !layout.has_spatial && !layout.has_latent && !layout.has_svc &&
        !layout.has_re_slopes &&
        data.temporal_gp_data.cov_type == tulpa_temporal_gp::TemporalCovType::EXPONENTIAL)
        return &compute_gradient_temporal_gp_handcoded;
    if (layout.has_multiscale_temporal && !layout.is_gp && !layout.is_multiscale_gp &&
        !layout.is_hsgp && !layout.has_svc && !layout.has_tvc &&
        !layout.has_latent && !layout.has_spatiotemporal &&
        !layout.has_spatial && !layout.has_re_slopes)
        return &compute_gradient_ms_temporal_handcoded;
    if (layout.has_latent && data.latent_n_factors > 0 &&
        !layout.has_spatial && !layout.has_temporal && !layout.has_svc &&
        !layout.is_hsgp && !layout.is_gp && !layout.is_multiscale_gp &&
        !layout.has_re_slopes)
        return &compute_gradient_latent_handcoded;

    // Composite H-mode: catch-all for exotic multi-feature combinations
    // that no specialized function above handles (e.g., HSGP+TVC, SVC+RW1,
    // latent+spatial, crossed+slopes). Slower than specialized but faster than A_r.
    // ST+AR1 also falls through here and routes to A_r below (via !NONE + AR1 guard above).
    return &compute_gradient_composite;
}

// =====================================================================
// Dual averaging for step size adaptation
// =====================================================================

DualAveraging::DualAveraging(double epsilon_init, int n_params, double target_boost)
  : mu(std::log(10.0 * epsilon_init)), log_epsilon_bar(std::log(epsilon_init)), H_bar(0.0),
    gamma(0.05), t0(10.0), kappa(0.75),
    target_accept(compute_target(n_params, target_boost)), m(0) {}

double DualAveraging::update(double alpha) {
  m++;
  double w = 1.0 / (m + t0);
  H_bar = (1.0 - w) * H_bar + w * (target_accept - alpha);
  double log_epsilon = mu - std::sqrt((double)m) / gamma * H_bar;
  // Clamp log_epsilon to reasonable range
  // Lower bound: exp(-14) ≈ 8e-7, Upper bound: exp(2) ≈ 7.4
  log_epsilon = std::max(-14.0, std::min(log_epsilon, 2.0));
  double epsilon = std::exp(log_epsilon);
  double m_w = std::pow((double)m, -kappa);
  log_epsilon_bar = m_w * log_epsilon + (1.0 - m_w) * log_epsilon_bar;
  return epsilon;
}

double DualAveraging::final_epsilon() const {
  return std::exp(log_epsilon_bar);
}

// WelfordStats defined in hmc_sampler.h — not duplicated here

// =====================================================================
// Leapfrog integrator
// =====================================================================

// Unified leapfrog step: identity mass when inv_mass is nullptr
LeapfrogResult leapfrog_step(
    const std::vector<double>& q,
    const std::vector<double>& p,
    double epsilon,
    const ModelData& data,
    const ParamLayout& layout,
    const double* inv_mass
) {
  int n = q.size();
  LeapfrogResult result;
  result.q = q;
  result.p = p;
  result.divergent = false;

  std::vector<double> grad(n);

  // Half step for momentum
  compute_gradient(result.q, data, layout, grad);
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  // Full step for position (scaled by inverse mass if provided)
  if (inv_mass) {
    for (int i = 0; i < n; i++) {
      result.q[i] += epsilon * inv_mass[i] * result.p[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      result.q[i] += epsilon * result.p[i];
    }
  }

  // Half step for momentum (fused gradient + log_prob)
  compute_gradient(result.q, data, layout, grad, &result.log_prob);
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  }

  for (int i = 0; i < n; i++) {
    if (std::abs(result.q[i]) > 1e10 || !std::isfinite(result.q[i])) {
      result.divergent = true;
      break;
    }
  }

  return result;
}

// =====================================================================
// Find reasonable initial step size
// =====================================================================

// Compute diagonal mass matrix from gradient magnitudes
std::vector<double> compute_diagonal_mass(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout
) {
  int n = q.size();
  std::vector<double> grad(n);
  compute_gradient(q, data, layout, grad);

  std::vector<double> mass(n);
  for (int i = 0; i < n; i++) {
    double abs_grad = std::abs(grad[i]);
    mass[i] = std::max(1.0, std::min(abs_grad, 1000.0));
  }

  return mass;
}

// =====================================================================
// Unified find_reasonable_epsilon: handles identity, diagonal, and dense mass
// Stan-style algorithm: start at epsilon=1, double or halve until
// acceptance probability crosses 0.5
// =====================================================================

// Helper: compute kinetic energy with mass matrix
static inline double kinetic_energy_mass(
    const double* p, int n,
    const double* inv_mass,          // nullptr = identity
    const DenseMassMatrix* dense_mass // nullptr = not dense
) {
    if (dense_mass) return dense_mass->kinetic_energy(p);
    if (inv_mass) {
        double ke = 0.0;
        for (int i = 0; i < n; i++) ke += p[i] * p[i] * inv_mass[i];
        return 0.5 * ke;
    }
    return 0.5 * tulpa_linalg::norm_squared(p, n);
}

// Helper: single leapfrog step respecting mass matrix type
static inline LeapfrogResult leapfrog_for_epsilon(
    const std::vector<double>& q, const std::vector<double>& p,
    double epsilon, const ModelData& data, const ParamLayout& layout,
    const double* inv_mass, const DenseMassMatrix* dense_mass
) {
    int n = q.size();
    LeapfrogResult result;
    result.q = q;
    result.p = p;
    result.divergent = false;

    std::vector<double> grad(n);
    compute_gradient(result.q, data, layout, grad);
    for (int i = 0; i < n; i++) result.p[i] += 0.5 * epsilon * grad[i];

    // Full step for position: q += eps * M^{-1} * p
    if (dense_mass) {
        std::vector<double> Mp(n);
        dense_mass->inv_mass_times_p(result.p.data(), Mp.data());
        for (int i = 0; i < n; i++) result.q[i] += epsilon * Mp[i];
    } else if (inv_mass) {
        for (int i = 0; i < n; i++) result.q[i] += epsilon * inv_mass[i] * result.p[i];
    } else {
        for (int i = 0; i < n; i++) result.q[i] += epsilon * result.p[i];
    }

    compute_gradient(result.q, data, layout, grad, &result.log_prob);
    for (int i = 0; i < n; i++) result.p[i] += 0.5 * epsilon * grad[i];

    if (!std::isfinite(result.log_prob)) result.divergent = true;
    for (int i = 0; i < n; i++) {
        if (std::abs(result.q[i]) > 1e10 || !std::isfinite(result.q[i])) {
            result.divergent = true;
            break;
        }
    }
    return result;
}

// Unified find_reasonable_epsilon: works with identity, diagonal, or dense mass.
// inv_mass_diag = nullptr for identity/dense, mass_dense = nullptr for identity/diagonal.
double find_reasonable_epsilon_impl(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const double* inv_mass_diag,
    const DenseMassMatrix* mass_dense
) {
    int n = q.size();
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> p(n);

    // Sample momentum based on mass type
    if (mass_dense) {
        const_cast<DenseMassMatrix*>(mass_dense)->sample_momentum(p.data(), rng);
    } else if (inv_mass_diag) {
        for (int i = 0; i < n; i++) p[i] = normal(rng) / std::sqrt(inv_mass_diag[i]);
    } else {
        for (int i = 0; i < n; i++) p[i] = normal(rng);
    }

    double log_prob_init;
    std::vector<double> grad_init(n);
    compute_gradient(q, data, layout, grad_init, &log_prob_init);
    double H_init = -log_prob_init + kinetic_energy_mass(p.data(), n, inv_mass_diag, mass_dense);

    double epsilon = 1.0;
    auto lf = leapfrog_for_epsilon(q, p, epsilon, data, layout, inv_mass_diag, mass_dense);

    // For dense mass, leapfrog_for_epsilon may not compute log_prob correctly;
    // recompute if needed
    double lp_first = lf.log_prob;
    if (mass_dense) {
        std::vector<double> grad_tmp(n);
        compute_gradient(lf.q, data, layout, grad_tmp, &lp_first);
    }
    double delta_H = (-lp_first + kinetic_energy_mass(lf.p.data(), n, inv_mass_diag, mass_dense)) - H_init;

    int direction = (!std::isfinite(delta_H) || delta_H > std::log(2.0)) ? -1 : 1;
    for (int iter = 0; iter < 50; iter++) {
        epsilon *= (direction == 1) ? 2.0 : 0.5;
        if (epsilon < 1e-10 || epsilon > 1e5) break;
        lf = leapfrog_for_epsilon(q, p, epsilon, data, layout, inv_mass_diag, mass_dense);
        double lp_try = lf.log_prob;
        if (mass_dense) {
            std::vector<double> grad_tmp(n);
            compute_gradient(lf.q, data, layout, grad_tmp, &lp_try);
        }
        if (!std::isfinite(lp_try)) { if (direction == 1) break; continue; }
        delta_H = (-lp_try + kinetic_energy_mass(lf.p.data(), n, inv_mass_diag, mass_dense)) - H_init;
        if (direction == 1 && (!std::isfinite(delta_H) || delta_H > std::log(2.0))) break;
        if (direction == -1 && std::isfinite(delta_H) && delta_H < std::log(2.0)) break;
    }
    return std::max(1e-10, std::min(epsilon, 1e3));
}

// Backward-compatible overloads (delegate to impl)
double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng
) {
    return find_reasonable_epsilon_impl(q, data, layout, rng, nullptr, nullptr);
}

double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const std::vector<double>& inv_mass
) {
    return find_reasonable_epsilon_impl(q, data, layout, rng, inv_mass.data(), nullptr);
}

double find_reasonable_epsilon_dense(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const DenseMassMatrix& mass
) {
    return find_reasonable_epsilon_impl(q, data, layout, rng, nullptr, &mass);
}

// =====================================================================
// NUTS (No-U-Turn Sampler) helper functions
// =====================================================================

double nuts_log_sum_exp(double a, double b) {
  double m = std::max(a, b);
  if (!std::isfinite(m)) return m;
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

double nuts_compute_hamiltonian(double log_prob, const std::vector<double>& p,
                                const std::vector<double>& inv_mass, int n) {
  double kinetic = 0.0;
  for (int i = 0; i < n; i++) {
    kinetic += p[i] * p[i] * inv_mass[i];
  }
  return -log_prob + 0.5 * kinetic;
}

bool nuts_check_uturn(const std::vector<double>& q_minus, const std::vector<double>& q_plus,
                      const std::vector<double>& p_minus, const std::vector<double>& p_plus,
                      const std::vector<double>& inv_mass, int n) {
  // Generalized U-turn criterion (Betancourt 2017, Section 3.2)
  // Check both directions: (q+ - q-) . (M^-1 p-) and (q+ - q-) . (M^-1 p+)
  double dot_fwd = 0.0, dot_bwd = 0.0;
  for (int i = 0; i < n; i++) {
    double dq = q_plus[i] - q_minus[i];
    dot_fwd += dq * (inv_mass[i] * p_plus[i]);
    dot_bwd += dq * (inv_mass[i] * p_minus[i]);
  }
  return (dot_fwd < 0.0) || (dot_bwd < 0.0);
}

LeapfrogResultWithGrad leapfrog_step_with_grad(
    const std::vector<double>& q, const std::vector<double>& p,
    const std::vector<double>& grad,
    double epsilon, const std::vector<double>& inv_mass,
    bool use_mass, const ModelData& data, const ParamLayout& layout) {

  int n = q.size();
  LeapfrogResultWithGrad result;
  result.q = q;
  result.p = p;
  result.grad.resize(n);
  result.divergent = false;

  // Half step for momentum using provided gradient
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * grad[i];
  }

  // Full step for position
  if (use_mass) {
    for (int i = 0; i < n; i++) {
      result.q[i] += epsilon * inv_mass[i] * result.p[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      result.q[i] += epsilon * result.p[i];
    }
  }

  // Compute gradient and log_prob at new position (fused: single O(N) pass)
  compute_gradient(result.q, data, layout, result.grad, &result.log_prob);

  // Half step for momentum using new gradient
  for (int i = 0; i < n; i++) {
    result.p[i] += 0.5 * epsilon * result.grad[i];
  }

  // Check for divergence
  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  }
  for (int i = 0; i < n; i++) {
    if (std::abs(result.q[i]) > 1e10 || !std::isfinite(result.q[i])) {
      result.divergent = true;
      break;
    }
  }

  return result;
}

// =====================================================================
// Optimized NUTS: zero-allocation infrastructure
// =====================================================================

// Pointer-based Hamiltonian (avoids std::vector overhead)
double nuts_compute_hamiltonian_fast(double log_prob, const double* p,
                                     const DenseMassMatrix& mass, int n) {
  return -log_prob + mass.kinetic_energy(p);
}

// Pointer-based U-turn check
// scratch: temporary buffer of size n (for dense matvec result)
bool nuts_check_uturn_fast(const double* q_minus, const double* q_plus,
                           const double* p_minus, const double* p_plus,
                           const DenseMassMatrix& mass, double* scratch, int n) {
  if (mass.type == MassMatrixType::BLOCK_DIAG && mass.adapted) {
    // Block-diagonal: use inv_mass_times_p which handles blocks correctly
    mass.inv_mass_times_p(p_plus, scratch);
    double dot_fwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_fwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    mass.inv_mass_times_p(p_minus, scratch);
    double dot_bwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_bwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    return (dot_fwd < 0.0) || (dot_bwd < 0.0);
  } else if (mass.type == MassMatrixType::DIAG || !mass.adapted) {
    // Diagonal path (fast, unrolled)
    double dot_fwd = 0.0, dot_bwd = 0.0;
    const double* inv_mass = mass.inv_mass_diag.data();
    int i = 0;
    for (; i + 3 < n; i += 4) {
      double dq0 = q_plus[i]   - q_minus[i];
      double dq1 = q_plus[i+1] - q_minus[i+1];
      double dq2 = q_plus[i+2] - q_minus[i+2];
      double dq3 = q_plus[i+3] - q_minus[i+3];
      dot_fwd += dq0 * (inv_mass[i]   * p_plus[i])
               + dq1 * (inv_mass[i+1] * p_plus[i+1])
               + dq2 * (inv_mass[i+2] * p_plus[i+2])
               + dq3 * (inv_mass[i+3] * p_plus[i+3]);
      dot_bwd += dq0 * (inv_mass[i]   * p_minus[i])
               + dq1 * (inv_mass[i+1] * p_minus[i+1])
               + dq2 * (inv_mass[i+2] * p_minus[i+2])
               + dq3 * (inv_mass[i+3] * p_minus[i+3]);
    }
    for (; i < n; i++) {
      double dq = q_plus[i] - q_minus[i];
      dot_fwd += dq * (inv_mass[i] * p_plus[i]);
      dot_bwd += dq * (inv_mass[i] * p_minus[i]);
    }
    return (dot_fwd < 0.0) || (dot_bwd < 0.0);
  } else {
    // Dense path: compute (q+ - q-) . (C * p+) and (q+ - q-) . (C * p-)
    // Use scratch for C * p
    mass.inv_mass_times_p(p_plus, scratch);
    double dot_fwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_fwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    mass.inv_mass_times_p(p_minus, scratch);
    double dot_bwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_bwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    return (dot_fwd < 0.0) || (dot_bwd < 0.0);
  }
}

// In-place leapfrog step operating on a workspace slot
// Mutates q, p, grad in the slot directly — zero heap allocation
LeapfrogInPlaceResult leapfrog_step_inplace(
    NUTSWorkspace& ws, int slot, double epsilon,
    const DenseMassMatrix& mass,
    const ModelData& data, const ParamLayout& layout) {

  double* q = ws.q_at(slot);
  double* p = ws.p_at(slot);
  double* grad = ws.grad_at(slot);
  int n = ws.n;

  LeapfrogInPlaceResult result;
  result.divergent = false;

  // Half step for momentum using current gradient
  tulpa_linalg::axpy(0.5 * epsilon, grad, p, n);

  // Full step for position: q += eps * C * p
  if (!mass.adapted) {
    // Identity mass: q += eps * p
    tulpa_linalg::axpy(epsilon, p, q, n);
  } else if (mass.type == MassMatrixType::BLOCK_DIAG) {
    // Block-diagonal: diagonal for non-block params, dense for block params
    // First pass: diagonal for all params
    tulpa_linalg::axpy_weighted(epsilon, mass.inv_mass_diag.data(), p, q, n);
    // Second pass: overwrite block params with dense contribution
    for (const auto& blk : mass.blocks) {
      if (blk.adapted) {
        double tmp[4];
        blk.matvec(p, tmp);
        for (int i = 0; i < blk.size; i++) {
          // Undo diagonal contribution, apply block contribution
          q[blk.start + i] += epsilon * (tmp[i] - mass.inv_mass_diag[blk.start + i] * p[blk.start + i]);
        }
      }
    }
  } else if (mass.type == MassMatrixType::DIAG) {
    // Diagonal: q[i] += eps * inv_mass[i] * p[i]
    tulpa_linalg::axpy_weighted(epsilon, mass.inv_mass_diag.data(), p, q, n);
  } else {
    // Dense: q += eps * C * p  (Eigen BLAS for n>=16, scalar fallback below)
    if (n >= 16) {
      Eigen::Map<const Eigen::MatrixXd> Am(mass.inv_mass_dense.data(), n, n);
      Eigen::Map<const Eigen::VectorXd> pv(p, n);
      Eigen::Map<Eigen::VectorXd> qv(q, n);
      qv.noalias() += epsilon * (Am.selfadjointView<Eigen::Lower>() * pv);
    } else {
      tulpa_linalg::axpy_matvec(epsilon, mass.inv_mass_dense.data(), p, q, n);
    }
  }
  // Precision/Kronecker blocks: undo base contribution, apply block M^{-1}
  if (mass.precision_block.active) {
    const auto& pb = mass.precision_block;
    std::vector<double> tmp(pb.size);
    pb.matvec(p, tmp.data());
    for (int i = 0; i < pb.size; i++) {
      // Undo the diagonal (or dense) contribution that was already applied
      q[pb.start + i] -= epsilon * mass.inv_mass_diag[pb.start + i] * p[pb.start + i];
      // Apply precision block contribution
      q[pb.start + i] += epsilon * tmp[i];
    }
  }
  if (mass.kronecker_block.active) {
    const auto& kb = mass.kronecker_block;
    int ST = kb.S * kb.T;
    std::vector<double> tmp(ST);
    kb.matvec(p, tmp.data());
    for (int i = 0; i < ST; i++) {
      q[kb.start + i] -= epsilon * mass.inv_mass_diag[kb.start + i] * p[kb.start + i];
      q[kb.start + i] += epsilon * tmp[i];
    }
  }

  // Compute gradient + log_prob at new position (fused: single O(N) pass)
  // Uses pre-resolved function pointer to skip 15+ branch dispatch per leapfrog step
  std::memcpy(ws.params_buf.data(), q, n * sizeof(double));
  ws.gradient_fn(ws.params_buf, data, layout, ws.grad_buf, &ws.logp_at(slot));
  std::memcpy(grad, ws.grad_buf.data(), n * sizeof(double));
  result.log_prob = ws.logp_at(slot);

  // Half step for momentum using new gradient
  tulpa_linalg::axpy(0.5 * epsilon, grad, p, n);

  // Divergence check (skip param scan if log_prob already non-finite)
  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  } else {
    for (int i = 0; i < n; i++) {
      if (std::abs(q[i]) > 1e10 || !std::isfinite(q[i])) {
        result.divergent = true;
        break;
      }
    }
  }

  return result;
}

// Zero-allocation recursive tree builder
// Uses workspace slot indices instead of vector copies
TreeStats build_tree_fast(
    NUTSWorkspace& ws, int input_slot, int direction, int depth,
    double epsilon, const DenseMassMatrix& mass,
    double H0, double delta_max,
    const ModelData& data, const ParamLayout& layout,
    std::mt19937& rng) {

  int n = ws.n;
  TreeStats stats;

  if (depth == 0) {
    stats.init_vectors(n);  // Pre-allocate U-turn vectors (avoids per-leaf heap allocation)
    // Base case: single leapfrog step in-place on input_slot
    LeapfrogInPlaceResult lf = leapfrog_step_inplace(
      ws, input_slot, direction * epsilon, mass, data, layout
    );

    double H_new = nuts_compute_hamiltonian_fast(
      lf.log_prob, ws.p_at(input_slot), mass, n
    );
    double delta_H = H_new - H0;

    // Both endpoints are the same slot (single node)
    stats.left_slot = input_slot;
    stats.right_slot = input_slot;
    stats.proposal_slot = input_slot;
    stats.log_prob_proposal = lf.log_prob;

    // Multinomial weight: log(weight) = H0 - H_new (relative, Stan-style)
    stats.sum_log_weight = H0 - H_new;

    // Divergence check
    stats.divergent = lf.divergent || (delta_H > delta_max);
    stats.stop = stats.divergent;
    stats.n_valid = stats.divergent ? 0 : 1;

    // Acceptance statistic
    double accept_stat = std::min(1.0, std::exp(-delta_H));
    if (!std::isfinite(accept_stat)) accept_stat = 0.0;
    stats.sum_accept_prob = accept_stat;
    stats.n_leapfrog = 1;

    // Generalized U-turn: track rho, p_sharp, p at this leaf
    const double* p_ptr = ws.p_at(input_slot);

    std::memcpy(stats.rho.data(), p_ptr, n * sizeof(double));
    std::memcpy(stats.p_beg.data(), p_ptr, n * sizeof(double));
    std::memcpy(stats.p_end.data(), p_ptr, n * sizeof(double));

    // p_sharp = M^{-1} * p  — use full mass matrix for U-turn criterion.
    // Dense mass captures correlation structure; using diagonal p_sharp would
    // make NUTS unable to detect turns in correlated directions, causing
    // trees to grow to max depth on correlated posteriors (slopes, BYM2, HSGP).
    mass.inv_mass_times_p(p_ptr, stats.p_sharp_beg.data());
    std::memcpy(stats.p_sharp_end.data(), stats.p_sharp_beg.data(), n * sizeof(double));

    return stats;
  }

  // Recursive case: build inner subtree
  TreeStats inner = build_tree_fast(
    ws, input_slot, direction, depth - 1,
    epsilon, mass, H0, delta_max, data, layout, rng
  );

  stats = std::move(inner);

  if (stats.stop) return stats;

  // Copy the appropriate endpoint to a fresh slot for outer start
  int start_slot = ws.alloc_slot();
  if (start_slot < 0) {
    stats.stop = true;
    return stats;
  }
  if (direction == 1) {
    ws.copy_node(start_slot, stats.right_slot);
  } else {
    ws.copy_node(start_slot, stats.left_slot);
  }

  // Build outer subtree from the copy
  TreeStats outer = build_tree_fast(
    ws, start_slot, direction, depth - 1,
    epsilon, mass, H0, delta_max, data, layout, rng
  );

  // Combine results
  stats.n_leapfrog += outer.n_leapfrog;
  stats.sum_accept_prob += outer.sum_accept_prob;
  stats.divergent = stats.divergent || outer.divergent;

  // Multinomial sampling
  double new_sum_log_weight = nuts_log_sum_exp(stats.sum_log_weight, outer.sum_log_weight);
  double accept_prob_outer = std::exp(outer.sum_log_weight - new_sum_log_weight);
  if (!std::isfinite(accept_prob_outer)) accept_prob_outer = 0.0;

  std::uniform_real_distribution<double> unif(0.0, 1.0);
  if (unif(rng) < accept_prob_outer) {
    stats.proposal_slot = outer.proposal_slot;
    stats.log_prob_proposal = outer.log_prob_proposal;
  }

  stats.sum_log_weight = new_sum_log_weight;
  stats.n_valid = stats.n_valid + outer.n_valid;

  // === SAVE BOUNDARY VALUES AS COPIES BEFORE MOVES ===
  // "init" = inner (built first), "final" = outer (extends from init)
  // These copies are needed because moves below invalidate the originals
  // Uses pre-allocated depth-indexed merge buffers (no per-merge heap allocation)
  double* p_init_end = ws.merge_buf(depth, NUTSWorkspace::MERGE_P_INIT_END);
  double* p_sharp_init_end = ws.merge_buf(depth, NUTSWorkspace::MERGE_PSHARP_INIT_END);
  double* rho_init = ws.merge_buf(depth, NUTSWorkspace::MERGE_RHO_INIT);
  double* rho_check = ws.merge_buf(depth, NUTSWorkspace::MERGE_RHO_CHECK);

  const double* src_p = (direction == 1) ? stats.p_end.data() : stats.p_beg.data();
  std::memcpy(p_init_end, src_p, n * sizeof(double));
  const double* src_ps = (direction == 1) ? stats.p_sharp_end.data() : stats.p_sharp_beg.data();
  std::memcpy(p_sharp_init_end, src_ps, n * sizeof(double));
  std::memcpy(rho_init, stats.rho.data(), n * sizeof(double));

  // Pointers to final's boundary (safe: these outer members are NOT moved)
  const double* p_final_beg_ptr = (direction == 1) ? outer.p_beg.data() : outer.p_end.data();
  const double* p_sharp_final_beg_ptr = (direction == 1) ? outer.p_sharp_beg.data() : outer.p_sharp_end.data();

  // === UPDATE TREE ENDPOINTS (moves invalidate init's boundary refs) ===
  if (direction == 1) {
    stats.right_slot = outer.right_slot;
    stats.p_sharp_end = std::move(outer.p_sharp_end);
    stats.p_end = std::move(outer.p_end);
  } else {
    stats.left_slot = outer.left_slot;
    stats.p_sharp_beg = std::move(outer.p_sharp_beg);
    stats.p_beg = std::move(outer.p_beg);
  }

  // Combine rho = rho_init + rho_final
  for (int i = 0; i < n; i++) {
    stats.rho[i] = rho_init[i] + outer.rho[i];
  }

  // === GENERALIZED U-TURN CRITERION (Stan-style, 3 juncture checks) ===
  // Check 1: Full merged trajectory — merged endpoints vs merged rho
  bool persist = compute_criterion(stats.p_sharp_beg.data(), stats.p_sharp_end.data(),
                                   stats.rho.data(), n);

  // After update, far endpoints depend on direction:
  // direction == 1: init's far = stats.beg (left, unchanged), final's far = stats.end (right, updated)
  // direction == -1: init's far = stats.end (right, unchanged), final's far = stats.beg (left, updated)
  const double* init_far_psharp = (direction == 1) ? stats.p_sharp_beg.data() : stats.p_sharp_end.data();
  const double* final_far_psharp = (direction == 1) ? stats.p_sharp_end.data() : stats.p_sharp_beg.data();

  // Check 2: Init subtree + seam from final (rho = rho_init + p_final_beg)
  // Fused: rho construction + dot product in single O(n) pass
  persist &= compute_criterion_fused(init_far_psharp, p_sharp_final_beg_ptr,
                                     rho_init, p_final_beg_ptr, rho_check, n);

  // Check 3: Seam from init + final subtree (rho = rho_final + p_init_end)
  persist &= compute_criterion_fused(p_sharp_init_end, final_far_psharp,
                                     outer.rho.data(), p_init_end, rho_check, n);

  stats.stop = outer.stop || !persist;

  return stats;
}

// =====================================================================
// SoftAbs per-trajectory metric (Riemannian-like divergence retry)
// =====================================================================

void compute_hessian_finite_diff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& hessian,
    double h
) {
  int p = static_cast<int>(params.size());
  hessian.resize(static_cast<size_t>(p) * p);

  // Base gradient
  std::vector<double> grad_base(p);
  compute_gradient(params, data, layout, grad_base);

  // Perturb each parameter and compute column of Hessian
  std::vector<double> params_pert = params;
  std::vector<double> grad_pert(p);
  for (int i = 0; i < p; i++) {
    double orig = params_pert[i];
    double hi = std::max(h, h * std::abs(orig));  // relative step for large params
    params_pert[i] = orig + hi;
    compute_gradient(params_pert, data, layout, grad_pert);
    for (int j = 0; j < p; j++) {
      hessian[static_cast<size_t>(i) * p + j] = (grad_pert[j] - grad_base[j]) / hi;
    }
    params_pert[i] = orig;
  }

  // Symmetrize: H = 0.5 * (H + H^T)
  for (int i = 0; i < p; i++) {
    for (int j = i + 1; j < p; j++) {
      double avg = 0.5 * (hessian[static_cast<size_t>(i) * p + j] +
                          hessian[static_cast<size_t>(j) * p + i]);
      hessian[static_cast<size_t>(i) * p + j] = avg;
      hessian[static_cast<size_t>(j) * p + i] = avg;
    }
  }
}

bool compute_softabs_metric(
    const std::vector<double>& neg_hessian,
    int p,
    double alpha,
    std::vector<double>& G_inv,
    std::vector<double>& L_G_inv
) {
  // Map to Eigen (column-major)
  Eigen::Map<const Eigen::MatrixXd> H_map(neg_hessian.data(), p, p);

  // Eigendecomposition (symmetric)
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(H_map);
  if (eigen.info() != Eigen::Success) return false;

  const auto& lambdas = eigen.eigenvalues();
  const auto& Q = eigen.eigenvectors();

  // Apply SoftAbs: f(λ) = λ * coth(α * λ)
  // Properties: always positive, f(|λ|>>0) ≈ |λ|, f(0) → 1/α
  Eigen::VectorXd softabs_inv_eig(p);
  for (int i = 0; i < p; i++) {
    double lam = lambdas(i);
    double al = alpha * lam;
    double f;
    if (std::abs(al) > 20.0) {
      f = std::abs(lam);
    } else if (std::abs(al) < 1e-10) {
      f = 1.0 / alpha;
    } else {
      f = lam * std::cosh(al) / std::sinh(al);
    }
    f = std::max(f, 1e-6);  // floor to ensure positive definiteness
    softabs_inv_eig(i) = 1.0 / f;
  }

  // Reconstruct G^{-1} = Q diag(1/f(λ)) Q^T
  Eigen::MatrixXd G_inv_mat = Q * softabs_inv_eig.asDiagonal() * Q.transpose();

  // Cholesky of G^{-1}
  Eigen::LLT<Eigen::MatrixXd> llt(G_inv_mat);
  if (llt.info() != Eigen::Success) return false;
  Eigen::MatrixXd L_mat = llt.matrixL();

  // Copy to output (column-major)
  G_inv.resize(static_cast<size_t>(p) * p);
  L_G_inv.resize(static_cast<size_t>(p) * p);
  Eigen::Map<Eigen::MatrixXd>(G_inv.data(), p, p) = G_inv_mat;
  Eigen::Map<Eigen::MatrixXd>(L_G_inv.data(), p, p) = L_mat;

  return true;
}

// =====================================================================
// Mass matrix selection and initialization helpers
// =====================================================================

// DENSE_MAX_PARAMS and MassMatrixConfig now defined in hmc_sampler.h

// Select mass matrix type (AUTO resolution, block detection, DENSE override)
// and initialize the DenseMassMatrix object.
MassMatrixConfig select_and_init_mass_matrix(
    DenseMassMatrix& mass,
    const ModelData& data,
    const ParamLayout& layout,
    int n_params,
    MassMatrixType metric_type,
    bool verbose
) {
  // Mass matrix adaptation
  // Resolve AUTO metric: DIAG default with DIAG→DENSE identity recovery.
  //
  // Key insight: adapted DENSE mass (where adapted=true) incurs O(n²) per leapfrog
  // step for matvec/kinetic/p_sharp operations. For n=54 (RE models), this is 22x
  // slower per step than identity mass (adapted=false). DIAG→identity recovery gives
  // fast per-step execution while still finding correct epsilon via dual averaging.
  //
  // Strategy: Start with DIAG for most models. If DIAG fails catastrophically
  // (epsilon > 2.0 at warmup end), recover to DENSE identity (adapted=false).
  // This gives:
  //   - O(n) per-step cost (identity path)
  //   - Correct epsilon from find_reasonable_epsilon
  //   - No divergences (epsilon small enough for the geometry)
  //
  // Only genuinely complex posteriors (correlated slopes, BYM2, GP, SVC) start
  // with DENSE, where the adapted covariance actually helps sampling efficiency
  // enough to justify the O(n²) per-step cost.
  //
  // HISTORY:
  // 2026-02-27: has_re/has_temporal in needs_dense → 1.25s PG+RE (adapted dense)
  // 2026-02-28: TVC gradient fix → removed TVC
  // 2026-03-03: Removed has_re/has_temporal → DIAG + recovery. PG+RE: 0.8s
  //   (identity dense, 22x faster per step). Adapted dense measured at 17.7s
  //   due to O(n²) per-step cost for n=54.
  MassMatrixType effective_metric = metric_type;
  bool auto_selected_diag = false;
  // Block specs for BLOCK_DIAG: (start_index, block_size) pairs
  std::vector<std::pair<int,int>> block_specs;

  if (effective_metric == MassMatrixType::AUTO) {
    // Build block_specs from param layout: detect pairs of correlated hyperparameters.
    // Each block captures a small dense correlation (2-4 params) at O(block²) cost,
    // avoiding full O(n²) DENSE while handling the key correlations DIAG misses.
    // NOTE: temporal_gp excluded — DIAG is faster (2.11s vs 2.39s, 5 seeds Bin+GP_t).
    // NOTE: HSGP excluded from AUTO blocks — DIAG is faster for HSGP-only (29k LF
    // vs 39k LF), and HSGP+temporal uses full DENSE (tested BLOCK_DIAG 2026-03-10:
    // worse performance and more divergences than DENSE).
    if (layout.is_bym2 && layout.log_sigma_bym2_idx >= 0 &&
        layout.logit_rho_bym2_idx == layout.log_sigma_bym2_idx + 1) {
      block_specs.push_back({layout.log_sigma_bym2_idx, 2});
    }
    if (layout.is_gp && layout.log_sigma2_gp_idx >= 0 &&
        layout.log_phi_gp_idx == layout.log_sigma2_gp_idx + 1) {
      block_specs.push_back({layout.log_sigma2_gp_idx, 2});
    }
    if (layout.is_multiscale_gp) {
      if (layout.log_sigma2_gp_local_idx >= 0 &&
          layout.log_phi_gp_local_idx == layout.log_sigma2_gp_local_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_local_idx, 2});
      }
      if (layout.log_sigma2_gp_regional_idx >= 0 &&
          layout.log_phi_gp_regional_idx == layout.log_sigma2_gp_regional_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_regional_idx, 2});
      }
    }
    if (layout.has_svc && layout.log_sigma2_svc_start >= 0 &&
        layout.log_phi_svc_start >= 0) {
      int n_svc = layout.log_sigma2_svc_end - layout.log_sigma2_svc_start;
      for (int t = 0; t < n_svc; t++) {
        int sigma_idx = layout.log_sigma2_svc_start + t;
        int phi_idx = layout.log_phi_svc_start + t;
        if (phi_idx == sigma_idx + n_svc) {
          // SVC sigma2 and phi are in separate contiguous blocks; can't form a 2x2 block
          // unless they're consecutive. Skip non-consecutive pairs.
        }
      }
      // SVC layout: [sigma2_0, sigma2_1, ..., phi_0, phi_1, ...]
      // These aren't consecutive pairs, so we'd need per-SVC blocks of non-contiguous params.
      // For now, only handle n_svc=1 where sigma2 and phi are adjacent:
      if (n_svc == 1 && layout.log_phi_svc_start == layout.log_sigma2_svc_start + 1) {
        block_specs.push_back({layout.log_sigma2_svc_start, 2});
      }
    }
    if (layout.is_st_gp && layout.log_phi_st_space_idx >= 0 &&
        layout.log_phi_st_time_idx == layout.log_phi_st_space_idx + 1) {
      block_specs.push_back({layout.log_phi_st_space_idx, 2});
    }
    if (layout.has_multiscale_temporal) {
      // Multiscale temporal hyperparams form a natural block (3-4 params):
      // log_sigma2_trend, log_sigma2_seasonal, log_sigma2_short [, logit_rho_short]
      // These are correlated but the temporal effects themselves (phi) are not
      // strongly correlated with the hyperparams → BLOCK_DIAG, not full DENSE.
      int ms_block_start = -1;
      int ms_block_size = 0;
      if (layout.log_sigma2_trend_idx >= 0) {
        ms_block_start = layout.log_sigma2_trend_idx;
        ms_block_size = 1;
      }
      if (layout.log_sigma2_seasonal_idx >= 0) {
        if (ms_block_start < 0) ms_block_start = layout.log_sigma2_seasonal_idx;
        ms_block_size++;
      }
      if (layout.log_sigma2_short_idx >= 0) {
        if (ms_block_start < 0) ms_block_start = layout.log_sigma2_short_idx;
        ms_block_size++;
      }
      if (layout.logit_rho_short_idx >= 0) {
        ms_block_size++;
      }
      if (ms_block_start >= 0 && ms_block_size >= 2 && ms_block_size <= 4) {
        block_specs.push_back({ms_block_start, ms_block_size});
      }
    }
    // Correlated slopes: Cholesky params form a natural block
    if (layout.has_re_correlated_slopes) {
      for (size_t t = 0; t < layout.chol_re_start_multi.size(); t++) {
        if (layout.re_correlated_multi[t]) {
          int chol_start = layout.chol_re_start_multi[t];
          int chol_size = layout.chol_re_end_multi[t] - chol_start;
          // Also include the corresponding sigma params
          // For now, just handle the Cholesky block if size <= 4
          if (chol_size >= 2 && chol_size <= 4) {
            block_specs.push_back({chol_start, chol_size});
          }
        }
      }
    }

    // NB phi params: overdispersion params are often correlated
    // A 2×2 block captures their joint curvature cheaply (4 extra multiplies/step)
    if (layout.legacy.has_phi_num && layout.legacy.has_phi_denom &&
        layout.legacy.log_phi_denom_idx == layout.legacy.log_phi_num_idx + 1) {
      block_specs.push_back({layout.legacy.log_phi_num_idx, 2});
    }

    // First check if model needs full DENSE (before block decision)
    // NB+ICAR: NegBin's digamma curvature creates strong correlations between
    // spatial phi params and overdispersion that BLOCK_DIAG's small blocks can't
    // capture. DENSE mass doubles the step size, cutting treedepth from 8-9 to 5-6,
    // which more than pays for the O(n²) per-step cost at p~108.
    // GP_t: NC z-sigma2-phi funnel creates erratic treedepth (2-10) with DIAG.
    // At p≤50, DENSE overhead is negligible.
    bool is_nb_family = (data.legacy.model_type == ModelType::NEGBIN_NEGBIN ||
                         data.legacy.model_type == ModelType::NEGBIN_GAMMA);
    bool is_icar = (data.spatial_type == SpatialType::ICAR);
    bool is_binomial_family = (data.legacy.model_type == ModelType::BINOMIAL ||
                              data.legacy.model_type == ModelType::BETA_BINOMIAL);
    // HSGP+temporal: 36 HSGP basis coefs and 20 temporal effects have complex
    // cross-correlations that DIAG can't handle (106 div) and BLOCK_DIAG misses
    // (16 div, eps~0.006). DENSE with eigenvalue conditioning captures the geometry
    // correctly (0-1 div). Tested BLOCK_DIAG (2026-03-10): 303s/0div PG, 214s/16div NB,
    // 133s/3div Bin — worse than DENSE (211s/3div, 176s/1div, 142s/0div).
    // Also applies to HSGP+TVC and HSGP+MS_t (same cross-correlation issue).
    // Only use DENSE when p <= 200 to avoid O(n²) per-step overhead dominating.
    bool hsgp_temporal = layout.is_hsgp && data.has_hsgp && n_params <= DENSE_MAX_PARAMS &&
                         (layout.has_temporal || layout.has_tvc || layout.has_multiscale_temporal);

    bool needs_full_dense = layout.has_latent ||  // N×K latent factors
                            hsgp_temporal ||  // HSGP+temporal cross-correlations
                            (is_nb_family && is_icar && n_params <= DENSE_MAX_PARAMS) ||  // NB+ICAR
                            (is_binomial_family && is_icar && n_params <= DENSE_MAX_PARAMS);  // Bin+ICAR

    // HSGP-only (no temporal): DIAG outperforms BLOCK_DIAG (29k LF/6 div vs 39k LF/15 div).
    // HSGP-only (no temporal): DIAG outperforms BLOCK_DIAG (29k LF/6 div vs
    // 39k LF/15 div). The real correlations are between lengthscale and m^2 basis
    // coefficients, which small blocks can't capture. Block adaptation adds noise.
    // HSGP+temporal uses full DENSE (handled above in needs_full_dense).
    bool prefer_diag = layout.is_hsgp && data.has_hsgp && !layout.has_temporal;

    if (needs_full_dense) {
      effective_metric = MassMatrixType::DENSE;
      block_specs.clear();
      auto_selected_diag = false;
    } else if (!block_specs.empty() && !prefer_diag) {
      // BLOCK_DIAG: captures key correlations without full O(n²)
      effective_metric = MassMatrixType::BLOCK_DIAG;
      auto_selected_diag = false;
    } else {
      // No blocks detected, no DENSE needed — fall back to DIAG
      effective_metric = MassMatrixType::DIAG;
      auto_selected_diag = true;
    }

    if (verbose) {
      REprintf("  [METRIC] auto -> %s (p=%d", metric_name(effective_metric), n_params);
      if (effective_metric == MassMatrixType::BLOCK_DIAG) {
        REprintf(", %d blocks:", (int)block_specs.size());
        for (const auto& bs : block_specs) {
          REprintf(" [%d,%d)", bs.first, bs.first + bs.second);
        }
      }
      if (layout.has_re) REprintf(", re");
      if (layout.has_re_correlated_slopes) REprintf(", correlated_slopes");
      if (layout.has_temporal) REprintf(", temporal");
      if (layout.is_bym2) REprintf(", bym2");
      if (layout.is_hsgp) REprintf(", hsgp");
      if (layout.is_gp) REprintf(", gp");
      if (layout.is_multiscale_gp) REprintf(", msgp");
      if (layout.is_temporal_gp) REprintf(", temporal_gp");
      if (layout.has_multiscale_temporal) REprintf(", ms_temporal");
      if (layout.has_svc) REprintf(", svc");
      if (layout.has_spatiotemporal) REprintf(", spatiotemporal");
      if (layout.has_latent) REprintf(", latent");
      if (layout.has_tvc) REprintf(", tvc");
      REprintf(")\n");
    }
  }
  // Also build block_specs when user explicitly requests BLOCK_DIAG
  if (effective_metric == MassMatrixType::BLOCK_DIAG && block_specs.empty()) {
    // User forced block_diag but AUTO didn't run — detect blocks from layout
    if (layout.is_temporal_gp && layout.log_sigma2_temporal_gp_idx >= 0 &&
        layout.logit_phi_temporal_gp_idx == layout.log_sigma2_temporal_gp_idx + 1) {
      block_specs.push_back({layout.log_sigma2_temporal_gp_idx, 2});
    }
    if (layout.is_hsgp && layout.log_sigma2_hsgp_idx >= 0 &&
        layout.log_lengthscale_hsgp_idx == layout.log_sigma2_hsgp_idx + 1) {
      block_specs.push_back({layout.log_sigma2_hsgp_idx, 2});
    }
    if (layout.is_bym2 && layout.log_sigma_bym2_idx >= 0 &&
        layout.logit_rho_bym2_idx == layout.log_sigma_bym2_idx + 1) {
      block_specs.push_back({layout.log_sigma_bym2_idx, 2});
    }
    if (layout.is_gp && layout.log_sigma2_gp_idx >= 0 &&
        layout.log_phi_gp_idx == layout.log_sigma2_gp_idx + 1) {
      block_specs.push_back({layout.log_sigma2_gp_idx, 2});
    }
    if (layout.is_multiscale_gp) {
      if (layout.log_sigma2_gp_local_idx >= 0 &&
          layout.log_phi_gp_local_idx == layout.log_sigma2_gp_local_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_local_idx, 2});
      }
      if (layout.log_sigma2_gp_regional_idx >= 0 &&
          layout.log_phi_gp_regional_idx == layout.log_sigma2_gp_regional_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_regional_idx, 2});
      }
    }
    if (layout.is_st_gp && layout.log_phi_st_space_idx >= 0 &&
        layout.log_phi_st_time_idx == layout.log_phi_st_space_idx + 1) {
      block_specs.push_back({layout.log_phi_st_space_idx, 2});
    }
    if (layout.legacy.has_phi_num && layout.legacy.has_phi_denom &&
        layout.legacy.log_phi_denom_idx == layout.legacy.log_phi_num_idx + 1) {
      block_specs.push_back({layout.legacy.log_phi_num_idx, 2});
    }
    // If still no blocks found, fall back to DIAG
    if (block_specs.empty()) {
      effective_metric = MassMatrixType::DIAG;
      if (verbose) {
        REprintf("  [BLOCK_DIAG] No correlated hyperparameter pairs found, falling back to DIAG\n");
      }
    }
  }

  // Auto-downgrade dense to diagonal when n_params too large
  // Dense needs O(p^2) storage and O(p^3) Cholesky; also needs n_warmup >= p samples
  if (effective_metric == MassMatrixType::DENSE && n_params > DENSE_MAX_PARAMS) {
    if (verbose) {
      REprintf("  [DENSE] n_params=%d > %d: auto-downgrading to diagonal\n",
               n_params, DENSE_MAX_PARAMS);
    }
    effective_metric = MassMatrixType::DIAG;
  }

  try {
    if (effective_metric == MassMatrixType::BLOCK_DIAG && !block_specs.empty()) {
      mass.init_block_diag(n_params, block_specs);
    } else {
      mass.init(n_params, effective_metric);
    }
  } catch (const std::bad_alloc&) {
    if (effective_metric == MassMatrixType::DENSE) {
      if (verbose) {
        REprintf("  [DENSE] Allocation failed for p=%d, falling back to diagonal\n", n_params);
      }
      effective_metric = MassMatrixType::DIAG;
      mass.init(n_params, effective_metric);
    } else {
      throw;
    }
  }

  // Initialize sparse GMRF block for ST_IV spatiotemporal interaction.
  // Uses sparse Cholesky of posterior precision Q = tau*(Q_s⊗Q_t) + diag(H_lik).
  // At warmup end, extracts diag(Q^{-1}) to set precision-informed diagonal mass.
  // NOTE: Factorization happens later (after warmup discovers tau and H_lik).
  bool use_sparse_gmrf_mass = true;
  if (use_sparse_gmrf_mass && data.has_spatiotemporal && data.spatiotemporal_data.type == STType::TYPE_IV) {
    int st_S = data.spatiotemporal_data.n_spatial;
    int st_T = data.spatiotemporal_data.n_times;
    mass.sparse_gmrf.init(layout.st_delta_start, st_S, st_T);
    if (verbose) {
      REprintf("  [SPARSE_GMRF] ST_IV block initialized: %dx%d=%d params at offset %d\n",
               st_S, st_T, st_S * st_T, layout.st_delta_start);
    }
  }

  return {effective_metric, auto_selected_diag, std::move(block_specs)};
}

// Warm-start mass matrix diagonal from model structure.
// Sets informed diagonal entries for parameter groups with known posterior scale,
// giving the step size tuner a reasonable starting point even before warmup
// samples are collected. This is critical for HSGP (m^2 basis coefficients),
// BYM2 (spatial + IID), and correlated slopes (z ~ N(0,1)) models where the
// identity mass causes excessively small epsilon -> deep NUTS trees.
void warm_start_mass_matrix(
    DenseMassMatrix& mass,
    const ModelData& data,
    const ParamLayout& layout,
    int n_params,
    bool verbose
) {
  std::vector<double> inv_m(n_params, 1.0);
  std::vector<double> sqrt_m(n_params, 1.0);
  bool any_informed = false;

  // HSGP basis coefficients: beta_j ~ N(0, 1) → posterior variance ≈ 1
  // Hyperparameters: log_sigma2 ~ prior with moderate variance,
  //                  log_lengthscale ~ LogNormal(0,1) → variance ≈ 1
  if (layout.is_hsgp) {
    for (int j = layout.hsgp_beta_start; j < layout.hsgp_beta_end; j++) {
      inv_m[j] = 1.0;  // N(0,1) prior → unit scale
    }
    inv_m[layout.log_sigma2_hsgp_idx] = 1.0;
    inv_m[layout.log_lengthscale_hsgp_idx] = 1.0;
    any_informed = true;
  }

  // ICAR: phi[s] precision ≈ degree (number of neighbors)
  // Higher degree → smaller variance → tighter mass
  if (layout.has_spatial && !layout.is_bym2 &&
      data.spatial_type == SpatialType::ICAR && !data.adj_row_ptr.empty()) {
    for (int s = 0; s < (layout.spatial_end - layout.spatial_start); s++) {
      int degree = data.adj_row_ptr[s + 1] - data.adj_row_ptr[s];
      // ICAR precision diagonal ≈ degree; variance ≈ 1/degree
      double var_est = 1.0 / std::max(1.0, (double)degree);
      inv_m[layout.spatial_start + s] = var_est;
    }
    any_informed = true;
  }

  // BYM2: spatial phi ~ ICAR (eigenvalue-scaled), theta ~ N(0, I)
  // Riebler parameterization: phi[s] ≈ scale_factor variance
  if (layout.is_bym2) {
    double sf = std::max(data.bym2_scale_factor, 0.1);
    for (int s = layout.spatial_start; s < layout.spatial_end; s++) {
      inv_m[s] = sf * sf;  // ICAR variance ~ scale_factor^2
    }
    for (int s = layout.theta_bym2_start; s < layout.theta_bym2_end; s++) {
      inv_m[s] = 1.0;  // IID: N(0,1)
    }
    inv_m[layout.log_sigma_bym2_idx] = 1.0;
    inv_m[layout.logit_rho_bym2_idx] = 4.0;  // logit scale: wider
    any_informed = true;
  }

  // Correlated slopes: z ~ N(0, 1) (non-centered), Cholesky raw ~ tanh
  if (layout.has_re_correlated_slopes) {
    // RE slopes z values
    for (int j = layout.re_start; j < layout.re_end; j++) {
      inv_m[j] = 1.0;  // z ~ N(0,1)
    }
    any_informed = true;
  }

  // Temporal effects: RW1/RW2 have known precision structure
  if (layout.has_temporal) {
    // Temporal effects: moderate scale (tau-dependent, start at 1.0)
    for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
      inv_m[j] = 1.0;
    }
    // AR1 rho: logit scale variance ≈ 4
    if (layout.logit_rho_ar1_idx >= 0) {
      inv_m[layout.logit_rho_ar1_idx] = 4.0;
    }
    any_informed = true;
  }

  // Non-centered RE: z ~ N(0, 1) → unit scale
  if (layout.has_re && data.re_parameterization == 1) {  // 1 = non-centered
    for (int j = layout.re_start; j < layout.re_end; j++) {
      inv_m[j] = 1.0;
    }
    any_informed = true;
  }

  // GP/SVC/TVC hyperparameters: moderate scale
  if (layout.is_gp) {
    if (layout.log_sigma2_gp_idx >= 0) inv_m[layout.log_sigma2_gp_idx] = 1.0;
    if (layout.log_phi_gp_idx >= 0) inv_m[layout.log_phi_gp_idx] = 1.0;
    any_informed = true;
  }

  // Temporal GP: NC z ~ N(0,1), logit_phi has wider posterior scale
  if (layout.is_temporal_gp) {
    for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
      inv_m[j] = 1.0;  // z ~ N(0,1) for NC
    }
    if (layout.log_sigma2_temporal_gp_idx >= 0)
      inv_m[layout.log_sigma2_temporal_gp_idx] = 1.0;
    if (layout.logit_phi_temporal_gp_idx >= 0)
      inv_m[layout.logit_phi_temporal_gp_idx] = 4.0;  // logit scale: wider
    any_informed = true;
  }

  if (any_informed) {
    // Compute sqrt_mass from inv_mass
    for (int i = 0; i < n_params; i++) {
      inv_m[i] = std::max(1e-3, std::min(inv_m[i], 1e3));
      sqrt_m[i] = 1.0 / std::sqrt(inv_m[i]);
    }
    mass.set_diagonal(inv_m, sqrt_m);
    if (verbose) {
      REprintf("  [WARMSTART] Initialized mass matrix from model structure\n");
    }
  }
}

// =====================================================================
// Run single HMC chain
// =====================================================================

// Pure C++ version - safe for OpenMP parallel regions
HMCResultCpp run_hmc_chain_cpp(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    int riemannian
) {
  int n_params = q_init.size();
  int n_sample = n_iter - n_warmup;
  bool use_nuts = (L == 0);

  HMCResultCpp result;
  result.n_params_stored = n_params;
  result.samples_flat.resize(static_cast<size_t>(n_sample) * n_params);
  result.log_prob.resize(n_sample);
  result.accept_prob.resize(n_sample);
  result.n_leapfrog.resize(n_sample, L);
  result.divergent.resize(n_sample, 0);
  result.treedepth.resize(n_sample, 0);
  result.n_warmup = n_warmup;
  result.n_sample = n_sample;
  result.chain_id = chain_id;
  result.n_max_treedepth = 0;

  // Collapsed GP: allocate w* storage
  if (data.gp_collapsed && data.has_gp) {
      result.n_gp_collapsed = data.gp_data.n_obs;
      result.gp_w_star_flat.resize(static_cast<size_t>(n_sample) * data.gp_data.n_obs, 0.0);
  }

  // Collapsed ICAR/BYM2: allocate phi*/theta* storage
  if (data.icar_collapsed || data.bym2_collapsed) {
      int S = data.n_spatial_units;
      result.n_icar_collapsed = S;
      result.icar_phi_star_flat.resize(static_cast<size_t>(n_sample) * S, 0.0);
      if (data.bym2_collapsed) {
          result.bym2_theta_star_flat.resize(static_cast<size_t>(n_sample) * S, 0.0);
      }
  }

  std::mt19937 rng(seed + chain_id * 12345);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  // Reset VecGradWorkspace cache for new model fit
  vec_grad_ws.cached_data_id = 0;

  std::vector<double> q = q_init;

  // For NUTS: fuse initial log_post + gradient into single O(N) pass
  std::vector<double> current_grad(n_params);
  double log_prob_current;
  if (use_nuts) {
    compute_gradient(q, data, layout, current_grad, &log_prob_current);
  } else {
    // Use same log-post function as the active gradient mode
    if (g_gradient_mode == GradientMode::AUTODIFF_ARENA ||
        g_gradient_mode == GradientMode::AUTODIFF_FWD ||
        g_gradient_mode == GradientMode::AUTODIFF_TAPE) {
      log_prob_current = tulpa::compute_log_post_impl(q, data, layout);
    } else {
      log_prob_current = compute_log_post(q, data, layout);
    }
  }

  double epsilon = find_reasonable_epsilon(q, data, layout, rng);

  // Compute target_boost for challenging model combinations
  // MSGP and GP with temporal are particularly challenging
  double target_boost = 0.0;
  if (data.has_multiscale_gp) {
    target_boost += 0.10;  // MSGP models need higher target acceptance
    if (layout.has_temporal) {
      target_boost += 0.05;  // MSGP + temporal is even more challenging
    }
  } else if (data.spatial_type == SpatialType::GP) {
    target_boost += 0.05;  // GP models moderately challenging
    if (layout.has_temporal) {
      target_boost += 0.05;  // GP + temporal combination
    }
  }
  DualAveraging da(epsilon, n_params, target_boost);

  // For NUTS: model-adaptive target acceptance
  // Store in nuts_target_accept for reuse at mass window boundaries (avoids bug
  // where da.target_accept was reset to 0.80 at each window reset).
  double nuts_target_accept = 0.80;
  if (use_nuts) {
    if (adapt_delta > 0) {
      // User override
      nuts_target_accept = adapt_delta;
    } else {
      // Auto-select based on model complexity
      nuts_target_accept = 0.80;  // Stan default base

      // BYM2: high correlation between ICAR phi + unstructured theta
      if (data.spatial_type == SpatialType::BYM2) {
        nuts_target_accept = 0.90;
      }
      // ICAR: correlated spatial params need slightly higher target
      else if (data.spatial_type == SpatialType::ICAR) {
        nuts_target_accept = 0.85;
      }

      // Correlated random slopes add funnel geometry
      if (data.has_re_correlated_slopes) {
        nuts_target_accept = std::max(nuts_target_accept, 0.90);
      }

      // Temporal GP NC: z ~ N(0,1) decorrelates parameters, lower target OK
      // Benchmarked: 0.70 gives 20% fewer LF steps and 50% less seed variance
      if (layout.is_temporal_gp && nuts_target_accept > 0.70) {
        nuts_target_accept = 0.70;
      }

      nuts_target_accept = std::min(0.99, nuts_target_accept);
    }
    da.target_accept = nuts_target_accept;
  }

  // current_grad already computed above (fused with log_prob for NUTS)

  // Select and initialize mass matrix (AUTO resolution, block detection, sparse GMRF)
  DenseMassMatrix mass;
  MassMatrixConfig mm_config = select_and_init_mass_matrix(mass, data, layout, n_params, metric_type, verbose);
  MassMatrixType effective_metric = mm_config.effective_metric;
  bool auto_selected_diag = mm_config.auto_selected_diag;
  std::vector<std::pair<int,int>> block_specs = std::move(mm_config.block_specs);

  // Warm-start mass matrix diagonal from model structure
  warm_start_mass_matrix(mass, data, layout, n_params, verbose);

  // Recompute epsilon with warm-start mass (if informed)
  // This gives the dual averaging a better starting point when mass is pre-set
  if (mass.type != MassMatrixType::DIAG && !mass.inv_mass_diag.empty()) {
    epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
  }

  WelfordStats mass_stats(n_params);              // Always track diagonal
  WelfordCovStats cov_stats(n_params);            // Only used when dense
  bool use_mass_matrix = false;

  // L-BFGS mass matrix adaptation (warmup-only)
  // Uses L-BFGS to learn curvature during warmup, then switches to standard HMC
  bool use_lbfgs = data.has_multiscale_gp &&
                   data.multiscale_gp_data.sampler == tulpa_gp::MSGPSampler::LBFGS;
  tulpa_gp::LBFGSState lbfgs_state;
  std::vector<double> q_prev, grad_prev;
  bool lbfgs_initialized = false;
  bool lbfgs_warmup_done = false;  // After warmup, use standard HMC
  if (use_lbfgs) {
    lbfgs_state = tulpa_gp::LBFGSState(10, n_params);
    q_prev.resize(n_params);
    grad_prev.resize(n_params);
  }

  // Stan-style expanding warmup windows for mass matrix adaptation
  // Phase 1: [0, init_buffer) - step size adaptation only
  // Phase 2: [init_buffer, n_warmup - term_buffer) - mass matrix adaptation
  //   Windows double in size: 25, 50, 100, 200, ...
  //   Last window extends to fill remaining space
  // Phase 3: [n_warmup - term_buffer, n_warmup) - final step size tuning
  // Models with structured warm-start (ICAR degree, BYM2 scale, HSGP) already
  // have reasonable mass, so we can start adaptation earlier (init_buffer=25).
  // This saves ~50 iterations of deep-tree warmup. Temporal_gp warm-start is
  // trivial (identity-like) so it still needs the full 75 iterations.
  bool has_structured_warmstart = layout.is_hsgp || layout.is_bym2 ||
    (layout.has_spatial && !layout.is_bym2 && data.spatial_type == SpatialType::ICAR && !data.adj_row_ptr.empty());
  int init_buffer = has_structured_warmstart ? 25 : 75;
  int term_buffer = 50;
  // For high-dimensional models (p>80), a 25-sample first mass window gives
  // very noisy variance estimates (25 samples / 108 params = 0.23 samples/param).
  // Skip the tiny first window by using a larger init_window (=50), so the first
  // mass update has ~50 samples. This trades one less mass update for better quality.
  int init_window = (n_params > 80) ? 50 : 25;

  // Dense mass models: balance final step size tuning vs warmup budget.
  // Models with p>100 need sufficient mass adaptation windows (warmup is fixed),
  // so keep term_buffer moderate. Previous 75 was too aggressive — used 30%
  // of warmup for final tuning, leaving fewer samples for mass adaptation.
  if (effective_metric == MassMatrixType::DENSE && n_params > 100) {
    term_buffer = 60;  // Reduced from 75 — saves 15 iterations for mass adaptation
  }

  // Note: For p~24, first mass window (25 samples < 29 needed) fails,
  // but this is fine — better to wait for more samples than set a poor
  // mass estimate early. The second window (100+ samples) gives good mass.

  // Adjust for short warmup
  if (n_warmup < init_buffer + term_buffer + init_window) {
    init_buffer = std::max(1, n_warmup / 5);
    term_buffer = std::max(1, n_warmup / 10);
    init_window = std::max(1, n_warmup - init_buffer - term_buffer);
  }

  // Compute mass adaptation window endpoints
  std::vector<int> mass_window_ends;
  {
    int adapt_end = n_warmup - term_buffer;
    if (adapt_end <= init_buffer) {
      // No room for mass adaptation windows
      mass_window_ends.push_back(std::max(1, adapt_end));
    } else {
      int next_end = init_buffer + init_window;
      int win_size = init_window;
      while (next_end < adapt_end) {
        int next_win = 2 * win_size;
        if (next_end + next_win > adapt_end) {
          // Extend current window to fill remaining space
          mass_window_ends.push_back(adapt_end);
          break;
        }
        mass_window_ends.push_back(next_end);
        win_size = next_win;
        next_end += win_size;
      }
      if (mass_window_ends.empty() || mass_window_ends.back() < adapt_end) {
        mass_window_ends.push_back(adapt_end);
      }
    }
  }
  int next_window_idx = 0;

  // Pre-allocate NUTS workspace (zero-allocation tree building)
  NUTSWorkspace nuts_ws;
  std::vector<double> _nuts_p;              // Momentum sampling buffer
  std::vector<double> _nuts_q_proposal;     // Persistent proposal (survives tree resets)
  std::vector<double> _nuts_grad_proposal;  // Persistent proposal gradient
  if (use_nuts) {
    nuts_ws.init(n_params, max_treedepth);
    nuts_ws.gradient_fn = resolve_gradient_fn(g_gradient_mode, data, layout);
    _nuts_p.resize(n_params);
    _nuts_q_proposal.resize(n_params);
    _nuts_grad_proposal.resize(n_params);
  }

  int sample_idx = 0;
  int n_accept = 0;
  int n_divergent = 0;
  // Adaptive NUTS→fixed-L switching: monitor early sampling for max treedepth
  int nuts_probe_window = std::min(20, n_sample);  // Check first 20 sampling iterations
  int nuts_probe_maxd = 0;  // Count of maxd hits in probe window
  bool nuts_probing = use_nuts && (L == 0);  // Only probe when using NUTS by default

  // SoftAbs divergence retry: compute local Hessian-based metric on divergent
  // trajectories and retry. Only active for BYM2/ICAR + dense mass (auto) or
  // when explicitly forced on.
  bool use_softabs_retry = false;
  if (riemannian == 1) {
    use_softabs_retry = true;
  } else if (riemannian == -1) {
    // Auto: enable for BYM2/ICAR with dense mass
    use_softabs_retry = (mass.type == MassMatrixType::DENSE &&
                         (data.spatial_type == SpatialType::BYM2 ||
                          data.spatial_type == SpatialType::ICAR));
  }
  // Disable if not using NUTS (SoftAbs retry only makes sense with NUTS)
  if (!use_nuts) use_softabs_retry = false;
  int softabs_retries = 0;
  int softabs_successes = 0;
  constexpr int SOFTABS_MAX_RETRIES = 3;  // Up to 3 retry attempts per divergence

  // Persistent SoftAbs metric (improvement #2): once computed, reuse for
  // all subsequent trajectories. Initialized at warmup→sampling transition
  // (improvement #4) or on first divergence, whichever comes first.
  bool softabs_metric_active = false;
  DenseMassMatrix softabs_persistent_mass;
  double softabs_persistent_eps = 0.0;
  if (use_softabs_retry) {
    softabs_persistent_mass.init(n_params, MassMatrixType::DENSE);
  }

  int warmup_total_leapfrog = 0;  // TEMP: diagnostic counter
  // Note: warmup divergences are normal for DIAG models and resolve via dual
  // averaging — only final epsilon matters (checked at warmup end).

  if (verbose && layout.is_temporal_gp) {
    REprintf("  [MASS-WINDOWS] n_warmup=%d, windows=[", n_warmup);
    for (size_t w = 0; w < mass_window_ends.size(); w++) {
      REprintf("%d%s", mass_window_ends[w], w + 1 < mass_window_ends.size() ? "," : "");
    }
    REprintf("]\n");
  }
  for (int iter = 0; iter < n_iter; iter++) {
    bool is_warmup = (iter < n_warmup);
    // Check if we've reached a mass adaptation window boundary
    if (is_warmup && next_window_idx < (int)mass_window_ends.size() &&
        iter == mass_window_ends[next_window_idx]) {
      bool dense_covariance_set = false;  // Track if DENSE covariance (not just diagonal) succeeded this window
      // Dense mass matrix: try full covariance first
      // OAS shrinkage guarantees PD even when n < p, so we can lower the
      // threshold from n_params+5.  For large p the original threshold is
      // unreachable during warmup (e.g. p=159, need 164 but only get 125).
      // New threshold: min(p+5, max(50, p/2))  — for p=159 this is 79.
      int dense_threshold = std::min(n_params + 5,
                                     std::max(50, n_params / 2));
      if (mass.type == MassMatrixType::DENSE && cov_stats.n >= dense_threshold) {
        auto cov = cov_stats.covariance();
        if (mass.update_from_covariance(cov.data(), cov_stats.n)) {
          use_mass_matrix = true;
          dense_covariance_set = true;
          if (verbose) {
            REprintf("  [DENSE] Window %d (iter %d): dense mass SET (n=%d, p=%d, OAS shrinkage=%.3f)\n",
                     next_window_idx, iter, cov_stats.n, n_params,
                     cov_stats.shrinkage_intensity);
          }
        } else {
          // Cholesky failed — mass auto-degraded to DIAG, use diagonal stats
          if (verbose) {
            REprintf("  [DENSE] Window %d (iter %d): Cholesky FAILED (cov_stats.n=%d, p=%d)\n",
                     next_window_idx, iter, cov_stats.n, n_params);
          }
          if (mass_stats.n >= 10) {
            mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
            use_mass_matrix = true;
          }
        }
      } else if (mass.type == MassMatrixType::DENSE) {
        // Not enough samples for dense yet — use diagonal as interim
        if (verbose) {
          REprintf("  [DENSE] Window %d (iter %d): not enough samples (cov_stats.n=%d, need=%d)\n",
                   next_window_idx, iter, cov_stats.n, dense_threshold);
        }
        if (mass_stats.n >= 10) {
          mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
          use_mass_matrix = true;
        }
      } else if (mass.type == MassMatrixType::BLOCK_DIAG) {
        // Block-diagonal: set diagonal for all params, then adapt block covariances
        if (mass_stats.n >= 10) {
          mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
          use_mass_matrix = true;
        }
        int n_adapted = 0;
        for (auto& blk : mass.blocks) {
          if (blk.update_from_welford()) {
            n_adapted++;
          }
        }
        if (verbose && n_adapted > 0) {
          REprintf("  [BLOCK_DIAG] Window %d (iter %d): %d/%d blocks adapted (n=%d)\n",
                   next_window_idx, iter, n_adapted, (int)mass.blocks.size(), mass_stats.n);
        }
        // Reset block Welford accumulators for next window
        for (auto& blk : mass.blocks) {
          blk.reset_welford();
        }
      } else if (mass_stats.n >= 10) {
        // Diagonal path
        mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
        use_mass_matrix = true;
      }

      // Temporal GP NC: z ~ N(0,1) by construction → optimal diag mass ≈ 1.0.
      // With limited warmup samples, noisy variance estimates for 20 z params
      // create unbalanced mass → small epsilon. Fix z entries to 1.0 so the
      // step size is driven by the hyperparameters (beta, sigma2, phi) only.
      if (verbose && layout.is_temporal_gp) {
        REprintf("  [Z-DEBUG] Window %d (iter %d): use_mass=%d, tgp=%d, nc=%d, ts=%d, te=%d, mass_n=%d\n",
                 next_window_idx, iter, (int)use_mass_matrix,
                 (int)layout.is_temporal_gp, data.temporal_gp_parameterization,
                 layout.temporal_start, layout.temporal_end, mass_stats.n);
      }
      if (use_mass_matrix && layout.is_temporal_gp &&
          data.temporal_gp_parameterization == 1 &&
          layout.temporal_start >= 0 && layout.temporal_end > layout.temporal_start) {
        if (verbose) {
          REprintf("  [Z-FREEZE] Window %d: z mass before=[", next_window_idx);
          for (int j = layout.temporal_start; j < std::min(layout.temporal_end, layout.temporal_start + 5); j++) {
            REprintf("%.3f%s", mass.inv_mass_diag[j], j < layout.temporal_start + 4 ? "," : "");
          }
          REprintf("...], hyper=[");
          // Print beta and hyperparams
          for (int j = 0; j < std::min(4, layout.temporal_start); j++) {
            REprintf("%.3f%s", mass.inv_mass_diag[j], j < 3 ? "," : "");
          }
          REprintf("], sigma2=%.3f, phi=%.3f\n",
                   layout.log_sigma2_temporal_gp_idx >= 0 ? mass.inv_mass_diag[layout.log_sigma2_temporal_gp_idx] : -1.0,
                   layout.logit_phi_temporal_gp_idx >= 0 ? mass.inv_mass_diag[layout.logit_phi_temporal_gp_idx] : -1.0);
        }
        for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
          mass.inv_mass_diag[j] = 1.0;
          mass.sqrt_mass_diag[j] = 1.0;
        }
      }

      mass_stats.reset();
      // For dense: only reset cov_stats when full covariance was successfully
      // computed THIS window. Otherwise keep accumulating across windows until
      // we have enough samples. This prevents the chicken-and-egg problem
      // where short windows never collect enough.
      // NOTE: We use dense_covariance_set (not mass.adapted) because
      // set_diagonal() also sets adapted=true, which would incorrectly
      // trigger a reset when we're still building up covariance samples.
      if (mass.type != MassMatrixType::DENSE || dense_covariance_set) {
        cov_stats.reset();
      }
      // Re-initialize step size with current mass matrix (A3)
      // Use dense-aware version when dense mass is adapted, so the step size
      // is calibrated for the rotated phase space (not just the diagonal).
      if (use_mass_matrix && mass.type == MassMatrixType::DENSE && mass.adapted) {
        epsilon = find_reasonable_epsilon_dense(q, data, layout, rng, mass);
      } else if (use_mass_matrix) {
        epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
      } else {
        epsilon = find_reasonable_epsilon(q, data, layout, rng);
      }
      da = DualAveraging(epsilon, n_params, target_boost);
      if (use_nuts) da.target_accept = nuts_target_accept;  // Preserve model-adaptive target

      next_window_idx++;
    }

    // L-BFGS: transition from L-BFGS to standard HMC at end of warmup
    // Extract diagonal mass matrix from learned curvature
    if (use_lbfgs && !lbfgs_warmup_done && iter == n_warmup - 1 && lbfgs_initialized) {
      // Use gamma from L-BFGS as uniform scaling for mass matrix
      // gamma = (s^T y) / (y^T y) approximates average inverse Hessian scaling
      double gamma = lbfgs_state.gamma;
      if (gamma > 0.01 && gamma < 100.0) {
        // Set inv_mass = gamma * I (larger gamma = larger variance = larger step in that direction)
        std::vector<double> inv_m(n_params, gamma);
        std::vector<double> sqrt_m(n_params, 1.0 / std::sqrt(gamma));
        mass.set_diagonal(inv_m, sqrt_m);
        use_mass_matrix = true;
      }
      lbfgs_warmup_done = true;
    }

    // =========================================================================
    // NUTS or fixed-trajectory HMC
    // =========================================================================
    double alpha = 0.0;
    bool divergent = false;
    int iter_n_leapfrog = L;
    int iter_treedepth = 0;

    if (use_nuts && !(use_lbfgs && !lbfgs_warmup_done)) {
      // -----------------------------------------------------------------
      // NUTS: No-U-Turn Sampler (optimized zero-allocation path)
      // -----------------------------------------------------------------

      auto& p = _nuts_p;

      // Step size jitter (improvement #5): ±20% random noise per trajectory
      // Prevents systematic step-size resonances that cause divergences.
      // Only during post-warmup sampling — warmup needs stable epsilon for adaptation.
      double eps_iter = epsilon;
      if (!is_warmup) {
        double jitter = 1.0 + 0.2 * (2.0 * unif(rng) - 1.0);  // U[0.8, 1.2]
        eps_iter = epsilon * jitter;
      }

      // Sample momentum p ~ N(0, M) where M = C^{-1}
      mass.sample_momentum(p.data(), rng);

      // Initial Hamiltonian (pointer-based, no vector overhead)
      double H0 = nuts_compute_hamiltonian_fast(
        log_prob_current, p.data(), mass, n_params
      );
      double delta_max = 1000.0;

      // Load current state into workspace persistent slots
      nuts_ws.load_node(NUTSWorkspace::NODE_LEFT_SLOT,
                        q.data(), p.data(), current_grad.data(), log_prob_current);
      nuts_ws.load_node(NUTSWorkspace::NODE_RIGHT_SLOT,
                        q.data(), p.data(), current_grad.data(), log_prob_current);

      // Initialize persistent proposal buffers (pre-allocated, no per-iter malloc)
      auto& q_proposal_data = _nuts_q_proposal;
      auto& grad_proposal_data = _nuts_grad_proposal;
      std::memcpy(q_proposal_data.data(), q.data(), n_params * sizeof(double));
      std::memcpy(grad_proposal_data.data(), current_grad.data(), n_params * sizeof(double));
      double log_prob_proposal = log_prob_current;
      double sum_log_weight = 0.0;  // Relative weights: log(exp(H0 - H0)) = 0

      int total_leapfrog = 0;
      double sum_accept_prob = 0.0;
      divergent = false;

      // Generalized U-turn tracking at top level (Stan-style)
      // rho = total momentum sum. rho_bck/rho_fwd = halves for 3-juncture checks.
      // At each iteration the entire old trajectory becomes one half,
      // the new subtree becomes the other half (Stan's approach).
      // Uses pre-allocated workspace vectors (no per-iteration heap allocation).
      auto& rho = nuts_ws.iter_rho;
      std::memcpy(rho.data(), p.data(), n_params * sizeof(double));
      auto& rho_bck = nuts_ws.iter_rho_bck;
      auto& rho_fwd = nuts_ws.iter_rho_fwd;
      std::fill(rho_bck.begin(), rho_bck.end(), 0.0);
      std::fill(rho_fwd.begin(), rho_fwd.end(), 0.0);

      // p_sharp = M^{-1} * p at initial point — full mass for correct U-turn geometry
      auto& p_sharp_init = nuts_ws.iter_p_sharp_init;
      mass.inv_mass_times_p(p.data(), p_sharp_init.data());

      // Boundary momenta: _end = far endpoint, _beg = origin-facing boundary
      // Stan naming: bck_end=bck_bck, bck_beg=bck_fwd, fwd_beg=fwd_bck, fwd_end=fwd_fwd
      auto& p_fwd_beg = nuts_ws.iter_p_fwd_beg;
      auto& p_fwd_end = nuts_ws.iter_p_fwd_end;
      auto& p_bck_beg = nuts_ws.iter_p_bck_beg;
      auto& p_bck_end = nuts_ws.iter_p_bck_end;
      std::memcpy(p_fwd_beg.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_fwd_end.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_bck_beg.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_bck_end.data(), p.data(), n_params * sizeof(double));
      auto& p_sharp_fwd_beg = nuts_ws.iter_p_sharp_fwd_beg;
      auto& p_sharp_fwd_end = nuts_ws.iter_p_sharp_fwd_end;
      auto& p_sharp_bck_beg = nuts_ws.iter_p_sharp_bck_beg;
      auto& p_sharp_bck_end = nuts_ws.iter_p_sharp_bck_end;
      std::memcpy(p_sharp_fwd_beg.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_fwd_end.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_bck_beg.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_bck_end.data(), p_sharp_init.data(), n_params * sizeof(double));

      // Build tree until U-turn or max depth
      for (int j = 0; j < max_treedepth; j++) {
        std::uniform_int_distribution<int> dir_dist(0, 1);
        int direction = 2 * dir_dist(rng) - 1;

        nuts_ws.reset_tree();

        int start_slot = nuts_ws.alloc_slot();
        if (start_slot < 0) break;
        if (direction == 1) {
          nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_RIGHT_SLOT);
        } else {
          nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_LEFT_SLOT);
        }

        // Stan: relabel halves before building subtree
        // Entire old trajectory becomes one half; new subtree is the other
        if (direction == 1) {
          // Extending forward: old trajectory → backward half
          std::memcpy(rho_bck.data(), rho.data(), n_params * sizeof(double));
          std::memcpy(p_bck_beg.data(), p_fwd_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(), n_params * sizeof(double));
        } else {
          // Extending backward: old trajectory → forward half
          std::memcpy(rho_fwd.data(), rho.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_beg.data(), p_bck_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_beg.data(), p_sharp_bck_end.data(), n_params * sizeof(double));
        }

        TreeStats subtree = build_tree_fast(
          nuts_ws, start_slot, direction, j,
          eps_iter, mass, H0, delta_max,
          data, layout, rng
        );

        total_leapfrog += subtree.n_leapfrog;
        sum_accept_prob += subtree.sum_accept_prob;

        if (subtree.divergent) {
          divergent = true;
        }

        if (!subtree.stop) {
          // Multinomial acceptance
          double log_sum_weight_subtree = subtree.sum_log_weight;
          double new_sum_log_weight = nuts_log_sum_exp(sum_log_weight, log_sum_weight_subtree);

          double accept_prob_subtree;
          if (log_sum_weight_subtree > new_sum_log_weight) {
            accept_prob_subtree = 1.0;
          } else {
            accept_prob_subtree = std::exp(log_sum_weight_subtree - new_sum_log_weight);
          }
          if (!std::isfinite(accept_prob_subtree)) accept_prob_subtree = 0.0;

          std::uniform_real_distribution<double> unif01(0.0, 1.0);
          if (unif01(rng) < accept_prob_subtree) {
            std::memcpy(q_proposal_data.data(), nuts_ws.q_at(subtree.proposal_slot),
                        n_params * sizeof(double));
            std::memcpy(grad_proposal_data.data(), nuts_ws.grad_at(subtree.proposal_slot),
                        n_params * sizeof(double));
            log_prob_proposal = subtree.log_prob_proposal;
          }

          sum_log_weight = new_sum_log_weight;
        }

        // Update direction endpoints and rho half from subtree
        // Use memcpy instead of std::move to preserve pre-allocated buffers
        if (direction == 1) {
          nuts_ws.copy_node(NUTSWorkspace::NODE_RIGHT_SLOT, subtree.right_slot);
          std::memcpy(rho_fwd.data(), subtree.rho.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_end.data(), subtree.p_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
        } else {
          nuts_ws.copy_node(NUTSWorkspace::NODE_LEFT_SLOT, subtree.left_slot);
          std::memcpy(rho_bck.data(), subtree.rho.data(), n_params * sizeof(double));
          std::memcpy(p_bck_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
          std::memcpy(p_bck_end.data(), subtree.p_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
        }

        // Combine rho = rho_bck + rho_fwd
        for (int i = 0; i < n_params; i++) {
          rho[i] = rho_bck[i] + rho_fwd[i];
        }

        iter_treedepth = j + 1;

        // Generalized U-turn check at top level (3 junctures)
        if (subtree.stop) break;

        // Check 1: Full trajectory — far endpoints vs total rho
        bool persist = compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_end.data(),
                                         rho.data(), n_params);

        // Check 2: Backward half + seam from forward (rho = rho_bck + p_fwd_beg)
        auto& rho_seam = nuts_ws.iter_rho_seam;
        for (int i = 0; i < n_params; i++) {
          rho_seam[i] = rho_bck[i] + p_fwd_beg[i];
        }
        persist &= compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_beg.data(),
                                      rho_seam.data(), n_params);

        // Check 3: Seam from backward + forward half (rho = rho_fwd + p_bck_beg)
        for (int i = 0; i < n_params; i++) {
          rho_seam[i] = rho_fwd[i] + p_bck_beg[i];
        }
        persist &= compute_criterion(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(),
                                      rho_seam.data(), n_params);

        if (!persist) break;
      }

      // SoftAbs divergence retry (improvements #1, #2): if trajectory diverged,
      // compute local Hessian-based metric and retry up to SOFTABS_MAX_RETRIES
      // times, halving step size each attempt. On first successful metric
      // computation, persist it for all subsequent trajectories.
      if (divergent && !is_warmup && use_softabs_retry) {
        softabs_retries++;

        // Compute fresh Hessian at current position (p+1 gradient evals)
        std::vector<double> hessian_buf;
        compute_hessian_finite_diff(q, data, layout, hessian_buf);
        for (auto& v : hessian_buf) v = -v;  // Negate: -H = curvature

        std::vector<double> G_inv_buf, L_G_inv_buf;
        bool metric_ok = compute_softabs_metric(
          hessian_buf, n_params, 1.0, G_inv_buf, L_G_inv_buf
        );

        if (metric_ok) {
          // Update persistent SoftAbs metric for retry use only (improvement #2).
          // Do NOT override main mass/epsilon — warmup-adapted values work better
          // for general trajectories. SoftAbs is rescue-only.
          softabs_persistent_mass.set_from_metric(G_inv_buf, L_G_inv_buf);
          double eps_base = find_reasonable_epsilon_dense(
            q, data, layout, rng, softabs_persistent_mass);
          softabs_persistent_eps = eps_base;
          softabs_metric_active = true;

          // Multiple retry attempts (improvement #1): try up to 3 times
          // with halving step size each attempt
          for (int retry_attempt = 0; retry_attempt < SOFTABS_MAX_RETRIES; retry_attempt++) {
            double eps_retry = eps_base * std::pow(0.5, retry_attempt);

            // Sample new momentum and re-run NUTS trajectory
            softabs_persistent_mass.sample_momentum(p.data(), rng);
            double H0_retry = nuts_compute_hamiltonian_fast(
              log_prob_current, p.data(), softabs_persistent_mass, n_params
            );

            // Load current state into workspace
            nuts_ws.load_node(NUTSWorkspace::NODE_LEFT_SLOT,
                              q.data(), p.data(), current_grad.data(), log_prob_current);
            nuts_ws.load_node(NUTSWorkspace::NODE_RIGHT_SLOT,
                              q.data(), p.data(), current_grad.data(), log_prob_current);

            std::memcpy(q_proposal_data.data(), q.data(), n_params * sizeof(double));
            std::memcpy(grad_proposal_data.data(), current_grad.data(), n_params * sizeof(double));
            log_prob_proposal = log_prob_current;
            sum_log_weight = 0.0;
            total_leapfrog = 0;
            sum_accept_prob = 0.0;
            bool retry_divergent = false;

            // Full NUTS tree with SoftAbs metric + 3-juncture U-turn
            std::memcpy(rho.data(), p.data(), n_params * sizeof(double));
            std::fill(rho_bck.begin(), rho_bck.end(), 0.0);
            std::fill(rho_fwd.begin(), rho_fwd.end(), 0.0);
            softabs_persistent_mass.inv_mass_times_p(p.data(), p_sharp_init.data());
            std::copy(p.begin(), p.end(), p_fwd_beg.begin());
            std::copy(p.begin(), p.end(), p_fwd_end.begin());
            std::copy(p.begin(), p.end(), p_bck_beg.begin());
            std::copy(p.begin(), p.end(), p_bck_end.begin());
            std::copy(p_sharp_init.begin(), p_sharp_init.end(), p_sharp_fwd_beg.begin());
            std::copy(p_sharp_init.begin(), p_sharp_init.end(), p_sharp_fwd_end.begin());
            std::copy(p_sharp_init.begin(), p_sharp_init.end(), p_sharp_bck_beg.begin());
            std::copy(p_sharp_init.begin(), p_sharp_init.end(), p_sharp_bck_end.begin());

            int retry_treedepth = 0;
            for (int j = 0; j < max_treedepth; j++) {
              std::uniform_int_distribution<int> dir_dist(0, 1);
              int direction = 2 * dir_dist(rng) - 1;

              nuts_ws.reset_tree();
              int start_slot = nuts_ws.alloc_slot();
              if (start_slot < 0) break;
              if (direction == 1) {
                nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_RIGHT_SLOT);
              } else {
                nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_LEFT_SLOT);
              }

              if (direction == 1) {
                std::memcpy(rho_bck.data(), rho.data(), n_params * sizeof(double));
                std::memcpy(p_bck_beg.data(), p_fwd_end.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(), n_params * sizeof(double));
              } else {
                std::memcpy(rho_fwd.data(), rho.data(), n_params * sizeof(double));
                std::memcpy(p_fwd_beg.data(), p_bck_end.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_fwd_beg.data(), p_sharp_bck_end.data(), n_params * sizeof(double));
              }

              TreeStats subtree = build_tree_fast(
                nuts_ws, start_slot, direction, j,
                eps_retry, softabs_persistent_mass, H0_retry, 1000.0,
                data, layout, rng
              );

              total_leapfrog += subtree.n_leapfrog;
              sum_accept_prob += subtree.sum_accept_prob;
              if (subtree.divergent) retry_divergent = true;

              if (!subtree.stop) {
                double log_sum_weight_subtree = subtree.sum_log_weight;
                double new_sum_log_weight = nuts_log_sum_exp(sum_log_weight, log_sum_weight_subtree);
                double accept_prob_subtree;
                if (log_sum_weight_subtree > new_sum_log_weight) {
                  accept_prob_subtree = 1.0;
                } else {
                  accept_prob_subtree = std::exp(log_sum_weight_subtree - new_sum_log_weight);
                }
                if (!std::isfinite(accept_prob_subtree)) accept_prob_subtree = 0.0;

                std::uniform_real_distribution<double> unif01(0.0, 1.0);
                if (unif01(rng) < accept_prob_subtree) {
                  std::memcpy(q_proposal_data.data(), nuts_ws.q_at(subtree.proposal_slot),
                              n_params * sizeof(double));
                  std::memcpy(grad_proposal_data.data(), nuts_ws.grad_at(subtree.proposal_slot),
                              n_params * sizeof(double));
                  log_prob_proposal = subtree.log_prob_proposal;
                }
                sum_log_weight = new_sum_log_weight;
              }

              if (direction == 1) {
                nuts_ws.copy_node(NUTSWorkspace::NODE_RIGHT_SLOT, subtree.right_slot);
                std::memcpy(rho_fwd.data(), subtree.rho.data(), n_params * sizeof(double));
                std::memcpy(p_fwd_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
                std::memcpy(p_fwd_end.data(), subtree.p_end.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_fwd_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_fwd_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
              } else {
                nuts_ws.copy_node(NUTSWorkspace::NODE_LEFT_SLOT, subtree.left_slot);
                std::memcpy(rho_bck.data(), subtree.rho.data(), n_params * sizeof(double));
                std::memcpy(p_bck_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
                std::memcpy(p_bck_end.data(), subtree.p_end.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_bck_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_bck_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
              }

              for (int i = 0; i < n_params; i++) {
                rho[i] = rho_bck[i] + rho_fwd[i];
              }
              retry_treedepth = j + 1;

              if (subtree.stop) break;

              bool persist = compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_end.data(),
                                               rho.data(), n_params);
              auto& rho_seam_retry = nuts_ws.iter_rho_seam;
              for (int i = 0; i < n_params; i++) {
                rho_seam_retry[i] = rho_bck[i] + p_fwd_beg[i];
              }
              persist &= compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_beg.data(),
                                            rho_seam_retry.data(), n_params);
              for (int i = 0; i < n_params; i++) {
                rho_seam_retry[i] = rho_fwd[i] + p_bck_beg[i];
              }
              persist &= compute_criterion(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(),
                                            rho_seam_retry.data(), n_params);
              if (!persist) break;
            }

            // If retry succeeded (no divergence), accept and stop retrying
            if (!retry_divergent) {
              divergent = false;
              iter_treedepth = retry_treedepth;
              softabs_successes++;
              alpha = (total_leapfrog > 0) ? (sum_accept_prob / total_leapfrog) : 0.0;
              iter_n_leapfrog = total_leapfrog;
              break;  // Success — stop retry loop
            }
            // Otherwise: try again with halved step size (next iteration)
          }  // end retry_attempt loop

          // If all retries failed, update stats from last attempt
          if (divergent) {
            alpha = (total_leapfrog > 0) ? (sum_accept_prob / total_leapfrog) : 0.0;
            iter_n_leapfrog = total_leapfrog;
          }
        }
        // else: metric computation failed, keep original divergent result
      }

      // Accept proposal: copy from persistent proposal buffers (memcpy, no alloc)
      std::memcpy(q.data(), q_proposal_data.data(), n_params * sizeof(double));
      std::memcpy(current_grad.data(), grad_proposal_data.data(), n_params * sizeof(double));
      log_prob_current = log_prob_proposal;
      n_accept++;

      // Average acceptance statistic for dual averaging
      alpha = (total_leapfrog > 0) ? (sum_accept_prob / total_leapfrog) : 0.0;
      iter_n_leapfrog = total_leapfrog;

      if (divergent) n_divergent++;
      if (iter_treedepth >= max_treedepth) result.n_max_treedepth++;

      // Adaptation during warmup
      if (is_warmup) {
        epsilon = da.update(alpha);

        // Early detection of catastrophic dense mass during terminal buffer.
        // Normal epsilon with dense mass is 0.1-0.5. If it exceeds 2.0, the mass
        // matrix eigenvalues are pathological. Fall back to DIAG immediately so
        // the remaining terminal buffer iterations (~48) properly adapt epsilon.
        // inv_mass_diag is always kept in sync with the dense diagonal (line 62).
        if (iter >= n_warmup - term_buffer && iter < n_warmup - 1 &&
            mass.type == MassMatrixType::DENSE && mass.adapted && epsilon > 2.0) {
          if (verbose) {
            REprintf("  [DENSE] WARNING at iter %d: epsilon=%.4f (catastrophic). "
                     "Falling back to DIAG mass.\n", iter, epsilon);
          }
          mass.type = MassMatrixType::DIAG;
          // inv_mass_diag already populated from dense diagonal (update_from_covariance)
          epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
          da = DualAveraging(epsilon, n_params, target_boost);
          if (use_nuts) da.target_accept = nuts_target_accept;
        }

        // DIAG→DENSE recovery is checked at warmup end (after da.final_epsilon)
        // rather than during warmup — warmup divergences are normal for DIAG models
        // and resolve via dual averaging. Only catastrophic final epsilon matters.

        if (iter >= init_buffer && iter < n_warmup - term_buffer) {
          mass_stats.update(q);
          if (mass.type == MassMatrixType::DENSE) {
            cov_stats.update(q);
          }
          if (mass.type == MassMatrixType::BLOCK_DIAG) {
            for (auto& blk : mass.blocks) {
              blk.welford_update(q.data());
            }
          }
        }
        if (iter == n_warmup - 1) {
          epsilon = da.final_epsilon();

          // DIAG→BLOCK_DIAG→DENSE recovery at warmup end: if AUTO selected DIAG but the
          // final adapted epsilon is catastrophic (>2.0), DIAG can't capture the
          // posterior geometry. Try BLOCK_DIAG first if block_specs available,
          // otherwise fall back to DENSE with identity mass.
          if (auto_selected_diag && mass.type == MassMatrixType::DIAG &&
              epsilon > 2.0) {
            if (!block_specs.empty()) {
              // Try BLOCK_DIAG recovery first (cheaper than full DENSE)
              if (verbose) {
                REprintf("  [DIAG->BLOCK_DIAG] Warmup end: final epsilon=%.4f (catastrophic). "
                         "Switching to BLOCK_DIAG (adapted=false).\n", epsilon);
              }
              mass.init_block_diag(n_params, block_specs);
              effective_metric = MassMatrixType::BLOCK_DIAG;
              auto_selected_diag = false;
              epsilon = find_reasonable_epsilon(q, data, layout, rng);
              da = DualAveraging(epsilon, n_params, target_boost);
              if (use_nuts) da.target_accept = nuts_target_accept;
              epsilon = da.final_epsilon();
            } else if (n_params <= DENSE_MAX_PARAMS) {
              if (verbose) {
                REprintf("  [DIAG->DENSE] Warmup end: final epsilon=%.4f (catastrophic). "
                         "Switching to DENSE identity mass.\n", epsilon);
              }
              mass.init(n_params, MassMatrixType::DENSE);
              effective_metric = MassMatrixType::DENSE;
              auto_selected_diag = false;
              epsilon = find_reasonable_epsilon(q, data, layout, rng);
              da = DualAveraging(epsilon, n_params, target_boost);
              if (use_nuts) da.target_accept = nuts_target_accept;
              epsilon = da.final_epsilon();
            }
          }

          // BLOCK_DIAG→DIAG fallback: if epsilon still catastrophic after BLOCK_DIAG
          if (epsilon > 2.0 && mass.type == MassMatrixType::BLOCK_DIAG) {
            if (verbose) {
              REprintf("  [BLOCK_DIAG->DIAG] WARNING: epsilon=%.4f still catastrophic. "
                       "Falling back to DIAG.\n", epsilon);
            }
            mass.init(n_params, MassMatrixType::DIAG);
            effective_metric = MassMatrixType::DIAG;
            epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
            da = DualAveraging(epsilon, n_params, target_boost);
            if (use_nuts) da.target_accept = nuts_target_accept;
            epsilon = da.final_epsilon();
          }

          // Final safety net: if epsilon is still > 1.0 with dense mass after
          // the full terminal buffer, fall back to DIAG. This catches cases
          // where the catastrophe develops slowly.
          if (epsilon > 1.0 && mass.type == MassMatrixType::DENSE && mass.adapted) {
            if (verbose) {
              REprintf("  [DENSE] WARNING: epsilon=%.4f after warmup (catastrophic). "
                       "Falling back to DIAG mass.\n", epsilon);
            }
            mass.type = MassMatrixType::DIAG;
            epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
            da = DualAveraging(epsilon, n_params, target_boost);
            if (use_nuts) da.target_accept = nuts_target_accept;
            epsilon = da.final_epsilon();
          }

          // Precision-informed diagonal mass for ST_IV at warmup end.
          // Build Q_post = tau*(Q_s⊗Q_t) + diag(h_lik), factorize, extract diag(Q^{-1}).
          if (mass.sparse_gmrf.active && !mass.sparse_gmrf.factorized) {
            int st_S = data.spatiotemporal_data.n_spatial;
            int st_T = data.spatiotemporal_data.n_times;
            int ST = st_S * st_T;
            double tau_st = std::exp(q[layout.log_tau_st_idx]);

            // Compute likelihood Hessian diagonal for each ST cell
            std::vector<double> h_lik(ST, 0.0);
            for (int i = 0; i < data.N; i++) {
              if (data.spatiotemporal_data.st_flat[i] <= 0) continue;
              int k = data.spatiotemporal_data.st_flat[i] - 1;
              double eta_i = 0.0;
              for (int p2 = 0; p2 < data.legacy.p_num; p2++)
                eta_i += data.legacy.X_num_flat[static_cast<size_t>(i) * data.legacy.p_num + p2] * q[layout.legacy.beta_num_start + p2];
              if (layout.has_re && data.re_group[i] > 0)
                eta_i += re_value_for_eta(&q[layout.re_start], data.re_group[i] - 1,
                                           std::exp(q[layout.log_sigma_re_idx]), data.re_parameterization);
              if (layout.has_spatial && data.spatial_group[i] > 0)
                eta_i += q[layout.spatial_start + data.spatial_group[i] - 1];
              if (layout.has_temporal && !data.temporal_time_idx.empty() &&
                  i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t_idx = data.temporal_time_idx[i] - 1;
                int g_idx = data.temporal_group_idx[i] - 1;
                int t_base = g_idx * data.n_times + t_idx;
                if (t_base >= 0 && t_base < (int)q.size()) eta_i += q[layout.temporal_start + t_base];
              }
              eta_i += q[layout.st_delta_start + k];
              double mu_i = std::exp(eta_i);
              double h_i = 0.0;
              if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
                h_i = mu_i;
              } else if (data.legacy.model_type == ModelType::BINOMIAL) {
                double p_i = 1.0 / (1.0 + std::exp(-eta_i));
                h_i = data.legacy.y_denom[i] * p_i * (1.0 - p_i);
              } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN ||
                         data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
                double phi = std::exp(q[layout.legacy.log_phi_num_idx]);
                h_i = mu_i / (1.0 + mu_i / phi);
              } else {
                h_i = mu_i;
              }
              if (data.spatiotemporal_data.shared) h_i *= 2.0;
              h_lik[k] += std::max(h_i, 1e-6);
            }

            mass.sparse_gmrf.build_and_factorize(
                data.spatiotemporal_data.adj_row_ptr,
                data.spatiotemporal_data.adj_col_idx,
                data.spatiotemporal_data.temporal_type,
                data.spatiotemporal_data.temporal_cyclic,
                tau_st, h_lik.data(), 0.001
            );

            if (mass.sparse_gmrf.factorized) {
              // Extract diag(Q^{-1}) and set diagonal mass for ST params
              int n_set = 0;
              double sum_var = 0.0;
              for (int k = 0; k < ST; k++) {
                Eigen::VectorXd ek = Eigen::VectorXd::Zero(ST);
                ek[k] = 1.0;
                Eigen::VectorXd col_k = mass.sparse_gmrf.llt.solve(ek);
                double var_k = col_k[k];
                if (var_k > 1e-10 && var_k < 100.0) {
                  mass.inv_mass_diag[layout.st_delta_start + k] = var_k;
                  n_set++;
                  sum_var += var_k;
                }
              }
              // Deactivate sparse GMRF — diagonal mass is now informed
              mass.sparse_gmrf.active = false;
              // Recompute epsilon with new mass
              epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
              da = DualAveraging(epsilon, n_params, target_boost);
              if (use_nuts) da.target_accept = nuts_target_accept;
              epsilon = da.final_epsilon();
              if (verbose) {
                REprintf("  [SPARSE_GMRF] Diagonal mass set for %d/%d ST params, avg_var=%.6f, tau=%.4f, new epsilon=%.6f\n",
                         n_set, ST, n_set > 0 ? sum_var / n_set : 0.0, tau_st, epsilon);
              }
            } else if (verbose) {
              REprintf("  [SPARSE_GMRF] WARNING: Cholesky failed, keeping adapted diagonal mass\n");
            }
          }

          if (verbose) {
            REprintf("  [METRIC] Warmup done: epsilon=%.6f, mass.type=%s, mass.adapted=%d\n",
                     epsilon, metric_name(mass.type), (int)mass.adapted);
          }
          // Proactive SoftAbs at warmup→sampling transition (improvement #4):
          // Pre-compute SoftAbs metric so it's ready for retry attempts.
          // Do NOT override main mass/epsilon — warmup-adapted values are better
          // for general sampling. SoftAbs is only used as rescue on divergences.
          if (use_softabs_retry && !softabs_metric_active) {
            std::vector<double> hessian_warmup_end;
            compute_hessian_finite_diff(q, data, layout, hessian_warmup_end);
            for (auto& v : hessian_warmup_end) v = -v;

            std::vector<double> G_inv_init, L_G_inv_init;
            if (compute_softabs_metric(hessian_warmup_end, n_params, 1.0,
                                       G_inv_init, L_G_inv_init)) {
              softabs_persistent_mass.set_from_metric(G_inv_init, L_G_inv_init);
              softabs_persistent_eps = find_reasonable_epsilon_dense(
                q, data, layout, rng, softabs_persistent_mass);
              softabs_metric_active = true;
              // Note: main mass and epsilon are NOT overridden
              if (verbose) {
                REprintf("  [SoftAbs] Proactive metric pre-computed at warmup end: retry_eps=%.6f\n",
                         softabs_persistent_eps);
              }
            }
          }
        }
        // Print tree depth for last 10 warmup iterations
        if (verbose && iter >= n_warmup - 10) {
          REprintf("  [%s] warmup iter %d: treedepth=%d, epsilon=%.6f\n",
                   metric_name(mass.type),
                   iter, iter_treedepth, epsilon);
        }
      }

      // Adaptive NUTS probe: warn if most early iterations hit max treedepth
      // (Stan's approach: warn but keep NUTS running — truncated NUTS picks
      // from up to 2^depth candidates, far better than HMC(L=10) with tiny epsilon)
      if (nuts_probing && !is_warmup && sample_idx < nuts_probe_window) {
        if (iter_treedepth >= max_treedepth) nuts_probe_maxd++;
        if (sample_idx == nuts_probe_window - 1) {
          nuts_probing = false;  // Probe window complete
          if (nuts_probe_maxd >= (nuts_probe_window * 8 + 9) / 10) {
            result.n_max_treedepth += 0;  // Already counted above
            if (verbose) {
              REprintf("  [NUTS] %d/%d initial sampling iterations hit max treedepth (%d). "
                       "Consider increasing max_treedepth or reparameterizing.\n",
                       nuts_probe_maxd, nuts_probe_window, max_treedepth);
            }
          }
        }
      }
    } else {
      // -----------------------------------------------------------------
      // Fixed-trajectory HMC (original code)
      // -----------------------------------------------------------------

      // Sample momentum and compute kinetic energy
      std::vector<double> p(n_params);
      double kinetic_current = 0.0;
      double H_current;

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        // L-BFGS: Sample p ~ N(0, B) where B ≈ 1/gamma * I (warmup only)
        std::vector<double> sqrt_diag = lbfgs_state.get_sqrt_B_diag();
        if ((int)sqrt_diag.size() == n_params) {
          for (int i = 0; i < n_params; i++) {
            p[i] = normal(rng) * sqrt_diag[i];
          }
          kinetic_current = lbfgs_state.kinetic_energy(p);
          H_current = -log_prob_current + kinetic_current;
        } else {
          mass.sample_momentum(p.data(), rng);
          kinetic_current = mass.kinetic_energy(p.data());
          H_current = -log_prob_current + kinetic_current;
        }
      } else {
        mass.sample_momentum(p.data(), rng);
        kinetic_current = mass.kinetic_energy(p.data());
        H_current = -log_prob_current + kinetic_current;
      }

      // Leapfrog integration
      std::vector<double> q_prop = q;
      std::vector<double> p_prop = p;

      // Determine effective L for this iteration
      int L_eff = L;
      if (use_nuts && use_lbfgs && !lbfgs_warmup_done) {
        // During L-BFGS warmup with NUTS mode, use fixed L=20
        L_eff = 20;
      }

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        // L-BFGS leapfrog
        std::vector<double> grad(n_params);
        compute_gradient(q_prop, data, layout, grad);

        for (int l = 0; l < L_eff; l++) {
          for (int i = 0; i < n_params; i++) {
            p_prop[i] += 0.5 * epsilon * grad[i];
          }
          std::vector<double> Hp(n_params);
          lbfgs_state.multiply_H(p_prop, Hp);
          for (int i = 0; i < n_params; i++) {
            q_prop[i] += epsilon * Hp[i];
            if (!std::isfinite(q_prop[i])) {
              divergent = true;
              break;
            }
          }
          if (divergent) break;
          compute_gradient(q_prop, data, layout, grad);
          for (int i = 0; i < n_params; i++) {
            p_prop[i] += 0.5 * epsilon * grad[i];
          }
          for (int i = 0; i < n_params; i++) {
            if (!std::isfinite(p_prop[i]) || std::abs(p_prop[i]) > 1e10) {
              divergent = true;
              break;
            }
          }
          if (divergent) break;
        }
      } else {
        // Standard leapfrog
        for (int l = 0; l < L_eff; l++) {
          LeapfrogResult lf;
          if (use_mass_matrix) {
            lf = leapfrog_step(q_prop, p_prop, epsilon, data, layout, mass.inv_mass_diag.data());
          } else {
            lf = leapfrog_step(q_prop, p_prop, epsilon, data, layout);
          }
          q_prop = lf.q;
          p_prop = lf.p;
          if (lf.divergent) {
            divergent = true;
            break;
          }
        }
      }

      // Compute proposed Hamiltonian (use same log-post as gradient mode)
      double log_prob_prop;
      if (g_gradient_mode == GradientMode::AUTODIFF_ARENA ||
          g_gradient_mode == GradientMode::AUTODIFF_FWD ||
          g_gradient_mode == GradientMode::AUTODIFF_TAPE) {
        log_prob_prop = tulpa::compute_log_post_impl(q_prop, data, layout);
      } else {
        log_prob_prop = compute_log_post(q_prop, data, layout);
      }
      double kinetic_prop = 0.0;

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        kinetic_prop = lbfgs_state.kinetic_energy(p_prop);
      } else {
        kinetic_prop = mass.kinetic_energy(p_prop.data());
      }
      double H_prop = -log_prob_prop + kinetic_prop;

      // Metropolis accept/reject
      alpha = std::min(1.0, std::exp(H_current - H_prop));
      if (!std::isfinite(alpha)) alpha = 0.0;

      std::uniform_real_distribution<double> unif01(0.0, 1.0);
      bool accepted = (unif01(rng) < alpha) && !divergent;
      if (accepted) {
        q = q_prop;
        log_prob_current = log_prob_prop;
        n_accept++;
        // Update cached gradient for transition to NUTS after L-BFGS warmup
        if (use_nuts) {
          compute_gradient(q, data, layout, current_grad);
        }
      }
      if (divergent) n_divergent++;

      // Adaptation during warmup
      if (is_warmup) {
        epsilon = da.update(alpha);
        // Only collect mass stats during mass adaptation phase (A5)
        if (iter >= init_buffer && iter < n_warmup - term_buffer) {
          mass_stats.update(q);
          if (mass.type == MassMatrixType::DENSE) {
            cov_stats.update(q);
          }
        }
        // On last warmup iteration, use averaged step size for sampling (A1)
        if (iter == n_warmup - 1) {
          epsilon = da.final_epsilon();
        }
      }

      // L-BFGS update: collect (s, y) pairs from accepted samples (warmup only)
      if (use_lbfgs && !lbfgs_warmup_done) {
        std::vector<double> grad_current(n_params);
        compute_gradient(q, data, layout, grad_current);

        if (!lbfgs_initialized) {
          q_prev = q;
          grad_prev = grad_current;
          lbfgs_initialized = true;
        } else if (accepted) {
          std::vector<double> s(n_params), y(n_params);
          for (int i = 0; i < n_params; i++) {
            s[i] = q[i] - q_prev[i];
            y[i] = grad_current[i] - grad_prev[i];
          }
          lbfgs_state.add_pair(s, y);
          q_prev = q;
          grad_prev = grad_current;
        }
      }

      iter_n_leapfrog = L_eff;
    }  // end fixed-trajectory HMC

    // Store sample (flat row-major storage, single memcpy)
    if (!is_warmup) {
      // NC GP: transform z -> w for stored samples (keep q as z for sampling)
      if (data.gp_parameterization == 1 && data.has_gp && layout.is_gp) {
          double sigma2_store = std::exp(q[layout.log_sigma2_gp_idx]);
          double phi_store = std::exp(q[layout.log_phi_gp_idx]);
          static thread_local tulpa_gp::NNGPNCWorkspace nc_ws_store;
          tulpa_gp::nngp_nc_forward(&q[layout.gp_w_start], sigma2_store, phi_store,
                                     data.gp_data, nc_ws_store);
          // Copy q, replace z with w
          std::memcpy(result.sample_row(sample_idx), q.data(),
                      n_params * sizeof(double));
          double* row = result.sample_row(sample_idx);
          int N_gp = data.gp_data.n_obs;
          for (int i = 0; i < N_gp; i++) {
              row[layout.gp_w_start + i] = nc_ws_store.w[i];
          }
      } else {
          std::memcpy(result.sample_row(sample_idx), q.data(),
                      n_params * sizeof(double));
      }
      result.log_prob[sample_idx] = log_prob_current;
      result.accept_prob[sample_idx] = alpha;
      result.n_leapfrog[sample_idx] = iter_n_leapfrog;
      result.divergent[sample_idx] = divergent ? 1 : 0;
      result.treedepth[sample_idx] = iter_treedepth;

      // Collapsed: store mode values from the last gradient evaluation
      if (data.gp_collapsed && data.has_gp && result.n_gp_collapsed > 0) {
          collapsed_gp_store_sample(sample_idx, collapsed_gp_ws,
              result.gp_w_star_flat, result.n_gp_collapsed);
      }
      if ((data.icar_collapsed || data.bym2_collapsed) && result.n_icar_collapsed > 0) {
          collapsed_icar_store_sample(sample_idx, data, collapsed_icar_ws,
              result.icar_phi_star_flat, result.bym2_theta_star_flat,
              result.n_icar_collapsed);
      }

      sample_idx++;
    } else {
      warmup_total_leapfrog += iter_n_leapfrog;  // TEMP: diagnostic
    }

    // Note: verbose output disabled in parallel - not thread-safe
    // Progress will be reported after parallel region
  }

  result.epsilon = da.final_epsilon();

  // Diagnostic stats - only when verbose
  if (verbose) {
    int sampling_total_lf = 0;
    for (int i = 0; i < result.n_sample; i++) sampling_total_lf += result.n_leapfrog[i];
    REprintf("  [STATS] Chain %d: metric=%s, adapted=%d, warmup_LF=%d, sampling_LF=%d, total_LF=%d, epsilon=%.6f\n",
             chain_id + 1,
             metric_name(mass.type),
             (int)mass.adapted,
             warmup_total_leapfrog, sampling_total_lf,
             warmup_total_leapfrog + sampling_total_lf, result.epsilon);
  }

  if (verbose && (softabs_retries > 0 || softabs_metric_active)) {
    REprintf("  [SoftAbs] Chain %d: metric=%s, %d divergent retried (up to %d attempts), %d resolved (%d remained)\n",
             chain_id + 1,
             softabs_metric_active ? "active" : "inactive",
             softabs_retries, SOFTABS_MAX_RETRIES, softabs_successes,
             softabs_retries - softabs_successes);
  }

  return result;
}

// R wrapper version - for single chain or non-parallel use
HMCResult run_hmc_chain(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    int riemannian
) {
  // Runtime gradient check: compare active gradient function against numerical
  if (g_gradient_mode != GradientMode::NUMERICAL) {
    bool grad_ok = verify_gradient_runtime(q_init, data, layout, 1e-4);
    if (!grad_ok) {
      g_gradient_mode = GradientMode::NUMERICAL;
      Rcpp::warning(
        "Gradient mismatch detected: active gradient function disagrees with "
        "numerical gradients (max rel diff > 1e-4). Falling back to numerical "
        "gradients (mode='N'). This is slower but correct. Please report this "
        "as a bug at https://github.com/gcol33/numdenom/issues"
      );
    }
  }

  // Run C++ version - pass verbose through for debugging
  HMCResultCpp cpp_result = run_hmc_chain_cpp(
    q_init, data, layout, n_iter, n_warmup, L, chain_id, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian
  );

  // Convert to R result
  int n_params = q_init.size();
  HMCResult result = cpp_to_r_result(cpp_result, n_params);

  if (verbose) {
    int n_div = 0;
    for (int i = 0; i < cpp_result.n_sample; i++) {
      n_div += cpp_result.divergent[i];
    }
    Rcpp::Rcerr << "Chain " << (chain_id + 1) << " complete. "
                << "Divergent: " << n_div << std::endl;
  }

  return result;
}

// =====================================================================
// Run multiple chains in parallel using OpenMP
// =====================================================================

std::vector<HMCResult> run_hmc_parallel_chains(
    const std::vector<double>& q_init,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    int riemannian
) {
  ParamLayout layout = compute_param_layout(data);
  int n_params = layout.total_params;

  // Runtime gradient check: compare active gradient function against numerical
  // BEFORE spawning parallel chains. This is single-threaded, so R API and
  // g_gradient_mode modification are safe.
  if (g_gradient_mode != GradientMode::NUMERICAL) {
    bool grad_ok = verify_gradient_runtime(q_init, data, layout, 1e-4);
    if (!grad_ok) {
      g_gradient_mode = GradientMode::NUMERICAL;
      REprintf("[numdenom] Falling back to numerical gradients for all chains.\n");
    }
  }

  // Use pure C++ containers in parallel region
  std::vector<HMCResultCpp> cpp_results(n_chains);

  // Thread-safe autodiff: Each chain creates its own tape via TapeScope (RAII).
  // All gradient modes (N, A, A_t, H) are now thread-safe and can run in parallel.
  // The old global tape limitation has been removed.

#ifdef _OPENMP
  if (n_chains > 1) {
    // Run chains in parallel - all gradient modes are now thread-safe
    #pragma omp parallel for schedule(static) num_threads(n_chains)
    for (int c = 0; c < n_chains; c++) {
      cpp_results[c] = run_hmc_chain_cpp(
        q_init, data, layout,
        n_iter, n_warmup, L, c, seed, false, max_treedepth, metric_type, adapt_delta, riemannian
      );
    }
  } else {
    // Single chain - run sequentially with verbose output
    cpp_results[0] = run_hmc_chain_cpp(
      q_init, data, layout,
      n_iter, n_warmup, L, 0, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian
    );
  }
#else
  // Sequential fallback when OpenMP not available
  for (int c = 0; c < n_chains; c++) {
    cpp_results[c] = run_hmc_chain_cpp(
      q_init, data, layout,
      n_iter, n_warmup, L, c, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian
    );
  }
#endif

  // Convert to R objects outside parallel region (single-threaded)
  std::vector<HMCResult> results(n_chains);
  for (int c = 0; c < n_chains; c++) {
    results[c] = cpp_to_r_result(cpp_results[c], n_params);

    if (verbose && n_chains > 1) {
      // Print summary if we ran in parallel (verbose was disabled during parallel run)
      int n_div = 0;
      for (int i = 0; i < cpp_results[c].n_sample; i++) {
        n_div += cpp_results[c].divergent[i];
      }
      Rcpp::Rcerr << "Chain " << (c + 1) << " complete. "
                  << "Divergent: " << n_div << std::endl;
    }
  }

  return results;
}

} // namespace tulpa_hmc

// =====================================================================
// R EXPORTS
// =====================================================================

// HMC sampler with bundled list arguments to avoid R's 65-arg limit for .Call
// Parameters are bundled into logical groups:
//   re_params: random effects (group, n_groups, n_terms, group_matrix, slopes, etc.)
//   spatial_params: spatial structure (type, group, adjacency, etc.)
//   temporal_params: temporal structure (type, time_idx, group_idx, etc.)
//   prior_params: prior hyperparameters
//   zi_params: zero-inflation (type, X_zi, prior_sd)
//   latent_params: latent factors
//   st_params: spatiotemporal interaction
// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_num_cont,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    std::string model_type_str,
    Rcpp::List re_params,
    Rcpp::List spatial_params,
    Rcpp::List temporal_params,
    Rcpp::List prior_params,
    Rcpp::List zi_params,
    Rcpp::List latent_params,
    Rcpp::List st_params,
    Rcpp::List tvc_params,  // Time-varying coefficients
    Rcpp::List svc_params,  // Spatially-varying coefficients
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    int n_threads,
    bool verbose,
    std::string gradient_mode_str = "auto",
    int max_treedepth = 10,
    std::string metric_str = "auto",
    double adapt_delta = -1.0,
    int riemannian = -1,
    bool gradient_check_only = false
) {
  using namespace tulpa_hmc;

  // Set global gradient mode from R parameter
  GradientMode grad_mode = parse_gradient_mode(gradient_mode_str);
  set_gradient_mode(grad_mode);

  // Parse metric type
  MassMatrixType metric_type = parse_metric_type(metric_str);

  // =========================================================================
  // Extract bundled parameters from lists with defensive checks
  // =========================================================================

  // Random effects parameters
  // Use eager deep copies to prevent R GC from invalidating memory during HMC
  std::vector<int> re_group = Rcpp::as<std::vector<int>>(re_params["group"]);
  int n_re_groups = Rcpp::as<int>(re_params["n_groups"]);
  int n_re_terms = Rcpp::as<int>(re_params["n_terms"]);

  // Handle group_matrix which may be numeric or integer
  Rcpp::IntegerMatrix re_group_matrix;
  SEXP group_mat_sexp = re_params["group_matrix"];
  if (Rf_isMatrix(group_mat_sexp)) {
    if (TYPEOF(group_mat_sexp) == INTSXP) {
      re_group_matrix = Rcpp::as<Rcpp::IntegerMatrix>(group_mat_sexp);
    } else {
      // Convert numeric to integer
      Rcpp::NumericMatrix nm(group_mat_sexp);
      re_group_matrix = Rcpp::IntegerMatrix(nm.nrow(), nm.ncol());
      for (int i = 0; i < nm.nrow(); i++) {
        for (int j = 0; j < nm.ncol(); j++) {
          re_group_matrix(i, j) = static_cast<int>(nm(i, j));
        }
      }
    }
  } else {
    // Create dummy matrix
    re_group_matrix = Rcpp::IntegerMatrix(1, 1);
    re_group_matrix(0, 0) = 0;
  }

  std::vector<int> re_n_groups_vec = Rcpp::as<std::vector<int>>(re_params["n_groups_vec"]);
  bool has_re_slopes = Rcpp::as<bool>(re_params["has_slopes"]);
  bool has_re_correlated_slopes = Rcpp::as<bool>(re_params["has_correlated_slopes"]);
  std::vector<int> re_n_coefs_vec = Rcpp::as<std::vector<int>>(re_params["n_coefs_vec"]);
  std::vector<int> re_correlated_vec = Rcpp::as<std::vector<int>>(re_params["correlated_vec"]);
  std::vector<int> re_n_chol_vec = Rcpp::as<std::vector<int>>(re_params["n_chol_vec"]);
  Rcpp::List slope_matrices_list = re_params["slope_matrices"];

  // Spatial parameters (eager deep copies)
  std::string spatial_type_str = Rcpp::as<std::string>(spatial_params["type"]);
  std::vector<int> spatial_group = Rcpp::as<std::vector<int>>(spatial_params["group"]);
  int n_spatial_units = Rcpp::as<int>(spatial_params["n_units"]);
  std::vector<int> adj_row_ptr = Rcpp::as<std::vector<int>>(spatial_params["adj_row_ptr"]);
  std::vector<int> adj_col_idx = Rcpp::as<std::vector<int>>(spatial_params["adj_col_idx"]);
  std::vector<int> n_neighbors = Rcpp::as<std::vector<int>>(spatial_params["n_neighbors"]);
  double bym2_scale_factor = Rcpp::as<double>(spatial_params["bym2_scale"]);

  // Precision mass matrix data (Q_inv and L_Q for ICAR/BYM2)
  std::vector<double> spatial_Q_inv;
  std::vector<double> spatial_L_Q;
  if (spatial_params.containsElementNamed("Q_inv") &&
      spatial_params.containsElementNamed("L_Q")) {
    SEXP qi_sexp = spatial_params["Q_inv"];
    SEXP lq_sexp = spatial_params["L_Q"];
    if (!Rf_isNull(qi_sexp) && !Rf_isNull(lq_sexp)) {
      spatial_Q_inv = Rcpp::as<std::vector<double>>(qi_sexp);
      spatial_L_Q = Rcpp::as<std::vector<double>>(lq_sexp);
    }
  }

  // Temporal parameters (eager deep copies)
  std::string temporal_type_str = Rcpp::as<std::string>(temporal_params["type"]);
  std::vector<int> temporal_time_idx = Rcpp::as<std::vector<int>>(temporal_params["time_idx"]);
  std::vector<int> temporal_group_idx = Rcpp::as<std::vector<int>>(temporal_params["group_idx"]);
  int n_times = Rcpp::as<int>(temporal_params["n_times"]);
  int n_temporal_groups = Rcpp::as<int>(temporal_params["n_groups"]);
  int n_temporal_params = Rcpp::as<int>(temporal_params["n_params"]);
  bool temporal_cyclic = Rcpp::as<bool>(temporal_params["cyclic"]);
  bool temporal_shared = Rcpp::as<bool>(temporal_params["shared"]);
  double tau_temporal_shape = Rcpp::as<double>(temporal_params["tau_shape"]);
  double tau_temporal_rate = Rcpp::as<double>(temporal_params["tau_rate"]);

  // Prior parameters
  double sigma_beta = Rcpp::as<double>(prior_params["sigma_beta"]);
  double sigma_re_scale = Rcpp::as<double>(prior_params["sigma_re_scale"]);
  double phi_prior_shape = Rcpp::as<double>(prior_params["phi_shape"]);
  double phi_prior_rate = Rcpp::as<double>(prior_params["phi_rate"]);
  double tau_spatial_shape = Rcpp::as<double>(prior_params["tau_spatial_shape"]);
  double tau_spatial_rate = Rcpp::as<double>(prior_params["tau_spatial_rate"]);

  // Zero-inflation parameters
  std::string zi_type_str = Rcpp::as<std::string>(zi_params["type"]);

  // Handle X_zi which may be numeric matrix or NULL
  Rcpp::NumericMatrix X_zi;
  SEXP xzi_sexp = zi_params["X"];
  if (!Rf_isNull(xzi_sexp) && Rf_isMatrix(xzi_sexp)) {
    X_zi = Rcpp::as<Rcpp::NumericMatrix>(xzi_sexp);
  } else {
    // Create empty dummy matrix
    X_zi = Rcpp::NumericMatrix(1, 1);
    X_zi(0, 0) = 1.0;
  }
  double zi_prior_sd = Rcpp::as<double>(zi_params["prior_sd"]);

  // One-inflation parameters (for OI-binomial and ZOIB)
  Rcpp::NumericMatrix X_oi;
  SEXP xoi_sexp = zi_params["X_oi"];
  if (!Rf_isNull(xoi_sexp) && Rf_isMatrix(xoi_sexp)) {
    X_oi = Rcpp::as<Rcpp::NumericMatrix>(xoi_sexp);
  } else {
    // Create empty dummy matrix
    X_oi = Rcpp::NumericMatrix(1, 1);
    X_oi(0, 0) = 1.0;
  }
  int p_oi = 0;
  SEXP p_oi_sexp = zi_params["p_oi"];
  if (!Rf_isNull(p_oi_sexp)) {
    p_oi = Rcpp::as<int>(p_oi_sexp);
  }
  double oi_prior_sd = zi_prior_sd;  // Default to same as ZI
  SEXP oi_prior_sd_sexp = zi_params["oi_prior_sd"];
  if (!Rf_isNull(oi_prior_sd_sexp)) {
    oi_prior_sd = Rcpp::as<double>(oi_prior_sd_sexp);
  }

  // Latent factor parameters
  bool has_latent = Rcpp::as<bool>(latent_params["has_latent"]);
  int latent_n_factors = Rcpp::as<int>(latent_params["n_factors"]);
  bool latent_shared = Rcpp::as<bool>(latent_params["shared"]);
  bool latent_scale = Rcpp::as<bool>(latent_params["scale"]);
  int latent_constraint = Rcpp::as<int>(latent_params["constraint"]);
  double latent_sigma_prior_rate = Rcpp::as<double>(latent_params["sigma_prior_rate"]);

  // =========================================================================
  // Set up model data
  // =========================================================================
  ModelData data;

  // Copy response data
  data.legacy.y_num = std::vector<int>(y_num.begin(), y_num.end());
  data.legacy.y_denom = std::vector<int>(y_denom.begin(), y_denom.end());
  data.legacy.y_num_cont = std::vector<double>(y_num_cont.begin(), y_num_cont.end());
  data.legacy.y_denom_cont = std::vector<double>(y_denom_cont.begin(), y_denom_cont.end());

  // Flatten design matrices for cache efficiency
  data.legacy.p_num = X_num.ncol();
  data.legacy.p_denom = X_denom.ncol();
  data.N = y_num.size();

  data.legacy.X_num_flat.resize(data.N * data.legacy.p_num);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.legacy.p_num; j++) {
      data.legacy.X_num_flat[i * data.legacy.p_num + j] = X_num(i, j);
    }
  }

  data.legacy.X_denom_flat.resize(data.N * data.legacy.p_denom);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.legacy.p_denom; j++) {
      data.legacy.X_denom_flat[i * data.legacy.p_denom + j] = X_denom(i, j);
    }
  }

  // Random effects (already deep copied above)
  data.re_group = re_group;
  data.n_re_groups = n_re_groups;
  data.n_re_terms = n_re_terms;

  // Random slopes flags
  data.has_re_slopes = has_re_slopes;
  data.has_re_correlated_slopes = has_re_correlated_slopes;

  // RE parameterization: 0 = centered, 1 = non-centered
  // Non-centered uses z ~ N(0,1) prior, centered uses re ~ N(0, sigma^2)
  data.re_parameterization = Rcpp::as<int>(re_params["parameterization"]);

  if (n_re_terms > 0) {
    // Multi-term RE structure (with or without slopes)
    data.re_group_multi.resize(n_re_terms);
    data.re_n_groups_multi.resize(n_re_terms);
    data.re_offsets.resize(n_re_terms);
    data.re_n_coefs.resize(n_re_terms);
    data.re_correlated.resize(n_re_terms);
    data.re_n_chol.resize(n_re_terms);
    data.re_n_slopes.resize(n_re_terms);
    data.re_slope_matrices.resize(n_re_terms);

    int offset = 0;
    int total_re_params = 0;
    int total_sigma_params = 0;
    int total_chol_params = 0;

    for (int t = 0; t < n_re_terms; t++) {
      data.re_n_groups_multi[t] = re_n_groups_vec[t];
      data.re_offsets[t] = offset;

      // Slopes metadata
      int n_coefs = has_re_slopes ? re_n_coefs_vec[t] : 1;
      data.re_n_coefs[t] = n_coefs;
      data.re_correlated[t] = has_re_slopes ? (re_correlated_vec[t] != 0) : false;
      data.re_n_chol[t] = has_re_slopes ? re_n_chol_vec[t] : 0;
      data.re_n_slopes[t] = n_coefs - 1;  // Number of slopes (excluding intercept)

      // Process slope matrix for this term
      if (has_re_slopes && data.re_n_slopes[t] > 0 && slope_matrices_list.size() > t) {
        SEXP mat_sexp = slope_matrices_list[t];
        if (!Rf_isNull(mat_sexp)) {
          Rcpp::NumericMatrix slope_mat(mat_sexp);
          int n_rows = slope_mat.nrow();
          int n_cols = slope_mat.ncol();
          data.re_slope_matrices[t].resize(n_rows * n_cols);
          for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
              data.re_slope_matrices[t][i * n_cols + j] = slope_mat(i, j);
            }
          }
        }
      }

      offset += re_n_groups_vec[t];
      total_re_params += re_n_groups_vec[t] * n_coefs;
      total_sigma_params += n_coefs;
      total_chol_params += data.re_n_chol[t];

      // Extract column t from re_group_matrix
      data.re_group_multi[t].resize(data.N);
      for (int i = 0; i < data.N; i++) {
        data.re_group_multi[t][i] = re_group_matrix(i, t);
      }
    }
    data.total_re_groups = offset;

    // Build contiguous flat array: obs-major layout re_group_multi_flat[i * n_re_terms + t]
    // Obs-major is cache-friendly: inner loop over terms for each observation
    data.re_group_multi_flat.resize(n_re_terms * data.N);
    for (int i = 0; i < data.N; i++) {
      for (int t = 0; t < n_re_terms; t++) {
        data.re_group_multi_flat[i * n_re_terms + t] = data.re_group_multi[t][i];
      }
    }
    data.total_re_params = total_re_params;
    data.total_sigma_params = total_sigma_params;
    data.total_chol_params = total_chol_params;
  } else {
    // No RE terms
    data.total_re_groups = n_re_groups;
    data.total_re_params = n_re_groups;
    data.total_sigma_params = (n_re_groups > 0) ? 1 : 0;
    data.total_chol_params = 0;
  }

  // Model type
  if (model_type_str == "binomial") {
    data.legacy.model_type = ModelType::BINOMIAL;
  } else if (model_type_str == "negbin_negbin") {
    data.legacy.model_type = ModelType::NEGBIN_NEGBIN;
  } else if (model_type_str == "poisson_gamma") {
    data.legacy.model_type = ModelType::POISSON_GAMMA;
  } else if (model_type_str == "negbin_gamma") {
    data.legacy.model_type = ModelType::NEGBIN_GAMMA;
  } else if (model_type_str == "gamma_gamma") {
    data.legacy.model_type = ModelType::GAMMA_GAMMA;
  } else if (model_type_str == "lognormal") {
    data.legacy.model_type = ModelType::LOGNORMAL;
  } else if (model_type_str == "beta_binomial") {
    data.legacy.model_type = ModelType::BETA_BINOMIAL;
  } else {
    data.legacy.model_type = ModelType::POISSON_GAMMA;  // fallback
  }

  // Spatial structure
  if (spatial_type_str == "icar") {
    data.spatial_type = SpatialType::ICAR;
  } else if (spatial_type_str == "bym2") {
    data.spatial_type = SpatialType::BYM2;
  } else {
    data.spatial_type = SpatialType::NONE;
  }

  data.spatial_group = spatial_group;  // Already deep copied above
  data.n_spatial_units = n_spatial_units;
  data.adj_row_ptr = adj_row_ptr;
  data.adj_col_idx = adj_col_idx;
  data.n_neighbors = n_neighbors;
  data.bym2_scale_factor = bym2_scale_factor;
  data.spatial_Q_inv = std::move(spatial_Q_inv);
  data.spatial_L_Q = std::move(spatial_L_Q);

  // Collapsed ICAR/BYM2 parameterization
  data.icar_collapsed = false;
  data.bym2_collapsed = false;
  if (spatial_params.containsElementNamed("parameterization")) {
      std::string spatial_param_str = Rcpp::as<std::string>(spatial_params["parameterization"]);
      if (spatial_param_str == "collapsed") {
          if (data.spatial_type == SpatialType::ICAR) {
              data.icar_collapsed = true;
          } else if (data.spatial_type == SpatialType::BYM2) {
              data.bym2_collapsed = true;
          }
      }
  }

  // Temporal structure
  if (temporal_type_str == "rw1") {
    data.temporal_type = TemporalType::RW1;
  } else if (temporal_type_str == "rw2") {
    data.temporal_type = TemporalType::RW2;
  } else if (temporal_type_str == "ar1") {
    data.temporal_type = TemporalType::AR1;
  } else if (temporal_type_str == "gp") {
    data.temporal_type = TemporalType::GP;

    // GP-specific parameters
    data.has_temporal_gp = true;
    data.temporal_gp_data.n_obs = data.n_times;  // Use n_times, not N (total obs)
    data.temporal_gp_data.n_groups = n_temporal_groups;
    data.temporal_gp_data.time_values = Rcpp::as<std::vector<double>>(temporal_params["time_values"]);
    data.temporal_gp_data.group_index = temporal_group_idx;

    // Parse covariance type
    std::string cov_type_str = Rcpp::as<std::string>(temporal_params["cov_type"]);
    data.temporal_gp_data.cov_type = tulpa_temporal_gp::parse_temporal_cov_type(cov_type_str);
    data.temporal_gp_data.nu = Rcpp::as<double>(temporal_params["nu"]);
    data.temporal_gp_data.period = Rcpp::as<double>(temporal_params["period"]);
    data.temporal_gp_data.shared = temporal_shared;

    // GP priors
    data.temporal_gp_sigma2_prior_U = Rcpp::as<double>(temporal_params["gp_sigma2_prior_U"]);
    data.temporal_gp_sigma2_prior_alpha = Rcpp::as<double>(temporal_params["gp_sigma2_prior_alpha"]);
    data.temporal_gp_phi_prior_lower = Rcpp::as<double>(temporal_params["gp_phi_prior_lower"]);
    data.temporal_gp_phi_prior_upper = Rcpp::as<double>(temporal_params["gp_phi_prior_upper"]);

    // Parameterization: 0=centered, 1=non-centered (default NC)
    std::string gp_param_str = "noncentered";
    if (temporal_params.containsElementNamed("gp_parameterization")) {
        gp_param_str = Rcpp::as<std::string>(temporal_params["gp_parameterization"]);
    }
    data.temporal_gp_parameterization = (gp_param_str == "centered") ? 0 : 1;
  } else if (temporal_type_str == "multiscale") {
    data.temporal_type = TemporalType::NONE;  // MS_t uses its own data path
    data.has_temporal_gp = false;
    data.has_multiscale_temporal = true;

    // Extract MS_t data from temporal_params
    data.multiscale_temporal_data.n_times = n_times;
    data.multiscale_temporal_data.n_groups = n_temporal_groups;
    data.multiscale_temporal_data.n_obs = data.N;
    data.multiscale_temporal_data.time_index = temporal_time_idx;
    data.multiscale_temporal_data.group_index = temporal_group_idx;
    data.multiscale_temporal_data.shared = temporal_shared;

    int ms_seasonal_period = 0;
    if (temporal_params.containsElementNamed("seasonal_period"))
      ms_seasonal_period = Rcpp::as<int>(temporal_params["seasonal_period"]);
    data.multiscale_temporal_data.seasonal_period = ms_seasonal_period;

    std::string ms_trend_type = "rw1";
    if (temporal_params.containsElementNamed("trend_type"))
      ms_trend_type = Rcpp::as<std::string>(temporal_params["trend_type"]);
    std::string ms_short_type = "ar1";
    if (temporal_params.containsElementNamed("short_term_type"))
      ms_short_type = Rcpp::as<std::string>(temporal_params["short_term_type"]);
    data.multiscale_temporal_data.trend_type = tulpa_temporal::parse_temporal_type(ms_trend_type);
    data.multiscale_temporal_data.short_term_type = tulpa_temporal::parse_temporal_type(ms_short_type);

    // MS_t priors
    data.ms_sigma2_trend_prior_U = 1.0;
    data.ms_sigma2_trend_prior_alpha = 0.01;
    data.ms_sigma2_seasonal_prior_U = 1.0;
    data.ms_sigma2_seasonal_prior_alpha = 0.01;
    data.ms_sigma2_short_prior_U = 1.0;
    data.ms_sigma2_short_prior_alpha = 0.01;
    if (temporal_params.containsElementNamed("sigma2_trend_prior_U"))
      data.ms_sigma2_trend_prior_U = Rcpp::as<double>(temporal_params["sigma2_trend_prior_U"]);
    if (temporal_params.containsElementNamed("sigma2_trend_prior_alpha"))
      data.ms_sigma2_trend_prior_alpha = Rcpp::as<double>(temporal_params["sigma2_trend_prior_alpha"]);
    if (temporal_params.containsElementNamed("sigma2_short_prior_U"))
      data.ms_sigma2_short_prior_U = Rcpp::as<double>(temporal_params["sigma2_short_prior_U"]);
    if (temporal_params.containsElementNamed("sigma2_short_prior_alpha"))
      data.ms_sigma2_short_prior_alpha = Rcpp::as<double>(temporal_params["sigma2_short_prior_alpha"]);
  } else {
    data.temporal_type = TemporalType::NONE;
    data.has_temporal_gp = false;
  }

  data.temporal_time_idx = temporal_time_idx;  // Already deep copied above
  data.temporal_group_idx = temporal_group_idx;
  data.n_times = n_times;
  data.n_temporal_groups = n_temporal_groups;
  data.n_temporal_params = n_temporal_params;
  data.temporal_cyclic = temporal_cyclic;
  data.temporal_shared = temporal_shared;
  data.tau_temporal_shape = tau_temporal_shape;
  data.tau_temporal_rate = tau_temporal_rate;

  // Zero-inflation structure
  data.zi_type = tulpa_zi::parse_zi_type(zi_type_str);
  // Use explicit p_zi from R (not X_zi.ncol()) because OI-only models
  // pass a 1-column placeholder X_zi but p_zi=0
  {
    SEXP p_zi_sexp = zi_params["p_zi"];
    data.p_zi = (!Rf_isNull(p_zi_sexp)) ? Rcpp::as<int>(p_zi_sexp) : X_zi.ncol();
  }
  data.zi_prior_sd = zi_prior_sd;
  data.X_zi_flat.resize(data.N * data.p_zi);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_zi; j++) {
      data.X_zi_flat[i * data.p_zi + j] = X_zi(i, j);
    }
  }

  // One-inflation structure (for OI-binomial and ZOIB)
  data.p_oi = p_oi;
  data.oi_prior_sd = oi_prior_sd;
  if (p_oi > 0) {
    data.X_oi_flat.resize(data.N * data.p_oi);
    for (int i = 0; i < data.N; i++) {
      for (int j = 0; j < data.p_oi; j++) {
        data.X_oi_flat[i * data.p_oi + j] = X_oi(i, j);
      }
    }
  }

  // Priors
  data.sigma_beta = sigma_beta;
  data.sigma_re_scale = sigma_re_scale;
  data.phi_prior_shape = phi_prior_shape;
  data.phi_prior_rate = phi_prior_rate;
  data.tau_spatial_shape = tau_spatial_shape;
  data.tau_spatial_rate = tau_spatial_rate;

  // Parallelization
  data.n_threads = n_threads;

  // Initialize feature flags that are not used in cpp_hmc_fit (only in cpp_hmc_fit_gp)
  data.has_gp = false;
  data.has_multiscale_gp = false;
  data.has_multiscale_temporal = false;
  data.has_rsr = false;
  data.has_svc = false;
  data.has_hsgp = false;

  // Latent factors
  data.has_latent = has_latent;
  data.latent_n_factors = latent_n_factors;
  data.latent_shared = latent_shared;
  data.latent_scale = latent_scale;
  data.latent_constraint = latent_constraint;
  data.latent_sigma_prior_rate = latent_sigma_prior_rate;

  // Spatiotemporal interaction - extract from list
  bool has_spatiotemporal = st_params.size() > 0 && Rcpp::as<bool>(st_params["has_spatiotemporal"]);
  data.has_spatiotemporal = has_spatiotemporal;
  if (has_spatiotemporal) {
    // Extract parameters from list (eager deep copies to prevent R GC issues)
    std::string st_type_str = Rcpp::as<std::string>(st_params["type"]);
    bool st_shared = Rcpp::as<bool>(st_params["shared"]);
    int st_n_spatial = Rcpp::as<int>(st_params["n_spatial"]);
    int st_n_times = Rcpp::as<int>(st_params["n_times"]);
    int st_n_params = Rcpp::as<int>(st_params["n_params"]);
    std::vector<int> st_s_idx = Rcpp::as<std::vector<int>>(st_params["s_idx"]);
    std::vector<int> st_t_idx = Rcpp::as<std::vector<int>>(st_params["t_idx"]);
    std::vector<int> st_flat = Rcpp::as<std::vector<int>>(st_params["st_flat"]);
    std::string st_temporal_type_str = Rcpp::as<std::string>(st_params["temporal_type"]);
    bool st_temporal_cyclic = Rcpp::as<bool>(st_params["temporal_cyclic"]);
    std::vector<int> st_adj_row_ptr = Rcpp::as<std::vector<int>>(st_params["adj_row_ptr"]);
    std::vector<int> st_adj_col_idx = Rcpp::as<std::vector<int>>(st_params["adj_col_idx"]);
    double st_sigma2_prior_U = Rcpp::as<double>(st_params["sigma2_prior_U"]);
    double st_sigma2_prior_alpha = Rcpp::as<double>(st_params["sigma2_prior_alpha"]);

    // Parse ST type (accept both R-side "I"/"IV" and legacy "type_i"/"type_iv")
    if (st_type_str == "I" || st_type_str == "type_i") {
      data.spatiotemporal_data.type = STType::TYPE_I;
    } else if (st_type_str == "II" || st_type_str == "type_ii") {
      data.spatiotemporal_data.type = STType::TYPE_II;
    } else if (st_type_str == "III" || st_type_str == "type_iii") {
      data.spatiotemporal_data.type = STType::TYPE_III;
    } else if (st_type_str == "IV" || st_type_str == "type_iv") {
      data.spatiotemporal_data.type = STType::TYPE_IV;
    } else if (st_type_str == "separable") {
      data.spatiotemporal_data.type = STType::SEPARABLE;
    } else if (st_type_str == "nonsep_gp") {
      data.spatiotemporal_data.type = STType::NONSEP_GP;
    } else {
      Rcpp::stop("Unknown spatiotemporal type: '%s'. Expected one of: I, II, III, IV, separable, nonsep_gp",
                 st_type_str.c_str());
    }

    data.spatiotemporal_data.shared = st_shared;
    data.spatiotemporal_data.n_spatial = st_n_spatial;
    data.spatiotemporal_data.n_times = st_n_times;
    data.spatiotemporal_data.n_params = st_n_params;

    // Observation indexing (already deep copied above)
    data.spatiotemporal_data.s_idx = st_s_idx;
    data.spatiotemporal_data.t_idx = st_t_idx;
    data.spatiotemporal_data.st_flat = st_flat;

    // Temporal type for Type II/IV
    if (st_temporal_type_str == "rw1") {
      data.spatiotemporal_data.temporal_type = TemporalType::RW1;
    } else if (st_temporal_type_str == "rw2") {
      data.spatiotemporal_data.temporal_type = TemporalType::RW2;
    } else if (st_temporal_type_str == "ar1") {
      data.spatiotemporal_data.temporal_type = TemporalType::AR1;
    } else {
      data.spatiotemporal_data.temporal_type = TemporalType::RW1;  // Default
    }
    data.spatiotemporal_data.temporal_cyclic = st_temporal_cyclic;

    // Spatial adjacency for Type III/IV (already deep copied above)
    data.spatiotemporal_data.adj_row_ptr = st_adj_row_ptr;
    data.spatiotemporal_data.adj_col_idx = st_adj_col_idx;

    // Prior parameters
    data.st_sigma2_prior_U = st_sigma2_prior_U;
    data.st_sigma2_prior_alpha = st_sigma2_prior_alpha;

    // Parameterization: centered by default. NC requires spectral decomposition
    // (Kronecker eigenvectors of Q_s ⊗ Q_t) which is not yet implemented.
    // Simple scaling NC (z = delta * sqrt(tau)) preserves GMRF anisotropy
    // and makes performance worse (eps=0.003, td=11.5 vs eps=0.006, td=10).
    data.st_parameterization = 0;  // Always centered for now
    if (st_params.containsElementNamed("parameterization")) {
      std::string st_param_str = Rcpp::as<std::string>(st_params["parameterization"]);
      data.st_parameterization = (st_param_str == "centered") ? 0 : 1;
    }

    // Kronecker precision data for ST_IV (precomputed in R)
    if (st_params.containsElementNamed("Qs_inv") &&
        st_params.containsElementNamed("Qt_inv")) {
      SEXP qs_sexp = st_params["Qs_inv"];
      SEXP ls_sexp = st_params["Ls"];
      SEXP qt_sexp = st_params["Qt_inv"];
      SEXP lt_sexp = st_params["Lt"];
      if (!Rf_isNull(qs_sexp) && !Rf_isNull(qt_sexp) &&
          !Rf_isNull(ls_sexp) && !Rf_isNull(lt_sexp)) {
        data.st_Qs_inv = Rcpp::as<std::vector<double>>(qs_sexp);
        data.st_Ls = Rcpp::as<std::vector<double>>(ls_sexp);
        data.st_Qt_inv = Rcpp::as<std::vector<double>>(qt_sexp);
        data.st_Lt = Rcpp::as<std::vector<double>>(lt_sexp);
      }
    }

    // HSGP-ST: spectral basis spatiotemporal interaction
    data.st_is_hsgp = false;
    if (st_params.containsElementNamed("st_is_hsgp") &&
        Rcpp::as<bool>(st_params["st_is_hsgp"])) {
      data.st_is_hsgp = true;
      data.spatiotemporal_data.is_hsgp = true;
      int st_hsgp_m = Rcpp::as<int>(st_params["hsgp_m"]);
      double st_hsgp_c = Rcpp::as<double>(st_params["hsgp_c"]);
      std::vector<double> st_hsgp_coords = Rcpp::as<std::vector<double>>(st_params["hsgp_coords"]);
      bool st_hsgp_scale = true;
      if (st_params.containsElementNamed("hsgp_scale_coords"))
        st_hsgp_scale = Rcpp::as<bool>(st_params["hsgp_scale_coords"]);

      // Setup HSGP basis (Phi matrix + eigenvalues)
      tulpa_hsgp::setup_hsgp_2d(
        st_hsgp_coords, data.N,
        st_hsgp_m, st_hsgp_c, st_hsgp_scale,
        data.st_hsgp_data);
      data.spatiotemporal_data.hsgp_m_total = data.st_hsgp_data.m_total;
      // Override n_spatial and n_params for HSGP-ST
      data.spatiotemporal_data.n_spatial = data.st_hsgp_data.m_total;
      data.spatiotemporal_data.n_params = data.st_hsgp_data.m_total * st_n_times;
    }
  } else {
    data.spatiotemporal_data.type = STType::NONE;
  }

  // TVC (Temporally-Varying Coefficients) parameters
  bool has_tvc = tvc_params.size() > 0 && Rcpp::as<bool>(tvc_params["has_tvc"]);
  data.has_tvc = has_tvc;
  if (has_tvc) {
    // Extract TVC parameters (eager deep copies to prevent R GC issues)
    int tvc_n_tvc = Rcpp::as<int>(tvc_params["n_tvc"]);
    int tvc_n_times = Rcpp::as<int>(tvc_params["n_times"]);
    int tvc_n_groups = Rcpp::as<int>(tvc_params["n_groups"]);
    std::string tvc_structure_str = Rcpp::as<std::string>(tvc_params["structure"]);
    bool tvc_shared = Rcpp::as<bool>(tvc_params["shared"]);
    bool tvc_cyclic = Rcpp::as<bool>(tvc_params["cyclic"]);
    std::vector<int> tvc_indices = Rcpp::as<std::vector<int>>(tvc_params["tvc_indices"]);
    std::vector<int> tvc_time_index = Rcpp::as<std::vector<int>>(tvc_params["time_index"]);
    std::vector<int> tvc_group_index = Rcpp::as<std::vector<int>>(tvc_params["group_index"]);
    std::vector<double> tvc_X_tvc = Rcpp::as<std::vector<double>>(tvc_params["X_tvc"]);
    double tvc_tau_shape = Rcpp::as<double>(tvc_params["tau_shape"]);
    double tvc_tau_rate = Rcpp::as<double>(tvc_params["tau_rate"]);

    // Populate TVC data structure
    data.tvc_data.n_obs = data.N;
    data.tvc_data.n_tvc = tvc_n_tvc;
    data.tvc_data.n_times = tvc_n_times;
    data.tvc_data.n_groups = tvc_n_groups;
    data.tvc_data.shared = tvc_shared;
    data.tvc_data.cyclic = tvc_cyclic;
    data.tvc_data.tvc_indices = tvc_indices;
    data.tvc_data.time_index = tvc_time_index;
    data.tvc_data.group_index = tvc_group_index;
    data.tvc_data.X_tvc = tvc_X_tvc;

    // Parse TVC temporal structure
    if (tvc_structure_str == "rw1") {
      data.tvc_data.structure = tulpa_temporal::TemporalType::RW1;
    } else if (tvc_structure_str == "rw2") {
      data.tvc_data.structure = tulpa_temporal::TemporalType::RW2;
    } else if (tvc_structure_str == "ar1") {
      data.tvc_data.structure = tulpa_temporal::TemporalType::AR1;
    } else if (tvc_structure_str == "iid") {
      data.tvc_data.structure = tulpa_temporal::TemporalType::IID;
    } else {
      data.tvc_data.structure = tulpa_temporal::TemporalType::RW1;  // Default
    }

    // Prior parameters
    data.tvc_tau_shape = tvc_tau_shape;
    data.tvc_tau_rate = tvc_tau_rate;

    // Pre-allocate gradient workspace buffers (avoids per-call heap allocation)
    data.tvc_data.init_workspace();
  } else {
    data.tvc_data.n_tvc = 0;
    data.tvc_data.n_times = 0;
    data.tvc_data.n_groups = 1;
  }

  // SVC (Spatially-Varying Coefficients) parameters
  bool has_svc = svc_params.size() > 0 && Rcpp::as<bool>(svc_params["has_svc"]);
  data.has_svc = has_svc;
  if (has_svc) {
    // Extract SVC parameters (eager deep copies to prevent R GC issues)
    int svc_n_svc = Rcpp::as<int>(svc_params["n_svc"]);
    int svc_nn = Rcpp::as<int>(svc_params["nn"]);
    bool svc_shared = Rcpp::as<bool>(svc_params["shared"]);
    std::string svc_cov_type_str = Rcpp::as<std::string>(svc_params["cov_type"]);
    std::vector<double> svc_coords = Rcpp::as<std::vector<double>>(svc_params["coords"]);
    std::vector<int> svc_indices = Rcpp::as<std::vector<int>>(svc_params["svc_indices"]);
    std::vector<double> svc_X_svc = Rcpp::as<std::vector<double>>(svc_params["X_svc"]);
    double svc_sigma2_scale = Rcpp::as<double>(svc_params["sigma2_prior_scale"]);
    double svc_phi_lower = Rcpp::as<double>(svc_params["phi_prior_lower"]);
    double svc_phi_upper = Rcpp::as<double>(svc_params["phi_prior_upper"]);

    // Check if this is HSGP-based SVC
    std::string svc_approx = "nngp";
    if (svc_params.containsElementNamed("svc_approx")) {
      svc_approx = Rcpp::as<std::string>(svc_params["svc_approx"]);
    }
    data.svc_is_hsgp = (svc_approx == "hsgp");

    // Populate SVC data structure (shared fields)
    data.svc_data.n_obs = data.N;
    data.svc_data.n_svc = svc_n_svc;
    data.svc_data.shared = svc_shared;
    data.svc_data.coords = svc_coords;
    data.svc_data.svc_indices = svc_indices;
    data.svc_data.X_svc = svc_X_svc;

    if (data.svc_is_hsgp) {
      // HSGP-based SVC: set up basis functions
      int hsgp_m = Rcpp::as<int>(svc_params["hsgp_m"]);
      double hsgp_c = Rcpp::as<double>(svc_params["hsgp_c"]);
      data.svc_hsgp_m_per_dim = hsgp_m;
      data.svc_hsgp_boundary_factor = hsgp_c;

      // Set up HSGP basis (shared across all SVC terms)
      tulpa_hsgp::setup_hsgp_2d(svc_coords, data.N, hsgp_m, hsgp_c,
                                  svc_shared, data.svc_hsgp_data);

      // No NNGP data needed
      data.svc_data.nn = 0;
    } else {
      // NNGP-based SVC: set up neighbor structure
      data.svc_data.nn = svc_nn;
      data.svc_data.nn_idx = Rcpp::as<std::vector<int>>(svc_params["nn_idx"]);
      data.svc_data.nn_dist = Rcpp::as<std::vector<double>>(svc_params["nn_dist"]);
      data.svc_data.nn_order = Rcpp::as<std::vector<int>>(svc_params["nn_order"]);
      data.svc_data.nn_order_inv = Rcpp::as<std::vector<int>>(svc_params["nn_order_inv"]);

      // Parse SVC covariance type
      if (svc_cov_type_str == "exponential") {
        data.svc_data.cov_type = tulpa_svc::CovType::EXPONENTIAL;
      } else if (svc_cov_type_str == "matern") {
        data.svc_data.cov_type = tulpa_svc::CovType::MATERN;
      } else if (svc_cov_type_str == "gaussian") {
        data.svc_data.cov_type = tulpa_svc::CovType::GAUSSIAN;
      } else if (svc_cov_type_str == "spherical") {
        data.svc_data.cov_type = tulpa_svc::CovType::SPHERICAL;
      } else {
        data.svc_data.cov_type = tulpa_svc::CovType::EXPONENTIAL;
      }
    }

    // Prior parameters
    data.svc_sigma2_prior_scale = svc_sigma2_scale;
    data.svc_phi_prior_lower = svc_phi_lower;
    data.svc_phi_prior_upper = svc_phi_upper;

    // Pre-allocate SVC workspace buffers
    data.svc_data.init_workspace();
  } else {
    data.svc_data.n_svc = 0;
    data.svc_data.n_obs = data.N;
    data.svc_data.nn = 0;
    data.svc_is_hsgp = false;
  }

  // Initialize parameters
  std::vector<double> q0(q_init.begin(), q_init.end());

  // Memory barrier to ensure all copies complete before HMC execution
  // This prevents R GC from invalidating memory during sampling
  std::atomic_thread_fence(std::memory_order_seq_cst);

  // =========================================================================
  // Gradient check only mode: compare N, A, A_r, H without sampling
  // =========================================================================
  if (gradient_check_only) {
    ParamLayout layout = compute_param_layout(data);
    double tol = 1e-4;

    // Compute numerical gradient (reference)
    std::vector<double> grad_N;
    compute_gradient_numerical(q0, data, layout, grad_N);

    // Also compute numerical gradient for impl-based log_post (used by A/A_r)
    std::vector<double> grad_N_impl;
    compute_gradient_numerical_impl(q0, data, layout, grad_N_impl);

    // Helper: compute max relative difference
    auto max_rel_diff = [](const std::vector<double>& a, const std::vector<double>& b) -> double {
      double mx = 0.0;
      for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        double diff = std::abs(a[i] - b[i]);
        double scale = std::max(1.0, std::max(std::abs(a[i]), std::abs(b[i])));
        mx = std::max(mx, diff / scale);
      }
      return mx;
    };

    // Helper: check if a function is autodiff-based (differentiates log_post_impl<T>)
    auto is_autodiff_fn = [](GradientFn fn) -> bool {
      return fn == &compute_gradient_arena ||
             fn == &compute_gradient_forward ||
             fn == &compute_gradient_autodiff;
    };

    // Try H mode
    double h_vs_n = -1.0;  // -1 means not available
    GradientFn h_fn = resolve_gradient_fn(GradientMode::HANDCODED, data, layout);
    if (h_fn != &compute_gradient_numerical && h_fn != &compute_gradient_numerical_impl) {
      // If H resolved to an autodiff function (fallback), compare against N_impl
      // since autodiff differentiates log_post_impl, not compute_log_post
      const auto& ref = is_autodiff_fn(h_fn) ? grad_N_impl : grad_N;
      std::vector<double> grad_H;
      h_fn(q0, data, layout, grad_H, nullptr);
      h_vs_n = max_rel_diff(grad_H, ref);

      // Print top-5 worst parameters when H fails
      if (h_vs_n > 1e-4) {
        std::vector<std::pair<double,int>> diffs;
        for (size_t i = 0; i < grad_H.size() && i < ref.size(); i++) {
          double diff = std::abs(grad_H[i] - ref[i]);
          double scale = std::max(1.0, std::max(std::abs(grad_H[i]), std::abs(ref[i])));
          diffs.push_back({diff/scale, (int)i});
        }
        std::sort(diffs.begin(), diffs.end(), [](auto&a,auto&b){return a.first>b.first;});
        Rprintf("  H gradient mismatch (top 5 of %d params):\n", (int)grad_H.size());
        for (int k = 0; k < std::min(5,(int)diffs.size()); k++) {
          int idx = diffs[k].second;
          Rprintf("    param[%d]: H=%.6e N=%.6e  rel_diff=%.2e\n",
                  idx, grad_H[idx], ref[idx], diffs[k].first);
        }
        // Print layout info for worst param
        int worst = diffs[0].second;
        Rprintf("  Layout: beta_num[%d-%d] beta_denom[%d-%d] re[%d+]\n",
                layout.legacy.beta_num_start, layout.legacy.beta_num_start+data.legacy.p_num-1,
                layout.legacy.beta_denom_start, layout.legacy.beta_denom_start+data.legacy.p_denom-1,
                layout.re_start);
        if (layout.has_spatial) Rprintf("  spatial[%d-%d]\n", layout.spatial_start, layout.spatial_end-1);
        if (layout.has_temporal) Rprintf("  temporal[%d-%d]\n", layout.temporal_start, layout.temporal_end-1);
        if (layout.has_spatiotemporal) Rprintf("  ST delta[%d+] tau_st[%d]\n",
                                               layout.st_delta_start, layout.log_tau_st_idx);
        if (layout.has_tvc) Rprintf("  TVC w[%d+] tau[%d+]\n", layout.tvc_w_start, layout.log_tau_tvc_start);
        if (layout.has_svc) Rprintf("  SVC w[%d+] sigma2[%d+] phi[%d+]\n",
                                    layout.svc_w_start, layout.log_sigma2_svc_start, layout.log_phi_svc_start);
        if (layout.has_re_slopes) Rprintf("  has_re_slopes=true n_re_terms=%d\n", data.n_re_terms);
        if (layout.is_temporal_gp) Rprintf("  temporal_gp: sigma2[%d] phi[%d]\n",
                                           layout.log_sigma2_temporal_gp_idx, layout.logit_phi_temporal_gp_idx);
        if (layout.has_multiscale_temporal) Rprintf("  ms_temporal\n");
        if (layout.has_latent) Rprintf("  latent\n");
      }
    }

    // Try A_r mode (arena autodiff)
    double ar_vs_n = -1.0;
    GradientFn ar_fn = resolve_gradient_fn(GradientMode::AUTODIFF_ARENA, data, layout);
    if (ar_fn != &compute_gradient_numerical && ar_fn != &compute_gradient_numerical_impl) {
      std::vector<double> grad_Ar;
      ar_fn(q0, data, layout, grad_Ar, nullptr);
      ar_vs_n = max_rel_diff(grad_Ar, grad_N_impl);
    }

    // Try A mode (forward autodiff)
    double a_vs_n = -1.0;
    GradientFn a_fn = resolve_gradient_fn(GradientMode::AUTODIFF_FWD, data, layout);
    if (a_fn != &compute_gradient_numerical && a_fn != &compute_gradient_numerical_impl) {
      std::vector<double> grad_A;
      a_fn(q0, data, layout, grad_A, nullptr);
      a_vs_n = max_rel_diff(grad_A, grad_N_impl);
    }

    // H vs A_r cross-check
    double h_vs_ar = -1.0;
    if (h_vs_n >= 0 && ar_vs_n >= 0) {
      std::vector<double> grad_H2, grad_Ar2;
      h_fn(q0, data, layout, grad_H2, nullptr);
      ar_fn(q0, data, layout, grad_Ar2, nullptr);
      h_vs_ar = max_rel_diff(grad_H2, grad_Ar2);

      // Print top-3 worst parameters when cross-check fails
      if (h_vs_ar > 1e-4) {
        std::vector<std::pair<double,int>> diffs;
        for (size_t i = 0; i < grad_H2.size() && i < grad_Ar2.size(); i++) {
          double diff = std::abs(grad_H2[i] - grad_Ar2[i]);
          double scale = std::max(1.0, std::max(std::abs(grad_H2[i]), std::abs(grad_Ar2[i])));
          diffs.push_back({diff/scale, (int)i});
        }
        std::sort(diffs.begin(), diffs.end(), [](auto&a,auto&b){return a.first>b.first;});
        Rprintf("  H vs A_r cross-check DIVERGE (top 3 of %d params):\n", (int)grad_H2.size());
        for (int k = 0; k < std::min(3,(int)diffs.size()); k++) {
          int idx = diffs[k].second;
          Rprintf("    param[%d]: H=%.8e A_r=%.8e  rel_diff=%.2e\n",
                  idx, grad_H2[idx], grad_Ar2[idx], diffs[k].first);
        }
      }
    }

    return Rcpp::List::create(
      Rcpp::Named("h_vs_n") = h_vs_n,
      Rcpp::Named("ar_vs_n") = ar_vs_n,
      Rcpp::Named("a_vs_n") = a_vs_n,
      Rcpp::Named("h_vs_ar") = h_vs_ar,
      Rcpp::Named("tol") = tol,
      // h_ok: check h_vs_n first; if it fails but h_vs_ar passes, H is still
      // correct (compute_log_post vs log_post_impl diverge for some ZI models)
      Rcpp::Named("h_ok") = (h_vs_n >= 0)
        ? ((h_vs_n < tol) ? true : (h_vs_ar >= 0 && h_vs_ar < tol))
        : NA_LOGICAL,
      Rcpp::Named("ar_ok") = (ar_vs_n >= 0) ? (ar_vs_n < tol) : NA_LOGICAL,
      Rcpp::Named("a_ok") = (a_vs_n >= 0) ? (a_vs_n < tol) : NA_LOGICAL,
      Rcpp::Named("n_params") = (int)q0.size()
    );
  }

  // Run sampler
  if (n_chains == 1) {
    ParamLayout layout = compute_param_layout(data);
    HMCResult result = run_hmc_chain(
      q0, data, layout, n_iter, n_warmup, L, 0, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian
    );

    Rcpp::List ret = Rcpp::List::create(
      Rcpp::Named("samples") = result.samples,
      Rcpp::Named("log_prob") = result.log_prob,
      Rcpp::Named("accept_prob") = result.accept_prob,
      Rcpp::Named("n_leapfrog") = result.n_leapfrog,
      Rcpp::Named("treedepth") = result.treedepth,
      Rcpp::Named("divergent") = result.divergent,
      Rcpp::Named("epsilon") = result.epsilon,
      Rcpp::Named("n_warmup") = result.n_warmup,
      Rcpp::Named("n_sample") = result.n_sample,
      Rcpp::Named("n_chains") = 1,
      Rcpp::Named("sampler") = result.sampler.empty()
        ? ((L == 0) ? std::string("NUTS") : std::string("HMC"))
        : result.sampler
    );
    if (result.n_gp_collapsed > 0) {
      ret["gp_w_star"] = result.gp_w_star;
    }
    if (result.n_icar_collapsed > 0) {
      ret["icar_phi_star"] = result.icar_phi_star;
      if (result.bym2_theta_star.nrow() > 0) {
        ret["bym2_theta_star"] = result.bym2_theta_star;
      }
    }
    return ret;
  } else {
    // Multiple chains
    std::vector<HMCResult> results = run_hmc_parallel_chains(
      q0, data, n_iter, n_warmup, L, n_chains, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian
    );

    // Combine results
    int n_sample = results[0].n_sample;
    int n_params = results[0].samples.ncol();

    Rcpp::List samples_list(n_chains);
    Rcpp::List log_prob_list(n_chains);
    Rcpp::List accept_prob_list(n_chains);
    Rcpp::List n_leapfrog_list(n_chains);
    Rcpp::List treedepth_list(n_chains);
    Rcpp::List divergent_list(n_chains);
    Rcpp::NumericVector epsilon_vec(n_chains);

    // Determine sampler name: if any chain switched, report it
    std::string sampler_name = (L == 0) ? "NUTS" : "HMC";
    for (int c = 0; c < n_chains; c++) {
      samples_list[c] = results[c].samples;
      log_prob_list[c] = results[c].log_prob;
      accept_prob_list[c] = results[c].accept_prob;
      n_leapfrog_list[c] = results[c].n_leapfrog;
      treedepth_list[c] = results[c].treedepth;
      divergent_list[c] = results[c].divergent;
      epsilon_vec[c] = results[c].epsilon;
      if (!results[c].sampler.empty()) {
        sampler_name = results[c].sampler;
      }
    }

    return Rcpp::List::create(
      Rcpp::Named("samples") = samples_list,
      Rcpp::Named("log_prob") = log_prob_list,
      Rcpp::Named("accept_prob") = accept_prob_list,
      Rcpp::Named("n_leapfrog") = n_leapfrog_list,
      Rcpp::Named("treedepth") = treedepth_list,
      Rcpp::Named("divergent") = divergent_list,
      Rcpp::Named("epsilon") = epsilon_vec,
      Rcpp::Named("n_warmup") = n_warmup,
      Rcpp::Named("n_sample") = n_sample,
      Rcpp::Named("n_chains") = n_chains,
      Rcpp::Named("sampler") = sampler_name
    );
  }
}

// [[Rcpp::export]]
int cpp_get_max_threads() {
  #ifdef _OPENMP
  return omp_get_max_threads();
  #else
  return 1;
  #endif
}

// HMC sampler for GP-based spatial models
// Parameters are bundled into lists to avoid R's .Call argument limit:
//   gp_params: GP spatial parameters
//   ms_gp_params: multiscale GP parameters
//   ms_temporal_params: multiscale temporal parameters
//   rsr_params: RSR parameters
// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit_gp(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_num_cont,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    std::string model_type_str,
    Rcpp::List gp_params,
    Rcpp::List ms_gp_params,
    Rcpp::List ms_temporal_params,
    Rcpp::List rsr_params,
    Rcpp::List temporal_params,  // Regular temporal (RW1/RW2/AR1/GP) — was missing, caused silent drop
    double sigma_beta,
    double sigma_re_scale,
    double phi_prior_shape,
    double phi_prior_rate,
    std::string zi_type_str,
    Rcpp::NumericMatrix X_zi,
    double zi_prior_sd,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    int n_threads,
    bool verbose,
    int max_treedepth = 10,
    double adapt_delta = -1.0,
    std::string metric_str = "auto",
    std::string gradient_mode_str = "auto",
    Rcpp::List tvc_params = Rcpp::List(),
    bool gradient_check_only = false
) {
  using namespace tulpa_hmc;

  // Parse metric and gradient mode from string parameters
  GradientMode grad_mode = parse_gradient_mode(gradient_mode_str);
  set_gradient_mode(grad_mode);
  MassMatrixType metric_type = parse_metric_type(metric_str);
  // Force all Rcpp parameter extractions into eagerly-copied std::vectors FIRST
  // This prevents R garbage collection from invalidating lazy Rcpp views during C++ execution
  // The original debug output workaround worked because I/O forced R to sync; this achieves
  // the same effect through explicit eager copying without visible output.

  // Extract GP parameters - convert to native C++ types immediately
  std::string gp_type_str = Rcpp::as<std::string>(gp_params["gp_type"]);

  // Force eager copy into std::vectors (not Rcpp views that could be GC'd)
  std::vector<double> coords_vec = Rcpp::as<std::vector<double>>(gp_params["coords"]);
  std::vector<int> nn_idx_vec = Rcpp::as<std::vector<int>>(gp_params["nn_idx"]);
  std::vector<double> nn_dist_vec = Rcpp::as<std::vector<double>>(gp_params["nn_dist"]);
  std::vector<int> nn_order_vec = Rcpp::as<std::vector<int>>(gp_params["nn_order"]);
  std::vector<int> nn_order_inv_vec = Rcpp::as<std::vector<int>>(gp_params["nn_order_inv"]);
  std::vector<double> nn_neighbor_dist_vec = Rcpp::as<std::vector<double>>(gp_params["nn_neighbor_dist"]);  // Phase 1.3

  int nn = Rcpp::as<int>(gp_params["nn"]);
  std::string cov_type_str = Rcpp::as<std::string>(gp_params["cov_type"]);
  double nu = Rcpp::as<double>(gp_params["nu"]);
  bool gp_shared = Rcpp::as<bool>(gp_params["shared"]);
  double gp_sigma2_prior_U = Rcpp::as<double>(gp_params["sigma2_prior_U"]);
  double gp_sigma2_prior_alpha = Rcpp::as<double>(gp_params["sigma2_prior_alpha"]);
  double gp_phi_prior_lower = Rcpp::as<double>(gp_params["phi_prior_lower"]);
  double gp_phi_prior_upper = Rcpp::as<double>(gp_params["phi_prior_upper"]);

  // GP solver configuration
  std::string gp_solver_str = Rcpp::as<std::string>(gp_params["solver"]);
  double gp_cg_tol = Rcpp::as<double>(gp_params["cg_tol"]);
  int gp_cg_maxiter = Rcpp::as<int>(gp_params["cg_maxiter"]);

  // Observation-to-location mapping (1-based from R, convert to 0-based)
  std::vector<int> gp_obs_to_loc_r = Rcpp::as<std::vector<int>>(gp_params["gp_obs_to_loc"]);
  int gp_n_unique = Rcpp::as<int>(gp_params["n_unique"]);

  // Memory barrier to ensure all extractions complete before proceeding
  std::atomic_thread_fence(std::memory_order_seq_cst);

  // Extract multiscale GP parameters - eager copy to std::vectors
  std::vector<int> nn_idx_local_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_idx_local"]);
  std::vector<double> nn_dist_local_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_dist_local"]);
  std::vector<int> nn_order_local_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_local"]);
  std::vector<int> nn_order_inv_local_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_inv_local"]);
  int nn_local = Rcpp::as<int>(ms_gp_params["nn_local"]);
  std::vector<int> nn_idx_regional_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_idx_regional"]);
  std::vector<double> nn_dist_regional_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_dist_regional"]);
  std::vector<int> nn_order_regional_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_regional"]);
  std::vector<int> nn_order_inv_regional_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_inv_regional"]);
  int nn_regional = Rcpp::as<int>(ms_gp_params["nn_regional"]);
  std::vector<double> nn_neighbor_dist_local_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_neighbor_dist_local"]);  // Phase 1.3
  std::vector<double> nn_neighbor_dist_regional_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_neighbor_dist_regional"]);  // Phase 1.3
  double range_local_lower = Rcpp::as<double>(ms_gp_params["range_local_lower"]);
  double range_local_upper = Rcpp::as<double>(ms_gp_params["range_local_upper"]);
  double range_regional_lower = Rcpp::as<double>(ms_gp_params["range_regional_lower"]);
  double range_regional_upper = Rcpp::as<double>(ms_gp_params["range_regional_upper"]);
  double ms_sigma2_local_prior_U = Rcpp::as<double>(ms_gp_params["sigma2_local_prior_U"]);
  double ms_sigma2_local_prior_alpha = Rcpp::as<double>(ms_gp_params["sigma2_local_prior_alpha"]);
  double ms_sigma2_regional_prior_U = Rcpp::as<double>(ms_gp_params["sigma2_regional_prior_U"]);
  double ms_sigma2_regional_prior_alpha = Rcpp::as<double>(ms_gp_params["sigma2_regional_prior_alpha"]);
  std::string msgp_sampler_str = Rcpp::as<std::string>(ms_gp_params["sampler"]);

  // Extract multiscale temporal parameters - eager copy
  std::string ms_temporal_type_str = Rcpp::as<std::string>(ms_temporal_params["type"]);
  std::vector<int> ms_time_index_vec = Rcpp::as<std::vector<int>>(ms_temporal_params["time_index"]);
  std::vector<int> ms_group_index_vec = Rcpp::as<std::vector<int>>(ms_temporal_params["group_index"]);
  int ms_n_times = Rcpp::as<int>(ms_temporal_params["n_times"]);
  int ms_n_groups = Rcpp::as<int>(ms_temporal_params["n_groups"]);
  std::string trend_type_str = Rcpp::as<std::string>(ms_temporal_params["trend_type"]);
  int seasonal_period = Rcpp::as<int>(ms_temporal_params["seasonal_period"]);
  std::string short_term_type_str = Rcpp::as<std::string>(ms_temporal_params["short_term_type"]);
  bool ms_temporal_shared = Rcpp::as<bool>(ms_temporal_params["shared"]);
  double ms_sigma2_trend_prior_U = Rcpp::as<double>(ms_temporal_params["sigma2_trend_prior_U"]);
  double ms_sigma2_trend_prior_alpha = Rcpp::as<double>(ms_temporal_params["sigma2_trend_prior_alpha"]);
  double ms_sigma2_seasonal_prior_U = Rcpp::as<double>(ms_temporal_params["sigma2_seasonal_prior_U"]);
  double ms_sigma2_seasonal_prior_alpha = Rcpp::as<double>(ms_temporal_params["sigma2_seasonal_prior_alpha"]);
  double ms_sigma2_short_prior_U = Rcpp::as<double>(ms_temporal_params["sigma2_short_prior_U"]);
  double ms_sigma2_short_prior_alpha = Rcpp::as<double>(ms_temporal_params["sigma2_short_prior_alpha"]);

  // Extract RSR parameters - eager copy
  bool has_rsr = Rcpp::as<bool>(rsr_params["has_rsr"]);
  std::vector<double> rsr_projection_vec = Rcpp::as<std::vector<double>>(rsr_params["projection"]);
  int rsr_n = Rcpp::as<int>(rsr_params["n"]);

  // Second memory barrier after all Rcpp extractions
  std::atomic_thread_fence(std::memory_order_seq_cst);
  using namespace tulpa_hmc;

  // Set up model data
  ModelData data;

  // Copy response data
  data.legacy.y_num = std::vector<int>(y_num.begin(), y_num.end());
  data.legacy.y_denom = std::vector<int>(y_denom.begin(), y_denom.end());
  data.legacy.y_num_cont = std::vector<double>(y_num_cont.begin(), y_num_cont.end());
  data.legacy.y_denom_cont = std::vector<double>(y_denom_cont.begin(), y_denom_cont.end());
  data.N = y_num.size();

  // Flatten design matrices
  data.legacy.p_num = X_num.ncol();
  data.legacy.p_denom = X_denom.ncol();

  data.legacy.X_num_flat.resize(data.N * data.legacy.p_num);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.legacy.p_num; j++) {
      data.legacy.X_num_flat[i * data.legacy.p_num + j] = X_num(i, j);
    }
  }

  data.legacy.X_denom_flat.resize(data.N * data.legacy.p_denom);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.legacy.p_denom; j++) {
      data.legacy.X_denom_flat[i * data.legacy.p_denom + j] = X_denom(i, j);
    }
  }

  // Random effects (single-term legacy path)
  data.re_group = std::vector<int>(re_group.begin(), re_group.end());
  data.n_re_groups = n_re_groups;

  // Initialize multi-term RE fields to indicate single-term mode
  data.n_re_terms = 0;  // 0 means use legacy single-term path
  data.total_re_groups = n_re_groups;
  data.has_re_slopes = false;  // GP interface doesn't support random slopes
  data.has_re_correlated_slopes = false;

  // Model type
  if (model_type_str == "binomial") {
    data.legacy.model_type = ModelType::BINOMIAL;
  } else if (model_type_str == "negbin_negbin") {
    data.legacy.model_type = ModelType::NEGBIN_NEGBIN;
  } else if (model_type_str == "poisson_gamma") {
    data.legacy.model_type = ModelType::POISSON_GAMMA;
  } else if (model_type_str == "negbin_gamma") {
    data.legacy.model_type = ModelType::NEGBIN_GAMMA;
  } else if (model_type_str == "gamma_gamma") {
    data.legacy.model_type = ModelType::GAMMA_GAMMA;
  } else if (model_type_str == "lognormal") {
    data.legacy.model_type = ModelType::LOGNORMAL;
  } else if (model_type_str == "beta_binomial") {
    data.legacy.model_type = ModelType::BETA_BINOMIAL;
  } else {
    data.legacy.model_type = ModelType::POISSON_GAMMA;  // fallback
  }

  // Covariance type
  tulpa_gp::CovType cov_type;
  if (cov_type_str == "exponential") {
    cov_type = tulpa_gp::CovType::EXPONENTIAL;
  } else if (cov_type_str == "matern") {
    cov_type = tulpa_gp::CovType::MATERN;
  } else if (cov_type_str == "gaussian") {
    cov_type = tulpa_gp::CovType::GAUSSIAN;
  } else {
    cov_type = tulpa_gp::CovType::SPHERICAL;
  }

  // GP spatial structure
  if (gp_type_str == "gp") {
    data.spatial_type = SpatialType::GP;
    data.has_gp = true;
    data.has_multiscale_gp = false;
    data.has_hsgp = false;

    data.gp_data.n_obs = gp_n_unique;  // Unique locations, not total observations
    data.gp_data.nn = nn;
    data.gp_data.coords = coords_vec;  // Already std::vector from eager copy
    data.gp_data.nn_idx = nn_idx_vec;
    data.gp_data.nn_dist = nn_dist_vec;
    data.gp_data.nn_neighbor_dist = nn_neighbor_dist_vec;  // Phase 1.3: cached pairwise distances
    // Convert obs_to_loc from R's 1-based to C++'s 0-based indexing
    data.gp_data.obs_to_loc.resize(gp_obs_to_loc_r.size());
    for (size_t i = 0; i < gp_obs_to_loc_r.size(); i++) {
      data.gp_data.obs_to_loc[i] = gp_obs_to_loc_r[i] - 1;
    }
    // Convert from R's 1-based to C++'s 0-based indexing
    data.gp_data.nn_order.resize(nn_order_vec.size());
    for (size_t i = 0; i < nn_order_vec.size(); i++) {
      data.gp_data.nn_order[i] = nn_order_vec[i] - 1;
    }
    data.gp_data.nn_order_inv.resize(nn_order_inv_vec.size());
    for (size_t i = 0; i < nn_order_inv_vec.size(); i++) {
      data.gp_data.nn_order_inv[i] = nn_order_inv_vec[i] - 1;
    }
    data.gp_data.cov_type = cov_type;
    data.gp_data.nu = nu;
    data.gp_data.shared = gp_shared;

    // Set solver configuration
    data.gp_data.solver_config.solver = tulpa_gp::parse_gp_solver(gp_solver_str);
    data.gp_data.solver_config.cg_tol = gp_cg_tol;
    data.gp_data.solver_config.cg_maxiter = gp_cg_maxiter;
    data.gp_data.solver_config.n_obs = gp_n_unique;  // Unique locations

    data.gp_sigma2_prior_U = gp_sigma2_prior_U;
    data.gp_sigma2_prior_alpha = gp_sigma2_prior_alpha;
    data.gp_phi_prior_lower = gp_phi_prior_lower;
    data.gp_phi_prior_upper = gp_phi_prior_upper;

    // GP parameterization: centered (default), noncentered, or collapsed
    if (gp_params.containsElementNamed("parameterization")) {
        std::string gp_param_str = Rcpp::as<std::string>(gp_params["parameterization"]);
        if (gp_param_str == "collapsed") {
            data.gp_parameterization = 0;  // Not relevant for collapsed
            data.gp_collapsed = true;
        } else {
            data.gp_parameterization = (gp_param_str == "centered") ? 0 : 1;
            data.gp_collapsed = false;
        }
    } else {
        data.gp_parameterization = 0;  // Default: centered
        data.gp_collapsed = false;
    }

  } else if (gp_type_str == "multiscale_gp") {
    data.spatial_type = SpatialType::MULTISCALE_GP;
    data.has_gp = false;
    data.has_multiscale_gp = true;
    data.has_hsgp = false;

    // Check if using HSGP approximation
    std::string msgp_approx = "nngp";
    if (gp_params.containsElementNamed("msgp_approx")) {
      msgp_approx = Rcpp::as<std::string>(gp_params["msgp_approx"]);
    }
    data.msgp_is_hsgp = (msgp_approx == "hsgp");

    if (data.msgp_is_hsgp) {
      // HSGP-MSGP: set up shared basis functions, no NNGP neighbor computation
      int hsgp_m = Rcpp::as<int>(gp_params["hsgp_m"]);
      double hsgp_c = Rcpp::as<double>(gp_params["hsgp_c"]);
      tulpa_hsgp::setup_hsgp_2d(coords_vec, data.N, hsgp_m, hsgp_c,
                                  gp_shared, data.msgp_hsgp_data);
      data.multiscale_gp_data.shared = gp_shared;
      // Set n_obs for consistency (used by param layout check)
      data.multiscale_gp_data.n_obs = data.N;

      // Lengthscale prior means from range bounds (geometric mean on log scale)
      data.ms_log_ls_local_mean = 0.5 * (std::log(range_local_lower) + std::log(range_local_upper));
      data.ms_log_ls_local_sd = 0.5;
      data.ms_log_ls_regional_mean = 0.5 * (std::log(range_regional_lower) + std::log(range_regional_upper));
      data.ms_log_ls_regional_sd = 0.5;

      if (verbose) {
        Rcpp::Rcout << "  HSGP-MSGP: m=" << hsgp_m << ", c=" << hsgp_c
                    << ", m_total=" << data.msgp_hsgp_data.m_total
                    << " (local+regional: " << 2 * data.msgp_hsgp_data.m_total << " basis coefficients)\n";
        Rcpp::Rcout << "  Lengthscale priors: local LogN(" << data.ms_log_ls_local_mean
                    << ", " << data.ms_log_ls_local_sd << "), regional LogN("
                    << data.ms_log_ls_regional_mean << ", " << data.ms_log_ls_regional_sd << ")\n";
      }
    } else {
      // NNGP-MSGP: standard neighbor-based computation
      data.multiscale_gp_data.n_obs = gp_n_unique;  // Unique locations, not total observations
      data.multiscale_gp_data.coords = coords_vec;  // Already std::vector from eager copy
      // Convert obs_to_loc from R's 1-based to C++'s 0-based indexing
      data.multiscale_gp_data.obs_to_loc.resize(gp_obs_to_loc_r.size());
      for (size_t i = 0; i < gp_obs_to_loc_r.size(); i++) {
        data.multiscale_gp_data.obs_to_loc[i] = gp_obs_to_loc_r[i] - 1;
      }

      // Local scale - use pre-copied std::vectors
      data.multiscale_gp_data.nn_local = nn_local;
      data.multiscale_gp_data.nn_idx_local = nn_idx_local_vec;
      data.multiscale_gp_data.nn_dist_local = nn_dist_local_vec;
      // Convert from R's 1-based to C++'s 0-based indexing
      data.multiscale_gp_data.nn_order_local.resize(nn_order_local_vec.size());
      for (size_t i = 0; i < nn_order_local_vec.size(); i++) {
        data.multiscale_gp_data.nn_order_local[i] = nn_order_local_vec[i] - 1;
      }
      data.multiscale_gp_data.nn_order_inv_local.resize(nn_order_inv_local_vec.size());
      for (size_t i = 0; i < nn_order_inv_local_vec.size(); i++) {
        data.multiscale_gp_data.nn_order_inv_local[i] = nn_order_inv_local_vec[i] - 1;
      }
      data.multiscale_gp_data.nn_neighbor_dist_local = nn_neighbor_dist_local_vec;  // Phase 1.3

      // Regional scale - use pre-copied std::vectors
      data.multiscale_gp_data.nn_regional = nn_regional;
      data.multiscale_gp_data.nn_idx_regional = nn_idx_regional_vec;
      data.multiscale_gp_data.nn_dist_regional = nn_dist_regional_vec;
      // Convert from R's 1-based to C++'s 0-based indexing
      data.multiscale_gp_data.nn_order_regional.resize(nn_order_regional_vec.size());
      for (size_t i = 0; i < nn_order_regional_vec.size(); i++) {
        data.multiscale_gp_data.nn_order_regional[i] = nn_order_regional_vec[i] - 1;
      }
      data.multiscale_gp_data.nn_order_inv_regional.resize(nn_order_inv_regional_vec.size());
      for (size_t i = 0; i < nn_order_inv_regional_vec.size(); i++) {
        data.multiscale_gp_data.nn_order_inv_regional[i] = nn_order_inv_regional_vec[i] - 1;
      }
      data.multiscale_gp_data.nn_neighbor_dist_regional = nn_neighbor_dist_regional_vec;  // Phase 1.3

      // Range constraints
      data.multiscale_gp_data.range_local_lower = range_local_lower;
      data.multiscale_gp_data.range_local_upper = range_local_upper;
      data.multiscale_gp_data.range_regional_lower = range_regional_lower;
      data.multiscale_gp_data.range_regional_upper = range_regional_upper;

      data.multiscale_gp_data.cov_type = cov_type;
      data.multiscale_gp_data.nu = nu;
      data.multiscale_gp_data.sampler = tulpa_gp::parse_msgp_sampler(msgp_sampler_str);
    }

    data.multiscale_gp_data.shared = gp_shared;
    data.ms_sigma2_local_prior_U = ms_sigma2_local_prior_U;
    data.ms_sigma2_local_prior_alpha = ms_sigma2_local_prior_alpha;
    data.ms_sigma2_regional_prior_U = ms_sigma2_regional_prior_U;
    data.ms_sigma2_regional_prior_alpha = ms_sigma2_regional_prior_alpha;

  } else if (gp_type_str == "hsgp") {
    data.spatial_type = SpatialType::HSGP;
    data.has_gp = false;
    data.has_multiscale_gp = false;
    data.has_hsgp = true;

    // HSGP parameters from gp_params
    int hsgp_m = Rcpp::as<int>(gp_params["hsgp_m"]);
    double hsgp_c = Rcpp::as<double>(gp_params["hsgp_c"]);
    bool hsgp_shared = gp_shared;

    // Setup HSGP data structure with precomputed basis functions
    tulpa_hsgp::setup_hsgp_2d(coords_vec, data.N, hsgp_m, hsgp_c,
                                hsgp_shared, data.hsgp_data);

    data.hsgp_m_per_dim = hsgp_m;
    data.hsgp_boundary_factor = hsgp_c;

  } else {
    data.spatial_type = SpatialType::NONE;
    data.has_gp = false;
    data.has_multiscale_gp = false;
    data.has_hsgp = false;
  }

  // Initialize adjacency for ICAR/BYM2 (not used with GP)
  data.n_spatial_units = 0;
  data.bym2_scale_factor = 1.0;

  // Multi-scale temporal structure
  if (ms_temporal_type_str == "multiscale") {
    data.has_multiscale_temporal = true;

    data.multiscale_temporal_data.n_times = ms_n_times;
    data.multiscale_temporal_data.n_groups = ms_n_groups;
    data.multiscale_temporal_data.n_obs = data.N;
    data.multiscale_temporal_data.time_index = ms_time_index_vec;  // Already std::vector from eager copy
    data.multiscale_temporal_data.group_index = ms_group_index_vec;
    data.multiscale_temporal_data.shared = ms_temporal_shared;
    data.multiscale_temporal_data.seasonal_period = seasonal_period;

    // Parse temporal component types
    data.multiscale_temporal_data.trend_type = tulpa_temporal::parse_temporal_type(trend_type_str);
    data.multiscale_temporal_data.short_term_type = tulpa_temporal::parse_temporal_type(short_term_type_str);

    data.ms_sigma2_trend_prior_U = ms_sigma2_trend_prior_U;
    data.ms_sigma2_trend_prior_alpha = ms_sigma2_trend_prior_alpha;
    data.ms_sigma2_seasonal_prior_U = ms_sigma2_seasonal_prior_U;
    data.ms_sigma2_seasonal_prior_alpha = ms_sigma2_seasonal_prior_alpha;
    data.ms_sigma2_short_prior_U = ms_sigma2_short_prior_U;
    data.ms_sigma2_short_prior_alpha = ms_sigma2_short_prior_alpha;

  } else {
    data.has_multiscale_temporal = false;
    data.multiscale_temporal_data.trend_type = tulpa_temporal::TemporalType::NONE;
    data.multiscale_temporal_data.short_term_type = tulpa_temporal::TemporalType::NONE;
    data.multiscale_temporal_data.seasonal_period = 0;
  }

  // Regular temporal (RW1/RW2/AR1/GP) — now supported in GP interface
  {
    std::string temporal_type_str = Rcpp::as<std::string>(temporal_params["type"]);
    int n_temporal_groups = Rcpp::as<int>(temporal_params["n_groups"]);
    bool temporal_shared = Rcpp::as<bool>(temporal_params["shared"]);

    if (temporal_type_str == "rw1") {
      data.temporal_type = TemporalType::RW1;
    } else if (temporal_type_str == "rw2") {
      data.temporal_type = TemporalType::RW2;
    } else if (temporal_type_str == "ar1") {
      data.temporal_type = TemporalType::AR1;
    } else if (temporal_type_str == "gp") {
      data.temporal_type = TemporalType::GP;

      // GP-specific parameters (same parsing as cpp_hmc_fit)
      data.has_temporal_gp = true;
      data.temporal_gp_data.n_obs = Rcpp::as<int>(temporal_params["n_times"]);
      data.temporal_gp_data.n_groups = n_temporal_groups;
      data.temporal_gp_data.time_values = Rcpp::as<std::vector<double>>(temporal_params["time_values"]);
      data.temporal_gp_data.group_index = Rcpp::as<std::vector<int>>(temporal_params["group_idx"]);

      std::string cov_type_str = Rcpp::as<std::string>(temporal_params["cov_type"]);
      data.temporal_gp_data.cov_type = tulpa_temporal_gp::parse_temporal_cov_type(cov_type_str);
      data.temporal_gp_data.nu = Rcpp::as<double>(temporal_params["nu"]);
      data.temporal_gp_data.period = Rcpp::as<double>(temporal_params["period"]);
      data.temporal_gp_data.shared = temporal_shared;

      data.temporal_gp_sigma2_prior_U = Rcpp::as<double>(temporal_params["gp_sigma2_prior_U"]);
      data.temporal_gp_sigma2_prior_alpha = Rcpp::as<double>(temporal_params["gp_sigma2_prior_alpha"]);
      data.temporal_gp_phi_prior_lower = Rcpp::as<double>(temporal_params["gp_phi_prior_lower"]);
      data.temporal_gp_phi_prior_upper = Rcpp::as<double>(temporal_params["gp_phi_prior_upper"]);

      std::string gp_param_str = "noncentered";
      if (temporal_params.containsElementNamed("gp_parameterization")) {
        gp_param_str = Rcpp::as<std::string>(temporal_params["gp_parameterization"]);
      }
      data.temporal_gp_parameterization = (gp_param_str == "centered") ? 0 : 1;
    } else {
      data.temporal_type = TemporalType::NONE;
    }
    data.temporal_time_idx = Rcpp::as<std::vector<int>>(temporal_params["time_idx"]);
    data.temporal_group_idx = Rcpp::as<std::vector<int>>(temporal_params["group_idx"]);
    data.n_times = Rcpp::as<int>(temporal_params["n_times"]);
    data.n_temporal_groups = n_temporal_groups;
    data.n_temporal_params = Rcpp::as<int>(temporal_params["n_params"]);
    data.temporal_cyclic = Rcpp::as<bool>(temporal_params["cyclic"]);
    data.temporal_shared = temporal_shared;
    data.tau_temporal_shape = Rcpp::as<double>(temporal_params["tau_shape"]);
    data.tau_temporal_rate = Rcpp::as<double>(temporal_params["tau_rate"]);
  }

  // Multi-term RE structure (not used in GP interface - single term only)
  data.total_re_params = 0;
  data.total_sigma_params = 0;
  data.total_chol_params = 0;

  // RSR structure - use pre-copied std::vector
  data.has_rsr = has_rsr;
  if (has_rsr && !rsr_projection_vec.empty()) {
    data.rsr_projection = rsr_projection_vec;
    data.rsr_n = rsr_n;
  } else {
    data.rsr_n = 0;
  }

  // Zero-inflation structure (GP interface: no OI support, use matrix directly)
  data.zi_type = tulpa_zi::parse_zi_type(zi_type_str);
  data.p_zi = X_zi.ncol();
  data.zi_prior_sd = zi_prior_sd;
  data.X_zi_flat.resize(data.N * data.p_zi);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_zi; j++) {
      data.X_zi_flat[i * data.p_zi + j] = X_zi(i, j);
    }
  }

  // Standard priors
  data.sigma_beta = sigma_beta;
  data.sigma_re_scale = sigma_re_scale;
  data.phi_prior_shape = phi_prior_shape;
  data.phi_prior_rate = phi_prior_rate;
  data.tau_spatial_shape = 1.0;
  data.tau_spatial_rate = 0.01;

  // SVC not used in GP interface
  data.has_svc = false;

  // Latent factors not used in GP interface
  data.has_latent = false;
  data.latent_n_factors = 0;
  data.latent_shared = false;
  data.latent_scale = false;
  data.latent_constraint = 0;
  data.latent_sigma_prior_rate = 1.0;

  // Spatiotemporal not used in GP interface
  data.has_spatiotemporal = false;
  data.spatiotemporal_data.type = STType::NONE;

  // TVC (Temporally-Varying Coefficients) — now supported in GP interface
  {
    bool has_tvc = tvc_params.size() > 0 && tvc_params.containsElementNamed("has_tvc") &&
                   Rcpp::as<bool>(tvc_params["has_tvc"]);
    data.has_tvc = has_tvc;
    if (has_tvc) {
      int tvc_n_tvc = Rcpp::as<int>(tvc_params["n_tvc"]);
      int tvc_n_times = Rcpp::as<int>(tvc_params["n_times"]);
      int tvc_n_groups = Rcpp::as<int>(tvc_params["n_groups"]);
      std::string tvc_structure_str = Rcpp::as<std::string>(tvc_params["structure"]);
      bool tvc_shared = Rcpp::as<bool>(tvc_params["shared"]);
      bool tvc_cyclic = Rcpp::as<bool>(tvc_params["cyclic"]);
      std::vector<int> tvc_indices = Rcpp::as<std::vector<int>>(tvc_params["tvc_indices"]);
      std::vector<int> tvc_time_index = Rcpp::as<std::vector<int>>(tvc_params["time_index"]);
      std::vector<int> tvc_group_index = Rcpp::as<std::vector<int>>(tvc_params["group_index"]);
      std::vector<double> tvc_X_tvc = Rcpp::as<std::vector<double>>(tvc_params["X_tvc"]);
      double tvc_tau_shape = Rcpp::as<double>(tvc_params["tau_shape"]);
      double tvc_tau_rate = Rcpp::as<double>(tvc_params["tau_rate"]);

      data.tvc_data.n_obs = data.N;
      data.tvc_data.n_tvc = tvc_n_tvc;
      data.tvc_data.n_times = tvc_n_times;
      data.tvc_data.n_groups = tvc_n_groups;
      data.tvc_data.shared = tvc_shared;
      data.tvc_data.cyclic = tvc_cyclic;
      data.tvc_data.tvc_indices = tvc_indices;
      data.tvc_data.time_index = tvc_time_index;
      data.tvc_data.group_index = tvc_group_index;
      data.tvc_data.X_tvc = tvc_X_tvc;

      if (tvc_structure_str == "rw1") {
        data.tvc_data.structure = tulpa_temporal::TemporalType::RW1;
      } else if (tvc_structure_str == "rw2") {
        data.tvc_data.structure = tulpa_temporal::TemporalType::RW2;
      } else if (tvc_structure_str == "ar1") {
        data.tvc_data.structure = tulpa_temporal::TemporalType::AR1;
      } else if (tvc_structure_str == "iid") {
        data.tvc_data.structure = tulpa_temporal::TemporalType::IID;
      } else {
        data.tvc_data.structure = tulpa_temporal::TemporalType::RW1;
      }

      data.tvc_tau_shape = tvc_tau_shape;
      data.tvc_tau_rate = tvc_tau_rate;
      data.tvc_data.init_workspace();
    } else {
      data.tvc_data.n_tvc = 0;
      data.tvc_data.n_times = 0;
      data.tvc_data.n_groups = 1;
    }
  }

  // Parallelization
  data.n_threads = n_threads;

  // Final memory barrier before HMC execution
  std::atomic_thread_fence(std::memory_order_seq_cst);

  // Initialize parameters - use explicit std::vector copy from Rcpp
  std::vector<double> q0(q_init.begin(), q_init.end());

  // =========================================================================
  // Gradient check only mode: compare N, A, A_r, H without sampling
  // =========================================================================
  if (gradient_check_only) {
    ParamLayout layout = compute_param_layout(data);

    std::vector<double> grad_N;
    compute_gradient_numerical(q0, data, layout, grad_N);

    std::vector<double> grad_N_impl;
    compute_gradient_numerical_impl(q0, data, layout, grad_N_impl);

    auto max_rel_diff = [](const std::vector<double>& a, const std::vector<double>& b) -> double {
      double mx = 0.0;
      for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        double diff = std::abs(a[i] - b[i]);
        double scale = std::max(1.0, std::max(std::abs(a[i]), std::abs(b[i])));
        mx = std::max(mx, diff / scale);
      }
      return mx;
    };

    auto is_autodiff_fn = [](GradientFn fn) -> bool {
      return fn == &compute_gradient_arena ||
             fn == &compute_gradient_forward ||
             fn == &compute_gradient_autodiff;
    };

    double h_vs_n = -1.0;
    GradientFn h_fn = resolve_gradient_fn(GradientMode::HANDCODED, data, layout);
    if (h_fn != &compute_gradient_numerical && h_fn != &compute_gradient_numerical_impl) {
      const auto& ref = is_autodiff_fn(h_fn) ? grad_N_impl : grad_N;
      std::vector<double> grad_H;
      h_fn(q0, data, layout, grad_H, nullptr);
      h_vs_n = max_rel_diff(grad_H, ref);
    }

    double ar_vs_n = -1.0;
    GradientFn ar_fn = resolve_gradient_fn(GradientMode::AUTODIFF_ARENA, data, layout);
    if (ar_fn != &compute_gradient_numerical && ar_fn != &compute_gradient_numerical_impl) {
      std::vector<double> grad_Ar;
      ar_fn(q0, data, layout, grad_Ar, nullptr);
      ar_vs_n = max_rel_diff(grad_Ar, grad_N_impl);
    }

    double a_vs_n = -1.0;
    GradientFn a_fn = resolve_gradient_fn(GradientMode::AUTODIFF_FWD, data, layout);
    if (a_fn != &compute_gradient_numerical && a_fn != &compute_gradient_numerical_impl) {
      std::vector<double> grad_A;
      a_fn(q0, data, layout, grad_A, nullptr);
      a_vs_n = max_rel_diff(grad_A, grad_N_impl);
    }

    double h_vs_ar = -1.0;
    if (h_vs_n >= 0 && ar_vs_n >= 0) {
      std::vector<double> grad_H2, grad_Ar2;
      h_fn(q0, data, layout, grad_H2, nullptr);
      ar_fn(q0, data, layout, grad_Ar2, nullptr);
      h_vs_ar = max_rel_diff(grad_H2, grad_Ar2);

      if (h_vs_ar > 1e-4) {
        std::vector<std::pair<double,int>> diffs;
        for (size_t i = 0; i < grad_H2.size() && i < grad_Ar2.size(); i++) {
          double diff = std::abs(grad_H2[i] - grad_Ar2[i]);
          double scale = std::max(1.0, std::max(std::abs(grad_H2[i]), std::abs(grad_Ar2[i])));
          diffs.push_back({diff/scale, (int)i});
        }
        std::sort(diffs.begin(), diffs.end(), [](auto&a,auto&b){return a.first>b.first;});
        Rprintf("  H vs A_r cross-check DIVERGE (top 3 of %d params):\n", (int)grad_H2.size());
        for (int k = 0; k < std::min(3,(int)diffs.size()); k++) {
          int idx = diffs[k].second;
          Rprintf("    param[%d]: H=%.8e A_r=%.8e  rel_diff=%.2e\n",
                  idx, grad_H2[idx], grad_Ar2[idx], diffs[k].first);
        }
        // Print MSGP layout info
        if (layout.is_multiscale_gp) {
          Rprintf("  MSGP: sigma2_local[%d] phi_local[%d] sigma2_reg[%d] phi_reg[%d]\n",
                  layout.log_sigma2_gp_local_idx, layout.log_phi_gp_local_idx,
                  layout.log_sigma2_gp_regional_idx, layout.log_phi_gp_regional_idx);
          Rprintf("  MSGP: beta_local[%d-%d] beta_reg[%d-%d]\n",
                  layout.gp_local_start, layout.gp_local_end-1,
                  layout.gp_regional_start, layout.gp_regional_end-1);
        }
        if (layout.has_temporal)
          Rprintf("  temporal[%d-%d] log_tau[%d]\n",
                  layout.temporal_start, layout.temporal_end-1, layout.log_tau_temporal_idx);
      }
    }

    double tol = 1e-4;
    return Rcpp::List::create(
      Rcpp::Named("h_vs_n") = h_vs_n,
      Rcpp::Named("ar_vs_n") = ar_vs_n,
      Rcpp::Named("a_vs_n") = a_vs_n,
      Rcpp::Named("h_vs_ar") = h_vs_ar,
      Rcpp::Named("tol") = tol,
      // h_ok: check h_vs_n first; if it fails but h_vs_ar passes, H is still
      // correct (compute_log_post vs log_post_impl diverge for some ZI models)
      Rcpp::Named("h_ok") = (h_vs_n >= 0)
        ? ((h_vs_n < tol) ? true : (h_vs_ar >= 0 && h_vs_ar < tol))
        : NA_LOGICAL,
      Rcpp::Named("ar_ok") = (ar_vs_n >= 0) ? (ar_vs_n < tol) : NA_LOGICAL,
      Rcpp::Named("a_ok") = (a_vs_n >= 0) ? (a_vs_n < tol) : NA_LOGICAL,
      Rcpp::Named("n_params") = (int)q0.size()
    );
  }

  // Run sampler
  if (n_chains == 1) {
    ParamLayout layout = compute_param_layout(data);
    HMCResult result = run_hmc_chain(
      q0, data, layout, n_iter, n_warmup, L, 0, seed, verbose, max_treedepth,
      metric_type, adapt_delta, -1
    );

    return Rcpp::List::create(
      Rcpp::Named("samples") = result.samples,
      Rcpp::Named("log_prob") = result.log_prob,
      Rcpp::Named("accept_prob") = result.accept_prob,
      Rcpp::Named("n_leapfrog") = result.n_leapfrog,
      Rcpp::Named("treedepth") = result.treedepth,
      Rcpp::Named("divergent") = result.divergent,
      Rcpp::Named("epsilon") = result.epsilon,
      Rcpp::Named("n_warmup") = result.n_warmup,
      Rcpp::Named("n_sample") = result.n_sample,
      Rcpp::Named("n_chains") = 1,
      Rcpp::Named("sampler") = result.sampler.empty()
        ? ((L == 0) ? std::string("NUTS") : std::string("HMC"))
        : result.sampler
    );
  } else {
    // Multiple chains
    std::vector<HMCResult> results = run_hmc_parallel_chains(
      q0, data, n_iter, n_warmup, L, n_chains, seed, verbose, max_treedepth,
      metric_type, adapt_delta, -1
    );

    // Combine results
    int n_sample = results[0].n_sample;
    int n_params = results[0].samples.ncol();

    Rcpp::List samples_list(n_chains);
    Rcpp::List log_prob_list(n_chains);
    Rcpp::List accept_prob_list(n_chains);
    Rcpp::List n_leapfrog_list(n_chains);
    Rcpp::List treedepth_list(n_chains);
    Rcpp::List divergent_list(n_chains);
    Rcpp::NumericVector epsilon_vec(n_chains);

    std::string sampler_name = (L == 0) ? "NUTS" : "HMC";
    for (int c = 0; c < n_chains; c++) {
      samples_list[c] = results[c].samples;
      log_prob_list[c] = results[c].log_prob;
      accept_prob_list[c] = results[c].accept_prob;
      n_leapfrog_list[c] = results[c].n_leapfrog;
      treedepth_list[c] = results[c].treedepth;
      divergent_list[c] = results[c].divergent;
      epsilon_vec[c] = results[c].epsilon;
      if (!results[c].sampler.empty()) {
        sampler_name = results[c].sampler;
      }
    }

    return Rcpp::List::create(
      Rcpp::Named("samples") = samples_list,
      Rcpp::Named("log_prob") = log_prob_list,
      Rcpp::Named("accept_prob") = accept_prob_list,
      Rcpp::Named("n_leapfrog") = n_leapfrog_list,
      Rcpp::Named("treedepth") = treedepth_list,
      Rcpp::Named("divergent") = divergent_list,
      Rcpp::Named("epsilon") = epsilon_vec,
      Rcpp::Named("n_warmup") = n_warmup,
      Rcpp::Named("n_sample") = n_sample,
      Rcpp::Named("n_chains") = n_chains,
      Rcpp::Named("sampler") = sampler_name
    );
  }
}

// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit_gp_v2(Rcpp::List args) {
  // O2-safe interface: single List parameter to minimize Rcpp template instantiation
  // at ABI boundary. All parameter extraction happens inside function body where
  // compiler has full visibility.

  // Extract all parameters from the list - matching cpp_hmc_fit_gp signature
  Rcpp::NumericVector q_init = Rcpp::as<Rcpp::NumericVector>(args["q_init"]);
  Rcpp::IntegerVector y_num = Rcpp::as<Rcpp::IntegerVector>(args["y_num"]);
  Rcpp::IntegerVector y_denom = Rcpp::as<Rcpp::IntegerVector>(args["y_denom"]);
  Rcpp::NumericVector y_num_cont = args.containsElementNamed("y_num_cont")
    ? Rcpp::as<Rcpp::NumericVector>(args["y_num_cont"])
    : Rcpp::NumericVector(y_num.size(), 0.0);
  Rcpp::NumericVector y_denom_cont = Rcpp::as<Rcpp::NumericVector>(args["y_denom_cont"]);
  Rcpp::NumericMatrix X_num = Rcpp::as<Rcpp::NumericMatrix>(args["X_num"]);
  Rcpp::NumericMatrix X_denom = Rcpp::as<Rcpp::NumericMatrix>(args["X_denom"]);
  Rcpp::IntegerVector re_group = Rcpp::as<Rcpp::IntegerVector>(args["re_group"]);
  int n_re_groups = Rcpp::as<int>(args["n_re_groups"]);
  std::string model_type_str = Rcpp::as<std::string>(args["model_type_str"]);
  Rcpp::List gp_params = Rcpp::as<Rcpp::List>(args["gp_params"]);
  Rcpp::List ms_gp_params = Rcpp::as<Rcpp::List>(args["ms_gp_params"]);
  Rcpp::List ms_temporal_params = Rcpp::as<Rcpp::List>(args["ms_temporal_params"]);
  Rcpp::List rsr_params = Rcpp::as<Rcpp::List>(args["rsr_params"]);
  Rcpp::List temporal_params = Rcpp::as<Rcpp::List>(args["temporal_params"]);
  double sigma_beta = Rcpp::as<double>(args["sigma_beta"]);
  double sigma_re_scale = Rcpp::as<double>(args["sigma_re_scale"]);
  double phi_prior_shape = Rcpp::as<double>(args["phi_prior_shape"]);
  double phi_prior_rate = Rcpp::as<double>(args["phi_prior_rate"]);
  std::string zi_type_str = Rcpp::as<std::string>(args["zi_type_str"]);
  Rcpp::NumericMatrix X_zi = Rcpp::as<Rcpp::NumericMatrix>(args["X_zi"]);
  double zi_prior_sd = Rcpp::as<double>(args["zi_prior_sd"]);
  int n_iter = Rcpp::as<int>(args["n_iter"]);
  int n_warmup = Rcpp::as<int>(args["n_warmup"]);
  int L = Rcpp::as<int>(args["L"]);
  int n_chains = Rcpp::as<int>(args["n_chains"]);
  unsigned int seed = Rcpp::as<unsigned int>(args["seed"]);
  int n_threads = Rcpp::as<int>(args["n_threads"]);
  bool verbose = Rcpp::as<bool>(args["verbose"]);
  int max_treedepth = 10;
  if (args.containsElementNamed("max_treedepth")) {
    max_treedepth = Rcpp::as<int>(args["max_treedepth"]);
  }
  double adapt_delta = -1.0;
  if (args.containsElementNamed("adapt_delta")) {
    adapt_delta = Rcpp::as<double>(args["adapt_delta"]);
  }
  std::string metric_str = "auto";
  if (args.containsElementNamed("metric_str")) {
    metric_str = Rcpp::as<std::string>(args["metric_str"]);
  }
  std::string gradient_mode_str = "auto";
  if (args.containsElementNamed("gradient_mode_str")) {
    gradient_mode_str = Rcpp::as<std::string>(args["gradient_mode_str"]);
  }

  // Extract TVC params if present
  Rcpp::List tvc_params_extracted;
  if (args.containsElementNamed("tvc_params")) {
    tvc_params_extracted = Rcpp::as<Rcpp::List>(args["tvc_params"]);
  }

  bool gradient_check_only = false;
  if (args.containsElementNamed("gradient_check_only")) {
    gradient_check_only = Rcpp::as<bool>(args["gradient_check_only"]);
  }

  // Delegate to the original implementation (metric/gradient parsed inside)
  return cpp_hmc_fit_gp(
    q_init, y_num, y_denom, y_num_cont, y_denom_cont,
    X_num, X_denom, re_group, n_re_groups,
    model_type_str, gp_params, ms_gp_params, ms_temporal_params, rsr_params,
    temporal_params,
    sigma_beta, sigma_re_scale, phi_prior_shape, phi_prior_rate,
    zi_type_str, X_zi, zi_prior_sd,
    n_iter, n_warmup, L, n_chains, seed, n_threads, verbose, max_treedepth, adapt_delta,
    metric_str, gradient_mode_str,
    tvc_params_extracted,
    gradient_check_only
  );
}
