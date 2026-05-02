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
#include "hmc_car_proper.h"
#include "hmc_temporal_autodiff.h"
#include "hmc_tvc_grad.h"
#include "hmc_multiscale_temporal_grad.h"
#include "lkj_chol_helpers.h"
#include "hmc_observation_likelihood.h"
#include "hmc_log_posterior_split.h"
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
thread_local CollapsedGPWorkspace collapsed_gp_ws;

// Shared collapsed ICAR/BYM2 workspace
thread_local CollapsedICARWorkspace collapsed_icar_ws;

// =====================================================================
// Log-posterior computation with OpenMP parallelization
// =====================================================================

double accumulate_log_prior_and_state(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    LogPosteriorState& state,
    ObservationLikelihoodContext& ctx,
    bool populate_obs_state,
    const double* precomputed_st_log_prior,
    const double* precomputed_tgp_log_prior
) {
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
  // this heap allocation on every gradient call ? saves ~100ns per call).
  // re_nc_flat lives in state (aliased above).
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
      // Tanh-parameterized L plus LKJ(eta=2) prior ? see lkj_chol_helpers.h.
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
    } else if (layout.is_car_proper) {
      // Proper CAR: Q(rho) = D - rho*W is full-rank for rho ? (rho_lower, rho_upper).
      // Prior: phi | tau, rho ~ N(0, (tau*Q)^{-1})
      //   log p = 0.5*log|tau*Q| - 0.5*tau*phi'Q*phi
      //         = 0.5*J*log(tau) + 0.5*log|Q(rho)| - 0.5*tau*phi'Q(rho)*phi
      // tau ~ Gamma(shape, rate) (with log-Jacobian).
      // rho via logit transform with uniform prior on (rho_lower, rho_upper):
      //   p(logit_rho) ? u*(1-u) where u = (rho-lower)/(upper-lower).

      // tau prior (Gamma + log-Jacobian)
      log_post += (data.tau_spatial_shape - 1.0) * log_tau_spatial
                - data.tau_spatial_rate * tau_spatial + log_tau_spatial;

      // Logit-rho Jacobian: log(u) + log(1-u), where u ? (0,1)
      double u_logit = (rho_car - data.car_rho_lower) /
                       (data.car_rho_upper - data.car_rho_lower);
      // Guard against numerical edges
      double u_clip = std::min(std::max(u_logit, 1e-12), 1.0 - 1e-12);
      log_post += std::log(u_clip) + std::log(1.0 - u_clip);

      // phi quadratic form: phi' Q(rho) phi
      double quad = 0.0;
      for (int i = 0; i < J; i++) {
        quad += data.n_neighbors[i] * phi_spatial[i] * phi_spatial[i];
        int row_start = data.adj_row_ptr[i];
        int row_end = data.adj_row_ptr[i + 1];
        for (int k = row_start; k < row_end; k++) {
          int j = data.adj_col_idx[k];
          if (j > i) {
            quad -= 2.0 * rho_car * phi_spatial[i] * phi_spatial[j];
          }
        }
      }

      // Log-determinant of Q(rho) ? recompute each call (rho changes Q).
      // Dense O(J^3) Cholesky is fine for small J; switch to sparse for large J.
      std::vector<double> Q = tulpa_car_proper::compute_car_precision(
          J, data.adj_row_ptr, data.adj_col_idx, data.n_neighbors, rho_car);
      double log_det_Q = tulpa_car_proper::car_log_det(J, Q);
      if (std::isinf(log_det_Q)) {
        return -std::numeric_limits<double>::infinity();
      }

      log_post += 0.5 * log_det_Q + 0.5 * J * log_tau_spatial - 0.5 * tau_spatial * quad;
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
  // temporal_f_nc lives in state (aliased above).

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
        log_post += tulpa_temporal_gp::log_prior_sigma2_temporal_pc(
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
  // gp_w_nc_buf lives in state (aliased above).

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
  // msgp_hsgp_f_local / msgp_hsgp_f_regional live in state (aliased above).

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
  // hsgp_f lives in state (aliased above).

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
  // latent_sigma / latent_factors_vec live in state (aliased above).
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
    // tau_st2 always 1.0 ? removed non-identifiable second precision

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

        // NC prior: -0.5 * z^T (Q_s ? Q_t) z  (tau-free GMRF)
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
  // tvc_eta lives in state (aliased above).

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

  // Populate the obs-loop context (only meaningful when populate_obs_state).
  // Pointers reference either `params`, `data`, thread_local workspaces, or
  // buffers in `state` that outlive this call.
  if (populate_obs_state) {
    ctx.beta_num        = beta_num;
    ctx.beta_denom      = beta_denom;
    ctx.re              = re;
    ctx.re_nc_flat      = re_nc_flat.empty() ? nullptr : re_nc_flat.data();
    ctx.phi_spatial     = phi_spatial;
    ctx.theta_bym2      = theta_bym2;
    ctx.sigma_s_bym2    = sigma_s_bym2;
    ctx.sigma_u_bym2    = sigma_u_bym2;
    ctx.phi_temporal    = phi_temporal;
    ctx.gp_w            = gp_w;
    ctx.gp_local        = gp_local;
    ctx.gp_regional     = gp_regional;
    ctx.msgp_hsgp_f_local    = &msgp_hsgp_f_local;
    ctx.msgp_hsgp_f_regional = &msgp_hsgp_f_regional;
    ctx.hsgp_f          = &hsgp_f;
    ctx.trend           = trend;
    ctx.seasonal        = seasonal;
    ctx.short_term      = short_term;
    ctx.latent_sigma    = &latent_sigma;
    ctx.latent_factors  = &latent_factors_vec;
    ctx.st_delta        = st_delta;
    ctx.tvc_eta         = &tvc_eta;
    ctx.svc_eta         = svc_eta_ptr;
    ctx.beta_zi         = beta_zi;
    ctx.beta_oi         = beta_oi;
    ctx.phi_num         = phi_num;
    ctx.phi_denom       = phi_denom;
  }

  return log_post;
}

// =====================================================================
// Observation-loop helper (parallel reduction over data.N).
// =====================================================================

double accumulate_obs_log_lik(const ObservationLikelihoodContext& ctx) {
  const auto& data = ctx.data;
  const auto& layout = ctx.layout;
  double log_lik = 0.0;

  // NOTE: Disable OpenMP for GP models to avoid race conditions.
  // The GP NNGP likelihood accesses shared data structures that may not be thread-safe.
  #ifdef _OPENMP
  int use_threads = (layout.is_gp || layout.is_multiscale_gp) ? 1 : data.n_threads;
  #pragma omp parallel for reduction(+:log_lik) schedule(static) \
          num_threads(use_threads)
  #endif
  for (int i = 0; i < data.N; i++) {
    log_lik += compute_observation_log_lik(ctx, i);
  }

  return log_lik;
}

// =====================================================================
// Orchestrators: compute_log_post / compute_log_prior / compute_log_lik_only
//
// The split satisfies the contract verified by tests/testthat/test-log-post-split.R:
//   compute_log_post == compute_log_prior + compute_log_lik_only
// (modulo IEEE arithmetic), and compute_log_lik_only now runs in a single
// pass instead of two compute_log_post calls (gcol33/tulpa#6 follow-up).
// =====================================================================

double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop,
    const double* precomputed_st_log_prior,
    const double* precomputed_tgp_log_prior
) {
  if (data.n_processes > 0 && data.likelihood_spec != nullptr) {
    return tulpa::compute_log_post_generic_spec_double(
        params, data, layout, skip_obs_loop);
  }

  LogPosteriorState state;
  ObservationLikelihoodContext ctx{params, data, layout};
  const bool populate = !skip_obs_loop;

  double log_prior = accumulate_log_prior_and_state(
      params, data, layout, state, ctx, populate,
      precomputed_st_log_prior, precomputed_tgp_log_prior);

  if (skip_obs_loop || (std::isinf(log_prior) && log_prior < 0.0)) {
    return log_prior;
  }
  return log_prior + accumulate_obs_log_lik(ctx);
}

double compute_log_prior(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
) {
  // Prior + structural terms only (skip the O(N) observation loop).
  return compute_log_post(params, data, layout, /*skip_obs_loop=*/true);
}

double compute_log_lik_only(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
) {
  if (data.n_processes > 0 && data.likelihood_spec != nullptr) {
    // Generic-spec path still derives by subtraction; only the legacy
    // (n_processes == 0) path has been split into separable stages.
    const double lp = tulpa::compute_log_post_generic_spec_double(
        params, data, layout, /*skip_obs_loop=*/false);
    const double lpr = tulpa::compute_log_post_generic_spec_double(
        params, data, layout, /*skip_obs_loop=*/true);
    return lp - lpr;
  }

  LogPosteriorState state;
  ObservationLikelihoodContext ctx{params, data, layout};
  const double log_prior = accumulate_log_prior_and_state(
      params, data, layout, state, ctx, /*populate_obs_state=*/true,
      /*precomputed_st_log_prior=*/nullptr,
      /*precomputed_tgp_log_prior=*/nullptr);

  // Match the historical (post - prior) behavior at -INFINITY: the obs
  // context may be partially populated, so the result is undefined and
  // tests don't probe this regime. Return 0 to keep the contract finite
  // when prior was finite, NaN-equivalent otherwise.
  if (std::isinf(log_prior) && log_prior < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return accumulate_obs_log_lik(ctx);
}

// =====================================================================

} // namespace tulpa_hmc
