// hmc_gradient_vectorized_passes.h
// Fragment of hmc_gradient_vectorized.h.
// Self-contained: opens namespace tulpa_hmc::vectorized.
// Pass 1 (expand grouped effects), Pass 2 (templated residual kernels +
// dispatcher), Pass 3 (scatter-add residuals to grouped-effect gradients).
#ifndef TULPA_HMC_GRADIENT_VECTORIZED_PASSES_H
#define TULPA_HMC_GRADIENT_VECTORIZED_PASSES_H

#include <cmath>
#include <cstring>
#include <vector>

#include <RcppEigen.h>

#include "hmc_gradient_helpers_impl.h"          // re/temporal/spatial _contribution kernels
#include "hmc_gradient_vectorized_workspace.h"  // VecGradWorkspace
#include "hmc_likelihood.h"                     // log_lik_*
#include "hmc_sampler.h"                        // ModelData, ParamLayout, ModelType
#include "portable_math.h"                      // tulpa::math::portable_digamma*

namespace tulpa_hmc {
namespace vectorized {

// ============================================================================
// Pass 1: Expand grouped effects to dense N-vectors
// ============================================================================

// Each expand_* helper is a thin wrapper that calls the per-i contribution
// kernel from hmc_gradient_helpers_impl.h in a tight loop. The same per-i
// kernels are used by the scalar fallback path, so the per-feature linear
// algebra has a single source of truth.

inline void expand_re_single(
    const ModelData& data,
    const ParamLayout& layout,
    const double* params,
    double* dense,
    int N,
    double sigma_re = 0.0  // >0 means non-centered: multiply z by sigma
) {
  for (int i = 0; i < N; i++) {
    dense[i] = re_single_contribution(i, data, layout, params, sigma_re);
  }
}

inline void expand_re_crossed(
    const ModelData& data,
    const ParamLayout& layout,
    const double* params,
    double* dense,
    int N,
    const double* sigma_re_terms = nullptr  // non-null means non-centered: multiply z by sigma per term
) {
  for (int i = 0; i < N; i++) {
    dense[i] = re_crossed_contribution(i, data, layout, params,
                                       sigma_re_terms, /*re_idx_out=*/nullptr);
  }
}

inline void expand_spatial_icar(
    const ModelData& data,
    const double* phi_spatial,
    double* dense,
    int N
) {
  for (int i = 0; i < N; i++) {
    dense[i] = spatial_icar_contribution(i, data, phi_spatial, /*s_idx_out=*/nullptr);
  }
}

inline void expand_spatial_bym2(
    const ModelData& data,
    const double* phi_spatial,
    const double* theta_bym2,
    double sigma_s, double sigma_u,
    double* dense,
    int N
) {
  for (int i = 0; i < N; i++) {
    dense[i] = spatial_bym2_contribution(i, data, phi_spatial, theta_bym2,
                                         sigma_s, sigma_u,
                                         /*s_idx_out=*/nullptr,
                                         /*d_phi_out=*/nullptr,
                                         /*d_theta_out=*/nullptr);
  }
}

inline void expand_temporal(
    const ModelData& data,
    const double* phi_temporal,
    double* dense,
    int N
) {
  for (int i = 0; i < N; i++) {
    dense[i] = temporal_contribution(i, data, phi_temporal, /*t_idx_out=*/nullptr);
  }
}

// ============================================================================
// Pass 2: Templated residual kernels (one per ModelType)
// Each computes resid_num[i], resid_denom[i], and phi gradient contributions.
// ============================================================================

template<ModelType MT> struct ModelTag {};

// --- BINOMIAL ---
// Fused: precomputes lchoose(n,y); uses log-sum-exp stable log-lik.
inline void compute_residuals_impl(
    ModelTag<ModelType::BINOMIAL>,
    int N,
    const double* eta_num, const double* /*eta_denom*/,
    const ModelData& data,
    double /*phi_num*/, double /*phi_denom*/,
    double* resid_num, double* resid_denom,
    double& grad_phi_num_lik, double& grad_phi_denom_lik,
    double& obs_ll, bool compute_lp,
    const VecGradWorkspace& ws
) {
  grad_phi_num_lik = 0.0;
  grad_phi_denom_lik = 0.0;
  double ll = 0.0;

  for (int i = 0; i < N; i++) {
    const double eta = eta_num[i];
    const int n_trials = data.legacy.y_denom[i];
    const int y = data.legacy.y_num[i];

    // Numerically stable logistic: 1 exp + 1 log (was 3 exp + 2 log)
    // Identity: log(1-p) = -eta + log(p) when eta >= 0
    //           log(p) = eta + log(1-p) when eta < 0
    double p, log_p, log_1mp;
    if (eta >= 0.0) {
      const double exp_neg = std::exp(-eta);
      const double denom = 1.0 + exp_neg;
      p = 1.0 / denom;
      log_p = -std::log(denom);
      log_1mp = -eta + log_p;
    } else {
      const double exp_pos = std::exp(eta);
      const double denom = 1.0 + exp_pos;
      p = exp_pos / denom;
      log_1mp = -std::log(denom);
      log_p = eta + log_1mp;
    }

    resid_num[i] = y - n_trials * p;
    resid_denom[i] = 0.0;

    if (compute_lp) {
      ll += y * log_p + (n_trials - y) * log_1mp + ws.lchoose_cache[i];
    }
  }
  obs_ll += ll;
}

// --- NEGBIN_NEGBIN ---
// Fused gradient + log-lik with hoisted phi-constants, precomputed lgamma(y+1),
// and digamma/lgamma lookup tables indexed by integer y.
// Per-obs transcendentals: 0 in multipass (exp/log via SIMD, digamma/lgamma via table).
inline void compute_residuals_impl(
    ModelTag<ModelType::NEGBIN_NEGBIN>,
    int N,
    const double* eta_num, const double* eta_denom,
    const ModelData& data,
    double phi_num, double phi_denom,
    double* resid_num, double* resid_denom,
    double& grad_phi_num_lik, double& grad_phi_denom_lik,
    double& obs_ll, bool compute_lp,
    const VecGradWorkspace& ws
) {
  // Hoist phi-dependent constants (computed once, not N times)
  const double digamma_phi_n = tulpa::math::portable_digamma(phi_num);
  const double digamma_phi_d = tulpa::math::portable_digamma(phi_denom);
  const double log_phi_n = std::log(phi_num);
  const double log_phi_d = std::log(phi_denom);
  const double lgamma_phi_n = compute_lp ? std::lgamma(phi_num) : 0.0;
  const double lgamma_phi_d = compute_lp ? std::lgamma(phi_denom) : 0.0;

  double gpn = 0.0, gpd = 0.0;
  double ll = 0.0;

  for (int i = 0; i < N; i++) {
    const double mu_num = std::exp(eta_num[i]);
    const double mu_denom = std::exp(eta_denom[i]);
    const int y_n = data.legacy.y_num[i];
    const int y_d = data.legacy.y_denom[i];

    // Shared intermediates (reciprocal: 1 div replaces 2 per component)
    const double dn = mu_num + phi_num;
    const double dd = mu_denom + phi_denom;
    const double inv_dn = 1.0 / dn;
    const double inv_dd = 1.0 / dd;

    // Gradient residuals
    resid_num[i] = y_n - mu_num * (y_n + phi_num) * inv_dn;
    resid_denom[i] = y_d - mu_denom * (y_d + phi_denom) * inv_dd;

    // Table lookup: digamma/lgamma indexed by integer y (table built per gradient call)
    const double digamma_yn_phi = ws.dig_table_num[y_n];
    const double lgamma_yn_phi = ws.lg_table_num[y_n];
    const double digamma_yd_phi = ws.dig_table_denom[y_d];
    const double lgamma_yd_phi = ws.lg_table_denom[y_d];
    const double log_dn = std::log(dn);
    const double log_dd = std::log(dd);
    const double log_phi_dn = log_phi_n - log_dn;
    const double log_phi_dd = log_phi_d - log_dd;

    gpn += digamma_yn_phi - digamma_phi_n + log_phi_dn + (mu_num - y_n) * inv_dn;
    gpd += digamma_yd_phi - digamma_phi_d + log_phi_dd + (mu_denom - y_d) * inv_dd;

    if (compute_lp) {
      // Fused log-lik using precomputed lgamma(y+1) and hoisted lgamma(phi)
      // log(mu/(mu+phi)) = log(mu) - log(mu+phi) = eta - log_dn (no extra log call)
      const double log_mu_dn = eta_num[i] - log_dn;
      const double log_mu_dd = eta_denom[i] - log_dd;
      ll += lgamma_yn_phi - ws.lgamma_y_num_p1[i] - lgamma_phi_n
          + phi_num * log_phi_dn + y_n * log_mu_dn;
      ll += lgamma_yd_phi - ws.lgamma_y_denom_p1[i] - lgamma_phi_d
          + phi_denom * log_phi_dd + y_d * log_mu_dd;
    }
  }
  grad_phi_num_lik += gpn;
  grad_phi_denom_lik += gpd;
  obs_ll += ll;
}

// --- POISSON_GAMMA ---
// Fused: hoists lgamma(phi), digamma(phi), log(phi); precomputes lgamma(y_num+1), log(y_denom).
inline void compute_residuals_impl(
    ModelTag<ModelType::POISSON_GAMMA>,
    int N,
    const double* eta_num, const double* eta_denom,
    const ModelData& data,
    double /*phi_num*/, double phi_denom,
    double* resid_num, double* resid_denom,
    double& grad_phi_num_lik, double& grad_phi_denom_lik,
    double& obs_ll, bool compute_lp,
    const VecGradWorkspace& ws
) {
  // Hoist phi-dependent constants
  const double digamma_phi = tulpa::math::portable_digamma(phi_denom);
  const double log_phi = std::log(phi_denom);
  const double lgamma_phi = compute_lp ? std::lgamma(phi_denom) : 0.0;

  double gpd = 0.0;
  double ll = 0.0;
  grad_phi_num_lik = 0.0;

  for (int i = 0; i < N; i++) {
    const double mu_num = std::exp(eta_num[i]);
    const double mu_denom = std::exp(eta_denom[i]);
    const int y_n = data.legacy.y_num[i];
    const double y_d = data.legacy.y_denom_cont[i];

    resid_num[i] = y_n - mu_num;
    // Gamma requires y > 0; skip denom contribution for invalid obs
    if (y_d > 0.0) {
      const double inv_mu_d = 1.0 / mu_denom;
      const double yd_over_mud = y_d * inv_mu_d;
      resid_denom[i] = phi_denom * (yd_over_mud - 1.0);
      const double log_rate = log_phi - eta_denom[i];
      gpd += log_rate + 1.0 + ws.log_y_denom_cont[i] - digamma_phi
           - yd_over_mud;
      if (compute_lp) {
        ll += y_n * eta_num[i] - mu_num - ws.lgamma_y_num_p1[i];
        ll += (phi_denom - 1.0) * ws.log_y_denom_cont[i] + phi_denom * log_rate
            - phi_denom * yd_over_mud - lgamma_phi;
      }
    } else {
      resid_denom[i] = 0.0;
      if (compute_lp) {
        ll += y_n * eta_num[i] - mu_num - ws.lgamma_y_num_p1[i];
        ll += -1e10;
      }
    }
  }
  grad_phi_denom_lik += gpd;
  obs_ll += ll;
}

// --- NEGBIN_GAMMA ---
// NB numerator (integer y, digamma/lgamma tables) + Gamma denominator (continuous y).
inline void compute_residuals_impl(
    ModelTag<ModelType::NEGBIN_GAMMA>,
    int N,
    const double* eta_num, const double* eta_denom,
    const ModelData& data,
    double phi_num, double phi_denom,
    double* resid_num, double* resid_denom,
    double& grad_phi_num_lik, double& grad_phi_denom_lik,
    double& obs_ll, bool compute_lp,
    const VecGradWorkspace& ws
) {
  // Hoist phi-dependent constants
  const double digamma_phi_n = tulpa::math::portable_digamma(phi_num);
  const double digamma_phi_d = tulpa::math::portable_digamma(phi_denom);
  const double log_phi_n = std::log(phi_num);
  const double log_phi_d = std::log(phi_denom);
  const double lgamma_phi_n = compute_lp ? std::lgamma(phi_num) : 0.0;
  const double lgamma_phi_d = compute_lp ? std::lgamma(phi_denom) : 0.0;

  double gpn = 0.0, gpd = 0.0;
  double ll = 0.0;

  for (int i = 0; i < N; i++) {
    const double mu_num = std::exp(eta_num[i]);
    const double mu_denom = std::exp(eta_denom[i]);
    const int y_n = data.legacy.y_num[i];
    const double y_d = data.legacy.y_denom_cont[i];

    // NB numerator
    const double dn = mu_num + phi_num;
    const double inv_dn = 1.0 / dn;
    resid_num[i] = y_n - mu_num * (y_n + phi_num) * inv_dn;

    // NB phi_num gradient (using digamma table)
    const double digamma_yn_phi = ws.dig_table_num[y_n];
    const double lgamma_yn_phi = ws.lg_table_num[y_n];
    const double log_dn = std::log(dn);
    const double log_phi_dn = log_phi_n - log_dn;
    gpn += digamma_yn_phi - digamma_phi_n + log_phi_dn + (mu_num - y_n) * inv_dn;

    // Gamma denominator — skip if y <= 0
    if (y_d > 0.0) {
      const double inv_mu_d = 1.0 / mu_denom;
      const double yd_over_mud = y_d * inv_mu_d;
      resid_denom[i] = phi_denom * (yd_over_mud - 1.0);
      const double log_rate = log_phi_d - eta_denom[i];
      gpd += log_rate + 1.0 + ws.log_y_denom_cont[i] - digamma_phi_d - yd_over_mud;
      if (compute_lp) {
        ll += lgamma_yn_phi - ws.lgamma_y_num_p1[i] - lgamma_phi_n
            + phi_num * log_phi_dn + y_n * (eta_num[i] - log_dn);
        ll += (phi_denom - 1.0) * ws.log_y_denom_cont[i] + phi_denom * log_rate
            - phi_denom * yd_over_mud - lgamma_phi_d;
      }
    } else {
      resid_denom[i] = 0.0;
      if (compute_lp) {
        ll += lgamma_yn_phi - ws.lgamma_y_num_p1[i] - lgamma_phi_n
            + phi_num * log_phi_dn + y_n * (eta_num[i] - log_dn);
        ll += -1e10;
      }
    }
  }
  grad_phi_num_lik += gpn;
  grad_phi_denom_lik += gpd;
  obs_ll += ll;
}

// --- GAMMA_GAMMA ---
// Fused: hoists lgamma(phi), digamma(phi), log(phi).
inline void compute_residuals_impl(
    ModelTag<ModelType::GAMMA_GAMMA>,
    int N,
    const double* eta_num, const double* eta_denom,
    const ModelData& data,
    double phi_num, double phi_denom,
    double* resid_num, double* resid_denom,
    double& grad_phi_num_lik, double& grad_phi_denom_lik,
    double& obs_ll, bool compute_lp,
    const VecGradWorkspace& ws
) {
  // Hoist phi-dependent constants
  const double digamma_phi_n = tulpa::math::portable_digamma(phi_num);
  const double digamma_phi_d = tulpa::math::portable_digamma(phi_denom);
  const double log_phi_n = std::log(phi_num);
  const double log_phi_d = std::log(phi_denom);
  const double lgamma_phi_n = compute_lp ? std::lgamma(phi_num) : 0.0;
  const double lgamma_phi_d = compute_lp ? std::lgamma(phi_denom) : 0.0;

  double gpn = 0.0, gpd = 0.0;
  double ll = 0.0;

  for (int i = 0; i < N; i++) {
    const double mu_num = std::exp(eta_num[i]);
    const double mu_denom = std::exp(eta_denom[i]);
    const double y_n = data.legacy.y_num_cont[i];
    const double y_d = data.legacy.y_denom_cont[i];

    // Gamma requires y > 0
    if (y_n > 0.0) {
      const double inv_mu_n = 1.0 / mu_num;
      const double yn_over_mun = y_n * inv_mu_n;
      resid_num[i] = phi_num * (yn_over_mun - 1.0);
      const double log_rate_n = log_phi_n - eta_num[i];
      gpn += log_rate_n + 1.0 + ws.log_y_num_cont[i] - digamma_phi_n - yn_over_mun;
      if (compute_lp)
        ll += (phi_num - 1.0) * ws.log_y_num_cont[i] + phi_num * log_rate_n
            - phi_num * yn_over_mun - lgamma_phi_n;
    } else {
      resid_num[i] = 0.0;
      if (compute_lp) ll += -1e10;
    }
    if (y_d > 0.0) {
      const double inv_mu_d = 1.0 / mu_denom;
      const double yd_over_mud = y_d * inv_mu_d;
      resid_denom[i] = phi_denom * (yd_over_mud - 1.0);
      const double log_rate_d = log_phi_d - eta_denom[i];
      gpd += log_rate_d + 1.0 + ws.log_y_denom_cont[i] - digamma_phi_d - yd_over_mud;
      if (compute_lp)
        ll += (phi_denom - 1.0) * ws.log_y_denom_cont[i] + phi_denom * log_rate_d
            - phi_denom * yd_over_mud - lgamma_phi_d;
    } else {
      resid_denom[i] = 0.0;
      if (compute_lp) ll += -1e10;
    }
  }
  grad_phi_num_lik += gpn;
  grad_phi_denom_lik += gpd;
  obs_ll += ll;
}

// --- LOGNORMAL ---
// Fused: hoists log(sigma) outside loop (no lgamma/digamma needed).
inline void compute_residuals_impl(
    ModelTag<ModelType::LOGNORMAL>,
    int N,
    const double* eta_num, const double* eta_denom,
    const ModelData& data,
    double phi_num, double phi_denom,
    double* resid_num, double* resid_denom,
    double& grad_phi_num_lik, double& grad_phi_denom_lik,
    double& obs_ll, bool compute_lp,
    const VecGradWorkspace& ws
) {
  const double sigma_num_sq = phi_num * phi_num;
  const double sigma_denom_sq = phi_denom * phi_denom;
  const double log_sigma_n = compute_lp ? std::log(phi_num) : 0.0;
  const double log_sigma_d = compute_lp ? std::log(phi_denom) : 0.0;

  double gpn = 0.0, gpd = 0.0;
  double ll = 0.0;

  for (int i = 0; i < N; i++) {
    const double mu_num = eta_num[i];
    const double mu_denom = eta_denom[i];
    const double log_yn = ws.log_y_num_cont[i];
    const double log_yd = ws.log_y_denom_cont[i];

    resid_num[i] = (log_yn - mu_num) / sigma_num_sq;
    resid_denom[i] = (log_yd - mu_denom) / sigma_denom_sq;

    const double z_n = (log_yn - mu_num) / phi_num;
    const double z_d = (log_yd - mu_denom) / phi_denom;
    gpn += (-1.0 + z_n * z_n) / phi_num;
    gpd += (-1.0 + z_d * z_d) / phi_denom;

    if (compute_lp) {
      ll += -log_yn - log_sigma_n - 0.5 * z_n * z_n;
      ll += -log_yd - log_sigma_d - 0.5 * z_d * z_d;
    }
  }
  grad_phi_num_lik += gpn;
  grad_phi_denom_lik += gpd;
  obs_ll += ll;
}

// --- BETA_BINOMIAL ---
// Fused: hoists digamma(phi), lgamma(phi); precomputes lchoose(n,y).
inline void compute_residuals_impl(
    ModelTag<ModelType::BETA_BINOMIAL>,
    int N,
    const double* eta_num, const double* /*eta_denom*/,
    const ModelData& data,
    double phi_num, double /*phi_denom*/,
    double* resid_num, double* resid_denom,
    double& grad_phi_num_lik, double& grad_phi_denom_lik,
    double& obs_ll, bool compute_lp,
    const VecGradWorkspace& ws
) {
  // Hoist phi-dependent constants
  const double psi_phi = tulpa::math::portable_digamma(phi_num);
  const double lgamma_phi = compute_lp ? std::lgamma(phi_num) : 0.0;

  double gpn = 0.0;
  grad_phi_denom_lik = 0.0;
  double ll = 0.0;

  for (int i = 0; i < N; i++) {
    // Numerically stable sigmoid (avoids exp overflow for |eta| > 700)
    const double eta = eta_num[i];
    const double p = (eta >= 0.0)
        ? 1.0 / (1.0 + std::exp(-eta))
        : std::exp(eta) / (1.0 + std::exp(eta));
    const int y_i = data.legacy.y_num[i];
    const int n_i = data.legacy.y_denom[i];
    const double alpha = p * phi_num;
    const double beta_param = (1.0 - p) * phi_num;

    // Fused digamma+lgamma: saves 5 log() per obs vs separate calls
    auto [psi_y_alpha, lg_y_alpha] = tulpa::math::portable_digamma_lgamma(y_i + alpha);
    auto [psi_nmy_beta, lg_nmy_beta] = tulpa::math::portable_digamma_lgamma(n_i - y_i + beta_param);
    auto [psi_alpha, lg_alpha] = tulpa::math::portable_digamma_lgamma(alpha);
    auto [psi_beta, lg_beta] = tulpa::math::portable_digamma_lgamma(beta_param);
    auto [psi_n_phi, lg_n_phi] = tulpa::math::portable_digamma_lgamma(n_i + phi_num);
    const double dLL_dp = phi_num * (psi_y_alpha - psi_nmy_beta - psi_alpha + psi_beta);
    resid_num[i] = dLL_dp * p * (1.0 - p);
    resid_denom[i] = 0.0;

    gpn += p * psi_y_alpha + (1.0 - p) * psi_nmy_beta - psi_n_phi
         - p * psi_alpha - (1.0 - p) * psi_beta + psi_phi;

    if (compute_lp) {
      ll += lg_y_alpha + lg_nmy_beta - lg_n_phi;
      ll += -lg_alpha - lg_beta + lgamma_phi;
      ll += ws.lchoose_cache[i];
    }
  }
  grad_phi_num_lik += gpn;
  obs_ll += ll;
}

// ============================================================================
// Pass 2 dispatcher
// ============================================================================

template<ModelType MT>
inline void compute_residuals(
    int N,
    const double* eta_num, const double* eta_denom,
    const ModelData& data,
    double phi_num, double phi_denom,
    double* resid_num, double* resid_denom,
    double& grad_phi_num_lik, double& grad_phi_denom_lik,
    double& obs_ll, bool compute_lp,
    const VecGradWorkspace& ws
) {
  compute_residuals_impl(
    ModelTag<MT>{}, N, eta_num, eta_denom, data,
    phi_num, phi_denom, resid_num, resid_denom,
    grad_phi_num_lik, grad_phi_denom_lik, obs_ll, compute_lp, ws
  );
}

// ============================================================================
// Pass 3: Scatter-add residuals back to grouped effect gradients
// ============================================================================

inline void accumulate_re_gradient_single(
    const ModelData& data,
    const ParamLayout& layout,
    const double* resid_num, const double* resid_denom,
    bool is_binomial,
    double* grad,
    int N
) {
  for (int i = 0; i < N; i++) {
    int gidx = data.re_group[i];
    if (gidx > 0) {
      double re_grad_i = resid_num[i];
      if (!is_binomial) re_grad_i += resid_denom[i];
      grad[layout.re_start + gidx - 1] += re_grad_i;
    }
  }
}

inline void accumulate_re_gradient_crossed(
    const ModelData& data,
    const ParamLayout& layout,
    const double* resid_num, const double* resid_denom,
    bool is_binomial,
    double* grad,
    int N
) {
  const int n_terms = data.n_re_terms;
  for (int i = 0; i < N; i++) {
    double re_grad_i = resid_num[i];
    if (!is_binomial) re_grad_i += resid_denom[i];
    for (int t = 0; t < n_terms; t++) {
      int gidx = data.re_group_multi_flat[i * n_terms + t];
      if (gidx > 0) {
        grad[layout.re_start_multi[t] + gidx - 1] += re_grad_i;
      }
    }
  }
}

inline void accumulate_temporal_gradient(
    const ModelData& data,
    const double* resid_num, const double* resid_denom,
    bool is_binomial,
    double* grad_temporal_lik,
    int N
) {
  const auto& tidx = data.temporal_time_idx;
  const auto& gidx = data.temporal_group_idx;
  int T = data.n_times;
  for (int i = 0; i < N; i++) {
    if (tidx[i] > 0) {
      double temp_grad_i = resid_num[i];
      if (!is_binomial) temp_grad_i += resid_denom[i];
      int t = tidx[i] - 1;
      int g = gidx[i] - 1;
      grad_temporal_lik[g * T + t] += temp_grad_i;  // Panel temporal: flat index
    }
  }
}

// Accumulate BYM2 spatial gradient from dense residual vectors
inline void accumulate_spatial_gradient_bym2(
    const ModelData& data,
    double sigma_s, double sigma_u,
    const double* resid_num, const double* resid_denom,
    bool is_binomial,
    double* grad_spatial_lik,
    double* grad_theta,
    int N
) {
  double scale = data.bym2_scale_factor;
  const auto& group = data.spatial_group;
  for (int i = 0; i < N; i++) {
    if (group[i] > 0) {
      int s = group[i] - 1;
      double lik_grad = resid_num[i];
      if (!is_binomial) lik_grad += resid_denom[i];
      grad_spatial_lik[s] += lik_grad * sigma_s * scale;
      grad_theta[s] += lik_grad * sigma_u;
    }
  }
}

}  // namespace vectorized
}  // namespace tulpa_hmc

#endif  // TULPA_HMC_GRADIENT_VECTORIZED_PASSES_H
