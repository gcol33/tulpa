// hmc_gradient_vectorized.h
// Vectorized gradient computation with template specialization per model type.
// Replaces scalar per-observation loop in compute_gradient_analytical() for
// non-ZI, non-slopes model configurations.
//
// Architecture (three-pass):
//   Pass 1: Vectorized eta = X * beta  (Eigen matvec)
//           + dense-expanded RE/spatial/temporal additions
//   Pass 2: Scalar model-specific residual kernel (templated per ModelType)
//   Pass 3: Vectorized grad_beta = X^T * resid  (Eigen matvec)
//
// Template dispatch eliminates dead branches at compile time, reducing
// instruction cache pressure from ~30-50KB to ~5KB per instantiation.
//
// IMPORTANT: This header must be included from hmc_sampler.cpp AFTER
// hmc_sampler.h and the log_lik_* inline functions are defined.

#ifndef TULPA_HMC_GRADIENT_VECTORIZED_H
#define TULPA_HMC_GRADIENT_VECTORIZED_H

#include <vector>
#include <cmath>
#include <cstring>

// Assumes hmc_sampler.h and RcppEigen.h are already included by the .cpp file.
// Assumes log_lik_binomial/negbin/poisson/gamma are defined above in the .cpp.
// IMPORTANT: This file is included from within namespace tulpa_hmc {} in hmc_sampler.cpp.
// Do NOT wrap contents in namespace tulpa_hmc — it would be doubly nested.

namespace vectorized {

// ============================================================================
// Pre-allocated workspace for vectorized gradient (avoids per-call allocation)
// ============================================================================
struct VecGradWorkspace {
  int N = 0;
  std::vector<double> eta_num;
  std::vector<double> eta_denom;
  std::vector<double> mu_num;     // exp(eta_num) — vectorized via Eigen
  std::vector<double> mu_denom;   // exp(eta_denom) — vectorized via Eigen
  std::vector<double> resid_num;
  std::vector<double> resid_denom;
  std::vector<double> effect_dense;  // Pre-expanded RE+spatial+temporal

  // Parameter-dependent intermediates (recomputed each gradient call via Eigen SIMD)
  std::vector<double> log_dn;    // log(mu_num + phi_num) for NB
  std::vector<double> log_dd;    // log(mu_denom + phi_denom) for NB

  // Precomputed data-dependent constants (computed once per dataset)
  uint64_t cached_data_id = 0;
  std::vector<double> lgamma_y_num_p1;    // lgamma(y_num[i] + 1)
  std::vector<double> lgamma_y_denom_p1;  // lgamma(y_denom[i] + 1)
  std::vector<double> lchoose_cache;      // lchoose(n_trials[i], y[i]) for binomial
  std::vector<double> log_y_num_cont;     // log(y_num_cont[i]) for GG, LN
  std::vector<double> log_y_denom_cont;   // log(y_denom_cont[i]) for PG, GG, LN

  // Digamma/lgamma lookup tables for NB/PG: indexed by integer y value.
  // Rebuilt per gradient call (phi changes), but only max_y+1 entries vs N observations.
  // Typically ~50-100 entries vs 500+ observations = 5-10x fewer digamma calls.
  int max_y_num_val = -1;   // max(y_num) across dataset
  int max_y_denom_val = -1; // max(y_denom) across dataset
  std::vector<double> dig_table_num;   // digamma(y + phi_num) for y = 0..max_y_num
  std::vector<double> lg_table_num;    // lgamma(y + phi_num) for y = 0..max_y_num
  std::vector<double> dig_table_denom; // digamma(y + phi_denom) for y = 0..max_y_denom
  std::vector<double> lg_table_denom;  // lgamma(y + phi_denom) for y = 0..max_y_denom

  void init(int n) {
    if (n == N) return;  // Already sized
    N = n;
    eta_num.resize(n);
    eta_denom.resize(n);
    mu_num.resize(n);
    mu_denom.resize(n);
    resid_num.resize(n);
    resid_denom.resize(n);
    effect_dense.resize(n);
    log_dn.resize(n);
    log_dd.resize(n);
    lgamma_y_num_p1.resize(n);
    lgamma_y_denom_p1.resize(n);
    lchoose_cache.resize(n);
    log_y_num_cont.resize(n);
    log_y_denom_cont.resize(n);
    cached_data_id = 0;  // Force recompute on next use
  }

  // Precompute data-only constants (lgamma(y+1), lchoose, log(y)) — called once per dataset
  void precompute(const ModelData& data) {
    if (data.unique_id == cached_data_id) return;  // Already computed for this data
    cached_data_id = data.unique_id;
    max_y_num_val = 0;
    max_y_denom_val = 0;
    for (int i = 0; i < N; i++) {
      lgamma_y_num_p1[i] = std::lgamma(data.legacy.y_num[i] + 1.0);
      lgamma_y_denom_p1[i] = std::lgamma(data.legacy.y_denom[i] + 1.0);
      // lchoose for binomial/beta-binomial: lgamma(n+1) - lgamma(y+1) - lgamma(n-y+1)
      lchoose_cache[i] = std::lgamma(data.legacy.y_denom[i] + 1.0)
                        - std::lgamma(data.legacy.y_num[i] + 1.0)
                        - std::lgamma(data.legacy.y_denom[i] - data.legacy.y_num[i] + 1.0);
      // log(y) for PG/GG/LN families (data-only, never changes)
      log_y_num_cont[i] = (i < static_cast<int>(data.legacy.y_num_cont.size()) && data.legacy.y_num_cont[i] > 0.0)
                         ? std::log(data.legacy.y_num_cont[i]) : 0.0;
      log_y_denom_cont[i] = (i < static_cast<int>(data.legacy.y_denom_cont.size()) && data.legacy.y_denom_cont[i] > 0.0)
                           ? std::log(data.legacy.y_denom_cont[i]) : 0.0;
      // Track max y values for lookup table sizing
      if (data.legacy.y_num[i] > max_y_num_val) max_y_num_val = data.legacy.y_num[i];
      if (data.legacy.y_denom[i] > max_y_denom_val) max_y_denom_val = data.legacy.y_denom[i];
    }
  }

  // Build digamma/lgamma lookup tables for current phi values.
  // Called once per gradient evaluation (phi changes each leapfrog step).
  // Cost: ~(max_y_num + max_y_denom) digamma calls vs N*2 without table.
  void build_digamma_tables(double phi_num, double phi_denom, bool need_denom) {
    int tn = max_y_num_val + 1;
    if (static_cast<int>(dig_table_num.size()) < tn) {
      dig_table_num.resize(tn);
      lg_table_num.resize(tn);
    }
    for (int y = 0; y < tn; y++) {
      auto [d, l] = tulpa::math::portable_digamma_lgamma(y + phi_num);
      dig_table_num[y] = d;
      lg_table_num[y] = l;
    }
    if (need_denom) {
      int td = max_y_denom_val + 1;
      if (static_cast<int>(dig_table_denom.size()) < td) {
        dig_table_denom.resize(td);
        lg_table_denom.resize(td);
      }
      for (int y = 0; y < td; y++) {
        auto [d, l] = tulpa::math::portable_digamma_lgamma(y + phi_denom);
        dig_table_denom[y] = d;
        lg_table_denom[y] = l;
      }
    }
  }
};

// ============================================================================
// Pass 1: Expand grouped effects to dense N-vectors
// ============================================================================

inline void expand_re_single(
    const ModelData& data,
    const ParamLayout& layout,
    const double* params,
    double* dense,
    int N,
    double sigma_re = 0.0  // >0 means non-centered: multiply z by sigma
) {
  const double* re = &params[layout.re_start];
  const auto& group = data.re_group;
  if (sigma_re > 0.0) {
    // Non-centered: dense[i] = sigma * z[g]
    for (int i = 0; i < N; i++) {
      dense[i] = (group[i] > 0) ? sigma_re * re[group[i] - 1] : 0.0;
    }
  } else {
    // Centered: dense[i] = re[g]
    for (int i = 0; i < N; i++) {
      dense[i] = (group[i] > 0) ? re[group[i] - 1] : 0.0;
    }
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
  std::memset(dense, 0, N * sizeof(double));
  const int n_terms = data.n_re_terms;
  for (int i = 0; i < N; i++) {
    for (int t = 0; t < n_terms; t++) {
      int gidx = data.re_group_multi_flat[i * n_terms + t];
      if (gidx > 0) {
        double scale = (sigma_re_terms != nullptr) ? sigma_re_terms[t] : 1.0;
        dense[i] += scale * params[layout.re_start_multi[t] + gidx - 1];
      }
    }
  }
}

inline void expand_spatial_icar(
    const ModelData& data,
    const double* phi_spatial,
    double* dense,
    int N
) {
  const auto& group = data.spatial_group;
  for (int i = 0; i < N; i++) {
    dense[i] = (group[i] > 0) ? phi_spatial[group[i] - 1] : 0.0;
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
  double scale = data.bym2_scale_factor;
  const auto& group = data.spatial_group;
  for (int i = 0; i < N; i++) {
    if (group[i] > 0) {
      int s = group[i] - 1;
      dense[i] = sigma_s * phi_spatial[s] * scale + sigma_u * theta_bym2[s];
    } else {
      dense[i] = 0.0;
    }
  }
}

inline void expand_temporal(
    const ModelData& data,
    const double* phi_temporal,
    double* dense,
    int N
) {
  const auto& tidx = data.temporal_time_idx;
  const auto& gidx = data.temporal_group_idx;
  int T = data.n_times;
  for (int i = 0; i < N; i++) {
    if (tidx[i] > 0) {
      int t = tidx[i] - 1;
      int g = gidx[i] - 1;
      dense[i] = phi_temporal[g * T + t];  // Panel temporal: flat index
    } else {
      dense[i] = 0.0;
    }
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

// ============================================================================
// Shared vectorized residual + beta-grad kernel for specialized gradient fns
// ============================================================================
//
// These functions allow specialized gradient functions (HSGP, temporal GP,
// spatiotemporal, slopes) to delegate the expensive obs loop to the same
// template-specialized, Eigen-vectorized infrastructure used by
// compute_obs_gradients_vectorized(). The caller:
//   1. Builds eta vectors (X*beta + RE + custom effects) using Eigen matvec
//   2. Calls compute_residuals_and_beta_grads() for residuals + beta grads
//   3. Scatters residuals to custom effect gradient buffers
//
// This eliminates:
//   - Scalar dot products for eta (replaced by Eigen matvec in caller)
//   - Runtime if/else on model_type (replaced by template dispatch)
//   - Scalar scatter for X^T * resid (replaced by Eigen matvec here)

template<ModelType MT>
void compute_residuals_and_beta_grads(
    const ModelData& data,
    const ParamLayout& layout,
    const double* eta_num,             // INPUT: pre-built [N] with all effects
    const double* eta_denom,           // INPUT: pre-built [N] with all effects
    double* resid_num_out,             // OUTPUT: [N] residuals
    double* resid_denom_out,           // OUTPUT: [N] residuals
    double* grad,                      // beta grads accumulated here
    double& grad_phi_num_lik,          // OUTPUT: phi_num likelihood gradient
    double& grad_phi_denom_lik,        // OUTPUT: phi_denom likelihood gradient
    double& obs_ll,                    // OUTPUT: obs log-likelihood (if compute_lp)
    bool compute_lp,
    double phi_num,
    double phi_denom,
    VecGradWorkspace& ws
) {
    const int N = data.N;
    constexpr bool is_binomial = (MT == ModelType::BINOMIAL) || (MT == ModelType::BETA_BINOMIAL);

    // Ensure workspace is initialized and precomputed
    ws.init(N);
    ws.precompute(data);

    // Build digamma/lgamma lookup tables for NB families
    if constexpr (MT == ModelType::NEGBIN_NEGBIN) {
        ws.build_digamma_tables(phi_num, phi_denom, true);
    } else if constexpr (MT == ModelType::NEGBIN_GAMMA) {
        ws.build_digamma_tables(phi_num, phi_denom, false);
    }

    // === Pass 2: Template-specialized residual computation ===
    // Copy eta into workspace (residual kernels read from ws for multipass SIMD)
    // Skip if caller already built eta directly in ws (avoids UB from self-memcpy)
    if (eta_num != ws.eta_num.data()) {
        std::memcpy(ws.eta_num.data(), eta_num, N * sizeof(double));
    }
    if constexpr (!is_binomial) {
        if (eta_denom != ws.eta_denom.data()) {
            std::memcpy(ws.eta_denom.data(), eta_denom, N * sizeof(double));
        }
    }

    grad_phi_num_lik = 0.0;
    grad_phi_denom_lik = 0.0;

    compute_residuals<MT>(
        N, ws.eta_num.data(), ws.eta_denom.data(), data,
        phi_num, phi_denom,
        ws.resid_num.data(), ws.resid_denom.data(),
        grad_phi_num_lik, grad_phi_denom_lik,
        obs_ll, compute_lp, ws
    );

    // Copy residuals to output (skip if output IS the workspace buffer)
    if (resid_num_out != ws.resid_num.data()) {
        std::memcpy(resid_num_out, ws.resid_num.data(), N * sizeof(double));
    }
    if constexpr (!is_binomial) {
        if (resid_denom_out != ws.resid_denom.data()) {
            std::memcpy(resid_denom_out, ws.resid_denom.data(), N * sizeof(double));
        }
    } else {
        // Always zero resid_denom for binomial — memset has no UB with self-target
        // (unlike memcpy). Workspace may have stale data from prior non-binomial call.
        std::memset(resid_denom_out, 0, N * sizeof(double));
    }

    // === Pass 3: Vectorized beta gradient accumulation ===
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    {
        Eigen::Map<const RowMajorMatrix> X_num(data.legacy.X_num_flat.data(), N, data.legacy.p_num);
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
}

// Runtime dispatcher: switches on data.legacy.model_type, calls template instantiation
inline bool dispatch_residuals_and_beta_grads(
    const ModelData& data,
    const ParamLayout& layout,
    const double* eta_num,
    const double* eta_denom,
    double* resid_num_out,
    double* resid_denom_out,
    double* grad,
    double& grad_phi_num_lik,
    double& grad_phi_denom_lik,
    double& obs_ll,
    bool compute_lp,
    double phi_num,
    double phi_denom,
    VecGradWorkspace& ws
) {
    #define DISPATCH_CASE(MT) \
        case MT: \
            compute_residuals_and_beta_grads<MT>( \
                data, layout, eta_num, eta_denom, \
                resid_num_out, resid_denom_out, grad, \
                grad_phi_num_lik, grad_phi_denom_lik, \
                obs_ll, compute_lp, phi_num, phi_denom, ws); \
            return true;

    switch (data.legacy.model_type) {
        DISPATCH_CASE(ModelType::BINOMIAL)
        DISPATCH_CASE(ModelType::NEGBIN_NEGBIN)
        DISPATCH_CASE(ModelType::POISSON_GAMMA)
        DISPATCH_CASE(ModelType::NEGBIN_GAMMA)
        DISPATCH_CASE(ModelType::GAMMA_GAMMA)
        DISPATCH_CASE(ModelType::LOGNORMAL)
        DISPATCH_CASE(ModelType::BETA_BINOMIAL)
    }
    #undef DISPATCH_CASE
    return false;
}

}  // namespace vectorized
// Note: namespace tulpa_hmc is NOT closed here — the .cpp file handles it

#endif  // TULPA_HMC_GRADIENT_VECTORIZED_H
