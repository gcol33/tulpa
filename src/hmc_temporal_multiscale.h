// hmc_temporal_multiscale.h
// Multi-scale temporal decomposition: trend + seasonal + short-term
// Builds on existing RW1/RW2/AR1 infrastructure
//
// Templated over the scalar type: double for evaluation, the autodiff
// types (arena::Var) for sampler gradients.

#ifndef TULPA_HMC_TEMPORAL_MULTISCALE_H
#define TULPA_HMC_TEMPORAL_MULTISCALE_H

#include <vector>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include "hmc_temporal.h"  // For temporal kernels (RW1, RW2, AR1)
#include "pc_prior.h"
#include "tulpa/sum_to_zero.h"  // s2z_aug_coef / s2z_aug_rank / component sums

namespace tulpa_temporal {

// Types imported via hmc_temporal.h -> tulpa/temporal_data.h
// TemporalType, TemporalData, MultiscaleTemporalData are in tulpa:: namespace

using tulpa::math::safe_sqrt;

// -----------------------------------------------------------------------------
// Variance parameterization
// -----------------------------------------------------------------------------

// The precision a variance-parameterized arm evaluates at. The value functions
// below and the analytic gradients in hmc_multiscale_temporal_grad.h both take
// their precision from here, so a value and its gradient describe one density.
template <typename T>
inline T variance_to_precision(const T& sigma2) {
  return T(1.0) / sigma2;
}

// -----------------------------------------------------------------------------
// RW1 log-likelihood (intrinsic first-order random walk)
// -----------------------------------------------------------------------------

// Log-likelihood for RW1: sum of (phi[t] - phi[t-1])^2 / (2*sigma2)
template <typename T>
inline T rw1_log_lik(
    const std::vector<T>& phi,  // Length T
    const T& sigma2,
    bool cyclic = false,
    bool augment = false        // Q -> Q + 11'/n, the sum-to-zero augmentation
) {
  int n = static_cast<int>(phi.size());
  if (n < 2) return T(0.0);

  T quad = rw1_quadratic_form(phi.data(), n, cyclic);
  int rank = tulpa_temporal::rw1_rank(n, cyclic);

  // RW1's null space is the constant alone, so the augmentation fills it and
  // the field becomes full rank. Quadratic and rank move together here rather
  // than the caller adding one and this adding the other.
  if (augment) {
    const T s = tulpa::s2z_component_sum(phi.data(), 0, n);
    quad = quad + tulpa::s2z_aug_coef(T(1.0), n) * s * s;
    rank = tulpa::s2z_aug_rank(rank, 1);
  }

  const T tau = variance_to_precision(sigma2);
  return gmrf_log_norm(rank, safe_log(tau)) - T(0.5) * tau * quad;
}

// -----------------------------------------------------------------------------
// RW2 log-likelihood (intrinsic second-order random walk)
// -----------------------------------------------------------------------------

// Log-likelihood for RW2: sum of (phi[t] - 2*phi[t-1] + phi[t-2])^2 / (2*sigma2)
template <typename T>
inline T rw2_log_lik(
    const std::vector<T>& phi,  // Length T
    const T& sigma2,
    bool cyclic = false,
    bool augment = false        // Q -> Q + 11'/n, the sum-to-zero augmentation
) {
  int n = static_cast<int>(phi.size());
  if (n < 3) return T(0.0);

  T quad = rw2_quadratic_form(phi.data(), n, cyclic);
  int rank = tulpa_temporal::rw2_rank(n, cyclic);

  // A non-cyclic RW2 also carries a LINEAR null direction, which a sum-to-zero
  // augmentation does not touch, so it stays deficient by one: s2z_aug_rank
  // takes rank(Q) and the number of directions actually filled rather than
  // assuming the field length.
  if (augment) {
    const T s = tulpa::s2z_component_sum(phi.data(), 0, n);
    quad = quad + tulpa::s2z_aug_coef(T(1.0), n) * s * s;
    rank = tulpa::s2z_aug_rank(rank, 1);
  }

  const T tau = variance_to_precision(sigma2);
  return gmrf_log_norm(rank, safe_log(tau)) - T(0.5) * tau * quad;
}

// -----------------------------------------------------------------------------
// Non-centred intrinsic walk (the multiscale trend)
// -----------------------------------------------------------------------------

// Dimension of an intrinsic walk's null space: the constant for RW1, the
// constant and the linear trend for RW2. Zero for any other type.
inline int rw_walk_order(TemporalType type) {
  if (type == TemporalType::RW1) return 1;
  if (type == TemporalType::RW2) return 2;
  return 0;
}

// The walk f as a function of sampled coordinates u, in the spectral basis of
// its structure matrix R (R = D'D, D the first or second difference):
//
//   f = (RW2) u_1 l  +  sigma sum_j b_j u_{k + j},    b_j = v_j / sqrt(lambda_j),
//
// where (lambda_j, v_j) are the positive eigenpairs of R and l the unit
// centred linear direction (`basis`: n x (n - k) column-major, the b_j;
// `slope`: l, empty for RW1). Returns the log density of u. The coordinates
// u_{k + j} are N(0, 1), so the scale enters the field rather than the
// coordinates' density and the field/scale funnel of the centred density is
// gone. In this basis the prior is diagonal, and so is a likelihood whose
// precision is constant over time, so the conditional posterior of the
// coordinates is close to diagonal too: a diagonal mass matrix reaches it,
// where the walk's own innovations (cumulative sums of u) couple every
// coordinate to every later time point. u_0 stands for the level, which
// temporal_effects_by_time centres out of the linear predictor: it carries a proper
// N(0, 1) and enters no field value. The RW2 slope u_1 is flat, as the linear
// null direction is under the centred density.
//
// This is the centred model exactly. The centred density is augmented on the
// constant (rw1_log_lik / rw2_log_lik with `augment`); integrating the level
// out of it leaves tau^{(n - k)/2} exp(-tau/2 f'Rf) on the centred walk, with
// the slope flat for RW2, and f'Rf = sum_j lambda_j c_j^2 for the walk's
// coordinates c_j = sigma u_{k + j} / sqrt(lambda_j) in the eigenbasis, whose
// Jacobian sigma^{n - k} prod_j lambda_j^{-1/2} turns this into the standard
// normal density of u up to a constant. Every quantity eta reads -- the centred
// walk and sigma -- has the same joint distribution under both.
template <typename T>
inline T rw_walk_noncentered(const std::vector<T>& u, TemporalType type,
                             const std::vector<double>& basis,
                             const std::vector<double>& slope,
                             const T& sigma2, std::vector<T>& f) {
  const int n = static_cast<int>(u.size());
  const int k = rw_walk_order(type);
  f.assign(n, T(0.0));
  if (n == 0 || k == 0) return T(0.0);
  const int m = n - k;
  const T sigma = safe_sqrt(sigma2);

  T log_dens = T(-0.5) * u[0] * u[0];
  std::vector<T> g(n, T(0.0));
  for (int j = 0; j < m; j++) {
    const T z = u[k + j];
    log_dens = log_dens - T(0.5) * z * z;
    const double* bj = basis.data() + static_cast<std::size_t>(j) * n;
    for (int i = 0; i < n; i++) g[i] = g[i] + bj[i] * z;
  }
  for (int i = 0; i < n; i++) f[i] = sigma * g[i];
  if (k == 2) {
    for (int i = 0; i < n; i++) f[i] = f[i] + slope[i] * u[1];
  }
  return log_dens - T(0.5 * kLogTwoPi * (m + 1));
}

// -----------------------------------------------------------------------------
// AR1 log-likelihood (stationary first-order autoregressive)
// -----------------------------------------------------------------------------

// Log-likelihood for AR1: phi[t] = rho * phi[t-1] + epsilon[t], innovation
// variance sigma2. The density is ar1_log_density taken at the corresponding
// precision, so the stationary factor 1 - rho^2 is floored in one place.
template <typename T>
inline T ar1_log_lik(
    const std::vector<T>& phi,  // Length T
    const T& sigma2,            // Innovation variance
    const T& rho                // Autocorrelation (-1 < rho < 1)
) {
  int n = static_cast<int>(phi.size());
  return ar1_log_density(phi.data(), n, rho, variance_to_precision(sigma2));
}

// -----------------------------------------------------------------------------
// IID log-likelihood (independent identically distributed)
// -----------------------------------------------------------------------------

template <typename T>
inline T iid_log_lik(
    const std::vector<T>& phi,  // Length T
    const T& sigma2
) {
  int n = static_cast<int>(phi.size());
  const T tau = variance_to_precision(sigma2);
  return log_prior_temporal(phi.data(), n, TemporalType::IID,
                            tau, T(0.0), false);
}

// -----------------------------------------------------------------------------
// Non-centred short-term arm (AR1 / IID)
// -----------------------------------------------------------------------------

// The short-term field s as a function of sampled coordinates u:
//
//   IID:  s_t = sigma u_t
//   AR1:  s_1 = sigma u_1 / sqrt(1 - rho^2),  s_t = rho s_{t-1} + sigma u_t,
//
// sigma the innovation SD. Returns the log density of u, standard normal in
// every coordinate. This is the centred model exactly: tau Q(s) = sum u_t^2
// for the AR1 quadratic form Q (ar1_cross_form) at tau = 1 / sigma^2, and the
// Jacobian sigma^n / sqrt(1 - rho^2) of u -> s cancels the centred density's
// normalizer, with 1 - rho^2 the same floored ar1_one_minus_rho2 the centred
// density reads. The scale and the correlation enter the field rather than the
// coordinates' density, so the field / scale funnel of the centred density,
// whose curvature in s grows as sigma falls, is gone.
template <typename T>
inline T short_term_noncentered(const std::vector<T>& u, TemporalType type,
                                const T& sigma2, const T& rho,
                                std::vector<T>& s) {
  const int n = static_cast<int>(u.size());
  s.assign(n, T(0.0));
  if (n == 0 || (type != TemporalType::AR1 && type != TemporalType::IID)) {
    return T(0.0);
  }
  const T sigma = safe_sqrt(sigma2);
  T log_dens = T(0.0);
  for (int t = 0; t < n; t++) log_dens = log_dens - T(0.5) * u[t] * u[t];
  if (type == TemporalType::IID) {
    for (int t = 0; t < n; t++) s[t] = sigma * u[t];
  } else {
    s[0] = sigma * u[0] / safe_sqrt(ar1_one_minus_rho2(rho));
    for (int t = 1; t < n; t++) s[t] = rho * s[t - 1] + sigma * u[t];
  }
  return log_dens - T(0.5 * kLogTwoPi * n);
}

// -----------------------------------------------------------------------------
// Multi-scale temporal log-likelihood
// -----------------------------------------------------------------------------

// Combined log-likelihood for trend + seasonal + short-term
template <typename T>
inline T multiscale_temporal_log_lik(
    const std::vector<T>& trend,       // Length n_times (or empty)
    const std::vector<T>& seasonal,    // Length seasonal_period (or empty)
    const std::vector<T>& short_term,  // Length n_times (or empty)
    const T& sigma2_trend,
    const T& sigma2_seasonal,
    const T& sigma2_short,
    const T& rho_short,                // Only used if short_term is AR1
    const MultiscaleTemporalData& temp_data
) {
  T log_lik = T(0.0);

  // Trend and seasonal are both intrinsic and both enter the SAME linear
  // predictor, so each carries a constant null direction that is unidentified
  // against the intercept and against the other component. Both are augmented
  // -- Q -> Q + 11'/n, with temporal_effects_by_time centring
  // each arm before it reaches eta -- so the direction is removed rather than
  // penalised. The short-term arm is proper (AR1/IID), identifies its own
  // level, and is left alone.

  // Trend component
  if (temp_data.trend_type == TemporalType::RW1 && !trend.empty()) {
    log_lik = log_lik + rw1_log_lik(trend, sigma2_trend, false, true);
  } else if (temp_data.trend_type == TemporalType::RW2 && !trend.empty()) {
    log_lik = log_lik + rw2_log_lik(trend, sigma2_trend, false, true);
  }

  // Seasonal component (always cyclic RW1)
  if (temp_data.seasonal_period > 0 && !seasonal.empty()) {
    log_lik = log_lik + rw1_log_lik(seasonal, sigma2_seasonal, true, true);
  }

  // Short-term component
  if (temp_data.short_term_type == TemporalType::AR1 && !short_term.empty()) {
    log_lik = log_lik + ar1_log_lik(short_term, sigma2_short, rho_short);
  } else if (temp_data.short_term_type == TemporalType::IID && !short_term.empty()) {
    log_lik = log_lik + iid_log_lik(short_term, sigma2_short);
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// Total temporal effect at each time point, and at each observation
// -----------------------------------------------------------------------------

// effect[t] = trend[t] + seasonal[t % period] + short[t], each intrinsic arm
// centred, for t = 0 .. n_times - 1. This is the field the linear predictor
// reads: an observation's effect is the effect at its time
// (`gather_temporal_eta()`), and a prediction at a fitted time reads the same
// value.
template <typename T>
inline void temporal_effects_by_time(
    const std::vector<T>& trend,
    const std::vector<T>& seasonal,
    const std::vector<T>& short_term,
    const MultiscaleTemporalData& temp_data,
    std::vector<T>& effect  // Output: length n_times
) {
  const int n_t = temp_data.n_times;
  effect.assign(n_t > 0 ? n_t : 0, T(0.0));

  // The intrinsic arms are centred on their way in. Their augmented prior gives
  // each constant direction the arm's own precision (order 1) rather than the
  // stiff pin that preceded it, so leaving the constant in eta would free the
  // level instead of fixing it -- and trend and seasonal land on the same
  // linear predictor, so their constants are unidentified against each other as
  // well as against the intercept. The short-term arm is proper and keeps its
  // own level.
  const T trend_mean =
      trend.empty() ? T(0.0)
                    : tulpa::s2z_component_mean(trend.data(), 0,
                                                static_cast<int>(trend.size()));
  const T seasonal_mean =
      seasonal.empty() ? T(0.0)
                       : tulpa::s2z_component_mean(
                             seasonal.data(), 0,
                             static_cast<int>(seasonal.size()));

  for (int t_idx = 0; t_idx < n_t; t_idx++) {
    T e = T(0.0);

    // Trend contribution
    if (t_idx < static_cast<int>(trend.size())) {
      e = e + (trend[t_idx] - trend_mean);
    }

    // Seasonal contribution (wrap around using modulo)
    if (temp_data.seasonal_period > 0 && !seasonal.empty()) {
      int s_idx = t_idx % temp_data.seasonal_period;
      if (s_idx < static_cast<int>(seasonal.size())) {
        e = e + (seasonal[s_idx] - seasonal_mean);
      }
    }

    // Short-term contribution
    if (t_idx < static_cast<int>(short_term.size())) {
      e = e + short_term[t_idx];
    }

    effect[t_idx] = e;
  }
}

// eta_temporal[i] = effect[time_idx[i] - 1]; an index outside the time points
// contributes nothing.
template <typename T>
inline void gather_temporal_eta(
    const std::vector<T>& effect,
    const MultiscaleTemporalData& temp_data,
    std::vector<T>& eta_temporal  // Output: length n_obs
) {
  int N = temp_data.n_obs;
  // n_obs and time_index are separate fields of MultiscaleTemporalData, which
  // ships under inst/include: a LinkingTo caller can declare more observations
  // than it supplied indices for, and the loop below reads time_index[i] for
  // every one of them.
  if (temp_data.time_index.size() < static_cast<std::size_t>(N)) {
    throw std::invalid_argument(
        "tulpa: MultiscaleTemporalData.time_index is shorter than n_obs.");
  }
  eta_temporal.resize(N);
  const int n_t = static_cast<int>(effect.size());
  for (int i = 0; i < N; i++) {
    const int t_idx = temp_data.time_index[i] - 1;  // Convert to 0-based
    eta_temporal[i] = (t_idx >= 0 && t_idx < n_t) ? effect[t_idx] : T(0.0);
  }
}

// -----------------------------------------------------------------------------
// Priors for temporal hyperparameters
// -----------------------------------------------------------------------------

// PC prior for temporal variance (favor simpler models with smaller variance)
template <typename T>
inline T log_prior_sigma2_temporal_pc(const T& sigma2, double U, double alpha) {
  return tulpa::log_prior_sigma2_pc(sigma2, U, alpha);
}

// Parse temporal type from string
inline TemporalType parse_temporal_type(const std::string& type_str) {
  static const tulpa::EnumEntry<TemporalType> table[] = {
    {"rw1", TemporalType::RW1}, {"rw2", TemporalType::RW2},
    {"ar1", TemporalType::AR1}, {"iid", TemporalType::IID},
    {"none", TemporalType::NONE}
  };
  return tulpa::parse_enum(type_str, table, TemporalType::NONE);
}

} // namespace tulpa_temporal

#endif // TULPA_HMC_TEMPORAL_MULTISCALE_H
