// hmc_temporal_multiscale.h
// Multi-scale temporal decomposition: trend + seasonal + short-term
// Builds on existing RW1/RW2/AR1 infrastructure
//
// Templated over the scalar type: double for evaluation, the autodiff
// types (ad::Var, fwd::Dual, arena::Var) for gradient modes.

#ifndef TULPA_HMC_TEMPORAL_MULTISCALE_H
#define TULPA_HMC_TEMPORAL_MULTISCALE_H

#include <vector>
#include <cmath>
#include <string>
#include "hmc_temporal.h"  // For temporal kernels (RW1, RW2, AR1)
#include "pc_prior.h"
#include "tulpa/sum_to_zero.h"  // s2z_aug_coef / s2z_aug_rank / component sums

namespace tulpa_temporal {

// Types imported via hmc_temporal.h -> tulpa/temporal_data.h
// TemporalType, TemporalData, MultiscaleTemporalData are in tulpa:: namespace

using tulpa::math::safe_sqrt;

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

  // Normalizing constant (improper prior, omit for sampling)
  return T(-0.5) * quad / sigma2
       - T(0.5 * rank) * safe_log(T(2.0 * M_PI) * sigma2);
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

  // Normalizing constant
  return T(-0.5) * quad / sigma2
       - T(0.5 * rank) * safe_log(T(2.0 * M_PI) * sigma2);
}

// -----------------------------------------------------------------------------
// AR1 log-likelihood (stationary first-order autoregressive)
// -----------------------------------------------------------------------------

// Log-likelihood for AR1: phi[t] = rho * phi[t-1] + epsilon[t]
// Innovation-variance parameterization (cf. ar1_log_density, which takes
// the precision); the stationary variance is regularized so the density
// stays finite as |rho| -> 1.
template <typename T>
inline T ar1_log_lik(
    const std::vector<T>& phi,  // Length T
    const T& sigma2,            // Innovation variance
    const T& rho                // Autocorrelation (-1 < rho < 1)
) {
  int n = static_cast<int>(phi.size());
  if (n < 2) return T(0.0);

  T log_lik = T(0.0);

  // Marginal distribution of first observation
  T one_minus_rho2 = T(1.0) - rho * rho;
  T marginal_var = sigma2 / (one_minus_rho2 + T(1e-10));
  log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * marginal_var);
  log_lik = log_lik - T(0.5) * phi[0] * phi[0] / marginal_var;

  // Conditional distributions
  for (int t = 1; t < n; t++) {
    T resid = phi[t] - rho * phi[t - 1];
    log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2);
    log_lik = log_lik - T(0.5) * resid * resid / sigma2;
  }

  return log_lik;
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
  T log_lik = T(0.0);

  for (int t = 0; t < n; t++) {
    log_lik = log_lik - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2);
    log_lik = log_lik - T(0.5) * phi[t] * phi[t] / sigma2;
  }

  return log_lik;
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
  // (gcol33/tulpa#241) -- Q -> Q + 11'/n, with compute_temporal_eta centring
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
// Compute total temporal effect at each observation
// -----------------------------------------------------------------------------

// eta_temporal[i] = trend[time_idx[i]] + seasonal[time_idx[i] % period] + short[time_idx[i]]
template <typename T>
inline void compute_temporal_eta(
    const std::vector<T>& trend,
    const std::vector<T>& seasonal,
    const std::vector<T>& short_term,
    const MultiscaleTemporalData& temp_data,
    std::vector<T>& eta_temporal  // Output: length n_obs
) {
  int N = temp_data.n_obs;
  eta_temporal.resize(N);

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

  for (int i = 0; i < N; i++) {
    T effect = T(0.0);
    int t_idx = temp_data.time_index[i] - 1;  // Convert to 0-based

    // Trend contribution
    if (!trend.empty() && t_idx >= 0 &&
        t_idx < static_cast<int>(trend.size())) {
      effect = effect + (trend[t_idx] - trend_mean);
    }

    // Seasonal contribution (wrap around using modulo)
    if (temp_data.seasonal_period > 0 && !seasonal.empty()) {
      int s_idx = t_idx % temp_data.seasonal_period;
      if (s_idx >= 0 && s_idx < static_cast<int>(seasonal.size())) {
        effect = effect + (seasonal[s_idx] - seasonal_mean);
      }
    }

    // Short-term contribution
    if (!short_term.empty() && t_idx >= 0 &&
        t_idx < static_cast<int>(short_term.size())) {
      effect = effect + short_term[t_idx];
    }

    eta_temporal[i] = effect;
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

// Prior for AR1 rho: Beta(a, b) on (rho + 1) / 2
template <typename T>
inline T log_prior_rho(const T& rho, double a = 2.0, double b = 2.0) {
  // Transform to [0, 1]
  T x = (rho + T(1.0)) / T(2.0);

  // Beta log density (unnormalized)
  return T(a - 1.0) * safe_log(x) + T(b - 1.0) * safe_log(T(1.0) - x);
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
