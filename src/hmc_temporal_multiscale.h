// hmc_temporal_multiscale.h
// Multi-scale temporal decomposition: trend + seasonal + short-term
// Builds on existing RW1/RW2/AR1 infrastructure

#ifndef TULPA_HMC_TEMPORAL_MULTISCALE_H
#define TULPA_HMC_TEMPORAL_MULTISCALE_H

#include <vector>
#include <cmath>
#include <string>
#include "hmc_temporal.h"  // For temporal functions (RW1, RW2, AR1)
#include "pc_prior.h"

namespace tulpa_temporal {

// Types imported via hmc_temporal.h -> tulpa/temporal_data.h
// TemporalType, TemporalData, MultiscaleTemporalData are in tulpa:: namespace

// -----------------------------------------------------------------------------
// RW1 log-likelihood (intrinsic first-order random walk)
// -----------------------------------------------------------------------------

// Log-likelihood for RW1: sum of (phi[t] - phi[t-1])^2 / (2*sigma2)
inline double rw1_log_lik(
    const std::vector<double>& phi,  // Length T
    double sigma2,
    bool cyclic = false
) {
  int T = static_cast<int>(phi.size());
  if (T < 2) return 0.0;

  double log_lik = 0.0;

  // First differences
  for (int t = 1; t < T; t++) {
    double diff = phi[t] - phi[t - 1];
    log_lik += -0.5 * diff * diff / sigma2;
  }

  // Cyclic: add connection from last to first
  if (cyclic) {
    double diff = phi[0] - phi[T - 1];
    log_lik += -0.5 * diff * diff / sigma2;
  }

  // Normalizing constant (improper prior, omit for sampling)
  int n_diffs = cyclic ? T : (T - 1);
  log_lik += -0.5 * n_diffs * std::log(2.0 * M_PI * sigma2);

  return log_lik;
}

// -----------------------------------------------------------------------------
// RW2 log-likelihood (intrinsic second-order random walk)
// -----------------------------------------------------------------------------

// Log-likelihood for RW2: sum of (phi[t] - 2*phi[t-1] + phi[t-2])^2 / (2*sigma2)
inline double rw2_log_lik(
    const std::vector<double>& phi,  // Length T
    double sigma2,
    bool cyclic = false
) {
  int T = static_cast<int>(phi.size());
  if (T < 3) return 0.0;

  double log_lik = 0.0;

  // Second differences
  for (int t = 2; t < T; t++) {
    double diff2 = phi[t] - 2.0 * phi[t - 1] + phi[t - 2];
    log_lik += -0.5 * diff2 * diff2 / sigma2;
  }

  // Cyclic: add wrap-around connections
  if (cyclic) {
    double diff2_1 = phi[0] - 2.0 * phi[T - 1] + phi[T - 2];
    double diff2_2 = phi[1] - 2.0 * phi[0] + phi[T - 1];
    log_lik += -0.5 * diff2_1 * diff2_1 / sigma2;
    log_lik += -0.5 * diff2_2 * diff2_2 / sigma2;
  }

  // Normalizing constant
  int n_diffs = cyclic ? T : (T - 2);
  log_lik += -0.5 * n_diffs * std::log(2.0 * M_PI * sigma2);

  return log_lik;
}

// -----------------------------------------------------------------------------
// AR1 log-likelihood (stationary first-order autoregressive)
// -----------------------------------------------------------------------------

// Log-likelihood for AR1: phi[t] = rho * phi[t-1] + epsilon[t]
inline double ar1_log_lik(
    const std::vector<double>& phi,  // Length T
    double sigma2,                   // Innovation variance
    double rho                       // Autocorrelation (-1 < rho < 1)
) {
  int T = static_cast<int>(phi.size());
  if (T < 2) return 0.0;

  double log_lik = 0.0;

  // Marginal distribution of first observation
  double marginal_var = sigma2 / (1.0 - rho * rho);
  log_lik += -0.5 * std::log(2.0 * M_PI * marginal_var) -
             0.5 * phi[0] * phi[0] / marginal_var;

  // Conditional distributions
  for (int t = 1; t < T; t++) {
    double resid = phi[t] - rho * phi[t - 1];
    log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
               0.5 * resid * resid / sigma2;
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// IID log-likelihood (independent identically distributed)
// -----------------------------------------------------------------------------

inline double iid_log_lik(
    const std::vector<double>& phi,  // Length T
    double sigma2
) {
  int T = static_cast<int>(phi.size());
  double log_lik = 0.0;

  for (int t = 0; t < T; t++) {
    log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
               0.5 * phi[t] * phi[t] / sigma2;
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// Multi-scale temporal log-likelihood
// -----------------------------------------------------------------------------

// Combined log-likelihood for trend + seasonal + short-term
inline double multiscale_temporal_log_lik(
    const std::vector<double>& trend,       // Length n_times (or empty)
    const std::vector<double>& seasonal,    // Length seasonal_period (or empty)
    const std::vector<double>& short_term,  // Length n_times (or empty)
    double sigma2_trend,
    double sigma2_seasonal,
    double sigma2_short,
    double rho_short,                       // Only used if short_term is AR1
    const MultiscaleTemporalData& temp_data
) {
  double log_lik = 0.0;

  // Trend component
  if (temp_data.trend_type == TemporalType::RW1 && !trend.empty()) {
    log_lik += rw1_log_lik(trend, sigma2_trend, false);
  } else if (temp_data.trend_type == TemporalType::RW2 && !trend.empty()) {
    log_lik += rw2_log_lik(trend, sigma2_trend, false);
  }

  // Seasonal component (always cyclic RW1)
  if (temp_data.seasonal_period > 0 && !seasonal.empty()) {
    log_lik += rw1_log_lik(seasonal, sigma2_seasonal, true);  // Cyclic
  }

  // Short-term component
  if (temp_data.short_term_type == TemporalType::AR1 && !short_term.empty()) {
    log_lik += ar1_log_lik(short_term, sigma2_short, rho_short);
  } else if (temp_data.short_term_type == TemporalType::IID && !short_term.empty()) {
    log_lik += iid_log_lik(short_term, sigma2_short);
  }

  return log_lik;
}

// -----------------------------------------------------------------------------
// Compute total temporal effect at each observation
// -----------------------------------------------------------------------------

// eta_temporal[i] = trend[time_idx[i]] + seasonal[time_idx[i] % period] + short[time_idx[i]]
inline void compute_temporal_eta(
    const std::vector<double>& trend,
    const std::vector<double>& seasonal,
    const std::vector<double>& short_term,
    const MultiscaleTemporalData& temp_data,
    std::vector<double>& eta_temporal  // Output: length n_obs
) {
  int N = temp_data.n_obs;
  eta_temporal.resize(N);

  for (int i = 0; i < N; i++) {
    double effect = 0.0;
    int t_idx = temp_data.time_index[i] - 1;  // Convert to 0-based

    // Trend contribution
    if (!trend.empty() && t_idx < static_cast<int>(trend.size())) {
      effect += trend[t_idx];
    }

    // Seasonal contribution (wrap around using modulo)
    if (temp_data.seasonal_period > 0 && !seasonal.empty()) {
      int s_idx = t_idx % temp_data.seasonal_period;
      if (s_idx < static_cast<int>(seasonal.size())) {
        effect += seasonal[s_idx];
      }
    }

    // Short-term contribution
    if (!short_term.empty() && t_idx < static_cast<int>(short_term.size())) {
      effect += short_term[t_idx];
    }

    eta_temporal[i] = effect;
  }
}

// -----------------------------------------------------------------------------
// Priors for temporal hyperparameters
// -----------------------------------------------------------------------------

// PC prior for temporal variance (favor simpler models with smaller variance)
inline double log_prior_sigma2_temporal_pc(double sigma2, double U, double alpha) {
  return tulpa::log_prior_sigma2_pc(sigma2, U, alpha);
}

// Prior for AR1 rho: Beta(a, b) on (rho + 1) / 2
inline double log_prior_rho(double rho, double a = 2.0, double b = 2.0) {
  if (rho <= -1.0 || rho >= 1.0) return -INFINITY;

  // Transform to [0, 1]
  double x = (rho + 1.0) / 2.0;

  // Beta log density (unnormalized)
  return (a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0 - x);
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
