// hmc_tvc.h
// Temporally-Varying Coefficients (TVC) for HMC backend
// Supports RW1, RW2, AR1, and GP temporal structures for coefficients

#ifndef TULPA_HMC_TVC_H
#define TULPA_HMC_TVC_H

#include <vector>
#include <cmath>
#include "hmc_temporal.h"  // Reuse RW1/RW2/AR1 implementations

// Use canonical type definitions from exported headers
#include "tulpa/tvc_data.h"
#include "tulpa/types.h"

namespace tulpa_tvc {

using tulpa::TemporalType;
using tulpa::TVCData;
using tulpa_temporal::rw1_quadratic_form;
using tulpa_temporal::rw2_quadratic_form;
using tulpa_temporal::ar1_log_density;

// -----------------------------------------------------------------------------
// TVC log-prior
// -----------------------------------------------------------------------------

// Compute log-prior for a single TVC term's temporal trajectory
// w: temporal trajectory (length n_times)
// tau: precision parameter
// rho: AR1 correlation (only used if structure == AR1)
inline double tvc_term_log_prior(
    const double* w,
    int n_times,
    TemporalType structure,
    double tau,
    double rho = 0.0
) {
  double log_prior = 0.0;

  if (structure == TemporalType::RW1) {
    double quad = rw1_quadratic_form(w, n_times, false);
    int rank = n_times - 1;
    log_prior += 0.5 * rank * std::log(tau);
    log_prior -= 0.5 * tau * quad;

  } else if (structure == TemporalType::RW2) {
    double quad = rw2_quadratic_form(w, n_times, false);
    int rank = n_times - 2;
    log_prior += 0.5 * rank * std::log(tau);
    log_prior -= 0.5 * tau * quad;

  } else if (structure == TemporalType::AR1) {
    log_prior += ar1_log_density(w, n_times, rho, tau);

  } else if (structure == TemporalType::IID) {
    // IID: independent N(0, 1/tau) for each time point
    log_prior += 0.5 * n_times * std::log(tau);
    double quad = 0.0;
    for (int t = 0; t < n_times; t++) {
      quad += w[t] * w[t];
    }
    log_prior -= 0.5 * tau * quad;
  }

  return log_prior;
}

// Compute log-prior for all TVC terms
// w_flat: all TVC values (n_times * n_tvc * n_groups, flattened)
// tau: vector of precisions (length n_tvc)
// rho: vector of AR1 correlations (length n_tvc, only for AR1)
inline double tvc_log_prior(
    const std::vector<double>& w_flat,
    const TVCData& tvc_data,
    const std::vector<double>& tau,
    const std::vector<double>& rho
) {
  int n_times = tvc_data.n_times;
  int n_tvc = tvc_data.n_tvc;
  int n_groups = tvc_data.n_groups;

  double log_prior = 0.0;

  // Layout: w_flat[g * n_tvc * n_times + j * n_times + t]
  for (int g = 0; g < n_groups; g++) {
    for (int j = 0; j < n_tvc; j++) {
      const double* w_jg = &w_flat[(g * n_tvc + j) * n_times];
      double rho_j = (tvc_data.structure == TemporalType::AR1) ? rho[j] : 0.0;
      log_prior += tvc_term_log_prior(w_jg, n_times, tvc_data.structure,
                                       tau[j], rho_j);
    }
  }

  return log_prior;
}

// -----------------------------------------------------------------------------
// TVC contribution to linear predictor
// -----------------------------------------------------------------------------

// Compute TVC contribution to linear predictor for all observations
// eta_tvc[i] = sum_j X_tvc[i,j] * w[time_index[i], j, group_index[i]]
inline void compute_tvc_eta(
    const std::vector<double>& w_flat,  // n_groups * n_tvc * n_times
    const TVCData& tvc_data,
    std::vector<double>& eta_tvc         // Output: length n_obs
) {
  int N = tvc_data.n_obs;
  int n_times = tvc_data.n_times;
  int n_tvc = tvc_data.n_tvc;

  std::fill(eta_tvc.begin(), eta_tvc.end(), 0.0);

  for (int i = 0; i < N; i++) {
    int t = tvc_data.time_index[i] - 1;  // 0-based
    int g = tvc_data.group_index[i] - 1;  // 0-based

    for (int j = 0; j < n_tvc; j++) {
      // w_flat layout: [g * n_tvc * n_times + j * n_times + t]
      double w_ijg = w_flat[(g * n_tvc + j) * n_times + t];
      double x_ij = tvc_data.X_tvc[i * n_tvc + j];
      eta_tvc[i] += x_ij * w_ijg;
    }
  }
}

// -----------------------------------------------------------------------------
// Sum-to-zero constraint for identifiability
// -----------------------------------------------------------------------------

// Apply soft sum-to-zero constraint to TVC (for each term and group)
inline double tvc_sum_to_zero_penalty(
    const std::vector<double>& w_flat,
    const TVCData& tvc_data,
    double lambda = 0.001
) {
  int n_times = tvc_data.n_times;
  int n_tvc = tvc_data.n_tvc;
  int n_groups = tvc_data.n_groups;

  double penalty = 0.0;

  for (int g = 0; g < n_groups; g++) {
    for (int j = 0; j < n_tvc; j++) {
      double sum = 0.0;
      for (int t = 0; t < n_times; t++) {
        sum += w_flat[(g * n_tvc + j) * n_times + t];
      }
      penalty -= 0.5 * lambda * sum * sum;
    }
  }

  return penalty;
}

// -----------------------------------------------------------------------------
// Prior on TVC hyperparameters
// -----------------------------------------------------------------------------

// Log prior for TVC precision (PC prior style)
// Favors smaller variance = simpler (more constant) coefficients
inline double log_prior_tau_pc(double tau, double U, double alpha) {
  // sigma ~ Exponential(rate = -log(alpha)/U)
  // tau = 1/sigma^2, apply Jacobian
  double sigma = 1.0 / std::sqrt(tau);
  double rate = -std::log(alpha) / U;
  return std::log(rate) - rate * sigma - std::log(2.0 * sigma) - 2.0 * std::log(tau);
}

// Log prior for AR1 correlation
// Uniform on (-1, 1)
inline double log_prior_rho_uniform(double rho) {
  if (rho <= -1.0 || rho >= 1.0) return -INFINITY;
  return -std::log(2.0);  // Uniform(-1, 1)
}

// Beta prior on (rho + 1) / 2
inline double log_prior_rho_beta(double rho, double a, double b) {
  if (rho <= -1.0 || rho >= 1.0) return -INFINITY;
  double u = (rho + 1.0) / 2.0;  // Transform to (0, 1)
  // Beta(a, b) density: u^{a-1} * (1-u)^{b-1} / B(a, b)
  return (a - 1.0) * std::log(u) + (b - 1.0) * std::log(1.0 - u) -
         std::lgamma(a) - std::lgamma(b) + std::lgamma(a + b) -
         std::log(2.0);  // Jacobian for rho -> u
}

// -----------------------------------------------------------------------------
// Gradient helpers (for HMC)
// -----------------------------------------------------------------------------
//
// The RW1 log-prior gradient lives in hmc_tvc_grad.h as
// tulpa_tvc::rw1_grad_w. A previous duplicate in this file used the
// opposite sign convention (gradient of -log_prior) and was never
// actually called — removed to avoid divergence.

// Gradient of RW2 log-prior w.r.t. w
inline void rw2_gradient(
    const double* w,
    int n_times,
    double tau,
    double* grad_w
) {
  // Second difference: d_t = w_t - 2*w_{t+1} + w_{t+2}
  // Quadratic form: sum(d_t^2)
  // Gradient is more complex, compute numerically if needed

  // For simplicity, use finite differences
  std::vector<double> w_copy(w, w + n_times);
  double eps = 1e-6;

  // Base quadratic form
  double base_quad = rw2_quadratic_form(w, n_times, false);

  for (int t = 0; t < n_times; t++) {
    w_copy[t] = w[t] + eps;
    double quad_plus = rw2_quadratic_form(w_copy.data(), n_times, false);
    grad_w[t] = -tau * (quad_plus - base_quad) / eps;
    w_copy[t] = w[t];
  }
}

// Parse temporal structure type from string
inline TemporalType parse_tvc_structure(const std::string& struct_str) {
  static const tulpa::EnumEntry<TemporalType> table[] = {
      {"rw1", TemporalType::RW1},
      {"rw2", TemporalType::RW2},
      {"ar1", TemporalType::AR1},
      {"iid", TemporalType::IID}
  };
  return tulpa::parse_enum(struct_str, table, TemporalType::RW1);
}

} // namespace tulpa_tvc

#endif // TULPA_HMC_TVC_H
