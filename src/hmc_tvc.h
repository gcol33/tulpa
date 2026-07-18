// hmc_tvc.h
// Temporally-Varying Coefficients (TVC) for HMC backend
// Supports RW1, RW2, AR1, and GP temporal structures for coefficients

#ifndef TULPA_HMC_TVC_H
#define TULPA_HMC_TVC_H

#include <vector>
#include <cmath>
#include "hmc_temporal.h"  // Reuse RW1/RW2/AR1 implementations
#include "pc_prior.h"      // single-source PC prior on every sampled scale

// Use canonical type definitions from exported headers
#include "tulpa/tvc_data.h"
#include "tulpa/types.h"

namespace tulpa_tvc {

using tulpa::TemporalType;
using tulpa::TVCData;
using tulpa_temporal::rw1_quadratic_form;
using tulpa_temporal::rw2_quadratic_form;
using tulpa_temporal::ar1_log_density;
using tulpa::math::safe_log;

// -----------------------------------------------------------------------------
// TVC log-prior
// -----------------------------------------------------------------------------

// Compute log-prior for a single TVC term's temporal trajectory.
// Templated over the scalar type (double for evaluation, autodiff types
// for gradients).
// w: temporal trajectory (length n_times)
// tau: precision parameter
// rho: AR1 correlation (only used if structure == AR1)
template <typename T>
inline T tvc_term_log_prior(
    const T* w,
    int n_times,
    TemporalType structure,
    const T& tau,
    const T& rho,
    bool cyclic = false
) {
  T log_prior = T(0.0);

  if (structure == TemporalType::RW1) {
    T quad = rw1_quadratic_form(w, n_times, cyclic);
    int rank = tulpa_temporal::rw1_rank(n_times, cyclic);
    log_prior = log_prior + T(0.5 * rank) * safe_log(tau);
    log_prior = log_prior - T(0.5) * tau * quad;

  } else if (structure == TemporalType::RW2) {
    T quad = rw2_quadratic_form(w, n_times, cyclic);
    int rank = tulpa_temporal::rw2_rank(n_times, cyclic);
    log_prior = log_prior + T(0.5 * rank) * safe_log(tau);
    log_prior = log_prior - T(0.5) * tau * quad;

  } else if (structure == TemporalType::AR1) {
    log_prior = log_prior + ar1_log_density(w, n_times, rho, tau, 1e-10);

  } else if (structure == TemporalType::IID) {
    // IID: independent N(0, 1/tau) for each time point
    log_prior = log_prior + T(0.5 * n_times) * safe_log(tau);
    T quad = T(0.0);
    for (int t = 0; t < n_times; t++) {
      quad = quad + w[t] * w[t];
    }
    log_prior = log_prior - T(0.5) * tau * quad;
  }

  return log_prior;
}

// Compute log-prior for all TVC terms
// w_flat: all TVC values (n_groups * n_tvc * n_times, flattened)
// tau: vector of precisions (length n_tvc)
// rho: vector of AR1 correlations (length n_tvc, only for AR1)
template <typename T>
inline T tvc_log_prior(
    const std::vector<T>& w_flat,
    const TVCData& tvc_data,
    const std::vector<T>& tau,
    const std::vector<T>& rho
) {
  int n_times = tvc_data.n_times;
  int n_tvc = tvc_data.n_tvc;
  int n_groups = tvc_data.n_groups;

  T log_prior = T(0.0);

  // Layout: w_flat[(g * n_tvc + j) * n_times + t]
  for (int g = 0; g < n_groups; g++) {
    for (int j = 0; j < n_tvc; j++) {
      const T* w_jg = w_flat.data() + (g * n_tvc + j) * n_times;
      T rho_j = (tvc_data.structure == TemporalType::AR1) ? rho[j] : T(0.0);
      log_prior = log_prior + tvc_term_log_prior(
          w_jg, n_times, tvc_data.structure, tau[j], rho_j, tvc_data.cyclic);
    }
  }

  return log_prior;
}

// -----------------------------------------------------------------------------
// TVC contribution to linear predictor
// -----------------------------------------------------------------------------

// Compute TVC contribution to linear predictor for all observations
// eta_tvc[i] = sum_j X_tvc[i,j] * w[group_index[i], j, time_index[i]]
template <typename T>
inline void compute_tvc_eta(
    const std::vector<T>& w_flat,  // n_groups * n_tvc * n_times
    const TVCData& tvc_data,
    std::vector<T>& eta_tvc         // Output: length n_obs
) {
  int N = tvc_data.n_obs;
  int n_times = tvc_data.n_times;
  int n_tvc = tvc_data.n_tvc;

  eta_tvc.assign(N, T(0.0));

  for (int i = 0; i < N; i++) {
    int t = tvc_data.time_index[i] - 1;  // 0-based
    int g = tvc_data.group_index[i] - 1;  // 0-based

    for (int j = 0; j < n_tvc; j++) {
      // w_flat layout: [(g * n_tvc + j) * n_times + t]
      T w_jgt = w_flat[(g * n_tvc + j) * n_times + t];
      double x_ij = tvc_data.X_tvc[i * n_tvc + j];
      eta_tvc[i] = eta_tvc[i] + T(x_ij) * w_jgt;
    }
  }
}

// -----------------------------------------------------------------------------
// Sum-to-zero constraint for identifiability
// -----------------------------------------------------------------------------

// Apply soft sum-to-zero constraint to TVC (for each term and group)
template <typename T>
inline T tvc_sum_to_zero_penalty(
    const std::vector<T>& w_flat,
    const TVCData& tvc_data,
    double lambda = 0.001
) {
  int n_times = tvc_data.n_times;
  int n_tvc = tvc_data.n_tvc;
  int n_groups = tvc_data.n_groups;

  T penalty = T(0.0);

  for (int g = 0; g < n_groups; g++) {
    for (int j = 0; j < n_tvc; j++) {
      T sum = T(0.0);
      for (int t = 0; t < n_times; t++) {
        sum = sum + w_flat[(g * n_tvc + j) * n_times + t];
      }
      penalty = penalty - T(0.5 * lambda) * sum * sum;
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
  return tulpa::log_prior_tau_pc<double>(tau, U, alpha);
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
