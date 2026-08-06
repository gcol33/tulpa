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
#include "tulpa/soft_sum_to_zero.h"  // s2z_precision
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

// Apply soft sum-to-zero constraint to TVC (for each term and group). Each
// pinned sum runs over the n_times coefficients of one (group, term), so the
// precision is s2z_precision(n_times); it is derived here rather than taken
// from the caller so no call site can pass a kappa where a precision is meant.
template <typename T>
inline T tvc_sum_to_zero_penalty(
    const std::vector<T>& w_flat,
    const TVCData& tvc_data
) {
  int n_times = tvc_data.n_times;
  int n_tvc = tvc_data.n_tvc;
  int n_groups = tvc_data.n_groups;

  const double lambda = tulpa::s2z_precision(n_times);
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

} // namespace tulpa_tvc

#endif // TULPA_HMC_TVC_H
