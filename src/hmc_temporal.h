// hmc_temporal.h
// Temporal random effects support for HMC backend
// Supports RW1, RW2, and AR1 temporal structures

#ifndef TULPA_HMC_TEMPORAL_H
#define TULPA_HMC_TEMPORAL_H

#define _USE_MATH_DEFINES  // For M_PI on Windows
#include <vector>
#include <cmath>
#include <utility>

// Fallback definition of M_PI if not provided by <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Use canonical type definitions from exported headers
#include "tulpa/temporal_data.h"
#include "tulpa/types.h"
#include "autodiff_utils.h"  // safe_log for the templated kernels

namespace tulpa_temporal {

using tulpa::TemporalType;
using tulpa::TemporalData;
using tulpa::MultiscaleTemporalData;
using tulpa::math::safe_log;

// =====================================================================
// RW1 / RW2 quadratic forms and AR1 log-density (single source)
// =====================================================================
//
// Templated over the scalar type: double for the plain sampler, the
// autodiff types (ad::Var, fwd::Dual, arena::Var) for gradient modes.
// Pointer-based so strided per-group callers pass `phi + g * T_len`.

// Sum of products of first differences of two series. With a == b this
// is the acyclic part of phi' Q_RW1 phi; with a != b it is the
// off-diagonal temporal term of a Kronecker (Q_s (x) Q_t) prior.
template <typename T>
inline T rw1_cross_form(const T* a, const T* b, int T_len) {
  T quad = T(0.0);
  for (int t = 1; t < T_len; t++) {
    quad = quad + (a[t] - a[t - 1]) * (b[t] - b[t - 1]);
  }
  return quad;
}

// Compute phi' Q_RW1 phi for RW1 prior
// Q_RW1 is the first-order random walk precision matrix
template <typename T>
inline T rw1_quadratic_form(
    const T* phi,
    int T_len,
    bool cyclic
) {
  T quad = rw1_cross_form(phi, phi, T_len);

  // Cyclic: add edge from T to 1
  if (cyclic && T_len > 1) {
    T diff = phi[0] - phi[T_len - 1];
    quad = quad + diff * diff;
  }

  return quad;
}

// Sum of products of second differences of two series (acyclic part of
// the RW2 quadratic / Kronecker cross term).
template <typename T>
inline T rw2_cross_form(const T* a, const T* b, int T_len) {
  T quad = T(0.0);
  for (int t = 2; t < T_len; t++) {
    T d2_a = a[t] - T(2.0) * a[t - 1] + a[t - 2];
    T d2_b = b[t] - T(2.0) * b[t - 1] + b[t - 2];
    quad = quad + d2_a * d2_b;
  }
  return quad;
}

// Compute phi' Q_RW2 phi for RW2 prior
// Q_RW2 penalizes second differences (curvature)
template <typename T>
inline T rw2_quadratic_form(
    const T* phi,
    int T_len,
    bool cyclic
) {
  if (T_len < 3) return T(0.0);

  T quad = rw2_cross_form(phi, phi, T_len);

  // Cyclic: wrap around
  if (cyclic) {
    T diff2_1 = phi[T_len - 2] - T(2.0) * phi[T_len - 1] + phi[0];
    quad = quad + diff2_1 * diff2_1;
    T diff2_2 = phi[T_len - 1] - T(2.0) * phi[0] + phi[1];
    quad = quad + diff2_2 * diff2_2;
  }

  return quad;
}

// Compute log-density for AR1 process
// phi[t] | phi[t-1] ~ N(rho * phi[t-1], sigma^2)
// Marginal variance: sigma^2 / (1 - rho^2)
// `stationary_eps` regularizes the stationary-precision denominator
// tau * (1 - rho^2): 0 keeps the exact density, a small positive value
// keeps it (and its gradient) finite as |rho| -> 1.
template <typename T>
inline T ar1_log_density(
    const T* phi,
    int T_len,
    const T& rho,
    const T& tau,  // precision = 1/sigma^2
    double stationary_eps = 0.0
) {
  if (T_len < 2) return T(0.0);

  T log_dens = T(0.0);

  // First observation: phi[0] ~ N(0, sigma^2 / (1 - rho^2))
  T marginal_var = T(1.0) / (tau * (T(1.0) - rho * rho) + T(stationary_eps));
  log_dens = log_dens - T(0.5) * phi[0] * phi[0] / marginal_var;
  log_dens = log_dens - T(0.5) * safe_log(T(2.0 * M_PI) * marginal_var);

  // Conditional: phi[t] | phi[t-1] ~ N(rho * phi[t-1], sigma^2)
  T sigma2 = T(1.0) / tau;
  for (int t = 1; t < T_len; t++) {
    T resid = phi[t] - rho * phi[t - 1];
    log_dens = log_dens - T(0.5) * resid * resid / sigma2;
    log_dens = log_dens - T(0.5) * safe_log(T(2.0 * M_PI) * sigma2);
  }

  return log_dens;
}

// =====================================================================
// Temporal log-prior contribution
// =====================================================================

// Compute log-prior for temporal effects (single group)
inline double temporal_log_prior(
    const double* phi,
    int T,
    TemporalType type,
    double tau,           // precision for RW1/RW2, or conditional precision for AR1
    double rho,           // AR1 autocorrelation (ignored for RW)
    bool cyclic
) {
  double log_prior = 0.0;

  if (type == TemporalType::RW1) {
    // RW1: p(phi|tau) propto tau^{(T-1)/2} exp(-0.5 * tau * phi' Q phi)
    double quad = rw1_quadratic_form(phi, T, cyclic);
    // Cyclic RW1 adds the wrap edge, so Q is the cycle-graph Laplacian: a
    // single null direction (the constant), rank T-1 -- same as acyclic RW1.
    int rank = T - 1;  // Rank of precision matrix
    log_prior += 0.5 * rank * std::log(tau);
    log_prior -= 0.5 * tau * quad;

  } else if (type == TemporalType::RW2) {
    // RW2: p(phi|tau) propto tau^{(T-2)/2} exp(-0.5 * tau * phi' Q phi)
    double quad = rw2_quadratic_form(phi, T, cyclic);
    // Cyclic RW2 annihilates only constants on a cycle (a linear ramp is not
    // periodic), so rank T-1; acyclic RW2 also annihilates the ramp, rank T-2.
    int rank = cyclic ? T - 1 : T - 2;  // Rank of precision matrix
    log_prior += 0.5 * rank * std::log(tau);
    log_prior -= 0.5 * tau * quad;

  } else if (type == TemporalType::AR1) {
    // AR1: proper prior
    log_prior += ar1_log_density(phi, T, rho, tau);

  } else if (type == TemporalType::IID) {
    // IID N(0, 1/tau): sum of independent normal log-densities
    double sigma2 = 1.0 / tau;
    double log_norm = -0.5 * std::log(2.0 * M_PI * sigma2);
    for (int t = 0; t < T; t++) {
      log_prior += log_norm - 0.5 * phi[t] * phi[t] / sigma2;
    }
  }

  return log_prior;
}

// =====================================================================
// Sum-to-zero constraint (soft)
// =====================================================================

// Apply soft sum-to-zero constraint penalty for RW models
inline double sum_to_zero_penalty(const double* phi, int T, double lambda = 0.001) {
  double sum = 0.0;
  for (int t = 0; t < T; t++) {
    sum += phi[t];
  }
  return -0.5 * lambda * sum * sum;
}

} // namespace tulpa_temporal

#endif // QUOTR_HMC_TEMPORAL_H
