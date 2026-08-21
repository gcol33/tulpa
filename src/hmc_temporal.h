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
#include "tulpa/soft_sum_to_zero.h"  // s2z_precision
#include "tulpa/sum_to_zero.h"       // rw1_rank / rw2_rank
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

// Rank of the intrinsic-GMRF precision, defined in tulpa/sum_to_zero.h next to
// the augmented rank it feeds and visible to packages that link against the
// engine. Named here so the temporal kernels below keep reading unqualified.
using tulpa::rw1_rank;
using tulpa::rw2_rank;

// log(2*pi).
constexpr double kLogTwoPi = 1.8378770664093454835606594728112;

// Floor on the AR1 stationary factor 1 - rho^2.
//
// The stationary variance sigma^2 / (1 - rho^2) diverges at the boundary of the
// stationary region, and with it the density and its gradient. The floor is on
// the CORRELATION factor rather than on the stationary precision
// tau * (1 - rho^2), so where it engages is a property of rho alone and does not
// move with tau: a precision floor engages at 1 - rho^2 = 1e-12 when tau = 100
// and at 1e-8 when tau = 0.01, for the same model.
constexpr double kAr1StationaryFloor = 1e-10;

template <typename T>
inline T ar1_one_minus_rho2(const T& rho) {
  return tulpa::math::safe_max(T(1.0) - rho * rho, T(kAr1StationaryFloor));
}

// Gaussian normalizer of a GMRF whose precision is tau * Q0 and whose Q0 has
// `rank` non-null eigenvalues: 0.5 * rank * (log tau - log 2pi). Taken on the
// log-precision, which is the coordinate every sampler carries.
//
// The pseudo-determinant 0.5 * log|Q0|_+ is deliberately NOT included. It is
// constant in the hyperparameters, so every quantity within one model is on a
// common scale and the Laplace and sampler tiers agree; it differs between RW1
// and RW2 and between cyclic and acyclic, so a marginal likelihood from this
// normalizer is NOT comparable across temporal structures.
template <typename T>
inline T gmrf_log_norm(int rank, const T& log_tau) {
  return T(0.5 * rank) * (log_tau - T(kLogTwoPi));
}

// Log-density of a stationary AR1 of length T_len:
//   phi[0]            ~ N(0, sigma^2 / (1 - rho^2))
//   phi[t] | phi[t-1] ~ N(rho * phi[t-1], sigma^2),   sigma^2 = 1 / tau
//
// Assembled as one rank-T_len GMRF: the joint precision is
// tau * Q_AR1 with |Q_AR1| = 1 - rho^2, so the whole normalizer is
// 0.5 * T_len * (log tau - log 2pi) + 0.5 * log(1 - rho^2).
template <typename T>
inline T ar1_log_density(
    const T* phi,
    int T_len,
    const T& rho,
    const T& tau  // precision = 1/sigma^2
) {
  if (T_len < 1) return T(0.0);

  const T omr2 = ar1_one_minus_rho2(rho);

  T quad = omr2 * phi[0] * phi[0];
  for (int t = 1; t < T_len; t++) {
    const T resid = phi[t] - rho * phi[t - 1];
    quad = quad + resid * resid;
  }

  return gmrf_log_norm(T_len, safe_log(tau))
       + T(0.5) * safe_log(omr2)
       - T(0.5) * tau * quad;
}

// =====================================================================
// Temporal log-prior
// =====================================================================

// Log-density of one temporal field of length T_len under the structure
// `type`, at precision `tau` and (AR1 only) correlation `rho`.
//
// The single implementation behind every temporal prior in the engine: the
// sampler kernels, the TVC terms and the Laplace adapters in
// laplace_temporal_priors.h all evaluate this, so no two tiers can normalize
// the same field differently.
//
// RW1 and RW2 are intrinsic, so their normalizer takes the RANK of Q0 rather
// than the field length; AR1 and IID are proper and their rank is T_len.
template <typename T>
inline T log_prior_temporal(
    const T* phi,
    int T_len,
    TemporalType type,
    const T& tau,         // precision for RW1/RW2/IID, conditional precision for AR1
    const T& rho,         // AR1 autocorrelation (ignored for the rest)
    bool cyclic
) {
  if (type == TemporalType::AR1) return ar1_log_density(phi, T_len, rho, tau);
  if (type == TemporalType::NONE) return T(0.0);

  const T log_tau = safe_log(tau);

  if (type == TemporalType::RW1) {
    return gmrf_log_norm(rw1_rank(T_len, cyclic), log_tau)
         - T(0.5) * tau * rw1_quadratic_form(phi, T_len, cyclic);
  }
  if (type == TemporalType::RW2) {
    return gmrf_log_norm(rw2_rank(T_len, cyclic), log_tau)
         - T(0.5) * tau * rw2_quadratic_form(phi, T_len, cyclic);
  }
  if (type == TemporalType::IID) {
    T quad = T(0.0);
    for (int t = 0; t < T_len; t++) quad = quad + phi[t] * phi[t];
    return gmrf_log_norm(T_len, log_tau) - T(0.5) * tau * quad;
  }
  return T(0.0);
}

// =====================================================================
// Sum-to-zero constraint (soft)
// =====================================================================

// Apply soft sum-to-zero constraint penalty for RW models. The precision is
// derived from the field length (tulpa/soft_sum_to_zero.h) rather than taken
// from the caller, so no call site can pass a kappa where a precision is meant.
inline double sum_to_zero_penalty(const double* phi, int T) {
  double sum = 0.0;
  for (int t = 0; t < T; t++) {
    sum += phi[t];
  }
  return -0.5 * tulpa::s2z_precision(T) * sum * sum;
}

} // namespace tulpa_temporal

#endif // QUOTR_HMC_TEMPORAL_H
