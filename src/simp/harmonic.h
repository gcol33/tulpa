// Vendored from gcol33/SIMP (cc1de1d) by vendor_simp.sh -- do not edit.
// Upstream: SIMP inst/include/simp/harmonic.h. Edit there and re-vendor.

#ifndef SIMP_HARMONIC_H
#define SIMP_HARMONIC_H

#include "scheme.h"
#include <Eigen/Dense>
#include <limits>
#include <algorithm>
#include <cmath>

namespace simp {

// One-step propagation matrix of a splitting scheme on the unit harmonic
// oscillator H = (q^2 + p^2) / 2, at dimensionless step nu = omega * eps. The
// scheme's elementary flows become shears on (q, p): a kick with weight c sends
// p -= (c * nu) q, a drift with weight c sends q += (c * nu) p. Every separable
// quadratic system diagonalizes into independent oscillators, so this 2x2 map
// at nu = omega_k * eps characterizes the scheme's action on mode k exactly.
inline Eigen::Matrix2d harmonic_map(const Scheme& s, double nu) {
  Eigen::Matrix2d M = Eigen::Matrix2d::Identity();
  for (const auto& op : s.ops) {
    Eigen::Matrix2d F = Eigen::Matrix2d::Identity();
    if (op.first == Op::Kick) {
      F(1, 0) = -op.second * nu;  // p -= c nu q
    } else {
      F(0, 1) = op.second * nu;   // q += c nu p
    }
    M = F * M;
  }
  return M;
}

// True iff the scheme is linearly stable at nu, i.e. the harmonic map is
// elliptic (|trace| < 2), so the modified energy stays bounded rather than
// growing geometrically.
inline bool harmonic_stable(const Scheme& s, double nu) {
  return std::abs(harmonic_map(s, nu).trace()) < 2.0;
}

// Relative amplitude of the energy oscillation the scheme induces on the unit
// harmonic oscillator at dimensionless step nu, in [0, 1). An elliptic
// symplectic map preserves a modified energy whose level sets are ellipses; the
// true energy H = (q^2 + p^2) / 2 therefore oscillates as the state traverses
// one, between the ellipse's minor and major extents. With lambda_min,
// lambda_max the eigenvalues of the invariant form, H_max / H_min =
// lambda_max / lambda_min, so the peak-to-peak relative swing is
// (lambda_max - lambda_min) / (lambda_max + lambda_min): zero for the exact
// rotation, rising to one at the stability boundary. This is the quantity the
// minimum-error and step-adapted coefficients minimize. Returns +inf when the
// scheme is unstable at nu.
inline double harmonic_energy_error(const Scheme& s, double nu) {
  const Eigen::Matrix2d M = harmonic_map(s, nu);
  if (!(std::abs(M.trace()) < 2.0)) {
    return std::numeric_limits<double>::infinity();
  }
  // Invariant quadratic form of a symplectic 2x2 map M = [[a, b], [c, d]]:
  // S = [[-c, (a - d) / 2], [(a - d) / 2, b]] satisfies M' S M = S.
  const double a = M(0, 0), b = M(0, 1), c = M(1, 0), d = M(1, 1);
  Eigen::Matrix2d S;
  S << -c, 0.5 * (a - d), 0.5 * (a - d), b;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(S);
  double l0 = es.eigenvalues()(0), l1 = es.eigenvalues()(1);
  if (l0 + l1 < 0.0) { l0 = -l0; l1 = -l1; }  // orient to the PD representative
  const double lo = std::min(l0, l1), hi = std::max(l0, l1);
  if (!(lo > 0.0)) return std::numeric_limits<double>::infinity();
  return (hi - lo) / (hi + lo);
}

// Largest dimensionless step at which the scheme stays stable: the upper end of
// the elliptic interval nu in (0, nu_lim), found by scanning outward from zero
// and bisecting the first crossing of |trace| = 2. Returns nu_hi if no crossing
// is found below it.
inline double harmonic_stability_limit(const Scheme& s, double nu_hi = 10.0,
                                       double scan = 0.01) {
  double prev = 0.0;
  for (double nu = scan; nu <= nu_hi; nu += scan) {
    if (!harmonic_stable(s, nu)) {
      double lo = prev, hi = nu;
      for (int it = 0; it < 60; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (harmonic_stable(s, mid)) lo = mid; else hi = mid;
      }
      return lo;
    }
    prev = nu;
  }
  return nu_hi;
}

}  // namespace simp

#endif  // SIMP_HARMONIC_H
