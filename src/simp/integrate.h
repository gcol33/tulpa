// Vendored from gcol33/SIMP (cc1de1d) by vendor_simp.sh -- do not edit.
// Upstream: SIMP inst/include/simp/integrate.h. Edit there and re-vendor.

#ifndef SIMP_INTEGRATE_H
#define SIMP_INTEGRATE_H

#include "scheme.h"
#include <Eigen/Dense>

namespace simp {

// One step of a splitting scheme for a separable Hamiltonian. force(q) returns
// the force -dU/dq (for a target with log-density lp, force = d(lp)/dq), and
// applyMinv(p) returns M^{-1} p. Both are caller-supplied, so a diagonal,
// dense, or matrix-free mass all use this one stepper, as does any scalar T
// (double or an automatic-differentiation type) through the Eigen vectors. The
// mass is constant in q, which keeps H separable and the step symplectic.
template <typename T, typename Force, typename ApplyMinv>
void step(Eigen::Matrix<T, Eigen::Dynamic, 1>& q,
          Eigen::Matrix<T, Eigen::Dynamic, 1>& p,
          const T& eps, const Scheme& s,
          Force&& force, ApplyMinv&& applyMinv) {
  for (const auto& op : s.ops) {
    if (op.first == Op::Kick) {
      p += (T(op.second) * eps) * force(q);
    } else {
      q += (T(op.second) * eps) * applyMinv(p);
    }
  }
}

// Advance n_steps steps in place.
template <typename T, typename Force, typename ApplyMinv>
void integrate(Eigen::Matrix<T, Eigen::Dynamic, 1>& q,
               Eigen::Matrix<T, Eigen::Dynamic, 1>& p,
               const T& eps, int n_steps, const Scheme& s,
               Force&& force, ApplyMinv&& applyMinv) {
  for (int i = 0; i < n_steps; ++i) {
    step(q, p, eps, s, force, applyMinv);
  }
}

}  // namespace simp

#endif  // SIMP_INTEGRATE_H
