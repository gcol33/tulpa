// Vendored from gcol33/SIMP (cc1de1d) by vendor_simp.sh -- do not edit.
// Upstream: SIMP inst/include/simp/mts.h. Edit there and re-vendor.
// Copyright (c) 2026 Gilles Colling. MIT license (see inst/COPYRIGHTS).

#ifndef SIMP_MTS_H
#define SIMP_MTS_H

#include <Eigen/Dense>

namespace simp {

// One outer step of the RESPA / Verlet-I multiple-time-stepping integrator for
// a potential that splits as U = U_slow + U_fast. The slow force (expensive and
// smooth -- a likelihood, say) is evaluated once per outer step; the fast force
// (cheap and stiff -- a Gaussian latent prior, say) drives m inner leapfrog
// substeps of size eps / m. The step is symmetric, time reversible, and
// symplectic, so a Hamiltonian Monte Carlo proposal built on it stays exact
// under the Metropolis correction while spending far fewer slow-force
// evaluations than a single-rate integrator at the step the stiff part demands.
//
// force_slow(q) and force_fast(q) each return the force -dU/dq of their part
// (for a log-density lp, force = d(lp)/dq); applyMinv(p) returns M^{-1} p. Any
// scalar T (double or an automatic-differentiation type) flows through the
// Eigen vectors, as in the single-rate stepper.
template <typename T, typename FSlow, typename FFast, typename ApplyMinv>
void step_mts(Eigen::Matrix<T, Eigen::Dynamic, 1>& q,
              Eigen::Matrix<T, Eigen::Dynamic, 1>& p, const T& eps, int m,
              FSlow&& force_slow, FFast&& force_fast, ApplyMinv&& applyMinv) {
  const T half = T(0.5) * eps;
  const T inner = eps / T(m);
  const T inner_half = T(0.5) * inner;

  p += half * force_slow(q);                    // outer half kick (slow)
  for (int i = 0; i < m; ++i) {                 // inner leapfrog chain (fast)
    p += inner_half * force_fast(q);
    q += inner * applyMinv(p);
    p += inner_half * force_fast(q);
  }
  p += half * force_slow(q);                    // outer half kick (slow)
}

// Advance n_steps outer steps in place.
template <typename T, typename FSlow, typename FFast, typename ApplyMinv>
void integrate_mts(Eigen::Matrix<T, Eigen::Dynamic, 1>& q,
                   Eigen::Matrix<T, Eigen::Dynamic, 1>& p, const T& eps,
                   int n_steps, int m, FSlow&& force_slow, FFast&& force_fast,
                   ApplyMinv&& applyMinv) {
  for (int i = 0; i < n_steps; ++i) {
    step_mts(q, p, eps, m, force_slow, force_fast, applyMinv);
  }
}

}  // namespace simp

#endif  // SIMP_MTS_H
