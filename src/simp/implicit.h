// Vendored from gcol33/SIMP (a87642d) by vendor_simp.sh -- do not edit.
// Upstream: SIMP inst/include/simp/implicit.h. Edit there and re-vendor.

#ifndef SIMP_IMPLICIT_H
#define SIMP_IMPLICIT_H

#include <Eigen/Dense>
#include <algorithm>

namespace simp {

// Outcome of a generalized-leapfrog step: whether both implicit stages reached
// the fixed-point tolerance, and the largest iteration count they used.
struct ImplicitResult {
  bool converged;
  int iters;
};

// Generalized (implicit) leapfrog for a non-separable Hamiltonian H(q, p),
// as in Riemannian Hamiltonian Monte Carlo where the mass depends on position.
// One symmetric, time-reversible, symplectic step of size eps.
//
// dHdq(q, p) and dHdp(q, p) return the partial gradients dH/dq and dH/dp. The
// step is
//   p_half = p      - (eps/2) dH/dq(q,      p_half)      (implicit in p_half)
//   q_new  = q      + (eps/2)[dH/dp(q, p_half) + dH/dp(q_new, p_half)]
//                                                        (implicit in q_new)
//   p_new  = p_half - (eps/2) dH/dq(q_new,  p_half)      (explicit)
// The two implicit stages are solved by fixed-point iteration to tol, capped at
// max_iter. For a separable H = U(q) + K(p) the partials drop their cross
// dependence and both stages become explicit, recovering the standard leapfrog.
template <typename DHDQ, typename DHDP>
ImplicitResult step_implicit(Eigen::VectorXd& q, Eigen::VectorXd& p, double eps,
                             DHDQ&& dHdq, DHDP&& dHdp,
                             double tol = 1e-10, int max_iter = 100) {
  const double half = 0.5 * eps;
  bool converged = true;
  int iters = 0;

  // Stage 1: p_half = p - half * dHdq(q, p_half).
  Eigen::VectorXd p_half = p - half * dHdq(q, p);  // explicit warm start
  bool stage1 = false;
  for (int it = 1; it <= max_iter; ++it) {
    Eigen::VectorXd cand = p - half * dHdq(q, p_half);
    double err = (cand - p_half).cwiseAbs().maxCoeff();
    p_half = cand;
    iters = std::max(iters, it);
    if (err < tol) { stage1 = true; break; }
  }
  converged = converged && stage1;

  // Stage 2: q_new = q + half * (dHdp(q, p_half) + dHdp(q_new, p_half)).
  const Eigen::VectorXd dHdp_q = dHdp(q, p_half);
  Eigen::VectorXd q_new = q + eps * dHdp_q;  // explicit warm start
  bool stage2 = false;
  for (int it = 1; it <= max_iter; ++it) {
    Eigen::VectorXd cand = q + half * (dHdp_q + dHdp(q_new, p_half));
    double err = (cand - q_new).cwiseAbs().maxCoeff();
    q_new = cand;
    iters = std::max(iters, it);
    if (err < tol) { stage2 = true; break; }
  }
  converged = converged && stage2;

  // Stage 3: explicit momentum half step at the new position.
  Eigen::VectorXd p_new = p_half - half * dHdq(q_new, p_half);

  q = q_new;
  p = p_new;
  return ImplicitResult{converged, iters};
}

}  // namespace simp

#endif  // SIMP_IMPLICIT_H
