// hmc_gp_cg.h
// Conjugate Gradient based linear solvers for NNGP neighbor systems.
//
// The neighbor system C * alpha = c is small (k x k where k = nn, typically
// 10-20) but solved O(N) times per log-lik / gradient call. CG is an
// alternative to direct Cholesky that can be cheaper for ill-conditioned
// or very large NNGP problems and serves as a stepping stone toward a
// future GPU-batched solver.
//
// Wired into the sampler via `GPSolverConfig::solver = CG/PCG` set from
// `R/spatial.R` (`solver = "cg"` argument). The default ("cholesky") path
// is unchanged.

#ifndef TULPA_HMC_GP_CG_H
#define TULPA_HMC_GP_CG_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "linalg_fast.h"

namespace tulpa_gp {

// =============================================================================
// CG-based linear system solvers for small dense SPD systems (k x k)
// =============================================================================

// Solve C * x = b for a small dense SPD C, optionally preconditioned.
//
// C_flat: k x k in row-major order. Returns the iteration count, positive when
// converged and negative on breakdown or exhaustion.
//
// `precond` selects the Jacobi (diagonal) preconditioner, which is what makes
// this robust on an ill-conditioned neighbour system (small phi, or repeated
// coordinates). Without it M^-1 is the identity and every line below reduces to
// plain CG term for term: z == r, so r_dot_z is r_dot_r, the residual norm in
// the convergence test is the same quantity, and beta and the direction update
// are unchanged. That is why there is one body rather than two.
inline int dense_pcg_solve_impl(
    const double* C_flat, int k,
    const double* b,
    double* x,
    double tol,
    int maxiter,
    bool precond
) {
  std::fill(x, x + k, 0.0);

  std::vector<double> M_inv(k, 1.0);
  if (precond) {
    for (int i = 0; i < k; i++) {
      double diag = C_flat[i * k + i];
      M_inv[i] = (std::abs(diag) > 1e-14) ? 1.0 / diag : 1.0;
    }
  }

  std::vector<double> r(k), z(k), p(k), Cp(k);
  std::copy(b, b + k, r.begin());

  for (int i = 0; i < k; i++) {
    z[i] = M_inv[i] * r[i];
  }
  std::copy(z.begin(), z.end(), p.begin());

  double r_dot_z = tulpa_linalg::dot_product(r.data(), z.data(), k);
  double b_norm = std::sqrt(tulpa_linalg::dot_product(b, b, k));
  if (b_norm < 1e-14) b_norm = 1.0;

  for (int iter = 0; iter < maxiter; iter++) {
    for (int i = 0; i < k; i++) {
      Cp[i] = 0.0;
      for (int j = 0; j < k; j++) {
        Cp[i] += C_flat[i * k + j] * p[j];
      }
    }

    double p_dot_Cp = tulpa_linalg::dot_product(p.data(), Cp.data(), k);
    if (std::abs(p_dot_Cp) < 1e-30) {
      return -(iter + 1);  // breakdown
    }

    double alpha = r_dot_z / p_dot_Cp;

    for (int i = 0; i < k; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Cp[i];
    }

    double r_norm = std::sqrt(tulpa_linalg::dot_product(r.data(), r.data(), k));
    if (r_norm / b_norm < tol) {
      return iter + 1;  // converged
    }

    for (int i = 0; i < k; i++) {
      z[i] = M_inv[i] * r[i];
    }

    double r_dot_z_new = tulpa_linalg::dot_product(r.data(), z.data(), k);
    double beta = r_dot_z_new / r_dot_z;
    r_dot_z = r_dot_z_new;

    for (int i = 0; i < k; i++) {
      p[i] = z[i] + beta * p[i];
    }
  }

  return -maxiter;  // did not converge
}

inline int dense_cg_solve(
    const double* C_flat, int k,
    const double* b,
    double* x,
    double tol = 1e-6,
    int maxiter = 100
) {
  return dense_pcg_solve_impl(C_flat, k, b, x, tol, maxiter, /*precond=*/false);
}

inline int dense_pcg_solve(
    const double* C_flat, int k,
    const double* b,
    double* x,
    double tol = 1e-6,
    int maxiter = 100
) {
  return dense_pcg_solve_impl(C_flat, k, b, x, tol, maxiter, /*precond=*/true);
}

}  // namespace tulpa_gp

#endif  // TULPA_HMC_GP_CG_H
