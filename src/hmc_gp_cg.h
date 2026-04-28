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

// Solve C * x = b using CG for small dense SPD matrix C.
// C_flat: k x k matrix in row-major order
// Returns: number of iterations (positive = converged, negative = failure)
inline int dense_cg_solve(
    const double* C_flat, int k,
    const double* b,
    double* x,
    double tol = 1e-6,
    int maxiter = 100
) {
  std::fill(x, x + k, 0.0);

  // r = b - C*x = b (since x=0)
  std::vector<double> r(k), p(k), Cp(k);
  std::copy(b, b + k, r.begin());
  std::copy(r.begin(), r.end(), p.begin());

  double r_dot_r = tulpa_linalg::dot_product(r.data(), r.data(), k);
  double b_norm = std::sqrt(tulpa_linalg::dot_product(b, b, k));
  if (b_norm < 1e-14) b_norm = 1.0;

  for (int iter = 0; iter < maxiter; iter++) {
    // Cp = C * p
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

    double alpha = r_dot_r / p_dot_Cp;

    for (int i = 0; i < k; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Cp[i];
    }

    double r_dot_r_new = tulpa_linalg::dot_product(r.data(), r.data(), k);
    if (std::sqrt(r_dot_r_new) / b_norm < tol) {
      return iter + 1;  // converged
    }

    double beta = r_dot_r_new / r_dot_r;
    r_dot_r = r_dot_r_new;

    for (int i = 0; i < k; i++) {
      p[i] = r[i] + beta * p[i];
    }
  }

  return -maxiter;  // did not converge
}

// Solve C * x = b using PCG with Jacobi (diagonal) preconditioner.
// Numerically more robust than plain CG for ill-conditioned NNGP neighbor
// systems (e.g. small phi or repeated coordinates).
inline int dense_pcg_solve(
    const double* C_flat, int k,
    const double* b,
    double* x,
    double tol = 1e-6,
    int maxiter = 100
) {
  std::fill(x, x + k, 0.0);

  std::vector<double> M_inv(k);
  for (int i = 0; i < k; i++) {
    double diag = C_flat[i * k + i];
    M_inv[i] = (std::abs(diag) > 1e-14) ? 1.0 / diag : 1.0;
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
      return -(iter + 1);
    }

    double alpha = r_dot_z / p_dot_Cp;

    for (int i = 0; i < k; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Cp[i];
    }

    double r_norm = std::sqrt(tulpa_linalg::dot_product(r.data(), r.data(), k));
    if (r_norm / b_norm < tol) {
      return iter + 1;
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

  return -maxiter;
}

}  // namespace tulpa_gp

#endif  // TULPA_HMC_GP_CG_H
