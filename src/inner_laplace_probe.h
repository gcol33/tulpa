// inner_laplace_probe.h
//
// The Gaussian-conditional-mean curve at a probed latent index, shared by both
// inner-layer diagnostics.
//
// For a fixed latent index i, write H for the converged Newton Hessian
// (posterior precision), Sigma = H^{-1}, and v_i = Sigma e_i. Then
// v_i[i] = Sigma_ii is the inner Laplace's conditional variance sigma_i^2, and
//
//   x(t) = mode + (t / sigma_i^2) v_i
//
// is E_piG(x_{-i} | x_i = mode_i + t), the Gaussian conditional-mean curve. It
// is linear in t and passes through the joint mode at t = 0, and the Gaussian
// approximation restricted to it is exactly N(mode_i, sigma_i^2) -- the inner
// Laplace's own marginal for x_i, since v_i' H v_i = e_i' Sigma e_i = sigma_i^2.
//
// Both inner-layer diagnostics run along this one curve: the cubic Edgeworth
// term gamma_3 (inner_laplace_skew.h) expands the joint log density along it,
// and the inner importance k-hat (inner_laplace_is.h) importance-samples it.
// The solve is the same, so it lives here rather than in either of them.

#ifndef TULPA_INNER_LAPLACE_PROBE_H
#define TULPA_INNER_LAPLACE_PROBE_H

#include "laplace_cholesky.h"
#include "sparse_cholesky.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace tulpa {

// Solve H v = e_i against the LIVE factor (dense scratch.chol, or the sparse
// solver when use_sparse) without refactorizing -- the same pattern the
// inv_block_layout diagonal-block extraction in laplace_newton.h uses.
//
// rhs / v / z_work are caller-supplied scratch sized n_x (z_work is read only
// on the dense path). On success `v` holds Sigma e_i and `sigma_i` its
// conditional standard deviation. Returns false, leaving `sigma_i` untouched,
// when the index is out of range, the solve produced a non-finite entry, or the
// implied conditional variance is non-positive (an unidentified or degenerate
// component) -- the caller then leaves that index unscored rather than
// reporting a number derived from a broken solve.
inline bool inner_probe_column(int n_x, int i,
                               DenseCholeskyScratch& chol,
                               SparseCholeskySolver& sparse_solver,
                               bool use_sparse,
                               std::vector<double>& rhs,
                               std::vector<double>& v,
                               std::vector<double>& z_work,
                               double& sigma_i) {
  if (i < 0 || i >= n_x) return false;

  std::fill(rhs.begin(), rhs.end(), 0.0);
  rhs[i] = 1.0;
  if (use_sparse) {
    sparse_solver.solve(rhs.data(), v.data(), n_x);
    for (int k = 0; k < n_x; k++) {
      if (!std::isfinite(v[k])) return false;
    }
  } else if (!chol_substitute_raw(chol.L.data(), n_x, rhs.data(), v.data(),
                                  z_work.data())) {
    return false;
  }

  const double sigma2_i = v[i];
  if (!(sigma2_i > 0.0) || !std::isfinite(sigma2_i)) return false;
  sigma_i = std::sqrt(sigma2_i);
  return true;
}

} // namespace tulpa

#endif // TULPA_INNER_LAPLACE_PROBE_H
