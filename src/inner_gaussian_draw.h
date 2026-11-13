#ifndef TULPA_INNER_GAUSSIAN_DRAW_H
#define TULPA_INNER_GAUSSIAN_DRAW_H

#include "laplace_cholesky.h"
#include "sparse_cholesky.h"

namespace tulpa {

// Apply the inverse square root of the live precision to column-major normals.
inline bool inner_gaussian_transform(
    int n, int ncol, const DenseCholeskyScratch& chol,
    SparseCholeskySolver& solver, bool sparse,
    const double* eps, double* delta) {
  if (sparse) return solver.apply_inv_chol_factor(eps, delta, n, ncol);
  for (int c = 0; c < ncol; ++c) {
    const double* z = eps + static_cast<std::size_t>(c) * n;
    double* d = delta + static_cast<std::size_t>(c) * n;
    for (int j = n - 1; j >= 0; --j) {
      double sum = z[j];
      for (int k = j + 1; k < n; ++k) sum -= chol.L[k + j * n] * d[k];
      d[j] = sum / chol.L[j + j * n];
    }
  }
  return true;
}

} // namespace tulpa

#endif
