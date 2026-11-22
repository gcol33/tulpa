// laplace_hessian_target.h
// Overloads that let one prior scatter write either Hessian storage: the dense
// DenseMat, which holds both triangles, and the SparseHessianBuilder, which
// holds the lower triangle and normalizes (row, col) itself.

#ifndef TULPA_LAPLACE_HESSIAN_TARGET_H
#define TULPA_LAPLACE_HESSIAN_TARGET_H

#include "laplace_types.h"
#include "sparse_hessian.h"
#include "laplace_s2z.h"
#include <Rcpp.h>

namespace tulpa {

// Add v to the symmetric entry (i, j): both orientations on the dense matrix,
// the one stored entry on the sparse builder.
inline void hess_add_sym(DenseMat& H, int i, int j, double v) {
    H[i][j] += v;
    if (i != j) H[j][i] += v;
}

inline void hess_add_sym(SparseHessianBuilder& H, int i, int j, double v) {
    H.add(i, j, v);
}

// Add v to (row, col) for a scatter that visits every off-diagonal entry once
// from each end: the dense matrix takes every visit into its own orientation,
// the sparse builder takes only the visit from the lower index.
inline void hess_add_visit(DenseMat& H, int row, int col, double v) {
    H[row][col] += v;
}

inline void hess_add_visit(SparseHessianBuilder& H, int row, int col, double v) {
    if (col >= row) H.add(row, col, v);
}

// The sum-to-zero pin of laplace_s2z.h on either storage. The dense matrix
// always stores the exact 1 1' block, so `storage` only steers the sparse one.
inline void add_s2z_pin_to(DenseVec& grad, DenseMat& H,
                           const Rcpp::NumericVector& x,
                           int start, const int* idx, int n, double tau,
                           S2ZStorage /*storage*/) {
    add_s2z_pin(grad, H, x, start, idx, n, tau);
}

inline void add_s2z_pin_to(DenseVec& grad, SparseHessianBuilder& H,
                           const Rcpp::NumericVector& x,
                           int start, const int* idx, int n, double tau,
                           S2ZStorage storage) {
    add_s2z_pin_sparse(grad, H, x, start, idx, n, tau, storage);
}

} // namespace tulpa

#endif // TULPA_LAPLACE_HESSIAN_TARGET_H
