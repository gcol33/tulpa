// joint_inner_vcov.h
// Per-cell constrained inner-covariance block extraction for the joint
// nested-Laplace post-grid step.
//
// Given a grid cell's latent precision Qk (= H at the inner mode, sparse SPD,
// lower-triangle CSC) and a latent index set idx = [dense | field], return the
// sum-to-zero-constrained covariance block on the read sub-blocks:
//   block[D, D]            (fixed-effect betas: full)
//   block[D, F], block[F, D]  (betas x field cross: full)
//   diag(block[F, F])      (field marginal variances)
// with the field x field off-diagonal left at zero (never read by the summary
// or the predict path; see dev_notes/plans/joint_inner_vcov_selinv.md).
//
// The constrained block is
//   C[idx,idx] - G' M^{-1} G,  C = Qk^{-1},  W = C A',  G = W[idx,:]',  M = A C A'
// (conditioning by kriging, Rue & Held 2005). The cheap recipe avoids the
// ~|idx|-column inverse: it solves only the |D| dense columns of C plus the
// k_constr constraint columns W, and gets diag(C[F,F]) from one Takahashi
// selected-inversion pass (sparse_cholesky.h). field_marginal = false instead
// forms the full idx x idx block (every idx column solved) for the full-block
// fallback consumers; still one factorize per cell, parallel across cells.
//
// The core takes POD inputs + a caller-owned SparseCholeskySolver (CHOLMOD
// common is not thread-safe, so concurrent callers pass one solver per thread),
// so the joint kernel can reuse the factor it already forms at its store_Q site
// and the Rcpp entry can drive a parallel per-cell loop
// through the same single source.

#ifndef TULPA_JOINT_INNER_VCOV_H
#define TULPA_JOINT_INNER_VCOV_H

#include "sparse_cholesky.h"
#include <vector>

namespace tulpa {

// Per-cell extraction. Returns true on success (out_block filled), false if the
// cell's precision could not be factorized (caller treats as a dropped cell).
//
//   Qp/Qi/Qx : Qk lower-triangle CSC (Qp length n_x+1, Qi 0-based rows, Qx
//              values), the dgCMatrix / stype = -1 convention the kernel stores.
//   idx0     : 0-based latent indices, length p = n_dense + n_field. The first
//              n_dense are the dense (fixed-effect) block; the remaining are the
//              field block (diagonal only when field_marginal).
//   A_cols   : sum-to-zero constraint groups, each a vector of 0-based latent
//              indices summed to zero (one group per ICAR/CAR field, two per
//              BYM2 field). Empty => unconstrained block.
//   field_marginal : true => cheap recipe (dense columns + field diagonal);
//                    false => full p x p block (all idx columns solved).
//   solver   : caller-owned CHOLMOD context (reset() + analyze() + factorize()
//              are driven here). One per thread under concurrency.
//   out_block: resized to p*p, column-major, p = idx0.size().
bool extract_inner_vcov_block_cell(
    const int* Qp, const int* Qi, const double* Qx, int n_x, int nnz,
    const std::vector<int>& idx0, int n_dense,
    const std::vector<std::vector<int>>& A_cols,
    bool field_marginal,
    SparseCholeskySolver& solver,
    std::vector<double>& out_block
);

} // namespace tulpa

#endif // TULPA_JOINT_INNER_VCOV_H
