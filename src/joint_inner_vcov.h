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
// or the predict path).
//
// The constrained block is
//   C[idx,idx] - G' Cap^{-1} G,  W = C Vb,  G = W[idx,:]',  Cap = Psi^{-1} + Vb' W
// (conditioning by kriging, Rue & Held 2005), with C the inverse of the
// cell's precision anchored at one node per constraint group and Vb, Psi the
// constraint and anchor terms that undo the anchor (InvBlockConstraint). The cheap recipe avoids the
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

#include "linear_constraint_rows.h"
#include "row_classes.h"
#include "sparse_cholesky.h"
#include <algorithm>
#include <functional>
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
//   cons     : the cell's constraint (CellConstraintRows): the hard rows its
//              Gaussian is conditioned on and the soft pins the stored Q
//              leaves off. Empty => the unconstrained inverse of Q.
//   field_marginal : true => cheap recipe (dense columns + field diagonal);
//                    false => full p x p block (all idx columns solved).
//   solver   : caller-owned CHOLMOD context (reset() + analyze() + factorize()
//              are driven here). One per thread under concurrency.
//   out_block: resized to p*p, column-major, p = idx0.size().
bool extract_inner_vcov_block_cell(
    const int* Qp, const int* Qi, const double* Qx, int n_x, int nnz,
    const std::vector<int>& idx0, int n_dense,
    const CellConstraintRows& cons,
    bool field_marginal,
    SparseCholeskySolver& solver,
    std::vector<double>& out_block
);

// The same cell's block in the shape it is actually computed in, for callers
// that accumulate it rather than hold it: the first `s` columns as a p x s
// column-major strip (lower triangle, rows a >= b; the rows above the diagonal
// stay 0), and the diagonal of the remaining p - s columns, whose off-diagonal
// is never formed. `s` is n_dense under field_marginal and p otherwise, so a
// field-marginal cell costs O(p n_dense) rather than O(p^2).
struct InnerVcovStrip {
    int p = 0;
    int s = 0;
    std::vector<double> strip;  // p x s, column-major
    std::vector<double> diag;   // length p - s
};

inline int inner_vcov_strip_width(int p, int n_dense, bool field_marginal) {
    if (!field_marginal) return p;
    return std::max(0, std::min(n_dense, p));
}

bool extract_inner_vcov_strip_cell(
    const int* Qp, const int* Qi, const double* Qx, int n_x, int nnz,
    const std::vector<int>& idx0, int n_dense,
    const CellConstraintRows& cons,
    bool field_marginal,
    SparseCholeskySolver& solver,
    InnerVcovStrip& out
);

// The joint tier's per-cell fixed-effect retention, asked for
// on the inner Newton loop itself rather than off a grid-wide store of every
// cell's precision.
//
// `p` is the leading latent block -- both joint layouts stack every arm's
// coefficients as a contiguous prefix, so the fixed effects are latent indices
// [0, p). The block is read under the cell's own constraint (the one the solve
// carries, handed to extract_joint_fixed_block), so it is the covariance the
// fit's own posterior draws are generated from.
//
// `extra_idx` (0-based, each >= p) widens the block to the latent indices
// [0, p) followed by `extra_idx`, so a block whose prediction needs its own
// coefficients' covariance with the fixed effects -- a varying-coefficient
// basis read at new coordinates -- carries it from the same cell solve. The
// leading p x p of the widened block is the fixed-effect block.
struct JointFixedBlockRequest {
    int p = 0;
    std::vector<int> extra_idx;
    bool active() const { return p > 0; }
};

// Extract the request's block from one cell's lower-triangle CSC and append it
// to (`flat_out`, `sizes_out`) in the LaplaceResult re_cov contract. The CSC is
// the caller's working Hessian, alive only for this cell, so the whole grid's
// precision is never resident. `solver` is the caller's own CHOLMOD context (one
// per thread; it is reset here, so it must NOT be the loop's live factor).
//
// Identical by construction to what cpp_joint_inner_vcov_blocks() returns for
// the same cell at `idx = 1:p`, `n_dense = p`, `field_marginal = FALSE`: same
// bytes in, same routine.
bool extract_joint_fixed_block(
    const int* Qp, const int* Qi, const double* Qx, int n_x, int nnz,
    const JointFixedBlockRequest& req,
    const CellConstraintRows& cons,
    SparseCholeskySolver& solver,
    std::vector<double>& flat_out,
    std::vector<int>& sizes_out
);

// The per-row predictive variance of the linear predictor at one cell,
// var(eta_r | theta_k, y) = a_r' Sigma_k a_r, asked for on the inner Newton loop
// the way the fixed-effect block is. `loadings` fills every row's loading vector
// a_r = d eta_r / d x at the latent point it is handed (joint_row_loadings), so
// the request carries the cell's own block amplitudes and the loop needs no
// knowledge of the arms.
struct JointEtaVarRequest {
    std::function<void(const double* x, RowLoadings& out)> loadings;
    bool active() const { return static_cast<bool>(loadings); }
};

// Sigma_k is the covariance of the cell's Gaussian: the CSC `(Qp, Qi, Qx)` plus
// the soft pins of `cons`, conditioned on its hard rows, folded by Woodbury
// against the CSC anchored at one node per row, so a direction the stored
// precision holds only through the uniform ridge never enters the fold
// (InvBlockConstraint). Rows
// sharing a loading vector share one solve (row_classes.h). Returns false, with
// `var_out` all NaN, when the precision does not factor; a row whose solve loses
// positivity reads NaN on its own.
bool extract_joint_eta_var(
    const int* Qp, const int* Qi, const double* Qx, int n_x, int nnz,
    const RowLoadings& L,
    const CellConstraintRows& cons,
    SparseCholeskySolver& solver,
    std::vector<double>& var_out
);

// Draws of N(0, V) for one cell, V = B (B' M B)^{-1} B' the covariance of the
// cell's Gaussian on null(A) (B a basis of it, A the hard rows of `cons`;
// V = M^{-1} with none), M = Q + S' D S with its soft rows S and their
// precision D, from the anchored factor
// every other read of the cell uses (InvBlockConstraint::condition_draw):
// column c of the n_x x n_draws column-major `eps` (standard normals) gives
// column c of `out`. M need be positive definite on null(A) only and Q
// positive semi-definite on the whole space. Returns false, with `why` set,
// when the cell does not factor or M is indefinite off the constraint set.
bool draw_constrained_cell(
    const int* Qp, const int* Qi, const double* Qx, int n_x, int nnz,
    const CellConstraintRows& cons,
    const double* eps, int n_draws,
    SparseCholeskySolver& solver,
    std::vector<double>& out, const char** why
);

// The fold of `cons` over the cell's stored precision: factors Q anchored at
// one node per row (`solver` keeps that factor) and builds `constr` over the
// rows [soft; hard], D^{-1} on the soft block and the hard limit on the rest.
// False, with `why` set when given, when Q does not factor or D is not
// positive definite; a fold whose target is not positive definite is built
// with `constr.usable = false`, which the caller reads.
struct InvBlockConstraint;
bool fold_cell_constraint(
    SparseCholeskySolver& solver,
    const int* Qp, const int* Qi, const double* Qx, int n_x, int nnz,
    const CellConstraintRows& cons, InvBlockConstraint& constr,
    const char** why = nullptr
);

} // namespace tulpa

#endif // TULPA_JOINT_INNER_VCOV_H
