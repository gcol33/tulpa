// latent_block.h
// LatentBlock: a single rank-deficient or full-rank latent prior block plugged
// into the multi-block nested-Laplace driver (src/nested_laplace_multi.h) or
// the joint multi-arm driver (src/nested_laplace_joint_multi.h).
//
// Each block lives at [start, start + size) in the joint latent vector x[].
// An observation i (within its arm) sees the block via one entry
// x[start + idx(i, k_arm) - 1] (1-based to match spatial_idx / temporal_idx
// conventions; idx returns -1 if the obs does not see the block, e.g. an
// observation with no time stamp).
//
// At outer-grid point k the block's contribution to arm k_arm's eta is
//   eta_i += arm_scale(k_arm, k) * d_fac(k) * x[start + idx(i, k_arm) - 1].
// d_fac is the BYM2 hook: for ICAR/RW1/RW2/AR1/CAR_proper it is constant 1.0;
// for BYM2-phi it is sqrt(rho_k) * scale_factor; for BYM2-theta it is
// sqrt(1 - rho_k); for IID it is sigma_k. Carrying this coefficient through
// the block lets the scatter handle plain blocks, IID-as-sigma, and BYM2-style
// reparameterizations with one code path.
//
// arm_scale is the joint-driver hook for INLA-style `copy=` semantics: donor
// arms see one sigma per grid point, the copy arm sees another. Empty in the
// single-arm driver and in shared (non-copied) blocks of the joint driver.
//
// This header is internal to tulpa (uses DenseVec/DenseMat from laplace_types).
// Model packages do not consume it directly — they reach the multi-block
// driver through the cpp_nested_laplace_* Rcpp entries.

#ifndef TULPA_LATENT_BLOCK_H
#define TULPA_LATENT_BLOCK_H

#include "laplace_types.h"
#include <Rcpp.h>
#include <functional>
#include <utility>
#include <vector>

namespace tulpa {

// Forward decl — the sparse-builder fields below are referenced by
// pointer/reference only at function-signature level. Full definition
// in sparse_hessian.h, only needed at call sites.
class SparseHessianBuilder;

// How obs contribute to this block's latent indices.
//
// - INDEXED_SINGLE: one block-local DOF per obs (BYM2, ICAR, BYM2-phi,
//   BYM2-theta, CAR_proper, RW1, RW2, AR1, IID, NNGP, tgmrf). Use the
//   existing `idx(i, k_arm)` callback. Most blocks in the registry are
//   this kind.
// - INDEXED_MULTI: a small set of block-local DOFs per obs (SPDE: ~3
//   mesh nodes via barycentric interpolation; FEM-P2: more). Use
//   `obs_indices(i, k_arm, out)` to fill (idx, weight) pairs.
// - DENSE_BASIS: every block coefficient is touched by every obs (HSGP,
//   HSGP-SVC, HSGP-MSGP). Use `basis_eval(i, k_arm, k_grid, out)` to fill the
//   full `size`-vector of basis weights. Pattern builder bypasses per-
//   obs enumeration and fills the full block × block sub-pattern
//   unconditionally.
// - BILINEAR_FACTOR: each obs touches exactly TWO block-local DOFs —
//   one factor-field slot u_i and one per-arm loading lambda_k — and
//   the eta contribution is the PRODUCT `u_i * lambda_k` (bilinear, not
//   linear in x). Use `obs_factor_lambda(i, k_arm)` to fetch the two
//   GLOBAL slot indices; the scatter reads `x[u_slot]` and
//   `x[lambda_slot]` to weight the Gauss-Newton scatter entries
//   (weight on u = lambda * d_e, weight on lambda = u * d_e). Eta
//   accumulator must NOT use the standard active-dof linear formula
//   (it double-counts the product); see compute_eta_joint_sparse_dispatch.
enum class BlockContribKind {
    INDEXED_SINGLE,
    INDEXED_MULTI,
    DENSE_BASIS,
    BILINEAR_FACTOR
};

// Sparsity shape of this block's prior precision Q (governs nnz/row
// the pattern builder appends via `add_prior_pattern`):
//
// - NONE: diagonal only (IID, BYM2-theta with d_fac on it). No off-
//   diagonal entries beyond what the data scatter contributes.
// - ADJACENCY: ~5-9 nnz/row from graph neighbors (ICAR, BYM2-phi,
//   CAR_proper, RW1, RW2, AR1).
// - NN_K: ~nn^2 nnz/row from squared neighbor lists (NNGP).
// - SPDE_Q: FEM stiffness + mass matrix pattern.
// - USER_CSC: user-supplied Q with arbitrary pattern (tgmrf).
// - DIAGONAL_LOWRANK: prior is diagonal (typically I or tau*I), but the
//   data-induced fill on the block is rank-N with a dense basis-block
//   structure (HSGP, latent factors). The pattern builder treats this
//   exactly like DENSE_BASIS contrib + no prior-side off-diagonals.
enum class PriorFillKind {
    NONE,
    ADJACENCY,
    NN_K,
    SPDE_Q,
    USER_CSC,
    DIAGONAL_LOWRANK
};

struct LatentBlock {
    int start = 0;
    int size  = 0;

    // (i, k_arm) -> latent index within block (1-based). Return -1 to skip
    // obs i. Single-arm callers pass k_arm = 0; single-arm factories ignore
    // it. Joint factories dispatch on k_arm to per-arm index vectors.
    //
    // For INDEXED_SINGLE blocks this is the canonical mapping; the sparse
    // scatter reads it directly. For INDEXED_MULTI / DENSE_BASIS, use
    // `obs_indices` / `basis_eval` instead — `idx` may be empty.
    std::function<int(int /*i*/, int /*k_arm*/)> idx;

    // Grid-dependent linear-mixing coefficient on the block's eta contribution
    // (per outer-grid point k). 1.0 for plain indexed blocks (ICAR/RW1/RW2/
    // AR1/CAR_proper in single-arm mode and shared mode); sqrt(rho_k) *
    // scale_factor for BYM2-phi; sqrt(1 - rho_k) for BYM2-theta; sigma_k for
    // IID. Combined with arm_scale (when present) to form the per-arm
    // effective coefficient on x[idx].
    std::function<double(int /*k_grid*/)> d_fac;

    // Per-arm linear scale on the block's eta contribution (joint driver
    // only). Multiplied into d_fac(k_grid) when the block contributes to arm
    // k_arm's eta. Used for INLA `copy=` semantics: donor arms see one sigma,
    // the copy arm sees another. Empty = no per-arm scaling (every arm sees
    // d_fac(k_grid) directly). The single-arm driver does not call arm_scale.
    std::function<double(int /*k_arm*/, int /*k_grid*/)> arm_scale;

    // Per-arm per-row design weight on the block's eta contribution (joint
    // driver, INDEXED_SINGLE only). Multiplied into the block-local weight of
    // observation i in arm k_arm, so its eta contribution becomes
    //   eta_i += arm_scale(k_arm, k_grid) * d_fac(k_grid)
    //            * row_weight(i, k_arm) * x[start + idx(i, k_arm) - 1].
    // This is the areal-block analogue of the HSGP `svc_column` row-scaling:
    // an INLA f(cell, weight, copy=...) spatially-varying coefficient where
    // the same field is multiplied by a per-observation covariate (e.g. a
    // `time` column). The weight enters at exactly one layer (the block-local
    // weight resolved alongside idx); the gradient and z-block Hessian inherit
    // it through the chain rule (slope = d_eff * weight, Hessian accumulates
    // weight^2), so it must NOT also be folded into d_fac / arm_scale.
    // Empty = uniform weight 1 (byte-identical to a block without it).
    std::function<double(int /*i*/, int /*k_arm*/)> row_weight;

    // Adds block's prior Q at grid point k to (grad, H). x is the full latent
    // vector; the helper indexes into [start, start + size).
    std::function<void(DenseVec&, DenseMat&, const Rcpp::NumericVector&, int)>
        add_prior;

    // log p(x_block | theta_k). Summed across blocks gives the latent log-prior
    // contribution (the driver adds RE/beta log-priors on top).
    std::function<double(const Rcpp::NumericVector&, int)> log_prior;

    // Per-grid-point feasibility (e.g. proper-CAR PD check via log|D - rho*W|).
    // Return false to short-circuit the inner solve at this k with
    // log_marginal = -inf. May be empty (treated as always true).
    std::function<bool(int)> prep;

    // Optional sum-to-zero / soft centering, applied after each Newton step
    // on the block's sub-vector. Returns the mean offset that was applied
    // (single-arm callers ignore the return value; joint drivers use it to
    // shift per-arm intercepts so eta is preserved). Return 0 means no
    // centering happened. May be empty (no centering at all, e.g. AR1).
    std::function<double(Rcpp::NumericVector&)> center;

    // ----- Sparse-builder fields -----
    //
    // Every existing factory defaults to INDEXED_SINGLE + ADJACENCY, matching
    // the historical block registry shape (one block-local DOF per obs, sparse
    // adjacency-graph prior). Factories for SPDE / HSGP / NNGP / latent
    // factors / tgmrf override these in Stage 1.3.
    //
    // The dense `add_prior` field above is kept as the legacy path used by the
    // dense fallback (n_x < SPARSE_THRESHOLD). `add_prior_sparse` below is the
    // new path used at scale. Both must agree on values when both are present.

    BlockContribKind contrib_kind = BlockContribKind::INDEXED_SINGLE;
    PriorFillKind    prior_kind   = PriorFillKind::ADJACENCY;

    // INDEXED_MULTI only. Append (block-local 1-based index, weight) pairs
    // for obs i in arm k_arm. Caller passes a reusable thread-local scratch
    // vector — implementations should `clear()` then `emplace_back(...)`.
    // For INDEXED_SINGLE this is unused (use `idx` instead); for DENSE_BASIS
    // unused (use `basis_eval`).
    std::function<void(int /*i*/, int /*k_arm*/,
                       std::vector<std::pair<int,double>>& /*out*/)>
        obs_indices;

    // DENSE_BASIS only. Fill `out[0..size)` with the basis weights for obs i
    // in arm k_arm at outer-grid cell k_grid. E.g. HSGP fills
    // out[j] = Phi(s_i, j) * sqrt(S_j(theta_k_grid)). Caller owns the buffer
    // (sized to `size`); implementations write in place. k_grid selects the
    // per-cell state written by prep(k_grid) — the joint driver runs cells
    // concurrently, so implementations must not read a single shared cache.
    std::function<void(int /*i*/, int /*k_arm*/, int /*k_grid*/,
                       double* /*out*/)>
        basis_eval;

    // DENSE_BASIS only. Batched view of the per-arm basis matrix for the
    // SYRK / GEMM scatter path (Stage 2.1).
    //
    //   data           : row-major (N_k * m_per_arm) buffer of the RAW basis
    //                    Phi_k (NOT pre-scaled by sqrt_S); the scatter helper
    //                    applies sqrt_S on the fly so the buffer can be a
    //                    long-lived cache that does not change with theta.
    //   sqrt_S         : length-m_per_arm vector of spectral-density square
    //                    roots cached by the block's `prep` step (HSGP and
    //                    derivatives) or 1.0 placeholders for blocks whose
    //                    column scaling lives elsewhere. May be nullptr to
    //                    mean "no spectral scaling".
    //   N_k            : rows in the per-arm Phi
    //   m_per_arm      : columns in the per-arm Phi (= per-output basis dim)
    //   m_offset_in_block : the per-arm Phi's contribution lands at
    //                       [start + m_offset_in_block, + m_per_arm). For
    //                       single-output HSGP this is 0; for HSGP-MO it is
    //                       k_arm * m_per_arm (output-major layout).
    //
    // The scatter helper batches every obs across the arm in a single SYRK /
    // GEMM call. The sqrt_S pointer refers to cell k_grid's state written by
    // prep(k_grid); it stays valid for the duration of that cell's solve
    // (per-cell slot storage), and the scatter must hold it only across one
    // solve.
    //
    // Optional: when this callback is empty the per-obs `basis_eval` path is
    // used (current behavior pre-Stage 2.1). Both should agree on values.
    struct DenseBasisBatch {
        const double* data              = nullptr;
        const double* sqrt_S            = nullptr;
        int           N_k               = 0;
        int           m_per_arm         = 0;
        int           m_offset_in_block = 0;
    };
    std::function<DenseBasisBatch(int /*k_arm*/, int /*k_grid*/)>
        dense_basis_batch;

    // BILINEAR_FACTOR only. Return the pair (u_slot_global, lambda_slot_global)
    // for obs i in arm k_arm. Both indices are absolute (i.e. include the
    // block's start offset). Return any negative slot to skip the obs
    // (e.g. when obs_idx_per_arm returns 0/-1 for "no factor contribution
    // at this obs"). The scatter reads x[u_slot] and x[lambda_slot] to
    // compute the bilinear linearization weights at the current Newton
    // iterate.
    std::function<std::pair<int,int>(int /*i*/, int /*k_arm*/)>
        obs_factor_lambda;

    // Append (row, col) lower-triangle entries the prior contributes to the
    // joint H sparsity pattern. Indices are global in the joint latent x
    // (i.e. include `start` offsets). Called ONCE at fit-time, before any
    // Newton iteration; the resulting pattern is reused across all outer-
    // grid cells. May be empty for blocks whose prior contributes only to
    // the diagonal (which is always present via the data-induced fill).
    std::function<void(std::vector<std::pair<int,int>>& /*out*/)>
        add_prior_pattern;

    // Sparse-builder analogue of `add_prior`. Scatters the prior's gradient
    // and Hessian contributions into a SparseHessianBuilder rather than a
    // DenseMat. Same semantics as the dense version; only the H container
    // type differs. Required when contrib_kind != INDEXED_SINGLE or when the
    // joint driver chooses the sparse Newton path.
    std::function<void(SparseHessianBuilder& /*H*/, DenseVec& /*grad*/,
                       const Rcpp::NumericVector& /*x*/, int /*k_grid*/)>
        add_prior_sparse;
};

} // namespace tulpa

#endif // TULPA_LATENT_BLOCK_H
