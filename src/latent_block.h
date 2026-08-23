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

#include <stdexcept>
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

// One constant removed by a block's centering, and the coefficient it aliases
// with. `beta_offset` is relative to the arm's beta_start (0 = the intercept);
// `amount` is the removed value, which the driver adds to that coefficient
// after the block's own arm_scale / d_fac, so eta is preserved.
struct CenterFold {
    int    beta_offset = 0;
    double amount      = 0.0;
};

// The common case: a block seen uniformly by its observations removes one
// constant, which aliases with the arm intercept. Centres [start, start+length)
// in place and reports the fold.
inline std::vector<CenterFold> center_intercept(Rcpp::NumericVector& x,
                                                int start, int length) {
    if (length <= 0) return {};
    double mean = 0.0;
    for (int i = 0; i < length; i++) mean += x[start + i];
    mean /= length;
    for (int i = 0; i < length; i++) x[start + i] -= mean;
    return { CenterFold{0, mean} };
}

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

    // The ONLY way to invoke `obs_indices`, because it carries the scratch
    // contract: the output buffer is cleared HERE rather than by each
    // implementation. The buffer is `static thread_local` and reused across the
    // whole observation loop, so an implementation that appended instead of
    // replacing would not merely duplicate within one observation -- every
    // stale (index, weight) pair would be scattered into eta as though it
    // belonged to the current row, growing with position in the loop. That is a
    // silent wrong answer, not a crash.
    void fill_obs_indices(int i, int k_arm,
                          std::vector<std::pair<int, double>>& out) const {
        out.clear();
        obs_indices(i, k_arm, out);
    }

    // The ONLY way to read `d_fac`, so its contract lives in ONE place. `d_fac`
    // is REQUIRED: every block factory sets it unconditionally. A block that
    // omitted it would otherwise be silently amplitude-1.0 wherever a call site
    // guarded the read and an uncaught `std::bad_function_call` wherever one
    // did not, on the same block vector.
    double d_fac_at(int k_grid) const {
        if (!d_fac) {
            throw std::logic_error(
                "LatentBlock: d_fac is required and was not set by the block "
                "factory. It is the block's grid-dependent eta mixing "
                "coefficient; a block with no mixing sets it to a constant 1.");
        }
        return d_fac(k_grid);
    }

    // Per-arm linear scale on the block's eta contribution (joint driver
    // only). Multiplied into d_fac(k_grid) when the block contributes to arm
    // k_arm's eta. Used for INLA `copy=` semantics: donor arms see one sigma,
    // the copy arm sees another. Empty = no per-arm scaling (every arm sees
    // d_fac(k_grid) directly). The single-arm driver does not call arm_scale.
    std::function<double(int /*k_arm*/, int /*k_grid*/)> arm_scale;

    // Per-arm per-row design weight on the block's eta contribution, read on
    // every contribution kind by every walker through block_row_weight() below.
    // Multiplied into the block-local weight of observation i in arm k_arm, so
    // its eta contribution becomes
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

    // Optional sum-to-zero / soft centering, applied after each Newton step on
    // the block's sub-vector. Returns every constant that was removed together
    // with WHERE it belongs, so the drivers can shift the aliased coefficient
    // and leave eta unchanged. An empty vector means no centering happened; the
    // field itself may be empty (no centering at all, e.g. AR1).
    //
    // `beta_offset` is relative to the arm's beta_start, so 0 is the arm
    // intercept -- what a block contributing eta_i += x[...] uniformly aliases
    // with, and what every scalar block returns via center_intercept().
    //
    // A non-zero offset exists because that aliasing is not universal. A block
    // whose observations see it through a per-observation weight
    // (eta_i += X_{ia} * u_a[cell_i], as multivariate CAR and the varying-
    // coefficient blocks do) shifts eta along that covariate's column, not
    // uniformly, so its constant aliases with the COEFFICIENT on X_{.a} rather
    // than with the intercept. Folding such a constant into the intercept would
    // change eta instead of preserving it.
    std::function<std::vector<CenterFold>(Rcpp::NumericVector&)> center;

    // ----- Sparse-builder fields -----
    //
    // The defaults describe an areal block: one block-local DOF per
    // observation, reached through `idx`, with a sparse adjacency-graph prior.
    // The SPDE / HSGP / NNGP / latent-factor / tgmrf factories set their own.
    //
    // `add_prior` above is the dense route, taken when n_x < SPARSE_THRESHOLD;
    // `add_prior_sparse` below is the sparse one. Both must agree on values
    // when both are present.

    BlockContribKind contrib_kind = BlockContribKind::INDEXED_SINGLE;
    PriorFillKind    prior_kind   = PriorFillKind::ADJACENCY;

    // INDEXED_MULTI only. Fill (block-local 1-based index, weight) pairs
    // for obs i in arm k_arm. Call it through `fill_obs_indices()`, never
    // directly: that clears the caller's reusable thread-local scratch first,
    // so an implementation may assume an empty vector and just
    // `emplace_back(...)`.
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
    // SYRK / GEMM scatter path.
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
    // used instead. Both must agree on values.
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

// Does this block set force the sparse Newton path regardless of n_x?
//
// Two reasons a block cannot go through the dense scatter:
//   * contrib_kind != INDEXED_SINGLE -- the dense scatter resolves a block
//     only through `idx`, so it cannot see an INDEXED_MULTI block's per-obs
//     weights or a DENSE_BASIS block's basis row.
//   * a prior that scatters only into a SparseHessianBuilder (add_prior_sparse
//     set, dense add_prior empty) -- the dense path calls `add_prior` if it is
//     present and silently contributes NOTHING if it is not, which drops the
//     block's prior from H and from the mode it implies.
//
// Reading it off the blocks makes the requirement a property of the block
// rather than something each caller has to remember to pass as force_sparse.
inline bool blocks_require_sparse(const std::vector<LatentBlock>& blocks) {
    for (const auto& b : blocks) {
        if (b.contrib_kind != BlockContribKind::INDEXED_SINGLE) return true;
        if (b.add_prior_sparse && !b.add_prior) return true;
    }
    return false;
}

// The block's per-row design weight at observation i on arm k_arm, 1 where the
// block declares none. Every walker over the block list reads it through this
// one accessor: a weighted block whose contribution kind is INDEXED_MULTI or
// DENSE_BASIS is as weighted as an INDEXED_SINGLE one, and a walker that read
// the member on one branch only would fit the unweighted model with nothing to
// say so.
inline double block_row_weight(const LatentBlock& blk, int i, int k_arm) {
    return blk.row_weight ? blk.row_weight(i, k_arm) : 1.0;
}

// Which convention each block's reported field follows.
//
// TRUE where the block carries a post-hoc centerer: its levels are reported
// sum-to-zero, with the constant that was removed folded into an intercept, so
// the field and the intercept are a different split of the same eta. FALSE
// where the field is reported as the Newton mode found it.
//
// The rule is the prior's rank. An intrinsic prior (ICAR, BYM2's structured
// component, RW1 / RW2) has a null direction that the data cannot identify
// against the intercept, and centering picks the representative of it; a
// full-rank prior (proper CAR, AR1, IID) has none, so there is nothing to
// choose and the mode is reported unmoved. Recorded on the fit rather than
// left to be inferred from the block type, so comparing two fits of the same
// field -- a copy block against a non-copy one, a temporal field against a
// spatial one -- does not silently compare two splits.
inline Rcpp::LogicalVector block_center_flags(
    const std::vector<LatentBlock>& blocks
) {
    Rcpp::LogicalVector out((R_xlen_t)blocks.size());
    for (std::size_t b = 0; b < blocks.size(); b++) {
        out[(R_xlen_t)b] = static_cast<bool>(blocks[b].center);
    }
    return out;
}

} // namespace tulpa

#endif // TULPA_LATENT_BLOCK_H
