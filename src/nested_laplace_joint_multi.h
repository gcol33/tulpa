// nested_laplace_joint_multi.h
// Shared driver for joint multi-likelihood nested Laplace with one or more
// latent prior blocks.
//
// This is the joint analogue of run_multi_block_nested_laplace (single-arm,
// see nested_laplace_multi.h). For each outer-grid point k the inner Newton
// solves
//
//   eta_{k_arm,i} = X_{k_arm} beta_{k_arm} + RE_{k_arm}[g(i)]
//                 + Σ_b arm_scale_b(k_arm, k) * d_fac_b(k)
//                       * x[start_b + idx_b(i, k_arm) - 1]
//
//   grad/H from per-arm scatter (β/β, β/RE, RE/RE diagonal blocks
//          per arm, plus latent/{β, RE, latent} cross-terms per block)
//          + Σ_b add_prior_b(k)
//          + add_per_arm_beta_re_priors
//
//   center: each block's centerer applied to its sub-vector, with each
//           arm's first beta column shifted by
//           arm_scale_b(k_arm, k) * d_fac_b(k) * delta_b
//           to preserve eta after centering rank-deficient blocks.
//
//   log_prior: Σ_b log_prior_b(k) + log_prior_per_arm_re(x, parsed),
//              which carries every beta / RE term the gradient does
//
// Per-block prep is invoked once at grid point k before the inner solve. If
// any block reports infeasible (e.g. proper-CAR with rho outside the PD
// interval), the inner solve short-circuits with log_marginal = -inf.

#ifndef TULPA_NESTED_LAPLACE_JOINT_MULTI_H
#define TULPA_NESTED_LAPLACE_JOINT_MULTI_H

#include "coupled_scatter_plan.h"
#include "joint_hessian_pattern.h"
#include "laplace_core.h"
#include "laplace_family_link.h"
#include "laplace_newton_joint.h"
#include "laplace_newton_joint_sparse.h"
#include "laplace_profile.h"
#include "laplace_re_priors.h"
#include "latent_block.h"
#include "nested_laplace_grid.h"
#include "nested_laplace_joint_core.h"
#include "row_classes.h"
#include "scatter_dense_basis.h"
#include "scatter_indexed_cache.h"
#include "sparse_cholesky.h"
#include "sparse_hessian.h"
#include "tulpa/cell_coupling.h"
#include <Rcpp.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <memory>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// Initial capacity of the two chain scratch buffers. Pure allocation hints:
// both grow on demand and nothing depends on the value. One cell's rank-1
// self-cross touches at most its own rows, a row's chain at most one entry per
// latent block, so these are sized for the common case and never a bound.
inline constexpr int kCellChainScratchHint = 64;
inline constexpr int kRowChainScratchHint  = 32;

// Threshold below which a centering fold is skipped: the level the centerer
// removed is numerically zero, so folding it would only perturb the coefficient
// it lands in. One definition, so the dense, sparse and batched centering paths
// cannot drift apart.
inline constexpr double kCenterFoldMinAmount = 1e-15;

// Apply every block's centerer to x and compensate each arm's aliased
// coefficient so eta is preserved: arm k's aliased beta column absorbs
// arm_scale_b(k_arm, k_grid) * d_fac_b(k_grid) * amount for each constant the
// centerer removed from x[block]. Offset 0 is the intercept, the column an
// unweighted block aliases with; a weighted block reports the column of the
// covariate it rides on. `d_fac_at_block(b)` supplies block b's amplitude at
// the cell being centred (the dense driver reads a per-cell cache, the sparse
// and batched drivers call the block).
template <typename DFacFn>
inline void center_joint_blocks(
    Rcpp::NumericVector&            x,
    const std::vector<LatentBlock>& blocks,
    const std::vector<ParsedArm>&   parsed,
    int                             n_arms,
    int                             k_grid,
    DFacFn                          d_fac_at_block
) {
    const int n_blocks = static_cast<int>(blocks.size());
    for (int b = 0; b < n_blocks; b++) {
        if (!blocks[b].center) continue;
        const double dfac = d_fac_at_block(b);
        for (const auto& fold : blocks[b].center(x)) {
            if (std::abs(fold.amount) < kCenterFoldMinAmount) continue;
            for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                if (parsed[k_arm].p == 0) continue;
                if (fold.beta_offset < 0 ||
                    fold.beta_offset >= parsed[k_arm].p) continue;
                const double s = blocks[b].arm_scale
                                 ? blocks[b].arm_scale(k_arm, k_grid)
                                 : 1.0;
                x[parsed[k_arm].beta_start + fold.beta_offset] +=
                    s * dfac * fold.amount;
            }
        }
    }
}

// Joint log-prior at x: the per-arm beta / RE priors plus every block's own
// log-prior at the grid cell.
inline double log_prior_joint_blocks(
    const Rcpp::NumericVector&      x,
    const std::vector<LatentBlock>& blocks,
    const std::vector<ParsedArm>&   parsed,
    int                             k_grid
) {
    double lp = log_prior_per_arm_re(x, parsed);
    for (const auto& b : blocks) {
        if (b.log_prior) lp += b.log_prior(x, k_grid);
    }
    return lp;
}

// Row i of arm k_arm through the prior blocks: every (block, global latent
// index, weight) entry of d eta_i / d x, in block order. The weight is the
// block amplitude `d_eff_cache[b]` (arm_scale * d_fac at the cell) times the
// row's design weight (block_row_weight, applied once on every kind) times the
// block-local weight the kind carries: the obs_indices weight (INDEXED_MULTI),
// the basis value at cell `k_grid` (DENSE_BASIS), or the paired loading / factor
// value read off `x` (BILINEAR_FACTOR, whose eta term is the product
// d * u * lambda). Zero-weight entries are not emitted, and `skip(b)` drops a
// block before any of its per-row work runs.
//
// The one walk behind every joint consumer of a row's block loadings: the
// coupled row scatter (collect_coupled_row_latents), the per-observation sparse
// scatter, and the per-row predictive variance (joint_row_loadings). The eta
// accumulators form the same sum as values rather than as a Jacobian.
template <typename Skip, typename Emit>
inline void for_each_row_block_latent(
    int                                 i,
    int                                 k_arm,
    int                                 k_grid,
    const std::vector<LatentBlock>&     blocks,
    const std::vector<double>&          d_eff_cache,
    const double*                       x,
    std::vector<std::pair<int,double>>& multi_scratch,
    std::vector<double>&                basis_scratch,
    Skip&&                              skip,
    Emit&&                              emit
) {
    const int B = static_cast<int>(blocks.size());
    for (int b = 0; b < B; b++) {
        const LatentBlock& blk = blocks[b];
        const double d_b = d_eff_cache[b] * block_row_weight(blk, i, k_arm);
        // A field_coef = 0 arm, a rho = 0 BYM2 component or a zero row weight
        // contributes nothing to this row.
        if (d_b == 0.0) continue;
        if (skip(b)) continue;
        switch (blk.contrib_kind) {
        case BlockContribKind::INDEXED_SINGLE: {
            if (!blk.idx) break;
            const int l = blk.idx(i, k_arm);
            if (l > 0 && l <= blk.size) emit(b, blk.start + l - 1, d_b);
            break;
        }
        case BlockContribKind::INDEXED_MULTI: {
            if (!blk.obs_indices) break;
            blk.fill_obs_indices(i, k_arm, multi_scratch);
            for (const auto& jw : multi_scratch) {
                const int l = jw.first;
                if (l <= 0 || l > blk.size) continue;
                const double w = d_b * jw.second;
                if (w != 0.0) emit(b, blk.start + l - 1, w);
            }
            break;
        }
        case BlockContribKind::DENSE_BASIS: {
            if (!blk.basis_eval) break;
            if (static_cast<int>(basis_scratch.size()) < blk.size)
                basis_scratch.resize(blk.size);
            blk.basis_eval(i, k_arm, k_grid, basis_scratch.data());
            for (int j = 0; j < blk.size; j++) {
                const double w = basis_scratch[j] * d_b;
                if (w != 0.0) emit(b, blk.start + j, w);
            }
            break;
        }
        case BlockContribKind::BILINEAR_FACTOR: {
            if (!blk.obs_factor_lambda || x == nullptr) break;
            // obs_factor_lambda owns both bounds: it returns {-1, -1} for a row
            // outside the factor's own latent range.
            const auto slots = blk.obs_factor_lambda(i, k_arm);
            if (slots.first < 0 || slots.second < 0) break;
            const double w_u      = x[slots.second] * d_b;
            const double w_lambda = x[slots.first]  * d_b;
            if (w_u != 0.0)      emit(b, slots.first,  w_u);
            if (w_lambda != 0.0) emit(b, slots.second, w_lambda);
            break;
        }
        }
    }
}

// Resolve the active (latent index, chain weight) entries one arm row contributes
// through the prior blocks, for the block kinds the coupled per-cell scatter
// supports:
//   * INDEXED_SINGLE -- one latent dof per row (an areal/ICAR field, optionally
//     SVC-weighted via `row_weight`);
//   * INDEXED_MULTI  -- several latent dofs per row (a separable-MCAR block's p
//     coupled fields, weights carried in `obs_indices`).
// `d_eff_cache[b]` is the per-block arm_scale * d_fac amplitude (a zero entry
// skips the block). Entries are APPENDED to out_idx / out_w (the caller clears
// them). Single source of truth for the active-latent resolution used by both
// `scatter_one_arm_row_dense` (gradient + Hessian) and `build_arm_row_chain`
// (cross-arm Hessian chain); the sparse row scatter reads the same dofs off the
// coupled plan (coupled_cell_arm_dofs / build_arm_indexed_cache resolve them
// from the same blocks). DENSE_BASIS / BILINEAR_FACTOR are not coupled-scatter
// kinds (no coupled family uses them) and are skipped.
//
// PRECONDITION: the appended entries resolve to DISTINCT global latent indices.
// Block ranges are laid out disjointly (each factory returns start + size) and
// every INDEXED_MULTI producer emits one slot per row per field, so it holds by
// construction. The dense row scatter sums all (a, b) pairs and writes both
// directions while the sparse row body sums the lower triangle once; the two
// agree only for distinct indices, since a repeated index would gain
// 2 * d_a * d_b on the dense diagonal against d_a * d_b on the sparse one.
inline void collect_coupled_row_latents(
    int                              i,
    int                              k_arm,
    const std::vector<LatentBlock>&  blocks,
    const std::vector<double>&       d_eff_cache,
    std::vector<int>&                out_idx,
    std::vector<double>&             out_w
) {
    // Per-thread scratch via a constant-initialized POD pointer: a lazily
    // dynamic-initialized `thread_local std::vector` inside an OpenMP region
    // corrupts the heap under the mingw toolchain (init guard + thread-atexit
    // destructor registration). The pointee intentionally leaks per thread.
    static thread_local std::vector<std::pair<int,double>>* multi_scratch_p = nullptr;
    if (!multi_scratch_p) multi_scratch_p = new std::vector<std::pair<int,double>>();
    // Never written: the two kinds that read a basis or x are skipped below.
    std::vector<double> no_basis;
    for_each_row_block_latent(
        i, k_arm, /*k_grid=*/0, blocks, d_eff_cache, /*x=*/nullptr,
        *multi_scratch_p, no_basis,
        [&](int b) {
            const BlockContribKind kind = blocks[b].contrib_kind;
            return kind == BlockContribKind::DENSE_BASIS ||
                   kind == BlockContribKind::BILINEAR_FACTOR;
        },
        [&](int, int latent, double w) {
            out_idx.push_back(latent);
            out_w.push_back(w);
        });
}

// d eta / d x for every row of every arm at latent point `x` and outer cell
// `k_grid`, packed arm-major into `L` (arm 0's rows first). Row r's loading
// vector is its fixed-effect design row, its random-effect indicator and its
// block entries (for_each_row_block_latent). `d_eff[b][k_arm]` is the block
// amplitude at the cell. The linear predictor is linear in every coordinate but
// a bilinear factor's, so at `x` this is the Jacobian the inner Laplace's
// Gaussian maps to a variance.
inline void joint_row_loadings(
    const double*                           x,
    const std::vector<JointArm>&            arms,
    const std::vector<ParsedArm>&           parsed,
    const std::vector<LatentBlock>&         blocks,
    int                                     k_grid,
    const std::vector<std::vector<double>>& d_eff,
    RowLoadings&                            L
) {
    L.clear();
    const int n_arms = static_cast<int>(arms.size());
    const int B      = static_cast<int>(blocks.size());
    std::vector<double> d_eff_arm(B, 0.0);
    std::vector<std::pair<int,double>> multi_scratch;
    std::vector<double> basis_scratch;
    for (int k_arm = 0; k_arm < n_arms; k_arm++) {
        const ParsedArm& pa = parsed[k_arm];
        for (int b = 0; b < B; b++) d_eff_arm[b] = d_eff[b][k_arm];
        for (int i = 0; i < arms[k_arm].N; i++) {
            for (int j = 0; j < pa.p; j++) L.push(pa.beta_start + j, pa.X(i, j));
            if (pa.n_re_groups > 0) {
                const int g = static_cast<int>(pa.re_idx[i]) - 1;
                if (g >= 0 && g < pa.n_re_groups) L.push(pa.re_start + g, 1.0);
            }
            for_each_row_block_latent(
                i, k_arm, k_grid, blocks, d_eff_arm, x,
                multi_scratch, basis_scratch,
                [](int) { return false; },
                [&](int, int latent, double w) { L.push(latent, w); });
            L.end_row();
        }
    }
}

// Scatter the gradient and Fisher curvature of ONE row of arm k_arm
// (per-obs path: row = obs index i; per-cell path: row = a row from a
// coupled cell's CellDerivs) into the joint (grad, H). Single source of
// truth for both the per-obs scatter (called once per obs with the
// family's `gh = arm_grad_hess(...)`) and the cell-coupling per-cell
// scatter (called once per cell row with the spec's
// `out.arm_grad[kk][j]` / `out.arm_neg_hess_diag[kk][j]`).
//
// `d_eff_cache[b]` carries `arm_scale_b(k_arm, k_grid) * d_fac_b(k_grid)`
// (resolved once per scatter call by the caller); a zero entry skips
// the block entirely (e.g. `field_coef = 0` arm, or `rho = 0` BYM2
// phi-component). `active_idx` / `active_d` are caller-owned scratch
// reused across rows.
inline void scatter_one_arm_row_dense(
    int                              i,
    double                           g_row,
    double                           H_row,
    const ParsedArm&                 pa,
    int                              k_arm,
    const std::vector<LatentBlock>&  blocks,
    const std::vector<double>&       d_eff_cache,
    DenseVec&                        grad,
    DenseMat&                        H,
    std::vector<int>&                active_idx,
    std::vector<double>&             active_d
) {
    const int p_k    = pa.p;
    const int n_re_k = pa.n_re_groups;
    const int bstart = pa.beta_start;
    const int rstart = pa.re_start;

    int g_re = -1;
    if (n_re_k > 0) {
        int gi = static_cast<int>(pa.re_idx[i]) - 1;
        if (gi >= 0 && gi < n_re_k) g_re = rstart + gi;
    }

    // Resolve active latent dofs for row i (INDEXED_SINGLE one-per-row +
    // INDEXED_MULTI several-per-row). -1 / out-of-range from a block means "this
    // row doesn't see this block"; a block with d_eff == 0 (field_coef = 0 arm,
    // rho = 0 BYM2 phi-component, etc.) is skipped. Shared resolver so the
    // gradient/Hessian scatter and the cross-arm chain see the same dofs.
    active_idx.clear();
    active_d.clear();
    collect_coupled_row_latents(i, k_arm, blocks, d_eff_cache, active_idx, active_d);
    const int A = static_cast<int>(active_idx.size());

    // β block: gradient + diagonal-block Hessian + cross with RE and
    // every active latent block.
    for (int j = 0; j < p_k; j++) {
        const double Xij = pa.X(i, j);
        grad[bstart + j] += g_row * Xij;
        for (int l = 0; l < p_k; l++) {
            H[bstart + j][bstart + l] += H_row * Xij * pa.X(i, l);
        }
        if (g_re >= 0) {
            H[bstart + j][g_re] += H_row * Xij;
            H[g_re][bstart + j] += H_row * Xij;
        }
        for (int a = 0; a < A; a++) {
            H[bstart + j][active_idx[a]] += H_row * Xij * active_d[a];
            H[active_idx[a]][bstart + j] += H_row * Xij * active_d[a];
        }
    }

    // RE block: gradient + diagonal + cross with active latent indices.
    if (g_re >= 0) {
        grad[g_re] += g_row;
        H[g_re][g_re] += H_row;
        for (int a = 0; a < A; a++) {
            H[g_re][active_idx[a]] += H_row * active_d[a];
            H[active_idx[a]][g_re] += H_row * active_d[a];
        }
    }

    // Latent x latent block (intra-block + inter-block). Includes both
    // the diagonal at (idx_a, idx_a) and the off-diagonal (idx_a, idx_b)
    // for a != b.
    for (int a = 0; a < A; a++) {
        grad[active_idx[a]] += g_row * active_d[a];
        for (int b = 0; b < A; b++) {
            H[active_idx[a]][active_idx[b]] +=
                H_row * active_d[a] * active_d[b];
        }
    }
}

// Per-observation latent-block scatter for one arm at one grid point.
//
// Variable-length analogue of the single-arm multi-block scatter
// (accumulate_latent_cross_terms in nested_laplace_multi.h), with the
// β/β and β/RE diagonal blocks evaluated *per arm* using ParsedArm
// offsets so multiple likelihood arms can share the same latent vector.
// Each block's eta contribution carries an optional per-arm scaling
// factor (arm_scale) for INLA `copy=` semantics.
//
// Per-row work is factored into `scatter_one_arm_row_dense()` so the
// cell-coupling per-cell branch can share it; this function only owns
// the d_eff cache + the per-obs loop over (i, gh.grad, gh.neg_hess).
inline void scatter_arm_obs_joint_multi(
    const Rcpp::NumericVector& /*x*/,
    const Rcpp::NumericVector&    eta,
    const ParsedArm&              pa,
    const JointArm&               arm,
    const ArmSpecView&            view,
    int                           k_arm,
    const std::vector<LatentBlock>& blocks,
    int                           k_grid,
    DenseVec&                     grad,
    DenseMat&                     H
) {
    const int B = static_cast<int>(blocks.size());

    std::vector<double> d_eff_cache(B);
    for (int b = 0; b < B; b++) {
        double s = blocks[b].arm_scale
                    ? blocks[b].arm_scale(k_arm, k_grid)
                    : 1.0;
        d_eff_cache[b] = s * blocks[b].d_fac_at(k_grid);
    }

    std::vector<int>    active_idx;
    std::vector<double> active_d;
    active_idx.reserve(B);
    active_d.reserve(B);

    for (int i = 0; i < arm.N; i++) {
        auto gh = arm_grad_hess(view, i, eta[i]);
        scatter_one_arm_row_dense(
            i, gh.grad, gh.neg_hess,
            pa, k_arm, blocks, d_eff_cache,
            grad, H, active_idx, active_d
        );
    }
}

// ============================================================================
// Cell-coupling per-cell branch (dense path).
//
// When the joint fit registers a non-separable `CellCouplingSpec` (one
// whose `arm_ids()` lists at least one arm), the per-obs scatter is
// skipped for the coupled arms (the outer `scatter_joint` closure
// guards on `arms[k_arm].coupled`) and the per-cell branch below is
// fired once per call. For each cell, it builds CellEtas / CellResponse
// / CellDerivs views, dispatches `spec->evaluate_cell()`, and scatters
// the per-arm row derivatives through the same `scatter_one_arm_row_dense()`
// helper the per-obs path uses. Single source of truth for the per-row
// (β, RE, latent) bookkeeping.
//
// `cell_rows[kk][c]` = row indices (0-based, into arm.N) for cell c of
// the kk-th coupled arm (where kk indexes `coupled_arms`). Pre-computed
// once per fit by `build_cell_rows_from_arms()`.
//
// The branch reads the whole `CellDerivs` contract: `arm_grad` and
// `arm_neg_hess_diag` per row, the dense cross-arm block `arm_cross_hess[kk][ll]`
// for every pair the spec declares, and the rank-1 self-cross descriptor a spec
// may supply in place of its own (kk, kk) block. The dense and sparse wrappers
// share this body; the sparse one additionally splits the cell loop across idle
// team threads.
// ============================================================================

// Evaluate the cell-coupling spec's per-cell log-density sum at the current
// etas, discarding derivatives. Used by the log-lik functor during the Newton
// line search; the scatter pass runs the same call with the derivatives kept,
// which keeps the line-search objective consistent with the scatter at the cost
// of a second evaluate per accepted step.
//
// The call sets `grad_only`, so a spec may skip its curvature work here, and it
// still hands over the zero-filled `arm_neg_hess_diag` buffers and an outer
// `arm_cross_hess` array the contract promises -- every inner block null, which
// is the documented spelling of "this pair contributes nothing at this cell".
// `phi_override` (optional, length n_coupled): per-coupled-arm dispersion to use
// in place of the shared `arms[k].phi`. The threaded sparse outer-grid driver
// passes a per-thread snapshot taken under the phi-sync critical, because the
// gridded coupled-arm dispersion (e.g. the beta precision on `phi.grid.pos`) is
// rewritten in `arms` per cell by a concurrent thread's `prep_at_grid`; reading
// the shared `arms[k].phi` lock-free here would race it.
// nullptr keeps the direct `arms[k].phi` read for the serial / dense callers.
// What a caller supplies per coupled arm besides its eta. The single-species
// path reads it off the JointArm; the batched path reads species s's slice of
// BatchArmBuffers. Everything else about the cell loop is the same, which is
// why the loop below takes this rather than being written twice.
struct CellArmResponse {
    const double* y        = nullptr;
    const int*    n_trials = nullptr;
    double        phi      = 0.0;
};

// Per-solve views over the coupled arms, and the per-cell slices of them.
//
// Both walkers over the coupled cells -- the objective-only evaluation and the
// scatter -- need the same two things: per-arm pointers that are stable across
// cells (eta, y, n_trials, family, phi) and per-arm slices that are rebuilt at
// every cell (the row list, its length, and zeroed gradient / curvature
// buffers). Written out twice they were line for line identical, which is one
// definition of the CellEtas / CellResponse / CellDerivs contract per caller.
//
// `bind_cell` grows each buffer monotonically and zeroes only the prefix in
// use, so a run over cells of varying size allocates at the largest cell seen
// and not once per cell.
struct CoupledArmViews {
    int n_coupled = 0;

    // Stable across cells.
    std::vector<const double*> eta_ptr;
    std::vector<const double*> y_ptr;
    std::vector<const int*>    n_trials_ptr;
    std::vector<std::string>   family_holder;   // owns what family_ptr points at
    std::vector<const char*>   family_ptr;
    std::vector<double>        phi;

    // Rebuilt per cell by bind_cell().
    std::vector<int>                 row_count;
    std::vector<const int*>          rows_ptr;
    std::vector<std::vector<double>> grad_buf;
    std::vector<std::vector<double>> neg_hess_diag_buf;
    std::vector<double*>             grad_ptr;
    std::vector<double*>             neg_hess_diag_ptr;

    template <class ArmResponseFn>
    CoupledArmViews(const std::vector<int>&                 coupled_arms,
                    const std::vector<JointArm>&            arms,
                    const std::vector<Rcpp::NumericVector>& etas,
                    ArmResponseFn&&                         arm_response)
        : n_coupled((int)coupled_arms.size()),
          eta_ptr(n_coupled), y_ptr(n_coupled), n_trials_ptr(n_coupled),
          family_holder(n_coupled), family_ptr(n_coupled), phi(n_coupled),
          row_count(n_coupled), rows_ptr(n_coupled),
          grad_buf(n_coupled), neg_hess_diag_buf(n_coupled),
          grad_ptr(n_coupled), neg_hess_diag_ptr(n_coupled) {
        for (int kk = 0; kk < n_coupled; kk++) {
            const int k = coupled_arms[kk];
            const CellArmResponse r = arm_response(kk, k);
            eta_ptr[kk]       = REAL(etas[k]);
            y_ptr[kk]         = r.y;
            n_trials_ptr[kk]  = r.n_trials;
            family_holder[kk] = arms[k].family;
            family_ptr[kk]    = family_holder[kk].c_str();
            phi[kk]           = r.phi;
        }
    }

    void bind_cell(int c,
                   const std::vector<std::vector<std::vector<int>>>& cell_rows) {
        for (int kk = 0; kk < n_coupled; kk++) {
            const int rc = (int)cell_rows[kk][c].size();
            row_count[kk] = rc;
            rows_ptr[kk]  = cell_rows[kk][c].data();
            if ((int)grad_buf[kk].size() < rc) {
                grad_buf[kk].assign(rc, 0.0);
                neg_hess_diag_buf[kk].assign(rc, 0.0);
            } else {
                std::fill(grad_buf[kk].begin(), grad_buf[kk].begin() + rc, 0.0);
                std::fill(neg_hess_diag_buf[kk].begin(),
                          neg_hess_diag_buf[kk].begin() + rc, 0.0);
            }
            grad_ptr[kk]          = grad_buf[kk].data();
            neg_hess_diag_ptr[kk] = neg_hess_diag_buf[kk].data();
        }
    }

    CellEtas etas_view() {
        CellEtas v;
        v.arm_eta_ptr   = eta_ptr.data();
        v.arm_rows      = rows_ptr.data();
        v.arm_row_count = row_count.data();
        v.n_arms_       = n_coupled;
        return v;
    }

    CellResponse response_view() {
        CellResponse v;
        v.arm_y         = y_ptr.data();
        v.arm_n_trials  = n_trials_ptr.data();
        v.arm_family    = family_ptr.data();
        v.arm_phi       = phi.data();
        v.arm_rows      = rows_ptr.data();
        v.arm_row_count = row_count.data();
        v.n_arms_       = n_coupled;
        return v;
    }
};

// The arm's own response, with `phi_override` (the batched multi-species path's
// per-arm dispersion) replacing the arm's when supplied. The single-response
// shape both coupled walkers take.
inline CellArmResponse coupled_arm_own_response(const JointArm& arm, int kk,
                                                const double* phi_override) {
    CellArmResponse r;
    r.y        = arm.y.size() > 0 ? REAL(arm.y) : nullptr;
    r.n_trials = arm.n_trials.size() > 0 ? INTEGER(arm.n_trials) : nullptr;
    r.phi      = phi_override ? phi_override[kk] : arm.phi;
    return r;
}

// The cell-coupling log-likelihood at the given per-arm etas, summed over
// cells. `arm_response(kk, k)` returns arm k's response for this evaluation.
//
// The cells are summed over the coupled scatter's fixed chunk partition
// (coupled_chunk_count): each chunk in cell order, then the chunk sums in chunk
// order, so the value is the same at every thread count and the chunks can run
// on whatever threads are idle (run_coupled_chunks; `n_threads` is the team it
// may open outside a parallel region).
//
// Derivatives are discarded here, so the spec may skip its curvature work; the
// zero-filled diagonal buffers and the outer cross array are still supplied,
// which is what the CellDerivs contract promises.
template <class ArmResponseFn>
inline double eval_cell_coupling_log_lik_impl(
    const CellCouplingSpec&                       spec,
    const std::vector<int>&                       coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int                                           n_cells,
    const std::vector<JointArm>&                  arms,
    const std::vector<Rcpp::NumericVector>&       etas,
    ArmResponseFn&&                               arm_response,
    int                                           n_threads
) {
    const int n_coupled = (int)coupled_arms.size();
    if (n_coupled == 0 || n_cells == 0) return 0.0;

    const int n_chunks = coupled_chunk_count(n_cells);
    std::vector<double> chunk_ll(n_chunks, 0.0);
    run_coupled_chunks(n_chunks, n_threads, spec.thread_safe(), [&](int ch) {
        CoupledArmViews views(coupled_arms, arms, etas, arm_response);

        // Outer cross-Hessian array with every inner block null. A spec is
        // required to test the INNER pointer only, so an objective-only call
        // supplies the outer array rather than a null one.
        std::vector<double*>        cross_hess_null_inner(n_coupled, nullptr);
        std::vector<double* const*> cross_hess_outer(
            n_coupled, cross_hess_null_inner.data());

        double total = 0.0;
        const int c_hi = coupled_chunk_lo(ch + 1, n_chunks, n_cells);
        for (int c = coupled_chunk_lo(ch, n_chunks, n_cells); c < c_hi; c++) {
            views.bind_cell(c, cell_rows);
            CellEtas     etas_view = views.etas_view();
            CellResponse y_view    = views.response_view();
            CellDerivs out;
            out.arm_grad           = views.grad_ptr.data();
            out.arm_neg_hess_diag  = views.neg_hess_diag_ptr.data();
            out.arm_cross_hess     = cross_hess_outer.data();
            out.arm_row_count      = views.row_count.data();
            out.n_arms_            = n_coupled;
            out.grad_only          = true;
            total += spec.evaluate_cell(c, etas_view, y_view, out);
        }
        chunk_ll[ch] = total;
    });

    double total = 0.0;
    for (int ch = 0; ch < n_chunks; ch++) total += chunk_ll[ch];
    return total;
}

// Single-species entry: the response is the arm's own, with `phi_override`
// replacing the arm's dispersion when supplied.
inline double eval_cell_coupling_log_lik(
    const CellCouplingSpec&                       spec,
    const std::vector<int>&                       coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int                                           n_cells,
    const std::vector<JointArm>&                  arms,
    const std::vector<Rcpp::NumericVector>&       etas,
    const double*                                 phi_override,
    int                                           n_threads
) {
    return eval_cell_coupling_log_lik_impl(
        spec, coupled_arms, cell_rows, n_cells, arms, etas,
        [&](int kk, int k) {
            return coupled_arm_own_response(arms[k], kk, phi_override);
        },
        n_threads);
}

// Per-cell row index inversion. For each coupled arm kk (= index into
// `coupled_arms`), `cell_rows[kk][c]` lists the 0-based rows of
// arms[coupled_arms[kk]] that belong to cell c. Built once per fit at
// the top of the joint driver. Returns the number of cells inferred
// from the max of all coupled arms' `cell_obs_map` entries.
inline int build_cell_rows_from_arms(
    const std::vector<JointArm>&                  arms,
    const std::vector<int>&                       coupled_arms,
    std::vector<std::vector<std::vector<int>>>&   cell_rows_out
) {
    cell_rows_out.clear();
    if (coupled_arms.empty()) return 0;
    int n_cells = 0;
    for (int k : coupled_arms) {
        const Rcpp::IntegerVector& m = arms[k].cell_obs_map;
        for (int i = 0; i < (int)m.size(); i++) {
            if (m[i] > n_cells) n_cells = m[i];
        }
    }
    cell_rows_out.assign(coupled_arms.size(),
                         std::vector<std::vector<int>>(n_cells));
    for (size_t kk = 0; kk < coupled_arms.size(); kk++) {
        int k = coupled_arms[kk];
        const Rcpp::IntegerVector& m = arms[k].cell_obs_map;
        for (int i = 0; i < (int)m.size(); i++) {
            int c = m[i] - 1;
            if (c >= 0 && c < n_cells) cell_rows_out[kk][c].push_back(i);
        }
    }
    return n_cells;
}

// One (idx, weight) entry in an eta -> joint-vector chain for a single
// (arm, row) of a coupled cell. The within-row scatter writes
// `H_row * outer(chain, chain)` into the joint H; the cross-arm scatter
// writes `Hkl * (chain_k outer chain_l + transpose)` for a (k_row, l_row)
// pair given the spec's `arm_cross_hess[kk][ll][j*Nl + m]`.
// Chain-entry block group, matching the block order the per-row sparse/dense
// scatter loops process (beta, then RE, then latent). Used by the batched
// within-row slot cache to reproduce the per-row helper's exact multiply
// association (same-group pair: (H_row * w_a) * w_b; cross-group pair, where the
// earlier block owns the oracle's outer loop: (H_row * w_b) * w_a).
enum ChainGroup : int { CHAIN_BETA = 0, CHAIN_RE = 1, CHAIN_LATENT = 2 };

struct ArmRowChainEntry {
    int    idx;  // global index into the joint latent vector
    double w;    // chain weight: X(j, a) for beta, 1.0 for RE, d_eff for latent
    int    grp;  // ChainGroup: beta / RE / latent (block-iteration order)
};

// Resolve the eta -> joint-vector chain for arm `pa` row `j` at grid point
// `k_grid`. Matches the contract of `scatter_one_arm_row_{dense,sparse}`
// (INDEXED_SINGLE + INDEXED_MULTI blocks, via collect_coupled_row_latents); the
// entries are exactly the joint dofs that the per-row helper would write nonzero
// contributions for in its `H_row * outer(chain, chain)` term.
inline void build_arm_row_chain(
    int                              j,
    const ParsedArm&                 pa,
    int                              k_arm,
    const std::vector<LatentBlock>&  blocks,
    const std::vector<double>&       d_eff_cache,
    std::vector<ArmRowChainEntry>&   out_chain
) {
    out_chain.clear();
    for (int a = 0; a < pa.p; a++) {
        out_chain.push_back({pa.beta_start + a, pa.X(j, a), CHAIN_BETA});
    }
    if (pa.n_re_groups > 0) {
        int gi = static_cast<int>(pa.re_idx[j]) - 1;
        if (gi >= 0 && gi < pa.n_re_groups) {
            out_chain.push_back({pa.re_start + gi, 1.0, CHAIN_RE});
        }
    }
    // Active latent dofs (INDEXED_SINGLE + INDEXED_MULTI), via the shared
    // resolver so the chain matches the (idx, weight) entries the gradient /
    // Hessian scatter writes for this row. POD-pointer TLS for the same
    // mingw/OpenMP reason as collect_coupled_row_latents' scratch.
    static thread_local std::vector<int>*    lat_idx_p = nullptr;
    static thread_local std::vector<double>* lat_w_p   = nullptr;
    if (!lat_idx_p) lat_idx_p = new std::vector<int>();
    if (!lat_w_p)   lat_w_p   = new std::vector<double>();
    std::vector<int>&    lat_idx = *lat_idx_p;
    std::vector<double>& lat_w   = *lat_w_p;
    lat_idx.clear();
    lat_w.clear();
    collect_coupled_row_latents(j, k_arm, blocks, d_eff_cache, lat_idx, lat_w);
    for (std::size_t t = 0; t < lat_idx.size(); ++t) {
        out_chain.push_back({lat_idx[t], lat_w[t], CHAIN_LATENT});
    }
}

// Cross-chain scatter (dense). Adds `Hkl * (chain_k chain_l^T + transpose)`
// to the joint H. The symmetric-matrix entry at (a, b) is
// `Hkl * (w_k(a) * w_l(b) + w_l(a) * w_k(b))`; iterating chain_k x chain_l
// once and writing `val` to both (a, b) and (b, a) reproduces that for all
// four cases (a-only in chain_k, b-only in chain_l, shared dofs, diagonal).
// For shared diagonal a == b, the two writes to the same dense cell sum
// to 2*val, which is the correct symmetric value.
inline void scatter_cross_chain_dense(
    double                                Hkl,
    const std::vector<ArmRowChainEntry>&  chain_k,
    const std::vector<ArmRowChainEntry>&  chain_l,
    DenseMat&                             H
) {
    if (Hkl == 0.0) return;
    for (const auto& e_k : chain_k) {
        for (const auto& e_l : chain_l) {
            double val = Hkl * e_k.w * e_l.w;
            H[e_k.idx][e_l.idx] += val;
            H[e_l.idx][e_k.idx] += val;
        }
    }
}

// ============================================================================
// Rank-1 self-cross fast path.
//
// When a coupled arm's (k, k) off-diagonal cross-Hessian is the symmetric
// rank-1 `coef * v v^T` over the cell's rows (the all-undetected occupancy
// mixture: the density depends on the rows only through the scalar
// P0 = prod(1 - p), so every cross-row second derivative factors through it),
// the whole block collapses in joint-dof space to a single `coef * u u^T` with
//     u = sum_r v[r] * chain(row_r),
// because each row's eta is linear in the joint dofs (chain(row_r)). The spec
// folds the term's own eta-diagonal `coef * v[r]^2` into arm_neg_hess_diag, so
// the full u u^T (including its dof-diagonal) is scattered here. This replaces
// the O(rc^2) dense arm_cross_hess[k][k] loop with O(rc) chain accumulation +
// one small outer product over the (few) joint dofs the arm's rows touch --
// the same dof pairs the dense path would write, so the joint Hessian pattern
// already covers them.
// ============================================================================

// Accumulate u = sum_r vrow[r] * chain(row_r) for arm k's `rc` rows, merged so
// each joint dof appears once. `chain_scratch` is reused per row; the merged
// (idx, weight) entries land in `u_out`.
inline void accumulate_self_rank1_u(
    const double*                    vrow,
    int                              rc,
    const int*                       rows,
    const ParsedArm&                 pa,
    int                              k_arm,
    const std::vector<LatentBlock>&  blocks,
    const std::vector<double>&       d_eff_cache,
    std::vector<ArmRowChainEntry>&   chain_scratch,
    std::vector<ArmRowChainEntry>&   u_out
) {
    u_out.clear();
    for (int r = 0; r < rc; r++) {
        const double vr = vrow[r];
        if (vr == 0.0) continue;
        build_arm_row_chain(rows[r], pa, k_arm, blocks, d_eff_cache, chain_scratch);
        for (const auto& e : chain_scratch) {
            u_out.push_back({e.idx, vr * e.w, e.grp});
        }
    }
    if (u_out.empty()) return;
    std::sort(u_out.begin(), u_out.end(),
              [](const ArmRowChainEntry& a, const ArmRowChainEntry& b) {
                  return a.idx < b.idx;
              });
    std::size_t w = 0;
    for (std::size_t r = 1; r < u_out.size(); r++) {
        if (u_out[r].idx == u_out[w].idx) {
            u_out[w].w += u_out[r].w;
        } else {
            ++w;
            u_out[w] = u_out[r];
        }
    }
    u_out.resize(w + 1);
}

// Scatter the symmetric rank-1 `coef * u u^T` (dense). Writes every (a, b)
// cell once with coef * u_a * u_b -- both triangles plus the diagonal -- to
// match the dense symmetric storage the per-obs scatter uses.
inline void scatter_self_rank1_dense(
    double                                coef,
    const std::vector<ArmRowChainEntry>&  u,
    DenseMat&                             H
) {
    if (coef == 0.0) return;
    for (const auto& ea : u) {
        for (const auto& eb : u) {
            H[ea.idx][eb.idx] += coef * ea.w * eb.w;
        }
    }
}

// Where the coupled per-cell scatter's writes land. The cell walk
// (scatter_cell_coupling_branch_impl) owns the cell iteration, the spec call
// and the chain construction; a target owns the three kinds of write:
//   row(kk, k, i, g, h, pa, blocks, d_eff)  -- row i of arm k (coupled index
//       kk), through its design, from its eta-space score g and negative
//       curvature h;
//   cross(kk, ll, Hkl, chain_k, chain_l)    -- Hkl * (chain_k chain_l' + its
//       transpose) for a row of arm kk against a row of arm ll;
//   rank1(kk, coef, u)                      -- the rank-1 self-cross coef u u';
// with begin_cell(c) before a cell's first write.

// Dense target: the n_x x n_x matrix, both triangles written.
struct CoupledDenseScatter {
    DenseVec&           grad;
    DenseMat&           H;
    std::vector<int>    active_idx;
    std::vector<double> active_d;

    CoupledDenseScatter(DenseVec& g, DenseMat& h, int n_blocks)
        : grad(g), H(h) {
        active_idx.reserve(n_blocks);
        active_d.reserve(n_blocks);
    }
    void begin_cell(int) {}
    void row(int, int k, int i, double g, double h, const ParsedArm& pa,
             const std::vector<LatentBlock>& blocks,
             const std::vector<double>& d_eff) {
        scatter_one_arm_row_dense(i, g, h, pa, k, blocks, d_eff, grad, H,
                                  active_idx, active_d);
    }
    void cross(int, int, double Hkl, const std::vector<ArmRowChainEntry>& ck,
               const std::vector<ArmRowChainEntry>& cl) {
        scatter_cross_chain_dense(Hkl, ck, cl, H);
    }
    void rank1(int, double coef, const std::vector<ArmRowChainEntry>& u) {
        scatter_self_rank1_dense(coef, u, H);
    }
};

// Sparse target for one chunk of cells: every write goes through the coupled
// plan's flat offsets into the chunk's sink (coupled_scatter_plan.h), lower
// triangle only. A chain entry's position is a fixed effect's place in the
// coupled fixed-effect block, or a dof's place in the current cell's own list.
struct CoupledPlannedScatter {
    const CoupledScatterPlan&           plan;
    const std::vector<ArmIndexedView>&  views;   // [kk]
    CoupledChunkSink                    sink;
    std::vector<double>                 w_buf;
    CoupledCellSlots                    cell;
    std::vector<int>                    pos_a, pos_b;

    CoupledPlannedScatter(const CoupledScatterPlan&           p,
                          const std::vector<ArmIndexedView>&  v,
                          CoupledChunkSink                    s)
        : plan(p), views(v), sink(s) {
        int max_A = 1;
        for (const auto& av : v) max_A = std::max(max_A, av.max_A);
        w_buf.assign(max_A, 0.0);
        pos_a.reserve(kRowChainScratchHint);
        pos_b.reserve(kRowChainScratchHint);
    }

    void begin_cell(int c) { cell = plan.cell(c); }

    void row(int kk, int, int i, double g, double h, const ParsedArm& pa,
             const std::vector<LatentBlock>&, const std::vector<double>& d_eff) {
        scatter_row_indexed(i, g, h, pa, d_eff.data(), views[kk],
                            /*xv=*/nullptr, w_buf.data(), sink);
    }

    void positions(int kk, const std::vector<ArmRowChainEntry>& ch,
                   std::vector<int>& pos) const {
        pos.resize(ch.size());
        for (std::size_t t = 0; t < ch.size(); t++) {
            pos[t] = (ch[t].grp == CHAIN_BETA)
                     ? ch[t].idx + plan.beta_delta[kk]
                     : cell.local_pos(ch[t].idx);
        }
    }

    // Both directions of a pair land in one lower-triangle entry, so each
    // (a, b) is written once; a dof the two chains share is a diagonal entry,
    // which takes the pair's term from each direction.
    void cross(int kk, int ll, double Hkl,
               const std::vector<ArmRowChainEntry>& ck,
               const std::vector<ArmRowChainEntry>& cl) {
        if (Hkl == 0.0) return;
        positions(kk, ck, pos_a);
        positions(ll, cl, pos_b);
        for (std::size_t a = 0; a < ck.size(); a++) {
            for (std::size_t b = 0; b < cl.size(); b++) {
                const double val = Hkl * ck[a].w * cl[b].w;
                const int s = cell.slot(pos_a[a], pos_b[b]);
                sink.hess(s, val);
                if (ck[a].idx == cl[b].idx) sink.hess(s, val);
            }
        }
    }

    // `u` carries unique dofs (merged by accumulate_self_rank1_u), so the
    // `idx >= idx` guard writes each lower-triangle entry exactly once.
    void rank1(int kk, double coef, const std::vector<ArmRowChainEntry>& u) {
        if (coef == 0.0) return;
        positions(kk, u, pos_a);
        for (std::size_t a = 0; a < u.size(); a++) {
            for (std::size_t b = 0; b < u.size(); b++) {
                if (u[a].idx >= u[b].idx) {
                    sink.hess(cell.slot(pos_a[a], pos_a[b]),
                              coef * u[a].w * u[b].w);
                }
            }
        }
    }
};

// Per-cell scatter branch over the cells [c0, c1). Walks cells, dispatches to
// `spec->evaluate_cell()`, and hands each row's eta-space derivatives, each
// nonzero cross-row entry and each rank-1 self-cross to the target `sc`
// (CoupledDenseScatter or CoupledPlannedScatter). Single source of truth for
// the cell iteration, view construction, per-cell cross_hess buffer allocation
// and per-arm bookkeeping; the targets differ only in where a write lands. All
// scratch is function-local, so distinct cell ranges run independently.
template <typename Scatter>
inline void scatter_cell_coupling_branch_impl(
    const CellCouplingSpec&                       spec,
    const std::vector<int>&                       coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int                                           n_cells,
    const std::vector<JointArm>&                  arms,
    const std::vector<ParsedArm>&                 parsed,
    const std::vector<Rcpp::NumericVector>&       etas,
    const std::vector<LatentBlock>&               blocks,
    int                                           k_grid,
    Scatter&                                      sc,
    CurvatureMode                                 curvature,
    bool                                          grad_only,
    const double*                                 phi_override,
    int                                           c0,
    int                                           c1
) {
    const int n_coupled = (int)coupled_arms.size();
    const int B         = (int)blocks.size();
    if (n_coupled == 0 || n_cells == 0) return;
    const int cend = c1;

    // Per-arm d_eff cache (one entry per coupled arm × block).
    std::vector<std::vector<double>> d_eff_per_arm(n_coupled,
                                                    std::vector<double>(B));
    for (int kk = 0; kk < n_coupled; kk++) {
        int k = coupled_arms[kk];
        for (int b = 0; b < B; b++) {
            double s = blocks[b].arm_scale
                        ? blocks[b].arm_scale(k, k_grid)
                        : 1.0;
            d_eff_per_arm[kk][b] = s * blocks[b].d_fac_at(k_grid);
        }
    }

    // Per-arm views and the per-cell slices of them: the same holder the
    // objective-only walker uses, so the two cannot describe the cell
    // differently.
    CoupledArmViews views(coupled_arms, arms, etas,
                          [&](int kk, int k) {
                              return coupled_arm_own_response(arms[k], kk,
                                                              phi_override);
                          });
    std::vector<int>&        arm_row_count = views.row_count;
    std::vector<const int*>& arm_rows_ptr  = views.rows_ptr;

    // Per-cell cross-arm Hessian scratch. arm_cross_hess[kk][ll] is a
    // J_kk x J_ll row-major buffer (kk <= ll only; kk > ll left nullptr
    // per the CellDerivs contract). The outer two dims have stable shape
    // n_coupled x n_coupled across cells; only the per-cell J_kk * J_ll
    // backing storage is grown monotonically.
    std::vector<std::vector<std::vector<double>>> cross_hess_buf(n_coupled,
        std::vector<std::vector<double>>(n_coupled));
    std::vector<std::vector<double*>> cross_hess_ptr_inner(n_coupled,
        std::vector<double*>(n_coupled, nullptr));
    std::vector<double* const*> cross_hess_outer(n_coupled, nullptr);
    for (int kk = 0; kk < n_coupled; kk++) {
        cross_hess_outer[kk] = cross_hess_ptr_inner[kk].data();
    }

    // Per-cell rank-1 self-cross descriptor scratch. A
    // coupled arm may declare its (kk, kk) off-diagonal cross block as one
    // symmetric rank-1 coef * v v^T instead of the dense cross_hess_buf[kk][kk];
    // the kernel collapses it to one coef * u u^T in dof space below. coef
    // is re-zeroed per cell (0 = dense path); the rc_k weight buffer is grown
    // monotonically alongside arm_grad_buf.
    std::vector<double>              rank1_coef(n_coupled, 0.0);
    std::vector<std::vector<double>> rank1_vec_buf(n_coupled);
    std::vector<double*>             rank1_vec_ptr(n_coupled, nullptr);
    std::vector<ArmRowChainEntry>    rank1_u_scratch;
    rank1_u_scratch.reserve(kCellChainScratchHint);

    // Per-row chain scratch reused across rows / pairs.
    std::vector<ArmRowChainEntry> chain_k_scratch;
    std::vector<ArmRowChainEntry> chain_l_scratch;
    chain_k_scratch.reserve(kRowChainScratchHint);
    chain_l_scratch.reserve(kRowChainScratchHint);

    // Which (kk, ll) dense cross-Hessian slabs to allocate. The spec declares the
    // pairs it actually writes densely; a self block it emits as the rank-1
    // self-cross, or a cross it factorises to zero, is omitted, and its buffer
    // stays nullptr (the scatter below already guards null). This is what bounds a
    // cell with J observations on a self-coupled arm to O(J) rather than O(J^2):
    // the dense rc_kk * rc_ll slab is never allocated for the omitted pairs. This
    // is the single-response path, which supplies the rank-1 descriptor.
    std::vector<std::vector<char>> alloc_pair(
        n_coupled, std::vector<char>(n_coupled, 0));
    for (const auto& pr :
         spec.dense_cross_pairs(n_coupled, /*rank1_self_supported=*/true)) {
        const int a = std::min(pr.first, pr.second);
        const int b = std::max(pr.first, pr.second);
        if (a >= 0 && b < n_coupled) alloc_pair[a][b] = 1;
    }

    for (int c = c0; c < cend; c++) {
        views.bind_cell(c, cell_rows);
        sc.begin_cell(c);
        for (int kk = 0; kk < n_coupled; kk++) {
            const int rc = arm_row_count[kk];
            // Rank-1 self-cross descriptor: re-zero the coefficient (the spec
            // sets it only when it declares the (kk, kk) block rank-1) and grow
            // the rc_k weight buffer. Its contents are only read when the spec
            // sets the coefficient, in which case the spec writes all rc rows.
            rank1_coef[kk] = 0.0;
            if ((int)rank1_vec_buf[kk].size() < rc) rank1_vec_buf[kk].assign(rc, 0.0);
            rank1_vec_ptr[kk] = rank1_vec_buf[kk].data();
        }

        // Cross-arm Hessian buffers: allocate one J_kk * J_ll slab per
        // (kk, ll) pair with kk <= ll. kk > ll stays nullptr per the
        // CellDerivs contract; integration symmetrises.
        for (int kk = 0; kk < n_coupled; kk++) {
            int rc_k = arm_row_count[kk];
            for (int ll = kk; ll < n_coupled; ll++) {
                if (!alloc_pair[kk][ll]) {        // spec does not write this slab
                    cross_hess_ptr_inner[kk][ll] = nullptr;
                    continue;
                }
                int rc_l = arm_row_count[ll];
                std::size_t n_pair = (std::size_t)rc_k * (std::size_t)rc_l;
                auto& buf = cross_hess_buf[kk][ll];
                if (buf.size() < n_pair) {
                    buf.assign(n_pair, 0.0);
                } else {
                    std::fill(buf.begin(), buf.begin() + n_pair, 0.0);
                }
                cross_hess_ptr_inner[kk][ll] = buf.data();
            }
            for (int ll = 0; ll < kk; ll++) {
                cross_hess_ptr_inner[kk][ll] = nullptr;
            }
        }

        CellEtas     etas_view = views.etas_view();
        CellResponse y_view    = views.response_view();

        CellDerivs out;
        out.arm_grad             = views.grad_ptr.data();
        out.arm_neg_hess_diag    = views.neg_hess_diag_ptr.data();
        out.arm_cross_hess       = cross_hess_outer.data();
        out.arm_row_count        = arm_row_count.data();
        out.n_arms_              = n_coupled;
        out.curvature            = curvature;
        out.grad_only            = grad_only;
        out.arm_cross_rank1_coef = rank1_coef.data();
        out.arm_cross_rank1_vec  = rank1_vec_ptr.data();

        spec.evaluate_cell(c, etas_view, y_view, out);

        // Within-arm per-row scatter (eta diagonal Hessian + gradient).
        for (int kk = 0; kk < n_coupled; kk++) {
            int k  = coupled_arms[kk];
            int rc = arm_row_count[kk];
            const int*    rows = arm_rows_ptr[kk];
            const double* g    = views.grad_buf[kk].data();
            const double* h    = views.neg_hess_diag_buf[kk].data();
            for (int j = 0; j < rc; j++) {
                sc.row(kk, k, rows[j], g[j], h[j], parsed[k], blocks,
                       d_eff_per_arm[kk]);
            }
        }

        // Cross-arm Hessian scatter. Pure curvature with no gradient
        // contribution, so a grad-only step (cached-factor reuse) skips it.
        // The self block (kk, kk) takes the rank-1 fast path when the spec
        // declared one (a single coef * u u^T over the arm's joint dofs, as in
        // the all-undetected occupancy mixture); otherwise it walks the dense
        // off-diagonal. Cross blocks (kk < ll) are always dense.
        if (!grad_only) {
            for (int kk = 0; kk < n_coupled; kk++) {
                int k     = coupled_arms[kk];
                int rc_k  = arm_row_count[kk];
                const int* rows_k = arm_rows_ptr[kk];

                // Self block (kk, kk).
                if (rank1_coef[kk] != 0.0) {
                    accumulate_self_rank1_u(
                        rank1_vec_ptr[kk], rc_k, rows_k, parsed[k], k, blocks,
                        d_eff_per_arm[kk], chain_k_scratch, rank1_u_scratch);
                    sc.rank1(kk, rank1_coef[kk], rank1_u_scratch);
                } else if (const double* ch = cross_hess_ptr_inner[kk][kk]) {
                    for (int j = 0; j < rc_k; j++) {
                        build_arm_row_chain(rows_k[j], parsed[k], k, blocks,
                                            d_eff_per_arm[kk], chain_k_scratch);
                        for (int m = j + 1; m < rc_k; m++) {
                            double Hkl = ch[(std::size_t)j * rc_k + m];
                            if (Hkl == 0.0) continue;
                            build_arm_row_chain(rows_k[m], parsed[k], k, blocks,
                                                d_eff_per_arm[kk], chain_l_scratch);
                            sc.cross(kk, kk, Hkl, chain_k_scratch, chain_l_scratch);
                        }
                    }
                }

                // Cross blocks (kk, ll), ll > kk: always dense.
                for (int ll = kk + 1; ll < n_coupled; ll++) {
                    const double* ch = cross_hess_ptr_inner[kk][ll];
                    if (!ch) continue;
                    int l     = coupled_arms[ll];
                    int rc_l  = arm_row_count[ll];
                    const int* rows_l = arm_rows_ptr[ll];
                    for (int j = 0; j < rc_k; j++) {
                        build_arm_row_chain(rows_k[j], parsed[k], k, blocks,
                                            d_eff_per_arm[kk], chain_k_scratch);
                        for (int m = 0; m < rc_l; m++) {
                            double Hkl = ch[(std::size_t)j * rc_l + m];
                            if (Hkl == 0.0) continue;
                            build_arm_row_chain(rows_l[m], parsed[l], l, blocks,
                                                d_eff_per_arm[ll], chain_l_scratch);
                            sc.cross(kk, ll, Hkl, chain_k_scratch, chain_l_scratch);
                        }
                    }
                }
            }
        }
    }
}

// Dense wrapper: every cell in one pass, written into the n_x x n_x DenseMat
// (both triangles).
inline void scatter_cell_coupling_dense_branch(
    const CellCouplingSpec&                       spec,
    const std::vector<int>&                       coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int                                           n_cells,
    const std::vector<JointArm>&                  arms,
    const std::vector<ParsedArm>&                 parsed,
    const std::vector<Rcpp::NumericVector>&       etas,
    const std::vector<LatentBlock>&               blocks,
    int                                           k_grid,
    DenseVec&                                     grad,
    DenseMat&                                     H,
    CurvatureMode                                 curvature = CurvatureMode::Observed,
    const double*                                 phi_override = nullptr
) {
    CoupledDenseScatter sc(grad, H, static_cast<int>(blocks.size()));
    scatter_cell_coupling_branch_impl(
        spec, coupled_arms, cell_rows, n_cells,
        arms, parsed, etas, blocks, k_grid, sc,
        curvature, /*grad_only=*/false, phi_override, 0, n_cells);
}

// Sparse wrapper: the cells in the plan's fixed chunks, each written through
// the plan's flat offsets (lower triangle) into the joint SparseHessianBuilder,
// then the shared entries' chunk partials added in chunk order. The chunks run
// on whatever threads are idle -- inside the outer grid's parallel region the
// grid threads that have run out of cells take a long cell's chunks -- and the
// answer is the same however many do. `n_threads` is the team opened when the
// call is not already inside a parallel region (a serial pilot, a one-thread
// grid).
inline void scatter_cell_coupling_sparse_branch(
    const CellCouplingSpec&                       spec,
    const std::vector<int>&                       coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int                                           n_cells,
    const std::vector<JointArm>&                  arms,
    const std::vector<ParsedArm>&                 parsed,
    const std::vector<Rcpp::NumericVector>&       etas,
    const std::vector<LatentBlock>&               blocks,
    int                                           k_grid,
    DenseVec&                                     grad,
    SparseHessianBuilder&                         H,
    const CoupledScatterPlan&                     plan,
    CurvatureMode                                 curvature,
    bool                                          grad_only,
    const double*                                 phi_override,
    int                                           n_threads
) {
    if (coupled_arms.empty() || n_cells == 0) return;
    if (!plan.valid_for(H, n_cells)) {
        throw std::logic_error(
            "coupled scatter plan was not resolved against this Hessian "
            "pattern");
    }
    std::vector<ArmIndexedView> views;
    views.reserve(plan.arm.size());
    for (const auto& ac : plan.arm) views.emplace_back(ac);

    CoupledChunkPartials partials(plan);
    double* gv = grad.data();
    double* Hv = H.values.data();
    run_coupled_chunks(plan.n_chunks, n_threads, spec.thread_safe(),
                       [&](int ch) {
        CoupledPlannedScatter sc(plan, views, partials.sink(ch, gv, Hv));
        scatter_cell_coupling_branch_impl(
            spec, coupled_arms, cell_rows, n_cells,
            arms, parsed, etas, blocks, k_grid, sc,
            curvature, grad_only, phi_override,
            plan.chunk_lo(ch), plan.chunk_lo(ch + 1));
    });
    partials.reduce(gv, Hv);
}

// Sparse-builder analogue of scatter_arm_obs_joint_multi. Writes into a
// SparseHessianBuilder (which owns the joint Hessian pattern built once
// by build_joint_hessian_pattern) rather than into an n_x × n_x DenseMat.
//
// Semantics match the dense version exactly. Differences:
//   - Only the lower triangle is written (single H.add() per off-diagonal
//     pair, not two). SparseHessianBuilder normalizes (r, c) → (max, min)
//     internally; calling H.add(r, c) and H.add(c, r) would double-count.
//   - DENSE_BASIS blocks contribute every coefficient of the block to the
//     active-dofs list, weighted by basis_eval(i, k_arm) values.
//   - INDEXED_MULTI blocks contribute the dofs returned by obs_indices.
//   - INDEXED_SINGLE matches the existing behavior (one dof per obs).
//
// Caller owns scratch buffers (active_scratch, basis_scratch, multi_scratch).
// They should live on a per-thread NewtonScratchJoint so concurrent outer-
// grid threads do not contend. The pattern in H must already cover every
// (row, col) this scatter will touch — H.add() silently drops entries not
// present in the pattern.
inline void scatter_arm_obs_joint_multi_sparse(
    const Rcpp::NumericVector&    x,
    const Rcpp::NumericVector&    eta,
    const ParsedArm&              pa,
    const JointArm&               arm,
    const ArmSpecView&            view,
    int                           k_arm,
    const std::vector<LatentBlock>& blocks,
    int                           k_grid,
    DenseVec&                     grad,
    SparseHessianBuilder&         H,
    std::vector<std::pair<int,double>>& active_scratch,
    std::vector<double>&          basis_scratch,
    std::vector<std::pair<int,double>>& multi_scratch,
    std::vector<DenseBasisActive>&    active_db_scratch,
    std::vector<DenseBasisScratch>&   db_buffers,
    const ScatterIndexCache*          idx_cache = nullptr,
    int                               n_threads = 1
) {
    const int p_k      = pa.p;
    const int n_re_k   = pa.n_re_groups;
    const int bstart   = pa.beta_start;
    const int rstart   = pa.re_start;
    const int B = static_cast<int>(blocks.size());

    // Cache per-block d_eff = arm_scale(k_arm, k_grid) * d_fac(k_grid),
    // contrib_kind, and the max DENSE_BASIS size so basis_scratch is
    // sized once per call. Count DENSE_BASIS vs INDEXED blocks to decide
    // whether basis_eval must run inside the per-obs loop (for cross
    // emissions) or can be skipped entirely in favor of the batch helper.
    std::vector<double> d_eff_cache(B);
    std::vector<BlockContribKind> kind_cache(B);
    int max_basis_size  = 0;
    int n_db_with_batch = 0;
    int n_db_legacy     = 0;
    int n_indexed       = 0;
    for (int b = 0; b < B; b++) {
        double s = blocks[b].arm_scale
                    ? blocks[b].arm_scale(k_arm, k_grid)
                    : 1.0;
        d_eff_cache[b] = s * blocks[b].d_fac_at(k_grid);
        kind_cache[b]  = blocks[b].contrib_kind;
        if (kind_cache[b] == BlockContribKind::DENSE_BASIS) {
            if (blocks[b].size > max_basis_size) max_basis_size = blocks[b].size;
            if (blocks[b].dense_basis_batch) n_db_with_batch++;
            else                              n_db_legacy++;
        } else {
            n_indexed++;
        }
    }
    if (static_cast<int>(basis_scratch.size()) < max_basis_size) {
        basis_scratch.resize(max_basis_size);
    }

    const int n_db_total = n_db_with_batch + n_db_legacy;
    // basis_eval must run in the per-obs loop when there are cross terms
    // to emit (DB x INDEXED, DB x DB inter-block) or when a legacy DB
    // block lacks the batch hook. The pure-single-DB case (HSGP /
    // HSGP-MO / HSGP-SVC alone) skips per-obs DB entirely; the batch
    // helper handles every block-internal contribution.
    const bool need_db_in_perobs =
        (n_db_total > 0) &&
        ((n_indexed > 0) || (n_db_total >= 2) || (n_db_legacy > 0));

    // Fast path: when no DENSE_BASIS block is present, the
    // per-obs scatter resolves to a (mostly) static (i, k_arm) ->
    // (active_dof, flat_idx) mapping. INDEXED_SINGLE / INDEXED_MULTI have
    // fully static weights; BILINEAR_FACTOR active weights are computed
    // from x[paired_slot] inside the cached scatter (paired-slot is also
    // cached). Mixed DB + INDEXED cases fall through to the legacy per-
    // obs path.
    const bool use_indexed_cache =
        idx_cache
        && scatter_index_cache_valid(*idx_cache, H)
        && !idx_cache->any_dense_basis
        && n_db_total == 0
        && static_cast<int>(idx_cache->arm.size()) > k_arm
        && static_cast<int>(idx_cache->arm[k_arm].plans.size()) == arm.N;
    if (use_indexed_cache) {
        scatter_arm_obs_indexed_cached(
            x, eta, pa, arm, view, d_eff_cache, idx_cache->arm[k_arm],
            grad, H, n_threads
        );
        return;
    }

    for (int i = 0; i < arm.N; i++) {
        auto gh = arm_grad_hess(view, i, eta[i]);

        int g_re = -1;
        if (n_re_k > 0) {
            int gi = static_cast<int>(pa.re_idx[i]) - 1;
            if (gi >= 0 && gi < n_re_k) g_re = rstart + gi;
        }

        // Active dofs of every block this row reads: INDEXED entries into
        // active_scratch, DENSE_BASIS entries into active_db_scratch. The basis
        // row is resolved here only when a cross term needs it (see
        // need_db_in_perobs); otherwise the batch helper below covers the block.
        active_scratch.clear();
        active_db_scratch.clear();
        for_each_row_block_latent(
            i, k_arm, k_grid, blocks, d_eff_cache, x.begin(),
            multi_scratch, basis_scratch,
            [&](int b) {
                return kind_cache[b] == BlockContribKind::DENSE_BASIS &&
                       !need_db_in_perobs &&
                       static_cast<bool>(blocks[b].dense_basis_batch);
            },
            [&](int b, int latent, double w) {
                if (kind_cache[b] == BlockContribKind::DENSE_BASIS) {
                    DenseBasisActive e;
                    e.dof       = latent;
                    e.weight    = w;
                    e.block_idx = b;
                    e.has_batch = static_cast<bool>(blocks[b].dense_basis_batch);
                    active_db_scratch.push_back(e);
                } else {
                    active_scratch.emplace_back(latent, w);
                }
            });
        const int A_idx = static_cast<int>(active_scratch.size());
        const int A_db  = static_cast<int>(active_db_scratch.size());

        // β block: gradient + β/β diagonal + β × RE + β × active crosses.
        // DENSE_BASIS contributions with a batch hook are skipped — the
        // post-loop scatter_dense_basis_block writes β × block via GEMM.
        for (int j = 0; j < p_k; j++) {
            const double Xij = pa.X(i, j);
            grad[bstart + j] += gh.grad * Xij;
            for (int l = 0; l <= j; l++) {
                H.add(bstart + j, bstart + l,
                      gh.neg_hess * Xij * pa.X(i, l));
            }
            if (g_re >= 0) {
                H.add(bstart + j, g_re, gh.neg_hess * Xij);
            }
            for (int a = 0; a < A_idx; a++) {
                H.add(bstart + j, active_scratch[a].first,
                      gh.neg_hess * Xij * active_scratch[a].second);
            }
            for (int a = 0; a < A_db; a++) {
                if (active_db_scratch[a].has_batch) continue;
                H.add(bstart + j, active_db_scratch[a].dof,
                      gh.neg_hess * Xij * active_db_scratch[a].weight);
            }
        }

        // RE block: gradient + diagonal + cross with active dofs.
        // DENSE_BASIS with batch hook handled in post-loop.
        if (g_re >= 0) {
            grad[g_re] += gh.grad;
            H.add(g_re, g_re, gh.neg_hess);
            for (int a = 0; a < A_idx; a++) {
                H.add(g_re, active_scratch[a].first,
                      gh.neg_hess * active_scratch[a].second);
            }
            for (int a = 0; a < A_db; a++) {
                if (active_db_scratch[a].has_batch) continue;
                H.add(g_re, active_db_scratch[a].dof,
                      gh.neg_hess * active_db_scratch[a].weight);
            }
        }

        // INDEXED × INDEXED intra/inter (lower triangle including diagonal):
        // gradient + H. Existing semantics unchanged.
        for (int a = 0; a < A_idx; a++) {
            const int d_a = active_scratch[a].first;
            const double w_a = active_scratch[a].second;
            grad[d_a] += gh.grad * w_a;
            for (int b = 0; b <= a; b++) {
                const int d_b = active_scratch[b].first;
                const double w_b = active_scratch[b].second;
                H.add(d_a, d_b, gh.neg_hess * w_a * w_b);
            }
        }

        // DENSE_BASIS active interactions.
        //   * Gradient: skip when batch hook covers it (batch helper
        //     accumulates Phi^T g_obs). Emit when legacy (no batch hook).
        //   * DB × INDEXED cross: always emit (batch doesn't see INDEXED).
        //   * DB × DB intra-block: skip when both have batch hook (SYRK
        //     covers it). Emit when at least one is legacy or when the
        //     pair is across different blocks (inter-block cross).
        for (int a = 0; a < A_db; a++) {
            const int    d_a       = active_db_scratch[a].dof;
            const double w_a       = active_db_scratch[a].weight;
            const int    blk_a     = active_db_scratch[a].block_idx;
            const bool   a_batch   = active_db_scratch[a].has_batch;

            if (!a_batch) {
                grad[d_a] += gh.grad * w_a;
            }

            // DB × INDEXED (cross with all INDEXED active dofs)
            for (int b = 0; b < A_idx; b++) {
                const int d_b    = active_scratch[b].first;
                const double w_b = active_scratch[b].second;
                H.add(d_a, d_b, gh.neg_hess * w_a * w_b);
            }
            // DB × DB lower triangle including diagonal
            for (int b = 0; b <= a; b++) {
                const int    d_b     = active_db_scratch[b].dof;
                const double w_b     = active_db_scratch[b].weight;
                const int    blk_b   = active_db_scratch[b].block_idx;
                const bool   b_batch = active_db_scratch[b].has_batch;
                const bool   same    = (blk_a == blk_b);
                // Skip the case the batch helper covers: both sides have
                // a batch hook AND both live in the same block (intra-
                // block SYRK output).
                if (a_batch && b_batch && same) continue;
                H.add(d_a, d_b, gh.neg_hess * w_a * w_b);
            }
        }
    }

    // Post-loop batched scatter for every DENSE_BASIS block that exposes a
    // dense_basis_batch hook. One SYRK + one GEMM + per-group RE reduction
    // + one GEMV per block; replaces the per-obs M^2 scalar loop.
    //
    // db_buffers is indexed by (k_arm * B + b). Each (arm, block) slot holds
    // its own scatter index cache; cache keys (blk_off_row, M, p_k, n_re_k,
    // bstart, rstart, H_ptr) are stable across outer-grid cells, so each
    // slot rebuilds at most once per fit.
    if (n_db_with_batch > 0) {
        const size_t needed = static_cast<size_t>(k_arm + 1) * B;
        if (db_buffers.size() < needed) db_buffers.resize(needed);
        for (int b = 0; b < B; b++) {
            if (kind_cache[b] != BlockContribKind::DENSE_BASIS) continue;
            if (!blocks[b].dense_basis_batch) continue;
            scatter_dense_basis_block(blocks[b], k_arm, k_grid, pa, arm, view,
                                       eta, d_eff_cache[b], grad, H,
                                       db_buffers[static_cast<size_t>(k_arm) * B + b]);
        }
    }
}

// Sparse-path joint compute_eta accumulator. Dispatches on each block's
// contrib_kind so INDEXED_SINGLE / INDEXED_MULTI / DENSE_BASIS all flow
// through the same per-arm obs loop.
//
// d_eff_per_block_arm[b][k_arm] caches d_fac(k_grid) * arm_scale(k_arm, k_grid).
// basis_scratch_per_block sized to max(block.size for DENSE_BASIS blocks);
// reused per obs to avoid per-call allocation.
inline void compute_eta_joint_sparse_dispatch(
    const Rcpp::NumericVector&        x,
    std::vector<Rcpp::NumericVector>& etas,
    const std::vector<JointArm>&      arms,
    const std::vector<ParsedArm>&     parsed,
    const std::vector<LatentBlock>&   blocks,
    int                                k_grid,
    const std::vector<std::vector<double>>& d_eff,
    std::vector<double>&              basis_scratch,
    std::vector<std::pair<int,double>>& multi_scratch
) {
    const int n_arms = static_cast<int>(arms.size());
    const int B      = static_cast<int>(blocks.size());
    for (int k_arm = 0; k_arm < n_arms; k_arm++) {
        const ParsedArm& pa = parsed[k_arm];
        const int N_k    = arms[k_arm].N;
        const int p_k    = pa.p;
        const int n_re_k = pa.n_re_groups;
        const int bstart = pa.beta_start;
        const int rstart = pa.re_start;

        for (int i = 0; i < N_k; i++) {
            double e = (pa.offset.size() != 0) ? pa.offset[i] : 0.0;
            for (int j = 0; j < p_k; j++) e += pa.X(i, j) * x[bstart + j];
            if (n_re_k > 0) {
                int g = static_cast<int>(pa.re_idx[i]) - 1;
                if (g >= 0 && g < n_re_k) e += x[rstart + g];
            }
            for (int b = 0; b < B; b++) {
                const LatentBlock& blk = blocks[b];
                // Same fold as the scatter walkers: the per-row SVC weight
                // rides the block amplitude on every kind, and is 1.0 where
                // the block declares none.
                const double d_e = d_eff[b][k_arm]
                                 * block_row_weight(blk, i, k_arm);
                if (d_e == 0.0) continue;
                switch (blk.contrib_kind) {
                case BlockContribKind::INDEXED_SINGLE: {
                    if (!blk.idx) break;
                    int l = blk.idx(i, k_arm);
                    if (l > 0 && l <= blk.size) {
                        e += d_e * x[blk.start + l - 1];
                    }
                    break;
                }
                case BlockContribKind::INDEXED_MULTI: {
                    if (!blk.obs_indices) break;
                    blk.fill_obs_indices(i, k_arm, multi_scratch);
                    for (const auto& jw : multi_scratch) {
                        int l = jw.first;
                        if (l > 0 && l <= blk.size) {
                            e += d_e * jw.second * x[blk.start + l - 1];
                        }
                    }
                    break;
                }
                case BlockContribKind::DENSE_BASIS: {
                    if (!blk.basis_eval) break;
                    if (static_cast<int>(basis_scratch.size()) < blk.size) {
                        basis_scratch.assign(blk.size, 0.0);
                    }
                    blk.basis_eval(i, k_arm, k_grid, basis_scratch.data());
                    double acc = 0.0;
                    for (int j = 0; j < blk.size; j++) {
                        acc += basis_scratch[j] * x[blk.start + j];
                    }
                    e += d_e * acc;
                    break;
                }
                case BlockContribKind::BILINEAR_FACTOR: {
                    if (!blk.obs_factor_lambda) break;
                    // Both slots are bounded by obs_factor_lambda itself, which
                    // returns {-1, -1} outside the factor's latent range.
                    auto [u_slot, lambda_slot] = blk.obs_factor_lambda(i, k_arm);
                    if (u_slot >= 0 && lambda_slot >= 0) {
                        e += d_e * x[u_slot] * x[lambda_slot];
                    }
                    break;
                }
                }
            }
            etas[k_arm][i] = e;
        }
    }
}

// Sparse-path inner driver. Built ONCE per outer-grid pass (pattern
// computed at fit-time; SparseHessianBuilder values rebuilt per cell).
// Serial outer-grid: see needs_sparse branch in the public driver for the
// rationale. Forward-declared here, defined just below.
Rcpp::List run_multi_block_nested_laplace_joint_sparse_impl(
    int                              n_grid,
    std::vector<JointArm>&           arms,
    const std::vector<ParsedArm>&    parsed,
    const std::vector<LatentBlock>&  blocks,
    int                              n_x,
    int                              max_iter,
    double                           tol,
    int                              n_threads,
    bool                             store_modes,
    const Rcpp::NumericVector&       x_init,
    bool                             store_Q,
    std::function<void(int)>         prep_at_grid,
    const std::vector<int>&          tile_ids,
    const std::vector<int>&          tile_pilot_cells,
    double                           prune_tol,
    std::shared_ptr<CellCouplingSpec> cell_coupling_spec,
    const std::vector<int>&          coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int                              n_cells,
    JointPDMode                      pd_mode = JointPDMode::LM,
    StepCurvature                    step_curvature = StepCurvature::Observed,
    int                              hessian_refresh = 1,
    int                              n_threads_outer = 1,
    tulpa_progress::GridProgress*    progress = nullptr,
    GridCheckpoint*                  checkpoint = nullptr,
    const std::vector<double>&       x_init_per_cell = std::vector<double>(),
    // Inner-Laplace skewness diagnostic (inner_laplace_skew.h), opt-in like
    // store_Q. See laplace_newton_joint.h's build_joint_curvature3_fns: a
    // coupled arm (cell_coupling_spec->arm_ids()) is excluded from the
    // per-arm oracle automatically, so its observations drop out of gamma_3
    // rather than being scored against the wrong (unused) per-obs likelihood.
    bool                             compute_skew = false,
    const std::vector<int>*          skew_probe_idx = nullptr,
    // Per-cell fixed-effect covariance block, extracted inside each cell's own
    // solve so the grid never holds every cell's precision at once.
    const JointFixedBlockRequest*    fixed_block = nullptr,
    // Subspace debias (subspace_debias.h). Runs on every integrated cell (never
    // the cheap screen), so the corrected coordinates enter the reported
    // marginal as a mixture over the whole outer grid.
    const SubspaceDebiasOptions*     debias = nullptr,
    // Corrected integrated Laplace (inner_cila.h). Runs on every integrated
    // cell (never the cheap screen), so the corrected cell weights and
    // particles cover the whole outer grid.
    const CilaOptions*               cila = nullptr,
    // Inner Newton steps per cell in the cheap screening sweep. A positional
    // forward from the entry stays valid as the tail grows.
    int                              screen_iters = CHEAP_SCREEN_ITERS,
    // Whether every fully-solved cell reports the per-row predictive variance
    // of the linear predictor (LaplaceResult::eta_var, emitted as
    // `fitted_eta_var`). Never the cheap screen.
    bool                             compute_eta_var = false,
    // Per-cell log hyperprior + log cell measure the cheap screen ranks with
    // (run_nested_laplace_grid); empty ranks on the log-marginal alone.
    const std::vector<double>&       screen_log_offset = std::vector<double>(),
    // Return the screened surface without the full pass (a placement pilot's
    // detecting grid; see run_nested_laplace_grid).
    bool                             screen_only = false
);

// Outer-grid driver. n_x_after_re is the latent dimension after all per-arm
// (β + RE) blocks; each LatentBlock's start field must point above that
// offset (typically built by appending sizes as blocks are constructed).
//
// `prep_at_grid` is an optional per-grid-point callback that runs before
// block.prep and the inner Newton at each outer-grid index. Joint kernels
// use it to apply per-grid dispersion overrides on `arms` (e.g.
// phi_grid_per_arm rewrites arm.phi for the current outer-grid index) or
// any other grid-dependent state that doesn't fit cleanly inside a
// LatentBlock callback. Pass `nullptr` (default) to disable.
Rcpp::List run_multi_block_nested_laplace_joint(
    int                              n_grid,
    std::vector<JointArm>&           arms,
    const std::vector<ParsedArm>&    parsed,
    const std::vector<LatentBlock>&  blocks,
    int                              n_x_after_re,
    int                              max_iter,
    double                           tol,
    int                              n_threads,
    bool                             store_modes,
    const Rcpp::NumericVector&       x_init,
    bool                             store_Q = false,
    std::function<void(int)>         prep_at_grid = nullptr,
    int                              n_threads_outer = 1,
    const std::vector<int>&          tile_ids = std::vector<int>(),
    const std::vector<int>&          tile_pilot_cells = std::vector<int>(),
    double                           prune_tol = 0.0,
    bool                             force_sparse = false,
    std::shared_ptr<CellCouplingSpec> cell_coupling_spec = nullptr,
    JointPDMode                      pd_mode = JointPDMode::LM,
    StepCurvature                    step_curvature = StepCurvature::Observed,
    int                              hessian_refresh = 1,
    tulpa_progress::GridProgress*    progress = nullptr,
    GridCheckpoint*                  checkpoint = nullptr,
    const std::vector<double>&       x_init_per_cell = std::vector<double>(),
    bool                             compute_skew = false,
    const std::vector<int>*          skew_probe_idx = nullptr,
    // Per-cell fixed-effect covariance block, extracted inside each cell's own
    // solve so the grid never holds every cell's precision at once.
    const JointFixedBlockRequest*    fixed_block = nullptr,
    // Subspace debias (subspace_debias.h). Runs on every integrated cell (never
    // the cheap screen), so the corrected coordinates enter the reported
    // marginal as a mixture over the whole outer grid.
    const SubspaceDebiasOptions*     debias = nullptr,
    // Corrected integrated Laplace (inner_cila.h). Runs on every integrated
    // cell (never the cheap screen), so the corrected cell weights and
    // particles cover the whole outer grid.
    const CilaOptions*               cila = nullptr,
    // Factorization backend of the dense inner Newton: 0 auto (size
    // threshold), >0 CHOLMOD, <0 dense. Orthogonal to force_sparse above,
    // which chooses between this driver and the sparse-assembly one.
    int                              inner_sparse_override = 0,
    // Inner Newton steps per cell in the cheap screening sweep.
    int                              screen_iters = CHEAP_SCREEN_ITERS,
    // Whether every fully-solved cell reports the per-row predictive variance
    // of the linear predictor (LaplaceResult::eta_var, emitted as
    // `fitted_eta_var`). Never the cheap screen.
    bool                             compute_eta_var = false,
    // Per-cell log hyperprior + log cell measure the cheap screen ranks with
    // (run_nested_laplace_grid); empty ranks on the log-marginal alone.
    const std::vector<double>&       screen_log_offset = std::vector<double>(),
    // Return the screened surface without the full pass (a placement pilot's
    // detecting grid; see run_nested_laplace_grid).
    bool                             screen_only = false
);

// Per-cell linear predictor at each cell's mode, for a single-arm fit run
// through the joint driver (nngp / hsgp / spde / the spatiotemporal entries,
// and cpp_nested_laplace_joint_multi). The multi-block driver fills
// `fitted_eta` itself; this one reads the same quantity through the joint
// driver's own eta accumulator, so the arm's offset, every block kind and each
// cell's block scaling are the ones the inner solve used.
//
// Two kinds of cell report NaN rows rather than a number: one whose block
// preparation fails at its coordinate, and one the cheap screen pruned, which
// was never solved. A pruned cell's `modes` row is not missing but ZERO -- the
// grid runner allocates the matrix zeroed and a skipped cell never writes into
// it -- so the predicate is the cell's own `log_marginal`, which the screen
// leaves at -Inf, and not the mode row's contents. Both kinds carry zero outer
// weight, so the mixture these rows feed never draws them; writing NaN keeps a
// row that is not a linear predictor from reading as one.
inline void nl_attach_fitted_eta_single_arm(
    Rcpp::List& out,
    const std::vector<JointArm>& arms,
    const std::vector<ParsedArm>& parsed,
    const std::vector<LatentBlock>& blocks
) {
    if (arms.size() != 1 || !out.containsElementNamed("modes")) return;
    Rcpp::NumericMatrix modes = out["modes"];
    const int ng = modes.nrow();
    const int N  = arms[0].N;
    const int B  = static_cast<int>(blocks.size());
    Rcpp::NumericVector log_marginal =
        out.containsElementNamed("log_marginal")
        ? Rcpp::as<Rcpp::NumericVector>(out["log_marginal"])
        : Rcpp::NumericVector();
    const bool have_lm = (log_marginal.size() == ng);
    Rcpp::NumericMatrix fitted_eta(ng, N);
    std::vector<Rcpp::NumericVector> etas(1, Rcpp::NumericVector(N));
    std::vector<std::vector<double>> d_eff(B, std::vector<double>(1, 0.0));
    std::vector<double> basis_scratch;
    std::vector<std::pair<int, double>> multi_scratch;
    for (int k = 0; k < ng; k++) {
        bool ok = !have_lm || R_finite(log_marginal[k]);
        for (int b = 0; ok && b < B; b++) {
            if (blocks[b].prep && !blocks[b].prep(k)) { ok = false; break; }
            const double s = blocks[b].arm_scale ? blocks[b].arm_scale(0, k) : 1.0;
            d_eff[b][0] = s * blocks[b].d_fac_at(k);
        }
        if (!ok) {
            for (int i = 0; i < N; i++) fitted_eta(k, i) = NA_REAL;
            continue;
        }
        Rcpp::NumericVector x = modes(k, Rcpp::_);
        compute_eta_joint_sparse_dispatch(x, etas, arms, parsed, blocks, k,
                                          d_eff, basis_scratch, multi_scratch);
        for (int i = 0; i < N; i++) fitted_eta(k, i) = etas[0][i];
    }
    out["fitted_eta"] = fitted_eta;
}

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_JOINT_MULTI_H
