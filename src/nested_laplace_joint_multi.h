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
//   log_prior: Σ_b log_prior_b(k) + log_prior_per_arm_re(x, parsed)
//
// Per-block prep is invoked once at grid point k before the inner solve. If
// any block reports infeasible (e.g. proper-CAR with rho outside the PD
// interval), the inner solve short-circuits with log_marginal = -inf.

#ifndef TULPA_NESTED_LAPLACE_JOINT_MULTI_H
#define TULPA_NESTED_LAPLACE_JOINT_MULTI_H

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
#include "scatter_dense_basis.h"
#include "scatter_indexed_cache.h"
#include "sparse_cholesky.h"
#include "sparse_hessian.h"
#include "tulpa/cell_coupling.h"
#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// Per-arm per-row design weight on an INDEXED_SINGLE block (areal SVC).
// Returns 1.0 when the block carries no row_weight, so the no-weight path is
// byte-identical. The weight enters the block-local weight that multiplies
// x[cell] at row i of arm k_arm; the per-block d_eff (arm_scale * d_fac)
// amplitude is applied on top, unchanged.
inline double block_row_weight(const LatentBlock& blk, int i, int k_arm) {
    return blk.row_weight ? blk.row_weight(i, k_arm) : 1.0;
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
// `scatter_one_arm_row_{dense,sparse}` (gradient + Hessian) and
// `build_arm_row_chain` (cross-arm Hessian chain). DENSE_BASIS / BILINEAR_FACTOR
// are not coupled-scatter kinds (no coupled family uses them) and are skipped.
inline void collect_coupled_row_latents(
    int                              i,
    int                              k_arm,
    const std::vector<LatentBlock>&  blocks,
    const std::vector<double>&       d_eff_cache,
    std::vector<int>&                out_idx,
    std::vector<double>&             out_w
) {
    static thread_local std::vector<std::pair<int,double>> multi_scratch;
    const int B = static_cast<int>(blocks.size());
    for (int b = 0; b < B; b++) {
        if (d_eff_cache[b] == 0.0) continue;
        const LatentBlock& blk = blocks[b];
        if (blk.contrib_kind == BlockContribKind::INDEXED_MULTI) {
            if (!blk.obs_indices) continue;
            blk.obs_indices(i, k_arm, multi_scratch);
            for (const auto& jw : multi_scratch) {
                const int l = jw.first;
                if (l > 0 && l <= blk.size) {
                    const double w = d_eff_cache[b] * jw.second;
                    if (w == 0.0) continue;
                    out_idx.push_back(blk.start + l - 1);
                    out_w.push_back(w);
                }
            }
        } else {  // INDEXED_SINGLE (the one-dof-per-row coupled-scatter kind)
            if (!blk.idx) continue;
            const int l_b = blk.idx(i, k_arm);
            if (l_b > 0 && l_b <= blk.size) {
                const double w = d_eff_cache[b] * block_row_weight(blk, i, k_arm);
                if (w == 0.0) continue;
                out_idx.push_back(blk.start + l_b - 1);
                out_w.push_back(w);
            }
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
    const int B      = static_cast<int>(blocks.size());

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
    (void) B;
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

// Sparse-builder analogue of scatter_one_arm_row_dense. Scope matches the
// dense helper: INDEXED_SINGLE and INDEXED_MULTI blocks (resolved by
// collect_coupled_row_latents); DENSE_BASIS / BILINEAR_FACTOR are handled only
// by the per-obs sparse scatter's own optimized fast path, not the coupled
// per-cell branch.
//
// Lower-triangle writes only -- H.add() normalizes (row, col) to
// (max, min) internally, so a single call covers both directions of an
// off-diagonal entry. Caller-owned scratch (active_idx, active_d) is
// reused across rows.
inline void scatter_one_arm_row_sparse(
    int                              i,
    double                           g_row,
    double                           H_row,
    const ParsedArm&                 pa,
    int                              k_arm,
    const std::vector<LatentBlock>&  blocks,
    const std::vector<double>&       d_eff_cache,
    DenseVec&                        grad,
    SparseHessianBuilder&            H,
    std::vector<int>&                active_idx,
    std::vector<double>&             active_d
) {
    const int p_k    = pa.p;
    const int n_re_k = pa.n_re_groups;
    const int bstart = pa.beta_start;
    const int rstart = pa.re_start;
    const int B      = static_cast<int>(blocks.size());

    int g_re = -1;
    if (n_re_k > 0) {
        int gi = static_cast<int>(pa.re_idx[i]) - 1;
        if (gi >= 0 && gi < n_re_k) g_re = rstart + gi;
    }

    (void) B;
    active_idx.clear();
    active_d.clear();
    collect_coupled_row_latents(i, k_arm, blocks, d_eff_cache, active_idx, active_d);
    const int A = static_cast<int>(active_idx.size());

    // beta block: gradient + lower-triangle beta/beta + beta/RE + beta/active.
    for (int j = 0; j < p_k; j++) {
        const double Xij = pa.X(i, j);
        grad[bstart + j] += g_row * Xij;
        for (int l = 0; l <= j; l++) {
            H.add(bstart + j, bstart + l, H_row * Xij * pa.X(i, l));
        }
        if (g_re >= 0) {
            H.add(bstart + j, g_re, H_row * Xij);
        }
        for (int a = 0; a < A; a++) {
            H.add(bstart + j, active_idx[a], H_row * Xij * active_d[a]);
        }
    }

    // RE block: gradient + diagonal + RE/active.
    if (g_re >= 0) {
        grad[g_re] += g_row;
        H.add(g_re, g_re, H_row);
        for (int a = 0; a < A; a++) {
            H.add(g_re, active_idx[a], H_row * active_d[a]);
        }
    }

    // Active x active (intra + inter block): gradient + lower triangle.
    for (int a = 0; a < A; a++) {
        grad[active_idx[a]] += g_row * active_d[a];
        for (int b = 0; b <= a; b++) {
            H.add(active_idx[a], active_idx[b],
                  H_row * active_d[a] * active_d[b]);
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
        d_eff_cache[b] = s * blocks[b].d_fac(k_grid);
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
// Cell-coupling per-cell branch (dense path, gcol33/tulpa#32 Change 2b).
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
// B.1 scope: single-arm cell coupling only (the per-cell `CellDerivs`
// writes only `arm_grad` + `arm_neg_hess_diag`; the cross-arm Hessian
// fields are left unread). Cross-arm Hessian + the sparse twin land
// with B.2 alongside the real tulpaObs `OccuCoverLognormalCoupling`
// consumer.
// ============================================================================

// Evaluate the cell-coupling spec's per-cell log-density sum at the
// current etas, discarding derivatives. Used by the log-lik functor
// during Newton line search (the scatter pass computes the same call
// with derivatives kept; B.1 accepts the 2x evaluation cost as the
// trivial way to keep the line-search objective consistent with the
// scatter -- caching the scatter's derivatives across line-search
// rejects is a B.2 micro-opt if the spec evaluate is expensive).
// `phi_override` (optional, length n_coupled): per-coupled-arm dispersion to use
// in place of the shared `arms[k].phi`. The threaded sparse outer-grid driver
// passes a per-thread snapshot taken under the phi-sync critical, because the
// gridded coupled-arm dispersion (e.g. the beta precision on `phi.grid.pos`) is
// rewritten in `arms` per cell by a concurrent thread's `prep_at_grid`; reading
// the shared `arms[k].phi` lock-free here would race it (gcol33/tulpaObs#42).
// nullptr keeps the direct `arms[k].phi` read for the serial / dense callers.
inline double eval_cell_coupling_log_lik(
    const CellCouplingSpec&                       spec,
    const std::vector<int>&                       coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int                                           n_cells,
    const std::vector<JointArm>&                  arms,
    const std::vector<Rcpp::NumericVector>&       etas,
    const double*                                 phi_override = nullptr
) {
    const int n_coupled = (int)coupled_arms.size();
    if (n_coupled == 0 || n_cells == 0) return 0.0;

    std::vector<const double*> arm_eta_ptr(n_coupled);
    std::vector<const double*> arm_y_ptr(n_coupled);
    std::vector<const int*>    arm_n_trials_ptr(n_coupled);
    std::vector<std::string>   family_holder(n_coupled);
    std::vector<const char*>   arm_family_ptr(n_coupled);
    std::vector<double>        arm_phi_vec(n_coupled);
    for (int kk = 0; kk < n_coupled; kk++) {
        int k = coupled_arms[kk];
        arm_eta_ptr[kk]      = REAL(etas[k]);
        arm_y_ptr[kk]        = arms[k].y.size()       > 0 ? REAL(arms[k].y)             : nullptr;
        arm_n_trials_ptr[kk] = arms[k].n_trials.size()> 0 ? INTEGER(arms[k].n_trials)   : nullptr;
        family_holder[kk]    = arms[k].family;
        arm_family_ptr[kk]   = family_holder[kk].c_str();
        arm_phi_vec[kk]      = phi_override ? phi_override[kk] : arms[k].phi;
    }

    std::vector<int>            arm_row_count(n_coupled);
    std::vector<const int*>     arm_rows_ptr(n_coupled);
    std::vector<std::vector<double>> arm_grad_buf(n_coupled);
    std::vector<std::vector<double>> arm_neg_hess_diag_buf(n_coupled);
    std::vector<double*>        arm_grad_ptr(n_coupled);
    std::vector<double*>        arm_neg_hess_diag_ptr(n_coupled);

    double total = 0.0;
    for (int c = 0; c < n_cells; c++) {
        for (int kk = 0; kk < n_coupled; kk++) {
            int rc = (int)cell_rows[kk][c].size();
            arm_row_count[kk] = rc;
            arm_rows_ptr[kk]  = cell_rows[kk][c].data();
            if ((int)arm_grad_buf[kk].size() < rc) {
                arm_grad_buf[kk].assign(rc, 0.0);
                arm_neg_hess_diag_buf[kk].assign(rc, 0.0);
            } else {
                std::fill(arm_grad_buf[kk].begin(),
                          arm_grad_buf[kk].begin() + rc, 0.0);
                std::fill(arm_neg_hess_diag_buf[kk].begin(),
                          arm_neg_hess_diag_buf[kk].begin() + rc, 0.0);
            }
            arm_grad_ptr[kk]          = arm_grad_buf[kk].data();
            arm_neg_hess_diag_ptr[kk] = arm_neg_hess_diag_buf[kk].data();
        }
        CellEtas etas_view;
        etas_view.arm_eta_ptr   = arm_eta_ptr.data();
        etas_view.arm_rows      = arm_rows_ptr.data();
        etas_view.arm_row_count = arm_row_count.data();
        etas_view.n_arms_       = n_coupled;
        CellResponse y_view;
        y_view.arm_y           = arm_y_ptr.data();
        y_view.arm_n_trials    = arm_n_trials_ptr.data();
        y_view.arm_family      = arm_family_ptr.data();
        y_view.arm_phi         = arm_phi_vec.data();
        y_view.arm_rows        = arm_rows_ptr.data();
        y_view.arm_row_count   = arm_row_count.data();
        y_view.n_arms_         = n_coupled;
        CellDerivs out;
        out.arm_grad           = arm_grad_ptr.data();
        out.arm_neg_hess_diag  = arm_neg_hess_diag_ptr.data();
        out.arm_cross_hess     = nullptr;
        out.arm_row_count      = arm_row_count.data();
        out.n_arms_            = n_coupled;
        total += spec.evaluate_cell(c, etas_view, y_view, out);
    }
    return total;
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
    // Hessian scatter writes for this row.
    static thread_local std::vector<int>    lat_idx;
    static thread_local std::vector<double> lat_w;
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

// Cross-chain scatter (sparse). H.add() normalizes to (max, min); a single
// write per (a, b) iter accumulates the correct off-diagonal contribution
// to the symmetric matrix. The diagonal (a == b) case needs an extra write
// to match the dense version's 2*val accumulation.
inline void scatter_cross_chain_sparse(
    double                                Hkl,
    const std::vector<ArmRowChainEntry>&  chain_k,
    const std::vector<ArmRowChainEntry>&  chain_l,
    SparseHessianBuilder&                 H
) {
    if (Hkl == 0.0) return;
    for (const auto& e_k : chain_k) {
        for (const auto& e_l : chain_l) {
            double val = Hkl * e_k.w * e_l.w;
            H.add(e_k.idx, e_l.idx, val);
            if (e_k.idx == e_l.idx) {
                H.add(e_k.idx, e_l.idx, val);
            }
        }
    }
}

// Per-cell scatter branch. Walks cells, dispatches to
// `spec->evaluate_cell()`, and scatters per-arm row derivatives via the
// caller-supplied per-row helper (`scatter_one_arm_row_dense` or
// `scatter_one_arm_row_sparse`) plus the cross-arm Hessian via the
// caller-supplied cross-chain helper. Single source of truth for the
// cell iteration, view construction, per-cell cross_hess buffer
// allocation, and per-arm bookkeeping; dense/sparse share everything
// except the H container and the two helpers.
template <typename HType, typename ScatterRowFn, typename ScatterCrossFn>
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
    DenseVec&                                     grad,
    HType&                                        H,
    ScatterRowFn                                  scatter_row,
    ScatterCrossFn                                scatter_cross,
    CurvatureMode                                 curvature = CurvatureMode::Observed,
    bool                                          grad_only = false,
    const double*                                 phi_override = nullptr
) {
    const int n_coupled = (int)coupled_arms.size();
    const int B         = (int)blocks.size();
    if (n_coupled == 0 || n_cells == 0) return;

    // Per-arm d_eff cache (one entry per coupled arm × block).
    std::vector<std::vector<double>> d_eff_per_arm(n_coupled,
                                                    std::vector<double>(B));
    for (int kk = 0; kk < n_coupled; kk++) {
        int k = coupled_arms[kk];
        for (int b = 0; b < B; b++) {
            double s = blocks[b].arm_scale
                        ? blocks[b].arm_scale(k, k_grid)
                        : 1.0;
            d_eff_per_arm[kk][b] = s * blocks[b].d_fac(k_grid);
        }
    }

    // Per-cell pointer arrays for the CellEtas/Response/Derivs views.
    // Reused across cells -- the per-arm pointers themselves are stable
    // across cells (etas / family / phi); only the per-arm row slice
    // changes per cell.
    std::vector<const double*> arm_eta_ptr(n_coupled);
    std::vector<const double*> arm_y_ptr(n_coupled);
    std::vector<const int*>    arm_n_trials_ptr(n_coupled);
    std::vector<std::string>   family_holder(n_coupled);
    std::vector<const char*>   arm_family_ptr(n_coupled);
    std::vector<double>        arm_phi_vec(n_coupled);
    for (int kk = 0; kk < n_coupled; kk++) {
        int k = coupled_arms[kk];
        arm_eta_ptr[kk]      = REAL(etas[k]);
        arm_y_ptr[kk]        = arms[k].y.size()       > 0 ? REAL(arms[k].y)             : nullptr;
        arm_n_trials_ptr[kk] = arms[k].n_trials.size()> 0 ? INTEGER(arms[k].n_trials)   : nullptr;
        family_holder[kk]    = arms[k].family;
        arm_family_ptr[kk]   = family_holder[kk].c_str();
        arm_phi_vec[kk]      = phi_override ? phi_override[kk] : arms[k].phi;
    }

    // Per-cell scratch (row counts + row pointer arrays + derivative
    // buffers). Sized at construction; capacity grows monotonically as
    // larger cells are encountered.
    std::vector<int>            arm_row_count(n_coupled);
    std::vector<const int*>     arm_rows_ptr(n_coupled);
    std::vector<std::vector<double>> arm_grad_buf(n_coupled);
    std::vector<std::vector<double>> arm_neg_hess_diag_buf(n_coupled);
    std::vector<double*>        arm_grad_ptr(n_coupled);
    std::vector<double*>        arm_neg_hess_diag_ptr(n_coupled);

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

    // Per-row chain scratch reused across rows / pairs.
    std::vector<int>    active_idx;
    std::vector<double> active_d;
    active_idx.reserve(B);
    active_d.reserve(B);
    std::vector<ArmRowChainEntry> chain_k_scratch;
    std::vector<ArmRowChainEntry> chain_l_scratch;
    chain_k_scratch.reserve(32);
    chain_l_scratch.reserve(32);

    for (int c = 0; c < n_cells; c++) {
        for (int kk = 0; kk < n_coupled; kk++) {
            int rc = (int)cell_rows[kk][c].size();
            arm_row_count[kk] = rc;
            arm_rows_ptr[kk]  = cell_rows[kk][c].data();
            if ((int)arm_grad_buf[kk].size() < rc) {
                arm_grad_buf[kk].assign(rc, 0.0);
                arm_neg_hess_diag_buf[kk].assign(rc, 0.0);
            } else {
                std::fill(arm_grad_buf[kk].begin(),
                          arm_grad_buf[kk].begin() + rc, 0.0);
                std::fill(arm_neg_hess_diag_buf[kk].begin(),
                          arm_neg_hess_diag_buf[kk].begin() + rc, 0.0);
            }
            arm_grad_ptr[kk]          = arm_grad_buf[kk].data();
            arm_neg_hess_diag_ptr[kk] = arm_neg_hess_diag_buf[kk].data();
        }

        // Cross-arm Hessian buffers: allocate one J_kk * J_ll slab per
        // (kk, ll) pair with kk <= ll. kk > ll stays nullptr per the
        // CellDerivs contract; integration symmetrises.
        for (int kk = 0; kk < n_coupled; kk++) {
            int rc_k = arm_row_count[kk];
            for (int ll = kk; ll < n_coupled; ll++) {
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

        CellEtas etas_view;
        etas_view.arm_eta_ptr   = arm_eta_ptr.data();
        etas_view.arm_rows      = arm_rows_ptr.data();
        etas_view.arm_row_count = arm_row_count.data();
        etas_view.n_arms_       = n_coupled;

        CellResponse y_view;
        y_view.arm_y           = arm_y_ptr.data();
        y_view.arm_n_trials    = arm_n_trials_ptr.data();
        y_view.arm_family      = arm_family_ptr.data();
        y_view.arm_phi         = arm_phi_vec.data();
        y_view.arm_rows        = arm_rows_ptr.data();
        y_view.arm_row_count   = arm_row_count.data();
        y_view.n_arms_         = n_coupled;

        CellDerivs out;
        out.arm_grad           = arm_grad_ptr.data();
        out.arm_neg_hess_diag  = arm_neg_hess_diag_ptr.data();
        out.arm_cross_hess     = cross_hess_outer.data();
        out.arm_row_count      = arm_row_count.data();
        out.n_arms_            = n_coupled;
        out.curvature          = curvature;
        out.grad_only          = grad_only;

        spec.evaluate_cell(c, etas_view, y_view, out);

        // Within-arm per-row scatter (eta diagonal Hessian + gradient).
        for (int kk = 0; kk < n_coupled; kk++) {
            int k  = coupled_arms[kk];
            int rc = arm_row_count[kk];
            const int*    rows = arm_rows_ptr[kk];
            const double* g    = arm_grad_buf[kk].data();
            const double* h    = arm_neg_hess_diag_buf[kk].data();
            for (int j = 0; j < rc; j++) {
                scatter_row(
                    rows[j], g[j], h[j],
                    parsed[k], k, blocks, d_eff_per_arm[kk],
                    grad, H, active_idx, active_d
                );
            }
        }

        // Cross-arm Hessian scatter. For each (kk, ll) pair with kk <= ll,
        // walk rows of arm kk x arm ll. Same-arm (kk == ll) skips the
        // j == m diagonal (already in arm_neg_hess_diag) and reads only
        // off-diagonal entries. Pure curvature with no gradient contribution,
        // so a grad-only step (cached-factor reuse) skips it entirely.
        if (!grad_only) {
            for (int kk = 0; kk < n_coupled; kk++) {
                int k     = coupled_arms[kk];
                int rc_k  = arm_row_count[kk];
                const int* rows_k = arm_rows_ptr[kk];
                for (int ll = kk; ll < n_coupled; ll++) {
                    const double* ch = cross_hess_ptr_inner[kk][ll];
                    if (!ch) continue;
                    int l     = coupled_arms[ll];
                    int rc_l  = arm_row_count[ll];
                    const int* rows_l = arm_rows_ptr[ll];
                    for (int j = 0; j < rc_k; j++) {
                        build_arm_row_chain(rows_k[j], parsed[k], k, blocks,
                                            d_eff_per_arm[kk], chain_k_scratch);
                        int m_start = (kk == ll) ? (j + 1) : 0;
                        for (int m = m_start; m < rc_l; m++) {
                            double Hkl = ch[(std::size_t)j * rc_l + m];
                            if (Hkl == 0.0) continue;
                            build_arm_row_chain(rows_l[m], parsed[l], l, blocks,
                                                d_eff_per_arm[ll], chain_l_scratch);
                            scatter_cross(Hkl, chain_k_scratch, chain_l_scratch, H);
                        }
                    }
                }
            }
        }
    }
}

// Dense wrapper: per-cell branch with `scatter_one_arm_row_dense` (writes
// both directions into the n_x x n_x DenseMat).
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
    scatter_cell_coupling_branch_impl(
        spec, coupled_arms, cell_rows, n_cells,
        arms, parsed, etas, blocks, k_grid, grad, H,
        scatter_one_arm_row_dense,
        scatter_cross_chain_dense,
        curvature, /*grad_only=*/false, phi_override
    );
}

// Sparse wrapper: per-cell branch with `scatter_one_arm_row_sparse`
// (lower-triangle writes into the joint SparseHessianBuilder; the pattern
// must already cover every (row, col) the per-row helper touches, which
// build_joint_hessian_pattern guarantees per arm regardless of coupling
// since it iterates all arms).
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
    CurvatureMode                                 curvature = CurvatureMode::Observed,
    bool                                          grad_only = false,
    const double*                                 phi_override = nullptr
) {
    scatter_cell_coupling_branch_impl(
        spec, coupled_arms, cell_rows, n_cells,
        arms, parsed, etas, blocks, k_grid, grad, H,
        scatter_one_arm_row_sparse,
        scatter_cross_chain_sparse,
        curvature, grad_only, phi_override
    );
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
        d_eff_cache[b] = s * blocks[b].d_fac(k_grid);
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

    // Stage 2.2b fast path: when no DENSE_BASIS block is present, the
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

        // INDEXED active dofs across all non-DENSE_BASIS blocks.
        active_scratch.clear();
        // DENSE_BASIS active dofs — populated only when cross terms are
        // needed; otherwise left empty so the per-obs loop costs O(p_k +
        // n_re_k + A_idx^2) instead of O(M^2).
        active_db_scratch.clear();

        for (int b = 0; b < B; b++) {
            const LatentBlock& blk = blocks[b];
            const double d_eff = d_eff_cache[b];
            // Field_coef = 0 arms (or any block with arm_scale * d_fac == 0)
            // contribute nothing to gradient or Hessian for this arm; skip
            // the per-block resolution work entirely.
            if (d_eff == 0.0) continue;

            if (kind_cache[b] == BlockContribKind::DENSE_BASIS) {
                const bool has_batch = static_cast<bool>(blk.dense_basis_batch);
                if (!need_db_in_perobs && has_batch) continue;

                blk.basis_eval(i, k_arm, basis_scratch.data());
                for (int j = 0; j < blk.size; j++) {
                    double w = basis_scratch[j] * d_eff;
                    if (w != 0.0) {
                        DenseBasisActive e;
                        e.dof       = blk.start + j;
                        e.weight    = w;
                        e.block_idx = b;
                        e.has_batch = has_batch;
                        active_db_scratch.push_back(e);
                    }
                }
            } else if (kind_cache[b] == BlockContribKind::INDEXED_SINGLE) {
                int l = blk.idx(i, k_arm);
                if (l > 0 && l <= blk.size) {
                    double w = d_eff * block_row_weight(blk, i, k_arm);
                    if (w != 0.0) {
                        active_scratch.emplace_back(blk.start + l - 1, w);
                    }
                }
            } else if (kind_cache[b] == BlockContribKind::BILINEAR_FACTOR) {
                // eta_i += d_eff * u * lambda. Gauss-Newton linearization:
                // emit u_slot with weight (lambda * d_eff) and lambda_slot
                // with weight (u * d_eff). Active × active fills the
                // mixed-curvature (u, lambda) Hessian entry naturally.
                auto [u_slot, lambda_slot] = blk.obs_factor_lambda(i, k_arm);
                if (u_slot >= 0 && lambda_slot >= 0) {
                    const double u_val      = x[u_slot];
                    const double lambda_val = x[lambda_slot];
                    active_scratch.emplace_back(u_slot,      lambda_val * d_eff);
                    active_scratch.emplace_back(lambda_slot, u_val      * d_eff);
                }
            } else {  // INDEXED_MULTI
                multi_scratch.clear();
                blk.obs_indices(i, k_arm, multi_scratch);
                for (const auto& [l, w_local] : multi_scratch) {
                    if (l > 0 && l <= blk.size) {
                        active_scratch.emplace_back(blk.start + l - 1,
                                                     w_local * d_eff);
                    }
                }
            }
        }
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
            scatter_dense_basis_block(blocks[b], k_arm, pa, arm, view, eta,
                                       d_eff_cache[b], grad, H,
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
                const double d_e = d_eff[b][k_arm];
                if (d_e == 0.0) continue;
                switch (blk.contrib_kind) {
                case BlockContribKind::INDEXED_SINGLE: {
                    if (!blk.idx) break;
                    int l = blk.idx(i, k_arm);
                    if (l > 0 && l <= blk.size) {
                        e += d_e * block_row_weight(blk, i, k_arm)
                                 * x[blk.start + l - 1];
                    }
                    break;
                }
                case BlockContribKind::INDEXED_MULTI: {
                    if (!blk.obs_indices) break;
                    blk.obs_indices(i, k_arm, multi_scratch);
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
                    blk.basis_eval(i, k_arm, basis_scratch.data());
                    double acc = 0.0;
                    for (int j = 0; j < blk.size; j++) {
                        acc += basis_scratch[j] * x[blk.start + j];
                    }
                    e += d_e * acc;
                    break;
                }
                case BlockContribKind::BILINEAR_FACTOR: {
                    if (!blk.obs_factor_lambda) break;
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
    CurvatureMode                    step_curvature = CurvatureMode::Observed,
    int                              hessian_refresh = 1,
    int                              n_threads_outer = 1,
    tulpa_progress::GridProgress*    progress = nullptr,
    GridCheckpoint*                  checkpoint = nullptr
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
    CurvatureMode                    step_curvature = CurvatureMode::Observed,
    int                              hessian_refresh = 1,
    tulpa_progress::GridProgress*    progress = nullptr,
    GridCheckpoint*                  checkpoint = nullptr
);

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_JOINT_MULTI_H
