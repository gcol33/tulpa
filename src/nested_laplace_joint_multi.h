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
#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// Per-observation latent-block scatter for one arm at one grid point.
//
// Variable-length analogue of the single-arm multi-block scatter
// (accumulate_latent_cross_terms in nested_laplace_multi.h), with the
// β/β and β/RE diagonal blocks evaluated *per arm* using ParsedArm
// offsets so multiple likelihood arms can share the same latent vector.
// Each block's eta contribution carries an optional per-arm scaling
// factor (arm_scale) for INLA `copy=` semantics.
//
// Accumulates β/RE/RE×β diagonal blocks (per arm, unchanged from
// joint_core), then for each obs i resolves the active subset of blocks
// (those with idx_b(i, k_arm) in [1, size_b]) and adds the latent
// gradient and β×latent / RE×latent / latent×latent Hessian cross-terms
// with the effective coefficient d_eff_b = arm_scale_b(k_arm, k_grid) *
// d_fac_b(k_grid).
inline void scatter_arm_obs_joint_multi(
    const Rcpp::NumericVector& /*x*/,
    const Rcpp::NumericVector&    eta,
    const ParsedArm&              pa,
    const JointArm&               arm,
    int                           k_arm,
    const std::vector<LatentBlock>& blocks,
    int                           k_grid,
    DenseVec&                     grad,
    DenseMat&                     H
) {
    const int p_k      = pa.p;
    const int n_re_k   = pa.n_re_groups;
    const int bstart   = pa.beta_start;
    const int rstart   = pa.re_start;
    const std::string& family = arm.family;
    const double phi_disp     = arm.phi;
    const int B = static_cast<int>(blocks.size());

    // Cache per-block effective coefficient for this (k_arm, k_grid).
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
        auto gh = grad_hess_for_family(
            arm.y[i], arm.n_trials[i], eta[i], family, phi_disp);

        int g_re = -1;
        if (n_re_k > 0) {
            int gi = static_cast<int>(pa.re_idx[i]) - 1;
            if (gi >= 0 && gi < n_re_k) g_re = rstart + gi;
        }

        // Resolve active blocks for obs i. -1 from idx means "this obs
        // doesn't see this block" (e.g. an obs with no spatial unit).
        active_idx.clear();
        active_d.clear();
        for (int b = 0; b < B; b++) {
            int l_b = blocks[b].idx(i, k_arm);
            if (l_b > 0 && l_b <= blocks[b].size) {
                active_idx.push_back(blocks[b].start + l_b - 1);
                active_d.push_back(d_eff_cache[b]);
            }
        }
        const int A = static_cast<int>(active_idx.size());

        // β block: gradient + diagonal-block Hessian + cross with RE and
        // every active latent block.
        for (int j = 0; j < p_k; j++) {
            const double Xij = pa.X(i, j);
            grad[bstart + j] += gh.grad * Xij;
            for (int l = 0; l < p_k; l++) {
                H[bstart + j][bstart + l] += gh.neg_hess * Xij * pa.X(i, l);
            }
            if (g_re >= 0) {
                H[bstart + j][g_re] += gh.neg_hess * Xij;
                H[g_re][bstart + j] += gh.neg_hess * Xij;
            }
            for (int a = 0; a < A; a++) {
                H[bstart + j][active_idx[a]] += gh.neg_hess * Xij * active_d[a];
                H[active_idx[a]][bstart + j] += gh.neg_hess * Xij * active_d[a];
            }
        }

        // RE block: gradient + diagonal + cross with active latent indices.
        if (g_re >= 0) {
            grad[g_re] += gh.grad;
            H[g_re][g_re] += gh.neg_hess;
            for (int a = 0; a < A; a++) {
                H[g_re][active_idx[a]] += gh.neg_hess * active_d[a];
                H[active_idx[a]][g_re] += gh.neg_hess * active_d[a];
            }
        }

        // Latent x latent block (intra-block + inter-block). Includes both
        // the diagonal at (idx_a, idx_a) and the off-diagonal (idx_a, idx_b)
        // for a != b.
        for (int a = 0; a < A; a++) {
            grad[active_idx[a]] += gh.grad * active_d[a];
            for (int b = 0; b < A; b++) {
                H[active_idx[a]][active_idx[b]] +=
                    gh.neg_hess * active_d[a] * active_d[b];
            }
        }
    }
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
    const ScatterIndexCache*          idx_cache = nullptr
) {
    const int p_k      = pa.p;
    const int n_re_k   = pa.n_re_groups;
    const int bstart   = pa.beta_start;
    const int rstart   = pa.re_start;
    const std::string& family = arm.family;
    const double phi_disp     = arm.phi;
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
            x, eta, pa, arm, d_eff_cache, idx_cache->arm[k_arm],
            grad, H, family, phi_disp
        );
        return;
    }

    for (int i = 0; i < arm.N; i++) {
        auto gh = grad_hess_for_family(
            arm.y[i], arm.n_trials[i], eta[i], family, phi_disp);

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
                    active_scratch.emplace_back(blk.start + l - 1, d_eff);
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
            scatter_dense_basis_block(blocks[b], k_arm, pa, arm, eta,
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
            double e = 0.0;
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
                        e += d_e * x[blk.start + l - 1];
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
inline Rcpp::List run_multi_block_nested_laplace_joint_sparse_impl(
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
    double                           prune_tol
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
inline Rcpp::List run_multi_block_nested_laplace_joint(
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
    bool                             force_sparse = false
) {
    const int n_arms = static_cast<int>(arms.size());
    if (static_cast<int>(parsed.size()) != n_arms) {
        Rcpp::stop("parsed and arms vectors must have the same length.");
    }

    int n_x = n_x_after_re;
    for (const auto& b : blocks) {
        n_x = std::max(n_x, b.start + b.size);
    }

    // Sparse-path detection. Triggered by (a) user opt-in via force_sparse,
    // (b) n_x crossing the dense-Newton threshold, or (c) any block whose
    // contrib_kind is not INDEXED_SINGLE — the dense scatter
    // (scatter_arm_obs_joint_multi) only handles INDEXED_SINGLE via the
    // `idx` callback, so SPDE/HSGP/etc. require the sparse path even at
    // small n_x.
    bool needs_sparse = force_sparse || (n_x >= SPARSE_THRESHOLD);
    if (!needs_sparse) {
        for (const auto& b : blocks) {
            if (b.contrib_kind != BlockContribKind::INDEXED_SINGLE) {
                needs_sparse = true;
                break;
            }
        }
    }
    if (needs_sparse) {
        // First-ship: sparse path is serial outer-grid. Each per-thread
        // SparseHessianBuilder would replicate the pattern (a few × 10^7
        // entries at n_sites = 10^6); serializing is the simplest correct
        // shape. Parallel sparse is a follow-up to this stage.
        return run_multi_block_nested_laplace_joint_sparse_impl(
            n_grid, arms, parsed, blocks, n_x,
            max_iter, tol, n_threads,
            store_modes, x_init, store_Q,
            prep_at_grid, tile_ids, tile_pilot_cells, prune_tol
        );
    }

    // Per-outer-thread Newton scratch + CHOLMOD solver pool. The solver pool
    // lives inside run_nested_laplace_grid (one per outer thread); scratch
    // pool lives here because it's joint-shaped (per-arm etas).
    int n_outer = std::max(1, n_threads_outer);
    std::vector<NewtonScratchJoint> scratch_pool(n_outer);
    for (auto& s : scratch_pool) s.allocate(n_x, arms);

    // Force inner OpenMP to single-thread when the outer grid is parallel —
    // see run_multi_block_nested_laplace for the rationale.
    int n_threads_inner_eff = (n_outer > 1) ? 1 : n_threads;

    // Inner implementation: takes max_iter as a parameter so the cheap-pass
    // path can call this with max_iter=1 for a one-Newton-step screen at the
    // pilot mode. The 3-arg `solve_at_theta` wrapper below threads the outer
    // `max_iter` through.
    auto solve_at_theta_impl = [&](int k_grid,
                                   const std::vector<double>& prev_mode,
                                   SparseCholeskySolver* shared_solver,
                                   int max_iter_use,
                                   NewtonScratchJoint* scratch_override
                                   = nullptr) -> LaplaceResult
    {
        if (prep_at_grid) prep_at_grid(k_grid);
        for (const auto& b : blocks) {
            if (b.prep && !b.prep(k_grid)) {
                LaplaceResult bad;
                bad.mode = (static_cast<int>(prev_mode.size()) == n_x)
                           ? prev_mode
                           : std::vector<double>(n_x, 0.0);
                bad.log_marginal = -std::numeric_limits<double>::infinity();
                bad.n_iter = 0;
                bad.converged = false;
                bad.log_det_Q = 0.0;
                return bad;
            }
        }

        // Cache per-block (k_arm, k_grid) -> d_eff. Per-block d_fac(k_grid)
        // is evaluated once; per-arm scaling is re-evaluated inside the
        // per-arm loops because compute_eta is called from inside the
        // Newton step many times.
        const int B = static_cast<int>(blocks.size());
        std::vector<double> d_fac_cache(B);
        for (int b = 0; b < B; b++) {
            d_fac_cache[b] = blocks[b].d_fac(k_grid);
        }

        auto compute_eta_joint = [&](const Rcpp::NumericVector& x,
                                     std::vector<Rcpp::NumericVector>& etas) {
            for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                const ParsedArm& pa = parsed[k_arm];
                const int N_k    = arms[k_arm].N;
                const int p_k    = pa.p;
                const int n_re_k = pa.n_re_groups;
                const int bstart = pa.beta_start;
                const int rstart = pa.re_start;

                // Per-arm effective coefficients per block.
                std::vector<double> d_eff(B);
                for (int b = 0; b < B; b++) {
                    double s = blocks[b].arm_scale
                                ? blocks[b].arm_scale(k_arm, k_grid)
                                : 1.0;
                    d_eff[b] = s * d_fac_cache[b];
                }

                #ifdef _OPENMP
                #pragma omp parallel for schedule(static) \
                    num_threads(n_threads_inner_eff > 0 ? n_threads_inner_eff : 1) \
                    if(n_threads_inner_eff > 1)
                #endif
                for (int i = 0; i < N_k; i++) {
                    double e = 0.0;
                    for (int j = 0; j < p_k; j++) e += pa.X(i, j) * x[bstart + j];
                    if (n_re_k > 0) {
                        int g = static_cast<int>(pa.re_idx[i]) - 1;
                        if (g >= 0 && g < n_re_k) e += x[rstart + g];
                    }
                    for (int b = 0; b < B; b++) {
                        int l = blocks[b].idx(i, k_arm);
                        if (l > 0 && l <= blocks[b].size) {
                            e += d_eff[b] * x[blocks[b].start + l - 1];
                        }
                    }
                    etas[k_arm][i] = e;
                }
            }
        };

        auto scatter_joint = [&](const Rcpp::NumericVector& x,
                                 const std::vector<Rcpp::NumericVector>& etas,
                                 DenseVec& grad, DenseMat& H) {
            for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                scatter_arm_obs_joint_multi(
                    x, etas[k_arm], parsed[k_arm], arms[k_arm], k_arm,
                    blocks, k_grid, grad, H
                );
            }
            for (const auto& b : blocks) {
                if (b.add_prior) b.add_prior(grad, H, x, k_grid);
            }
            add_per_arm_beta_re_priors(grad, H, x, parsed);
        };

        auto center_joint = [&](Rcpp::NumericVector& x) {
            for (int b = 0; b < B; b++) {
                if (!blocks[b].center) continue;
                double c_b = blocks[b].center(x);
                if (std::abs(c_b) < 1e-15) continue;
                // Per-arm intercept compensation so eta is preserved when a
                // rank-deficient block is re-centered after a Newton step.
                // arm k's first beta column absorbs the constant
                // arm_scale_b(k_arm, k_grid) * d_fac_b(k_grid) * c_b that
                // the centerer removed from x[block]. See the BYM2 / ICAR
                // joint kernel centerers in nested_laplace_joint.cpp for
                // the load-bearing rationale.
                for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                    if (parsed[k_arm].p == 0) continue;
                    double s = blocks[b].arm_scale
                                ? blocks[b].arm_scale(k_arm, k_grid)
                                : 1.0;
                    x[parsed[k_arm].beta_start] += s * d_fac_cache[b] * c_b;
                }
            }
        };

        auto log_prior_joint = [&](const Rcpp::NumericVector& x,
                                    const std::vector<Rcpp::NumericVector>&)
            -> double {
            double lp = log_prior_per_arm_re(x, parsed);
            for (const auto& b : blocks) {
                if (b.log_prior) lp += b.log_prior(x, k_grid);
            }
            return lp;
        };

        int tid;
        #ifdef _OPENMP
        tid = omp_in_parallel() ? omp_get_thread_num() : 0;
        #else
        tid = 0;
        #endif

        NewtonScratchJoint& scratch = scratch_override
                                       ? *scratch_override
                                       : scratch_pool[tid];

        return laplace_newton_solve_joint(
            arms, n_x,
            max_iter_use, tol, n_threads_inner_eff,
            compute_eta_joint, scatter_joint, center_joint, log_prior_joint,
            scratch, prev_mode, shared_solver,
            store_Q
        );
    };

    // 3-arg adapter for run_nested_laplace_grid (which calls
    // solve_at_theta(k, prev_mode, solver) without knowing about the
    // max_iter parameter). Threads the outer `max_iter` through.
    auto solve_at_theta = [&](int k_grid,
                              const std::vector<double>& prev_mode,
                              SparseCholeskySolver* shared_solver) -> LaplaceResult
    {
        return solve_at_theta_impl(k_grid, prev_mode, shared_solver,
                                    max_iter, nullptr);
    };

    // Cheap-pass screening (Phase 3, dev_notes/speedup.md): one Newton step
    // from the pilot mode, then report the Laplace log-marginal at that
    // quasi-mode. This is much more accurate than evaluating `log_lik +
    // log_prior` at the raw pilot mode — for cells whose data MAP differs
    // from x_pilot (especially when sigma or alpha shifts the design
    // multiplicatively), one Newton step corrects the worst of the pilot
    // bias.
    //
    // Cost per cell: 1 eta+scatter+Cholesky factor+step ≈ 20% of the full
    // 5-iter Newton. The cheap pass runs serially after the pilot solve and
    // before any parallel region, so a dedicated thread-local solver +
    // scratch keep it isolated from the parallel fan-out's pool (the pool's
    // entries are reserved for the inner Newton on survivors).
    SparseCholeskySolver cheap_solver;
    NewtonScratchJoint cheap_scratch;
    cheap_scratch.allocate(n_x, arms);
    auto cheap_eval = [&](int k_grid,
                          const std::vector<double>& x_pilot) -> double {
        LaplaceResult r = solve_at_theta_impl(
            k_grid, x_pilot, &cheap_solver,
            /*max_iter_use=*/1, &cheap_scratch);
        return r.log_marginal;
    };

    return run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, store_modes, n_outer,
        tile_ids, tile_pilot_cells,
        cheap_eval, prune_tol
    );
}

// Sparse-path implementation. Mirrors the dense run_multi_block_nested_
// laplace_joint structure: per-cell prep + Newton solve + log_marginal,
// pushed through run_nested_laplace_grid. Differences are isolated to:
//   * SparseHessianBuilder allocated once (pattern fixed across the grid)
//   * NewtonScratchJointSparse (no DenseMat H)
//   * compute_eta dispatches on contrib_kind (INDEXED_SINGLE/MULTI/DENSE_BASIS)
//   * scatter calls add_prior_sparse + scatter_arm_obs_joint_multi_sparse
//   * laplace_newton_solve_joint_sparse (CHOLMOD-only factor/solve path)
inline Rcpp::List run_multi_block_nested_laplace_joint_sparse_impl(
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
    double                           prune_tol
) {
    const int n_arms = static_cast<int>(arms.size());
    const int B      = static_cast<int>(blocks.size());

    // Build the joint H pattern ONCE. Reused for every outer-grid cell.
    SparseHessianBuilder H_builder;
    { TULPA_PROFILE_PHASE(PHASE_PATTERN_BUILD);
      build_joint_hessian_pattern(parsed, arms, blocks, n_x, H_builder); }

    NewtonScratchJointSparse scratch;
    scratch.allocate(n_x, arms);

    // Cheap-pass dedicated scratch / solver / builder. Pattern reused;
    // VALUES live in the builder, so each builder needs its own copy.
    SparseHessianBuilder cheap_builder;
    { TULPA_PROFILE_PHASE(PHASE_PATTERN_BUILD);
      build_joint_hessian_pattern(parsed, arms, blocks, n_x, cheap_builder); }
    NewtonScratchJointSparse cheap_scratch;
    cheap_scratch.allocate(n_x, arms);
    SparseCholeskySolver cheap_solver;

    // Stage 2.2b: precompute per-(k_arm, i) flat-index cache for the
    // scatter. Built once per H_builder; disabled automatically when any
    // block is BILINEAR_FACTOR (dynamic active weights). DENSE_BASIS
    // presence is recorded so scatter falls back to the per-obs path on
    // mixed shapes. Each builder needs its own cache (validity keyed on
    // the H pointer).
    ScatterIndexCache idx_cache_main;
    ScatterIndexCache idx_cache_cheap;
    { TULPA_PROFILE_PHASE(PHASE_PATTERN_BUILD);
      build_scatter_index_cache(parsed, arms, blocks, H_builder,    idx_cache_main);
      build_scatter_index_cache(parsed, arms, blocks, cheap_builder, idx_cache_cheap); }

    // DENSE_BASIS scatter scratch buffers — lifted outside the per-cell
    // lambda so each (k_arm, b) slot's index cache (built once on first
    // scatter call) survives across every outer-grid cell. Cache keys are
    // stable per slot (blk_off_row, M, p_k, n_re_k, bstart, rstart,
    // H_ptr) so the same scratch hits across cells. Separate buffer sets
    // per H_builder because the cache validity check pins the H pointer.
    std::vector<DenseBasisScratch> db_buffers_main(
        static_cast<size_t>(n_arms) * B);
    std::vector<DenseBasisScratch> db_buffers_cheap(
        static_cast<size_t>(n_arms) * B);

    auto solve_at_theta_impl = [&](int k_grid,
                                   const std::vector<double>& prev_mode,
                                   SparseCholeskySolver* shared_solver,
                                   int max_iter_use,
                                   bool use_cheap_scratch) -> LaplaceResult
    {
        { TULPA_PROFILE_PHASE(PHASE_PREP);
          if (prep_at_grid) prep_at_grid(k_grid); }
        for (const auto& b : blocks) {
            TULPA_PROFILE_PHASE(PHASE_PREP);
            if (b.prep && !b.prep(k_grid)) {
                LaplaceResult bad;
                bad.mode = (static_cast<int>(prev_mode.size()) == n_x)
                           ? prev_mode
                           : std::vector<double>(n_x, 0.0);
                bad.log_marginal = -std::numeric_limits<double>::infinity();
                bad.n_iter = 0;
                bad.converged = false;
                bad.log_det_Q = 0.0;
                return bad;
            }
        }

        // Per-block-per-arm d_eff cache: d_fac(k_grid) * arm_scale(k_arm, k_grid).
        std::vector<std::vector<double>> d_eff(B, std::vector<double>(n_arms, 0.0));
        for (int b = 0; b < B; b++) {
            double dfac = blocks[b].d_fac ? blocks[b].d_fac(k_grid) : 1.0;
            for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                double s = blocks[b].arm_scale
                            ? blocks[b].arm_scale(k_arm, k_grid)
                            : 1.0;
                d_eff[b][k_arm] = s * dfac;
            }
        }

        // Per-call scratch for the inner scatter / eta dispatch.
        std::vector<double>              basis_scratch;
        std::vector<std::pair<int,double>> active_scratch;
        std::vector<std::pair<int,double>> multi_scratch;
        std::vector<DenseBasisActive>     active_db_scratch;
        // db_buffers is a reference into the lifted-out per-fit storage so
        // each (k_arm, b) cache survives across outer-grid cells.
        std::vector<DenseBasisScratch>&   db_buffers =
            use_cheap_scratch ? db_buffers_cheap : db_buffers_main;

        auto compute_eta_joint = [&](const Rcpp::NumericVector& x,
                                     std::vector<Rcpp::NumericVector>& etas) {
            compute_eta_joint_sparse_dispatch(
                x, etas, arms, parsed, blocks, k_grid,
                d_eff, basis_scratch, multi_scratch);
        };

        SparseHessianBuilder& H_use = use_cheap_scratch ? cheap_builder : H_builder;
        const ScatterIndexCache* idx_cache_use =
            use_cheap_scratch ? &idx_cache_cheap : &idx_cache_main;

        auto scatter_joint_sparse = [&](const Rcpp::NumericVector& x,
                                         const std::vector<Rcpp::NumericVector>& etas,
                                         DenseVec& grad,
                                         SparseHessianBuilder& H) {
            for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                scatter_arm_obs_joint_multi_sparse(
                    x, etas[k_arm], parsed[k_arm], arms[k_arm], k_arm,
                    blocks, k_grid, grad, H,
                    active_scratch, basis_scratch, multi_scratch,
                    active_db_scratch, db_buffers,
                    idx_cache_use
                );
            }
            for (const auto& b : blocks) {
                if (b.add_prior_sparse) b.add_prior_sparse(H, grad, x, k_grid);
            }
            add_per_arm_beta_re_priors_sparse(grad, H, x, parsed);
        };

        auto center_joint = [&](Rcpp::NumericVector& x) {
            for (int b = 0; b < B; b++) {
                if (!blocks[b].center) continue;
                double c_b = blocks[b].center(x);
                if (std::abs(c_b) < 1e-15) continue;
                double dfac = blocks[b].d_fac ? blocks[b].d_fac(k_grid) : 1.0;
                for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                    if (parsed[k_arm].p == 0) continue;
                    double s = blocks[b].arm_scale
                                ? blocks[b].arm_scale(k_arm, k_grid)
                                : 1.0;
                    x[parsed[k_arm].beta_start] += s * dfac * c_b;
                }
            }
        };

        auto log_prior_joint = [&](const Rcpp::NumericVector& x,
                                    const std::vector<Rcpp::NumericVector>&)
            -> double {
            double lp = log_prior_per_arm_re(x, parsed);
            for (const auto& b : blocks) {
                if (b.log_prior) lp += b.log_prior(x, k_grid);
            }
            return lp;
        };

        NewtonScratchJointSparse& sc = use_cheap_scratch ? cheap_scratch : scratch;
        return laplace_newton_solve_joint_sparse(
            arms, n_x,
            max_iter_use, tol, n_threads,
            compute_eta_joint, scatter_joint_sparse,
            center_joint, log_prior_joint,
            H_use, sc, prev_mode, shared_solver, store_Q
        );
    };

    auto solve_at_theta = [&](int k_grid,
                              const std::vector<double>& prev_mode,
                              SparseCholeskySolver* shared_solver) -> LaplaceResult
    {
        return solve_at_theta_impl(k_grid, prev_mode, shared_solver,
                                    max_iter, /*use_cheap_scratch=*/false);
    };

    auto cheap_eval = [&](int k_grid,
                          const std::vector<double>& x_pilot) -> double {
        LaplaceResult r = solve_at_theta_impl(
            k_grid, x_pilot, &cheap_solver,
            /*max_iter_use=*/1, /*use_cheap_scratch=*/true);
        return r.log_marginal;
    };

    return run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, store_modes,
        /*n_outer=*/1,
        tile_ids, tile_pilot_cells,
        cheap_eval, prune_tol
    );
}

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_JOINT_MULTI_H
