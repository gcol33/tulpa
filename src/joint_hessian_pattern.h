// joint_hessian_pattern.h
// Build the sparsity pattern of the joint multi-arm Laplace Hessian and
// initialize a SparseHessianBuilder with it.
//
// Pattern is fit-level (does not depend on the outer-grid hyperparameter
// index k_grid), so it is built once at the start of a fit and reused
// across all 20M outer-grid cells in the joint nested-Laplace driver.
//
// Composition:
//   (1) per-arm β/β dense + β/RE dense + RE/RE diagonal
//   (2) per-block prior-induced fill via LatentBlock::add_prior_pattern
//       (adjacency / NN_K / SPDE_Q / USER_CSC / none); diagonal of every
//       latent block always added
//   (3) per-arm × per-block data-induced fill, dispatched on
//       LatentBlock::contrib_kind:
//         INDEXED_SINGLE: one block-local DOF per obs (via `idx`)
//         INDEXED_MULTI:  small DOF list per obs (via `obs_indices`)
//         DENSE_BASIS:    skip per-obs enumeration, add full β/block,
//                         RE/block, block/block sub-patterns once
//   (4) per-arm × pairwise-block cross coupling per obs (e.g. BYM2 φ × θ
//       at the same site). Dense-basis on either side → full cross block.
//
// Memory budget. The intermediate `entries` vector can grow large at
// n_sites = 10^6 (a few × 10^7 entries; ~4-5 GB peak before SparseHessianBuilder::init
// dedupes). Stays comfortably within a 64 GB workstation budget. Streaming
// the build into a std::set directly is a stage-1.5 optimization if
// n_sites ≥ 10^7 starts hitting memory walls.

#ifndef TULPA_JOINT_HESSIAN_PATTERN_H
#define TULPA_JOINT_HESSIAN_PATTERN_H

#include "latent_block.h"
#include "nested_laplace_joint_core.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <algorithm>
#include <utility>
#include <vector>

namespace tulpa {

namespace detail {

// Append every (r, c) entry in the rectangular block
// [r0, r0 + r_size) × [c0, c0 + c_size) to `out`, normalized to lower
// triangle (row >= col). SparseHessianBuilder::init dedupes.
inline void add_dense_block_pattern(
    std::vector<std::pair<int,int>>& out,
    int r0, int r_size, int c0, int c_size
) {
    for (int r = r0; r < r0 + r_size; r++) {
        for (int c = c0; c < c0 + c_size; c++) {
            int hi = (r > c) ? r : c;
            int lo = (r > c) ? c : r;
            out.emplace_back(hi, lo);
        }
    }
}

// Resolve the active block-local DOFs (as global indices) for obs i in
// arm k_arm for block `blk`. Fills `scratch` with (global_idx, weight)
// pairs. Weight is unused by the pattern builder but kept for API symmetry
// with the scatter side. Returns nothing for DENSE_BASIS — that case is
// handled at the block level (full sub-pattern, no per-obs work).
// For BILINEAR_FACTOR, emits both the factor slot u and the loading slot
// lambda; the active × active intra-block pass fills the (u, lambda)
// cross pattern entry naturally.
inline void resolve_indexed_dofs(
    const LatentBlock&                       blk,
    int                                       i,
    int                                       k_arm,
    std::vector<std::pair<int,double>>&       scratch
) {
    scratch.clear();
    if (blk.contrib_kind == BlockContribKind::INDEXED_SINGLE) {
        if (!blk.idx) return;
        int l = blk.idx(i, k_arm);
        if (l > 0 && l <= blk.size) {
            scratch.emplace_back(blk.start + l - 1, 1.0);
        }
    } else if (blk.contrib_kind == BlockContribKind::INDEXED_MULTI) {
        if (!blk.obs_indices) return;
        blk.obs_indices(i, k_arm, scratch);
        // Translate block-local 1-based indices to global.
        for (auto& [idx_local, w] : scratch) {
            idx_local = blk.start + idx_local - 1;
        }
    } else if (blk.contrib_kind == BlockContribKind::BILINEAR_FACTOR) {
        if (!blk.obs_factor_lambda) return;
        auto [u_slot, lambda_slot] = blk.obs_factor_lambda(i, k_arm);
        if (u_slot < 0 || lambda_slot < 0) return;
        scratch.emplace_back(u_slot, 1.0);
        scratch.emplace_back(lambda_slot, 1.0);
    }
    // DENSE_BASIS: caller adds full sub-pattern; scratch left empty.
}

} // namespace detail

// Build the joint Hessian sparsity pattern and initialize `out_builder`.
// See file header for algorithm details.
//
// `coupled_arms` lists the 0-based arm indices a CellCouplingSpec couples
// at this fit (empty if no spec is registered, or the spec is the
// separable default). Every (k, l) pair with k != l from `coupled_arms`
// adds cross-arm beta/beta, beta/RE, RE/beta, RE/RE dense pattern blocks
// (section 5 below). Latent-block cross-coverage is already provided by
// section 3's per-arm walk -- each arm's own per-obs path adds its beta/RE
// x latent entries, so a shared latent dof reached from two coupled arms
// already has both (beta_k, z) and (beta_l, z) in the pattern.
inline void build_joint_hessian_pattern(
    const std::vector<ParsedArm>&    parsed,
    const std::vector<JointArm>&     arms,
    const std::vector<LatentBlock>&  blocks,
    int                              n_x,
    SparseHessianBuilder&            out_builder,
    const std::vector<int>&          coupled_arms = std::vector<int>()
) {
    const int n_arms = static_cast<int>(arms.size());
    const int B      = static_cast<int>(blocks.size());

    std::vector<std::pair<int,int>> entries;

    // Capacity hint (rough — exact for β/β + diag, approx for data fill).
    {
        size_t cap = 0;
        for (int k = 0; k < n_arms; k++) {
            cap += static_cast<size_t>(parsed[k].p) * parsed[k].p;
            cap += static_cast<size_t>(parsed[k].p) * parsed[k].n_re_groups;
            cap += parsed[k].n_re_groups;
            cap += static_cast<size_t>(arms[k].N) * B * 4;
        }
        for (const auto& b : blocks) {
            if (b.contrib_kind == BlockContribKind::DENSE_BASIS) {
                cap += static_cast<size_t>(b.size) * b.size;
            } else {
                cap += b.size;  // diagonal
                if (b.prior_kind == PriorFillKind::ADJACENCY)
                    cap += static_cast<size_t>(b.size) * 8;
                else if (b.prior_kind == PriorFillKind::NN_K)
                    cap += static_cast<size_t>(b.size) * 200;
            }
        }
        entries.reserve(cap);
    }

    // ---- (1) Per-arm β/β dense + β/RE dense + RE/RE diagonal ----
    for (int k = 0; k < n_arms; k++) {
        const ParsedArm& pa = parsed[k];
        detail::add_dense_block_pattern(entries,
                                         pa.beta_start, pa.p,
                                         pa.beta_start, pa.p);
        if (pa.n_re_groups > 0) {
            detail::add_dense_block_pattern(entries,
                                             pa.re_start, pa.n_re_groups,
                                             pa.beta_start, pa.p);
        }
        for (int g = 0; g < pa.n_re_groups; g++) {
            entries.emplace_back(pa.re_start + g, pa.re_start + g);
        }
    }

    // ---- (2) Per-block prior-induced pattern + always-present diagonal ----
    for (int b = 0; b < B; b++) {
        const LatentBlock& blk = blocks[b];
        for (int j = 0; j < blk.size; j++) {
            entries.emplace_back(blk.start + j, blk.start + j);
        }
        if (blk.add_prior_pattern) {
            blk.add_prior_pattern(entries);
        }
    }

    // ---- (3) Per-arm × per-block data-induced pattern ----
    std::vector<std::pair<int,double>> dof_scratch;
    std::vector<std::pair<int,double>> dof_scratch2;

    for (int k = 0; k < n_arms; k++) {
        const ParsedArm& pa = parsed[k];
        const int N_k = arms[k].N;

        for (int b = 0; b < B; b++) {
            const LatentBlock& blk = blocks[b];

            if (blk.contrib_kind == BlockContribKind::DENSE_BASIS) {
                // Every obs hits every basis coefficient → full sub-blocks
                // unconditionally. No per-obs enumeration.
                detail::add_dense_block_pattern(entries,
                                                 blk.start, blk.size,
                                                 pa.beta_start, pa.p);
                if (pa.n_re_groups > 0) {
                    detail::add_dense_block_pattern(entries,
                                                     blk.start, blk.size,
                                                     pa.re_start, pa.n_re_groups);
                }
                detail::add_dense_block_pattern(entries,
                                                 blk.start, blk.size,
                                                 blk.start, blk.size);
                continue;
            }

            // INDEXED_SINGLE / INDEXED_MULTI: per-obs enumeration.
            for (int i = 0; i < N_k; i++) {
                detail::resolve_indexed_dofs(blk, i, k, dof_scratch);
                if (dof_scratch.empty()) continue;

                // β × {active dofs}
                for (const auto& dw : dof_scratch) {
                    int d = dw.first;
                    for (int j = 0; j < pa.p; j++) {
                        int bj = pa.beta_start + j;
                        int hi = (d > bj) ? d : bj;
                        int lo = (d > bj) ? bj : d;
                        entries.emplace_back(hi, lo);
                    }
                }
                // RE × {active dofs}
                if (pa.n_re_groups > 0) {
                    int gi = static_cast<int>(pa.re_idx[i]) - 1;
                    if (gi >= 0 && gi < pa.n_re_groups) {
                        int g_re = pa.re_start + gi;
                        for (const auto& dw : dof_scratch) {
                            int d = dw.first;
                            int hi = (d > g_re) ? d : g_re;
                            int lo = (d > g_re) ? g_re : d;
                            entries.emplace_back(hi, lo);
                        }
                    }
                }
                // {dofs} × {dofs} intra-block (diagonal already present).
                const int A = static_cast<int>(dof_scratch.size());
                for (int a1 = 0; a1 < A; a1++) {
                    int d1 = dof_scratch[a1].first;
                    for (int a2 = 0; a2 < A; a2++) {
                        int d2 = dof_scratch[a2].first;
                        if (d1 >= d2) entries.emplace_back(d1, d2);
                    }
                }
            }
        }

        // ---- (4) Per-arm × pairwise-block cross-coupling per obs ----
        for (int b1 = 0; b1 < B; b1++) {
            for (int b2 = b1 + 1; b2 < B; b2++) {
                const LatentBlock& blk1 = blocks[b1];
                const LatentBlock& blk2 = blocks[b2];

                bool b1_dense = (blk1.contrib_kind == BlockContribKind::DENSE_BASIS);
                bool b2_dense = (blk2.contrib_kind == BlockContribKind::DENSE_BASIS);

                if (b1_dense || b2_dense) {
                    // Dense-basis on either side → over-pattern full cross.
                    detail::add_dense_block_pattern(entries,
                                                     blk2.start, blk2.size,
                                                     blk1.start, blk1.size);
                    continue;
                }

                // Both indexed: per-obs enumeration.
                for (int i = 0; i < N_k; i++) {
                    detail::resolve_indexed_dofs(blk1, i, k, dof_scratch);
                    if (dof_scratch.empty()) continue;
                    detail::resolve_indexed_dofs(blk2, i, k, dof_scratch2);
                    if (dof_scratch2.empty()) continue;

                    for (const auto& dw1 : dof_scratch) {
                        int d1 = dw1.first;
                        for (const auto& dw2 : dof_scratch2) {
                            int d2 = dw2.first;
                            int hi = (d1 > d2) ? d1 : d2;
                            int lo = (d1 > d2) ? d2 : d1;
                            entries.emplace_back(hi, lo);
                        }
                    }
                }
            }
        }
    }

    // ---- (5) Cross-arm coupled blocks (CellCouplingSpec) ----
    // For each pair (kk, ll) of coupled arms with kk < ll, the per-cell
    // branch's cross_hess scatter writes Hkl * (chain_kk outer chain_ll
    // + transpose) into the joint H. The beta_k/beta_l, beta_k/RE_l,
    // RE_k/beta_l, RE_k/RE_l sub-blocks of that outer product are NOT
    // covered by sections 1-3 (those are per-arm only). Add them as
    // dense blocks here. Latent-block cross-coverage IS covered by
    // section 3 (each arm's per-obs walk hits its own beta/RE x latent
    // entries), so a shared latent dof reached from both kk and ll
    // already has both (beta_kk, z) and (beta_ll, z) in the pattern.
    for (size_t ii = 0; ii < coupled_arms.size(); ii++) {
        int k = coupled_arms[ii];
        if (k < 0 || k >= n_arms) continue;
        const ParsedArm& pa_k = parsed[k];
        for (size_t jj = ii + 1; jj < coupled_arms.size(); jj++) {
            int l = coupled_arms[jj];
            if (l < 0 || l >= n_arms) continue;
            const ParsedArm& pa_l = parsed[l];
            // beta_k x beta_l
            if (pa_k.p > 0 && pa_l.p > 0) {
                detail::add_dense_block_pattern(entries,
                                                 pa_k.beta_start, pa_k.p,
                                                 pa_l.beta_start, pa_l.p);
            }
            // beta_k x RE_l, beta_l x RE_k
            if (pa_k.p > 0 && pa_l.n_re_groups > 0) {
                detail::add_dense_block_pattern(entries,
                                                 pa_k.beta_start, pa_k.p,
                                                 pa_l.re_start, pa_l.n_re_groups);
            }
            if (pa_l.p > 0 && pa_k.n_re_groups > 0) {
                detail::add_dense_block_pattern(entries,
                                                 pa_l.beta_start, pa_l.p,
                                                 pa_k.re_start, pa_k.n_re_groups);
            }
            // RE_k x RE_l
            if (pa_k.n_re_groups > 0 && pa_l.n_re_groups > 0) {
                detail::add_dense_block_pattern(entries,
                                                 pa_k.re_start, pa_k.n_re_groups,
                                                 pa_l.re_start, pa_l.n_re_groups);
            }
        }
    }

    out_builder.init(n_x, entries);
}

} // namespace tulpa

#endif // TULPA_JOINT_HESSIAN_PATTERN_H
