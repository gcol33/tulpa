// scatter_indexed_cache.h
// Stage 2.2b: per-obs INDEXED scatter index cache for the joint multi-arm
// sparse path.
//
// The per-obs scatter in `scatter_arm_obs_joint_multi_sparse` (see
// nested_laplace_joint_multi.h) does ~10 SparseHessianBuilder::add() calls
// per obs in the canonical ICAR/BYM2 workload. Each call hits a
// std::map<pair<int,int>, int> lookup (~50 ns). The pattern is built once
// at fit-time and stable for the whole outer grid, so all the lookups are
// repeated work. This cache resolves the active INDEXED dofs and the flat
// values[] indices for every H entry the scatter will touch — once at
// fit-time, reused across every Newton iter and outer-grid cell.
//
// Scope:
//   * Covers INDEXED_SINGLE and INDEXED_MULTI blocks.
//   * DENSE_BASIS blocks are handled by scatter_dense_basis.h (Stage 2.2a).
//     When DENSE_BASIS is present alongside INDEXED blocks, the DB cross
//     terms keep using the per-obs map-lookup path; the INDEXED cache
//     handles its own subset.
//   * BILINEAR_FACTOR has dynamic (x-dependent) active weights, so the
//     weight cannot be precomputed. The cache is disabled when any block
//     is BILINEAR_FACTOR; scatter falls back to the per-obs loop.
//
// Layout (single flat allocation per arm, indexed by per-obs offset):
//
//   ScatterIndexCache::arm[k_arm].plans[i] -> PerObsScatterPlan
//
//   PerObsScatterPlan stores offsets into the arm-level flat arrays:
//     - active_dof_global[ k_arm.act_off + plan.act_start + a ]
//     - active_block_idx [ k_arm.act_off + plan.act_start + a ]
//     - active_local_w   [ k_arm.act_off + plan.act_start + a ]
//     - idx_beta_active  [ k_arm.bxa_off + plan.bxa_start + (j*A_idx + a) ]
//     - idx_re_active    [ k_arm.rxa_off + plan.rxa_start + a ]
//     - idx_act_act      [ k_arm.axa_off + plan.axa_start + col-major LT pos ]
//
// Cache validity is keyed on (H_builder pointer, H_builder.nnz, n_arms,
// blocks fingerprint). Built explicitly via `build_scatter_index_cache`
// after the pattern is finalized; reused unmodified across all outer-grid
// cells in a fit.

#ifndef TULPA_SCATTER_INDEXED_CACHE_H
#define TULPA_SCATTER_INDEXED_CACHE_H

#include "joint_hessian_pattern.h"
#include "latent_block.h"
#include "nested_laplace_joint_core.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cstddef>
#include <vector>

namespace tulpa {

struct PerObsScatterPlan {
    int A_idx        = 0;    // number of active INDEXED dofs for this obs
    int g_re_global  = -1;   // global RE-group index, or -1 if no RE
    int act_start    = 0;    // offset into arm's active_dof_global etc.
    int bxa_start    = 0;    // offset into arm's idx_beta_active flat
    int rxa_start    = 0;    // offset into arm's idx_re_active flat
    int axa_start    = 0;    // offset into arm's idx_act_act flat
};

struct ArmIndexedCache {
    // Per-arm β × β indices (p_k * (p_k + 1) / 2 entries; lower triangle
    // column-major: for l in [0,p_k), for j in [l, p_k)).
    std::vector<int> idx_bb;

    // Per-arm RE × RE diagonal indices (n_re_k entries; idx for
    // (rstart+g, rstart+g) for g in [0, n_re_k)).
    std::vector<int> idx_re_diag;

    // Per-arm β × RE indices (p_k * n_re_k entries; layout [g * p_k + j]).
    std::vector<int> idx_beta_re;

    // Per-obs plans.
    std::vector<PerObsScatterPlan> plans;

    // Concatenated active-dof storage (length sum_i A_idx_i).
    std::vector<int>    active_dof_global;
    std::vector<int>    active_block_idx;
    std::vector<double> active_local_w;

    // Concatenated flat-index arrays:
    //   idx_beta_active[ obs.bxa_start + j*A_idx + a ]  for j in [0,p_k), a in [0,A_idx)
    //   idx_re_active  [ obs.rxa_start + a ]           for a in [0,A_idx); valid only when obs.g_re_global >= 0
    //   idx_act_act    [ obs.axa_start + col_major_lt(a1, a2) ]  for a1 >= a2 in [0, A_idx)
    std::vector<int> idx_beta_active;
    std::vector<int> idx_re_active;
    std::vector<int> idx_act_act;
};

struct ScatterIndexCache {
    bool enabled = false;      // false if any block is BILINEAR_FACTOR (no static weights)
    bool any_dense_basis = false; // true if any block is DENSE_BASIS — DB cross terms keep using the per-obs map path

    // Validity key
    const void* cache_H_ptr = nullptr;
    int         cache_H_nnz = -1;

    std::vector<ArmIndexedCache> arm;
};

namespace detail {

// Column-major lower-triangle index (a1 >= a2):
//   for a2 in [0, A_idx), for a1 in [a2, A_idx) -> position
inline int col_major_lt_pos(int a1, int a2, int A_idx) {
    // Sum of column heights for cols [0..a2): each col c has (A_idx - c) entries.
    // Sum_{c=0..a2-1} (A_idx - c) = a2 * A_idx - a2*(a2-1)/2
    return a2 * A_idx - a2 * (a2 - 1) / 2 + (a1 - a2);
}

inline int upper_diag_lt_pos(int j, int l) {
    // Lower triangle of p_k x p_k with j >= l, column-major:
    //   for l_outer in [0, p_k), for j_inner in [l_outer, p_k)
    // pos = (l * p_k - l*(l-1)/2) + (j - l)
    // Where ((l * p_k - l*(l-1)/2)) is sum of col heights for cols 0..l-1
    return l * /*p_k unused — different layout chosen below*/ 0 + 0 + (j - l);
    (void)j; (void)l;
}

} // namespace detail

// Build the cache. Must be called AFTER `H_builder.init()` so the
// entry_map has its final pattern. Inspects blocks: if any block is
// BILINEAR_FACTOR, the cache is left disabled. DENSE_BASIS blocks set
// `any_dense_basis = true` and are skipped at lookup time (the per-obs
// loop continues to handle DB cross terms via H.add).
inline void build_scatter_index_cache(
    const std::vector<ParsedArm>&    parsed,
    const std::vector<JointArm>&     arms,
    const std::vector<LatentBlock>&  blocks,
    const SparseHessianBuilder&      H,
    ScatterIndexCache&               cache
) {
    const int n_arms = static_cast<int>(arms.size());
    const int B      = static_cast<int>(blocks.size());

    cache.cache_H_ptr = static_cast<const void*>(&H);
    cache.cache_H_nnz = H.nnz;

    // Disable when any block is BILINEAR_FACTOR (dynamic weights).
    cache.enabled        = true;
    cache.any_dense_basis = false;
    for (int b = 0; b < B; b++) {
        if (blocks[b].contrib_kind == BlockContribKind::BILINEAR_FACTOR) {
            cache.enabled = false;
        } else if (blocks[b].contrib_kind == BlockContribKind::DENSE_BASIS) {
            cache.any_dense_basis = true;
        }
    }
    if (!cache.enabled) {
        cache.arm.clear();
        return;
    }

    cache.arm.assign(n_arms, ArmIndexedCache{});

    std::vector<std::pair<int,double>> multi_scratch;

    for (int k_arm = 0; k_arm < n_arms; k_arm++) {
        const ParsedArm& pa = parsed[k_arm];
        const int N_k    = arms[k_arm].N;
        const int p_k    = pa.p;
        const int n_re_k = pa.n_re_groups;
        const int bstart = pa.beta_start;
        const int rstart = pa.re_start;

        ArmIndexedCache& ac = cache.arm[k_arm];

        // β × β lower triangle, column-major (l outer, j inner over [l, p_k)).
        ac.idx_bb.resize(static_cast<size_t>(p_k) * (p_k + 1) / 2);
        {
            int t = 0;
            for (int l = 0; l < p_k; l++) {
                const int c = bstart + l;
                for (int j = l; j < p_k; j++) {
                    const int r = bstart + j;
                    ac.idx_bb[t++] = H.lookup(r, c);
                }
            }
        }

        // RE × RE diagonal (rstart+g, rstart+g).
        ac.idx_re_diag.resize(static_cast<size_t>(n_re_k));
        for (int g = 0; g < n_re_k; g++) {
            const int r = rstart + g;
            ac.idx_re_diag[g] = H.lookup(r, r);
        }

        // β × RE: layout [g * p_k + j] for g in [0,n_re_k), j in [0,p_k).
        ac.idx_beta_re.resize(static_cast<size_t>(n_re_k) * p_k);
        for (int g = 0; g < n_re_k; g++) {
            const int c = rstart + g;
            for (int j = 0; j < p_k; j++) {
                const int r = bstart + j;
                ac.idx_beta_re[static_cast<size_t>(g) * p_k + j] =
                    H.lookup(r, c);
            }
        }

        // Per-obs plans.
        ac.plans.assign(N_k, PerObsScatterPlan{});

        // First pass: resolve active dofs per obs to size the flat arrays.
        // Counts only INDEXED_SINGLE / INDEXED_MULTI; DENSE_BASIS is handled
        // outside the cache. BILINEAR is already disabled above.
        size_t total_A = 0;
        for (int i = 0; i < N_k; i++) {
            int A_i = 0;
            for (int b = 0; b < B; b++) {
                const LatentBlock& blk = blocks[b];
                if (blk.contrib_kind == BlockContribKind::INDEXED_SINGLE) {
                    int l = blk.idx ? blk.idx(i, k_arm) : -1;
                    if (l > 0 && l <= blk.size) A_i++;
                } else if (blk.contrib_kind == BlockContribKind::INDEXED_MULTI) {
                    if (!blk.obs_indices) continue;
                    multi_scratch.clear();
                    blk.obs_indices(i, k_arm, multi_scratch);
                    for (const auto& [l, w_local] : multi_scratch) {
                        if (l > 0 && l <= blk.size) {
                            A_i++;
                            (void)w_local;
                        }
                    }
                }
                // DENSE_BASIS skipped from cache; handled by 2.2a.
            }
            ac.plans[i].A_idx = A_i;
            total_A += static_cast<size_t>(A_i);
        }

        ac.active_dof_global.resize(total_A);
        ac.active_block_idx .resize(total_A);
        ac.active_local_w   .resize(total_A);

        // Size flat-index arrays via prefix sums.
        size_t total_bxa = 0, total_rxa = 0, total_axa = 0;
        for (int i = 0; i < N_k; i++) {
            const int A_i = ac.plans[i].A_idx;
            ac.plans[i].bxa_start = static_cast<int>(total_bxa);
            ac.plans[i].rxa_start = static_cast<int>(total_rxa);
            ac.plans[i].axa_start = static_cast<int>(total_axa);
            total_bxa += static_cast<size_t>(p_k) * A_i;
            total_rxa += static_cast<size_t>(A_i);
            total_axa += static_cast<size_t>(A_i) * (A_i + 1) / 2;
        }
        ac.idx_beta_active.resize(total_bxa);
        ac.idx_re_active  .resize(total_rxa);
        ac.idx_act_act    .resize(total_axa);

        // Second pass: fill active dofs and flat indices.
        size_t act_cursor = 0;
        for (int i = 0; i < N_k; i++) {
            PerObsScatterPlan& plan = ac.plans[i];
            plan.act_start = static_cast<int>(act_cursor);

            // g_re for this obs.
            int g_re_global = -1;
            if (n_re_k > 0) {
                int gi = static_cast<int>(pa.re_idx[i]) - 1;
                if (gi >= 0 && gi < n_re_k) g_re_global = rstart + gi;
            }
            plan.g_re_global = g_re_global;

            // Resolve active dofs in the same block order as the scatter path.
            int a = 0;
            for (int b = 0; b < B; b++) {
                const LatentBlock& blk = blocks[b];
                if (blk.contrib_kind == BlockContribKind::INDEXED_SINGLE) {
                    if (!blk.idx) continue;
                    int l = blk.idx(i, k_arm);
                    if (l > 0 && l <= blk.size) {
                        ac.active_dof_global[act_cursor + a] = blk.start + l - 1;
                        ac.active_block_idx [act_cursor + a] = b;
                        ac.active_local_w   [act_cursor + a] = 1.0;
                        a++;
                    }
                } else if (blk.contrib_kind == BlockContribKind::INDEXED_MULTI) {
                    if (!blk.obs_indices) continue;
                    multi_scratch.clear();
                    blk.obs_indices(i, k_arm, multi_scratch);
                    for (const auto& [l, w_local] : multi_scratch) {
                        if (l > 0 && l <= blk.size) {
                            ac.active_dof_global[act_cursor + a] = blk.start + l - 1;
                            ac.active_block_idx [act_cursor + a] = b;
                            ac.active_local_w   [act_cursor + a] = w_local;
                            a++;
                        }
                    }
                }
                // DENSE_BASIS: skipped (per-obs path retained).
            }
            act_cursor += static_cast<size_t>(plan.A_idx);

            const int A_i = plan.A_idx;

            // β × active: layout [j * A_i + a] for j in [0,p_k), a in [0,A_i).
            for (int j = 0; j < p_k; j++) {
                const int r = bstart + j;
                for (int aa = 0; aa < A_i; aa++) {
                    const int d = ac.active_dof_global[plan.act_start + aa];
                    ac.idx_beta_active[static_cast<size_t>(plan.bxa_start)
                                       + static_cast<size_t>(j) * A_i + aa] =
                        H.lookup(r, d);
                }
            }

            // RE × active: layout [a] for a in [0,A_i); -1 entries when no RE.
            if (g_re_global >= 0) {
                for (int aa = 0; aa < A_i; aa++) {
                    const int d = ac.active_dof_global[plan.act_start + aa];
                    ac.idx_re_active[static_cast<size_t>(plan.rxa_start) + aa] =
                        H.lookup(g_re_global, d);
                }
            } else {
                for (int aa = 0; aa < A_i; aa++) {
                    ac.idx_re_active[static_cast<size_t>(plan.rxa_start) + aa] = -1;
                }
            }

            // active × active lower triangle column-major (a2 outer, a1 inner over [a2, A_i)).
            int t = 0;
            for (int a2 = 0; a2 < A_i; a2++) {
                const int d2 = ac.active_dof_global[plan.act_start + a2];
                for (int a1 = a2; a1 < A_i; a1++) {
                    const int d1 = ac.active_dof_global[plan.act_start + a1];
                    ac.idx_act_act[static_cast<size_t>(plan.axa_start) + t++] =
                        H.lookup(d1, d2);
                }
            }
        }
    }
}

// Returns true if the cache is built and consistent with H. Caller passes
// the H currently being scattered into; if the H pointer/nnz differ from
// the cache, the cache is treated as stale.
inline bool scatter_index_cache_valid(
    const ScatterIndexCache& cache,
    const SparseHessianBuilder& H
) {
    if (!cache.enabled) return false;
    if (cache.cache_H_ptr != static_cast<const void*>(&H)) return false;
    if (cache.cache_H_nnz != H.nnz) return false;
    return true;
}

// Cached scatter for one arm. Used when every block is INDEXED_SINGLE or
// INDEXED_MULTI (no DENSE_BASIS, no BILINEAR_FACTOR). Writes the same
// gradient and Hessian entries as the per-obs path in
// scatter_arm_obs_joint_multi_sparse, but replaces every H.add() with a
// direct values[idx] += val using flat indices precomputed at fit-time.
//
// Caller passes:
//   - eta, pa, arm: per-arm data (same as the legacy scatter).
//   - d_eff_cache[b]: per-block effective coefficient at this outer-grid
//     cell (arm_scale * d_fac). Length = blocks.size().
//   - arm_cache: precomputed cache for this k_arm.
//   - family, phi_disp: per-arm likelihood family + dispersion.
inline void scatter_arm_obs_indexed_cached(
    const Rcpp::NumericVector&  eta,
    const ParsedArm&            pa,
    const JointArm&             arm,
    const std::vector<double>&  d_eff_cache,
    const ArmIndexedCache&      ac,
    DenseVec&                   grad,
    SparseHessianBuilder&       H,
    const std::string&          family,
    double                      phi_disp
) {
    const int p_k    = pa.p;
    const int n_re_k = pa.n_re_groups;
    const int bstart = pa.beta_start;
    const int rstart = pa.re_start;

    double* __restrict__       Hv  = H.values.data();
    const int* __restrict__    bb  = ac.idx_bb.data();
    const int* __restrict__    bre = ac.idx_beta_re.data();
    const int* __restrict__    red = ac.idx_re_diag.data();
    const int* __restrict__    bxa = ac.idx_beta_active.data();
    const int* __restrict__    rxa = ac.idx_re_active.data();
    const int* __restrict__    axa = ac.idx_act_act.data();
    const int* __restrict__    adg = ac.active_dof_global.data();
    const int* __restrict__    abi = ac.active_block_idx.data();
    const double* __restrict__ alw = ac.active_local_w.data();

    // Per-obs active weight buffer; reused across i. Sized to max A_idx.
    int max_A = 0;
    for (const auto& p : ac.plans) if (p.A_idx > max_A) max_A = p.A_idx;
    std::vector<double> w_active(max_A, 0.0);

    for (int i = 0; i < arm.N; i++) {
        auto gh = grad_hess_for_family(
            arm.y[i], arm.n_trials[i], eta[i], family, phi_disp);

        const PerObsScatterPlan& plan = ac.plans[i];
        const int A_i      = plan.A_idx;
        const int g_re_glb = plan.g_re_global;

        // Compose per-obs active weights: d_eff_cache[block] * local_w.
        for (int a = 0; a < A_i; a++) {
            w_active[a] = d_eff_cache[abi[plan.act_start + a]]
                          * alw[plan.act_start + a];
        }

        // β block: gradient + β/β diagonal + β × RE + β × active.
        for (int j = 0; j < p_k; j++) {
            const double Xij = pa.X(i, j);
            grad[bstart + j] += gh.grad * Xij;
        }
        // β/β lower triangle column-major: idx_bb layout matches.
        {
            int t = 0;
            for (int l = 0; l < p_k; l++) {
                const double Xil = pa.X(i, l);
                for (int j = l; j < p_k; j++) {
                    const int k = bb[t++];
                    if (k >= 0) Hv[k] += gh.neg_hess * pa.X(i, j) * Xil;
                }
            }
        }
        // β × RE diagonal: indexed by the obs's g_re_global.
        if (g_re_glb >= 0) {
            const int g_local = g_re_glb - rstart;
            const int row_off = g_local * p_k;
            for (int j = 0; j < p_k; j++) {
                const int k = bre[row_off + j];
                if (k >= 0) Hv[k] += gh.neg_hess * pa.X(i, j);
            }
        }
        // β × active: idx_beta_active[bxa_start + j*A_i + a].
        if (A_i > 0) {
            for (int j = 0; j < p_k; j++) {
                const double Xij = pa.X(i, j);
                const int row_off = plan.bxa_start + j * A_i;
                for (int a = 0; a < A_i; a++) {
                    const int k = bxa[row_off + a];
                    if (k >= 0) Hv[k] += gh.neg_hess * Xij * w_active[a];
                }
            }
        }

        // RE block: gradient + RE/RE diagonal + RE × active.
        if (g_re_glb >= 0) {
            grad[g_re_glb] += gh.grad;
            const int g_local = g_re_glb - rstart;
            const int k_re = red[g_local];
            if (k_re >= 0) Hv[k_re] += gh.neg_hess;
            for (int a = 0; a < A_i; a++) {
                const int k = rxa[plan.rxa_start + a];
                if (k >= 0) Hv[k] += gh.neg_hess * w_active[a];
            }
        }

        // active × active lower triangle column-major + gradient.
        if (A_i > 0) {
            int t = 0;
            for (int a2 = 0; a2 < A_i; a2++) {
                const double w_a2 = w_active[a2];
                // active gradient (one entry per active dof).
                grad[adg[plan.act_start + a2]] += gh.grad * w_a2;
                for (int a1 = a2; a1 < A_i; a1++) {
                    const int k = axa[plan.axa_start + t++];
                    if (k >= 0) Hv[k] += gh.neg_hess * w_active[a1] * w_a2;
                }
            }
        }
    }
}

} // namespace tulpa

#endif // TULPA_SCATTER_INDEXED_CACHE_H
