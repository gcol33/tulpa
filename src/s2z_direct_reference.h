// s2z_direct_reference.h
// Dense-block reference for the sum-to-zero log-determinant, used by the test
// fixtures only: it factors B = A + sum_k coef_k 1_k 1_k' with every pinned
// block's lower triangle densified, the independent check s2z_block_schur is
// held against.

#ifndef TULPA_S2Z_DIRECT_REFERENCE_H
#define TULPA_S2Z_DIRECT_REFERENCE_H

#include "sparse_hessian.h"

namespace tulpa {

// The pattern-dependent parts of s2z_log_det_direct. The matrix
// B = A + sum_k coef_k 1_k 1_k' has A's structural pattern plus each s2z block's
// full lower triangle:
//   * `B_builder` — B's CSC pattern + entry_map;
//   * `a_slots` — for each A nonzero p, the flat values[] slot in B_builder, so
//     A's values scatter via `B.values[slot] += val` instead of a map lookup;
//   * `block_slots` — for each dense lower-triangle entry of every coef_k 1_k
//     1_k' block, the flat values[] slot, in block-then-(i,j) order;
//   * `B_solver` — the CHOLMOD solver holding B's symbolic and numeric factor.
// Each block contributes n_k (n_k + 1) / 2 entries to the pattern, the
// entry_map and the factor, about 1.5 GB on a field of a few thousand nodes, so
// this stays a reference for test-sized fields.
struct S2ZDirectFactor {
    SparseHessianBuilder B_builder;     // pattern + entry_map
    std::vector<int>     a_slots;       // flat slot per A nonzero
    std::vector<int>     block_slots;   // flat slot per dense block LT entry
    std::vector<int>     cross_slots;   // flat slot per (a>b) cross-block entry
    SparseCholeskySolver B_solver;      // B's symbolic + numeric factor
};

// Cross-block fill is quadratic in the total pinned length, so a dense coupling
// on a large field is refused here rather than allocated. s2z_block_schur folds
// the same D with no fill.
constexpr long long S2Z_COUPLED_DIRECT_MAX_ENTRIES = 4000000LL;

// Build the pattern-dependent parts for B = A + sum_k coef_k 1_k 1_k': B's CSC
// pattern + entry_map, the flat values[] slots for A's nonzeros and for each
// dense block lower-triangle entry, and the solver's symbolic factor.
inline void build_s2z_direct_factor(
    const SparseHessianBuilder& A_builder,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    bool coupled,
    S2ZDirectFactor& cache
) {
    const int n_x = A_builder.n;
    const int K   = static_cast<int>(r1.size());

    // B's pattern = A's nonzeros plus each block's full lower triangle (the only
    // entries the dense 1_k 1_k' touches), so the factor sees the same matrix the
    // dense densify path stores.
    std::vector<std::pair<int,int>> pattern;
    pattern.reserve(A_builder.nnz);
    for (int j = 0; j < n_x; ++j)
        for (int p = A_builder.col_ptr[j]; p < A_builder.col_ptr[j + 1]; ++p)
            pattern.emplace_back(A_builder.row_idx[p], j);
    for (int k = 0; k < K; ++k) {
        const int nk = r1[k].n;
        for (int i = 0; i < nk; ++i)
            for (int j = 0; j <= i; ++j)
                pattern.emplace_back(r1[k].node(i), r1[k].node(j));
    }
    // D[a,b] fills the whole (a,b) rectangle: (U D U')_{pq} = D[a,b] for p in
    // block a, q in block b. init() folds each pair into the lower triangle.
    if (coupled)
        for (int a = 0; a < K; ++a)
            for (int b = 0; b < a; ++b)
                for (int i = 0; i < r1[a].n; ++i)
                    for (int j = 0; j < r1[b].n; ++j)
                        pattern.emplace_back(r1[a].node(i), r1[b].node(j));

    cache.B_builder.init(n_x, pattern);

    // Resolve the flat values[] slot for every entry the per-call scatter writes,
    // in the SAME traversal order, so each call writes B.values[slot] += val.
    cache.a_slots.clear();
    cache.a_slots.reserve(A_builder.nnz);
    for (int j = 0; j < n_x; ++j)
        for (int p = A_builder.col_ptr[j]; p < A_builder.col_ptr[j + 1]; ++p)
            cache.a_slots.push_back(cache.B_builder.lookup(A_builder.row_idx[p], j));

    cache.block_slots.clear();
    for (int k = 0; k < K; ++k) {
        const int nk = r1[k].n;
        for (int i = 0; i < nk; ++i)
            for (int j = 0; j <= i; ++j)
                cache.block_slots.push_back(
                    cache.B_builder.lookup(r1[k].node(i), r1[k].node(j)));
    }
    cache.cross_slots.clear();
    if (coupled)
        for (int a = 0; a < K; ++a)
            for (int b = 0; b < a; ++b)
                for (int i = 0; i < r1[a].n; ++i)
                    for (int j = 0; j < r1[b].n; ++j)
                        cache.cross_slots.push_back(
                            cache.B_builder.lookup(r1[a].node(i), r1[b].node(j)));

    // The cholmod_sparse view aliases B_builder's arrays, so it stays valid as
    // long as `cache` (hence B_builder) lives.
    cholmod_sparse B_view = cache.B_builder.as_cholmod(&cache.B_solver.common());
    cache.B_solver.analyze(&B_view);
}

// Cancellation-free log-determinant for the sum-to-zero rank-1 penalties.
//
// Target: log|B|, B = A + sum_k coef_k 1_k 1_k', where A is a sparse
// Hessian (lower-triangle CSC in `A_builder`, already carrying
// LAPLACE_UNIFORM_RIDGE on its diagonal) and 1_k is the indicator of field block
// k over its node set (contiguous [start_k, start_k + n_k), or the component's
// arbitrary nodes for a disconnected map).
//
// log|B| is read directly from a Cholesky factor of B itself, with B's pattern
// = A's pattern plus each block k densified to its full lower triangle (where
// coef_k 1_k 1_k' has support). Shares no code with s2z_block_schur beyond the
// builder, which is what makes it the reference. Returns log|B| on success and
// `fallback` on any failure (allocation, non-PD, K == 0).
inline double s2z_log_det_direct(
    const SparseHessianBuilder& A_builder,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    double fallback
) {
    const int K = static_cast<int>(r1.size());
    if (K == 0) return fallback;
    const int n_x = A_builder.n;

    const std::vector<double>& coupling = A_builder.s2z_coupling;
    const bool coupled = !coupling.empty();
    if (coupled) {
        if ((int) coupling.size() != K * K) return fallback;
        long long entries = 0;
        for (int a = 0; a < K; ++a)
            for (int b = 0; b < a; ++b)
                entries += (long long) r1[a].n * r1[b].n;
        if (entries > S2Z_COUPLED_DIRECT_MAX_ENTRIES) return fallback;
    }

    S2ZDirectFactor cc;
    build_s2z_direct_factor(A_builder, r1, coupled, cc);

    SparseHessianBuilder& B_builder = cc.B_builder;

    // Zero, then scatter A's stored values and the dense rank-1 blocks through
    // the flat slots (same traversal order the build resolved). zero() also
    // clears any s2z rank-1 registered on B_builder, which this path never
    // sets, so B carries A + sum_k coef_k 1_k 1_k' exactly.
    B_builder.zero();
    double* __restrict__ Bv = B_builder.values.data();
    {
        int t = 0;
        for (int j = 0; j < n_x; ++j)
            for (int p = A_builder.col_ptr[j]; p < A_builder.col_ptr[j + 1]; ++p) {
                const int slot = cc.a_slots[t++];
                scatter_slot(Bv, slot, A_builder.values[p]);
            }
    }
    {
        int t = 0;
        for (int k = 0; k < K; ++k) {
            const double c = coupled ? coupling[(std::size_t) k * K + k] : r1[k].coef;
            const int nk = r1[k].n;
            for (int i = 0; i < nk; ++i)
                for (int j = 0; j <= i; ++j) {
                    const int slot = cc.block_slots[t++];
                    scatter_slot(Bv, slot, c);
                }
        }
    }
    if (coupled) {
        int t = 0;
        for (int a = 0; a < K; ++a)
            for (int b = 0; b < a; ++b) {
                const double d = coupling[(std::size_t) a * K + b];
                for (int i = 0; i < r1[a].n; ++i)
                    for (int j = 0; j < r1[b].n; ++j) {
                        const int slot = cc.cross_slots[t++];
                        scatter_slot(Bv, slot, d);
                    }
            }
    }

    cholmod_sparse B_cholmod = B_builder.as_cholmod(&cc.B_solver.common());
    if (!cc.B_solver.factorize(&B_cholmod)) return fallback;
    const double ld = cc.B_solver.log_determinant();
    return std::isfinite(ld) ? ld : fallback;
}

} // namespace tulpa

#endif // TULPA_S2Z_DIRECT_REFERENCE_H
