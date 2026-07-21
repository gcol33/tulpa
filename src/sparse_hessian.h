// sparse_hessian.h
// Sparse Hessian builder for Newton-Raphson in Laplace approximation.
// Eliminates the O(n²) DenseMat bottleneck for large spatial models.
//
// Usage:
//   1. Construct with sparsity pattern (list of (row, col) pairs)
//   2. Each Newton iteration: zero(), scatter entries, factorize+solve via CHOLMOD
//
// The sparsity pattern is fixed across iterations. Only values change.

#ifndef TULPA_SPARSE_HESSIAN_H
#define TULPA_SPARSE_HESSIAN_H

#include "laplace_newton.h"        // NewtonScratch (shared with dense solver)
#include "laplace_newton_loop.h"   // shared eval_*, line_search_backtrack, finalize_log_marginal
#include "laplace_profile.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <vector>
#include <map>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace tulpa {

class SparseHessianBuilder {
public:
    int n;                          // dimension
    std::vector<int> col_ptr;       // CSC column pointers
    std::vector<int> row_idx;       // CSC row indices
    std::vector<double> values;     // CSC values (zeroed each iteration)
    int nnz;

    // Map from (row, col) to flat index in values array, for lower triangle
    // only (symmetric, stype=-1).
    //
    // Held by shared_ptr-to-const because the map is fit-level immutable: it is
    // populated once in init() and only read thereafter (add() / lookup()). The
    // sparse joint driver replicates one builder across every outer thread (and
    // the cheap-pass workers); a deep copy of this map at n_sites ~ 10^6 is both
    // the dominant pre-grid time cost and ~4x the memory of the CSC arrays
    // (~48 B per std::map node vs 12 B per nonzero). Sharing it makes a builder
    // copy O(1) in the map and stores the pattern once for the whole fit
    //. A copied builder gets its own mutable `values` (deep
    // copied) but shares this read-only map; concurrent const lookups across
    // builders are safe since nothing reassigns it after init().
    using EntryMap = std::map<std::pair<int,int>, int>;
    std::shared_ptr<const EntryMap> entry_map;

    // Sum-to-zero rank-1 penalties registered by the scatter for intrinsic
    // fields too large to densify (see icar_s2z_densify). Each entry is the
    // dense rank-1 `coef * 1_block 1_block'` on indices [start, start+n) that
    // the adjacency sparsity pattern cannot hold. The sparse Newton solvers
    // fold them into the Newton step (Sherman-Morrison / Woodbury) from the
    // factor of the stored Hessian, and into the Laplace log-det by factoring
    // the well-conditioned `H + sum_k coef_k 1_k 1_k'` directly
    // (s2z_log_det_direct), so the result matches the dense full-11' path
    // without storing 11'. Cleared each zero(); typically length 1 (one
    // spatial field).
    struct S2ZRank1 { int start; int n; double coef; };
    std::vector<S2ZRank1> s2z_rank1;
    void add_s2z_rank1(int start, int n, double coef) {
        s2z_rank1.push_back({start, n, coef});
    }

    // Optional dense coupling D over the K registered vectors, so the fold is
    // the rank-K `U D U'` with U = [1_1 ... 1_K] rather than K independent
    // rank-1 terms. Row-major K*K, SPD; empty means D = diag(coef_k) and every
    // path behaves exactly as before.
    //
    // A Kronecker-structured field needs this: a multivariate CAR block carries
    // precision `Sigma^-1 (x) Q`, whose sum-to-zero augmentation is
    // `Sigma^-1 (x) 11'/J`. That couples FIELDS -- D[(a,c),(b,c)] =
    // Sinv[a,b]/J_c -- which K independent rank-1 terms cannot express.
    std::vector<double> s2z_coupling;
    void set_s2z_coupling(std::vector<double> D) { s2z_coupling = std::move(D); }

    // Initialize from a list of (row, col) pairs that define the sparsity pattern.
    // Only lower triangle entries needed (row >= col).
    void init(int dim, const std::vector<std::pair<int,int>>& pattern) {
        n = dim;

        // Deduplicate and sort by (col, row)
        std::set<std::pair<int,int>> unique_entries;
        for (auto& [r, c] : pattern) {
            int lo = std::min(r, c);
            int hi = std::max(r, c);
            unique_entries.insert({hi, lo});  // store as (row, col) with row >= col
        }

        // Build CSC (column-major, lower triangle)
        nnz = static_cast<int>(unique_entries.size());
        col_ptr.assign(n + 1, 0);
        row_idx.resize(nnz);
        values.resize(nnz, 0.0);

        // Sort by column then row
        struct Entry { int row, col; };
        std::vector<Entry> sorted;
        sorted.reserve(nnz);
        for (auto& [r, c] : unique_entries) {
            sorted.push_back({r, c});
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const Entry& a, const Entry& b) {
                      return a.col < b.col || (a.col == b.col && a.row < b.row);
                  });

        // Build the (row, col) -> flat-index map locally, then publish it as a
        // shared-to-const pattern (see the entry_map declaration). `sorted` is
        // ordered by (col, row), the map's own key order is (row, col); the
        // ordering only affects map-build cost, not correctness.
        auto local_map = std::make_shared<EntryMap>();
        int cur_col = 0;
        for (int e = 0; e < nnz; e++) {
            while (cur_col <= sorted[e].col) {
                col_ptr[cur_col] = e;
                cur_col++;
            }
            row_idx[e] = sorted[e].row;
            (*local_map)[{sorted[e].row, sorted[e].col}] = e;
        }
        while (cur_col <= n) {
            col_ptr[cur_col] = nnz;
            cur_col++;
        }
        entry_map = std::move(local_map);
    }

    // Zero all values (call at start of each Newton iteration)
    void zero() {
        std::fill(values.begin(), values.end(), 0.0);
        s2z_rank1.clear();
        s2z_coupling.clear();
    }

    // Add `ridge` to every diagonal entry. The CSC lower-triangle stores
    // each column's diagonal as the FIRST entry of the column (smallest
    // row index in the lower triangle is row == col), so the diagonal is
    // at `values[col_ptr[j]]` for column j with `row_idx[col_ptr[j]] == j`.
    // Used to apply the uniform upstream regularization (see
    // `LAPLACE_UNIFORM_RIDGE` in laplace_cholesky.h) after scatter and
    // before factorization.
    void add_uniform_ridge(double ridge) {
        for (int j = 0; j < n; j++) {
            const int idx = col_ptr[j];
            if (idx < col_ptr[j + 1] && row_idx[idx] == j) {
                values[idx] += ridge;
            }
        }
    }

    // Add value to entry (row, col). Handles symmetric storage (lower triangle).
    // If the entry doesn't exist in the pattern, it's silently dropped.
    void add(int row, int col, double val) {
        int lo = std::min(row, col);
        int hi = std::max(row, col);
        auto it = entry_map->find({hi, lo});
        if (it != entry_map->end()) {
            values[it->second] += val;
        }
    }

    // Return the flat index in `values` for the entry (row, col), or -1
    // if the entry is not in the pattern. Lower-triangle normalization
    // matches `add()`. Used by scatter callers that want to cache the
    // index once and then write via `values[idx] += val` directly,
    // avoiding the per-write std::map lookup. Stage 2.2 scatter index
    // cache.
    int lookup(int row, int col) const {
        int lo = std::min(row, col);
        int hi = std::max(row, col);
        auto it = entry_map->find({hi, lo});
        return (it == entry_map->end()) ? -1 : it->second;
    }

    // Create a cholmod_sparse view (does NOT allocate — reuses internal arrays).
    // The returned pointer is valid until the next init() or destruction.
    // Caller must NOT free the returned sparse matrix.
    cholmod_sparse as_cholmod(cholmod_common* common) const {
        cholmod_sparse A;
        A.nrow = n;
        A.ncol = n;
        A.nzmax = nnz;
        A.p = const_cast<int*>(col_ptr.data());
        A.i = const_cast<int*>(row_idx.data());
        A.x = const_cast<double*>(values.data());
        A.z = nullptr;
        A.stype = -1;   // lower triangle stored
        A.itype = CHOLMOD_INT;
        A.xtype = CHOLMOD_REAL;
        A.dtype = CHOLMOD_DOUBLE;
        A.sorted = 1;
        A.packed = 1;
        return A;
    }
};

// Fold the registered sum-to-zero rank-1 penalties into a Newton STEP already
// taken against the factor of A (the stored Hessian, WITHOUT the 11' terms). On
// entry `delta` = A^{-1} grad; on return delta solves
// (A + sum_k c_k 1_k 1_k') x = grad (Woodbury). One reuse-solve per penalty
// against the cached factor; K = #penalties is the count of intrinsic fields too
// large to densify (typically 1). Returns false on a non-finite solve (caller
// keeps the uncorrected delta).
//
// The capacitance  Cap = D^{-1} + U' A^{-1} U  (D = diag(coef), U = [1_block_k])
// is SPD; a tiny dense Cholesky solves it. This handles the STEP only. The
// log-determinant is NOT computed here: the matrix-determinant-lemma form
// log|A| + log|Cap| is a (-large)+(+large) cancellation along the constant
// direction (A pins it only through LAPLACE_UNIFORM_RIDGE, so 1'A^{-1}1 ~ 1/ridge
// and log|A| ~ log(ridge)). The step is unaffected (it is the Woodbury solution,
// not a determinant), so it stays exact; the determinant is obtained separately
// from a direct factorization of the well-conditioned A + UDU' (see
// s2z_log_det_direct).
// D^{-1} (dense K x K, row-major) and log|D| for the registered pins. An empty
// coupling is the diagonal D = diag(coef_k) every path used before Kronecker-
// coupled fields needed a dense D, so the two agree entry for entry there.
// Returns false if D is not SPD, which callers treat as "keep the uncorrected
// step / fall back" rather than as an error.
inline bool s2z_build_Dinv(
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    const std::vector<double>& coupling,
    std::vector<double>& Dinv,
    double& log_det_D
) {
    const int K = static_cast<int>(r1.size());
    Dinv.assign((std::size_t) K * K, 0.0);
    log_det_D = 0.0;
    if (K == 0) return true;

    if (coupling.empty()) {
        for (int k = 0; k < K; ++k) {
            const double c = r1[k].coef;
            if (!(c > 0.0) || !std::isfinite(c)) return false;
            Dinv[(std::size_t) k * K + k] = 1.0 / c;
            log_det_D += std::log(c);
        }
        return true;
    }
    if ((int) coupling.size() != K * K) return false;

    // D = L L': log|D| = 2 sum_j log L_jj, then D^{-1} column by column.
    std::vector<double> L((std::size_t) K * K, 0.0);
    for (int j = 0; j < K; ++j) {
        for (int k = 0; k <= j; ++k) {
            double sum = coupling[(std::size_t) j * K + k];
            for (int p = 0; p < k; ++p)
                sum -= L[(std::size_t) j * K + p] * L[(std::size_t) k * K + p];
            if (j == k) {
                if (!(sum > 0.0) || !std::isfinite(sum)) return false;
                L[(std::size_t) j * K + j] = std::sqrt(sum);
                log_det_D += 2.0 * std::log(L[(std::size_t) j * K + j]);
            } else {
                L[(std::size_t) j * K + k] = sum / L[(std::size_t) k * K + k];
            }
        }
    }
    std::vector<double> y(K, 0.0);
    for (int col = 0; col < K; ++col) {
        for (int j = 0; j < K; ++j) {
            double s = (j == col) ? 1.0 : 0.0;
            for (int p = 0; p < j; ++p) s -= L[(std::size_t) j * K + p] * y[p];
            y[j] = s / L[(std::size_t) j * K + j];
        }
        for (int j = K - 1; j >= 0; --j) {
            double s = y[j];
            for (int p = j + 1; p < K; ++p)
                s -= L[(std::size_t) p * K + j] * Dinv[(std::size_t) p * K + col];
            Dinv[(std::size_t) j * K + col] = s / L[(std::size_t) j * K + j];
        }
    }
    return true;
}

inline bool apply_s2z_rank1_correction(
    SparseCholeskySolver& solver, int n_x,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    double* delta,
    const std::vector<double>& coupling
) {
    const int K = static_cast<int>(r1.size());
    if (K == 0) return true;

    std::vector<std::vector<double>> W(K, std::vector<double>(n_x, 0.0));
    std::vector<double> rhs(n_x, 0.0);
    for (int k = 0; k < K; ++k) {
        std::fill(rhs.begin(), rhs.end(), 0.0);
        for (int i = 0; i < r1[k].n; ++i) rhs[r1[k].start + i] = 1.0;
        solver.solve(rhs.data(), W[k].data(), n_x);          // W_k = A^{-1} 1_k
        for (int i = 0; i < n_x; ++i)
            if (!std::isfinite(W[k][i])) return false;
    }

    // b_k = 1_k' delta = 1_k' A^{-1} grad;  M_{jk} = 1_j' W_k = (U'A^{-1}U)_{jk}.
    std::vector<double> b(K, 0.0);
    std::vector<double> M(K * K, 0.0);
    for (int k = 0; k < K; ++k) {
        double bk = 0.0;
        for (int i = 0; i < r1[k].n; ++i) bk += delta[r1[k].start + i];
        b[k] = bk;
        for (int j = 0; j < K; ++j) {
            double m = 0.0;
            for (int i = 0; i < r1[j].n; ++i) m += W[k][r1[j].start + i];
            M[j * K + k] = m;
        }
    }

    // Cap = D^{-1} + M (SPD). Dense Cholesky Cap = L L'.
    std::vector<double> Dinv; double log_det_D = 0.0;
    if (!s2z_build_Dinv(r1, coupling, Dinv, log_det_D)) return false;
    std::vector<double> Cap(K * K, 0.0);
    for (int j = 0; j < K; ++j)
        for (int k = 0; k < K; ++k)
            Cap[j * K + k] = M[j * K + k] + Dinv[(std::size_t) j * K + k];
    std::vector<double> L(K * K, 0.0);
    for (int j = 0; j < K; ++j) {
        for (int k = 0; k <= j; ++k) {
            double sum = Cap[j * K + k];
            for (int p = 0; p < k; ++p) sum -= L[j * K + p] * L[k * K + p];
            if (j == k) {
                if (sum <= 0.0) return false;   // not SPD: keep uncorrected
                L[j * K + j] = std::sqrt(sum);
            } else {
                L[j * K + k] = sum / L[k * K + k];
            }
        }
    }

    // Solve Cap y = b via L L' y = b (forward then back substitution).
    std::vector<double> y(K, 0.0);
    for (int j = 0; j < K; ++j) {
        double sum = b[j];
        for (int p = 0; p < j; ++p) sum -= L[j * K + p] * y[p];
        y[j] = sum / L[j * K + j];
    }
    for (int j = K - 1; j >= 0; --j) {
        double sum = y[j];
        for (int p = j + 1; p < K; ++p) sum -= L[p * K + j] * y[p];
        y[j] = sum / L[j * K + j];
    }

    // delta -= sum_k y_k W_k  (Woodbury step correction).
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < n_x; ++i)
            delta[i] -= y[k] * W[k][i];

    return true;
}

// Pattern-invariant cache for s2z_log_det_direct, reused across outer-grid cells
// and (in the batched driver) across species sharing one design. Within a fit
// the matrix B = A + sum_k coef_k 1_k 1_k' has a fixed sparsity pattern: A's
// structural pattern (the same for every grid cell and every species) plus each
// s2z block's full lower triangle. Only the numeric values change per call. The
// cache holds the parts that depend on the pattern alone:
//   * `B_builder` — B's CSC pattern + entry_map, built once;
//   * `a_slots` — for each A nonzero p, the flat values[] slot in B_builder, so
//     A's values scatter via `B.values[slot] += val` instead of a map lookup;
//   * `block_slots` — for each dense lower-triangle entry of every coef_k 1_k
//     1_k' block, the flat values[] slot, in block-then-(i,j) order;
//   * `B_solver` — a persistent CHOLMOD solver whose analyze() runs once and
//     whose factorize() re-runs per call (symbolic reuse via the analyzed()
//     guard, mirroring the Newton solver).
// Validity is keyed on n_x, A's nnz, and the s2z block layout (start/n per k),
// NOT on the builder identity, so a per-species builder with the same pattern
// reuses one cache. A different problem (changed n_x / nnz / block layout)
// triggers a rebuild.
struct S2ZLogDetCache {
    bool built = false;
    int  n_x   = -1;
    int  a_nnz = -1;
    std::vector<std::pair<int,int>> block_layout;   // (start, n) per rank-1 k

    SparseHessianBuilder B_builder;     // pattern + entry_map (once)
    std::vector<int>     a_slots;       // flat slot per A nonzero
    std::vector<int>     block_slots;   // flat slot per dense block LT entry
    std::vector<int>     cross_slots;   // flat slot per (a>b) cross-block entry
    SparseCholeskySolver B_solver;      // symbolic once, numeric per call

    // A dense coupling adds the cross-block rectangles to B's pattern, so a
    // cache built for one and reused for the other would scatter into the wrong
    // slots. Keyed on presence, not on D's values, which are numeric per call.
    bool coupled = false;

    bool matches(const SparseHessianBuilder& A_builder,
                 const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
                 bool want_coupled) const {
        if (!built || n_x != A_builder.n || a_nnz != A_builder.nnz) return false;
        if (coupled != want_coupled) return false;
        if (block_layout.size() != r1.size()) return false;
        for (std::size_t k = 0; k < r1.size(); ++k)
            if (block_layout[k].first != r1[k].start ||
                block_layout[k].second != r1[k].n)
                return false;
        return true;
    }
};

// Cross-block fill is quadratic in the total pinned length, so a dense coupling
// on a large field is refused here rather than allocated. s2z_block_schur folds
// the same D with no fill and is the path that carries those fits; this one is
// the small-n reference and the non-PD fallback.
constexpr long long S2Z_COUPLED_DIRECT_MAX_ENTRIES = 4000000LL;

// (Re)build the pattern-invariant cache for B = A + sum_k coef_k 1_k 1_k': B's
// CSC pattern + entry_map, the flat values[] slots for A's nonzeros and for each
// dense block lower-triangle entry, and the persistent solver's symbolic factor.
inline void build_s2z_log_det_cache(
    const SparseHessianBuilder& A_builder,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    bool coupled,
    S2ZLogDetCache& cache
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
        const int s = r1[k].start, nk = r1[k].n;
        for (int i = 0; i < nk; ++i)
            for (int j = 0; j <= i; ++j)
                pattern.emplace_back(s + i, s + j);
    }
    // D[a,b] fills the whole (a,b) rectangle: (U D U')_{pq} = D[a,b] for p in
    // block a, q in block b. init() folds each pair into the lower triangle.
    if (coupled)
        for (int a = 0; a < K; ++a)
            for (int b = 0; b < a; ++b)
                for (int i = 0; i < r1[a].n; ++i)
                    for (int j = 0; j < r1[b].n; ++j)
                        pattern.emplace_back(r1[a].start + i, r1[b].start + j);

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
        const int s = r1[k].start, nk = r1[k].n;
        for (int i = 0; i < nk; ++i)
            for (int j = 0; j <= i; ++j)
                cache.block_slots.push_back(cache.B_builder.lookup(s + i, s + j));
    }
    cache.cross_slots.clear();
    if (coupled)
        for (int a = 0; a < K; ++a)
            for (int b = 0; b < a; ++b)
                for (int i = 0; i < r1[a].n; ++i)
                    for (int j = 0; j < r1[b].n; ++j)
                        cache.cross_slots.push_back(
                            cache.B_builder.lookup(r1[a].start + i, r1[b].start + j));

    // Symbolic factor once; the numeric factorize re-runs per call against the
    // same B pattern. The cholmod_sparse view aliases B_builder's arrays, so it
    // stays valid as long as the cache (hence B_builder) lives.
    cache.B_solver.reset();
    cholmod_sparse B_view = cache.B_builder.as_cholmod(&cache.B_solver.common());
    cache.B_solver.analyze(&B_view);

    cache.n_x   = n_x;
    cache.a_nnz = A_builder.nnz;
    cache.block_layout.clear();
    cache.block_layout.reserve(K);
    for (int k = 0; k < K; ++k)
        cache.block_layout.emplace_back(r1[k].start, r1[k].n);
    cache.coupled = coupled;
    cache.built = true;
}

// Cancellation-free log-determinant for the sum-to-zero rank-1 penalties.
//
// Target: log|B|, B = A + sum_k coef_k 1_k 1_k', where A is the stored sparse
// Hessian (lower-triangle CSC in `A_builder`, already carrying
// LAPLACE_UNIFORM_RIDGE on its diagonal) and 1_k is the indicator of field block
// k on [start_k, start_k + n_k).
//
// Identity: log|B| is read directly from a Cholesky factor of B itself, the same
// well-conditioned matrix the dense densify path factors. This is exact and
// cancellation-free because the constant direction the penalty pins is held in B
// by coef_k 1_k 1_k' (an O(1) eigenvalue), not by the 1e-10 ridge that holds it
// in A. The matrix-determinant-lemma form log|A| + log|D^{-1} + U'A^{-1}U| is
// algebraically equal but numerically a (-large)+(+large) cancellation, because
// each pinned direction sits at the ridge in A (1'A^{-1}1 ~ 1/ridge,
// log|A| ~ log(ridge)). Factoring B directly never forms A^{-1} along 1, so no
// catastrophic subtraction occurs.
//
// B's pattern = A's pattern with each block k densified to its full lower
// triangle (where coef_k 1_k 1_k' has support). The pattern is invariant across
// grid cells and species within a fit; with a `cache` the CSC pattern, the flat
// scatter slots, and the symbolic factor are built once and reused, leaving each
// call to zero the values, scatter through the flat slots, numerically refactor,
// and read log|B|. Without a cache the same work is done statelessly. Returns
// log|B| on success. On any failure (allocation, non-PD) returns `fallback` (the
// bare log|A| the caller already holds) so the marginal stays finite.
// Small dense SPD Cholesky (row-major), for the K x K capacitance and the
// n_s x n_s Schur complement in the block-Schur log-determinant / solve.
struct SmallChol {
    int m = 0;
    std::vector<double> L;   // lower-triangle row-major
    bool ok = false;
    void factor(const std::vector<double>& M, int m_) {
        m = m_; L.assign((std::size_t) m * m, 0.0); ok = true;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j <= i; ++j) {
                double s = M[(std::size_t) i * m + j];
                for (int p = 0; p < j; ++p) s -= L[(std::size_t) i*m+p] * L[(std::size_t) j*m+p];
                if (i == j) {
                    if (s <= 0.0) { ok = false; return; }
                    L[(std::size_t) i*m+i] = std::sqrt(s);
                } else {
                    L[(std::size_t) i*m+j] = s / L[(std::size_t) j*m+j];
                }
            }
        }
    }
    void solve(const double* b, double* x) const {     // L L' x = b
        std::vector<double> y(m);
        for (int i = 0; i < m; ++i) {
            double s = b[i];
            for (int p = 0; p < i; ++p) s -= L[(std::size_t) i*m+p] * y[p];
            y[i] = s / L[(std::size_t) i*m+i];
        }
        for (int i = m - 1; i >= 0; --i) {
            double s = y[i];
            for (int p = i + 1; p < m; ++p) s -= L[(std::size_t) p*m+i] * x[p];
            x[i] = s / L[(std::size_t) i*m+i];
        }
    }
    double logdet() const {
        double d = 0.0;
        for (int i = 0; i < m; ++i) d += 2.0 * std::log(L[(std::size_t) i*m+i]);
        return d;
    }
};

// Pattern-invariant cache for s2z_block_schur, reused across outer-grid cells and
// (in the batched driver) across species sharing one design + s2z layout. Mirrors
// S2ZLogDetCache: the field/scalar partition, the A_FF sparsity pattern and its
// symbolic CHOLMOD factor, and the per-A-nonzero scatter destinations are fixed
// across a fit; only values change, so each call re-scatters + numerically
// re-factorizes. Validity keyed on n_x, A's nnz, and the s2z block layout.
struct S2ZBlockSchurCache {
    bool built = false;
    int  n_x = -1, a_nnz = -1, nf = 0, ns = 0;
    std::vector<std::pair<int,int>> block_layout;
    std::vector<int> floc, sloc;       // n_x: local field / scalar index, -1 otherwise
    std::vector<int> dest_kind;        // per A nonzero: 0=field-field, 1=scalar-scalar, 2=field-scalar
    std::vector<int> dest_a, dest_b;   // primary dest index; symmetric A_ss index (kind 1 off-diag) else -1
    SparseHessianBuilder aff;          // A_FF pattern + values (zeroed/scattered per call)
    SparseCholeskySolver aff_solver;   // symbolic factor once, numeric per call

    bool matches(const SparseHessianBuilder& A,
                 const std::vector<SparseHessianBuilder::S2ZRank1>& r1) const {
        if (!built || n_x != A.n || a_nnz != A.nnz) return false;
        if (block_layout.size() != r1.size()) return false;
        for (std::size_t k = 0; k < r1.size(); ++k)
            if (block_layout[k].first != r1[k].start || block_layout[k].second != r1[k].n)
                return false;
        return true;
    }
};

// (Re)build the pattern-invariant block-Schur cache: field/scalar partition, the
// A_FF CSC pattern + its symbolic factor, and each A nonzero's scatter target.
inline void build_s2z_block_schur_cache(
    const SparseHessianBuilder& A,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    S2ZBlockSchurCache& cache
) {
    const int n_x = A.n;
    const int K = static_cast<int>(r1.size());
    cache.floc.assign(n_x, -1);
    cache.sloc.assign(n_x, -1);
    int nf = 0;
    for (int g = 0; g < n_x; ++g) {
        bool isf = false;
        for (int k = 0; k < K; ++k)
            if (g >= r1[k].start && g < r1[k].start + r1[k].n) { isf = true; break; }
        if (isf) cache.floc[g] = nf++;
    }
    int ns = 0;
    for (int g = 0; g < n_x; ++g) if (cache.floc[g] < 0) cache.sloc[g] = ns++;
    cache.nf = nf; cache.ns = ns;

    std::vector<std::pair<int,int>> ffpat;
    cache.dest_kind.assign(A.nnz, 0);
    cache.dest_a.assign(A.nnz, -1);
    cache.dest_b.assign(A.nnz, -1);
    int t = 0;
    for (int c = 0; c < n_x; ++c)
        for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p, ++t) {
            const int r = A.row_idx[p];
            const int fr = cache.floc[r], fc = cache.floc[c], sr = cache.sloc[r], sc = cache.sloc[c];
            if (fr >= 0 && fc >= 0)      { cache.dest_kind[t] = 0; ffpat.emplace_back(fr, fc); }
            else if (sr >= 0 && sc >= 0) { cache.dest_kind[t] = 1; cache.dest_a[t] = sr*ns+sc;
                                           cache.dest_b[t] = (sr != sc) ? sc*ns+sr : -1; }
            else if (fr >= 0 && sc >= 0) { cache.dest_kind[t] = 2; cache.dest_a[t] = fr*ns + sc; }
            else                          { cache.dest_kind[t] = 2; cache.dest_a[t] = fc*ns + sr; }
        }
    cache.aff.init(nf, ffpat);
    // Resolve field-field value slots now the A_FF pattern (entry_map) exists.
    t = 0;
    for (int c = 0; c < n_x; ++c)
        for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p, ++t)
            if (cache.dest_kind[t] == 0)
                cache.dest_a[t] = cache.aff.lookup(cache.floc[A.row_idx[p]], cache.floc[c]);

    cache.aff_solver.reset();
    cholmod_sparse view = cache.aff.as_cholmod(&cache.aff_solver.common());
    cache.aff_solver.analyze(&view);

    cache.n_x = n_x; cache.a_nnz = A.nnz;
    cache.block_layout.clear(); cache.block_layout.reserve(K);
    for (int k = 0; k < K; ++k) cache.block_layout.emplace_back(r1[k].start, r1[k].n);
    cache.built = true;
}

// Block-Schur log-determinant for B = A + sum_k coef_k 1_k 1_k'. Partitions the
// latent into the field indices F (the union of the s2z block ranges) and the
// scalar complement S (intercepts, betas, REs). Factors the sparse field
// sub-block A_FF (PD -- pinned off the constant by per-cell data curvature at low
// tau and by the ICAR at high tau, verified across the grid), folds the K rank-1
// pins via the matrix-determinant lemma, and closes the field<->scalar coupling
// with a small dense Schur complement:
//   log|B| = log|A_FF| + log|C| + log|C^-1 + U' A_FF^-1 U|      (= log|B_FF|)
//          + log|A_ss - A_sf B_FF^-1 A_fs|.                     (Schur, n_s x n_s)
// U = [1_k] in local field coords, C = diag(coef_k). All sparse + O(K^3) +
// O(n_s^3); no dense field block, no uniform ridge. Exact and well-conditioned:
// A_FF is PD, so U'A_FF^-1 U is bounded -- the catastrophic cancellation would
// need the UNPINNED full A, which this never factors. Returns `fallback` on any non-PD /
// allocation failure.
inline bool s2z_block_schur(
    const SparseHessianBuilder& A,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    const double* grad,        // n_x gradient, or nullptr for determinant only
    double* delta,             // n_x Newton step B^-1 grad output, or nullptr
    double* logdet_out,        // log|B| output, or nullptr
    S2ZBlockSchurCache* cache = nullptr
) {
    const int K = static_cast<int>(r1.size());
    if (K == 0) return false;
    const int n_x = A.n;

    S2ZBlockSchurCache local_cache;
    S2ZBlockSchurCache& cc = cache ? *cache : local_cache;
    if (!cc.matches(A, r1)) build_s2z_block_schur_cache(A, r1, cc);
    const int nf = cc.nf, ns = cc.ns;
    const std::vector<int>& floc = cc.floc;
    const std::vector<int>& sloc = cc.sloc;

    // Scatter A's values into the cached A_FF pattern + dense A_ss / A_fs via the
    // precomputed per-nonzero destinations (no membership recompute, no re-sort).
    cc.aff.zero();
    double* __restrict__ affv = cc.aff.values.data();
    std::vector<double> Ass((std::size_t) ns * ns, 0.0);
    std::vector<double> Afs((std::size_t) nf * ns, 0.0);
    {
        int t = 0;
        for (int c = 0; c < n_x; ++c)
            for (int p = A.col_ptr[c]; p < A.col_ptr[c + 1]; ++p, ++t) {
                const double v = A.values[p];
                const int a = cc.dest_a[t];
                switch (cc.dest_kind[t]) {
                    case 0:  if (a >= 0) affv[a] += v; break;
                    case 1:  Ass[a] += v; if (cc.dest_b[t] >= 0) Ass[cc.dest_b[t]] += v; break;
                    default: Afs[a] += v; break;
                }
            }
    }

    cholmod_sparse vff = cc.aff.as_cholmod(&cc.aff_solver.common());
    if (!cc.aff_solver.analyzed()) cc.aff_solver.analyze(&vff);
    if (!cc.aff_solver.factorize(&vff)) return false;   // A_FF indefinite -> caller falls back to LM
    const double logAFF = cc.aff_solver.log_determinant();
    SparseCholeskySolver& sf = cc.aff_solver;

    // W = A_FF^-1 U (U = [1_k] local). cap = C^-1 + U'A_FF^-1 U (K x K, SPD).
    std::vector<std::vector<double>> W(K, std::vector<double>(nf, 0.0));
    {
        std::vector<double> uk(nf, 0.0);
        for (int k = 0; k < K; ++k) {
            std::fill(uk.begin(), uk.end(), 0.0);
            for (int i = 0; i < r1[k].n; ++i) uk[floc[r1[k].start + i]] = 1.0;
            sf.solve(uk.data(), W[k].data(), nf);
        }
    }
    // D enters only as D^{-1} in the cap and log|D| in the determinant, so a
    // Kronecker-coupled field costs nothing here beyond a dense K x K inverse:
    // no cross-block fill, no extra factorization.
    std::vector<double> Dinv; double logC = 0.0;
    if (!s2z_build_Dinv(r1, A.s2z_coupling, Dinv, logC)) return false;
    std::vector<double> cap((std::size_t) K * K, 0.0);
    for (int a = 0; a < K; ++a)
        for (int b = 0; b < K; ++b) {
            double m = 0.0;   // 1_a' W_b
            for (int i = 0; i < r1[a].n; ++i) m += W[b][floc[r1[a].start + i]];
            cap[(std::size_t) a*K + b] = m + Dinv[(std::size_t) a * K + b];
        }
    SmallChol capL; capL.factor(cap, K);
    if (!capL.ok) return false;
    const double logBFF = logAFF + logC + capL.logdet();

    // Schur S = A_ss - A_sf B_FF^-1 A_fs, with B_FF^-1 y = A_FF^-1 y - W cap^-1 (W'y).
    // Z[j] = B_FF^-1 A_fs[:,j] is reused for the step's field back-substitution.
    std::vector<std::vector<double>> Z(ns, std::vector<double>(nf, 0.0));
    std::vector<double> S = Ass;
    {
        std::vector<double> col(nf), ainv(nf), wty(K), capy(K);
        for (int j = 0; j < ns; ++j) {
            for (int i = 0; i < nf; ++i) col[i] = Afs[(std::size_t) i*ns + j];
            sf.solve(col.data(), ainv.data(), nf);                 // A_FF^-1 A_fs[:,j]
            for (int k = 0; k < K; ++k) {                           // W' col = U' A_FF^-1 col
                double s = 0.0;
                for (int i = 0; i < nf; ++i) s += W[k][i] * col[i];
                wty[k] = s;
            }
            capL.solve(wty.data(), capy.data());
            for (int i = 0; i < nf; ++i) {
                double corr = 0.0;
                for (int k = 0; k < K; ++k) corr += W[k][i] * capy[k];
                Z[j][i] = ainv[i] - corr;
            }
        }
        for (int i = 0; i < ns; ++i)
            for (int j = 0; j < ns; ++j) {
                double m = 0.0;                                    // A_fs[:,i]' Z[j]
                for (int t = 0; t < nf; ++t) m += Afs[(std::size_t) t*ns + i] * Z[j][t];
                S[(std::size_t) i*ns + j] -= m;
            }
    }
    SmallChol SL;
    double logS = 0.0;
    if (ns > 0) {
        SL.factor(S, ns);
        if (!SL.ok) return false;
        logS = SL.logdet();
    }
    if (logdet_out) {
        const double ld = logBFF + logS;
        if (!std::isfinite(ld)) return false;
        *logdet_out = ld;
    }

    // Step: delta = B^-1 grad via block elimination.
    //   bf_g = B_FF^-1 g_f ;  S s = g_s - A_fs' bf_g ;  f = bf_g - sum_j Z[j] s_j.
    if (grad && delta) {
        std::vector<double> gf(nf, 0.0), gs(ns, 0.0);
        for (int g = 0; g < n_x; ++g) {
            if (floc[g] >= 0) gf[floc[g]] = grad[g];
            else              gs[sloc[g]] = grad[g];
        }
        std::vector<double> ainv(nf, 0.0), bf_g(nf, 0.0), wty(K), capy(K);
        sf.solve(gf.data(), ainv.data(), nf);
        for (int k = 0; k < K; ++k) {
            double s = 0.0;
            for (int i = 0; i < nf; ++i) s += W[k][i] * gf[i];
            wty[k] = s;
        }
        capL.solve(wty.data(), capy.data());
        for (int i = 0; i < nf; ++i) {
            double corr = 0.0;
            for (int k = 0; k < K; ++k) corr += W[k][i] * capy[k];
            bf_g[i] = ainv[i] - corr;
        }
        std::vector<double> svec(ns, 0.0);
        if (ns > 0) {
            std::vector<double> rhs(ns, 0.0);
            for (int j = 0; j < ns; ++j) {
                double m = 0.0;
                for (int t = 0; t < nf; ++t) m += Afs[(std::size_t) t*ns + j] * bf_g[t];
                rhs[j] = gs[j] - m;
            }
            SL.solve(rhs.data(), svec.data());
        }
        for (int g = 0; g < n_x; ++g) {
            if (floc[g] >= 0) {
                const int i = floc[g];
                double acc = bf_g[i];
                for (int j = 0; j < ns; ++j) acc -= Z[j][i] * svec[j];
                delta[g] = acc;
            } else {
                delta[g] = svec[sloc[g]];
            }
        }
        for (int g = 0; g < n_x; ++g) if (!std::isfinite(delta[g])) return false;
    }
    return true;
}

// Determinant-only convenience wrapper over s2z_block_schur.
inline double s2z_log_det_block_schur(
    const SparseHessianBuilder& A,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    double fallback,
    S2ZBlockSchurCache* cache = nullptr
) {
    double ld;
    return s2z_block_schur(A, r1, nullptr, nullptr, &ld, cache) ? ld : fallback;
}

inline double s2z_log_det_direct(
    const SparseHessianBuilder& A_builder,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    double fallback,
    S2ZLogDetCache* cache = nullptr
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

    S2ZLogDetCache local_cache;
    S2ZLogDetCache& cc = cache ? *cache : local_cache;
    if (!cc.matches(A_builder, r1, coupled))
        build_s2z_log_det_cache(A_builder, r1, coupled, cc);

    SparseHessianBuilder& B_builder = cc.B_builder;

    // Zero, then scatter A's stored values and the dense rank-1 blocks through
    // the cached flat slots (same traversal order the cache resolved). zero()
    // also clears any s2z rank-1 registered on B_builder, which this path never
    // sets, so B carries A + sum_k coef_k 1_k 1_k' exactly.
    B_builder.zero();
    double* __restrict__ Bv = B_builder.values.data();
    {
        int t = 0;
        for (int j = 0; j < n_x; ++j)
            for (int p = A_builder.col_ptr[j]; p < A_builder.col_ptr[j + 1]; ++p) {
                const int slot = cc.a_slots[t++];
                if (slot >= 0) Bv[slot] += A_builder.values[p];
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
                    if (slot >= 0) Bv[slot] += c;
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
                        if (slot >= 0) Bv[slot] += d;
                    }
            }
    }

    cholmod_sparse B_cholmod = B_builder.as_cholmod(&cc.B_solver.common());
    if (!cc.B_solver.analyzed()) cc.B_solver.analyze(&B_cholmod);
    if (!cc.B_solver.factorize(&B_cholmod)) return fallback;
    const double ld = cc.B_solver.log_determinant();
    return std::isfinite(ld) ? ld : fallback;
}

// =====================================================================
// Sparse Newton solver: like laplace_newton_solve but with sparse H
// =====================================================================
//
// ComputeEta: same as before
// ScatterSparse: fn(x, eta, grad, SparseHessianBuilder&) — writes to sparse H
// CenterEffects: same
// ComputeLogPrior: same

// Scratch-aware sparse-Hessian Newton solver. Same scratch contract as the
// dense-Hessian solver in laplace_newton.h: caller owns NewtonScratch
// allocation outside the parallel region.
template<typename ComputeEta, typename ScatterSparse,
         typename CenterEffects, typename ComputeLogPrior>
LaplaceResult laplace_newton_solve_sparse(
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const std::string& family,
    double phi,
    int N, int n_x,
    int max_iter, double tol, int n_threads,
    ComputeEta compute_eta,
    ScatterSparse scatter_sparse,
    CenterEffects center_effects_fn,
    ComputeLogPrior compute_log_prior,
    SparseHessianBuilder& H_builder,
    NewtonScratch& scratch,
    const std::vector<double>& x_init,
    SparseCholeskySolver* shared_solver,
    bool store_Q
) {
    LaplaceResult result;
    result.mode.assign(n_x, 0.0);
    result.converged = false;
    result.n_iter = 0;
    result.log_det_Q = 0.0;
    result.log_marginal = 0.0;

    Rcpp::NumericVector& x = scratch.x;
    if (static_cast<int>(x_init.size()) == n_x) {
        for (int j = 0; j < n_x; j++) x[j] = x_init[j];
    } else {
        for (int j = 0; j < n_x; j++) x[j] = 0.0;
    }

    SparseCholeskySolver local_solver;
    SparseCholeskySolver& solver = shared_solver ? *shared_solver : local_solver;
    NewtonConvState conv_state;

    for (int iter = 0; iter < max_iter; iter++) {
        { TULPA_PROFILE_PHASE(PHASE_ETA);
          compute_eta(x, scratch.eta); }

        // Gradient (dense — it's only n_x entries, always manageable)
        DenseVec grad(n_x, 0.0);

        // Hessian (sparse!)
        H_builder.zero();
        { TULPA_PROFILE_PHASE(PHASE_SCATTER);
          scatter_sparse(x, scratch.eta, grad, H_builder); }

        // Uniform upstream ridge so the dense pivot-clamp and sparse-dbound
        // hacks aren't needed (see LAPLACE_UNIFORM_RIDGE in laplace_cholesky.h).
        H_builder.add_uniform_ridge(LAPLACE_UNIFORM_RIDGE);

        // Solve H * delta = grad via CHOLMOD
        cholmod_sparse H_cholmod = H_builder.as_cholmod(&solver.common());

        if (!solver.analyzed()) {
            TULPA_PROFILE_PHASE(PHASE_ANALYZE);
            solver.analyze(&H_cholmod);
        }

        std::vector<double> delta(n_x, 0.0);
        bool factorize_ok = false;
        { TULPA_PROFILE_PHASE(PHASE_FACTORIZE);
          factorize_ok = solver.factorize(&H_cholmod); }
        bool solve_ok = false;

        if (factorize_ok) {
            { TULPA_PROFILE_PHASE(PHASE_SOLVE);
              solver.solve(grad.data(), delta.data(), n_x);
              apply_s2z_rank1_correction(solver, n_x, H_builder.s2z_rank1,
                                         delta.data(), H_builder.s2z_coupling); }
            solve_ok = true;
            for (int j = 0; j < n_x; j++) {
                if (!std::isfinite(delta[j])) { solve_ok = false; break; }
            }
        }

        if (!solve_ok) {
            for (int j = 0; j < n_x; j++) {
                if (std::isfinite(delta[j])) x[j] += 0.1 * delta[j];
            }
            result.n_iter = iter + 1;
            continue;
        }

        // Step halving on penalized deviance (shared helper).
        auto eval_obj = [&](const Rcpp::NumericVector& xv) -> double {
            return eval_penalized_log_lik(
                xv, y, n_trials, N, family, phi, n_threads,
                compute_eta, compute_log_prior, scratch.eta_tmp
            );
        };

        double obj_old, step_scale;
        { TULPA_PROFILE_PHASE(PHASE_LINE_SEARCH);
          obj_old = eval_obj(x);
          double obj_unused = 0.0;
          double slope = newton_decrement(grad, delta, n_x);
          step_scale = line_search_backtrack(
              x, delta, n_x, obj_old, slope, eval_obj, obj_unused, scratch.x_try
          ); }

        result.n_iter = iter + 1;
        if (newton_converged(delta, grad, step_scale, n_x, tol, conv_state)) {
            result.converged = true;
            break;
        }
    }

    { TULPA_PROFILE_PHASE(PHASE_LOG_LIK_PRIOR);
      center_effects_fn(x); }
    for (int j = 0; j < n_x; j++) result.mode[j] = x[j];

    // Finalize: log-det and log-marginal. Reuse scratch.eta as eta_final.
    { TULPA_PROFILE_PHASE(PHASE_ETA);
      compute_eta(x, scratch.eta); }

    DenseVec grad_final(n_x, 0.0);
    H_builder.zero();
    { TULPA_PROFILE_PHASE(PHASE_SCATTER);
      scatter_sparse(x, scratch.eta, grad_final, H_builder); }
    H_builder.add_uniform_ridge(LAPLACE_UNIFORM_RIDGE);

    cholmod_sparse H_final = H_builder.as_cholmod(&solver.common());
    if (!solver.analyzed()) {
        TULPA_PROFILE_PHASE(PHASE_ANALYZE);
        solver.analyze(&H_final);
    }
    bool final_fact_ok;
    { TULPA_PROFILE_PHASE(PHASE_FACTORIZE);
      final_fact_ok = solver.factorize(&H_final); }
    if (final_fact_ok) {
        TULPA_PROFILE_PHASE(PHASE_LOG_DET);
        result.log_det_Q = solver.log_determinant();
        // Fold the sum-to-zero rank-1 penalties into log|H| by factoring the
        // well-conditioned H + sum_k coef_k 1_k 1_k' directly (cancellation-free;
        // see s2z_log_det_direct). Matches the dense full-11' path.
        result.log_det_Q = s2z_log_det_direct(H_builder, H_builder.s2z_rank1,
                                              result.log_det_Q);
    }

    double log_lik, log_prior;
    { TULPA_PROFILE_PHASE(PHASE_LOG_LIK_PRIOR);
      log_lik = compute_total_log_lik(y, n_trials, scratch.eta, N, family, phi, n_threads);
      log_prior = compute_log_prior(x, scratch.eta); }

    // A failed final factorize leaves no valid log|Q| (result.log_det_Q stays at
    // its init 0); finalizing anyway would hand an ill-conditioned cell an
    // inflated marginal that dominates the outer-grid weights. Drop it to -Inf
    // (weight 0) instead, matching the joint sparse path's PD-failure handling.
    result.log_marginal = final_fact_ok
        ? finalize_log_marginal(log_lik, log_prior, result.log_det_Q, n_x)
        : -INFINITY;

    if (store_Q) {
        // H_builder already holds Q in CSC lower-triangle form (stype = -1).
        // Copy out the arrays so the caller doesn't need to keep the builder
        // alive past Newton convergence.
        result.Q_csc_p = H_builder.col_ptr;
        result.Q_csc_i = H_builder.row_idx;
        result.Q_csc_x = H_builder.values;
        result.Q_csc_n = n_x;
    }

    return result;
}

// Convenience overload that allocates scratch locally. Use only outside
// parallel regions.
template<typename ComputeEta, typename ScatterSparse,
         typename CenterEffects, typename ComputeLogPrior>
LaplaceResult laplace_newton_solve_sparse(
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const std::string& family,
    double phi,
    int N, int n_x,
    int max_iter, double tol, int n_threads,
    ComputeEta compute_eta,
    ScatterSparse scatter_sparse,
    CenterEffects center_effects_fn,
    ComputeLogPrior compute_log_prior,
    SparseHessianBuilder& H_builder,
    const Rcpp::NumericVector& x_init = Rcpp::NumericVector(),
    SparseCholeskySolver* shared_solver = nullptr,
    bool store_Q = false
) {
    NewtonScratch scratch;
    scratch.allocate(n_x, N);
    std::vector<double> x_init_vec;
    if (x_init.size() == n_x) {
        x_init_vec.assign(x_init.begin(), x_init.end());
    }
    // n_threads flows into the scratch-taking overload, which sizes its own
    // regions; no process-global omp_set_num_threads here.
    return laplace_newton_solve_sparse(
        y, n_trials, family, phi, N, n_x,
        max_iter, tol, n_threads,
        compute_eta, scatter_sparse, center_effects_fn, compute_log_prior,
        H_builder, scratch, x_init_vec, shared_solver, store_Q
    );
}

} // namespace tulpa

#endif // TULPA_SPARSE_HESSIAN_H
