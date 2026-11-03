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

#include "cholmod_view.h"
#include "hessian_pattern_guard.h" // scatter_slot, out-of-pattern drop counter
#include "sparse_cholesky.h"
#include "small_dense_sym.h"  // SmallChol, SmallSymEig
#include <Rcpp.h>
#include <vector>
#include <map>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <limits>
#include <set>
#include <stdexcept>

namespace tulpa {

// Process-wide source of SparseHessianBuilder pattern generations. Unique across
// builders, so a generation also names a pattern to a SparseCholeskySolver
// shared between Hessians (SparseCholeskySolver::ensure_analyzed).
inline unsigned long long next_hessian_pattern_generation() {
    static std::atomic<unsigned long long> counter{0};
    return ++counter;
}

class SparseHessianBuilder {
public:
    int n = 0;                      // dimension
    std::vector<int> col_ptr;       // CSC column pointers
    std::vector<int> row_idx;       // CSC row indices
    std::vector<double> values;     // CSC values (zeroed each iteration)
    int nnz = 0;

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
    // fields too large to densify (see s2z_densify). Each entry is the dense
    // rank-1 `coef * 1_block 1_block'` on a component's nodes that the adjacency
    // sparsity pattern cannot hold. `idx` is the component's field-local node
    // list (nullptr for the contiguous run [start, start + n)): a connected map
    // and a replicate fold contiguously, a genuine disconnected map folds the
    // arbitrary node set. `idx` must point at a buffer that outlives the solve
    // (the fit's GraphPartition). The absolute index of member i is node(i).
    // The sparse Newton solvers fold them into the Newton step
    // (Sherman-Morrison / Woodbury) from the factor of the stored Hessian, and
    // into the Laplace log-det through the block-Schur factor of
    // `H + sum_k coef_k 1_k 1_k'` (s2z_block_schur), so the result matches the
    // dense full-11' path without storing 11'. Cleared each zero();
    // typically length 1 (one spatial field).
    struct S2ZRank1 {
        int start; int n; const int* idx; double coef;
        int node(int i) const { return start + (idx ? idx[i] : i); }
    };
    std::vector<S2ZRank1> s2z_rank1;
    void add_s2z_rank1(int start, int n, const int* idx, double coef) {
        s2z_rank1.push_back({start, n, idx, coef});
    }
    void add_s2z_rank1(int start, int n, double coef) {
        s2z_rank1.push_back({start, n, nullptr, coef});
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

    // Renewed by every init(), from a process-wide counter, so no two patterns
    // share a generation. The flat-write-index caches (scatter_dense_basis.h,
    // scatter_indexed_cache.h) hold offsets into `values` resolved by
    // lookup(row, col), and keyed on this builder's identity plus its shape.
    // Two different patterns installed into the same builder object with equal
    // nnz would satisfy such a key and hand back offsets pointing at other
    // entries -- a silently wrong Hessian, and one the pattern guard's drop
    // counter cannot see because the offset is valid, just not the one meant.
    // The generation makes re-initialisation visible to every cache at once.
    unsigned long long pattern_generation = 0;

    // Initialize from a list of (row, col) pairs that define the sparsity pattern.
    // Only lower triangle entries needed (row >= col).
    void init(int dim, const std::vector<std::pair<int,int>>& pattern) {
        n = dim;
        pattern_generation = next_hessian_pattern_generation();

        // Deduplicate and sort by (col, row). A pair outside [0, dim) is not
        // registered: col_ptr is sized dim + 1 and the CSC build below walks
        // cur_col up to each entry's column, so an out-of-range column writes
        // past the end of col_ptr. A rejected pair leaves its entry absent from
        // the pattern, which is the state add() / scatter_slot() already report
        // through the drop counter the enclosing HessianPatternGuard raises on
        // (see hessian_pattern_guard.h) as soon as anything nonzero is written
        // to it -- the reporting channel that works inside a parallel region.
        std::set<std::pair<int,int>> unique_entries;
        for (auto& [r, c] : pattern) {
            if (r < 0 || r >= n || c < 0 || c >= n) continue;
            int lo = std::min(r, c);
            int hi = std::max(r, c);
            unique_entries.insert({hi, lo});  // store as (row, col) with row >= col
        }

        // Build CSC (column-major, lower triangle). The entry count indexes
        // row_idx / values as an int everywhere below, so a pattern past
        // INT_MAX is rejected rather than silently narrowed.
        if (unique_entries.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::length_error(
                "SparseHessianBuilder::init: sparsity pattern has more than "
                "INT_MAX unique entries.");
        }
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

    // Install a pattern already in lower-triangle CSC form: column j holds rows
    // >= j in strictly ascending order, its diagonal first. Same result as
    // init() over the equivalent pair list, without the set-based dedupe. A
    // malformed CSC is refused rather than installed, since every slot read off
    // it would be wrong.
    void init_csc(int dim, std::vector<int> csc_p, std::vector<int> csc_i) {
        if (dim < 0 || static_cast<int>(csc_p.size()) != dim + 1 ||
            csc_p[0] != 0 || csc_p[dim] != static_cast<int>(csc_i.size())) {
            throw std::invalid_argument(
                "SparseHessianBuilder::init_csc: inconsistent column pointers.");
        }
        for (int j = 0; j < dim; j++) {
            const int b = csc_p[j], e = csc_p[j + 1];
            if (e <= b || csc_i[b] != j) {
                throw std::invalid_argument(
                    "SparseHessianBuilder::init_csc: a column lacks its diagonal.");
            }
            for (int k = b + 1; k < e; k++) {
                if (csc_i[k] <= csc_i[k - 1] || csc_i[k] >= dim) {
                    throw std::invalid_argument(
                        "SparseHessianBuilder::init_csc: rows must ascend "
                        "strictly within [0, dim).");
                }
            }
        }
        n = dim;
        pattern_generation = next_hessian_pattern_generation();
        nnz = static_cast<int>(csc_i.size());
        col_ptr = std::move(csc_p);
        row_idx = std::move(csc_i);
        values.assign(nnz, 0.0);
        auto local_map = std::make_shared<EntryMap>();
        for (int j = 0; j < n; j++) {
            for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) {
                local_map->emplace(std::make_pair(row_idx[k], j), k);
            }
        }
        entry_map = std::move(local_map);
        s2z_rank1.clear();
        s2z_coupling.clear();
    }

    // Flat index of the lower-triangle entry (row, col), row >= col, or -1 off
    // the pattern. Constant time when column `col` stores every row from col to
    // `row` contiguously -- a dense panel, or a dense diagonal block -- and a
    // binary search over the column otherwise; no map lookup either way.
    int csc_slot(int row, int col) const {
        const int b = col_ptr[col], e = col_ptr[col + 1];
        const int k = b + (row - col);
        if (k < e && row_idx[k] == row) return k;
        const int* first = row_idx.data() + b;
        const int* last  = row_idx.data() + e;
        const int* it = std::lower_bound(first, last, row);
        return (it != last && *it == row)
                   ? static_cast<int>(it - row_idx.data()) : -1;
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
    //
    // Every Laplace factorize callsite depends on this ridge landing on EVERY
    // column for positive-definiteness on a rank-deficient prior. A column with
    // no (j, j) in its pattern cannot take it, which is a malformed pattern
    // rather than a column to pass over, so the miss is counted through the
    // same channel a scatter miss uses and the driver's HessianPatternGuard
    // raises on it once its parallel region has joined. Raising here is not
    // available: this runs inside those regions.
    void add_uniform_ridge(double ridge) {
        for (int j = 0; j < n; j++) {
            const int idx = col_ptr[j];
            if (idx < col_ptr[j + 1] && row_idx[idx] == j) {
                values[idx] += ridge;
            } else if (ridge != 0.0) {
                record_hessian_pattern_drop();
            }
        }
    }

    // Add value to entry (row, col). Handles symmetric storage (lower triangle).
    // An entry outside the pattern cannot be stored; a nonzero contribution to
    // one is recorded so the enclosing driver's HessianPatternGuard can raise
    // (see hessian_pattern_guard.h). With TULPA_DEBUG_HESSIAN_DROPS set, the
    // first 200 misses also print their (row, col) so a pattern/scatter
    // mismatch can be localized to a block without a debugger.
    void add(int row, int col, double val) {
        int lo = std::min(row, col);
        int hi = std::max(row, col);
        auto it = entry_map->find({hi, lo});
        if (it != entry_map->end()) {
            values[it->second] += val;
        } else if (val != 0.0) {
            static const char* dbg = std::getenv("TULPA_DEBUG_HESSIAN_DROPS");
            if (dbg) {
                static std::atomic<int> printed{0};
                if (printed.fetch_add(1) < 200) {
                    Rprintf("HESSIAN_DROP row=%d col=%d val=%g\n", hi, lo, val);
                }
            }
            record_hessian_pattern_drop();
        }
    }

    // Return the flat index in `values` for the entry (row, col), or -1
    // if the entry is not in the pattern. Lower-triangle normalization
    // matches `add()`. Used by scatter callers that want to cache the
    // index once and then write via `values[idx] += val` directly,
    // avoiding the per-write std::map lookup.
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
        (void) common;
        return cholmod_lower_view(static_cast<std::size_t>(n),
                                  col_ptr.data(), row_idx.data(),
                                  values.data(),
                                  static_cast<std::size_t>(nnz));
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
// from the block-Schur factor of A + UDU' (see s2z_block_schur).
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

// `delta` holds `ncol` columns of length n_x, each corrected against the same
// W = A^{-1} U, solved in one call.
inline bool apply_s2z_rank1_correction(
    SparseCholeskySolver& solver, int n_x,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    double* delta,
    const std::vector<double>& coupling,
    int ncol = 1
) {
    const int K = static_cast<int>(r1.size());
    if (K == 0 || ncol <= 0) return true;
    const std::size_t n = static_cast<std::size_t>(n_x);

    std::vector<double> U(n * K, 0.0), W(n * K, 0.0);
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < r1[k].n; ++i) U[n * k + r1[k].node(i)] = 1.0;
    solver.solve(U.data(), W.data(), n_x, K);                  // W_k = A^{-1} 1_k
    for (double w : W) if (!std::isfinite(w)) return false;

    // M_{jk} = 1_j' W_k = (U'A^{-1}U)_{jk}.
    std::vector<double> M(K * K, 0.0);
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < K; ++j) {
            double m = 0.0;
            for (int i = 0; i < r1[j].n; ++i) m += W[n * k + r1[j].node(i)];
            M[j * K + k] = m;
        }

    // Cap = D^{-1} + M (SPD). Dense Cholesky Cap = L L'.
    std::vector<double> Dinv; double log_det_D = 0.0;
    if (!s2z_build_Dinv(r1, coupling, Dinv, log_det_D)) return false;
    std::vector<double> L(K * K, 0.0);
    for (int j = 0; j < K; ++j) {
        for (int k = 0; k <= j; ++k) {
            double sum = M[j * K + k] + Dinv[(std::size_t) j * K + k];
            for (int p = 0; p < k; ++p) sum -= L[j * K + p] * L[k * K + p];
            if (j == k) {
                if (sum <= 0.0) return false;   // not SPD: keep uncorrected
                L[j * K + j] = std::sqrt(sum);
            } else {
                L[j * K + k] = sum / L[k * K + k];
            }
        }
    }

    std::vector<double> y(K, 0.0);
    for (int c = 0; c < ncol; ++c) {
        double* d = delta + n * c;
        // b_k = 1_k' delta = 1_k' A^{-1} grad; Cap y = b via L L' y = b.
        for (int j = 0; j < K; ++j) {
            double sum = 0.0;
            for (int i = 0; i < r1[j].n; ++i) sum += d[r1[j].node(i)];
            for (int p = 0; p < j; ++p) sum -= L[j * K + p] * y[p];
            y[j] = sum / L[j * K + j];
        }
        for (int j = K - 1; j >= 0; --j) {
            double sum = y[j];
            for (int p = j + 1; p < K; ++p) sum -= L[p * K + j] * y[p];
            y[j] = sum / L[j * K + j];
        }
        // delta -= sum_k y_k W_k  (Woodbury step correction).
        for (int k = 0; k < K; ++k) {
            const double* wk = W.data() + n * k;
            for (std::size_t i = 0; i < n; ++i) d[i] -= y[k] * wk[i];
        }
    }
    return true;
}

// Absolute node indices of a rank-1 block (contiguous [start, start+n) or the
// component's arbitrary node set). Caches key on these so a changed partition
// (or a different fit reusing the scratch) rebuilds rather than scatters into
// stale slots; the per-call cost is negligible against the factorization.
inline std::vector<int> s2z_block_nodes(
    const SparseHessianBuilder::S2ZRank1& r
) {
    std::vector<int> v(r.n);
    for (int i = 0; i < r.n; ++i) v[i] = r.node(i);
    return v;
}

inline bool s2z_layout_matches(
    const std::vector<std::vector<int>>& layout,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1
) {
    if (layout.size() != r1.size()) return false;
    for (std::size_t k = 0; k < r1.size(); ++k) {
        if ((int) layout[k].size() != r1[k].n) return false;
        for (int i = 0; i < r1[k].n; ++i)
            if (layout[k][i] != r1[k].node(i)) return false;
    }
    return true;
}

// Largest relative backward error accepted from the LDL' factor of an
// indefinite field block, measured on the K solves A_FF W = U. CHOLMOD's LDL'
// does not pivot for stability, so a small pivot met early in the elimination
// can grow the factor's entries; a backward-stable factor leaves residuals near
// n * eps, and this bound refuses one whose growth reaches the solves.
constexpr double S2Z_LDL_BACKWARD_TOL = 1e-8;

// Pattern-invariant cache for s2z_block_schur, reused across outer-grid cells and
// (in the batched driver) across species sharing one design + s2z layout. It
// holds no densified block, so it stays sparse: the field/scalar partition, the A_FF sparsity pattern and its
// symbolic CHOLMOD factor, and the per-A-nonzero scatter destinations are fixed
// across a fit; only values change, so each call re-scatters + numerically
// re-factorizes. Validity keyed on n_x, A's nnz, and the s2z block layout.
struct S2ZBlockSchurCache {
    bool built = false;
    int  n_x = -1, a_nnz = -1, nf = 0, ns = 0;
    std::vector<std::vector<int>> block_layout;   // absolute nodes per rank-1 k
    std::vector<int> floc, sloc;       // n_x: local field / scalar index, -1 otherwise
    std::vector<int> dest_kind;        // per A nonzero: 0=field-field, 1=scalar-scalar, 2=field-scalar
    std::vector<int> dest_a, dest_b;   // primary dest index; symmetric A_ss index (kind 1 off-diag) else -1
    SparseHessianBuilder aff;          // A_FF pattern + values (zeroed/scattered per call)
    SparseCholeskySolver aff_solver;   // symbolic factor once, numeric per call
    SparseCholeskySolver aff_ldl;      // simplicial LDL' of A_FF, analyzed on first need
    unsigned long long pattern_generation = 0;   // A's pattern the partition read

    bool matches(const SparseHessianBuilder& A,
                 const std::vector<SparseHessianBuilder::S2ZRank1>& r1) const {
        if (!built || n_x != A.n || a_nnz != A.nnz ||
            pattern_generation != A.pattern_generation) return false;
        return s2z_layout_matches(block_layout, r1);
    }
};

// Row degree above which a coordinate of the block-Schur step is a dense
// scalar: AMD's own dense-row rule (CHOLMOD's default ordering), a row with
// more than max(16, 10 sqrt(n)) off-diagonal entries. Such a row would close
// the sparse factor as a dense trailing block anyway; the Schur complement
// carries it there without the fill.
inline int s2z_dense_degree(int n_x) {
    return std::max(16, static_cast<int>(10.0 * std::sqrt(static_cast<double>(n_x))));
}

// (Re)build the pattern-invariant block-Schur cache: field/scalar partition, the
// A_FF CSC pattern + its symbolic factor, and each A nonzero's scatter target.
void build_s2z_block_schur_cache(
    const SparseHessianBuilder& A,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    S2ZBlockSchurCache& cache
);

// Block-Schur log-determinant for B = A + sum_k coef_k 1_k 1_k'. Partitions the
// latent by the sparsity of A: the dense scalars S are the coordinates whose
// row of A exceeds s2z_dense_degree() (intercepts and coefficients loading
// every observation, a dense basis), the field F is every other coordinate and
// every pinned node. A pin decides nothing else: a block without one -- an
// NNGP or HSGP neighbour of an ICAR field, a random effect, a Knorr-Held
// interaction -- is sparse and factored as such, where a pin-based partition
// put its coordinates in the dense Schur complement (an n x n dense
// factorization per step for an n-node block). Factors the sparse field
// sub-block anchored at one node per pin, A_a = A_FF + E C_a E'
// (InvBlockConstraint), folds the K rank-1 pins and the anchors back out by the
// matrix-determinant lemma, and closes the field<->scalar coupling with a small
// dense Schur complement:
//   log|B| = log|A_a| + log|C| + sum log c_a + log|cap|          (= log|B_FF|)
//          + log|A_ss - A_sf B_FF^-1 A_fs|,                      (Schur, n_s x n_s)
//   cap = diag(C^-1, -C_a^-1) + Vb' A_a^-1 Vb,   Vb = [U E],
// U = [1_k] in local field coords, C = diag(coef_k) (or the coupling D). All
// sparse + O((2K)^3) + O(n_s^3); no dense field block, and no factor of the
// unpinned full A. The anchor is what keeps it well-conditioned: A_FF alone is
// PD at a converged mode for one intrinsic field, but two intrinsic fields
// loading the same rows (an additive space + time model) leave their opposite
// shift in A_FF with the uniform ridge as its only curvature, and a fold read
// off A_FF^-1 is then the difference of O(1 / ridge) terms.
//
// `field_inertia` extends this to an A_a that is not PD, which happens away
// from the mode (a screen's two-step solve, an iteration cap). B_FF can still
// be PD when A_a's negative directions lie in the span of the pins. A_a is then
// factored as LDL', and Haynsworth's inertia additivity on
// [[A_a, Vb], [Vb', -diag(C^-1, -C_a^-1)]] gives
//   neg(B_FF) = neg(A_a) + pos(cap) - K,
// so B_FF is PD exactly when neg(cap) = (number of anchors) + neg(A_a), and the
// determinant identity above holds with absolute values. The LDL' factor is
// accepted only within S2Z_LDL_BACKWARD_TOL on the W solves.
// `field_indefinite`, when given, reports whether A_a failed its Cholesky
// factorization, whether or not the inertia route then carried the call.
//
// Returns false on any non-PD / allocation failure, and on K == 0, where there
// is no penalty to fold in and log|B| = log|A|; the determinant-only wrapper
// turns both into the caller's `fallback`, so a caller that can reach K == 0
// has to separate the two itself. Both drivers gate the call on a non-empty
// penalty list instead.
bool s2z_block_schur(
    const SparseHessianBuilder& A,
    const std::vector<SparseHessianBuilder::S2ZRank1>& r1,
    const double* grad,        // n_x gradient, or nullptr for determinant only
    double* delta,             // n_x Newton step B^-1 grad output, or nullptr
    double* logdet_out,        // log|B| output, or nullptr
    S2ZBlockSchurCache* cache = nullptr,
    bool field_inertia = false,
    bool* field_indefinite = nullptr,
    int n_rhs = 1              // grad / delta hold n_rhs columns of length n_x
);

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


} // namespace tulpa

#endif // TULPA_SPARSE_HESSIAN_H
