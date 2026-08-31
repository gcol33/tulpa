// sparse_cholesky.cpp
// CHOLMOD sparse Cholesky solver implementation for tulpa
// Feature 1: CHOLMOD integration via R's Matrix package
//
// This file includes Matrix/stubs.c which provides the runtime stubs
// that resolve CHOLMOD functions via R_GetCCallable. This inclusion
// must happen in exactly one translation unit.
//
// IMPORTANT: The Matrix package remaps cholmod_* to M_cholmod_* via stubs.
// All CHOLMOD calls must use M_cholmod_* names (or R_MATRIX_CHOLMOD() macro).

#include "sparse_cholesky.h"
#include "hessian_pattern_guard.h"

#include <limits>

// Include the Matrix stubs — this defines the actual function bodies
// that call into the Matrix DLL at runtime. Must be in exactly one .cpp.
#include <Matrix/stubs.c>

namespace tulpa {

// =====================================================================
// SparseCholeskySolver implementation
// =====================================================================

SparseCholeskySolver::SparseCholeskySolver()
    : factor_(nullptr), A_owned_(nullptr), analyzed_(false), factored_(false)
{
    M_cholmod_start(&common_);
    // Prefer supernodal factorization (converts sparse to many small dense
    // BLAS ops → 5-20x faster than column-by-column on irregular sparsity)
    common_.supernodal = CHOLMOD_SUPERNODAL;
    // Silence CHOLMOD's "matrix not positive definite" prints: non-PD is an
    // expected, handled outcome here (the LM ridge escalates and refactorizes,
    // the caller checks the return value), not an error to surface to the user.
    common_.print = 0;
    common_.error_handler = nullptr;
}

SparseCholeskySolver::~SparseCholeskySolver() {
    if (factor_) {
        M_cholmod_free_factor(&factor_, &common_);
    }
    if (A_owned_) {
        M_cholmod_free_sparse(&A_owned_, &common_);
    }
    M_cholmod_finish(&common_);
}

void SparseCholeskySolver::reset() {
    if (factor_) {
        M_cholmod_free_factor(&factor_, &common_);
        factor_ = nullptr;
    }
    if (A_owned_) {
        M_cholmod_free_sparse(&A_owned_, &common_);
        A_owned_ = nullptr;
    }
    analyzed_ = false;
    factored_ = false;
}

// Allocate A_owned_ over the union of `keep`'s pattern (may be null) and the
// entries of H above drop_tol, and fill the values from H. Every diagonal is
// always present. Both row lists ascend within a column, so the union is a
// two-cursor merge. Any previously analyzed factor belongs to the old pattern
// and is dropped; the dispatch re-analyzes on the next call because
// analyzed() then reads false.
bool SparseCholeskySolver::build_owned_pattern(
    const DenseMat& H, int n, double drop_tol, cholmod_sparse* keep
) {
    const int* Kp = keep ? static_cast<const int*>(keep->p) : nullptr;
    const int* Ki = keep ? static_cast<const int*>(keep->i) : nullptr;

    auto in_keep = [&](int j, int& cursor, int i) {
        if (!Kp) return false;
        const int kend = Kp[j + 1];
        while (cursor < kend && Ki[cursor] < i) cursor++;
        return cursor < kend && Ki[cursor] == i;
    };

    size_t nnz = 0;
    for (int j = 0; j < n; j++) {
        int cursor = Kp ? Kp[j] : 0;
        for (int i = j; i < n; i++) {
            if (i == j || std::abs(H[i][j]) > drop_tol || in_keep(j, cursor, i))
                nnz++;
        }
    }

    cholmod_sparse* A = M_cholmod_allocate_sparse(
        n, n, nnz, 1, 1, -1, CHOLMOD_REAL, &common_
    );
    if (!A) return false;

    int* Ap = static_cast<int*>(A->p);
    int* Ai = static_cast<int*>(A->i);
    double* Ax = static_cast<double*>(A->x);

    size_t idx = 0;
    for (int j = 0; j < n; j++) {
        Ap[j] = static_cast<int>(idx);
        int cursor = Kp ? Kp[j] : 0;
        for (int i = j; i < n; i++) {
            if (i == j || std::abs(H[i][j]) > drop_tol || in_keep(j, cursor, i)) {
                Ai[idx] = i;
                Ax[idx] = H[i][j];
                idx++;
            }
        }
    }
    Ap[n] = static_cast<int>(idx);

    if (factor_) {
        M_cholmod_free_factor(&factor_, &common_);
        factor_ = nullptr;
    }
    analyzed_ = false;
    factored_ = false;
    if (A_owned_) M_cholmod_free_sparse(&A_owned_, &common_);
    A_owned_ = A;
    return true;
}

cholmod_sparse* SparseCholeskySolver::refill_from_dense(
    const DenseMat& H, int n, double drop_tol
) {
    if (!A_owned_) {
        // First call: discover the pattern from this H and cache it.
        return build_owned_pattern(H, n, drop_tol, nullptr) ? A_owned_ : nullptr;
    }

    // Subsequent calls: refill Ax in place from the cached (Ap, Ai). No
    // allocator traffic. The lower triangle is walked in full because the
    // cached pattern came from the first H's VALUES, and that is a property of
    // one iterate rather than of the model: an entry at or below drop_tol then
    // has no slot, yet the same entry can carry real curvature at another
    // Newton iterate, outer-grid cell or hyperparameter value. A varying-
    // coefficient block is the ordinary case -- its cross terms scale with the
    // per-row design weight and the block amplitude, so which of them clear the
    // tolerance changes from cell to cell.
    //
    // So an entry found off the pattern GROWS the pattern rather than being
    // discarded. Growth is monotone (the union keeps every cached entry), so it
    // converges after the first few cells and costs one reallocation and one
    // re-analyze each time. Discarding instead would factor a matrix that is
    // missing curvature, and the Newton direction, log|H| and the standard
    // errors would all inherit it.
    const int* Ap = static_cast<const int*>(A_owned_->p);
    const int* Ai = static_cast<const int*>(A_owned_->i);
    double* Ax    = static_cast<double*>(A_owned_->x);
    bool grow = false;
    for (int j = 0; j < n && !grow; j++) {
        int idx = Ap[j];
        const int end = Ap[j + 1];
        for (int i = j; i < n; i++) {
            const double v = H[i][j];
            if (idx < end && Ai[idx] == i) {
                idx++;
            } else if (i == j || std::abs(v) > drop_tol) {
                grow = true;
                break;
            }
        }
    }

    if (grow) {
        // Union with the cached pattern, so an entry that mattered at an
        // earlier cell keeps its slot even where it vanishes at this one.
        if (build_owned_pattern(H, n, drop_tol, A_owned_)) return A_owned_;
        // Out of memory for the grown pattern. Keep the cached one and record
        // what it cannot hold, so the enclosing HessianPatternGuard reports a
        // factorization that is missing curvature instead of it passing
        // silently.
        Ap = static_cast<const int*>(A_owned_->p);
        Ai = static_cast<const int*>(A_owned_->i);
        Ax = static_cast<double*>(A_owned_->x);
    }

    for (int j = 0; j < n; j++) {
        int idx = Ap[j];
        const int end = Ap[j + 1];
        for (int i = j; i < n; i++) {
            const double v = H[i][j];
            if (idx < end && Ai[idx] == i) {
                Ax[idx] = v;
                idx++;
            } else if (i == j || std::abs(v) > drop_tol) {
                record_hessian_pattern_drop();
            }
        }
    }
    return A_owned_;
}

void SparseCholeskySolver::analyze(cholmod_sparse* A) {
    if (factor_) {
        M_cholmod_free_factor(&factor_, &common_);
        factor_ = nullptr;
    }
    factor_ = M_cholmod_analyze(A, &common_);
    analyzed_ = (factor_ != nullptr);
    factored_ = false;
}

bool SparseCholeskySolver::factorize(cholmod_sparse* A) {
    if (!analyzed_ || !factor_) return false;
    int ok = M_cholmod_factorize(A, factor_, &common_);
    factored_ = (ok != 0) && (common_.status == CHOLMOD_OK);
    return factored_;
}

std::size_t SparseCholeskySolver::analyzed_factor_bytes() const {
    if (!factor_) return 0;
    const cholmod_factor* L = factor_;
    if (L->is_super) {
        // Supernodal: factorize() allocates L->x with xsize doubles; the
        // integer supernode structure (s, plus super/pi/px of size nsuper+1)
        // is already present from the symbolic analyze.
        const std::size_t x_bytes = static_cast<std::size_t>(L->xsize) * sizeof(double);
        const std::size_t s_bytes = static_cast<std::size_t>(L->ssize) * sizeof(int);
        const std::size_t super_bytes =
            3 * (static_cast<std::size_t>(L->nsuper) + 1) * sizeof(int);
        return x_bytes + s_bytes + super_bytes;
    }
    // Simplicial: values + row indices + column pointers.
    const std::size_t x_bytes = static_cast<std::size_t>(L->nzmax) * sizeof(double);
    const std::size_t i_bytes = static_cast<std::size_t>(L->nzmax) * sizeof(int);
    const std::size_t p_bytes = (static_cast<std::size_t>(L->n) + 1) * sizeof(int);
    return x_bytes + i_bytes + p_bytes;
}

bool SparseCholeskySolver::solve(const double* b, double* x, int n) {
    // A failed solve fills NaN, never zero: zero is a valid step and reads as
    // convergence to a caller testing max|delta| < tol. See the header.
    const double failed = std::numeric_limits<double>::quiet_NaN();
    if (!factored_ || !factor_) {
        for (int i = 0; i < n; i++) x[i] = failed;
        return false;
    }

    // Create dense RHS from raw pointer (stack-allocated, no CHOLMOD alloc)
    cholmod_dense b_dense;
    b_dense.nrow = n;
    b_dense.ncol = 1;
    b_dense.nzmax = n;
    b_dense.d = n;
    b_dense.x = const_cast<double*>(b);
    b_dense.z = nullptr;
    b_dense.xtype = CHOLMOD_REAL;
    b_dense.dtype = CHOLMOD_DOUBLE;

    // Solve Ax = b (CHOLMOD_A = 0: full solve using LL' or LDL')
    cholmod_dense* x_dense = M_cholmod_solve(CHOLMOD_A, factor_, &b_dense, &common_);

    if (!x_dense) {
        for (int i = 0; i < n; i++) x[i] = failed;
        return false;
    }
    double* xp = static_cast<double*>(x_dense->x);
    for (int i = 0; i < n; i++) x[i] = xp[i];
    M_cholmod_free_dense(&x_dense, &common_);
    return true;
}

void SparseCholeskySolver::SolveWorkspace::release() {
    if (common_) {
        if (X_) M_cholmod_free_dense(&X_, common_);
        if (Y_) M_cholmod_free_dense(&Y_, common_);
        if (E_) M_cholmod_free_dense(&E_, common_);
    }
    X_ = nullptr;
    Y_ = nullptr;
    E_ = nullptr;
    common_ = nullptr;
}

SparseCholeskySolver::SolveWorkspace::~SolveWorkspace() {
    release();
}

bool SparseCholeskySolver::solve(const double* b, double* x, int n,
                                 SolveWorkspace& ws) {
    // Same NaN-on-failure contract as the allocating solve above.
    const double failed = std::numeric_limits<double>::quiet_NaN();
    if (!factored_ || !factor_) {
        for (int i = 0; i < n; i++) x[i] = failed;
        return false;
    }
    // A handle allocated against another solver's cholmod_common cannot be
    // resized or freed against this one. Refusing keeps the failure inside the
    // return value; this runs inside OpenMP regions, where a throw is
    // std::terminate rather than an R error.
    if (ws.common_ && ws.common_ != &common_) {
        for (int i = 0; i < n; i++) x[i] = failed;
        return false;
    }
    ws.common_ = &common_;

    // Dense RHS over the caller's buffer (stack-allocated, no CHOLMOD alloc).
    cholmod_dense b_dense;
    b_dense.nrow = n;
    b_dense.ncol = 1;
    b_dense.nzmax = n;
    b_dense.d = n;
    b_dense.x = const_cast<double*>(b);
    b_dense.z = nullptr;
    b_dense.xtype = CHOLMOD_REAL;
    b_dense.dtype = CHOLMOD_DOUBLE;

    // CHOLMOD_A: full solve using LL' or LDL'. cholmod_solve2 allocates each
    // handle that is still null and resizes one that no longer fits, so the
    // first call pays the allocation and the rest reuse it.
    int ok = M_cholmod_solve2(CHOLMOD_A, factor_, &b_dense,
                              &ws.X_, &ws.Y_, &ws.E_, &common_);
    if (!ok || !ws.X_ || !ws.X_->x) {
        for (int i = 0; i < n; i++) x[i] = failed;
        return false;
    }
    const double* xp = static_cast<const double*>(ws.X_->x);
    for (int i = 0; i < n; i++) x[i] = xp[i];
    return true;
}

bool SparseCholeskySolver::apply_inv_chol_factor(const double* eps, double* x,
                                                 int n, int ncol) {
    if (!factored_ || !factor_ || n <= 0 || ncol <= 0) return false;
    // LDL' has no square root on this route -- see the header.
    if (!factor_->is_ll) return false;

    cholmod_dense b_dense;
    b_dense.nrow  = n;
    b_dense.ncol  = ncol;
    b_dense.nzmax = static_cast<std::size_t>(n) * ncol;
    b_dense.d     = n;
    b_dense.x     = const_cast<double*>(eps);
    b_dense.z     = nullptr;
    b_dense.xtype = CHOLMOD_REAL;
    b_dense.dtype = CHOLMOD_DOUBLE;

    // L' y = eps, then undo the fill-reducing permutation: x = P' y.
    cholmod_dense* y = M_cholmod_solve(CHOLMOD_Lt, factor_, &b_dense, &common_);
    if (!y) return false;
    cholmod_dense* z = M_cholmod_solve(CHOLMOD_Pt, factor_, y, &common_);
    M_cholmod_free_dense(&y, &common_);
    if (!z) return false;

    const double* zp = static_cast<const double*>(z->x);
    const std::size_t total = static_cast<std::size_t>(n) * ncol;
    for (std::size_t i = 0; i < total; i++) x[i] = zp[i];
    M_cholmod_free_dense(&z, &common_);
    return true;
}

double SparseCholeskySolver::log_determinant() const {
    if (!factored_ || !factor_) return 0.0;
    return M_cholmod_factor_ldetA(factor_);
}

// =====================================================================
// Free functions: Takahashi recursion on caller-provided L
// =====================================================================
//
// Selected inversion via Takahashi equations for LL' factorization.
// Reference: Rue & Held (2005), Appendix B; Erisman & Tinney (1975).
//
// For Q = LL', compute Z = Q^{-1} at the sparsity positions of L.
// Process columns j from n-1 down to 0:
//
//   Z[i,j] = -1/L[j,j] * Σ_{k: L[k,j]≠0, k>j} L[k,j] * Z[i,k]   for i > j
//   Z[j,j] =  1/L[j,j] * (1/L[j,j] - Σ_{k: L[k,j]≠0, k>j} L[k,j] * Z[k,j])
//
// Z is symmetric, so Z[i,k] = Z[k,i] when needed.

// The three structural preconditions the recursion below reads L under. One
// pass over Lp / Li; cheap against the recursion, and the only thing standing
// between a foreign factor and silently wrong marginal variances.
static bool takahashi_valid_factor(int n, const int* Lp, const int* Li) {
    if (n < 0 || Lp[0] != 0) return false;
    for (int j = 0; j < n; j++) {
        const int col_start = Lp[j];
        const int col_end = Lp[j + 1];
        if (col_end < col_start) return false;
        if (col_start == col_end) continue;
        // Diagonal in the first slot: the recursion reads L[j,j] as Lx[Lp[j]].
        if (Li[col_start] != j) return false;
        // Strictly ascending, which is both "sorted" and "no duplicates": the
        // entry lookup stops at the first row index past the one it wants.
        for (int s = col_start + 1; s < col_end; s++) {
            if (Li[s] <= Li[s - 1]) return false;
            if (Li[s] >= n) return false;
        }
    }
    return true;
}

bool takahashi_partial_inverse_csc(
    int n,
    const int* Lp,
    const int* Li,
    const double* Lx,
    double* Zx_out
) {
    if (n <= 0) return n == 0;
    int nnz = Lp[n];
    for (int idx = 0; idx < nnz; idx++) Zx_out[idx] = 0.0;

    if (!takahashi_valid_factor(n, Lp, Li)) return false;

    for (int j = n - 1; j >= 0; j--) {
        int col_start = Lp[j];
        int col_end = Lp[j + 1];
        if (col_start >= col_end) continue;

        double Ljj = Lx[col_start];
        // A pivot this small is not a column to skip: every later column's
        // recursion reads Z entries from this one, so continuing produces a
        // partial inverse whose zeros are indistinguishable from computed
        // values, including a zero marginal variance at node j.
        if (!(std::abs(Ljj) >= 1e-15)) {
            for (int idx = 0; idx < nnz; idx++) Zx_out[idx] = 0.0;
            return false;
        }
        double Ljj_inv = 1.0 / Ljj;

        // Off-diagonal entries Z[i,j] for i > j (bottom-up within column j).
        for (int idx_i = col_end - 1; idx_i > col_start; idx_i--) {
            int i = Li[idx_i];

            // Z[i,j] = -1/L[j,j] * Σ_{k>j, L[k,j]≠0} L[k,j] * Z[i,k].
            // Z is stored in lower triangle: Z[hi, lo] in column lo at row hi.
            double sum = 0.0;
            for (int idx_k = col_start + 1; idx_k < col_end; idx_k++) {
                int k = Li[idx_k];

                int lo = std::min(i, k);
                int hi = std::max(i, k);

                double z_ik = 0.0;
                if (lo == hi) {
                    z_ik = Zx_out[Lp[lo]];  // diagonal
                } else {
                    for (int s = Lp[lo]; s < Lp[lo + 1]; s++) {
                        if (Li[s] == hi) { z_ik = Zx_out[s]; break; }
                        if (Li[s] > hi) break;
                    }
                }

                sum += Lx[idx_k] * z_ik;
            }
            Zx_out[idx_i] = -Ljj_inv * sum;
        }

        // Diagonal: Z[j,j] = 1/L[j,j] * (1/L[j,j] - Σ_{k>j} L[k,j] * Z[k,j]).
        double sum_diag = 0.0;
        for (int idx = col_start + 1; idx < col_end; idx++) {
            sum_diag += Lx[idx] * Zx_out[idx];
        }
        Zx_out[col_start] = Ljj_inv * (Ljj_inv - sum_diag);
    }
    return true;
}

bool takahashi_partial_inverse_dense(
    int n,
    const int* Lp,
    const int* Li,
    const double* Lx,
    double* Z_out
) {
    // Zero the dense buffer first: a refused recursion leaves it zeroed rather
    // than holding whatever the caller allocated.
    for (size_t idx = 0; idx < (size_t)n * n; idx++) Z_out[idx] = 0.0;
    if (n <= 0) return n == 0;

    int nnz = Lp[n];
    std::vector<double> Zx(nnz, 0.0);
    if (!takahashi_partial_inverse_csc(n, Lp, Li, Lx, Zx.data())) return false;

    // Scatter Zx onto the dense layout (column-major) and mirror to upper.
    for (int j = 0; j < n; j++) {
        for (int idx = Lp[j]; idx < Lp[j + 1]; idx++) {
            int i = Li[idx];
            double z = Zx[idx];
            Z_out[i + (size_t)j * n] = z;     // [i, j]
            Z_out[j + (size_t)i * n] = z;     // [j, i] (symmetric)
        }
    }
    return true;
}

double SparseCholeskySolver::SelectedInverse::at(int i_orig, int j_orig) const {
    // An unfactorized solve returns a default-constructed SelectedInverse
    // (n == 0, every array empty); an index outside the factored dimension has
    // no entry either. Both read 0.0 rather than dereferencing out of range.
    if (n == 0 || i_orig < 0 || i_orig >= n || j_orig < 0 || j_orig >= n) {
        return 0.0;
    }
    int pi = perm_inv[i_orig];
    int pj = perm_inv[j_orig];
    int lo = std::min(pi, pj);
    int hi = std::max(pi, pj);
    for (int s = Lp[lo]; s < Lp[lo + 1]; s++) {
        if (Li[s] == hi) return Zx[s];
        if (Li[s] > hi) break;
    }
    return 0.0;
}

SparseCholeskySolver::SelectedInverse SparseCholeskySolver::selected_inversion_full() {
    SelectedInverse out;
    if (!factored_ || !factor_) return out;

    int n = static_cast<int>(factor_->n);

    // Convert to simplicial LL' if supernodal (Takahashi needs element access).
    // The conversion allocates, so it can fail; a failed conversion leaves the
    // factor supernodal, where L->p is supernode pointers and L->i is not the
    // CSC row index array, so reading Lp[n] as nnz below would walk unrelated
    // memory. Return the empty struct, which every caller treats as failure.
    if (factor_->is_super) {
        int ok = M_cholmod_change_factor(
            CHOLMOD_REAL,  // xtype
            1,             // to_ll (LL')
            0,             // to_super = false (simplicial)
            1,             // to_packed
            1,             // to_monotonic
            factor_, &common_
        );
        if (!ok || common_.status != CHOLMOD_OK) return SelectedInverse{};
    }

    // Ensure LL' form (not LDL'): the Takahashi recursion reads the diagonal of
    // L at Lx[Lp[j]] and assumes a true Cholesky factor.
    if (!factor_->is_ll) {
        int ok = M_cholmod_change_factor(
            CHOLMOD_REAL, 1, 0, 1, 1,
            factor_, &common_
        );
        if (!ok || common_.status != CHOLMOD_OK) return SelectedInverse{};
    }

    // Read the achieved form rather than inferring it from the calls having
    // been made.
    if (factor_->is_super || !factor_->is_ll) return SelectedInverse{};

    int* Lp = static_cast<int*>(factor_->p);
    int* Li = static_cast<int*>(factor_->i);
    double* Lx = static_cast<double*>(factor_->x);
    int* Perm = static_cast<int*>(factor_->Perm);

    int nnz = Lp[n];
    out.n = n;
    out.Lp.assign(Lp, Lp + n + 1);
    out.Li.assign(Li, Li + nnz);
    out.Zx.assign(nnz, 0.0);
    // A refused recursion (malformed factor, unusable pivot) returns the empty
    // struct every caller already treats as failure, rather than a Z of zeros.
    if (!takahashi_partial_inverse_csc(n, Lp, Li, Lx, out.Zx.data())) {
        return SelectedInverse{};
    }

    // original index -> permuted index (inverse of the factor's Perm).
    out.perm_inv.resize(n);
    for (int j = 0; j < n; j++) {
        int orig_j = Perm ? Perm[j] : j;
        out.perm_inv[orig_j] = j;
    }

    return out;
}

std::vector<double> SparseCholeskySolver::selected_inversion_diagonal() {
    SelectedInverse si = selected_inversion_full();
    if (si.n == 0) return {};

    // Diagonal of Z in original ordering: Z[orig_j, orig_j] is the diagonal
    // entry of column perm_inv[orig_j], which is the first slot of that column.
    std::vector<double> diag_inv(si.n, 0.0);
    for (int orig_j = 0; orig_j < si.n; orig_j++) {
        int pj = si.perm_inv[orig_j];
        diag_inv[orig_j] = si.Zx[si.Lp[pj]];
    }

    return diag_inv;
}

// =====================================================================
// CSC conversion helpers
// =====================================================================

cholmod_sparse* dense_to_cholmod_sparse(
    const DenseMat& H, int n,
    cholmod_common* common
) {
    // Full lower triangle: n*(n+1)/2 entries
    size_t nnz = (size_t)n * (n + 1) / 2;

    // stype = -1: lower triangle stored, matrix treated as symmetric
    cholmod_sparse* A = M_cholmod_allocate_sparse(
        n, n, nnz,
        1,    // sorted
        1,    // packed
        -1,   // stype: lower triangle
        CHOLMOD_REAL,
        common
    );

    if (!A) return nullptr;

    int* Ap = static_cast<int*>(A->p);
    int* Ai = static_cast<int*>(A->i);
    double* Ax = static_cast<double*>(A->x);

    // Fill CSC: column j has rows j..n-1 (lower triangle)
    size_t idx = 0;
    for (int j = 0; j < n; j++) {
        Ap[j] = static_cast<int>(idx);
        for (int i = j; i < n; i++) {
            Ai[idx] = i;
            Ax[idx] = H[i][j];
            idx++;
        }
    }
    Ap[n] = static_cast<int>(idx);

    return A;
}

cholmod_sparse* dense_to_cholmod_sparse_drop(
    const DenseMat& H, int n,
    double drop_tol,
    cholmod_common* common
) {
    // First pass: count non-zeros in lower triangle above threshold
    size_t nnz = 0;
    for (int j = 0; j < n; j++) {
        for (int i = j; i < n; i++) {
            if (i == j || std::abs(H[i][j]) > drop_tol) {
                nnz++;
            }
        }
    }

    cholmod_sparse* A = M_cholmod_allocate_sparse(
        n, n, nnz,
        1, 1, -1,
        CHOLMOD_REAL,
        common
    );

    if (!A) return nullptr;

    int* Ap = static_cast<int*>(A->p);
    int* Ai = static_cast<int*>(A->i);
    double* Ax = static_cast<double*>(A->x);

    size_t idx = 0;
    for (int j = 0; j < n; j++) {
        Ap[j] = static_cast<int>(idx);
        for (int i = j; i < n; i++) {
            if (i == j || std::abs(H[i][j]) > drop_tol) {
                Ai[idx] = i;
                Ax[idx] = H[i][j];
                idx++;
            }
        }
    }
    Ap[n] = static_cast<int>(idx);

    return A;
}

} // namespace tulpa

// Rcpp export: compute marginal variances (diagonal of Q^{-1}) via selected inversion
// Q is provided as a symmetric sparse matrix (CSC, lower triangle, from R's Matrix package)
// [[Rcpp::export]]
Rcpp::NumericVector cpp_selected_inversion_diagonal(
    Rcpp::NumericVector Q_x, Rcpp::IntegerVector Q_i, Rcpp::IntegerVector Q_p,
    int n
) {
    // n is a plain argument, unrelated to the three CSC vectors, and the copy
    // loops below read all of them unconditionally. Validate the triple
    // against n before anything is allocated or dereferenced.
    int nnz = (int)Q_x.size();
    if (n <= 0) {
        Rcpp::stop("cpp_selected_inversion_diagonal: n (%d) must be positive.", n);
    }
    if ((int)Q_p.size() != n + 1) {
        Rcpp::stop("cpp_selected_inversion_diagonal: length(Q_p) (%d) != n + 1 (%d).",
                   (int)Q_p.size(), n + 1);
    }
    if ((int)Q_i.size() != nnz) {
        Rcpp::stop("cpp_selected_inversion_diagonal: length(Q_i) (%d) != "
                   "length(Q_x) (%d).",
                   (int)Q_i.size(), nnz);
    }
    if (Q_p[0] != 0 || Q_p[n] != nnz) {
        Rcpp::stop("cpp_selected_inversion_diagonal: Q_p must run 0 .. nnz "
                   "(got Q_p[0] = %d, Q_p[n] = %d, nnz = %d).",
                   (int)Q_p[0], (int)Q_p[n], nnz);
    }
    for (int j = 0; j < n; j++) {
        if (Q_p[j + 1] < Q_p[j]) {
            Rcpp::stop("cpp_selected_inversion_diagonal: Q_p is not "
                       "non-decreasing at column %d (%d > %d).",
                       j + 1, (int)Q_p[j], (int)Q_p[j + 1]);
        }
    }
    for (int e = 0; e < nnz; e++) {
        if (Q_i[e] < 0 || Q_i[e] >= n) {
            Rcpp::stop("cpp_selected_inversion_diagonal: Q_i[%d] (%d) outside "
                       "[0, n) with n = %d.",
                       e + 1, (int)Q_i[e], n);
        }
    }

    tulpa::SparseCholeskySolver solver;

    // Build cholmod_sparse from CSC components
    cholmod_sparse* A = M_cholmod_allocate_sparse(
        n, n, nnz, 1, 1, -1, CHOLMOD_REAL, &solver.common()
    );
    if (!A) return Rcpp::NumericVector(n, NA_REAL);

    int* Ap = static_cast<int*>(A->p);
    int* Ai = static_cast<int*>(A->i);
    double* Ax = static_cast<double*>(A->x);
    for (int j = 0; j <= n; j++) Ap[j] = Q_p[j];
    for (int e = 0; e < nnz; e++) { Ai[e] = Q_i[e]; Ax[e] = Q_x[e]; }

    solver.analyze(A);
    if (!solver.factorize(A)) {
        M_cholmod_free_sparse(&A, &solver.common());
        return Rcpp::NumericVector(n, NA_REAL);
    }

    std::vector<double> diag_inv = solver.selected_inversion_diagonal();
    M_cholmod_free_sparse(&A, &solver.common());

    return Rcpp::wrap(diag_inv);
}
