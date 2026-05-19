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

cholmod_sparse* SparseCholeskySolver::refill_from_dense(
    const DenseMat& H, int n, double drop_tol
) {
    if (!A_owned_) {
        // First call: discover pattern, allocate, fill values.
        size_t nnz = 0;
        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) {
                if (i == j || std::abs(H[i][j]) > drop_tol) nnz++;
            }
        }
        A_owned_ = M_cholmod_allocate_sparse(
            n, n, nnz, 1, 1, -1, CHOLMOD_REAL, &common_
        );
        if (!A_owned_) return nullptr;

        int* Ap = static_cast<int*>(A_owned_->p);
        int* Ai = static_cast<int*>(A_owned_->i);
        double* Ax = static_cast<double*>(A_owned_->x);

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
        return A_owned_;
    }

    // Subsequent calls: refill Ax in place from the cached (Ap, Ai).
    // No allocator traffic, no n^2 discovery scan.
    const int* Ap = static_cast<const int*>(A_owned_->p);
    const int* Ai = static_cast<const int*>(A_owned_->i);
    double* Ax    = static_cast<double*>(A_owned_->x);
    for (int j = 0; j < n; j++) {
        const int end = Ap[j + 1];
        for (int idx = Ap[j]; idx < end; idx++) {
            Ax[idx] = H[Ai[idx]][j];
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

void SparseCholeskySolver::solve(const double* b, double* x, int n) {
    if (!factored_ || !factor_) {
        for (int i = 0; i < n; i++) x[i] = 0.0;
        return;
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

    if (x_dense) {
        double* xp = static_cast<double*>(x_dense->x);
        for (int i = 0; i < n; i++) x[i] = xp[i];
        M_cholmod_free_dense(&x_dense, &common_);
    } else {
        for (int i = 0; i < n; i++) x[i] = 0.0;
    }
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

void takahashi_partial_inverse_csc(
    int n,
    const int* Lp,
    const int* Li,
    const double* Lx,
    double* Zx_out
) {
    int nnz = Lp[n];
    for (int idx = 0; idx < nnz; idx++) Zx_out[idx] = 0.0;

    for (int j = n - 1; j >= 0; j--) {
        int col_start = Lp[j];
        int col_end = Lp[j + 1];
        if (col_start >= col_end) continue;

        double Ljj = Lx[col_start];
        if (std::abs(Ljj) < 1e-15) continue;
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
}

void takahashi_partial_inverse_dense(
    int n,
    const int* Lp,
    const int* Li,
    const double* Lx,
    double* Z_out
) {
    int nnz = Lp[n];
    std::vector<double> Zx(nnz, 0.0);
    takahashi_partial_inverse_csc(n, Lp, Li, Lx, Zx.data());

    // Zero the dense buffer.
    for (int idx = 0; idx < n * n; idx++) Z_out[idx] = 0.0;

    // Scatter Zx onto the dense layout (column-major) and mirror to upper.
    for (int j = 0; j < n; j++) {
        for (int idx = Lp[j]; idx < Lp[j + 1]; idx++) {
            int i = Li[idx];
            double z = Zx[idx];
            Z_out[i + (size_t)j * n] = z;     // [i, j]
            Z_out[j + (size_t)i * n] = z;     // [j, i] (symmetric)
        }
    }
}

std::vector<double> SparseCholeskySolver::selected_inversion_diagonal() {
    if (!factored_ || !factor_) return {};

    int n = static_cast<int>(factor_->n);

    // Convert to simplicial LL' if supernodal (Takahashi needs element access)
    if (factor_->is_super) {
        M_cholmod_change_factor(
            CHOLMOD_REAL,  // xtype
            1,             // to_ll (LL')
            0,             // to_super = false (simplicial)
            1,             // to_packed
            1,             // to_monotonic
            factor_, &common_
        );
    }

    // Ensure LL' form (not LDL')
    if (!factor_->is_ll) {
        M_cholmod_change_factor(
            CHOLMOD_REAL, 1, 0, 1, 1,
            factor_, &common_
        );
    }

    int* Lp = static_cast<int*>(factor_->p);
    int* Li = static_cast<int*>(factor_->i);
    double* Lx = static_cast<double*>(factor_->x);
    int* Perm = static_cast<int*>(factor_->Perm);

    int nnz = Lp[n];
    std::vector<double> Zx(nnz, 0.0);
    takahashi_partial_inverse_csc(n, Lp, Li, Lx, Zx.data());

    // Extract diagonal of Z in permuted space, then unpermute.
    std::vector<double> diag_inv(n, 0.0);
    for (int j = 0; j < n; j++) {
        int col_start = Lp[j];
        double z_jj = Zx[col_start];
        int orig_j = Perm ? Perm[j] : j;
        diag_inv[orig_j] = z_jj;
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
    tulpa::SparseCholeskySolver solver;

    // Build cholmod_sparse from CSC components
    int nnz = Q_x.size();
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
