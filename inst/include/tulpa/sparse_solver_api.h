// sparse_solver_api.h
// Cross-package sparse-Cholesky / log-determinant API for tulpa downstreams.
//
// Wraps tulpa's CHOLMOD-backed SparseCholeskySolver and stochastic Lanczos
// log-det routines behind opaque handles, so model packages do not need to
// link CHOLMOD directly. All operations route through tulpa's DLL, which
// owns the Matrix-package CHOLMOD stub initialisation.
//
// Sparsity convention: lower-triangle CSC of a symmetric matrix.
//   col_ptr [n + 1] — column pointers
//   row_idx [nnz]   — row indices (each entry has row >= col)
//   values  [nnz]   — numeric values
//   stype = -1, sorted = 1, packed = 1 (always)
//
// Lifecycle:
//   handle = create();
//   analyze(handle, A);                  // once per sparsity pattern
//   for (...) {
//       factorize(handle, A);            // each new value pass
//       solve(handle, b, x, n);
//       ld = log_determinant(handle);
//   }
//   destroy(handle);

#ifndef TULPA_SPARSE_SOLVER_API_H
#define TULPA_SPARSE_SOLVER_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"  // brings in TULPA_ABI_VERSION
#include "nuts_api.h"    // for check_abi_version()

namespace tulpa {

// Opaque handle. Pointer is owned by tulpa; do not deref.
typedef void* sparse_chol_handle;

// ----------------------------------------------------------------------------
// Lifecycle
// ----------------------------------------------------------------------------
typedef sparse_chol_handle (*SparseCholCreateFn)();
typedef void (*SparseCholDestroyFn)(sparse_chol_handle);

inline SparseCholCreateFn get_sparse_chol_create_fn() {
    static SparseCholCreateFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (SparseCholCreateFn)R_GetCCallable("tulpa", "tulpa_sparse_chol_create");
    }
    return fn;
}

inline SparseCholDestroyFn get_sparse_chol_destroy_fn() {
    static SparseCholDestroyFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (SparseCholDestroyFn)R_GetCCallable("tulpa", "tulpa_sparse_chol_destroy");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// Symbolic analysis. Determines fill-reducing ordering and factor allocation.
// Must be called before factorize(). values may be nullptr; only structure is
// inspected. Returns 1 on success, 0 on failure.
// ----------------------------------------------------------------------------
typedef int (*SparseCholAnalyzeFn)(
    sparse_chol_handle handle,
    int n,
    const int* col_ptr,
    const int* row_idx,
    const double* values,
    int nnz
);

inline SparseCholAnalyzeFn get_sparse_chol_analyze_fn() {
    static SparseCholAnalyzeFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (SparseCholAnalyzeFn)R_GetCCallable("tulpa", "tulpa_sparse_chol_analyze");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// Numeric factorization. Sparsity pattern must match the analyze() call.
// Returns 1 on success, 0 on failure (e.g. not positive definite).
// ----------------------------------------------------------------------------
typedef int (*SparseCholFactorizeFn)(
    sparse_chol_handle handle,
    int n,
    const int* col_ptr,
    const int* row_idx,
    const double* values,
    int nnz
);

inline SparseCholFactorizeFn get_sparse_chol_factorize_fn() {
    static SparseCholFactorizeFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (SparseCholFactorizeFn)R_GetCCallable("tulpa", "tulpa_sparse_chol_factorize");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// Solve A x = b using the current factorization. b and x have length n.
// x may alias b. If the factor is invalid, fills x with zeros.
// ----------------------------------------------------------------------------
typedef void (*SparseCholSolveFn)(
    sparse_chol_handle handle,
    const double* b,
    double* x,
    int n
);

inline SparseCholSolveFn get_sparse_chol_solve_fn() {
    static SparseCholSolveFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (SparseCholSolveFn)R_GetCCallable("tulpa", "tulpa_sparse_chol_solve");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// log|A| from the current factorization. Returns NaN if not factored.
// ----------------------------------------------------------------------------
typedef double (*SparseCholLogDetFn)(sparse_chol_handle handle);

inline SparseCholLogDetFn get_sparse_chol_log_det_fn() {
    static SparseCholLogDetFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (SparseCholLogDetFn)R_GetCCallable("tulpa", "tulpa_sparse_chol_log_det");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// Selected inversion: diagonal of A^{-1} via Takahashi equations from the
// current factor. Caller provides diag_out [n]. Returns 1 on success, 0 on
// failure. May convert the internal factor from supernodal to simplicial.
// ----------------------------------------------------------------------------
typedef int (*SparseCholSelInvDiagFn)(
    sparse_chol_handle handle,
    double* diag_out,
    int n
);

inline SparseCholSelInvDiagFn get_sparse_chol_sel_inv_diag_fn() {
    static SparseCholSelInvDiagFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (SparseCholSelInvDiagFn)R_GetCCallable(
            "tulpa", "tulpa_sparse_chol_sel_inv_diag");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// Pure-function Takahashi partial inverse (no factorization required, no
// handle). Caller supplies L = lower-triangular CSC of an LL^T factorisation
// of Q. Z_out is filled column-major as a dense n*n with Q^{-1} on
// pattern(L + L^T) and zeros elsewhere. Returns 1 on success, 0 on bad args.
// ----------------------------------------------------------------------------
typedef int (*TakahashiPartialInverseDenseFn)(
    int n,
    const int* L_col_ptr,
    const int* L_row_idx,
    const double* L_values,
    int L_nnz,
    double* Z_out
);

inline TakahashiPartialInverseDenseFn get_takahashi_partial_inverse_dense_fn() {
    static TakahashiPartialInverseDenseFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (TakahashiPartialInverseDenseFn)R_GetCCallable(
            "tulpa", "tulpa_takahashi_partial_inverse_dense");
        if (!fn) {
            Rf_error(
                "tulpa lacks tulpa_takahashi_partial_inverse_dense — "
                "rebuild tulpa from current source.");
        }
    }
    return fn;
}

// ----------------------------------------------------------------------------
// Stochastic Lanczos Quadrature for log|A| (no factorization required).
// Cost is O(n_probes * n_lanczos * nnz). Use when n is too large for
// Cholesky (e.g. > 1e5). seed = 42 for reproducibility.
// ----------------------------------------------------------------------------
typedef double (*StochasticLogDetFn)(
    int n,
    const int* col_ptr,
    const int* row_idx,
    const double* values,
    int nnz,
    int n_probes,
    int n_lanczos,
    unsigned int seed
);

inline StochasticLogDetFn get_stochastic_log_det_fn() {
    static StochasticLogDetFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (StochasticLogDetFn)R_GetCCallable(
            "tulpa", "tulpa_stochastic_log_det");
    }
    return fn;
}

} // namespace tulpa

#endif // TULPA_SPARSE_SOLVER_API_H
