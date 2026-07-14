// ============================================================================
// Sparse-Cholesky / log-det shims
//
// The handle is a tulpa::SparseCholeskySolver*. All operations route back
// through tulpa's DLL, which owns the Matrix-package CHOLMOD stub init.
// Each analyze/factorize call builds a stack-resident cholmod_sparse view
// over the caller's CSC arrays; nothing is copied or freed in the wrapper.
// ============================================================================

#include "shim_guard.h"

namespace {

inline cholmod_sparse make_cholmod_view(
    int n,
    const int* col_ptr,
    const int* row_idx,
    const double* values,
    int nnz
) {
    cholmod_sparse A;
    A.nrow = n;
    A.ncol = n;
    A.nzmax = nnz;
    A.p = const_cast<int*>(col_ptr);
    A.i = const_cast<int*>(row_idx);
    A.x = const_cast<double*>(values);
    A.z = nullptr;
    A.stype = -1;   // lower triangle stored
    A.itype = CHOLMOD_INT;
    A.xtype = CHOLMOD_REAL;
    A.dtype = CHOLMOD_DOUBLE;
    A.sorted = 1;
    A.packed = 1;
    return A;
}

} // namespace

extern "C" tulpa::sparse_chol_handle tulpa_sparse_chol_create_impl() {
    TULPA_SHIM_GUARD_BEGIN
    return reinterpret_cast<tulpa::sparse_chol_handle>(
        new tulpa::SparseCholeskySolver());
    TULPA_SHIM_GUARD_END("tulpa_sparse_chol_create")
    return nullptr;
}

extern "C" void tulpa_sparse_chol_destroy_impl(tulpa::sparse_chol_handle handle) {
    if (!handle) return;
    TULPA_SHIM_GUARD_BEGIN
    delete reinterpret_cast<tulpa::SparseCholeskySolver*>(handle);
    TULPA_SHIM_GUARD_END("tulpa_sparse_chol_destroy")
}

extern "C" int tulpa_sparse_chol_analyze_impl(
    tulpa::sparse_chol_handle handle,
    int n,
    const int* col_ptr,
    const int* row_idx,
    const double* values,
    int nnz
) {
    if (!handle) return 0;
    TULPA_SHIM_GUARD_BEGIN
    auto* solver = reinterpret_cast<tulpa::SparseCholeskySolver*>(handle);
    cholmod_sparse A = make_cholmod_view(n, col_ptr, row_idx, values, nnz);
    solver->analyze(&A);
    return solver->analyzed() ? 1 : 0;
    TULPA_SHIM_GUARD_END("tulpa_sparse_chol_analyze")
    return 0;
}

extern "C" int tulpa_sparse_chol_factorize_impl(
    tulpa::sparse_chol_handle handle,
    int n,
    const int* col_ptr,
    const int* row_idx,
    const double* values,
    int nnz
) {
    if (!handle) return 0;
    TULPA_SHIM_GUARD_BEGIN
    auto* solver = reinterpret_cast<tulpa::SparseCholeskySolver*>(handle);
    cholmod_sparse A = make_cholmod_view(n, col_ptr, row_idx, values, nnz);
    return solver->factorize(&A) ? 1 : 0;
    TULPA_SHIM_GUARD_END("tulpa_sparse_chol_factorize")
    return 0;
}

extern "C" void tulpa_sparse_chol_solve_impl(
    tulpa::sparse_chol_handle handle,
    const double* b,
    double* x,
    int n
) {
    if (!handle) {
        for (int i = 0; i < n; i++) x[i] = 0.0;
        return;
    }
    TULPA_SHIM_GUARD_BEGIN
    auto* solver = reinterpret_cast<tulpa::SparseCholeskySolver*>(handle);
    solver->solve(b, x, n);
    TULPA_SHIM_GUARD_END("tulpa_sparse_chol_solve")
}

extern "C" double tulpa_sparse_chol_log_det_impl(
    tulpa::sparse_chol_handle handle
) {
    if (!handle) return std::numeric_limits<double>::quiet_NaN();
    TULPA_SHIM_GUARD_BEGIN
    auto* solver = reinterpret_cast<tulpa::SparseCholeskySolver*>(handle);
    if (!solver->factored()) return std::numeric_limits<double>::quiet_NaN();
    return solver->log_determinant();
    TULPA_SHIM_GUARD_END("tulpa_sparse_chol_log_det")
    return std::numeric_limits<double>::quiet_NaN();
}

extern "C" int tulpa_sparse_chol_sel_inv_diag_impl(
    tulpa::sparse_chol_handle handle,
    double* diag_out,
    int n
) {
    if (!handle || !diag_out) return 0;
    TULPA_SHIM_GUARD_BEGIN
    auto* solver = reinterpret_cast<tulpa::SparseCholeskySolver*>(handle);
    std::vector<double> d = solver->selected_inversion_diagonal();
    if ((int)d.size() != n) return 0;
    for (int i = 0; i < n; i++) diag_out[i] = d[i];
    return 1;
    TULPA_SHIM_GUARD_END("tulpa_sparse_chol_sel_inv_diag")
    return 0;
}

// Pure free-function Takahashi: caller supplies L (lower-tri CSC) directly.
// No CHOLMOD factor / solver state. Z_out is column-major n*n, fully overwritten.
extern "C" int tulpa_takahashi_partial_inverse_dense_impl(
    int n,
    const int* L_col_ptr,
    const int* L_row_idx,
    const double* L_values,
    int /* L_nnz */,
    double* Z_out
) {
    if (n <= 0 || !L_col_ptr || !L_row_idx || !L_values || !Z_out) return 0;
    TULPA_SHIM_GUARD_BEGIN
    tulpa::takahashi_partial_inverse_dense(n, L_col_ptr, L_row_idx, L_values, Z_out);
    return 1;
    TULPA_SHIM_GUARD_END("tulpa_takahashi_partial_inverse_dense")
    return 0;
}

extern "C" double tulpa_stochastic_log_det_impl(
    int n,
    const int* col_ptr,
    const int* row_idx,
    const double* values,
    int nnz,
    int n_probes,
    int n_lanczos,
    unsigned int seed
) {
    TULPA_SHIM_GUARD_BEGIN
    std::vector<int>    cp(col_ptr, col_ptr + n + 1);
    std::vector<int>    ri(row_idx, row_idx + nnz);
    std::vector<double> vv(values,  values  + nnz);
    return tulpa::stochastic_log_determinant(
        cp, ri, vv, n,
        n_probes  > 0 ? n_probes  : 30,
        n_lanczos > 0 ? n_lanczos : 50,
        seed);
    TULPA_SHIM_GUARD_END("tulpa_stochastic_log_det")
    return std::numeric_limits<double>::quiet_NaN();
}
