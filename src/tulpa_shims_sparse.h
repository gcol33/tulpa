// ============================================================================
// Sparse-Cholesky / log-det shims
//
// The handle is a tulpa::SparseCholeskySolver*. All operations route back
// through tulpa's DLL, which owns the Matrix-package CHOLMOD stub init.
// Each analyze/factorize call builds a stack-resident cholmod_sparse view
// over the caller's CSC arrays; nothing is copied or freed in the wrapper.
// ============================================================================

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
    return reinterpret_cast<tulpa::sparse_chol_handle>(
        new tulpa::SparseCholeskySolver());
}

extern "C" void tulpa_sparse_chol_destroy_impl(tulpa::sparse_chol_handle handle) {
    if (!handle) return;
    delete reinterpret_cast<tulpa::SparseCholeskySolver*>(handle);
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
    auto* solver = reinterpret_cast<tulpa::SparseCholeskySolver*>(handle);
    cholmod_sparse A = make_cholmod_view(n, col_ptr, row_idx, values, nnz);
    solver->analyze(&A);
    return solver->analyzed() ? 1 : 0;
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
    auto* solver = reinterpret_cast<tulpa::SparseCholeskySolver*>(handle);
    cholmod_sparse A = make_cholmod_view(n, col_ptr, row_idx, values, nnz);
    return solver->factorize(&A) ? 1 : 0;
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
    auto* solver = reinterpret_cast<tulpa::SparseCholeskySolver*>(handle);
    solver->solve(b, x, n);
}

extern "C" double tulpa_sparse_chol_log_det_impl(
    tulpa::sparse_chol_handle handle
) {
    if (!handle) return std::numeric_limits<double>::quiet_NaN();
    auto* solver = reinterpret_cast<tulpa::SparseCholeskySolver*>(handle);
    if (!solver->factored()) return std::numeric_limits<double>::quiet_NaN();
    return solver->log_determinant();
}

extern "C" int tulpa_sparse_chol_sel_inv_diag_impl(
    tulpa::sparse_chol_handle handle,
    double* diag_out,
    int n
) {
    if (!handle || !diag_out) return 0;
    auto* solver = reinterpret_cast<tulpa::SparseCholeskySolver*>(handle);
    std::vector<double> d = solver->selected_inversion_diagonal();
    if ((int)d.size() != n) return 0;
    for (int i = 0; i < n; i++) diag_out[i] = d[i];
    return 1;
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
    std::vector<int>    cp(col_ptr, col_ptr + n + 1);
    std::vector<int>    ri(row_idx, row_idx + nnz);
    std::vector<double> vv(values,  values  + nnz);
    return tulpa::stochastic_log_determinant(
        cp, ri, vv, n,
        n_probes  > 0 ? n_probes  : 30,
        n_lanczos > 0 ? n_lanczos : 50,
        seed);
}
