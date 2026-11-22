// nested_laplace_api.h
// The result struct of the cross-package nested-Laplace shims (the SPDE grid
// kernel, spde_api.h): registered C-callable shims with raw POD inputs and a
// caller-allocated POD result, so the ABI is stable across separately
// compiled DLLs (Rcpp::List would not be). Hyperparameter posterior weights
// and moments are computed in R from log_marginal and the input grid.

#ifndef TULPA_NESTED_LAPLACE_API_H
#define TULPA_NESTED_LAPLACE_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"  // brings in TULPA_ABI_VERSION
#include "nuts_api.h"    // for check_abi_version()

namespace tulpa {

// ----------------------------------------------------------------------------
// Result struct shared by all nested-Laplace shims.
//
// log_marginal : [n_grid] inner Laplace log-marginal at each grid point.
// n_iter       : [n_grid] inner Newton iterations at each grid point.
// modes        : [n_grid * n_x] row-major; index (k, j) = modes[k*n_x + j].
//                Only populated when store_modes == 1; nullptr otherwise.
// store_modes  : 0 / 1 (output: indicates whether modes was populated).
//
// Per-grid Q at the mode (Tulpa ABI v5+):
// store_Q       : 0 / 1 (output: 1 iff the shim retained Q per grid point).
// Q_n           : == n_x when store_Q == 1, else 0.
// Q_grid_nnz    : [n_grid] nnz_k of Q_k's lower triangle (incl. diagonal).
// Q_p_offsets   : [n_grid + 1] prefix sum, length(Q_p_flat) == sum(Q_n+1)
//                 across grid points; here Q_n+1 is constant per grid so
//                 Q_p_offsets[k] = k * (Q_n + 1). Kept explicit for callers
//                 that want a single offset table to seek into Q_p_flat.
// Q_x_offsets   : [n_grid + 1] prefix sum into Q_i_flat / Q_x_flat;
//                 Q_x_offsets[k+1] - Q_x_offsets[k] == Q_grid_nnz[k].
// Q_p_flat      : column pointers concatenated across grid points,
//                 length n_grid * (Q_n + 1). Each block is a length
//                 (Q_n+1) CSC column-pointer array (lower-triangle).
// Q_i_flat      : row indices concatenated, length sum(Q_grid_nnz).
// Q_x_flat      : values concatenated, same length as Q_i_flat.
//
// Caller writes the input flag into a *separate* function parameter (see
// each shim signature). The struct's store_Q field is output-only.
//
// n_grid, n_x  : dimensions filled by the shim.
//
// Caller allocates the struct itself. The shim allocates the buffers; caller
// frees them via free_buffers().
// ----------------------------------------------------------------------------
struct NestedLaplaceShimResult {
    // Every member is default-initialized so free_buffers() is safe on an
    // instance the shim never populated; its null tests are what decide
    // whether a buffer is deleted.
    int n_grid      = 0;
    int n_x         = 0;
    int store_modes = 0;
    double* log_marginal = nullptr;  // [n_grid]
    int*    n_iter       = nullptr;  // [n_grid]
    double* modes        = nullptr;  // [n_grid * n_x] or nullptr

    int     store_Q     = 0;        // output: 1 iff Q was populated
    int     Q_n         = 0;        // output: == n_x when store_Q == 1, else 0
    int*    Q_grid_nnz  = nullptr;  // [n_grid] or nullptr
    int*    Q_p_offsets = nullptr;  // [n_grid + 1] or nullptr
    int*    Q_x_offsets = nullptr;  // [n_grid + 1] or nullptr
    int*    Q_p_flat    = nullptr;  // [n_grid * (Q_n + 1)] or nullptr
    int*    Q_i_flat    = nullptr;  // [sum Q_grid_nnz] or nullptr
    double* Q_x_flat    = nullptr;  // [sum Q_grid_nnz] or nullptr

    void free_buffers() {
        if (log_marginal) { delete[] log_marginal; log_marginal = nullptr; }
        if (n_iter)       { delete[] n_iter;       n_iter       = nullptr; }
        if (modes)        { delete[] modes;        modes        = nullptr; }
        if (Q_grid_nnz)   { delete[] Q_grid_nnz;   Q_grid_nnz   = nullptr; }
        if (Q_p_offsets)  { delete[] Q_p_offsets;  Q_p_offsets  = nullptr; }
        if (Q_x_offsets)  { delete[] Q_x_offsets;  Q_x_offsets  = nullptr; }
        if (Q_p_flat)     { delete[] Q_p_flat;     Q_p_flat     = nullptr; }
        if (Q_i_flat)     { delete[] Q_i_flat;     Q_i_flat     = nullptr; }
        if (Q_x_flat)     { delete[] Q_x_flat;     Q_x_flat     = nullptr; }
    }
};

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_API_H
