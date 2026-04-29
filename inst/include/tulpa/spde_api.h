// spde_api.h
// Cross-package SPDE nested-Laplace API for model packages (tulpaGlmm,
// tulpaMesh, etc.). Wraps the cpp_nested_laplace_spde driver in
// src/spde_laplace.cpp.
//
// The SPDE backend is exposed separately from the other nested-Laplace
// backends because:
//   - it takes a sparse projection A and sparse precision base matrices
//     (C0, G1) instead of an areal/temporal index;
//   - the latent layout is [beta (p)] [w (n_mesh)] (no separate iid-RE
//     block — RE handling is left to the caller via X);
//   - it does not store per-grid-point modes.

#ifndef TULPA_SPDE_API_H
#define TULPA_SPDE_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"  // brings in TULPA_ABI_VERSION
#include "nuts_api.h"    // for check_abi_version()

namespace tulpa {

// ----------------------------------------------------------------------------
// Result struct for the SPDE driver. modes are not stored — only the
// log-marginal at each grid point and the per-grid-point Newton iteration
// counts. Q_nnz is reported once (sparsity is fixed across the grid).
// ----------------------------------------------------------------------------
struct SpdeNestedLaplaceShimResult {
    int n_grid;
    int Q_nnz;
    double* log_marginal;  // [n_grid]
    int*    n_iter;        // [n_grid]

    void free_buffers() {
        if (log_marginal) { delete[] log_marginal; log_marginal = nullptr; }
        if (n_iter)       { delete[] n_iter;       n_iter       = nullptr; }
    }
};

// ----------------------------------------------------------------------------
// nested_laplace_spde: 2D grid over (range, σ).
//
//   y         : [n_obs] response (binomial / poisson / ... per family).
//   n_trials  : [n_obs] trials per obs (binomial); 1 elsewhere.
//   X_flat    : [n_obs * p] column-major covariate matrix (no iid-RE block).
//
//   Sparse projection A (n_obs × n_mesh), CSC format:
//     A_x  : [A_nnz] non-zero values
//     A_i  : [A_nnz] row indices
//     A_p  : [n_mesh + 1] column pointers
//   A_nnz = A_p[n_mesh].
//
//   SPDE base matrices:
//     C0_diag : [n_mesh] mass-matrix diagonal (C0 is diagonal by lumping)
//     G1_x / G1_i / G1_p : [G1_nnz] / [G1_nnz] / [n_mesh + 1] CSC stiffness.
//   G1_nnz = G1_p[n_mesh].
//
//   range_grid, sigma_grid : [n_grid] hyperparameter grid.
//   nu       : Matern smoothness (typically 1.0; α = nu + d/2 = 2 in 2D).
//
//   Optional rational-SPDE coefficients (poles + weights of length
//   n_rational); pass nullptr / 0 for the integer-α path.
// ----------------------------------------------------------------------------
typedef void (*NestedLaplaceSpdeFn)(
    const double* y, const int* n_trials,
    const double* X_flat,
    int n_obs, int p, int n_mesh,
    const double* A_x, const int* A_i, const int* A_p,
    const double* C0_diag,
    const double* G1_x, const int* G1_i, const int* G1_p,
    const double* range_grid, const double* sigma_grid, int n_grid,
    double nu,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    const double* rational_poles, int n_rational,
    const double* rational_weights,
    SpdeNestedLaplaceShimResult* result_out
);

inline NestedLaplaceSpdeFn get_nested_laplace_spde_fn() {
    static NestedLaplaceSpdeFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (NestedLaplaceSpdeFn)R_GetCCallable("tulpa", "tulpa_nested_laplace_spde");
    }
    return fn;
}

} // namespace tulpa

#endif // TULPA_SPDE_API_H
