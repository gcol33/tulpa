// spde_api.h
// Cross-package SPDE nested-Laplace API for model packages (tulpaGlmm,
// tulpaMesh, etc.). Wraps the cpp_nested_laplace_spde driver in
// src/spde_laplace.cpp.
//
// ABI v14 alignment: the SPDE backend now uses the same universal
// NestedLaplaceShimResult block (modes + optional per-grid Q) as the other
// nested-Laplace shims (ICAR, BYM2, NNGP, HSGP, ...), so downstream glue
// can reuse the same plumbing. The latent layout is
//   [beta (p)] [re (n_re_groups)] [w_mesh (n_mesh)]
// matching the v10-style formula-side iid-RE block used elsewhere.
//
// SPDE-specific inputs (sparse projection A, sparse precision base
// matrices C0 / G1, paired range / sigma hyperparameter grids, Matern
// smoothness nu, optional rational poles / weights) come through as
// regular function parameters.

#ifndef TULPA_SPDE_API_H
#define TULPA_SPDE_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"            // brings in TULPA_ABI_VERSION
#include "nuts_api.h"              // for check_abi_version()
#include "nested_laplace_api.h"    // NestedLaplaceShimResult

namespace tulpa {

// ----------------------------------------------------------------------------
// nested_laplace_spde: 2D grid over (range, σ), paired (no Cartesian product).
//
//   y         : [n_obs] response (binomial / poisson / ... per family).
//   n_trials  : [n_obs] trials per obs (binomial); 1 elsewhere.
//   X_flat    : [n_obs * p] column-major covariate matrix.
//   re_idx    : [n_obs] 1-based formula-side iid-RE group index per obs
//               (double-typed to match the rest of the laplace_mode_* API).
//   n_re_groups : iid-RE group count (0 disables the RE block).
//   sigma_re  : iid-RE standard deviation.
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
//   range_grid, sigma_grid : [n_grid] paired hyperparameter grid.
//   nu       : Matern smoothness (typically 1.0; α = nu + d/2 = 2 in 2D).
//
//   Optional rational-SPDE coefficients (poles + weights of length
//   n_rational); pass nullptr / 0 for the integer-α path.
//
//   store_Q  : if 1, the per-grid Q at the mode is retained in result_out
//              (NestedLaplaceShimResult::Q_*). Modes are always stored.
//
// Latent layout reported back through result_out (n_x = p + n_re_groups +
// n_mesh):
//   [beta (p)] [re (n_re_groups)] [w_mesh (n_mesh)]
// ----------------------------------------------------------------------------
typedef void (*NestedLaplaceSpdeFn)(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int n_obs, int p, int n_mesh,
    int n_re_groups, double sigma_re,
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
    int store_Q,
    NestedLaplaceShimResult* result_out
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
