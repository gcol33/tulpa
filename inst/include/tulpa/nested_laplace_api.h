// nested_laplace_api.h
// Cross-package nested-Laplace API for model packages (tulpaGlmm, tulpaOcc).
//
// Mirrors laplace_api.h: registered C-callable shims with raw POD inputs and
// caller-allocated POD result structs, so the ABI is stable across separately
// compiled DLLs (Rcpp::List would not be).
//
// Each shim wraps the matching cpp_nested_laplace_<backend> entry point in
// src/nested_laplace.cpp. The universal output block (log_marginal[n_grid],
// n_iter[n_grid], optional modes[n_grid x n_x]) is filled into a
// NestedLaplaceShimResult; backend-specific grid vectors (tau_grid,
// sigma_grid, rho_grid, ...) are passed in by the caller, who already has
// them, and are not echoed back through the shim.
//
// Hyperparameter posterior weights / theta_mean / theta_sd are computed in
// R from log_marginal + the input grid (see R/nested_laplace.R). Keeping
// that work R-side lets the shim stay model-agnostic.

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
// store_modes  : 0 / 1.
// n_grid, n_x  : dimensions filled by the shim.
//
// Caller allocates the struct itself. The shim allocates the buffers; caller
// frees them via free_buffers().
// ----------------------------------------------------------------------------
struct NestedLaplaceShimResult {
    int n_grid;
    int n_x;
    int store_modes;
    double* log_marginal;  // [n_grid]
    int*    n_iter;        // [n_grid]
    double* modes;         // [n_grid * n_x] or nullptr

    void free_buffers() {
        if (log_marginal) { delete[] log_marginal; log_marginal = nullptr; }
        if (n_iter)       { delete[] n_iter;       n_iter       = nullptr; }
        if (modes)        { delete[] modes;        modes        = nullptr; }
    }
};

// Common arg block (re-used in the per-backend descriptions below):
//   y              : [N] response (double; cast internally per family).
//   n_trials       : [N] trials per obs (binomial); 1 elsewhere.
//   X_flat         : [N * p] column-major design matrix.
//   re_idx         : [N] 0/1-based RE group index per obs (mirrors the
//                    underlying laplace_mode_* convention; double-typed).
//   N, p           : dimensions.
//   n_re_groups    : iid-RE group count (0 disables the RE block).
//   sigma_re       : iid-RE standard deviation.
//   family         : "binomial", "poisson", "neg_binomial_2", "gaussian", ...
//                    See laplace_api.h for the full list.
//   phi            : dispersion (negbin / gaussian-sd / gamma-shape).
//   max_iter, tol  : inner Newton controls (per grid point).
//   n_threads      : OpenMP threads.
//   x_init / n_x_init : warm-start mode for grid point 0; nullptr / 0 ok.

// ----------------------------------------------------------------------------
// ICAR: 1D grid over τ (precision).
// Latent: [beta (p)] [re (n_re_groups)] [w_spatial (n_spatial_units)].
// store_modes = 1.
// ----------------------------------------------------------------------------
typedef void (*NestedLaplaceIcarFn)(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* spatial_idx, int n_spatial_units,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    const double* tau_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    NestedLaplaceShimResult* result_out
);

inline NestedLaplaceIcarFn get_nested_laplace_icar_fn() {
    static NestedLaplaceIcarFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (NestedLaplaceIcarFn)R_GetCCallable("tulpa", "tulpa_nested_laplace_icar");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// BYM2: 2D grid over (σ_spatial, ρ).
// Latent: [beta] [re] [phi (n_spatial)] [theta (n_spatial)] (length 2*n_spatial).
// store_modes = 0.
// scale_factor is the Riebler et al. (2016) ICAR scaling constant.
// ----------------------------------------------------------------------------
typedef void (*NestedLaplaceBym2Fn)(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* spatial_idx, int n_spatial_units,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    double scale_factor,
    const double* sigma_spatial_grid, const double* rho_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    NestedLaplaceShimResult* result_out
);

inline NestedLaplaceBym2Fn get_nested_laplace_bym2_fn() {
    static NestedLaplaceBym2Fn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (NestedLaplaceBym2Fn)R_GetCCallable("tulpa", "tulpa_nested_laplace_bym2");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// Proper CAR: 2D grid over (τ, ρ).
// Latent: [beta] [re] [w_spatial (n_spatial)]. store_modes = 1.
// ----------------------------------------------------------------------------
typedef void (*NestedLaplaceCarProperFn)(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* spatial_idx, int n_spatial_units,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    const double* tau_grid, const double* rho_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    NestedLaplaceShimResult* result_out
);

inline NestedLaplaceCarProperFn get_nested_laplace_car_proper_fn() {
    static NestedLaplaceCarProperFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (NestedLaplaceCarProperFn)R_GetCCallable(
            "tulpa", "tulpa_nested_laplace_car_proper");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// RW1: 1D grid over τ (precision). Latent [beta] [re] [w_temporal (n_times)].
// cyclic = 1 closes the chain (RW1 on a circular index). store_modes = 1.
// ----------------------------------------------------------------------------
typedef void (*NestedLaplaceRw1Fn)(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* temporal_idx, int n_times, int cyclic,
    const double* tau_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    NestedLaplaceShimResult* result_out
);

inline NestedLaplaceRw1Fn get_nested_laplace_rw1_fn() {
    static NestedLaplaceRw1Fn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (NestedLaplaceRw1Fn)R_GetCCallable("tulpa", "tulpa_nested_laplace_rw1");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// RW2: 1D grid over τ. Same latent layout as RW1; no cyclic flag.
// store_modes = 1.
// ----------------------------------------------------------------------------
typedef void (*NestedLaplaceRw2Fn)(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* temporal_idx, int n_times,
    const double* tau_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    NestedLaplaceShimResult* result_out
);

inline NestedLaplaceRw2Fn get_nested_laplace_rw2_fn() {
    static NestedLaplaceRw2Fn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (NestedLaplaceRw2Fn)R_GetCCallable("tulpa", "tulpa_nested_laplace_rw2");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// AR1: 2D grid over (τ, ρ). store_modes = 1.
// ----------------------------------------------------------------------------
typedef void (*NestedLaplaceAr1Fn)(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* temporal_idx, int n_times,
    const double* tau_grid, const double* rho_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    NestedLaplaceShimResult* result_out
);

inline NestedLaplaceAr1Fn get_nested_laplace_ar1_fn() {
    static NestedLaplaceAr1Fn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (NestedLaplaceAr1Fn)R_GetCCallable("tulpa", "tulpa_nested_laplace_ar1");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// NNGP: 2D grid over (σ², φ_gp). Continuous-spatial GP via nearest-neighbor
// conditional decomposition.
// Latent: [beta] [re] [w (n_spatial)]. store_modes = 0.
//
// coords_flat   : [n_spatial * coord_dim] column-major. Currently only the
//                 first 2 columns are read; pass coord_dim = 2.
// nn_idx_flat   : [n_spatial * nn] column-major; 1-based neighbor indices
//                 (0 sentinels for "no neighbor").
// nn_dist_flat  : [n_spatial * nn] column-major; same layout as nn_idx_flat.
// nn_order      : [n_spatial] permutation NNGP order → original index.
// cov_type      : 0 = exponential, 1 = Matern 3/2, 2 = Matern 5/2.
// ----------------------------------------------------------------------------
typedef void (*NestedLaplaceNngpFn)(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const double* coords_flat, int coord_dim,
    const int* nn_idx_flat, const double* nn_dist_flat,
    const int* nn_order,
    int n_spatial, int nn,
    const double* sigma2_grid, const double* phi_gp_grid, int n_grid,
    int cov_type,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    NestedLaplaceShimResult* result_out
);

inline NestedLaplaceNngpFn get_nested_laplace_nngp_fn() {
    static NestedLaplaceNngpFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (NestedLaplaceNngpFn)R_GetCCallable("tulpa", "tulpa_nested_laplace_nngp");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// HSGP: 2D grid over (σ², ℓ). Hilbert-space GP with M precomputed basis
// functions and eigenvalues.
// Latent: [beta] [re] [beta_M (n_basis)]. store_modes = 0.
//
// phi_basis_flat : [N * n_basis] column-major basis matrix Φ.
// lambda_eig     : [n_basis] eigenvalues λ_j (used to evaluate the
//                  spectral density per grid point).
// ----------------------------------------------------------------------------
typedef void (*NestedLaplaceHsgpFn)(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const double* phi_basis_flat, int n_basis,
    const double* lambda_eig,
    const double* sigma2_grid, const double* lengthscale_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    NestedLaplaceShimResult* result_out
);

inline NestedLaplaceHsgpFn get_nested_laplace_hsgp_fn() {
    static NestedLaplaceHsgpFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (NestedLaplaceHsgpFn)R_GetCCallable("tulpa", "tulpa_nested_laplace_hsgp");
    }
    return fn;
}

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_API_H
