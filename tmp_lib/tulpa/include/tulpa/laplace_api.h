// laplace_api.h
// Cross-package Laplace API for model packages (tulpaObs, tulpaGlmm, etc.)
//
// Mirrors nuts_api.h: registered C-callable shims with raw POD inputs and
// caller-allocated POD result structs, so the ABI is stable across separately
// compiled DLLs (Rcpp::List / NumericVector / NumericMatrix would not be).
//
// Usage from a model package:
//   #include <tulpa/laplace_api.h>
//   tulpa::LaplaceShimResult res{};
//   tulpa::get_laplace_mode_dense_fn()(
//       y, n_trials, X_flat, re_idx, N, p, n_re_groups, sigma_re,
//       family, phi, max_iter, tol, n_threads, x_init, n_x_init,
//       &res);
//   // ... use res.mode[0..n_x-1] ...
//   res.free_buffers();

#ifndef TULPA_LAPLACE_API_H
#define TULPA_LAPLACE_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"  // brings in TULPA_ABI_VERSION
#include "nuts_api.h"    // for check_abi_version()

namespace tulpa {

// ----------------------------------------------------------------------------
// Result struct shared by all Laplace shims.
//   mode:   [n_x] caller must free via free_buffers
//   n_x is set by the shim. Other scalars are filled by the shim.
// ----------------------------------------------------------------------------
struct LaplaceShimResult {
    int n_x;
    double* mode;        // [n_x]
    double log_det_Q;
    double log_marginal;
    int n_iter;
    int converged;       // 0 or 1

    void free_buffers() {
        if (mode) { delete[] mode; mode = nullptr; }
    }
};

// ----------------------------------------------------------------------------
// laplace_mode_dense: GLMM with a single iid-RE block.
//
//   y         : [N] response (double; cast internally for binomial/poisson)
//   n_trials  : [N] trials per obs (binomial) — 1 for poisson/gaussian
//   X_flat    : [N * p] column-major design matrix (R convention)
//   re_idx    : [N] 0-based RE group index per obs (-1 for none); double-typed
//               to mirror the existing internal API which uses NumericVector
//   N, p      : dimensions
//   n_re_groups, sigma_re : RE structure
//   family    : null-terminated; "binomial", "poisson", "neg_binomial_2",
//               "gaussian", "gamma", "inverse_gaussian", or "<fam>_<link>"
//   phi       : dispersion (negbin / gaussian-sd / gamma-shape)
//   max_iter, tol, n_threads : Newton controls
//   x_init    : [n_x_init] initial value for x = c(beta, re); nullptr or len 0 ok
//   result_out: caller allocates the struct; shim allocates result_out->mode
// ----------------------------------------------------------------------------
typedef void (*LaplaceModeDenseFn)(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    const double* x_init,
    int n_x_init,
    LaplaceShimResult* result_out
);

inline LaplaceModeDenseFn get_laplace_mode_dense_fn() {
    static LaplaceModeDenseFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (LaplaceModeDenseFn)R_GetCCallable("tulpa", "tulpa_laplace_mode_dense");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// laplace_mode_spatial: dense GLMM + ICAR spatial component.
//
//   spatial_idx [N]   : 0-based spatial-unit index per obs
//   adj_row_ptr [n_spatial_units + 1]
//   adj_col_idx [adj_row_ptr[n_spatial_units]]
//   n_neighbors [n_spatial_units]
//   tau_spatial : ICAR precision
// All other args mirror laplace_mode_dense.
// ----------------------------------------------------------------------------
typedef void (*LaplaceModeSpatialFn)(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const int* spatial_idx,
    int n_spatial_units,
    const int* adj_row_ptr,
    const int* adj_col_idx,
    const int* n_neighbors,
    double tau_spatial,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    const double* x_init,
    int n_x_init,
    LaplaceShimResult* result_out
);

inline LaplaceModeSpatialFn get_laplace_mode_spatial_fn() {
    static LaplaceModeSpatialFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (LaplaceModeSpatialFn)R_GetCCallable("tulpa", "tulpa_laplace_mode_spatial");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// laplace_mode_dense_multi_re: GLMM with multiple RE terms, optional
// correlated random slopes, observation weights, and offset.
//
// Encoding of variable-size per-term arrays:
//
//   re_idx_flat   : [n_terms * N] term-major; term k occupies
//                   re_idx_flat[k*N .. k*N + N - 1] (1-based group indices).
//   re_ngroups    : [n_terms] number of groups per term
//   re_ncoefs     : [n_terms] coefficients per group per term (1 = intercept).
//                   nullptr → all 1s (intercept-only for every term).
//   sigma_offsets : [n_terms + 1] prefix offsets into sigma_flat.
//                   Term k occupies sigma_flat[sigma_offsets[k]..sigma_offsets[k+1]-1].
//                   Length per term: ck (uncorrelated → diagonal sds) OR
//                   ck*(ck+1)/2 (correlated → packed lower-tri Cholesky factor
//                   in column-major order: ck=2 → [L11, L21, L22]).
//   sigma_flat    : flat doubles per the offsets above.
//   Z_offsets     : [n_terms + 1] prefix offsets into Z_flat. nullptr → all
//                   terms intercept-only (Z_flat ignored).
//   Z_flat        : column-major obs-major matrices stacked per term;
//                   term k occupies N * re_ncoefs[k] doubles starting at
//                   Z_offsets[k]. Element (obs i, coef c) = Z_flat[off + c*N + i].
//
//   weights       : [N] observation weights or nullptr (defaults to 1.0).
//   offset_vec   : [N] linear-predictor offset or nullptr (defaults to 0.0).
//   x_init       : [n_x_init] initial mode (only used when length matches
//                   the internally computed n_x); nullptr / 0 ok.
// ----------------------------------------------------------------------------
typedef void (*LaplaceModeDenseMultiReFn)(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    int N, int p,
    int n_terms,
    const int* re_idx_flat,
    const int* re_ngroups,
    const int* re_ncoefs,
    const int* sigma_offsets,
    const double* sigma_flat,
    const int* Z_offsets,
    const double* Z_flat,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    const double* weights,
    const double* offset_vec,
    const double* x_init,
    int n_x_init,
    LaplaceShimResult* result_out
);

inline LaplaceModeDenseMultiReFn get_laplace_mode_dense_multi_re_fn() {
    static LaplaceModeDenseMultiReFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (LaplaceModeDenseMultiReFn)R_GetCCallable(
            "tulpa", "tulpa_laplace_mode_dense_multi_re");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// laplace_mode_bym2: BYM2 reparametrisation of ICAR + IID spatial.
//
// Latent-field layout (length p + n_re_groups + 2*n_spatial_units):
//   [beta (p)] [re (n_re_groups)] [phi_scaled (n_spatial_units)] [theta (n_spatial_units)]
// Linear predictor adds  sigma_spatial * (sqrt(rho) * scale_factor * phi + sqrt(1-rho) * theta)
// to obs i for spatial unit s = spatial_idx[i] - 1.
//
//   spatial_idx [N]                : 1-based spatial-unit index per obs
//                                     (matches Rcpp / cpp_laplace_fit_bym2 convention)
//   adj_row_ptr [n_spatial_units + 1]
//   adj_col_idx [adj_row_ptr[n_spatial_units]]
//   n_neighbors [n_spatial_units]
//   sigma_spatial : combined (phi + theta) standard deviation
//   rho           : mixing weight in [0,1]; sqrt(rho) * scaled-ICAR + sqrt(1-rho) * IID
//   scale_factor  : ICAR scaling constant (Riebler et al. 2016) so var(phi_scaled) ≈ 1
// All other args mirror laplace_mode_spatial. No x_init in this variant.
// ----------------------------------------------------------------------------
typedef void (*LaplaceModeBym2Fn)(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const int* spatial_idx,
    int n_spatial_units,
    const int* adj_row_ptr,
    const int* adj_col_idx,
    const int* n_neighbors,
    double sigma_spatial,
    double rho,
    double scale_factor,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    LaplaceShimResult* result_out
);

inline LaplaceModeBym2Fn get_laplace_mode_bym2_fn() {
    static LaplaceModeBym2Fn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (LaplaceModeBym2Fn)R_GetCCallable("tulpa", "tulpa_laplace_mode_bym2");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// laplace_mode_gp: continuous-spatial GP (NNGP) Laplace.
//
// Latent layout: [beta (p)] [re (n_re_groups)] [w (n_spatial)]
// The first n_spatial observations carry the GP effect (eta_i += w_i for
// i < n_spatial); subsequent obs do not. This mirrors the underlying
// tulpa::laplace_mode_gp convention.
//
//   coords_flat   : [n_spatial * coord_dim] column-major (R convention).
//                   Currently only the first 2 columns are used; pass
//                   coord_dim = 2 unless / until the underlying NNGP kernel
//                   is generalised.
//   coord_dim     : columns in coords (≥ 2).
//   nn_idx_flat   : [n_spatial * nn] column-major; 1-based neighbour indices
//                   (0 = "no neighbour"). Element (loc i, slot j) =
//                   nn_idx_flat[j*n_spatial + i].
//   nn_dist_flat  : [n_spatial * nn] column-major; same layout as nn_idx_flat.
//   nn_order      : [n_spatial] permutation mapping NNGP order → original
//                   location index.
//   sigma2_gp, phi_gp : NNGP marginal variance and range parameter.
//   cov_type      : 0 = exponential, 1 = Matern 3/2, 2 = Matern 5/2.
// ----------------------------------------------------------------------------
typedef void (*LaplaceModeGpFn)(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const double* coords_flat,
    int coord_dim,
    const int* nn_idx_flat,
    const double* nn_dist_flat,
    const int* nn_order,
    int n_spatial, int nn,
    double sigma2_gp, double phi_gp, int cov_type,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    LaplaceShimResult* result_out
);

inline LaplaceModeGpFn get_laplace_mode_gp_fn() {
    static LaplaceModeGpFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (LaplaceModeGpFn)R_GetCCallable("tulpa", "tulpa_laplace_mode_gp");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// laplace_mode_multiscale_gp: two NNGP scales (local + regional) added to η.
//
// Latent layout: [beta (p)] [re (n_re_groups)] [w_local (n_spatial)]
//                [w_regional (n_spatial)]. eta_i += w_local_i + w_regional_i
// for i < n_spatial.
//
// Both nn_idx_*/nn_dist_* matrices share `coords_flat` and follow the same
// column-major convention as in laplace_mode_gp.
// ----------------------------------------------------------------------------
typedef void (*LaplaceModeMultiscaleGpFn)(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const double* coords_flat,
    int coord_dim,
    const int* nn_idx_local_flat,
    const double* nn_dist_local_flat,
    const int* nn_order_local,
    int nn_local,
    const int* nn_idx_regional_flat,
    const double* nn_dist_regional_flat,
    const int* nn_order_regional,
    int nn_regional,
    int n_spatial,
    double sigma2_local, double phi_local,
    double sigma2_regional, double phi_regional,
    int cov_type,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    LaplaceShimResult* result_out
);

inline LaplaceModeMultiscaleGpFn get_laplace_mode_multiscale_gp_fn() {
    static LaplaceModeMultiscaleGpFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (LaplaceModeMultiscaleGpFn)R_GetCCallable(
            "tulpa", "tulpa_laplace_mode_multiscale_gp");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// laplace_mode_multiscale_temporal: trend (RW1/RW2) + seasonal (RW1) +
// short-range (AR1 / iid) latent temporal blocks.
//
//   time_idx [N]     : 1-based time index per obs.
//   trend_type       : 0 = none, 1 = RW1, 2 = RW2.
//   seasonal_period  : 0 = none, otherwise length of the seasonal block.
//   short_type       : 0 = none, 1 = AR1, 2 = iid.
// ----------------------------------------------------------------------------
typedef void (*LaplaceModeMultiscaleTemporalFn)(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const int* time_idx,
    int n_times,
    int seasonal_period,
    int trend_type,
    int short_type,
    double sigma2_trend,
    double sigma2_seasonal,
    double sigma2_short,
    double rho_short,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    LaplaceShimResult* result_out
);

inline LaplaceModeMultiscaleTemporalFn get_laplace_mode_multiscale_temporal_fn() {
    static LaplaceModeMultiscaleTemporalFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (LaplaceModeMultiscaleTemporalFn)R_GetCCallable(
            "tulpa", "tulpa_laplace_mode_multiscale_temporal");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// laplace_mode_rsr: Restricted Spatial Regression. Adds a P_perp-projected
// spatial random effect to η, with ICAR-style adjacency precision tau_spatial.
//
//   spatial_idx [N]                : 1-based spatial-unit index per obs.
//   adj_row_ptr / adj_col_idx      : ICAR adjacency in CSR form.
//   rsr_projection_flat            : [rsr_n * rsr_n] orthogonal projection
//                                     P_perp (column-major). rsr_n must
//                                     equal n_spatial_units.
// ----------------------------------------------------------------------------
typedef void (*LaplaceModeRsrFn)(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const int* spatial_idx,
    int n_spatial_units,
    const int* adj_row_ptr,
    const int* adj_col_idx,
    const int* n_neighbors,
    double tau_spatial,
    const double* rsr_projection_flat,
    int rsr_n,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    LaplaceShimResult* result_out
);

inline LaplaceModeRsrFn get_laplace_mode_rsr_fn() {
    static LaplaceModeRsrFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (LaplaceModeRsrFn)R_GetCCallable("tulpa", "tulpa_laplace_mode_rsr");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// laplace_newton_solve / _sparse are not exposed here: both are templated on
// lambdas (compute_eta / scatter / center / log_prior) and cannot pass through
// R_RegisterCCallable. Downstream packages should call the high-level
// laplace_mode_* shims instead, or build a fresh entry that hides the lambdas.
// ----------------------------------------------------------------------------

} // namespace tulpa

#endif // TULPA_LAPLACE_API_H
