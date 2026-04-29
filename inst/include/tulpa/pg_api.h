// pg_api.h
// Cross-package Pólya-Gamma Gibbs samplers for binomial / negbin GLMMs.
//
// Salvage logs (numdenom benchmarks) put PG-Gibbs at 1.9-3.7x faster than
// HMC on these likelihoods, so the model packages need a stable hook.

#ifndef TULPA_PG_API_H
#define TULPA_PG_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"
#include "nuts_api.h"  // check_abi_version

namespace tulpa {

// All PG samplers fill this same flat result. Caller frees buffers.
//   beta:     [n_save * n_beta] row-major
//   re:       [n_save * n_re]   row-major (n_re = n_groups for IID; 0 if absent)
//   sigma_re: [n_save]
//   eta:      [n_save * n_obs]  row-major if store_eta != 0; nullptr otherwise
//   r_disp:   [n_save] negbin dispersion samples; nullptr for binomial
//   spatial:  [n_save * n_spatial_units] row-major; nullptr if absent
//   tau_spatial: [n_save]; nullptr if no spatial component
struct PGShimResult {
    int n_save;
    int n_beta;
    int n_re;
    int n_obs;
    int n_spatial_units;
    double* beta;
    double* re;
    double* sigma_re;
    double* eta;
    double* r_disp;
    double* spatial;
    double* tau_spatial;

    void free_buffers() {
        if (beta)        { delete[] beta;        beta = nullptr; }
        if (re)          { delete[] re;          re = nullptr; }
        if (sigma_re)    { delete[] sigma_re;    sigma_re = nullptr; }
        if (eta)         { delete[] eta;         eta = nullptr; }
        if (r_disp)      { delete[] r_disp;      r_disp = nullptr; }
        if (spatial)     { delete[] spatial;     spatial = nullptr; }
        if (tau_spatial) { delete[] tau_spatial; tau_spatial = nullptr; }
    }
};

// ----------------------------------------------------------------------------
// pg_binomial_gibbs: PG-augmented Gibbs for logistic GLMM with iid RE.
//   y, n_trials [N_obs]     : binomial counts and trials
//   X_flat      [N_obs * p] : column-major design matrix (R convention)
//   group       [N_obs]     : 1-based group index (matches existing internal API)
//   n_groups
//   n_iter, n_warmup, thin
//   prior_beta_sd, prior_sigma_scale (half-Cauchy)
//   store_eta, verbose      : 0/1 flags
// ----------------------------------------------------------------------------
typedef void (*PGBinomialFn)(
    const int* y,
    const int* n_trials,
    const double* X_flat,
    int N_obs, int p,
    const int* group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    int store_eta,
    int verbose,
    int n_threads,
    PGShimResult* result_out
);

inline PGBinomialFn get_pg_binomial_fn() {
    static PGBinomialFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (PGBinomialFn)R_GetCCallable("tulpa", "tulpa_pg_binomial_gibbs");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// pg_negbin_gibbs: PG+CRT Gibbs for NB GLMM with iid RE.
// Adds three more priors and an init for the dispersion r.
// ----------------------------------------------------------------------------
typedef void (*PGNegbinFn)(
    const int* y,
    const double* X_flat,
    int N_obs, int p,
    const int* group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    int store_eta,
    int verbose,
    int n_threads,
    PGShimResult* result_out
);

inline PGNegbinFn get_pg_negbin_fn() {
    static PGNegbinFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (PGNegbinFn)R_GetCCallable("tulpa", "tulpa_pg_negbin_gibbs");
    }
    return fn;
}

// ----------------------------------------------------------------------------
// pg_negbin_spatial_gibbs: NB GLMM with iid RE + ICAR spatial.
//   spatial_group       [N_obs] : 1-based spatial-unit index
//   adj_row_ptr [n_spatial_units + 1], adj_col_idx [...] : CSR adjacency
//   n_neighbors [n_spatial_units]
// ----------------------------------------------------------------------------
typedef void (*PGNegbinSpatialFn)(
    const int* y,
    const double* X_flat,
    int N_obs, int p,
    const int* re_group,
    int n_re_groups,
    const int* spatial_group,
    int n_spatial_units,
    const int* adj_row_ptr,
    const int* adj_col_idx,
    const int* n_neighbors,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_re_scale,
    double prior_tau_shape,
    double prior_tau_rate,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    int store_eta,
    int verbose,
    int n_threads,
    PGShimResult* result_out
);

inline PGNegbinSpatialFn get_pg_negbin_spatial_fn() {
    static PGNegbinSpatialFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (PGNegbinSpatialFn)R_GetCCallable("tulpa", "tulpa_pg_negbin_spatial_gibbs");
    }
    return fn;
}

} // namespace tulpa

#endif // TULPA_PG_API_H
