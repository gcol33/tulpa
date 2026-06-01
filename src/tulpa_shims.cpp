// tulpa_shims.cpp
// C-callable shims for inference drivers (Laplace, PG, VI, ESS).
// Pattern mirrors src/tulpa_init.cpp: raw pointers in, POD result struct out,
// caller owns the result buffers via free_buffers().
//
// Registration is wired from tulpa_init.cpp::tulpa_register_callables() so the
// existing R_init_tulpa hook picks these up at DLL load.

#include <Rcpp.h>
#include <RcppEigen.h>
#include <R_ext/Rdynload.h>
#include <vector>
#include <cstring>
#include <string>
#include <limits>

#include "laplace_core.h"
#include "laplace_re_priors.h"
#include "pg_binomial.h"
#include "pg_negbin.h"
#include "ess_sampler.h"
#include "hmc_sampler.h"
#include "vi_types.h"
#include "sparse_cholesky.h"
#include "stochastic_logdet.h"

#include "tulpa/laplace_api.h"
#include "tulpa/laplace_spec_api.h"
#include "tulpa/pg_api.h"
#include "tulpa/vi_api.h"
#include "tulpa/ess_api.h"
#include "tulpa/sparse_solver_api.h"
#include "tulpa/nested_laplace_api.h"
#include "tulpa/spde_api.h"
#include "tulpa/sghmc_api.h"
#include "tulpa/mclmc_api.h"
#include "tulpa/smc_api.h"
#include "sghmc_sampler.h"
#include "mclmc_modeldata.h"
#include "smc_modeldata.h"

// ============================================================================
// Forward declarations: these definitions live in their .cpp files but are
// not exposed in their public headers. Re-declare here to call them.
// ============================================================================
namespace tulpa {

namespace vi {
VIResult fit_vi(
    const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu
);
} // namespace vi

} // namespace tulpa

// Nested-Laplace + SPDE entries live at global scope (Rcpp::export functions
// without an explicit namespace). Re-declare the ones we call below.
Rcpp::List cpp_nested_laplace_icar(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::NumericVector tau_grid,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable,
    bool store_Q,
    std::string checkpoint_path = ""
);

Rcpp::List cpp_nested_laplace_bym2(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double scale_factor,
    Rcpp::NumericVector sigma_spatial_grid,
    Rcpp::NumericVector rho_grid,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable,
    bool store_Q,
    std::string checkpoint_path = ""
);

Rcpp::List cpp_nested_laplace_car_proper(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::NumericVector tau_grid, Rcpp::NumericVector rho_grid,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable,
    bool store_Q,
    std::string checkpoint_path = ""
);

Rcpp::List cpp_nested_laplace_temporal(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector temporal_idx, int n_times,
    std::string temporal_type,
    Rcpp::NumericVector tau_grid, Rcpp::NumericVector rho_grid, bool cyclic,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable,
    bool store_Q,
    std::string checkpoint_path = ""
);

Rcpp::List cpp_nested_laplace_nngp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int n_spatial, int nn,
    Rcpp::NumericVector sigma2_grid, Rcpp::NumericVector phi_gp_grid,
    int cov_type,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable,
    bool store_Q,
    std::string checkpoint_path = ""
);

Rcpp::List cpp_nested_laplace_hsgp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix phi_basis,
    Rcpp::NumericVector lambda_eig,
    Rcpp::NumericVector sigma2_grid,
    Rcpp::NumericVector lengthscale_grid,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable,
    bool store_Q,
    std::string checkpoint_path = ""
);

// Spatio-temporal nested-Laplace entries (defined in nested_laplace.cpp):
// one per spatial family, temporal kernel selected at runtime via
// `temporal_type`. Declared here so the shims below can call them.
Rcpp::List cpp_nested_laplace_st_icar(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_spatial_grid,
    std::string temporal_type,
    Rcpp::NumericVector tau_temporal_grid,
    Rcpp::Nullable<Rcpp::NumericVector> rho_temporal_grid,
    bool cyclic,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable,
    bool store_Q,
    bool force_sparse = false,
    std::string checkpoint_path = ""
);

Rcpp::List cpp_nested_laplace_st_car_proper(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_spatial_grid,
    Rcpp::NumericVector rho_spatial_grid,
    std::string temporal_type,
    Rcpp::NumericVector tau_temporal_grid,
    Rcpp::Nullable<Rcpp::NumericVector> rho_temporal_grid,
    bool cyclic,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable,
    bool store_Q,
    bool force_sparse = false,
    std::string checkpoint_path = ""
);

Rcpp::List cpp_nested_laplace_st_bym2(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double scale_factor,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector sigma_spatial_grid,
    Rcpp::NumericVector rho_spatial_grid,
    std::string temporal_type,
    Rcpp::NumericVector tau_temporal_grid,
    Rcpp::Nullable<Rcpp::NumericVector> rho_temporal_grid,
    bool cyclic,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable,
    bool store_Q,
    bool force_sparse = false,
    std::string checkpoint_path = ""
);

Rcpp::List cpp_nested_laplace_st_hsgp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix phi_basis,
    Rcpp::NumericVector lambda_eig,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector sigma2_spatial_grid,
    Rcpp::NumericVector lengthscale_spatial_grid,
    std::string temporal_type,
    Rcpp::NumericVector tau_temporal_grid,
    Rcpp::Nullable<Rcpp::NumericVector> rho_temporal_grid,
    bool cyclic,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable,
    bool store_Q,
    bool force_sparse = false,
    std::string checkpoint_path = ""
);

Rcpp::List cpp_nested_laplace_st_nngp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order, int nn, int cov_type,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector sigma2_spatial_grid,
    Rcpp::NumericVector phi_gp_spatial_grid,
    std::string temporal_type,
    Rcpp::NumericVector tau_temporal_grid,
    Rcpp::Nullable<Rcpp::NumericVector> rho_temporal_grid,
    bool cyclic,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable,
    bool store_Q,
    bool force_sparse = false,
    std::string checkpoint_path = ""
);

Rcpp::List cpp_nested_laplace_spde(
    Rcpp::NumericVector y, Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx, int n_re_groups, double sigma_re,
    Rcpp::NumericVector A_x, Rcpp::IntegerVector A_i, Rcpp::IntegerVector A_p,
    int n_obs, int n_mesh,
    Rcpp::NumericVector C0_diag,
    Rcpp::NumericVector G1_x, Rcpp::IntegerVector G1_i, Rcpp::IntegerVector G1_p,
    Rcpp::NumericVector range_grid,
    Rcpp::NumericVector sigma_grid,
    double nu,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable,
    Rcpp::Nullable<Rcpp::NumericVector> rational_poles_nullable,
    Rcpp::Nullable<Rcpp::NumericVector> rational_weights_nullable,
    bool store_Q,
    std::string checkpoint_path = ""
);

// ============================================================================
// Helpers
// ============================================================================
namespace {

// Build a column-major Rcpp::NumericMatrix from a flat caller buffer.
inline Rcpp::NumericMatrix build_matrix_colmajor(
    const double* X_flat, int N, int p
) {
    Rcpp::NumericMatrix X(N, p);
    if (N > 0 && p > 0 && X_flat) {
        std::memcpy(&X[0], X_flat, sizeof(double) * (size_t)N * (size_t)p);
    }
    return X;
}

// Build a column-major Rcpp::IntegerMatrix from a flat caller buffer.
inline Rcpp::IntegerMatrix build_int_matrix_colmajor(
    const int* M_flat, int N, int p
) {
    Rcpp::IntegerMatrix M(N, p);
    if (N > 0 && p > 0 && M_flat) {
        std::memcpy(&M[0], M_flat, sizeof(int) * (size_t)N * (size_t)p);
    }
    return M;
}

// Copy an Rcpp::NumericMatrix (column-major) into a row-major raw buffer.
// Allocates the buffer; caller owns it.
inline double* matrix_to_row_major(const Rcpp::NumericMatrix& M) {
    int nrow = M.nrow();
    int ncol = M.ncol();
    if (nrow <= 0 || ncol <= 0) return nullptr;
    double* buf = new double[(size_t)nrow * (size_t)ncol];
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            buf[(size_t)i * ncol + j] = M(i, j);
        }
    }
    return buf;
}

inline double* vector_to_buf(const Rcpp::NumericVector& v) {
    int n = v.size();
    if (n <= 0) return nullptr;
    double* buf = new double[n];
    for (int i = 0; i < n; i++) buf[i] = v[i];
    return buf;
}

// Common preamble shared by all (Nested-)Laplace shims that take the
// canonical (y, n_trials, X_flat, re_idx, family) input bundle. Two outlier
// shims do not use this: laplace_mode_dense_multi_re_impl (uses re_idx_list
// rather than re_idx) and nested_laplace_spde_impl (no re_idx).
struct LaplaceShimInputs {
    Rcpp::NumericVector  yv;
    Rcpp::IntegerVector  nv;
    Rcpp::NumericMatrix  Xm;
    Rcpp::NumericVector  rv;
    std::string          fam;
};

inline LaplaceShimInputs pack_laplace_shim_inputs(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, const char* family
) {
    LaplaceShimInputs in;
    in.yv  = Rcpp::NumericVector(y, y + N);
    in.nv  = Rcpp::IntegerVector(n_trials, n_trials + N);
    in.Xm  = build_matrix_colmajor(X_flat, N, p);
    in.rv  = Rcpp::NumericVector(re_idx, re_idx + N);
    in.fam = family ? std::string(family) : std::string("binomial");
    return in;
}

// Marshal the four adjacency CSR vectors shared by all spatial shims.
inline void marshal_adj(
    const int* spatial_idx, int N,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    int n_spatial_units,
    Rcpp::IntegerVector& sidx, Rcpp::IntegerVector& arp,
    Rcpp::IntegerVector& aci,  Rcpp::IntegerVector& nn
) {
    sidx = Rcpp::IntegerVector(spatial_idx, spatial_idx + N);
    int nadj = adj_row_ptr ? adj_row_ptr[n_spatial_units] : 0;
    arp  = Rcpp::IntegerVector(adj_row_ptr, adj_row_ptr + n_spatial_units + 1);
    aci  = Rcpp::IntegerVector(adj_col_idx, adj_col_idx + nadj);
    nn   = Rcpp::IntegerVector(n_neighbors, n_neighbors + n_spatial_units);
}

} // namespace

#include "tulpa_shims_nested_laplace.h"

#include "tulpa_shims_pg.h"

#include "tulpa_shims_vi_ess.h"

#include "tulpa_shims_sparse.h"

#include "tulpa_shims_stochastic.h"

#include "tulpa_shims_laplace_spec.h"


// ============================================================================
// Registration: called from tulpa_init.cpp::tulpa_register_callables.
// ============================================================================

void tulpa_register_shims(DllInfo* dll) {
    R_RegisterCCallable("tulpa", "tulpa_laplace_spec_dense",
        (DL_FUNC)&tulpa_laplace_spec_dense_impl);

    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_icar",
        (DL_FUNC)&tulpa_nested_laplace_icar_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_bym2",
        (DL_FUNC)&tulpa_nested_laplace_bym2_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_car_proper",
        (DL_FUNC)&tulpa_nested_laplace_car_proper_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_temporal",
        (DL_FUNC)&tulpa_nested_laplace_temporal_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_nngp",
        (DL_FUNC)&tulpa_nested_laplace_nngp_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_hsgp",
        (DL_FUNC)&tulpa_nested_laplace_hsgp_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_st_icar",
        (DL_FUNC)&tulpa_nested_laplace_st_icar_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_st_car_proper",
        (DL_FUNC)&tulpa_nested_laplace_st_car_proper_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_st_bym2",
        (DL_FUNC)&tulpa_nested_laplace_st_bym2_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_st_hsgp",
        (DL_FUNC)&tulpa_nested_laplace_st_hsgp_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_st_nngp",
        (DL_FUNC)&tulpa_nested_laplace_st_nngp_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_spde",
        (DL_FUNC)&tulpa_nested_laplace_spde_impl);

    // tulpa_nested_laplace_joint_bym2 shim removed in Phase J-E -- the
    // legacy single-block joint kernels were folded into the multi-block
    // path. See R/nested_laplace_joint.R::.joint_call_kernel_via_multi.

    R_RegisterCCallable("tulpa", "tulpa_pg_binomial_gibbs",
        (DL_FUNC)&tulpa_pg_binomial_gibbs_impl);
    R_RegisterCCallable("tulpa", "tulpa_pg_negbin_gibbs",
        (DL_FUNC)&tulpa_pg_negbin_gibbs_impl);
    R_RegisterCCallable("tulpa", "tulpa_pg_negbin_spatial_gibbs",
        (DL_FUNC)&tulpa_pg_negbin_spatial_gibbs_impl);

    R_RegisterCCallable("tulpa", "tulpa_fit_vi",
        (DL_FUNC)&tulpa_fit_vi_impl);

    R_RegisterCCallable("tulpa", "tulpa_run_ess_sampler",
        (DL_FUNC)&tulpa_run_ess_sampler_impl);

    R_RegisterCCallable("tulpa", "tulpa_sparse_chol_create",
        (DL_FUNC)&tulpa_sparse_chol_create_impl);
    R_RegisterCCallable("tulpa", "tulpa_sparse_chol_destroy",
        (DL_FUNC)&tulpa_sparse_chol_destroy_impl);
    R_RegisterCCallable("tulpa", "tulpa_sparse_chol_analyze",
        (DL_FUNC)&tulpa_sparse_chol_analyze_impl);
    R_RegisterCCallable("tulpa", "tulpa_sparse_chol_factorize",
        (DL_FUNC)&tulpa_sparse_chol_factorize_impl);
    R_RegisterCCallable("tulpa", "tulpa_sparse_chol_solve",
        (DL_FUNC)&tulpa_sparse_chol_solve_impl);
    R_RegisterCCallable("tulpa", "tulpa_sparse_chol_log_det",
        (DL_FUNC)&tulpa_sparse_chol_log_det_impl);
    R_RegisterCCallable("tulpa", "tulpa_sparse_chol_sel_inv_diag",
        (DL_FUNC)&tulpa_sparse_chol_sel_inv_diag_impl);
    R_RegisterCCallable("tulpa", "tulpa_takahashi_partial_inverse_dense",
        (DL_FUNC)&tulpa_takahashi_partial_inverse_dense_impl);
    R_RegisterCCallable("tulpa", "tulpa_stochastic_log_det",
        (DL_FUNC)&tulpa_stochastic_log_det_impl);

    R_RegisterCCallable("tulpa", "tulpa_sghmc_fit",
        (DL_FUNC)&tulpa_sghmc_fit_impl);
    R_RegisterCCallable("tulpa", "tulpa_sgld_fit",
        (DL_FUNC)&tulpa_sgld_fit_impl);

    R_RegisterCCallable("tulpa", "tulpa_mclmc_fit",
        (DL_FUNC)&tulpa_mclmc_fit_impl);
    R_RegisterCCallable("tulpa", "tulpa_smc_fit",
        (DL_FUNC)&tulpa_smc_fit_impl);
}
