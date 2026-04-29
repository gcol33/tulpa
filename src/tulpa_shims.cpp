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
#include "pg_binomial.h"
#include "pg_negbin.h"
#include "ess_sampler.h"
#include "hmc_sampler.h"
#include "vi_types.h"
#include "sparse_cholesky.h"
#include "stochastic_logdet.h"

#include "tulpa/laplace_api.h"
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

LaplaceResult laplace_mode_dense(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads
);

LaplaceResult laplace_mode_spatial(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const Rcpp::IntegerVector& spatial_idx, int n_spatial_units,
    const Rcpp::IntegerVector& adj_row_ptr, const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors, double tau_spatial,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    const Rcpp::NumericVector& x_init
);

LaplaceResult laplace_mode_dense_multi_re(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X,
    const Rcpp::List& re_idx_list,
    const Rcpp::IntegerVector& re_ngroups,
    const Rcpp::List& re_sigma_list,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::List> re_Z_list_,
    Rcpp::Nullable<Rcpp::IntegerVector> re_ncoefs_,
    Rcpp::Nullable<Rcpp::NumericVector> weights_,
    Rcpp::Nullable<Rcpp::NumericVector> offset_,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_
);

LaplaceResult laplace_mode_bym2(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const Rcpp::IntegerVector& spatial_idx, int n_spatial_units,
    const Rcpp::IntegerVector& adj_row_ptr, const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    double sigma_spatial, double rho, double scale_factor,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads
);

LaplaceResult laplace_mode_gp(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx, const Rcpp::NumericMatrix& nn_dist,
    const Rcpp::IntegerVector& nn_order,
    int n_spatial, int nn,
    double sigma2_gp, double phi_gp, int cov_type,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads
);

LaplaceResult laplace_mode_multiscale_gp(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx_local, const Rcpp::NumericMatrix& nn_dist_local,
    const Rcpp::IntegerVector& nn_order_local, int nn_local,
    const Rcpp::IntegerMatrix& nn_idx_regional, const Rcpp::NumericMatrix& nn_dist_regional,
    const Rcpp::IntegerVector& nn_order_regional, int nn_regional,
    int n_spatial,
    double sigma2_local, double phi_local,
    double sigma2_regional, double phi_regional,
    int cov_type, const std::string& family, double phi,
    int max_iter, double tol, int n_threads
);

LaplaceResult laplace_mode_multiscale_temporal(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const Rcpp::IntegerVector& time_idx, int n_times,
    int seasonal_period, int trend_type, int short_type,
    double sigma2_trend, double sigma2_seasonal, double sigma2_short, double rho_short,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads
);

LaplaceResult laplace_mode_rsr(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const Rcpp::IntegerVector& spatial_idx, int n_spatial_units,
    const Rcpp::IntegerVector& adj_row_ptr, const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors, double tau_spatial,
    const Rcpp::NumericVector& rsr_projection, int rsr_n,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads
);

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
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable
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
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable
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
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable
);

Rcpp::List cpp_nested_laplace_rw1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector temporal_idx, int n_times, bool cyclic,
    Rcpp::NumericVector tau_grid,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable
);

Rcpp::List cpp_nested_laplace_rw2(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_grid,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable
);

Rcpp::List cpp_nested_laplace_ar1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_grid, Rcpp::NumericVector rho_grid,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable
);

Rcpp::List cpp_nested_laplace_nngp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int n_spatial, int nn,
    Rcpp::NumericVector sigma2_grid, Rcpp::NumericVector phi_gp_grid,
    int cov_type,
    std::string family, double phi,
    int max_iter, double tol, int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable
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
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable
);

Rcpp::List cpp_nested_laplace_spde(
    Rcpp::NumericVector y, Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
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
    Rcpp::Nullable<Rcpp::NumericVector> rational_weights_nullable
);

// ============================================================================
// Helpers
// ============================================================================
namespace {

inline void copy_mode_result(
    const tulpa::LaplaceResult& result,
    tulpa::LaplaceShimResult* result_out
) {
    int n_x = result.mode.size();
    result_out->n_x = n_x;
    result_out->log_det_Q = result.log_det_Q;
    result_out->log_marginal = result.log_marginal;
    result_out->n_iter = result.n_iter;
    result_out->converged = result.converged ? 1 : 0;

    result_out->mode = new double[n_x];
    for (int j = 0; j < n_x; j++) {
        result_out->mode[j] = result.mode[j];
    }
}

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

// ============================================================================
// Laplace shims
// ============================================================================

extern "C" void tulpa_laplace_mode_dense_impl(
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
    const double* /*x_init*/,
    int /*n_x_init*/,
    tulpa::LaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    std::string fam = family ? std::string(family) : std::string("binomial");

    // laplace_mode_dense does not accept x_init — kept in the shim signature
    // for API symmetry with the spatial variant.
    tulpa::LaplaceResult result = tulpa::laplace_mode_dense(
        yv, nv, Xm, rv, n_re_groups, sigma_re, fam, phi,
        max_iter, tol, n_threads
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_spatial_impl(
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
    tulpa::LaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::NumericVector  xinit;
    if (x_init && n_x_init > 0) xinit = Rcpp::NumericVector(x_init, x_init + n_x_init);

    std::string fam = family ? std::string(family) : std::string("binomial");

    tulpa::LaplaceResult result = tulpa::laplace_mode_spatial(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn, tau_spatial,
        fam, phi, max_iter, tol, n_threads, xinit
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_dense_multi_re_impl(
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
    tulpa::LaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);

    Rcpp::List re_idx_list(n_terms);
    for (int k = 0; k < n_terms; k++) {
        Rcpp::IntegerVector v(re_idx_flat + (size_t)k * N,
                              re_idx_flat + (size_t)(k + 1) * N);
        re_idx_list[k] = v;
    }

    Rcpp::IntegerVector re_ngroups_v(re_ngroups, re_ngroups + n_terms);

    Rcpp::List re_sigma_list(n_terms);
    for (int k = 0; k < n_terms; k++) {
        int s = sigma_offsets[k];
        int e = sigma_offsets[k + 1];
        Rcpp::NumericVector v(sigma_flat + s, sigma_flat + e);
        re_sigma_list[k] = v;
    }

    // re_ncoefs is needed both to drive the internal API and to size Z slices.
    std::vector<int> ncoefs_vec(n_terms, 1);
    if (re_ncoefs) {
        for (int k = 0; k < n_terms; k++) ncoefs_vec[k] = re_ncoefs[k];
    }

    Rcpp::Nullable<Rcpp::IntegerVector> ncoefs_nullable = R_NilValue;
    if (re_ncoefs) {
        ncoefs_nullable = Rcpp::Nullable<Rcpp::IntegerVector>(
            Rcpp::IntegerVector(re_ncoefs, re_ncoefs + n_terms));
    }

    Rcpp::Nullable<Rcpp::List> Z_list_nullable = R_NilValue;
    if (Z_offsets && Z_flat) {
        Rcpp::List re_Z_list(n_terms);
        for (int k = 0; k < n_terms; k++) {
            int ck = ncoefs_vec[k];
            int off = Z_offsets[k];
            Rcpp::NumericMatrix Zk(N, ck);
            if (N > 0 && ck > 0) {
                std::memcpy(&Zk[0], Z_flat + off,
                            sizeof(double) * (size_t)N * ck);
            }
            re_Z_list[k] = Zk;
        }
        Z_list_nullable = Rcpp::Nullable<Rcpp::List>(re_Z_list);
    }

    Rcpp::Nullable<Rcpp::NumericVector> weights_nullable = R_NilValue;
    if (weights) {
        weights_nullable = Rcpp::Nullable<Rcpp::NumericVector>(
            Rcpp::NumericVector(weights, weights + N));
    }

    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue;
    if (offset_vec) {
        offset_nullable = Rcpp::Nullable<Rcpp::NumericVector>(
            Rcpp::NumericVector(offset_vec, offset_vec + N));
    }

    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue;
    if (x_init && n_x_init > 0) {
        x_init_nullable = Rcpp::Nullable<Rcpp::NumericVector>(
            Rcpp::NumericVector(x_init, x_init + n_x_init));
    }

    std::string fam = family ? std::string(family) : std::string("binomial");

    tulpa::LaplaceResult result = tulpa::laplace_mode_dense_multi_re(
        yv, nv, Xm, re_idx_list, re_ngroups_v, re_sigma_list,
        fam, phi, max_iter, tol, n_threads,
        Z_list_nullable, ncoefs_nullable,
        weights_nullable, offset_nullable, x_init_nullable
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_bym2_impl(
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
    tulpa::LaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);

    std::string fam = family ? std::string(family) : std::string("binomial");

    tulpa::LaplaceResult result = tulpa::laplace_mode_bym2(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        sigma_spatial, rho, scale_factor,
        fam, phi, max_iter, tol, n_threads
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_gp_impl(
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
    tulpa::LaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::NumericMatrix  cm = build_matrix_colmajor(coords_flat, n_spatial, coord_dim);
    Rcpp::IntegerMatrix  nim = build_int_matrix_colmajor(nn_idx_flat, n_spatial, nn);
    Rcpp::NumericMatrix  ndm = build_matrix_colmajor(nn_dist_flat, n_spatial, nn);
    Rcpp::IntegerVector  nord(nn_order, nn_order + n_spatial);

    std::string fam = family ? std::string(family) : std::string("binomial");

    tulpa::LaplaceResult result = tulpa::laplace_mode_gp(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        cm, nim, ndm, nord, n_spatial, nn,
        sigma2_gp, phi_gp, cov_type,
        fam, phi, max_iter, tol, n_threads
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_multiscale_gp_impl(
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
    tulpa::LaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::NumericMatrix  cm = build_matrix_colmajor(coords_flat, n_spatial, coord_dim);

    Rcpp::IntegerMatrix  nim_l = build_int_matrix_colmajor(nn_idx_local_flat,    n_spatial, nn_local);
    Rcpp::NumericMatrix  ndm_l = build_matrix_colmajor    (nn_dist_local_flat,   n_spatial, nn_local);
    Rcpp::IntegerVector  nord_l(nn_order_local,    nn_order_local    + n_spatial);
    Rcpp::IntegerMatrix  nim_r = build_int_matrix_colmajor(nn_idx_regional_flat, n_spatial, nn_regional);
    Rcpp::NumericMatrix  ndm_r = build_matrix_colmajor    (nn_dist_regional_flat, n_spatial, nn_regional);
    Rcpp::IntegerVector  nord_r(nn_order_regional, nn_order_regional + n_spatial);

    std::string fam = family ? std::string(family) : std::string("binomial");

    tulpa::LaplaceResult result = tulpa::laplace_mode_multiscale_gp(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        cm,
        nim_l, ndm_l, nord_l, nn_local,
        nim_r, ndm_r, nord_r, nn_regional,
        n_spatial,
        sigma2_local, phi_local, sigma2_regional, phi_regional,
        cov_type, fam, phi, max_iter, tol, n_threads
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_multiscale_temporal_impl(
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
    tulpa::LaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  tv(time_idx, time_idx + N);

    std::string fam = family ? std::string(family) : std::string("binomial");

    tulpa::LaplaceResult result = tulpa::laplace_mode_multiscale_temporal(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        tv, n_times, seasonal_period, trend_type, short_type,
        sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
        fam, phi, max_iter, tol, n_threads
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_rsr_impl(
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
    tulpa::LaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::NumericVector  proj(rsr_projection_flat,
                              rsr_projection_flat + (size_t)rsr_n * (size_t)rsr_n);

    std::string fam = family ? std::string(family) : std::string("binomial");

    tulpa::LaplaceResult result = tulpa::laplace_mode_rsr(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        tau_spatial, proj, rsr_n,
        fam, phi, max_iter, tol, n_threads
    );
    copy_mode_result(result, result_out);
}

// ============================================================================
// Nested-Laplace shims
// ============================================================================
//
// Each shim packs raw pointers into Rcpp::* on entry, calls the matching
// cpp_nested_laplace_<backend> wrapper, and unpacks the universal output
// block (log_marginal[n_grid], n_iter[n_grid], optional modes[n_grid x n_x])
// into the caller's NestedLaplaceShimResult. Backend-specific grid vectors
// are passed in by the caller and not echoed back through the shim.

namespace {

inline Rcpp::Nullable<Rcpp::NumericVector> wrap_x_init(
    const double* x_init, int n_x_init
) {
    if (x_init && n_x_init > 0) {
        return Rcpp::wrap(Rcpp::NumericVector(x_init, x_init + n_x_init));
    }
    return R_NilValue;
}

// Copy the universal output block out of a nested-Laplace result list.
inline void copy_nested_laplace_result(
    const Rcpp::List& out,
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector lm = out["log_marginal"];
    Rcpp::IntegerVector ni = out["n_iter"];
    int n_grid = (int)lm.size();

    result_out->n_grid = n_grid;
    result_out->log_marginal = new double[n_grid];
    result_out->n_iter       = new int[n_grid];
    for (int k = 0; k < n_grid; k++) {
        result_out->log_marginal[k] = lm[k];
        result_out->n_iter[k]       = ni[k];
    }

    bool has_modes = out.containsElementNamed("modes");
    if (has_modes) {
        Rcpp::NumericMatrix modes = out["modes"];
        int n_x = modes.ncol();
        result_out->store_modes = 1;
        result_out->n_x = n_x;
        result_out->modes = new double[(size_t)n_grid * (size_t)n_x];
        // modes is column-major in Rcpp; emit row-major in the shim buffer
        // so callers can index modes[k*n_x + j] without thinking about it.
        for (int k = 0; k < n_grid; k++) {
            for (int j = 0; j < n_x; j++) {
                result_out->modes[(size_t)k * n_x + j] = modes(k, j);
            }
        }
    } else {
        result_out->store_modes = 0;
        result_out->n_x = 0;
        result_out->modes = nullptr;
    }
}

} // namespace

extern "C" void tulpa_nested_laplace_icar_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* spatial_idx, int n_spatial_units,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    const double* tau_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::NumericVector  tg (tau_grid,    tau_grid + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_icar(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        tg, fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_bym2_impl(
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
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::NumericVector  sg (sigma_spatial_grid, sigma_spatial_grid + n_grid);
    Rcpp::NumericVector  rg (rho_grid,           rho_grid           + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_bym2(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        scale_factor, sg, rg,
        fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_car_proper_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* spatial_idx, int n_spatial_units,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    const double* tau_grid, const double* rho_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::NumericVector  tg (tau_grid, tau_grid + n_grid);
    Rcpp::NumericVector  rg (rho_grid, rho_grid + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_car_proper(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        tg, rg, fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_rw1_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* temporal_idx, int n_times, int cyclic,
    const double* tau_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  tv(temporal_idx, temporal_idx + N);
    Rcpp::NumericVector  tg(tau_grid, tau_grid + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_rw1(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        tv, n_times, (cyclic != 0),
        tg, fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_rw2_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* temporal_idx, int n_times,
    const double* tau_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  tv(temporal_idx, temporal_idx + N);
    Rcpp::NumericVector  tg(tau_grid, tau_grid + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_rw2(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        tv, n_times,
        tg, fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_ar1_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* temporal_idx, int n_times,
    const double* tau_grid, const double* rho_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  tv(temporal_idx, temporal_idx + N);
    Rcpp::NumericVector  tg(tau_grid, tau_grid + n_grid);
    Rcpp::NumericVector  rg(rho_grid, rho_grid + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_ar1(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        tv, n_times, tg, rg,
        fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_nngp_impl(
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
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::NumericMatrix  cm = build_matrix_colmajor(coords_flat, n_spatial, coord_dim);
    Rcpp::IntegerMatrix  nim = build_int_matrix_colmajor(nn_idx_flat, n_spatial, nn);
    Rcpp::NumericMatrix  ndm = build_matrix_colmajor(nn_dist_flat, n_spatial, nn);
    Rcpp::IntegerVector  nord(nn_order, nn_order + n_spatial);
    Rcpp::NumericVector  s2g(sigma2_grid,  sigma2_grid  + n_grid);
    Rcpp::NumericVector  phg(phi_gp_grid,  phi_gp_grid  + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_nngp(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        cm, nim, ndm, nord, n_spatial, nn,
        s2g, phg, cov_type,
        fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_hsgp_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const double* phi_basis_flat, int n_basis,
    const double* lambda_eig,
    const double* sigma2_grid, const double* lengthscale_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::NumericMatrix  pb = build_matrix_colmajor(phi_basis_flat, N, n_basis);
    Rcpp::NumericVector  le(lambda_eig, lambda_eig + n_basis);
    Rcpp::NumericVector  s2g(sigma2_grid,      sigma2_grid      + n_grid);
    Rcpp::NumericVector  lsg(lengthscale_grid, lengthscale_grid + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_hsgp(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        pb, le, s2g, lsg,
        fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
    );
    copy_nested_laplace_result(out, result_out);
}

// ============================================================================
// SPDE nested-Laplace shim
// ============================================================================

extern "C" void tulpa_nested_laplace_spde_impl(
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
    tulpa::SpdeNestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + n_obs);
    Rcpp::IntegerVector  nv(n_trials, n_trials + n_obs);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, n_obs, p);

    int A_nnz  = A_p  ? A_p [n_mesh] : 0;
    int G1_nnz = G1_p ? G1_p[n_mesh] : 0;
    Rcpp::NumericVector  axv(A_x, A_x + A_nnz);
    Rcpp::IntegerVector  aiv(A_i, A_i + A_nnz);
    Rcpp::IntegerVector  apv(A_p, A_p + n_mesh + 1);
    Rcpp::NumericVector  c0  (C0_diag, C0_diag + n_mesh);
    Rcpp::NumericVector  g1xv(G1_x, G1_x + G1_nnz);
    Rcpp::IntegerVector  g1iv(G1_i, G1_i + G1_nnz);
    Rcpp::IntegerVector  g1pv(G1_p, G1_p + n_mesh + 1);
    Rcpp::NumericVector  rng (range_grid, range_grid + n_grid);
    Rcpp::NumericVector  sig (sigma_grid, sigma_grid + n_grid);

    Rcpp::Nullable<Rcpp::NumericVector> poles_n   = R_NilValue;
    Rcpp::Nullable<Rcpp::NumericVector> weights_n = R_NilValue;
    if (rational_poles && rational_weights && n_rational > 0) {
        poles_n   = Rcpp::wrap(Rcpp::NumericVector(rational_poles,   rational_poles   + n_rational));
        weights_n = Rcpp::wrap(Rcpp::NumericVector(rational_weights, rational_weights + n_rational));
    }

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_spde(
        yv, nv, Xm,
        axv, aiv, apv, n_obs, n_mesh,
        c0, g1xv, g1iv, g1pv,
        rng, sig, nu,
        fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        poles_n, weights_n
    );

    Rcpp::NumericVector lm = out["log_marginal"];
    Rcpp::IntegerVector ni = out["n_iter"];
    int ng = (int)lm.size();
    result_out->n_grid = ng;
    result_out->Q_nnz  = (int)Rcpp::as<int>(out["Q_nnz"]);
    result_out->log_marginal = new double[ng];
    result_out->n_iter       = new int[ng];
    for (int k = 0; k < ng; k++) {
        result_out->log_marginal[k] = lm[k];
        result_out->n_iter[k]       = ni[k];
    }
}

// ============================================================================
// PG shims
// ============================================================================

namespace {

inline void fill_pg_result(
    const Rcpp::List& draws,
    int n_obs, int n_spatial_units, bool has_r,
    tulpa::PGShimResult* out
) {
    Rcpp::NumericMatrix beta = draws["beta"];
    Rcpp::NumericMatrix re   = draws["re"];
    Rcpp::NumericVector sig  = draws["sigma_re"];

    out->n_save = beta.nrow();
    out->n_beta = beta.ncol();
    out->n_re   = re.ncol();
    out->n_obs  = n_obs;
    out->n_spatial_units = n_spatial_units;

    out->beta     = matrix_to_row_major(beta);
    out->re       = matrix_to_row_major(re);
    out->sigma_re = vector_to_buf(sig);
    out->eta      = nullptr;
    out->r_disp   = nullptr;
    out->spatial  = nullptr;
    out->tau_spatial = nullptr;

    if (draws.containsElementNamed("eta")) {
        Rcpp::NumericMatrix e = draws["eta"];
        out->eta = matrix_to_row_major(e);
    }
    if (has_r && draws.containsElementNamed("r")) {
        Rcpp::NumericVector r = draws["r"];
        out->r_disp = vector_to_buf(r);
    }
    if (n_spatial_units > 0 && draws.containsElementNamed("spatial")) {
        Rcpp::NumericMatrix s = draws["spatial"];
        out->spatial = matrix_to_row_major(s);
    }
    if (n_spatial_units > 0 && draws.containsElementNamed("tau")) {
        Rcpp::NumericVector t = draws["tau"];
        out->tau_spatial = vector_to_buf(t);
    }
}

} // namespace

// Forward declaration: defined inside namespace tulpa in pg_binomial.cpp.
namespace tulpa {
Rcpp::List pg_binomial_gibbs_impl(
    Rcpp::IntegerVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::IntegerVector group,
    int n_groups, int n_iter, int n_warmup, int thin,
    double prior_beta_sd, double prior_sigma_scale,
    bool store_eta, bool verbose, int n_threads
);
}

extern "C" void tulpa_pg_binomial_gibbs_impl(
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
    tulpa::PGShimResult* result_out
) {
    Rcpp::IntegerVector yv(y, y + N_obs);
    Rcpp::IntegerVector nv(n_trials, n_trials + N_obs);
    Rcpp::NumericMatrix Xm = build_matrix_colmajor(X_flat, N_obs, p);
    Rcpp::IntegerVector gv(group, group + N_obs);

    GetRNGstate();
    Rcpp::List draws = tulpa::pg_binomial_gibbs_impl(
        yv, nv, Xm, gv, n_groups,
        n_iter, n_warmup, thin,
        prior_beta_sd, prior_sigma_scale,
        store_eta != 0, verbose != 0, n_threads
    );
    PutRNGstate();

    fill_pg_result(draws, N_obs, /*n_spatial_units=*/0, /*has_r=*/false, result_out);
}

extern "C" void tulpa_pg_negbin_gibbs_impl(
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
    tulpa::PGShimResult* result_out
) {
    Rcpp::IntegerVector yv(y, y + N_obs);
    Rcpp::NumericMatrix Xm = build_matrix_colmajor(X_flat, N_obs, p);
    Rcpp::IntegerVector gv(group, group + N_obs);

    GetRNGstate();
    Rcpp::List draws = tulpa::pg_negbin_gibbs(
        yv, Xm, gv, n_groups,
        n_iter, n_warmup, thin,
        prior_beta_sd, prior_sigma_scale,
        prior_r_shape, prior_r_rate, r_init,
        store_eta != 0, verbose != 0, n_threads
    );
    PutRNGstate();

    fill_pg_result(draws, N_obs, /*n_spatial_units=*/0, /*has_r=*/true, result_out);
}

extern "C" void tulpa_pg_negbin_spatial_gibbs_impl(
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
    tulpa::PGShimResult* result_out
) {
    Rcpp::IntegerVector yv(y, y + N_obs);
    Rcpp::NumericMatrix Xm = build_matrix_colmajor(X_flat, N_obs, p);
    Rcpp::IntegerVector reg(re_group, re_group + N_obs);
    Rcpp::IntegerVector sg(spatial_group, spatial_group + N_obs);
    Rcpp::IntegerVector nn(n_neighbors, n_neighbors + n_spatial_units);

    // Internal API takes adjacency as a List<IntegerVector>; build it from CSR.
    Rcpp::List adj_list(n_spatial_units);
    for (int i = 0; i < n_spatial_units; i++) {
        int s = adj_row_ptr[i], e = adj_row_ptr[i + 1];
        Rcpp::IntegerVector neighbors(e - s);
        for (int k = s; k < e; k++) neighbors[k - s] = adj_col_idx[k];
        adj_list[i] = neighbors;
    }

    GetRNGstate();
    Rcpp::List draws = tulpa::pg_negbin_gibbs_spatial(
        yv, Xm, reg, n_re_groups, sg, n_spatial_units,
        adj_list, nn,
        n_iter, n_warmup, thin,
        prior_beta_sd, prior_sigma_re_scale,
        prior_tau_shape, prior_tau_rate,
        prior_r_shape, prior_r_rate, r_init,
        store_eta != 0, verbose != 0, n_threads
    );
    PutRNGstate();

    fill_pg_result(draws, N_obs, n_spatial_units, /*has_r=*/true, result_out);
}

// ============================================================================
// VI shim
// ============================================================================

extern "C" void tulpa_fit_vi_impl(
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    int D,
    const tulpa::VIShimConfig* shim_config,
    const double* init_mu,
    int n_init_mu,
    tulpa::VIShimResult* result_out
) {
    using tulpa::vi::VIConfig;
    using tulpa::vi::VIVariant;

    VIConfig cfg;
    cfg.variant = static_cast<VIVariant>(shim_config->variant);
    cfg.max_iter = shim_config->max_iter;
    cfg.mc_samples = shim_config->mc_samples;
    cfg.tol_grad = shim_config->tol_grad;
    cfg.tol_rel_elbo = shim_config->tol_rel_elbo;
    cfg.patience = shim_config->patience;
    cfg.adam_alpha = shim_config->adam_alpha;
    cfg.adam_beta1 = shim_config->adam_beta1;
    cfg.adam_beta2 = shim_config->adam_beta2;
    cfg.adam_eps = shim_config->adam_eps;
    cfg.rank = shim_config->rank;
    cfg.use_laplace_init = shim_config->use_laplace_init != 0;
    cfg.fullrank_threshold = shim_config->fullrank_threshold;
    cfg.lowrank_threshold = shim_config->lowrank_threshold;
    cfg.verbose = shim_config->verbose != 0;
    cfg.print_every = shim_config->print_every;
    cfg.seed = shim_config->seed;

    Eigen::VectorXd init_vec;
    const Eigen::VectorXd* init_ptr = nullptr;
    if (init_mu && n_init_mu == D) {
        init_vec = Eigen::Map<const Eigen::VectorXd>(init_mu, D);
        init_ptr = &init_vec;
    }

    tulpa::vi::VIResult res = tulpa::vi::fit_vi(*data, *layout, D, cfg, init_ptr);

    result_out->variant_used = static_cast<int>(res.variant_used);
    result_out->D = D;
    result_out->rank_used = res.rank_used;
    result_out->iterations = res.iterations;
    result_out->converged = res.converged ? 1 : 0;
    result_out->final_elbo = res.final_elbo;
    result_out->psis_k = res.psis_k;

    result_out->mu       = nullptr;
    result_out->Sigma    = nullptr;
    result_out->L_factor = nullptr;
    result_out->d_diag   = nullptr;
    result_out->elbo_history = nullptr;

    if (res.mu.size() > 0) {
        int d = res.mu.size();
        result_out->mu = new double[d];
        for (int i = 0; i < d; i++) result_out->mu[i] = res.mu(i);
    }
    if (res.Sigma.rows() > 0 && res.Sigma.cols() > 0) {
        int rows = res.Sigma.rows(), cols = res.Sigma.cols();
        result_out->Sigma = new double[(size_t)rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result_out->Sigma[(size_t)i * cols + j] = res.Sigma(i, j);
            }
        }
    }
    if (res.L_factor.rows() > 0 && res.L_factor.cols() > 0) {
        int rows = res.L_factor.rows(), cols = res.L_factor.cols();
        result_out->L_factor = new double[(size_t)rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result_out->L_factor[(size_t)i * cols + j] = res.L_factor(i, j);
            }
        }
    }
    if (res.d_diag.size() > 0) {
        int d = res.d_diag.size();
        result_out->d_diag = new double[d];
        for (int i = 0; i < d; i++) result_out->d_diag[i] = res.d_diag(i);
    }
    if (!res.elbo_history.empty()) {
        size_t n = res.elbo_history.size();
        result_out->elbo_history = new double[n];
        for (size_t i = 0; i < n; i++) result_out->elbo_history[i] = res.elbo_history[i];
    }
}

// ============================================================================
// ESS shim
// ============================================================================

extern "C" void tulpa_run_ess_sampler_impl(
    const double* init_params,
    int n_params,
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    const tulpa::ESSShimConfig* shim_config,
    tulpa::ESSShimResult* result_out
) {
    std::vector<double> init(init_params, init_params + n_params);

    tulpa_ess::ESSConfig cfg;
    cfg.n_iter = shim_config->n_iter;
    cfg.n_warmup = shim_config->n_warmup;
    cfg.n_thin = shim_config->n_thin;
    cfg.verbose = shim_config->verbose != 0;
    cfg.print_every = shim_config->print_every;
    cfg.seed = shim_config->seed;
    cfg.use_cholesky = shim_config->use_cholesky != 0;
    cfg.adapt_during_warmup = shim_config->adapt_during_warmup != 0;
    cfg.adapt_interval = shim_config->adapt_interval;
    cfg.joint_sigma_re = shim_config->joint_sigma_re != 0;

    tulpa_ess::ESSResult res = tulpa_ess::run_ess_sampler(init, *data, *layout, cfg);

    int n_save = res.samples.rows();
    int n_p    = res.samples.cols();
    result_out->n_save = n_save;
    result_out->n_params = n_p;
    result_out->n_slice_evals = res.n_slice_evals;
    result_out->avg_slice_evals = res.avg_slice_evals;
    result_out->success = res.success ? 1 : 0;
    result_out->samples = nullptr;
    result_out->log_lik = nullptr;
    std::strncpy(result_out->error_msg,
                 res.error_msg.c_str(),
                 sizeof(result_out->error_msg) - 1);
    result_out->error_msg[sizeof(result_out->error_msg) - 1] = '\0';

    if (n_save > 0 && n_p > 0) {
        result_out->samples = new double[(size_t)n_save * n_p];
        for (int i = 0; i < n_save; i++) {
            for (int j = 0; j < n_p; j++) {
                result_out->samples[(size_t)i * n_p + j] = res.samples(i, j);
            }
        }
    }
    if (!res.log_lik.empty()) {
        size_t n = res.log_lik.size();
        result_out->log_lik = new double[n];
        for (size_t i = 0; i < n; i++) result_out->log_lik[i] = res.log_lik[i];
    }
}

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

// ============================================================================
// SGHMC / SGLD shims
// ============================================================================
//
// Both samplers run on the same NUTS-style ModelData + ParamLayout interface
// (see nuts_api.h). The shim forwards raw scalar config, builds the sampler
// config struct, calls the underlying tulpa::run_sghmc_sampler /
// run_sgld_sampler, and copies the result into a flat
// SGSamplerShimResult buffer.

namespace {

inline void copy_sg_sampler_result(
    const Eigen::MatrixXd& samples,
    const std::vector<double>& log_lik,
    const std::vector<double>& epsilon_history,
    bool success,
    const std::string& error_msg,
    tulpa::SGSamplerShimResult* out
) {
    int n_save   = (int)samples.rows();
    int n_params = (int)samples.cols();
    out->n_sample = n_save;
    out->n_params = n_params;
    out->success  = success ? 1 : 0;
    std::strncpy(out->error_msg, error_msg.c_str(), sizeof(out->error_msg) - 1);
    out->error_msg[sizeof(out->error_msg) - 1] = '\0';

    out->samples = new double[(size_t)n_save * (size_t)n_params];
    for (int i = 0; i < n_save; i++) {
        for (int j = 0; j < n_params; j++) {
            out->samples[(size_t)i * n_params + j] = samples(i, j);
        }
    }

    out->log_lik = new double[n_save];
    for (int i = 0; i < n_save; i++) out->log_lik[i] = log_lik[i];

    int n_eps = (int)epsilon_history.size();
    out->n_eps_history  = n_eps;
    out->epsilon_history = new double[n_eps > 0 ? n_eps : 1];
    for (int i = 0; i < n_eps; i++) out->epsilon_history[i] = epsilon_history[i];
    out->final_epsilon = (n_eps > 0) ? epsilon_history[n_eps - 1] : 0.0;
}

} // namespace

extern "C" void tulpa_sghmc_fit_impl(
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    const double* init,
    int n_params,
    int n_iter,
    int n_warmup,
    int batch_size,
    double epsilon,
    double alpha,
    int L,
    unsigned int seed,
    int adapt_eps,
    double grad_clip,
    int verbose,
    tulpa::SGSamplerShimResult* result_out
) {
    std::vector<double> q_init(init, init + n_params);

    tulpa_sghmc::SGHMCConfig cfg;
    cfg.n_iter        = n_iter;
    cfg.n_warmup      = n_warmup;
    cfg.batch_size    = batch_size;
    cfg.epsilon       = epsilon;
    cfg.alpha         = alpha;
    cfg.L             = L;
    cfg.seed          = seed;
    cfg.adapt_epsilon = (adapt_eps != 0);
    cfg.grad_clip     = grad_clip;
    cfg.verbose       = (verbose != 0);

    tulpa_sghmc::SGHMCResult res =
        tulpa_sghmc::run_sghmc_sampler(q_init, *data, *layout, cfg);

    copy_sg_sampler_result(res.samples, res.log_lik, res.epsilon_history,
                           res.success, res.error_msg, result_out);
}

extern "C" void tulpa_sgld_fit_impl(
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    const double* init,
    int n_params,
    int n_iter,
    int n_warmup,
    int batch_size,
    double epsilon,
    double schedule_a,
    double schedule_b,
    double schedule_gamma,
    int use_schedule,
    double grad_clip,
    unsigned int seed,
    int verbose,
    tulpa::SGSamplerShimResult* result_out
) {
    std::vector<double> q_init(init, init + n_params);

    tulpa_sghmc::SGLDConfig cfg;
    cfg.n_iter         = n_iter;
    cfg.n_warmup       = n_warmup;
    cfg.batch_size     = batch_size;
    cfg.epsilon        = epsilon;
    cfg.schedule_a     = schedule_a;
    cfg.schedule_b     = schedule_b;
    cfg.schedule_gamma = schedule_gamma;
    cfg.use_schedule   = (use_schedule != 0);
    cfg.grad_clip      = grad_clip;
    cfg.seed           = seed;
    cfg.verbose        = (verbose != 0);

    tulpa_sghmc::SGLDResult res =
        tulpa_sghmc::run_sgld_sampler(q_init, *data, *layout, cfg);

    copy_sg_sampler_result(res.samples, res.log_lik, res.epsilon_history,
                           res.success, res.error_msg, result_out);
}

// ============================================================================
// MCLMC / MAMCLMC shim
// ============================================================================
//
// Both samplers internally take a std::function log_prob_grad callback that
// cannot cross a DLL boundary. The shim builds it inside tulpa from the
// existing compute_log_post + compute_gradient (see mclmc_modeldata.h),
// then dispatches to mclmc_sample / mamclmc_sample based on the `adjusted`
// flag. Result reuses SGSamplerShimResult — the flat shape fits.

extern "C" void tulpa_mclmc_fit_impl(
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    const double* init,
    int n_params,
    int n_iter,
    int n_warmup,
    double step_size,
    int L,
    unsigned int seed,
    int adjusted,
    int verbose,
    tulpa::SGSamplerShimResult* result_out
) {
    std::vector<double> q_init(init, init + n_params);

    tulpa_mclmc::MCLMCConfig cfg;
    cfg.n_iter   = n_iter;
    cfg.n_warmup = n_warmup;
    cfg.step_size = step_size;
    cfg.L        = L;
    cfg.adjusted = adjusted;
    cfg.seed     = seed;
    cfg.verbose  = (verbose != 0);
    // mass_diag stays empty (identity); extending the shim signature would
    // require an ABI bump.

    tulpa_mclmc::MCLMCFitResult res =
        tulpa_mclmc::run_mclmc_sampler(q_init, *data, *layout, cfg);

    copy_sg_sampler_result(res.samples, res.log_lik, res.epsilon_history,
                           res.success, res.error_msg, result_out);
}

// ============================================================================
// SMC shim
// ============================================================================
//
// Drives tulpa::run_smc_sampler. The mutation kernel is pluggable: if
// the model package passes nullptr, tulpa uses its built-in random-walk
// Metropolis kernel scaled by 1 / sqrt(beta) targeting
// log_prior + beta * log_lik.

extern "C" void tulpa_smc_fit_impl(
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    const double* init,
    int n_params,
    int n_particles,
    int n_mcmc_steps,
    double ess_threshold,
    double prior_sigma,
    tulpa::SmcMutationFn mutation,
    void* user_data,
    unsigned int seed,
    int verbose,
    tulpa::SMCShimResult* result_out
) {
    std::vector<double> init_vec(init, init + n_params);

    tulpa::SMCConfig cfg;
    cfg.n_particles    = n_particles;
    cfg.n_mcmc_steps   = n_mcmc_steps;
    cfg.ess_threshold  = ess_threshold;
    cfg.prior_sigma    = prior_sigma;
    cfg.seed           = seed;
    cfg.verbose        = (verbose != 0);

    tulpa::SMCDriverResult res = tulpa::run_smc_sampler(
        init_vec, *data, *layout, cfg, mutation, user_data);

    int N = (int)res.particles.size();
    int P = (N > 0) ? (int)res.particles[0].size() : n_params;

    result_out->n_particles  = N;
    result_out->n_params     = P;
    result_out->log_evidence = res.log_evidence;
    result_out->success      = res.success ? 1 : 0;

    std::strncpy(result_out->error_msg, res.error_msg.c_str(),
                 sizeof(result_out->error_msg) - 1);
    result_out->error_msg[sizeof(result_out->error_msg) - 1] = '\0';

    if (N > 0 && P > 0) {
        result_out->particles = new double[(size_t)N * (size_t)P];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < P; j++) {
                result_out->particles[(size_t)i * P + j] = res.particles[i][j];
            }
        }
    } else {
        result_out->particles = new double[1];
    }

    {
        int W = (int)res.log_weights.size();
        result_out->log_weights = new double[W > 0 ? W : 1];
        for (int i = 0; i < W; i++) result_out->log_weights[i] = res.log_weights[i];
    }
}

// ============================================================================
// Registration: called from tulpa_init.cpp::tulpa_register_callables.
// ============================================================================

void tulpa_register_shims(DllInfo* dll) {
    R_RegisterCCallable("tulpa", "tulpa_laplace_mode_dense",
        (DL_FUNC)&tulpa_laplace_mode_dense_impl);
    R_RegisterCCallable("tulpa", "tulpa_laplace_mode_spatial",
        (DL_FUNC)&tulpa_laplace_mode_spatial_impl);
    R_RegisterCCallable("tulpa", "tulpa_laplace_mode_dense_multi_re",
        (DL_FUNC)&tulpa_laplace_mode_dense_multi_re_impl);
    R_RegisterCCallable("tulpa", "tulpa_laplace_mode_bym2",
        (DL_FUNC)&tulpa_laplace_mode_bym2_impl);
    R_RegisterCCallable("tulpa", "tulpa_laplace_mode_gp",
        (DL_FUNC)&tulpa_laplace_mode_gp_impl);
    R_RegisterCCallable("tulpa", "tulpa_laplace_mode_multiscale_gp",
        (DL_FUNC)&tulpa_laplace_mode_multiscale_gp_impl);
    R_RegisterCCallable("tulpa", "tulpa_laplace_mode_multiscale_temporal",
        (DL_FUNC)&tulpa_laplace_mode_multiscale_temporal_impl);
    R_RegisterCCallable("tulpa", "tulpa_laplace_mode_rsr",
        (DL_FUNC)&tulpa_laplace_mode_rsr_impl);

    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_icar",
        (DL_FUNC)&tulpa_nested_laplace_icar_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_bym2",
        (DL_FUNC)&tulpa_nested_laplace_bym2_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_car_proper",
        (DL_FUNC)&tulpa_nested_laplace_car_proper_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_rw1",
        (DL_FUNC)&tulpa_nested_laplace_rw1_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_rw2",
        (DL_FUNC)&tulpa_nested_laplace_rw2_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_ar1",
        (DL_FUNC)&tulpa_nested_laplace_ar1_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_nngp",
        (DL_FUNC)&tulpa_nested_laplace_nngp_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_hsgp",
        (DL_FUNC)&tulpa_nested_laplace_hsgp_impl);
    R_RegisterCCallable("tulpa", "tulpa_nested_laplace_spde",
        (DL_FUNC)&tulpa_nested_laplace_spde_impl);

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
