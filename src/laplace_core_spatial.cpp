// laplace_core_spatial.cpp
// Spatial / BYM2 Laplace mode finders + their R exports.
// Split from laplace_core.cpp on 2026-05-02.
//
// Each mode finder still uses laplace_newton_solve (in laplace_newton.h) and the
// scatter helpers from laplace_scatter.h — see laplace_core.cpp for the shared
// library context.

#include "laplace_core.h"
#include "laplace_cholesky.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include "laplace_scatter.h"
#include "laplace_spatial_priors.h"
#include "icar_kernel.h"           // count_graph_components
#include "laplace_spec_fit.h"     // spec-solver marshalling for the single-point fits
#include "latent_block.h"
#include "linalg_fast.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;


// =====================================================================
// R exports (call into tulpa:: functions defined above)
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_spatial(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors, double tau_spatial,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    int force_sparse = 0
) {
    // Fixed effects + optional iid RE + a single ICAR latent block, through the
    // unified spec solver (the family-enum laplace_mode_spatial was retired in
    // B2-live). The block is built exactly as cpp_nested_laplace_icar's, so the
    // mode + log-marginal match the nested kernel at one tau cell.
    const int N = y.size();
    const int p = X.ncol();
    const bool has_re = n_re_groups > 0;
    std::vector<int> re_group;
    if (has_re) {
        re_group.resize(N);
        for (int i = 0; i < N; i++) re_group[i] = (int)re_idx[i];
    }
    const int block_start = p + (has_re ? n_re_groups : 0);

    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);

    tulpa::SpecFamilyInputs in;
    tulpa::build_spec_family_inputs(
        in, y, n, X, re_group, n_re_groups, sigma_re, family, phi,
        /*sigma_beta=*/100.0, /*n_block_latent=*/n_spatial_units,
        /*weights=*/nullptr, offset.empty() ? nullptr : offset.data());

    // One constant null direction per connected component; the sampler and the
    // Polya-Gamma kernels derive the same count from the same adjacency.
    const int n_comp = tulpa::count_graph_components(
        n_spatial_units, adj_row_ptr.begin(), adj_col_idx.begin());

    tulpa::LatentBlock block;
    block.start = block_start;
    block.size  = n_spatial_units;
    block.idx   = [&](int i, int /*k_arm*/) { return spatial_idx[i]; };
    block.d_fac = [](int) { return 1.0; };
    block.add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                          const Rcpp::NumericVector& x, int /*k*/) {
        tulpa::add_icar_prior(grad, H, x, block_start, n_spatial_units,
                               tau_spatial, adj_row_ptr, adj_col_idx,
                               n_neighbors, n_comp);
    };
    block.log_prior = [&](const Rcpp::NumericVector& x, int /*k*/) {
        return tulpa::log_prior_icar(x, block_start, n_spatial_units, tau_spatial,
                                       adj_row_ptr, adj_col_idx, n_neighbors,
                                       n_comp);
    };
    block.center = [&](Rcpp::NumericVector& x) {
        return tulpa::center_intercept(x, block_start, n_spatial_units);
    };
    std::vector<tulpa::LatentBlock> blocks{ block };

    std::vector<double> params(in.layout.total_params, 0.0);
    if (has_re) params[in.layout.log_sigma_re_idx] = std::log(sigma_re);
    // Warm start: the [beta | RE | spatial] latent occupies the leading
    // block_start + n_spatial_units params (contiguous before log_sigma_re).
    if (x_init_nullable.isNotNull()) {
        Rcpp::NumericVector xi = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
        const int n_lat = block_start + n_spatial_units;
        for (int j = 0; j < n_lat && j < (int)xi.size(); j++) params[j] = xi[j];
    }

    tulpa::LaplaceResult res = tulpa::laplace_mode_spec_dense_solve(
        in.data, in.layout, params, in.re_group, max_iter, tol, n_threads,
        &blocks, /*k_grid=*/0, /*beta_prior=*/nullptr,
        /*return_re_cov=*/false, /*sparse_override=*/force_sparse);
    return tulpa::laplace_result_to_list(res);
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_bym2(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double sigma_spatial, double rho, double scale_factor,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    int force_sparse = 0
) {
    // Fixed effects + optional iid RE + BYM2's two latent blocks (phi:
    // ICAR-structured & centered, theta: IID) through the unified spec solver
    // (the family-enum laplace_mode_bym2 was retired in B2-live). Blocks built
    // exactly as cpp_nested_laplace_bym2's, with the grid-dependent d_fac.
    const int N = y.size();
    const int p = X.ncol();
    const bool has_re = n_re_groups > 0;
    std::vector<int> re_group;
    if (has_re) {
        re_group.resize(N);
        for (int i = 0; i < N; i++) re_group[i] = (int)re_idx[i];
    }
    const int phi_start   = p + (has_re ? n_re_groups : 0);
    const int theta_start = phi_start + n_spatial_units;

    // One constant null direction per connected component, as for plain ICAR.
    const int n_comp = tulpa::count_graph_components(
        n_spatial_units, adj_row_ptr.begin(), adj_col_idx.begin());

    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);

    tulpa::SpecFamilyInputs in;
    tulpa::build_spec_family_inputs(
        in, y, n, X, re_group, n_re_groups, sigma_re, family, phi,
        /*sigma_beta=*/100.0, /*n_block_latent=*/2 * n_spatial_units,
        /*weights=*/nullptr, offset.empty() ? nullptr : offset.data());

    // phi block: ICAR-structured, d = sigma * sqrt(rho) * scale_factor, centered.
    tulpa::LatentBlock phi_block;
    phi_block.start = phi_start;
    phi_block.size  = n_spatial_units;
    phi_block.idx   = [&](int i, int /*k_arm*/) { return spatial_idx[i]; };
    phi_block.d_fac = [&, scale_factor](int) {
        return sigma_spatial * std::sqrt(rho + 1e-10) * scale_factor;
    };
    phi_block.add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                              const Rcpp::NumericVector& x, int /*k*/) {
        tulpa::add_icar_prior(grad, H, x, phi_start, n_spatial_units, 1.0,
                               adj_row_ptr, adj_col_idx, n_neighbors, n_comp);
    };
    phi_block.log_prior = [&](const Rcpp::NumericVector& x, int /*k*/) {
        // Structured ICAR component (tau = 1); shares the quadratic form and the
        // augmentation with add_icar_prior so the objective stays consistent
        // with the gradient, instead of re-deriving them inline.
        return tulpa::log_prior_icar_structured(x, phi_start, n_spatial_units,
                                                /*tau=*/1.0, adj_row_ptr,
                                                adj_col_idx, n_neighbors,
                                                n_comp);
    };
    phi_block.center = [&](Rcpp::NumericVector& x) {
        return tulpa::center_intercept(x, phi_start, n_spatial_units);
    };

    // theta block: IID, d = sigma * sqrt(1 - rho), no centering.
    tulpa::LatentBlock theta_block;
    theta_block.start = theta_start;
    theta_block.size  = n_spatial_units;
    theta_block.idx   = [&](int i, int /*k_arm*/) { return spatial_idx[i]; };
    theta_block.d_fac = [&](int) {
        return sigma_spatial * std::sqrt(1.0 - rho + 1e-10);
    };
    theta_block.add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                                const Rcpp::NumericVector& x, int /*k*/) {
        for (int s = 0; s < n_spatial_units; s++) {
            int idx = theta_start + s;
            grad[idx] -= x[idx];
            H[idx][idx] += 1.0;
        }
    };
    theta_block.log_prior = [&](const Rcpp::NumericVector& x, int /*k*/) {
        double lp = 0.0;
        for (int s = 0; s < n_spatial_units; s++) {
            lp -= 0.5 * x[theta_start + s] * x[theta_start + s];
        }
        lp -= 0.5 * n_spatial_units * std::log(2.0 * M_PI);
        return lp;
    };

    std::vector<tulpa::LatentBlock> blocks{ phi_block, theta_block };

    std::vector<double> params(in.layout.total_params, 0.0);
    if (has_re) params[in.layout.log_sigma_re_idx] = std::log(sigma_re);

    tulpa::LaplaceResult res = tulpa::laplace_mode_spec_dense_solve(
        in.data, in.layout, params, in.re_group, max_iter, tol, n_threads,
        &blocks, /*k_grid=*/0, /*beta_prior=*/nullptr,
        /*return_re_cov=*/false, /*sparse_override=*/force_sparse);
    return tulpa::laplace_result_to_list(res);
}
