// nested_laplace.cpp
// Fixed-hyperparameter (single-point) Laplace fits of a spatial block whose
// conditional kernel needs a family-specific factory: proper CAR and HSGP.
// Each builds the block the joint driver builds for the same family at a
// one-cell grid and solves it with the shared dense spec solver
// (laplace_fit_blocks), so the mode and log marginal equal a one-cell nested
// fit. Every nested-Laplace grid fit runs on the joint driver
// (nested_laplace_joint_multi.cpp); SPDE has its own entry in spde_laplace.cpp.

#include "bym2_mixing.h"           // BYM2_RHO_EPS + the mixing amplitudes
#include "tulpa_stop.h"
#include "laplace_core.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include "laplace_spatial_priors.h"
#include "icar_kernel.h"           // count_graph_components
#include "laplace_temporal_priors.h"
#include "latent_block.h"
#include "nl_cell_cache.h"
#include "nested_laplace_grid.h"
#include "areal_input_check.h"
#include "nested_laplace_joint_core.h"      // ParsedArm / JointArm (single-arm joint setup)
#include "nested_laplace_joint_multi.h"     // run_multi_block_nested_laplace_joint_sparse_impl
#include "hmc_car_proper.h"
#include "gpu_nngp_laplace.h"
#include "hmc_hsgp_kernels.h"  // Eigen-free spectral density only
#include "hsgp_block_factory.h"             // make_hsgp_block
#include "laplace_spec_fit.h"               // build_spec_family_inputs + laplace_mode_spec_dense_solve
#include "laplace_block_fit.h"              // laplace_fit_blocks
#include "nl_entry_inputs.h"
#include "nngp_block_factory.h"             // make_nngp_block
#include "unit_precision_block.h"           // set_unit_precision_block_priors
#include "car_proper_block.h"                // set_car_proper_block_priors
#include "field_block_priors.h"              // set_icar / set_temporal_block_priors
#include "sparse_hessian.h"     // SparseHessianBuilder
#include <Rcpp.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// =====================================================================
// Proper-CAR block factory
// =====================================================================
// The proper-CAR LatentBlock: dense add_prior / log_prior and the sparse
// add_prior_pattern / add_prior_sparse, filled by set_car_proper_block_priors.
//
// Lifetime: the returned blocks' callbacks capture the Rcpp argument vectors by
// reference (mirroring make_<x>_spatial_ops). The CALLER must keep those vectors
// alive across the outer-grid solve -- every caller does, they are function-local
// and the solve runs before return.
namespace {

inline std::vector<tulpa::LatentBlock> make_car_proper_latent_blocks(
    int start, int n_units,
    const Rcpp::IntegerVector& spatial_idx,
    const Rcpp::NumericVector& tau_grid,
    const Rcpp::NumericVector& rho_grid,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
) {
    tulpa::LatentBlock block;
    block.start = start;
    block.size  = n_units;
    block.idx   = [&spatial_idx](int i, int /*k_arm*/) { return spatial_idx[i]; };
    block.d_fac = [](int) { return 1.0; };
    // The five prior callbacks are the shared proper-CAR set; this entry's
    // (tau, rho) come straight off its own two grids. No centerer: Q is full
    // rank, so the field carries no null direction to identify against the
    // intercept. One rule across every driver -- center an intrinsic prior
    // (ICAR, BYM2's structured component, RW1 / RW2), leave a full-rank one
    // (proper CAR, AR1, IID) at its mode -- and `block_centered` on the fit
    // reports which a block took.
    tulpa::set_car_proper_block_priors(
        block, start, n_units,
        [tau_grid](int k) { return tau_grid[k]; },
        [rho_grid](int k) { return rho_grid[k]; },
        adj_row_ptr, adj_col_idx, n_neighbors);
    return { block };
}

} // namespace

// Fixed-hyperparameter (single-point) proper-CAR Laplace: the
// make_car_proper_latent_blocks factory at a one-cell (tau, rho) grid and the
// shared dense spec solver. Full-rank Q(rho) = D - rho W, with log|Q(rho)| from
// the block's prep. Routed from dispatch_laplace_spatial /
// tulpa_laplace(spatial = list(type = "car_proper", ...)).
// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_car_proper(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double tau_spatial, double rho,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    bool compute_skew = false,
    Rcpp::Nullable<Rcpp::IntegerVector> skew_idx = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> weights_nullable = R_NilValue
) {
    tulpa::check_areal_inputs(adj_row_ptr, adj_col_idx, n_neighbors,
                              spatial_idx, static_cast<int>(y.size()),
                              n_spatial_units,
                              "cpp_laplace_fit_car_proper");
    const int N = y.size();
    const int p = X.ncol();
    const bool has_re = n_re_groups > 0;
    std::vector<int> re_group =
        tulpa::as_re_group_vec(re_idx, n_re_groups, N);
    const int block_start = p + (has_re ? n_re_groups : 0);
    std::vector<double> offset = tulpa::as_obs_vec(offset_nullable, N, "offset");
    std::vector<double> weights =
        tulpa::as_obs_vec(weights_nullable, N, "weights");

    // One-cell (tau, rho) grids. The block's callbacks hold their own handles
    // on these (and run the log|D - rho W| determinant in block.prep).
    Rcpp::NumericVector tau_grid = Rcpp::NumericVector::create(tau_spatial);
    Rcpp::NumericVector rho_grid = Rcpp::NumericVector::create(rho);
    std::vector<tulpa::LatentBlock> blocks = make_car_proper_latent_blocks(
        block_start, n_spatial_units, spatial_idx, tau_grid, rho_grid,
        adj_row_ptr, adj_col_idx, n_neighbors);

    return tulpa::laplace_fit_blocks(
        y, n, X, re_group, n_re_groups, sigma_re, family, phi, offset, weights,
        blocks, /*n_block_latent=*/n_spatial_units, x_init_nullable,
        max_iter, tol, n_threads, /*sparse_override=*/0,
        compute_skew, skew_idx);
}

// Hilbert-space GP: f_i = sum_j Phi_ij * sqrt(S(lambda_j; sigma2, ell)) * beta_j
// with beta ~ N(0, I_M). Phi (basis) and lambda_j (eigenvalues) are FIXED
// across the grid; only the spectral diagonal sqrt(S) varies per grid
// point. Inner solve reduces to a fixed-design regression on the
// rescaled basis B = Phi * diag(sqrt(S)), S the spectral density of the
// field's kernel (hsgp_spectral.h).

// Fixed-hyperparameter (single-point) HSGP Laplace: the make_hsgp_block
// factory at a one-row (sigma2, lengthscale) grid and the shared dense spec
// solver, which scatters DENSE_BASIS blocks. Routed from
// dispatch_laplace_spatial / tulpa_laplace(spatial = spatial_hsgp(...)).
// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_hsgp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix phi_basis, Rcpp::NumericVector lambda_eig,
    int kernel,
    double sigma2, double lengthscale,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    bool compute_skew = false,
    Rcpp::Nullable<Rcpp::IntegerVector> skew_idx = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> weights_nullable = R_NilValue
) {
    const int N = y.size();
    const int p = X.ncol();
    const int M = phi_basis.ncol();
    if (phi_basis.nrow() != N)
        tulpa::stop("phi_basis must have N rows (one per observation)");
    if (lambda_eig.size() != M)
        tulpa::stop("lambda_eig must have length ncol(phi_basis)");
    const bool has_re = n_re_groups > 0;
    std::vector<int> re_group =
        tulpa::as_re_group_vec(re_idx, n_re_groups, N);
    const int block_start = p + (has_re ? n_re_groups : 0);
    std::vector<double> offset = tulpa::as_obs_vec(offset_nullable, N, "offset");
    std::vector<double> weights =
        tulpa::as_obs_vec(weights_nullable, N, "weights");

    // One-row (sigma2, lengthscale) grid; make_hsgp_block reads the amplitude /
    // lengthscale from theta_grid(k, axis) in block.prep, so the row must
    // outlive the solve.
    Rcpp::NumericMatrix theta_grid(1, 2);
    theta_grid(0, 0) = sigma2;
    theta_grid(0, 1) = lengthscale;
    Rcpp::List phi_per_arm = Rcpp::List::create(phi_basis);
    Rcpp::IntegerVector n_obs_per_arm = Rcpp::IntegerVector::create(N);
    std::vector<tulpa::LatentBlock> blocks;
    blocks.push_back(tulpa::make_hsgp_block(
        block_start, M, phi_per_arm, n_obs_per_arm, /*n_arms=*/1,
        /*block_index=*/0, lambda_eig,
        /*axis_sigma2=*/0, /*axis_ell=*/1, theta_grid,
        tulpa::priors::hsgp_kernel_from_code(kernel, "cpp_laplace_fit_hsgp")));

    return tulpa::laplace_fit_blocks(
        y, n, X, re_group, n_re_groups, sigma_re, family, phi, offset, weights,
        blocks, /*n_block_latent=*/blocks.back().size, x_init_nullable,
        max_iter, tol, n_threads, /*sparse_override=*/0,
        compute_skew, skew_idx);
}
