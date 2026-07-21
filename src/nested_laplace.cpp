// nested_laplace.cpp
// Nested Laplace outer-grid drivers for latent Gaussian backends.
//
// Two flavours of backend live here:
//
// 1. LatentBlock-based backends (ICAR, BYM2, CAR_proper, RW1, RW2, AR1): each
//    latent prior block is described by a tulpa::LatentBlock (idx, d_fac,
//    add_prior, log_prior, prep, center). The generic driver
//    run_multi_block_nested_laplace takes a std::vector<LatentBlock> and
//    handles compute_eta, scatter (incl. cross-block Hessian terms), center,
//    and log_prior in one place. Single-block kernels (ICAR/RW1/RW2/AR1/
//    CAR_proper) build one block with d_fac = 1.0; BYM2 builds two blocks
//    (phi structured + theta IID) sharing spatial_idx with grid-dependent
//    d_fac = sigma_k * sqrt(rho_k) * scale_factor / sigma_k * sqrt(1-rho_k).
//
// 2. Non-LatentBlock backends (NNGP, HSGP): batch scatter or basis-rescaled
//    design — the eta/scatter shape doesn't fit the LatentBlock abstraction,
//    so each builds its own solve_at_theta lambda directly.
//
// SPDE has its own Q-rebuild pattern and lives in spde_laplace.cpp.
// Outer-grid loop + warm-starting + result aggregation live in
// nested_laplace_grid.h.

#include "laplace_core.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include "laplace_scatter.h"
#include "laplace_spatial_priors.h"
#include "laplace_temporal_priors.h"
#include "latent_block.h"
#include "nl_cell_cache.h"
#include "nested_laplace_grid.h"
#include "nested_laplace_joint_core.h"      // ParsedArm / JointArm (single-arm joint setup)
#include "nested_laplace_joint_multi.h"     // run_multi_block_nested_laplace_joint_sparse_impl
#include "nested_laplace_multi.h"  // accumulate_latent_cross_terms, run_multi_block_nested_laplace
#include "hmc_car_proper.h"
#include "gpu_nngp_laplace.h"
#include "hmc_hsgp_kernels.h"  // Eigen-free spectral density only
#include "hsgp_block_factory.h"             // make_hsgp_block
#include "laplace_spec_fit.h"               // build_spec_family_inputs + laplace_mode_spec_dense_solve
#include "nngp_block_factory.h"             // make_nngp_block
#include "sparse_hessian.h"     // SparseHessianBuilder + laplace_newton_solve_sparse
#include <Rcpp.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// Convenience: extract optional warm-start mode from an Rcpp Nullable into an
// owning NumericVector (empty vector when null). Used by every indexed shim.
inline Rcpp::NumericVector unwrap_x_init(
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable
) {
    if (x_init_nullable.isNotNull()) {
        return Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }
    return Rcpp::NumericVector();
}

} // namespace

// Bring the multi-block driver into the anonymous-namespace caller scope so
// the per-kernel Rcpp wrappers below can dispatch without a tulpa:: prefix.
namespace { using tulpa::run_multi_block_nested_laplace; }

// =====================================================================
// Areal spatial-block factories (single source of truth)
// =====================================================================
// One factory per areal spatial family returns the LatentBlock(s) the
// outer-grid drivers consume. Both the pure-spatial entries
// (cpp_nested_laplace_<x> -> run_multi_block_nested_laplace) and the
// spatio-temporal entries (cpp_nested_laplace_st_<x>, which stack a temporal
// block alongside) build their spatial side here, so the per-family block
// wiring -- dense add_prior / log_prior / center AND the sparse
// add_prior_pattern / add_prior_sparse the joint-sparse driver needs -- lives
// once.
//
// Lifetime: the returned blocks' callbacks capture the Rcpp argument vectors by
// reference (mirroring make_<x>_spatial_ops). The CALLER must keep those vectors
// alive across the outer-grid solve -- every caller does, they are function-local
// and the solve runs before return.
namespace {

inline std::vector<tulpa::LatentBlock> make_icar_latent_blocks(
    int start, int n_units,
    const Rcpp::IntegerVector& spatial_idx,
    const Rcpp::NumericVector& tau_grid,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
) {
    tulpa::LatentBlock block;
    block.start = start;
    block.size  = n_units;
    block.idx   = [&spatial_idx](int i, int /*k_arm*/) { return spatial_idx[i]; };
    block.d_fac = [](int) { return 1.0; };
    block.add_prior = [start, n_units, &tau_grid,
                       &adj_row_ptr, &adj_col_idx, &n_neighbors]
                      (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                       const Rcpp::NumericVector& x, int k) {
        tulpa::add_icar_prior(grad, H, x, start, n_units, tau_grid[k],
                               adj_row_ptr, adj_col_idx, n_neighbors);
    };
    block.log_prior = [start, n_units, &tau_grid,
                       &adj_row_ptr, &adj_col_idx, &n_neighbors]
                      (const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_icar(x, start, n_units, tau_grid[k],
                                       adj_row_ptr, adj_col_idx, n_neighbors);
    };
    block.center = [start, n_units](Rcpp::NumericVector& x) {
        return tulpa::center_intercept(x, start, n_units);
    };
    block.add_prior_pattern = [start, n_units, &adj_row_ptr, &adj_col_idx]
                              (std::vector<std::pair<int,int>>& out) {
        tulpa::add_icar_pattern(out, start, n_units, adj_row_ptr, adj_col_idx);
    };
    block.add_prior_sparse = [start, n_units, &tau_grid,
                              &adj_row_ptr, &adj_col_idx, &n_neighbors]
                             (tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                              const Rcpp::NumericVector& x, int k) {
        tulpa::add_icar_prior_sparse(grad, H, x, start, n_units, tau_grid[k],
                                       adj_row_ptr, adj_col_idx, n_neighbors);
    };
    return { block };
}

inline std::vector<tulpa::LatentBlock> make_car_proper_latent_blocks(
    int start, int n_units,
    const Rcpp::IntegerVector& spatial_idx,
    const Rcpp::NumericVector& tau_grid,
    const Rcpp::NumericVector& rho_grid,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
) {
    // Adjacency CSR copied into std::vector for the dense log|Q(rho)| helper;
    // owned by shared_ptr so the prep / log_prior closures outlive this factory.
    auto adj_rp_v = std::make_shared<std::vector<int>>(
        adj_row_ptr.begin(), adj_row_ptr.end());
    auto adj_ci_v = std::make_shared<std::vector<int>>(
        adj_col_idx.begin(), adj_col_idx.end());
    auto n_nbr_v  = std::make_shared<std::vector<int>>(
        n_neighbors.begin(), n_neighbors.end());
    // Per-cell log|Q(rho)| (nl_cell_cache.h). The single-arm kernels run the
    // outer grid serially today, but nothing at the type level prevents this
    // block from meeting a parallel grid; cell-keyed state costs nothing and
    // removes the hazard.
    auto log_det_Q_rho = std::make_shared<tulpa::NlCellCache<double>>();

    tulpa::LatentBlock block;
    block.start = start;
    block.size  = n_units;
    block.idx   = [&spatial_idx](int i, int /*k_arm*/) { return spatial_idx[i]; };
    block.d_fac = [](int) { return 1.0; };
    block.prep  = [n_units, &rho_grid, adj_rp_v, adj_ci_v, n_nbr_v,
                   log_det_Q_rho](int k) -> bool {
        std::vector<double> Qmat = tulpa_car_proper::compute_car_precision(
            n_units, *adj_rp_v, *adj_ci_v, *n_nbr_v, rho_grid[k]);
        double ld_val = tulpa_car_proper::car_log_det(n_units, Qmat);
        log_det_Q_rho->claim() = ld_val;
        log_det_Q_rho->publish(k);
        return std::isfinite(ld_val);
    };
    block.add_prior = [start, n_units, &tau_grid, &rho_grid,
                       &adj_row_ptr, &adj_col_idx, &n_neighbors]
                      (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                       const Rcpp::NumericVector& x, int k) {
        tulpa::add_car_proper_prior(grad, H, x, start, n_units,
                                     tau_grid[k], rho_grid[k],
                                     adj_row_ptr, adj_col_idx, n_neighbors);
    };
    block.log_prior = [start, n_units, &tau_grid, &rho_grid,
                       &adj_row_ptr, &adj_col_idx, &n_neighbors, log_det_Q_rho]
                      (const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_car_proper(x, start, n_units,
                                             tau_grid[k], rho_grid[k],
                                             log_det_Q_rho->find(k),
                                             adj_row_ptr, adj_col_idx, n_neighbors);
    };
    block.center = [start, n_units](Rcpp::NumericVector& x) {
        return tulpa::center_intercept(x, start, n_units);
    };
    block.add_prior_pattern = [start, n_units, &adj_row_ptr, &adj_col_idx]
                              (std::vector<std::pair<int,int>>& out) {
        tulpa::add_car_pattern(out, start, n_units, adj_row_ptr, adj_col_idx);
    };
    block.add_prior_sparse = [start, n_units, &tau_grid, &rho_grid,
                              &adj_row_ptr, &adj_col_idx, &n_neighbors]
                             (tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                              const Rcpp::NumericVector& x, int k) {
        tulpa::add_car_proper_prior_sparse(grad, H, x, start, n_units,
                                             tau_grid[k], rho_grid[k],
                                             adj_row_ptr, adj_col_idx, n_neighbors);
    };
    return { block };
}

// BYM2: two blocks (phi structured ICAR + theta IID) sharing spatial_idx. The
// structured reparameterisation lives in the per-block d_fac:
//   eta_s = sigma_k * (sqrt(rho_k) * scale_factor * phi_s + sqrt(1 - rho_k) * theta_s)
// so phi has a bare ICAR prior (tau = 1) and theta is N(0, I); the multi-block
// driver mixes d_fac_b(k) * x[idx_b] into eta. The block exposes both the dense
// and the sparse callbacks the joint-sparse path needs.
inline std::vector<tulpa::LatentBlock> make_bym2_latent_blocks(
    int start, int n_s,
    const Rcpp::IntegerVector& spatial_idx,
    double scale_factor,
    const Rcpp::NumericVector& sigma_spatial_grid,
    const Rcpp::NumericVector& rho_grid,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
) {
    int phi_start   = start;
    int theta_start = start + n_s;

    tulpa::LatentBlock phi_block;
    phi_block.start = phi_start;
    phi_block.size  = n_s;
    phi_block.idx   = [&spatial_idx](int i, int /*k_arm*/) { return spatial_idx[i]; };
    phi_block.d_fac = [&sigma_spatial_grid, &rho_grid, scale_factor](int k) {
        return sigma_spatial_grid[k] * std::sqrt(rho_grid[k] + 1e-10) * scale_factor;
    };
    phi_block.add_prior = [phi_start, n_s, &adj_row_ptr, &adj_col_idx, &n_neighbors]
                          (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                           const Rcpp::NumericVector& x, int /*k*/) {
        tulpa::add_icar_prior(grad, H, x, phi_start, n_s, 1.0,
                               adj_row_ptr, adj_col_idx, n_neighbors);
    };
    phi_block.log_prior = [phi_start, n_s, &adj_row_ptr, &adj_col_idx, &n_neighbors]
                          (const Rcpp::NumericVector& x, int /*k*/) {
        // Structured ICAR component (tau = 1); shares the quadratic form and the
        // sum-to-zero penalty with add_icar_prior so the objective stays
        // consistent with the gradient, instead of re-deriving them inline.
        return tulpa::log_prior_icar_structured(x, phi_start, n_s, /*tau=*/1.0,
                                                adj_row_ptr, adj_col_idx, n_neighbors);
    };
    phi_block.center = [phi_start, n_s](Rcpp::NumericVector& x) {
        return tulpa::center_intercept(x, phi_start, n_s);
    };
    phi_block.add_prior_pattern = [phi_start, n_s, &adj_row_ptr, &adj_col_idx]
                                  (std::vector<std::pair<int,int>>& out) {
        tulpa::add_icar_pattern(out, phi_start, n_s, adj_row_ptr, adj_col_idx);
    };
    phi_block.add_prior_sparse = [phi_start, n_s, &adj_row_ptr, &adj_col_idx, &n_neighbors]
                                 (tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                                  const Rcpp::NumericVector& x, int /*k*/) {
        tulpa::add_icar_prior_sparse(grad, H, x, phi_start, n_s, 1.0,
                                       adj_row_ptr, adj_col_idx, n_neighbors);
    };

    tulpa::LatentBlock theta_block;
    theta_block.start = theta_start;
    theta_block.size  = n_s;
    theta_block.idx   = [&spatial_idx](int i, int /*k_arm*/) { return spatial_idx[i]; };
    theta_block.d_fac = [&sigma_spatial_grid, &rho_grid](int k) {
        return sigma_spatial_grid[k] * std::sqrt(1.0 - rho_grid[k] + 1e-10);
    };
    theta_block.add_prior = [theta_start, n_s]
                            (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                             const Rcpp::NumericVector& x, int /*k*/) {
        for (int s = 0; s < n_s; s++) {
            int idx = theta_start + s;
            grad[idx] -= x[idx];
            H[idx][idx] += 1.0;
        }
    };
    theta_block.log_prior = [theta_start, n_s](const Rcpp::NumericVector& x, int /*k*/) {
        double lp = 0.0;
        for (int s = 0; s < n_s; s++) {
            lp -= 0.5 * x[theta_start + s] * x[theta_start + s];
        }
        lp -= 0.5 * n_s * std::log(2.0 * M_PI);
        return lp;
    };
    // theta_block.center left empty (IID, identifiability anchored by intercept).
    theta_block.add_prior_sparse = [theta_start, n_s]
                                   (tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                                    const Rcpp::NumericVector& x, int /*k*/) {
        for (int s = 0; s < n_s; s++) {
            int idx = theta_start + s;
            grad[idx] -= x[idx];
            H.add(idx, idx, 1.0);
        }
    };
    // theta has no off-diagonal prior pattern (the pattern builder contributes
    // the diagonal for every block index unconditionally).

    return { phi_block, theta_block };
}

} // namespace

// =====================================================================
// Nested Laplace: ICAR (1D grid over tau_spatial)
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_icar(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::NumericVector tau_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    std::string checkpoint_path = ""
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int spatial_start = p + n_re_groups;

    std::vector<tulpa::LatentBlock> blocks = make_icar_latent_blocks(
        spatial_start, n_spatial_units, spatial_idx, tau_grid,
        adj_row_ptr, adj_col_idx, n_neighbors);

    tulpa::Fingerprint sfp;
    sfp.fold_str("icar");
    sfp.fold_pod(n_spatial_units);
    if (adj_row_ptr.size()) sfp.fold(adj_row_ptr.begin(),
                                     (std::size_t)adj_row_ptr.size() * sizeof(int));
    if (adj_col_idx.size()) sfp.fold(adj_col_idx.begin(),
                                     (std::size_t)adj_col_idx.size() * sizeof(int));
    auto ckpt = tulpa::make_nl_grid_checkpoint(
        checkpoint_path, sfp.value(), max_iter, tol, y, n, X, re_idx,
        n_re_groups, sigma_re, family, phi, {tau_grid});

    Rcpp::List out = run_multi_block_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        blocks,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        store_Q, /*n_threads_outer=*/1, /*prune_tol=*/0.0,
        /*ext_spec=*/nullptr, /*ext_response=*/nullptr,
        /*progress=*/nullptr, ckpt.get()
    );
    out["tau_grid"] = tau_grid;
    return out;
}

// =====================================================================
// Nested Laplace: BYM2 (2D grid over (sigma_spatial, rho))
// =====================================================================
// BYM2 reparameterization: phi structured (ICAR) + theta IID,
//   eta_s = sigma * (sqrt(rho) * scale_factor * phi + sqrt(1 - rho) * theta)
//
// This kernel routes through run_multi_block_nested_laplace with phi and theta
// as two LatentBlocks. The multi-block compute_eta accumulates
// `d_fac_b(k) * x[idx_b]` per block, which re-associates the FP multiplication
// relative to the single-expression `sigma_k * (sqrt_rho * x * scale_factor +
// sqrt_1_rho * x)` factoring, so log_marginal can differ at ~1e-8 (well below
// the 1e-6 Newton tolerance).

// [[Rcpp::export]]
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
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    std::string checkpoint_path = ""
) {
    int n_grid = sigma_spatial_grid.size();
    int N = y.size();
    int p = X.ncol();
    std::vector<tulpa::LatentBlock> blocks = make_bym2_latent_blocks(
        p + n_re_groups, n_spatial_units, spatial_idx, scale_factor,
        sigma_spatial_grid, rho_grid,
        adj_row_ptr, adj_col_idx, n_neighbors);

    tulpa::Fingerprint sfp;
    sfp.fold_str("bym2");
    sfp.fold_pod(n_spatial_units);
    sfp.fold_pod(scale_factor);
    if (adj_row_ptr.size()) sfp.fold(adj_row_ptr.begin(),
                                     (std::size_t)adj_row_ptr.size() * sizeof(int));
    if (adj_col_idx.size()) sfp.fold(adj_col_idx.begin(),
                                     (std::size_t)adj_col_idx.size() * sizeof(int));
    auto ckpt = tulpa::make_nl_grid_checkpoint(
        checkpoint_path, sfp.value(), max_iter, tol, y, n, X, re_idx,
        n_re_groups, sigma_re, family, phi, {sigma_spatial_grid, rho_grid});

    Rcpp::List out = run_multi_block_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        blocks,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        store_Q, /*n_threads_outer=*/1, /*prune_tol=*/0.0,
        /*ext_spec=*/nullptr, /*ext_response=*/nullptr,
        /*progress=*/nullptr, ckpt.get()
    );
    out["sigma_spatial_grid"] = sigma_spatial_grid;
    out["rho_grid"] = rho_grid;
    return out;
}

// =====================================================================
// Nested Laplace: proper CAR (2D grid over (tau, rho))
// =====================================================================
// Full-rank Q(rho) = D - rho*W; identical inner shape to ICAR but with
// rho scaling the off-diagonals AND a non-vanishing log|Q(rho)| term in
// the per-grid-point log-marginal. log|Q(rho)| is computed once per grid
// point via the dense Cholesky in tulpa_car_proper::car_log_det.

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_car_proper(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::NumericVector tau_grid, Rcpp::NumericVector rho_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    std::string checkpoint_path = ""
) {
    int n_grid = tau_grid.size();
    if (rho_grid.size() != n_grid) {
        Rcpp::stop("tau_grid and rho_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int spatial_start = p + n_re_groups;

    std::vector<tulpa::LatentBlock> blocks = make_car_proper_latent_blocks(
        spatial_start, n_spatial_units, spatial_idx, tau_grid, rho_grid,
        adj_row_ptr, adj_col_idx, n_neighbors);

    tulpa::Fingerprint sfp;
    sfp.fold_str("car_proper");
    sfp.fold_pod(n_spatial_units);
    if (adj_row_ptr.size()) sfp.fold(adj_row_ptr.begin(),
                                     (std::size_t)adj_row_ptr.size() * sizeof(int));
    if (adj_col_idx.size()) sfp.fold(adj_col_idx.begin(),
                                     (std::size_t)adj_col_idx.size() * sizeof(int));
    auto ckpt = tulpa::make_nl_grid_checkpoint(
        checkpoint_path, sfp.value(), max_iter, tol, y, n, X, re_idx,
        n_re_groups, sigma_re, family, phi, {tau_grid, rho_grid});

    Rcpp::List out = run_multi_block_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        blocks,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        store_Q, /*n_threads_outer=*/1, /*prune_tol=*/0.0,
        /*ext_spec=*/nullptr, /*ext_response=*/nullptr,
        /*progress=*/nullptr, ckpt.get()
    );
    out["tau_grid"] = tau_grid;
    out["rho_grid"] = rho_grid;
    return out;
}

// Fixed-hyperparameter (single-point) proper-CAR Laplace. The conditional
// counterpart of cpp_nested_laplace_car_proper: it reuses the SAME
// make_car_proper_latent_blocks factory at a one-cell (tau, rho) grid and the
// shared dense spec solver, so the mode + log-marginal equal the nested kernel
// evaluated at that single grid cell. Routed from dispatch_laplace_spatial /
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
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue
) {
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

    // One-cell (tau, rho) grids; make_car_proper_latent_blocks captures them by
    // reference (and runs the log|D - rho W| determinant in block.prep), so they
    // must outlive the solve.
    Rcpp::NumericVector tau_grid = Rcpp::NumericVector::create(tau_spatial);
    Rcpp::NumericVector rho_grid = Rcpp::NumericVector::create(rho);
    std::vector<tulpa::LatentBlock> blocks = make_car_proper_latent_blocks(
        block_start, n_spatial_units, spatial_idx, tau_grid, rho_grid,
        adj_row_ptr, adj_col_idx, n_neighbors);

    tulpa::SpecFamilyInputs in;
    tulpa::build_spec_family_inputs(
        in, y, n, X, re_group, n_re_groups, sigma_re, family, phi,
        /*sigma_beta=*/100.0, /*n_block_latent=*/n_spatial_units,
        /*weights=*/nullptr, offset.empty() ? nullptr : offset.data());

    std::vector<double> params(in.layout.total_params, 0.0);
    if (has_re) params[in.layout.log_sigma_re_idx] = std::log(sigma_re);
    if (x_init_nullable.isNotNull()) {
        Rcpp::NumericVector xi = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
        const int n_lat = block_start + n_spatial_units;
        for (int j = 0; j < n_lat && j < (int)xi.size(); j++) params[j] = xi[j];
    }

    tulpa::LaplaceResult res = tulpa::laplace_mode_spec_dense_solve(
        in.data, in.layout, params, in.re_group, max_iter, tol, n_threads,
        &blocks, /*k_grid=*/0);
    return tulpa::laplace_result_to_list(res);
}

// =====================================================================
// Nested Laplace: single-block temporal (rw1 / rw2 / ar1)
//
// Collapsed into one runtime-dispatched entry, cpp_nested_laplace_temporal,
// defined further down next to the spatio-temporal entries so it can reuse the
// make_temporal_ops registry (declared later in this file). See that entry.
// =====================================================================

// =====================================================================
// Nested Laplace: NNGP (2D grid over (sigma2_gp, phi_gp))
// =====================================================================
// Continuous-spatial GP via nearest-neighbor conditional decomposition.
// Inner Newton matches laplace_mode_gp() in laplace_core.cpp; warm-start
// across grid points via run_nested_laplace_grid. cov_type: 0
// exponential, 1 Matern-3/2, 2 Matern-5/2 (matches nngp_cov_gpu).
//
// ABI v10+: spatial_idx (1-based, length N) maps observation -> spatial unit.
// store_modes = true; pass store_Q = true to retain Q at each grid point so
// the caller can build mixture-of-MVN posterior draws.

// [[Rcpp::export]]
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
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    std::string checkpoint_path = ""
) {
    const int n_grid = sigma2_grid.size();
    if (phi_gp_grid.size() != n_grid)
        Rcpp::stop("sigma2_grid and phi_gp_grid must have the same length");
    const int N = y.size();
    const int p = X.ncol();
    if (spatial_idx.size() != N)
        Rcpp::stop("length(spatial_idx) must equal length(y)");
    if (coords.nrow() != n_spatial)
        Rcpp::stop("nrow(coords) must equal n_spatial");

    // ---- Single-arm joint setup ----
    // Layout: [beta (p), re (n_re_groups), w_nngp (n_spatial)]. NNGP is
    // INDEXED_SINGLE — each obs maps to one spatial unit via spatial_idx.
    // The prior precision Λ = (I-A)' D⁻¹ (I-A) is NN_K-shaped sparse and
    // is applied by make_nngp_block's add_prior_sparse using
    // apply_nngp_full_prior_sparse.
    const int n_x = p + n_re_groups + n_spatial;
    const int gp_start = p + n_re_groups;

    std::vector<tulpa::ParsedArm> parsed(1);
    {
        tulpa::ParsedArm& pa = parsed[0];
        pa.X           = X;
        pa.re_idx      = re_idx;
        pa.spatial_idx = spatial_idx;
        pa.p           = p;
        pa.n_re_groups = n_re_groups;
        pa.sigma_re    = sigma_re;
        pa.beta_start  = 0;
        pa.re_start    = p;
        pa.tau_re      = (n_re_groups > 0)
                         ? 1.0 / (sigma_re * sigma_re + 1e-10)
                         : 0.0;
    }

    std::vector<tulpa::JointArm> arms(1);
    {
        tulpa::JointArm& a = arms[0];
        a.y        = y;
        a.n_trials = n;
        a.family   = family;
        a.phi      = phi;
        a.N        = N;
    }

    // ---- theta_grid: (sigma2, phi_gp). ----
    Rcpp::NumericMatrix theta_grid(n_grid, 2);
    for (int k = 0; k < n_grid; k++) {
        theta_grid(k, 0) = sigma2_grid[k];
        theta_grid(k, 1) = phi_gp_grid[k];
    }

    Rcpp::List spatial_idx_per_arm = Rcpp::List::create(spatial_idx);
    Rcpp::IntegerVector n_obs_per_arm = Rcpp::IntegerVector::create(N);

    std::vector<tulpa::LatentBlock> blocks;
    blocks.push_back(tulpa::make_nngp_block(
        /*start=*/gp_start, /*n_spatial=*/n_spatial,
        spatial_idx_per_arm, n_obs_per_arm,
        /*n_arms=*/1, /*block_index=*/0,
        nn, cov_type, coords, nn_idx, nn_dist, nn_order,
        /*axis_sigma2=*/0, /*axis_phi_gp=*/1, theta_grid
    ));

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull())
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    tulpa::Fingerprint sfp;
    sfp.fold_str("nngp");
    sfp.fold_pod(n_spatial);
    sfp.fold_pod(nn);
    sfp.fold_pod(cov_type);
    if (coords.size())      sfp.fold(coords.begin(),
                                     (std::size_t)coords.size() * sizeof(double));
    if (nn_idx.size())      sfp.fold(nn_idx.begin(),
                                     (std::size_t)nn_idx.size() * sizeof(int));
    if (spatial_idx.size()) sfp.fold(spatial_idx.begin(),
                                     (std::size_t)spatial_idx.size() * sizeof(int));
    auto ckpt = tulpa::make_nl_grid_checkpoint(
        checkpoint_path, sfp.value(), max_iter, tol, y, n, X, re_idx,
        n_re_groups, sigma_re, family, phi, {sigma2_grid, phi_gp_grid});

    Rcpp::List out = tulpa::run_multi_block_nested_laplace_joint_sparse_impl(
        n_grid, arms, parsed, blocks, n_x,
        max_iter, tol, n_threads,
        /*store_modes=*/true, x_init, store_Q,
        /*prep_at_grid=*/nullptr,
        /*tile_ids=*/std::vector<int>(),
        /*tile_pilot_cells=*/std::vector<int>(),
        /*prune_tol=*/0.0,
        /*cell_coupling_spec=*/nullptr,
        /*coupled_arms=*/std::vector<int>(),
        /*cell_rows=*/std::vector<std::vector<std::vector<int>>>(),
        /*n_cells=*/0,
        tulpa::JointPDMode::LM, tulpa::CurvatureMode::Observed,
        /*hessian_refresh=*/1, /*n_threads_outer=*/1,
        /*progress=*/nullptr, ckpt.get()
    );
    out["sigma2_grid"] = sigma2_grid;
    out["phi_gp_grid"] = phi_gp_grid;
    return out;
}

// =====================================================================
// Nested Laplace: HSGP (2D grid over (sigma2, lengthscale))
// =====================================================================
// Hilbert-space GP: f_i = sum_j Phi_ij * sqrt(S(lambda_j; sigma2, ell)) * beta_j
// with beta ~ N(0, I_M). Phi (basis) and lambda_j (eigenvalues) are FIXED
// across the grid; only the spectral diagonal sqrt(S) varies per grid
// point. Inner solve reduces to a fixed-design regression on the
// rescaled basis B = Phi * diag(sqrt(S)). spectral_density_se comes from
// hmc_hsgp_kernels.h (Eigen-free split out of hmc_hsgp.h).

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_hsgp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix phi_basis,
    Rcpp::NumericVector lambda_eig,
    Rcpp::NumericVector sigma2_grid,
    Rcpp::NumericVector lengthscale_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    std::string checkpoint_path = ""
) {
    const int n_grid = sigma2_grid.size();
    if (lengthscale_grid.size() != n_grid)
        Rcpp::stop("sigma2_grid and lengthscale_grid must have the same length");
    const int N = y.size();
    const int p = X.ncol();
    const int M = phi_basis.ncol();
    if (phi_basis.nrow() != N)
        Rcpp::stop("phi_basis must have N rows (one per observation)");
    if (lambda_eig.size() != M)
        Rcpp::stop("lambda_eig must have length ncol(phi_basis)");

    // ---- Single-arm joint setup ----
    // Layout: [beta (p), re (n_re_groups), beta_gp (M)]. The HSGP block is
    // DENSE_BASIS — every obs touches every basis coefficient via
    // Phi[i, j] * sqrt(S_j) — so the LatentBlock factory exposes basis_eval
    // and the joint sparse impl handles the pattern and scatter via the
    // unified path.
    const int n_x = p + n_re_groups + M;
    const int beta_gp_start = p + n_re_groups;

    std::vector<tulpa::ParsedArm> parsed(1);
    {
        tulpa::ParsedArm& pa = parsed[0];
        pa.X           = X;
        pa.re_idx      = re_idx;
        pa.spatial_idx = Rcpp::IntegerVector(N, 0);  // unused for DENSE_BASIS
        pa.p           = p;
        pa.n_re_groups = n_re_groups;
        pa.sigma_re    = sigma_re;
        pa.beta_start  = 0;
        pa.re_start    = p;
        pa.tau_re      = (n_re_groups > 0)
                         ? 1.0 / (sigma_re * sigma_re + 1e-10)
                         : 0.0;
    }

    std::vector<tulpa::JointArm> arms(1);
    {
        tulpa::JointArm& a = arms[0];
        a.y        = y;
        a.n_trials = n;
        a.family   = family;
        a.phi      = phi;
        a.N        = N;
    }

    // ---- theta_grid: (log_sigma2, log_lengthscale). The factory works in log
    // space because PC priors on (sigma2, ell) are typically applied in log
    // space upstream. ----
    Rcpp::NumericMatrix theta_grid(n_grid, 2);
    for (int k = 0; k < n_grid; k++) {
        theta_grid(k, 0) = std::log(sigma2_grid[k]);
        theta_grid(k, 1) = std::log(lengthscale_grid[k]);
    }

    Rcpp::List phi_per_arm = Rcpp::List::create(phi_basis);
    Rcpp::IntegerVector n_obs_per_arm = Rcpp::IntegerVector::create(N);

    std::vector<tulpa::LatentBlock> blocks;
    blocks.push_back(tulpa::make_hsgp_block(
        /*start=*/beta_gp_start, /*m_total=*/M,
        phi_per_arm, n_obs_per_arm, /*n_arms=*/1, /*block_index=*/0,
        lambda_eig,
        /*axis_log_sigma2=*/0, /*axis_log_ell=*/1, theta_grid
    ));

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull())
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    tulpa::Fingerprint sfp;
    sfp.fold_str("hsgp");
    sfp.fold_pod(M);
    if (phi_basis.size())  sfp.fold(phi_basis.begin(),
                                    (std::size_t)phi_basis.size() * sizeof(double));
    if (lambda_eig.size()) sfp.fold(lambda_eig.begin(),
                                    (std::size_t)lambda_eig.size() * sizeof(double));
    auto ckpt = tulpa::make_nl_grid_checkpoint(
        checkpoint_path, sfp.value(), max_iter, tol, y, n, X, re_idx,
        n_re_groups, sigma_re, family, phi, {sigma2_grid, lengthscale_grid});

    Rcpp::List out = tulpa::run_multi_block_nested_laplace_joint_sparse_impl(
        n_grid, arms, parsed, blocks, n_x,
        max_iter, tol, n_threads,
        /*store_modes=*/true, x_init, store_Q,
        /*prep_at_grid=*/nullptr,
        /*tile_ids=*/std::vector<int>(),
        /*tile_pilot_cells=*/std::vector<int>(),
        /*prune_tol=*/0.0,
        /*cell_coupling_spec=*/nullptr,
        /*coupled_arms=*/std::vector<int>(),
        /*cell_rows=*/std::vector<std::vector<std::vector<int>>>(),
        /*n_cells=*/0,
        tulpa::JointPDMode::LM, tulpa::CurvatureMode::Observed,
        /*hessian_refresh=*/1, /*n_threads_outer=*/1,
        /*progress=*/nullptr, ckpt.get()
    );
    out["sigma2_grid"]      = sigma2_grid;
    out["lengthscale_grid"] = lengthscale_grid;
    return out;
}

// Fixed-hyperparameter (single-point) HSGP Laplace. The conditional counterpart
// of cpp_nested_laplace_hsgp: it reuses the SAME make_hsgp_block factory at a
// one-row (log sigma2, log lengthscale) grid and the shared dense spec solver
// (which now scatters DENSE_BASIS blocks), so the mode + log-marginal equal the
// nested kernel evaluated at that single grid cell. Routed from
// dispatch_laplace_spatial / tulpa_laplace(spatial = spatial_hsgp(...)).
// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_hsgp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix phi_basis, Rcpp::NumericVector lambda_eig,
    double sigma2, double lengthscale,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue
) {
    const int N = y.size();
    const int p = X.ncol();
    const int M = phi_basis.ncol();
    if (phi_basis.nrow() != N)
        Rcpp::stop("phi_basis must have N rows (one per observation)");
    if (lambda_eig.size() != M)
        Rcpp::stop("lambda_eig must have length ncol(phi_basis)");
    const bool has_re = n_re_groups > 0;
    std::vector<int> re_group;
    if (has_re) {
        re_group.resize(N);
        for (int i = 0; i < N; i++) re_group[i] = (int)re_idx[i];
    }
    const int block_start = p + (has_re ? n_re_groups : 0);
    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);

    // One-row (log sigma2, log lengthscale) grid; make_hsgp_block reads the
    // amplitude / lengthscale from theta_grid(k, axis) in block.prep, so the row
    // must outlive the solve.
    Rcpp::NumericMatrix theta_grid(1, 2);
    theta_grid(0, 0) = std::log(sigma2);
    theta_grid(0, 1) = std::log(lengthscale);
    Rcpp::List phi_per_arm = Rcpp::List::create(phi_basis);
    Rcpp::IntegerVector n_obs_per_arm = Rcpp::IntegerVector::create(N);
    std::vector<tulpa::LatentBlock> blocks;
    blocks.push_back(tulpa::make_hsgp_block(
        block_start, M, phi_per_arm, n_obs_per_arm, /*n_arms=*/1,
        /*block_index=*/0, lambda_eig,
        /*axis_log_sigma2=*/0, /*axis_log_ell=*/1, theta_grid));

    tulpa::SpecFamilyInputs in;
    tulpa::build_spec_family_inputs(
        in, y, n, X, re_group, n_re_groups, sigma_re, family, phi,
        /*sigma_beta=*/100.0, /*n_block_latent=*/M,
        /*weights=*/nullptr, offset.empty() ? nullptr : offset.data());

    std::vector<double> params(in.layout.total_params, 0.0);
    if (has_re) params[in.layout.log_sigma_re_idx] = std::log(sigma_re);
    if (x_init_nullable.isNotNull()) {
        Rcpp::NumericVector xi = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
        const int n_lat = block_start + M;
        for (int j = 0; j < n_lat && j < (int)xi.size(); j++) params[j] = xi[j];
    }

    tulpa::LaplaceResult res = tulpa::laplace_mode_spec_dense_solve(
        in.data, in.layout, params, in.re_group, max_iter, tol, n_threads,
        &blocks, /*k_grid=*/0);
    return tulpa::laplace_result_to_list(res);
}

// =====================================================================
// Temporal prior callbacks (rw1 / rw2 / ar1) + the temporal LatentBlock
// =====================================================================
// IndexedPriorOps is a prior-only bundle for the indexed-temporal kinds (RW1 /
// RW2 / AR1): the temporal side is always single-DOF indexed, so the design
// contribution is `eta[i] += x[t_start + t_idx[i] - 1]`. make_temporal_ops
// selects the kernel at runtime; make_temporal_latent_block (below) wraps it as
// a LatentBlock.
//
// Spatio-temporal models stack a spatial block (make_<x>_latent_blocks for the
// areal families above, make_nngp_block / make_hsgp_block for the GP families)
// and a temporal block, then solve through the unified multi-block joint driver
// (run_indexed_st_nested_laplace_joint). Every obs contributes to both, so the
// off-diagonal cross-block H[w_spatial, w_temporal] is non-zero -- the two
// latent fields cannot be Laplace-marginalised separately, and the joint inner
// solve assembles the cross term from each block's own idx.
//
// The lambdas capture per-kind state (grids, adjacency) by reference; the
// caller (the Rcpp entry) keeps those Rcpp vectors alive across the solve.

namespace {

struct IndexedPriorOps {
    std::function<bool(int)> prep;
    std::function<void(tulpa::DenseVec&, tulpa::DenseMat&,
                       const Rcpp::NumericVector&, int)> add_prior;
    std::function<double(const Rcpp::NumericVector&, int)> log_prior;

    // Sparse-path fields. Populated by all RW1/RW2/AR1 ops factories using
    // the sparse precision twins.
    std::function<void(std::vector<std::pair<int,int>>&)>
        add_prior_pattern;
    std::function<void(tulpa::SparseHessianBuilder&, tulpa::DenseVec&,
                       const Rcpp::NumericVector&, int)>
        add_prior_sparse;
};

// Panel (grouped) temporal: a separate walk per group, all sharing one tau
// (and one rho for AR1). The G groups occupy contiguous blocks of n_times each
// at [start + g*n_times, ...), so every per-chain precision / log-prior /
// pattern builder runs once per group with the chains never connected across a
// group boundary. n_groups == 1 is the single-walk case (one call). The total
// prior pseudo-rank is the per-group rank summed over groups, which the
// log_prior_* helpers deliver block-by-block; identifiability against the
// intercept is the single global mean (a lone null direction once the
// likelihood pins the per-group level differences), pinned by the block's
// global center -- see make_temporal_latent_block.

// RW1 — 1D τ grid; cyclic flag closes each group's chain.
inline IndexedPriorOps make_rw1_ops(
    int start, int n_groups, int n_times,
    const Rcpp::NumericVector& tau_grid,
    bool cyclic
) {
    IndexedPriorOps ops;
    ops.prep = [](int) { return true; };
    ops.add_prior = [start, n_groups, n_times, &tau_grid, cyclic]
                    (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                     const Rcpp::NumericVector& x, int k) {
        for (int g = 0; g < n_groups; g++)
            tulpa::add_rw1_precision(grad, H, x, start + g * n_times, n_times,
                                      tau_grid[k], cyclic);
    };
    ops.log_prior = [start, n_groups, n_times, &tau_grid, cyclic]
                    (const Rcpp::NumericVector& x, int k) {
        double lp = 0.0;
        for (int g = 0; g < n_groups; g++)
            lp += tulpa::log_prior_rw1(x, start + g * n_times, n_times,
                                        tau_grid[k], cyclic);
        return lp;
    };
    ops.add_prior_pattern = [start, n_groups, n_times, cyclic]
                            (std::vector<std::pair<int,int>>& out) {
        for (int g = 0; g < n_groups; g++)
            tulpa::add_rw1_pattern(out, start + g * n_times, n_times, cyclic);
    };
    ops.add_prior_sparse = [start, n_groups, n_times, &tau_grid, cyclic]
                           (tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                            const Rcpp::NumericVector& x, int k) {
        for (int g = 0; g < n_groups; g++)
            tulpa::add_rw1_precision_sparse(grad, H, x, start + g * n_times,
                                             n_times, tau_grid[k], cyclic);
    };
    return ops;
}

// RW2 — 1D τ grid, optional cyclic (ring) closure.
inline IndexedPriorOps make_rw2_ops(
    int start, int n_groups, int n_times,
    const Rcpp::NumericVector& tau_grid, bool cyclic
) {
    IndexedPriorOps ops;
    ops.prep = [](int) { return true; };
    ops.add_prior = [start, n_groups, n_times, &tau_grid, cyclic]
                    (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                     const Rcpp::NumericVector& x, int k) {
        for (int g = 0; g < n_groups; g++)
            tulpa::add_rw2_precision(grad, H, x, start + g * n_times, n_times,
                                      tau_grid[k], cyclic);
    };
    ops.log_prior = [start, n_groups, n_times, &tau_grid, cyclic]
                    (const Rcpp::NumericVector& x, int k) {
        double lp = 0.0;
        for (int g = 0; g < n_groups; g++)
            lp += tulpa::log_prior_rw2(x, start + g * n_times, n_times,
                                        tau_grid[k], cyclic);
        return lp;
    };
    ops.add_prior_pattern = [start, n_groups, n_times, cyclic]
                            (std::vector<std::pair<int,int>>& out) {
        for (int g = 0; g < n_groups; g++)
            tulpa::add_rw2_pattern(out, start + g * n_times, n_times, cyclic);
    };
    ops.add_prior_sparse = [start, n_groups, n_times, &tau_grid, cyclic]
                           (tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                            const Rcpp::NumericVector& x, int k) {
        for (int g = 0; g < n_groups; g++)
            tulpa::add_rw2_precision_sparse(grad, H, x, start + g * n_times,
                                             n_times, tau_grid[k], cyclic);
    };
    return ops;
}

// AR1 — 2D (τ, ρ) grid. AR1 is proper (full rank) so each group's chain needs
// no constraint, but the per-group loop is identical to RW1/RW2.
inline IndexedPriorOps make_ar1_ops(
    int start, int n_groups, int n_times,
    const Rcpp::NumericVector& tau_grid,
    const Rcpp::NumericVector& rho_grid
) {
    IndexedPriorOps ops;
    ops.prep = [](int) { return true; };
    ops.add_prior = [start, n_groups, n_times, &tau_grid, &rho_grid]
                    (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                     const Rcpp::NumericVector& x, int k) {
        for (int g = 0; g < n_groups; g++)
            tulpa::add_ar1_precision(grad, H, x, start + g * n_times, n_times,
                                      tau_grid[k], rho_grid[k]);
    };
    ops.log_prior = [start, n_groups, n_times, &tau_grid, &rho_grid]
                    (const Rcpp::NumericVector& x, int k) {
        double lp = 0.0;
        for (int g = 0; g < n_groups; g++)
            lp += tulpa::log_prior_ar1(x, start + g * n_times, n_times,
                                        tau_grid[k], rho_grid[k]);
        return lp;
    };
    ops.add_prior_pattern = [start, n_groups, n_times]
                            (std::vector<std::pair<int,int>>& out) {
        for (int g = 0; g < n_groups; g++)
            tulpa::add_ar1_pattern(out, start + g * n_times, n_times);
    };
    ops.add_prior_sparse = [start, n_groups, n_times, &tau_grid, &rho_grid]
                           (tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                            const Rcpp::NumericVector& x, int k) {
        for (int g = 0; g < n_groups; g++)
            tulpa::add_ar1_precision_sparse(grad, H, x, start + g * n_times,
                                             n_times, tau_grid[k], rho_grid[k]);
    };
    return ops;
}

// Temporal prior dispatcher: pick the RW1 / RW2 / AR1 builder by name.
// The temporal axis is uniform (tau, optional rho, optional cyclic), so one
// registry keeps adding a temporal kernel O(1) -- a new kernel registers here
// and is immediately available to every spatial family, with no new spatial x
// temporal entry function. rho_grid is consulted only for AR1; cyclic applies
// to RW1 and RW2.
inline IndexedPriorOps make_temporal_ops(
    const std::string& temporal_type,
    int start, int n_groups, int n_times,
    const Rcpp::NumericVector& tau_grid,
    const Rcpp::NumericVector& rho_grid,
    bool cyclic
) {
    if (temporal_type == "rw1") {
        return make_rw1_ops(start, n_groups, n_times, tau_grid, cyclic);
    }
    if (temporal_type == "rw2") {
        return make_rw2_ops(start, n_groups, n_times, tau_grid, cyclic);
    }
    if (temporal_type == "ar1") {
        if (rho_grid.size() != tau_grid.size()) {
            Rcpp::stop("ar1 temporal prior requires rho_temporal_grid of the "
                       "same length as tau_temporal_grid");
        }
        return make_ar1_ops(start, n_groups, n_times, tau_grid, rho_grid);
    }
    Rcpp::stop("unknown temporal_type '%s' (expected 'rw1', 'rw2', or 'ar1')",
               temporal_type.c_str());
}

// Temporal LatentBlock: wrap make_temporal_ops (rw1 / rw2 / ar1, selected at
// runtime) as a LatentBlock with idx = temporal_idx, d_fac = 1, sum-to-zero
// centering, and ALL prior callbacks (dense add_prior / log_prior + the sparse
// add_prior_pattern / add_prior_sparse the joint-sparse driver uses). Shared by
// the temporal-only entry and every spatio-temporal entry (the temporal half of
// the [spatial, temporal] block stack). For panel (grouped) data the block
// spans n_groups * n_times nodes -- one chain per group -- and the per-chain
// ops loop over the groups (the chains stay disconnected); a lone single walk is
// the n_groups == 1 case. The center is the single GLOBAL mean: the per-group
// rank deficiency is carried by the prior pseudo-determinant (log_prior summed
// over groups), and the likelihood pins the per-group level differences, so only
// the one overall level confounds the intercept and needs centering. Lifetime:
// the callbacks capture temporal_idx / tau_grid / rho_grid by reference -- the
// caller keeps them alive across the outer-grid solve.
inline tulpa::LatentBlock make_temporal_latent_block(
    int start, int n_groups, int n_times,
    const Rcpp::IntegerVector& temporal_idx,
    const std::string& temporal_type,
    const Rcpp::NumericVector& tau_grid,
    const Rcpp::NumericVector& rho_grid,
    bool cyclic
) {
    const int n_units = n_groups * n_times;
    IndexedPriorOps ops_t = make_temporal_ops(temporal_type, start, n_groups,
                                              n_times, tau_grid, rho_grid, cyclic);
    tulpa::LatentBlock block;
    block.start = start;
    block.size  = n_units;
    block.idx   = [&temporal_idx](int i, int /*k_arm*/) { return temporal_idx[i]; };
    block.d_fac = [](int) { return 1.0; };
    block.prep              = ops_t.prep;
    block.add_prior         = ops_t.add_prior;
    block.log_prior         = ops_t.log_prior;
    block.add_prior_pattern = ops_t.add_prior_pattern;
    block.add_prior_sparse  = ops_t.add_prior_sparse;
    block.center = [start, n_units](Rcpp::NumericVector& x) {
        return tulpa::center_intercept(x, start, n_units);
    };
    return block;
}



// Areal spatio-temporal nested Laplace as a 1-arm joint over
// [beta | re | spatial block(s) | temporal block]. Dispatches dense/sparse
// through run_multi_block_nested_laplace_joint -- one spec-driven inner solve
// and one beta/RE convention, so the dense and sparse paths agree by
// construction. This retires the bespoke run_spatial_x_indexed_temporal_*
// driver (and its family-enum scatter) for the areal families: the spatial x
// temporal Hessian is just two INDEXED_SINGLE blocks sharing observations,
// which scatter_arm_obs_joint_multi already assembles (block x block cross
// terms via each block's own idx). `blocks` holds the spatial block(s) then the
// temporal block; their callbacks capture the caller's Rcpp vectors, which
// outlive this call.
inline Rcpp::List run_indexed_st_nested_laplace_joint(
    int n_grid,
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const Rcpp::IntegerVector& spatial_idx,
    const std::vector<tulpa::LatentBlock>& blocks,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    const Rcpp::NumericVector& x_init, bool store_Q, bool force_sparse,
    tulpa::GridCheckpoint* ckpt = nullptr
) {
    const int n_x_after_re = p + n_re_groups;

    std::vector<tulpa::ParsedArm> parsed(1);
    {
        tulpa::ParsedArm& pa = parsed[0];
        pa.X           = X;
        pa.re_idx      = re_idx;
        pa.spatial_idx = spatial_idx;
        pa.p           = p;
        pa.n_re_groups = n_re_groups;
        pa.sigma_re    = sigma_re;
        pa.beta_start  = 0;
        pa.re_start    = p;
        pa.tau_re      = (n_re_groups > 0)
                         ? 1.0 / (sigma_re * sigma_re + 1e-10) : 0.0;
    }
    std::vector<tulpa::JointArm> arms(1);
    {
        tulpa::JointArm& a = arms[0];
        a.y        = y;
        a.n_trials = n_trials;
        a.family   = family;
        a.phi      = phi;
        a.N        = N;
    }

    return tulpa::run_multi_block_nested_laplace_joint(
        n_grid, arms, parsed, blocks, n_x_after_re,
        max_iter, tol, n_threads, /*store_modes=*/true, x_init, store_Q,
        /*prep_at_grid=*/nullptr, /*n_threads_outer=*/1,
        std::vector<int>(), std::vector<int>(), /*prune_tol=*/0.0,
        force_sparse,
        /*cell_coupling_spec=*/nullptr,
        tulpa::JointPDMode::LM, tulpa::CurvatureMode::Observed,
        /*hessian_refresh=*/1, /*progress=*/nullptr, ckpt
    );
}

// Shared tail for every cpp_nested_laplace_st_<spatial> entry: stack the
// temporal latent block onto the caller's spatial block(s) at the right latent
// offset and run the joint inner solve. The temporal block, the [beta | re |
// spatial | temporal] offset bookkeeping, and the driver call live here once;
// each entry supplies only its spatial Q policy (the prebuilt block(s) and their
// latent dimension), its spatial_idx, and the force_sparse routing. `blocks` is
// taken by value so the temporal block appends without disturbing the caller.
inline Rcpp::List run_st_spatial_kernel(
    int n_grid, int spatial_latent_dim,
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const Rcpp::IntegerVector& spatial_idx,
    std::vector<tulpa::LatentBlock> blocks,
    const Rcpp::IntegerVector& temporal_idx, int n_times,
    const std::string& temporal_type,
    const Rcpp::NumericVector& tau_temporal_grid,
    const Rcpp::NumericVector& rho_t, bool cyclic,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    const Rcpp::NumericVector& x_init, bool store_Q, bool force_sparse,
    tulpa::GridCheckpoint* ckpt = nullptr
) {
    const int N = y.size();
    const int p = X.ncol();
    const int t_start = p + n_re_groups + spatial_latent_dim;
    // Space-time front door is single-walk temporal (no panel grouping yet).
    blocks.push_back(make_temporal_latent_block(
        t_start, /*n_groups=*/1, n_times, temporal_idx, temporal_type,
        tau_temporal_grid, rho_t, cyclic));
    return run_indexed_st_nested_laplace_joint(
        n_grid, y, n_trials, X, re_idx, N, p, n_re_groups, sigma_re,
        spatial_idx, blocks, family, phi, max_iter, tol, n_threads,
        x_init, store_Q, force_sparse, ckpt);
}

} // namespace

// =====================================================================
// Spatio-temporal nested Laplace: one typed entry per spatial family.
//
// The temporal kernel (rw1 / rw2 / ar1) is selected at runtime via
// `temporal_type` and built by make_temporal_ops, so the spatial x temporal
// cross-product is no longer one hand-written function per pair. Each entry
// owns the typed spatial inputs (adjacency / HSGP basis / NNGP neighbours),
// validates its own paired grids, and shares run_..._dispatch for the joint
// inner Newton. Adding a temporal kernel touches only make_temporal_ops.
//
// Joint over the spatial hyperparameter(s) x the temporal hyperparameter(s).
// Caller passes paired vectors of length n_grid (the Cartesian product is
// built R-side). Temporal axis: tau_temporal_grid always; rho_temporal_grid
// only for ar1 (pass NULL otherwise); cyclic only for rw1.
//   Latent: [beta (p)] [re (n_re_groups)] [w_spatial] [w_temporal (n_t)].
// =====================================================================

namespace {
// Materialise an optional rho grid (ar1) into a concrete vector that outlives
// the temporal ops (whose lambdas capture it by reference).
inline Rcpp::NumericVector nl_unwrap_rho_temporal(
    Rcpp::Nullable<Rcpp::NumericVector> rho_temporal_grid
) {
    return rho_temporal_grid.isNotNull()
        ? Rcpp::NumericVector(rho_temporal_grid)
        : Rcpp::NumericVector(0);
}
} // namespace

// ---- Temporal-only (rw1 / rw2 / ar1) ---------------------------------------
// Single latent temporal block, no spatial side. `temporal_type` selects the
// kernel at runtime through the same make_temporal_ops registry the ST entries
// use, so rw1 / rw2 / ar1 share one entry / shim / ABI typedef instead of
// three. tau_grid drives all kernels; rho_grid is the ar1 lag-1 grid (empty for
// rw1 / rw2); cyclic closes the rw1 chain (ignored by rw2 / ar1). `n_groups > 1`
// is panel (grouped) data: a separate walk per group sharing one tau (one rho
// for ar1), with temporal_idx the flattened 1-based node (group-1)*n_times+time.
//   Latent: [beta (p)] [re (n_re_groups)] [w_temporal (n_groups*n_times)].

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_temporal(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector temporal_idx, int n_times,
    std::string temporal_type,
    Rcpp::NumericVector tau_grid, Rcpp::NumericVector rho_grid, bool cyclic,
    std::string family, double phi = 1.0,
    int n_groups = 1,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    std::string checkpoint_path = ""
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int temporal_start = p + n_re_groups;

    // One temporal LatentBlock from the shared factory (rw1 / rw2 / ar1 selected
    // at runtime; n_groups walks for panel data); its callbacks close over
    // tau_grid / rho_grid / temporal_idx, which this frame keeps alive across the
    // run_multi_block call below.
    std::vector<tulpa::LatentBlock> blocks{ make_temporal_latent_block(
        temporal_start, n_groups, n_times, temporal_idx, temporal_type,
        tau_grid, rho_grid, cyclic) };

    tulpa::Fingerprint sfp;
    sfp.fold_str("temporal");
    sfp.fold_str(temporal_type);
    sfp.fold_pod(n_times);
    sfp.fold_pod(n_groups);
    sfp.fold_pod(cyclic);
    if (temporal_idx.size()) sfp.fold(temporal_idx.begin(),
                                      (std::size_t)temporal_idx.size() * sizeof(int));
    auto ckpt = tulpa::make_nl_grid_checkpoint(
        checkpoint_path, sfp.value(), max_iter, tol, y, n, X, re_idx,
        n_re_groups, sigma_re, family, phi, {tau_grid, rho_grid});

    Rcpp::List out = run_multi_block_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        blocks,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        store_Q, /*n_threads_outer=*/1, /*prune_tol=*/0.0,
        /*ext_spec=*/nullptr, /*ext_response=*/nullptr,
        /*progress=*/nullptr, ckpt.get()
    );
    out["tau_grid"] = tau_grid;
    if (rho_grid.size() > 0) out["rho_grid"] = rho_grid;
    return out;
}

// ---- ICAR (spatial) --------------------------------------------------------
// Grid axes: tau_spatial (1D) x temporal.

// [[Rcpp::export]]
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
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false,
    std::string checkpoint_path = ""
) {
    int n_grid = tau_spatial_grid.size();
    if (tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("tau_spatial_grid and tau_temporal_grid must have the same length");
    }
    Rcpp::NumericVector rho_t = nl_unwrap_rho_temporal(rho_temporal_grid);
    int s_start = X.ncol() + n_re_groups;

    std::vector<tulpa::LatentBlock> blocks = make_icar_latent_blocks(
        s_start, n_spatial_units, spatial_idx, tau_spatial_grid,
        adj_row_ptr, adj_col_idx, n_neighbors);

    tulpa::Fingerprint sfp;
    sfp.fold_str("st_icar");
    sfp.fold_pod(n_spatial_units);
    if (adj_row_ptr.size()) sfp.fold(adj_row_ptr.begin(),
                                     (std::size_t)adj_row_ptr.size() * sizeof(int));
    if (adj_col_idx.size()) sfp.fold(adj_col_idx.begin(),
                                     (std::size_t)adj_col_idx.size() * sizeof(int));
    sfp.fold_str(temporal_type);
    sfp.fold_pod(n_times);
    sfp.fold_pod(cyclic);
    if (temporal_idx.size()) sfp.fold(temporal_idx.begin(),
                                      (std::size_t)temporal_idx.size() * sizeof(int));
    auto ckpt = tulpa::make_nl_grid_checkpoint(
        checkpoint_path, sfp.value(), max_iter, tol, y, n, X, re_idx,
        n_re_groups, sigma_re, family, phi,
        {tau_spatial_grid, tau_temporal_grid, rho_t});

    Rcpp::List out = run_st_spatial_kernel(
        n_grid, /*spatial_latent_dim=*/n_spatial_units,
        y, n, X, re_idx, n_re_groups, sigma_re, spatial_idx, std::move(blocks),
        temporal_idx, n_times, temporal_type, tau_temporal_grid, rho_t, cyclic,
        family, phi, max_iter, tol, n_threads,
        unwrap_x_init(x_init_nullable), store_Q, force_sparse, ckpt.get());
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    if (temporal_type == "ar1") out["rho_temporal_grid"] = rho_t;
    return out;
}

// ---- CAR_proper (spatial) --------------------------------------------------
// Grid axes: (tau_spatial, rho_spatial) (paired 2D) x temporal.

// [[Rcpp::export]]
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
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false,
    std::string checkpoint_path = ""
) {
    int n_grid = tau_spatial_grid.size();
    if (rho_spatial_grid.size() != n_grid || tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("tau_spatial_grid, rho_spatial_grid, tau_temporal_grid must have the same length");
    }
    Rcpp::NumericVector rho_t = nl_unwrap_rho_temporal(rho_temporal_grid);
    int s_start = X.ncol() + n_re_groups;

    std::vector<tulpa::LatentBlock> blocks = make_car_proper_latent_blocks(
        s_start, n_spatial_units, spatial_idx, tau_spatial_grid, rho_spatial_grid,
        adj_row_ptr, adj_col_idx, n_neighbors);

    tulpa::Fingerprint sfp;
    sfp.fold_str("st_car_proper");
    sfp.fold_pod(n_spatial_units);
    if (adj_row_ptr.size()) sfp.fold(adj_row_ptr.begin(),
                                     (std::size_t)adj_row_ptr.size() * sizeof(int));
    if (adj_col_idx.size()) sfp.fold(adj_col_idx.begin(),
                                     (std::size_t)adj_col_idx.size() * sizeof(int));
    sfp.fold_str(temporal_type);
    sfp.fold_pod(n_times);
    sfp.fold_pod(cyclic);
    if (temporal_idx.size()) sfp.fold(temporal_idx.begin(),
                                      (std::size_t)temporal_idx.size() * sizeof(int));
    auto ckpt = tulpa::make_nl_grid_checkpoint(
        checkpoint_path, sfp.value(), max_iter, tol, y, n, X, re_idx,
        n_re_groups, sigma_re, family, phi,
        {tau_spatial_grid, rho_spatial_grid, tau_temporal_grid, rho_t});

    Rcpp::List out = run_st_spatial_kernel(
        n_grid, /*spatial_latent_dim=*/n_spatial_units,
        y, n, X, re_idx, n_re_groups, sigma_re, spatial_idx, std::move(blocks),
        temporal_idx, n_times, temporal_type, tau_temporal_grid, rho_t, cyclic,
        family, phi, max_iter, tol, n_threads,
        unwrap_x_init(x_init_nullable), store_Q, force_sparse, ckpt.get());
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["rho_spatial_grid"]  = rho_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    if (temporal_type == "ar1") out["rho_temporal_grid"] = rho_t;
    return out;
}

// ---- BYM2 (spatial) --------------------------------------------------------
// Grid axes: (sigma_spatial, rho_spatial) (paired 2D) x temporal.
//   Latent spatial block is [phi (n_s) + theta (n_s)] -> 2 * n_spatial_units.

// [[Rcpp::export]]
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
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false,
    std::string checkpoint_path = ""
) {
    int n_grid = sigma_spatial_grid.size();
    if (rho_spatial_grid.size() != n_grid || tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("sigma_spatial_grid, rho_spatial_grid, tau_temporal_grid must have the same length");
    }
    Rcpp::NumericVector rho_t = nl_unwrap_rho_temporal(rho_temporal_grid);
    int s_start = X.ncol() + n_re_groups;

    std::vector<tulpa::LatentBlock> blocks = make_bym2_latent_blocks(
        s_start, n_spatial_units, spatial_idx, scale_factor,
        sigma_spatial_grid, rho_spatial_grid,
        adj_row_ptr, adj_col_idx, n_neighbors);

    tulpa::Fingerprint sfp;
    sfp.fold_str("st_bym2");
    sfp.fold_pod(n_spatial_units);
    sfp.fold_pod(scale_factor);
    if (adj_row_ptr.size()) sfp.fold(adj_row_ptr.begin(),
                                     (std::size_t)adj_row_ptr.size() * sizeof(int));
    if (adj_col_idx.size()) sfp.fold(adj_col_idx.begin(),
                                     (std::size_t)adj_col_idx.size() * sizeof(int));
    sfp.fold_str(temporal_type);
    sfp.fold_pod(n_times);
    sfp.fold_pod(cyclic);
    if (temporal_idx.size()) sfp.fold(temporal_idx.begin(),
                                      (std::size_t)temporal_idx.size() * sizeof(int));
    auto ckpt = tulpa::make_nl_grid_checkpoint(
        checkpoint_path, sfp.value(), max_iter, tol, y, n, X, re_idx,
        n_re_groups, sigma_re, family, phi,
        {sigma_spatial_grid, rho_spatial_grid, tau_temporal_grid, rho_t});

    Rcpp::List out = run_st_spatial_kernel(
        n_grid, /*spatial_latent_dim=*/2 * n_spatial_units,
        y, n, X, re_idx, n_re_groups, sigma_re, spatial_idx, std::move(blocks),
        temporal_idx, n_times, temporal_type, tau_temporal_grid, rho_t, cyclic,
        family, phi, max_iter, tol, n_threads,
        unwrap_x_init(x_init_nullable), store_Q, force_sparse, ckpt.get());
    out["sigma_spatial_grid"] = sigma_spatial_grid;
    out["rho_spatial_grid"]   = rho_spatial_grid;
    out["tau_temporal_grid"]  = tau_temporal_grid;
    if (temporal_type == "ar1") out["rho_temporal_grid"] = rho_t;
    return out;
}

// ---- HSGP (spatial) --------------------------------------------------------
// Grid axes: (sigma2_spatial, lengthscale_spatial) (paired 2D) x temporal.
//   Spatial block is the M-dim basis-weight vector beta_M.

// [[Rcpp::export]]
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
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false,
    std::string checkpoint_path = ""
) {
    int n_grid = sigma2_spatial_grid.size();
    if (lengthscale_spatial_grid.size() != n_grid ||
        tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("sigma2_spatial_grid, lengthscale_spatial_grid, tau_temporal_grid must have the same length");
    }
    Rcpp::NumericVector rho_t = nl_unwrap_rho_temporal(rho_temporal_grid);
    int N = y.size();
    int p = X.ncol();
    int M = phi_basis.ncol();
    if (phi_basis.nrow() != N)
        Rcpp::stop("phi_basis must have N rows (one per observation)");
    if (lambda_eig.size() != M)
        Rcpp::stop("lambda_eig must have length ncol(phi_basis)");
    int s_start = p + n_re_groups;
    (void) force_sparse;  // HSGP is DENSE_BASIS -> the joint driver always takes
                          // the sparse path; the legacy force_sparse knob is moot.

    // theta_grid for the HSGP block: (log_sigma2, log_lengthscale), matching the
    // pure-spatial HSGP entry (PC priors applied in log space upstream).
    Rcpp::NumericMatrix theta_grid(n_grid, 2);
    for (int k = 0; k < n_grid; k++) {
        theta_grid(k, 0) = std::log(sigma2_spatial_grid[k]);
        theta_grid(k, 1) = std::log(lengthscale_spatial_grid[k]);
    }
    Rcpp::List phi_per_arm = Rcpp::List::create(phi_basis);
    Rcpp::IntegerVector n_obs_per_arm = Rcpp::IntegerVector::create(N);

    std::vector<tulpa::LatentBlock> blocks;
    blocks.push_back(tulpa::make_hsgp_block(
        /*start=*/s_start, /*m_total=*/M,
        phi_per_arm, n_obs_per_arm, /*n_arms=*/1, /*block_index=*/0,
        lambda_eig,
        /*axis_log_sigma2=*/0, /*axis_log_ell=*/1, theta_grid));
    // DENSE_BASIS HSGP block forces the joint sparse path regardless of n_x.
    Rcpp::IntegerVector spatial_idx_unused(N, 0);  // HSGP has no per-obs unit idx

    tulpa::Fingerprint sfp;
    sfp.fold_str("st_hsgp");
    sfp.fold_pod(M);
    if (phi_basis.size())  sfp.fold(phi_basis.begin(),
                                    (std::size_t)phi_basis.size() * sizeof(double));
    if (lambda_eig.size()) sfp.fold(lambda_eig.begin(),
                                    (std::size_t)lambda_eig.size() * sizeof(double));
    sfp.fold_str(temporal_type);
    sfp.fold_pod(n_times);
    sfp.fold_pod(cyclic);
    if (temporal_idx.size()) sfp.fold(temporal_idx.begin(),
                                      (std::size_t)temporal_idx.size() * sizeof(int));
    auto ckpt = tulpa::make_nl_grid_checkpoint(
        checkpoint_path, sfp.value(), max_iter, tol, y, n, X, re_idx,
        n_re_groups, sigma_re, family, phi,
        {sigma2_spatial_grid, lengthscale_spatial_grid, tau_temporal_grid, rho_t});

    Rcpp::List out = run_st_spatial_kernel(
        n_grid, /*spatial_latent_dim=*/M,
        y, n, X, re_idx, n_re_groups, sigma_re, spatial_idx_unused, std::move(blocks),
        temporal_idx, n_times, temporal_type, tau_temporal_grid, rho_t, cyclic,
        family, phi, max_iter, tol, n_threads,
        unwrap_x_init(x_init_nullable), store_Q, /*force_sparse=*/true, ckpt.get());
    out["sigma2_spatial_grid"]      = sigma2_spatial_grid;
    out["lengthscale_spatial_grid"] = lengthscale_spatial_grid;
    out["tau_temporal_grid"]        = tau_temporal_grid;
    if (temporal_type == "ar1") out["rho_temporal_grid"] = rho_t;
    return out;
}

// ---- NNGP (spatial) --------------------------------------------------------
// Grid axes: (sigma2_spatial, phi_gp_spatial) (paired 2D) x temporal.

// [[Rcpp::export]]
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
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false,
    std::string checkpoint_path = ""
) {
    int n_grid = sigma2_spatial_grid.size();
    if (phi_gp_spatial_grid.size() != n_grid ||
        tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("sigma2_spatial_grid, phi_gp_spatial_grid, tau_temporal_grid must have the same length");
    }
    Rcpp::NumericVector rho_t = nl_unwrap_rho_temporal(rho_temporal_grid);
    int N = y.size();
    int p = X.ncol();
    if (spatial_idx.size() != N)
        Rcpp::stop("length(spatial_idx) must equal length(y)");
    if (coords.nrow() != n_spatial)
        Rcpp::stop("nrow(coords) must equal n_spatial");
    int s_start = p + n_re_groups;
    (void) force_sparse;  // NNGP's prior is sparse-native -> the joint driver
                          // always takes the sparse path; force_sparse is moot.

    // theta_grid for the NNGP block: (sigma2, phi_gp), matching the pure-spatial
    // NNGP entry (linear, not log).
    Rcpp::NumericMatrix theta_grid(n_grid, 2);
    for (int k = 0; k < n_grid; k++) {
        theta_grid(k, 0) = sigma2_spatial_grid[k];
        theta_grid(k, 1) = phi_gp_spatial_grid[k];
    }
    Rcpp::List spatial_idx_per_arm = Rcpp::List::create(spatial_idx);
    Rcpp::IntegerVector n_obs_per_arm = Rcpp::IntegerVector::create(N);

    std::vector<tulpa::LatentBlock> blocks;
    blocks.push_back(tulpa::make_nngp_block(
        /*start=*/s_start, /*n_spatial=*/n_spatial,
        spatial_idx_per_arm, n_obs_per_arm,
        /*n_arms=*/1, /*block_index=*/0,
        nn, cov_type, coords, nn_idx, nn_dist, nn_order,
        /*axis_sigma2=*/0, /*axis_phi_gp=*/1, theta_grid));

    tulpa::Fingerprint sfp;
    sfp.fold_str("st_nngp");
    sfp.fold_pod(n_spatial);
    sfp.fold_pod(nn);
    sfp.fold_pod(cov_type);
    if (coords.size())   sfp.fold(coords.begin(),
                                  (std::size_t)coords.size() * sizeof(double));
    if (nn_idx.size())   sfp.fold(nn_idx.begin(),
                                  (std::size_t)nn_idx.size() * sizeof(int));
    if (spatial_idx.size()) sfp.fold(spatial_idx.begin(),
                                     (std::size_t)spatial_idx.size() * sizeof(int));
    sfp.fold_str(temporal_type);
    sfp.fold_pod(n_times);
    sfp.fold_pod(cyclic);
    if (temporal_idx.size()) sfp.fold(temporal_idx.begin(),
                                      (std::size_t)temporal_idx.size() * sizeof(int));
    auto ckpt = tulpa::make_nl_grid_checkpoint(
        checkpoint_path, sfp.value(), max_iter, tol, y, n, X, re_idx,
        n_re_groups, sigma_re, family, phi,
        {sigma2_spatial_grid, phi_gp_spatial_grid, tau_temporal_grid, rho_t});

    Rcpp::List out = run_st_spatial_kernel(
        n_grid, /*spatial_latent_dim=*/n_spatial,
        y, n, X, re_idx, n_re_groups, sigma_re, spatial_idx, std::move(blocks),
        temporal_idx, n_times, temporal_type, tau_temporal_grid, rho_t, cyclic,
        family, phi, max_iter, tol, n_threads,
        unwrap_x_init(x_init_nullable), store_Q, /*force_sparse=*/true, ckpt.get());
    out["sigma2_spatial_grid"] = sigma2_spatial_grid;
    out["phi_gp_spatial_grid"] = phi_gp_spatial_grid;
    out["tau_temporal_grid"]   = tau_temporal_grid;
    if (temporal_type == "ar1") out["rho_temporal_grid"] = rho_t;
    return out;
}
