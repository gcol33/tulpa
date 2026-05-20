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
#include "nested_laplace_grid.h"
#include "nested_laplace_multi.h"  // accumulate_latent_cross_terms, run_multi_block_nested_laplace
#include "hmc_car_proper.h"
#include "gpu_nngp_laplace.h"
#include "hmc_hsgp_kernels.h"  // Eigen-free spectral density only
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
    bool store_Q = false
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int spatial_start = p + n_re_groups;

    tulpa::LatentBlock block;
    block.start = spatial_start;
    block.size  = n_spatial_units;
    block.idx   = [&](int i, int /*k_arm*/) { return spatial_idx[i]; };
    block.d_fac = [](int) { return 1.0; };
    block.add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                          const Rcpp::NumericVector& x, int k) {
        tulpa::add_icar_prior(grad, H, x, spatial_start, n_spatial_units,
                               tau_grid[k], adj_row_ptr, adj_col_idx, n_neighbors);
    };
    block.log_prior = [&](const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_icar(x, spatial_start, n_spatial_units, tau_grid[k],
                                       adj_row_ptr, adj_col_idx, n_neighbors);
    };
    block.center = [&](Rcpp::NumericVector& x) -> double {
        return tulpa::center_effects(x, spatial_start, n_spatial_units);
    };

    std::vector<tulpa::LatentBlock> blocks{ block };

    Rcpp::List out = run_multi_block_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        blocks,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        store_Q
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
// Port note (Phase A5, plan_multi_block.md): this kernel routes through
// run_multi_block_nested_laplace with phi and theta as two LatentBlocks.
// The multi-block compute_eta accumulates `d_fac_b(k) * x[idx_b]` per block,
// which re-associates the FP multiplication relative to the original
// `sigma_k * (sqrt_rho * x * scale_factor + sqrt_1_rho * x)` factoring.
// Result: log_marginal differs at ~1e-8 from pre-refactor (well below the
// 1e-6 Newton tolerance). See dev_notes/refactor_notes_bym2.md.

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
    bool store_Q = false
) {
    int n_grid = sigma_spatial_grid.size();
    int N = y.size();
    int p = X.ncol();
    int phi_start   = p + n_re_groups;
    int theta_start = phi_start + n_spatial_units;

    // phi block: ICAR-structured, mixed with d = sigma_k * sqrt(rho_k) * scale_factor.
    // theta block: IID, mixed with d = sigma_k * sqrt(1 - rho_k).
    // Both blocks share spatial_idx. Centering applies only to phi (theta is IID,
    // identifiability is anchored by the global intercept).
    tulpa::LatentBlock phi_block;
    phi_block.start = phi_start;
    phi_block.size  = n_spatial_units;
    phi_block.idx   = [&](int i, int /*k_arm*/) { return spatial_idx[i]; };
    phi_block.d_fac = [&, scale_factor](int k) {
        return sigma_spatial_grid[k] *
               std::sqrt(rho_grid[k] + 1e-10) * scale_factor;
    };
    phi_block.add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                              const Rcpp::NumericVector& x, int /*k*/) {
        tulpa::add_icar_prior(grad, H, x, phi_start, n_spatial_units, 1.0,
                               adj_row_ptr, adj_col_idx, n_neighbors);
    };
    phi_block.log_prior = [&](const Rcpp::NumericVector& x, int /*k*/) {
        // Bare ICAR quadratic at tau = 1 (matches the original BYM2 log_prior).
        double quad_form = 0.0;
        for (int s = 0; s < n_spatial_units; s++) {
            double phi_s = x[phi_start + s];
            quad_form += n_neighbors[s] * phi_s * phi_s;
            for (int kk = adj_row_ptr[s]; kk < adj_row_ptr[s + 1]; kk++) {
                int neighbor = adj_col_idx[kk];
                if (neighbor > s) quad_form -= 2.0 * phi_s * x[phi_start + neighbor];
            }
        }
        return -0.5 * quad_form;
    };
    phi_block.center = [&](Rcpp::NumericVector& x) -> double {
        return tulpa::center_effects(x, phi_start, n_spatial_units);
    };

    tulpa::LatentBlock theta_block;
    theta_block.start = theta_start;
    theta_block.size  = n_spatial_units;
    theta_block.idx   = [&](int i, int /*k_arm*/) { return spatial_idx[i]; };
    theta_block.d_fac = [&](int k) {
        return sigma_spatial_grid[k] * std::sqrt(1.0 - rho_grid[k] + 1e-10);
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
    // theta_block.center left empty (IID, no constraint)

    std::vector<tulpa::LatentBlock> blocks{ phi_block, theta_block };

    Rcpp::List out = run_multi_block_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        blocks,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        store_Q
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
    bool store_Q = false
) {
    int n_grid = tau_grid.size();
    if (rho_grid.size() != n_grid) {
        Rcpp::stop("tau_grid and rho_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int spatial_start = p + n_re_groups;

    // Adjacency CSR copied into std::vector for the dense log|Q(rho)| helper.
    std::vector<int> adj_rp_v(adj_row_ptr.begin(), adj_row_ptr.end());
    std::vector<int> adj_ci_v(adj_col_idx.begin(), adj_col_idx.end());
    std::vector<int> n_nbr_v(n_neighbors.begin(), n_neighbors.end());

    // Per-grid-point log|Q(rho_k)|, refreshed by block.prep and consumed by
    // block.log_prior. Dense O(n^3) is fine for areal graphs (n_spatial < ~500);
    // swap in sparse Cholesky if it becomes the bottleneck.
    auto log_det_Q_rho = std::make_shared<double>(0.0);

    tulpa::LatentBlock block;
    block.start = spatial_start;
    block.size  = n_spatial_units;
    block.idx   = [&](int i, int /*k_arm*/) { return spatial_idx[i]; };
    block.d_fac = [](int) { return 1.0; };
    block.prep  = [&, log_det_Q_rho](int k) -> bool {
        std::vector<double> Qmat = tulpa_car_proper::compute_car_precision(
            n_spatial_units, adj_rp_v, adj_ci_v, n_nbr_v, rho_grid[k]);
        *log_det_Q_rho = tulpa_car_proper::car_log_det(n_spatial_units, Qmat);
        return std::isfinite(*log_det_Q_rho);
    };
    block.add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                          const Rcpp::NumericVector& x, int k) {
        tulpa::add_car_proper_prior(grad, H, x, spatial_start, n_spatial_units,
                                     tau_grid[k], rho_grid[k],
                                     adj_row_ptr, adj_col_idx, n_neighbors);
    };
    block.log_prior = [&, log_det_Q_rho](const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_car_proper(x, spatial_start, n_spatial_units,
                                             tau_grid[k], rho_grid[k],
                                             *log_det_Q_rho,
                                             adj_row_ptr, adj_col_idx, n_neighbors);
    };
    block.center = [&](Rcpp::NumericVector& x) -> double {
        return tulpa::center_effects(x, spatial_start, n_spatial_units);
    };

    std::vector<tulpa::LatentBlock> blocks{ block };

    Rcpp::List out = run_multi_block_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        blocks,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        store_Q
    );
    out["tau_grid"] = tau_grid;
    out["rho_grid"] = rho_grid;
    return out;
}

// =====================================================================
// Nested Laplace: RW1 (1D grid over tau_temporal)
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_rw1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector temporal_idx, int n_times, bool cyclic,
    Rcpp::NumericVector tau_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int temporal_start = p + n_re_groups;

    tulpa::LatentBlock block;
    block.start = temporal_start;
    block.size  = n_times;
    block.idx   = [&](int i, int /*k_arm*/) { return temporal_idx[i]; };
    block.d_fac = [](int) { return 1.0; };
    block.add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                          const Rcpp::NumericVector& x, int k) {
        tulpa::add_rw1_precision(grad, H, x, temporal_start, n_times,
                                  tau_grid[k], cyclic);
    };
    block.log_prior = [&](const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_rw1(x, temporal_start, n_times, tau_grid[k], cyclic);
    };
    block.center = [&](Rcpp::NumericVector& x) -> double {
        return tulpa::center_effects(x, temporal_start, n_times);
    };

    std::vector<tulpa::LatentBlock> blocks{ block };

    Rcpp::List out = run_multi_block_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        blocks,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        store_Q
    );
    out["tau_grid"] = tau_grid;
    return out;
}

// =====================================================================
// Nested Laplace: RW2 (1D grid over tau_temporal)
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_rw2(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int temporal_start = p + n_re_groups;

    tulpa::LatentBlock block;
    block.start = temporal_start;
    block.size  = n_times;
    block.idx   = [&](int i, int /*k_arm*/) { return temporal_idx[i]; };
    block.d_fac = [](int) { return 1.0; };
    block.add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                          const Rcpp::NumericVector& x, int k) {
        tulpa::add_rw2_precision(grad, H, x, temporal_start, n_times,
                                  tau_grid[k], false);
    };
    block.log_prior = [&](const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_rw2(x, temporal_start, n_times, tau_grid[k], false);
    };
    block.center = [&](Rcpp::NumericVector& x) -> double {
        return tulpa::center_effects(x, temporal_start, n_times);
    };

    std::vector<tulpa::LatentBlock> blocks{ block };

    Rcpp::List out = run_multi_block_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        blocks,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        store_Q
    );
    out["tau_grid"] = tau_grid;
    return out;
}

// =====================================================================
// Nested Laplace: AR1 (2D grid over (tau, rho))
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_ar1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_grid, Rcpp::NumericVector rho_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false
) {
    int n_grid = tau_grid.size();
    int N = y.size();
    int p = X.ncol();
    int temporal_start = p + n_re_groups;

    // AR1 is proper (full rank) — no sum-to-zero constraint required, but
    // we still center the latent block to stabilise identifiability against
    // the intercept (matches RW1/RW2/ICAR).
    tulpa::LatentBlock block;
    block.start = temporal_start;
    block.size  = n_times;
    block.idx   = [&](int i, int /*k_arm*/) { return temporal_idx[i]; };
    block.d_fac = [](int) { return 1.0; };
    block.add_prior = [&](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                          const Rcpp::NumericVector& x, int k) {
        tulpa::add_ar1_precision(grad, H, x, temporal_start, n_times,
                                  tau_grid[k], rho_grid[k]);
    };
    block.log_prior = [&](const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_ar1(x, temporal_start, n_times,
                                      tau_grid[k], rho_grid[k]);
    };
    block.center = [&](Rcpp::NumericVector& x) -> double {
        return tulpa::center_effects(x, temporal_start, n_times);
    };

    std::vector<tulpa::LatentBlock> blocks{ block };

    Rcpp::List out = run_multi_block_nested_laplace(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        blocks,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true, unwrap_x_init(x_init_nullable),
        store_Q
    );
    out["tau_grid"] = tau_grid;
    out["rho_grid"] = rho_grid;
    return out;
}

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
    bool store_Q = false
) {
    int n_grid = sigma2_grid.size();
    if (phi_gp_grid.size() != n_grid)
        Rcpp::stop("sigma2_grid and phi_gp_grid must have the same length");
    int N = y.size();
    int p = X.ncol();
    if (spatial_idx.size() != N)
        Rcpp::stop("length(spatial_idx) must equal length(y)");
    if (coords.nrow() != n_spatial)
        Rcpp::stop("nrow(coords) must equal n_spatial");
    int n_x = p + n_re_groups + n_spatial;
    int gp_start = p + n_re_groups;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

    // Per-thread NewtonScratch pool. Outer-grid parallelism for NNGP is
    // gated behind GPU batching, not OpenMP — keep n_outer = 1 here.
    const int n_outer = 1;
    std::vector<tulpa::NewtonScratch> scratch_pool(n_outer);
    for (auto& s : scratch_pool) s.allocate(n_x, N);

    auto solve_at_theta = [&](int k,
                              const std::vector<double>& prev_mode,
                              tulpa::SparseCholeskySolver* solver)
        -> tulpa::LaplaceResult
    {
        double sigma2_k = sigma2_grid[k];
        double phi_gp_k = phi_gp_grid[k];

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
            #endif
            for (int i = 0; i < N; i++) {
                eta[i] = 0.0;
                for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) eta[i] += x[p + g];
                }
                int s = spatial_idx[i] - 1;
                if (s >= 0 && s < n_spatial) eta[i] += x[gp_start + s];
            }
        };

        auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                           tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            tulpa::scatter_obs_grad_hess_base(y, n, X, re_idx, N, p, n_re_groups,
                                                eta, family, phi, grad, H, n_threads);
            for (int i = 0; i < N; i++) {
                int s = spatial_idx[i] - 1;
                if (s < 0 || s >= n_spatial) continue;
                auto gh = tulpa::grad_hess_for_family(y[i], n[i], eta[i], family, phi);
                int gp_idx = gp_start + s;
                grad[gp_idx] += gh.grad;
                H[gp_idx][gp_idx] += gh.neg_hess;
                for (int j = 0; j < p; j++) {
                    H[gp_idx][j] += gh.neg_hess * X(i, j);
                    H[j][gp_idx] += gh.neg_hess * X(i, j);
                }
                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) {
                        H[p + g][gp_idx] += gh.neg_hess;
                        H[gp_idx][p + g] += gh.neg_hess;
                    }
                }
            }
            // NNGP prior: full precision Λ = (I-A)' D⁻¹ (I-A) — not diagonal.
            // Diagonal-on-w gives wrong w-mode + wrong field covariance on
            // smooth latent fields; full scatter restores pointwise recovery.
            std::vector<double> w(n_spatial);
            for (int s = 0; s < n_spatial; s++) w[s] = x[gp_start + s];
            std::vector<double> cm, cv, nngp_alpha;
            bool gpu_used;
            tulpa::batch_nngp_scatter(w, n_spatial, nn, sigma2_k, phi_gp_k, cov_type,
                                        coords, nn_idx, nn_dist, nn_order,
                                        cm, cv, gpu_used, &nngp_alpha);
            tulpa::apply_nngp_full_prior_dense(grad, H, w, nngp_alpha, cv,
                                                 nn_idx, nn_order,
                                                 n_spatial, nn, gp_start);
            tulpa::add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [](Rcpp::NumericVector&) {};

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            std::vector<double> w(n_spatial);
            for (int s = 0; s < n_spatial; s++) w[s] = x[gp_start + s];
            std::vector<double> cm, cv;
            bool gpu_used;
            tulpa::batch_nngp_scatter(w, n_spatial, nn, sigma2_k, phi_gp_k, cov_type,
                                        coords, nn_idx, nn_dist, nn_order,
                                        cm, cv, gpu_used);
            for (int s = 0; s < n_spatial; s++) {
                double resid = w[s] - cm[s];
                lp += -0.5 * std::log(2.0 * M_PI * cv[s]) -
                       0.5 * resid * resid / cv[s];
            }
            return lp;
        };

        int tid;
        #ifdef _OPENMP
        tid = omp_in_parallel() ? omp_get_thread_num() : 0;
        #else
        tid = 0;
        #endif
        return tulpa::laplace_newton_solve(
            y, n, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter, center, log_prior,
            scratch_pool[tid], prev_mode, solver,
            store_Q
        );
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    Rcpp::List out = tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, /*store_modes=*/true, n_outer
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
    bool store_Q = false
) {
    int n_grid = sigma2_grid.size();
    if (lengthscale_grid.size() != n_grid)
        Rcpp::stop("sigma2_grid and lengthscale_grid must have the same length");
    int N = y.size();
    int p = X.ncol();
    int M = phi_basis.ncol();
    if (phi_basis.nrow() != N)
        Rcpp::stop("phi_basis must have N rows (one per observation)");
    if (lambda_eig.size() != M)
        Rcpp::stop("lambda_eig must have length ncol(phi_basis)");
    int n_x = p + n_re_groups + M;
    int beta_gp_start = p + n_re_groups;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

    // Per-thread NewtonScratch pool. omp_get_thread_num() returns 0 outside
    // parallel regions so serial callers correctly land on scratch_pool[0].
    // No outer-grid parallelism is exposed at this kernel yet (HSGP is a
    // small-M problem; the joint multi-block path is where the tulpaObs win
    // lives) — keep n_outer hardcoded at 1 here.
    const int n_outer = 1;
    std::vector<tulpa::NewtonScratch> scratch_pool(n_outer);
    for (auto& s : scratch_pool) s.allocate(n_x, N);

    auto solve_at_theta = [&](int k,
                              const std::vector<double>& prev_mode,
                              tulpa::SparseCholeskySolver* solver)
        -> tulpa::LaplaceResult
    {
        double sigma2_k = sigma2_grid[k];
        double ell_k = lengthscale_grid[k];

        std::vector<double> sqrt_S(M);
        for (int j = 0; j < M; j++) {
            double S = tulpa_hsgp::spectral_density_se(lambda_eig[j], sigma2_k, ell_k);
            sqrt_S[j] = std::sqrt(std::max(S, 1e-30));
        }

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
            #endif
            for (int i = 0; i < N; i++) {
                double e = 0.0;
                for (int j = 0; j < p; j++) e += X(i, j) * x[j];
                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) e += x[p + g];
                }
                for (int j = 0; j < M; j++) {
                    e += phi_basis(i, j) * sqrt_S[j] * x[beta_gp_start + j];
                }
                eta[i] = e;
            }
        };

        auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                           tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            tulpa::scatter_obs_grad_hess_base(y, n, X, re_idx, N, p, n_re_groups,
                                                eta, family, phi, grad, H, n_threads);
            for (int i = 0; i < N; i++) {
                auto gh = tulpa::grad_hess_for_family(y[i], n[i], eta[i], family, phi);
                for (int j = 0; j < M; j++) {
                    double bij = phi_basis(i, j) * sqrt_S[j];
                    int idx_j = beta_gp_start + j;
                    grad[idx_j] += gh.grad * bij;
                    H[idx_j][idx_j] += gh.neg_hess * bij * bij;
                    for (int kk = j + 1; kk < M; kk++) {
                        double bik = phi_basis(i, kk) * sqrt_S[kk];
                        double cross = gh.neg_hess * bij * bik;
                        H[idx_j][beta_gp_start + kk] += cross;
                        H[beta_gp_start + kk][idx_j] += cross;
                    }
                    for (int jj = 0; jj < p; jj++) {
                        double cross = gh.neg_hess * bij * X(i, jj);
                        H[idx_j][jj] += cross;
                        H[jj][idx_j] += cross;
                    }
                    if (n_re_groups > 0) {
                        int g = (int)re_idx[i] - 1;
                        if (g >= 0 && g < n_re_groups) {
                            double cross = gh.neg_hess * bij;
                            H[idx_j][p + g] += cross;
                            H[p + g][idx_j] += cross;
                        }
                    }
                }
            }
            // beta_gp prior N(0, I): grad -= beta_gp; H_diag += 1.
            for (int j = 0; j < M; j++) {
                int idx = beta_gp_start + j;
                grad[idx] -= x[idx];
                H[idx][idx] += 1.0;
            }
            tulpa::add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [](Rcpp::NumericVector&) {};

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            double q = 0.0;
            for (int j = 0; j < M; j++) {
                double b = x[beta_gp_start + j];
                q += b * b;
            }
            lp += -0.5 * q - 0.5 * M * std::log(2.0 * M_PI);
            return lp;
        };

        int tid;
        #ifdef _OPENMP
        tid = omp_in_parallel() ? omp_get_thread_num() : 0;
        #else
        tid = 0;
        #endif
        return tulpa::laplace_newton_solve(
            y, n, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter, center, log_prior,
            scratch_pool[tid], prev_mode, solver,
            store_Q
        );
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    Rcpp::List out = tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, /*store_modes=*/true, n_outer
    );
    out["sigma2_grid"] = sigma2_grid;
    out["lengthscale_grid"] = lengthscale_grid;
    return out;
}

// =====================================================================
// Joint (spatial × temporal) nested Laplace — shared helpers
// =====================================================================
// Each observation contributes a spatial term and a temporal term to the
// linear predictor, both of which live in their own latent block of the
// joint state [beta] [re] [w_spatial (s_block_len)] [w_temporal (n_t)].
// Both blocks are first-class in the joint Hessian; the off-diagonal
// cross-block H[w_s, w_t] is non-zero for every obs that contributes to
// both sides, which is why the two latent fields cannot be Laplace-
// marginalised separately — the math forces a joint inner solve.
//
// The spatial side is encoded as a SpatialBlockOps bundle (design + prior).
// Three shapes are supported:
//   - "indexed" (ICAR, CAR_proper): single DOF per obs, weight 1.0;
//     block length = n_units.
//   - "two-DOF indexed" (BYM2): two DOFs per obs (phi half + theta half),
//     weights derived from (σ_k, ρ_k); block length = 2·n_units.
//   - "dense basis" (HSGP): M DOFs per obs (every basis function),
//     weights = Φ_ij · √S(λ_j; σ²_k, ℓ_k); block length = M = n_basis.
// The temporal side stays indexed-with-weight-1 (RW1/RW2/AR1).
//
// The hyperparameter grid is supplied caller-side as paired vectors of
// length n_grid; entry k is one (θ_s_k, θ_t_k) tuple. The Cartesian
// product is built on the R side.

namespace {

// Per-obs design store for the spatial latent block. For obs i, the partial
// derivative ∂η_i/∂x[start + obs_local_idx[e]] is obs_weight[e] for
// e ∈ [obs_p[i], obs_p[i+1]). The store is CSR-shaped to allow zero or
// many entries per obs; prep_at_k() refreshes weights when the per-k
// spectrum changes (BYM2, HSGP), and is a no-op otherwise (ICAR, CAR_proper).
struct SpatialBlockOps {
    int start;
    int block_len;

    std::function<bool(int)> prep_at_k;
    std::function<void(const Rcpp::NumericVector&, Rcpp::NumericVector&,
                       int, int)> add_eta;

    std::shared_ptr<std::vector<int>>    obs_p;
    std::shared_ptr<std::vector<int>>    obs_local_idx;
    std::shared_ptr<std::vector<double>> obs_weight;

    std::function<void(tulpa::DenseVec&, tulpa::DenseMat&,
                       const Rcpp::NumericVector&, int)> add_prior_at_k;
    std::function<double(const Rcpp::NumericVector&, int)> log_prior_at_k;
    std::function<void(Rcpp::NumericVector&)>             center;

    // Sparse-path fields (Stage 1.4b). Populated by spatial ops factories
    // that support the sparse Newton path. May be empty for kinds that
    // require dense scatter (HSGP / NNGP — implementation deferred).
    std::function<void(std::vector<std::pair<int,int>>&)>
        add_prior_pattern;          // appends prior nonzero entries (lower triangle)
    std::function<void(tulpa::SparseHessianBuilder&, tulpa::DenseVec&,
                       const Rcpp::NumericVector&, int)>
        add_prior_sparse;
};

// Base eta: X·β + re[g(i)] + temporal[t_idx[i]]. The spatial contribution is
// added on top by spatial_ops.add_eta().
inline void nl_compute_eta_base_x_indexed_temporal(
    const Rcpp::NumericVector& x,
    Rcpp::NumericVector& eta,
    int N, int p, int n_re_groups,
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& re_idx,
    int t_start, int n_t, const Rcpp::IntegerVector& t_idx,
    int n_threads
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
#endif
    for (int i = 0; i < N; i++) {
        eta[i] = 0.0;
        for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
        if (n_re_groups > 0) {
            int g = (int)re_idx[i] - 1;
            if (g >= 0 && g < n_re_groups) eta[i] += x[p + g];
        }
        int t = t_idx[i] - 1;
        if (t >= 0 && t < n_t) eta[i] += x[t_start + t];
    }
}

// Joint scatter for (spatial via SpatialBlockOps store) × (indexed temporal).
// Calls scatter_obs_grad_hess_base for the (β, re) self / cross blocks, then
// walks every obs once and accumulates: spatial diagonal + within-spatial
// cross + (β, spatial), (re, spatial), (spatial, temporal) cross + temporal
// diagonal + (β, temporal) + (re, temporal). Each per-obs spatial DOF is
// read from the flat CSR-like (obs_p, obs_local_idx, obs_weight) store.
inline void nl_scatter_obs_spatial_x_indexed_temporal(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups,
    const Rcpp::NumericVector& eta,
    const std::string& family, double phi,
    int s_start,
    const std::vector<int>&    obs_p,
    const std::vector<int>&    obs_local_idx,
    const std::vector<double>& obs_weight,
    int t_start, int n_t, const Rcpp::IntegerVector& t_idx,
    tulpa::DenseVec& grad, tulpa::DenseMat& H,
    int n_threads
) {
    tulpa::scatter_obs_grad_hess_base(y, n_trials, X, re_idx, N, p, n_re_groups,
                                       eta, family, phi, grad, H, n_threads);

    for (int i = 0; i < N; i++) {
        int t = t_idx[i] - 1;
        bool has_t = (t >= 0 && t < n_t);
        int idx_t = has_t ? (t_start + t) : -1;
        int s_beg = obs_p[i];
        int s_end = obs_p[i + 1];
        bool has_s = (s_end > s_beg);
        if (!has_s && !has_t) continue;

        auto gh = tulpa::grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi);

        // Spatial-side contributions: diagonal + within-spatial cross
        // (upper triangle once, mirrored), spatial×β, spatial×re,
        // spatial×temporal. For single-DOF blocks (ICAR / CAR_proper) the
        // inner loop runs once; for BYM2 it runs twice; for HSGP it runs
        // M times and the M×M cross block dominates the cost.
        for (int e = s_beg; e < s_end; e++) {
            int idx_a = s_start + obs_local_idx[e];
            double w_a = obs_weight[e];
            grad[idx_a] += gh.grad * w_a;
            H[idx_a][idx_a] += gh.neg_hess * w_a * w_a;
            for (int e2 = e + 1; e2 < s_end; e2++) {
                int idx_b = s_start + obs_local_idx[e2];
                double w_b = obs_weight[e2];
                double cross = gh.neg_hess * w_a * w_b;
                H[idx_a][idx_b] += cross;
                H[idx_b][idx_a] += cross;
            }
            for (int j = 0; j < p; j++) {
                double cross = gh.neg_hess * w_a * X(i, j);
                H[j][idx_a]     += cross;
                H[idx_a][j]     += cross;
            }
            if (n_re_groups > 0) {
                int g = (int)re_idx[i] - 1;
                if (g >= 0 && g < n_re_groups) {
                    double cross = gh.neg_hess * w_a;
                    H[p + g][idx_a] += cross;
                    H[idx_a][p + g] += cross;
                }
            }
            if (has_t) {
                double cross = gh.neg_hess * w_a;
                H[idx_t][idx_a] += cross;
                H[idx_a][idx_t] += cross;
            }
        }

        // Temporal-only contributions (single-DOF, weight 1).
        if (has_t) {
            grad[idx_t] += gh.grad;
            H[idx_t][idx_t] += gh.neg_hess;
            for (int j = 0; j < p; j++) {
                H[j][idx_t]     += gh.neg_hess * X(i, j);
                H[idx_t][j]     += gh.neg_hess * X(i, j);
            }
            if (n_re_groups > 0) {
                int g = (int)re_idx[i] - 1;
                if (g >= 0 && g < n_re_groups) {
                    H[p + g][idx_t] += gh.neg_hess;
                    H[idx_t][p + g] += gh.neg_hess;
                }
            }
        }
    }
}

// Generic outer-grid driver for joint (spatial via SpatialBlockOps) ×
// (indexed-temporal) nested Laplace. Both blocks contribute to the joint
// inner Newton; each owns its own (prep_at_k, add_prior_at_k, log_prior_at_k)
// callbacks. n_grid pairs the two hyperparameter trajectories — caller
// builds the Cartesian product of the per-axis grids R-side.
template<typename TempPrepFn, typename TempAddPriorFn, typename TempLogPriorFn>
inline Rcpp::List run_spatial_x_indexed_temporal_nested_laplace(
    int n_grid,
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const SpatialBlockOps& spatial_ops,
    int t_start, int n_t, const Rcpp::IntegerVector& t_idx,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    bool store_modes,
    const Rcpp::NumericVector& x_init,
    TempPrepFn prep_t_at_k,
    TempAddPriorFn add_prior_t_at_k,
    TempLogPriorFn log_prior_t_at_k,
    bool store_Q = false
) {
    int n_x = p + n_re_groups + spatial_ops.block_len + n_t;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    // Per-thread NewtonScratch pool. Outer-grid parallelism for spatio-
    // temporal kernels is not yet exposed to R (the win lives in the joint
    // multi-block driver); n_outer = 1 here.
    const int n_outer = 1;
    std::vector<tulpa::NewtonScratch> scratch_pool(n_outer);
    for (auto& s : scratch_pool) s.allocate(n_x, N);
    int s_start = spatial_ops.start;

    auto solve_at_theta = [&](int k,
                              const std::vector<double>& prev_mode,
                              tulpa::SparseCholeskySolver* solver)
        -> tulpa::LaplaceResult
    {
        if (!spatial_ops.prep_at_k(k) || !prep_t_at_k(k)) {
            tulpa::LaplaceResult bad;
            bad.mode = (static_cast<int>(prev_mode.size()) == n_x)
                       ? prev_mode
                       : std::vector<double>(n_x, 0.0);
            bad.log_marginal = -std::numeric_limits<double>::infinity();
            bad.n_iter = 0;
            bad.converged = false;
            bad.log_det_Q = 0.0;
            return bad;
        }

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            nl_compute_eta_base_x_indexed_temporal(
                x, eta, N, p, n_re_groups, X, re_idx,
                t_start, n_t, t_idx, n_threads);
            spatial_ops.add_eta(x, eta, N, n_threads);
        };

        auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                           tulpa::DenseVec& grad, tulpa::DenseMat& H) {
            nl_scatter_obs_spatial_x_indexed_temporal(
                y, n_trials, X, re_idx, N, p, n_re_groups,
                eta, family, phi,
                s_start,
                *spatial_ops.obs_p, *spatial_ops.obs_local_idx, *spatial_ops.obs_weight,
                t_start, n_t, t_idx,
                grad, H, n_threads);
            spatial_ops.add_prior_at_k(grad, H, x, k);
            add_prior_t_at_k(grad, H, x, k);
            tulpa::add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
        };

        auto center = [&](Rcpp::NumericVector& x) {
            spatial_ops.center(x);
            tulpa::center_effects(x, t_start, n_t);
        };

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            lp += spatial_ops.log_prior_at_k(x, k);
            lp += log_prior_t_at_k(x, k);
            return lp;
        };

        int tid;
        #ifdef _OPENMP
        tid = omp_in_parallel() ? omp_get_thread_num() : 0;
        #else
        tid = 0;
        #endif
        return tulpa::laplace_newton_solve(
            y, n_trials, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter, center, log_prior,
            scratch_pool[tid], prev_mode, solver, store_Q
        );
    };

    return tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, store_modes, n_outer
    );
}

// ---- per-kind prior callbacks ---------------------------------------------
// Two bundles:
//   - IndexedPriorOps: prior-only bundle for indexed-temporal kinds (RW1,
//     RW2, AR1). The temporal side is always single-DOF indexed; the design
//     contribution is `eta[i] += x[t_start + t_idx[i] - 1]` and lives in
//     the joint driver, not in the bundle.
//   - SpatialBlockOps (defined above): design + prior bundle for any
//     spatial kind. Adding a new (spatial_kind × temporal_kind) ST
//     combination is now a matter of picking the matching pair of factories
//     below — the shared joint driver
//     (run_spatial_x_indexed_temporal_nested_laplace) is parameter-free in
//     the spatial / temporal shape.
//
// The lambdas capture the per-kind state (grids, adjacency, basis, ...)
// by reference. The caller (the Rcpp entry function) is responsible for
// keeping those references alive: the Rcpp::Vector / Rcpp::IntegerVector
// arguments live on the entry's stack and outlive the ops bundle.

struct IndexedPriorOps {
    std::function<bool(int)> prep;
    std::function<void(tulpa::DenseVec&, tulpa::DenseMat&,
                       const Rcpp::NumericVector&, int)> add_prior;
    std::function<double(const Rcpp::NumericVector&, int)> log_prior;

    // Sparse-path fields (Stage 1.4b). Populated by all RW1/RW2/AR1 ops
    // factories using the sparse twins added in 1.3a.
    std::function<void(std::vector<std::pair<int,int>>&)>
        add_prior_pattern;
    std::function<void(tulpa::SparseHessianBuilder&, tulpa::DenseVec&,
                       const Rcpp::NumericVector&, int)>
        add_prior_sparse;
};

// RW1 — 1D τ grid; cyclic flag closes the chain.
inline IndexedPriorOps make_rw1_ops(
    int start, int n_units,
    const Rcpp::NumericVector& tau_grid,
    bool cyclic
) {
    IndexedPriorOps ops;
    ops.prep = [](int) { return true; };
    ops.add_prior = [start, n_units, &tau_grid, cyclic]
                    (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                     const Rcpp::NumericVector& x, int k) {
        tulpa::add_rw1_precision(grad, H, x, start, n_units, tau_grid[k], cyclic);
    };
    ops.log_prior = [start, n_units, &tau_grid, cyclic]
                    (const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_rw1(x, start, n_units, tau_grid[k], cyclic);
    };
    ops.add_prior_pattern = [start, n_units, cyclic]
                            (std::vector<std::pair<int,int>>& out) {
        tulpa::add_rw1_pattern(out, start, n_units, cyclic);
    };
    ops.add_prior_sparse = [start, n_units, &tau_grid, cyclic]
                           (tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                            const Rcpp::NumericVector& x, int k) {
        tulpa::add_rw1_precision_sparse(grad, H, x, start, n_units,
                                          tau_grid[k], cyclic);
    };
    return ops;
}

// RW2 — 1D τ grid; no cyclic flag (the RW2 implementation ignores it).
inline IndexedPriorOps make_rw2_ops(
    int start, int n_units,
    const Rcpp::NumericVector& tau_grid
) {
    IndexedPriorOps ops;
    ops.prep = [](int) { return true; };
    ops.add_prior = [start, n_units, &tau_grid]
                    (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                     const Rcpp::NumericVector& x, int k) {
        tulpa::add_rw2_precision(grad, H, x, start, n_units, tau_grid[k], false);
    };
    ops.log_prior = [start, n_units, &tau_grid]
                    (const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_rw2(x, start, n_units, tau_grid[k], false);
    };
    ops.add_prior_pattern = [start, n_units]
                            (std::vector<std::pair<int,int>>& out) {
        tulpa::add_rw2_pattern(out, start, n_units, /*cyclic=*/false);
    };
    ops.add_prior_sparse = [start, n_units, &tau_grid]
                           (tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                            const Rcpp::NumericVector& x, int k) {
        tulpa::add_rw2_precision_sparse(grad, H, x, start, n_units,
                                          tau_grid[k], false);
    };
    return ops;
}

// AR1 — 2D (τ, ρ) grid.
inline IndexedPriorOps make_ar1_ops(
    int start, int n_units,
    const Rcpp::NumericVector& tau_grid,
    const Rcpp::NumericVector& rho_grid
) {
    IndexedPriorOps ops;
    ops.prep = [](int) { return true; };
    ops.add_prior = [start, n_units, &tau_grid, &rho_grid]
                    (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                     const Rcpp::NumericVector& x, int k) {
        tulpa::add_ar1_precision(grad, H, x, start, n_units,
                                  tau_grid[k], rho_grid[k]);
    };
    ops.log_prior = [start, n_units, &tau_grid, &rho_grid]
                    (const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_ar1(x, start, n_units, tau_grid[k], rho_grid[k]);
    };
    ops.add_prior_pattern = [start, n_units]
                            (std::vector<std::pair<int,int>>& out) {
        tulpa::add_ar1_pattern(out, start, n_units);
    };
    ops.add_prior_sparse = [start, n_units, &tau_grid, &rho_grid]
                           (tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                            const Rcpp::NumericVector& x, int k) {
        tulpa::add_ar1_precision_sparse(grad, H, x, start, n_units,
                                          tau_grid[k], rho_grid[k]);
    };
    return ops;
}

// ---- spatial block factories ---------------------------------------------
// Each factory returns a SpatialBlockOps bundle that owns its per-obs design
// store (obs_p / obs_local_idx / obs_weight) and the prior callbacks. The
// store is shared by the joint driver via shared_ptr so per-k weight
// updates (BYM2, HSGP) are visible to the next scatter call without
// re-allocating.

// ICAR — single-DOF design (∂η_i/∂x[start + s_idx[i]-1] = 1). The design
// store is built once and never modified.
inline SpatialBlockOps make_icar_spatial_ops(
    int start, int n_units, int N,
    const Rcpp::IntegerVector& spatial_idx,
    const Rcpp::NumericVector& tau_grid,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
) {
    SpatialBlockOps ops;
    ops.start = start;
    ops.block_len = n_units;
    auto obs_p_v  = std::make_shared<std::vector<int>>(N + 1, 0);
    auto obs_li_v = std::make_shared<std::vector<int>>();
    auto obs_w_v  = std::make_shared<std::vector<double>>();
    obs_li_v->reserve(N);
    obs_w_v->reserve(N);
    int cur = 0;
    for (int i = 0; i < N; i++) {
        (*obs_p_v)[i] = cur;
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_units) {
            obs_li_v->push_back(s);
            obs_w_v->push_back(1.0);
            cur++;
        }
    }
    (*obs_p_v)[N] = cur;
    ops.obs_p = obs_p_v;
    ops.obs_local_idx = obs_li_v;
    ops.obs_weight = obs_w_v;

    ops.prep_at_k = [](int) { return true; };
    ops.add_eta = [start, n_units, &spatial_idx]
                  (const Rcpp::NumericVector& x, Rcpp::NumericVector& eta,
                   int Nn, int nt) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt > 0 ? nt : 1)
#endif
        for (int i = 0; i < Nn; i++) {
            int s = spatial_idx[i] - 1;
            if (s >= 0 && s < n_units) eta[i] += x[start + s];
        }
    };
    ops.add_prior_at_k = [start, n_units, &tau_grid,
                          &adj_row_ptr, &adj_col_idx, &n_neighbors]
                         (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                          const Rcpp::NumericVector& x, int k) {
        tulpa::add_icar_prior(grad, H, x, start, n_units, tau_grid[k],
                               adj_row_ptr, adj_col_idx, n_neighbors);
    };
    ops.log_prior_at_k = [start, n_units, &tau_grid,
                          &adj_row_ptr, &adj_col_idx, &n_neighbors]
                         (const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_icar(x, start, n_units, tau_grid[k],
                                       adj_row_ptr, adj_col_idx, n_neighbors);
    };
    ops.add_prior_pattern = [start, n_units, &adj_row_ptr, &adj_col_idx]
                            (std::vector<std::pair<int,int>>& out) {
        tulpa::add_car_pattern(out, start, n_units, adj_row_ptr, adj_col_idx);
    };
    ops.add_prior_sparse = [start, n_units, &tau_grid,
                             &adj_row_ptr, &adj_col_idx, &n_neighbors]
                            (tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                             const Rcpp::NumericVector& x, int k) {
        tulpa::add_icar_prior_sparse(grad, H, x, start, n_units, tau_grid[k],
                                       adj_row_ptr, adj_col_idx, n_neighbors);
    };
    ops.center = [start, n_units](Rcpp::NumericVector& x) {
        tulpa::center_effects(x, start, n_units);
    };
    return ops;
}

// CAR_proper — single-DOF design like ICAR; prep_at_k recomputes log|Q(ρ_k)|.
inline SpatialBlockOps make_car_proper_spatial_ops(
    int start, int n_units, int N,
    const Rcpp::IntegerVector& spatial_idx,
    const Rcpp::NumericVector& tau_grid,
    const Rcpp::NumericVector& rho_grid,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
) {
    auto adj_rp_v = std::make_shared<std::vector<int>>(
        adj_row_ptr.begin(), adj_row_ptr.end());
    auto adj_ci_v = std::make_shared<std::vector<int>>(
        adj_col_idx.begin(), adj_col_idx.end());
    auto n_nbr_v  = std::make_shared<std::vector<int>>(
        n_neighbors.begin(),  n_neighbors.end());
    auto log_det_Q_rho = std::make_shared<double>(0.0);

    SpatialBlockOps ops;
    ops.start = start;
    ops.block_len = n_units;
    auto obs_p_v  = std::make_shared<std::vector<int>>(N + 1, 0);
    auto obs_li_v = std::make_shared<std::vector<int>>();
    auto obs_w_v  = std::make_shared<std::vector<double>>();
    obs_li_v->reserve(N);
    obs_w_v->reserve(N);
    int cur = 0;
    for (int i = 0; i < N; i++) {
        (*obs_p_v)[i] = cur;
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_units) {
            obs_li_v->push_back(s);
            obs_w_v->push_back(1.0);
            cur++;
        }
    }
    (*obs_p_v)[N] = cur;
    ops.obs_p = obs_p_v;
    ops.obs_local_idx = obs_li_v;
    ops.obs_weight = obs_w_v;

    ops.prep_at_k = [n_units, &rho_grid, adj_rp_v, adj_ci_v, n_nbr_v,
                     log_det_Q_rho](int k) -> bool {
        std::vector<double> Qmat = tulpa_car_proper::compute_car_precision(
            n_units, *adj_rp_v, *adj_ci_v, *n_nbr_v, rho_grid[k]);
        *log_det_Q_rho = tulpa_car_proper::car_log_det(n_units, Qmat);
        return std::isfinite(*log_det_Q_rho);
    };
    ops.add_eta = [start, n_units, &spatial_idx]
                  (const Rcpp::NumericVector& x, Rcpp::NumericVector& eta,
                   int Nn, int nt) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt > 0 ? nt : 1)
#endif
        for (int i = 0; i < Nn; i++) {
            int s = spatial_idx[i] - 1;
            if (s >= 0 && s < n_units) eta[i] += x[start + s];
        }
    };
    ops.add_prior_at_k = [start, n_units, &tau_grid, &rho_grid,
                          &adj_row_ptr, &adj_col_idx, &n_neighbors]
                         (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                          const Rcpp::NumericVector& x, int k) {
        tulpa::add_car_proper_prior(grad, H, x, start, n_units,
                                     tau_grid[k], rho_grid[k],
                                     adj_row_ptr, adj_col_idx, n_neighbors);
    };
    ops.log_prior_at_k = [start, n_units, &tau_grid, &rho_grid,
                          &adj_row_ptr, &adj_col_idx, &n_neighbors,
                          log_det_Q_rho]
                         (const Rcpp::NumericVector& x, int k) {
        return tulpa::log_prior_car_proper(x, start, n_units,
                                             tau_grid[k], rho_grid[k],
                                             *log_det_Q_rho,
                                             adj_row_ptr, adj_col_idx, n_neighbors);
    };
    ops.add_prior_pattern = [start, n_units, &adj_row_ptr, &adj_col_idx]
                            (std::vector<std::pair<int,int>>& out) {
        tulpa::add_car_pattern(out, start, n_units, adj_row_ptr, adj_col_idx);
    };
    ops.add_prior_sparse = [start, n_units, &tau_grid, &rho_grid,
                             &adj_row_ptr, &adj_col_idx, &n_neighbors]
                            (tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                             const Rcpp::NumericVector& x, int k) {
        tulpa::add_car_proper_prior_sparse(grad, H, x, start, n_units,
                                             tau_grid[k], rho_grid[k],
                                             adj_row_ptr, adj_col_idx, n_neighbors);
    };
    ops.center = [start, n_units](Rcpp::NumericVector& x) {
        tulpa::center_effects(x, start, n_units);
    };
    return ops;
}

// BYM2 — two-DOF design per obs. The structured reparameterisation is
//   eta_s = σ_k · (√ρ_k · scale_factor · φ_s  +  √(1−ρ_k) · θ_s)
// where φ has an ICAR prior (with τ = 1, since σ is folded into the design)
// and θ is iid N(0, 1). Per-k weights (d_phi, d_theta) are refreshed by
// prep_at_k and broadcast to the obs_weight store.
inline SpatialBlockOps make_bym2_spatial_ops(
    int start, int n_s, int N,
    const Rcpp::IntegerVector& spatial_idx,
    double scale_factor,
    const Rcpp::NumericVector& sigma_spatial_grid,
    const Rcpp::NumericVector& rho_grid,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
) {
    SpatialBlockOps ops;
    ops.start = start;
    ops.block_len = 2 * n_s;
    int phi_start   = start;
    int theta_start = start + n_s;

    auto obs_p_v  = std::make_shared<std::vector<int>>(N + 1, 0);
    auto obs_li_v = std::make_shared<std::vector<int>>();
    auto obs_w_v  = std::make_shared<std::vector<double>>();
    obs_li_v->reserve(2 * N);
    obs_w_v->reserve(2 * N);
    int cur = 0;
    for (int i = 0; i < N; i++) {
        (*obs_p_v)[i] = cur;
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_s) {
            obs_li_v->push_back(s);          // phi local index
            obs_li_v->push_back(n_s + s);    // theta local index
            obs_w_v->push_back(0.0);
            obs_w_v->push_back(0.0);
            cur += 2;
        }
    }
    (*obs_p_v)[N] = cur;
    ops.obs_p = obs_p_v;
    ops.obs_local_idx = obs_li_v;
    ops.obs_weight = obs_w_v;

    auto sigma_k_p    = std::make_shared<double>(0.0);
    auto sqrt_rho_p   = std::make_shared<double>(0.0);
    auto sqrt_1_rho_p = std::make_shared<double>(0.0);

    ops.prep_at_k = [obs_w_v, &sigma_spatial_grid, &rho_grid, scale_factor,
                     sigma_k_p, sqrt_rho_p, sqrt_1_rho_p]
                    (int k) -> bool {
        double sigma_k    = sigma_spatial_grid[k];
        double rho_k      = rho_grid[k];
        double sqrt_rho   = std::sqrt(rho_k + 1e-10);
        double sqrt_1_rho = std::sqrt(1.0 - rho_k + 1e-10);
        *sigma_k_p    = sigma_k;
        *sqrt_rho_p   = sqrt_rho;
        *sqrt_1_rho_p = sqrt_1_rho;
        double d_phi   = sigma_k * sqrt_rho * scale_factor;
        double d_theta = sigma_k * sqrt_1_rho;
        for (size_t e = 0; e + 1 < obs_w_v->size(); e += 2) {
            (*obs_w_v)[e]     = d_phi;
            (*obs_w_v)[e + 1] = d_theta;
        }
        return true;
    };
    ops.add_eta = [phi_start, theta_start, n_s, &spatial_idx, scale_factor,
                   sigma_k_p, sqrt_rho_p, sqrt_1_rho_p]
                  (const Rcpp::NumericVector& x, Rcpp::NumericVector& eta,
                   int Nn, int nt) {
        double sigma_k    = *sigma_k_p;
        double sqrt_rho   = *sqrt_rho_p;
        double sqrt_1_rho = *sqrt_1_rho_p;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt > 0 ? nt : 1)
#endif
        for (int i = 0; i < Nn; i++) {
            int s = spatial_idx[i] - 1;
            if (s >= 0 && s < n_s) {
                eta[i] += sigma_k * (sqrt_rho * x[phi_start + s] * scale_factor +
                                       sqrt_1_rho * x[theta_start + s]);
            }
        }
    };
    ops.add_prior_at_k = [phi_start, theta_start, n_s,
                          &adj_row_ptr, &adj_col_idx, &n_neighbors]
                         (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                          const Rcpp::NumericVector& x, int /*k*/) {
        // phi: ICAR with τ = 1 (σ folded into the design)
        tulpa::add_icar_prior(grad, H, x, phi_start, n_s, 1.0,
                               adj_row_ptr, adj_col_idx, n_neighbors);
        // theta: N(0, I)
        for (int s = 0; s < n_s; s++) {
            int idx = theta_start + s;
            grad[idx] -= x[idx];
            H[idx][idx] += 1.0;
        }
    };
    ops.log_prior_at_k = [phi_start, theta_start, n_s,
                          &adj_row_ptr, &adj_col_idx, &n_neighbors]
                         (const Rcpp::NumericVector& x, int /*k*/) {
        double lp_theta = 0.0;
        for (int s = 0; s < n_s; s++) {
            lp_theta -= 0.5 * x[theta_start + s] * x[theta_start + s];
        }
        lp_theta -= 0.5 * n_s * std::log(2.0 * M_PI);
        double quad_form = 0.0;
        for (int s = 0; s < n_s; s++) {
            double phi_s = x[phi_start + s];
            quad_form += n_neighbors[s] * phi_s * phi_s;
            for (int kk = adj_row_ptr[s]; kk < adj_row_ptr[s + 1]; kk++) {
                int neighbor = adj_col_idx[kk];
                if (neighbor > s) quad_form -= 2.0 * phi_s * x[phi_start + neighbor];
            }
        }
        return lp_theta + (-0.5 * quad_form);
    };
    // Pattern: phi gets ICAR adjacency at phi_start; theta gets only its
    // diagonal (the pattern builder adds it unconditionally for every block
    // index in [start, start + block_len)).
    ops.add_prior_pattern = [phi_start, n_s, &adj_row_ptr, &adj_col_idx]
                            (std::vector<std::pair<int,int>>& out) {
        tulpa::add_car_pattern(out, phi_start, n_s, adj_row_ptr, adj_col_idx);
    };
    ops.add_prior_sparse = [phi_start, theta_start, n_s,
                             &adj_row_ptr, &adj_col_idx, &n_neighbors]
                            (tulpa::SparseHessianBuilder& H, tulpa::DenseVec& grad,
                             const Rcpp::NumericVector& x, int /*k*/) {
        // phi: ICAR with τ = 1 (σ folded into the design — matches dense).
        tulpa::add_icar_prior_sparse(grad, H, x, phi_start, n_s, 1.0,
                                       adj_row_ptr, adj_col_idx, n_neighbors);
        // theta: N(0, I)
        for (int s = 0; s < n_s; s++) {
            int idx = theta_start + s;
            grad[idx] -= x[idx];
            H.add(idx, idx, 1.0);
        }
    };
    ops.center = [phi_start, n_s](Rcpp::NumericVector& x) {
        // Only center phi (the ICAR half); theta is N(0, I) and stays free.
        tulpa::center_effects(x, phi_start, n_s);
    };
    return ops;
}

// HSGP — dense M-DOF design per obs. eta_i = Σ_j Φ_ij · √S(λ_j; σ²_k, ℓ_k) · β_M_j
// with β_M ~ N(0, I_M). prep_at_k refreshes the per-k spectral table
// √S_j and broadcasts the full N×M weight matrix Φ · diag(√S) into
// obs_weight (stored row-major: obs i has its M weights at positions
// [i*M, (i+1)*M) of obs_weight).
inline SpatialBlockOps make_hsgp_spatial_ops(
    int start, int N,
    const Rcpp::NumericMatrix& phi_basis,
    const Rcpp::NumericVector& lambda_eig,
    const Rcpp::NumericVector& sigma2_grid,
    const Rcpp::NumericVector& lengthscale_grid
) {
    int M = phi_basis.ncol();
    SpatialBlockOps ops;
    ops.start = start;
    ops.block_len = M;

    auto obs_p_v  = std::make_shared<std::vector<int>>(N + 1, 0);
    auto obs_li_v = std::make_shared<std::vector<int>>(
        static_cast<size_t>(N) * static_cast<size_t>(M), 0);
    auto obs_w_v  = std::make_shared<std::vector<double>>(
        static_cast<size_t>(N) * static_cast<size_t>(M), 0.0);
    auto sqrt_S_v = std::make_shared<std::vector<double>>(M, 0.0);
    for (int i = 0; i < N; i++) {
        (*obs_p_v)[i] = i * M;
        for (int j = 0; j < M; j++) {
            (*obs_li_v)[static_cast<size_t>(i) * M + j] = j;
        }
    }
    (*obs_p_v)[N] = N * M;
    ops.obs_p = obs_p_v;
    ops.obs_local_idx = obs_li_v;
    ops.obs_weight = obs_w_v;

    ops.prep_at_k = [N, M, &phi_basis, &lambda_eig, &sigma2_grid,
                     &lengthscale_grid, obs_w_v, sqrt_S_v]
                    (int k) -> bool {
        double sigma2_k = sigma2_grid[k];
        double ell_k    = lengthscale_grid[k];
        for (int j = 0; j < M; j++) {
            double S = tulpa_hsgp::spectral_density_se(lambda_eig[j], sigma2_k, ell_k);
            (*sqrt_S_v)[j] = std::sqrt(std::max(S, 1e-30));
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                (*obs_w_v)[static_cast<size_t>(i) * M + j] =
                    phi_basis(i, j) * (*sqrt_S_v)[j];
            }
        }
        return true;
    };
    ops.add_eta = [start, M, &phi_basis, sqrt_S_v]
                  (const Rcpp::NumericVector& x, Rcpp::NumericVector& eta,
                   int Nn, int nt) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt > 0 ? nt : 1)
#endif
        for (int i = 0; i < Nn; i++) {
            double e = 0.0;
            for (int j = 0; j < M; j++) {
                e += phi_basis(i, j) * (*sqrt_S_v)[j] * x[start + j];
            }
            eta[i] += e;
        }
    };
    ops.add_prior_at_k = [start, M](tulpa::DenseVec& grad, tulpa::DenseMat& H,
                                     const Rcpp::NumericVector& x, int /*k*/) {
        for (int j = 0; j < M; j++) {
            int idx = start + j;
            grad[idx] -= x[idx];
            H[idx][idx] += 1.0;
        }
    };
    ops.log_prior_at_k = [start, M](const Rcpp::NumericVector& x, int /*k*/) {
        double q = 0.0;
        for (int j = 0; j < M; j++) {
            double b = x[start + j];
            q += b * b;
        }
        return -0.5 * q - 0.5 * M * std::log(2.0 * M_PI);
    };
    ops.center = [](Rcpp::NumericVector&) {};
    return ops;
}

// NNGP — single-DOF design per obs (∂η_i/∂x[start + s_idx[i]-1] = 1), like ICAR.
// prep_at_k captures (σ²_k, φ_gp_k); the NNGP prior contribution to the
// scatter is recomputed inside add_prior_at_k (batch_nngp_scatter depends on
// the current latent w, so it must be re-run every Newton iteration).
//
// The prior contribution is the full NNGP precision Λ = (I-A)' D⁻¹ (I-A):
// each row's q_i = w_i - sum_k a_{i,k} w_{N(i)_k} pushes into the focal
// point, every neighbor, and every (neighbor_k, neighbor_kp) pair, so the
// scatter is off-diagonal in the spatial block. Matches the spatial-only
// nngp entry's full-prior scatter (laplace_mode_gp pattern, post-upgrade).
inline SpatialBlockOps make_nngp_spatial_ops(
    int start, int n_s, int N,
    const Rcpp::IntegerVector& spatial_idx,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::NumericMatrix& nn_dist,
    const Rcpp::IntegerVector& nn_order,
    int nn,
    const Rcpp::NumericVector& sigma2_grid,
    const Rcpp::NumericVector& phi_gp_grid,
    int cov_type
) {
    SpatialBlockOps ops;
    ops.start = start;
    ops.block_len = n_s;

    auto obs_p_v  = std::make_shared<std::vector<int>>(N + 1, 0);
    auto obs_li_v = std::make_shared<std::vector<int>>();
    auto obs_w_v  = std::make_shared<std::vector<double>>();
    obs_li_v->reserve(N);
    obs_w_v->reserve(N);
    int cur = 0;
    for (int i = 0; i < N; i++) {
        (*obs_p_v)[i] = cur;
        int s = spatial_idx[i] - 1;
        if (s >= 0 && s < n_s) {
            obs_li_v->push_back(s);
            obs_w_v->push_back(1.0);
            cur++;
        }
    }
    (*obs_p_v)[N] = cur;
    ops.obs_p = obs_p_v;
    ops.obs_local_idx = obs_li_v;
    ops.obs_weight = obs_w_v;

    auto sigma2_k_p = std::make_shared<double>(0.0);
    auto phi_gp_k_p = std::make_shared<double>(0.0);

    ops.prep_at_k = [sigma2_k_p, phi_gp_k_p, &sigma2_grid, &phi_gp_grid]
                    (int k) -> bool {
        *sigma2_k_p = sigma2_grid[k];
        *phi_gp_k_p = phi_gp_grid[k];
        return std::isfinite(*sigma2_k_p) && std::isfinite(*phi_gp_k_p) &&
               *sigma2_k_p > 0.0 && *phi_gp_k_p > 0.0;
    };
    ops.add_eta = [start, n_s, &spatial_idx]
                  (const Rcpp::NumericVector& x, Rcpp::NumericVector& eta,
                   int Nn, int nt) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt > 0 ? nt : 1)
#endif
        for (int i = 0; i < Nn; i++) {
            int s = spatial_idx[i] - 1;
            if (s >= 0 && s < n_s) eta[i] += x[start + s];
        }
    };
    ops.add_prior_at_k = [start, n_s, nn, cov_type,
                          &coords, &nn_idx, &nn_dist, &nn_order,
                          sigma2_k_p, phi_gp_k_p]
                         (tulpa::DenseVec& grad, tulpa::DenseMat& H,
                          const Rcpp::NumericVector& x, int /*k*/) {
        std::vector<double> w(n_s);
        for (int s = 0; s < n_s; s++) w[s] = x[start + s];
        std::vector<double> cm, cv, nngp_alpha;
        bool gpu_used;
        tulpa::batch_nngp_scatter(w, n_s, nn, *sigma2_k_p, *phi_gp_k_p, cov_type,
                                    coords, nn_idx, nn_dist, nn_order,
                                    cm, cv, gpu_used, &nngp_alpha);
        tulpa::apply_nngp_full_prior_dense(grad, H, w, nngp_alpha, cv,
                                             nn_idx, nn_order,
                                             n_s, nn, start);
    };
    ops.log_prior_at_k = [start, n_s, nn, cov_type,
                          &coords, &nn_idx, &nn_dist, &nn_order,
                          sigma2_k_p, phi_gp_k_p]
                         (const Rcpp::NumericVector& x, int /*k*/) {
        std::vector<double> w(n_s);
        for (int s = 0; s < n_s; s++) w[s] = x[start + s];
        std::vector<double> cm, cv;
        bool gpu_used;
        tulpa::batch_nngp_scatter(w, n_s, nn, *sigma2_k_p, *phi_gp_k_p, cov_type,
                                    coords, nn_idx, nn_dist, nn_order,
                                    cm, cv, gpu_used);
        double lp = 0.0;
        for (int s = 0; s < n_s; s++) {
            double resid = w[s] - cm[s];
            lp += -0.5 * std::log(2.0 * M_PI * cv[s]) -
                   0.5 * resid * resid / cv[s];
        }
        return lp;
    };
    ops.center = [](Rcpp::NumericVector&) {};
    return ops;
}

// =====================================================================
// Sparse-Hessian path for (spatial × indexed-temporal) nested Laplace.
//
// Mirrors the dense run_spatial_x_indexed_temporal_nested_laplace exactly,
// but: (a) builds the joint H sparsity pattern once at fit time, (b) writes
// values into a SparseHessianBuilder per Newton iteration, (c) factor/solves
// via CHOLMOD with no DenseMat H allocation. At n_x = 10^6 the dense H is
// 8 TB; this path bypasses it entirely.
//
// Wired for spatial kinds that populate add_prior_pattern / add_prior_sparse
// on SpatialBlockOps (ICAR, BYM2, CAR_proper). HSGP / NNGP carry the
// dense-only path until 1.4c migrates them to the LatentBlock interface.
// =====================================================================

// Pattern enumeration for ST joint Hessian. Latent layout:
//   [beta(p)] [re(n_re_groups)] [w_spatial(s_block_len)] [w_temporal(n_t)].
// Pattern is fit-level — does not depend on (k_grid, x) — so it is built
// once outside the outer-grid loop and reused.
inline void build_st_hessian_pattern(
    int N, int p, int n_re_groups,
    const Rcpp::NumericVector& re_idx,
    int s_start,
    const std::vector<int>&    obs_p,
    const std::vector<int>&    obs_local_idx,
    int s_block_len,
    int t_start, int n_t, const Rcpp::IntegerVector& t_idx,
    const SpatialBlockOps& spatial_ops,
    const IndexedPriorOps& temporal_ops,
    int n_x,
    tulpa::SparseHessianBuilder& out_builder
) {
    std::vector<std::pair<int,int>> entries;

    // β/β dense + β/RE dense + RE/RE diagonal.
    for (int j = 0; j < p; j++) {
        for (int l = 0; l <= j; l++) entries.emplace_back(j, l);
    }
    for (int g = 0; g < n_re_groups; g++) {
        for (int j = 0; j < p; j++) entries.emplace_back(p + g, j);
        entries.emplace_back(p + g, p + g);
    }

    // Spatial / temporal block diagonals.
    for (int s = 0; s < s_block_len; s++) {
        entries.emplace_back(s_start + s, s_start + s);
    }
    for (int t = 0; t < n_t; t++) {
        entries.emplace_back(t_start + t, t_start + t);
    }

    // Spatial / temporal prior patterns (off-diagonal — diagonals already
    // covered above).
    if (spatial_ops.add_prior_pattern) spatial_ops.add_prior_pattern(entries);
    if (temporal_ops.add_prior_pattern) temporal_ops.add_prior_pattern(entries);

    // Per-obs cross fill. Mirrors the dense scatter's inner-loop touches.
    for (int i = 0; i < N; i++) {
        int s_beg = obs_p[i];
        int s_end = obs_p[i + 1];
        bool has_s = (s_end > s_beg);
        int t = t_idx[i] - 1;
        bool has_t = (t >= 0 && t < n_t);
        if (!has_s && !has_t) continue;

        int g_re = -1;
        if (n_re_groups > 0) {
            int gi = static_cast<int>(re_idx[i]) - 1;
            if (gi >= 0 && gi < n_re_groups) g_re = p + gi;
        }

        // Spatial-side fill.
        for (int e = s_beg; e < s_end; e++) {
            int idx_a = s_start + obs_local_idx[e];
            // diagonal already covered.
            for (int e2 = e + 1; e2 < s_end; e2++) {
                int idx_b = s_start + obs_local_idx[e2];
                int hi = std::max(idx_a, idx_b);
                int lo = std::min(idx_a, idx_b);
                entries.emplace_back(hi, lo);
            }
            // β × spatial.
            for (int j = 0; j < p; j++) entries.emplace_back(idx_a, j);
            // RE × spatial.
            if (g_re >= 0) entries.emplace_back(idx_a, g_re);
            // temporal × spatial.
            if (has_t) {
                int idx_t = t_start + t;
                int hi = std::max(idx_t, idx_a);
                int lo = std::min(idx_t, idx_a);
                entries.emplace_back(hi, lo);
            }
        }

        // Temporal-side fill.
        if (has_t) {
            int idx_t = t_start + t;
            // diagonal already covered.
            for (int j = 0; j < p; j++) entries.emplace_back(idx_t, j);
            if (g_re >= 0) entries.emplace_back(idx_t, g_re);
        }
    }

    out_builder.init(n_x, entries);
}

// Sparse twin of nl_scatter_obs_spatial_x_indexed_temporal. Lower-triangle
// single-write semantics: every (a, b) pair with a != b is written exactly
// once via H.add(hi, lo, val). H.add() internally normalizes (r, c) →
// (max, min) — calling both H.add(a, b) and H.add(b, a) would double-count.
inline void nl_scatter_obs_spatial_x_indexed_temporal_sparse(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups,
    const Rcpp::NumericVector& eta,
    const std::string& family, double phi,
    int s_start,
    const std::vector<int>&    obs_p,
    const std::vector<int>&    obs_local_idx,
    const std::vector<double>& obs_weight,
    int t_start, int n_t, const Rcpp::IntegerVector& t_idx,
    tulpa::DenseVec& grad, tulpa::SparseHessianBuilder& H
) {
    for (int i = 0; i < N; i++) {
        auto gh = tulpa::grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi);

        // β block: gradient + diagonal Hessian (lower triangle l <= j).
        for (int j = 0; j < p; j++) {
            double Xij = X(i, j);
            grad[j] += gh.grad * Xij;
            for (int l = 0; l <= j; l++) {
                H.add(j, l, gh.neg_hess * Xij * X(i, l));
            }
        }

        int g_re = -1;
        if (n_re_groups > 0) {
            int gi = static_cast<int>(re_idx[i]) - 1;
            if (gi >= 0 && gi < n_re_groups) g_re = p + gi;
        }
        // RE block: gradient + diagonal + β cross.
        if (g_re >= 0) {
            grad[g_re] += gh.grad;
            H.add(g_re, g_re, gh.neg_hess);
            for (int j = 0; j < p; j++) {
                H.add(g_re, j, gh.neg_hess * X(i, j));
            }
        }

        int t = t_idx[i] - 1;
        bool has_t = (t >= 0 && t < n_t);
        int idx_t = has_t ? (t_start + t) : -1;
        int s_beg = obs_p[i];
        int s_end = obs_p[i + 1];
        bool has_s = (s_end > s_beg);
        if (!has_s && !has_t) continue;

        // Spatial-side contributions: diagonal + within-spatial cross +
        // β/RE/temporal cross.
        for (int e = s_beg; e < s_end; e++) {
            int idx_a = s_start + obs_local_idx[e];
            double w_a = obs_weight[e];
            grad[idx_a] += gh.grad * w_a;
            H.add(idx_a, idx_a, gh.neg_hess * w_a * w_a);
            for (int e2 = e + 1; e2 < s_end; e2++) {
                int idx_b = s_start + obs_local_idx[e2];
                double w_b = obs_weight[e2];
                int hi = std::max(idx_a, idx_b);
                int lo = std::min(idx_a, idx_b);
                H.add(hi, lo, gh.neg_hess * w_a * w_b);
            }
            for (int j = 0; j < p; j++) {
                H.add(idx_a, j, gh.neg_hess * w_a * X(i, j));
            }
            if (g_re >= 0) {
                H.add(idx_a, g_re, gh.neg_hess * w_a);
            }
            if (has_t) {
                int hi = std::max(idx_t, idx_a);
                int lo = std::min(idx_t, idx_a);
                H.add(hi, lo, gh.neg_hess * w_a);
            }
        }

        // Temporal-only contributions (single-DOF, weight 1).
        if (has_t) {
            grad[idx_t] += gh.grad;
            H.add(idx_t, idx_t, gh.neg_hess);
            for (int j = 0; j < p; j++) {
                H.add(idx_t, j, gh.neg_hess * X(i, j));
            }
            if (g_re >= 0) {
                H.add(idx_t, g_re, gh.neg_hess);
            }
        }
    }
}

// Sparse-path outer-grid driver for ST. Mirrors the dense runner exactly
// except H is sparse and the inner Newton calls laplace_newton_solve_sparse.
// Outer-grid serial — parallel sparse is a follow-up (the pattern would
// need per-thread builders; ~few × 10^7 entries at n_sites = 10^6 makes
// replication costly).
inline Rcpp::List run_spatial_x_indexed_temporal_nested_laplace_sparse_impl(
    int n_grid,
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const SpatialBlockOps& spatial_ops,
    int t_start, int n_t, const Rcpp::IntegerVector& t_idx,
    const IndexedPriorOps& temporal_ops,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    bool store_modes,
    const Rcpp::NumericVector& x_init,
    bool store_Q
) {
    int n_x = p + n_re_groups + spatial_ops.block_len + n_t;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    int s_start = spatial_ops.start;

    // Pattern built ONCE. Reused for every outer-grid cell. Note that the
    // spatial ops keep their obs_p/obs_local_idx pointers stable across
    // prep_at_k calls (only obs_weight values change for BYM2/HSGP), so
    // the data-induced pattern is grid-invariant.
    tulpa::SparseHessianBuilder H_builder;
    build_st_hessian_pattern(
        N, p, n_re_groups, re_idx, s_start,
        *spatial_ops.obs_p, *spatial_ops.obs_local_idx, spatial_ops.block_len,
        t_start, n_t, t_idx,
        spatial_ops, temporal_ops,
        n_x, H_builder
    );

    tulpa::NewtonScratch scratch;
    scratch.allocate(n_x, N);

    auto solve_at_theta = [&](int k,
                              const std::vector<double>& prev_mode,
                              tulpa::SparseCholeskySolver* solver)
        -> tulpa::LaplaceResult
    {
        if (!spatial_ops.prep_at_k(k) || !temporal_ops.prep(k)) {
            tulpa::LaplaceResult bad;
            bad.mode = (static_cast<int>(prev_mode.size()) == n_x)
                       ? prev_mode
                       : std::vector<double>(n_x, 0.0);
            bad.log_marginal = -std::numeric_limits<double>::infinity();
            bad.n_iter = 0;
            bad.converged = false;
            bad.log_det_Q = 0.0;
            return bad;
        }

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            nl_compute_eta_base_x_indexed_temporal(
                x, eta, N, p, n_re_groups, X, re_idx,
                t_start, n_t, t_idx, n_threads);
            spatial_ops.add_eta(x, eta, N, n_threads);
        };

        auto scatter_sparse = [&](const Rcpp::NumericVector& x,
                                   const Rcpp::NumericVector& eta,
                                   tulpa::DenseVec& grad,
                                   tulpa::SparseHessianBuilder& H) {
            nl_scatter_obs_spatial_x_indexed_temporal_sparse(
                y, n_trials, X, re_idx, N, p, n_re_groups,
                eta, family, phi,
                s_start,
                *spatial_ops.obs_p, *spatial_ops.obs_local_idx, *spatial_ops.obs_weight,
                t_start, n_t, t_idx,
                grad, H);
            spatial_ops.add_prior_sparse(H, grad, x, k);
            temporal_ops.add_prior_sparse(H, grad, x, k);
            // RE + β regularization (mirrors add_re_beta_priors).
            for (int g = 0; g < n_re_groups; g++) {
                grad[p + g] -= tau_re * x[p + g];
                H.add(p + g, p + g, tau_re);
            }
            const double tau_beta = 1e-4;
            for (int j = 0; j < p; j++) {
                grad[j] -= tau_beta * x[j];
                H.add(j, j, tau_beta);
            }
        };

        auto center = [&](Rcpp::NumericVector& x) {
            spatial_ops.center(x);
            tulpa::center_effects(x, t_start, n_t);
        };

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double lp = tulpa::compute_log_prior_re(x, p, n_re_groups, tau_re);
            lp += spatial_ops.log_prior_at_k(x, k);
            lp += temporal_ops.log_prior(x, k);
            return lp;
        };

        return tulpa::laplace_newton_solve_sparse(
            y, n_trials, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter_sparse, center, log_prior,
            H_builder, scratch,
            prev_mode, solver, store_Q
        );
    };

    return tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, store_modes, /*n_outer=*/1
    );
}

// Dispatch wrapper. Each Rcpp ST entry calls this — it builds the ops
// bundles, then this routes to sparse vs dense based on force_sparse and
// n_x. Sparse routing requires spatial_ops.add_prior_sparse and
// temporal_ops.add_prior_sparse to be populated; HSGP / NNGP spatial ops
// leave these empty until 1.4c, so force_sparse = TRUE on those errors out
// explicitly rather than silently falling back to dense.
inline Rcpp::List run_spatial_x_indexed_temporal_nested_laplace_dispatch(
    int n_grid,
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const SpatialBlockOps& spatial_ops,
    int t_start, int n_t, const Rcpp::IntegerVector& t_idx,
    const IndexedPriorOps& temporal_ops,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    bool store_modes,
    const Rcpp::NumericVector& x_init,
    bool store_Q,
    bool force_sparse
) {
    int n_x = p + n_re_groups + spatial_ops.block_len + n_t;
    bool sparse_supported = static_cast<bool>(spatial_ops.add_prior_sparse) &&
                            static_cast<bool>(temporal_ops.add_prior_sparse) &&
                            static_cast<bool>(spatial_ops.add_prior_pattern) &&
                            static_cast<bool>(temporal_ops.add_prior_pattern);
    if (force_sparse && !sparse_supported) {
        Rcpp::stop("force_sparse = TRUE but the sparse path is not yet wired "
                   "for this spatial kind (HSGP / NNGP ST). See 1.4c follow-up.");
    }
    if (force_sparse || (sparse_supported && n_x >= tulpa::SPARSE_THRESHOLD)) {
        return run_spatial_x_indexed_temporal_nested_laplace_sparse_impl(
            n_grid, y, n_trials, X, re_idx, N, p, n_re_groups, sigma_re,
            spatial_ops, t_start, n_t, t_idx, temporal_ops,
            family, phi, max_iter, tol, n_threads,
            store_modes, x_init, store_Q
        );
    }
    return run_spatial_x_indexed_temporal_nested_laplace(
        n_grid, y, n_trials, X, re_idx, N, p, n_re_groups, sigma_re,
        spatial_ops, t_start, n_t, t_idx,
        family, phi, max_iter, tol, n_threads,
        store_modes, x_init,
        temporal_ops.prep, temporal_ops.add_prior, temporal_ops.log_prior,
        store_Q
    );
}

} // namespace

// =====================================================================
// Nested Laplace: ICAR (spatial) × AR1 (temporal)
//   Joint over (τ_s) × (τ_t, ρ_t). Caller passes paired vectors of length
//   n_grid (the Cartesian product is built R-side).
//   Latent: [beta (p)] [re (n_re_groups)] [w_spatial (n_s)] [w_temporal (n_t)].
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_icar_ar1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    Rcpp::NumericVector rho_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = tau_spatial_grid.size();
    if (tau_temporal_grid.size() != n_grid || rho_temporal_grid.size() != n_grid) {
        Rcpp::stop("tau_spatial_grid, tau_temporal_grid, rho_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial_units;

    auto ops_s = make_icar_spatial_ops(s_start, n_spatial_units, N,
                                        spatial_idx, tau_spatial_grid,
                                        adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_ar1_ops(t_start, n_times, tau_temporal_grid, rho_temporal_grid);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    out["rho_temporal_grid"] = rho_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: ICAR (spatial) × RW1 (temporal)
//   1D τ_s × 1D τ_t grid. Caller passes paired vectors of length n_grid.
//   Latent: [beta] [re] [w_spatial (n_s)] [w_temporal (n_t)].
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_icar_rw1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times, bool cyclic,
    Rcpp::NumericVector tau_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = tau_spatial_grid.size();
    if (tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("tau_spatial_grid and tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial_units;

    auto ops_s = make_icar_spatial_ops(s_start, n_spatial_units, N,
                                        spatial_idx, tau_spatial_grid,
                                        adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_rw1_ops(t_start, n_times, tau_temporal_grid, cyclic);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: ICAR (spatial) × RW2 (temporal)
//   1D τ_s × 1D τ_t grid.
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_icar_rw2(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = tau_spatial_grid.size();
    if (tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("tau_spatial_grid and tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial_units;

    auto ops_s = make_icar_spatial_ops(s_start, n_spatial_units, N,
                                        spatial_idx, tau_spatial_grid,
                                        adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_rw2_ops(t_start, n_times, tau_temporal_grid);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: CAR_proper (spatial) × RW1 (temporal)
//   2D (τ_s, ρ_s) × 1D τ_t grid.
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_car_proper_rw1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times, bool cyclic,
    Rcpp::NumericVector tau_spatial_grid,
    Rcpp::NumericVector rho_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = tau_spatial_grid.size();
    if (rho_spatial_grid.size() != n_grid || tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("tau_spatial_grid, rho_spatial_grid, tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial_units;

    auto ops_s = make_car_proper_spatial_ops(s_start, n_spatial_units, N,
                                              spatial_idx,
                                              tau_spatial_grid, rho_spatial_grid,
                                              adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_rw1_ops(t_start, n_times, tau_temporal_grid, cyclic);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["rho_spatial_grid"]  = rho_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: CAR_proper (spatial) × RW2 (temporal)
//   2D (τ_s, ρ_s) × 1D τ_t grid.
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_car_proper_rw2(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_spatial_grid,
    Rcpp::NumericVector rho_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = tau_spatial_grid.size();
    if (rho_spatial_grid.size() != n_grid || tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("tau_spatial_grid, rho_spatial_grid, tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial_units;

    auto ops_s = make_car_proper_spatial_ops(s_start, n_spatial_units, N,
                                              spatial_idx,
                                              tau_spatial_grid, rho_spatial_grid,
                                              adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_rw2_ops(t_start, n_times, tau_temporal_grid);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["rho_spatial_grid"]  = rho_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: CAR_proper (spatial) × AR1 (temporal)
//   2D (τ_s, ρ_s) × 2D (τ_t, ρ_t) grid.
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_car_proper_ar1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector tau_spatial_grid,
    Rcpp::NumericVector rho_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    Rcpp::NumericVector rho_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = tau_spatial_grid.size();
    if (rho_spatial_grid.size() != n_grid ||
        tau_temporal_grid.size() != n_grid ||
        rho_temporal_grid.size() != n_grid) {
        Rcpp::stop("All four paired grids must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial_units;

    auto ops_s = make_car_proper_spatial_ops(s_start, n_spatial_units, N,
                                              spatial_idx,
                                              tau_spatial_grid, rho_spatial_grid,
                                              adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_ar1_ops(t_start, n_times, tau_temporal_grid, rho_temporal_grid);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["tau_spatial_grid"]  = tau_spatial_grid;
    out["rho_spatial_grid"]  = rho_spatial_grid;
    out["tau_temporal_grid"] = tau_temporal_grid;
    out["rho_temporal_grid"] = rho_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: BYM2 (spatial) × RW1 (temporal)
//   2D (σ_s, ρ_s) × 1D τ_t grid. cyclic flag closes the temporal chain.
//   Latent: [beta] [re] [phi (n_s) + theta (n_s)] [w_temporal (n_t)].
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_bym2_rw1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial_units,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    double scale_factor,
    Rcpp::IntegerVector temporal_idx, int n_times, bool cyclic,
    Rcpp::NumericVector sigma_spatial_grid,
    Rcpp::NumericVector rho_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = sigma_spatial_grid.size();
    if (rho_spatial_grid.size() != n_grid || tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("sigma_spatial_grid, rho_spatial_grid, tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + 2 * n_spatial_units;

    auto ops_s = make_bym2_spatial_ops(s_start, n_spatial_units, N,
                                        spatial_idx, scale_factor,
                                        sigma_spatial_grid, rho_spatial_grid,
                                        adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_rw1_ops(t_start, n_times, tau_temporal_grid, cyclic);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["sigma_spatial_grid"] = sigma_spatial_grid;
    out["rho_spatial_grid"]   = rho_spatial_grid;
    out["tau_temporal_grid"]  = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: BYM2 (spatial) × RW2 (temporal)
//   2D (σ_s, ρ_s) × 1D τ_t grid.
//   Latent: [beta] [re] [phi (n_s) + theta (n_s)] [w_temporal (n_t)].
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_bym2_rw2(
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
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = sigma_spatial_grid.size();
    if (rho_spatial_grid.size() != n_grid || tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("sigma_spatial_grid, rho_spatial_grid, tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + 2 * n_spatial_units;

    auto ops_s = make_bym2_spatial_ops(s_start, n_spatial_units, N,
                                        spatial_idx, scale_factor,
                                        sigma_spatial_grid, rho_spatial_grid,
                                        adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_rw2_ops(t_start, n_times, tau_temporal_grid);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["sigma_spatial_grid"] = sigma_spatial_grid;
    out["rho_spatial_grid"]   = rho_spatial_grid;
    out["tau_temporal_grid"]  = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: BYM2 (spatial) × AR1 (temporal)
//   2D (σ_s, ρ_s) × 2D (τ_t, ρ_t) grid (full 4-axis product).
//   Latent: [beta] [re] [phi (n_s) + theta (n_s)] [w_temporal (n_t)].
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_bym2_ar1(
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
    Rcpp::NumericVector tau_temporal_grid,
    Rcpp::NumericVector rho_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = sigma_spatial_grid.size();
    if (rho_spatial_grid.size() != n_grid ||
        tau_temporal_grid.size() != n_grid ||
        rho_temporal_grid.size() != n_grid) {
        Rcpp::stop("All four paired BYM2 x AR1 grids must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int s_start = p + n_re_groups;
    int t_start = s_start + 2 * n_spatial_units;

    auto ops_s = make_bym2_spatial_ops(s_start, n_spatial_units, N,
                                        spatial_idx, scale_factor,
                                        sigma_spatial_grid, rho_spatial_grid,
                                        adj_row_ptr, adj_col_idx, n_neighbors);
    auto ops_t = make_ar1_ops(t_start, n_times, tau_temporal_grid, rho_temporal_grid);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["sigma_spatial_grid"] = sigma_spatial_grid;
    out["rho_spatial_grid"]   = rho_spatial_grid;
    out["tau_temporal_grid"]  = tau_temporal_grid;
    out["rho_temporal_grid"]  = rho_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: HSGP (spatial) × RW1 (temporal)
//   2D (σ²_s, ℓ_s) × 1D τ_t grid. cyclic flag closes the temporal chain.
//   Latent: [beta] [re] [beta_M (n_basis)] [w_temporal (n_t)].
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_hsgp_rw1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix phi_basis,
    Rcpp::NumericVector lambda_eig,
    Rcpp::IntegerVector temporal_idx, int n_times, bool cyclic,
    Rcpp::NumericVector sigma2_spatial_grid,
    Rcpp::NumericVector lengthscale_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = sigma2_spatial_grid.size();
    if (lengthscale_spatial_grid.size() != n_grid ||
        tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("sigma2_spatial_grid, lengthscale_spatial_grid, tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int M = phi_basis.ncol();
    if (phi_basis.nrow() != N)
        Rcpp::stop("phi_basis must have N rows (one per observation)");
    if (lambda_eig.size() != M)
        Rcpp::stop("lambda_eig must have length ncol(phi_basis)");
    int s_start = p + n_re_groups;
    int t_start = s_start + M;

    auto ops_s = make_hsgp_spatial_ops(s_start, N, phi_basis, lambda_eig,
                                        sigma2_spatial_grid, lengthscale_spatial_grid);
    auto ops_t = make_rw1_ops(t_start, n_times, tau_temporal_grid, cyclic);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["sigma2_spatial_grid"]      = sigma2_spatial_grid;
    out["lengthscale_spatial_grid"] = lengthscale_spatial_grid;
    out["tau_temporal_grid"]        = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: HSGP (spatial) × RW2 (temporal)
//   2D (σ²_s, ℓ_s) × 1D τ_t grid.
//   Latent: [beta] [re] [beta_M (n_basis)] [w_temporal (n_t)].
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_hsgp_rw2(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix phi_basis,
    Rcpp::NumericVector lambda_eig,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector sigma2_spatial_grid,
    Rcpp::NumericVector lengthscale_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = sigma2_spatial_grid.size();
    if (lengthscale_spatial_grid.size() != n_grid ||
        tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("sigma2_spatial_grid, lengthscale_spatial_grid, tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int M = phi_basis.ncol();
    if (phi_basis.nrow() != N)
        Rcpp::stop("phi_basis must have N rows (one per observation)");
    if (lambda_eig.size() != M)
        Rcpp::stop("lambda_eig must have length ncol(phi_basis)");
    int s_start = p + n_re_groups;
    int t_start = s_start + M;

    auto ops_s = make_hsgp_spatial_ops(s_start, N, phi_basis, lambda_eig,
                                        sigma2_spatial_grid, lengthscale_spatial_grid);
    auto ops_t = make_rw2_ops(t_start, n_times, tau_temporal_grid);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["sigma2_spatial_grid"]      = sigma2_spatial_grid;
    out["lengthscale_spatial_grid"] = lengthscale_spatial_grid;
    out["tau_temporal_grid"]        = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: HSGP (spatial) × AR1 (temporal)
//   2D (σ²_s, ℓ_s) × 2D (τ_t, ρ_t) grid (full 4-axis product).
//   Latent: [beta] [re] [beta_M (n_basis)] [w_temporal (n_t)].
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_hsgp_ar1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix phi_basis,
    Rcpp::NumericVector lambda_eig,
    Rcpp::IntegerVector temporal_idx, int n_times,
    Rcpp::NumericVector sigma2_spatial_grid,
    Rcpp::NumericVector lengthscale_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    Rcpp::NumericVector rho_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = sigma2_spatial_grid.size();
    if (lengthscale_spatial_grid.size() != n_grid ||
        tau_temporal_grid.size() != n_grid ||
        rho_temporal_grid.size() != n_grid) {
        Rcpp::stop("All four paired HSGP x AR1 grids must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    int M = phi_basis.ncol();
    if (phi_basis.nrow() != N)
        Rcpp::stop("phi_basis must have N rows (one per observation)");
    if (lambda_eig.size() != M)
        Rcpp::stop("lambda_eig must have length ncol(phi_basis)");
    int s_start = p + n_re_groups;
    int t_start = s_start + M;

    auto ops_s = make_hsgp_spatial_ops(s_start, N, phi_basis, lambda_eig,
                                        sigma2_spatial_grid, lengthscale_spatial_grid);
    auto ops_t = make_ar1_ops(t_start, n_times, tau_temporal_grid, rho_temporal_grid);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["sigma2_spatial_grid"]      = sigma2_spatial_grid;
    out["lengthscale_spatial_grid"] = lengthscale_spatial_grid;
    out["tau_temporal_grid"]        = tau_temporal_grid;
    out["rho_temporal_grid"]        = rho_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: NNGP (spatial) × RW1 (temporal)
//   2D (σ²_s, φ_gp_s) × 1D τ_t grid. cyclic flag closes the temporal chain.
//   Latent: [beta] [re] [w_spatial (n_spatial)] [w_temporal (n_t)].
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_nngp_rw1(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::IntegerVector spatial_idx, int n_spatial,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order, int nn, int cov_type,
    Rcpp::IntegerVector temporal_idx, int n_times, bool cyclic,
    Rcpp::NumericVector sigma2_spatial_grid,
    Rcpp::NumericVector phi_gp_spatial_grid,
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = sigma2_spatial_grid.size();
    if (phi_gp_spatial_grid.size() != n_grid ||
        tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("sigma2_spatial_grid, phi_gp_spatial_grid, tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    if (spatial_idx.size() != N)
        Rcpp::stop("length(spatial_idx) must equal length(y)");
    if (coords.nrow() != n_spatial)
        Rcpp::stop("nrow(coords) must equal n_spatial");
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial;

    auto ops_s = make_nngp_spatial_ops(s_start, n_spatial, N, spatial_idx,
                                        coords, nn_idx, nn_dist, nn_order, nn,
                                        sigma2_spatial_grid, phi_gp_spatial_grid,
                                        cov_type);
    auto ops_t = make_rw1_ops(t_start, n_times, tau_temporal_grid, cyclic);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["sigma2_spatial_grid"] = sigma2_spatial_grid;
    out["phi_gp_spatial_grid"] = phi_gp_spatial_grid;
    out["tau_temporal_grid"]   = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: NNGP (spatial) × RW2 (temporal)
//   2D (σ²_s, φ_gp_s) × 1D τ_t grid.
//   Latent: [beta] [re] [w_spatial (n_spatial)] [w_temporal (n_t)].
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_nngp_rw2(
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
    Rcpp::NumericVector tau_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = sigma2_spatial_grid.size();
    if (phi_gp_spatial_grid.size() != n_grid ||
        tau_temporal_grid.size() != n_grid) {
        Rcpp::stop("sigma2_spatial_grid, phi_gp_spatial_grid, tau_temporal_grid must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    if (spatial_idx.size() != N)
        Rcpp::stop("length(spatial_idx) must equal length(y)");
    if (coords.nrow() != n_spatial)
        Rcpp::stop("nrow(coords) must equal n_spatial");
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial;

    auto ops_s = make_nngp_spatial_ops(s_start, n_spatial, N, spatial_idx,
                                        coords, nn_idx, nn_dist, nn_order, nn,
                                        sigma2_spatial_grid, phi_gp_spatial_grid,
                                        cov_type);
    auto ops_t = make_rw2_ops(t_start, n_times, tau_temporal_grid);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["sigma2_spatial_grid"] = sigma2_spatial_grid;
    out["phi_gp_spatial_grid"] = phi_gp_spatial_grid;
    out["tau_temporal_grid"]   = tau_temporal_grid;
    return out;
}

// =====================================================================
// Nested Laplace: NNGP (spatial) × AR1 (temporal)
//   2D (σ²_s, φ_gp_s) × 2D (τ_t, ρ_t) grid (full 4-axis product).
//   Latent: [beta] [re] [w_spatial (n_spatial)] [w_temporal (n_t)].
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_st_nngp_ar1(
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
    Rcpp::NumericVector tau_temporal_grid,
    Rcpp::NumericVector rho_temporal_grid,
    std::string family, double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    bool store_Q = false,
    bool force_sparse = false
) {
    int n_grid = sigma2_spatial_grid.size();
    if (phi_gp_spatial_grid.size() != n_grid ||
        tau_temporal_grid.size() != n_grid ||
        rho_temporal_grid.size() != n_grid) {
        Rcpp::stop("All four paired NNGP x AR1 grids must have the same length");
    }
    int N = y.size();
    int p = X.ncol();
    if (spatial_idx.size() != N)
        Rcpp::stop("length(spatial_idx) must equal length(y)");
    if (coords.nrow() != n_spatial)
        Rcpp::stop("nrow(coords) must equal n_spatial");
    int s_start = p + n_re_groups;
    int t_start = s_start + n_spatial;

    auto ops_s = make_nngp_spatial_ops(s_start, n_spatial, N, spatial_idx,
                                        coords, nn_idx, nn_dist, nn_order, nn,
                                        sigma2_spatial_grid, phi_gp_spatial_grid,
                                        cov_type);
    auto ops_t = make_ar1_ops(t_start, n_times, tau_temporal_grid, rho_temporal_grid);

    Rcpp::List out = run_spatial_x_indexed_temporal_nested_laplace_dispatch(
        n_grid, y, n, X, re_idx, N, p, n_re_groups, sigma_re,
        ops_s,
        t_start, n_times, temporal_idx,
        ops_t,
        family, phi, max_iter, tol, n_threads,
        /*store_modes=*/true,
        unwrap_x_init(x_init_nullable),
        store_Q,
        force_sparse
    );
    out["sigma2_spatial_grid"] = sigma2_spatial_grid;
    out["phi_gp_spatial_grid"] = phi_gp_spatial_grid;
    out["tau_temporal_grid"]   = tau_temporal_grid;
    out["rho_temporal_grid"]   = rho_temporal_grid;
    return out;
}

