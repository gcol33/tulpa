// laplace_core_gp.cpp
// GP (NNGP) Laplace at a FIXED (sigma2_gp, phi_gp).
//
// The conditional counterpart of cpp_nested_laplace_nngp: it builds the SAME
// single arm (make_single_arm) and the SAME NNGP LatentBlock (make_nngp_block)
// at a one-row (sigma2, phi_gp) grid and runs the shared joint multi-block
// driver, so the mode and log-marginal equal the nested kernel evaluated at that
// single grid cell -- bit-for-bit, at any n_x.
//
// The NNGP block scatters its prior into the sparse builder only, so
// blocks_require_sparse() routes this to the sparse Newton whatever n_x is. A
// small field therefore pays CHOLMOD overhead it could in principle skip; that
// is deliberate. The dense route disagrees with the sparse one on this block
// (measurably at nn = 5, divergently at nn = 8), and the spatio-temporal NNGP
// entry already pinned itself to sparse for the same reason.
//
// Routed from dispatch_laplace_spatial / tulpa_laplace(spatial = spatial_gp()).

#include "laplace_core.h"
#include "hessian_pattern_guard.h"          // HessianPatternGuard
#include "laplace_spec_fit.h"               // as_offset_vec, unwrap_skew_idx
#include "nested_laplace_grid.h"            // nl_grid_cell_to_result_list
#include "nested_laplace_joint_core.h"      // ParsedArm / JointArm, make_single_arm
#include "nested_laplace_joint_multi.h"     // run_multi_block_nested_laplace_joint
#include "nngp_block_factory.h"             // make_nngp_block
#include <Rcpp.h>
#include <vector>

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_gp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int n_spatial, int nn,
    double sigma2_gp, double phi_gp, int cov_type,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> obs_to_loc_nullable = R_NilValue,
    bool compute_skew = false,
    Rcpp::Nullable<Rcpp::IntegerVector> skew_idx = R_NilValue
) {
    const int N = y.size();
    const int p = X.ncol();
    // Layout [beta (p), re (n_re_groups), w_gp (n_spatial)], matching the
    // nested NNGP entry (cpp_nested_laplace_nngp).
    const int n_x = p + n_re_groups + n_spatial;
    const int gp_start = p + n_re_groups;

    if (coords.nrow() != n_spatial) {
        Rcpp::stop("nrow(coords) (%d) must equal n_spatial (%d).",
                   static_cast<int>(coords.nrow()), n_spatial);
    }

    // Per-observation 1-based field-node index. A supplied obs_to_loc attaches
    // each observation to its actual node, which is required when coordinates
    // repeat (n_spatial unique locations < N). Absent, the map is the identity
    // (obs i -> node i); observations past n_spatial get 0, which the driver
    // reads as "this obs does not see the block" -- the same rows the identity
    // bounds check used to drop.
    Rcpp::IntegerVector spatial_idx(N);
    if (obs_to_loc_nullable.isNotNull()) {
        Rcpp::IntegerVector otl(obs_to_loc_nullable);
        if (otl.size() != N) {
            Rcpp::stop("length(obs_to_loc) (%d) must equal length(y) (%d).",
                       static_cast<int>(otl.size()), N);
        }
        spatial_idx = otl;
    } else {
        for (int i = 0; i < N; i++) {
            spatial_idx[i] = (i < n_spatial) ? (i + 1) : 0;
        }
    }

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm> arms;
    tulpa::make_single_arm(parsed, arms, X, re_idx, spatial_idx,
                           p, n_re_groups, sigma_re, y, n, family, phi, N,
                           offset_nullable);

    // One-row (sigma2, phi_gp) grid; make_nngp_block reads the pair from
    // theta_grid(k, axis) in block.prep, so the row must outlive the solve.
    Rcpp::NumericMatrix theta_grid(1, 2);
    theta_grid(0, 0) = sigma2_gp;
    theta_grid(0, 1) = phi_gp;

    Rcpp::List spatial_idx_per_arm = Rcpp::List::create(spatial_idx);
    Rcpp::IntegerVector n_obs_per_arm = Rcpp::IntegerVector::create(N);

    std::vector<tulpa::LatentBlock> blocks;
    blocks.push_back(tulpa::make_nngp_block(
        gp_start, n_spatial, spatial_idx_per_arm, n_obs_per_arm,
        /*n_arms=*/1, /*block_index=*/0,
        nn, cov_type, coords, nn_idx, nn_dist, nn_order,
        /*axis_sigma2=*/0, /*axis_phi_gp=*/1, theta_grid));

    std::vector<int> skew_idx_vec;
    const std::vector<int>* skew_idx_ptr =
        tulpa::unwrap_skew_idx(compute_skew, skew_idx, skew_idx_vec);

    // The GP / NNGP Laplace path runs serially: the Vecchia prior scatter that
    // dominates is serial, while running the observation scatter multi-threaded
    // triggers a flaky heap corruption under the mingw OpenMP toolchain.
    n_threads = 1;

    const tulpa::HessianPatternGuard pattern_guard;
    Rcpp::List grid = tulpa::run_multi_block_nested_laplace_joint(
        /*n_grid=*/1, arms, parsed, blocks, n_x,
        max_iter, tol, n_threads,
        /*store_modes=*/true, Rcpp::NumericVector(), /*store_Q=*/false,
        /*prep_at_grid=*/nullptr, /*n_threads_outer=*/1,
        /*tile_ids=*/std::vector<int>(),
        /*tile_pilot_cells=*/std::vector<int>(),
        /*prune_tol=*/0.0, /*force_sparse=*/false,
        /*cell_coupling_spec=*/nullptr,
        tulpa::JointPDMode::LM, tulpa::CurvatureMode::Observed,
        /*hessian_refresh=*/1, /*progress=*/nullptr, /*checkpoint=*/nullptr,
        /*x_init_per_cell=*/std::vector<double>(),
        compute_skew, skew_idx_ptr);
    pattern_guard.check("the GP / NNGP Laplace solve");

    return tulpa::nl_grid_cell_to_result_list(grid, 0);
}
