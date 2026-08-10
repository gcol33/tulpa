// spde_laplace.cpp
// SPDE spatial Laplace: single fits and nested Laplace over (range, sigma).
//
// All three entries delegate to the shared joint-multi sparse impl via a
// single-arm ParsedArm + one SPDE LatentBlock, so they run one pattern
// enumerator, one scatter, one prior scatter and one Newton loop: the nested
// integrator over its grid, the two fixed-hyperparameter fits over a one-row
// grid projected back by nl_grid_cell_to_result_list. The block differs only in
// where its precision comes from -- make_spde_block assembles it from the FEM
// operators per cell, make_spde_block_precomputed takes the rational assembly's
// finished CSC.

#include "hessian_pattern_guard.h"  // HessianPatternGuard
#include "laplace_spec_fit.h"       // unwrap_skew_idx
#include "nested_laplace_grid.h"    // nl_grid_cell_to_result_list, checkpointing
#include "latent_block.h"
#include "nested_laplace_joint_core.h"
#include "nested_laplace_joint_multi.h"
#include "spde_block_factory.h"
#include <Rcpp.h>
#include <string>
#include <vector>

// =====================================================================
// Single SPDE Laplace fit
// =====================================================================

// One-cell run of the shared joint driver over a single arm and one SPDE
// LatentBlock. `q_nnz` receives the block's precision nonzero count, reported
// alongside the projected single-fit result.
static Rcpp::List spde_single_cell_fit(
    std::vector<tulpa::ParsedArm>& parsed,
    std::vector<tulpa::JointArm>& arms,
    std::vector<tulpa::LatentBlock>& blocks,
    int n_x, int max_iter, double tol, int n_threads,
    const Rcpp::NumericVector& x_init,
    bool compute_skew, const std::vector<int>* skew_idx_ptr,
    int q_nnz, const char* what
) {
    const tulpa::HessianPatternGuard pattern_guard;
    Rcpp::List grid = tulpa::run_multi_block_nested_laplace_joint_sparse_impl(
        /*n_grid=*/1, arms, parsed, blocks, n_x,
        max_iter, tol, n_threads,
        /*store_modes=*/true, x_init, /*store_Q=*/false,
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
        /*progress=*/nullptr, /*checkpoint=*/nullptr,
        /*x_init_per_cell=*/std::vector<double>(),
        compute_skew, skew_idx_ptr);
    pattern_guard.check(what);
    Rcpp::List out = tulpa::nl_grid_cell_to_result_list(grid, 0);
    out["Q_nnz"] = q_nnz;
    return out;
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_spde(
    Rcpp::NumericVector y, Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx, int n_re_groups, double sigma_re,
    Rcpp::NumericVector A_x, Rcpp::IntegerVector A_i, Rcpp::IntegerVector A_p,
    int n_obs, int n_mesh,
    Rcpp::NumericVector C0_diag,
    Rcpp::NumericVector G1_x, Rcpp::IntegerVector G1_i, Rcpp::IntegerVector G1_p,
    double kappa, double tau_spde,
    std::string family, double phi = 1.0,
    int alpha = 2,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> rational_poles_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> rational_weights_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    bool compute_skew = false,
    Rcpp::Nullable<Rcpp::IntegerVector> skew_idx = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> weights_nullable = R_NilValue
) {
    int N = n_obs;
    int p = X.ncol();
    // Layout [beta (p), re (n_re_groups), w_mesh (n_mesh)], matching the
    // nested SPDE entry (cpp_nested_laplace_spde).
    int n_x = p + n_re_groups + n_mesh;
    int mesh_start = p + n_re_groups;
    if (n_re_groups > 0 && (int)re_idx.size() != N)
        Rcpp::stop("length(re_idx) (%d) must equal n_obs (%d).",
                   (int)re_idx.size(), N);

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) {
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }

    std::vector<int> skew_idx_vec;
    const std::vector<int>* skew_idx_ptr =
        tulpa::unwrap_skew_idx(compute_skew, skew_idx, skew_idx_vec);

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm> arms;
    // SPDE is INDEXED_MULTI (obs -> several mesh nodes via A), so the arm's
    // spatial_idx is never indexed; a dummy zero vector keeps lifetime safe.
    tulpa::make_single_arm(parsed, arms, X, re_idx, Rcpp::IntegerVector(N, 0),
                           p, n_re_groups, sigma_re, y, n_trials, family, phi,
                           N, offset_nullable, weights_nullable);

    // One-row grid holding the operator parameters directly: this entry is
    // handed (kappa, tau_spde) by its caller, so direct_kappa_tau skips the
    // block's Matern (range, sigma) conversion rather than composing it with
    // its own inverse. nu = alpha - 1 reproduces the requested alpha inside the
    // block (which derives alpha = round(nu) + 1).
    Rcpp::NumericMatrix theta_grid(1, 2);
    theta_grid(0, 0) = kappa;
    theta_grid(0, 1) = tau_spde;
    const double nu_equiv = static_cast<double>(alpha) - 1.0;

    Rcpp::List A_x_per_arm = Rcpp::List::create(A_x);
    Rcpp::List A_i_per_arm = Rcpp::List::create(A_i);
    Rcpp::List A_p_per_arm = Rcpp::List::create(A_p);
    Rcpp::IntegerVector n_obs_per_arm = Rcpp::IntegerVector::create(N);

    std::vector<double> rat_poles, rat_weights;
    const bool use_rational = rational_poles_nullable.isNotNull() &&
                              rational_weights_nullable.isNotNull();
    if (use_rational) {
        rat_poles   = Rcpp::as<std::vector<double>>(rational_poles_nullable);
        rat_weights = Rcpp::as<std::vector<double>>(rational_weights_nullable);
    }

    int q_nnz = 0;
    std::vector<tulpa::LatentBlock> blocks;
    blocks.push_back(tulpa::make_spde_block(
        mesh_start, n_mesh,
        A_x_per_arm, A_i_per_arm, A_p_per_arm, n_obs_per_arm,
        /*n_arms=*/1, /*block_index=*/0,
        C0_diag, G1_x, G1_i, G1_p, nu_equiv,
        /*axis_range=*/0, /*axis_sigma=*/1, theta_grid,
        use_rational, rat_poles, rat_weights,
        /*direct_kappa_tau=*/true, &q_nnz));

    return spde_single_cell_fit(parsed, arms, blocks, n_x, max_iter, tol,
                                n_threads, x_init, compute_skew, skew_idx_ptr,
                                q_nnz, "the SPDE Laplace solve");
}

// =====================================================================
// Fractional SPDE single fit from a PRECOMPUTED rational precision + obs map.
// The rational rSPDE construction makes the latent the auxiliary weights
// x ~ N(0, Q^{-1}) with field u = Pr x, so the obs map is A_eff = A Pr and the
// precision is Q = Pl' Ci Pl. Both are assembled in R (.spde_rational_assemble,
// the validated oracle) and passed here as CSC. Like the integer entry above,
// this is a one-cell run of the joint driver: make_spde_block_precomputed seeds
// its builder from the supplied CSC instead of the FEM operators, and leaves the
// latent uncentred (the proper SPDE prior identifies the constant mode).
// =====================================================================
// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_spde_precomputed(
    Rcpp::NumericVector y, Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx, int n_re_groups, double sigma_re,
    int n_obs, int n_mesh,
    Rcpp::IntegerVector Q_p, Rcpp::IntegerVector Q_i, Rcpp::NumericVector Q_x,
    Rcpp::NumericVector Aeff_x, Rcpp::IntegerVector Aeff_i, Rcpp::IntegerVector Aeff_p,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    bool compute_skew = false,
    Rcpp::Nullable<Rcpp::IntegerVector> skew_idx = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> weights_nullable = R_NilValue
) {
    int N = n_obs;
    int p = X.ncol();
    int n_x = p + n_re_groups + n_mesh;
    int mesh_start = p + n_re_groups;
    if (n_re_groups > 0 && (int)re_idx.size() != N)
        Rcpp::stop("length(re_idx) (%d) must equal n_obs (%d).", (int)re_idx.size(), N);

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull())
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    std::vector<int> skew_idx_vec;
    const std::vector<int>* skew_idx_ptr =
        tulpa::unwrap_skew_idx(compute_skew, skew_idx, skew_idx_vec);

    std::vector<tulpa::ParsedArm> parsed;
    std::vector<tulpa::JointArm> arms;
    // INDEXED_MULTI: obs reach the latent through A_eff, so spatial_idx is
    // never indexed; a dummy zero vector keeps lifetime safe.
    tulpa::make_single_arm(parsed, arms, X, re_idx, Rcpp::IntegerVector(N, 0),
                           p, n_re_groups, sigma_re, y, n_trials, family, phi,
                           N, offset_nullable, weights_nullable);

    Rcpp::List Aeff_x_per_arm = Rcpp::List::create(Aeff_x);
    Rcpp::List Aeff_i_per_arm = Rcpp::List::create(Aeff_i);
    Rcpp::List Aeff_p_per_arm = Rcpp::List::create(Aeff_p);
    Rcpp::IntegerVector n_obs_per_arm = Rcpp::IntegerVector::create(N);

    int q_nnz = 0;
    std::vector<tulpa::LatentBlock> blocks;
    blocks.push_back(tulpa::make_spde_block_precomputed(
        mesh_start, n_mesh,
        Aeff_x_per_arm, Aeff_i_per_arm, Aeff_p_per_arm, n_obs_per_arm,
        /*n_arms=*/1, /*block_index=*/0,
        Q_p, Q_i, Q_x, &q_nnz));

    return spde_single_cell_fit(parsed, arms, blocks, n_x, max_iter, tol,
                                n_threads, x_init, compute_skew, skew_idx_ptr,
                                q_nnz, "the precomputed SPDE Laplace solve");
}

// =====================================================================
// Nested Laplace for SPDE: paired (range, sigma) grid, v10 ABI
// =====================================================================
// Mirrors the NNGP v10 entry (cpp_nested_laplace_nngp) so downstream
// glue can dispatch on the shared output shape:
//   - paired range_grid / sigma_grid (no Cartesian product)
//   - formula-side RE block (length n_re_groups, sigma_re prior)
//   - latent layout [beta (p), re (n_re_groups), w_mesh (n_mesh)]
//   - store_modes = true (matrix n_grid x n_x of inner-Newton modes)
//   - store_Q = true (per-grid mesh+beta+re Q for posterior draws)
// Replaces the v0 entry that only emitted log_marginal and the grid echo.

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_spde(
    Rcpp::NumericVector y, Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx, int n_re_groups, double sigma_re,
    Rcpp::NumericVector A_x, Rcpp::IntegerVector A_i, Rcpp::IntegerVector A_p,
    int n_obs, int n_mesh,
    Rcpp::NumericVector C0_diag,
    Rcpp::NumericVector G1_x, Rcpp::IntegerVector G1_i, Rcpp::IntegerVector G1_p,
    Rcpp::NumericVector range_grid, Rcpp::NumericVector sigma_grid,
    double nu = 1.0,
    std::string family = "gaussian", double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> rational_poles_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> rational_weights_nullable = R_NilValue,
    bool store_Q = false,
    std::string checkpoint_path = "",
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    bool compute_skew = false,
    Rcpp::Nullable<Rcpp::IntegerVector> skew_idx = R_NilValue
) {
    int N = n_obs;
    int p = X.ncol();
    int n_grid = range_grid.size();

    if (sigma_grid.size() != n_grid)
        Rcpp::stop("range_grid and sigma_grid must have the same length");
    if (re_idx.size() != N)
        Rcpp::stop("length(re_idx) must equal n_obs");
    if (C0_diag.size() != n_mesh)
        Rcpp::stop("length(C0_diag) must equal n_mesh");
    if (G1_p.size() != n_mesh + 1)
        Rcpp::stop("length(G1_p) must equal n_mesh + 1");
    if (G1_x.size() != G1_i.size())
        Rcpp::stop("G1_x and G1_i must have the same length");
    if (A_x.size() != A_i.size())
        Rcpp::stop("A_x and A_i must have the same length");
    if (A_p.size() != n_mesh + 1)
        Rcpp::stop("length(A_p) must equal n_mesh + 1");

    // ---- Single-arm joint setup ----
    // Layout: [beta (p), re (n_re_groups), w_mesh (n_mesh)]. parse_joint_arms
    // would build the same offsets for n_arms=1; we set them inline to avoid
    // a round-trip through an Rcpp::List arms wrapper.
    const int n_x = p + n_re_groups + n_mesh;
    const int mesh_start = p + n_re_groups;

    std::vector<tulpa::ParsedArm> parsed(1);
    {
        tulpa::ParsedArm& pa = parsed[0];
        pa.X           = X;
        pa.re_idx      = re_idx;
        // spatial_idx is unused for INDEXED_MULTI blocks (SPDE uses
        // obs_indices via the A matrix), but ParsedArm requires the field;
        // a dummy zero vector keeps lifetime safe.
        pa.spatial_idx = Rcpp::IntegerVector(N, 0);
        pa.p           = p;
        pa.n_re_groups = n_re_groups;
        pa.sigma_re    = sigma_re;
        pa.beta_start  = 0;
        pa.re_start    = p;
        pa.tau_re      = (n_re_groups > 0)
                         ? 1.0 / (sigma_re * sigma_re + 1e-10)
                         : 0.0;
        if (offset_nullable.isNotNull()) {
            Rcpp::NumericVector off(offset_nullable);
            if ((int)off.size() != N)
                Rcpp::stop("length(offset) (%d) must equal n_obs (%d).",
                           (int)off.size(), N);
            pa.offset = off;
        }
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

    // ---- theta_grid: n_grid x 2 (range, sigma). axis_range=0, axis_sigma=1. ----
    Rcpp::NumericMatrix theta_grid(n_grid, 2);
    for (int k = 0; k < n_grid; k++) {
        theta_grid(k, 0) = range_grid[k];
        theta_grid(k, 1) = sigma_grid[k];
    }

    // ---- Rational coefficients (constant across the grid). ----
    bool use_rational = rational_poles_nullable.isNotNull() &&
                        rational_weights_nullable.isNotNull();
    std::vector<double> rat_poles, rat_weights;
    if (use_rational) {
        rat_poles   = Rcpp::as<std::vector<double>>(rational_poles_nullable);
        rat_weights = Rcpp::as<std::vector<double>>(rational_weights_nullable);
    }

    // ---- SPDE LatentBlock via shared factory. ----
    // Per-arm A as length-1 Rcpp::Lists (the factory expects n_arms entries).
    Rcpp::List A_x_per_arm    = Rcpp::List::create(A_x);
    Rcpp::List A_i_per_arm    = Rcpp::List::create(A_i);
    Rcpp::List A_p_per_arm    = Rcpp::List::create(A_p);
    Rcpp::IntegerVector n_obs_per_arm = Rcpp::IntegerVector::create(N);

    std::vector<tulpa::LatentBlock> blocks;
    blocks.push_back(tulpa::make_spde_block(
        /*start=*/mesh_start, n_mesh,
        A_x_per_arm, A_i_per_arm, A_p_per_arm, n_obs_per_arm,
        /*n_arms=*/1, /*block_index=*/0,
        C0_diag, G1_x, G1_i, G1_p, nu,
        /*axis_range=*/0, /*axis_sigma=*/1, theta_grid,
        use_rational, rat_poles, rat_weights
    ));

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull())
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    // Grid-cell checkpoint/resume. Structure fingerprint folds
    // the SPDE FEM operators (A, C0, G1), nu, and any rational coefficients;
    // keys are the paired (range, sigma) grid coordinates.
    tulpa::Fingerprint sfp;
    sfp.fold_str("spde");
    sfp.fold_pod(n_mesh);
    sfp.fold_pod(nu);
    if (A_x.size())     sfp.fold(A_x.begin(),    (std::size_t)A_x.size() * sizeof(double));
    if (A_i.size())     sfp.fold(A_i.begin(),    (std::size_t)A_i.size() * sizeof(int));
    if (A_p.size())     sfp.fold(A_p.begin(),    (std::size_t)A_p.size() * sizeof(int));
    if (C0_diag.size()) sfp.fold(C0_diag.begin(),(std::size_t)C0_diag.size() * sizeof(double));
    if (G1_x.size())    sfp.fold(G1_x.begin(),   (std::size_t)G1_x.size() * sizeof(double));
    if (G1_i.size())    sfp.fold(G1_i.begin(),   (std::size_t)G1_i.size() * sizeof(int));
    if (G1_p.size())    sfp.fold(G1_p.begin(),   (std::size_t)G1_p.size() * sizeof(int));
    sfp.fold_pod(use_rational);
    if (!rat_poles.empty())   sfp.fold(rat_poles.data(),   rat_poles.size() * sizeof(double));
    if (!rat_weights.empty()) sfp.fold(rat_weights.data(), rat_weights.size() * sizeof(double));
    auto ckpt = tulpa::make_nl_grid_checkpoint(
        checkpoint_path, sfp.value(), max_iter, tol, y, n_trials, X, re_idx,
        n_re_groups, sigma_re, family, phi, {range_grid, sigma_grid});

    std::vector<int> skew_idx_vec;
    const std::vector<int>* skew_idx_ptr =
        tulpa::unwrap_skew_idx(compute_skew, skew_idx, skew_idx_vec);

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
        /*progress=*/nullptr, ckpt.get(),
        /*x_init_per_cell=*/std::vector<double>(),
        compute_skew, skew_idx_ptr
    );
    out["range_grid"] = range_grid;
    out["sigma_grid"] = sigma_grid;
    out["nu"] = nu;
    return out;
}
