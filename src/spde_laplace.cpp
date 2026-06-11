// spde_laplace.cpp
// SPDE spatial Laplace: single-fit and nested Laplace over (range, sigma).
// Uses SpdeQBuilder from spde_qbuilder.h for pattern-preserving Q construction.
//
// The nested-Laplace entry (cpp_nested_laplace_spde) delegates to the
// shared joint-multi sparse impl via a single-arm ParsedArm + LatentBlock
// built by make_spde_block. This keeps SPDE on the same code path as the
// joint multi-arm driver (one pattern enumerator, one scatter, one prior
// scatter), removing the previous bespoke pattern/scatter/log_prior path.

#include "spde_qbuilder.h"
#include "spde_logdet.h"        // SpdeQLogDet (0.5 log|Q| prior normalizer)
#include "laplace_spec_fit.h"   // as_offset_vec (offset marshalling)
#include "sparse_hessian.h"
#include "nested_laplace_grid.h"
#include "laplace_re_priors.h"
#include "latent_block.h"
#include "nested_laplace_joint_core.h"
#include "nested_laplace_joint_multi.h"
#include "spde_block_factory.h"
#include <Rcpp.h>
#include <cmath>
#include <functional>
#include <set>
#include <utility>
#include <vector>

// =====================================================================
// Shared SPDE single-fit driver: given a precision builder `qb` and the obs
// projector rows `a_rows`, run the Laplace solve (sparse for large meshes,
// dense otherwise). Both the FEM-built integer/alpha path (cpp_laplace_fit_spde)
// and the precomputed rational path (cpp_laplace_fit_spde_precomputed) build qb
// + a_rows their own way, then call this. `center_mesh` is false for the
// rational path (its latent is the auxiliary weights x, not the field).
// =====================================================================
static Rcpp::List spde_run_single_fit(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, int N, int p,
    const Rcpp::NumericVector& re_idx, int n_re_groups, double sigma_re,
    int n_mesh, int mesh_start, int n_x,
    const tulpa::SpdeQBuilder& qb, const tulpa::ARows& a_rows,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    const Rcpp::NumericVector& x_init, const double* off_ptr,
    bool center_mesh
) {
    const double tau_re = (n_re_groups > 0)
                          ? 1.0 / (sigma_re * sigma_re + 1e-10) : 0.0;
    Rcpp::List out;

    // Prior normalizer 0.5 log|Q(theta)|. A constant in the latent solve (it
    // does not move the mode or the SEs), but required for this fit's
    // log_marginal to be a proper marginal when compared across (range, sigma)
    // -- e.g. the CCD mode-find's final refit and fixed-hyper model comparison.
    // See spde_logdet.h.
    double half_ldQ = 0.0;
    {
        tulpa::SpdeQLogDet qld;
        if (!qld.half_logdet(qb, half_ldQ)) half_ldQ = 0.0;
    }

    if (n_x >= tulpa::SPARSE_THRESHOLD) {
        std::vector<std::pair<int,int>> pattern;
        for (int j1 = 0; j1 < p; j1++)
            for (int j2 = j1; j2 < p; j2++)
                pattern.push_back({j2, j1});
        for (int g = 0; g < n_re_groups; g++)
            pattern.push_back({p + g, p + g});
        for (int col = 0; col < n_mesh; col++)
            for (int idx = qb.Q_p[col]; idx < qb.Q_p[col + 1]; idx++)
                pattern.push_back({mesh_start + qb.Q_i[idx], mesh_start + col});
        for (int i = 0; i < N; i++) {
            int g = (n_re_groups > 0) ? (int)re_idx[i] - 1 : -1;
            if (!(g >= 0 && g < n_re_groups)) g = -1;
            if (g >= 0)
                for (int j = 0; j < p; j++) pattern.push_back({p + g, j});
            for (const auto& ae : a_rows[i]) {
                int m_idx = mesh_start + ae.mesh_idx;
                for (int j = 0; j < p; j++) pattern.push_back({m_idx, j});
                if (g >= 0) pattern.push_back({m_idx, p + g});
                for (const auto& ae2 : a_rows[i]) {
                    int m_idx2 = mesh_start + ae2.mesh_idx;
                    if (m_idx >= m_idx2) pattern.push_back({m_idx, m_idx2});
                }
            }
        }

        tulpa::SparseHessianBuilder H_builder;
        H_builder.init(n_x, pattern);

        auto re_group = [&](int i) -> int {
            if (n_re_groups <= 0) return -1;
            int g = (int)re_idx[i] - 1;
            return (g >= 0 && g < n_re_groups) ? g : -1;
        };
        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            for (int i = 0; i < N; i++) {
                eta[i] = off_ptr ? off_ptr[i] : 0.0;
                for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
                int g = re_group(i);
                if (g >= 0) eta[i] += x[p + g];
                for (const auto& ae : a_rows[i])
                    eta[i] += ae.weight * x[mesh_start + ae.mesh_idx];
            }
        };
        auto scatter_sparse = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                                   tulpa::DenseVec& grad, tulpa::SparseHessianBuilder& H) {
            for (int i = 0; i < N; i++) {
                auto gh = tulpa::grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi);
                for (int j = 0; j < p; j++) {
                    grad[j] += gh.grad * X(i, j);
                    for (int k = 0; k <= j; k++)
                        H.add(j, k, gh.neg_hess * X(i, j) * X(i, k));
                }
                int g = re_group(i);
                if (g >= 0) {
                    int re_i = p + g;
                    grad[re_i] += gh.grad;
                    H.add(re_i, re_i, gh.neg_hess);
                    for (int j = 0; j < p; j++)
                        H.add(re_i, j, gh.neg_hess * X(i, j));
                }
                const auto& row = a_rows[i];
                for (size_t s1 = 0; s1 < row.size(); s1++) {
                    int idx1 = mesh_start + row[s1].mesh_idx;
                    double a1 = row[s1].weight;
                    grad[idx1] += gh.grad * a1;
                    H.add(idx1, idx1, gh.neg_hess * a1 * a1);
                    for (int j = 0; j < p; j++)
                        H.add(idx1, j, gh.neg_hess * X(i, j) * a1);
                    if (g >= 0)
                        H.add(idx1, p + g, gh.neg_hess * a1);
                    for (size_t s2 = s1 + 1; s2 < row.size(); s2++) {
                        int idx2 = mesh_start + row[s2].mesh_idx;
                        H.add(std::max(idx1, idx2), std::min(idx1, idx2),
                              gh.neg_hess * a1 * row[s2].weight);
                    }
                }
            }
            for (int col = 0; col < n_mesh; col++) {
                for (int qidx = qb.Q_p[col]; qidx < qb.Q_p[col + 1]; qidx++) {
                    int row = qb.Q_i[qidx];
                    double q = qb.Q_x[qidx];
                    grad[mesh_start + row] -= q * x[mesh_start + col];
                    if (row >= col) H.add(mesh_start + row, mesh_start + col, q);
                }
            }
            double tau_beta = 1e-4;
            for (int j = 0; j < p; j++) { grad[j] -= tau_beta * x[j]; H.add(j, j, tau_beta); }
            for (int g = 0; g < n_re_groups; g++) {
                grad[p + g] -= tau_re * x[p + g];
                H.add(p + g, p + g, tau_re);
            }
        };
        auto center = [&](Rcpp::NumericVector& x) {
            if (center_mesh) tulpa::center_effects(x, mesh_start, n_mesh);
        };
        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double qf = 0.0;
            for (int col = 0; col < n_mesh; col++)
                for (int qidx = qb.Q_p[col]; qidx < qb.Q_p[col + 1]; qidx++)
                    qf += x[mesh_start + qb.Q_i[qidx]] * qb.Q_x[qidx] * x[mesh_start + col];
            for (int g = 0; g < n_re_groups; g++) qf += tau_re * x[p + g] * x[p + g];
            return half_ldQ - 0.5 * qf;
        };

        tulpa::LaplaceResult res = tulpa::laplace_newton_solve_sparse(
            y, n_trials, family, phi, N, n_x, max_iter, tol, n_threads,
            compute_eta, scatter_sparse, center, log_prior, H_builder, x_init);
        return Rcpp::List::create(
            Rcpp::Named("mode") = res.mode,
            Rcpp::Named("log_det_Q") = res.log_det_Q,
            Rcpp::Named("log_marginal") = res.log_marginal,
            Rcpp::Named("n_iter") = res.n_iter,
            Rcpp::Named("converged") = res.converged,
            Rcpp::Named("Q_nnz") = qb.nnz(),
            Rcpp::Named("H_nnz") = H_builder.nnz);
    }

    tulpa::run_spde_laplace(
        y, n_trials, X, N, p, n_mesh, mesh_start, n_x,
        a_rows, qb, family, phi, max_iter, tol, n_threads, x_init, nullptr, off_ptr,
        [&](const tulpa::LaplaceResult& res) {
            out = Rcpp::List::create(
                Rcpp::Named("mode") = res.mode,
                Rcpp::Named("log_det_Q") = res.log_det_Q,
                Rcpp::Named("log_marginal") = res.log_marginal,
                Rcpp::Named("n_iter") = res.n_iter,
                Rcpp::Named("converged") = res.converged,
                Rcpp::Named("Q_nnz") = qb.nnz());
        },
        re_idx, n_re_groups, sigma_re, center_mesh, half_ldQ);
    return out;
}

// =====================================================================
// Single SPDE Laplace fit
// =====================================================================

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
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue
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
    const double tau_re = (n_re_groups > 0)
                          ? 1.0 / (sigma_re * sigma_re + 1e-10) : 0.0;

    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);
    const double* off_ptr = offset.empty() ? nullptr : offset.data();

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) {
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }

    // Build Q
    tulpa::SpdeQBuilder qb;
    qb.init(n_mesh, C0_diag, G1_x, G1_i, G1_p);

    if (rational_poles_nullable.isNotNull() && rational_weights_nullable.isNotNull()) {
        // Fractional alpha: rational approximation
        std::vector<double> poles = Rcpp::as<std::vector<double>>(rational_poles_nullable);
        std::vector<double> weights = Rcpp::as<std::vector<double>>(rational_weights_nullable);
        qb.rebuild_rational(kappa, tau_spde, poles, weights);
    } else {
        qb.rebuild(kappa, tau_spde, alpha);
    }

    // Build sparse A and run the shared single-fit driver (integer field is
    // sum-to-zero centred).
    tulpa::ARows a_rows = tulpa::build_A_rows(N, n_mesh, A_x, A_i, A_p);
    return spde_run_single_fit(
        y, n_trials, X, N, p, re_idx, n_re_groups, sigma_re,
        n_mesh, mesh_start, n_x, qb, a_rows, family, phi,
        max_iter, tol, n_threads, x_init, off_ptr, /*center_mesh=*/true);
}

// =====================================================================
// Fractional SPDE single fit from a PRECOMPUTED rational precision + obs map
// (gcol33/tulpa#71). The rational rSPDE construction makes the latent the
// auxiliary weights x ~ N(0, Q^{-1}) with field u = Pr x, so the obs map is
// A_eff = A Pr and the precision is Q = Pl' Ci Pl. Both are assembled in R
// (.spde_rational_assemble, the validated oracle) and passed here as CSC; this
// entry wraps them in an SpdeQBuilder + ARows and reuses the shared driver. The
// latent x is NOT centred (the proper SPDE prior identifies the constant mode).
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
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue
) {
    int N = n_obs;
    int p = X.ncol();
    int n_x = p + n_re_groups + n_mesh;
    int mesh_start = p + n_re_groups;
    if (n_re_groups > 0 && (int)re_idx.size() != N)
        Rcpp::stop("length(re_idx) (%d) must equal n_obs (%d).", (int)re_idx.size(), N);

    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);
    const double* off_ptr = offset.empty() ? nullptr : offset.data();
    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull())
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    // Wrap the precomputed lower-triangular CSC precision in an SpdeQBuilder.
    tulpa::SpdeQBuilder qb;
    qb.n_mesh = n_mesh;
    qb.Q_p.assign(Q_p.begin(), Q_p.end());
    qb.Q_i.assign(Q_i.begin(), Q_i.end());
    qb.Q_x.assign(Q_x.begin(), Q_x.end());

    tulpa::ARows a_rows = tulpa::build_A_rows(N, n_mesh, Aeff_x, Aeff_i, Aeff_p);
    return spde_run_single_fit(
        y, n_trials, X, N, p, re_idx, n_re_groups, sigma_re,
        n_mesh, mesh_start, n_x, qb, a_rows, family, phi,
        max_iter, tol, n_threads, x_init, off_ptr, /*center_mesh=*/false);
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
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue
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

    // Grid-cell checkpoint/resume (gcol33/tulpa#50). Structure fingerprint folds
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
        checkpoint_path, sfp.value(), max_iter, tol, y, n_trials, X,
        n_re_groups, sigma_re, family, phi, {range_grid, sigma_grid});

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
    out["range_grid"] = range_grid;
    out["sigma_grid"] = sigma_grid;
    out["nu"] = nu;
    return out;
}
