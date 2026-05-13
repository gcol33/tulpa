// spde_laplace.cpp
// SPDE spatial Laplace: single-fit and nested Laplace over (range, sigma).
// Uses SpdeQBuilder from spde_qbuilder.h for pattern-preserving Q construction.

#include "spde_qbuilder.h"
#include "sparse_hessian.h"
#include "nested_laplace_grid.h"
#include "laplace_re_priors.h"
#include <Rcpp.h>
#include <cmath>
#include <set>
#include <utility>
#include <vector>

// =====================================================================
// Single SPDE Laplace fit
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_spde(
    Rcpp::NumericVector y, Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
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
    Rcpp::Nullable<Rcpp::NumericVector> rational_weights_nullable = R_NilValue
) {
    int N = n_obs;
    int p = X.ncol();
    int n_x = p + n_mesh;
    int mesh_start = p;

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

    // Build sparse A
    tulpa::ARows a_rows = tulpa::build_A_rows(N, n_mesh, A_x, A_i, A_p);

    Rcpp::List out;

    // Use sparse Newton for large meshes
    if (n_x >= tulpa::SPARSE_THRESHOLD) {
        // Build H sparsity pattern from Q pattern + A structure + beta block
        std::vector<std::pair<int,int>> pattern;

        // Beta×beta dense block
        for (int j1 = 0; j1 < p; j1++)
            for (int j2 = j1; j2 < p; j2++)
                pattern.push_back({j2, j1});

        // Q pattern (mesh×mesh block, shifted by mesh_start)
        for (int col = 0; col < n_mesh; col++) {
            for (int idx = qb.Q_p[col]; idx < qb.Q_p[col + 1]; idx++) {
                int row = qb.Q_i[idx];
                pattern.push_back({mesh_start + row, mesh_start + col});
            }
        }

        // A pattern: for each obs, its A row connects beta to mesh nodes
        for (int i = 0; i < N; i++) {
            for (const auto& ae : a_rows[i]) {
                int m_idx = mesh_start + ae.mesh_idx;
                // Beta×mesh cross
                for (int j = 0; j < p; j++) pattern.push_back({m_idx, j});
                // Mesh×mesh from A'diag(h)A (pairs within same obs triangle)
                for (const auto& ae2 : a_rows[i]) {
                    int m_idx2 = mesh_start + ae2.mesh_idx;
                    if (m_idx >= m_idx2) pattern.push_back({m_idx, m_idx2});
                }
            }
        }

        tulpa::SparseHessianBuilder H_builder;
        H_builder.init(n_x, pattern);

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            for (int i = 0; i < N; i++) {
                eta[i] = 0.0;
                for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
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
                const auto& row = a_rows[i];
                for (size_t s1 = 0; s1 < row.size(); s1++) {
                    int idx1 = mesh_start + row[s1].mesh_idx;
                    double a1 = row[s1].weight;
                    grad[idx1] += gh.grad * a1;
                    H.add(idx1, idx1, gh.neg_hess * a1 * a1);
                    for (int j = 0; j < p; j++)
                        H.add(idx1, j, gh.neg_hess * X(i, j) * a1);
                    for (size_t s2 = s1 + 1; s2 < row.size(); s2++) {
                        int idx2 = mesh_start + row[s2].mesh_idx;
                        H.add(std::max(idx1, idx2), std::min(idx1, idx2),
                              gh.neg_hess * a1 * row[s2].weight);
                    }
                }
            }
            // Q prior: gradient uses full Q, Hessian uses lower triangle only
            for (int col = 0; col < n_mesh; col++) {
                for (int qidx = qb.Q_p[col]; qidx < qb.Q_p[col + 1]; qidx++) {
                    int row = qb.Q_i[qidx];
                    double q = qb.Q_x[qidx];
                    grad[mesh_start + row] -= q * x[mesh_start + col];
                    // Only add lower triangle (row >= col) to avoid double-counting
                    if (row >= col) {
                        H.add(mesh_start + row, mesh_start + col, q);
                    }
                }
            }
            double tau_beta = 1e-4;
            for (int j = 0; j < p; j++) { grad[j] -= tau_beta * x[j]; H.add(j, j, tau_beta); }
        };

        auto center = [&](Rcpp::NumericVector& x) {
            tulpa::center_effects(x, mesh_start, n_mesh);
        };

        auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
            double qf = 0.0;
            for (int col = 0; col < n_mesh; col++) {
                for (int qidx = qb.Q_p[col]; qidx < qb.Q_p[col + 1]; qidx++)
                    qf += x[mesh_start + qb.Q_i[qidx]] * qb.Q_x[qidx] * x[mesh_start + col];
            }
            return -0.5 * qf;
        };

        tulpa::LaplaceResult res = tulpa::laplace_newton_solve_sparse(
            y, n_trials, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter_sparse, center, log_prior,
            H_builder, x_init);

        out = Rcpp::List::create(
            Rcpp::Named("mode") = res.mode,
            Rcpp::Named("log_det_Q") = res.log_det_Q,
            Rcpp::Named("log_marginal") = res.log_marginal,
            Rcpp::Named("n_iter") = res.n_iter,
            Rcpp::Named("converged") = res.converged,
            Rcpp::Named("Q_nnz") = qb.nnz(),
            Rcpp::Named("H_nnz") = H_builder.nnz
        );
        return out;
    }

    // Dense path for small meshes
    tulpa::run_spde_laplace(
        y, n_trials, X, N, p, n_mesh, mesh_start, n_x,
        a_rows, qb, family, phi,
        max_iter, tol, n_threads, x_init, nullptr,
        [&](const tulpa::LaplaceResult& res) {
            out = Rcpp::List::create(
                Rcpp::Named("mode") = res.mode,
                Rcpp::Named("log_det_Q") = res.log_det_Q,
                Rcpp::Named("log_marginal") = res.log_marginal,
                Rcpp::Named("n_iter") = res.n_iter,
                Rcpp::Named("converged") = res.converged,
                Rcpp::Named("Q_nnz") = qb.nnz()
            );
        }
    );
    return out;
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
    bool store_Q = false
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

    int re_start = p;
    int mesh_start = p + n_re_groups;
    int n_x = p + n_re_groups + n_mesh;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);

    // Build Q pattern once
    tulpa::SpdeQBuilder qb;
    qb.init(n_mesh, C0_diag, G1_x, G1_i, G1_p);

    // Build sparse A once
    tulpa::ARows a_rows = tulpa::build_A_rows(N, n_mesh, A_x, A_i, A_p);

    // Determine alpha and rational coefficients (constant across the grid).
    int alpha = static_cast<int>(std::round(nu)) + 1;  // alpha = nu + d/2, d=2
    bool use_rational = rational_poles_nullable.isNotNull() &&
                        rational_weights_nullable.isNotNull();
    std::vector<double> rat_poles, rat_weights;
    if (use_rational) {
        rat_poles = Rcpp::as<std::vector<double>>(rational_poles_nullable);
        rat_weights = Rcpp::as<std::vector<double>>(rational_weights_nullable);
    }

    // ------------------------------------------------------------------
    // Build sparse Hessian sparsity pattern once (fixed across the grid).
    // Pattern entries are (row, col) with row >= col after dedup.
    // ------------------------------------------------------------------
    std::set<std::pair<int,int>> pat;
    auto push_pat = [&](int r, int c) {
        int lo = std::min(r, c), hi = std::max(r, c);
        pat.insert({hi, lo});
    };

    // Beta x beta dense block
    for (int j1 = 0; j1 < p; j1++)
        for (int j2 = j1; j2 < p; j2++) push_pat(j2, j1);

    // RE block (only if n_re_groups > 0)
    if (n_re_groups > 0) {
        // RE diagonal
        for (int g = 0; g < n_re_groups; g++) push_pat(re_start + g, re_start + g);
        // RE x beta cross — observed groups only
        for (int i = 0; i < N; i++) {
            int g = static_cast<int>(re_idx[i]) - 1;
            if (g < 0 || g >= n_re_groups) continue;
            for (int j = 0; j < p; j++) push_pat(re_start + g, j);
        }
    }

    // Q pattern (mesh x mesh, lower triangle), shifted by mesh_start
    for (int col = 0; col < n_mesh; col++) {
        for (int idx = qb.Q_p[col]; idx < qb.Q_p[col + 1]; idx++) {
            int row = qb.Q_i[idx];
            if (row >= col) push_pat(mesh_start + row, mesh_start + col);
        }
    }

    // A-row driven cross terms
    for (int i = 0; i < N; i++) {
        int g = (n_re_groups > 0) ? static_cast<int>(re_idx[i]) - 1 : -1;
        bool g_valid = (g >= 0 && g < n_re_groups);
        for (const auto& ae : a_rows[i]) {
            int m_idx = mesh_start + ae.mesh_idx;
            // Beta x mesh cross
            for (int j = 0; j < p; j++) push_pat(m_idx, j);
            // RE x mesh cross
            if (g_valid) push_pat(m_idx, re_start + g);
            // Mesh x mesh pairs (within obs, lower triangle)
            for (const auto& ae2 : a_rows[i]) {
                int m_idx2 = mesh_start + ae2.mesh_idx;
                if (m_idx >= m_idx2) push_pat(m_idx, m_idx2);
            }
        }
    }

    std::vector<std::pair<int,int>> pattern(pat.begin(), pat.end());
    tulpa::SparseHessianBuilder H_builder;
    H_builder.init(n_x, pattern);

    // Shared CHOLMOD across grid points (warm symbolic factorization).
    tulpa::SparseCholeskySolver shared_solver;

    auto solve_at_theta = [&](int k, const Rcpp::NumericVector& prev_mode)
        -> tulpa::LaplaceResult
    {
        double kappa_k = std::sqrt(8.0 * nu) / range_grid[k];
        double tau_k   = 1.0 / (std::sqrt(4.0 * M_PI) * kappa_k * sigma_grid[k]);

        if (use_rational) {
            qb.rebuild_rational(kappa_k, tau_k, rat_poles, rat_weights);
        } else {
            qb.rebuild(kappa_k, tau_k, alpha);
        }

        auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
            for (int i = 0; i < N; i++) {
                double e = 0.0;
                for (int j = 0; j < p; j++) e += X(i, j) * x[j];
                if (n_re_groups > 0) {
                    int g = static_cast<int>(re_idx[i]) - 1;
                    if (g >= 0 && g < n_re_groups) e += x[re_start + g];
                }
                for (const auto& ae : a_rows[i])
                    e += ae.weight * x[mesh_start + ae.mesh_idx];
                eta[i] = e;
            }
        };

        auto scatter_sparse = [&](const Rcpp::NumericVector& x,
                                   const Rcpp::NumericVector& eta,
                                   tulpa::DenseVec& grad,
                                   tulpa::SparseHessianBuilder& H) {
            // Likelihood contribution
            for (int i = 0; i < N; i++) {
                auto gh = tulpa::grad_hess_for_family(y[i], n_trials[i], eta[i],
                                                       family, phi);
                int g = (n_re_groups > 0) ? static_cast<int>(re_idx[i]) - 1 : -1;
                bool g_valid = (g >= 0 && g < n_re_groups);

                // Beta self + lower triangle
                for (int j = 0; j < p; j++) {
                    grad[j] += gh.grad * X(i, j);
                    for (int k2 = 0; k2 <= j; k2++)
                        H.add(j, k2, gh.neg_hess * X(i, j) * X(i, k2));
                }
                // RE self + RE x beta
                if (g_valid) {
                    grad[re_start + g] += gh.grad;
                    H.add(re_start + g, re_start + g, gh.neg_hess);
                    for (int j = 0; j < p; j++)
                        H.add(re_start + g, j, gh.neg_hess * X(i, j));
                }
                // Mesh self + mesh x beta + mesh x re + mesh x mesh (pairs)
                const auto& row = a_rows[i];
                for (size_t s1 = 0; s1 < row.size(); s1++) {
                    int m_idx = mesh_start + row[s1].mesh_idx;
                    double a1 = row[s1].weight;
                    grad[m_idx] += gh.grad * a1;
                    H.add(m_idx, m_idx, gh.neg_hess * a1 * a1);
                    for (int j = 0; j < p; j++)
                        H.add(m_idx, j, gh.neg_hess * X(i, j) * a1);
                    if (g_valid)
                        H.add(m_idx, re_start + g, gh.neg_hess * a1);
                    for (size_t s2 = s1 + 1; s2 < row.size(); s2++) {
                        int m_idx2 = mesh_start + row[s2].mesh_idx;
                        double cross = gh.neg_hess * a1 * row[s2].weight;
                        H.add(std::max(m_idx, m_idx2),
                              std::min(m_idx, m_idx2), cross);
                    }
                }
            }
            // Q prior on mesh block: grad uses full Q, H uses lower triangle.
            for (int col = 0; col < n_mesh; col++) {
                for (int qidx = qb.Q_p[col]; qidx < qb.Q_p[col + 1]; qidx++) {
                    int row = qb.Q_i[qidx];
                    double q = qb.Q_x[qidx];
                    grad[mesh_start + row] -= q * x[mesh_start + col];
                    if (row >= col)
                        H.add(mesh_start + row, mesh_start + col, q);
                }
            }
            // RE prior + small beta ridge (inline because add_re_beta_priors
            // writes to a DenseMat, not a SparseHessianBuilder).
            for (int g = 0; g < n_re_groups; g++) {
                grad[re_start + g] -= tau_re * x[re_start + g];
                H.add(re_start + g, re_start + g, tau_re);
            }
            const double tau_beta = 1e-4;
            for (int j = 0; j < p; j++) {
                grad[j] -= tau_beta * x[j];
                H.add(j, j, tau_beta);
            }
        };

        auto center = [&](Rcpp::NumericVector& x) {
            tulpa::center_effects(x, mesh_start, n_mesh);
        };

        auto log_prior = [&](const Rcpp::NumericVector& x,
                              const Rcpp::NumericVector&) {
            // Mesh quadratic form
            double qf = 0.0;
            for (int col = 0; col < n_mesh; col++) {
                for (int qidx = qb.Q_p[col]; qidx < qb.Q_p[col + 1]; qidx++)
                    qf += x[mesh_start + qb.Q_i[qidx]] * qb.Q_x[qidx]
                          * x[mesh_start + col];
            }
            double lp = -0.5 * qf;
            // RE prior log-density (offset by re_start, hence pass shifted
            // view via the existing helper signature which uses p as offset)
            lp += tulpa::compute_log_prior_re(x, re_start, n_re_groups, tau_re);
            return lp;
        };

        return tulpa::laplace_newton_solve_sparse(
            y, n_trials, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter_sparse, center, log_prior,
            H_builder, prev_mode, &shared_solver, store_Q
        );
    };

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull())
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);

    Rcpp::List out = tulpa::run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, /*store_modes=*/true
    );
    out["range_grid"] = range_grid;
    out["sigma_grid"] = sigma_grid;
    out["nu"] = nu;
    return out;
}
