// spde_laplace.cpp
// SPDE spatial Laplace: single-fit and nested Laplace over (range, sigma).
// Uses SpdeQBuilder from spde_qbuilder.h for pattern-preserving Q construction.

#include "spde_qbuilder.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>

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
// Nested Laplace for SPDE: 2D grid over (range, sigma)
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_spde(
    Rcpp::NumericVector y, Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector A_x, Rcpp::IntegerVector A_i, Rcpp::IntegerVector A_p,
    int n_obs, int n_mesh,
    Rcpp::NumericVector C0_diag,
    Rcpp::NumericVector G1_x, Rcpp::IntegerVector G1_i, Rcpp::IntegerVector G1_p,
    Rcpp::NumericVector range_grid,
    Rcpp::NumericVector sigma_grid,
    double nu = 1.0,
    std::string family = "binomial", double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> rational_poles_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> rational_weights_nullable = R_NilValue
) {
    int N = n_obs;
    int p = X.ncol();
    int n_x = p + n_mesh;
    int mesh_start = p;
    int n_grid = range_grid.size();

    Rcpp::NumericVector log_marginals(n_grid);
    Rcpp::IntegerVector n_iters(n_grid);

    // Build Q pattern once
    tulpa::SpdeQBuilder qb;
    qb.init(n_mesh, C0_diag, G1_x, G1_i, G1_p);

    // Build sparse A once
    tulpa::ARows a_rows = tulpa::build_A_rows(N, n_mesh, A_x, A_i, A_p);

    // Shared CHOLMOD
    tulpa::SparseCholeskySolver shared_solver;

    // Warm-start chain
    Rcpp::NumericVector prev_mode;
    if (x_init_nullable.isNotNull()) {
        prev_mode = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }

    // Determine alpha and rational coefficients
    int alpha = static_cast<int>(std::round(nu)) + 1;  // alpha = nu + d/2, d=2
    bool use_rational = rational_poles_nullable.isNotNull() && rational_weights_nullable.isNotNull();
    std::vector<double> poles, weights;
    if (use_rational) {
        poles = Rcpp::as<std::vector<double>>(rational_poles_nullable);
        weights = Rcpp::as<std::vector<double>>(rational_weights_nullable);
    }

    for (int k = 0; k < n_grid; k++) {
        double kappa_k = std::sqrt(8.0 * nu) / range_grid[k];
        double tau_k = 1.0 / (std::sqrt(4.0 * M_PI) * kappa_k * sigma_grid[k]);

        if (use_rational) {
            qb.rebuild_rational(kappa_k, tau_k, poles, weights);
        } else {
            qb.rebuild(kappa_k, tau_k, alpha);
        }

        tulpa::run_spde_laplace(
            y, n_trials, X, N, p, n_mesh, mesh_start, n_x,
            a_rows, qb, family, phi,
            max_iter, tol, n_threads,
            prev_mode, &shared_solver,
            [&](const tulpa::LaplaceResult& res) {
                log_marginals[k] = res.log_marginal;
                n_iters[k] = res.n_iter;
                prev_mode = res.mode;
            }
        );
    }

    return Rcpp::List::create(
        Rcpp::Named("log_marginal") = log_marginals,
        Rcpp::Named("n_iter") = n_iters,
        Rcpp::Named("n_grid") = n_grid,
        Rcpp::Named("range_grid") = range_grid,
        Rcpp::Named("sigma_grid") = sigma_grid,
        Rcpp::Named("Q_nnz") = qb.nnz()
    );
}
