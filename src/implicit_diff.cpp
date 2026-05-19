// implicit_diff.cpp
// Rcpp exports for implicit differentiation of Laplace log-marginal.
// Enables NUTS sampling over SPDE hyperparameters.

#include "implicit_diff.h"
#include "spde_qbuilder.h"
#include "sparse_hessian.h"
#include <Rcpp.h>

// Compute gradient of Laplace log-marginal w.r.t. (log_range, log_sigma)
// at a given point. Runs inner Laplace to find x*(theta), then implicit diff.

// [[Rcpp::export]]
Rcpp::List cpp_spde_laplace_gradient(
    Rcpp::NumericVector y, Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector A_x, Rcpp::IntegerVector A_i, Rcpp::IntegerVector A_p,
    int n_obs, int n_mesh,
    Rcpp::NumericVector C0_diag,
    Rcpp::NumericVector G1_x, Rcpp::IntegerVector G1_i, Rcpp::IntegerVector G1_p,
    double log_range, double log_sigma,
    double nu = 1.0,
    std::string family = "binomial", double phi = 1.0,
    int max_iter = 50, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue
) {
    int N = n_obs;
    int p = X.ncol();
    int n_x = p + n_mesh;
    int mesh_start = p;

    double range = std::exp(log_range);
    double sigma_spde = std::exp(log_sigma);
    double kappa = std::sqrt(8.0 * nu) / range;
    double tau = 1.0 / (std::sqrt(4.0 * M_PI) * kappa * sigma_spde);

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) {
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }

    // Build Q and A
    tulpa::SpdeQBuilder qb;
    qb.init(n_mesh, C0_diag, G1_x, G1_i, G1_p);
    qb.rebuild(kappa, tau, 2);

    tulpa::ARows a_rows = tulpa::build_A_rows(N, n_mesh, A_x, A_i, A_p);

    // Run inner Laplace to find mode
    tulpa::SparseCholeskySolver solver;
    Rcpp::List dummy;

    // Use the dense run_spde_laplace for the inner solve
    tulpa::LaplaceResult inner_result;
    tulpa::run_spde_laplace(
        y, n_trials, X, N, p, n_mesh, mesh_start, n_x,
        a_rows, qb, family, phi,
        max_iter, tol, n_threads, x_init, &solver,
        [&](const tulpa::LaplaceResult& res) { inner_result = res; }
    );

    // Compute gradient via implicit differentiation. Both sides now use
    // std::vector<double> for the mode (see laplace_core.h on the parallel-
    // safety motivation), so no wrap copy is needed here.
    tulpa::ImplicitDiffResult grad = tulpa::spde_implicit_gradient(
        inner_result.mode, y, n_trials, X, N, p, n_mesh, mesh_start,
        a_rows, qb, range, sigma_spde, nu,
        C0_diag, G1_x, G1_i, G1_p,
        family, phi, solver
    );

    return Rcpp::List::create(
        Rcpp::Named("log_marginal") = inner_result.log_marginal,
        Rcpp::Named("grad_log_range") = grad.grad_log_range,
        Rcpp::Named("grad_log_sigma") = grad.grad_log_sigma,
        Rcpp::Named("mode") = inner_result.mode,
        Rcpp::Named("converged") = inner_result.converged,
        Rcpp::Named("n_iter") = inner_result.n_iter
    );
}
