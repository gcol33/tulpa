// implicit_diff.cpp
// Rcpp exports for implicit differentiation of Laplace log-marginal.
// Enables NUTS sampling over SPDE hyperparameters.

#include "implicit_diff.h"
#include "spde_qbuilder.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <tuple>

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

    // The analytic dQ/dtheta kernel (implicit_diff.h) differentiates the
    // alpha = 2 assembly term by term, so this entry is the nu = 1 operator.
    if (std::abs(nu - 1.0) > 1e-10) {
        Rcpp::stop("cpp_spde_laplace_gradient implements the alpha = 2 "
                   "(nu = 1) operator; got nu = %g. Its analytic dQ/dtheta is "
                   "written for that assembly.", nu);
    }
    double range = std::exp(log_range);
    double sigma_spde = std::exp(log_sigma);
    double kappa, tau;
    std::tie(kappa, tau) =
        tulpa::spde_range_sigma_to_kappa_tau(range, sigma_spde, nu);

    Rcpp::NumericVector x_init;
    if (x_init_nullable.isNotNull()) {
        x_init = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }

    // Build Q and A
    tulpa::SpdeQBuilder qb;
    qb.init(n_mesh, C0_diag, G1_x, G1_i, G1_p, 2);
    qb.rebuild(kappa, tau);

    tulpa::ARows a_rows = tulpa::build_A_rows(N, n_mesh, A_x, A_i, A_p);

    // Run inner Laplace to find mode
    tulpa::SparseCholeskySolver solver;
    Rcpp::List dummy;

    // Use the dense run_spde_laplace for the inner solve
    tulpa::LaplaceResult inner_result;
    tulpa::run_spde_laplace(
        y, n_trials, X, N, p, n_mesh, mesh_start, n_x,
        a_rows, qb, family, phi,
        max_iter, tol, n_threads, x_init, &solver, /*offset=*/nullptr,
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

    Rcpp::List out = tulpa::laplace_result_to_list(inner_result);
    out["grad_log_range"] = grad.grad_log_range;
    out["grad_log_sigma"] = grad.grad_log_sigma;
    return out;
}
