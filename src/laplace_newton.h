// laplace_newton.h
// PIRLS-equivalent Newton/Fisher scoring solver for Laplace modes.

#ifndef TULPA_LAPLACE_NEWTON_H
#define TULPA_LAPLACE_NEWTON_H

#include "laplace_cholesky.h"
#include "laplace_cholesky_dispatch.h"  // dispatch_factor_solve, dispatch_factor_log_det
#include "laplace_family_link.h"
#include "laplace_newton_loop.h"        // eval_*, step_halving_update, finalize_log_marginal
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

constexpr int SPARSE_THRESHOLD = 200;
// SPARSE_DROP_TOL lives in laplace_cholesky_dispatch.h as SPARSE_DROP_TOL_DISPATCH.
// MAX_HALVING is defined in laplace_newton_loop.h.

template<typename ComputeEta, typename ScatterGradHess,
         typename CenterEffects, typename ComputeLogPrior>
LaplaceResult laplace_newton_solve(
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const std::string& family,
    double phi,
    int N, int n_x,
    int max_iter, double tol, int n_threads,
    ComputeEta compute_eta,
    ScatterGradHess scatter_grad_hess,
    CenterEffects center_effects_fn,
    ComputeLogPrior compute_log_prior,
    const Rcpp::NumericVector& x_init = Rcpp::NumericVector(),
    SparseCholeskySolver* shared_solver = nullptr
) {
    LaplaceResult result;
    result.mode = Rcpp::NumericVector(n_x, 0.0);
    result.converged = false;
    result.n_iter = 0;
    result.log_det_Q = 0.0;
    result.log_marginal = 0.0;

    Rcpp::NumericVector x(n_x, 0.0);
    if (x_init.size() == n_x) {
        for (int j = 0; j < n_x; j++) x[j] = x_init[j];
    }
    bool use_sparse = (n_x >= SPARSE_THRESHOLD);

    SparseCholeskySolver local_solver;
    SparseCholeskySolver& sparse_solver = shared_solver ? *shared_solver : local_solver;

    #ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    #endif

    auto eval_objective = [&](const Rcpp::NumericVector& xv) -> double {
        return eval_penalized_log_lik(xv, y, n_trials, N, family, phi, n_threads,
                                       compute_eta, compute_log_prior);
    };

    auto cholesky_solve = [&](DenseMat& H, DenseVec& grad,
                              std::vector<double>& delta) -> bool {
        return dispatch_factor_solve(H, grad, delta, n_x, sparse_solver, use_sparse);
    };

    double obj_current = -1e300;
    bool obj_valid = false;

    for (int iter = 0; iter < max_iter; iter++) {
        Rcpp::NumericVector eta(N, 0.0);
        compute_eta(x, eta);

        DenseVec grad(n_x, 0.0);
        DenseMat H(n_x, DenseVec(n_x, 0.0));
        scatter_grad_hess(x, eta, grad, H);

        std::vector<double> delta(n_x, 0.0);
        bool solve_ok = cholesky_solve(H, grad, delta);

        if (!solve_ok) {
            for (int j = 0; j < n_x; j++) {
                if (std::isfinite(delta[j])) x[j] += 0.1 * delta[j];
            }
            obj_valid = false;
            result.n_iter = iter + 1;
            continue;
        }

        if (!obj_valid) {
            obj_current = eval_objective(x);
            obj_valid = true;
        }

        double step_scale = step_halving_update(x, delta, n_x, obj_current,
                                                  eval_objective, obj_current);

        result.n_iter = iter + 1;
        if (max_abs_step(delta, step_scale, n_x) < tol) {
            result.converged = true;
            break;
        }
    }

    center_effects_fn(x);
    result.mode = x;

    Rcpp::NumericVector eta_final(N, 0.0);
    compute_eta(x, eta_final);

    DenseVec grad_final(n_x, 0.0);
    DenseMat H_final(n_x, DenseVec(n_x, 0.0));
    scatter_grad_hess(x, eta_final, grad_final, H_final);

    dispatch_factor_log_det(H_final, n_x, sparse_solver, use_sparse, result.log_det_Q);

    double log_lik = compute_total_log_lik(y, n_trials, eta_final, N, family, phi, n_threads);
    double log_prior = compute_log_prior(x, eta_final);

    result.log_marginal = finalize_log_marginal(log_lik, log_prior, result.log_det_Q, n_x);

    return result;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_H
