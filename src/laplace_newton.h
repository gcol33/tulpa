// laplace_newton.h
// PIRLS-equivalent Newton/Fisher scoring solver for Laplace modes.

#ifndef TULPA_LAPLACE_NEWTON_H
#define TULPA_LAPLACE_NEWTON_H

#include "laplace_cholesky.h"
#include "laplace_family_link.h"
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
constexpr double SPARSE_DROP_TOL = 1e-12;
constexpr int MAX_HALVING = 12;

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
        Rcpp::NumericVector eta_tmp(N, 0.0);
        compute_eta(xv, eta_tmp);
        double ll = compute_total_log_lik(y, n_trials, eta_tmp, N, family, phi, n_threads);
        double lp = compute_log_prior(xv, eta_tmp);
        return ll + lp;
    };

    auto cholesky_solve = [&](DenseMat& H, DenseVec& grad,
                              std::vector<double>& delta) -> bool {
        bool ok = false;
        if (use_sparse) {
            cholmod_sparse* A = dense_to_cholmod_sparse_drop(
                H, n_x, SPARSE_DROP_TOL, &sparse_solver.common());
            if (A) {
                if (!sparse_solver.analyzed()) sparse_solver.analyze(A);
                if (sparse_solver.factorize(A)) {
                    sparse_solver.solve(grad.data(), delta.data(), n_x);
                    ok = true;
                    for (int j = 0; j < n_x; j++) {
                        if (!std::isfinite(delta[j])) { ok = false; break; }
                    }
                }
                M_cholmod_free_sparse(&A, &sparse_solver.common());
            }
            if (!ok) {
                auto chol = dense_cholesky_solve(H, grad, n_x);
                ok = chol.success;
                for (int j = 0; j < n_x; j++) delta[j] = chol.delta[j];
            }
        } else {
            auto chol = dense_cholesky_solve(H, grad, n_x);
            ok = chol.success;
            for (int j = 0; j < n_x; j++) delta[j] = chol.delta[j];
        }
        return ok;
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

        double step_scale = 1.0;
        for (int half = 0; half <= MAX_HALVING; half++) {
            Rcpp::NumericVector x_try(n_x);
            for (int j = 0; j < n_x; j++) x_try[j] = x[j] + step_scale * delta[j];

            double obj_try = eval_objective(x_try);
            if (obj_try >= obj_current - 1e-8 || half == MAX_HALVING) {
                for (int j = 0; j < n_x; j++) x[j] = x_try[j];
                obj_current = obj_try;
                break;
            }
            step_scale *= 0.5;
        }

        double max_delta = 0.0;
        for (int j = 0; j < n_x; j++) {
            max_delta = std::max(max_delta, std::abs(step_scale * delta[j]));
        }

        result.n_iter = iter + 1;
        if (max_delta < tol) {
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

    if (use_sparse) {
        cholmod_sparse* A_final = dense_to_cholmod_sparse_drop(
            H_final, n_x, SPARSE_DROP_TOL, &sparse_solver.common());
        if (A_final) {
            if (!sparse_solver.analyzed()) sparse_solver.analyze(A_final);
            if (sparse_solver.factorize(A_final)) {
                result.log_det_Q = sparse_solver.log_determinant();
            } else {
                Rcpp::NumericMatrix L_final(n_x, n_x);
                dense_cholesky_factorize(H_final, n_x, L_final, result.log_det_Q);
            }
            M_cholmod_free_sparse(&A_final, &sparse_solver.common());
        } else {
            Rcpp::NumericMatrix L_final(n_x, n_x);
            dense_cholesky_factorize(H_final, n_x, L_final, result.log_det_Q);
        }
    } else {
        Rcpp::NumericMatrix L_final(n_x, n_x);
        dense_cholesky_factorize(H_final, n_x, L_final, result.log_det_Q);
    }

    double log_lik = compute_total_log_lik(y, n_trials, eta_final, N, family, phi, n_threads);
    double log_prior = compute_log_prior(x, eta_final);

    result.log_marginal = log_lik + log_prior
                          - 0.5 * result.log_det_Q
                          + 0.5 * n_x * std::log(2.0 * M_PI);

    return result;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_H
