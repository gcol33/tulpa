// laplace_newton_joint.h
// PIRLS-equivalent Newton solver for *joint multi-likelihood* Laplace modes.
//
// Mirrors laplace_newton.h::laplace_newton_solve, but the data side is a
// vector of arms rather than a single (y, family, phi) bundle. Each arm
// carries its own (y, n_trials, family, phi, N); the prior side and the
// latent vector x are shared across arms.
//
// Callbacks operate on a vector of per-arm eta vectors:
//   - compute_eta_joint(x, etas): caller fills etas[k] for each arm.
//   - scatter_joint(x, etas, grad, H): caller scatters per-arm contributions
//                                       and shared prior into the joint (g, H).
//   - compute_log_prior_joint(x, etas): joint log p(x | theta).
//   - center_effects_fn(x): post-step centering of structured blocks.
//
// The Newton loop, Cholesky dispatch, step halving, log_det and final
// log-marginal machinery are identical to the single-arm path — they
// operate on the joint (g, H, x) without caring how many arms contributed.

#ifndef TULPA_LAPLACE_NEWTON_JOINT_H
#define TULPA_LAPLACE_NEWTON_JOINT_H

#include "laplace_cholesky.h"
#include "laplace_cholesky_dispatch.h"
#include "laplace_family_link.h"
#include "laplace_newton.h"          // SPARSE_THRESHOLD
#include "laplace_newton_loop.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// One arm of a joint Laplace fit. POD-ish wrapper over Rcpp containers so
// the joint solver can iterate per arm without templating on the arm count.
struct JointArm {
    Rcpp::NumericVector y;         // [N]
    Rcpp::IntegerVector n_trials;  // [N]
    std::string         family;
    double              phi;
    int                 N;
};

// Sum of per-arm log-likelihoods at the current per-arm eta vectors.
inline double compute_total_log_lik_joint(
    const std::vector<JointArm>& arms,
    const std::vector<Rcpp::NumericVector>& etas,
    int n_threads
) {
    double total = 0.0;
    for (size_t k = 0; k < arms.size(); k++) {
        total += compute_total_log_lik(
            arms[k].y, arms[k].n_trials, etas[k], arms[k].N,
            arms[k].family, arms[k].phi, n_threads
        );
    }
    return total;
}

// Penalised log-lik (joint) used by the line search.
template<typename ComputeEtaJoint, typename ComputeLogPriorJoint>
inline double eval_penalized_log_lik_joint(
    const Rcpp::NumericVector& x,
    const std::vector<JointArm>& arms,
    int n_threads,
    ComputeEtaJoint compute_eta_joint,
    ComputeLogPriorJoint compute_log_prior_joint
) {
    std::vector<Rcpp::NumericVector> etas;
    etas.reserve(arms.size());
    for (size_t k = 0; k < arms.size(); k++) {
        etas.emplace_back(arms[k].N, 0.0);
    }
    compute_eta_joint(x, etas);
    double ll = compute_total_log_lik_joint(arms, etas, n_threads);
    double lp = compute_log_prior_joint(x, etas);
    return ll + lp;
}

// Joint Newton solver.
template<typename ComputeEtaJoint, typename ScatterJoint,
         typename CenterEffects, typename ComputeLogPriorJoint>
LaplaceResult laplace_newton_solve_joint(
    const std::vector<JointArm>& arms,
    int n_x,
    int max_iter, double tol, int n_threads,
    ComputeEtaJoint compute_eta_joint,
    ScatterJoint scatter_joint,
    CenterEffects center_effects_fn,
    ComputeLogPriorJoint compute_log_prior_joint,
    const Rcpp::NumericVector& x_init = Rcpp::NumericVector(),
    SparseCholeskySolver* shared_solver = nullptr,
    bool store_Q = false
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
        return eval_penalized_log_lik_joint(
            xv, arms, n_threads, compute_eta_joint, compute_log_prior_joint
        );
    };

    auto cholesky_solve = [&](DenseMat& H, DenseVec& grad,
                              std::vector<double>& delta) -> bool {
        return dispatch_factor_solve(H, grad, delta, n_x, sparse_solver, use_sparse);
    };

    // Reusable per-arm eta buffers — same lifetime as the Newton loop, so we
    // skip a vector<NumericVector> reallocation per iteration.
    std::vector<Rcpp::NumericVector> etas;
    etas.reserve(arms.size());
    for (size_t k = 0; k < arms.size(); k++) {
        etas.emplace_back(arms[k].N, 0.0);
    }

    double obj_current = -1e300;
    bool obj_valid = false;

    for (int iter = 0; iter < max_iter; iter++) {
        compute_eta_joint(x, etas);

        DenseVec grad(n_x, 0.0);
        DenseMat H(n_x, DenseVec(n_x, 0.0));
        scatter_joint(x, etas, grad, H);

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

    compute_eta_joint(x, etas);

    DenseVec grad_final(n_x, 0.0);
    DenseMat H_final(n_x, DenseVec(n_x, 0.0));
    scatter_joint(x, etas, grad_final, H_final);

    dispatch_factor_log_det(H_final, n_x, sparse_solver, use_sparse, result.log_det_Q);

    double log_lik   = compute_total_log_lik_joint(arms, etas, n_threads);
    double log_prior = compute_log_prior_joint(x, etas);

    result.log_marginal = finalize_log_marginal(log_lik, log_prior, result.log_det_Q, n_x);

    if (store_Q) {
        dense_to_csc_lower_drop_raw(
            H_final, n_x, SPARSE_DROP_TOL_DISPATCH,
            result.Q_csc_p, result.Q_csc_i, result.Q_csc_x
        );
        result.Q_csc_n = n_x;
    }

    return result;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_JOINT_H
