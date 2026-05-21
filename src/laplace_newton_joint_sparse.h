// laplace_newton_joint_sparse.h
// Sparse-Hessian PIRLS-equivalent Newton solver for *joint multi-likelihood*
// Laplace modes.
//
// Sibling of laplace_newton_joint.h::laplace_newton_solve_joint: same arm
// vector, same callback shapes for compute_eta_joint / center / log_prior,
// but the scatter writes into a SparseHessianBuilder rather than a DenseMat
// and the Newton step factorizes the sparse H directly via CHOLMOD.
//
// Why a separate header (not a unified solver with branching):
//   * NewtonScratchJointSparse never allocates n_x x n_x DenseMat H.
//     At n_x ~ 10^6 the dense allocation alone is 8 TB; the entire point
//     of the sparse path is to never touch it.
//   * The factorize/solve dispatch is sparse-only — no dense-to-CSC
//     conversion path, no DenseCholeskyScratch. CHOLMOD handles everything.
//   * The sparsity pattern is built ONCE at fit-time (outside this solver)
//     via build_joint_hessian_pattern; only values change per iteration.
//
// Eta accumulator: lives in the caller (run_multi_block_nested_laplace_joint).
// Same shape as the dense compute_eta_joint but must dispatch on
// LatentBlock::contrib_kind to handle INDEXED_SINGLE / INDEXED_MULTI /
// DENSE_BASIS. The caller supplies the dispatch lambda.
//
// Scatter: caller supplies a lambda that calls
// scatter_arm_obs_joint_multi_sparse (from nested_laplace_joint_multi.h) for
// each arm, plus add_prior_sparse on each block, plus per-arm beta/RE priors.

#ifndef TULPA_LAPLACE_NEWTON_JOINT_SPARSE_H
#define TULPA_LAPLACE_NEWTON_JOINT_SPARSE_H

#include "laplace_family_link.h"
#include "laplace_newton_joint.h"     // JointArm, compute_total_log_lik_joint
#include "laplace_newton_loop.h"
#include "laplace_profile.h"
#include "sparse_cholesky.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// Per-thread scratch for the sparse joint Newton solver.
// IMPORTANT: no DenseMat H — that's the whole point. The Hessian lives
// inside the caller-supplied SparseHessianBuilder.
struct NewtonScratchJointSparse {
    Rcpp::NumericVector x;       // size n_x
    Rcpp::NumericVector x_try;   // size n_x
    std::vector<Rcpp::NumericVector> etas;      // per-arm, length N_k
    std::vector<Rcpp::NumericVector> etas_tmp;  // line-search buffer
    DenseVec  grad;              // size n_x, zeroed per iter
    std::vector<double> delta;   // size n_x, written by sparse solve

    void allocate(int n_x, const std::vector<JointArm>& arms) {
        x      = Rcpp::NumericVector(n_x, 0.0);
        x_try  = Rcpp::NumericVector(n_x, 0.0);
        etas.clear();     etas.reserve(arms.size());
        etas_tmp.clear(); etas_tmp.reserve(arms.size());
        for (const JointArm& a : arms) {
            etas.emplace_back(a.N, 0.0);
            etas_tmp.emplace_back(a.N, 0.0);
        }
        grad.assign(n_x, 0.0);
        delta.assign(n_x, 0.0);
    }

    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0);
        std::fill(delta.begin(), delta.end(), 0.0);
    }
};

// Sparse-H joint Newton solver. Compositional skeleton mirrors
// laplace_newton_solve_joint exactly; differences are isolated to the H
// container (SparseHessianBuilder vs DenseMat) and the factor/solve path
// (CHOLMOD only, no dense fallback).
//
// scatter_joint_sparse signature:
//   (const NumericVector& x, const vector<NumericVector>& etas,
//    DenseVec& grad, SparseHessianBuilder& H) -> void
//
// The H_builder must already be init()ed with the joint pattern before
// this function is called.
template<typename ComputeEtaJoint, typename ScatterJointSparse,
         typename CenterEffects, typename ComputeLogPriorJoint>
LaplaceResult laplace_newton_solve_joint_sparse(
    const std::vector<JointArm>& arms,
    int n_x,
    int max_iter, double tol, int n_threads,
    ComputeEtaJoint compute_eta_joint,
    ScatterJointSparse scatter_joint_sparse,
    CenterEffects center_effects_fn,
    ComputeLogPriorJoint compute_log_prior_joint,
    SparseHessianBuilder& H_builder,
    NewtonScratchJointSparse& scratch,
    const std::vector<double>& x_init,
    SparseCholeskySolver* shared_solver,
    bool store_Q
) {
    LaplaceResult result;
    result.mode.assign(n_x, 0.0);
    result.converged = false;
    result.n_iter = 0;
    result.log_det_Q = 0.0;
    result.log_marginal = 0.0;

    Rcpp::NumericVector& x = scratch.x;
    if (static_cast<int>(x_init.size()) == n_x) {
        for (int j = 0; j < n_x; j++) x[j] = x_init[j];
    } else {
        for (int j = 0; j < n_x; j++) x[j] = 0.0;
    }

    SparseCholeskySolver local_solver;
    SparseCholeskySolver& solver = shared_solver ? *shared_solver : local_solver;

    auto eval_objective = [&](const Rcpp::NumericVector& xv) -> double {
        return eval_penalized_log_lik_joint(
            xv, arms, n_threads, compute_eta_joint, compute_log_prior_joint,
            scratch.etas_tmp
        );
    };

    double obj_current = -1e300;
    bool obj_valid = false;

    for (int iter = 0; iter < max_iter; iter++) {
        { TULPA_PROFILE_PHASE(PHASE_ETA);
          compute_eta_joint(x, scratch.etas); }

        scratch.zero_grad();
        H_builder.zero();
        { TULPA_PROFILE_PHASE(PHASE_SCATTER);
          scatter_joint_sparse(x, scratch.etas, scratch.grad, H_builder); }

        // Uniform upstream ridge so the dense pivot-clamp and sparse-dbound
        // hacks aren't needed (see LAPLACE_UNIFORM_RIDGE in laplace_cholesky.h).
        H_builder.add_uniform_ridge(LAPLACE_UNIFORM_RIDGE);

        // Factorize and solve via CHOLMOD.
        cholmod_sparse H_cholmod = H_builder.as_cholmod(&solver.common());
        if (!solver.analyzed()) {
            TULPA_PROFILE_PHASE(PHASE_ANALYZE);
            solver.analyze(&H_cholmod);
        }

        bool factorize_ok = false;
        { TULPA_PROFILE_PHASE(PHASE_FACTORIZE);
          factorize_ok = solver.factorize(&H_cholmod); }
        bool solve_ok = false;
        if (factorize_ok) {
            { TULPA_PROFILE_PHASE(PHASE_SOLVE);
              solver.solve(scratch.grad.data(), scratch.delta.data(), n_x); }
            solve_ok = true;
            for (int j = 0; j < n_x; j++) {
                if (!std::isfinite(scratch.delta[j])) { solve_ok = false; break; }
            }
        }

        if (!solve_ok) {
            for (int j = 0; j < n_x; j++) {
                if (std::isfinite(scratch.delta[j])) {
                    x[j] += 0.1 * scratch.delta[j];
                }
            }
            obj_valid = false;
            result.n_iter = iter + 1;
            continue;
        }

        if (!obj_valid) {
            obj_current = eval_objective(x);
            obj_valid = true;
        }

        double step_scale;
        { TULPA_PROFILE_PHASE(PHASE_LINE_SEARCH);
          step_scale = step_halving_update(
              x, scratch.delta, n_x, obj_current, eval_objective, obj_current,
              scratch.x_try
          ); }

        result.n_iter = iter + 1;
        if (max_abs_step(scratch.delta, step_scale, n_x) < tol) {
            result.converged = true;
            break;
        }
    }

    // Final pass at the Newton mode (uncentered): rebuild grad + H for
    // log_det evaluation, then center post-hoc. See laplace_newton_joint.h
    // line 207 for the BYM2/ICAR rank-deficiency rationale.
    { TULPA_PROFILE_PHASE(PHASE_ETA);
      compute_eta_joint(x, scratch.etas); }
    scratch.zero_grad();
    H_builder.zero();
    { TULPA_PROFILE_PHASE(PHASE_SCATTER);
      scatter_joint_sparse(x, scratch.etas, scratch.grad, H_builder); }
    H_builder.add_uniform_ridge(LAPLACE_UNIFORM_RIDGE);

    cholmod_sparse H_final = H_builder.as_cholmod(&solver.common());
    if (!solver.analyzed()) {
        TULPA_PROFILE_PHASE(PHASE_ANALYZE);
        solver.analyze(&H_final);
    }
    bool final_fact_ok = false;
    { TULPA_PROFILE_PHASE(PHASE_FACTORIZE);
      final_fact_ok = solver.factorize(&H_final); }
    if (final_fact_ok) {
        TULPA_PROFILE_PHASE(PHASE_LOG_DET);
        result.log_det_Q = solver.log_determinant();
    }

    double log_lik, log_prior;
    { TULPA_PROFILE_PHASE(PHASE_LOG_LIK_PRIOR);
      log_lik   = compute_total_log_lik_joint(arms, scratch.etas, n_threads);
      log_prior = compute_log_prior_joint(x, scratch.etas); }

    result.log_marginal = finalize_log_marginal(log_lik, log_prior,
                                                  result.log_det_Q, n_x);

    { TULPA_PROFILE_PHASE(PHASE_LOG_LIK_PRIOR);
      center_effects_fn(x); }
    for (int j = 0; j < n_x; j++) result.mode[j] = x[j];

    if (store_Q) {
        // Copy the CSC arrays out so the caller doesn't depend on H_builder
        // staying alive.
        result.Q_csc_p = H_builder.col_ptr;
        result.Q_csc_i = H_builder.row_idx;
        result.Q_csc_x = H_builder.values;
        result.Q_csc_n = n_x;
    }

    return result;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_JOINT_SPARSE_H
