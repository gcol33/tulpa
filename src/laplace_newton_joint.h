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

// Joint data log-likelihood as a functor of the per-arm eta vectors. The joint
// analog of FamilyLogLik: the joint Newton loop reads the data log-lik only
// through `log_lik_fn(etas) -> double`, so the family-enum joint driver passes
// this (summing compute_total_log_lik over arms) while the spec-driven joint
// path (L4) passes a functor backed by each arm's LikelihoodSpec. The borrowed
// arms vector must outlive the fit.
struct JointFamilyLogLik {
    const std::vector<JointArm>* arms = nullptr;
    int n_threads = 1;
    double operator()(const std::vector<Rcpp::NumericVector>& etas) const {
        return compute_total_log_lik_joint(*arms, etas, n_threads);
    }
};

// Penalised log-lik (joint), likelihood-agnostic: the data log-lik enters as a
// functor of the per-arm etas, mirroring eval_penalized_log_lik_ll. `etas_scratch`
// is the caller's pre-allocated per-arm eta buffer set; we never reallocate.
template<typename ComputeEtaJoint, typename ComputeLogPriorJoint, typename JointLogLik>
inline double eval_penalized_log_lik_joint_ll(
    const Rcpp::NumericVector& x,
    ComputeEtaJoint compute_eta_joint,
    ComputeLogPriorJoint compute_log_prior_joint,
    JointLogLik log_lik_fn,
    std::vector<Rcpp::NumericVector>& etas_scratch
) {
    compute_eta_joint(x, etas_scratch);
    double ll = log_lik_fn(etas_scratch);
    double lp = compute_log_prior_joint(x, etas_scratch);
    return ll + lp;
}

// Family-enum convenience overload: wraps the per-arm built-in log-lik as the
// functor. Single source of truth for the joint loop body.
template<typename ComputeEtaJoint, typename ComputeLogPriorJoint>
inline double eval_penalized_log_lik_joint(
    const Rcpp::NumericVector& x,
    const std::vector<JointArm>& arms,
    int n_threads,
    ComputeEtaJoint compute_eta_joint,
    ComputeLogPriorJoint compute_log_prior_joint,
    std::vector<Rcpp::NumericVector>& etas_scratch
) {
    JointFamilyLogLik ll{&arms, n_threads};
    return eval_penalized_log_lik_joint_ll(x, compute_eta_joint,
                                           compute_log_prior_joint, ll,
                                           etas_scratch);
}

// Per-thread scratch for the joint Newton solver. Same role as NewtonScratch
// but with per-arm eta vectors. Allocate once, single-threaded outside any
// OpenMP region. See NewtonScratch comment for why grad / H / delta are
// hoisted: per-iter std::vector allocation is thread-safe but contends on the
// central allocator under concurrent outer-grid threads, eating parallel
// efficiency.
struct NewtonScratchJoint {
    Rcpp::NumericVector x;       // size n_x
    Rcpp::NumericVector x_try;   // size n_x
    std::vector<Rcpp::NumericVector> etas;      // size = arms.size(), each N_k
    std::vector<Rcpp::NumericVector> etas_tmp;  // same shape, line-search buffer
    DenseVec  grad;              // size n_x, zeroed per iter
    DenseMat  H;                 // n_x x n_x, zeroed per iter
    DenseVec  delta;             // size n_x, zeroed per iter
    DenseCholeskyScratch chol;   // raw L/z buffers for dense fallback

    void allocate(int n_x, const std::vector<JointArm>& arms) {
        x       = Rcpp::NumericVector(n_x, 0.0);
        x_try   = Rcpp::NumericVector(n_x, 0.0);
        etas.clear();     etas.reserve(arms.size());
        etas_tmp.clear(); etas_tmp.reserve(arms.size());
        for (const JointArm& a : arms) {
            etas.emplace_back(a.N, 0.0);
            etas_tmp.emplace_back(a.N, 0.0);
        }
        grad.assign(n_x, 0.0);
        H.assign(n_x, DenseVec(n_x, 0.0));
        delta.assign(n_x, 0.0);
        chol.ensure(n_x);
    }

    void zero_for_iter() {
        std::fill(grad.begin(), grad.end(), 0.0);
        H.zero();
        std::fill(delta.begin(), delta.end(), 0.0);
    }
};

// Scratch-aware, likelihood-agnostic joint Newton solver. Like the single-arm
// laplace_newton_solve_ll, the data log-lik enters ONLY through
// `log_lik_fn(etas) -> double`, so the loop carries no family knowledge: the
// family-enum forwarder below passes a JointFamilyLogLik and the spec-driven
// joint path (L4) passes a functor backed by each arm's LikelihoodSpec.
// Performs zero Rcpp allocations once `scratch` is supplied; safe inside an
// OpenMP parallel region with a thread-local `shared_solver`.
template<typename ComputeEtaJoint, typename ScatterJoint,
         typename CenterEffects, typename ComputeLogPriorJoint, typename JointLogLik>
LaplaceResult laplace_newton_solve_joint_ll(
    int n_x,
    int max_iter, double tol,
    ComputeEtaJoint compute_eta_joint,
    ScatterJoint scatter_joint,
    CenterEffects center_effects_fn,
    ComputeLogPriorJoint compute_log_prior_joint,
    JointLogLik log_lik_fn,
    NewtonScratchJoint& scratch,
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
    bool use_sparse = (n_x >= SPARSE_THRESHOLD);

    SparseCholeskySolver local_solver;
    SparseCholeskySolver& sparse_solver = shared_solver ? *shared_solver : local_solver;

    auto eval_objective = [&](const Rcpp::NumericVector& xv) -> double {
        return eval_penalized_log_lik_joint_ll(
            xv, compute_eta_joint, compute_log_prior_joint, log_lik_fn,
            scratch.etas_tmp
        );
    };

    auto cholesky_solve = [&](DenseMat& H, DenseVec& grad,
                              std::vector<double>& delta) -> bool {
        return dispatch_factor_solve(H, grad, delta, n_x, sparse_solver,
                                     use_sparse, scratch.chol);
    };

    double obj_current = -1e300;
    bool obj_valid = false;

    for (int iter = 0; iter < max_iter; iter++) {
        compute_eta_joint(x, scratch.etas);
        scratch.zero_for_iter();
        scatter_joint(x, scratch.etas, scratch.grad, scratch.H);

        bool solve_ok = cholesky_solve(scratch.H, scratch.grad, scratch.delta);

        if (!solve_ok) {
            for (int j = 0; j < n_x; j++) {
                if (std::isfinite(scratch.delta[j])) x[j] += 0.1 * scratch.delta[j];
            }
            obj_valid = false;
            result.n_iter = iter + 1;
            continue;
        }

        if (!obj_valid) {
            obj_current = eval_objective(x);
            obj_valid = true;
        }

        double step_scale = step_halving_update(
            x, scratch.delta, n_x, obj_current, eval_objective, obj_current,
            scratch.x_try
        );

        result.n_iter = iter + 1;
        if (max_abs_step(scratch.delta, step_scale, n_x) < tol) {
            result.converged = true;
            break;
        }
    }

    // Compute log_marginal at the Newton mode (uncentered). For BYM2/ICAR the
    // ICAR prior is rank-deficient along the constant-shift direction in phi,
    // and the obs Hessian is what pins that direction down, so the joint MAP
    // is at the uncentered Newton iterate. Centering phi without compensating
    // each arm's intercept would shift eta off the mode and corrupt log_lik
    // (and would also shift the proper-CAR log_prior). We center post-hoc
    // *after* log_marginal so the reported mode block has mean(phi) = 0 with
    // the equivalent intercept shift absorbed into each arm's first beta
    // column. Net effect: same eta, same log_marginal, centered phi block.
    compute_eta_joint(x, scratch.etas);
    scratch.zero_for_iter();
    scatter_joint(x, scratch.etas, scratch.grad, scratch.H);

    dispatch_factor_log_det(scratch.H, n_x, sparse_solver, use_sparse,
                             scratch.chol, result.log_det_Q);

    double log_lik   = log_lik_fn(scratch.etas);
    double log_prior = compute_log_prior_joint(x, scratch.etas);

    result.log_marginal = finalize_log_marginal(log_lik, log_prior, result.log_det_Q, n_x);

    center_effects_fn(x);
    for (int j = 0; j < n_x; j++) result.mode[j] = x[j];

    if (store_Q) {
        dense_to_csc_lower_drop_raw(
            scratch.H, n_x, SPARSE_DROP_TOL_DISPATCH,
            result.Q_csc_p, result.Q_csc_i, result.Q_csc_x
        );
        result.Q_csc_n = n_x;
    }

    return result;
}

// Family-enum forwarder (scratch-aware): wraps the per-arm built-in log-lik as
// the functor and delegates to the shared loop above, keeping the existing
// nested joint driver's call sites unchanged while the loop body lives in one
// place.
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
    NewtonScratchJoint& scratch,
    const std::vector<double>& x_init,
    SparseCholeskySolver* shared_solver,
    bool store_Q
) {
    JointFamilyLogLik ll{&arms, n_threads};
    return laplace_newton_solve_joint_ll(
        n_x, max_iter, tol,
        compute_eta_joint, scatter_joint, center_effects_fn,
        compute_log_prior_joint, ll, scratch, x_init, shared_solver, store_Q
    );
}

// Convenience overload that allocates scratch locally. Safe to call from
// non-parallel call sites only; the outer driver passes scratch through
// when it parallelises the grid.
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
    NewtonScratchJoint scratch;
    scratch.allocate(n_x, arms);
    std::vector<double> x_init_vec;
    if (x_init.size() == n_x) {
        x_init_vec.assign(x_init.begin(), x_init.end());
    }
    #ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    #endif
    return laplace_newton_solve_joint(
        arms, n_x, max_iter, tol, n_threads,
        compute_eta_joint, scatter_joint, center_effects_fn,
        compute_log_prior_joint,
        scratch, x_init_vec, shared_solver, store_Q
    );
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_JOINT_H
