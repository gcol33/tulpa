// laplace_newton.h
// PIRLS-equivalent Newton/Fisher scoring solver for Laplace modes.

#ifndef TULPA_LAPLACE_NEWTON_H
#define TULPA_LAPLACE_NEWTON_H

#include "laplace_cholesky.h"
#include "laplace_cholesky_dispatch.h"  // dispatch_factor_solve, dispatch_factor_log_det
#include "laplace_family_link.h"
#include "laplace_newton_loop.h"        // eval_*, line_search_backtrack, finalize_log_marginal
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

constexpr int SPARSE_THRESHOLD = 200;
// SPARSE_DROP_TOL lives in laplace_cholesky_dispatch.h as SPARSE_DROP_TOL_DISPATCH.
// MAX_HALVING is defined in laplace_newton_loop.h.

// Per-thread scratch for the single-arm Newton solver. All buffers are
// allocated once by the caller (single-threaded, outside any OpenMP parallel
// region) and reused across grid points and Newton iterations.
//
// Two reasons to hoist:
//   1. Rcpp::NumericVector — Rf_allocVector is not thread-safe.
//   2. DenseVec/DenseMat   — std::vector allocation is thread-safe but the
//      per-iter (n_x + 1) mallocs for grad / H / delta hit the central
//      allocator under concurrent outer-grid threads, manifesting as
//      Amdahl-shaped serial overhead inside the parallel region. Allocating
//      once and zero-ing each iter eliminates the contention.
//
// DenseCholeskyScratch holds raw std::vector buffers for the dense-fallback
// Cholesky factorization so that path is also Rcpp-free on the hot loop.
struct NewtonScratch {
    Rcpp::NumericVector x;       // size n_x — current Newton iterate
    Rcpp::NumericVector x_try;   // size n_x — step-halving trial
    Rcpp::NumericVector eta;     // size N — current linear predictor
    Rcpp::NumericVector eta_tmp; // size N — eval_objective trial
    DenseVec  grad;              // size n_x — Newton gradient, zeroed per iter
    DenseMat  H;                 // n_x x n_x — Newton Hessian, zeroed per iter
    DenseVec  delta;             // size n_x — Newton step, zeroed per iter
    DenseCholeskyScratch chol;   // raw L/z buffers for dense fallback

    void allocate(int n_x, int N) {
        x       = Rcpp::NumericVector(n_x, 0.0);
        x_try   = Rcpp::NumericVector(n_x, 0.0);
        eta     = Rcpp::NumericVector(N, 0.0);
        eta_tmp = Rcpp::NumericVector(N, 0.0);
        grad.assign(n_x, 0.0);
        H.assign(n_x, DenseVec(n_x, 0.0));
        delta.assign(n_x, 0.0);
        chol.ensure(n_x);
    }

    // Clear grad / H / delta before a fresh scatter. Sizes are fixed at
    // allocate(); this only zeros values, no malloc traffic.
    void zero_for_iter() {
        std::fill(grad.begin(), grad.end(), 0.0);
        H.zero();
        std::fill(delta.begin(), delta.end(), 0.0);
    }
};

// Scratch-aware, likelihood-agnostic Newton solver. The four pre-allocated
// buffers in `scratch` must be sized to (n_x, n_x, N, N). The solver does not
// allocate any Rcpp objects; the caller can therefore drive this from a
// parallel region as long as the SparseCholeskySolver is also thread-local.
//
// The data log-likelihood enters ONLY through `log_lik_fn(eta) -> double`, so
// the loop carries no family knowledge: the family-enum mode finders pass a
// FamilyLogLik (see the forwarding overload below) and the LikelihoodSpec path
// passes a functor backed by spec.ll_double. This is the single Newton loop the
// whole engine shares.
template<typename ComputeEta, typename ScatterGradHess,
         typename CenterEffects, typename ComputeLogPrior, typename LogLik>
LaplaceResult laplace_newton_solve_ll(
    int N, int n_x,
    int max_iter, double tol,
    ComputeEta compute_eta,
    ScatterGradHess scatter_grad_hess,
    CenterEffects center_effects_fn,
    ComputeLogPrior compute_log_prior,
    LogLik log_lik_fn,
    NewtonScratch& scratch,
    const std::vector<double>& x_init,
    SparseCholeskySolver* shared_solver,
    bool store_Q,
    const std::vector<std::pair<int, int>>* inv_block_layout = nullptr,
    int sparse_override = 0
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
    // sparse_override: 0 = auto (size threshold), >0 = force sparse, <0 = force
    // dense. The override lets the test suite drive the same problem through both
    // factorization paths for a byte-level dense == sparse equivalence gate,
    // mirroring the joint path's force_sparse control.
    bool use_sparse = (sparse_override == 0)
                          ? (n_x >= SPARSE_THRESHOLD)
                          : (sparse_override > 0);

    SparseCholeskySolver local_solver;
    SparseCholeskySolver& sparse_solver = shared_solver ? *shared_solver : local_solver;

    // Do NOT call omp_set_num_threads here. When the outer driver runs us
    // from inside a parallel region we want the inner kernels (per-obs
    // scatter, etc.) to inherit the per-thread context. The closures and the
    // log-lik functor own their own threading.

    auto eval_objective = [&](const Rcpp::NumericVector& xv) -> double {
        return eval_penalized_log_lik_ll(
            xv, compute_eta, compute_log_prior, log_lik_fn, scratch.eta_tmp
        );
    };

    auto cholesky_solve = [&](DenseMat& H, DenseVec& grad,
                              std::vector<double>& delta) -> bool {
        return dispatch_factor_solve(H, grad, delta, n_x, sparse_solver,
                                     use_sparse, scratch.chol);
    };

    double obj_current = -1e300;
    bool obj_valid = false;
    NewtonConvState conv_state;

    for (int iter = 0; iter < max_iter; iter++) {
        compute_eta(x, scratch.eta);
        scratch.zero_for_iter();
        scatter_grad_hess(x, scratch.eta, scratch.grad, scratch.H);

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

        double slope = newton_decrement(scratch.grad, scratch.delta, n_x);
        double step_scale = line_search_backtrack(
            x, scratch.delta, n_x, obj_current, slope, eval_objective,
            obj_current, scratch.x_try
        );

        result.n_iter = iter + 1;
        if (newton_converged(scratch.delta, scratch.grad, step_scale, n_x, tol,
                             conv_state)) {
            result.converged = true;
            break;
        }
    }

    center_effects_fn(x);
    for (int j = 0; j < n_x; j++) result.mode[j] = x[j];

    compute_eta(x, scratch.eta);
    scratch.zero_for_iter();
    scatter_grad_hess(x, scratch.eta, scratch.grad, scratch.H);

    dispatch_factor_log_det(scratch.H, n_x, sparse_solver, use_sparse,
                             scratch.chol, result.log_det_Q);

    // Diagonal blocks of H^{-1} for the requested index ranges. Reuses the
    // factor just built for the log-determinant (no refactorization): for each
    // unit column e_j inside a block we solve H v = e_j and read the block
    // rows of v, giving the FULL-inverse block (fixed effects and other blocks
    // marginalized out). Sparse path solves against the live CHOLMOD factor;
    // dense path back-substitutes the live scratch.chol.L. Each block is
    // symmetrized and stored column-major.
    if (inv_block_layout && !inv_block_layout->empty()) {
        bool used_sparse_factor = use_sparse && sparse_solver.factored();
        std::vector<double> rhs(n_x, 0.0), col(n_x, 0.0);
        std::vector<double> z_work;
        if (!used_sparse_factor) z_work.assign(n_x, 0.0);

        for (const auto& blk : *inv_block_layout) {
            int s = blk.first, m = blk.second;
            std::vector<double> block(static_cast<std::size_t>(m) * m, 0.0);

            for (int c = 0; c < m; c++) {
                std::fill(rhs.begin(), rhs.end(), 0.0);
                rhs[s + c] = 1.0;
                if (used_sparse_factor) {
                    sparse_solver.solve(rhs.data(), col.data(), n_x);
                } else {
                    chol_substitute_raw(scratch.chol.L.data(), n_x,
                                        rhs.data(), col.data(), z_work.data());
                }
                for (int r = 0; r < m; r++) block[r * m + c] = col[s + r];
            }

            // Symmetrize (numerical asymmetry only) and append column-major.
            for (int cc = 0; cc < m; cc++) {
                for (int r = 0; r < m; r++) {
                    result.re_cov_flat.push_back(
                        0.5 * (block[r * m + cc] + block[cc * m + r]));
                }
            }
            result.re_cov_block_sizes.push_back(m);
        }
    }

    double log_lik = log_lik_fn(scratch.eta);
    double log_prior = compute_log_prior(x, scratch.eta);

    result.log_marginal = finalize_log_marginal(log_lik, log_prior, result.log_det_Q, n_x);

    if (store_Q) {
        // Drop tolerance matches the sparse-Cholesky dispatch path so the
        // exported CSC pattern is consistent with the in-loop solve when
        // n_x >= SPARSE_THRESHOLD.
        dense_to_csc_lower_drop_raw(
            scratch.H, n_x, SPARSE_DROP_TOL_DISPATCH,
            result.Q_csc_p, result.Q_csc_i, result.Q_csc_x
        );
        result.Q_csc_n = n_x;
    }

    return result;
}

// Family-enum forwarder (scratch-aware). Wraps the built-in family log-lik as
// the functor and delegates to the shared loop above, so the family-string
// callers (laplace_core*, the nested ST driver, spde_qbuilder) keep their exact
// signature while the loop body lives in one place.
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
    NewtonScratch& scratch,
    const std::vector<double>& x_init,
    SparseCholeskySolver* shared_solver,
    bool store_Q,
    const std::vector<std::pair<int, int>>* inv_block_layout = nullptr
) {
    FamilyLogLik ll{&y, &n_trials, N, family, phi, n_threads};
    return laplace_newton_solve_ll(
        N, n_x, max_iter, tol,
        compute_eta, scatter_grad_hess, center_effects_fn, compute_log_prior,
        ll, scratch, x_init, shared_solver, store_Q, inv_block_layout
    );
}

// Convenience overload: allocates scratch locally. Used by the standalone
// laplace_mode_* entry points that are called once per R-export and do not
// participate in any outer-grid parallelism. NOT safe to call from inside an
// OpenMP parallel region — use the scratch-aware overload above for that.
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
    SparseCholeskySolver* shared_solver = nullptr,
    bool store_Q = false,
    const std::vector<std::pair<int, int>>* inv_block_layout = nullptr
) {
    NewtonScratch scratch;
    scratch.allocate(n_x, N);
    std::vector<double> x_init_vec;
    if (x_init.size() == n_x) {
        x_init_vec.assign(x_init.begin(), x_init.end());
    }
    // n_threads flows into laplace_newton_solve, which sizes its own
    // regions; no process-global omp_set_num_threads here.
    return laplace_newton_solve(
        y, n_trials, family, phi, N, n_x,
        max_iter, tol, n_threads,
        compute_eta, scatter_grad_hess, center_effects_fn, compute_log_prior,
        scratch, x_init_vec, shared_solver, store_Q, inv_block_layout
    );
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_H
