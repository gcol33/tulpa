// laplace_cholesky_dispatch.h
// Sparse-with-dense-fallback Cholesky dispatch for the dense Newton driver.
//
// The dense PIRLS solver in laplace_newton.h tries CHOLMOD on the sparsified
// dense Hessian when n_x >= SPARSE_THRESHOLD; on any failure (allocation,
// factorize, non-finite delta) it falls back to the hand-rolled dense
// Cholesky. That same dispatch happens twice per call (per-iteration solve,
// post-loop log-det), so it lives here as a single helper.

#ifndef TULPA_LAPLACE_CHOLESKY_DISPATCH_H
#define TULPA_LAPLACE_CHOLESKY_DISPATCH_H

#include "laplace_cholesky.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

inline constexpr double SPARSE_DROP_TOL_DISPATCH = 1e-12;

// Thread-safety contract:
//
// Both dispatch paths run without a critical section. CHOLMOD 5.x holds its
// allocator hooks on the global SuiteSparse_config rather than on
// `cholmod_common`, and M_cholmod_start overrides only `Common->error_handler`,
// so a per-thread `cholmod_common` is safe as long as SuiteSparse_config keeps
// its system-malloc defaults in Matrix's build. The dense fallback allocates
// raw `std::vector<double>` scratch, never an Rcpp vector: Rf_allocVector
// touches R's GC and is not safe off the main thread.

// Factor H and solve H * delta = grad. Try sparse CHOLMOD first if
// `prefer_sparse`, fall back to the dense hand-rolled Cholesky on any failure.
// Returns true if either path produced a finite delta.
//
// Both paths see `H + LAPLACE_UNIFORM_RIDGE * I`. The uniform upstream ridge
// guarantees positive-definiteness even on rank-deficient priors (ICAR, RW1,
// RW2, ...) so dense and sparse factor the same matrix and agree on
// log_det / mode to numerical tolerance. Neither path carries a pivot clamp or
// an LDL' retry of its own: a per-path repair fires on different pivot subsets
// in different elimination orders, which makes the two paths factor different
// matrices and diverge by O(1)-O(10) in log_marginal on a doubly
// rank-deficient input.
// Factor an H that ALREADY carries its diagonal ridge and solve H delta = grad.
// `log_det_out`, when non-null, receives log|H| from whichever factor succeeded.
// Split out of dispatch_factor_solve below so a caller escalating the diagonal
// across several attempts (joint_pd_step_solve_dense) loads the ridge itself
// rather than having the base ridge re-applied on every attempt.
inline bool dispatch_factor_solve_ridged(
    DenseMat& H, DenseVec& grad, std::vector<double>& delta, int n_x,
    SparseCholeskySolver& sparse_solver, bool prefer_sparse,
    DenseCholeskyScratch& dense_scratch,
    double* log_det_out = nullptr
) {
    bool ok = false;
    if (prefer_sparse) {
        // Owned-sparse path: first call discovers + caches the pattern; later
        // calls just refill Ax. No per-iter cholmod_sparse alloc/free, no
        // O(n^2) discovery scan. Solver owns A; do NOT free here.
        cholmod_sparse* A = sparse_solver.refill_from_dense(
            H, n_x, SPARSE_DROP_TOL_DISPATCH);
        if (A) {
            if (!sparse_solver.analyzed()) sparse_solver.analyze(A);
            if (sparse_solver.factorize(A)) {
                ok = sparse_solver.solve(grad.data(), delta.data(), n_x);
                for (int j = 0; ok && j < n_x; j++) {
                    if (!std::isfinite(delta[j])) { ok = false; break; }
                }
                if (ok && log_det_out) *log_det_out = sparse_solver.log_determinant();
            }
        }
    }
    if (!ok) {
        double log_det = 0.0;
        ok = dense_cholesky_solve_raw(H, grad, n_x, dense_scratch, delta,
                                       log_det);
        if (ok && log_det_out) *log_det_out = log_det;
    }
    return ok;
}

inline bool dispatch_factor_solve(
    DenseMat& H, DenseVec& grad, std::vector<double>& delta, int n_x,
    SparseCholeskySolver& sparse_solver, bool prefer_sparse,
    DenseCholeskyScratch& dense_scratch
) {
    // Apply the uniform upstream ridge once. H is rebuilt fresh per Newton
    // iter, so each call re-applies it on top of the unridged assembly.
    add_uniform_ridge_dense(H, n_x, LAPLACE_UNIFORM_RIDGE);
    return dispatch_factor_solve_ridged(H, grad, delta, n_x, sparse_solver,
                                        prefer_sparse, dense_scratch);
}

// Factor an H that ALREADY carries its diagonal ridge and return log|H| via the
// diagonal of L. Same sparse/dense dispatch as dispatch_factor_solve_ridged, and
// split out for the same reason: a caller that factors ONE assembled H twice --
// once for the log-determinant, then again through an escalating solve when that
// log-determinant is not finite -- loads the base ridge itself, once, instead of
// each entry re-applying it on top of the previous one.
//
// Returns whether either path produced a FINITE log-determinant. False is the
// Cholesky reporting that H is not PD at this point. Carrying the value on
// instead turns a failed cell into an undefined outer-grid weight (NaN), or, on
// an exactly singular direction, into a cell that takes the whole grid: a -Inf
// log-determinant is a +Inf log-marginal.
inline bool dispatch_factor_log_det_ridged(
    DenseMat& H, int n_x,
    SparseCholeskySolver& sparse_solver, bool prefer_sparse,
    DenseCholeskyScratch& dense_scratch,
    double& log_det_out
) {
    log_det_out = 0.0;
    bool sparse_ok = false;
    if (prefer_sparse) {
        cholmod_sparse* A = sparse_solver.refill_from_dense(
            H, n_x, SPARSE_DROP_TOL_DISPATCH);
        if (A) {
            if (!sparse_solver.analyzed()) sparse_solver.analyze(A);
            sparse_ok = sparse_solver.factorize(A);
            if (sparse_ok) log_det_out = sparse_solver.log_determinant();
        }
    }
    if (!sparse_ok) {
        dense_cholesky_log_det_raw(H, n_x, dense_scratch, log_det_out);
    }
    return std::isfinite(log_det_out);
}

// Factor H and return log|H + ridge*I| via the diagonal of L. Same
// sparse/dense dispatch and uniform upstream regularization as
// dispatch_factor_solve.
inline bool dispatch_factor_log_det(
    DenseMat& H, int n_x,
    SparseCholeskySolver& sparse_solver, bool prefer_sparse,
    DenseCholeskyScratch& dense_scratch,
    double& log_det_out
) {
    add_uniform_ridge_dense(H, n_x, LAPLACE_UNIFORM_RIDGE);
    return dispatch_factor_log_det_ridged(H, n_x, sparse_solver, prefer_sparse,
                                          dense_scratch, log_det_out);
}

} // namespace tulpa

#endif // TULPA_LAPLACE_CHOLESKY_DISPATCH_H
