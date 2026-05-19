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

// Thread-safety contract (experiment, 2026-05-19):
//
// We previously wrapped both dispatch paths in `omp critical(tulpa_cholmod)`
// on the assumption that CHOLMOD-via-Matrix routes allocations through R's
// GC and is therefore unsafe under concurrent threads. The dense fallback
// path also allocated Rcpp::NumericMatrix scratch, which IS unsafe under
// threads (Rf_allocVector touches R's GC).
//
// Closer reading of Matrix 1.7+ headers shows CHOLMOD 5.x moved its
// allocator hooks off `cholmod_common` and onto the global SuiteSparse_config.
// M_cholmod_start only overrides `Common->error_handler`, not the allocator.
// So *if* SuiteSparse_config keeps its system-malloc defaults in Matrix's
// build, per-thread `cholmod_common` instances are thread-safe out of the
// box and the critical section was masking the Rcpp-scratch unsafety in the
// dense fallback.
//
// This file now uses raw `std::vector<double>` scratch in the dense fallback
// (no R allocations on the parallel hot path) and runs the CHOLMOD calls
// uncritical. If CHOLMOD-via-Matrix still races, the test harness will fail
// and we'll know we need to bundle SuiteSparse or switch to Eigen.

// Factor H and solve H * delta = grad. Try sparse CHOLMOD first if
// `prefer_sparse`, fall back to the dense hand-rolled Cholesky on any failure.
// Returns true if either path produced a finite delta.
inline bool dispatch_factor_solve(
    DenseMat& H, DenseVec& grad, std::vector<double>& delta, int n_x,
    SparseCholeskySolver& sparse_solver, bool prefer_sparse,
    DenseCholeskyScratch& dense_scratch
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
                sparse_solver.solve(grad.data(), delta.data(), n_x);
                ok = true;
                for (int j = 0; j < n_x; j++) {
                    if (!std::isfinite(delta[j])) { ok = false; break; }
                }
            }
        }
    }
    if (!ok) {
        double log_det_unused = 0.0;
        ok = dense_cholesky_solve_raw(H, grad, n_x, dense_scratch, delta,
                                       log_det_unused);
    }
    return ok;
}

// Factor H and return log|H| via the diagonal of L. Same sparse/dense
// dispatch as dispatch_factor_solve.
inline void dispatch_factor_log_det(
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
}

} // namespace tulpa

#endif // TULPA_LAPLACE_CHOLESKY_DISPATCH_H
