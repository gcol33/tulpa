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
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>
#include <limits>
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
            if (sparse_solver.ensure_analyzed(
                    A, SparseCholeskySolver::kOwnedPatternTag) &&
                sparse_solver.factorize(A)) {
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
            sparse_ok = sparse_solver.ensure_analyzed(
                            A, SparseCholeskySolver::kOwnedPatternTag) &&
                        sparse_solver.factorize(A);
            if (sparse_ok) log_det_out = sparse_solver.log_determinant();
        }
    }
    if (!sparse_ok) {
        dense_cholesky_log_det_raw(H, n_x, dense_scratch, log_det_out);
    }
    return std::isfinite(log_det_out);
}

// g' H^-1 g off the factor a dispatch above left live: the CHOLMOD factor when
// `sparse_live`, the dense lower factor in `dense_scratch` otherwise. It is the
// Newton decrement at the point H and g were scattered at
// (`LaplaceResult::newton_decrement`). NaN where the solve fails. `step_out`,
// when given, receives the step H^-1 g the decrement is read off.
inline double newton_decrement_live(const double* grad, int n_x,
                                    bool sparse_live,
                                    SparseCholeskySolver& sparse_solver,
                                    DenseCholeskyScratch& dense_scratch,
                                    std::vector<double>* step_out = nullptr) {
    std::vector<double> local;
    std::vector<double>& step = step_out ? *step_out : local;
    step.assign(n_x, 0.0);
    bool ok;
    if (sparse_live) {
        ok = sparse_solver.solve(grad, step.data(), n_x);
    } else {
        std::vector<double> z_work(n_x, 0.0);
        ok = chol_substitute_raw(dense_scratch.L.data(), n_x, grad, step.data(),
                                 z_work.data());
    }
    if (!ok) return std::numeric_limits<double>::quiet_NaN();
    double dec = 0.0;
    for (int j = 0; j < n_x; j++) dec += grad[j] * step[j];
    return std::isfinite(dec) ? dec : std::numeric_limits<double>::quiet_NaN();
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

// ---- The same dispatch over a Hessian assembled into a SparseHessianBuilder.
//
// The builder holds only the structural nonzeros, so a Newton step neither
// zeroes nor reads an n x n matrix. CHOLMOD factors the builder's own CSC
// arrays in place; the symbolic factor is tied to the builder's pattern
// generation, so a solver shared with the dense route above re-analyzes when the
// pattern it last saw was the other one. Where CHOLMOD fails the builder is
// expanded into `dense_fallback` and the dense Cholesky runs on it, which is the
// fallback the dense route takes on the same matrix.

// Expand the builder's lower triangle into `H` (whose upper triangle is left
// zero, as no reader of a Newton Hessian touches it).
inline void sparse_builder_to_dense_lower(const SparseHessianBuilder& B,
                                          DenseMat& H) {
    H.zero();
    for (int j = 0; j < B.n; j++) {
        for (int k = B.col_ptr[j]; k < B.col_ptr[j + 1]; k++) {
            H[B.row_idx[k]][j] = B.values[k];
        }
    }
}

// The builder's lower triangle as CSC arrays under dense_to_csc_lower_drop_raw's
// rule: every diagonal kept, an off-diagonal kept when |value| > drop_tol.
inline void sparse_builder_to_csc_lower_drop(
    const SparseHessianBuilder& B, double drop_tol,
    std::vector<int>& csc_p, std::vector<int>& csc_i, std::vector<double>& csc_x
) {
    csc_p.assign(B.n + 1, 0);
    csc_i.clear();
    csc_x.clear();
    csc_i.reserve(B.nnz);
    csc_x.reserve(B.nnz);
    for (int j = 0; j < B.n; j++) {
        csc_p[j] = static_cast<int>(csc_i.size());
        for (int k = B.col_ptr[j]; k < B.col_ptr[j + 1]; k++) {
            const int r = B.row_idx[k];
            const double v = B.values[k];
            if (r == j || std::fabs(v) > drop_tol) {
                csc_i.push_back(r);
                csc_x.push_back(v);
            }
        }
    }
    csc_p[B.n] = static_cast<int>(csc_i.size());
}

// Factor a builder that ALREADY carries its diagonal ridge and solve
// H delta = grad. Counterpart of dispatch_factor_solve_ridged.
inline bool dispatch_factor_solve_sparse_ridged(
    SparseHessianBuilder& H, const DenseVec& grad, std::vector<double>& delta,
    int n_x, SparseCholeskySolver& sparse_solver,
    DenseMat& dense_fallback, DenseCholeskyScratch& dense_scratch
) {
    cholmod_sparse A = H.as_cholmod(&sparse_solver.common());
    bool ok = sparse_solver.ensure_analyzed(&A, H.pattern_generation) &&
              sparse_solver.factorize(&A) &&
              sparse_solver.solve(grad.data(), delta.data(), n_x);
    for (int j = 0; ok && j < n_x; j++) {
        if (!std::isfinite(delta[j])) ok = false;
    }
    if (!ok) {
        sparse_builder_to_dense_lower(H, dense_fallback);
        double log_det = 0.0;
        ok = dense_cholesky_solve_raw(dense_fallback, grad, n_x, dense_scratch,
                                      delta, log_det);
    }
    return ok;
}

inline bool dispatch_factor_solve_sparse(
    SparseHessianBuilder& H, const DenseVec& grad, std::vector<double>& delta,
    int n_x, SparseCholeskySolver& sparse_solver,
    DenseMat& dense_fallback, DenseCholeskyScratch& dense_scratch
) {
    H.add_uniform_ridge(LAPLACE_UNIFORM_RIDGE);
    return dispatch_factor_solve_sparse_ridged(H, grad, delta, n_x,
                                               sparse_solver, dense_fallback,
                                               dense_scratch);
}

// Ridge, factor and return log|H| from the builder. Counterpart of
// dispatch_factor_log_det, with the same finiteness contract.
inline bool dispatch_factor_log_det_sparse(
    SparseHessianBuilder& H, int n_x, SparseCholeskySolver& sparse_solver,
    DenseMat& dense_fallback, DenseCholeskyScratch& dense_scratch,
    double& log_det_out
) {
    H.add_uniform_ridge(LAPLACE_UNIFORM_RIDGE);
    log_det_out = 0.0;
    cholmod_sparse A = H.as_cholmod(&sparse_solver.common());
    const bool sparse_ok =
        sparse_solver.ensure_analyzed(&A, H.pattern_generation) &&
        sparse_solver.factorize(&A);
    if (sparse_ok) {
        log_det_out = sparse_solver.log_determinant();
    } else {
        sparse_builder_to_dense_lower(H, dense_fallback);
        dense_cholesky_log_det_raw(dense_fallback, n_x, dense_scratch,
                                   log_det_out);
    }
    return std::isfinite(log_det_out);
}

} // namespace tulpa

#endif // TULPA_LAPLACE_CHOLESKY_DISPATCH_H
