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

namespace tulpa {

inline constexpr double SPARSE_DROP_TOL_DISPATCH = 1e-12;

// Factor H and solve H * delta = grad. Try sparse CHOLMOD first if
// `prefer_sparse`, fall back to the dense hand-rolled Cholesky on any failure.
// Returns true if either path produced a finite delta.
inline bool dispatch_factor_solve(
    DenseMat& H, DenseVec& grad, std::vector<double>& delta, int n_x,
    SparseCholeskySolver& sparse_solver, bool prefer_sparse
) {
    bool ok = false;
    if (prefer_sparse) {
        cholmod_sparse* A = dense_to_cholmod_sparse_drop(
            H, n_x, SPARSE_DROP_TOL_DISPATCH, &sparse_solver.common());
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
    }
    if (!ok) {
        auto chol = dense_cholesky_solve(H, grad, n_x);
        ok = chol.success;
        for (int j = 0; j < n_x; j++) delta[j] = chol.delta[j];
    }
    return ok;
}

// Factor H and return log|H| via the diagonal of L. Same sparse/dense
// dispatch as dispatch_factor_solve; the dense fallback's log_det semantics
// match dense_cholesky_factorize (which never throws and returns 0 on failure).
inline void dispatch_factor_log_det(
    DenseMat& H, int n_x,
    SparseCholeskySolver& sparse_solver, bool prefer_sparse,
    double& log_det_out
) {
    log_det_out = 0.0;
    if (prefer_sparse) {
        cholmod_sparse* A = dense_to_cholmod_sparse_drop(
            H, n_x, SPARSE_DROP_TOL_DISPATCH, &sparse_solver.common());
        if (A) {
            if (!sparse_solver.analyzed()) sparse_solver.analyze(A);
            bool sparse_ok = sparse_solver.factorize(A);
            if (sparse_ok) log_det_out = sparse_solver.log_determinant();
            M_cholmod_free_sparse(&A, &sparse_solver.common());
            if (sparse_ok) return;
        }
    }
    Rcpp::NumericMatrix L_unused(n_x, n_x);
    dense_cholesky_factorize(H, n_x, L_unused, log_det_out);
}

} // namespace tulpa

#endif // TULPA_LAPLACE_CHOLESKY_DISPATCH_H
