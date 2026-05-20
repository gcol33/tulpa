// laplace_cholesky.h
// Dense Cholesky helpers used by Laplace solvers.

#ifndef TULPA_LAPLACE_CHOLESKY_H
#define TULPA_LAPLACE_CHOLESKY_H

#include "laplace_types.h"
#include <Rcpp.h>
#include <cmath>

namespace tulpa {

// Uniform upstream diagonal ridge.
//
// Every Laplace factorize-call site adds `LAPLACE_UNIFORM_RIDGE * I` to H
// before factorization. This is a tiny vague Gaussian prior on the latent
// (precision = ridge per coordinate) that does two jobs:
//
//   1. Guarantees H + ridge*I is PD whenever H is PSD with rank deficit k.
//      Cholesky pivots stay strictly positive, so dense and sparse paths
//      agree even on rank-deficient priors (ICAR with k=1, RW1 with k=1,
//      ICAR x RW1 with k=2, etc.). No asymmetric pivot clamp, no
//      symmetric CHOLMOD dbound fallback that activates on a different
//      pivot subset than the dense path.
//   2. Bounds log_det contributions from rank-deficient directions at
//      log(ridge) per zero eigenvalue, a theoretically interpretable
//      baseline (vague Gaussian prior on those directions). Replaces
//      both the dense clamp's elimination-order-dependent log(1e-6) and
//      CHOLMOD's permutation-dependent log(1e-6).
//
// The value 1e-10 is well below typical full-rank pivots (where ICAR
// adjacencies produce O(1)-O(10) Schur complements at the n_sites scale
// we care about), so healthy fits are unaffected to numerical tolerance.
// Rank-deficient log_marginal baselines shift by k*log(1e-10/1e-6) =
// -k*log(1e4) ~ -9.2*k units versus the pre-ridge dense baseline. This
// is a documented one-time baseline shift, not a regression.
inline constexpr double LAPLACE_UNIFORM_RIDGE = 1e-10;

// Add `ridge` to the diagonal of a DenseMat-storage Hessian. Used by
// `dispatch_factor_solve` / `dispatch_factor_log_det` and the LikelihoodSpec
// dense entry before factorization.
inline void add_uniform_ridge_dense(DenseMat& H, int n, double ridge) {
    for (int j = 0; j < n; j++) H[j][j] += ridge;
}

struct CholeskyResult {
    Rcpp::NumericMatrix L;
    Rcpp::NumericVector delta;
    double log_det;
    bool success;
};

// Per-thread scratch for the raw-buffer dense Cholesky path.
// Holds L (n*n, column-major: L_data[i + j*n]) and z (n).
// Allocate once outside the parallel region; reuse across cells/iterations.
struct DenseCholeskyScratch {
    std::vector<double> L;   // column-major, capacity >= n*n
    std::vector<double> z;   // capacity >= n
    int n_alloc = 0;

    void ensure(int n) {
        if (n_alloc < n) {
            L.assign(static_cast<size_t>(n) * n, 0.0);
            z.assign(n, 0.0);
            n_alloc = n;
        }
    }
};

namespace detail {

// Element accessors that unify DenseMat (vector-of-vectors) and
// Rcpp::NumericMatrix under one template. Adding a new matrix type only
// needs one more overload here.
inline double chol_at(const DenseMat& H, int j, int k)            { return H[j][k]; }
inline double chol_at(const Rcpp::NumericMatrix& H, int j, int k) { return H(j, k); }

// Templated Cholesky factorization core: writes L (lower-triangular) and
// log_det = log|H| from any matrix-like H supported by chol_at().
//
// Rank-deficient handling lives upstream: every Laplace callsite adds
// `LAPLACE_UNIFORM_RIDGE * I` to H before calling here (see
// `add_uniform_ridge_dense` in this header). The factorize routines below
// therefore assume H is PD; a non-positive pivot during elimination
// surfaces as NaN through sqrt and propagates to log_det / delta so the
// caller can detect it. There is no in-pivot clamp.
template <class MatLike>
inline void cholesky_factorize_impl(
    const MatLike& H, int n,
    Rcpp::NumericMatrix& L, double& log_det
) {
    log_det = 0.0;
    for (int j = 0; j < n; j++) {
        for (int k = 0; k <= j; k++) {
            double sum = chol_at(H, j, k);
            for (int i = 0; i < k; i++) sum -= L(j, i) * L(k, i);
            if (j == k) {
                // No in-pivot clamp: every callsite adds
                // LAPLACE_UNIFORM_RIDGE * I upstream, so a non-positive
                // pivot here means the upstream ridge wasn't applied
                // (a bug) — let sqrt produce NaN and propagate.
                L(j, k) = std::sqrt(sum);
            } else {
                L(j, k) = sum / L(k, k);
            }
        }
    }
    for (int j = 0; j < n; j++) {
        log_det += std::log(L(j, j));
    }
    log_det *= 2.0;
}

// Raw-buffer variant. L_data is column-major, leading dimension n
// (i.e. L(j,k) lives at L_data[j + k*n]). Thread-safe: no Rcpp allocation.
template <class MatLike>
inline void cholesky_factorize_impl_raw(
    const MatLike& H, int n,
    double* L_data, double& log_det
) {
    log_det = 0.0;
    for (int j = 0; j < n; j++) {
        for (int k = 0; k <= j; k++) {
            double sum = chol_at(H, j, k);
            for (int i = 0; i < k; i++) {
                sum -= L_data[j + i * n] * L_data[k + i * n];
            }
            if (j == k) {
                // No in-pivot clamp: every callsite adds
                // LAPLACE_UNIFORM_RIDGE * I upstream, so a non-positive
                // pivot here means the upstream ridge wasn't applied
                // (a bug) — let sqrt produce NaN and propagate.
                L_data[j + k * n] = std::sqrt(sum);
            } else {
                L_data[j + k * n] = sum / L_data[k + k * n];
            }
        }
    }
    for (int j = 0; j < n; j++) {
        log_det += std::log(L_data[j + j * n]);
    }
    log_det *= 2.0;
}

} // namespace detail

inline CholeskyResult dense_cholesky_solve(
    const DenseMat& H, const DenseVec& rhs, int n
) {
    CholeskyResult res;
    res.L = Rcpp::NumericMatrix(n, n);
    res.delta = Rcpp::NumericVector(n);
    res.success = true;

    detail::cholesky_factorize_impl(H, n, res.L, res.log_det);

    Rcpp::NumericVector z(n);
    for (int j = 0; j < n; j++) {
        double sum = rhs[j];
        for (int k = 0; k < j; k++) sum -= res.L(j, k) * z[k];
        z[j] = sum / res.L(j, j);
    }

    for (int j = n - 1; j >= 0; j--) {
        double sum = z[j];
        for (int k = j + 1; k < n; k++) sum -= res.L(k, j) * res.delta[k];
        res.delta[j] = sum / res.L(j, j);
    }

    for (int j = 0; j < n; j++) {
        if (!std::isfinite(res.delta[j])) {
            res.success = false;
            break;
        }
    }

    return res;
}

inline void dense_cholesky_factorize(
    const DenseMat& H, int n,
    Rcpp::NumericMatrix& L, double& log_det
) {
    detail::cholesky_factorize_impl(H, n, L, log_det);
}

inline void dense_cholesky_factorize(
    const Rcpp::NumericMatrix& H, int n,
    Rcpp::NumericMatrix& L, double& log_det
) {
    detail::cholesky_factorize_impl(H, n, L, log_det);
}

// Raw-buffer solve: writes delta_out in-place, uses scratch.L/z, no Rcpp alloc.
// Returns true if the back-substituted delta is finite. Singular-pivot guard
// inside cholesky_factorize_impl_raw clamps tiny pivots so factorization
// always completes; success here only tracks back-substitution finiteness.
inline bool dense_cholesky_solve_raw(
    const DenseMat& H, const DenseVec& rhs, int n,
    DenseCholeskyScratch& scratch,
    std::vector<double>& delta_out,
    double& log_det_out
) {
    scratch.ensure(n);
    if (static_cast<int>(delta_out.size()) < n) delta_out.assign(n, 0.0);
    double* L_data = scratch.L.data();
    double* z = scratch.z.data();

    detail::cholesky_factorize_impl_raw(H, n, L_data, log_det_out);

    // Forward substitution: L z = rhs  (L is column-major lower-triangular)
    for (int j = 0; j < n; j++) {
        double sum = rhs[j];
        for (int k = 0; k < j; k++) sum -= L_data[j + k * n] * z[k];
        z[j] = sum / L_data[j + j * n];
    }

    // Back substitution: L' delta = z
    for (int j = n - 1; j >= 0; j--) {
        double sum = z[j];
        for (int k = j + 1; k < n; k++) {
            sum -= L_data[k + j * n] * delta_out[k];
        }
        delta_out[j] = sum / L_data[j + j * n];
    }

    for (int j = 0; j < n; j++) {
        if (!std::isfinite(delta_out[j])) return false;
    }
    return true;
}

// Raw-buffer factorize-for-log-det: no solve, no Rcpp alloc.
inline void dense_cholesky_log_det_raw(
    const DenseMat& H, int n,
    DenseCholeskyScratch& scratch,
    double& log_det_out
) {
    scratch.ensure(n);
    detail::cholesky_factorize_impl_raw(H, n, scratch.L.data(), log_det_out);
}

// Extract a symmetric DenseMat into raw CSC lower-triangle arrays
// (stype = -1, dgCMatrix-compatible). Entries with |value| <= drop_tol are
// dropped; the diagonal is always retained even when zero so downstream
// CHOLMOD analyzes a structurally complete pattern.
inline void dense_to_csc_lower_drop_raw(
    const DenseMat& H, int n, double drop_tol,
    std::vector<int>& csc_p,
    std::vector<int>& csc_i,
    std::vector<double>& csc_x
) {
    csc_p.assign(n + 1, 0);
    csc_i.clear();
    csc_x.clear();
    for (int c = 0; c < n; ++c) {
        csc_p[c] = static_cast<int>(csc_i.size());
        // Diagonal first, always retained.
        csc_i.push_back(c);
        csc_x.push_back(H[c][c]);
        for (int r = c + 1; r < n; ++r) {
            double v = H[r][c];
            if (std::fabs(v) > drop_tol) {
                csc_i.push_back(r);
                csc_x.push_back(v);
            }
        }
    }
    csc_p[n] = static_cast<int>(csc_i.size());
}

} // namespace tulpa

#endif // TULPA_LAPLACE_CHOLESKY_H
