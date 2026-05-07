// laplace_cholesky.h
// Dense Cholesky helpers used by Laplace solvers.

#ifndef TULPA_LAPLACE_CHOLESKY_H
#define TULPA_LAPLACE_CHOLESKY_H

#include "laplace_types.h"
#include <Rcpp.h>
#include <cmath>

namespace tulpa {

struct CholeskyResult {
    Rcpp::NumericMatrix L;
    Rcpp::NumericVector delta;
    double log_det;
    bool success;
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
// Singularity guard: tiny / non-positive pivots are clamped to 1e-6 (the
// Laplace solver tolerates this — Newton-Raphson inflates the Hessian
// regularization on the next outer iteration).
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
                if (sum <= 0) sum = 1e-6;
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
