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

inline CholeskyResult dense_cholesky_solve(
    const DenseMat& H, const DenseVec& rhs, int n
) {
    CholeskyResult res;
    res.L = Rcpp::NumericMatrix(n, n);
    res.delta = Rcpp::NumericVector(n);
    res.log_det = 0.0;
    res.success = true;

    for (int j = 0; j < n; j++) {
        for (int k = 0; k <= j; k++) {
            double sum = H[j][k];
            for (int i = 0; i < k; i++) sum -= res.L(j, i) * res.L(k, i);
            if (j == k) {
                if (sum <= 0) sum = 1e-6;
                res.L(j, k) = std::sqrt(sum);
            } else {
                res.L(j, k) = sum / res.L(k, k);
            }
        }
    }

    for (int j = 0; j < n; j++) {
        res.log_det += std::log(res.L(j, j));
    }
    res.log_det *= 2.0;

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
    log_det = 0.0;
    for (int j = 0; j < n; j++) {
        for (int k = 0; k <= j; k++) {
            double sum = H[j][k];
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

inline void dense_cholesky_factorize(
    const Rcpp::NumericMatrix& H, int n,
    Rcpp::NumericMatrix& L, double& log_det
) {
    log_det = 0.0;
    for (int j = 0; j < n; j++) {
        for (int k = 0; k <= j; k++) {
            double sum = H(j, k);
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

} // namespace tulpa

#endif // TULPA_LAPLACE_CHOLESKY_H
