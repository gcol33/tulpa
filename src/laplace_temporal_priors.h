// laplace_temporal_priors.h
// Temporal prior utilities and precision interfaces for Laplace solvers.
//
// Each precision (RW1/RW2/AR1) is exposed in two scatter flavors:
//   * Dense:  `add_*_precision`         -> DenseMat& H  (legacy fallback)
//   * Sparse: `add_*_precision_sparse`  -> SparseHessianBuilder& H  (scale path)
// Both must agree numerically. Pattern enumeration is provided alongside.

#ifndef TULPA_LAPLACE_TEMPORAL_PRIORS_H
#define TULPA_LAPLACE_TEMPORAL_PRIORS_H

#include "laplace_types.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

// Forward decl — full type in sparse_hessian.h, only needed in the .cpp.
class SparseHessianBuilder;

inline double log_prior_rw1(
    const Rcpp::NumericVector& x, int start_idx, int n_times,
    double tau, bool cyclic
) {
    if (n_times < 2) return 0.0;
    double quad = 0.0;
    for (int t = 1; t < n_times; t++) {
        double d = x[start_idx + t] - x[start_idx + t - 1];
        quad += d * d;
    }
    int rank = n_times - 1;
    if (cyclic) {
        double d = x[start_idx] - x[start_idx + n_times - 1];
        quad += d * d;
        rank = n_times;
    }
    return 0.5 * rank * std::log(tau / (2.0 * M_PI)) - 0.5 * tau * quad;
}

inline double log_prior_rw2(
    const Rcpp::NumericVector& x, int start_idx, int n_times,
    double tau, bool /*cyclic*/
) {
    if (n_times < 3) return 0.0;
    double quad = 0.0;
    for (int t = 1; t < n_times - 1; t++) {
        double d = x[start_idx + t - 1] - 2.0 * x[start_idx + t] + x[start_idx + t + 1];
        quad += d * d;
    }
    int rank = n_times - 2;
    return 0.5 * rank * std::log(tau / (2.0 * M_PI)) - 0.5 * tau * quad;
}

inline double log_prior_ar1(
    const Rcpp::NumericVector& x, int start_idx, int n_times,
    double tau, double rho
) {
    if (n_times < 1) return 0.0;
    double rho2 = rho * rho;
    double quad = (1.0 - rho2) * x[start_idx] * x[start_idx];
    for (int t = 1; t < n_times; t++) {
        double r = x[start_idx + t] - rho * x[start_idx + t - 1];
        quad += r * r;
    }
    return 0.5 * n_times * std::log(tau / (2.0 * M_PI))
           + 0.5 * std::log(std::max(1.0 - rho2, 1e-15))
           - 0.5 * tau * quad;
}

void add_rw1_precision(
    DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x,
    int start_idx, int n_times, double tau, bool cyclic
);

void add_rw1_precision_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const Rcpp::NumericVector& x,
    int start_idx, int n_times, double tau, bool cyclic
);

void add_rw2_precision(
    DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x,
    int start_idx, int n_times, double tau, bool cyclic
);

void add_rw2_precision_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const Rcpp::NumericVector& x,
    int start_idx, int n_times, double tau, bool cyclic
);

void add_ar1_precision(
    DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x,
    int start_idx, int n_times, double tau, double rho
);

void add_ar1_precision_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const Rcpp::NumericVector& x,
    int start_idx, int n_times, double tau, double rho
);

// Append (row, col) lower-triangle entries the RW1 / RW2 / AR1 precision
// contributes to the joint H sparsity pattern. Diagonal entries are omitted
// (pattern builder adds them unconditionally).
void add_rw1_pattern(
    std::vector<std::pair<int,int>>& out,
    int start_idx, int n_times, bool cyclic
);

void add_rw2_pattern(
    std::vector<std::pair<int,int>>& out,
    int start_idx, int n_times, bool cyclic
);

void add_ar1_pattern(
    std::vector<std::pair<int,int>>& out,
    int start_idx, int n_times
);

} // namespace tulpa

#endif // TULPA_LAPLACE_TEMPORAL_PRIORS_H
