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
    // Both the path-graph (acyclic) and ring-graph (cyclic) RW1 precisions have
    // exactly one null vector (the constant), so rank = n_times - 1 in both.
    int rank = n_times - 1;
    if (cyclic) {
        double d = x[start_idx] - x[start_idx + n_times - 1];
        quad += d * d;
    }
    return 0.5 * rank * std::log(tau / (2.0 * M_PI)) - 0.5 * tau * quad;
}

inline double log_prior_rw2(
    const Rcpp::NumericVector& x, int start_idx, int n_times,
    double tau, bool cyclic
) {
    if (n_times < 3) return 0.0;
    double quad = 0.0;
    for (int t = 1; t < n_times - 1; t++) {
        double d = x[start_idx + t - 1] - 2.0 * x[start_idx + t] + x[start_idx + t + 1];
        quad += d * d;
    }
    // On a cycle the second-difference operator annihilates only constants
    // (a linear ramp is not periodic), so rank n-1; the wrap adds the two
    // second differences centred at t = n-1 and t = 0.
    int rank = cyclic ? n_times - 1 : n_times - 2;
    if (cyclic) {
        int i0 = start_idx, iL = start_idx + n_times - 1;
        double d1 = x[iL - 1] - 2.0 * x[iL] + x[i0];      // centre n-1
        double d2 = x[iL]     - 2.0 * x[i0] + x[i0 + 1];  // centre 0
        quad += d1 * d1 + d2 * d2;
    }
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

// A whole temporal field: n_groups contiguous walks of n_times at
// [start + g*n_times, ...), sharing one tau, never connected across a group
// boundary.
//
// RW1 and RW2 are intrinsic, so the field carries a constant null direction
// that is jointly unidentified with the intercept. These entry points add the
// per-group precision AND the sum-to-zero pin that identifies it (laplace_s2z.h),
// which the per-group `add_rw*_precision` helpers below deliberately do not: the
// pin is over the field's GLOBAL constant, not per walk. The G - 1 between-group
// level contrasts stay improper by design -- the front door does not fit them --
// so this pins exactly the one direction the sampler path augments
// (tulpa_priors_temporal.h), and the two paths identify the same model.
//
// The pin is stored as S2ZStorage::Fold, so an RW chain's pattern stays
// tridiagonal (RW1) / pentadiagonal (RW2) and the block keeps the ADJACENCY
// fill kind it declares; the solver folds the rank-1 in exactly.
//
// AR1 is proper (full rank), needs no pin, and keeps the per-group helpers.
void add_rw1_field(
    DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
);

void add_rw1_field_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const Rcpp::NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
);

void add_rw1_field_pattern(
    std::vector<std::pair<int,int>>& out,
    int start, int n_groups, int n_times, bool cyclic
);

double log_prior_rw1_field(
    const Rcpp::NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
);

void add_rw2_field(
    DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
);

void add_rw2_field_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const Rcpp::NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
);

void add_rw2_field_pattern(
    std::vector<std::pair<int,int>>& out,
    int start, int n_groups, int n_times, bool cyclic
);

double log_prior_rw2_field(
    const Rcpp::NumericVector& x,
    int start, int n_groups, int n_times, double tau, bool cyclic
);

} // namespace tulpa

#endif // TULPA_LAPLACE_TEMPORAL_PRIORS_H
