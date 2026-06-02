// laplace_spatial_priors.h
// Spatial prior interfaces for Laplace solvers.
//
// Each Q(rho) family is exposed in two scatter flavors that share the same
// underlying CAR(rho) kernel:
//   * Dense:  `add_*_prior`         -> DenseMat& H  (legacy fallback)
//   * Sparse: `add_*_prior_sparse`  -> SparseHessianBuilder& H  (scale path)
// Both must agree numerically on values; the only difference is the H storage.
// `add_*_pattern` enumerates the (row, col) lower-triangle entries the prior
// contributes, called once at fit-time to seed the sparse Hessian builder.

#ifndef TULPA_LAPLACE_SPATIAL_PRIORS_H
#define TULPA_LAPLACE_SPATIAL_PRIORS_H

#include "laplace_types.h"
#include <Rcpp.h>
#include <utility>
#include <vector>

namespace tulpa {

// Forward decl. Definition in sparse_hessian.h. Only the sparse-scatter
// helpers below need the full type, and they live in the .cpp.
class SparseHessianBuilder;

void add_icar_prior(
    DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x,
    int spatial_start, int n_spatial_units, double tau_spatial,
    const Rcpp::IntegerVector& adj_row_ptr, const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
);

void add_icar_prior_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const Rcpp::NumericVector& x,
    int spatial_start, int n_spatial_units, double tau_spatial,
    const Rcpp::IntegerVector& adj_row_ptr, const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
);

double log_prior_icar(
    const Rcpp::NumericVector& x, int spatial_start, int n_spatial_units,
    double tau_spatial,
    const Rcpp::IntegerVector& adj_row_ptr, const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
);

// Structured intrinsic (ICAR, rank-deficient) field log-prior contribution,
// WITHOUT the rank-deficient normalization constant:
//   -0.5 * tau * phi'Q phi  -  0.5 * tau_s2z * (sum phi)^2
// The sum-to-zero penalty matches the one `add_icar_prior[_sparse]` fold into
// the gradient/Hessian. `log_prior_icar` is this plus the normalization; the
// BYM2 structured component calls it directly (tau = 1) so its objective is a
// single shared derivation rather than an inline copy of the quadratic form.
double log_prior_icar_structured(
    const Rcpp::NumericVector& x, int spatial_start, int n_spatial_units,
    double tau_spatial,
    const Rcpp::IntegerVector& adj_row_ptr, const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
);

void add_car_proper_prior(
    DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x,
    int spatial_start, int n_spatial_units, double tau, double rho,
    const Rcpp::IntegerVector& adj_row_ptr, const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
);

void add_car_proper_prior_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const Rcpp::NumericVector& x,
    int spatial_start, int n_spatial_units, double tau, double rho,
    const Rcpp::IntegerVector& adj_row_ptr, const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
);

double log_prior_car_proper(
    const Rcpp::NumericVector& x, int spatial_start, int n_spatial_units,
    double tau, double rho, double log_det_Q_rho,
    const Rcpp::IntegerVector& adj_row_ptr, const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
);

// Append (row, col) lower-triangle entries the CAR/ICAR adjacency contributes
// to the joint H sparsity pattern. Indices are global (`spatial_start +
// site_idx`). Diagonal entries omitted — pattern builder adds them
// unconditionally. Both ICAR and proper-CAR share this pattern because their
// Q's have identical nonzero structure (rho only changes values, not
// sparsity).
void add_car_pattern(
    std::vector<std::pair<int,int>>& out,
    int spatial_start, int n_spatial_units,
    const Rcpp::IntegerVector& adj_row_ptr, const Rcpp::IntegerVector& adj_col_idx
);

} // namespace tulpa

#endif // TULPA_LAPLACE_SPATIAL_PRIORS_H
