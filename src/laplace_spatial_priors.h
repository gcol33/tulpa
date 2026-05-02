// laplace_spatial_priors.h
// Spatial prior interfaces for Laplace solvers.

#ifndef TULPA_LAPLACE_SPATIAL_PRIORS_H
#define TULPA_LAPLACE_SPATIAL_PRIORS_H

#include "laplace_types.h"
#include <Rcpp.h>

namespace tulpa {

void add_icar_prior(
    DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x,
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

void add_car_proper_prior(
    DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x,
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

} // namespace tulpa

#endif // TULPA_LAPLACE_SPATIAL_PRIORS_H
