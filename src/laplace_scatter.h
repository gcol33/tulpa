// laplace_scatter.h
// Observation gradient/Hessian scatter interfaces for Laplace solvers.

#ifndef TULPA_LAPLACE_SCATTER_H
#define TULPA_LAPLACE_SCATTER_H

#include "laplace_types.h"
#include <Rcpp.h>
#include <string>
#include <vector>

namespace tulpa {

void scatter_obs_grad_hess_base(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups,
    const Rcpp::NumericVector& eta, const std::string& family, double phi,
    DenseVec& grad, DenseMat& H, int n_threads,
    const double* det_prob = nullptr
);

void scatter_obs_with_latent(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups,
    const Rcpp::NumericVector& eta, const std::string& family, double phi,
    const std::vector<int>& effect_idx, const std::vector<double>& d_factors,
    DenseVec& grad, DenseMat& H, int n_threads
);

} // namespace tulpa

#endif // TULPA_LAPLACE_SCATTER_H
