// interval_gaussian_export.cpp
// Internal Rcpp entry exposing the interval-censored Gaussian (ordered-probit
// with known thresholds) per-observation kernel for unit / FD-gradient tests.
// Not user-facing; the family is consumed through tulpa_nested_laplace_joint()
// (family = "interval_gaussian") and tulpaObs's cover(positive = "ordinal").

#include "laplace_family_link.h"
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::NumericVector cpp_interval_gaussian_terms(double lower, double upper,
                                                double eta, double sigma) {
    const tulpa::IntervalGaussian r =
        tulpa::interval_gaussian_core(lower, upper, eta, sigma);
    return Rcpp::NumericVector::create(
        Rcpp::Named("ll")       = r.ll,
        Rcpp::Named("grad")     = r.grad,
        Rcpp::Named("neg_hess") = r.neg_hess);
}
