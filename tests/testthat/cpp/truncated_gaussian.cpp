// truncated_gaussian.cpp
// Internal Rcpp entry exposing the truncated Gaussian (truncated lognormal on
// the natural scale) per-observation kernel for unit / FD-gradient tests. Not user-facing; the family is consumed through
// tulpa_nested_laplace_joint() (family = "truncated_gaussian") and tulpaObs's
// cover(positive = "lognormal_trunc")..

#include "laplace_family_link.h"
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::NumericVector cpp_truncated_gaussian_terms(double y, double upper,
                                                 double eta, double sigma,
                                                 double lower = NA_REAL) {
    // NA (the default) leaves the lower side open.
    const tulpa::TruncatedGaussian r = tulpa::truncated_gaussian_core(
        y, ISNAN(lower) ? R_NegInf : lower, upper, eta, sigma);
    return Rcpp::NumericVector::create(
        Rcpp::Named("ll")       = r.ll,
        Rcpp::Named("grad")     = r.grad,
        Rcpp::Named("neg_hess") = r.neg_hess);
}
