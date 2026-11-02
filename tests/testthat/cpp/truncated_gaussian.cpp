// truncated_gaussian_export.cpp
// Internal Rcpp entry exposing the upper-truncated Gaussian (upper-truncated
// lognormal on the natural scale) per-observation kernel for unit / FD-gradient
// tests. Not user-facing; the family is consumed through
// tulpa_nested_laplace_joint() (family = "truncated_gaussian") and tulpaObs's
// cover(positive = "lognormal_trunc")..

#include "laplace_family_link.h"
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::NumericVector cpp_truncated_gaussian_terms(double y, double upper,
                                                 double eta, double sigma) {
    const tulpa::TruncatedGaussian r =
        tulpa::truncated_gaussian_core(y, upper, eta, sigma);
    return Rcpp::NumericVector::create(
        Rcpp::Named("ll")       = r.ll,
        Rcpp::Named("grad")     = r.grad,
        Rcpp::Named("neg_hess") = r.neg_hess);
}
