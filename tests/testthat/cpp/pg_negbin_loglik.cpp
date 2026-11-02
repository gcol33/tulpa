// pg_negbin_loglik.cpp
#include <Rcpp.h>
#include "pg_negbin.h"

// Scores the negative-binomial log-likelihood the Gibbs kernels evaluate, at a
// linear predictor supplied directly, so it can be held against dnbinom() over
// a range of eta the sampler never visits.
// [[Rcpp::export]]
double cpp_test_negbin_loglik_eta(
    Rcpp::IntegerVector y,
    Rcpp::NumericVector eta,
    double r
) {
  if (eta.size() != y.size()) {
    Rcpp::stop("eta holds %d values for %d observations.",
               (int) eta.size(), (int) y.size());
  }
  return tulpa::negbin_loglik_eta(y, eta, r);
}
