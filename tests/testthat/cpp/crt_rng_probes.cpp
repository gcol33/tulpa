// Test probes over src/crt_rng.cpp, compiled against the engine by load_cpp_fixture().

#include <RcppEigen.h>
#include "crt_rng.h"
#include "tulpa_stop.h"
#include "laplace_likelihoods.h"   // polygamma_rising_difference
#include <Rcpp.h>
#include <cmath>

// [[Rcpp::export]]
Rcpp::IntegerVector cpp_sample_crt(Rcpp::IntegerVector y, double r) {
  return tulpa::sample_crt_vec(y, r);
}

// [[Rcpp::export]]
int cpp_sample_crt_sum(Rcpp::IntegerVector y, double r) {
  return tulpa::sample_crt_sum(y, r);
}
