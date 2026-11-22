// brasil_export.cpp
// R entry point for the compiled BRASIL rational-approximation search
// (src/brasil.h). The fractional-SPDE assembler calls this once per
// (range, sigma) cell; the R implementation in
// tests/testthat/helper-brasil-reference.R is the oracle the port is pinned
// against.

#include <RcppEigen.h>
#include "tulpa_stop.h"
#include "brasil.h"

// [[Rcpp::export]]
Rcpp::List cpp_spde_rational_roots(int order, double beta, double spectrum_ratio,
                                   double tol = 1e-7) {
  if (order < 1) {
    tulpa::stop("rational order must be >= 1.");
  }
  if (!(spectrum_ratio > 0.0 && spectrum_ratio < 1.0)) {
    tulpa::stop("spectrum_ratio must be in (0, 1).");
  }

  const tulpa::brasil::RationalRoots rr =
      tulpa::brasil::rational_roots(order, beta, spectrum_ratio, tol);

  return Rcpp::List::create(
      Rcpp::Named("rb")         = Rcpp::wrap(rr.rb),
      Rcpp::Named("rc")         = Rcpp::wrap(rr.rc),
      Rcpp::Named("scale")      = rr.scale,
      Rcpp::Named("m_beta")     = rr.m_beta,
      Rcpp::Named("beta_rem")   = rr.beta_rem,
      Rcpp::Named("error")      = rr.error,
      Rcpp::Named("deviation")  = rr.deviation,
      Rcpp::Named("converged")  = rr.converged,
      Rcpp::Named("iterations") = rr.iterations);
}

