// brasil_export.cpp
// R entry point for the compiled BRASIL rational-approximation search
// (src/brasil.h). The fractional-SPDE assembler calls this once per
// (range, sigma) cell; the R implementation in R/brasil.R stays as the oracle
// the port is pinned against.

#include <RcppEigen.h>
#include "brasil.h"

// [[Rcpp::export]]
Rcpp::List cpp_spde_rational_roots(int order, double beta, double spectrum_ratio,
                                   double tol = 1e-7) {
  if (order < 1) {
    Rcpp::stop("rational order must be >= 1.");
  }
  if (!(spectrum_ratio > 0.0 && spectrum_ratio < 1.0)) {
    Rcpp::stop("spectrum_ratio must be in (0, 1).");
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

// Evaluate the fitted barycentric rational on a grid, for the equivalence test
// that pins this port against the R oracle: comparing the APPROXIMATION is what
// matters (the weights are defined up to scale and the roots up to order),
// so the test reads the function both implementations define.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_brasil_approx(const Rcpp::NumericVector& x,
                                      double a, double b, int m, double f_exp,
                                      double tol = 1e-7) {
  tulpa::brasil::BrasilResult res = tulpa::brasil::brasil(
      [f_exp](double t) { return std::pow(t, f_exp); }, a, b, m, tol);
  Rcpp::NumericVector out(x.size());
  for (R_xlen_t i = 0; i < x.size(); ++i) {
    out[i] = tulpa::brasil::bary_eval1(res.br, x[i]);
  }
  out.attr("converged") = res.converged;
  out.attr("error") = res.error;
  out.attr("deviation") = res.deviation;
  return out;
}
