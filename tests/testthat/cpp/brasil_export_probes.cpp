// Test probes over src/brasil_export.cpp, compiled against the engine by load_cpp_fixture().

#include <RcppEigen.h>
#include "tulpa_stop.h"
#include "brasil.h"

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
