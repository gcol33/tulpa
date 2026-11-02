// sobol_export.cpp
// R entry point for the native Sobol' generator in sobol.h.
#include "sobol.h"
#include <Rcpp.h>

// Sobol' points 1 .. n in dimension d, as an n x d matrix in the unit cube.
// The origin (point 0) is not emitted; see the note in sobol.h.
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_sobol_points(int n, int d) {
  if (n <= 0) {
    Rcpp::stop("cpp_sobol_points: n must be positive (got %d).", n);
  }
  if (d <= 0) {
    Rcpp::stop("cpp_sobol_points: d must be positive (got %d).", d);
  }
  if (d > tulpa::SOBOL_MAX_DIM) {
    Rcpp::stop(
        "cpp_sobol_points: d = %d exceeds the tabulated maximum dimension %d.",
        d, tulpa::SOBOL_MAX_DIM);
  }

  Rcpp::NumericMatrix out(n, d);
  if (!tulpa::sobol_points(n, d, out.begin())) {
    Rcpp::stop("cpp_sobol_points: generator refused n = %d, d = %d.", n, d);
  }
  return out;
}

// The tabulated maximum dimension, so the R side never carries a second copy
// of the number.
// [[Rcpp::export]]
int cpp_sobol_max_dim() { return tulpa::SOBOL_MAX_DIM; }
