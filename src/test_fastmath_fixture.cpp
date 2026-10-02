// test_fastmath_fixture.cpp
// Exposes tulpa::fastmath::exp / log elementwise so testthat can compare them
// with the system libm over ordinary, extreme and special inputs.

#include "fastmath.h"
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_fastmath(Rcpp::NumericVector x, std::string fn) {
    Rcpp::NumericVector out(x.size());
    if (fn == "exp") {
        for (R_xlen_t i = 0; i < x.size(); i++) out[i] = tulpa::fastmath::exp(x[i]);
    } else if (fn == "log") {
        for (R_xlen_t i = 0; i < x.size(); i++) out[i] = tulpa::fastmath::log(x[i]);
    } else {
        Rcpp::stop("fn must be \"exp\" or \"log\"");
    }
    return out;
}
