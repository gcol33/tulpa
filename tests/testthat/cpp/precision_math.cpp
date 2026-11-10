#include <Rcpp.h>
#include <cmath>

static int precision_lgamma_shift = 0;
static int precision_lgamma_calls = 0;

namespace std {
inline long double precision_test_lgamma(long double x) {
    ++precision_lgamma_calls;
    const long double exact = std::lgamma(x);
    if (precision_lgamma_shift == 0) return exact;
    double result = static_cast<double>(exact);
    for (int j=0; j<std::abs(precision_lgamma_shift); ++j)
        result = std::nextafter(result, precision_lgamma_shift > 0 ? INFINITY : -INFINITY);
    return result;
}
}

#define lgamma precision_test_lgamma
#include <tulpa/precision_math.h>
#undef lgamma

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_precision_perturb(Rcpp::NumericVector x, int order) {
    Rcpp::NumericMatrix result(x.size(),3);
    precision_lgamma_calls = 0;
    for (int j=0; j<3; ++j) {
        precision_lgamma_shift = 4*(j-1);
        for (R_xlen_t i=0; i<x.size(); ++i)
            result(i,j) = tulpa::math::stirling_remainder_derivative(x[i],order);
    }
    result.attr("lgamma_calls") = precision_lgamma_calls;
    return result;
}
