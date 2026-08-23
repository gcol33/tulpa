// portable_math_export.cpp
// Internal Rcpp entry exposing the thread-safe replacements in
// tulpa/portable_math.h so their agreement with R's own Rmath routines is
// checkable rather than assumed. Every one of these is evaluated on OpenMP
// worker threads inside the family log-likelihood and curvature ladders, where
// an Rmath domain warning would reach R's global error state from a worker; the
// replacements are what makes that path thread-safe, and this entry is what
// says they compute the same numbers.
//
// Not user-facing.

#include "tulpa/portable_math.h"
#include <Rcpp.h>
#include <string>

// [[Rcpp::export]]
Rcpp::NumericVector cpp_portable_math(std::string fn, Rcpp::NumericVector x,
                                      Rcpp::NumericVector y) {
    const R_xlen_t n = x.size();
    Rcpp::NumericVector out(n);
    const bool binary = (fn == "lchoose");
    if (binary && y.size() != n) {
        Rcpp::stop("cpp_portable_math: '%s' needs y the same length as x.",
                   fn.c_str());
    }
    for (R_xlen_t i = 0; i < n; i++) {
        const double xi = x[i];
        if (fn == "lgamma")          out[i] = tulpa::math::portable_lgamma(xi);
        else if (fn == "digamma")    out[i] = tulpa::math::portable_digamma(xi);
        else if (fn == "trigamma")   out[i] = tulpa::math::portable_trigamma(xi);
        else if (fn == "tetragamma") out[i] = tulpa::math::portable_tetragamma(xi);
        else if (fn == "pentagamma") out[i] = tulpa::math::portable_pentagamma(xi);
        else if (fn == "pnorm")      out[i] = tulpa::math::portable_pnorm(xi);
        else if (fn == "pnorm_log")  out[i] = tulpa::math::portable_pnorm_log(xi);
        else if (fn == "dnorm")      out[i] = tulpa::math::portable_dnorm(xi);
        else if (fn == "dnorm_log")  out[i] = tulpa::math::portable_dnorm_log(xi);
        else if (fn == "log1m_exp")  out[i] = tulpa::math::log1m_exp(xi);
        else if (fn == "lchoose")    out[i] = tulpa::math::portable_lchoose(xi, y[i]);
        else Rcpp::stop("cpp_portable_math: unknown function '%s'.", fn.c_str());
    }
    return out;
}
