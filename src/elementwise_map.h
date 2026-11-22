// elementwise_map.h
// Maps a scalar function over equal-length Rcpp vectors, the shape every
// vectorized R entry over a scalar sampler or density takes.

#ifndef TULPA_ELEMENTWISE_MAP_H
#define TULPA_ELEMENTWISE_MAP_H

#include <Rcpp.h>
#include "tulpa_stop.h"

namespace tulpa {

// out[i] = f(v0[i], vs[i]...), in index order. Stops with `msg` when any
// vector's length differs from the first one's.
template <int RTYPE, typename F, typename V0, typename... Vs>
inline Rcpp::Vector<RTYPE> map_elementwise(const char* msg, F&& f,
                                           const V0& v0, const Vs&... vs) {
  const R_xlen_t n = v0.size();
  if (((static_cast<R_xlen_t>(vs.size()) != n) || ...)) tulpa::stop(msg);
  Rcpp::Vector<RTYPE> out(n);
  for (R_xlen_t i = 0; i < n; ++i) out[i] = f(v0[i], vs[i]...);
  return out;
}

// As map_elementwise over double inputs, with NA wherever any input is NA or
// NaN; f is not called there.
template <typename F, typename... Vs>
inline Rcpp::NumericVector map_elementwise_na(const char* msg, F&& f,
                                              const Vs&... vs) {
  return map_elementwise<REALSXP>(msg, [&f](auto... x) -> double {
    return (ISNAN(x) || ...) ? NA_REAL : f(x...);
  }, vs...);
}

} // namespace tulpa

#endif // TULPA_ELEMENTWISE_MAP_H
