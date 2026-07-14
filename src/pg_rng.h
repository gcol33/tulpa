// pg_rng.h
// Polya-Gamma random number generator
// Based on Polson, Scott & Windle (2013) JASA

#ifndef TULPA_PG_RNG_H
#define TULPA_PG_RNG_H

#include <Rcpp.h>
#include <cmath>

namespace tulpa {

// Sample from PG(1, z) using the exact Devroye method.
// This is the core sampler for Polya-Gamma augmentation.
double rpg1(double z);

// Sample from PG(b, z) for integer b.
// Exact sum of b Devroye draws for small/moderate b; Gaussian moment-match
// for large b.
double rpg_int(int b, double z);

// Sample from PG(b, z) for real b > 0. Integer part by rpg_int; fractional
// part by the truncated sum-of-gammas representation with a tail-mean
// correction (Polson, Scott & Windle 2013).
double rpg_real(double b, double z);

// Vectorized PG(1, z) sampler
Rcpp::NumericVector rpg1_vec(Rcpp::NumericVector z);

// Vectorized PG(b, z) sampler for integer b vector
Rcpp::NumericVector rpg_vec(Rcpp::IntegerVector b, Rcpp::NumericVector z);

// Inverse Gaussian sampler (IG(mu, lambda)).
double rinvgauss(double mu, double lambda);

// Numerically stable cosh for large arguments.
inline double cosh_safe(double x) {
  if (std::abs(x) > 500) {
    return std::exp(std::abs(x)) / 2.0;
  }
  return std::cosh(x);
}

} // namespace tulpa

#endif // TULPA_PG_RNG_H
