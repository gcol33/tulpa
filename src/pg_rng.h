// pg_rng.h
// Pólya-Gamma random number generator
// Based on Polson, Scott & Windle (2013) JASA

#ifndef TULPA_PG_RNG_H
#define TULPA_PG_RNG_H

#include <Rcpp.h>
#include <cmath>

namespace tulpa {

// Sample from PG(1, z) using the Devroye method
// This is the core sampler for Pólya-Gamma augmentation
double rpg1(double z);

// Sample from PG(b, z) for integer b
// Uses sum of b independent PG(1, z) draws
double rpg_int(int b, double z);

// Vectorized PG(1, z) sampler
Rcpp::NumericVector rpg1_vec(Rcpp::NumericVector z);

// Vectorized PG(b, z) sampler for integer b vector
Rcpp::NumericVector rpg_vec(Rcpp::IntegerVector b, Rcpp::NumericVector z);

// ---------------------------------------------------------------------
// Internal helper functions
// ---------------------------------------------------------------------

// Sample from J*(1, z) - the tilted Jacobi distribution
// Used in the Devroye method for PG(1, z)
double sample_jacobi_tilted(double z);

// Compute a_n(x, t) coefficients for the alternating series
double a_coef(int n, double x, double t);

// Mass function ratio for accept/reject
double mass_ratio(double x, double z);

// Inverse Gaussian sampler (needed for large z)
double rinvgauss(double mu, double lambda);

// Truncated inverse Gaussian sampler
double rinvgauss_trunc(double mu, double lambda, double trunc);

// Exponential tilting constant
inline double cosh_safe(double x) {
  if (std::abs(x) > 500) {
    return std::exp(std::abs(x)) / 2.0;
  }
  return std::cosh(x);
}

} // namespace tulpa

#endif // QUOTR_PG_RNG_H
