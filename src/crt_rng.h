// crt_rng.h
// Chinese Restaurant Table (CRT) distribution sampler
// For Negative Binomial Pólya-Gamma augmentation
// Reference: Zhou et al. (2012) "Lognormal and Gamma Mixed Negative Binomial Regression" ICML

#ifndef TULPA_CRT_RNG_H
#define TULPA_CRT_RNG_H

#include <Rcpp.h>

namespace tulpa {

// ---------------------------------------------------------------------
// Chinese Restaurant Table Distribution
// ---------------------------------------------------------------------
//
// The CRT distribution arises from the Chinese Restaurant Process (CRP).
// When y customers are seated sequentially with concentration parameter r,
// each new customer either:
//   - Joins an existing table with probability proportional to (number at table)
//   - Starts a new table with probability proportional to r
//
// L ~ CRT(y, r) counts the total number of tables.
//
// PMF: P(L = l | y, r) = Gamma(r) / Gamma(y + r) * |s(y, l)| * r^l
// where |s(y, l)| are unsigned Stirling numbers of the first kind.
//
// Properties:
//   - E[L | y, r] = r * (psi(y + r) - psi(r)) where psi is digamma
//   - E[L | y, r] ≈ r * log(1 + y/r) for large y
//   - When y = 0, L = 0 deterministically
//   - When r → ∞, L → y (every customer gets own table)
//   - When r → 0, L → 1{y > 0} (all customers at one table)
//
// ---------------------------------------------------------------------

// Sample L ~ CRT(y, r) using sequential Bernoulli trials
// O(y) algorithm
// @param y Non-negative integer (number of "customers")
// @param r Positive real (concentration parameter)
// @return Number of "tables" L in {0, 1, ..., y}
int sample_crt(int y, double r);

// Vectorized CRT sampler
// @param y Integer vector of counts
// @param r Concentration parameter (scalar, shared across all)
// @return Integer vector of table counts
Rcpp::IntegerVector sample_crt_vec(Rcpp::IntegerVector y, double r);

// Vectorized CRT sampler with observation-specific r
// @param y Integer vector of counts
// @param r Numeric vector of concentration parameters
// @return Integer vector of table counts
Rcpp::IntegerVector sample_crt_vec2(Rcpp::IntegerVector y, Rcpp::NumericVector r);

// ---------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------

// Expected value of CRT(y, r): r * (psi(y + r) - psi(r))
double crt_mean(int y, double r);

// Sum of CRT draws (sufficient statistic for r update)
// Returns sum(L_i) where each L_i ~ CRT(y_i, r)
int sample_crt_sum(Rcpp::IntegerVector y, double r);

} // namespace tulpa

#endif // TULPA_CRT_RNG_H
