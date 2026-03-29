// crt_rng.cpp
// Chinese Restaurant Table (CRT) distribution sampler implementation
// Reference: Zhou et al. (2012) "Lognormal and Gamma Mixed Negative Binomial Regression" ICML

#include "crt_rng.h"
#include <Rcpp.h>
#include <cmath>

using namespace Rcpp;

namespace tulpa {

// Sample L ~ CRT(y, r) using sequential Bernoulli trials
// This is the standard O(y) algorithm from Zhou et al. (2012)
//
// The algorithm simulates the Chinese Restaurant Process:
// For customer i = 1, ..., y:
//   - Customer i starts a new table with probability r / (r + i - 1)
//   - Otherwise joins an existing table
// L = total number of tables created
int sample_crt(int y, double r) {
  if (y <= 0) return 0;
  if (r <= 0.0) return 0;  // Degenerate case: all at one table

  int L = 0;

  for (int i = 1; i <= y; i++) {
    // Probability that customer i starts a new table
    double p_new_table = r / (r + static_cast<double>(i) - 1.0);

    // Bernoulli trial
    if (R::runif(0.0, 1.0) < p_new_table) {
      L++;
    }
  }

  return L;
}

// Vectorized CRT sampler (scalar r)
IntegerVector sample_crt_vec(IntegerVector y, double r) {
  int n = y.size();
  IntegerVector L(n);

  for (int j = 0; j < n; j++) {
    L[j] = sample_crt(y[j], r);
  }

  return L;
}

// Vectorized CRT sampler (vector r)
IntegerVector sample_crt_vec2(IntegerVector y, NumericVector r) {
  int n = y.size();
  if (r.size() != n) {
    stop("y and r must have the same length");
  }

  IntegerVector L(n);

  for (int j = 0; j < n; j++) {
    L[j] = sample_crt(y[j], r[j]);
  }

  return L;
}

// Expected value of CRT(y, r)
// E[L] = r * (psi(y + r) - psi(r))
// where psi is the digamma function
double crt_mean(int y, double r) {
  if (y <= 0 || r <= 0.0) return 0.0;

  // Use R's digamma function
  return r * (R::digamma(y + r) - R::digamma(r));
}

// Sum of CRT draws (sufficient statistic for dispersion update)
int sample_crt_sum(IntegerVector y, double r) {
  int n = y.size();
  int L_total = 0;

  for (int j = 0; j < n; j++) {
    L_total += sample_crt(y[j], r);
  }

  return L_total;
}

} // namespace tulpa

// ---------------------------------------------------------------------
// R exports
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::IntegerVector cpp_sample_crt(Rcpp::IntegerVector y, double r) {
  return tulpa::sample_crt_vec(y, r);
}

// [[Rcpp::export]]
int cpp_sample_crt_sum(Rcpp::IntegerVector y, double r) {
  return tulpa::sample_crt_sum(y, r);
}

// [[Rcpp::export]]
double cpp_crt_mean(int y, double r) {
  return tulpa::crt_mean(y, r);
}
