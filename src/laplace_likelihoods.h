// laplace_likelihoods.h
// Canonical likelihood functions used by the Laplace approximation engine.

#ifndef TULPA_LAPLACE_LIKELIHOODS_H
#define TULPA_LAPLACE_LIKELIHOODS_H

#include "fastmath.h"      // fastmath::exp / log
#include "linalg_fast.h"   // tulpa::math::clamp_exp_arg
#include "tulpa/family_density.h"  // log_lik_gaussian
#include <cmath>

namespace tulpa {

double grad_log_lik_gaussian(double y, double eta, double phi);
double neg_hess_log_lik_gaussian(double y, double eta, double phi);

// The binomial and Poisson log-densities each split into an eta-dependent
// kernel and a per-observation constant, so a fit can evaluate the constant
// once instead of on every objective evaluation. The full
// densities below are `kernel + const` in that order, so a caller holding a
// precomputed constant reproduces them bit for bit.
//
// The eta-dependent kernels, score and Fisher weight are inline: they are
// evaluated once per observation per Newton step and per line-search trial, and
// a call into another translation unit cannot share the exp a score and its
// weight both need (laplace_family_link.h's grad_hess_for_family_core reads
// binomial_mean_logit once for both).
//
// Every binomial quantity below reads e = exp(-|eta|), taken on the side that
// keeps the argument non-positive; the `_e` forms take it precomputed, so a
// caller wanting the density and its derivatives together pays one exp. The
// binomial and Poisson kernels read fastmath::exp / log, which run the same
// code on every platform.
inline double binomial_exp_neg_abs(double eta) {
  return (eta > 0) ? fastmath::exp(-eta) : fastmath::exp(eta);
}
inline double log_lik_binomial_kernel_e(int y, int n, double eta, double e) {
  if (eta > 0) {
    return y * eta - n * eta - n * fastmath::log(1.0 + e);
  }
  return y * eta - n * fastmath::log(1.0 + e);
}
inline double log_lik_binomial_kernel(int y, int n, double eta) {
  return log_lik_binomial_kernel_e(y, n, eta, binomial_exp_neg_abs(eta));
}
double log_lik_binomial_const(int y, int n);
double log_lik_binomial(int y, int n, double eta);

// Inverse logit from e = exp(-|eta|).
inline double binomial_mean_logit_e(double eta, double e) {
  if (eta > 0) return 1.0 / (1.0 + e);
  return e / (1.0 + e);
}
inline double binomial_mean_logit(double eta) {
  return binomial_mean_logit_e(eta, binomial_exp_neg_abs(eta));
}
inline double grad_log_lik_binomial(int y, int n, double eta) {
  const double p = binomial_mean_logit(eta);
  return y - n * p;
}
inline double neg_hess_log_lik_binomial(int y, int n, double eta) {
  const double p = binomial_mean_logit(eta);
  return n * p * (1.0 - p);
}

// log Gamma(y + size) - log Gamma(size) for a count y >= 0, formed without the
// cancellation of the two lgammas at a large size (tulpa::math::lgamma_ratio_count).
// Every negative binomial normalizer in the package reads it.
double log_rising_factorial(int y, double size);
// psi^(k)(size + y) - psi^(k)(size) for k in 0..3 and a count y >= 0: the sum
// of (-1)^k k! / (size + j)^(k+1) over j < y, which is formed directly for
// y <= 32. Past that the argument is shifted to at least 10 by the same
// recurrence and the asymptotic series of psi^(k) is differenced term by term,
// each power difference as size^-q expm1(-q log1p(y / size)), so no step
// subtracts two nearly equal numbers at a large size.
double polygamma_rising_difference(int k, int y, double size);
inline double digamma_rising_difference(int y, double size) {
  return polygamma_rising_difference(0, y, size);
}
double log_lik_negbin(int y, double eta, double phi);
double grad_log_lik_negbin(int y, double eta, double phi);
double neg_hess_log_lik_negbin(int y, double eta, double phi);

// The Poisson mean exp(eta), with eta clamped as tulpa_linalg::safe_exp clamps.
inline double poisson_mean_log(double eta) {
  return fastmath::exp(tulpa::math::clamp_exp_arg(eta));
}
inline double log_lik_poisson_kernel(int y, double eta) {
  return y * eta - poisson_mean_log(eta);
}
double log_lik_poisson_const(int y);
double log_lik_poisson(int y, double eta);
inline double grad_log_lik_poisson(int y, double eta) {
  return y - poisson_mean_log(eta);
}
inline double neg_hess_log_lik_poisson(int /*y*/, double eta) {
  return poisson_mean_log(eta);
}

} // namespace tulpa

#endif // TULPA_LAPLACE_LIKELIHOODS_H
