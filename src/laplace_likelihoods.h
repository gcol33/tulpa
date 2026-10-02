// laplace_likelihoods.h
// Canonical likelihood functions used by the Laplace approximation engine.

#ifndef TULPA_LAPLACE_LIKELIHOODS_H
#define TULPA_LAPLACE_LIKELIHOODS_H

#include "fastmath.h"      // fastmath::exp / log
#include "linalg_fast.h"   // tulpa::math::clamp_exp_arg
#include <cmath>

namespace tulpa {

double log_lik_gaussian(double y, double eta, double phi);
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
