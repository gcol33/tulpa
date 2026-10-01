// laplace_likelihoods.h
// Canonical likelihood functions used by the Laplace approximation engine.

#ifndef TULPA_LAPLACE_LIKELIHOODS_H
#define TULPA_LAPLACE_LIKELIHOODS_H

#include "linalg_fast.h"   // tulpa_linalg::safe_exp
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
inline double log_lik_binomial_kernel(int y, int n, double eta) {
  if (eta > 0) {
    return y * eta - n * eta - n * std::log(1.0 + std::exp(-eta));
  }
  return y * eta - n * std::log(1.0 + std::exp(eta));
}
double log_lik_binomial_const(int y, int n);
double log_lik_binomial(int y, int n, double eta);

// Inverse logit, evaluated on the side that keeps exp's argument non-positive.
inline double binomial_mean_logit(double eta) {
  if (eta > 0) {
    double exp_neg_eta = std::exp(-eta);
    return 1.0 / (1.0 + exp_neg_eta);
  }
  double exp_eta = std::exp(eta);
  return exp_eta / (1.0 + exp_eta);
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

inline double log_lik_poisson_kernel(int y, double eta) {
  return y * eta - tulpa_linalg::safe_exp(eta);
}
double log_lik_poisson_const(int y);
double log_lik_poisson(int y, double eta);
inline double grad_log_lik_poisson(int y, double eta) {
  return y - tulpa_linalg::safe_exp(eta);
}
inline double neg_hess_log_lik_poisson(int /*y*/, double eta) {
  return tulpa_linalg::safe_exp(eta);
}

} // namespace tulpa

#endif // TULPA_LAPLACE_LIKELIHOODS_H
