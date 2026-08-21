// laplace_likelihoods.cpp
// Canonical likelihood functions used by the Laplace approximation engine.

#include "laplace_likelihoods.h"
#include "linalg_fast.h"
#include <Rcpp.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

double log_lik_gaussian(double y, double eta, double phi) {
  double resid = y - eta;
  return -0.5 * std::log(2.0 * M_PI * phi * phi) - resid * resid / (2.0 * phi * phi);
}

double grad_log_lik_gaussian(double y, double eta, double phi) {
  return (y - eta) / (phi * phi);
}

double neg_hess_log_lik_gaussian(double y, double eta, double phi) {
  return 1.0 / (phi * phi);
}

double log_lik_binomial_kernel(int y, int n, double eta) {
  if (eta > 0) {
    return y * eta - n * eta - n * std::log(1.0 + std::exp(-eta));
  }
  return y * eta - n * std::log(1.0 + std::exp(eta));
}

// lchoose is eta-independent, so it never moves the mode, the gradient, or a
// normalized grid weight. It is kept so the density below is a true
// log-density, matching dbinom(), the autodiff path and the GLMM oracle --
// otherwise a binomial logLik / WAIC / cross-backend comparison is off by
// sum(lchoose(n_i, y_i)) whenever n > 1. Three lgamma calls, so a fit
// precomputes it per observation rather than paying it per evaluation.
double log_lik_binomial_const(int y, int n) {
  return R::lchoose((double) n, (double) y);
}

double log_lik_binomial(int y, int n, double eta) {
  return log_lik_binomial_kernel(y, n, eta) + log_lik_binomial_const(y, n);
}

double grad_log_lik_binomial(int y, int n, double eta) {
  double p;
  if (eta > 0) {
    double exp_neg_eta = std::exp(-eta);
    p = 1.0 / (1.0 + exp_neg_eta);
  } else {
    double exp_eta = std::exp(eta);
    p = exp_eta / (1.0 + exp_eta);
  }
  return y - n * p;
}

double neg_hess_log_lik_binomial(int y, int n, double eta) {
  double p;
  if (eta > 0) {
    double exp_neg_eta = std::exp(-eta);
    p = 1.0 / (1.0 + exp_neg_eta);
  } else {
    double exp_eta = std::exp(eta);
    p = exp_eta / (1.0 + exp_eta);
  }
  return n * p * (1.0 - p);
}

double log_lik_negbin(int y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  double log_p = R::lgammafn(y + phi) - R::lgammafn(phi) - R::lgammafn(y + 1.0)
               + phi * std::log(phi / (mu + phi))
               + y * std::log(mu / (mu + phi));
  return log_p;
}

double grad_log_lik_negbin(int y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  double p = mu / (mu + phi);
  return y - (y + phi) * p;
}

double neg_hess_log_lik_negbin(int y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  double denom = mu + phi;
  return (y + phi) * mu * phi / (denom * denom);
}

double log_lik_poisson_kernel(int y, double eta) {
  return y * eta - tulpa_linalg::safe_exp(eta);
}

// -lgamma(y + 1), eta-independent and precomputed per observation for the same
// reason log_lik_binomial_const is. Negated here so the full density below is
// `kernel + const` like the binomial one, which is the same floating-point
// operation as the subtraction it replaces.
double log_lik_poisson_const(int y) {
  return -R::lgammafn(y + 1.0);
}

double log_lik_poisson(int y, double eta) {
  return log_lik_poisson_kernel(y, eta) + log_lik_poisson_const(y);
}

double grad_log_lik_poisson(int y, double eta) {
  return y - tulpa_linalg::safe_exp(eta);
}

double neg_hess_log_lik_poisson(int y, double eta) {
  return tulpa_linalg::safe_exp(eta);
}

} // namespace tulpa
