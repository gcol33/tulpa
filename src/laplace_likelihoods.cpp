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

double log_lik_binomial(int y, int n, double eta) {
  double log_p;
  if (eta > 0) {
    log_p = y * eta - n * eta - n * std::log(1.0 + std::exp(-eta));
  } else {
    log_p = y * eta - n * std::log(1.0 + std::exp(eta));
  }
  // lchoose is eta-independent, so it never moves the mode, the gradient, or a
  // normalized grid weight. It is kept so this is a true log-density, matching
  // dbinom(), the autodiff path and the GLMM oracle -- otherwise a binomial
  // logLik / WAIC / cross-backend comparison is off by sum(lchoose(n_i, y_i))
  // whenever n > 1. Evaluated once per fit, not inside the Newton loop.
  return log_p + R::lchoose((double) n, (double) y);
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

double log_lik_poisson(int y, double eta) {
  return y * eta - tulpa_linalg::safe_exp(eta) - R::lgammafn(y + 1.0);
}

double grad_log_lik_poisson(int y, double eta) {
  return y - tulpa_linalg::safe_exp(eta);
}

double neg_hess_log_lik_poisson(int y, double eta) {
  return tulpa_linalg::safe_exp(eta);
}

double log_lik_gamma(double y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  return phi * std::log(phi) - R::lgammafn(phi) + (phi - 1.0) * std::log(y)
         - phi * std::log(mu) - phi * y / mu;
}

double grad_log_lik_gamma(double y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  return phi * (y / mu - 1.0);
}

double neg_hess_log_lik_gamma(double y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  return phi * y / mu;
}

double log_lik_gamma_inverse(double y, double eta, double phi) {
  return phi * std::log(phi) + phi * std::log(eta) - R::lgammafn(phi)
         + (phi - 1.0) * std::log(y) - phi * y * eta;
}

double grad_log_lik_gamma_inverse(double y, double eta, double phi) {
  return phi * (1.0 / eta - y);
}

double neg_hess_log_lik_gamma_inverse(double y, double eta, double phi) {
  return phi / (eta * eta);
}

double log_lik_binomial_probit(int y, int n, double eta) {
  return y * R::pnorm(eta, 0.0, 1.0, 1, 1)
       + (n - y) * R::pnorm(-eta, 0.0, 1.0, 1, 1);
}

double grad_log_lik_binomial_probit(int y, int n, double eta) {
  double phi_eta = R::dnorm(eta, 0.0, 1.0, 0);
  double p = R::pnorm(eta, 0.0, 1.0, 1, 0);
  double q = 1.0 - p;
  if (p < 1e-15) p = 1e-15;
  if (q < 1e-15) q = 1e-15;
  return phi_eta * (y / p - (n - y) / q);
}

double neg_hess_log_lik_binomial_probit(int y, int n, double eta) {
  double phi_eta = R::dnorm(eta, 0.0, 1.0, 0);
  double p = R::pnorm(eta, 0.0, 1.0, 1, 0);
  double q = 1.0 - p;
  if (p < 1e-15) p = 1e-15;
  if (q < 1e-15) q = 1e-15;
  return n * phi_eta * phi_eta / (p * q);
}

double log_lik_binomial_cloglog(int y, int n, double eta) {
  double u = tulpa_linalg::safe_exp(eta);
  double log_q = -u;
  double log_p = std::log1p(-std::exp(log_q));
  if (log_p < -700.0) log_p = -700.0;
  return y * log_p + (n - y) * log_q;
}

double grad_log_lik_binomial_cloglog(int y, int n, double eta) {
  double u = tulpa_linalg::safe_exp(eta);
  double exp_neg_u = std::exp(-u);
  double p = 1.0 - exp_neg_u;
  if (p < 1e-15) p = 1e-15;
  double dp = u * exp_neg_u;
  return y * dp / p - (n - y) * u;
}

double neg_hess_log_lik_binomial_cloglog(int y, int n, double eta) {
  double u = tulpa_linalg::safe_exp(eta);
  double exp_neg_u = std::exp(-u);
  double p = 1.0 - exp_neg_u;
  double q = exp_neg_u;
  if (p < 1e-15) p = 1e-15;
  if (q < 1e-15) q = 1e-15;
  double dp = u * exp_neg_u;
  return n * dp * dp / (p * q);
}

double log_lik_inverse_gaussian(double y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  double resid = y - mu;
  return -0.5 * std::log(2.0 * M_PI * phi * y * y * y)
         - resid * resid / (2.0 * phi * mu * mu * y);
}

double grad_log_lik_inverse_gaussian(double y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  return (y - mu) / (phi * mu * mu);
}

double neg_hess_log_lik_inverse_gaussian(double y, double eta, double phi) {
  double mu = tulpa_linalg::safe_exp(eta);
  double fisher = 1.0 / (phi * mu);
  double observed = (2.0 * y - mu) / (phi * mu * mu);
  return (observed > fisher) ? observed : fisher;
}

} // namespace tulpa
