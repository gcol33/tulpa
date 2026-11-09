// laplace_likelihoods.cpp
// Canonical likelihood functions used by the Laplace approximation engine.

#include "laplace_likelihoods.h"
#include "linalg_fast.h"
#include "autodiff_utils.h"  // tulpa::math::log_lik_negbin_log
#include <Rcpp.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

double grad_log_lik_gaussian(double y, double eta, double phi) {
  return (y - eta) / (phi * phi);
}

double neg_hess_log_lik_gaussian(double y, double eta, double phi) {
  return 1.0 / (phi * phi);
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

double log_lik_negbin(int y, double eta, double phi) {
  return tulpa::math::log_lik_negbin_log(
      y, tulpa::math::clamp_exp_arg(eta), std::log(phi));
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

} // namespace tulpa

// Elementwise gamma or beta density and precision derivatives.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_precision_density(const Rcpp::NumericVector& y,
    const Rcpp::NumericVector& mu, const Rcpp::NumericVector& phi,
    bool beta, int order = 0) {
  const R_xlen_t n = y.size();
  if (mu.size() != n || phi.size() != n || order < 0 || order > 2)
    Rcpp::stop("Equal input lengths and derivative order 0, 1 or 2 are required.");
  Rcpp::NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    if (ISNAN(y[i]) || ISNAN(mu[i]) || ISNAN(phi[i])) out[i] = NA_REAL;
    else if (order == 0) out[i] = beta ? tulpa::log_lik_beta(y[i], mu[i], phi[i])
                                     : tulpa::log_lik_gamma(y[i], mu[i], phi[i]);
    else out[i] = beta ? tulpa::math::beta_phi_derivative(y[i], mu[i], phi[i], order)
                       : tulpa::math::gamma_phi_derivative(y[i], mu[i], phi[i], order);
  }
  return out;
}

// Beta mean score or its precision derivative on the logit scale.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_beta_precision_score(const Rcpp::NumericVector& y,
    const Rcpp::NumericVector& mu, const Rcpp::NumericVector& phi, bool derivative) {
  if (mu.size() != y.size() || phi.size() != y.size()) Rcpp::stop("Equal input lengths are required.");
  Rcpp::NumericVector out(y.size());
  for (R_xlen_t i = 0; i < y.size(); ++i)
    out[i] = ISNAN(y[i]) || ISNAN(mu[i]) || ISNAN(phi[i]) ? NA_REAL : derivative
      ? tulpa::math::beta_score_phi_derivative(y[i], mu[i], phi[i])
      : tulpa::math::beta_mu_score(y[i], mu[i], phi[i]);
  return out;
}

// Stirling remainder derivatives or d - log(1+d).
// [[Rcpp::export]]
Rcpp::NumericVector cpp_precision_math(const Rcpp::NumericVector& x,
                                       bool remainder, int order = 0) {
  if (order < 0 || order > 4) Rcpp::stop("Derivative order must be in 0..4.");
  Rcpp::NumericVector out(x.size());
  for (R_xlen_t i = 0; i < x.size(); ++i)
    out[i] = ISNAN(x[i]) ? NA_REAL : remainder
      ? tulpa::math::stirling_remainder_derivative(x[i], order)
      : tulpa::math::log1pmx(x[i]);
  return out;
}

// Elementwise log Gamma(y + size) - log Gamma(size) over equal-length count
// and size vectors (NA where either is NA).
// [[Rcpp::export]]
Rcpp::NumericVector cpp_log_rising_factorial(const Rcpp::NumericVector& y,
                                             const Rcpp::NumericVector& size) {
  const R_xlen_t n = y.size();
  if (size.size() != n) Rcpp::stop("`y` and `size` must have the same length.");
  Rcpp::NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    out[i] = (ISNAN(y[i]) || ISNAN(size[i])) ? NA_REAL
           : tulpa::log_rising_factorial((int) y[i], size[i]);
  }
  return out;
}

// Elementwise psi^(k)(y + size) - psi^(k)(size), k in 0..3, over equal-length
// count and size vectors (NA where either is NA).
// [[Rcpp::export]]
Rcpp::NumericVector cpp_polygamma_rising_difference(int k,
                                                    const Rcpp::NumericVector& y,
                                                    const Rcpp::NumericVector& size) {
  if (k < 0 || k > 3) Rcpp::stop("`k` must be 0, 1, 2 or 3.");
  const R_xlen_t n = y.size();
  if (size.size() != n) Rcpp::stop("`y` and `size` must have the same length.");
  Rcpp::NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    out[i] = (ISNAN(y[i]) || ISNAN(size[i])) ? NA_REAL
           : tulpa::polygamma_rising_difference(k, (int) y[i], size[i]);
  }
  return out;
}

// Elementwise negative binomial (type 2) log density at mean mu and size phi
// over equal-length vectors (NA where any input is NA).
// [[Rcpp::export]]
Rcpp::NumericVector cpp_negbin2_log_density(const Rcpp::NumericVector& y,
                                            const Rcpp::NumericVector& mu,
                                            const Rcpp::NumericVector& phi) {
  const R_xlen_t n = y.size();
  if (mu.size() != n || phi.size() != n)
    Rcpp::stop("`y`, `mu` and `phi` must have the same length.");
  Rcpp::NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    out[i] = (ISNAN(y[i]) || ISNAN(mu[i]) || ISNAN(phi[i])) ? NA_REAL
           : tulpa::log_lik_negbin((int) y[i], std::log(mu[i]), phi[i]);
  }
  return out;
}
