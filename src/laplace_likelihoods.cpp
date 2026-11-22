// laplace_likelihoods.cpp
// Canonical likelihood functions used by the Laplace approximation engine.

#include "laplace_likelihoods.h"
#include "tulpa_stop.h"
#include "elementwise_map.h"
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
  return log_lik_negbin(y,eta,phi,tulpa::math::portable_lgamma(y+1.0));
}

double log_lik_negbin(int y, double eta, double phi, double log_factorial) {
  return tulpa::math::negbin_log_pmf_log(y,tulpa::math::clamp_exp_arg(eta),
                                       std::log(phi),log_factorial);
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
  const char* msg = "Equal input lengths and derivative order 0, 1 or 2 are required.";
  if (order < 0 || order > 2) tulpa::stop(msg);
  return tulpa::map_elementwise_na(msg, [beta, order](double yi, double mi, double phii) -> double {
    if (order == 0) return beta ? tulpa::log_lik_beta(yi, mi, phii)
                                : tulpa::log_lik_gamma(yi, mi, phii);
    return beta ? tulpa::math::beta_phi_derivative(yi, mi, phii, order)
                : tulpa::math::gamma_phi_derivative(yi, mi, phii, order);
  }, y, mu, phi);
}

// Beta mean score or its precision derivative on the logit scale.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_beta_precision_score(const Rcpp::NumericVector& y,
    const Rcpp::NumericVector& mu, const Rcpp::NumericVector& phi, bool derivative) {
  return tulpa::map_elementwise_na(
      "Equal input lengths are required.",
      [derivative](double yi, double mi, double phii) {
        return derivative ? tulpa::math::beta_score_phi_derivative(yi, mi, phii)
                          : tulpa::math::beta_mu_score(yi, mi, phii);
      },
      y, mu, phi);
}

// Elementwise log Gamma(y + size) - log Gamma(size) over equal-length count
// and size vectors (NA where either is NA).
// [[Rcpp::export]]
Rcpp::NumericVector cpp_log_rising_factorial(const Rcpp::NumericVector& y,
                                             const Rcpp::NumericVector& size) {
  return tulpa::map_elementwise_na(
      "`y` and `size` must have the same length.",
      [](double yi, double si) { return tulpa::log_rising_factorial((int) yi, si); },
      y, size);
}

// Elementwise psi^(k)(y + size) - psi^(k)(size), k in 0..3, over equal-length
// count and size vectors (NA where either is NA).
// [[Rcpp::export]]
Rcpp::NumericVector cpp_polygamma_rising_difference(int k,
                                                    const Rcpp::NumericVector& y,
                                                    const Rcpp::NumericVector& size) {
  if (k < 0 || k > 3) tulpa::stop("`k` must be 0, 1, 2 or 3.");
  return tulpa::map_elementwise_na(
      "`y` and `size` must have the same length.",
      [k](double yi, double si) {
        return tulpa::polygamma_rising_difference(k, (int) yi, si);
      },
      y, size);
}

// Elementwise negative binomial (type 2) log density at mean mu and size phi
// over equal-length vectors (NA where any input is NA).
// [[Rcpp::export]]
Rcpp::NumericVector cpp_negbin2_log_density(const Rcpp::NumericVector& y,
                                            const Rcpp::NumericVector& mu,
                                            const Rcpp::NumericVector& phi) {
  return tulpa::map_elementwise_na(
      "`y`, `mu` and `phi` must have the same length.",
      [](double yi, double mi, double phii) {
        return tulpa::log_lik_negbin((int) yi, std::log(mi), phii);
      },
      y, mu, phi);
}
