// laplace_core.h
// Core Laplace approximation engine for ratiod
// Implements nested Laplace approximation for latent Gaussian models

#ifndef TULPA_LAPLACE_CORE_H
#define TULPA_LAPLACE_CORE_H

#include <Rcpp.h>
#include <vector>

namespace tulpa {

// ---------------------------------------------------------------------
// Likelihood functions for different families
// ---------------------------------------------------------------------

// Log-likelihood for Gaussian with identity link (phi = residual SD)
double log_lik_gaussian(double y, double eta, double phi);

// Gradient of Gaussian log-likelihood w.r.t. eta
double grad_log_lik_gaussian(double y, double eta, double phi);

// Negative Hessian of Gaussian log-likelihood
double neg_hess_log_lik_gaussian(double y, double eta, double phi);

// Log-likelihood for binomial with logit link
double log_lik_binomial(int y, int n, double eta);

// Gradient of binomial log-likelihood w.r.t. eta
double grad_log_lik_binomial(int y, int n, double eta);

// Negative Hessian (second derivative) of binomial log-likelihood
double neg_hess_log_lik_binomial(int y, int n, double eta);

// Log-likelihood for negative binomial with log link
double log_lik_negbin(int y, double eta, double phi);

// Gradient of negbin log-likelihood
double grad_log_lik_negbin(int y, double eta, double phi);

// Negative Hessian of negbin log-likelihood
double neg_hess_log_lik_negbin(int y, double eta, double phi);

// Log-likelihood for Poisson with log link
double log_lik_poisson(int y, double eta);

// Gradient of Poisson log-likelihood
double grad_log_lik_poisson(int y, double eta);

// Negative Hessian of Poisson log-likelihood
double neg_hess_log_lik_poisson(int y, double eta);

// Log-likelihood for Gamma with log link (phi = shape)
double log_lik_gamma(double y, double eta, double phi);
double grad_log_lik_gamma(double y, double eta, double phi);
double neg_hess_log_lik_gamma(double y, double eta, double phi);

// Log-likelihood for Gamma with inverse link (eta = 1/mu, phi = shape)
double log_lik_gamma_inverse(double y, double eta, double phi);
double grad_log_lik_gamma_inverse(double y, double eta, double phi);
double neg_hess_log_lik_gamma_inverse(double y, double eta, double phi);

// Log-likelihood for binomial with probit link
double log_lik_binomial_probit(int y, int n, double eta);
double grad_log_lik_binomial_probit(int y, int n, double eta);
double neg_hess_log_lik_binomial_probit(int y, int n, double eta);

// Log-likelihood for binomial with cloglog link
double log_lik_binomial_cloglog(int y, int n, double eta);
double grad_log_lik_binomial_cloglog(int y, int n, double eta);
double neg_hess_log_lik_binomial_cloglog(int y, int n, double eta);

// Log-likelihood for inverse Gaussian with log link (phi = dispersion = 1/lambda)
double log_lik_inverse_gaussian(double y, double eta, double phi);
double grad_log_lik_inverse_gaussian(double y, double eta, double phi);
double neg_hess_log_lik_inverse_gaussian(double y, double eta, double phi);

// ---------------------------------------------------------------------
// Laplace approximation core
// ---------------------------------------------------------------------

// Result structure for Laplace mode finding
struct LaplaceResult {
  Rcpp::NumericVector mode;     // Mode of latent field x*(theta)
  double log_det_Q;             // Log determinant of posterior precision
  double log_marginal;          // Log p(y | theta) approximation
  int n_iter;                   // Newton iterations used
  bool converged;               // Convergence flag
};

// Find the mode of p(x | y, theta) using Newton-Raphson
// For latent Gaussian models:
//   log p(x | y, theta) = sum_i log p(y_i | eta_i) - 0.5 x' Q x + const
//
// Newton update: x_new = x - H^{-1} g
// where g = gradient, H = Hessian
//
// @param y Response vector
// @param n Trials (for binomial) or NULL
// @param X Design matrix
// @param Q_diag Diagonal of prior precision (sparse representation)
// @param Q_offdiag Off-diagonal elements (for spatial/RE structure)
// @param family Distribution family
// @param phi Overdispersion (for negbin)
// @param x_init Initial values for x
// @param max_iter Maximum Newton iterations
// @param tol Convergence tolerance
LaplaceResult laplace_mode(
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& Q_diag,
    const Rcpp::List& Q_sparse,
    const std::string& family,
    double phi,
    const Rcpp::NumericVector& x_init,
    int max_iter,
    double tol
);

// Compute log marginal likelihood approximation: log p(y | theta)
// Uses Laplace approximation:
//   log p(y | theta) ≈ log p(y | x*, theta) + log p(x* | theta)
//                      - 0.5 * log |H| + (n/2) log(2π)
double log_marginal_laplace(
    const LaplaceResult& result,
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X,
    const std::string& family,
    double phi
);

// ---------------------------------------------------------------------
// Hyperparameter optimization
// ---------------------------------------------------------------------

// Optimize hyperparameters theta by maximizing log p(y | theta)
// Uses L-BFGS-B optimization
//
// @param theta_init Initial hyperparameter values
// @param theta_lower Lower bounds for theta
// @param theta_upper Upper bounds for theta
// @param y, n, X, family Model specification
// @param max_iter Maximum optimization iterations
Rcpp::List optimize_theta(
    const Rcpp::NumericVector& theta_init,
    const Rcpp::NumericVector& theta_lower,
    const Rcpp::NumericVector& theta_upper,
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X,
    const Rcpp::List& re_structure,
    const Rcpp::List& spatial_structure,
    const std::string& family,
    int max_iter
);

// ---------------------------------------------------------------------
// Posterior sampling from Laplace approximation
// ---------------------------------------------------------------------

// Sample from the Laplace approximation to p(x | y, theta*)
// where theta* is the optimized hyperparameter value
//
// This gives approximate posterior draws that can be used
// for inference and ratio computation
Rcpp::NumericMatrix sample_laplace_posterior(
    const LaplaceResult& result,
    const Rcpp::NumericVector& Q_diag,
    const Rcpp::List& Q_sparse,
    int n_samples
);

} // namespace tulpa

#endif // QUOTR_LAPLACE_CORE_H
