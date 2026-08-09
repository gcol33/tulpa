// laplace_likelihoods.h
// Canonical likelihood functions used by the Laplace approximation engine.

#ifndef TULPA_LAPLACE_LIKELIHOODS_H
#define TULPA_LAPLACE_LIKELIHOODS_H

namespace tulpa {

double log_lik_gaussian(double y, double eta, double phi);
double grad_log_lik_gaussian(double y, double eta, double phi);
double neg_hess_log_lik_gaussian(double y, double eta, double phi);

// The binomial and Poisson log-densities each split into an eta-dependent
// kernel and a per-observation constant, so a fit can evaluate the constant
// once instead of on every objective evaluation (gcol33/tulpa#372). The full
// densities below are `kernel + const` in that order, so a caller holding a
// precomputed constant reproduces them bit for bit.
double log_lik_binomial_kernel(int y, int n, double eta);
double log_lik_binomial_const(int y, int n);
double log_lik_binomial(int y, int n, double eta);
double grad_log_lik_binomial(int y, int n, double eta);
double neg_hess_log_lik_binomial(int y, int n, double eta);

double log_lik_negbin(int y, double eta, double phi);
double grad_log_lik_negbin(int y, double eta, double phi);
double neg_hess_log_lik_negbin(int y, double eta, double phi);

double log_lik_poisson_kernel(int y, double eta);
double log_lik_poisson_const(int y);
double log_lik_poisson(int y, double eta);
double grad_log_lik_poisson(int y, double eta);
double neg_hess_log_lik_poisson(int y, double eta);

double log_lik_gamma(double y, double eta, double phi);
double grad_log_lik_gamma(double y, double eta, double phi);
double neg_hess_log_lik_gamma(double y, double eta, double phi);

double log_lik_gamma_inverse(double y, double eta, double phi);
double grad_log_lik_gamma_inverse(double y, double eta, double phi);
double neg_hess_log_lik_gamma_inverse(double y, double eta, double phi);

double log_lik_binomial_probit(int y, int n, double eta);
double grad_log_lik_binomial_probit(int y, int n, double eta);
double neg_hess_log_lik_binomial_probit(int y, int n, double eta);

double log_lik_binomial_cloglog(int y, int n, double eta);
double grad_log_lik_binomial_cloglog(int y, int n, double eta);
double neg_hess_log_lik_binomial_cloglog(int y, int n, double eta);

double log_lik_inverse_gaussian(double y, double eta, double phi);
double grad_log_lik_inverse_gaussian(double y, double eta, double phi);
double neg_hess_log_lik_inverse_gaussian(double y, double eta, double phi);

} // namespace tulpa

#endif // TULPA_LAPLACE_LIKELIHOODS_H
