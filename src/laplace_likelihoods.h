// laplace_likelihoods.h
// Canonical likelihood functions used by the Laplace approximation engine.

#ifndef TULPA_LAPLACE_LIKELIHOODS_H
#define TULPA_LAPLACE_LIKELIHOODS_H

namespace tulpa {

double log_lik_gaussian(double y, double eta, double phi);
double grad_log_lik_gaussian(double y, double eta, double phi);
double neg_hess_log_lik_gaussian(double y, double eta, double phi);

double log_lik_binomial(int y, int n, double eta);
double grad_log_lik_binomial(int y, int n, double eta);
double neg_hess_log_lik_binomial(int y, int n, double eta);

double log_lik_negbin(int y, double eta, double phi);
double grad_log_lik_negbin(int y, double eta, double phi);
double neg_hess_log_lik_negbin(int y, double eta, double phi);

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
