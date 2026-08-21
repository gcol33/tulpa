// pg_binomial.h
// Pólya-Gamma Gibbs sampler for binomial models with random effects

#ifndef TULPA_PG_BINOMIAL_H
#define TULPA_PG_BINOMIAL_H

#include <Rcpp.h>
#include <vector>

namespace tulpa {

// ---------------------------------------------------------------------
// Helper functions for blocked Gibbs updates
// ---------------------------------------------------------------------

// Update beta (fixed effects) given omega and random effects
// Uses normal-normal conjugacy after PG augmentation
Rcpp::NumericVector update_beta(
    const Rcpp::NumericVector& kappa,     // y - n/2
    const Rcpp::NumericVector& omega,     // PG draws
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& re_contrib, // Z * b
    double prior_sd
);

// Update random effects given omega and beta
// Uses normal-normal conjugacy
Rcpp::NumericVector update_re(
    const Rcpp::NumericVector& kappa,
    const Rcpp::NumericVector& omega,
    const Rcpp::NumericVector& X_beta,    // X * beta
    const Rcpp::IntegerVector& group,
    int n_groups,
    double sigma_re
);

} // namespace tulpa

#endif // TULPA_PG_BINOMIAL_H
