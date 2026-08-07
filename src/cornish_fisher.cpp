#include "cornish_fisher.h"
#include <Rcpp.h>

// Skew-corrected marginal quantiles for a block of latent coordinates.
//
// `mu` / `sigma` are the Gaussian inner-Laplace marginal centre and scale of
// each coordinate, `gamma3` its leading-order Edgeworth skewness (NaN where the
// cubic term was not computable), and `z` the standard-normal quantiles of the
// requested probabilities. Returns the [length(mu) x length(z)] quantile matrix
// and, per coordinate, whether the skew correction was used there -- a
// coordinate that declines gets the Gaussian quantiles mu + sigma z in the same
// matrix, so the caller always has a complete table and a record of how each
// row was produced.
//
// [[Rcpp::export]]
Rcpp::List cpp_cornish_fisher_quantile(Rcpp::NumericVector mu,
                                       Rcpp::NumericVector sigma,
                                       Rcpp::NumericVector gamma3,
                                       Rcpp::NumericVector z,
                                       double max_abs_gamma3) {
  const int n  = mu.size();
  const int nz = z.size();
  if (sigma.size() != n || gamma3.size() != n) {
    Rcpp::stop("cpp_cornish_fisher_quantile: mu, sigma and gamma3 must have "
               "the same length.");
  }

  double z_lo = 0.0, z_hi = 0.0;
  for (int k = 0; k < nz; k++) {
    if (z[k] < z_lo) z_lo = z[k];
    if (z[k] > z_hi) z_hi = z[k];
  }

  Rcpp::NumericMatrix q(n, nz);
  Rcpp::LogicalVector applied(n);

  for (int i = 0; i < n; i++) {
    const double g = gamma3[i];
    const bool use = tulpa::cornish_fisher_eligible(sigma[i], g, max_abs_gamma3) &&
                     tulpa::cornish_fisher_monotone(g, z_lo, z_hi);
    applied[i] = use;
    for (int k = 0; k < nz; k++) {
      const double w = use ? tulpa::cornish_fisher_z(z[k], g) : z[k];
      q(i, k) = mu[i] + sigma[i] * w;
    }
  }

  return Rcpp::List::create(Rcpp::_["q"] = q, Rcpp::_["applied"] = applied);
}
