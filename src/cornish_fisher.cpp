#include "cornish_fisher.h"
#include <Rcpp.h>

// Skew-corrected marginal quantiles for a block of latent coordinates.
//
// `mu` / `sigma` are the Gaussian inner-Laplace marginal centre and scale of
// each coordinate, `gamma3` its leading-order Edgeworth skewness (NaN where the
// cubic term was not computable), `gamma1` the location term (NaN where it
// was not computable, which declines the coordinate rather than
// standing in for zero), and `z` the standard-normal quantiles of the requested
// probabilities. `max_abs_gamma3` bands the shape and `max_abs_centre` the
// centre gamma_1 + gamma_3 / 2; both are
// `cornish_fisher_eligible()`'s, so one predicate decides and this loop only
// adds the level-dependent monotonicity check on top of it.
// Returns the [length(mu) x length(z)] quantile matrix and, per
// coordinate, whether the skew correction was used there -- a coordinate that
// declines gets the Gaussian quantiles mu + sigma z in the same matrix, so the
// caller always has a complete table and a record of how each row was produced.
//
// [[Rcpp::export]]
Rcpp::List cpp_cornish_fisher_quantile(Rcpp::NumericVector mu,
                                       Rcpp::NumericVector sigma,
                                       Rcpp::NumericVector gamma3,
                                       Rcpp::NumericVector gamma1,
                                       Rcpp::NumericVector z,
                                       double max_abs_gamma3,
                                       double max_abs_centre) {
  const int n  = mu.size();
  const int nz = z.size();
  if (sigma.size() != n || gamma3.size() != n || gamma1.size() != n) {
    Rcpp::stop("cpp_cornish_fisher_quantile: mu, sigma, gamma3 and gamma1 must "
               "have the same length.");
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
    const double m = tulpa::cornish_fisher_center(gamma1[i], g);
    const bool use = tulpa::cornish_fisher_eligible(sigma[i], gamma1[i], g,
                                                    max_abs_gamma3,
                                                    max_abs_centre) &&
                     tulpa::cornish_fisher_monotone(g, z_lo, z_hi);
    applied[i] = use;
    for (int k = 0; k < nz; k++) {
      const double w = use ? (m + tulpa::cornish_fisher_z(z[k], g)) : z[k];
      q(i, k) = mu[i] + sigma[i] * w;
    }
  }

  return Rcpp::List::create(Rcpp::_["q"] = q, Rcpp::_["applied"] = applied);
}

// The expansion's centre and its band decision, per coordinate.
//
// The eligibility RECORD a nested fit carries (`.nl_skew_correction_attach()`)
// has to say which coordinates the bands admit before any interval level is
// requested, and it must be the same decision the quantile path then makes.
// Reading it from `cornish_fisher_in_band()` is what keeps that one predicate;
// R classifies WHY a declined coordinate declined, and never re-derives WHETHER.
//
// [[Rcpp::export]]
Rcpp::List cpp_cornish_fisher_bands(Rcpp::NumericVector gamma3,
                                    Rcpp::NumericVector gamma1,
                                    double max_abs_gamma3,
                                    double max_abs_centre) {
  const int n = gamma3.size();
  if (gamma1.size() != n) {
    Rcpp::stop("cpp_cornish_fisher_bands: gamma3 and gamma1 must have the "
               "same length.");
  }
  Rcpp::NumericVector centre(n);
  Rcpp::LogicalVector in_band(n);
  for (int i = 0; i < n; i++) {
    centre[i] = tulpa::cornish_fisher_center(gamma1[i], gamma3[i]);
    in_band[i] = tulpa::cornish_fisher_in_band(gamma1[i], gamma3[i],
                                               max_abs_gamma3, max_abs_centre);
  }
  return Rcpp::List::create(Rcpp::_["centre"] = centre,
                            Rcpp::_["in_band"] = in_band);
}
