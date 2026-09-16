// cov_kernel_export.cpp
// Production R-side access to the canonical isotropic covariance kernel
// (inst/include/tulpa/cov_kernel.h), keyed by the tulpa::CovType code. This
// is what R-side NNGP precision rebuilds (.nngp_precision_Q()) read instead
// of restating the kernel formulas, so a cov_type code names one kernel on
// every path, R included.

#include <Rcpp.h>

#include "tulpa/cov_kernel.h"

// [[Rcpp::export]]
Rcpp::NumericVector cpp_gp_cov_value(Rcpp::NumericVector d, double sigma2,
                                      double phi, int cov_type) {
  const tulpa::CovType ct = static_cast<tulpa::CovType>(cov_type);
  Rcpp::NumericVector out(d.size());
  for (R_xlen_t i = 0; i < d.size(); ++i) {
    out[i] = tulpa::cov_value(d[i], sigma2, phi, ct);
  }
  return out;
}
