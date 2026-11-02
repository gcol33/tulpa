// cov_kernel_equiv_export.cpp
// Probes for the isotropic covariance kernel and its phi-derivative.
//
// dcov_dphi has ONE implementation, tulpa_svc::dcov_dphi_svc, which the GP
// path delegates to, and it covers every cov_type spatial_gp() advertises --
// exponential, matern 3/2, gaussian and spherical -- so a fit takes its
// covariance and its gradient from the same kernel whichever it names.
//
// These exports let test-cov-kernel.R check every cov_type's derivative
// against a numerical derivative of the value function it is supposed to
// differentiate, which is what keeps the pair from drifting.

#include <Rcpp.h>

#include "hmc_svc.h"

// Covariance value for a cov_type code (the tulpa::CovType enum ordering:
// 0 = exponential, 1 = matern 3/2, 2 = gaussian, 3 = spherical).
// [[Rcpp::export]]
double cpp_test_compute_cov(double d, double sigma2, double phi, int cov_type) {
  return tulpa_svc::compute_cov(d, sigma2, phi,
                                static_cast<tulpa::CovType>(cov_type));
}

// dk(d)/dphi for the same code, from the canonical kernel.
// [[Rcpp::export]]
double cpp_test_dcov_dphi(double d, double sigma2, double phi, int cov_type) {
  const tulpa::CovType ct = static_cast<tulpa::CovType>(cov_type);
  const double cov_val = tulpa_svc::compute_cov(d, sigma2, phi, ct);
  return tulpa_svc::dcov_dphi_svc(d, phi, cov_val, sigma2, ct);
}
