// cov_kernel_equiv_export.cpp
// Probes for the isotropic covariance kernel and its phi-derivative
// (A5).
//
// dcov_dphi was written twice -- tulpa_svc::dcov_dphi_svc and a copy in
// hmc_gp_gradients.h -- and the copies had drifted: the GP copy returned half
// the true Gaussian derivative (it dropped the factor of 2 in k*2*d^2/phi^3),
// and neither implemented SPHERICAL, so a spherical fit computed its covariance
// from one kernel and its gradient from another (the exponential). Both kernels
// are reachable: spatial_gp() advertises cov = "gaussian" and "spherical".
//
// The GP copy now delegates to the canonical one. These exports let
// test-cov-kernel.R check every cov_type's derivative against a numerical
// derivative of the value function it is supposed to differentiate, which pins
// both defects and any future drift.

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
