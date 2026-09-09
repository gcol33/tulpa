#ifndef TULPA_COV_KERNEL_H
#define TULPA_COV_KERNEL_H

// cov_kernel.h
// The isotropic spatial covariance kernels and their phi-derivative, in one
// place, keyed by the tulpa::CovType code.
//
// Every path that turns a (distance, sigma2, phi, cov_type) into a covariance
// reads this file: the exact-NUTS NNGP/SVC kernels (hmc_svc.h and its autodiff
// twin), the Laplace NNGP neighbour scatter (gpu_nngp_laplace.h), the
// Polya-Gamma Gibbs sweep (pg_shared.h) and the field predictor
// (gp_predict.cpp). One code therefore names ONE kernel wherever it is read,
// which is what a fit and a prediction from that fit need in order to be the
// same model.
//
// phi is the RANGE in every kernel (distance enters as d / phi), which is what
// makes a PC range prior apply to it directly.

#include <cmath>

#include "tulpa/types.h"

namespace tulpa {

// Exponential: sigma^2 * exp(-d / phi)   (Matern nu = 1/2)
inline double cov_exponential(double d, double sigma2, double phi) {
  return sigma2 * std::exp(-d / phi);
}

// Matern 3/2: sigma^2 * (1 + x) * exp(-x),  x = sqrt(3) d / phi
inline double cov_matern32(double d, double sigma2, double phi) {
  double x = std::sqrt(3.0) * d / phi;
  return sigma2 * (1.0 + x) * std::exp(-x);
}

// Matern 5/2: sigma^2 * (1 + x + x^2/3) * exp(-x),  x = sqrt(5) d / phi
inline double cov_matern52(double d, double sigma2, double phi) {
  double x = std::sqrt(5.0) * d / phi;
  return sigma2 * (1.0 + x + x * x / 3.0) * std::exp(-x);
}

// Gaussian (squared exponential): sigma^2 * exp(-(d / phi)^2)
inline double cov_gaussian(double d, double sigma2, double phi) {
  double r = d / phi;
  return sigma2 * std::exp(-r * r);
}

// Spherical: compact support at d = phi
inline double cov_spherical(double d, double sigma2, double phi) {
  if (d >= phi) return 0.0;
  double r = d / phi;
  return sigma2 * (1.0 - 1.5 * r + 0.5 * r * r * r);
}

// Covariance at one distance for a CovType code.
inline double cov_value(double d, double sigma2, double phi, CovType cov_type) {
  switch (cov_type) {
    case CovType::EXPONENTIAL: return cov_exponential(d, sigma2, phi);
    case CovType::MATERN32:    return cov_matern32(d, sigma2, phi);
    case CovType::MATERN52:    return cov_matern52(d, sigma2, phi);
    case CovType::GAUSSIAN:    return cov_gaussian(d, sigma2, phi);
    case CovType::SPHERICAL:   return cov_spherical(d, sigma2, phi);
  }
  return cov_exponential(d, sigma2, phi);
}

// dk(d)/dphi for the same code. Named cov_dphi rather than dcov_dphi so it
// cannot be found by ADL where tulpa_gp::dcov_dphi is the intended overload.
//
// sigma2 is needed for SPHERICAL alone: the other kernels' derivatives are
// proportional to k(d), so they can be written from cov_val, but the spherical
// polynomial's is not, so it takes sigma2 explicitly.
inline double cov_dphi(double d, double phi, double cov_val, double sigma2,
                        CovType cov_type) {
  if (d < 1e-10) return 0.0;
  switch (cov_type) {
    case CovType::EXPONENTIAL:
      // k = s2*exp(-d/phi) -> dk/dphi = k*d/phi^2
      return cov_val * d / (phi * phi);
    case CovType::MATERN32: {
      // k = s2*(1+x)*exp(-x), x = sqrt(3)*d/phi -> dk/dphi = k*x^2/(phi*(1+x))
      double x = 1.7320508075688772 * d / phi;
      return (1.0 + x > 1e-10) ? cov_val * x * x / (phi * (1.0 + x)) : 0.0;
    }
    case CovType::MATERN52: {
      // k = s2*g*exp(-x), g = 1+x+x^2/3, x = sqrt(5)*d/phi. With
      // dk/dx = -s2*exp(-x)*x*(1+x)/3 and dx/dphi = -x/phi,
      // dk/dphi = k * x^2 * (1+x) / (3*phi*g).
      double x = 2.2360679774997898 * d / phi;
      double g = 1.0 + x + x * x / 3.0;
      return (g > 1e-10) ? cov_val * x * x * (1.0 + x) / (3.0 * phi * g) : 0.0;
    }
    case CovType::GAUSSIAN:
      // k = s2*exp(-(d/phi)^2) -> dk/dphi = k*2*d^2/phi^3
      return cov_val * 2.0 * d * d / (phi * phi * phi);
    case CovType::SPHERICAL: {
      // k = s2*(1 - 1.5r + 0.5r^3) for r = d/phi < 1, else 0 (and flat there)
      // -> dk/dphi = s2 * 1.5 * r * (1 - r^2) / phi
      if (d >= phi) return 0.0;
      double r = d / phi;
      return sigma2 * 1.5 * r * (1.0 - r * r) / phi;
    }
  }
  return cov_val * d / (phi * phi);
}

}  // namespace tulpa

#endif  // TULPA_COV_KERNEL_H
