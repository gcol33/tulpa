// pc_prior.h
// Penalized-complexity (PC) prior on a scale parameter. Single source of truth
// shared by the GP, temporal, spatiotemporal, TVC and HSGP hyperparameter
// priors, on every scale any of them samples.
//
// The PC prior (Simpson et al. 2017) places an exponential prior on the
// standard deviation sigma with rate lambda = -log(alpha)/U, so that
// P(sigma > U) = alpha:
//
//   log p(sigma) = log(lambda) - lambda * sigma
//
// Samplers parameterize the scale differently (sigma, variance sigma2, log
// variance, precision tau = 1/sigma^2, log precision, log sigma). Each needs
// the density on ITS scale, i.e. the base density plus the log-Jacobian of
// sigma with respect to the sampled coordinate. Deriving that chain per site is
// what produced the tulpa_priors_tvc.h (+2*log_tau) and tulpa_priors_hsgp.h
// (missing +log(sigma)) errors, so every scale is provided here and derived
// once:
//
//   scale          sigma(x)          log|dsigma/dx|
//   ------------------------------------------------------------------
//   sigma          x                 0
//   log_sigma      exp(x)            log_sigma
//   sigma2         sqrt(x)           -log(2*sigma)
//   log_sigma2     exp(x/2)          -log(2) + 0.5*log_sigma2
//   tau            1/sqrt(x)         -log(2) - 1.5*log(tau)
//   log_tau        exp(-x/2)         -log(2) - 0.5*log_tau
//
// Templated over T so the double, tape (ad::Var), forward (fwd::Dual) and arena
// (arena::Var) paths all share one implementation -- a double-only helper is
// what forced the AD sites to hand-roll copies in the first place.

#ifndef TULPA_PC_PRIOR_H
#define TULPA_PC_PRIOR_H

#include <cmath>

#include "autodiff_utils.h"

namespace tulpa {

// Exponential rate lambda calibrated so that P(sigma > U) = alpha.
inline double pc_rate(double U, double alpha) {
  return -std::log(alpha) / U;
}

// Base density, over sigma itself. Every other scale is this plus a Jacobian.
template <typename T>
inline T log_prior_sigma_pc(const T& sigma, double U, double alpha) {
  const double rate = pc_rate(U, alpha);
  return T(std::log(rate)) - T(rate) * sigma;
}

// Over log(sigma).
template <typename T>
inline T log_prior_log_sigma_pc(const T& log_sigma, double U, double alpha) {
  const T sigma = math::safe_exp(log_sigma);
  return log_prior_sigma_pc(sigma, U, alpha) + log_sigma;
}

// Over the variance sigma2.
template <typename T>
inline T log_prior_sigma2_pc(const T& sigma2, double U, double alpha) {
  const T sigma = math::safe_sqrt(sigma2);
  return log_prior_sigma_pc(sigma, U, alpha) - math::safe_log(T(2.0) * sigma);
}

// Over log(sigma2).
template <typename T>
inline T log_prior_log_sigma2_pc(const T& log_sigma2, double U, double alpha) {
  const T sigma = math::safe_exp(log_sigma2 * T(0.5));
  return log_prior_sigma_pc(sigma, U, alpha)
       - T(std::log(2.0)) + T(0.5) * log_sigma2;
}

// Over the precision tau = 1 / sigma^2.
template <typename T>
inline T log_prior_tau_pc(const T& tau, double U, double alpha) {
  const T sigma = T(1.0) / math::safe_sqrt(tau);
  return log_prior_sigma_pc(sigma, U, alpha)
       - T(std::log(2.0)) - T(1.5) * math::safe_log(tau);
}

// Over log(tau).
template <typename T>
inline T log_prior_log_tau_pc(const T& log_tau, double U, double alpha) {
  const T sigma = math::safe_exp(log_tau * T(-0.5));
  return log_prior_sigma_pc(sigma, U, alpha)
       - T(std::log(2.0)) - T(0.5) * log_tau;
}

}  // namespace tulpa

#endif  // TULPA_PC_PRIOR_H
