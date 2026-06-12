// hmc_tvc_autodiff.h
// TVC hyperparameter priors on the unconstrained (log / logit) scale.
// The TVC log-prior, quadratic-form, eta, and sum-to-zero kernels are
// templated in hmc_tvc.h (namespace tulpa_tvc) and shared by the double
// and autodiff paths.

#ifndef TULPA_HMC_TVC_AUTODIFF_H
#define TULPA_HMC_TVC_AUTODIFF_H

#include <vector>
#include <cmath>
#include "autodiff_utils.h"
#include "hmc_tvc.h"  // For TVCData and TemporalType

namespace tulpa_tvc_ad {

using tulpa_tvc::TVCData;
using tulpa_temporal::TemporalType;
using namespace tulpa::math;

// =============================================================================
// TVC hyperparameter priors
// =============================================================================

// Half-Cauchy prior on tau (precision)
// Actually this is on sigma = 1/sqrt(tau), then transformed
template<typename T>
T log_prior_tau_half_cauchy(const T& log_tau, double scale) {
    T tau = safe_exp(log_tau);
    T sigma = T(1.0) / safe_sqrt(tau);
    // Half-Cauchy: 2 / (pi * scale * (1 + (sigma/scale)^2))
    // log form + Jacobian for log_tau -> tau -> sigma
    return T(std::log(2.0 / M_PI / scale))
           - safe_log(T(1.0) + sigma * sigma / T(scale * scale))
           - T(0.5) * log_tau;  // Jacobian: d(sigma)/d(tau) * d(tau)/d(log_tau)
}

// Gamma prior on tau
template<typename T>
T log_prior_tau_gamma(const T& log_tau, double shape, double rate) {
    T tau = safe_exp(log_tau);
    // Gamma(shape, rate): tau^{shape-1} * exp(-rate*tau) * rate^shape / Gamma(shape)
    // + Jacobian for log transform
    return T(shape - 1.0) * log_tau - rate * tau + log_tau;
}

// Beta prior on rho (AR1 correlation on (-1, 1))
// Transform: u = (rho + 1) / 2, u ~ Beta(a, b)
template<typename T>
T log_prior_rho_beta(const T& logit_rho_scaled, double a, double b) {
    // logit_rho_scaled = logit((rho + 1)/2) = log((rho+1)/(1-rho))
    // rho = (2 * inv_logit(logit_rho_scaled)) - 1
    T u = inv_logit(logit_rho_scaled);
    T rho = T(2.0) * u - T(1.0);

    // Beta density on u
    T log_dens = T(a - 1.0) * safe_log(u) + T(b - 1.0) * safe_log(T(1.0) - u);
    // Jacobian for logit transform: u * (1 - u)
    log_dens = log_dens + safe_log(u) + safe_log(T(1.0) - u);

    return log_dens;
}

} // namespace tulpa_tvc_ad

#endif // TULPA_HMC_TVC_AUTODIFF_H
