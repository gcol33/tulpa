// spde_matern_map.h
// The Whittle-Matern axis map in the operator's own log coordinates.
//
// For the d = 2 SPDE the marginal variance is
//
//   sigma^2 = Gamma(nu) / (Gamma(nu + 1) (4 pi) kappa^(2 nu) tau^2)
//           = 1 / (4 pi nu kappa^(2 nu) tau^2)
//
// (Lindgren, Rue & Lindstrom 2011, using Gamma(nu + 1) = nu Gamma(nu)), and the
// empirical range is range = sqrt(8 nu) / kappa. So nu enters BOTH coordinates:
//
//   log_range = 0.5 log(8 nu) - log_kappa
//   log_sigma = -0.5 log(4 pi nu) - nu log_kappa - log_tau
//
// At nu = 1 the sigma line collapses to -0.5 log(4 pi) - log_kappa - log_tau.
//
// This is the inverse of spde_range_sigma_to_kappa_tau() (spde_qbuilder.h) and
// of .spde_range_sigma() (R/marginal_se_spatial.R); the three must agree formula
// for formula. It is templated so the joint-NUTS hyper prior can evaluate it on
// an autodiff scalar and the sampler's draw loop on a double.

#ifndef TULPA_SPDE_MATERN_MAP_H
#define TULPA_SPDE_MATERN_MAP_H

#include <cmath>

namespace tulpa {

// Compile-time-stable pi (no <cmath> M_PI dependency on MSVC).
constexpr double SPDE_PI = 3.14159265358979323846;

template <typename T>
inline void spde_log_kappa_tau_to_log_range_sigma(
    const T& log_kappa, const T& log_tau, double nu,
    T& log_range, T& log_sigma
) {
    log_range = T(0.5 * std::log(8.0 * nu)) - log_kappa;
    log_sigma = T(-0.5 * std::log(4.0 * SPDE_PI * nu))
              - T(nu) * log_kappa - log_tau;
}

} // namespace tulpa

#endif // TULPA_SPDE_MATERN_MAP_H
