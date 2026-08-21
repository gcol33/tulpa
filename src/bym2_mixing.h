// bym2_mixing.h
// The BYM2 mixing weight rho splits a field's amplitude between its structured
// (ICAR) part and its unstructured (IID) part:
//
//   eta_s = sigma * ( sqrt(rho) * scale_factor * phi_s + sqrt(1 - rho) * theta_s )
//
// Both square roots are evaluated at every outer-grid cell and at every
// Metropolis proposal, so rho reaching an endpoint has to leave them finite.
// BYM2_RHO_EPS is the shift that does it, and it is one value for the Laplace
// grid kernels and the Polya-Gamma samplers alike.
//
// The shift is not free: at rho = 0 the structured amplitude is
// sqrt(BYM2_RHO_EPS) * sigma * scale_factor -- 1e-5 * sigma * scale_factor --
// rather than 0, and symmetrically for the unstructured amplitude at rho = 1.
//
// rho outside [0, 1] is rejected at the entry points rather than shifted into
// range: 1 - rho + BYM2_RHO_EPS goes negative past 1 + BYM2_RHO_EPS and the
// square root is then NaN, which propagates through d_fac into eta and returns
// a NaN cell instead of an error.

#ifndef TULPA_BYM2_MIXING_H
#define TULPA_BYM2_MIXING_H

#include <cmath>

namespace tulpa {

constexpr double BYM2_RHO_EPS = 1e-10;

// Amplitude of the structured (ICAR) component at unit sigma, before the
// scaling factor.
inline double bym2_sd_structured(double rho) {
    return std::sqrt(rho + BYM2_RHO_EPS);
}

// Amplitude of the unstructured (IID) component at unit sigma.
inline double bym2_sd_unstructured(double rho) {
    return std::sqrt(1.0 - rho + BYM2_RHO_EPS);
}

// The same shift on the log scale, for the Beta(alpha, beta) prior on rho.
inline double bym2_log_rho(double rho) {
    return std::log(rho + BYM2_RHO_EPS);
}

inline double bym2_log1m_rho(double rho) {
    return std::log(1.0 - rho + BYM2_RHO_EPS);
}

}  // namespace tulpa

#endif  // TULPA_BYM2_MIXING_H
