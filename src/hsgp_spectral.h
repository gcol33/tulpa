// hsgp_spectral.h
// Single source of truth for the squared-exponential spectral density the
// Hilbert-space GP bases evaluate.
//
// In D dimensions the squared-exponential spectral density at angular
// frequency omega is
//
//   S(omega) = sigma^2 (2 pi l^2)^{D/2} exp(-l^2 omega^2 / 2)
//
// TULPA'S HSGP BASES ARE TWO-DIMENSIONAL, so the prefactor written out is
// (2 pi l^2)^{2/2} = 2 pi l^2 and there is no dimension argument here. That is
// a layout constraint rather than a default: an HSGP basis and its eigenvalues
// are built from coordinates stored at a fixed 2-D stride, which is why
// `spatial_gp(approx = "hsgp")` and every sampler path take exactly two
// coordinate columns and `tulpa_linalg::require_coords_2col()` refuses anything
// else at the boundary. A 1-D or 3-D eigenvalue set reaching this function
// would get the wrong normalizing constant, so sigma^2 would stop meaning the
// marginal variance and the PC prior on it would be calibrated against the
// wrong quantity.
//
// `omega_sq` is the basis eigenvalue, i.e. omega^2 already squared.

#ifndef TULPA_HSGP_SPECTRAL_H
#define TULPA_HSGP_SPECTRAL_H

#include <cmath>
#include "autodiff_utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {
namespace priors {

template <typename T>
inline T hsgp_spectral_density_2d(const T& sigma2, const T& lengthscale,
                                  double omega_sq) {
    return sigma2 * T(2.0 * M_PI) * lengthscale * lengthscale
        * tulpa::math::safe_exp(T(-0.5) * lengthscale * lengthscale
                                * T(omega_sq));
}

}  // namespace priors
}  // namespace tulpa

#endif  // TULPA_HSGP_SPECTRAL_H
