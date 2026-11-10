// hsgp_spectral.h
// Spectral densities for two-dimensional Hilbert-space GP bases:
// the squared-exponential kernel and the Matern kernels of
// smoothness 3/2 and 5/2 (Riutort-Mayol et al. 2023, Statistics and Computing
// 33:17, Eqs. 1-3).
//
// In D dimensions, at angular frequency omega with |omega|^2 = omega_sq,
//
//   squared exponential: S(omega) = sigma^2 (2 pi l^2)^{D/2}
//                                   exp(-l^2 omega_sq / 2)
//   Matern nu:           S(omega) = sigma^2 2^D pi^{D/2} Gamma(nu + D/2)
//                                   (2 nu)^nu / (Gamma(nu) l^{2 nu})
//                                   (2 nu / l^2 + omega_sq)^{-(nu + D/2)}
//
// the Fourier duals of k(r) = sigma^2 exp(-r^2 / (2 l^2)) and of the Matern
// kernels sigma^2 (1 + sqrt(3) r / l) exp(-sqrt(3) r / l) (nu = 3/2) and
// sigma^2 (1 + sqrt(5) r / l + 5 r^2 / (3 l^2)) exp(-sqrt(5) r / l)
// (nu = 5/2). Each integrates to (2 pi)^D sigma^2, so sigma^2 is the marginal
// variance whatever the kernel.
//
// At D = 2 the spectra are:
//
//   squared exponential: 2 pi l^2 exp(-l^2 omega_sq / 2)
//   Matern 3/2:          6 pi 3^{3/2} l^{-3} (3 / l^2 + omega_sq)^{-5/2}
//   Matern 5/2:          10 pi 5^{5/2} l^{-5} (5 / l^2 + omega_sq)^{-7/2}
//
// (times sigma^2). The fixed dimension follows the basis layout: an HSGP
// basis and its eigenvalues are built from coordinates stored at a fixed 2-D
// stride, which is why every HSGP door takes exactly two coordinate columns and
// `tulpa_linalg::require_coords_2col()` refuses anything else at the boundary.
// A 1-D or 3-D eigenvalue set reaching this function would get the wrong
// normalizing constant, so sigma^2 would stop meaning the marginal variance and
// the PC prior on it would be calibrated against the wrong quantity.
//
// `omega_sq` is the basis eigenvalue, i.e. omega^2 already squared. The kernel
// travels as the integer code `HSGPData::kernel` stores and the R side writes
// (`.HSGP_KERNELS`, R/hsgp_basis.R).

#ifndef TULPA_HSGP_SPECTRAL_H
#define TULPA_HSGP_SPECTRAL_H

#include <cmath>
#include <Rcpp.h>
#include "autodiff_utils.h"
#include <tulpa/hsgp_spectral.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {
namespace priors {

enum class HsgpKernel : int {
    SQUARED_EXPONENTIAL = 0,
    MATERN32 = 1,
    MATERN52 = 2
};

// The kernel an integer code names; an unknown code is refused at the
// boundary that reads it, never inside a density.
inline HsgpKernel hsgp_kernel_from_code(int code, const char* who) {
    switch (code) {
    case 0: return HsgpKernel::SQUARED_EXPONENTIAL;
    case 1: return HsgpKernel::MATERN32;
    case 2: return HsgpKernel::MATERN52;
    default:
        Rcpp::stop("%s: HSGP kernel code %d is not one of 0 (squared "
                   "exponential), 1 (Matern 3/2), 2 (Matern 5/2).", who, code);
    }
    return HsgpKernel::SQUARED_EXPONENTIAL;
}

// The kernel a nested-Laplace block spec names in its `kernel` element; the
// R side writes it on every HSGP block it builds.
inline HsgpKernel hsgp_kernel_from_spec(const Rcpp::List& spec,
                                        const char* who) {
    if (!spec.containsElementNamed("kernel")) {
        Rcpp::stop("%s: an HSGP block spec carries no `kernel` code.", who);
    }
    return hsgp_kernel_from_code(Rcpp::as<int>(spec["kernel"]), who);
}

template <typename T>
inline T hsgp_spectral_density_2d(HsgpKernel kernel, const T& sigma2,
                                  const T& lengthscale, double omega_sq) {
    return tulpa_hsgp_spectral::spectral_density(omega_sq, sigma2, lengthscale,
        2, static_cast<int>(kernel));
}

// sqrt of the spectral density at each eigenvalue, the per-coefficient loading
// scale of an HSGP latent z ~ N(0, I) at (sigma2, lengthscale):
// f(s) = sum_j phi_j(s) sqrt(S(lambda_j)) z_j.
inline void hsgp_sqrt_spectral_2d(HsgpKernel kernel, double sigma2,
                                  double lengthscale, const double* eig,
                                  int m_total, double* out) {
    for (int j = 0; j < m_total; ++j) {
        const double S = hsgp_spectral_density_2d<double>(kernel, sigma2,
                                                          lengthscale, eig[j]);
        out[j] = std::sqrt(S > 0.0 ? S : 0.0);
    }
}

}  // namespace priors
}  // namespace tulpa

#endif  // TULPA_HSGP_SPECTRAL_H
