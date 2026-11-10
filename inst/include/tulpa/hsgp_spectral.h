#ifndef TULPA_EXPORTED_HSGP_SPECTRAL_H
#define TULPA_EXPORTED_HSGP_SPECTRAL_H

#include <tulpa/ad_scalar_math.h>
#include <cmath>

namespace tulpa_hsgp_spectral {

// Angular-frequency spectral density in d dimensions. Matern kernels use
// k(r) = sigma2 * Matern(sqrt(2 nu) r / ell), with nu = 3/2 or 5/2.
template<typename T>
inline T spectral_density(double omega_sq, const T& sigma2, const T& ell,
                          int dimension, int kernel = 0) {
    using tulpa::math::safe_exp;
    using tulpa::math::safe_log;
    const double pi = std::acos(-1.0);
    if (kernel == 0) {
        T scale = T(1.0);
        for (int j = 0; j < dimension; ++j) scale = scale * ell;
        return sigma2 * T(std::pow(2 * pi, 0.5 * dimension)) * scale *
            safe_exp(T(-0.5 * omega_sq) * ell * ell);
    }
    const double nu = kernel == 1 ? 1.5 : 2.5;
    const double power = nu + 0.5 * dimension;
    const double constant = std::pow(2.0, dimension) * std::pow(pi, 0.5 * dimension) *
        std::tgamma(power) * std::pow(2 * nu, nu) / std::tgamma(nu);
    return sigma2 * T(constant) * safe_exp(T(-2 * nu) * safe_log(ell) -
        T(power) * safe_log(T(2 * nu) / (ell * ell) + T(omega_sq)));
}

inline double spectral_log_lengthscale_derivative(double omega_sq, double ell,
                                                   int dimension, int kernel = 0) {
    if (kernel == 0) return dimension - ell * ell * omega_sq;
    const double nu = kernel == 1 ? 1.5 : 2.5;
    return -2 * nu + (2 * nu + dimension) * (2 * nu) /
        (2 * nu + ell * ell * omega_sq);
}

} // namespace tulpa_hsgp_spectral
#endif
