// hmc_hsgp_kernels.h
// Eigen-free portion of the HSGP basis: spectral density of the squared-
// exponential kernel and 1D Laplacian eigenfunctions / eigenvalues. Split
// out of hmc_hsgp.h so translation units that only need the spectral diagonal
// (e.g. nested-Laplace drivers) do not pull in RcppEigen and trigger
// inline-template instantiations of the matvec helpers.
//
// hmc_hsgp.h still includes this header, so the original API (with Eigen
// matvecs) remains intact.

#ifndef TULPA_HMC_HSGP_KERNELS_H
#define TULPA_HMC_HSGP_KERNELS_H

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa_hsgp {

// Spectral density of the squared-exponential kernel:
//   S(ω) = σ² · √(2π) · ℓ · exp(-½ ℓ² ω²)
// `omega_sq` is ω² (squared angular frequency / Laplacian eigenvalue).
inline double spectral_density_se(double omega_sq, double sigma2, double lengthscale) {
    double ell = lengthscale;
    double ell2 = ell * ell;
    return sigma2 * std::sqrt(2.0 * M_PI) * ell * std::exp(-0.5 * ell2 * omega_sq);
}

// dS/d(σ²) = S / σ²
inline double dS_dsigma2(double omega_sq, double sigma2, double lengthscale) {
    return spectral_density_se(omega_sq, sigma2, lengthscale) / sigma2;
}

// dS/d(ℓ) = S · (1/ℓ - ℓ ω²)
inline double dS_dlengthscale(double omega_sq, double sigma2, double lengthscale) {
    double S = spectral_density_se(omega_sq, sigma2, lengthscale);
    double ell = lengthscale;
    return S * (1.0 / ell - ell * omega_sq);
}

// 1D Laplacian eigenfunction on [-L, L]:
//   φ_j(x) = (1/√L) · sin(π j (x + L) / (2L)),  j = 1, 2, …
inline double phi_1d(double x, int j, double L) {
    double norm = 1.0 / std::sqrt(L);
    return norm * std::sin(M_PI * j * (x + L) / (2.0 * L));
}

// 1D Laplacian eigenvalue: λ_j = (π j / (2L))²
inline double lambda_1d(int j, double L) {
    double tmp = M_PI * j / (2.0 * L);
    return tmp * tmp;
}

} // namespace tulpa_hsgp

#endif // TULPA_HMC_HSGP_KERNELS_H
