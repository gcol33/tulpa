// hmc_hsgp_kernels.h
// Eigen-free portion of the HSGP basis: spectral density of the squared-
// exponential kernel and 1D Laplacian eigenfunctions / eigenvalues. A
// translation unit that needs only the spectral diagonal (a nested-Laplace
// driver, the prediction path) includes this header on its own and does not
// pull in RcppEigen.

#ifndef TULPA_HMC_HSGP_KERNELS_H
#define TULPA_HMC_HSGP_KERNELS_H

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa_hsgp {

// Spectral density of the 2-D isotropic squared-exponential kernel:
//   S(ω) = σ² · (2π) · ℓ² · exp(-½ ℓ² ω²)
// The HSGP basis here is 2-D (eigenvalue ω² = ‖ω‖² = λ_j1 + λ_j2), so the
// D-dimensional prefactor σ² · (2π)^{D/2} · ℓ^D is evaluated at D = 2; the
// 1-D form σ² · √(2π) · ℓ mis-scales the field's marginal variance.
inline double spectral_density_se(double omega_sq, double sigma2, double lengthscale) {
    double ell = lengthscale;
    double ell2 = ell * ell;
    return sigma2 * (2.0 * M_PI) * ell2 * std::exp(-0.5 * ell2 * omega_sq);
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
