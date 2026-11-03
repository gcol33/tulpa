// hmc_hsgp_kernels.h
// Eigen-free portion of the HSGP basis: the 1D Laplacian eigenfunctions /
// eigenvalues. The spectral densities the basis weights carry live in
// hsgp_spectral.h. A translation unit that needs only these includes this
// header on its own and does not pull in RcppEigen.

#ifndef TULPA_HMC_HSGP_KERNELS_H
#define TULPA_HMC_HSGP_KERNELS_H

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa_hsgp {

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
