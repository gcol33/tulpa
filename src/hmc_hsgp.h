// hmc_hsgp.h
// Hilbert Space Gaussian Process (HSGP) approximation
// Based on Riutort-Mayol et al. (2023) and Stan's implementation
//
// HSGP approximates GP as: f(x) = sum_j phi_j(x) * sqrt(S(lambda_j)) * beta_j
// where phi_j are Laplacian eigenfunctions and S is the spectral density

#ifndef TULPA_HMC_HSGP_H
#define TULPA_HMC_HSGP_H

#include <vector>
#include <cmath>
#include <RcppEigen.h>
#include "tulpa/hsgp_data.h"
#include "hmc_hsgp_kernels.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa_hsgp {

using tulpa::HSGPData;

// spectral_density_se, phi_1d and lambda_1d are declared in
// hmc_hsgp_kernels.h (Eigen-free).

// Fill the eigenvalues + basis matrix for a GIVEN centering (x_center, y_center)
// and boundary (L1, L2). Single source of truth for the eigenfunction /
// eigenvalue math; setup_hsgp_2d derives (center, L) from the coordinate range
// and calls this, and the prediction path (cpp_hsgp_field_predict) reuses it
// with the training-derived (center, L) so a new-coordinate basis is consistent
// with the fitted one (the basis depends on the training extent, not the
// prediction extent).
inline void hsgp_fill_basis_2d(
    const std::vector<double>& coords,
    int n_obs,
    int m,
    double x_center,
    double y_center,
    double L1,
    double L2,
    bool shared,
    HSGPData& data
) {
    data.n_obs = n_obs;
    data.n_dim = 2;
    data.m_per_dim = m;
    data.m_total = m * m;
    data.shared = shared;
    data.L1 = L1;
    data.L2 = L2;

    // Scale coordinates to [-L, L] about the supplied center.
    data.coords_scaled.resize(2 * n_obs);
    for (int i = 0; i < n_obs; i++) {
        data.coords_scaled[2*i]     = coords[2*i]     - x_center;
        data.coords_scaled[2*i + 1] = coords[2*i + 1] - y_center;
    }

    // Compute eigenvalues for 2D: lambda_{j1,j2} = lambda_j1 + lambda_j2
    data.eigenvalues.resize(data.m_total);
    for (int j1 = 1; j1 <= m; j1++) {
        for (int j2 = 1; j2 <= m; j2++) {
            int idx = (j1 - 1) * m + (j2 - 1);
            data.eigenvalues[idx] = lambda_1d(j1, data.L1) + lambda_1d(j2, data.L2);
        }
    }

    // Compute basis matrix: phi[i, j] = phi_{j1}(x_i) * phi_{j2}(y_i)
    data.phi_flat.resize(n_obs * data.m_total);
    for (int i = 0; i < n_obs; i++) {
        double x = data.coords_scaled[2*i];
        double y = data.coords_scaled[2*i + 1];

        for (int j1 = 1; j1 <= m; j1++) {
            double phi_x = phi_1d(x, j1, data.L1);
            for (int j2 = 1; j2 <= m; j2++) {
                double phi_y = phi_1d(y, j2, data.L2);
                int j_idx = (j1 - 1) * m + (j2 - 1);
                data.phi_flat[i * data.m_total + j_idx] = phi_x * phi_y;
            }
        }
    }
}

// Compute the training-consistent (center, L) an HSGP basis uses for a given
// coordinate set and boundary factor c. Exposed so the prediction path can
// recover the fitted centering from the training coordinates.
inline void hsgp_center_L_2d(
    const std::vector<double>& coords,
    int n_obs,
    double c,
    double& x_center,
    double& y_center,
    double& L1,
    double& L2
) {
    double x_min = coords[0], x_max = coords[0];
    double y_min = coords[1], y_max = coords[1];
    for (int i = 1; i < n_obs; i++) {
        double x = coords[2*i];
        double y = coords[2*i + 1];
        if (x < x_min) x_min = x;
        if (x > x_max) x_max = x;
        if (y < y_min) y_min = y;
        if (y > y_max) y_max = y;
    }
    x_center = (x_max + x_min) / 2.0;
    y_center = (y_max + y_min) / 2.0;
    L1 = c * (x_max - x_min) / 2.0;
    L2 = c * (y_max - y_min) / 2.0;
    if (L1 < 0.1) L1 = 0.1;
    if (L2 < 0.1) L2 = 0.1;
}

// Setup HSGP for 2D coordinates
// coords: flattened [x1, y1, x2, y2, ...] (length 2*n_obs)
// m: basis functions per dimension
// c: boundary factor (L = c * max_range)
inline void setup_hsgp_2d(
    const std::vector<double>& coords,
    int n_obs,
    int m,
    double c,
    bool shared,
    HSGPData& data
) {
    double x_center, y_center, L1, L2;
    hsgp_center_L_2d(coords, n_obs, c, x_center, y_center, L1, L2);
    hsgp_fill_basis_2d(coords, n_obs, m, x_center, y_center, L1, L2, shared, data);
}

} // namespace tulpa_hsgp

#endif // TULPA_HMC_HSGP_H
