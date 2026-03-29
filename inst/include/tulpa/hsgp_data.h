#ifndef TULPA_HSGP_DATA_H
#define TULPA_HSGP_DATA_H

#include <vector>

namespace tulpa {

// ============================================================================
// HSGP data (Hilbert Space GP approximation)
// ============================================================================
struct HSGPData {
    int n_obs = 0;                  // Number of observations
    int n_dim = 2;                  // Number of spatial dimensions (1 or 2)
    int m_per_dim = 15;             // Basis functions per dimension
    int m_total = 0;                // Total basis functions (m^d)
    double L1 = 0.0;               // Boundary half-length dim 1
    double L2 = 0.0;               // Boundary half-length dim 2
    std::vector<double> phi_flat;   // Basis matrix phi[i,j], flattened [n_obs x m_total]
    std::vector<double> eigenvalues;// Spectral eigenvalues [m_total]
    std::vector<double> coords_scaled; // Scaled coordinates [-1,1] [n_obs x n_dim]
    bool shared = true;             // Legacy: shared between num/denom
};

} // namespace tulpa

#endif // TULPA_HSGP_DATA_H
