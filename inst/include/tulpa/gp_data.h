#ifndef TULPA_GP_DATA_H
#define TULPA_GP_DATA_H

#include <vector>
#include "tulpa/types.h"

namespace tulpa {

// ============================================================================
// GP data (for NNGP spatial fields)
// ============================================================================
struct GPData {
    int n_obs = 0;                              // Number of unique spatial locations
    int nn = 15;                                // Number of nearest neighbors
    std::vector<double> coords;                 // Coordinates [n_obs x 2], flattened
    std::vector<int> nn_idx;                    // Neighbor indices [n_obs x nn]
    std::vector<double> nn_dist;                // Distances to neighbors [n_obs x nn]
    std::vector<double> nn_neighbor_dist;       // Pairwise distances among neighbors [n_obs x nn x nn]
    std::vector<int> nn_order;                  // Observation ordering (for NNGP)
    std::vector<int> nn_order_inv;              // Inverse ordering
    std::vector<int> obs_to_loc;                // Observation-to-location mapping
    CovType cov_type = CovType::EXPONENTIAL;
    double nu = 1.5;                            // Matern smoothness
    bool shared = true;                         // Legacy: shared between num/denom
    GPSolverConfig solver_config;
};

// ============================================================================
// Multi-scale GP data (local + regional scale)
// ============================================================================
struct MultiscaleGPData {
    int n_obs = 0;                              // Number of unique spatial locations
    std::vector<double> coords;                 // Coordinates [n_obs x 2], flattened
    std::vector<int> obs_to_loc;                // Observation-to-location mapping

    // Local scale NNGP
    int nn_local = 15;
    std::vector<int> nn_idx_local;
    std::vector<double> nn_dist_local;
    std::vector<double> nn_neighbor_dist_local;
    std::vector<int> nn_order_local;
    std::vector<int> nn_order_inv_local;

    // Regional scale NNGP
    int nn_regional = 15;
    std::vector<int> nn_idx_regional;
    std::vector<double> nn_dist_regional;
    std::vector<double> nn_neighbor_dist_regional;
    std::vector<int> nn_order_regional;
    std::vector<int> nn_order_inv_regional;

    // Range bounds
    double range_local_lower = 0.01;
    double range_local_upper = 10.0;
    double range_regional_lower = 0.01;
    double range_regional_upper = 100.0;

    CovType cov_type = CovType::EXPONENTIAL;
    double nu = 1.5;                            // Matern smoothness
    bool shared = true;                         // Legacy: shared between num/denom
    MSGPSampler sampler = MSGPSampler::AUTO;
};

} // namespace tulpa

#endif // TULPA_GP_DATA_H
