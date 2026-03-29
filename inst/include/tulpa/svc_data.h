#ifndef TULPA_SVC_DATA_H
#define TULPA_SVC_DATA_H

#include <vector>
#include "tulpa/types.h"

namespace tulpa {

// ============================================================================
// SVC data (Spatially-Varying Coefficients)
// ============================================================================
struct SVCData {
    int n_obs = 0;                      // Number of observations
    int n_svc = 0;                      // Number of spatially-varying coefficients
    int nn = 15;                        // Number of nearest neighbors
    std::vector<double> coords;         // Coordinates [n_obs x 2], flattened
    std::vector<int> svc_indices;       // Design matrix columns with SVCs
    std::vector<double> X_svc;          // Design matrix subset [n_obs x n_svc]
    std::vector<int> nn_idx;            // Neighbor indices [n_obs x nn]
    std::vector<double> nn_dist;        // Distances to neighbors [n_obs x nn]
    std::vector<int> nn_order;          // Observation ordering
    std::vector<int> nn_order_inv;      // Inverse ordering
    CovType cov_type = CovType::EXPONENTIAL;
    bool shared = true;                 // Legacy: shared between num/denom

    // Workspace (engine-allocated, not set by model packages)
    mutable std::vector<double> w_flat_ws;      // [n_obs x n_svc]
    mutable std::vector<double> sigma2_ws;      // [n_svc]
    mutable std::vector<double> phi_ws;         // [n_svc]
    mutable std::vector<double> w_j_ws;         // [n_obs]
    mutable std::vector<double> eta_ws;         // [n_obs]

    void init_workspace() {
        w_flat_ws.resize(n_obs * n_svc);
        sigma2_ws.resize(n_svc);
        phi_ws.resize(n_svc);
        w_j_ws.resize(n_obs);
        eta_ws.resize(n_obs);
    }
};

} // namespace tulpa

#endif // TULPA_SVC_DATA_H
