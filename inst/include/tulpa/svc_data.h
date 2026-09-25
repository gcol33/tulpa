#ifndef TULPA_SVC_DATA_H
#define TULPA_SVC_DATA_H

#include <vector>
#include "tulpa/types.h"

namespace tulpa {

// ============================================================================
// SVC data (Spatially-Varying Coefficients)
// ============================================================================
struct SVCData {
    // Field LOCATIONS. Equal to the observation count unless several rows share
    // a location, in which case each term's field lives on the distinct
    // locations and `obs_to_loc` maps a row to its one: two
    // rows at one site would otherwise be two field values at distance 0,
    // perfectly correlated, and the neighbour covariance singular.
    int n_obs = 0;
    int n_svc = 0;                      // Number of spatially-varying coefficients
    int nn = 15;                        // Number of nearest neighbors
    std::vector<double> coords;         // Coordinates [n_obs x 2], flattened
    std::vector<int> svc_indices;       // Design matrix columns with SVCs
    std::vector<double> X_svc;          // Design matrix subset [n_rows() x n_svc]
    std::vector<int> nn_idx;            // Neighbor indices [n_obs x nn]
    std::vector<double> nn_dist;        // Distances to neighbors [n_obs x nn]
    std::vector<int> nn_order;          // Observation ordering
    std::vector<int> nn_order_inv;      // Inverse ordering
    CovType cov_type = CovType::EXPONENTIAL;
    bool shared = true;                 // shared across processes
    // Row -> location map, 0-based, one entry per observation row. EMPTY is
    // the identity (one location per row, n_rows() == n_obs), which is every
    // HSGP field and every NNGP field on distinct coordinates.
    std::vector<int> obs_to_loc;

    int n_rows() const {
        return obs_to_loc.empty() ? n_obs : static_cast<int>(obs_to_loc.size());
    }
    int loc_of(int i) const { return obs_to_loc.empty() ? i : obs_to_loc[i]; }

    // Workspace (engine-allocated, not set by model packages)
    mutable std::vector<double> w_flat_ws;      // [n_obs x n_svc]
    mutable std::vector<double> sigma2_ws;      // [n_svc]
    mutable std::vector<double> phi_ws;         // [n_svc]
    mutable std::vector<double> w_j_ws;         // [n_obs]
    mutable std::vector<double> eta_ws;         // [n_rows()]

    void init_workspace() {
        w_flat_ws.resize(n_obs * n_svc);
        sigma2_ws.resize(n_svc);
        phi_ws.resize(n_svc);
        w_j_ws.resize(n_obs);
        eta_ws.resize(n_rows());
    }
};

} // namespace tulpa

#endif // TULPA_SVC_DATA_H
