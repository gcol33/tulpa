#ifndef TULPA_ST_DATA_H
#define TULPA_ST_DATA_H

#include <vector>
#include "tulpa/types.h"

namespace tulpa {

// ============================================================================
// Spatiotemporal interaction data
// ============================================================================
struct SpatiotemporalData {
    STType type = STType::NONE;
    bool shared = true;                 // Legacy: shared between num/denom

    int n_spatial = 0;                  // Number of spatial units (S)
    int n_times = 0;                    // Number of time points (T)
    int n_params = 0;                   // Total interaction parameters

    // Index mapping
    std::vector<int> s_idx;             // Spatial index per obs (1-based)
    std::vector<int> t_idx;             // Temporal index per obs (1-based)
    std::vector<int> st_flat;           // Flattened (s-1)*T + t

    // Spatial structure
    bool spatial_is_gp = false;
    std::vector<int> adj_row_ptr;       // CSR row pointers (CAR structure)
    std::vector<int> adj_col_idx;       // CSR column indices
    std::vector<int> n_neighbors;       // Neighbors per spatial unit
    bool spatial_proper = false;        // Proper CAR vs ICAR
    double bym2_scale = 1.0;           // BYM2 scaling factor

    // Temporal structure
    TemporalType temporal_type = TemporalType::RW1;
    bool temporal_cyclic = false;

    // GP-based ST
    NonsepType nonsep_type = NonsepType::PRODUCT;
    CovType cov_space = CovType::EXPONENTIAL;
    CovType cov_time = CovType::EXPONENTIAL;
    int nn = 15;                        // NNGP neighbors
    std::vector<double> coords;         // Coordinates [N x 2], flattened
    std::vector<double> time_values;    // Time values [N]
    std::vector<int> nn_idx;            // NNGP neighbor indices
    std::vector<double> nn_dist_space;  // Spatial distances to neighbors
    std::vector<double> nn_dist_time;   // Temporal distances to neighbors
    std::vector<int> nn_order;          // Observation ordering
    std::vector<int> nn_order_inv;      // Inverse ordering

    // Priors
    double sigma2_prior_U = 1.0;
    double sigma2_prior_alpha = 0.01;
    double phi_space_prior_lower = 0.01;
    double phi_space_prior_upper = 10.0;
    double phi_time_prior_lower = 0.01;
    double phi_time_prior_upper = 10.0;

    // HSGP-ST
    bool is_hsgp = false;
    int hsgp_m_total = 0;              // m^2 basis functions
};

} // namespace tulpa

#endif // TULPA_ST_DATA_H
