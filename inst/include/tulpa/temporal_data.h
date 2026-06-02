#ifndef TULPA_TEMPORAL_DATA_H
#define TULPA_TEMPORAL_DATA_H

#include <vector>
#include "tulpa/types.h"

namespace tulpa {

// ============================================================================
// Temporal data (RW1/RW2/AR1/IID/GP)
// ============================================================================
struct TemporalData {
    TemporalType type = TemporalType::NONE;
    std::vector<int> time_index;        // Maps obs → time point (1-based)
    std::vector<int> group_index;       // Maps obs → temporal group (1-based)
    int n_times = 0;                    // Number of time points
    int n_groups = 1;                   // Number of temporal groups
    int n_temporal_params = 0;          // Total temporal parameters
    bool cyclic = false;                // Cyclic RW flag
    bool shared = true;                 // shared across processes
    double tau_temporal_shape = 1.0;    // Gamma shape for temporal precision
    double tau_temporal_rate = 0.01;    // Gamma rate for temporal precision
};

// ============================================================================
// Multi-scale temporal data (trend + seasonal + short-term)
// ============================================================================
struct MultiscaleTemporalData {
    int n_times = 0;                    // Number of unique time points
    int n_groups = 0;                   // Number of groups (for panel data)
    int n_obs = 0;                      // Total observations

    std::vector<int> time_index;        // Time index per obs (1-based)
    std::vector<int> group_index;       // Group index per obs (1-based)

    // Component specifications
    TemporalType trend_type = TemporalType::NONE;
    int seasonal_period = 0;            // 0 if no seasonal, else period (e.g., 12)
    TemporalType short_term_type = TemporalType::NONE;

    bool shared = true;
};

// ============================================================================
// Temporal GP data (for irregularly-spaced time series)
// ============================================================================
struct TemporalGPData {
    int n_obs = 0;                      // Number of unique time points
    int n_groups = 1;                   // Number of groups
    std::vector<double> time_values;    // Numeric time values [n_obs]
    std::vector<int> group_index;       // Group index per obs (1-based)
    TemporalCovType cov_type = TemporalCovType::EXPONENTIAL;
    double nu = 1.5;                    // Matern smoothness
    double period = 1.0;               // Period for periodic covariance
    bool shared = true;                 // shared across processes
};

} // namespace tulpa

#endif // TULPA_TEMPORAL_DATA_H
