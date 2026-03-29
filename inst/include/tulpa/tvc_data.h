#ifndef TULPA_TVC_DATA_H
#define TULPA_TVC_DATA_H

#include <vector>
#include "tulpa/types.h"

namespace tulpa {

// ============================================================================
// TVC data (Temporally-Varying Coefficients)
// ============================================================================
struct TVCData {
    int n_obs = 0;                      // Number of observations
    int n_times = 0;                    // Number of unique time points
    int n_tvc = 0;                      // Number of TVC terms
    int n_groups = 1;                   // Number of groups
    std::vector<int> time_index;        // Maps obs → time point (1-based)
    std::vector<int> group_index;       // Maps obs → group (1-based)
    std::vector<int> tvc_indices;       // Design matrix columns with TVCs
    std::vector<double> X_tvc;          // Design matrix subset [n_obs x n_tvc]
    TemporalType structure = TemporalType::RW1;
    bool shared = false;                // Legacy: shared between num/denom
    bool cyclic = false;                // Cyclic temporal structure

    // Workspace (engine-allocated, not set by model packages)
    mutable std::vector<double> tau_ws;
    mutable std::vector<double> rho_ws;
    mutable std::vector<double> w_flat_ws;
    mutable std::vector<double> eta_ws;
    mutable std::vector<double> grad_w_ws;
    mutable std::vector<double> grad_log_tau_ws;
    mutable std::vector<double> grad_logit_rho_ws;
    mutable std::vector<double> grad_w_jg_ws;
    mutable std::vector<double> d_ws;

    void init_workspace() {
        int n_w = n_groups * n_tvc * n_times;
        tau_ws.resize(n_tvc);
        rho_ws.resize(n_tvc);
        w_flat_ws.resize(n_w);
        eta_ws.resize(n_obs);
        grad_w_ws.resize(n_w);
        grad_log_tau_ws.resize(n_tvc);
        grad_logit_rho_ws.resize(n_tvc);
        grad_w_jg_ws.resize(n_times);
        d_ws.resize(n_times);
    }
};

} // namespace tulpa

#endif // TULPA_TVC_DATA_H
