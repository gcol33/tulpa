// hmc_icar_collapsed_workspace.h
// Workspace struct + unit-obs CSR map for collapsed ICAR/BYM2.
// Self-contained: defines symbols inside namespace tulpa_hmc.

#ifndef TULPA_HMC_ICAR_COLLAPSED_WORKSPACE_H
#define TULPA_HMC_ICAR_COLLAPSED_WORKSPACE_H

#include <vector>

#include "hmc_sampler.h"  // tulpa_hmc::ModelData (via using-decls)

namespace tulpa_hmc {

// =========================================================================
// Workspace for collapsed ICAR/BYM2 computations
// =========================================================================

struct CollapsedICARWorkspace {
    int S = 0;                          // Number of spatial units
    bool is_bym2 = false;               // BYM2 mode (2S inner variables)
    int inner_dim = 0;                  // S for ICAR, 2S for BYM2

    // Mode variables
    std::vector<double> phi_star;       // Structured spatial mode (S)
    std::vector<double> theta_star;     // Unstructured mode (S, BYM2 only)

    // Data-level Hessian diagonal (per spatial unit)
    std::vector<double> W_data;         // sum_i(-d²LL/deta²) at unit s (length S)

    // Newton workspace
    std::vector<double> grad;           // gradient (inner_dim)
    std::vector<double> hess_diag;      // diagonal part of Hessian (inner_dim)
    std::vector<double> cg_r, cg_p, cg_Ap;  // CG workspace (inner_dim)

    // Laplace correction
    double laplace_log_det = 0.0;       // -0.5 * log det(H)

    bool mode_found = false;

    // Pre-built unit→obs mapping (avoids O(N) scan per unit in compute_unit_lik)
    std::vector<int> unit_obs_ptr;    // CSR: unit_obs_ptr[s]..unit_obs_ptr[s+1]
    std::vector<int> unit_obs_idx;    // Observation indices for each unit
    bool obs_map_built = false;
    int obs_map_N = 0;                // N when map was built

    void init(int n_units, bool bym2) {
        if (n_units != S || bym2 != is_bym2) {
            mode_found = false;
            obs_map_built = false;
        }
        S = n_units;
        is_bym2 = bym2;
        inner_dim = bym2 ? 2 * S : S;

        phi_star.assign(S, 0.0);
        W_data.assign(S, 0.0);
        grad.assign(inner_dim, 0.0);
        hess_diag.assign(inner_dim, 0.0);
        cg_r.assign(inner_dim, 0.0);
        cg_p.assign(inner_dim, 0.0);
        cg_Ap.assign(inner_dim, 0.0);

        if (bym2) {
            theta_star.assign(S, 0.0);
        } else {
            theta_star.clear();
        }
    }
};

// Build unit→obs CSR mapping for O(1) per-unit observation lookup
inline void build_unit_obs_map(CollapsedICARWorkspace& ws, const ModelData& data) {
    if (ws.obs_map_built && ws.obs_map_N == data.N && ws.S == (int)ws.unit_obs_ptr.size() - 1) return;
    int S = ws.S;
    int N = data.N;
    std::vector<int> counts(S, 0);
    for (int i = 0; i < N; i++) {
        int s = data.spatial_group[i] - 1;
        if (s >= 0 && s < S) counts[s]++;
    }
    ws.unit_obs_ptr.resize(S + 1);
    ws.unit_obs_ptr[0] = 0;
    for (int s = 0; s < S; s++) ws.unit_obs_ptr[s + 1] = ws.unit_obs_ptr[s] + counts[s];
    ws.unit_obs_idx.resize(ws.unit_obs_ptr[S]);
    std::fill(counts.begin(), counts.end(), 0);
    for (int i = 0; i < N; i++) {
        int s = data.spatial_group[i] - 1;
        if (s >= 0 && s < S) {
            ws.unit_obs_idx[ws.unit_obs_ptr[s] + counts[s]] = i;
            counts[s]++;
        }
    }
    ws.obs_map_built = true;
    ws.obs_map_N = N;
}

}  // namespace tulpa_hmc

#endif // TULPA_HMC_ICAR_COLLAPSED_WORKSPACE_H
