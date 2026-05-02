// hmc_gp_collapsed_ops.h
// NNGP workspace + precision-matrix matvecs for collapsed GP.
// Self-contained: defines symbols inside namespace tulpa_hmc.

#ifndef TULPA_HMC_GP_COLLAPSED_OPS_H
#define TULPA_HMC_GP_COLLAPSED_OPS_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <RcppEigen.h>

#include "hmc_gp.h"        // tulpa_gp::GPData
#include "hmc_sampler.h"   // tulpa_hmc::ModelData / ModelType
#include "hmc_svc.h"       // tulpa_svc::compute_cov

namespace tulpa_hmc {

using tulpa_gp::GPData;
using tulpa_svc::compute_cov;

// =========================================================================
// NNGP precision matrix operations (Q = (I-B)^T D^{-1} (I-B))
// =========================================================================

// Workspace for collapsed GP computations
struct CollapsedGPWorkspace {
    // NNGP structure coefficients
    std::vector<double> B_flat;     // NNGP regression coefficients (flattened)
    std::vector<int> n_nb;          // Number of neighbors per location
    std::vector<int> nb_idx_flat;   // Neighbor indices (flattened, 0-based)
    std::vector<double> d_cond;     // Conditional variances
    int nn = 0;                     // Max neighbors
    int N_gp = 0;                   // Number of unique locations

    // Transpose neighbor structure (for Q^T v efficiently)
    std::vector<int> rev_ptr;       // CSR-style: rev_ptr[i] .. rev_ptr[i+1] are locations j
    std::vector<int> rev_col;       // The j values
    std::vector<int> rev_nb_pos;    // Position of i in j's neighbor list (for B lookup)

    // Newton workspace
    std::vector<double> w_star;     // Current mode estimate
    std::vector<double> grad_w;     // Gradient w.r.t. w
    std::vector<double> hess_diag;  // Diagonal of data neg-Hessian (W)
    std::vector<double> newton_rhs; // RHS for Newton solve
    std::vector<double> cg_r, cg_p, cg_Ap;  // CG workspace

    // Laplace correction
    double laplace_log_det = 0.0;  // -0.5 * log det(W + Q)

    // Analytical Laplace gradient workspace
    std::vector<double> H_inv_diag;   // diagonal of (W+Q)^{-1}, length N_gp

    bool structure_built = false;
    bool mode_found = false;

    // Fix 2: NNGP structure cache — avoid rebuild when (sigma2, phi) unchanged
    double cached_sigma2 = -1.0;
    double cached_phi = -1.0;

    // Loc→obs mapping (avoids O(N) scan per location in compute_loc_lik)
    std::vector<int> loc_obs_ptr;    // CSR: loc_obs_ptr[loc]..loc_obs_ptr[loc+1]
    std::vector<int> loc_obs_idx;    // Observation indices for each location
    bool loc_map_built = false;

    void init(int n_gp, int max_nn) {
        if (n_gp != N_gp) {
            mode_found = false;
            loc_map_built = false;
            cached_sigma2 = -1.0;
            cached_phi = -1.0;
        }
        N_gp = n_gp;
        nn = max_nn;
        B_flat.assign(n_gp * max_nn, 0.0);
        n_nb.assign(n_gp, 0);
        nb_idx_flat.assign(n_gp * max_nn, -1);
        d_cond.assign(n_gp, 1.0);
        w_star.assign(n_gp, 0.0);
        grad_w.assign(n_gp, 0.0);
        hess_diag.assign(n_gp, 0.0);
        newton_rhs.assign(n_gp, 0.0);
        cg_r.assign(n_gp, 0.0);
        cg_p.assign(n_gp, 0.0);
        cg_Ap.assign(n_gp, 0.0);
        H_inv_diag.assign(n_gp, 0.0);
    }
};

// Build loc→obs CSR mapping for O(1) per-location observation lookup
inline void build_loc_obs_map(CollapsedGPWorkspace& ws, const ModelData& data) {
    if (ws.loc_map_built) return;
    int N_gp = ws.N_gp;
    int N = data.N;
    std::vector<int> counts(N_gp, 0);
    for (int i = 0; i < N; i++) {
        int loc = data.gp_data.obs_to_loc[i];
        if (loc >= 0 && loc < N_gp) counts[loc]++;
    }
    ws.loc_obs_ptr.resize(N_gp + 1);
    ws.loc_obs_ptr[0] = 0;
    for (int loc = 0; loc < N_gp; loc++)
        ws.loc_obs_ptr[loc + 1] = ws.loc_obs_ptr[loc] + counts[loc];
    ws.loc_obs_idx.resize(ws.loc_obs_ptr[N_gp]);
    std::fill(counts.begin(), counts.end(), 0);
    for (int i = 0; i < N; i++) {
        int loc = data.gp_data.obs_to_loc[i];
        if (loc >= 0 && loc < N_gp) {
            ws.loc_obs_idx[ws.loc_obs_ptr[loc] + counts[loc]] = i;
            counts[loc]++;
        }
    }
    ws.loc_map_built = true;
}

// Build the NNGP coefficients B and conditional variances D for given (sigma2, phi)
// B[i,j] = c_i^T C_i^{-1} e_j  (regression of w_i on neighbors)
// d[i] = sigma2 - c_i^T C_i^{-1} c_i  (conditional variance)
inline void build_nngp_B_D(
    double sigma2, double phi,
    const GPData& gp_data,
    CollapsedGPWorkspace& ws
) {
    int N = gp_data.n_obs;
    int nn = gp_data.nn;
    ws.init(N, nn);

    // Work arrays
    std::vector<double> c_vec(nn), C_mat(nn * nn), L(nn * nn);
    std::vector<double> alpha(nn);

    // First observation: marginal N(0, sigma2)
    int first_idx = gp_data.nn_order[0];
    ws.n_nb[first_idx] = 0;
    ws.d_cond[first_idx] = sigma2;

    for (int i = 1; i < N; i++) {
        int obs_idx = gp_data.nn_order[i];

        // Count neighbors
        int n_nb_i = 0;
        for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) n_nb_i++;
        ws.n_nb[obs_idx] = n_nb_i;

        if (n_nb_i == 0) {
            ws.d_cond[obs_idx] = sigma2;
            continue;
        }

        // Build c_vec (cross-covariance with neighbors)
        for (int j = 0; j < n_nb_i; j++) {
            double d = gp_data.nn_dist[i * nn + j];
            c_vec[j] = compute_cov(d, sigma2, phi, gp_data.cov_type);
        }

        // Build C_mat (neighbor-neighbor covariance) and get neighbor indices
        bool ok = true;
        for (int j1 = 0; j1 < n_nb_i && ok; j1++) {
            int raw1 = gp_data.nn_idx[i * nn + j1];
            if (raw1 - 1 < 0 || raw1 - 1 >= (int)gp_data.nn_order.size()) { ok = false; break; }
            int idx1 = gp_data.nn_order[raw1 - 1];
            ws.nb_idx_flat[obs_idx * nn + j1] = idx1;

            for (int j2 = 0; j2 < n_nb_i; j2++) {
                if (j1 == j2) {
                    C_mat[j1 * n_nb_i + j2] = sigma2;
                } else {
                    int raw2 = gp_data.nn_idx[i * nn + j2];
                    if (raw2 - 1 < 0 || raw2 - 1 >= (int)gp_data.nn_order.size()) { ok = false; break; }
                    int idx2 = gp_data.nn_order[raw2 - 1];
                    double dx = gp_data.coords[idx1 * 2] - gp_data.coords[idx2 * 2];
                    double dy = gp_data.coords[idx1 * 2 + 1] - gp_data.coords[idx2 * 2 + 1];
                    C_mat[j1 * n_nb_i + j2] = compute_cov(std::sqrt(dx*dx + dy*dy),
                                                           sigma2, phi, gp_data.cov_type);
                }
            }
        }
        if (!ok) {
            ws.d_cond[obs_idx] = sigma2;
            continue;
        }

        // Cholesky: C = LL'
        std::fill(L.begin(), L.begin() + n_nb_i * n_nb_i, 0.0);
        for (int j = 0; j < n_nb_i; j++) {
            double s = 0.0;
            for (int k = 0; k < j; k++) s += L[j * n_nb_i + k] * L[j * n_nb_i + k];
            double diag = C_mat[j * n_nb_i + j] - s;
            L[j * n_nb_i + j] = (diag > 0) ? std::sqrt(diag) : 1e-5;
            for (int k = j + 1; k < n_nb_i; k++) {
                double t = 0.0;
                for (int m = 0; m < j; m++) t += L[k * n_nb_i + m] * L[j * n_nb_i + m];
                L[k * n_nb_i + j] = (C_mat[k * n_nb_i + j] - t) / L[j * n_nb_i + j];
            }
        }

        // Solve L*y = c, L'*alpha = y => alpha = C^{-1}c
        for (int j = 0; j < n_nb_i; j++) {
            double s = 0.0;
            for (int k = 0; k < j; k++) s += L[j * n_nb_i + k] * alpha[k];
            alpha[j] = (c_vec[j] - s) / L[j * n_nb_i + j];
        }
        for (int j = n_nb_i - 1; j >= 0; j--) {
            double s = 0.0;
            for (int k = j + 1; k < n_nb_i; k++) s += L[k * n_nb_i + j] * alpha[k];
            alpha[j] = (alpha[j] - s) / L[j * n_nb_i + j];
        }

        // Store B coefficients and conditional variance
        double c_alpha = 0.0;
        for (int j = 0; j < n_nb_i; j++) {
            ws.B_flat[obs_idx * nn + j] = alpha[j];
            c_alpha += c_vec[j] * alpha[j];
        }
        ws.d_cond[obs_idx] = std::max(sigma2 - c_alpha, 1e-10);
    }

    // Build reverse (transpose) neighbor structure
    // For Q^T v computation: need to know which locations j have i as neighbor
    std::vector<std::vector<std::pair<int,int>>> rev_lists(N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < ws.n_nb[i]; j++) {
            int nb = ws.nb_idx_flat[i * nn + j];
            if (nb >= 0 && nb < N) {
                rev_lists[nb].push_back({i, j});
            }
        }
    }

    // Flatten into CSR format
    ws.rev_ptr.resize(N + 1, 0);
    int total_rev = 0;
    for (int i = 0; i < N; i++) {
        ws.rev_ptr[i] = total_rev;
        total_rev += (int)rev_lists[i].size();
    }
    ws.rev_ptr[N] = total_rev;
    ws.rev_col.resize(total_rev);
    ws.rev_nb_pos.resize(total_rev);
    for (int i = 0; i < N; i++) {
        int off = ws.rev_ptr[i];
        for (int k = 0; k < (int)rev_lists[i].size(); k++) {
            ws.rev_col[off + k] = rev_lists[i][k].first;
            ws.rev_nb_pos[off + k] = rev_lists[i][k].second;
        }
    }
    ws.structure_built = true;
}

// Compute Qv = (I-B)^T D^{-1} (I-B) v  (NNGP precision times vector)
// O(N * nn) per call
inline void nngp_precision_matvec(
    const double* v,
    double* result,
    const CollapsedGPWorkspace& ws
) {
    int N = ws.N_gp;
    int nn = ws.nn;

    // Step 1: r = (I-B) v   [forward pass]
    // (I-B) is lower triangular in NNGP ordering
    // r[i] = v[i] - sum_j B[i,j] * v[nb_j]
    std::vector<double> r(N);
    for (int i = 0; i < N; i++) {
        double ri = v[i];
        for (int j = 0; j < ws.n_nb[i]; j++) {
            int nb = ws.nb_idx_flat[i * nn + j];
            if (nb >= 0) ri -= ws.B_flat[i * nn + j] * v[nb];
        }
        r[i] = ri;
    }

    // Step 2: s = D^{-1} r
    std::vector<double> s(N);
    for (int i = 0; i < N; i++) {
        s[i] = r[i] / ws.d_cond[i];
    }

    // Step 3: result = (I-B)^T s   [backward pass using reverse structure]
    // result[i] = s[i] - sum_{j where i is neighbor of j} B[j, pos] * s[j]
    for (int i = 0; i < N; i++) {
        double res_i = s[i];
        for (int k = ws.rev_ptr[i]; k < ws.rev_ptr[i + 1]; k++) {
            int j = ws.rev_col[k];
            int pos = ws.rev_nb_pos[k];
            res_i -= ws.B_flat[j * nn + pos] * s[j];
        }
        result[i] = res_i;
    }
}

// Compute (W + Q) v  where W = diag(hess_diag) and Q = NNGP precision
inline void wplusq_matvec(
    const double* v,
    double* result,
    const double* hess_diag,
    const CollapsedGPWorkspace& ws
) {
    // Qv
    nngp_precision_matvec(v, result, ws);
    // Add Wv
    for (int i = 0; i < ws.N_gp; i++) {
        result[i] += hess_diag[i] * v[i];
    }
}

// Solve (W + Q) x = b using Conjugate Gradient
// Returns number of CG iterations
inline int cg_solve(
    double* x,           // Solution (in/out, initial guess)
    const double* b,     // RHS
    const double* hess_diag,
    CollapsedGPWorkspace& ws,
    int max_iter = 100,
    double tol = 1e-8
) {
    int N = ws.N_gp;

    // r = b - (W+Q)x
    wplusq_matvec(x, ws.cg_Ap.data(), hess_diag, ws);
    for (int i = 0; i < N; i++) ws.cg_r[i] = b[i] - ws.cg_Ap[i];

    // p = r
    std::memcpy(ws.cg_p.data(), ws.cg_r.data(), N * sizeof(double));

    double rr = 0.0;
    for (int i = 0; i < N; i++) rr += ws.cg_r[i] * ws.cg_r[i];

    double b_norm = 0.0;
    for (int i = 0; i < N; i++) b_norm += b[i] * b[i];
    if (b_norm < 1e-30) return 0;

    for (int iter = 0; iter < max_iter; iter++) {
        if (rr / b_norm < tol * tol) return iter;

        // Ap = (W+Q) p
        wplusq_matvec(ws.cg_p.data(), ws.cg_Ap.data(), hess_diag, ws);

        double pAp = 0.0;
        for (int i = 0; i < N; i++) pAp += ws.cg_p[i] * ws.cg_Ap[i];
        if (pAp < 1e-30) return iter;

        double alpha = rr / pAp;

        for (int i = 0; i < N; i++) {
            x[i] += alpha * ws.cg_p[i];
            ws.cg_r[i] -= alpha * ws.cg_Ap[i];
        }

        double rr_new = 0.0;
        for (int i = 0; i < N; i++) rr_new += ws.cg_r[i] * ws.cg_r[i];

        double beta = rr_new / rr;
        for (int i = 0; i < N; i++) {
            ws.cg_p[i] = ws.cg_r[i] + beta * ws.cg_p[i];
        }
        rr = rr_new;
    }
    return max_iter;
}

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_GP_COLLAPSED_OPS_H
