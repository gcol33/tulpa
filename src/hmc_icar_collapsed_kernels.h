// hmc_icar_collapsed_kernels.h
// ICAR/BYM2 precision and Hessian matvecs + CG solver.
// Self-contained: defines symbols inside namespace tulpa_hmc.

#ifndef TULPA_HMC_ICAR_COLLAPSED_KERNELS_H
#define TULPA_HMC_ICAR_COLLAPSED_KERNELS_H

#include <cstring>
#include <vector>

#include "hmc_icar_collapsed_workspace.h"
#include "hmc_sampler.h"
#include "icar_kernel.h"

namespace tulpa_hmc {

// =========================================================================
// ICAR precision matrix operations (Q = D - W, adjacency-based)
// =========================================================================

// Compute Q*v where Q[i,i] = n_neighbors[i], Q[i,j] = -1 if j~i.
// Thin wrapper over the shared CAR kernel (rho = 1 special case).
inline void icar_precision_matvec(
    const double* v,
    double* result,
    int S,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    const std::vector<int>& n_neighbors
) {
    tulpa::car_apply(v, result, S,
                     adj_row_ptr.data(), adj_col_idx.data(), n_neighbors.data(),
                     /*rho=*/1.0);
}

// =========================================================================
// (W + tau*Q + lambda*I) matvec for ICAR CG solver
// W = diag(W_data), Q = ICAR precision, lambda = sum-to-zero penalty
// =========================================================================

// ICAR mode: (W + tau*Q + lambda*I) v  +  lambda_s2z * sum(v) * ones
// The sum-to-zero penalty: -0.5 * lambda_s2z * (sum phi)^2
//   → adds lambda_s2z * 11^T to the Hessian
// For efficiency: compute Q*v, then add diagonal + dense rank-1 terms
inline void icar_hessian_matvec(
    const double* v,
    double* result,
    const CollapsedICARWorkspace& ws,
    double tau,
    const ModelData& data,
    double lambda_s2z = 0.001  // sum-to-zero penalty strength
) {
    int S = ws.S;

    // Q * v
    icar_precision_matvec(v, result, S, data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);

    // Scale by tau and add W diagonal + sum-to-zero rank-1 term
    double sum_v = 0.0;
    for (int i = 0; i < S; i++) sum_v += v[i];

    for (int i = 0; i < S; i++) {
        result[i] = ws.W_data[i] * v[i] + tau * result[i] + lambda_s2z * sum_v;
    }
}

// BYM2 mode: full 2S Hessian matvec
// Variables: [phi_0..phi_{S-1}, theta_0..theta_{S-1}]
// Block structure:
//   phi-phi:     a²*W_data + tau_icar*Q + lambda_s2z*11^T
//   theta-theta: c²*W_data + I
//   phi-theta:   a*c*W_data (diagonal)
// where a = sigma_s * scale, c = sigma_u
inline void bym2_hessian_matvec(
    const double* v,           // length 2S: [v_phi; v_theta]
    double* result,            // length 2S: [result_phi; result_theta]
    const CollapsedICARWorkspace& ws,
    double a,                  // sigma_s * scale = sigma_total * sqrt(rho) * scale
    double c,                  // sigma_u = sigma_total * sqrt(1-rho)
    const ModelData& data,
    double lambda_s2z = 0.001
) {
    int S = ws.S;
    const double* v_phi = v;
    const double* v_theta = v + S;
    double* r_phi = result;
    double* r_theta = result + S;

    // Q * v_phi (ICAR structure)
    std::vector<double> Qv(S);
    icar_precision_matvec(v_phi, Qv.data(), S, data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);

    double sum_vphi = 0.0;
    for (int i = 0; i < S; i++) sum_vphi += v_phi[i];

    double a2 = a * a;
    double c2 = c * c;
    double ac = a * c;

    for (int i = 0; i < S; i++) {
        double Wd = ws.W_data[i];
        r_phi[i] = a2 * Wd * v_phi[i] + Qv[i] + lambda_s2z * sum_vphi
                   + ac * Wd * v_theta[i];
        r_theta[i] = ac * Wd * v_phi[i] + (c2 * Wd + 1.0) * v_theta[i];
    }
}

// =========================================================================
// CG solver for collapsed ICAR/BYM2
// =========================================================================

// Generic CG: solves H*x = b where H is applied via function pointer
// For ICAR: H = (W + tau*Q + lambda_s2z*11^T)
// For BYM2: H = full 2S block Hessian
inline int icar_cg_solve(
    double* x,           // Solution (in/out, warm-started)
    const double* b,     // RHS
    CollapsedICARWorkspace& ws,
    double tau,          // ICAR precision (or 1.0 for BYM2 since Q has no tau)
    double a,            // BYM2: sigma_s * scale (unused for ICAR)
    double cc,           // BYM2: sigma_u (unused for ICAR)
    const ModelData& data,
    int max_iter = 100,
    double tol = 1e-8
) {
    int N = ws.inner_dim;

    // r = b - H*x
    if (ws.is_bym2) {
        bym2_hessian_matvec(x, ws.cg_Ap.data(), ws, a, cc, data);
    } else {
        icar_hessian_matvec(x, ws.cg_Ap.data(), ws, tau, data);
    }
    for (int i = 0; i < N; i++) ws.cg_r[i] = b[i] - ws.cg_Ap[i];

    std::memcpy(ws.cg_p.data(), ws.cg_r.data(), N * sizeof(double));

    double rr = 0.0;
    for (int i = 0; i < N; i++) rr += ws.cg_r[i] * ws.cg_r[i];

    double b_norm = 0.0;
    for (int i = 0; i < N; i++) b_norm += b[i] * b[i];
    if (b_norm < 1e-30) return 0;

    for (int iter = 0; iter < max_iter; iter++) {
        if (rr / b_norm < tol * tol) return iter;

        if (ws.is_bym2) {
            bym2_hessian_matvec(ws.cg_p.data(), ws.cg_Ap.data(), ws, a, cc, data);
        } else {
            icar_hessian_matvec(ws.cg_p.data(), ws.cg_Ap.data(), ws, tau, data);
        }

        double pAp = 0.0;
        for (int i = 0; i < N; i++) pAp += ws.cg_p[i] * ws.cg_Ap[i];
        if (pAp < 1e-30) return iter;

        double alpha_cg = rr / pAp;
        for (int i = 0; i < N; i++) {
            x[i] += alpha_cg * ws.cg_p[i];
            ws.cg_r[i] -= alpha_cg * ws.cg_Ap[i];
        }

        double rr_new = 0.0;
        for (int i = 0; i < N; i++) rr_new += ws.cg_r[i] * ws.cg_r[i];

        double beta_cg = rr_new / rr;
        for (int i = 0; i < N; i++) {
            ws.cg_p[i] = ws.cg_r[i] + beta_cg * ws.cg_p[i];
        }
        rr = rr_new;
    }
    return max_iter;
}

}  // namespace tulpa_hmc

#endif // TULPA_HMC_ICAR_COLLAPSED_KERNELS_H
