// hmc_icar_collapsed.h
// Collapsed/marginalized ICAR and BYM2 spatial effects via inner Laplace optimization
//
// Instead of sampling S (ICAR) or 2S (BYM2) spatial effects alongside hyperparameters,
// we marginalize them out by finding phi* = argmax [log p(y|phi,theta_outer) + log p(phi|tau)]
// at each HMC gradient evaluation.
//
// ICAR: reduces S+1 params (log_tau + S phi) to just 1 (log_tau)
// BYM2: reduces 2S+2 params (log_sigma, logit_rho, S phi, S theta) to 2 (log_sigma, logit_rho)
//
// The collapsed log-posterior is:
//   log p(theta|y) ~ log p(y|phi*,theta) + log p(phi*|tau) + log p(theta)
//                    - 0.5 * log det(W + tau*Q)  [Laplace correction]
//
// Key advantage over collapsed GP: Q is FIXED (adjacency-based), doesn't depend on
// hyperparameters. Only tau*Q changes. This makes numerical Laplace gradient cheaper.

#ifndef TULPA_HMC_ICAR_COLLAPSED_H
#define TULPA_HMC_ICAR_COLLAPSED_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppEigen.h>

// NOTE: Must be included AFTER hmc_sampler.h (which defines ModelData/ModelType).
using tulpa_hmc::ModelData;
using tulpa_hmc::ModelType;
using tulpa_hmc::SpatialType;

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

// =========================================================================
// ICAR precision matrix operations (Q = D - W, adjacency-based)
// =========================================================================

// Compute Q*v where Q[i,i] = n_neighbors[i], Q[i,j] = -1 if j~i
// Uses CSR adjacency from ModelData
inline void icar_precision_matvec(
    const double* v,
    double* result,
    int S,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    const std::vector<int>& n_neighbors
) {
    for (int i = 0; i < S; i++) {
        result[i] = n_neighbors[i] * v[i];
        for (int k = adj_row_ptr[i]; k < adj_row_ptr[i + 1]; k++) {
            result[i] -= v[adj_col_idx[k]];
        }
    }
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

// =========================================================================
// Laplace log-det correction via sparse Cholesky
// =========================================================================

// ICAR: build sparse A = W + tau*Q, then Woodbury for rank-1 sum-to-zero
// det(A + λ·11') = det(A) · (1 + λ · 1'A⁻¹1)  — keeps A truly sparse
inline double compute_laplace_log_det_icar(
    const CollapsedICARWorkspace& ws,
    double tau,
    const ModelData& data,
    double lambda_s2z = 0.001
) {
    int S = ws.S;

    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;
    triplets.reserve(S + 2 * data.adj_col_idx.size());

    // Build SPARSE A = W + tau*Q (NO rank-1 — handled via Woodbury)
    for (int i = 0; i < S; i++) {
        triplets.push_back(T(i, i, ws.W_data[i] + tau * data.n_neighbors[i]));
    }
    for (int i = 0; i < S; i++) {
        for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; k++) {
            int j = data.adj_col_idx[k];
            triplets.push_back(T(i, j, -tau));
        }
    }

    Eigen::SparseMatrix<double> A(S, S);
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;
    llt.compute(A);
    if (llt.info() != Eigen::Success) {
        for (int i = 0; i < S; i++) {
            triplets.push_back(T(i, i, 1e-6));
        }
        A.setFromTriplets(triplets.begin(), triplets.end());
        llt.compute(A);
        if (llt.info() != Eigen::Success) return 0.0;
    }

    // log det(A) from sparse Cholesky
    Eigen::SparseMatrix<double> L_sparse = llt.matrixL();
    double log_det_A = 0.0;
    for (int i = 0; i < S; i++) {
        log_det_A += std::log(L_sparse.coeff(i, i));
    }
    log_det_A *= 2.0;

    // Woodbury: log det(A + λ·11') = log det(A) + log(1 + λ · 1'A⁻¹·1)
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(S);
    Eigen::VectorXd u = llt.solve(ones);
    double one_t_u = ones.dot(u);
    double log_det = log_det_A + std::log(1.0 + lambda_s2z * one_t_u);

    return -0.5 * log_det;
}

// BYM2: build sparse 2S×2S block Hessian, Woodbury for rank-1 sum-to-zero in phi block
// H_base = [[a²*W + Q,  a*c*W], [a*c*W,  c²*W + I]]  (sparse)
// H = H_base + λ·vv'  where v = [1,...,1, 0,...,0]  (phi block only)
inline double compute_laplace_log_det_bym2(
    const CollapsedICARWorkspace& ws,
    double a, double cc,
    const ModelData& data,
    double lambda_s2z = 0.001
) {
    int S = ws.S;
    int dim = 2 * S;

    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;
    triplets.reserve(dim + 4 * S + 2 * data.adj_col_idx.size());

    double a2 = a * a;
    double c2 = cc * cc;
    double ac = a * cc;

    // phi-phi block (top-left S×S): a²*W + Q (NO rank-1 — Woodbury)
    for (int i = 0; i < S; i++) {
        triplets.push_back(T(i, i, a2 * ws.W_data[i] + data.n_neighbors[i]));
    }
    for (int i = 0; i < S; i++) {
        for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; k++) {
            int j = data.adj_col_idx[k];
            triplets.push_back(T(i, j, -1.0));
        }
    }

    // theta-theta block (bottom-right S×S): c²*W + I
    for (int i = 0; i < S; i++) {
        triplets.push_back(T(S + i, S + i, c2 * ws.W_data[i] + 1.0));
    }

    // phi-theta cross blocks (diagonal): a*c*W
    for (int i = 0; i < S; i++) {
        if (std::abs(ac * ws.W_data[i]) > 1e-15) {
            triplets.push_back(T(i, S + i, ac * ws.W_data[i]));
            triplets.push_back(T(S + i, i, ac * ws.W_data[i]));
        }
    }

    Eigen::SparseMatrix<double> H_base(dim, dim);
    H_base.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;
    llt.compute(H_base);
    if (llt.info() != Eigen::Success) {
        for (int i = 0; i < dim; i++) {
            triplets.push_back(T(i, i, 1e-6));
        }
        H_base.setFromTriplets(triplets.begin(), triplets.end());
        llt.compute(H_base);
        if (llt.info() != Eigen::Success) return 0.0;
    }

    // log det(H_base) from sparse Cholesky
    Eigen::SparseMatrix<double> L_sparse = llt.matrixL();
    double log_det_base = 0.0;
    for (int i = 0; i < dim; i++) {
        log_det_base += std::log(L_sparse.coeff(i, i));
    }
    log_det_base *= 2.0;

    // Woodbury: det(H_base + λ·vv') = det(H_base) · (1 + λ · v'H_base⁻¹·v)
    // where v = [1,...,1, 0,...,0] (first S entries are 1, last S are 0)
    Eigen::VectorXd v = Eigen::VectorXd::Zero(dim);
    v.head(S).setOnes();
    Eigen::VectorXd u = llt.solve(v);
    double v_t_u = v.dot(u);
    double log_det = log_det_base + std::log(1.0 + lambda_s2z * v_t_u);

    return -0.5 * log_det;
}

// =========================================================================
// Analytical Laplace gradient: tr(H^{-1} dH/dtheta) for hyperparameters
// Eliminates 2*p Newton re-solves per gradient call.
// Uses dense algebra — efficient for S <= 200 (collapsed is for small S).
// =========================================================================

struct LaplaceGradICARResult {
    double log_det;         // -0.5 * log det(H)
    double grad_log_tau;    // d/d(log_tau) [-0.5 log det H] = -0.5 * tau * tr(H^{-1} Q)
};

// ICAR: H = A + λ·11' where A = diag(W_data) + tau*Q
// Uses Woodbury: H⁻¹ = A⁻¹ - λ/(1+λv) · u·u' where u = A⁻¹·1, v = 1'u
// dH/d(log_tau) = tau * Q  =>  grad = -0.5 * tau * tr(H^{-1} Q)
inline LaplaceGradICARResult compute_laplace_grad_icar(
    const CollapsedICARWorkspace& ws,
    double tau,
    const ModelData& data,
    double lambda_s2z = 0.001
) {
    int S = ws.S;
    LaplaceGradICARResult result = {0.0, 0.0};

    // Build dense A = W + tau*Q (NO rank-1)
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(S, S);
    for (int i = 0; i < S; i++) {
        A(i, i) = ws.W_data[i] + tau * data.n_neighbors[i];
        for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; k++) {
            int j = data.adj_col_idx[k];
            A(i, j) = -tau;
        }
    }

    // Dense Cholesky of A
    Eigen::LLT<Eigen::MatrixXd> llt(A);
    if (llt.info() != Eigen::Success) {
        A.diagonal().array() += 1e-6;
        llt.compute(A);
        if (llt.info() != Eigen::Success) return result;
    }

    // log det(A) from Cholesky
    Eigen::MatrixXd L = llt.matrixL();
    double log_det_A = 0.0;
    for (int i = 0; i < S; i++) log_det_A += 2.0 * std::log(L(i, i));

    // Woodbury: u = A⁻¹·1, v = 1'u
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(S);
    Eigen::VectorXd u = llt.solve(ones);
    double v = ones.dot(u);  // 1'A⁻¹·1

    // log det(H) = log det(A) + log(1 + λv)
    double log_woodbury = std::log(1.0 + lambda_s2z * v);
    result.log_det = -0.5 * (log_det_A + log_woodbury);

    // A⁻¹
    Eigen::MatrixXd Ainv = llt.solve(Eigen::MatrixXd::Identity(S, S));

    // H⁻¹ = A⁻¹ - λ/(1+λv) · u·u'
    double woodbury_scale = lambda_s2z / (1.0 + lambda_s2z * v);

    // tr(H⁻¹ Q) = tr(A⁻¹ Q) - woodbury_scale · u'Qu
    double trace_AinvQ = 0.0;
    for (int i = 0; i < S; i++) {
        trace_AinvQ += data.n_neighbors[i] * Ainv(i, i);
        for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; k++) {
            trace_AinvQ -= Ainv(i, data.adj_col_idx[k]);
        }
    }

    // Qu = Q·u (sparse matvec)
    Eigen::VectorXd Qu(S);
    for (int i = 0; i < S; i++) {
        Qu(i) = data.n_neighbors[i] * u(i);
        for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; k++) {
            Qu(i) -= u(data.adj_col_idx[k]);
        }
    }
    double uQu = u.dot(Qu);

    double trace_HinvQ = trace_AinvQ - woodbury_scale * uQu;
    result.grad_log_tau = -0.5 * tau * trace_HinvQ;
    return result;
}

struct LaplaceGradBYM2Result {
    double log_det;           // -0.5 * log det(H)
    double grad_log_sigma;    // d/d(log_sigma) [-0.5 log det H]
    double grad_logit_rho;    // d/d(logit_rho) [-0.5 log det H]
};

// BYM2: H = A + λ·vv' where A is 2S×2S block Hessian, v=[1,...,1,0,...,0]
// Uses Woodbury: H⁻¹ = A⁻¹ - λ/(1+λ·v'A⁻¹v) · (A⁻¹v)(A⁻¹v)'
//
// dH/d(log_sigma) = [[2a²*diag(W), 2ac*diag(W)],
//                     [2ac*diag(W), 2c²*diag(W)]]
//
// dH/d(logit_rho) = [[a²*(1-rho)*diag(W), ac*(1-2rho)/2*diag(W)],
//                     [ac*(1-2rho)/2*diag(W), -c²*rho*diag(W)]]
inline LaplaceGradBYM2Result compute_laplace_grad_bym2(
    const CollapsedICARWorkspace& ws,
    double a, double cc,
    double rho,
    const ModelData& data,
    double lambda_s2z = 0.001
) {
    int S = ws.S;
    int dim = 2 * S;
    LaplaceGradBYM2Result result = {0.0, 0.0, 0.0};

    double a2 = a * a;
    double c2 = cc * cc;
    double ac = a * cc;

    // Build dense A (NO rank-1 in phi block)
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dim, dim);

    // phi-phi block (top-left S×S): a²*W + Q
    for (int i = 0; i < S; i++) {
        A(i, i) = a2 * ws.W_data[i] + data.n_neighbors[i];
        for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; k++) {
            A(i, data.adj_col_idx[k]) -= 1.0;
        }
    }

    // theta-theta block
    for (int i = 0; i < S; i++) {
        A(S + i, S + i) = c2 * ws.W_data[i] + 1.0;
    }

    // phi-theta cross blocks
    for (int i = 0; i < S; i++) {
        A(i, S + i) = ac * ws.W_data[i];
        A(S + i, i) = ac * ws.W_data[i];
    }

    // Dense Cholesky of A
    Eigen::LLT<Eigen::MatrixXd> llt(A);
    if (llt.info() != Eigen::Success) {
        A.diagonal().array() += 1e-6;
        llt.compute(A);
        if (llt.info() != Eigen::Success) return result;
    }

    // log det(A)
    Eigen::MatrixXd L = llt.matrixL();
    double log_det_A = 0.0;
    for (int i = 0; i < dim; i++) log_det_A += 2.0 * std::log(L(i, i));

    // Woodbury: v=[1,...,1,0,...,0], u=A⁻¹v
    Eigen::VectorXd v = Eigen::VectorXd::Zero(dim);
    v.head(S).setOnes();
    Eigen::VectorXd u = llt.solve(v);
    double vtu = v.dot(u);

    result.log_det = -0.5 * (log_det_A + std::log(1.0 + lambda_s2z * vtu));

    // A⁻¹
    Eigen::MatrixXd Ainv = llt.solve(Eigen::MatrixXd::Identity(dim, dim));

    // Woodbury H⁻¹ entries: Hinv(i,j) = Ainv(i,j) - scale * u(i)*u(j)
    double woodbury_scale = lambda_s2z / (1.0 + lambda_s2z * vtu);

    // tr(H⁻¹ dH/d(log_sigma)):
    double tr_sigma = 0.0;
    for (int i = 0; i < S; i++) {
        double w = ws.W_data[i];
        double Hinv_ii = Ainv(i, i) - woodbury_scale * u(i) * u(i);
        double Hinv_iSi = Ainv(i, S + i) - woodbury_scale * u(i) * u(S + i);
        double Hinv_SiSi = Ainv(S + i, S + i) - woodbury_scale * u(S + i) * u(S + i);
        tr_sigma += w * (2.0*a2 * Hinv_ii + 4.0*ac * Hinv_iSi + 2.0*c2 * Hinv_SiSi);
    }
    result.grad_log_sigma = -0.5 * tr_sigma;

    // tr(H⁻¹ dH/d(logit_rho)):
    double half_1m2r = (1.0 - 2.0 * rho) / 2.0;
    double tr_rho = 0.0;
    for (int i = 0; i < S; i++) {
        double w = ws.W_data[i];
        double Hinv_ii = Ainv(i, i) - woodbury_scale * u(i) * u(i);
        double Hinv_iSi = Ainv(i, S + i) - woodbury_scale * u(i) * u(S + i);
        double Hinv_SiSi = Ainv(S + i, S + i) - woodbury_scale * u(S + i) * u(S + i);
        tr_rho += w * (a2*(1.0-rho) * Hinv_ii
                     + 2.0*ac*half_1m2r * Hinv_iSi
                     - c2*rho * Hinv_SiSi);
    }
    result.grad_logit_rho = -0.5 * tr_rho;

    return result;
}

// =========================================================================
// Per-spatial-unit likelihood (aggregated over observations at each unit)
// =========================================================================

struct UnitLikResult {
    double ll;          // log-likelihood contribution
    double grad;        // d(ll)/d(spatial_effect)
    double neg_hess;    // -d²(ll)/d(spatial_effect)²
};

// Compute data log-likelihood, gradient, and Hessian at spatial unit s,
// where spatial_eff[s] enters eta for all observations at unit s.
// RE effects are included via re_vals (pre-computed actual RE values).
// Fast version: uses pre-filtered obs list instead of scanning all N observations.
inline UnitLikResult compute_unit_lik(
    int s,                      // spatial unit index (0-based)
    double spatial_eff,         // spatial effect at this unit
    const double* beta_num, const double* beta_denom,
    double phi_num, double phi_denom,
    const double* re_vals,      // pre-computed RE values (actual, not z), length n_re_groups or NULL
    const ModelData& data,
    bool is_binomial,
    const int* obs_list = nullptr,  // pre-filtered obs indices at this unit (or NULL)
    int n_obs = -1                  // number of obs at this unit (-1 = scan all)
) {
    UnitLikResult res = {0.0, 0.0, 0.0};
    int N = (n_obs >= 0) ? n_obs : data.N;

    for (int idx = 0; idx < N; idx++) {
        int i = (obs_list != nullptr) ? obs_list[idx] : idx;
        if (obs_list == nullptr && data.spatial_group[i] - 1 != s) continue;

        // Compute eta
        double eta_num_i = 0.0, eta_denom_i = 0.0;
        for (int p = 0; p < data.legacy.p_num; p++)
            eta_num_i += data.legacy.X_num_flat[i * data.legacy.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < data.legacy.p_denom; p++)
                eta_denom_i += data.legacy.X_denom_flat[i * data.legacy.p_denom + p] * beta_denom[p];
        }

        // Add spatial effect
        eta_num_i += spatial_eff;
        if (!is_binomial) eta_denom_i += spatial_eff;  // shared by default

        // Add RE
        if (re_vals != nullptr && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
            double re_val = re_vals[data.re_group[i] - 1];
            eta_num_i += re_val;
            if (!is_binomial) eta_denom_i += re_val;
        }

        // Per-family likelihood, gradient, Hessian
        double mu_num = std::exp(std::min(eta_num_i, 20.0));
        int y_num = data.legacy.y_num[i];

        switch (data.legacy.model_type) {
            case ModelType::POISSON_GAMMA: {
                res.ll += y_num * eta_num_i - mu_num;
                res.grad += y_num - mu_num;
                res.neg_hess += mu_num;
                if (!is_binomial) {
                    double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                    double y_denom = data.legacy.y_denom_cont[i];
                    double shape = phi_denom;
                    res.ll += shape * std::log(shape) - std::lgamma(shape)
                              + (shape - 1.0) * std::log(std::max(y_denom, 1e-10))
                              - shape * eta_denom_i - shape * y_denom / mu_denom;
                    res.grad += shape * (y_denom / mu_denom - 1.0);
                    res.neg_hess += shape * y_denom / mu_denom;
                }
                break;
            }
            case ModelType::NEGBIN_NEGBIN: {
                double r_num = phi_num;
                res.ll += std::lgamma(y_num + r_num) - std::lgamma(r_num) - std::lgamma(y_num + 1)
                          + y_num * eta_num_i - (y_num + r_num) * std::log(mu_num + r_num)
                          + r_num * std::log(r_num);
                double resid_num = y_num - mu_num * (y_num + r_num) / (mu_num + r_num);
                res.grad += resid_num;
                res.neg_hess += mu_num * r_num * (y_num + r_num) / ((mu_num + r_num) * (mu_num + r_num));
                if (!is_binomial) {
                    double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                    int y_denom = (int)data.legacy.y_denom[i];
                    double r_denom = phi_denom;
                    res.ll += std::lgamma(y_denom + r_denom) - std::lgamma(r_denom) - std::lgamma(y_denom + 1)
                              + y_denom * eta_denom_i - (y_denom + r_denom) * std::log(mu_denom + r_denom)
                              + r_denom * std::log(r_denom);
                    double resid_denom = y_denom - mu_denom * (y_denom + r_denom) / (mu_denom + r_denom);
                    res.grad += resid_denom;
                    res.neg_hess += mu_denom * r_denom * (y_denom + r_denom) / ((mu_denom + r_denom) * (mu_denom + r_denom));
                }
                break;
            }
            case ModelType::NEGBIN_GAMMA: {
                double r_num = phi_num;
                res.ll += std::lgamma(y_num + r_num) - std::lgamma(r_num) - std::lgamma(y_num + 1)
                          + y_num * eta_num_i - (y_num + r_num) * std::log(mu_num + r_num)
                          + r_num * std::log(r_num);
                double resid_num = y_num - mu_num * (y_num + r_num) / (mu_num + r_num);
                res.grad += resid_num;
                res.neg_hess += mu_num * r_num * (y_num + r_num) / ((mu_num + r_num) * (mu_num + r_num));
                // Gamma denominator
                {
                    double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                    double y_denom = data.legacy.y_denom_cont[i];
                    double shape = phi_denom;
                    res.ll += shape * std::log(shape) - std::lgamma(shape)
                              + (shape - 1.0) * std::log(std::max(y_denom, 1e-10))
                              - shape * eta_denom_i - shape * y_denom / mu_denom;
                    res.grad += shape * (y_denom / mu_denom - 1.0);
                    res.neg_hess += shape * y_denom / mu_denom;
                }
                break;
            }
            case ModelType::BINOMIAL: {
                int n_trials = (int)data.legacy.y_denom[i];
                double p_i = 1.0 / (1.0 + std::exp(-eta_num_i));
                res.ll += y_num * eta_num_i - n_trials * std::log(1.0 + std::exp(eta_num_i));
                res.grad += y_num - n_trials * p_i;
                res.neg_hess += n_trials * p_i * (1.0 - p_i);
                break;
            }
            default:
                res.ll += y_num * eta_num_i - mu_num;
                res.grad += y_num - mu_num;
                res.neg_hess += mu_num;
                break;
        }
    }
    return res;
}

// =========================================================================
// Newton-Raphson for finding phi* (ICAR inner optimization)
// =========================================================================

// Find phi* = argmax [LL(y|phi,outer) + (-0.5*tau*phi'Q*phi) + (-0.5*lambda*(sum phi)^2)]
// Returns data log-likelihood + ICAR prior at phi* (not including outer param priors)
inline double collapsed_icar_find_mode(
    const double* beta_num, const double* beta_denom,
    double tau,
    double phi_num, double phi_denom,
    const double* re_vals,      // pre-computed RE values or NULL
    const ModelData& data,
    CollapsedICARWorkspace& ws,
    int max_newton = 20,
    double newton_tol = 1e-6
) {
    int S = data.n_spatial_units;
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    ws.init(S, false);

    // Build unit→obs mapping (cached, O(N) first time, O(1) after)
    build_unit_obs_map(ws, data);

    // Warm-start from previous mode
    if (!ws.mode_found) {
        std::fill(ws.phi_star.begin(), ws.phi_star.end(), 0.0);
    } else {
        for (int i = 0; i < S; i++) {
            if (std::isnan(ws.phi_star[i]) || std::isinf(ws.phi_star[i])) {
                std::fill(ws.phi_star.begin(), ws.phi_star.end(), 0.0);
                break;
            }
        }
    }

    std::vector<double> Qphi(S);
    std::vector<double> delta(S, 0.0);

    for (int newton_iter = 0; newton_iter < max_newton; newton_iter++) {
        // Compute per-unit data likelihood, gradient, and Hessian
        std::fill(ws.grad.begin(), ws.grad.end(), 0.0);
        std::fill(ws.W_data.begin(), ws.W_data.end(), 0.0);

        for (int s = 0; s < S; s++) {
            int n_obs_s = ws.unit_obs_ptr[s + 1] - ws.unit_obs_ptr[s];
            const int* obs_s = &ws.unit_obs_idx[ws.unit_obs_ptr[s]];
            UnitLikResult lr = compute_unit_lik(s, ws.phi_star[s],
                                                 beta_num, beta_denom,
                                                 phi_num, phi_denom,
                                                 re_vals, data, is_binomial,
                                                 obs_s, n_obs_s);
            ws.grad[s] = lr.grad;
            ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
        }

        // Add ICAR prior gradient: -tau * Q * phi
        icar_precision_matvec(ws.phi_star.data(), Qphi.data(), S,
                              data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
        double sum_phi = 0.0;
        for (int i = 0; i < S; i++) sum_phi += ws.phi_star[i];
        for (int i = 0; i < S; i++) {
            ws.grad[i] -= tau * Qphi[i] + 0.001 * sum_phi;  // ICAR + sum-to-zero
        }

        // Check convergence
        double grad_norm = 0.0;
        for (int i = 0; i < S; i++) grad_norm += ws.grad[i] * ws.grad[i];
        grad_norm = std::sqrt(grad_norm);

        if (grad_norm < newton_tol) break;

        // Solve (W + tau*Q + lambda_s2z*11^T) delta = grad via CG
        std::fill(delta.begin(), delta.end(), 0.0);
        icar_cg_solve(delta.data(), ws.grad.data(), ws, tau, 0.0, 0.0, data, 100, 1e-8);

        for (int i = 0; i < S; i++) ws.phi_star[i] += delta[i];
    }

    ws.mode_found = true;

    // Compute log-posterior at phi*
    double data_ll = 0.0;
    for (int s = 0; s < S; s++) {
        int n_obs_s = ws.unit_obs_ptr[s + 1] - ws.unit_obs_ptr[s];
        const int* obs_s = &ws.unit_obs_idx[ws.unit_obs_ptr[s]];
        UnitLikResult lr = compute_unit_lik(s, ws.phi_star[s],
                                             beta_num, beta_denom,
                                             phi_num, phi_denom,
                                             re_vals, data, is_binomial,
                                             obs_s, n_obs_s);
        data_ll += lr.ll;
        ws.W_data[s] = std::max(lr.neg_hess, 1e-8);  // Update for Laplace
    }

    // ICAR prior: -0.5 * tau * phi' Q phi + 0.5*(S-1)*log(tau) - 0.5*0.001*(sum phi)^2
    icar_precision_matvec(ws.phi_star.data(), Qphi.data(), S,
                          data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
    double phiQphi = 0.0;
    for (int i = 0; i < S; i++) phiQphi += ws.phi_star[i] * Qphi[i];
    double sum_phi = 0.0;
    for (int i = 0; i < S; i++) sum_phi += ws.phi_star[i];
    double icar_prior = -0.5 * tau * phiQphi + 0.5 * (S - 1) * std::log(tau)
                        - 0.5 * 0.001 * sum_phi * sum_phi;

    // Laplace correction
    ws.laplace_log_det = compute_laplace_log_det_icar(ws, tau, data);

    return data_ll + icar_prior;
}

// =========================================================================
// BYM2 mode: Newton for (phi*, theta*)
// =========================================================================

inline double collapsed_bym2_find_mode(
    const double* beta_num, const double* beta_denom,
    double sigma_total, double rho, double scale_factor,
    double phi_num, double phi_denom,
    const double* re_vals,
    const ModelData& data,
    CollapsedICARWorkspace& ws,
    int max_newton = 20,
    double newton_tol = 1e-6
) {
    int S = data.n_spatial_units;
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    double sigma_s = sigma_total * std::sqrt(rho);
    double sigma_u = sigma_total * std::sqrt(1.0 - rho);
    double a = sigma_s * scale_factor;  // coefficient for phi
    double c = sigma_u;                  // coefficient for theta

    ws.init(S, true);

    // Build unit→obs mapping (cached)
    build_unit_obs_map(ws, data);

    // Warm-start
    if (!ws.mode_found) {
        std::fill(ws.phi_star.begin(), ws.phi_star.end(), 0.0);
        std::fill(ws.theta_star.begin(), ws.theta_star.end(), 0.0);
    } else {
        bool has_nan = false;
        for (int i = 0; i < S; i++) {
            if (std::isnan(ws.phi_star[i]) || std::isinf(ws.phi_star[i]) ||
                std::isnan(ws.theta_star[i]) || std::isinf(ws.theta_star[i])) {
                has_nan = true;
                break;
            }
        }
        if (has_nan) {
            std::fill(ws.phi_star.begin(), ws.phi_star.end(), 0.0);
            std::fill(ws.theta_star.begin(), ws.theta_star.end(), 0.0);
        }
    }

    // Combined inner variable: [phi; theta]
    std::vector<double> inner(2 * S, 0.0);
    for (int i = 0; i < S; i++) {
        inner[i] = ws.phi_star[i];
        inner[S + i] = ws.theta_star[i];
    }

    std::vector<double> Qphi(S);
    std::vector<double> delta(2 * S, 0.0);

    for (int newton_iter = 0; newton_iter < max_newton; newton_iter++) {
        // Compute spatial effect: b_s = a*phi_s + c*theta_s
        std::fill(ws.grad.begin(), ws.grad.end(), 0.0);
        std::fill(ws.W_data.begin(), ws.W_data.end(), 0.0);

        for (int s = 0; s < S; s++) {
            double b_s = a * inner[s] + c * inner[S + s];
            int n_obs_s = ws.unit_obs_ptr[s + 1] - ws.unit_obs_ptr[s];
            const int* obs_s = &ws.unit_obs_idx[ws.unit_obs_ptr[s]];
            UnitLikResult lr = compute_unit_lik(s, b_s,
                                                 beta_num, beta_denom,
                                                 phi_num, phi_denom,
                                                 re_vals, data, is_binomial,
                                                 obs_s, n_obs_s);
            // Data gradients w.r.t. phi and theta (chain rule through b_s)
            ws.grad[s] = lr.grad * a;          // dLL/dphi_s = dLL/db * a
            ws.grad[S + s] = lr.grad * c;      // dLL/dtheta_s = dLL/db * c
            ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
        }

        // Add prior gradients
        // phi: ICAR prior -0.5*phi'Q*phi → gradient = -Q*phi
        icar_precision_matvec(inner.data(), Qphi.data(), S,
                              data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
        double sum_phi = 0.0;
        for (int i = 0; i < S; i++) sum_phi += inner[i];
        for (int i = 0; i < S; i++) {
            ws.grad[i] -= Qphi[i] + 0.001 * sum_phi;  // ICAR + sum-to-zero
        }
        // theta: IID N(0,1) → gradient = -theta
        for (int i = 0; i < S; i++) {
            ws.grad[S + i] -= inner[S + i];
        }

        // Check convergence
        double grad_norm = 0.0;
        for (int i = 0; i < 2 * S; i++) grad_norm += ws.grad[i] * ws.grad[i];
        grad_norm = std::sqrt(grad_norm);

        if (grad_norm < newton_tol) break;

        // CG solve for 2S system
        std::fill(delta.begin(), delta.end(), 0.0);
        icar_cg_solve(delta.data(), ws.grad.data(), ws, 1.0, a, c, data, 100, 1e-8);

        for (int i = 0; i < 2 * S; i++) inner[i] += delta[i];
    }

    // Store back
    for (int i = 0; i < S; i++) {
        ws.phi_star[i] = inner[i];
        ws.theta_star[i] = inner[S + i];
    }
    ws.mode_found = true;

    // Compute log-posterior at mode
    double data_ll = 0.0;
    for (int s = 0; s < S; s++) {
        double b_s = a * ws.phi_star[s] + c * ws.theta_star[s];
        int n_obs_s = ws.unit_obs_ptr[s + 1] - ws.unit_obs_ptr[s];
        const int* obs_s = &ws.unit_obs_idx[ws.unit_obs_ptr[s]];
        UnitLikResult lr = compute_unit_lik(s, b_s,
                                             beta_num, beta_denom,
                                             phi_num, phi_denom,
                                             re_vals, data, is_binomial,
                                             obs_s, n_obs_s);
        data_ll += lr.ll;
        ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
    }

    // ICAR prior on phi: -0.5 * phi' Q phi - 0.5*0.001*(sum phi)^2
    icar_precision_matvec(ws.phi_star.data(), Qphi.data(), S,
                          data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
    double phiQphi = 0.0;
    double sum_phi = 0.0;
    for (int i = 0; i < S; i++) {
        phiQphi += ws.phi_star[i] * Qphi[i];
        sum_phi += ws.phi_star[i];
    }
    double phi_prior = -0.5 * phiQphi - 0.5 * 0.001 * sum_phi * sum_phi;

    // IID prior on theta: -0.5 * sum(theta^2)
    double theta_prior = 0.0;
    for (int i = 0; i < S; i++) {
        theta_prior -= 0.5 * ws.theta_star[i] * ws.theta_star[i];
    }

    // Laplace correction
    ws.laplace_log_det = compute_laplace_log_det_bym2(ws, a, c, data);

    return data_ll + phi_prior + theta_prior;
}

// =========================================================================
// Residual computation at mode (for scattering to outer param gradients)
// =========================================================================

// Compute per-observation residuals dLL/deta at the spatial mode
inline void collapsed_icar_compute_residuals(
    const CollapsedICARWorkspace& ws,
    const double* beta_num, const double* beta_denom,
    double phi_num, double phi_denom,
    const double* re_vals,
    double a_bym2, double c_bym2,  // BYM2 scaling factors (a=0, c=0 for ICAR)
    const ModelData& data,
    double* resid_num,  // length N
    double* resid_denom // length N
) {
    int N = data.N;
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    for (int i = 0; i < N; i++) {
        int s = data.spatial_group[i] - 1;  // 0-based

        // Spatial effect at this unit
        double spatial_eff;
        if (ws.is_bym2) {
            spatial_eff = a_bym2 * ws.phi_star[s] + c_bym2 * ws.theta_star[s];
        } else {
            spatial_eff = ws.phi_star[s];
        }

        double eta_num_i = 0.0, eta_denom_i = 0.0;
        for (int p = 0; p < data.legacy.p_num; p++)
            eta_num_i += data.legacy.X_num_flat[i * data.legacy.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < data.legacy.p_denom; p++)
                eta_denom_i += data.legacy.X_denom_flat[i * data.legacy.p_denom + p] * beta_denom[p];
        }

        eta_num_i += spatial_eff;
        if (!is_binomial) eta_denom_i += spatial_eff;

        if (re_vals != nullptr && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
            eta_num_i += re_vals[data.re_group[i] - 1];
            if (!is_binomial) eta_denom_i += re_vals[data.re_group[i] - 1];
        }

        double mu_num = std::exp(std::min(eta_num_i, 20.0));

        switch (data.legacy.model_type) {
            case ModelType::POISSON_GAMMA: {
                resid_num[i] = data.legacy.y_num[i] - mu_num;
                if (!is_binomial) {
                    double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                    resid_denom[i] = phi_denom * (data.legacy.y_denom_cont[i] / mu_denom - 1.0);
                } else {
                    resid_denom[i] = 0.0;
                }
                break;
            }
            case ModelType::NEGBIN_NEGBIN: {
                double r_num = phi_num;
                resid_num[i] = data.legacy.y_num[i] - mu_num * (data.legacy.y_num[i] + r_num) / (mu_num + r_num);
                if (!is_binomial) {
                    double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                    int y_denom = (int)data.legacy.y_denom[i];
                    double r_denom = phi_denom;
                    resid_denom[i] = y_denom - mu_denom * (y_denom + r_denom) / (mu_denom + r_denom);
                } else {
                    resid_denom[i] = 0.0;
                }
                break;
            }
            case ModelType::NEGBIN_GAMMA: {
                double r_num = phi_num;
                resid_num[i] = data.legacy.y_num[i] - mu_num * (data.legacy.y_num[i] + r_num) / (mu_num + r_num);
                double mu_denom = std::exp(std::min(eta_denom_i, 20.0));
                double shape = phi_denom;
                resid_denom[i] = shape * (data.legacy.y_denom_cont[i] / mu_denom - 1.0);
                break;
            }
            case ModelType::BINOMIAL: {
                int n_trials = (int)data.legacy.y_denom[i];
                double p_i = 1.0 / (1.0 + std::exp(-eta_num_i));
                resid_num[i] = data.legacy.y_num[i] - n_trials * p_i;
                resid_denom[i] = 0.0;
                break;
            }
            default:
                resid_num[i] = data.legacy.y_num[i] - mu_num;
                resid_denom[i] = 0.0;
                break;
        }
    }
}

// =========================================================================
// Full Laplace log-det with Newton re-solve (for numerical gradient)
// =========================================================================

// Compute Laplace log-det for given params, with warm-started Newton from phi*
// Used for central-difference numerical gradient of the Laplace correction.
inline double laplace_log_det_icar_full(
    const double* beta_num, const double* beta_denom,
    double tau,
    double phi_num, double phi_denom,
    const double* re_vals,
    const ModelData& data,
    const std::vector<double>& warm_phi,        // warm start
    const std::vector<double>& warm_theta = {}  // for BYM2
) {
    int S = data.n_spatial_units;
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    // Temporary workspace with warm start
    CollapsedICARWorkspace temp_ws;
    temp_ws.init(S, !warm_theta.empty());
    build_unit_obs_map(temp_ws, data);
    temp_ws.phi_star = warm_phi;
    if (!warm_theta.empty()) temp_ws.theta_star = warm_theta;
    temp_ws.mode_found = true;

    // Short Newton (warm-started, 5 iters max)
    if (temp_ws.is_bym2) {
        // For BYM2 we need sigma_total, rho, scale from the caller
        // This function is ICAR-only; BYM2 uses laplace_log_det_bym2_full
        return 0.0;
    }

    std::vector<double> Qphi(S);
    std::vector<double> delta(S, 0.0);

    for (int iter = 0; iter < 5; iter++) {
        std::fill(temp_ws.grad.begin(), temp_ws.grad.end(), 0.0);
        std::fill(temp_ws.W_data.begin(), temp_ws.W_data.end(), 0.0);

        for (int s = 0; s < S; s++) {
            int n_obs_s = temp_ws.unit_obs_ptr[s + 1] - temp_ws.unit_obs_ptr[s];
            const int* obs_s = &temp_ws.unit_obs_idx[temp_ws.unit_obs_ptr[s]];
            UnitLikResult lr = compute_unit_lik(s, temp_ws.phi_star[s],
                                                 beta_num, beta_denom,
                                                 phi_num, phi_denom,
                                                 re_vals, data, is_binomial,
                                                 obs_s, n_obs_s);
            temp_ws.grad[s] = lr.grad;
            temp_ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
        }

        icar_precision_matvec(temp_ws.phi_star.data(), Qphi.data(), S,
                              data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
        double sum_phi = 0.0;
        for (int i = 0; i < S; i++) sum_phi += temp_ws.phi_star[i];
        for (int i = 0; i < S; i++) {
            temp_ws.grad[i] -= tau * Qphi[i] + 0.001 * sum_phi;
        }

        double grad_norm = 0.0;
        for (int i = 0; i < S; i++) grad_norm += temp_ws.grad[i] * temp_ws.grad[i];
        if (std::sqrt(grad_norm) < 1e-6) break;

        std::fill(delta.begin(), delta.end(), 0.0);
        icar_cg_solve(delta.data(), temp_ws.grad.data(), temp_ws, tau, 0.0, 0.0, data, 50, 1e-8);
        for (int i = 0; i < S; i++) temp_ws.phi_star[i] += delta[i];
    }

    // Final Hessian at new mode
    for (int s = 0; s < S; s++) {
        int n_obs_s = temp_ws.unit_obs_ptr[s + 1] - temp_ws.unit_obs_ptr[s];
        const int* obs_s = &temp_ws.unit_obs_idx[temp_ws.unit_obs_ptr[s]];
        UnitLikResult lr = compute_unit_lik(s, temp_ws.phi_star[s],
                                             beta_num, beta_denom,
                                             phi_num, phi_denom,
                                             re_vals, data, is_binomial,
                                             obs_s, n_obs_s);
        temp_ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
    }

    return compute_laplace_log_det_icar(temp_ws, tau, data);
}

// BYM2 version of full Laplace log-det with Newton
inline double laplace_log_det_bym2_full(
    const double* beta_num, const double* beta_denom,
    double sigma_total, double rho, double scale_factor,
    double phi_num, double phi_denom,
    const double* re_vals,
    const ModelData& data,
    const std::vector<double>& warm_phi,
    const std::vector<double>& warm_theta
) {
    int S = data.n_spatial_units;
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    double sigma_s = sigma_total * std::sqrt(rho);
    double sigma_u = sigma_total * std::sqrt(1.0 - rho);
    double a = sigma_s * scale_factor;
    double c = sigma_u;

    CollapsedICARWorkspace temp_ws;
    temp_ws.init(S, true);
    build_unit_obs_map(temp_ws, data);
    temp_ws.phi_star = warm_phi;
    temp_ws.theta_star = warm_theta;
    temp_ws.mode_found = true;

    // Combined inner variable
    std::vector<double> inner(2 * S);
    for (int i = 0; i < S; i++) {
        inner[i] = temp_ws.phi_star[i];
        inner[S + i] = temp_ws.theta_star[i];
    }

    std::vector<double> Qphi(S);
    std::vector<double> delta(2 * S, 0.0);

    for (int iter = 0; iter < 5; iter++) {
        std::fill(temp_ws.grad.begin(), temp_ws.grad.end(), 0.0);
        std::fill(temp_ws.W_data.begin(), temp_ws.W_data.end(), 0.0);

        for (int s = 0; s < S; s++) {
            double b_s = a * inner[s] + c * inner[S + s];
            int n_obs_s = temp_ws.unit_obs_ptr[s + 1] - temp_ws.unit_obs_ptr[s];
            const int* obs_s = &temp_ws.unit_obs_idx[temp_ws.unit_obs_ptr[s]];
            UnitLikResult lr = compute_unit_lik(s, b_s,
                                                 beta_num, beta_denom,
                                                 phi_num, phi_denom,
                                                 re_vals, data, is_binomial,
                                                 obs_s, n_obs_s);
            temp_ws.grad[s] = lr.grad * a;
            temp_ws.grad[S + s] = lr.grad * c;
            temp_ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
        }

        icar_precision_matvec(inner.data(), Qphi.data(), S,
                              data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
        double sum_phi = 0.0;
        for (int i = 0; i < S; i++) sum_phi += inner[i];
        for (int i = 0; i < S; i++) {
            temp_ws.grad[i] -= Qphi[i] + 0.001 * sum_phi;
            temp_ws.grad[S + i] -= inner[S + i];
        }

        double grad_norm = 0.0;
        for (int i = 0; i < 2 * S; i++) grad_norm += temp_ws.grad[i] * temp_ws.grad[i];
        if (std::sqrt(grad_norm) < 1e-6) break;

        std::fill(delta.begin(), delta.end(), 0.0);
        icar_cg_solve(delta.data(), temp_ws.grad.data(), temp_ws, 1.0, a, c, data, 50, 1e-8);
        for (int i = 0; i < 2 * S; i++) inner[i] += delta[i];
    }

    // Update W_data at final mode
    for (int s = 0; s < S; s++) {
        double b_s = a * inner[s] + c * inner[S + s];
        int n_obs_s = temp_ws.unit_obs_ptr[s + 1] - temp_ws.unit_obs_ptr[s];
        const int* obs_s = &temp_ws.unit_obs_idx[temp_ws.unit_obs_ptr[s]];
        UnitLikResult lr = compute_unit_lik(s, b_s,
                                             beta_num, beta_denom,
                                             phi_num, phi_denom,
                                             re_vals, data, is_binomial,
                                             obs_s, n_obs_s);
        temp_ws.W_data[s] = std::max(lr.neg_hess, 1e-8);
    }

    return compute_laplace_log_det_bym2(temp_ws, a, c, data);
}

// =========================================================================
// H-mode: Analytical Laplace gradient for collapsed ICAR
//
// Eliminates the 2*p Newton re-solves of the numerical approach.
// Uses implicit function theorem + per-family third derivatives.
//
// Formula: d/dθⱼ[-0.5 log det H] = -0.5 * tr(H⁻¹ dH/dθⱼ)
// where dH/dθⱼ = diag(∂W/∂θⱼ + dW/dφ · dφ*/dθⱼ) + δ(j=log_τ)·τ·Q
// and dφ*/dθⱼ = H⁻¹ · cross_hessⱼ (implicit function theorem)
//
// Cost: O(S³) Cholesky + O(N) per-obs derivs + O(p·S) assembly
// vs numerical: O(p · S · Newton_iters · N/S)
// =========================================================================

struct LaplaceGradFullResult {
    double log_det;                    // -0.5 * log det(H)
    std::vector<double> laplace_grad;  // length n_params
    bool success;
};

inline LaplaceGradFullResult compute_laplace_gradient_icar_H(
    const CollapsedICARWorkspace& ws,
    const double* beta_num, const double* beta_denom,
    double tau, double phi_num, double phi_denom,
    const double* re_vals,
    const ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int n_params,
    double lambda_s2z = 0.001
) {
    int S = ws.S;
    int N = data.N;
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    LaplaceGradFullResult result;
    result.laplace_grad.assign(n_params, 0.0);
    result.success = false;
    result.log_det = 0.0;

    // ---- Phase 1: Sparse A + Woodbury for rank-1 sum-to-zero ----
    // H = A + λ·1·1'  where  A = diag(W_data) + tau*Q  (sparse)
    // H⁻¹ = A⁻¹ - ws·(A⁻¹·1)(A⁻¹·1)'  where ws = λ/(1 + λ·1'A⁻¹·1)
    typedef Eigen::Triplet<double> T_icar;
    std::vector<T_icar> triplets;
    triplets.reserve(S + 2 * (int)data.adj_col_idx.size());
    for (int i = 0; i < S; i++) {
        triplets.push_back(T_icar(i, i, ws.W_data[i] + tau * data.n_neighbors[i]));
    }
    for (int i = 0; i < S; i++) {
        for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; k++) {
            triplets.push_back(T_icar(i, data.adj_col_idx[k], -tau));
        }
    }
    Eigen::SparseMatrix<double> A_sp(S, S);
    A_sp.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> sp_llt;
    sp_llt.compute(A_sp);
    if (sp_llt.info() != Eigen::Success) {
        for (int i = 0; i < S; i++)
            triplets.push_back(T_icar(i, i, 1e-6));
        A_sp.setFromTriplets(triplets.begin(), triplets.end());
        sp_llt.compute(A_sp);
        if (sp_llt.info() != Eigen::Success) return result;
    }

    // Log-det(A) from sparse Cholesky
    Eigen::SparseMatrix<double> L_sp = sp_llt.matrixL();
    double log_det_A = 0.0;
    for (int i = 0; i < S; i++)
        log_det_A += std::log(L_sp.coeff(i, i));
    log_det_A *= 2.0;

    // Woodbury quantities: u_ones = A⁻¹·1, scale factor
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(S);
    Eigen::VectorXd u_ones = sp_llt.solve(ones);
    double one_t_u = ones.dot(u_ones);
    double wb_scale = lambda_s2z / (1.0 + lambda_s2z * one_t_u);

    // Log-det with Woodbury: log det(H) = log det(A) + log(1 + λ·1'A⁻¹·1)
    result.log_det = -0.5 * (log_det_A + std::log(1.0 + lambda_s2z * one_t_u));

    // Compute Ainv_diag and tr(A⁻¹·Q) via S column solves
    Eigen::VectorXd Ainv_diag(S);
    double trAinvQ = 0.0;
    Eigen::VectorXd e_s = Eigen::VectorXd::Zero(S);
    for (int s = 0; s < S; s++) {
        e_s(s) = 1.0;
        Eigen::VectorXd col_s = sp_llt.solve(e_s);
        Ainv_diag(s) = col_s(s);
        // tr(A⁻¹·Q) contribution: Q[s,:] · col_s = n_s·col_s[s] - Σ_{j∈adj(s)} col_s[j]
        trAinvQ += data.n_neighbors[s] * col_s(s);
        for (int k = data.adj_row_ptr[s]; k < data.adj_row_ptr[s + 1]; k++) {
            trAinvQ -= col_s(data.adj_col_idx[k]);
        }
        e_s(s) = 0.0;
    }

    // Woodbury-corrected diagonal: Hinv_diag[s] = Ainv_diag[s] - ws·u_ones[s]²
    Eigen::VectorXd Hinv_diag(S);
    for (int s = 0; s < S; s++) {
        Hinv_diag(s) = Ainv_diag(s) - wb_scale * u_ones(s) * u_ones(s);
    }

    // ---- Phase 2: Per-obs derivatives, aggregate per-site ----
    std::vector<double> dWdphi(S, 0.0);

    // Per-obs coefficients (will be finalized after u is computed)
    std::vector<double> obs_dw_deta_num(N, 0.0);
    std::vector<double> obs_dw_deta_den(N, 0.0);
    std::vector<double> obs_w_num(N, 0.0);
    std::vector<double> obs_w_den(N, 0.0);

    // Per-site dispersion aggregates
    std::vector<double> dW_dphi_num_site(S, 0.0);
    std::vector<double> dW_dphi_denom_site(S, 0.0);
    std::vector<double> cross_phi_num_site(S, 0.0);
    std::vector<double> cross_phi_denom_site(S, 0.0);

    for (int i = 0; i < N; i++) {
        int s = data.spatial_group[i] - 1;

        // Compute eta at mode
        double eta_num_i = 0.0, eta_denom_i = 0.0;
        for (int p = 0; p < data.legacy.p_num; p++)
            eta_num_i += data.legacy.X_num_flat[i * data.legacy.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < data.legacy.p_denom; p++)
                eta_denom_i += data.legacy.X_denom_flat[i * data.legacy.p_denom + p] * beta_denom[p];
        }
        eta_num_i += ws.phi_star[s];
        if (!is_binomial) eta_denom_i += ws.phi_star[s];
        if (re_vals && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
            eta_num_i += re_vals[data.re_group[i] - 1];
            if (!is_binomial) eta_denom_i += re_vals[data.re_group[i] - 1];
        }

        double mu_num = std::exp(std::min(eta_num_i, 20.0));
        double w_n = 0.0, w_d = 0.0;
        double dw_eta_n = 0.0, dw_eta_d = 0.0;
        double dw_phi_n = 0.0, dw_phi_d = 0.0;
        double dr_phi_n = 0.0, dr_phi_d = 0.0;

        switch (data.legacy.model_type) {
            case ModelType::POISSON_GAMMA: {
                w_n = mu_num;
                dw_eta_n = mu_num;
                if (!is_binomial) {
                    double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                    double y_d = data.legacy.y_denom_cont[i];
                    double alpha = phi_denom;
                    w_d = alpha * y_d / mu_d;
                    dw_eta_d = -w_d;
                    dw_phi_d = y_d / mu_d;
                    dr_phi_d = y_d / mu_d - 1.0;
                }
                break;
            }
            case ModelType::NEGBIN_NEGBIN: {
                double r_n = phi_num;
                double y_n = data.legacy.y_num[i];
                double A = mu_num + r_n, B = y_n + r_n;
                double A2 = A * A, A3 = A2 * A;
                w_n = mu_num * r_n * B / A2;
                dw_eta_n = mu_num * r_n * B * (r_n - mu_num) / A3;
                dw_phi_n = mu_num * (mu_num * (y_n + 2*r_n) - y_n * r_n) / A3;
                dr_phi_n = mu_num * (y_n - mu_num) / A2;
                if (!is_binomial) {
                    double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                    double y_d = (double)data.legacy.y_denom[i];
                    double r_d = phi_denom;
                    double Ad = mu_d + r_d, Bd = y_d + r_d;
                    double Ad2 = Ad * Ad, Ad3 = Ad2 * Ad;
                    w_d = mu_d * r_d * Bd / Ad2;
                    dw_eta_d = mu_d * r_d * Bd * (r_d - mu_d) / Ad3;
                    dw_phi_d = mu_d * (mu_d * (y_d + 2*r_d) - y_d * r_d) / Ad3;
                    dr_phi_d = mu_d * (y_d - mu_d) / Ad2;
                }
                break;
            }
            case ModelType::NEGBIN_GAMMA: {
                double r_n = phi_num;
                double y_n = data.legacy.y_num[i];
                double A = mu_num + r_n, B = y_n + r_n;
                double A2 = A * A, A3 = A2 * A;
                w_n = mu_num * r_n * B / A2;
                dw_eta_n = mu_num * r_n * B * (r_n - mu_num) / A3;
                dw_phi_n = mu_num * (mu_num * (y_n + 2*r_n) - y_n * r_n) / A3;
                dr_phi_n = mu_num * (y_n - mu_num) / A2;
                {
                    double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                    double y_d = data.legacy.y_denom_cont[i];
                    double alpha = phi_denom;
                    w_d = alpha * y_d / mu_d;
                    dw_eta_d = -w_d;
                    dw_phi_d = y_d / mu_d;
                    dr_phi_d = y_d / mu_d - 1.0;
                }
                break;
            }
            case ModelType::BINOMIAL: {
                int n_trials = (int)data.legacy.y_denom[i];
                double p_i = 1.0 / (1.0 + std::exp(-eta_num_i));
                w_n = n_trials * p_i * (1.0 - p_i);
                dw_eta_n = w_n * (1.0 - 2.0 * p_i);
                break;
            }
            default: {
                w_n = mu_num;
                dw_eta_n = mu_num;
                break;
            }
        }

        // Per-site dW/dphi (spatial enters both eta_num and eta_denom with coeff 1)
        dWdphi[s] += dw_eta_n + dw_eta_d;

        // Store per-obs
        obs_dw_deta_num[i] = dw_eta_n;
        obs_dw_deta_den[i] = dw_eta_d;
        obs_w_num[i] = w_n;
        obs_w_den[i] = w_d;

        // Dispersion aggregates per site
        dW_dphi_num_site[s] += dw_phi_n;
        dW_dphi_denom_site[s] += dw_phi_d;
        cross_phi_num_site[s] += dr_phi_n;
        cross_phi_denom_site[s] += dr_phi_d;
    }

    // ---- Phase 3: u = Hinv * (dWdphi .* Hinv_diag) via Woodbury ----
    Eigen::VectorXd a_vec(S);
    for (int s = 0; s < S; s++) a_vec(s) = dWdphi[s] * Hinv_diag(s);
    // H⁻¹·a = A⁻¹·a - ws·(1'·A⁻¹·a)·u_ones
    Eigen::VectorXd z_solve = sp_llt.solve(a_vec);
    double dot_za = ones.dot(z_solve);
    Eigen::VectorXd u = z_solve - wb_scale * dot_za * u_ones;

    // ---- Phase 4: tr(Hinv * Q) via Woodbury ----
    // tr(H⁻¹·Q) = tr(A⁻¹·Q) - ws · u_ones'·Q·u_ones
    Eigen::VectorXd Qu_ones(S);
    icar_precision_matvec(u_ones.data(), Qu_ones.data(), S,
                          data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
    double trHinvQ = trAinvQ - wb_scale * u_ones.dot(Qu_ones);

    // ---- Phase 5: Q * phi_star (for log_tau cross-Hessian) ----
    std::vector<double> Qphi(S);
    icar_precision_matvec(ws.phi_star.data(), Qphi.data(), S,
                          data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);

    // ---- Phase 6: Finalize per-obs coefficients ----
    // coeff[i] = Hinv_diag[s(i)] * dw_deta[i] - u[s(i)] * w[i]
    // These combine the direct (∂W/∂θ) and indirect (through φ* movement) terms
    for (int i = 0; i < N; i++) {
        int s = data.spatial_group[i] - 1;
        double h_s = Hinv_diag(s);
        double u_s = u(s);
        obs_dw_deta_num[i] = h_s * obs_dw_deta_num[i] - u_s * obs_w_num[i];
        obs_dw_deta_den[i] = h_s * obs_dw_deta_den[i] - u_s * obs_w_den[i];
    }
    // Now obs_dw_deta_num[i] is the combined coefficient for X_num[i,:] contributions
    // and obs_dw_deta_den[i] for X_denom[i,:] contributions

    // ---- Phase 7: Assemble Laplace gradient per param ----

    // Beta_num: -0.5 * X_num' * coeff_num
    for (int k = 0; k < data.legacy.p_num; k++) {
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += obs_dw_deta_num[i] * data.legacy.X_num_flat[i * data.legacy.p_num + k];
        }
        result.laplace_grad[layout.legacy.beta_num_start + k] = -0.5 * sum;
    }

    // Beta_denom
    if (!is_binomial) {
        for (int k = 0; k < data.legacy.p_denom; k++) {
            double sum = 0.0;
            for (int i = 0; i < N; i++) {
                sum += obs_dw_deta_den[i] * data.legacy.X_denom_flat[i * data.legacy.p_denom + k];
            }
            result.laplace_grad[layout.legacy.beta_denom_start + k] = -0.5 * sum;
        }
    }

    // RE (w.r.t. centered RE values — NC transform applied by caller)
    if (layout.has_re) {
        int n_re = layout.re_end - layout.re_start;
        for (int i = 0; i < N; i++) {
            if (data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                int g = data.re_group[i] - 1;
                if (g < n_re) {
                    result.laplace_grad[layout.re_start + g] +=
                        -0.5 * (obs_dw_deta_num[i] + obs_dw_deta_den[i]);
                }
            }
        }
    }

    // log_phi_num (chain rule: d/d(log_phi) = phi * d/d(phi))
    if (layout.legacy.has_phi_num) {
        double sum_dW = 0.0, sum_cross = 0.0;
        for (int s = 0; s < S; s++) {
            sum_dW += Hinv_diag(s) * dW_dphi_num_site[s];
            sum_cross += u(s) * cross_phi_num_site[s];
        }
        result.laplace_grad[layout.legacy.log_phi_num_idx] = -0.5 * phi_num * (sum_dW + sum_cross);
    }

    // log_phi_denom
    if (layout.legacy.has_phi_denom) {
        double sum_dW = 0.0, sum_cross = 0.0;
        for (int s = 0; s < S; s++) {
            sum_dW += Hinv_diag(s) * dW_dphi_denom_site[s];
            sum_cross += u(s) * cross_phi_denom_site[s];
        }
        result.laplace_grad[layout.legacy.log_phi_denom_idx] = -0.5 * phi_denom * (sum_dW + sum_cross);
    }

    // log_tau: -0.5 * tau * [trHinvQ - u'*Qphi]
    {
        double u_dot_Qphi = 0.0;
        for (int s = 0; s < S; s++) u_dot_Qphi += u(s) * Qphi[s];
        result.laplace_grad[layout.log_tau_spatial_idx] =
            -0.5 * tau * (trHinvQ - u_dot_Qphi);
    }

    result.success = true;
    return result;
}

// =========================================================================
// BYM2 H-mode: full analytical Laplace gradient for all outer parameters
// Uses 2S×2S dense inverse + S×S effective inverse G
// NOTE: BYM2 inherently needs O(S²) G matrix, so Woodbury only helps for
// very large S (avoids O(S³) dense Cholesky). Low priority for optimization.
// =========================================================================

inline LaplaceGradFullResult compute_laplace_gradient_bym2_H(
    const CollapsedICARWorkspace& ws,
    const double* beta_num, const double* beta_denom,
    double a, double cc, double rho,
    double phi_num, double phi_denom,
    const double* re_vals,
    const ModelData& data,
    const tulpa_hmc::ParamLayout& layout,
    int n_params,
    double lambda_s2z = 0.001
) {
    int S = ws.S;
    int dim = 2 * S;
    int N = data.N;
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    double a2 = a * a;
    double c2 = cc * cc;
    double ac = a * cc;

    LaplaceGradFullResult result;
    result.laplace_grad.assign(n_params, 0.0);
    result.success = false;
    result.log_det = 0.0;

    // ---- Phase 1: Build 2S×2S H, Cholesky, Hinv ----
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);

    // phi-phi block with lambda*11' sum-to-zero
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) H(i, j) = lambda_s2z;
        H(i, i) += a2 * ws.W_data[i] + data.n_neighbors[i];
        for (int k = data.adj_row_ptr[i]; k < data.adj_row_ptr[i + 1]; k++) {
            H(i, data.adj_col_idx[k]) -= 1.0;
        }
    }
    // theta-theta block
    for (int i = 0; i < S; i++) {
        H(S + i, S + i) = c2 * ws.W_data[i] + 1.0;
    }
    // phi-theta cross blocks
    for (int i = 0; i < S; i++) {
        H(i, S + i) = ac * ws.W_data[i];
        H(S + i, i) = ac * ws.W_data[i];
    }

    Eigen::LLT<Eigen::MatrixXd> llt(H);
    if (llt.info() != Eigen::Success) {
        H.diagonal().array() += 1e-6;
        llt.compute(H);
        if (llt.info() != Eigen::Success) return result;
    }

    // Log-det
    Eigen::MatrixXd L = llt.matrixL();
    double log_det = 0.0;
    for (int i = 0; i < dim; i++) log_det += 2.0 * std::log(L(i, i));
    result.log_det = -0.5 * log_det;

    // Full 2S×2S inverse
    Eigen::MatrixXd Hinv = llt.solve(Eigen::MatrixXd::Identity(dim, dim));

    // ---- Phase 2: Effective S×S inverse G and diagonal G_diag ----
    // G[s,t] = a²*Hinv[s,t] + ac*Hinv[s,S+t] + ac*Hinv[S+s,t] + c²*Hinv[S+s,S+t]
    // G_diag[s] = G[s,s] = tr(Hinv * ∂H/∂W_s)
    Eigen::MatrixXd G(S, S);
    Eigen::VectorXd G_diag(S);
    for (int s = 0; s < S; s++) {
        for (int t = 0; t < S; t++) {
            G(s, t) = a2 * Hinv(s, t) + ac * Hinv(s, S + t)
                     + ac * Hinv(S + s, t) + c2 * Hinv(S + s, S + t);
        }
        G_diag(s) = G(s, s);
    }

    // ---- Phase 3: Per-obs derivatives, per-site aggregates ----
    std::vector<double> dWdphi(S, 0.0);  // B[s] = Σ_{i∈s} (dw_n + dw_d)
    std::vector<double> R_site(S, 0.0);  // per-site residual sum (for sigma/rho cross-Hessian)
    std::vector<double> obs_dw_deta_num(N, 0.0);
    std::vector<double> obs_dw_deta_den(N, 0.0);
    std::vector<double> obs_w_num(N, 0.0);
    std::vector<double> obs_w_den(N, 0.0);

    // Dispersion aggregates per site
    std::vector<double> dW_dphi_num_site(S, 0.0);
    std::vector<double> dW_dphi_denom_site(S, 0.0);
    std::vector<double> cross_phi_num_site(S, 0.0);
    std::vector<double> cross_phi_denom_site(S, 0.0);

    for (int i = 0; i < N; i++) {
        int s = data.spatial_group[i] - 1;
        double b_s = a * ws.phi_star[s] + cc * ws.theta_star[s];

        // Compute eta at mode
        double eta_num_i = 0.0, eta_denom_i = 0.0;
        for (int p = 0; p < data.legacy.p_num; p++)
            eta_num_i += data.legacy.X_num_flat[i * data.legacy.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < data.legacy.p_denom; p++)
                eta_denom_i += data.legacy.X_denom_flat[i * data.legacy.p_denom + p] * beta_denom[p];
        }
        eta_num_i += b_s;
        if (!is_binomial) eta_denom_i += b_s;
        if (re_vals && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
            eta_num_i += re_vals[data.re_group[i] - 1];
            if (!is_binomial) eta_denom_i += re_vals[data.re_group[i] - 1];
        }

        double mu_num = std::exp(std::min(eta_num_i, 20.0));
        double w_n = 0.0, w_d = 0.0;
        double dw_eta_n = 0.0, dw_eta_d = 0.0;
        double dw_phi_n = 0.0, dw_phi_d = 0.0;
        double dr_phi_n = 0.0, dr_phi_d = 0.0;
        double resid_i = 0.0;  // dLL/dη for this obs

        switch (data.legacy.model_type) {
            case ModelType::POISSON_GAMMA: {
                w_n = mu_num;
                dw_eta_n = mu_num;
                resid_i = data.legacy.y_num[i] - mu_num;
                if (!is_binomial) {
                    double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                    double y_d = data.legacy.y_denom_cont[i];
                    double alpha = phi_denom;
                    w_d = alpha * y_d / mu_d;
                    dw_eta_d = -w_d;
                    dw_phi_d = y_d / mu_d;
                    dr_phi_d = y_d / mu_d - 1.0;
                    resid_i += alpha * (y_d / mu_d - 1.0);
                }
                break;
            }
            case ModelType::NEGBIN_NEGBIN: {
                double r_n = phi_num;
                double y_n = data.legacy.y_num[i];
                double A = mu_num + r_n, B = y_n + r_n;
                double A2 = A * A, A3 = A2 * A;
                w_n = mu_num * r_n * B / A2;
                dw_eta_n = mu_num * r_n * B * (r_n - mu_num) / A3;
                dw_phi_n = mu_num * (mu_num * (y_n + 2*r_n) - y_n * r_n) / A3;
                dr_phi_n = mu_num * (y_n - mu_num) / A2;
                resid_i = y_n - mu_num * B / A;
                if (!is_binomial) {
                    double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                    double y_d = (double)data.legacy.y_denom[i];
                    double r_d = phi_denom;
                    double Ad = mu_d + r_d, Bd = y_d + r_d;
                    double Ad2 = Ad * Ad, Ad3 = Ad2 * Ad;
                    w_d = mu_d * r_d * Bd / Ad2;
                    dw_eta_d = mu_d * r_d * Bd * (r_d - mu_d) / Ad3;
                    dw_phi_d = mu_d * (mu_d * (y_d + 2*r_d) - y_d * r_d) / Ad3;
                    dr_phi_d = mu_d * (y_d - mu_d) / Ad2;
                    resid_i += y_d - mu_d * Bd / Ad;
                }
                break;
            }
            case ModelType::NEGBIN_GAMMA: {
                double r_n = phi_num;
                double y_n = data.legacy.y_num[i];
                double A = mu_num + r_n, B = y_n + r_n;
                double A2 = A * A, A3 = A2 * A;
                w_n = mu_num * r_n * B / A2;
                dw_eta_n = mu_num * r_n * B * (r_n - mu_num) / A3;
                dw_phi_n = mu_num * (mu_num * (y_n + 2*r_n) - y_n * r_n) / A3;
                dr_phi_n = mu_num * (y_n - mu_num) / A2;
                resid_i = y_n - mu_num * B / A;
                {
                    double mu_d = std::exp(std::min(eta_denom_i, 20.0));
                    double y_d = data.legacy.y_denom_cont[i];
                    double alpha = phi_denom;
                    w_d = alpha * y_d / mu_d;
                    dw_eta_d = -w_d;
                    dw_phi_d = y_d / mu_d;
                    dr_phi_d = y_d / mu_d - 1.0;
                    resid_i += alpha * (y_d / mu_d - 1.0);
                }
                break;
            }
            case ModelType::BINOMIAL: {
                int n_trials = (int)data.legacy.y_denom[i];
                double p_i = 1.0 / (1.0 + std::exp(-eta_num_i));
                w_n = n_trials * p_i * (1.0 - p_i);
                dw_eta_n = w_n * (1.0 - 2.0 * p_i);
                resid_i = data.legacy.y_num[i] - n_trials * p_i;
                break;
            }
            default: {
                w_n = mu_num;
                dw_eta_n = mu_num;
                resid_i = data.legacy.y_num[i] - mu_num;
                break;
            }
        }

        dWdphi[s] += dw_eta_n + dw_eta_d;
        obs_dw_deta_num[i] = dw_eta_n;
        obs_dw_deta_den[i] = dw_eta_d;
        obs_w_num[i] = w_n;
        obs_w_den[i] = w_d;
        R_site[s] += resid_i;

        dW_dphi_num_site[s] += dw_phi_n;
        dW_dphi_denom_site[s] += dw_phi_d;
        cross_phi_num_site[s] += dr_phi_n;
        cross_phi_denom_site[s] += dr_phi_d;
    }

    // ---- Phase 4: u = G * (B .* G_diag) ----
    Eigen::VectorXd a_vec(S);
    for (int s = 0; s < S; s++) a_vec(s) = dWdphi[s] * G_diag(s);
    Eigen::VectorXd u = G * a_vec;

    // ---- Phase 5: Coeff and scatter for betas, RE ----
    // coeff[i] = G_diag[s(i)] * dw_deta[i] - u[s(i)] * w[i]
    for (int i = 0; i < N; i++) {
        int s = data.spatial_group[i] - 1;
        double g_s = G_diag(s);
        double u_s = u(s);
        obs_dw_deta_num[i] = g_s * obs_dw_deta_num[i] - u_s * obs_w_num[i];
        obs_dw_deta_den[i] = g_s * obs_dw_deta_den[i] - u_s * obs_w_den[i];
    }

    // Beta_num: -0.5 * X_num' * coeff_num
    for (int k = 0; k < data.legacy.p_num; k++) {
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += obs_dw_deta_num[i] * data.legacy.X_num_flat[i * data.legacy.p_num + k];
        }
        result.laplace_grad[layout.legacy.beta_num_start + k] = -0.5 * sum;
    }

    // Beta_denom
    if (!is_binomial) {
        for (int k = 0; k < data.legacy.p_denom; k++) {
            double sum = 0.0;
            for (int i = 0; i < N; i++) {
                sum += obs_dw_deta_den[i] * data.legacy.X_denom_flat[i * data.legacy.p_denom + k];
            }
            result.laplace_grad[layout.legacy.beta_denom_start + k] = -0.5 * sum;
        }
    }

    // RE
    if (layout.has_re) {
        int n_re = layout.re_end - layout.re_start;
        for (int i = 0; i < N; i++) {
            if (data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
                int g = data.re_group[i] - 1;
                if (g < n_re) {
                    result.laplace_grad[layout.re_start + g] +=
                        -0.5 * (obs_dw_deta_num[i] + obs_dw_deta_den[i]);
                }
            }
        }
    }

    // ---- Phase 6: Direct sigma/rho traces (structural H change) ----
    // dH/d(log_sigma)_direct: [[2a²W, 2acW], [2acW, 2c²W]]
    double tr_sigma_direct = 0.0;
    for (int s = 0; s < S; s++) {
        double w = ws.W_data[s];
        tr_sigma_direct += w * (2.0*a2 * Hinv(s, s) + 4.0*ac * Hinv(s, S + s)
                                + 2.0*c2 * Hinv(S + s, S + s));
    }

    // dH/d(logit_rho)_direct
    double half_1m2r = (1.0 - 2.0 * rho) / 2.0;
    double tr_rho_direct = 0.0;
    for (int s = 0; s < S; s++) {
        double w = ws.W_data[s];
        tr_rho_direct += w * (a2*(1.0-rho) * Hinv(s, s)
                            + 2.0*ac*half_1m2r * Hinv(s, S + s)
                            - c2*rho * Hinv(S + s, S + s));
    }

    // ---- Phase 7: Indirect sigma/rho via cross-Hessian + IFT ----
    // For sigma: cross = [a*(R-W*b*), c*(R-W*b*)]
    // For rho: cross = [a*(R*(1-ρ)/2 - W*d_rho), c*(-R*ρ/2 - W*d_rho)]
    {
        Eigen::VectorXd b_star(S);
        for (int s = 0; s < S; s++)
            b_star(s) = a * ws.phi_star[s] + cc * ws.theta_star[s];

        // sigma cross-Hessian: z_sigma[s] = R[s] - W[s]*b*[s]
        Eigen::VectorXd z_sigma(S);
        for (int s = 0; s < S; s++)
            z_sigma(s) = R_site[s] - ws.W_data[s] * b_star(s);

        // effective shift for sigma via G: shift = G * z_sigma
        Eigen::VectorXd shift_sigma = G * z_sigma;

        // Total dW[s]/d(log_sigma) = B[s] * (b*[s] + shift_sigma[s])
        // Indirect Laplace contribution for sigma
        double tr_sigma_indirect = 0.0;
        for (int s = 0; s < S; s++)
            tr_sigma_indirect += G_diag(s) * dWdphi[s] * (b_star(s) + shift_sigma(s));

        result.laplace_grad[layout.log_sigma_bym2_idx] =
            -0.5 * (tr_sigma_direct + tr_sigma_indirect);

        // rho cross-Hessian
        // d_rho[s] = a*(1-ρ)/2 * φ*[s] - c*ρ/2 * θ*[s]
        double da_drho = a * (1.0 - rho) / 2.0;
        double dc_drho = -cc * rho / 2.0;

        Eigen::VectorXd d_rho_site(S);
        for (int s = 0; s < S; s++)
            d_rho_site(s) = da_drho * ws.phi_star[s] + dc_drho * ws.theta_star[s];

        // cross_rho_phi[s] = a*(R[s]*(1-ρ)/2 - W[s]*d_rho[s])
        // cross_rho_theta[s] = c*(-R[s]*ρ/2 - W[s]*d_rho[s])
        // Combined shift: need full 2S cross, then Hinv * cross, then a*phi + c*theta
        Eigen::VectorXd cross_rho(dim);
        for (int s = 0; s < S; s++) {
            cross_rho(s) = a * (R_site[s] * (1.0 - rho) / 2.0
                               - ws.W_data[s] * d_rho_site(s));
            cross_rho(S + s) = cc * (-R_site[s] * rho / 2.0
                                     - ws.W_data[s] * d_rho_site(s));
        }

        // d(inner)/d(logit_rho) = Hinv * cross_rho
        Eigen::VectorXd d_inner_rho = Hinv * cross_rho;
        // effective shift: a*dphi + c*dtheta
        Eigen::VectorXd shift_rho(S);
        for (int s = 0; s < S; s++)
            shift_rho(s) = a * d_inner_rho(s) + cc * d_inner_rho(S + s);

        // Total dW[s]/d(logit_rho) = B[s] * (d_rho[s] + shift_rho[s])
        double tr_rho_indirect = 0.0;
        for (int s = 0; s < S; s++)
            tr_rho_indirect += G_diag(s) * dWdphi[s] * (d_rho_site(s) + shift_rho(s));

        result.laplace_grad[layout.logit_rho_bym2_idx] =
            -0.5 * (tr_rho_direct + tr_rho_indirect);
    }

    // ---- Phase 8: Dispersion gradients (through W change, same coeff framework) ----
    if (layout.legacy.has_phi_num) {
        // dW/d(phi_num) per site + cross-Hessian for phi_num
        double sum_dW = 0.0, sum_cross = 0.0;
        for (int s = 0; s < S; s++) {
            sum_dW += G_diag(s) * dW_dphi_num_site[s];
            sum_cross += u(s) * cross_phi_num_site[s];
        }
        result.laplace_grad[layout.legacy.log_phi_num_idx] = -0.5 * phi_num * (sum_dW + sum_cross);
    }

    if (layout.legacy.has_phi_denom) {
        double sum_dW = 0.0, sum_cross = 0.0;
        for (int s = 0; s < S; s++) {
            sum_dW += G_diag(s) * dW_dphi_denom_site[s];
            sum_cross += u(s) * cross_phi_denom_site[s];
        }
        result.laplace_grad[layout.legacy.log_phi_denom_idx] = -0.5 * phi_denom * (sum_dW + sum_cross);
    }

    result.success = true;
    return result;
}

// =========================================================================
// High-level wrappers for compute_log_post integration
// These encapsulate all collapsed ICAR/BYM2 logic so the main log_post
// function only needs a single call instead of 80+ lines of inline code.
// =========================================================================

struct CollapsedICARLogPostResult {
    double log_post_contribution = 0.0;  // Prior + Laplace correction
    const double* phi_spatial = nullptr;  // Points into ws.phi_star
    const double* theta_bym2 = nullptr;   // Points into ws.theta_star (BYM2 only)
};

// Compute the collapsed ICAR/BYM2 contribution to log_post:
//   1. Find phi* (and theta* for BYM2) via Newton
//   2. Evaluate ICAR/BYM2 prior at mode
//   3. Add Laplace correction
//   4. Return pointers to mode values for use in observation loop
//
// re_vals: actual RE values (after NC transform), or nullptr if no RE.
// Caller adds result.log_post_contribution to log_post and uses
// result.phi_spatial / result.theta_bym2 in the observation loop.
inline CollapsedICARLogPostResult collapsed_icar_log_post_contribution(
    bool is_bym2,
    double tau_spatial,        // exp(log_tau) for ICAR
    double sigma_total,        // exp(log_sigma_bym2) for BYM2
    double logit_rho,          // logit_rho for BYM2
    double bym2_scale_factor,  // scale factor for BYM2
    double phi_num, double phi_denom,
    const double* beta_num, const double* beta_denom,
    const double* re_vals,     // nullptr if no RE
    const ModelData& data,
    CollapsedICARWorkspace& ws) {

    CollapsedICARLogPostResult res;
    int S = data.n_spatial_units;

    if (is_bym2) {
        double rho = 1.0 / (1.0 + std::exp(-logit_rho));

        collapsed_bym2_find_mode(
            beta_num, beta_denom, sigma_total, rho, bym2_scale_factor,
            phi_num, phi_denom, re_vals, data, ws);

        res.phi_spatial = ws.phi_star.data();
        res.theta_bym2 = ws.theta_star.data();

        // ICAR prior on phi*: -0.5 * phi*^T Q phi* - 0.5 * lambda * (sum phi*)^2
        std::vector<double> Qphi(S);
        icar_precision_matvec(ws.phi_star.data(), Qphi.data(), S,
                              data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
        double phiQphi = 0.0, sum_phi = 0.0;
        for (int i = 0; i < S; i++) {
            phiQphi += ws.phi_star[i] * Qphi[i];
            sum_phi += ws.phi_star[i];
        }
        res.log_post_contribution += -0.5 * phiQphi - 0.5 * 0.001 * sum_phi * sum_phi;

        // N(0,1) prior on theta*
        for (int i = 0; i < S; i++) {
            res.log_post_contribution -= 0.5 * ws.theta_star[i] * ws.theta_star[i];
        }
    } else {
        // Collapsed ICAR
        collapsed_icar_find_mode(
            beta_num, beta_denom, tau_spatial, phi_num, phi_denom,
            re_vals, data, ws);

        res.phi_spatial = ws.phi_star.data();

        // ICAR prior at phi*: 0.5*(J-1)*log(tau) - 0.5*tau*phi*^T Q phi* - lambda*(sum phi*)^2
        std::vector<double> Qphi(S);
        icar_precision_matvec(ws.phi_star.data(), Qphi.data(), S,
                              data.adj_row_ptr, data.adj_col_idx, data.n_neighbors);
        double phiQphi = 0.0, sum_phi = 0.0;
        for (int i = 0; i < S; i++) {
            phiQphi += ws.phi_star[i] * Qphi[i];
            sum_phi += ws.phi_star[i];
        }
        res.log_post_contribution += -0.5 * tau_spatial * phiQphi
                                   + 0.5 * (S - 1) * std::log(tau_spatial)
                                   - 0.5 * 0.001 * sum_phi * sum_phi;
    }

    // Laplace correction: -0.5 * log det(H)
    res.log_post_contribution += ws.laplace_log_det;

    return res;
}

// Store collapsed ICAR/BYM2 mode values (phi*, theta*) into result buffer
// Called once per sampling iteration after accept/reject
inline void collapsed_icar_store_sample(
    int sample_idx,
    const ModelData& data,
    const CollapsedICARWorkspace& ws,
    std::vector<double>& icar_phi_star_flat,
    std::vector<double>& bym2_theta_star_flat,
    int S) {
    std::memcpy(&icar_phi_star_flat[sample_idx * S],
                ws.phi_star.data(), S * sizeof(double));
    if (data.bym2_collapsed && !bym2_theta_star_flat.empty()) {
        std::memcpy(&bym2_theta_star_flat[sample_idx * S],
                    ws.theta_star.data(), S * sizeof(double));
    }
}

#endif // TULPA_HMC_ICAR_COLLAPSED_H
