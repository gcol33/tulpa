// hmc_icar_collapsed_logdet.h
// Sparse-Cholesky + Woodbury Laplace log-det for ICAR/BYM2.
// Self-contained: defines symbols inside namespace tulpa_hmc.

#ifndef TULPA_HMC_ICAR_COLLAPSED_LOGDET_H
#define TULPA_HMC_ICAR_COLLAPSED_LOGDET_H

#include <cmath>
#include <vector>

#include <RcppEigen.h>

#include "hmc_icar_collapsed_workspace.h"
#include "hmc_sampler.h"

namespace tulpa_hmc {

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

}  // namespace tulpa_hmc

#endif // TULPA_HMC_ICAR_COLLAPSED_LOGDET_H
