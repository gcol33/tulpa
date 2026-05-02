// hmc_icar_collapsed_grad.h
// Analytical Laplace gradient (tr(H^-1 dH/dtheta)) for ICAR/BYM2 hyperparams.
// Self-contained: defines symbols inside namespace tulpa_hmc.

#ifndef TULPA_HMC_ICAR_COLLAPSED_GRAD_H
#define TULPA_HMC_ICAR_COLLAPSED_GRAD_H

#include <cmath>

#include <RcppEigen.h>

#include "hmc_icar_collapsed_workspace.h"
#include "hmc_sampler.h"

namespace tulpa_hmc {

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

}  // namespace tulpa_hmc

#endif // TULPA_HMC_ICAR_COLLAPSED_GRAD_H
