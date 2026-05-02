// hmc_icar_collapsed_grad_H.h
// H-mode (full Hessian) Laplace gradients for ICAR/BYM2.
// Self-contained: defines symbols inside namespace tulpa_hmc.

#ifndef TULPA_HMC_ICAR_COLLAPSED_GRAD_H_H
#define TULPA_HMC_ICAR_COLLAPSED_GRAD_H_H

#include <algorithm>
#include <cmath>
#include <vector>

#include <RcppEigen.h>

#include "hmc_icar_collapsed_kernels.h"
#include "hmc_icar_collapsed_workspace.h"
#include "hmc_sampler.h"

namespace tulpa_hmc {

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
    const ParamLayout& layout,
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
    const ParamLayout& layout,
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

}  // namespace tulpa_hmc

#endif // TULPA_HMC_ICAR_COLLAPSED_GRAD_H_H
