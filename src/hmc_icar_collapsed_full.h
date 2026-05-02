// hmc_icar_collapsed_full.h
// Full-model residuals + Laplace log-det helpers for ICAR/BYM2.
// Self-contained: defines symbols inside namespace tulpa_hmc.

#ifndef TULPA_HMC_ICAR_COLLAPSED_FULL_H
#define TULPA_HMC_ICAR_COLLAPSED_FULL_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "hmc_icar_collapsed_kernels.h"
#include "hmc_icar_collapsed_logdet.h"
#include "hmc_icar_collapsed_unit_lik.h"
#include "hmc_icar_collapsed_workspace.h"
#include "hmc_sampler.h"

namespace tulpa_hmc {

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

}  // namespace tulpa_hmc

#endif // TULPA_HMC_ICAR_COLLAPSED_FULL_H
