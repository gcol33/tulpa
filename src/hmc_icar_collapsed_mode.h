// hmc_icar_collapsed_mode.h
// Inner Newton/CG mode finders for collapsed ICAR and BYM2.
// Self-contained: defines symbols inside namespace tulpa_hmc.

#ifndef TULPA_HMC_ICAR_COLLAPSED_MODE_H
#define TULPA_HMC_ICAR_COLLAPSED_MODE_H

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

}  // namespace tulpa_hmc

#endif // TULPA_HMC_ICAR_COLLAPSED_MODE_H
