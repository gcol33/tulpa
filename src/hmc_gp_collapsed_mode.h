// hmc_gp_collapsed_mode.h
// Newton mode-finder + per-location likelihood for collapsed GP.
// Self-contained: defines symbols inside namespace tulpa_hmc.

#ifndef TULPA_HMC_GP_COLLAPSED_MODE_H
#define TULPA_HMC_GP_COLLAPSED_MODE_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "hmc_gp_collapsed_logdet.h"
#include "hmc_gp_collapsed_ops.h"
#include "hmc_sampler.h"

namespace tulpa_hmc {

// =========================================================================
// Newton-Raphson for finding w* (inner Laplace optimization)
// =========================================================================

// Find w* = argmax_w [ log p(y|beta,w) + log p_NNGP(w|sigma2,phi) ]
// using Newton-Raphson with CG inner solve
//
// Returns the log-posterior at w* (data + prior, no hyperparameter priors)
inline double collapsed_gp_find_mode(
    const double* beta_num, const double* beta_denom,
    double sigma2, double phi,
    double phi_num, double phi_denom,
    const ModelData& data,
    CollapsedGPWorkspace& ws,
    int max_newton = 20,
    double newton_tol = 1e-6
) {
    int N_gp = data.gp_data.n_obs;
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    // Fix 2: Only rebuild NNGP structure if (sigma2, phi) changed
    if (sigma2 != ws.cached_sigma2 || phi != ws.cached_phi || !ws.structure_built) {
        build_nngp_B_D(sigma2, phi, data.gp_data, ws);
        ws.cached_sigma2 = sigma2;
        ws.cached_phi = phi;
    }

    // Build loc→obs mapping (cached, O(N) first time only)
    build_loc_obs_map(ws, data);

    // Initialize w to previous mode or zero
    if (!ws.mode_found) {
        std::fill(ws.w_star.begin(), ws.w_star.end(), 0.0);
    } else {
        // Warm-start from previous w_star, but guard against NaN
        bool has_nan = false;
        for (int i = 0; i < N_gp; i++) {
            if (std::isnan(ws.w_star[i]) || std::isinf(ws.w_star[i])) {
                has_nan = true;
                break;
            }
        }
        if (has_nan) {
            std::fill(ws.w_star.begin(), ws.w_star.end(), 0.0);
        }
    }

    std::vector<double> Qw(N_gp);
    std::vector<double> delta_w(N_gp, 0.0);

    // DEBUG: Newton iteration tracking
    static thread_local int newton_debug_count = 0;
    bool newton_debug = (newton_debug_count < 3 && N_gp >= 20);

    for (int newton_iter = 0; newton_iter < max_newton; newton_iter++) {
        // Compute data gradient and Hessian per location
        std::fill(ws.grad_w.begin(), ws.grad_w.end(), 0.0);
        std::fill(ws.hess_diag.begin(), ws.hess_diag.end(), 0.0);

        double data_ll = 0.0;
        int obs_count = 0;
        for (int loc = 0; loc < N_gp; loc++) {
            int n_obs_loc = ws.loc_obs_ptr[loc + 1] - ws.loc_obs_ptr[loc];
            const int* obs_loc = &ws.loc_obs_idx[ws.loc_obs_ptr[loc]];
            LocLikResult lr = compute_loc_lik(loc, ws.w_star.data(),
                                               beta_num, beta_denom,
                                               phi_num, phi_denom,
                                               data, is_binomial,
                                               obs_loc, n_obs_loc);
            ws.grad_w[loc] = lr.grad;
            ws.hess_diag[loc] = std::max(lr.neg_hess, 1e-8);  // Ensure positive
            data_ll += lr.ll;
            if (newton_debug && newton_iter == 0) {
                obs_count += n_obs_loc;
                if (std::isnan(lr.ll) || std::isnan(lr.grad) || n_obs_loc == 0) {
                    Rprintf("  [NEWTON] loc=%d cnt=%d ll=%.4f grad=%.4f hess=%.8f\n",
                            loc, n_obs_loc, lr.ll, lr.grad, lr.neg_hess);
                }
            }
        }
        if (newton_debug && newton_iter == 0) {
            // Print obs_to_loc range
            int otl_min = data.N, otl_max = -1;
            for (int i = 0; i < data.N; i++) {
                otl_min = std::min(otl_min, data.gp_data.obs_to_loc[i]);
                otl_max = std::max(otl_max, data.gp_data.obs_to_loc[i]);
            }
            Rprintf("  obs_to_loc range=[%d,%d] N_gp=%d\n", otl_min, otl_max, N_gp);
        }

        // Add NNGP prior gradient: d/dw log p_NNGP(w|sigma2,phi) = -Q w
        nngp_precision_matvec(ws.w_star.data(), Qw.data(), ws);
        for (int i = 0; i < N_gp; i++) {
            ws.grad_w[i] -= Qw[i];  // gradient = data_grad - Q*w
        }

        // Check convergence
        double grad_norm = 0.0;
        for (int i = 0; i < N_gp; i++) grad_norm += ws.grad_w[i] * ws.grad_w[i];
        grad_norm = std::sqrt(grad_norm);

        if (newton_debug) {
            double w_min = 1e30, w_max = -1e30;
            double dw_min = 1e30, dw_max = -1e30;
            for (int i = 0; i < N_gp; i++) {
                w_min = std::min(w_min, ws.w_star[i]);
                w_max = std::max(w_max, ws.w_star[i]);
            }
            Rprintf("[NEWTON iter=%d] grad_norm=%.6e data_ll=%.4f w=[%.4f,%.4f]",
                    newton_iter, grad_norm, data_ll, w_min, w_max);
            if (newton_iter == 0)
                Rprintf(" obs_count=%d N_gp=%d N=%d", obs_count, N_gp, data.N);
            Rprintf("\n");
        }

        if (grad_norm < newton_tol) {
            break;
        }

        // Solve (W + Q) delta_w = grad_w using CG
        std::fill(delta_w.begin(), delta_w.end(), 0.0);
        int cg_iters = cg_solve(delta_w.data(), ws.grad_w.data(), ws.hess_diag.data(), ws, 100, 1e-8);

        if (newton_debug) {
            double dw_min = 1e30, dw_max = -1e30;
            for (int i = 0; i < N_gp; i++) {
                dw_min = std::min(dw_min, delta_w[i]);
                dw_max = std::max(dw_max, delta_w[i]);
            }
            Rprintf("  CG iters=%d delta_w=[%.4f,%.4f]\n", cg_iters, dw_min, dw_max);
        }

        // Newton update: w <- w + delta_w
        for (int i = 0; i < N_gp; i++) {
            ws.w_star[i] += delta_w[i];
        }
    }
    if (newton_debug) newton_debug_count++;

    ws.mode_found = true;

    // Compute log-posterior at w*
    double data_ll = 0.0;
    for (int loc = 0; loc < N_gp; loc++) {
        int n_obs_loc = ws.loc_obs_ptr[loc + 1] - ws.loc_obs_ptr[loc];
        const int* obs_loc = &ws.loc_obs_idx[ws.loc_obs_ptr[loc]];
        LocLikResult lr = compute_loc_lik(loc, ws.w_star.data(),
                                           beta_num, beta_denom,
                                           phi_num, phi_denom,
                                           data, is_binomial,
                                           obs_loc, n_obs_loc);
        data_ll += lr.ll;
    }

    // NNGP prior: -0.5 * w^T Q w - 0.5 * log det(Q) + 0.5 * N * log(2pi)
    // For the collapsed log-post we need: data_ll + nngp_prior
    nngp_precision_matvec(ws.w_star.data(), Qw.data(), ws);
    double wQw = 0.0;
    for (int i = 0; i < N_gp; i++) wQw += ws.w_star[i] * Qw[i];

    // log det(Q) = -2 * sum log(d_i) (from Q = (I-B)^T D^{-1} (I-B), det Q = 1/prod(d_i))
    double log_det_Q = 0.0;
    for (int i = 0; i < N_gp; i++) log_det_Q -= std::log(ws.d_cond[i]);

    double nngp_prior = -0.5 * wQw + 0.5 * log_det_Q;

    // Laplace correction: -0.5 * log det(W + Q) via sparse Cholesky
    compute_laplace_log_det(ws);

    return data_ll + nngp_prior;
}

// =========================================================================
// Collapsed GP gradient computation
// =========================================================================

// Compute residuals (dL/deta) at w* for scattering to beta gradients
inline void collapsed_gp_compute_residuals(
    const double* w_star,
    const double* beta_num, const double* beta_denom,
    double phi_num, double phi_denom,
    const ModelData& data,
    double* resid_num,  // length N (observations, not locations)
    double* resid_denom // length N
) {
    int N = data.N;
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    for (int i = 0; i < N; i++) {
        int loc_i = data.gp_data.obs_to_loc[i];

        double eta_num_i = 0.0, eta_denom_i = 0.0;
        for (int p = 0; p < data.legacy.p_num; p++)
            eta_num_i += data.legacy.X_num_flat[i * data.legacy.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < data.legacy.p_denom; p++)
                eta_denom_i += data.legacy.X_denom_flat[i * data.legacy.p_denom + p] * beta_denom[p];
        }

        eta_num_i += w_star[loc_i];
        if (!is_binomial && data.gp_data.shared) eta_denom_i += w_star[loc_i];

        double mu_num = std::exp(eta_num_i);

        switch (data.legacy.model_type) {
            case ModelType::POISSON_GAMMA: {
                resid_num[i] = data.legacy.y_num[i] - mu_num;
                if (!is_binomial && data.gp_data.shared) {
                    double mu_denom = std::exp(eta_denom_i);
                    resid_denom[i] = phi_denom * (data.legacy.y_denom_cont[i] / mu_denom - 1.0);
                } else {
                    resid_denom[i] = 0.0;
                }
                break;
            }
            case ModelType::NEGBIN_NEGBIN: {
                double r_num = phi_num;
                resid_num[i] = data.legacy.y_num[i] - mu_num * (data.legacy.y_num[i] + r_num) / (mu_num + r_num);
                if (data.gp_data.shared) {
                    double mu_denom = std::exp(eta_denom_i);
                    int y_denom = (int)data.legacy.y_denom[i];
                    double r_denom = phi_denom;
                    resid_denom[i] = y_denom - mu_denom * (y_denom + r_denom) / (mu_denom + r_denom);
                } else {
                    resid_denom[i] = 0.0;
                }
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
// Cheap Laplace log-det with fixed w* (no Newton, just rebuild Q + Cholesky)
// =========================================================================
// For numerical gradient of the Laplace correction w.r.t. phi:
// Perturb phi, rebuild Q for new phi, recompute W at fixed w*, Cholesky.
inline double laplace_log_det_fixed_w(
    double sigma2, double phi,
    const double* w_star,
    const double* beta_num, const double* beta_denom,
    double phi_num, double phi_denom,
    const ModelData& data
) {
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    CollapsedGPWorkspace temp_ws;
    build_nngp_B_D(sigma2, phi, data.gp_data, temp_ws);

    int N_gp = temp_ws.N_gp;
    for (int loc = 0; loc < N_gp; loc++) {
        LocLikResult lr = compute_loc_lik(loc, w_star,
                                           beta_num, beta_denom,
                                           phi_num, phi_denom, data, is_binomial);
        temp_ws.hess_diag[loc] = std::max(lr.neg_hess, 1e-8);
    }

    return compute_laplace_log_det(temp_ws);
}

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_GP_COLLAPSED_MODE_H
