// hmc_gp_collapsed_grad.h
// Analytical Laplace gradient for non-GP parameters in collapsed GP.
// Self-contained: defines symbols inside namespace tulpa_hmc.

#ifndef TULPA_HMC_GP_COLLAPSED_GRAD_H
#define TULPA_HMC_GP_COLLAPSED_GRAD_H

#include <algorithm>
#include <cmath>
#include <vector>

#include <RcppEigen.h>

#include "hmc_gp_collapsed_logdet.h"
#include "hmc_gp_collapsed_mode.h"
#include "hmc_gp_collapsed_ops.h"
#include "hmc_sampler.h"

namespace tulpa_hmc {

// =========================================================================
// Analytical Laplace gradient for non-GP parameters (beta, phi, RE)
// =========================================================================
// Uses H_inv_diag (Takahashi diagonal) and third derivatives dW/d(theta).
// The Laplace correction gradient is: -0.5 * tr((W+Q)^{-1} * dW/d(theta))
// Since W is diagonal: = -0.5 * sum_i (W+Q)^{-1}_{ii} * dW_i/d(theta)
//
// For parameter theta that affects eta via deta/dtheta:
//   dW_i/dtheta = (dW_i/deta) * (deta/dtheta)

inline void collapsed_gp_laplace_grad_nonGP(
    const double* w_star,
    const double* beta_num, const double* beta_denom,
    double phi_num, double phi_denom,
    double sigma_re, const double* re,
    const ModelData& data,
    const double* H_inv_diag,
    const ParamLayout& layout,
    double* grad
) {
    int N = data.N;
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    double grad_laplace_phi_num = 0.0;
    double grad_laplace_phi_denom = 0.0;

    for (int i = 0; i < N; i++) {
        int loc_i = data.gp_data.obs_to_loc[i];
        double h_inv_i = H_inv_diag[loc_i];

        // Compute eta at this observation
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
        int y_num = data.legacy.y_num[i];

        // Third derivatives: dW_i/d(eta_num), dW_i/d(eta_denom)
        // and dW_i/d(log_phi_num), dW_i/d(log_phi_denom)
        double dW_deta_num = 0.0, dW_deta_denom = 0.0;
        double dW_dlogphi_num = 0.0, dW_dlogphi_denom = 0.0;

        switch (data.legacy.model_type) {
            case ModelType::POISSON_GAMMA: {
                // neg_hess_num = mu, d(neg_hess)/d(eta) = mu
                dW_deta_num = mu_num;
                if (data.gp_data.shared) {
                    double mu_denom = std::exp(eta_denom_i);
                    double y_denom = data.legacy.y_denom_cont[i];
                    double shape = phi_denom;
                    double nh_d = shape * y_denom / mu_denom;
                    dW_deta_denom = -nh_d;
                    dW_dlogphi_denom = nh_d;
                }
                break;
            }
            case ModelType::NEGBIN_NEGBIN: {
                double r_num = phi_num;
                double yr = y_num + r_num;
                double mr = mu_num + r_num;
                double mr3 = mr * mr * mr;
                dW_deta_num = mu_num * r_num * yr * (r_num - mu_num) / mr3;
                double dnh_dr = mu_num * (yr * (mu_num - r_num) + r_num * mr) / mr3;
                dW_dlogphi_num = r_num * dnh_dr;

                if (data.gp_data.shared) {
                    double mu_d = std::exp(eta_denom_i);
                    int y_d = (int)data.legacy.y_denom[i];
                    double r_d = phi_denom;
                    double yr_d = y_d + r_d;
                    double mr_d = mu_d + r_d;
                    double mr_d3 = mr_d * mr_d * mr_d;
                    dW_deta_denom = mu_d * r_d * yr_d * (r_d - mu_d) / mr_d3;
                    double dnh_dr_d = mu_d * (yr_d * (mu_d - r_d) + r_d * mr_d) / mr_d3;
                    dW_dlogphi_denom = r_d * dnh_dr_d;
                }
                break;
            }
            case ModelType::BINOMIAL: {
                int n_trials = (int)data.legacy.y_denom[i];
                double p_i = 1.0 / (1.0 + std::exp(-eta_num_i));
                dW_deta_num = n_trials * p_i * (1.0 - p_i) * (1.0 - 2.0 * p_i);
                break;
            }
            case ModelType::NEGBIN_GAMMA: {
                double r_num_v = phi_num;
                double yr = y_num + r_num_v;
                double mr = mu_num + r_num_v;
                double mr3 = mr * mr * mr;
                dW_deta_num = mu_num * r_num_v * yr * (r_num_v - mu_num) / mr3;
                double dnh_dr = mu_num * (yr * (mu_num - r_num_v) + r_num_v * mr) / mr3;
                dW_dlogphi_num = r_num_v * dnh_dr;
                if (data.gp_data.shared) {
                    double mu_denom = std::exp(eta_denom_i);
                    double y_denom = data.legacy.y_denom_cont[i];
                    double shape = phi_denom;
                    double nh_d = shape * y_denom / mu_denom;
                    dW_deta_denom = -nh_d;
                    dW_dlogphi_denom = nh_d;
                }
                break;
            }
            default: {
                dW_deta_num = mu_num;
                break;
            }
        }

        // Laplace gradient weight for eta_num and eta_denom
        double w_num = -0.5 * h_inv_i * dW_deta_num;
        double w_denom = -0.5 * h_inv_i * dW_deta_denom;

        // Scatter to beta_num
        for (int p = 0; p < data.legacy.p_num; p++) {
            grad[layout.legacy.beta_num_start + p] += w_num * data.legacy.X_num_flat[i * data.legacy.p_num + p];
        }
        // Scatter to beta_denom
        if (!is_binomial) {
            for (int p = 0; p < data.legacy.p_denom; p++) {
                grad[layout.legacy.beta_denom_start + p] += w_denom * data.legacy.X_denom_flat[i * data.legacy.p_denom + p];
            }
        }
        // Scatter to RE
        if (layout.has_re && data.re_group.size() > (size_t)i && data.re_group[i] > 0) {
            int re_g = data.re_group[i] - 1;
            double w_re = w_num + w_denom;
            if (data.re_parameterization == 1) {
                grad[layout.re_start + re_g] += w_re * sigma_re;
            } else {
                grad[layout.re_start + re_g] += w_re;
            }
        }

        // Accumulate phi gradients
        grad_laplace_phi_num += -0.5 * h_inv_i * dW_dlogphi_num;
        grad_laplace_phi_denom += -0.5 * h_inv_i * dW_dlogphi_denom;
    }

    if (layout.legacy.has_phi_num) grad[layout.legacy.log_phi_num_idx] += grad_laplace_phi_num;
    if (layout.legacy.has_phi_denom) grad[layout.legacy.log_phi_denom_idx] += grad_laplace_phi_denom;
}

// =========================================================================
// Analytical sigma2 Laplace gradient + numerical phi Laplace gradient
// =========================================================================
// sigma2: tr(Z * dQ/d(log_sigma2)) = -(N - tr(Z*W)) because dQ/d(log_sigma2) = -Q
// phi: use laplace_log_det_fixed_w (cheap: rebuild Q + Cholesky, no Newton) with central diff

inline void compute_laplace_grad_gp_hypers(
    double sigma2, double phi,
    const double* w_star,
    const double* beta_num, const double* beta_denom,
    double phi_num, double phi_denom,
    const GPData& gp_data,
    const CollapsedGPWorkspace& ws,
    const ModelData& data,
    double& grad_log_sigma2,
    double& grad_log_phi
) {
    int N = ws.N_gp;

    // ---- sigma2 gradient (analytical) ----
    // -0.5 * d/d(log_sigma2) log det(W+Q)
    // = -0.5 * tr(Z * dW/d(log_sigma2)) + [-0.5 * tr(Z * dQ/d(log_sigma2))]
    // W doesn't depend on sigma2, so dW/d(log_sigma2) = 0
    // dQ/d(log_sigma2) = -Q (because Q ~ 1/sigma2 and B is scale-invariant)
    // => -0.5 * tr(Z * (-Q)) = 0.5 * tr(Z*Q)
    // tr(Z*Q) = tr(Z*(W+Q)) - tr(Z*W) = tr(I) - sum_i Z_ii * W_i = N - sum_i Z_ii * W_i
    double trace_ZW = 0.0;
    for (int i = 0; i < N; i++) {
        trace_ZW += ws.H_inv_diag[i] * ws.hess_diag[i];
    }
    grad_log_sigma2 += 0.5 * (N - trace_ZW);

    // ---- phi gradient (cheap numerical via fixed-w Cholesky) ----
    // Central difference: d/d(log_phi) [-0.5 log det(W+Q)]
    // Uses laplace_log_det_fixed_w which only rebuilds Q (no Newton solve)
    const double eps = 1e-5;
    double log_phi = std::log(phi);

    double phi_plus = std::exp(log_phi + eps);
    double ld_plus = laplace_log_det_fixed_w(sigma2, phi_plus, w_star,
                                              beta_num, beta_denom,
                                              phi_num, phi_denom, data);

    double phi_minus = std::exp(log_phi - eps);
    double ld_minus = laplace_log_det_fixed_w(sigma2, phi_minus, w_star,
                                               beta_num, beta_denom,
                                               phi_num, phi_denom, data);

    grad_log_phi += (ld_plus - ld_minus) / (2.0 * eps);
}

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_GP_COLLAPSED_GRAD_H

