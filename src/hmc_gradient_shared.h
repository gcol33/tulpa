// hmc_gradient_shared.h
// Shared building blocks for handcoded H-mode gradient functions.
//
// These helpers extract the duplicated preamble, vectorized eta computation,
// residual dispatch, RE scatter, and epilogue code that was previously
// copy-pasted across 11+ specialized gradient functions.
//
// All are static inline — zero overhead, compiler inlines them.
//
// Self-contained: opens namespace tulpa_hmc, includes its own dependencies.

#ifndef TULPA_HMC_GRADIENT_SHARED_H
#define TULPA_HMC_GRADIENT_SHARED_H

#include <cmath>
#include <cstring>
#include <vector>

#include <RcppEigen.h>

#include "hmc_gradient_helpers_impl.h"     // CommonGradParams, extract_common_params, ...
#include "hmc_gradient_vectorized.h"       // tulpa_hmc::vectorized::VecGradWorkspace, ...
#include "hmc_sampler.h"                   // ModelData, ParamLayout, ModelType

namespace tulpa_hmc {

// Shared per-thread vectorized gradient workspace. Defined in hmc_gradients.cpp,
// referenced from every handcoded gradient .cpp via this extern declaration.
extern thread_local vectorized::VecGradWorkspace vec_grad_ws;

// ---------------------------------------------------------------------------
// Gradient preamble: init grad, extract common params, set up fuse_lp
// ---------------------------------------------------------------------------

struct GradientPreamble {
    bool fuse_lp;
    double obs_log_lik;
    CommonGradParams cp;
    bool is_binomial;
    int N;
};

static inline GradientPreamble gradient_preamble(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    GradientPreamble pre;
    pre.fuse_lp = (log_post_out != nullptr) && !layout.has_zi;
    if (log_post_out && layout.has_zi) {
        *log_post_out = compute_log_post(params, data, layout);
    }
    pre.obs_log_lik = 0.0;
    grad.assign(static_cast<int>(params.size()), 0.0);
    pre.cp = extract_common_params(params, layout);
    pre.is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                       data.legacy.model_type == ModelType::BETA_BINOMIAL);
    pre.N = data.N;
    return pre;
}

// ---------------------------------------------------------------------------
// Base prior gradients: beta, RE, phi (shared across all gradient functions)
// ---------------------------------------------------------------------------

static inline void gradient_base_priors(
    const ModelData& data,
    const ParamLayout& layout,
    const CommonGradParams& cp,
    double* grad
) {
    beta_gradient_prior(data, layout, cp.beta_num, cp.beta_denom, grad);
    re_gradient_prior(data, layout, cp.re, grad, cp.sigma_re);
    phi_gradient_prior(data, layout, cp.phi_num, cp.phi_denom, grad);
}

// ---------------------------------------------------------------------------
// Vectorized eta computation: X * beta (Eigen matvec) + RE contribution
//
// After calling this, eta_num and eta_denom contain the fixed effects
// linear predictor plus RE contribution. Callers add feature-specific
// effects (GP, SVC, TVC, temporal, etc.) to eta before dispatching residuals.
// ---------------------------------------------------------------------------

static inline void compute_base_eta(
    const ModelData& data,
    const ParamLayout& layout,
    const CommonGradParams& cp,
    bool is_binomial,
    vectorized::VecGradWorkspace& ws
) {
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    const int N = data.N;
    ws.init(N);

    // X_num * beta_num
    Eigen::Map<const RowMajorMatrix> X_num(data.legacy.X_num_flat.data(), N, data.legacy.p_num);
    Eigen::Map<const VectorXd> b_num(cp.beta_num, data.legacy.p_num);
    Eigen::Map<VectorXd> eta_n(ws.eta_num.data(), N);
    eta_n.noalias() = X_num * b_num;

    // X_denom * beta_denom (or zero for binomial)
    if (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.legacy.X_denom_flat.data(), N, data.legacy.p_denom);
        Eigen::Map<const VectorXd> b_denom(cp.beta_denom, data.legacy.p_denom);
        Eigen::Map<VectorXd> eta_d(ws.eta_denom.data(), N);
        eta_d.noalias() = X_denom * b_denom;
    } else {
        std::memset(ws.eta_denom.data(), 0, N * sizeof(double));
    }

    // Add RE contribution
    if (layout.has_re) {
        for (int i = 0; i < N; i++) {
            if (data.re_group[i] > 0) {
                int g = data.re_group[i] - 1;
                double re_eff = re_value_for_eta(cp.re, g, cp.sigma_re, data.re_parameterization);
                ws.eta_num[i] += re_eff;
                if (!is_binomial) ws.eta_denom[i] += re_eff;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch residuals + beta gradients (vectorized pass 2+3)
// ---------------------------------------------------------------------------

static inline void dispatch_residuals(
    const ModelData& data,
    const ParamLayout& layout,
    const CommonGradParams& cp,
    bool fuse_lp,
    double& obs_log_lik,
    vectorized::VecGradWorkspace& ws,
    double* grad
) {
    double grad_phi_num_lik = 0.0, grad_phi_denom_lik = 0.0;
    vectorized::dispatch_residuals_and_beta_grads(
        data, layout,
        ws.eta_num.data(), ws.eta_denom.data(),
        ws.resid_num.data(), ws.resid_denom.data(),
        grad, grad_phi_num_lik, grad_phi_denom_lik,
        obs_log_lik, fuse_lp, cp.phi_num, cp.phi_denom, ws);
}

// ---------------------------------------------------------------------------
// Scatter residuals to RE gradients
// ---------------------------------------------------------------------------

static inline void scatter_re_residuals(
    const ModelData& data,
    const ParamLayout& layout,
    const vectorized::VecGradWorkspace& ws,
    double* grad
) {
    if (!layout.has_re) return;
    for (int i = 0; i < data.N; i++) {
        if (data.re_group[i] > 0) {
            grad[layout.re_start + data.re_group[i] - 1] +=
                ws.resid_num[i] + ws.resid_denom[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient epilogue: NC transform + fused log-posterior
// ---------------------------------------------------------------------------

static inline void gradient_epilogue(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    const CommonGradParams& cp,
    bool fuse_lp,
    double obs_log_lik,
    std::vector<double>& grad,
    double* log_post_out
) {
    re_gradient_nc_transform(data, layout, params.data(), grad.data(), cp.sigma_re);

    if (fuse_lp) {
        *log_post_out = compute_log_post(params, data, layout, /*skip_obs_loop=*/true) + obs_log_lik;
    }
}

}  // namespace tulpa_hmc

#endif // TULPA_HMC_GRADIENT_SHARED_H
