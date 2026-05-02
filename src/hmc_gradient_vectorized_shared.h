// hmc_gradient_vectorized_shared.h
// Fragment of hmc_gradient_vectorized.h.
// Included from the hmc_gradient_vectorized.h umbrella header inside
// namespace tulpa_hmc { namespace vectorized { ... } } in hmc_gradients.cpp.
// Do NOT wrap contents in any namespace — already inside namespace vectorized.
// Shared vectorized residual + beta-grad kernel for specialized gradient fns.
#ifndef TULPA_HMC_GRADIENT_VECTORIZED_SHARED_H
#define TULPA_HMC_GRADIENT_VECTORIZED_SHARED_H

// ============================================================================
// Shared vectorized residual + beta-grad kernel for specialized gradient fns
// ============================================================================
//
// These functions allow specialized gradient functions (HSGP, temporal GP,
// spatiotemporal, slopes) to delegate the expensive obs loop to the same
// template-specialized, Eigen-vectorized infrastructure used by
// compute_obs_gradients_vectorized(). The caller:
//   1. Builds eta vectors (X*beta + RE + custom effects) using Eigen matvec
//   2. Calls compute_residuals_and_beta_grads() for residuals + beta grads
//   3. Scatters residuals to custom effect gradient buffers
//
// This eliminates:
//   - Scalar dot products for eta (replaced by Eigen matvec in caller)
//   - Runtime if/else on model_type (replaced by template dispatch)
//   - Scalar scatter for X^T * resid (replaced by Eigen matvec here)

template<ModelType MT>
void compute_residuals_and_beta_grads(
    const ModelData& data,
    const ParamLayout& layout,
    const double* eta_num,             // INPUT: pre-built [N] with all effects
    const double* eta_denom,           // INPUT: pre-built [N] with all effects
    double* resid_num_out,             // OUTPUT: [N] residuals
    double* resid_denom_out,           // OUTPUT: [N] residuals
    double* grad,                      // beta grads accumulated here
    double& grad_phi_num_lik,          // OUTPUT: phi_num likelihood gradient
    double& grad_phi_denom_lik,        // OUTPUT: phi_denom likelihood gradient
    double& obs_ll,                    // OUTPUT: obs log-likelihood (if compute_lp)
    bool compute_lp,
    double phi_num,
    double phi_denom,
    VecGradWorkspace& ws
) {
    const int N = data.N;
    constexpr bool is_binomial = (MT == ModelType::BINOMIAL) || (MT == ModelType::BETA_BINOMIAL);

    // Ensure workspace is initialized and precomputed
    ws.init(N);
    ws.precompute(data);

    // Build digamma/lgamma lookup tables for NB families
    if constexpr (MT == ModelType::NEGBIN_NEGBIN) {
        ws.build_digamma_tables(phi_num, phi_denom, true);
    } else if constexpr (MT == ModelType::NEGBIN_GAMMA) {
        ws.build_digamma_tables(phi_num, phi_denom, false);
    }

    // === Pass 2: Template-specialized residual computation ===
    // Copy eta into workspace (residual kernels read from ws for multipass SIMD)
    // Skip if caller already built eta directly in ws (avoids UB from self-memcpy)
    if (eta_num != ws.eta_num.data()) {
        std::memcpy(ws.eta_num.data(), eta_num, N * sizeof(double));
    }
    if constexpr (!is_binomial) {
        if (eta_denom != ws.eta_denom.data()) {
            std::memcpy(ws.eta_denom.data(), eta_denom, N * sizeof(double));
        }
    }

    grad_phi_num_lik = 0.0;
    grad_phi_denom_lik = 0.0;

    compute_residuals<MT>(
        N, ws.eta_num.data(), ws.eta_denom.data(), data,
        phi_num, phi_denom,
        ws.resid_num.data(), ws.resid_denom.data(),
        grad_phi_num_lik, grad_phi_denom_lik,
        obs_ll, compute_lp, ws
    );

    // Copy residuals to output (skip if output IS the workspace buffer)
    if (resid_num_out != ws.resid_num.data()) {
        std::memcpy(resid_num_out, ws.resid_num.data(), N * sizeof(double));
    }
    if constexpr (!is_binomial) {
        if (resid_denom_out != ws.resid_denom.data()) {
            std::memcpy(resid_denom_out, ws.resid_denom.data(), N * sizeof(double));
        }
    } else {
        // Always zero resid_denom for binomial — memset has no UB with self-target
        // (unlike memcpy). Workspace may have stale data from prior non-binomial call.
        std::memset(resid_denom_out, 0, N * sizeof(double));
    }

    // === Pass 3: Vectorized beta gradient accumulation ===
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorXd = Eigen::VectorXd;

    {
        Eigen::Map<const RowMajorMatrix> X_num(data.legacy.X_num_flat.data(), N, data.legacy.p_num);
        Eigen::Map<const VectorXd> rn(ws.resid_num.data(), N);
        Eigen::Map<VectorXd> gb_num(&grad[layout.legacy.beta_num_start], data.legacy.p_num);
        gb_num.noalias() += X_num.transpose() * rn;
    }

    if constexpr (!is_binomial) {
        Eigen::Map<const RowMajorMatrix> X_denom(data.legacy.X_denom_flat.data(), N, data.legacy.p_denom);
        Eigen::Map<const VectorXd> rd(ws.resid_denom.data(), N);
        Eigen::Map<VectorXd> gb_denom(&grad[layout.legacy.beta_denom_start], data.legacy.p_denom);
        gb_denom.noalias() += X_denom.transpose() * rd;
    }

    // Phi gradients (with log-transform Jacobian)
    if (layout.legacy.has_phi_num) {
        grad[layout.legacy.log_phi_num_idx] += phi_num * grad_phi_num_lik;
    }
    if (layout.legacy.has_phi_denom) {
        grad[layout.legacy.log_phi_denom_idx] += phi_denom * grad_phi_denom_lik;
    }
}

// Runtime dispatcher: switches on data.legacy.model_type, calls template instantiation
inline bool dispatch_residuals_and_beta_grads(
    const ModelData& data,
    const ParamLayout& layout,
    const double* eta_num,
    const double* eta_denom,
    double* resid_num_out,
    double* resid_denom_out,
    double* grad,
    double& grad_phi_num_lik,
    double& grad_phi_denom_lik,
    double& obs_ll,
    bool compute_lp,
    double phi_num,
    double phi_denom,
    VecGradWorkspace& ws
) {
    #define DISPATCH_CASE(MT) \
        case MT: \
            compute_residuals_and_beta_grads<MT>( \
                data, layout, eta_num, eta_denom, \
                resid_num_out, resid_denom_out, grad, \
                grad_phi_num_lik, grad_phi_denom_lik, \
                obs_ll, compute_lp, phi_num, phi_denom, ws); \
            return true;

    switch (data.legacy.model_type) {
        DISPATCH_CASE(ModelType::BINOMIAL)
        DISPATCH_CASE(ModelType::NEGBIN_NEGBIN)
        DISPATCH_CASE(ModelType::POISSON_GAMMA)
        DISPATCH_CASE(ModelType::NEGBIN_GAMMA)
        DISPATCH_CASE(ModelType::GAMMA_GAMMA)
        DISPATCH_CASE(ModelType::LOGNORMAL)
        DISPATCH_CASE(ModelType::BETA_BINOMIAL)
    }
    #undef DISPATCH_CASE
    return false;
}

#endif  // TULPA_HMC_GRADIENT_VECTORIZED_SHARED_H
