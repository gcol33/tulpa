#ifndef TULPA_LIKELIHOOD_H
#define TULPA_LIKELIHOOD_H

#include <vector>
#include <string>
#include <functional>
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa/autodiff_fwd.h"

namespace tulpa {

// ============================================================================
// LikelihoodFn: per-observation log-likelihood callback
//
// Receives pre-computed linear predictor values (one per process) and returns
// the log-likelihood contribution for observation i.
//
// Templated for double (evaluation), arena::Var (reverse AD), fwd::Dual
// (forward AD). Model packages provide one implementation that works for all
// three via C++ templates.
// ============================================================================
template<typename T>
using LikelihoodFn = T(*)(
    int i,                           // Observation index
    const T* eta,                    // Linear predictor values [n_processes]
    const T& logit_zi,               // ZI linear predictor (0 if no ZI)
    const T& logit_oi,               // OI linear predictor (0 if no OI)
    const std::vector<T>& params,    // Full parameter vector (for extra params)
    const ModelData& data,           // Generic model data
    const ParamLayout& layout,       // Parameter layout
    const void* model_data           // Model-specific response data
);

// ============================================================================
// ResidualFn: per-observation residual for H-mode gradients
//
// Computes d(log_lik_i)/d(eta_k) for each process k. tulpa's vectorized
// gradient framework handles the chain rule from eta back through all
// latent structure (spatial, temporal, RE, etc.).
//
// Model packages only differentiate through their own likelihood, never
// through spatial fields or temporal effects.
// ============================================================================
using ResidualFn = void(*)(
    int i,                           // Observation index
    const double* eta,               // Linear predictor values [n_processes]
    double logit_zi,                 // ZI LP
    double logit_oi,                 // OI LP
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    const void* model_data,
    double* resid_out                // Output: d(ll_i)/d(eta_k) for each process
);

// ============================================================================
// ExtraGradFn: gradient contribution from model-specific extra parameters
//
// For H-mode: computes gradients for parameters that are not part of the
// linear predictor (e.g., dispersion phi, detection betas).
// ============================================================================
using ExtraGradFn = void(*)(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    const void* model_data,
    std::vector<double>& grad,       // Accumulate into existing gradient
    double* log_post_out             // Accumulate into log posterior
);

// ============================================================================
// EtaWeightsFn: per-observation eta-space gradient + negative-Hessian for
// Laplace IRLS / Fisher scoring.
//
// For obs i, fills:
//   grad_eta[k]                          = d log_lik_i / d eta_i_k
//   neg_hess_eta[k * n_processes + l]    = -d^2 log_lik_i / (d eta_i_k d eta_i_l)
//
// Returning weights in eta-space (rather than mu-space) is the natural IRLS
// contract: the eta-space negative-Hessian assembles directly into X' W X for
// the latent-field Hessian. n_processes == 1 → grad_eta and neg_hess_eta are
// scalars; for general n_processes, neg_hess_eta is row-major n x n.
//
// Optional. If null, the LikelihoodSpec is not usable from the spec-driven
// Laplace path (a clear error is raised). NUTS / VI / ESS / MCLMC / SMC do
// not require this callback.
// ============================================================================
using EtaWeightsFn = void(*)(
    int i,                           // Observation index
    const double* eta,               // Linear predictor values [n_processes]
    double logit_zi,                 // ZI LP
    double logit_oi,                 // OI LP
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    const void* model_data,
    double* grad_eta,                // [n_processes] out
    double* neg_hess_eta             // [n_processes * n_processes] out
);

// ============================================================================
// FullGradFn: fully hand-coded gradient + log-posterior for the entire
// parameter vector. Optional override for the auto N→A→A_r→H progression
// when a model package ships a tuned hand-coded gradient (e.g. ratio
// likelihoods porting their pre-tulpa H-kernel verbatim).
//
// Signature matches every other gradient driver in tulpa
// (src/hmc_gradient_*.cpp): no separate model_data argument because
// model-specific response data is reachable via data.model_response_data.
//
// Contract:
//   * grad is sized to params.size() and overwritten (not accumulated).
//   * If log_post_out is non-null, the log-posterior is written there in
//     the same pass (fused log-post + gradient, the standard tulpa contract).
//   * The function must compute d(log_posterior)/d(params) — that is,
//     log-likelihood + log-prior contributions, including all latent
//     structure (spatial, temporal, RE) that lives in `params`.
//
// When set on a LikelihoodSpec, the dispatcher prefers this callback over
// AD even in AUTO mode. Set to nullptr (the default) to keep the existing
// AD progression.
// ============================================================================
using FullGradFn = void(*)(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,       // Output: overwritten, length == params.size()
    double* log_post_out             // Optional: fused log-posterior (may be null)
);

// ============================================================================
// LikelihoodSpec: registration structure for model-specific likelihoods
//
// Model packages create one of these and pass it to tulpa's fitting functions.
//
// LAYOUT RULE: append-only. Existing fields keep their relative order so
// model packages compiled against an earlier ABI continue to bind. New
// optional fields go at the end and default to nullptr / safe values.
// ============================================================================
struct LikelihoodSpec {
    // ABI version this spec was compiled against (set automatically)
    int abi_version = TULPA_ABI_VERSION;

    int n_processes = 1;
    std::string name;

    // Templated likelihood for each AD mode
    LikelihoodFn<double> ll_double = nullptr;
    LikelihoodFn<arena::Var> ll_arena = nullptr;
    LikelihoodFn<::fwd::Dual> ll_fwd = nullptr;

    // H-mode: per-observation residual (optional, falls back to A_r if null)
    ResidualFn residual_fn = nullptr;

    // H-mode: extra parameter gradients (optional)
    ExtraGradFn extra_grad_fn = nullptr;

    // Spec-driven Laplace IRLS weights (optional). Required to use this spec
    // through tulpa_laplace_spec_*; ignored by NUTS / VI / ESS / MCLMC / SMC.
    EtaWeightsFn eta_weights_fn = nullptr;

    // Number of extra parameters this likelihood adds to the parameter vector
    int n_extra_params = 0;

    // Extend ParamLayout with model-specific parameter positions
    void (*extend_layout)(const ModelData& data,
                          ParamLayout& layout,
                          const void* model_data) = nullptr;

    // Prior contribution from model-specific extra parameters
    double (*extra_prior)(const std::vector<double>& params,
                          const ParamLayout& layout,
                          const void* model_data) = nullptr;

    // Optional fully hand-coded gradient over the entire parameter vector.
    // When non-null, the gradient dispatcher uses this callback instead of
    // the templated AD path. See FullGradFn for the contract. Lets a model
    // package port a pre-existing hand-tuned gradient verbatim without
    // routing through tulpa's per-kernel handcoded predicates.
    FullGradFn gradient_fn = nullptr;

    // Arena-AD variant of extra_prior. When non-null AND ll_arena is set,
    // the gradient dispatcher routes through arena reverse-mode AD instead
    // of finite differences over the full parameter vector. The signature
    // mirrors `extra_prior` but takes an arena::Var-vector view of params
    // and returns an arena::Var so the prior contribution flows into the
    // backward pass. Append-only on LikelihoodSpec — no ABI bump.
    arena::Var (*extra_prior_arena)(const std::vector<arena::Var>& params,
                                    const ParamLayout& layout,
                                    const void* model_data) = nullptr;
};

} // namespace tulpa

#endif // TULPA_LIKELIHOOD_H
