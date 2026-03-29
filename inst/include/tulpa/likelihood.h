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
// LikelihoodSpec: registration structure for model-specific likelihoods
//
// Model packages create one of these and pass it to tulpa's fitting functions.
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
};

} // namespace tulpa

#endif // TULPA_LIKELIHOOD_H
