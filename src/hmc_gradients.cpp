// hmc_gradients.cpp
// HMC backend gradient orchestrator.
//
// One consumer: the dispatcher in hmc_gradient_dispatch.cpp, which resolves
// every model to the generic-LikelihoodSpec gradient path.

#include "hmc_sampler.h"
#include "tulpa/likelihood.h"

#include <Rcpp.h>
#include <vector>

namespace tulpa_hmc {

// =====================================================================
// Public entry point: thin convenience wrapper around resolve_gradient_fn.
// Hot paths (NUTS leapfrog) should resolve once and reuse the pointer;
// this wrapper is for cold-path callers that need a one-shot gradient.
// =====================================================================
void compute_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    GradientFn fn = resolve_gradient_fn(g_gradient_mode, data, layout);
    fn(params, data, layout, grad, log_post_out);
}

// =====================================================================
// Global gradient mode getters/setters (g_gradient_mode itself is owned
// by hmc_gradient_fallback.cpp).
// =====================================================================
void set_gradient_mode(GradientMode mode) {
    g_gradient_mode = mode;
}

GradientMode get_gradient_mode() {
    return g_gradient_mode;
}

} // namespace tulpa_hmc
