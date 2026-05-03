// hmc_gradients.cpp
// Gradient implementations and dispatch for the HMC backend.

#include "hmc_sampler.h"
#include "linalg_fast.h"
#include <RcppEigen.h>
#include "autodiff.h"
#include "autodiff_utils.h"
#include "hmc_gp_autodiff.h"
#include "hmc_gp_collapsed.h"
#include "hmc_icar_collapsed.h"
#include "hmc_car_proper.h"
#include "hmc_temporal_autodiff.h"
#include "hmc_tvc_grad.h"
#include "hmc_multiscale_temporal_grad.h"
#include "lkj_chol_helpers.h"
#include "hmc_likelihood.h"
#include "log_post_impl.h"
#include "tulpa/likelihood.h"
#include <Rcpp.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace tulpa_hmc {

// Include vectorized gradient header after hmc_likelihood.h so legacy
// log_lik_* helpers are available to template definitions.
#include "hmc_gradient_vectorized.h"

// Thread-local vectorized gradient workspace (avoids per-call allocation).
// External linkage so the definition is shared across gradient translation units.
thread_local vectorized::VecGradWorkspace vec_grad_ws;

extern thread_local CollapsedGPWorkspace collapsed_gp_ws;
extern thread_local CollapsedICARWorkspace collapsed_icar_ws;

// Pointer-based ICAR quadratic form for hot gradient paths.
// Delegates to the shared CAR/ICAR kernel (rho = 1).
static inline double icar_quadratic_form_ptr(
    const double* phi, int J,
    const ModelData& data
) {
  return tulpa::car_quad_form(
      phi, J,
      data.adj_row_ptr.data(), data.adj_col_idx.data(), data.n_neighbors.data(),
      /*rho=*/1.0);
}

#include "hmc_gradient_analytical_impl.h"
#include "hmc_gradient_fallback_impl.h"
#include "hmc_gradient_helpers_impl.h"

// =====================================================================
// Shared gradient building blocks (preamble, vectorized eta, dispatch,
// RE scatter, epilogue). Used by all handcoded gradient functions below.
// =====================================================================
#include "hmc_gradient_shared.h"

#include "hmc_gradient_gp_impl.h"
#include "hmc_gradient_feature_impl.h"
#include "hmc_gradient_autodiff_impl.h"
#include "hmc_gradient_hsgp_impl.h"
#include "hmc_gradient_st_impl.h"
#include "hmc_gradient_composite_impl.h"

// g_gradient_mode defined earlier in file (before verify_gradient_runtime)

// Set global gradient mode (called at start of sampling)
void set_gradient_mode(GradientMode mode) {
    g_gradient_mode = mode;
}

GradientMode get_gradient_mode() {
    return g_gradient_mode;
}

void reset_grad_workspace_cache() {
    vec_grad_ws.cached_data_id = 0;
}

void compute_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    // Delegate to resolve_gradient_fn() - single source of truth for dispatch logic.
    // For hot paths (NUTS leapfrog), callers should resolve once via resolve_gradient_fn()
    // and reuse the pointer. This convenience wrapper is for cold-path callers.
    GradientFn fn = resolve_gradient_fn(g_gradient_mode, data, layout);
    fn(params, data, layout, grad, log_post_out);
}

// =====================================================================

} // namespace tulpa_hmc
