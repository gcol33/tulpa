// hmc_gradient_fallback.cpp
// Gradient implementations for the generic LikelihoodSpec path:
//   - compute_gradient_generic_numerical  (central differences)
//   - compute_gradient_generic_arena      (arena reverse-mode AD)
//   - g_gradient_mode  (global mode used by resolve_gradient_fn)
//   - verify_gradient_runtime  (warmup-time gradient sanity check)
//
// Phase D simplification (gcol33/tulpa#15): the legacy ratio
// compute_gradient_numerical / compute_gradient_autodiff / _arena /
// _forward / compute_gradient_numerical_impl were deleted with the
// H-mode kernels. The runtime check now finite-diffs against the
// generic-spec evaluator regardless of which active path is in use.

#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "tulpa/autodiff_arena.h"
#include "hmc_sampler.h"
#include "log_post_impl.h"
#include "tulpa/likelihood.h"

namespace tulpa_hmc {

// Gradient mode: controls which gradient function the dispatcher picks.
// Defined here so verify_gradient_runtime can read it without pulling
// hmc_gradients.cpp into its translation unit.
GradientMode g_gradient_mode = GradientMode::AUTO;

// =====================================================================
// Generic LikelihoodSpec gradient: central differences against
// compute_log_post_generic_spec_double. Used as the canonical
// numerical reference and as the fallback when the model package
// does not ship an arena-AD log-lik.
// =====================================================================

void compute_gradient_generic_numerical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    auto log_post_fn = [&](const std::vector<double>& p) -> double {
        return tulpa::compute_log_post_generic_spec_double(p, data, layout);
    };

    double f0 = log_post_fn(params);
    if (log_post_out) *log_post_out = f0;

    const double eps = 1e-6;
    const int n = static_cast<int>(params.size());
    grad.assign(n, 0.0);
    std::vector<double> pw = params;

    for (int j = 0; j < n; j++) {
        pw[j] = params[j] + eps;
        double fp = log_post_fn(pw);
        pw[j] = params[j] - eps;
        double fm = log_post_fn(pw);
        pw[j] = params[j];
        grad[j] = (fp - fm) / (2.0 * eps);
    }
}

// =====================================================================
// Generic LikelihoodSpec gradient: arena reverse-mode AD against
// compute_log_post_generic<Var>. Folds in the optional extra-prior
// arena variant when the model package ships one.
// =====================================================================

void compute_gradient_generic_arena(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    using namespace tulpa::arena;
    const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);

    ArenaScope scope;
    Arena* ar = scope.arena();
    std::vector<Var> params_ar = make_vars(ar, params);

    Var log_post = tulpa::compute_log_post_generic<Var>(
        params_ar, data, layout,
        spec->ll_arena, data.model_response_data);

    if (spec->extra_prior_arena != nullptr) {
        log_post = log_post + spec->extra_prior_arena(
            params_ar, layout, data.model_response_data);
    }

    if (log_post_out) *log_post_out = log_post.val();
    log_post.backward();
    grad = get_adjoints(params_ar);
}

// =====================================================================
// Prior-only gradient for multiple-time-stepping. The "fast" force is the
// gradient of the prior + structural terms with the O(N) observation loop
// skipped (skip_obs_loop = true): the stiff, cheap Gaussian-latent part. The
// slow force is then grad_full - grad_prior = the observation-likelihood
// gradient. Both the arena and the numerical variant fold in the optional
// model-package prior (extra_prior / extra_prior_arena), matching the full
// evaluators so grad_full - grad_prior is exactly the likelihood gradient.
// =====================================================================

void compute_gradient_prior_numerical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    auto log_prior_fn = [&](const std::vector<double>& p) -> double {
        return tulpa::compute_log_post_generic_spec_double(
            p, data, layout, /*skip_obs_loop=*/true);
    };

    double f0 = log_prior_fn(params);
    if (log_post_out) *log_post_out = f0;

    const double eps = 1e-6;
    const int n = static_cast<int>(params.size());
    grad.assign(n, 0.0);
    std::vector<double> pw = params;

    for (int j = 0; j < n; j++) {
        pw[j] = params[j] + eps;
        double fp = log_prior_fn(pw);
        pw[j] = params[j] - eps;
        double fm = log_prior_fn(pw);
        pw[j] = params[j];
        grad[j] = (fp - fm) / (2.0 * eps);
    }
}

void compute_gradient_prior_arena(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    using namespace tulpa::arena;
    const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);

    ArenaScope scope;
    Arena* ar = scope.arena();
    std::vector<Var> params_ar = make_vars(ar, params);

    Var log_prior = tulpa::compute_log_post_generic<Var>(
        params_ar, data, layout,
        spec->ll_arena, data.model_response_data, /*skip_obs_loop=*/true);

    if (spec->extra_prior_arena != nullptr) {
        log_prior = log_prior + spec->extra_prior_arena(
            params_ar, layout, data.model_response_data);
    }

    if (log_post_out) *log_post_out = log_prior.val();
    log_prior.backward();
    grad = get_adjoints(params_ar);
}

// =====================================================================
// Runtime gradient check: compares the active gradient (from the
// dispatcher) against compute_gradient_generic_numerical at the
// start of warmup. Catches sign/scale bugs in hand-coded full-gradient
// hooks (spec->gradient_fn) and in the arena-AD path.
// =====================================================================
bool verify_gradient_runtime(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    double tol
) {
  std::vector<double> grad_active, grad_numerical;
  GradientFn active_fn = resolve_gradient_fn(g_gradient_mode, data, layout);
  active_fn(params, data, layout, grad_active, nullptr);

  compute_gradient_generic_numerical(params, data, layout, grad_numerical, nullptr);

  double max_diff = 0.0;
  int worst_idx = -1;
  for (size_t i = 0; i < grad_active.size(); i++) {
    double diff = std::abs(grad_active[i] - grad_numerical[i]);
    double scale = std::max(1.0, std::max(std::abs(grad_active[i]),
                                           std::abs(grad_numerical[i])));
    double rel_diff = diff / scale;
    if (rel_diff > max_diff) {
      max_diff = rel_diff;
      worst_idx = static_cast<int>(i);
    }
  }

  if (max_diff > tol) {
    REprintf("[tulpa] WARNING: gradient mismatch detected at param %d!\n"
             "  max |active - numerical| / scale = %.6e (tol = %.1e)\n"
             "  active[%d] = %.8e, numerical[%d] = %.8e\n"
             "  Falling back to numerical gradients for safety.\n",
             worst_idx, max_diff, tol,
             worst_idx, grad_active[worst_idx],
             worst_idx, grad_numerical[worst_idx]);
    return false;
  }
  return true;
}

}  // namespace tulpa_hmc
