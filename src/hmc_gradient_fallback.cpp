// hmc_gradient_fallback.cpp
// Fallback gradient implementations: numerical (central differences),
// autodiff (tape, arena, forward), generic multi-process gradients,
// runtime gradient verification, and the global gradient mode.

#include <algorithm>
#include <cmath>
#include <vector>

#include <Rcpp.h>

#include "autodiff.h"
#include "autodiff_arena.h"
#include "autodiff_fwd.h"
#include "hmc_sampler.h"
#include "log_post_impl.h"
#include "tulpa/likelihood.h"

namespace tulpa_hmc {

// Gradient mode: controls which gradient function the dispatcher picks.
// Defined here so fallback verify_gradient_runtime can read/write it.
GradientMode g_gradient_mode = GradientMode::AUTO;

// =====================================================================
// Numerical gradient (fallback for complex models)
// =====================================================================

void compute_gradient_numerical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
  int n = params.size();
  grad.resize(n);

  if (log_post_out) {
    *log_post_out = compute_log_post(params, data, layout);
  }

  double h = 1e-5;

  for (int i = 0; i < n; i++) {
    std::vector<double> params_plus = params;
    std::vector<double> params_minus = params;

    params_plus[i] = params[i] + h;
    params_minus[i] = params[i] - h;

    double f_plus = compute_log_post(params_plus, data, layout);
    double f_minus = compute_log_post(params_minus, data, layout);

    grad[i] = (f_plus - f_minus) / (2.0 * h);
  }
}

// =====================================================================
// Numerical gradient using compute_log_post_impl<double>
// Used for verifying A_r/A modes which use compute_log_post_impl<T>.
// This ensures the numerical reference matches the same function that
// the autodiff mode differentiates (important when parameterization
// differs from H-mode, e.g. non-centered RE).
// =====================================================================

void compute_gradient_numerical_impl(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
  int n = params.size();
  grad.resize(n);

  if (log_post_out) {
    *log_post_out = tulpa::compute_log_post_impl(params, data, layout);
  }

  double h = 1e-5;

  for (int i = 0; i < n; i++) {
    std::vector<double> params_plus = params;
    std::vector<double> params_minus = params;

    params_plus[i] = params[i] + h;
    params_minus[i] = params[i] - h;

    double f_plus = tulpa::compute_log_post_impl(params_plus, data, layout);
    double f_minus = tulpa::compute_log_post_impl(params_minus, data, layout);

    grad[i] = (f_plus - f_minus) / (2.0 * h);
  }
}

// =====================================================================
// Runtime gradient check: compare compute_gradient() dispatcher output
// against numerical gradients at the first warmup iteration.
// Catches log-post/gradient mismatches in ALL specialized gradient functions
// (GP, HSGP, SVC, TVC, MSGP, spatiotemporal, etc.), not just the main
// compute_gradient_analytical().
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

  // Choose the correct numerical reference based on which function is active.
  // Autodiff functions differentiate compute_log_post_impl<T>, so the reference
  // must finite-diff the same template. H-mode functions use compute_log_post,
  // so the reference must finite-diff that.
  bool active_is_autodiff = (active_fn == &compute_gradient_arena ||
                             active_fn == &compute_gradient_forward ||
                             active_fn == &compute_gradient_autodiff);
  if (active_is_autodiff) {
    compute_gradient_numerical_impl(params, data, layout, grad_numerical);
  } else {
    compute_gradient_numerical(params, data, layout, grad_numerical);
  }

  double max_diff = 0.0;
  int worst_idx = -1;
  for (size_t i = 0; i < grad_active.size(); i++) {
    double diff = std::abs(grad_active[i] - grad_numerical[i]);
    double scale = std::max(1.0, std::max(std::abs(grad_active[i]),
                                           std::abs(grad_numerical[i])));
    double rel_diff = diff / scale;
    if (rel_diff > max_diff) {
      max_diff = rel_diff;
      worst_idx = i;
    }
  }

  if (max_diff > tol) {
    REprintf("[numdenom] WARNING: gradient mismatch detected at param %d!\n"
             "  max |active - numerical| / scale = %.6e (tol = %.1e)\n"
             "  active[%d] = %.8e, numerical[%d] = %.8e\n"
             "  This indicates a bug in the specialized gradient function.\n"
             "  Falling back to numerical gradients for safety.\n",
             worst_idx, max_diff, tol,
             worst_idx, grad_active[worst_idx],
             worst_idx, grad_numerical[worst_idx]);
    return false;
  }
  return true;
}

// =====================================================================
// Autodiff gradient (O(n) - works for ALL models)
// =====================================================================

void compute_gradient_autodiff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    using namespace tulpa::ad;

    TapeScope tape_scope;
    Tape* tape = tape_scope.tape;

    std::vector<Var> params_ad = make_vars(tape, params);

    Var log_post = tulpa::compute_log_post_impl(params_ad, data, layout);

    if (log_post_out) *log_post_out = log_post.val();

    log_post.backward();

    grad = get_adjoints(params_ad);
}

// =====================================================================
// Arena-based reverse-mode autodiff gradient (O(N) - fast, all models)
// Uses contiguous SoA memory layout with pre-computed partials.
// ~10-30x faster than tape autodiff, within 50% of hand-coded speed.
// =====================================================================

void compute_gradient_arena(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    using namespace tulpa::arena;

    int n_nodes_used = 0;
    {
        ArenaScope scope;
        Arena* arena = scope.arena();

        std::vector<Var> params_ar = make_vars(arena, params);

        Var log_post = tulpa::compute_log_post_impl(params_ar, data, layout);

        if (log_post_out) *log_post_out = log_post.val();

        log_post.backward();

        grad = get_adjoints(params_ar);

        n_nodes_used = arena->size();
    }

    (void)n_nodes_used;
}

// =====================================================================
// Forward-mode autodiff gradient (O(n*p) - but ~10x faster than tape)
// Uses dual numbers for efficient gradient computation without heap allocation
// =====================================================================

void compute_gradient_forward(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    int n_params = static_cast<int>(params.size());
    grad.assign(n_params, 0.0);

    std::vector<fwd::Dual> params_dual(n_params);

    for (int i = 0; i < n_params; i++) {
        for (int j = 0; j < n_params; j++) {
            params_dual[j].val = params[j];
            params_dual[j].grad = (j == i) ? 1.0 : 0.0;
        }

        fwd::Dual log_post = tulpa::compute_log_post_impl(params_dual, data, layout);

        grad[i] = log_post.grad;

        if (i == 0 && log_post_out) *log_post_out = log_post.val;
    }
}

// =====================================================================
// Generic multi-process gradient (central differences)
// Used by model packages (tulpaOcc, etc.) that plug in via LikelihoodSpec.
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
// Generic multi-process gradient via arena reverse-mode AD
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

    if (log_post_out) *log_post_out = log_post.val();
    log_post.backward();
    grad = get_adjoints(params_ar);
}

}  // namespace tulpa_hmc
