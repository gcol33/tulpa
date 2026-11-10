// hmc_gradient_fallback.cpp
// Gradient implementations for the generic LikelihoodSpec path:
//   - compute_gradient_generic_numerical  (central differences)
//   - compute_gradient_generic_arena      (arena reverse-mode AD)
//   - g_gradient_mode  (global mode used by resolve_gradient_fn)
//   - verify_gradient_runtime  (warmup-time gradient sanity check)
//
// The runtime check finite-differences against the generic-spec evaluator
// whichever of the three gradient sources is active.

#include <Rcpp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
bool g_gradient_verified = false;

static void validate_initial_state(double log_density,
                                   const std::vector<double>& gradient,
                                   int chain_number) {
    if (!std::isfinite(log_density)) {
        Rcpp::stop("non-finite log density at the initial position (chain %d)",
                   chain_number);
    }
    for (size_t j = 0; j < gradient.size(); ++j) {
        if (!std::isfinite(gradient[j])) {
            Rcpp::stop("non-finite gradient at the initial position, coordinate %d "
                       "(chain %d)", static_cast<int>(j + 1), chain_number);
        }
    }
}

void validate_hmc_initial_position(const std::vector<double>& params,
                                   const ModelData& data,
                                   const ParamLayout& layout,
                                   int chain_number) {
#ifdef _OPENMP
    if (omp_in_parallel()) return;
#endif
    double lp = compute_log_post(params, data, layout);
    validate_initial_state(lp, {}, chain_number);
    std::vector<double> gradient(params.size(), 0.0);
    compute_gradient(params, data, layout, gradient, &lp);
    validate_initial_state(lp, gradient, chain_number);
}

// =====================================================================
// Generic LikelihoodSpec gradient: central differences against
// compute_log_post_generic_spec_double. Used as the canonical
// numerical reference and as the fallback when the model package
// does not ship an arena-AD log-lik.
// =====================================================================

// Central-difference gradient of a scalar function of the parameter vector, at
// relative step `eps`. The step is scaled by the coordinate's own magnitude:
// with a fixed absolute step, a coordinate that lives far from the origin loses
// the low-order bits of params[j] + h to rounding, and the difference quotient
// reports that rounding rather than the derivative.
template <typename F>
static void central_difference_gradient(
    F&& f,
    const std::vector<double>& params,
    std::vector<double>& grad,
    double eps = 1e-6
) {
    const int n = static_cast<int>(params.size());
    grad.assign(n, 0.0);
    std::vector<double> pw = params;

    for (int j = 0; j < n; j++) {
        const double h = eps * std::max(1.0, std::abs(params[j]));
        pw[j] = params[j] + h;
        double fp = f(pw);
        pw[j] = params[j] - h;
        double fm = f(pw);
        pw[j] = params[j];
        grad[j] = (fp - fm) / (2.0 * h);
    }
}

// Richardson-extrapolated central difference: with D(h) the central quotient,
// D(h/2) + (D(h/2) - D(h)) / 3 cancels the h^2 term of the truncation error, so
// the step can be large enough that rounding noise in the evaluated function
// stays negligible after division by it. A target whose value carries noise of
// relative size eps * cond -- a dense covariance factorized near its nugget, a
// smooth Matern kernel over closely spaced times -- puts an error of
// ~ eps * cond * |f| / h on a central difference; at h = 1e-6 that error
// reaches 1e-4 of the derivative while the AD gradient is exact.
template <typename F>
static void richardson_difference_gradient(
    F&& f,
    const std::vector<double>& params,
    std::vector<double>& grad,
    double eps = 1e-3
) {
    std::vector<double> d_h, d_half;
    central_difference_gradient(f, params, d_h, eps);
    central_difference_gradient(f, params, d_half, 0.5 * eps);
    grad.resize(params.size());
    for (size_t j = 0; j < params.size(); j++) {
        grad[j] = d_half[j] + (d_half[j] - d_h[j]) / 3.0;
    }
}

void compute_gradient_generic_numerical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out
) {
    auto log_post_fn = [&](const std::vector<double>& p) -> double {
        return compute_log_post(p, data, layout);
    };

    if (log_post_out) *log_post_out = log_post_fn(params);
    central_difference_gradient(log_post_fn, params, grad);
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
        return compute_log_post(
            p, data, layout, /*skip_obs_loop=*/true);
    };

    if (log_post_out) *log_post_out = log_prior_fn(params);
    central_difference_gradient(log_prior_fn, params, grad);
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
  double lp = compute_log_post(params, data, layout);
  validate_initial_state(lp, {}, 1);
  std::vector<double> grad_active, grad_numerical;
  // The consumer gradient contract (likelihood.h) is "grad is sized to
  // params.size() and overwritten"; a hand-coded spec->gradient_fn that writes
  // grad[i] without resizing would overflow an empty buffer, so pre-size it.
  grad_active.assign(params.size(), 0.0);
  GradientFn active_fn = resolve_gradient_fn(g_gradient_mode, data, layout);
  active_fn(params, data, layout, grad_active, &lp);
  validate_initial_state(lp, grad_active, 1);

  compute_gradient_generic_numerical(params, data, layout, grad_numerical, nullptr);
  validate_initial_state(lp, grad_numerical, 1);

  // A second reference at a large step with the h^2 error extrapolated away.
  // The two references fail in opposite regimes -- the small step on a target
  // whose evaluation carries rounding noise, the large one where the target
  // varies on a scale below its step -- so the active gradient is accepted
  // when it agrees with either. A wrong derivative (a sign, a scale, a missing
  // term) disagrees with both.
  std::vector<double> grad_richardson;
  {
    auto log_post_fn = [&](const std::vector<double>& p) -> double {
      return compute_log_post(p, data, layout);
    };
    richardson_difference_gradient(log_post_fn, params, grad_richardson);
  }
  validate_initial_state(lp, grad_richardson, 1);

  if (grad_numerical.size() != grad_active.size()) {
    REprintf("[tulpa] WARNING: the active gradient returned %d entries for %d "
             "parameters.\n  Falling back to numerical gradients for safety.\n",
             (int)grad_active.size(), (int)grad_numerical.size());
    return false;
  }

  auto rel_gap = [](double ref, double act) {
    const double scale = std::max(1.0, std::max(std::abs(act), std::abs(ref)));
    return std::abs(act - ref) / scale;
  };

  double max_diff = 0.0;
  int worst_idx = -1;
  for (size_t i = 0; i < grad_active.size(); i++) {
    double rel_diff = rel_gap(grad_numerical[i], grad_active[i]);
    if (std::isfinite(grad_richardson[i])) {
      rel_diff = std::min(rel_diff, rel_gap(grad_richardson[i], grad_active[i]));
    }
    if (!std::isfinite(rel_diff)) {
      max_diff = rel_diff;
      worst_idx = static_cast<int>(i);
      break;
    }
    if (rel_diff > max_diff) {
      max_diff = rel_diff;
      worst_idx = static_cast<int>(i);
    }
  }

  if (worst_idx >= 0 && (!std::isfinite(max_diff) || max_diff > tol)) {
    REprintf("[tulpa] WARNING: gradient mismatch detected at param %d!\n"
             "  max |active - numerical| / scale = %.6e (tol = %.1e)\n"
             "  active[%d] = %.8e, numerical[%d] = %.8e (Richardson %.8e)\n"
             "  Falling back to numerical gradients for safety.\n",
             worst_idx, max_diff, tol,
             worst_idx, grad_active[worst_idx],
             worst_idx, grad_numerical[worst_idx], grad_richardson[worst_idx]);
    return false;
  }
  return true;
}

// =====================================================================
// The one gate every NUTS entry passes through (gcol33/tulpa#684).
//
// The check used to sit in two callers -- the R wrapper and the multi-chain
// producer -- while six other production sites entered run_hmc_chain_cpp
// directly, including the R_RegisterCCallable a LinkingTo consumer reaches for
// backend "hmc". A consumer's hand-coded spec->gradient_fn was therefore
// checked when it asked for several chains and unchecked when it asked for
// one, which is the reverse of what the check is for.
// =====================================================================
void ensure_gradient_verified(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout
) {
#ifdef _OPENMP
  // Inside an across-chain region the producer has already run this on the
  // main thread, and Rcpp::warning off-thread is not safe.
  if (omp_in_parallel()) return;
#endif
  if (g_gradient_verified) return;
  g_gradient_verified = true;
  if (g_gradient_mode == GradientMode::NUMERICAL) return;
  if (verify_gradient_runtime(q_init, data, layout, 1e-4)) return;
  g_gradient_mode = GradientMode::NUMERICAL;
  Rcpp::warning(
    "Gradient mismatch detected: active gradient function disagrees with "
    "numerical gradients (max rel diff > 1e-4). Falling back to numerical "
    "gradients (mode='N'). This is slower but correct. Please report this "
    "as a bug at https://github.com/gcol33/tulpa/issues"
  );
}

}  // namespace tulpa_hmc
