// vi_optimizer.h
// Adam optimizer for variational inference
// Shared across all VI variants

#ifndef TULPA_VI_OPTIMIZER_H
#define TULPA_VI_OPTIMIZER_H

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <RcppEigen.h>
#include "vi_types.h"

namespace tulpa {
namespace vi {

// ---------------------------------------------------------------------
// Adam Optimizer State
// ---------------------------------------------------------------------

struct AdamState {
  Eigen::VectorXd m;  // First moment estimate
  Eigen::VectorXd v;  // Second moment estimate
  int t;              // Iteration counter

  AdamState() : t(0) {}

  AdamState(int n_params) : m(Eigen::VectorXd::Zero(n_params)),
                            v(Eigen::VectorXd::Zero(n_params)),
                            t(0) {}

  void reset() {
    m.setZero();
    v.setZero();
    t = 0;
  }
};

// ---------------------------------------------------------------------
// Adam Optimizer
// ---------------------------------------------------------------------

class AdamOptimizer {
public:
  double alpha;   // Learning rate
  double beta1;   // First moment decay
  double beta2;   // Second moment decay
  double eps;     // Numerical stability

  AdamOptimizer(double alpha_ = 0.01,
                double beta1_ = 0.9,
                double beta2_ = 0.999,
                double eps_ = 1e-8)
    : alpha(alpha_), beta1(beta1_), beta2(beta2_), eps(eps_) {}

  // Initialize state for given parameter dimension
  AdamState init_state(int n_params) const {
    return AdamState(n_params);
  }

  // Perform one optimization step
  // Returns updated parameters
  Eigen::VectorXd step(const Eigen::VectorXd& params,
                       const Eigen::VectorXd& grads,
                       AdamState& state) const {
    state.t++;

    // Update biased first moment estimate
    state.m = beta1 * state.m + (1.0 - beta1) * grads;

    // Update biased second moment estimate
    state.v = beta2 * state.v + (1.0 - beta2) * grads.array().square().matrix();

    // Bias-corrected estimates
    double bias_correction1 = 1.0 - std::pow(beta1, state.t);
    double bias_correction2 = 1.0 - std::pow(beta2, state.t);

    Eigen::VectorXd m_hat = state.m / bias_correction1;
    Eigen::VectorXd v_hat = state.v / bias_correction2;

    // Update parameters (gradient ascent for ELBO maximization)
    Eigen::VectorXd update = alpha * m_hat.array() / (v_hat.array().sqrt() + eps);

    return params + update;
  }

  // Step with gradient clipping
  Eigen::VectorXd step_clipped(const Eigen::VectorXd& params,
                               const Eigen::VectorXd& grads,
                               AdamState& state,
                               double max_grad_norm) const {
    // Clip gradients by norm
    double grad_norm = grads.norm();
    Eigen::VectorXd clipped_grads = grads;
    if (grad_norm > max_grad_norm) {
      clipped_grads = grads * (max_grad_norm / grad_norm);
    }

    return step(params, clipped_grads, state);
  }
};

// ---------------------------------------------------------------------
// Convergence Checker
// ---------------------------------------------------------------------

class ConvergenceChecker {
public:
  double tol_grad;         // Gradient norm tolerance
  double tol_rel_elbo;     // Relative ELBO change tolerance
  int patience;            // Patience for early stopping

  // Internal state
  int no_improvement_count;
  double best_elbo;
  bool has_best;           // false until the first ELBO has been recorded

  ConvergenceChecker(double tol_grad_ = 1e-4,
                     double tol_rel_elbo_ = 0.01,
                     int patience_ = 50)
    : tol_grad(tol_grad_),
      tol_rel_elbo(tol_rel_elbo_),
      patience(patience_),
      no_improvement_count(0),
      best_elbo(-std::numeric_limits<double>::infinity()),
      has_best(false) {}

  void reset() {
    no_improvement_count = 0;
    best_elbo = -std::numeric_limits<double>::infinity();
    has_best = false;
  }

  // Check if converged, returns convergence reason or empty string
  std::string check(double elbo, double grad_norm) {
    // Check gradient norm
    if (grad_norm < tol_grad) {
      return "gradient_norm";
    }

    // Check relative ELBO improvement
    if (elbo > best_elbo) {
      if (!has_best) {
        // Nothing to improve on yet.
        no_improvement_count = 0;
      } else if (best_elbo == 0.0) {
        // Any improvement over zero is unbounded relative to it, so it counts
        // as a large one.
        no_improvement_count = 0;
      } else if ((elbo - best_elbo) / std::abs(best_elbo) < tol_rel_elbo) {
        no_improvement_count++;
      } else {
        no_improvement_count = 0;
      }
      best_elbo = elbo;
      has_best = true;
    } else {
      no_improvement_count++;
    }

    // Check patience
    if (no_improvement_count >= patience) {
      return "patience";
    }

    return "";  // Not converged
  }
};

// ---------------------------------------------------------------------
// Shared VI optimization step
// ---------------------------------------------------------------------

// The per-iteration tail every VI variant runs. The variants differ only in
// how they compute the ELBO and pack `grad_flat` (mean-field stacks
// [mu, log_sigma]; low-rank stacks [mu, L column-major, log_d]; full-rank
// stacks [mu, lower triangle of L]). From the gradient norm onward -- the
// convergence test, the Adam update, the progress print, the interrupt check
// and the iteration count -- they are identical, so that half lives here once.
//
// `post_update` runs after the Adam step and before the progress print, for a
// variant that has to repair its parameterization (full-rank clamps the
// diagonal of L positive so the Cholesky stays valid). Pass nothing when there
// is no repair to do.
//
// Returns true when the checker has declared convergence, which is the caller's
// signal to break out of the optimization loop.
template <typename Params, typename PostUpdate>
inline bool vi_adam_step(Params& params,
                         const Eigen::VectorXd& grad_flat,
                         double elbo,
                         int iter,
                         const VIConfig& config,
                         ConvergenceChecker& checker,
                         AdamOptimizer& optimizer,
                         AdamState& state,
                         VIResult& result,
                         PostUpdate post_update) {
  double grad_norm = grad_flat.norm();

  std::string converged = checker.check(elbo, grad_norm);
  if (!converged.empty()) {
    result.converged = true;
    result.iterations = iter + 1;
    if (config.verbose) {
      Rcpp::Rcout << "Converged at iteration " << iter + 1
                  << " (" << converged << ")\n";
    }
    return true;
  }

  Eigen::VectorXd params_flat = params.flatten();
  params_flat = optimizer.step_clipped(params_flat, grad_flat, state,
                                      config.max_grad_norm);
  params.unflatten(params_flat);

  post_update();

  if (config.verbose && (iter + 1) % config.print_every == 0) {
    Rcpp::Rcout << "Iter " << iter + 1 << ": ELBO = " << elbo
                << ", |grad| = " << grad_norm << "\n";
  }

  if ((iter + 1) % 100 == 0) {
    Rcpp::checkUserInterrupt();
  }

  result.iterations = iter + 1;
  return false;
}

template <typename Params>
inline bool vi_adam_step(Params& params,
                         const Eigen::VectorXd& grad_flat,
                         double elbo,
                         int iter,
                         const VIConfig& config,
                         ConvergenceChecker& checker,
                         AdamOptimizer& optimizer,
                         AdamState& state,
                         VIResult& result) {
  return vi_adam_step(params, grad_flat, elbo, iter, config, checker,
                      optimizer, state, result, []() {});
}

} // namespace vi
} // namespace tulpa

#endif // TULPA_VI_OPTIMIZER_H
