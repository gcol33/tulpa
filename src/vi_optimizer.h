// vi_optimizer.h
// Adam optimizer for variational inference
// Shared across all VI variants

#ifndef TULPA_VI_OPTIMIZER_H
#define TULPA_VI_OPTIMIZER_H

#include <Rcpp.h>
#include <deque>
#include <string>
#include <vector>
#include <cmath>
#include <RcppEigen.h>
#include "vi_types.h"
#include <tulpa/vi_math.h>

namespace tulpa {
namespace vi {

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

  std::string converged = checker.check(elbo, grad_flat);
  if (!converged.empty()) {
    result.converged = true;
    result.converged_reason = converged;
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
