// laplace_newton_loop.h
// Shared Newton-loop helpers used by both the dense (laplace_newton.h) and
// sparse (sparse_hessian.h) PIRLS-style solvers.
//
// The two solvers differ only in how the Hessian is assembled and factorized.
// Everything else - objective evaluation, step halving, convergence test, and
// the final log-marginal formula - is identical. Pulling these into one place
// keeps the two drivers in lockstep.

#ifndef TULPA_LAPLACE_NEWTON_LOOP_H
#define TULPA_LAPLACE_NEWTON_LOOP_H

#include "laplace_cholesky.h"
#include "laplace_family_link.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tulpa {

constexpr int MAX_HALVING = 12;

// Penalized log-likelihood: log p(y|x,theta) + log p(x|theta), with the data
// log-lik supplied as a functor of the current eta. This is the likelihood-
// agnostic form the shared Newton loop uses: `log_lik_fn(eta)` returns the data
// log-lik (family-enum, LikelihoodSpec, or any other) so the loop carries no
// family knowledge.
//
// `eta_scratch` is a caller-owned NumericVector of length N used as the
// objective's working eta buffer. Hoisting the allocation out of this
// function lets run_nested_laplace_grid call the Newton solver from inside
// an OpenMP parallel region — Rf_allocVector is not thread-safe.
template<typename ComputeEta, typename ComputeLogPrior, typename LogLik>
inline double eval_penalized_log_lik_ll(
    const Rcpp::NumericVector& x,
    ComputeEta compute_eta, ComputeLogPrior compute_log_prior, LogLik log_lik_fn,
    Rcpp::NumericVector& eta_scratch
) {
    compute_eta(x, eta_scratch);
    double ll = log_lik_fn(eta_scratch);
    double lp = compute_log_prior(x, eta_scratch);
    return ll + lp;
}

// Family-enum convenience overload: wraps the built-in family log-lik as the
// functor. Single source of truth for the loop body; callers that still pass a
// family string (the sparse solver) keep working unchanged.
template<typename ComputeEta, typename ComputeLogPrior>
inline double eval_penalized_log_lik(
    const Rcpp::NumericVector& x,
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    int N, const std::string& family, double phi, int n_threads,
    ComputeEta compute_eta, ComputeLogPrior compute_log_prior,
    Rcpp::NumericVector& eta_scratch
) {
    FamilyLogLik ll{&y, &n_trials, N, family, phi, n_threads};
    return eval_penalized_log_lik_ll(x, compute_eta, compute_log_prior, ll,
                                     eta_scratch);
}

// Backtracking line search with safeguarded quadratic interpolation. Maximizes
// the penalized log-posterior along the Newton direction `delta`. The full step
// (step_scale = 1) is always tried first, so a well-conditioned solve near the
// mode takes the full Newton step and keeps quadratic convergence -- fits that
// accept the full step are bit-for-bit unchanged from the pure-halving path.
//
// On an overshoot (objective decreases), the next trial step is the maximizer of
// the quadratic that interpolates the objective and its directional derivative
// `slope = grad . delta` (>= 0 on an ascent direction) at step 0 together with
// the failed trial value (Nocedal & Wright, Numerical Optimization, sec. 3.5):
//   q(a) = obj_old + slope a + c a^2,  c = (obj_try - obj_old - slope*step)/step^2
//   a*   = slope step^2 / (2 (obj_old + slope step - obj_try)).
// The interpolated step is safeguarded to [0.1, 0.5] x the current step so one
// backtrack lands on or near the line optimum where fixed halving needs several.
// Falls back to halving when the direction is not ascent (slope <= 0) or the
// model is degenerate / non-finite. Acceptance is monotone (objective non-
// decreasing) and each trial step is <= 0.5 x the previous, so the converged
// mode and the MAX_HALVING trial cap are identical to the pure-halving path;
// only the trial sequence inside a backtrack differs.
//
// `x_try_scratch` is a caller-owned NumericVector of length n_x used as the
// step-trial buffer. Reused across trials AND across Newton iterations.
// `n_evals_out`, if non-null, accumulates the number of objective evaluations
// (line-search instrumentation; production callers pass nullptr).
template<typename EvalObj>
inline double line_search_backtrack(
    Rcpp::NumericVector& x,
    const std::vector<double>& delta,
    int n_x, double obj_old, double slope,
    EvalObj eval_obj,
    double& obj_out,
    Rcpp::NumericVector& x_try_scratch,
    int* n_evals_out = nullptr
) {
    double step_scale = 1.0;
    for (int half = 0; half <= MAX_HALVING; half++) {
        for (int j = 0; j < n_x; j++) x_try_scratch[j] = x[j] + step_scale * delta[j];
        double obj_try = eval_obj(x_try_scratch);
        if (n_evals_out) ++(*n_evals_out);
        if (obj_try >= obj_old - 1e-8 || half == MAX_HALVING) {
            for (int j = 0; j < n_x; j++) x[j] = x_try_scratch[j];
            obj_out = obj_try;
            return step_scale;
        }
        double next = 0.5 * step_scale;  // halving fallback
        if (slope > 0.0 && std::isfinite(obj_try)) {
            double denom = 2.0 * (obj_old + slope * step_scale - obj_try);
            if (denom > 0.0) {
                double a  = slope * step_scale * step_scale / denom;
                double lo = 0.1 * step_scale, hi = 0.5 * step_scale;
                next = std::min(std::max(a, lo), hi);
            }
        }
        step_scale = next;
    }
    obj_out = obj_old;  // unreachable: MAX_HALVING branch above always accepts.
    return 1.0;
}

// Maximum |step_scale * delta_j|. The historic Newton convergence criterion:
// the largest absolute coordinate of the taken step. Scale-dependent, so it is
// inflated when H is ill-conditioned (see newton_converged below).
inline double max_abs_step(const std::vector<double>& delta, double step_scale, int n_x) {
    double m = 0.0;
    for (int j = 0; j < n_x; j++) {
        m = std::max(m, std::abs(step_scale * delta[j]));
    }
    return m;
}

// Newton decrement lambda^2 = grad' H^{-1} grad = grad . delta, the predicted
// increase of the (concave) log-posterior from the full Newton step. It is the
// affine-invariant measure of distance to the mode (Boyd & Vandenberghe,
// Convex Optimization, sec. 9.5.1): unlike max|delta| it is unaffected by a
// linear reparameterization, hence by the conditioning of H. Non-negative since
// H is PD.
inline double newton_decrement(const std::vector<double>& grad,
                               const std::vector<double>& delta, int n_x) {
    double d = 0.0;
    for (int j = 0; j < n_x; j++) d += grad[j] * delta[j];
    return d;
}

// Near-mode gate and patience for the stalled-decrement convergence path below.
// The gate (predicted objective gain < 1e-6) marks "essentially at a mode"; the
// patience is how many consecutive non-halving iterations past the gate count as
// a conditioning-limited stall rather than ongoing progress.
inline constexpr double NEWTON_NEARMODE_GATE  = 1e-6;
inline constexpr int    NEWTON_STALL_PATIENCE = 5;

// Per-solve state for the Newton convergence test (decrement history). One
// instance lives for the duration of a single Newton solve; the batch solver
// keeps one per stream.
struct NewtonConvState {
    double prev_decrement = std::numeric_limits<double>::infinity();
    int    stalled = 0;
};

// Newton convergence test, the single source of truth for every solver. Two
// paths:
//   1. max|delta| < tol -- the historic step criterion. On a well-conditioned
//      solve this fires exactly when it always did, so those fits are unchanged.
//   2. a stalled decrement -- the affine-invariant rescue for an ill-conditioned
//      H (a high-order rational SPDE precision Q = Pl' Ci Pl, which squares
//      cond(Pl); a near-singular GMRF). There the Cholesky solve loses too many
//      digits for the step to ever fall below tol, and step halving thrashes,
//      yet the mode is found. We declare convergence only once the decrement is
//      near-mode (< gate) AND has failed to halve for `patience` straight
//      iterations -- the signature of a solve that has extracted all the
//      accuracy its conditioning allows. A well-conditioned fit trips path 1 the
//      moment it crosses the gate, so its stall counter never reaches patience;
//      path 2 cannot preempt it.
inline bool newton_converged(const std::vector<double>& delta,
                             const std::vector<double>& grad,
                             double step_scale, int n_x, double tol,
                             NewtonConvState& st) {
    if (max_abs_step(delta, step_scale, n_x) < tol) return true;
    double dec = newton_decrement(grad, delta, n_x);
    bool improved = dec < st.prev_decrement * 0.5;
    st.prev_decrement = dec;
    if (dec >= NEWTON_NEARMODE_GATE || improved) { st.stalled = 0; return false; }
    return ++st.stalled >= NEWTON_STALL_PATIENCE;
}

// Laplace log-marginal: log_lik + log_prior - 0.5 log|H| + 0.5 n log(2 pi).
inline double finalize_log_marginal(
    double log_lik, double log_prior, double log_det_H, int n_x
) {
    return log_lik + log_prior - 0.5 * log_det_H + 0.5 * n_x * std::log(2.0 * M_PI);
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_LOOP_H
