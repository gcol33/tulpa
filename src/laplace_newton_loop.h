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

// Step halving: try x + step_scale*delta, halve until objective recovers or
// MAX_HALVING is reached. Updates x in place. obj_out receives the accepted
// objective value. Returns the accepted step_scale.
//
// `x_try_scratch` is a caller-owned NumericVector of length n_x used as the
// step-trial buffer. Reused across halvings AND across Newton iterations.
template<typename EvalObj>
inline double step_halving_update(
    Rcpp::NumericVector& x,
    const std::vector<double>& delta,
    int n_x, double obj_old,
    EvalObj eval_obj,
    double& obj_out,
    Rcpp::NumericVector& x_try_scratch
) {
    double step_scale = 1.0;
    for (int half = 0; half <= MAX_HALVING; half++) {
        for (int j = 0; j < n_x; j++) x_try_scratch[j] = x[j] + step_scale * delta[j];
        double obj_try = eval_obj(x_try_scratch);
        if (obj_try >= obj_old - 1e-8 || half == MAX_HALVING) {
            for (int j = 0; j < n_x; j++) x[j] = x_try_scratch[j];
            obj_out = obj_try;
            return step_scale;
        }
        step_scale *= 0.5;
    }
    obj_out = obj_old;  // unreachable: MAX_HALVING branch above always accepts.
    return 1.0;
}

// Maximum |step_scale * delta_j| - used as the Newton convergence criterion.
inline double max_abs_step(const std::vector<double>& delta, double step_scale, int n_x) {
    double m = 0.0;
    for (int j = 0; j < n_x; j++) {
        m = std::max(m, std::abs(step_scale * delta[j]));
    }
    return m;
}

// Laplace log-marginal: log_lik + log_prior - 0.5 log|H| + 0.5 n log(2 pi).
inline double finalize_log_marginal(
    double log_lik, double log_prior, double log_det_H, int n_x
) {
    return log_lik + log_prior - 0.5 * log_det_H + 0.5 * n_x * std::log(2.0 * M_PI);
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_LOOP_H
