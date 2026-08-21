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
// A NON-FINITE trial objective is never accepted, at any trial including the
// last. -Inf is the domain barrier of a constrained link (link_eta_in_domain in
// laplace_family_link.h) and NaN is a failed evaluation; moving the iterate onto
// either leaves the model undefined at x and every subsequent gradient, Hessian
// and log-determinant reading garbage. This is what makes the barrier hold: the
// search backtracks off an infeasible trial instead of walking through it.
// Accepting a FINITE non-improving value at the last trial is unchanged, so any
// solve whose objective stays finite -- every fit on an unconstrained link --
// takes exactly the trial sequence it always did.
//
// The acceptance slack is ABSOLUTE (1e-8), which makes it a real criterion far
// from the mode and no criterion at all near it: once the whole available gain
// drops below 1e-8 every finite trial passes. That is not a defect to tighten
// away -- see newton_trust_scale below, which is what steers in the regime where
// no rule reading the objective can.
//
// When no trial is acceptable the search returns 0, leaving `x` and `obj_out`
// untouched: the caller's iterate is still the best point known. newton_converged
// treats a zero step as a stall rather than convergence, so an exhausted search
// cannot be mistaken for a mode.
//
// `start_scale` is the trial step the search opens with. It is 1 (the full
// Newton step) for a solve whose direction is trustworthy, which is every solve
// away from the mode and every solve whose Newton weight is the observed
// curvature; newton_trust_scale supplies the damped value where it is not.
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
    int* n_evals_out = nullptr,
    double start_scale = 1.0
) {
    double step_scale = start_scale;
    for (int half = 0; half <= MAX_HALVING; half++) {
        for (int j = 0; j < n_x; j++) x_try_scratch[j] = x[j] + step_scale * delta[j];
        double obj_try = eval_obj(x_try_scratch);
        if (n_evals_out) ++(*n_evals_out);
        if (std::isfinite(obj_try) &&
            (obj_try >= obj_old - 1e-8 || half == MAX_HALVING)) {
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
    // Every trial was non-finite. Leave x and the objective as they were and
    // report a zero step; see the note above.
    obj_out = obj_old;
    return 0.0;
}

// Phase-I feasibility search for the Newton start.
//
// The line search and the constrained-link barrier both reduce to one predicate:
// is the penalized objective finite at this x? A start where it is NOT carries no
// usable information -- every trial along every direction is -Inf, nothing is
// accepted, and the solve stalls where it began. The default latent start x = 0
// is exactly that case for a link carried on eta > 0 (laplace_family_link.h):
// eta is 0, which is the boundary, not the interior.
//
// `coords` are latent slots that shift eta when moved -- the per-process
// intercept, the same slot center_effects_fn folds a removed field level into.
// The routine sweeps a common shift of those slots over a decade ladder in both
// signs and keeps the shift with the LARGEST finite objective, which is a coarse
// intercept-only fit: the analogue of starting a GLM from linkfun(mustart) with
// the slopes at zero, obtained without this layer needing to know the family, the
// link, or the response. It only has to land in the interior -- the Newton loop
// does the rest.
//
// The sweep runs ONLY when the supplied start is already infeasible, so a fit
// whose start has a finite objective -- every fit on an unconstrained link, and
// any warm start inherited from a previous grid point -- takes the single
// evaluation that establishes feasibility and is otherwise unchanged.
//
// Returns whether x is feasible on exit. On false, x is restored to what it was
// and the caller reports the failure rather than iterating on a stalled solve.
inline constexpr int FEASIBLE_START_EXP_LO = -3;
inline constexpr int FEASIBLE_START_EXP_HI = 6;

template<typename EvalObj>
inline bool make_start_feasible(
    Rcpp::NumericVector& x,
    const std::vector<int>& coords,
    int n_x,
    EvalObj eval_obj,
    double& obj_out
) {
    obj_out = eval_obj(x);
    if (std::isfinite(obj_out)) return true;
    if (coords.empty()) return false;

    std::vector<double> x0;
    x0.reserve(coords.size());
    for (int c : coords) x0.push_back((c >= 0 && c < n_x) ? x[c] : 0.0);

    auto apply_shift = [&](double shift) {
        for (std::size_t j = 0; j < coords.size(); j++) {
            const int c = coords[j];
            if (c >= 0 && c < n_x) x[c] = x0[j] + shift;
        }
    };

    double best_obj = 0.0, best_shift = 0.0;
    bool found = false;
    for (int e = FEASIBLE_START_EXP_LO; e <= FEASIBLE_START_EXP_HI; e++) {
        const double mag = std::pow(10.0, (double)e);
        for (int sgn = 0; sgn < 2; sgn++) {
            const double shift = (sgn == 0) ? mag : -mag;
            apply_shift(shift);
            const double obj = eval_obj(x);
            if (std::isfinite(obj) && (!found || obj > best_obj)) {
                best_obj = obj;
                best_shift = shift;
                found = true;
            }
        }
    }

    apply_shift(found ? best_shift : 0.0);
    if (found) obj_out = best_obj;
    return found;
}

// Largest absolute component of a vector. Applied to the gradient of the
// penalized log-posterior at the returned mode it is the solve's ACHIEVED
// residual -- how far from stationary the reported mode actually is. Every
// driver already re-scatters grad at that point for the log-determinant, so
// this costs one pass over n_x. See LaplaceResult::score_max.
inline double max_abs(const std::vector<double>& v) {
    double m = 0.0;
    for (double e : v) m = std::max(m, std::abs(e));
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

// Near-mode gate and patience for the stalled-step convergence path below.
// The gate (predicted objective gain < 1e-6) marks "essentially at a mode"; the
// patience is how many consecutive iterations without a new shortest step count
// as a conditioning-limited stall rather than ongoing progress.
inline constexpr double NEWTON_NEARMODE_GATE  = 1e-6;
inline constexpr int    NEWTON_STALL_PATIENCE = 5;

// Trust-factor schedule for newton_trust_scale below: how hard a step that grew
// the decrement is damped, how fast an undamped one is restored, and the floor.
// The floor is far below anything the fixtures reach (they settle around 0.5);
// it exists so a pathological curvature ratio still contracts rather than
// bottoming out at a scale that cannot.
inline constexpr double NEWTON_TRUST_SHRINK = 0.5;
inline constexpr double NEWTON_TRUST_GROW   = 1.5;
inline constexpr double NEWTON_TRUST_MIN    = 1.0 / 1048576.0;  // 2^-20

// Per-solve state for the Newton convergence test (shortest Newton step seen so
// far, how many iterations have passed without beating it) and for the near-mode
// trust factor. One instance lives for the duration of a single Newton solve;
// the batch solver keeps one per stream.
struct NewtonConvState {
    double best_step = std::numeric_limits<double>::infinity();
    int    stalled = 0;
    double trust = 1.0;
    double prev_decrement = -1.0;   // < 0 until the solve is near the mode
};

// The step scale the line search should open with, given the Newton decrement at
// the current iterate.
//
// Away from the mode this is 1: the full Newton step is tried first, so those
// solves take exactly the trial sequence they always did.
//
// Near the mode the objective can no longer steer. The penalized log-posterior
// is stationary there, so the gain from a step is second order -- about
// decrement / 2 -- while the objective's own accumulation noise is 8 eps |obj|.
// On the random-intercept fixtures of dev_notes/probe_inner_stationarity.R
// (|obj| ~ 4e2) those cross at a decrement near 1e-12, i.e. at a joint score
// near 1e-6, and below that crossing NO rule reading the objective discriminates:
// the absolute acceptance slack waves every trial through, and a tighter
// (Armijo) test is worse still, rejecting genuine progress as noise.
//
// That is only harmless while the Newton direction is trustworthy. It is when the
// Newton weight IS the observed curvature (working_weight_is_observed in
// laplace_family_link.h). Where it is not -- neg_binomial_1's quasi-likelihood
// weight mu / (1 + phi), inverse_gaussian's Fisher weight 1 / (phi mu) -- the
// Newton matrix can understate the true curvature by more than a factor of two,
// and the undamped iteration is then locally DIVERGENT rather than merely slow:
// at phi = 6, sigma_re = 5 the largest eigenvalue of Hw^-1 Ho at the mode is
// 2.23, so I - Hw^-1 Ho has spectral radius 1.23. The iterate walks away from the
// mode geometrically, losing a few parts in 1e9 of objective per step -- under
// any absolute slack -- until the stall test reads the growing step as a
// conditioning floor and stops. Damping to any scale below 2 / 2.23 restores
// contraction; 0.6 gives a spectral radius of 0.40.
//
// The decrement steers where the objective cannot. It is g' H^-1 g, the
// affine-invariant distance to the mode already computed every iteration, and it
// keeps full relative precision exactly where objective differences are noise. So
// below the gate the trust factor halves whenever the decrement GREW -- the step
// just taken overshot -- and relaxes back toward 1 whenever it fell. A solve whose
// decrement never grows below the gate holds trust at 1 throughout and is
// bit-for-bit what it was.
inline double newton_trust_scale(NewtonConvState& st, double decrement) {
    if (!(decrement < NEWTON_NEARMODE_GATE)) {
        st.trust = 1.0;
        st.prev_decrement = -1.0;
        return 1.0;
    }
    if (st.prev_decrement >= 0.0) {
        st.trust = (decrement > st.prev_decrement)
            ? std::max(st.trust * NEWTON_TRUST_SHRINK, NEWTON_TRUST_MIN)
            : std::min(st.trust * NEWTON_TRUST_GROW, 1.0);
    }
    st.prev_decrement = decrement;
    return st.trust;
}

// Newton convergence test, the single source of truth for every solver.
//
// Both paths read the FULL Newton step max|delta|, never the damped step
// max|step_scale delta| that was actually taken. Convergence is a property of the
// iterate -- the proposal H^-1 g at x is small exactly when x is stationary --
// while step_scale is the line search's choice about how much of that proposal to
// trust. Reading the taken step conflates the two, and in the direction that
// matters: a search that had to damp a large proposal to nothing has failed to
// move, which is the opposite of having arrived. With newton_trust_scale able to
// open at 2^-20 that is not hypothetical -- a proposal of 1e-6 damped to the floor
// would otherwise clear a 1e-12 tolerance.
//
// Two paths:
//   1. max|delta| < tol -- the historic step criterion. On a well-conditioned
//      solve this fires exactly when it always did, so those fits are unchanged.
//   2. a stalled step -- the rescue for an ill-conditioned H (a high-order
//      rational SPDE precision Q = Pl' Ci Pl, which squares cond(Pl); a
//      near-singular GMRF). There the Cholesky solve loses too many digits for
//      the step to ever fall below tol: H^-1 amplifies the gradient's rounding
//      residual into a spurious step, and the iteration thrashes around a floor
//      instead of walking down to it, yet the mode is found.
//
// What separates the two is whether the step is still SHRINKING. A solve that is
// converging -- at any rate, however slow -- shortens its step every iteration
// and so keeps setting a new minimum. A solve at its conditioning floor bounces:
// on the rational SPDE fixture of dev_notes/diag_spde_converge.R (cond(H) 1e13)
// the step wanders over 3e-6 .. 4e-5 with per-iteration ratios from 0.25 to 7,
// setting a new minimum only occasionally, so `patience` consecutive misses is
// its signature and not a converging solve's.
//
// The predecessor of this test asked instead whether the DECREMENT had halved,
// which conflates the two: a badly scaled Newton weight converges linearly at
// rate r, the decrement then shrinks by r^2, and any r >= 0.707 reads as a
// stall. That is not hypothetical -- neg_binomial_1's quasi-likelihood weight
// mu / (1 + phi) sits far below the observed curvature at large phi, and on the
// random-intercept fixture of gcol33/tulpa#255 the measured rate reaches the
// 0.707 boundary between phi = 3 (0.57) and phi = 4 (0.70), then passes it
// outright at phi = 6 (0.93). That is exactly where those fits began returning a
// mode whose score was 7e-04 rather than 1e-11, silently costing the exact outer
// gradient five digits. The step-based test leaves them running and they arrive:
// 81 iterations at phi = 4, 323 at phi = 6, both at 2e-11
// (dev_notes/probe_inner_stationarity.R). A solve that needs more than max_iter
// now exits with converged = false and a score_max the caller can gate on,
// rather than a convergence flag on a mode that never settled.
inline bool newton_converged(const std::vector<double>& delta,
                             const std::vector<double>& grad,
                             double step_scale, int n_x, double tol,
                             NewtonConvState& st) {
    // step_scale == 0 is line_search_backtrack reporting that no trial along the
    // direction had a finite objective, so x was left unchanged. Path 1 below
    // would read that stall as convergence -- max|0 * delta| is 0, under any tol
    // -- and stamp converged=true on an iterate the solver could not move. A
    // stalled search is not a mode: return false and let the solve either
    // recover on the next direction or exit at max_iter with converged=false.
    // The line search never returns 0 while the objective stays finite, so no
    // currently-working solve reaches this.
    if (step_scale <= 0.0) return false;
    const double step = max_abs(delta);
    if (step < tol) return true;
    // Tracked unconditionally, so `best_step` is the true running minimum over
    // the whole solve rather than over the near-mode tail only.
    const bool progress = step < st.best_step;
    if (progress) st.best_step = step;
    if (newton_decrement(grad, delta, n_x) >= NEWTON_NEARMODE_GATE || progress) {
        st.stalled = 0;
        return false;
    }
    return ++st.stalled >= NEWTON_STALL_PATIENCE;
}

// The step taken when the factorization of this iteration's Hessian failed:
// move a short damped distance along whatever finite direction the conditioner
// did return, and mark the objective stale so the next iteration re-evaluates
// it. `delta` may be entirely non-finite, in which case the iterate does not
// move and the next iteration re-factorizes from the same point.
//
// This is the damped NEWTON direction, not a gradient step: every conditioner in
// the engine returns a usable direction alongside its failure report, and that
// direction carries the curvature scaling a raw gradient step throws away -- a
// fixed-length gradient step is negligible or enormous depending on the units of
// the predictors. Every driver takes this step, so a solve that meets a failed
// factorization walks the same path whichever Hessian container it runs on.
inline void newton_damped_fallback(
    Rcpp::NumericVector& x, const std::vector<double>& delta, int n_x,
    bool& obj_valid
) {
    for (int j = 0; j < n_x; j++) {
        if (std::isfinite(delta[j])) x[j] += 0.1 * delta[j];
    }
    obj_valid = false;
}

// Everything a Newton iteration does AFTER a successful solve: the lazy
// objective refresh, the Newton decrement, the trust-scaled backtracking line
// search, and the convergence test. Reads `grad` / `delta` / `x_try` off the
// scratch, so it serves the dense and sparse containers alike -- neither is
// touched here, only the step the solve wrote.
//
// `obj_current` / `obj_valid` / `conv_state` carry across iterations, and
// `n_iter_out` records the iteration count the caller reports. Returns true when
// the solve has converged, which is the caller's signal to break.
template <typename Scratch, typename EvalObj>
inline bool newton_step_tail(
    Rcpp::NumericVector& x,
    Scratch& scratch,
    int n_x, int iter, double tol,
    EvalObj eval_objective,
    double& obj_current,
    bool& obj_valid,
    NewtonConvState& conv_state,
    int& n_iter_out
) {
    if (!obj_valid) {
        obj_current = eval_objective(x);
        obj_valid = true;
    }

    double slope = newton_decrement(scratch.grad, scratch.delta, n_x);
    double step_scale = line_search_backtrack(
        x, scratch.delta, n_x, obj_current, slope, eval_objective,
        obj_current, scratch.x_try, nullptr,
        newton_trust_scale(conv_state, slope)
    );

    n_iter_out = iter + 1;
    return newton_converged(scratch.delta, scratch.grad, step_scale, n_x, tol,
                            conv_state);
}

// One Newton iteration, shared by the single-arm and dense joint drivers.
//
// `refresh_grad_hess()` and `cholesky_solve()` are the only steps that differ
// between them -- the first recomputes the linear predictor(s) and scatters the
// gradient and Hessian into the scratch (compute_eta + scatter_grad_hess for one
// arm, compute_eta_joint + scatter_joint for several), the second factorizes
// whichever container holds it and writes `scratch.delta`. Everything else is
// newton_damped_fallback and newton_step_tail above, which the sparse joint
// driver calls directly around its own factor-reuse block.
template <typename Scratch, typename RefreshFn, typename SolveFn,
          typename EvalObj>
inline bool newton_step(
    Rcpp::NumericVector& x,
    Scratch& scratch,
    int n_x, int iter, double tol,
    RefreshFn refresh_grad_hess,
    SolveFn cholesky_solve,
    EvalObj eval_objective,
    double& obj_current,
    bool& obj_valid,
    NewtonConvState& conv_state,
    int& n_iter_out
) {
    refresh_grad_hess();

    if (!cholesky_solve()) {
        newton_damped_fallback(x, scratch.delta, n_x, obj_valid);
        n_iter_out = iter + 1;
        return false;
    }

    return newton_step_tail(x, scratch, n_x, iter, tol, eval_objective,
                            obj_current, obj_valid, conv_state, n_iter_out);
}

// Laplace log-marginal: log_lik + log_prior - 0.5 log|H| + 0.5 n log(2 pi).
inline double finalize_log_marginal(
    double log_lik, double log_prior, double log_det_H, int n_x
) {
    return log_lik + log_prior - 0.5 * log_det_H + 0.5 * n_x * std::log(2.0 * M_PI);
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_LOOP_H
