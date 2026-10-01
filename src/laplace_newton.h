// laplace_newton.h
// PIRLS-equivalent Newton/Fisher scoring solver for Laplace modes.

#ifndef TULPA_LAPLACE_NEWTON_H
#define TULPA_LAPLACE_NEWTON_H

#include "laplace_cholesky.h"
#include "laplace_cholesky_dispatch.h"  // dispatch_factor_solve, dispatch_factor_log_det
#include "laplace_family_curvature.h"   // curvature3_obs_for_family
#include "laplace_family_link.h"
#include "laplace_newton_loop.h"        // eval_*, line_search_backtrack, finalize_log_marginal
#include "inner_cila.h"                 // run_inner_cila
#include "inner_laplace_is.h"           // compute_inner_is_curve
#include "subspace_debias.h"            // compute_subspace_debias
#include "inner_laplace_skew.h"         // compute_inner_skew_gamma3
#include "inv_block_extract.h"          // extract_inv_diag_blocks
#include "sparse_cholesky.h"
#include "laplace_profile.h"            // TULPA_PROFILE_PHASE
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

constexpr int SPARSE_THRESHOLD = 200;

// The crossover for a solve that assembles into a structural-pattern builder
// (laplace_newton_solve_ll's `sparse_H`). That route neither zeroes nor reads a
// dense matrix, so CHOLMOD overtakes the dense Cholesky far earlier than on the
// dense route's discovered pattern: on a one-term random-intercept spec solve,
// timed per fit with its CHOLMOD analysis included, dense / sparse were 33 / 40
// us at n_x = 32, 90 / 67 at 62, 410 / 130 at 122 and 1703 / 210 at 192.
constexpr int STRUCTURAL_SPARSE_THRESHOLD = 50;

// Whether a Newton solve of dimension n_x factors sparsely. sparse_override:
// 0 = auto (size threshold), > 0 = force sparse, < 0 = force dense. The override
// lets the test suite drive one problem through both factorization paths for a
// dense == sparse equivalence gate, mirroring the joint path's force_sparse.
inline bool newton_use_sparse(int n_x, int sparse_override,
                              int threshold = SPARSE_THRESHOLD) {
    return (sparse_override == 0) ? (n_x >= threshold)
                                  : (sparse_override > 0);
}
// SPARSE_DROP_TOL lives in laplace_cholesky_dispatch.h as SPARSE_DROP_TOL_DISPATCH.
// MAX_HALVING is defined in laplace_newton_loop.h.

// Per-thread scratch for the single-arm Newton solver. All buffers are
// allocated once by the caller (single-threaded, outside any OpenMP parallel
// region) and reused across grid points and Newton iterations.
//
// Two reasons to hoist:
//   1. Rcpp::NumericVector — Rf_allocVector is not thread-safe.
//   2. DenseVec/DenseMat   — std::vector allocation is thread-safe but the
//      per-iter (n_x + 1) mallocs for grad / H / delta hit the central
//      allocator under concurrent outer-grid threads, manifesting as
//      Amdahl-shaped serial overhead inside the parallel region. Allocating
//      once and zero-ing each iter eliminates the contention.
//
// DenseCholeskyScratch holds raw std::vector buffers for the dense-fallback
// Cholesky factorization so that path is also Rcpp-free on the hot loop.
struct NewtonScratch {
    Rcpp::NumericVector x;       // size n_x — current Newton iterate
    Rcpp::NumericVector x_try;   // size n_x — step-halving trial
    Rcpp::NumericVector eta;     // size N — current linear predictor
    Rcpp::NumericVector eta_tmp; // size N — eval_objective trial
    DenseVec  grad;              // size n_x — Newton gradient, zeroed per iter
    DenseMat  H;                 // n_x x n_x — Newton Hessian, zeroed per iter
    DenseVec  delta;             // size n_x — Newton step, zeroed per iter
    DenseCholeskyScratch chol;   // raw L/z buffers for dense fallback
    // Structural-pattern Hessian for a caller whose scatter can assemble
    // sparsely (laplace_newton_solve_ll's `sparse_H`). The caller installs the
    // pattern; it persists across solves on this scratch, so a pattern is built
    // once per thread rather than once per solve. `H` stays allocated as the
    // matrix the dense Cholesky falls back to.
    SparseHessianBuilder H_sparse;

    void allocate(int n_x, int N) {
        x       = Rcpp::NumericVector(n_x, 0.0);
        x_try   = Rcpp::NumericVector(n_x, 0.0);
        eta     = Rcpp::NumericVector(N, 0.0);
        eta_tmp = Rcpp::NumericVector(N, 0.0);
        grad.assign(n_x, 0.0);
        H.assign(n_x, DenseVec(n_x, 0.0));
        delta.assign(n_x, 0.0);
        chol.ensure(n_x);
    }

    // Clear grad / H / delta before a fresh scatter. Sizes are fixed at
    // allocate(); this only zeros values, no malloc traffic.
    void zero_for_iter() {
        std::fill(grad.begin(), grad.end(), 0.0);
        H.zero();
        std::fill(delta.begin(), delta.end(), 0.0);
    }
};

// The Newton Hessian as laplace_newton_solve_ll holds it: assembled, factored,
// read for its log-determinant and exported. Two stores share the loop.
//
// DenseNewtonHessian is the row-major scratch.H, factored through
// dispatch_factor_solve (CHOLMOD on a pattern discovered from the values when
// prefer_sparse, the dense Cholesky otherwise). Every Newton step zeroes and
// reads all n_x^2 entries of it.
struct DenseNewtonHessian {
    NewtonScratch& s;
    SparseCholeskySolver& solver;
    bool prefer_sparse;
    int n_x;

    DenseMat& hessian() { return s.H; }
    void zero() { s.zero_for_iter(); }
    bool factor_solve() {
        return dispatch_factor_solve(s.H, s.grad, s.delta, n_x, solver,
                                     prefer_sparse, s.chol);
    }
    bool factor_log_det(double& log_det) {
        return dispatch_factor_log_det(s.H, n_x, solver, prefer_sparse, s.chol,
                                       log_det);
    }
    bool sparse_live() const { return prefer_sparse && solver.factored(); }
    void export_lower_csc(LaplaceResult& r) const {
        dense_to_csc_lower_drop_raw(s.H, n_x, SPARSE_DROP_TOL_DISPATCH,
                                    r.Q_csc_p, r.Q_csc_i, r.Q_csc_x);
    }
};

// SparseNewtonHessian assembles into a SparseHessianBuilder whose pattern the
// caller derived from the model's structure, so a step touches only the
// structural nonzeros, and CHOLMOD factors the builder's arrays in place.
struct SparseNewtonHessian {
    NewtonScratch& s;
    SparseHessianBuilder& H;
    SparseCholeskySolver& solver;
    int n_x;

    SparseHessianBuilder& hessian() { return H; }
    void zero() {
        std::fill(s.grad.begin(), s.grad.end(), 0.0);
        std::fill(s.delta.begin(), s.delta.end(), 0.0);
        H.zero();
    }
    bool factor_solve() {
        return dispatch_factor_solve_sparse(H, s.grad, s.delta, n_x, solver,
                                            s.H, s.chol);
    }
    bool factor_log_det(double& log_det) {
        return dispatch_factor_log_det_sparse(H, n_x, solver, s.H, s.chol,
                                              log_det);
    }
    bool sparse_live() const { return solver.factored(); }
    void export_lower_csc(LaplaceResult& r) const {
        sparse_builder_to_csc_lower_drop(H, SPARSE_DROP_TOL_DISPATCH,
                                         r.Q_csc_p, r.Q_csc_i, r.Q_csc_x);
    }
};

// Scratch-aware, likelihood-agnostic Newton solver. The four pre-allocated
// buffers in `scratch` must be sized to (n_x, n_x, N, N). The solver does not
// allocate any Rcpp objects; the caller can therefore drive this from a
// parallel region as long as the SparseCholeskySolver is also thread-local.
//
// The data log-likelihood enters ONLY through `log_lik_fn(eta) -> double`, so
// the loop carries no family knowledge: the family-enum mode finders pass a
// FamilyLogLik (see the forwarding overload below) and the LikelihoodSpec path
// passes a functor backed by spec.ll_double. This is the single Newton loop the
// whole engine shares.
template<typename ComputeEta, typename ScatterGradHess,
         typename CenterEffects, typename ComputeLogPrior, typename LogLik>
LaplaceResult laplace_newton_solve_ll(
    int N, int n_x,
    int max_iter, double tol,
    ComputeEta compute_eta,
    ScatterGradHess scatter_grad_hess,
    CenterEffects center_effects_fn,
    ComputeLogPrior compute_log_prior,
    LogLik log_lik_fn,
    NewtonScratch& scratch,
    const std::vector<double>& x_init,
    SparseCholeskySolver* shared_solver,
    bool store_Q,
    const std::vector<std::pair<int, int>>* inv_block_layout = nullptr,
    int sparse_override = 0,
    // Latent slots the feasibility sweep may shift when the start is outside the
    // likelihood's domain -- the per-process intercepts. nullptr (the default)
    // skips the sweep entirely, which is the behaviour every caller had before
    // it existed; see make_start_feasible in laplace_newton_loop.h.
    const std::vector<int>* feasible_start_coords = nullptr,
    // Inner-Laplace skewness diagnostic (inner_laplace_skew.h), opt-in like
    // store_Q. The caller builds the third-derivative oracle (family ladder,
    // a LikelihoodSpec finite-difference wrapper, or the per-observation tensor
    // contraction of a multi-process spec) because this loop is otherwise
    // likelihood-agnostic; the oracle carries its own decline reason, so a
    // likelihood that ships no third derivative does not report as an unset
    // knob. skew_probe_idx == nullptr with compute_skew = true probes every
    // latent index.
    bool compute_skew = false,
    const std::vector<int>* skew_probe_idx = nullptr,
    const Curvature3Oracle* curvature3 = nullptr,
    // Subspace debias (subspace_debias.h), opt-in and independent of the
    // diagnostics above: the caller selects the flagged coordinates from a
    // previous solve's inner-layer bands and passes them here. nullptr or an
    // empty index set never reaches the sampler, so the solve is unchanged and
    // consumes no random number.
    const SubspaceDebiasOptions* debias = nullptr,
    // Corrected integrated Laplace (inner_cila.h). Runs on the
    // pre-centering iterate and the same live factor, and presents its draws
    // under the same centering fold the reported mode carries.
    const CilaOptions* cila = nullptr,
    // Distinguishes this cell's auxiliary stream from its neighbours' on an
    // outer grid; irrelevant for the deterministic net, load-bearing for the
    // randomized-QMC shifts.
    std::uint64_t cila_cell_key = 0,
    // A builder carrying the structural pattern of this problem's Hessian. When
    // set, the solve factors sparsely from STRUCTURAL_SPARSE_THRESHOLD on, and
    // where `scatter_grad_hess` accepts a SparseHessianBuilder&, the Hessian is
    // assembled into it instead of the dense scratch.H. The scatter must write only inside the pattern; a write
    // off it is counted for the enclosing HessianPatternGuard.
    SparseHessianBuilder* sparse_H = nullptr
) {
    LaplaceResult result;
    result.mode.assign(n_x, 0.0);
    result.converged = false;
    result.n_iter = 0;
    result.log_det_Q = 0.0;
    result.log_marginal = 0.0;

    Rcpp::NumericVector& x = scratch.x;
    if (static_cast<int>(x_init.size()) == n_x) {
        for (int j = 0; j < n_x; j++) x[j] = x_init[j];
    } else {
        for (int j = 0; j < n_x; j++) x[j] = 0.0;
    }
    const bool use_sparse = newton_use_sparse(
        n_x, sparse_override,
        sparse_H ? STRUCTURAL_SPARSE_THRESHOLD : SPARSE_THRESHOLD);

    SparseCholeskySolver local_solver;
    SparseCholeskySolver& sparse_solver = shared_solver ? *shared_solver : local_solver;

    // Do NOT call omp_set_num_threads here. When the outer driver runs us
    // from inside a parallel region we want the inner kernels (per-obs
    // scatter, etc.) to inherit the per-thread context. The closures and the
    // log-lik functor own their own threading.

    // The last objective evaluation: the linear predictor it ran at, that
    // predictor's data log-likelihood, and the iterate it came from. An
    // accepted line-search trial is the next iterate, so the refresh that
    // follows reads its eta from here instead of recomputing it, and the
    // log-marginal after the loop reads the converged iterate's log-likelihood
    // the last trial already paid for. Both are keyed on bits (the iterate for
    // eta, eta for the log-likelihood), so a hit returns exactly what the call
    // would have; compute_eta and log_lik_fn are functions of their arguments.
    std::vector<double> ll_memo_eta;
    std::vector<double> eta_memo_x;
    double ll_memo_value = 0.0;
    auto log_lik_memo = [&](const Rcpp::NumericVector& eta) -> double {
        const std::size_t n = static_cast<std::size_t>(eta.size());
        if (n > 0 && ll_memo_eta.size() == n &&
            std::memcmp(ll_memo_eta.data(), eta.begin(),
                        n * sizeof(double)) == 0) {
            return ll_memo_value;
        }
        ll_memo_value = log_lik_fn(eta);
        ll_memo_eta.assign(eta.begin(), eta.end());
        eta_memo_x.clear();   // eval_objective re-keys it to its own iterate
        return ll_memo_value;
    };

    auto eval_objective = [&](const Rcpp::NumericVector& xv) -> double {
        const double obj = eval_penalized_log_lik_ll(
            xv, compute_eta, compute_log_prior, log_lik_memo, scratch.eta_tmp
        );
        eta_memo_x.assign(xv.begin(), xv.end());
        return obj;
    };
    auto eta_from_memo = [&](const Rcpp::NumericVector& xv,
                             Rcpp::NumericVector& eta_out) -> bool {
        const std::size_t nx = static_cast<std::size_t>(xv.size());
        if (nx == 0 || eta_memo_x.size() != nx ||
            ll_memo_eta.size() != static_cast<std::size_t>(eta_out.size()) ||
            std::memcmp(eta_memo_x.data(), xv.begin(),
                        nx * sizeof(double)) != 0) {
            return false;
        }
        std::copy(ll_memo_eta.begin(), ll_memo_eta.end(), eta_out.begin());
        return true;
    };

    double obj_current = -1e300;
    bool obj_valid = false;
    NewtonConvState conv_state;

    // Move the start into the likelihood's domain before iterating. Without a
    // finite objective at x the line search accepts nothing and the loop would
    // spin max_iter times over a point it cannot leave.
    if (feasible_start_coords) {
        double obj_start = 0.0;
        if (!make_start_feasible(x, *feasible_start_coords, n_x, eval_objective,
                                 obj_start)) {
            result.start_infeasible = true;
            for (int j = 0; j < n_x; j++) result.mode[j] = x[j];
            return result;
        }
        obj_current = obj_start;
        obj_valid = true;
    }

    // Everything from the iterations on runs against one Hessian store.
    auto run = [&](auto& store) {
        // Profiler scopes (tulpa_profile()): eta and scatter here, factorize and
        // line_search inside the shared newton_step, and the final pass below.
        auto refresh_grad_hess = [&]() {
            { TULPA_PROFILE_PHASE(PHASE_ETA);
              if (!eta_from_memo(x, scratch.eta)) compute_eta(x, scratch.eta); }
            store.zero();
            { TULPA_PROFILE_PHASE(PHASE_SCATTER);
              scatter_grad_hess(x, scratch.eta, scratch.grad, store.hessian()); }
        };
        auto cholesky_solve = [&]() -> bool { return store.factor_solve(); };

        for (int iter = 0; iter < max_iter; iter++) {
            if (newton_step(x, scratch, n_x, iter, tol, refresh_grad_hess,
                            cholesky_solve, eval_objective, obj_current, obj_valid,
                            conv_state, result.n_iter)) {
                result.converged = true;
                break;
            }
        }

        // log_marginal belongs to the Newton mode, so everything that enters it is
        // evaluated at the uncentered iterate and `center_effects_fn` runs last,
        // over the reported mode alone. The fold each caller
        // applies preserves eta, so the data log-lik is the same either way, but the
        // log-prior is not: a proper field prior (AR1, proper CAR) is not
        // shift-invariant, and even an intrinsic one moves the beta ridge through
        // the coefficient the fold lands in. Evaluating there reports a Laplace
        // expansion at a non-stationary point, and `score_max` is non-zero by
        // exactly the amount the shift moved off the mode. This is the ordering the
        // joint loops already take.
        refresh_grad_hess();
        result.score_max = max_abs(scratch.grad);

        // A non-finite log-determinant is the plain Cholesky reporting that the
        // Hessian at the returned point is not PD -- a point the solve stopped at
        // without reaching a mode. There is no Laplace expansion there, so the fit
        // says so rather than carrying the value into log_marginal: an -Inf
        // log-determinant is a +Inf log-marginal, which does not merely lose the
        // cell but makes it take the whole outer grid's weight. A PD Hessian never
        // reaches this, so every fit that factorizes is unchanged.
        { TULPA_PROFILE_PHASE(PHASE_LOG_DET);
          result.hessian_pd_at_mode = store.factor_log_det(result.log_det_Q); }
        if (!result.hessian_pd_at_mode) {
            result.log_det_Q = std::numeric_limits<double>::quiet_NaN();
            result.converged = false;
        }

        // Diagonal blocks of H^{-1} for the requested index ranges. Reuses the
        // factor just built for the log-determinant (no refactorization): for each
        // unit column e_j inside a block we solve H v = e_j and read the block
        // rows of v, giving the FULL-inverse block (fixed effects and other blocks
        // marginalized out). Sparse path solves against the live CHOLMOD factor;
        // dense path back-substitutes the live scratch.chol.L. Each block is
        // symmetrized and stored column-major.
        // Withheld where the Hessian at the returned point is not PD: its inverse
        // is not a covariance there.
        //
        // Nothing after the loop refactorizes, so this reading of which factor is
        // live serves both the inverse-block extraction and the skew probes below.
        const bool used_sparse_factor = store.sparse_live();
        result.sparse_factor_live = used_sparse_factor;

        if (result.hessian_pd_at_mode) {
            result.newton_decrement = newton_decrement_live(
                scratch.grad.data(), n_x, used_sparse_factor, sparse_solver,
                scratch.chol);
        }

        if (inv_block_layout && !inv_block_layout->empty() &&
            result.hessian_pd_at_mode) {
            TULPA_PROFILE_PHASE(PHASE_HESSIAN_EXTRACT);
            std::vector<double> z_work;
            if (!used_sparse_factor) z_work.assign(n_x, 0.0);
            auto solve_live = [&](const double* rhs, double* out) {
                if (used_sparse_factor) {
                    sparse_solver.solve(rhs, out, n_x);
                } else {
                    chol_substitute_raw(scratch.chol.L.data(), n_x, rhs, out,
                                        z_work.data());
                }
            };
            extract_inv_diag_blocks(solve_live, n_x, *inv_block_layout,
                                    /*constr=*/nullptr, InvBlockSymmetry::Average,
                                    result.re_cov_flat, result.re_cov_block_sizes);
        }

        double log_lik, log_prior;
        { TULPA_PROFILE_PHASE(PHASE_LOG_LIK_PRIOR);
          log_lik = log_lik_memo(scratch.eta);
          log_prior = compute_log_prior(x, scratch.eta); }

        result.log_marginal = finalize_log_marginal(log_lik, log_prior, result.log_det_Q, n_x);

        if (store_Q) {
            TULPA_PROFILE_PHASE(PHASE_HESSIAN_EXTRACT);
            // Drop tolerance matches the sparse-Cholesky dispatch path so the
            // exported CSC pattern is consistent with the in-loop solve when
            // n_x >= SPARSE_THRESHOLD.
            store.export_lower_csc(result);
            result.Q_csc_n = n_x;
        }

        // The Newton-converged iterate, before the presentation centering at the end
        // of the loop. Every probe of the inner layer -- gamma_3, the importance
        // curve, the subspace sampler and the correction -- reads this point,
        // because it is the one the live factor and the reported log_marginal
        // belong to.
        std::vector<double> pre_center_x(n_x);
        for (int j = 0; j < n_x; j++) pre_center_x[j] = x[j];

        {   // post-mode probes of the inner layer, timed as one phase
            TULPA_PROFILE_PHASE(PHASE_INNER_DIAG);
            if (compute_skew) {
                std::vector<int> all_idx;
                const std::vector<int>& probe =
                    inner_probe_indices(n_x, skew_probe_idx, all_idx);
                if (!result.converged) {
                    // gamma_3 is a cubic expansion ABOUT the mode and the inner k-hat an
                    // importance ratio against the Gaussian AT it, so neither exists at
                    // a point the solve stopped short of. Emitting the indices unscored
                    // is what separates that from the diagnostic never having been
                    // requested.
                    inner_probe_decline(result, probe, "not_converged");
                } else {
                    Curvature3Oracle no_oracle;
                    InnerSkewOutcome sk = compute_inner_skew_gamma3(
                        n_x, N, pre_center_x, scratch.chol, sparse_solver,
                        used_sparse_factor, compute_eta, x, scratch.eta, scratch.eta_tmp,
                        curvature3 ? *curvature3 : no_oracle, probe
                    );
                    result.inner_skew = std::move(sk.gamma3);
                    result.inner_skew_gamma1 = std::move(sk.gamma1);
                    result.inner_skew_gamma1_declined = sk.gamma1_declined;
                    result.inner_skew_idx = probe;
                    result.inner_skew_dropped = sk.n_nonfinite_dropped;
                    result.inner_skew_declined = sk.declined;

                    // The likelihood-agnostic inner k-hat over the same probed subspace,
                    // along the same conditional-mean curve the cubic term just walked.
                    // It reads the joint density through the loop's own penalized
                    // objective, so it does not depend on the third-derivative oracle
                    // and stands where gamma_3 declines.
                    InnerISOutcome is_out = compute_inner_is_curve(
                        n_x, pre_center_x, scratch.chol, sparse_solver,
                        used_sparse_factor, eval_objective, x, probe
                    );
                    result.inner_is_z          = std::move(is_out.z);
                    result.inner_is_log_joint  = std::move(is_out.log_joint);
                    result.inner_is_sigma      = std::move(is_out.sigma);
                    result.inner_is_declined   = is_out.declined;
                }
            }

            run_subspace_debias(result, n_x, pre_center_x, scratch.chol,
                                sparse_solver, used_sparse_factor,
                                eval_objective, x, debias);

            // The correction reads the same pre-centering iterate, and presents each
            // draw through the loop's own centering fold so a drawn coefficient is in
            // the coordinates the reported mode is in.
            run_inner_cila(result, n_x, pre_center_x, scratch.chol, sparse_solver,
                           used_sparse_factor, eval_objective,
                           [&](Rcpp::NumericVector& xv) { center_effects_fn(xv); },
                           x, cila, cila_cell_key);
        }

        center_effects_fn(x);
        for (int j = 0; j < n_x; j++) result.mode[j] = x[j];
    };

    // The sparse store is instantiated only for a scatter that can write into
    // a builder; any other scatter compiles against the dense store alone.
    if constexpr (std::is_invocable_v<ScatterGradHess&,
                                      const Rcpp::NumericVector&,
                                      const Rcpp::NumericVector&, DenseVec&,
                                      SparseHessianBuilder&>) {
        if (sparse_H != nullptr && use_sparse) {
            SparseNewtonHessian store{scratch, *sparse_H, sparse_solver, n_x};
            run(store);
            return result;
        }
    }
    DenseNewtonHessian store{scratch, sparse_solver, use_sparse, n_x};
    run(store);
    return result;
}

// Family-enum forwarder (scratch-aware). Wraps the built-in family log-lik as
// the functor and delegates to the shared loop above, so the family-string
// callers (laplace_core*, the nested ST driver, spde_qbuilder) keep their exact
// signature while the loop body lives in one place.
template<typename ComputeEta, typename ScatterGradHess,
         typename CenterEffects, typename ComputeLogPrior>
LaplaceResult laplace_newton_solve(
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const std::string& family,
    double phi,
    int N, int n_x,
    int max_iter, double tol, int n_threads,
    ComputeEta compute_eta,
    ScatterGradHess scatter_grad_hess,
    CenterEffects center_effects_fn,
    ComputeLogPrior compute_log_prior,
    NewtonScratch& scratch,
    const std::vector<double>& x_init,
    SparseCholeskySolver* shared_solver,
    bool store_Q,
    const std::vector<std::pair<int, int>>* inv_block_layout = nullptr,
    bool compute_skew = false,
    const std::vector<int>* skew_probe_idx = nullptr
) {
    FamilyLogLik ll{&y, &n_trials, N, family, phi, n_threads};
    ll.prepare();
    Curvature3Oracle curvature3;
    if (compute_skew) {
        curvature3.scalar = [&y, &n_trials, &family, phi](int j, double eta_j) -> double {
            return curvature3_obs_for_family(y[j], n_trials[j], eta_j, family, phi);
        };
    }
    return laplace_newton_solve_ll(
        N, n_x, max_iter, tol,
        compute_eta, scatter_grad_hess, center_effects_fn, compute_log_prior,
        ll, scratch, x_init, shared_solver, store_Q, inv_block_layout,
        0, nullptr, compute_skew, skew_probe_idx, &curvature3
    );
}

// Convenience overload: allocates scratch locally. Used by the standalone
// laplace_mode_* entry points that are called once per R-export and do not
// participate in any outer-grid parallelism. NOT safe to call from inside an
// OpenMP parallel region — use the scratch-aware overload above for that.
template<typename ComputeEta, typename ScatterGradHess,
         typename CenterEffects, typename ComputeLogPrior>
LaplaceResult laplace_newton_solve(
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const std::string& family,
    double phi,
    int N, int n_x,
    int max_iter, double tol, int n_threads,
    ComputeEta compute_eta,
    ScatterGradHess scatter_grad_hess,
    CenterEffects center_effects_fn,
    ComputeLogPrior compute_log_prior,
    const Rcpp::NumericVector& x_init = Rcpp::NumericVector(),
    SparseCholeskySolver* shared_solver = nullptr,
    bool store_Q = false,
    const std::vector<std::pair<int, int>>* inv_block_layout = nullptr,
    bool compute_skew = false,
    const std::vector<int>* skew_probe_idx = nullptr
) {
    NewtonScratch scratch;
    scratch.allocate(n_x, N);
    std::vector<double> x_init_vec;
    if (x_init.size() == n_x) {
        x_init_vec.assign(x_init.begin(), x_init.end());
    }
    // n_threads flows into laplace_newton_solve, which sizes its own
    // regions; no process-global omp_set_num_threads here.
    return laplace_newton_solve(
        y, n_trials, family, phi, N, n_x,
        max_iter, tol, n_threads,
        compute_eta, scatter_grad_hess, center_effects_fn, compute_log_prior,
        scratch, x_init_vec, shared_solver, store_Q, inv_block_layout,
        compute_skew, skew_probe_idx
    );
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_H
