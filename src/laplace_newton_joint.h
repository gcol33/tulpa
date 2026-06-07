// laplace_newton_joint.h
// PIRLS-equivalent Newton solver for *joint multi-likelihood* Laplace modes.
//
// Mirrors laplace_newton.h::laplace_newton_solve, but the data side is a
// vector of arms rather than a single (y, family, phi) bundle. Each arm
// carries its own (y, n_trials, family, phi, N); the prior side and the
// latent vector x are shared across arms.
//
// Callbacks operate on a vector of per-arm eta vectors:
//   - compute_eta_joint(x, etas): caller fills etas[k] for each arm.
//   - scatter_joint(x, etas, grad, H): caller scatters per-arm contributions
//                                       and shared prior into the joint (g, H).
//   - compute_log_prior_joint(x, etas): joint log p(x | theta).
//   - center_effects_fn(x): post-step centering of structured blocks.
//
// The Newton loop, Cholesky dispatch, line search, log_det and final
// log-marginal machinery are identical to the single-arm path — they
// operate on the joint (g, H, x) without caring how many arms contributed.

#ifndef TULPA_LAPLACE_NEWTON_JOINT_H
#define TULPA_LAPLACE_NEWTON_JOINT_H

#include "laplace_builtin_family_spec.h"  // builtin_family_spec, BuiltinFamilyResponse
#include "laplace_cholesky.h"
#include "laplace_cholesky_dispatch.h"
#include "laplace_family_link.h"
#include "laplace_newton.h"          // SPARSE_THRESHOLD
#include "laplace_newton_loop.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// One arm of a joint Laplace fit. POD-ish wrapper over Rcpp containers so
// the joint solver can iterate per arm without templating on the arm count.
struct JointArm {
    Rcpp::NumericVector y;         // [N]
    Rcpp::IntegerVector n_trials;  // [N]
    std::string         family;    // built-in family (ignored when spec != null)
    double              phi;        // built-in dispersion (ignored when spec)
    int                 N;
    // Grouped beta sufficient statistics (gcol33/tulpaObs#49). Optional; when
    // present (and family == "beta") row i collapses n_trials[i] exchangeable
    // beta obs sharing this row's linear predictor, with slog_y[i] = sum log(y)
    // and slog_1my[i] = sum log(1-y). Empty => ungrouped per-obs path.
    Rcpp::NumericVector slog_y;
    Rcpp::NumericVector slog_1my;
    // Per-arm field coefficient: multiplied into the field amplitude this
    // arm sees on each shared latent block (sigma_arm = field_coef * sigma).
    // Default 1.0 = donor behaviour (existing non-copy arms). `0.0` means
    // this arm carries no field; the scatter / compute_eta paths short-
    // circuit on a per-block per-arm `d_eff == 0` check.
    //
    // A hyperparam-driven field coefficient is materialized R-side onto the
    // copy block's per-arm sigma axes; `field_coef` carries only the per-arm
    // CONSTANT multiplier on top. (See nested_laplace_joint.R's desugaring
    // of `copy = list(arm, alpha_grid)` into `responses[[X]]$field_coef =
    // list(name = "alpha", grid = G)`.)
    double              field_coef = 1.0;
    // Per-arm cell coupling (gcol33/tulpa#32 Change 2b).
    //   coupled = true  -> the inner Newton skips this arm's per-obs scatter;
    //                      the CellCouplingSpec's evaluate_cell() writes its
    //                      gradient + (diagonal) Hessian contribution per cell,
    //                      and the joint kernel scatters them through the same
    //                      X / RE / latent bookkeeping as the per-obs path.
    //   cell_obs_map[i] = 1-based cell id for row i of this arm; the kernel
    //                      inverts it once into per-cell row lists for the
    //                      per-cell branch. Length must equal arms[k].N when
    //                      coupled == true; otherwise ignored.
    bool                       coupled = false;
    Rcpp::IntegerVector        cell_obs_map;
    // Optional model-supplied likelihood (tulpaGlmm / tulpaObs custom arms).
    // When spec != nullptr the joint solver routes this arm's score, Fisher
    // curvature and log-lik through it instead of the built-in family closed
    // forms; the pointees (spec, response, and any data/layout/params it reads)
    // must outlive the fit. The spec must be single-process (n_processes == 1).
    const LikelihoodSpec*      spec          = nullptr;
    const void*                response_data = nullptr;
    const ModelData*           data          = nullptr;
    const ParamLayout*         layout        = nullptr;
    const std::vector<double>* params        = nullptr;
};

// A resolved single-process likelihood for one joint arm: everything the spec
// callbacks need, with stable storage owned by JointArmSpecs. The joint scatter
// and log-lik read every arm only through this view, so the joint path has a
// single spec-driven likelihood boundary (dev_notes/plans/clean_migration.md Phase L / L4).
struct ArmSpecView {
    const LikelihoodSpec*      spec          = nullptr;
    const void*                response_data = nullptr;
    const ModelData*           data          = nullptr;
    const ParamLayout*         layout        = nullptr;
    const std::vector<double>* params        = nullptr;
};

// Per-obs eta-space score + Fisher working weight for one arm, sourced through
// its spec view (single process => scalars). Returns a GradHess so existing
// scatter bodies (gh.grad / gh.neg_hess) are unchanged.
inline GradHess arm_grad_hess(const ArmSpecView& view, int i, double eta_i) {
    GradHess gh;
    view.spec->eta_weights_fn(
        i, &eta_i, 0.0, 0.0,
        *view.params, *view.data, *view.layout, view.response_data,
        &gh.grad, &gh.neg_hess);
    return gh;
}

// Owns the resolved spec views for a joint fit. Arms with a model-supplied spec
// borrow it; otherwise a built-in family spec + response is materialized here
// (single source of truth: builtin_family_spec). The empty ModelData /
// ParamLayout / params satisfy the spec-callback signature for built-in
// families, which ignore them. Pointer stability: builtin_specs / responses are
// reserved to n_arms up front so no reallocation invalidates the view pointers.
struct JointArmSpecs {
    std::vector<LikelihoodSpec>        builtin_specs;
    std::vector<BuiltinFamilyResponse> builtin_responses;
    ModelData                          empty_data;
    ParamLayout                        empty_layout;
    std::vector<double>                empty_params;
    std::vector<ArmSpecView>           views;
    // Per-arm pointer into builtin_responses (nullptr for model-supplied arms).
    // Lets the driver refresh per-cell-mutable dispersion after prep.
    std::vector<BuiltinFamilyResponse*> arm_builtin_response;

    // Refresh built-in dispersion from the live arms. The phi_grid hyperparameter
    // axis rewrites arm.phi before each inner solve, so the response (snapshotted
    // at build) must track it. No-op for model-supplied arms, which own their
    // dispersion through the spec. Mirrors the pre-existing arm.phi rewrite, so
    // it inherits the same serial-outer-grid assumption as phi_grid.
    void sync_dispersion(const std::vector<JointArm>& arms) {
        for (size_t k = 0; k < arm_builtin_response.size(); k++) {
            if (arm_builtin_response[k]) arm_builtin_response[k]->phi = arms[k].phi;
        }
    }
};

// Populate an already-constructed JointArmSpecs in place. Required for building
// a per-outer-thread pool of specs: JointArmSpecs is self-referential (each
// ArmSpecView points at the owner's builtin_responses storage AND at the owner's
// own empty_data / empty_layout / empty_params members), so it cannot be moved
// or copied into a pool slot without dangling those pointers. Constructing each
// pool element in place via this routine keeps every pointer valid.
inline void build_joint_arm_specs_into(const std::vector<JointArm>& arms,
                                       JointArmSpecs& s) {
    const int n = static_cast<int>(arms.size());
    s.builtin_specs.clear();
    s.builtin_responses.clear();
    s.builtin_specs.reserve(n);
    s.builtin_responses.reserve(n);
    s.views.assign(n, ArmSpecView{});
    s.arm_builtin_response.assign(n, nullptr);
    for (int k = 0; k < n; k++) {
        const JointArm& a = arms[k];
        if (a.spec) {
            s.views[k] = ArmSpecView{
                a.spec, a.response_data,
                a.data   ? a.data   : &s.empty_data,
                a.layout ? a.layout : &s.empty_layout,
                a.params ? a.params : &s.empty_params
            };
        } else {
            s.builtin_specs.push_back(builtin_family_spec(a.family));
            BuiltinFamilyResponse r;
            r.y        = (a.N > 0)              ? REAL(a.y)            : nullptr;
            r.n_trials = (a.n_trials.size() > 0) ? INTEGER(a.n_trials) : nullptr;
            r.N        = a.N;
            r.family   = a.family;
            r.phi      = a.phi;
            r.slog_y   = (a.slog_y.size()   > 0) ? REAL(a.slog_y)   : nullptr;
            r.slog_1my = (a.slog_1my.size() > 0) ? REAL(a.slog_1my) : nullptr;
            s.builtin_responses.push_back(r);
            s.views[k] = ArmSpecView{
                &s.builtin_specs.back(), &s.builtin_responses.back(),
                &s.empty_data, &s.empty_layout, &s.empty_params
            };
            s.arm_builtin_response[k] = &s.builtin_responses.back();
        }
    }
}

// Convenience wrapper returning a freshly built JointArmSpecs by value. Safe for
// a single fit-scoped specs object (NRVO / move keeps the std::vector buffers, so
// the views into builtin_responses stay valid); for a per-thread POOL use
// build_joint_arm_specs_into on an in-place slot instead (see its note).
inline JointArmSpecs build_joint_arm_specs(const std::vector<JointArm>& arms) {
    JointArmSpecs s;
    build_joint_arm_specs_into(arms, s);
    return s;
}

// Sum of per-arm log-likelihoods at the current per-arm etas, sourced through
// each arm's resolved spec view (single process => ll_double sees one eta).
// When `skip_arm` is non-null, arms whose `skip_arm[k]` is true contribute 0
// (used by the cell-coupling path to skip coupled arms' per-obs sum so the
// per-cell branch can add its own log-density contribution).
inline double compute_total_log_lik_joint(
    const std::vector<ArmSpecView>& views,
    const std::vector<Rcpp::NumericVector>& etas,
    int n_threads,
    const std::vector<bool>* skip_arm = nullptr
) {
    const double zd = 0.0;  // logit_zi / logit_oi are unused at np == 1
    double total = 0.0;
    for (size_t k = 0; k < views.size(); k++) {
        if (skip_arm && k < skip_arm->size() && (*skip_arm)[k]) continue;
        const ArmSpecView& v = views[k];
        const Rcpp::NumericVector& eta = etas[k];
        const int N = static_cast<int>(eta.size());
        double sub = 0.0;
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:sub) schedule(static) \
            num_threads(n_threads > 0 ? n_threads : 1) if(n_threads > 1)
        #endif
        for (int i = 0; i < N; i++) {
            double eta_i = eta[i];
            sub += v.spec->ll_double(i, &eta_i, zd, zd,
                                     *v.params, *v.data, *v.layout,
                                     v.response_data);
        }
        total += sub;
    }
    return total;
}

// Joint data log-lik as a functor of the per-arm etas, sourced through each
// arm's resolved spec view. The joint Newton loop reads the data log-lik only
// through `log_lik_fn(etas) -> double`; the built-in family enters solely via
// build_joint_arm_specs, so model packages (tulpaGlmm / tulpaObs) can supply a
// custom per-arm likelihood with no family-enum extension. The borrowed views
// vector must outlive the fit.
struct JointSpecLogLik {
    const std::vector<ArmSpecView>* views = nullptr;
    int n_threads = 1;
    // When non-null, `skip_arm[k] = true` excludes arm k's per-obs sum.
    // The cell-coupling path uses this to skip coupled arms and add the
    // per-cell log-density via `cell_coupling_log_lik_fn` instead.
    const std::vector<bool>* skip_arm = nullptr;
    // Optional cell-coupling log-density adder. Called with `etas` once
    // per log-lik evaluation; returns the spec's sum of `evaluate_cell()`
    // log-densities across all cells. nullptr -> separable default
    // (per-obs sum over every arm).
    std::function<double(const std::vector<Rcpp::NumericVector>&)>
                                    cell_coupling_log_lik_fn = nullptr;
    double operator()(const std::vector<Rcpp::NumericVector>& etas) const {
        double total = compute_total_log_lik_joint(*views, etas, n_threads,
                                                    skip_arm);
        if (cell_coupling_log_lik_fn) total += cell_coupling_log_lik_fn(etas);
        return total;
    }
};

// Penalised log-lik (joint), likelihood-agnostic: the data log-lik enters as a
// functor of the per-arm etas, mirroring eval_penalized_log_lik_ll. `etas_scratch`
// is the caller's pre-allocated per-arm eta buffer set; we never reallocate.
template<typename ComputeEtaJoint, typename ComputeLogPriorJoint, typename JointLogLik>
inline double eval_penalized_log_lik_joint_ll(
    const Rcpp::NumericVector& x,
    ComputeEtaJoint compute_eta_joint,
    ComputeLogPriorJoint compute_log_prior_joint,
    JointLogLik log_lik_fn,
    std::vector<Rcpp::NumericVector>& etas_scratch
) {
    compute_eta_joint(x, etas_scratch);
    double ll = log_lik_fn(etas_scratch);
    double lp = compute_log_prior_joint(x, etas_scratch);
    return ll + lp;
}

// Per-thread scratch for the joint Newton solver. Same role as NewtonScratch
// but with per-arm eta vectors. Allocate once, single-threaded outside any
// OpenMP region. See NewtonScratch comment for why grad / H / delta are
// hoisted: per-iter std::vector allocation is thread-safe but contends on the
// central allocator under concurrent outer-grid threads, eating parallel
// efficiency.
struct NewtonScratchJoint {
    Rcpp::NumericVector x;       // size n_x
    Rcpp::NumericVector x_try;   // size n_x
    std::vector<Rcpp::NumericVector> etas;      // size = arms.size(), each N_k
    std::vector<Rcpp::NumericVector> etas_tmp;  // same shape, line-search buffer
    DenseVec  grad;              // size n_x, zeroed per iter
    DenseMat  H;                 // n_x x n_x, zeroed per iter
    DenseVec  delta;             // size n_x, zeroed per iter
    DenseCholeskyScratch chol;   // raw L/z buffers for dense fallback

    void allocate(int n_x, const std::vector<JointArm>& arms) {
        x       = Rcpp::NumericVector(n_x, 0.0);
        x_try   = Rcpp::NumericVector(n_x, 0.0);
        etas.clear();     etas.reserve(arms.size());
        etas_tmp.clear(); etas_tmp.reserve(arms.size());
        for (const JointArm& a : arms) {
            etas.emplace_back(a.N, 0.0);
            etas_tmp.emplace_back(a.N, 0.0);
        }
        grad.assign(n_x, 0.0);
        H.assign(n_x, DenseVec(n_x, 0.0));
        delta.assign(n_x, 0.0);
        chol.ensure(n_x);
    }

    void zero_for_iter() {
        std::fill(grad.begin(), grad.end(), 0.0);
        H.zero();
        std::fill(delta.begin(), delta.end(), 0.0);
    }
};

// Scratch-aware, likelihood-agnostic joint Newton solver. Like the single-arm
// laplace_newton_solve_ll, the data log-lik enters ONLY through
// `log_lik_fn(etas) -> double`, so the loop carries no family knowledge: the
// nested joint driver passes a JointSpecLogLik backed by each arm's resolved
// LikelihoodSpec (built-in family or model-supplied), so this is the single
// joint Newton loop for every likelihood. Performs zero Rcpp allocations once
// `scratch` is supplied; safe inside an OpenMP parallel region with a
// thread-local `shared_solver`.
template<typename ComputeEtaJoint, typename ScatterJoint,
         typename CenterEffects, typename ComputeLogPriorJoint, typename JointLogLik>
LaplaceResult laplace_newton_solve_joint_ll(
    int n_x,
    int max_iter, double tol,
    ComputeEtaJoint compute_eta_joint,
    ScatterJoint scatter_joint,
    CenterEffects center_effects_fn,
    ComputeLogPriorJoint compute_log_prior_joint,
    JointLogLik log_lik_fn,
    NewtonScratchJoint& scratch,
    const std::vector<double>& x_init,
    SparseCholeskySolver* shared_solver,
    bool store_Q
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
    bool use_sparse = (n_x >= SPARSE_THRESHOLD);

    SparseCholeskySolver local_solver;
    SparseCholeskySolver& sparse_solver = shared_solver ? *shared_solver : local_solver;

    auto eval_objective = [&](const Rcpp::NumericVector& xv) -> double {
        return eval_penalized_log_lik_joint_ll(
            xv, compute_eta_joint, compute_log_prior_joint, log_lik_fn,
            scratch.etas_tmp
        );
    };

    auto cholesky_solve = [&](DenseMat& H, DenseVec& grad,
                              std::vector<double>& delta) -> bool {
        return dispatch_factor_solve(H, grad, delta, n_x, sparse_solver,
                                     use_sparse, scratch.chol);
    };

    double obj_current = -1e300;
    bool obj_valid = false;
    NewtonConvState conv_state;

    for (int iter = 0; iter < max_iter; iter++) {
        compute_eta_joint(x, scratch.etas);
        scratch.zero_for_iter();
        scatter_joint(x, scratch.etas, scratch.grad, scratch.H, /*finalize=*/false);

        bool solve_ok = cholesky_solve(scratch.H, scratch.grad, scratch.delta);

        if (!solve_ok) {
            for (int j = 0; j < n_x; j++) {
                if (std::isfinite(scratch.delta[j])) x[j] += 0.1 * scratch.delta[j];
            }
            obj_valid = false;
            result.n_iter = iter + 1;
            continue;
        }

        if (!obj_valid) {
            obj_current = eval_objective(x);
            obj_valid = true;
        }

        double slope = newton_decrement(scratch.grad, scratch.delta, n_x);
        double step_scale = line_search_backtrack(
            x, scratch.delta, n_x, obj_current, slope, eval_objective,
            obj_current, scratch.x_try
        );

        result.n_iter = iter + 1;
        if (newton_converged(scratch.delta, scratch.grad, step_scale, n_x, tol,
                             conv_state)) {
            result.converged = true;
            break;
        }
    }

    // Compute log_marginal at the Newton mode (uncentered). For BYM2/ICAR the
    // ICAR prior is rank-deficient along the constant-shift direction in phi,
    // and the obs Hessian is what pins that direction down, so the joint MAP
    // is at the uncentered Newton iterate. Centering phi without compensating
    // each arm's intercept would shift eta off the mode and corrupt log_lik
    // (and would also shift the proper-CAR log_prior). We center post-hoc
    // *after* log_marginal so the reported mode block has mean(phi) = 0 with
    // the equivalent intercept shift absorbed into each arm's first beta
    // column. Net effect: same eta, same log_marginal, centered phi block.
    compute_eta_joint(x, scratch.etas);
    scratch.zero_for_iter();
    scatter_joint(x, scratch.etas, scratch.grad, scratch.H, /*finalize=*/true);

    dispatch_factor_log_det(scratch.H, n_x, sparse_solver, use_sparse,
                             scratch.chol, result.log_det_Q);

    double log_lik   = log_lik_fn(scratch.etas);
    double log_prior = compute_log_prior_joint(x, scratch.etas);

    result.log_marginal = finalize_log_marginal(log_lik, log_prior, result.log_det_Q, n_x);

    center_effects_fn(x);
    for (int j = 0; j < n_x; j++) result.mode[j] = x[j];

    if (store_Q) {
        dense_to_csc_lower_drop_raw(
            scratch.H, n_x, SPARSE_DROP_TOL_DISPATCH,
            result.Q_csc_p, result.Q_csc_i, result.Q_csc_x
        );
        result.Q_csc_n = n_x;
    }

    return result;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_JOINT_H
