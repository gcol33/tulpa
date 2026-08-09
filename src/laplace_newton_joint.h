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

#include "joint_pd_step.h"                // JointPDMode, pd_lm_escalate, pd_eigen_clamp_solve
#include "laplace_builtin_family_spec.h"  // builtin_family_spec, BuiltinFamilyResponse
#include "laplace_cholesky.h"
#include "laplace_cholesky_dispatch.h"
#include "laplace_family_link.h"
#include "laplace_newton.h"          // SPARSE_THRESHOLD
#include "laplace_newton_loop.h"
#include "laplace_spec_curvature3.h" // build_spec_curvature3_oracle (inner-skew diagnostic)
#include "inner_cila.h"              // run_inner_cila
#include "inner_laplace_is.h"        // compute_inner_is_curve
#include "inner_laplace_skew.h"      // compute_inner_skew_gamma3_joint
#include "joint_inner_vcov.h"        // JointFixedBlockRequest, extract_joint_fixed_block
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
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
    // Grouped beta sufficient statistics. Optional; when
    // present (and family == "beta") row i collapses n_trials[i] exchangeable
    // beta obs sharing this row's linear predictor, with slog_y[i] = sum log(y)
    // and slog_1my[i] = sum log(1-y). Empty => ungrouped per-obs path.
    Rcpp::NumericVector slog_y;
    Rcpp::NumericVector slog_1my;
    // Interval-censored Gaussian bounds (tulpaObs ordinal cover). Optional;
    // when present (and family == "interval_gaussian") row i records the latent
    // value fell in (lower[i], upper[i]] on the predictor scale, with +/-Inf the
    // open outer classes. Empty => not an interval arm.
    Rcpp::NumericVector lower;
    Rcpp::NumericVector upper;
    // Upper-truncated Gaussian ceiling. Optional; when present
    // (and family == "truncated_gaussian") row i's latent log-response is truncated
    // to <= trunc_upper[i] on the predictor scale (+Inf => no truncation). The point
    // response y[i] is still read. Empty => not a truncated arm.
    Rcpp::NumericVector trunc_upper;
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
    // Per-arm cell coupling (Change 2b).
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
// single spec-driven likelihood boundary.
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
            r.lower    = (a.lower.size()    > 0) ? REAL(a.lower)    : nullptr;
            r.upper    = (a.upper.size()    > 0) ? REAL(a.upper)    : nullptr;
            r.trunc_upper = (a.trunc_upper.size() > 0) ? REAL(a.trunc_upper) : nullptr;
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

// Per-arm third-log-lik-derivative oracles for the inner-Laplace skewness
// diagnostic (inner_laplace_skew.h's compute_inner_skew_gamma3_joint), one
// entry per arm in `views`. Mirrors compute_total_log_lik_joint's own
// separable-sum contract exactly: a coupled arm (`skip_arm[k] == true`) has
// its per-obs sum excluded from the log-lik there via `skip_arm`, so its
// per-observation oracle is excluded here the same way -- an empty entry, not
// whatever build_spec_curvature3_oracle would return for its (unused) per-obs
// spec. Every other arm gets build_spec_curvature3_oracle(*view.spec, ...).
//
// `coupled_scored` says the caller will supply the cell tensor contraction
// (cell_curvature3.h) covering those arms, which is what scores them instead
// (gcol33/tulpa#301). A coupled arm is then not declined at all -- it is
// carried by the other term -- so it neither appears in `arms_declined` nor
// contributes a fit-level reason. At `coupled_scored = false` it reads
// "coupled_arm" as before.
//
// The REASON each arm declined travels with the oracles (gcol33/tulpa#296).
// `declined` is the fit-level reason when no arm has an oracle at all -- the
// single reason when the arms agree on it, or the distinct reasons joined when
// they do not.
inline JointCurvature3Oracles build_joint_curvature3_fns(
    const std::vector<ArmSpecView>& views,
    const std::vector<bool>* skip_arm,
    bool coupled_scored = false
) {
    JointCurvature3Oracles out;
    out.arms.resize(views.size());
    std::vector<std::string> why;
    for (std::size_t k = 0; k < views.size(); k++) {
        if (skip_arm && k < skip_arm->size() && (*skip_arm)[k]) {
            if (coupled_scored) continue;
            out.arms_declined.push_back(static_cast<int>(k));
            why.push_back("coupled_arm");
            continue;
        }
        const ArmSpecView& v = views[k];
        out.arms[k] = build_spec_curvature3_oracle(*v.spec, v.response_data,
                                                   *v.data, *v.layout, *v.params);
        if (!out.arms[k].any()) {
            out.arms_declined.push_back(static_cast<int>(k));
            why.push_back(out.arms[k].declined.empty()
                              ? std::string("curvature3_unavailable")
                              : out.arms[k].declined);
        }
    }
    // `cell_cubic` is assigned by the caller after this returns (it needs the
    // per-solve dispersion), so this emptiness test runs over the per-arm
    // oracles alone, which is the right scope for the per-arm reasons.
    bool any_arm = false;
    for (const auto& o : out.arms) if (o.any()) { any_arm = true; break; }
    if (!any_arm && !coupled_scored) {
        std::vector<std::string> distinct;
        for (const std::string& w : why) {
            if (std::find(distinct.begin(), distinct.end(), w) == distinct.end()) {
                distinct.push_back(w);
            }
        }
        for (std::size_t i = 0; i < distinct.size(); i++) {
            if (i) out.declined += ", ";
            out.declined += distinct[i];
        }
    }
    return out;
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

// Factor + solve for one inner Newton step of the DENSE joint loop, enforcing
// positive-definiteness. Same two policies as the sparse loop's
// joint_pd_step_solve (joint_pd_step.h), over the dense H container and the
// dense/CHOLMOD dispatch instead of the SparseHessianBuilder.
//
// `H` arrives UNRIDGED, exactly as the scatter left it: the base
// LAPLACE_UNIFORM_RIDGE is applied here, once, and any LM escalation loads the
// diagonal on top of it. `out_modified`, when non-null, records whether the
// matrix that was finally factorized is the one handed in -- false for every
// solve whose first attempt succeeded, which is every solve at a PD Hessian.
inline bool joint_pd_step_solve_dense(
    DenseMat& H, DenseVec& grad, std::vector<double>& delta, int n_x,
    SparseCholeskySolver& solver, bool prefer_sparse,
    DenseCholeskyScratch& dense_scratch, JointPDMode pd_mode,
    double* out_log_det = nullptr,
    bool* out_modified = nullptr
) {
    add_uniform_ridge_dense(H, n_x, LAPLACE_UNIFORM_RIDGE);

    if (pd_mode == JointPDMode::PSD && n_x <= JOINT_PSD_MAX_DIM) {
        Eigen::MatrixXd Hd(n_x, n_x);
        for (int j = 0; j < n_x; ++j)
            for (int i = 0; i < n_x; ++i) Hd(i, j) = H[i][j];
        return pd_eigen_clamp_solve(Hd, n_x, grad.data(), delta.data(),
                                    out_log_det, out_modified);
    }

    return pd_lm_escalate(
        [&](double* log_det) -> bool {
            return dispatch_factor_solve_ridged(H, grad, delta, n_x, solver,
                                                prefer_sparse, dense_scratch,
                                                log_det);
        },
        [&](double bump) { add_uniform_ridge_dense(H, n_x, bump); },
        out_log_det, out_modified);
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

    // CHOLMOD context for the per-cell fixed-effect block extraction
    // (gcol33/tulpa#307). Separate from the loop's own solver: the extraction
    // resets and re-analyzes, which would throw away the pattern cache the
    // Newton iterations rely on. Built here, single-threaded, only when the
    // driver asked for the block -- one CHOLMOD common per outer thread, never
    // one per grid cell.
    std::unique_ptr<SparseCholeskySolver> extract_solver;

    void allocate(int n_x, const std::vector<JointArm>& arms,
                  bool want_fixed_block = false) {
        if (want_fixed_block && !extract_solver) {
            extract_solver.reset(new SparseCholeskySolver());
        }
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
    bool store_Q,
    // PD enforcement for the inner step (joint_pd_step.h). A coupled arm's
    // observed Hessian is indefinite wherever the mixture term is not concave,
    // which for the occupancy mixture is any sparse-detection data set at the
    // x = 0 start; without a conditioner the Cholesky yields a non-finite step,
    // nothing is accepted, and the loop reports the start vector as its mode
    // (gcol33/tulpa#344). LM (the default) reduces to the plain Newton step
    // wherever H is already PD.
    JointPDMode pd_mode = JointPDMode::LM,
    // Inner-Laplace skewness diagnostic (inner_laplace_skew.h), opt-in like
    // store_Q. curvature3_fns carries one per-observation oracle per arm plus the
    // optional coupled-cell tensor contraction (build_joint_curvature3_fns +
    // build_cell_curvature3_tensor); nullptr declines entirely. skew_probe_idx
    // == nullptr probes every latent
    // index. Computed BEFORE center_effects_fn(x) below -- unlike the single-arm
    // loop, this joint loop centers x POST-HOC, after log_marginal, purely to
    // present mean(phi) = 0 in the reported mode (see the comment on that call);
    // the Newton-converged x and its freshly-factored scratch.chol / scratch.H
    // (what dispatch_factor_log_det and log_lik_fn just used) are the consistent
    // point to probe, not the cosmetically-shifted one.
    bool compute_skew = false,
    const std::vector<int>* skew_probe_idx = nullptr,
    const JointCurvature3Oracles* curvature3_fns = nullptr,
    // Per-cell fixed-effect covariance block (gcol33/tulpa#307). Extracted from
    // this cell's precision here, so the caller no longer has to keep every
    // cell's precision alive to read it afterwards. Independent of store_Q: a
    // fit that did not ask for the precision still gets the block, and one that
    // did gets both. nullptr / p == 0 extracts nothing.
    const JointFixedBlockRequest* fixed_block = nullptr,
    // Subspace debias (subspace_debias.h, gcol33/tulpa#304/#306). Runs on the
    // same pre-centering iterate and the same live factor the two inner
    // diagnostics probe, since that is the point log_marginal belongs to. Empty
    // or absent leaves the solve untouched and consumes no random number.
    const SubspaceDebiasOptions* debias = nullptr,
    // Corrected integrated Laplace (inner_cila.h, gcol33/tulpa#351). Runs on
    // the same pre-centering iterate and the same live factor, and presents its
    // draws under the same centering fold the reported mode carries. Absent or
    // inactive leaves the solve untouched.
    const CilaOptions* cila = nullptr,
    // Distinguishes this cell's auxiliary stream from its neighbours' on an
    // outer grid; irrelevant for the deterministic net, load-bearing for the
    // randomized-QMC shifts.
    std::uint64_t cila_cell_key = 0
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
        return joint_pd_step_solve_dense(H, grad, delta, n_x, sparse_solver,
                                         use_sparse, scratch.chol, pd_mode);
    };

    double obj_current = -1e300;
    bool obj_valid = false;
    NewtonConvState conv_state;

    auto refresh_grad_hess = [&]() {
        compute_eta_joint(x, scratch.etas);
        scratch.zero_for_iter();
        scatter_joint(x, scratch.etas, scratch.grad, scratch.H, /*finalize=*/false);
    };

    for (int iter = 0; iter < max_iter; iter++) {
        if (newton_step(x, scratch, n_x, iter, tol, refresh_grad_hess,
                        cholesky_solve, eval_objective, obj_current, obj_valid,
                        conv_state, result.n_iter)) {
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
    result.score_max = max_abs(scratch.grad);

    dispatch_factor_log_det(scratch.H, n_x, sparse_solver, use_sparse,
                             scratch.chol, result.log_det_Q);

    // A non-finite log-determinant is the plain Cholesky reporting that the
    // Hessian at the returned point is not PD -- a point the solve stopped at
    // without reaching a mode. Condition it so the cell still carries a defined
    // log-marginal (an undefined one silently corrupts the outer-grid weights),
    // and record that the stored precision below is no longer the matrix the
    // scatter built. A PD Hessian never reaches this, so every fit that
    // factorizes on the first attempt is unchanged.
    const bool hessian_pd_at_mode = std::isfinite(result.log_det_Q);
    if (!hessian_pd_at_mode) {
        joint_pd_step_solve_dense(scratch.H, scratch.grad, scratch.delta, n_x,
                                  sparse_solver, use_sparse, scratch.chol,
                                  pd_mode, &result.log_det_Q);
    }

    double log_lik   = log_lik_fn(scratch.etas);
    double log_prior = compute_log_prior_joint(x, scratch.etas);

    result.log_marginal = finalize_log_marginal(log_lik, log_prior, result.log_det_Q, n_x);

    // The Newton-converged iterate, before the cosmetic post-hoc centering
    // below. Every probe of the inner layer -- gamma_3, the importance curve,
    // and the subspace sampler -- reads this point, because it is the one the
    // live factor and the reported log_marginal belong to.
    std::vector<double> pre_center_x(n_x);
    for (int j = 0; j < n_x; j++) pre_center_x[j] = x[j];
    const bool used_sparse_factor = use_sparse && sparse_solver.factored();

    if (compute_skew && result.converged) {
        std::vector<int> all_idx;
        const std::vector<int>* probe = skew_probe_idx;
        if (!probe) {
            all_idx.resize(n_x);
            for (int j = 0; j < n_x; j++) all_idx[j] = j;
            probe = &all_idx;
        }
        if (curvature3_fns) {
            InnerSkewOutcome sk = compute_inner_skew_gamma3_joint(
                n_x, pre_center_x, scratch.chol, sparse_solver, used_sparse_factor,
                compute_eta_joint, x, scratch.etas, scratch.etas_tmp,
                *curvature3_fns, *probe
            );
            result.inner_skew = std::move(sk.gamma3);
            result.inner_skew_gamma1 = std::move(sk.gamma1);
            result.inner_skew_gamma1_declined = sk.gamma1_declined;
            result.inner_skew_idx = *probe;
            result.inner_skew_dropped = sk.n_nonfinite_dropped;
            result.inner_skew_declined = sk.declined;
            result.inner_skew_arms_declined = sk.arms_declined;
        } else {
            // No oracle set was built at all: report the indices as unscored
            // rather than emit nothing, so the reason reaches the fit.
            result.inner_skew.assign(probe->size(),
                                     std::numeric_limits<double>::quiet_NaN());
            result.inner_skew_idx = *probe;
            result.inner_skew_declined = "curvature3_unavailable";
            result.inner_skew_gamma1_declined = "curvature3_unavailable";
        }

        // The likelihood-agnostic inner k-hat over the same probed subspace
        // (gcol33/tulpa#303). Evaluated at the pre-centering iterate for the
        // same reason gamma_3 is: that is the point the live factor and the
        // reported log_marginal belong to.
        InnerISOutcome is_out = compute_inner_is_curve(
            n_x, pre_center_x, scratch.chol, sparse_solver, used_sparse_factor,
            eval_objective, x, *probe
        );
        result.inner_is_z         = std::move(is_out.z);
        result.inner_is_log_joint = std::move(is_out.log_joint);
        result.inner_is_sigma     = std::move(is_out.sigma);
        result.inner_is_declined  = is_out.declined;
    }

    if (result.converged) {
        run_subspace_debias(result, n_x, pre_center_x, scratch.chol,
                            sparse_solver, used_sparse_factor,
                            eval_objective, x, debias);
    }

    // The correction reads the same pre-centering iterate for the same reason,
    // and presents each draw through the loop's own centering fold so a drawn
    // coefficient is in the coordinates the reported mode is in.
    run_inner_cila(result, n_x, pre_center_x, scratch.chol, sparse_solver,
                   used_sparse_factor, eval_objective,
                   [&](Rcpp::NumericVector& xv) { center_effects_fn(xv); },
                   x, cila, cila_cell_key);

    center_effects_fn(x);
    for (int j = 0; j < n_x; j++) result.mode[j] = x[j];

    // The precision at the mode in CSC. The fixed-effect block is read off it
    // here and the arrays are then released, so requesting the block costs one
    // cell's precision -- not the grid's. Centering does not touch H.
    //
    // Both are withheld where the Hessian at the returned point is not PD: its
    // inverse is not a covariance there, and the conditioned matrix the
    // log-determinant came from is not what the scatter built. The fit's
    // `converged` flag is what says why, and the R retention reads it.
    const bool want_block =
        fixed_block && fixed_block->active() && scratch.extract_solver;
    if ((store_Q || want_block) && hessian_pd_at_mode) {
        std::vector<int> csc_p, csc_i;
        std::vector<double> csc_x;
        dense_to_csc_lower_drop_raw(scratch.H, n_x, SPARSE_DROP_TOL_DISPATCH,
                                    csc_p, csc_i, csc_x);
        if (want_block) {
            extract_joint_fixed_block(
                csc_p.data(), csc_i.data(), csc_x.data(), n_x,
                static_cast<int>(csc_x.size()), *fixed_block,
                *scratch.extract_solver,
                result.re_cov_flat, result.re_cov_block_sizes);
        }
        if (store_Q) {
            result.Q_csc_p = std::move(csc_p);
            result.Q_csc_i = std::move(csc_i);
            result.Q_csc_x = std::move(csc_x);
            result.Q_csc_n = n_x;
        }
    }

    return result;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_JOINT_H
