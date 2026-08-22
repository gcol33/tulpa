// laplace_newton_joint_sparse.h
// Sparse-Hessian PIRLS-equivalent Newton solver for *joint multi-likelihood*
// Laplace modes.
//
// Sibling of laplace_newton_joint.h::laplace_newton_solve_joint_ll: same
// JointLogLik functor, same callback shapes for compute_eta_joint / center /
// log_prior, but the scatter writes into a SparseHessianBuilder rather than a
// DenseMat and the Newton step factorizes the sparse H directly via CHOLMOD.
//
// Why a separate header (not a unified solver with branching):
//   * NewtonScratchJointSparse never allocates n_x x n_x DenseMat H.
//     At n_x ~ 10^6 the dense allocation alone is 8 TB; the entire point
//     of the sparse path is to never touch it.
//   * The factorize/solve dispatch is sparse-only — no dense-to-CSC
//     conversion path, no DenseCholeskyScratch. CHOLMOD handles everything.
//   * The sparsity pattern is built ONCE at fit-time (outside this solver)
//     via build_joint_hessian_pattern; only values change per iteration.
//
// Eta accumulator: lives in the caller (run_multi_block_nested_laplace_joint).
// Same shape as the dense compute_eta_joint but must dispatch on
// LatentBlock::contrib_kind to handle INDEXED_SINGLE / INDEXED_MULTI /
// DENSE_BASIS. The caller supplies the dispatch lambda.
//
// Scatter: caller supplies a lambda that calls
// scatter_arm_obs_joint_multi_sparse (from nested_laplace_joint_multi.h) for
// each arm, plus add_prior_sparse on each block, plus per-arm beta/RE priors.

#ifndef TULPA_LAPLACE_NEWTON_JOINT_SPARSE_H
#define TULPA_LAPLACE_NEWTON_JOINT_SPARSE_H

#include "joint_pd_step.h"            // JointPDMode, pd_lm_escalate, pd_eigen_clamp_solve
#include "laplace_family_link.h"
#include "laplace_newton_joint.h"     // JointArm, compute_total_log_lik_joint
#include "laplace_newton_loop.h"
#include "laplace_profile.h"
#include "sparse_cholesky.h"
#include "sparse_hessian.h"
#include <RcppEigen.h>                 // PSD eigen-clamp inner step (small n_x)
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// Per-thread scratch for the sparse joint Newton solver.
// IMPORTANT: no DenseMat H — that's the whole point. The Hessian lives
// inside the caller-supplied SparseHessianBuilder.
struct NewtonScratchJointSparse {
    Rcpp::NumericVector x;       // size n_x
    Rcpp::NumericVector x_try;   // size n_x
    std::vector<Rcpp::NumericVector> etas;      // per-arm, length N_k
    std::vector<Rcpp::NumericVector> etas_tmp;  // line-search buffer
    DenseVec  grad;              // size n_x, zeroed per iter
    std::vector<double> delta;   // size n_x, written by sparse solve

    // Pattern-invariant cache for the final-pass s2z log-determinant, reused
    // across all outer-grid cells this scratch solves. Built lazily on first
    // use (no s2z field -> never touched). See S2ZLogDetCache.
    S2ZLogDetCache s2z_log_det_cache;

    // Pattern-invariant cache for the block-Schur inner step + log-determinant on
    // the s2z large-field path (A_FF pattern + symbolic factor + scatter slots).
    S2ZBlockSchurCache s2z_block_schur_cache;

    // CHOLMOD context for the per-cell fixed-effect block extraction. See the
    // twin member on NewtonScratchJoint.
    std::unique_ptr<SparseCholeskySolver> extract_solver;

    void allocate(int n_x, const std::vector<JointArm>& arms,
                  bool want_fixed_block = false) {
        if (want_fixed_block && !extract_solver) {
            extract_solver.reset(new SparseCholeskySolver());
        }
        x      = Rcpp::NumericVector(n_x, 0.0);
        x_try  = Rcpp::NumericVector(n_x, 0.0);
        etas.clear();     etas.reserve(arms.size());
        etas_tmp.clear(); etas_tmp.reserve(arms.size());
        for (const JointArm& a : arms) {
            etas.emplace_back(a.N, 0.0);
            etas_tmp.emplace_back(a.N, 0.0);
        }
        grad.assign(n_x, 0.0);
        delta.assign(n_x, 0.0);
    }

    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0);
        std::fill(delta.begin(), delta.end(), 0.0);
    }
};

// Factor + solve for one inner Newton step of the SPARSE joint loop, enforcing
// positive-definiteness. The policies live in joint_pd_step.h and are shared
// with the dense loop's joint_pd_step_solve_dense; this is the CHOLMOD /
// SparseHessianBuilder backend for them.
// `H` must already carry the base LAPLACE_UNIFORM_RIDGE on its diagonal.
// `out_log_det`, if non-null, receives the log-determinant of the PD-enforced
// Hessian (LM: from the CHOLMOD factor; PSD: sum of log-clamped eigenvalues).
// The final Laplace pass uses this so an indefinite "mode" still yields a
// defined log-marginal (a failed factorize previously left log_det = 0, which
// corrupted the outer-grid weights).
inline bool joint_pd_step_solve(
    SparseHessianBuilder& H, SparseCholeskySolver& solver,
    int n_x, JointPDMode pd_mode,
    const double* grad, double* delta,
    double* out_log_det = nullptr,
    bool* out_modified = nullptr
) {
    if (pd_mode == JointPDMode::PSD && n_x <= JOINT_PSD_MAX_DIM) {
        Eigen::MatrixXd Hd = Eigen::MatrixXd::Zero(n_x, n_x);
        for (int j = 0; j < H.n; ++j) {
            for (int p = H.col_ptr[j]; p < H.col_ptr[j + 1]; ++p) {
                const int i = H.row_idx[p];
                const double v = H.values[p];
                Hd(i, j) = v;
                if (i != j) Hd(j, i) = v;   // CSC stores lower triangle only
            }
        }
        return pd_eigen_clamp_solve(Hd, n_x, grad, delta, out_log_det,
                                    out_modified);
    }

    return pd_lm_escalate(
        [&](double* log_det) -> bool {
            cholmod_sparse H_cholmod = H.as_cholmod(&solver.common());
            if (!solver.analyzed()) solver.analyze(&H_cholmod);
            if (!solver.factorize(&H_cholmod)) return false;
            if (!solver.solve(grad, delta, n_x)) return false;
            for (int i = 0; i < n_x; ++i)
                if (!std::isfinite(delta[i])) return false;
            if (log_det) *log_det = solver.log_determinant();
            return true;
        },
        [&](double bump) { H.add_uniform_ridge(bump); },
        out_log_det, out_modified);
}

// Inner Newton step for the sum-to-zero large-field path. Prefers the exact
// block-Schur step: factor the PD field block A_FF, then fold the rank-1 pins
// coef_k 1_k 1_k' and the field<->scalar coupling via a small dense Schur. That
// is the TRUE Newton step (A + sum_k coef_k 1_k 1_k')^-1 grad with no perturbing
// ridge, so the inner solve converges quadratically. Falls back to the LM
// escalating-ridge step + Woodbury when A_FF or the Schur complement is not PD
// (which can happen far from the mode, where the observed Hessian is indefinite).
// With no rank-1 registered (small densified field, or no intrinsic field) it is
// exactly joint_pd_step_solve. Sets `used_block_schur` so the caller does not
// re-apply the Woodbury correction (block-Schur already includes the rank-1
// terms) and knows the CHOLMOD `solver` factor was NOT populated. Returns false
// on a non-finite step.
inline bool s2z_newton_step(
    SparseHessianBuilder& H, SparseCholeskySolver& solver,
    int n_x, JointPDMode pd_mode,
    const double* grad, double* delta,
    bool& used_block_schur,
    S2ZBlockSchurCache* bs_cache = nullptr
) {
    used_block_schur = false;
    if (pd_mode == JointPDMode::LM && !H.s2z_rank1.empty() &&
        s2z_block_schur(H, H.s2z_rank1, grad, delta, nullptr, bs_cache)) {
        used_block_schur = true;
        return true;
    }
    bool ok = joint_pd_step_solve(H, solver, n_x, pd_mode, grad, delta, nullptr);
    // The Woodbury fold applies the rank-1 terms through `solver`, and only the
    // LM branch of joint_pd_step_solve populates that factor: the PSD branch
    // eigen-solves a densified copy and leaves the CHOLMOD factor absent or
    // stale, so folding there would correct the step against a different matrix.
    if (ok && pd_mode == JointPDMode::LM)
        apply_s2z_rank1_correction(solver, n_x, H.s2z_rank1, delta,
                                   H.s2z_coupling);
    return ok;
}

// Sparse-H joint Newton solver. Compositional skeleton mirrors
// laplace_newton_solve_joint_ll exactly; differences are isolated to the H
// container (SparseHessianBuilder vs DenseMat) and the factor/solve path
// (CHOLMOD only, no dense fallback).
//
// scatter_joint_sparse signature:
//   (const NumericVector& x, const vector<NumericVector>& etas,
//    DenseVec& grad, SparseHessianBuilder& H) -> void
//
// The H_builder must already be init()ed with the joint pattern before
// this function is called.
template<typename ComputeEtaJoint, typename ScatterJointSparse,
         typename CenterEffects, typename ComputeLogPriorJoint, typename JointLogLik>
LaplaceResult laplace_newton_solve_joint_sparse_ll(
    int n_x,
    int max_iter, double tol,
    ComputeEtaJoint compute_eta_joint,
    ScatterJointSparse scatter_joint_sparse,
    CenterEffects center_effects_fn,
    ComputeLogPriorJoint compute_log_prior_joint,
    JointLogLik log_lik_fn,
    SparseHessianBuilder& H_builder,
    NewtonScratchJointSparse& scratch,
    const std::vector<double>& x_init,
    SparseCholeskySolver* shared_solver,
    bool store_Q,
    JointPDMode pd_mode = JointPDMode::LM,
    int hessian_refresh = 1,
    // Inner-Laplace skewness diagnostic (inner_laplace_skew.h), opt-in like
    // store_Q. Only computable on the plain-CHOLMOD final factor: the s2z
    // (sum-to-zero large-field) path solves against H + a rank-1 correction
    // the stored CHOLMOD factor of H_builder alone does not reflect (see
    // s2z_newton_step / apply_s2z_rank1_correction above), and the PSD path
    // never populates `solver` at all (its final factorize goes through the
    // dense eigen-clamp branch of joint_pd_step_solve). Probing either with a
    // plain solver.solve() would silently solve against the wrong matrix, so
    // both decline BY NAME -- the probed indices come back NaN carrying the
    // reason -- rather than report a wrong gamma_3 or emit nothing at all.
    bool compute_skew = false,
    const std::vector<int>* skew_probe_idx = nullptr,
    const JointCurvature3Oracles* curvature3_fns = nullptr,
    // Per-cell fixed-effect covariance block. Read off the
    // builder's own CSC -- the working Hessian is already in that layout, so on
    // this path the block costs no precision copy at all. The extraction
    // factorizes that CSC itself rather than reusing the loop's live factor,
    // which after joint_pd_step_solve may be of a ridge-escalated matrix (the
    // s2z path) or absent (the PSD path densifies instead). Reading the same
    // bytes store_Q would hand out is what makes the block available on every
    // path and identical to what cpp_joint_inner_vcov_blocks() returns.
    const JointFixedBlockRequest* fixed_block = nullptr,
    // Subspace debias (subspace_debias.h). Built from the
    // same live factor as the diagnostics above and so declined on exactly the
    // same two paths, for the same reason: the s2z rank-1 correction and the PSD
    // eigen-clamp both leave `solver` holding a factor of a different matrix.
    const SubspaceDebiasOptions* debias = nullptr,
    // Corrected integrated Laplace (inner_cila.h). A draw
    // from the inner Gaussian is P' L^-T applied to a standard normal vector,
    // which the live CHOLMOD factor supplies directly; an
    // LDL' factor has no such square root and is declined by name.
    const CilaOptions* cila = nullptr,
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

    SparseCholeskySolver local_solver;
    SparseCholeskySolver& solver = shared_solver ? *shared_solver : local_solver;

    auto eval_objective = [&](const Rcpp::NumericVector& xv) -> double {
        return eval_penalized_log_lik_joint_ll(
            xv, compute_eta_joint, compute_log_prior_joint, log_lik_fn,
            scratch.etas_tmp
        );
    };

    double obj_current = -1e300;
    bool obj_valid = false;

    // Shamanskii-style factor reuse (the chord method). For a non-quadratic
    // arm (e.g. the beta cover positive arm) the latent Hessian changes every
    // inner iteration, so a plain Newton loop re-factorizes the sparse
    // Cholesky on each step -- the dominant per-grid-cell cost. With
    // `hessian_refresh = m > 1` the factor is recomputed only every m-th
    // iteration (and on the first iteration, and whenever a reused solve
    // returns a non-finite step); the intervening iterations reuse the cached
    // CHOLMOD factor and re-solve with the refreshed gradient. The gradient is
    // exact on every iteration and each step is line-search safeguarded, so the
    // converged mode is unchanged; only the path to it uses a stale curvature.
    // The final mode-pass below always re-factorizes, so log_det_Q and the SEs
    // use the true Hessian at the mode regardless of `hessian_refresh`.
    //
    // Reuse is only valid in LM mode: the PSD path eigen-solves a densified
    // Hessian and never populates the CHOLMOD factor, so there is nothing to
    // reuse. In PSD mode the refresh interval collapses to 1.
    const int refresh = (hessian_refresh > 1) ? hessian_refresh : 1;
    const bool reuse_enabled = (refresh > 1) && (pd_mode == JointPDMode::LM);
    bool have_factor = false;
    NewtonConvState conv_state;

    for (int iter = 0; iter < max_iter; iter++) {
        { TULPA_PROFILE_PHASE(PHASE_ETA);
          compute_eta_joint(x, scratch.etas); }

        // Decide up front whether this iteration re-factorizes. A reuse
        // iteration applies the cached factor to a fresh gradient, so the
        // Hessian it would build is discarded -- the scatter is told to skip
        // the (expensive) likelihood curvature and emit the gradient only.
        const bool do_factor =
            !reuse_enabled || !have_factor || (iter % refresh == 0) ||
            !H_builder.s2z_rank1.empty();   // s2z large-field path: block-Schur every iter, no reuse

        scratch.zero_grad();
        H_builder.zero();
        { TULPA_PROFILE_PHASE(PHASE_SCATTER);
          scatter_joint_sparse(x, scratch.etas, scratch.grad, H_builder,
                               /*finalize=*/false, /*grad_only=*/!do_factor); }

        // Uniform upstream base ridge for numerical hygiene of an already-PD H.
        H_builder.add_uniform_ridge(LAPLACE_UNIFORM_RIDGE);

        // PD-enforced factor + solve. LM escalates the ridge until CHOLMOD
        // factorizes; PSD eigen-clamps the (small) dense Hessian. Either yields
        // a usable ascent step where a plain factorize of the indefinite
        // mixture Hessian would fail. On reuse iterations the cached factor is
        // re-applied to the refreshed gradient instead (see `reuse_enabled`).
        bool solve_ok;
        { TULPA_PROFILE_PHASE(PHASE_FACTORIZE);
          if (do_factor) {
              // Exact block-Schur step on the s2z large-field path (true Newton,
              // quadratic), else LM ridge + Woodbury. s2z_newton_step folds the
              // rank-1 correction itself. Block-Schur does NOT populate the
              // CHOLMOD `solver` factor; reuse is forced off on the s2z path
              // (see do_factor), so have_factor stays false there.
              bool used_block_schur = false;
              solve_ok = s2z_newton_step(H_builder, solver, n_x, pd_mode,
                                         scratch.grad.data(), scratch.delta.data(),
                                         used_block_schur, &scratch.s2z_block_schur_cache);
              have_factor = solve_ok && !used_block_schur;
          } else {
              solve_ok = solver.solve(scratch.grad.data(),
                                      scratch.delta.data(), n_x);
              for (int j = 0; solve_ok && j < n_x; j++)
                  if (!std::isfinite(scratch.delta[j])) { solve_ok = false; break; }
              // A reuse step builds only the gradient, so H_builder lacks the
              // likelihood curvature and must NOT be factorized here. On the
              // rare failure (non-finite step from a non-finite gradient), fall
              // through to the gradient-ascent guard and force a full
              // re-factorization on the next iteration. Reuse runs only with no
              // rank-1 registered, so the Woodbury fold below is a no-op there.
              if (!solve_ok) have_factor = false;
              else if (pd_mode == JointPDMode::LM)
                  apply_s2z_rank1_correction(solver, n_x, H_builder.s2z_rank1,
                                             scratch.delta.data(),
                                             H_builder.s2z_coupling);
          }
        }

        if (!solve_ok) {
            newton_damped_fallback(x, scratch.delta, n_x, obj_valid);
            result.n_iter = iter + 1;
            continue;
        }

        bool converged_now;
        { TULPA_PROFILE_PHASE(PHASE_LINE_SEARCH);
          converged_now = newton_step_tail(x, scratch, n_x, iter, tol,
                                           eval_objective, obj_current,
                                           obj_valid, conv_state,
                                           result.n_iter); }
        if (converged_now) {
            result.converged = true;
            break;
        }
    }

    // Final pass at the Newton mode (uncentered): rebuild grad + H for
    // log_det evaluation, then center post-hoc. See the log-marginal-at-the-
    // uncentered-mode note in laplace_newton_joint.h for the BYM2/ICAR
    // rank-deficiency rationale.
    { TULPA_PROFILE_PHASE(PHASE_ETA);
      compute_eta_joint(x, scratch.etas); }
    scratch.zero_grad();
    H_builder.zero();
    { TULPA_PROFILE_PHASE(PHASE_SCATTER);
      scatter_joint_sparse(x, scratch.etas, scratch.grad, H_builder,
                           /*finalize=*/true, /*grad_only=*/false); }
    // Read before joint_pd_step_solve below, which consumes scratch.grad.
    result.score_max = max_abs(scratch.grad);
    H_builder.add_uniform_ridge(LAPLACE_UNIFORM_RIDGE);

    // PD-enforced final factorize so log_det is defined even when the mode sits
    // at an indefinite point (the delta is discarded here; we only need the
    // conditioned log-determinant for the log-marginal).
    //
    // Sum-to-zero rank-1 fields take the direct route: log|H + sum_k coef_k 1_k
    // 1_k'| is read from a factor of that well-conditioned matrix (the constant
    // direction is pinned by the rank-1 block, so it is PD on the base ridge and
    // never triggers LM ridge escalation). This must be computed from the
    // freshly-scattered base-ridge H, BEFORE joint_pd_step_solve mutates the
    // diagonal: with the rank-1 left off the stored H, the constant direction of
    // H is unpinned and joint_pd_step_solve escalates the ridge to factor it,
    // which would inflate the determinant. Factoring H + 1 1' directly matches
    // the dense full-1 1' path. LM only (the PSD path densifies the small
    // Hessian and registers no rank-1).
    const bool s2z_direct =
        (pd_mode == JointPDMode::LM) && !H_builder.s2z_rank1.empty();
    const double S2Z_NA = std::numeric_limits<double>::quiet_NaN();
    double s2z_log_det = S2Z_NA;
    if (s2z_direct) {
        TULPA_PROFILE_PHASE(PHASE_LOG_DET);
        s2z_log_det = s2z_log_det_block_schur(H_builder, H_builder.s2z_rank1,
                                              /*fallback=*/S2Z_NA,
                                              &scratch.s2z_block_schur_cache);
        if (!std::isfinite(s2z_log_det))
            s2z_log_det = s2z_log_det_direct(H_builder, H_builder.s2z_rank1,
                                             /*fallback=*/S2Z_NA,
                                             &scratch.s2z_log_det_cache);
        // Neither reader could form a factor of the pinned matrix, so the
        // PD-enforced value below stands in for it.
        result.s2z_log_det_fallback = !std::isfinite(s2z_log_det);
    }
    // The values as the scatter left them, plus the base ridge. joint_pd_step_solve
    // below loads the diagonal further on every failed factorization and never
    // takes the load back off, so the builder afterwards holds H + lambda I --
    // not the precision at the mode. A covariance read off that is smaller than
    // the one read off H, so the block and the exported precision are both taken
    // from this snapshot instead. On the sum-to-zero path the escalation is the
    // normal case rather than the exception, which is why the snapshot is taken
    // unconditionally rather than the export being gated on a clean factor.
    const bool want_block =
        fixed_block && fixed_block->active() && scratch.extract_solver;
    std::vector<double> H_values_at_mode;
    if (store_Q || want_block) H_values_at_mode = H_builder.values;

    bool pd_conditioned = false;
    { TULPA_PROFILE_PHASE(PHASE_FACTORIZE);
      joint_pd_step_solve(H_builder, solver, n_x, pd_mode,
                          scratch.grad.data(), scratch.delta.data(),
                          &result.log_det_Q, &pd_conditioned); }
    // Prefer the cancellation-free direct factor; keep the PD-enforced value only
    // if the direct factor was non-PD (NaN fallback).
    if (s2z_direct && std::isfinite(s2z_log_det)) result.log_det_Q = s2z_log_det;

    // Whether the Hessian at the returned point is the PD matrix the expansion
    // needs. On the sum-to-zero path the escalation `pd_conditioned` reports is
    // an artefact of the rank-1 pins being left off the STORED H, so the reading
    // there is the direct factor of the pinned matrix, which is the true one.
    result.hessian_pd_at_mode =
        s2z_direct ? std::isfinite(s2z_log_det) : !pd_conditioned;

    double log_lik, log_prior;
    { TULPA_PROFILE_PHASE(PHASE_LOG_LIK_PRIOR);
      log_lik   = log_lik_fn(scratch.etas);
      log_prior = compute_log_prior_joint(x, scratch.etas); }

    result.log_marginal = finalize_log_marginal(log_lik, log_prior,
                                                  result.log_det_Q, n_x);

    // The three inner-layer probes read the live CHOLMOD factor directly, so
    // each condition below is a way for that factor to hold a different matrix
    // than the one the solve stepped with. They are separated rather than
    // collapsed into one flag so a declined fit says WHICH: a probe that never
    // ran because the solve stalled sends a reader to a different question than
    // one blocked by the PD mode this fit was configured with.
    const char* factor_declined = nullptr;
    if (!result.converged)                    factor_declined = "not_converged";
    else if (pd_mode != JointPDMode::LM)      factor_declined = "pd_eigen_clamp";
    else if (!H_builder.s2z_rank1.empty())    factor_declined = "s2z_rank1_factor";
    else if (!solver.factored())              factor_declined = "factor_unavailable";
    const bool skew_factor_valid = (factor_declined == nullptr);

    std::vector<double> pre_center_x(n_x);
    for (int j = 0; j < n_x; j++) pre_center_x[j] = x[j];
    DenseCholeskyScratch unused_dense_chol;  // sparse-only path never reads it
    if (compute_skew) {
        std::vector<int> all_idx;
        const std::vector<int>& probe =
            inner_probe_indices(n_x, skew_probe_idx, all_idx);
        if (!skew_factor_valid) {
            inner_probe_decline(result, probe, factor_declined);
        } else if (curvature3_fns) {
            InnerSkewOutcome sk = compute_inner_skew_gamma3_joint(
                n_x, pre_center_x, unused_dense_chol, solver, /*use_sparse=*/true,
                compute_eta_joint, x, scratch.etas, scratch.etas_tmp,
                *curvature3_fns, probe
            );
            result.inner_skew = std::move(sk.gamma3);
            result.inner_skew_gamma1 = std::move(sk.gamma1);
            result.inner_skew_gamma1_declined = sk.gamma1_declined;
            result.inner_skew_idx = probe;
            result.inner_skew_dropped = sk.n_nonfinite_dropped;
            result.inner_skew_declined = sk.declined;
            result.inner_skew_arms_declined = sk.arms_declined;
        } else {
            result.inner_skew.assign(probe.size(),
                                     std::numeric_limits<double>::quiet_NaN());
            result.inner_skew_idx = probe;
            result.inner_skew_declined = "curvature3_unavailable";
            result.inner_skew_gamma1_declined = "curvature3_unavailable";
        }

        // The likelihood-agnostic inner k-hat over the same probed subspace.
        if (skew_factor_valid) {
            InnerISOutcome is_out = compute_inner_is_curve(
                n_x, pre_center_x, unused_dense_chol, solver, /*use_sparse=*/true,
                eval_objective, x, probe
            );
            result.inner_is_z         = std::move(is_out.z);
            result.inner_is_log_joint = std::move(is_out.log_joint);
            result.inner_is_sigma     = std::move(is_out.sigma);
            result.inner_is_declined  = is_out.declined;
        }
    }

    run_subspace_debias(result, n_x, pre_center_x, unused_dense_chol,
                        solver, /*use_sparse=*/true,
                        eval_objective, x, debias,
                        result.converged ? factor_declined : nullptr);

    run_inner_cila(result, n_x, pre_center_x, unused_dense_chol, solver,
                   /*use_sparse=*/true, eval_objective,
                   [&](Rcpp::NumericVector& xv) { center_effects_fn(xv); },
                   x, cila, cila_cell_key);

    { TULPA_PROFILE_PHASE(PHASE_LOG_LIK_PRIOR);
      center_effects_fn(x); }
    for (int j = 0; j < n_x; j++) result.mode[j] = x[j];

    // Both read the snapshot taken before the PD-enforced factorize, so the
    // precision the caller gets is the one the scatter built and the block is a
    // covariance of that rather than of a ridge-inflated matrix. The pattern is
    // fit-level and untouched by the escalation, so only the values are held.
    if (want_block) {
        extract_joint_fixed_block(
            H_builder.col_ptr.data(), H_builder.row_idx.data(),
            H_values_at_mode.data(), n_x,
            static_cast<int>(H_values_at_mode.size()), *fixed_block,
            *scratch.extract_solver,
            result.re_cov_flat, result.re_cov_block_sizes);
    }

    if (store_Q) {
        // Copy the CSC arrays out so the caller doesn't depend on H_builder
        // staying alive.
        result.Q_csc_p = H_builder.col_ptr;
        result.Q_csc_i = H_builder.row_idx;
        result.Q_csc_x = std::move(H_values_at_mode);
        result.Q_csc_n = n_x;
    }

    return result;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_JOINT_SPARSE_H
