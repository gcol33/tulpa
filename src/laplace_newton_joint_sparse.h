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

#include "hard_constraints.h"         // HardConstraintState
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

    // Pattern-invariant cache for the block-Schur inner step + log-determinant on
    // the s2z large-field path (A_FF pattern + symbolic factor + scatter slots).
    S2ZBlockSchurCache s2z_block_schur_cache;

    // CHOLMOD context for the per-cell fixed-effect block extraction. See the
    // twin member on NewtonScratchJoint.
    std::unique_ptr<SparseCholeskySolver> extract_solver;

    void allocate(int n_x, const std::vector<JointArm>& arms,
                  bool want_extract = false) {
        if (want_extract && !extract_solver) {
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
    bool* out_modified = nullptr,
    int max_tries = JOINT_LM_MAX_TRIES
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
        out_log_det, out_modified, max_tries);
}

// Inner Newton step for the sum-to-zero large-field path. Prefers the exact
// block-Schur step: factor the field block A_FF, then fold the rank-1 pins
// coef_k 1_k 1_k' and the field<->scalar coupling via a small dense Schur. That
// is the TRUE Newton step (A + sum_k coef_k 1_k 1_k')^-1 grad with no perturbing
// ridge, so the inner solve converges quadratically. A_FF itself need not be PD:
// the inertia route accepts it wherever the pinned matrix is, which is the
// matrix the step solves against. Falls back to the LM escalating-ridge step +
// Woodbury when the pinned matrix is not PD (which can happen far from the
// mode, where the observed Hessian is indefinite).
// With no rank-1 registered (small densified field, or no intrinsic field) it is
// exactly joint_pd_step_solve. Sets `used_block_schur` so the caller does not
// re-apply the Woodbury correction (block-Schur already includes the rank-1
// terms) and knows the CHOLMOD `solver` factor was NOT populated. Returns false
// on a non-finite step.
//
// `guarded = false` asks whether H gives a Newton step as it stands
// (inner_step_under_curvature): with rank-1 pins that is the block-Schur
// factor alone, since the fallback factors the UNPINNED H, whose constant
// direction only the ridge ladder carries even at a PD point; without pins it
// is one factorization of H.
// The further right-hand sides of a multi-column solve, columns 1..n_rhs-1 of
// `rhs` / `out`, against the factor joint_pd_step_solve left in `solver` for
// column 0, with the sum-to-zero pins folded in as for that column. LM only:
// that is the branch that populates the factor.
inline bool lm_solve_further_columns(
    SparseHessianBuilder& H, SparseCholeskySolver& solver, int n_x,
    const double* rhs, double* out, int n_rhs
) {
    for (int c = 1; c < n_rhs; c++) {
        const std::size_t off = static_cast<std::size_t>(c) * n_x;
        if (!solver.solve(rhs + off, out + off, n_x)) return false;
        if (!apply_s2z_rank1_correction(solver, n_x, H.s2z_rank1, out + off,
                                        H.s2z_coupling))
            return false;
        for (int j = 0; j < n_x; j++)
            if (!std::isfinite(out[off + j])) return false;
    }
    return true;
}

// `grad` / `delta` hold n_rhs columns of length n_x; column 0 is the Newton
// step, the others are further solves against the same matrix (a hard
// constraint's A' columns), which the PSD branch cannot supply.
inline bool s2z_newton_step(
    SparseHessianBuilder& H, SparseCholeskySolver& solver,
    int n_x, JointPDMode pd_mode,
    const double* grad, double* delta,
    bool& used_block_schur,
    S2ZBlockSchurCache* bs_cache = nullptr,
    bool guarded = true,
    int n_rhs = 1
) {
    used_block_schur = false;
    if (n_rhs > 1 && pd_mode != JointPDMode::LM) return false;
    const bool pinned = pd_mode == JointPDMode::LM && !H.s2z_rank1.empty();
    if (pinned && s2z_block_schur(H, H.s2z_rank1, grad, delta, nullptr, bs_cache,
                                  /*field_inertia=*/true, nullptr, n_rhs)) {
        used_block_schur = true;
        return true;
    }
    if (pinned && !guarded) return false;
    bool ok = joint_pd_step_solve(H, solver, n_x, pd_mode, grad, delta, nullptr,
                                  nullptr, guarded ? JOINT_LM_MAX_TRIES : 1);
    // The Woodbury fold applies the rank-1 terms through `solver`, and only the
    // LM branch of joint_pd_step_solve populates that factor: the PSD branch
    // eigen-solves a densified copy and leaves the CHOLMOD factor absent or
    // stale, so folding there would correct the step against a different matrix.
    if (ok && pd_mode == JointPDMode::LM) {
        apply_s2z_rank1_correction(solver, n_x, H.s2z_rank1, delta,
                                   H.s2z_coupling);
        ok = lm_solve_further_columns(H, solver, n_x, grad, delta, n_rhs);
    }
    return ok;
}

// The KKT step of a sparse joint solve under hard constraints: the step and
// the A' columns solved against the same matrix (pins folded in), then the step
// projected onto the constraint set. LM: the multi-column solve needs the
// factor that branch leaves.
inline bool hard_kkt_step_sparse(
    SparseHessianBuilder& H, SparseCholeskySolver& solver, int n_x,
    HardConstraintState& hard, const double* x, const double* grad,
    double* delta, bool& used_block_schur, S2ZBlockSchurCache* bs_cache,
    bool guarded
) {
    const bool ok = s2z_newton_step(H, solver, n_x, JointPDMode::LM,
                                    hard.load_rhs(grad), hard.sol.data(),
                                    used_block_schur, bs_cache, guarded,
                                    hard.n_rhs());
    if (!ok || !hard.factor_cap_from_sol()) return false;
    std::copy(hard.sol.begin(), hard.sol.begin() + n_x, delta);
    hard.project_step(x, delta);
    return true;
}

// The final pass of the sparse joint Newton solve, from the gradient and the
// Hessian builder the likelihood + prior scatter filled at the returned iterate
// `scratch.x` (etas already at that point, no base ridge loaded): the
// conditioned log-determinant (the block-Schur factor of the pinned matrix
// where sum-to-zero pins are registered), the log-marginal, the inner-layer probes, the centred mode, the
// precision snapshot at the mode and the fixed-effect block. `result` arrives
// with the iteration's `converged` and `n_iter`. The one final pass behind the
// single-species loop and each species of the batched driver.
template<typename ComputeEtaJoint, typename CenterEffects,
         typename ComputeLogPriorJoint, typename JointLogLik,
         typename EvalObjective>
inline void joint_newton_finalize_sparse(
    LaplaceResult& result, int n_x, NewtonScratchJointSparse& scratch,
    SparseHessianBuilder& H_builder, DenseVec& grad,
    SparseCholeskySolver& solver,
    ComputeEtaJoint& compute_eta_joint, CenterEffects& center_effects_fn,
    ComputeLogPriorJoint& compute_log_prior_joint, JointLogLik& log_lik_fn,
    EvalObjective& eval_objective,
    bool store_Q, JointPDMode pd_mode,
    bool compute_skew, const std::vector<int>* skew_probe_idx,
    const JointCurvature3Oracles* curvature3_fns,
    const JointFixedBlockRequest* fixed_block,
    const SubspaceDebiasOptions* debias, const CilaOptions* cila,
    std::uint64_t cila_cell_key, const JointEtaVarRequest* eta_var,
    HardConstraintState* hard = nullptr
) {
    Rcpp::NumericVector& x = scratch.x;
    const bool hard_on = hard && hard->active();
    // The skewness and importance-curve probes walk coordinate curves of the
    // unconstrained Gaussian of the factored Hessian, a posterior
    // approximation of the model only where every hard row is a level row. The
    // debias and the corrected integrated Laplace condition their Gaussian on
    // the constraint set and run under any.
    const bool hard_probes_declined = hard_on && !hard->augmented_probes;
    // Read before joint_pd_step_solve below, which consumes grad. On the
    // constraint set the score's component along it is what vanishes.
    result.score_max = hard_on ? hard->max_abs_tangent(grad.data())
                               : max_abs(grad);
    H_builder.add_uniform_ridge(LAPLACE_UNIFORM_RIDGE);

    // PD-enforced final factorize so log_det is defined even when the mode sits
    // at an indefinite point (the delta is discarded here; we only need the
    // conditioned log-determinant for the log-marginal).
    //
    // Sum-to-zero rank-1 fields take the pinned route: log|H + sum_k coef_k 1_k
    // 1_k'| is read from a factor of that well-conditioned matrix (the constant
    // direction is pinned by the rank-1 block, so it is PD on the base ridge and
    // never triggers LM ridge escalation). This must be computed from the
    // freshly-scattered base-ridge H, BEFORE joint_pd_step_solve mutates the
    // diagonal: with the rank-1 left off the stored H, the constant direction of
    // H is unpinned and joint_pd_step_solve escalates the ridge to factor it,
    // which would inflate the determinant. Factoring H + 1 1' matches the dense
    // full-1 1' path. LM only (the PSD path densifies the small
    // Hessian and registers no rank-1).
    const bool s2z_direct =
        (pd_mode == JointPDMode::LM) && !H_builder.s2z_rank1.empty();
    const double S2Z_NA = std::numeric_limits<double>::quiet_NaN();
    double s2z_log_det = S2Z_NA;
    // The block-Schur factor of the pinned matrix yields the Newton step
    // (H + sum_k coef_k 1_k 1_k')^-1 grad together with the log-determinant,
    // so where it holds, the decrement below is read off it and the PD-enforced
    // solve of the unpinned H -- whose constant direction the base ridge alone
    // has to carry, so its escalation ladder is the normal case here -- is not
    // run. The corrected integrated Laplace reads the live CHOLMOD factor of
    // that solve, so a cell it is requested on keeps the solve.
    bool s2z_step_ok = false;
    if (s2z_direct) {
        TULPA_PROFILE_PHASE(PHASE_LOG_DET);
        double ld = S2Z_NA;
        // The field block need not be PD here (an unconverged iterate), so the
        // inertia route reads the pinned matrix wherever it is PD.
        // Under hard constraints the A' columns are solved with the step.
        const double* rhs = hard_on ? hard->load_rhs(grad.data()) : grad.data();
        double* sol = hard_on ? hard->sol.data() : scratch.delta.data();
        s2z_step_ok = s2z_block_schur(H_builder, H_builder.s2z_rank1,
                                      rhs, sol, &ld,
                                      &scratch.s2z_block_schur_cache,
                                      /*field_inertia=*/true,
                                      &result.s2z_field_indefinite,
                                      hard_on ? hard->n_rhs() : 1);
        if (s2z_step_ok && hard_on)
            std::copy(sol, sol + n_x, scratch.delta.begin());
        s2z_log_det = ld;
        // No factor of the pinned matrix could be formed, so the PD-enforced
        // value below stands in for it.
        result.s2z_log_det_fallback = !std::isfinite(s2z_log_det);
    }
    const bool run_pd_solve = !s2z_step_ok || (cila && cila->active());
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
    const bool want_eta_var =
        eta_var && eta_var->active() && scratch.extract_solver;
    std::vector<double> H_values_at_mode;
    if (store_Q || want_block || want_eta_var) H_values_at_mode = H_builder.values;

    bool pd_conditioned = false;
    bool step_ok = s2z_step_ok;
    bool folded = s2z_step_ok;
    bool hard_cols_ok = s2z_step_ok;
    if (run_pd_solve) {
        { TULPA_PROFILE_PHASE(PHASE_FACTORIZE);
          step_ok = joint_pd_step_solve(H_builder, solver, n_x, pd_mode,
                                        grad.data(), scratch.delta.data(),
                                        &result.log_det_Q, &pd_conditioned); }
        // The sum-to-zero pins folded into that solve's step the way the Newton
        // loop folds them.
        folded = step_ok;
        if (step_ok && pd_mode == JointPDMode::LM && !H_builder.s2z_rank1.empty()) {
            folded = apply_s2z_rank1_correction(solver, n_x, H_builder.s2z_rank1,
                                                scratch.delta.data(),
                                                H_builder.s2z_coupling);
        }
        // The A' columns against the same factor, where the block-Schur pass
        // did not already solve them.
        if (hard_on && !s2z_step_ok) {
            hard_cols_ok = step_ok && pd_mode == JointPDMode::LM;
            if (hard_cols_ok) {
                hard->load_rhs(grad.data());
                hard_cols_ok = lm_solve_further_columns(
                    H_builder, solver, n_x, hard->rhs.data(), hard->sol.data(),
                    hard->n_rhs());
            }
        }
    }
    // Prefer the cancellation-free factor of the pinned matrix; keep the
    // PD-enforced value only where that matrix is not PD (NaN fallback).
    if (s2z_direct && std::isfinite(s2z_log_det)) result.log_det_Q = s2z_log_det;
    // Constrained final pass: the step projected onto the constraint set, and
    // log|B' M B| on the (n_x - r)-dimensional subspace.
    bool hard_ok = false;
    if (hard_on) {
        hard_ok = hard_cols_ok && folded && hard->factor_cap_from_sol();
        if (hard_ok) {
            hard->project_step(x.begin(), scratch.delta.data());
            result.log_det_Q += hard->log_det_subspace_correction();
        } else {
            result.log_det_Q = std::numeric_limits<double>::quiet_NaN();
        }
        folded = hard_ok;
    }
    // The Newton decrement off the step one more iteration would take.
    if (folded) {
        double dec = 0.0;
        for (int j = 0; j < n_x; j++) dec += grad[j] * scratch.delta[j];
        if (std::isfinite(dec)) result.newton_decrement = dec;
    }

    // Whether the Hessian at the returned point is the PD matrix the expansion
    // needs. On the sum-to-zero path the escalation `pd_conditioned` reports is
    // an artefact of the rank-1 pins being left off the STORED H, so the reading
    // there is the factor of the pinned matrix, which is the true one.
    result.pd_conditioned = pd_conditioned;
    result.hessian_pd_at_mode =
        (s2z_direct ? std::isfinite(s2z_log_det) : !pd_conditioned) &&
        (!hard_on || hard_ok);

    double log_lik, log_prior;
    { TULPA_PROFILE_PHASE(PHASE_LOG_LIK_PRIOR);
      log_lik   = log_lik_fn(scratch.etas);
      log_prior = compute_log_prior_joint(x, scratch.etas); }

    // The constrained model integrates over n_x - r dimensions.
    result.log_marginal = hard_on
        ? hard->log_marginal(log_lik, log_prior, result.log_det_Q)
        : finalize_log_marginal(log_lik, log_prior, result.log_det_Q, n_x);
    if (std::isfinite(result.newton_decrement)) {
        result.newton_step_gain = newton_step_gain(
            x, scratch.delta.data(), n_x, log_lik + log_prior, eval_objective,
            scratch.x_try);
    }

    // The inner-layer probes solve against the Hessian the solve stepped with,
    // so each condition below is a way for the factor they read to hold a
    // different matrix. They are separated rather than collapsed into one flag
    // so a declined fit says WHICH: a probe that never ran because the solve
    // stalled sends a reader to a different question than one blocked by the PD
    // mode this fit was configured with.
    //
    // With sum-to-zero pins registered the stepped matrix is H + the pins, which
    // the live CHOLMOD factor of H alone does not hold; the skewness and
    // importance-curve probes then solve through the same block-Schur fold of
    // the pinned matrix the step and the log-determinant were read off
    // (`pinned_solve`). That needs the builder to still hold H at the mode on the
    // base ridge: a PD-enforced solve that had to raise the ridge leaves
    // H + lambda I in it. The subspace debias reads the live factor itself
    // (its triangular factor, not only solves), so it keeps declining there.
    const char* factor_declined = nullptr;
    if (!result.converged)                    factor_declined = "not_converged";
    else if (pd_mode != JointPDMode::LM)      factor_declined = "pd_eigen_clamp";
    else if (s2z_direct && (!s2z_step_ok || pd_conditioned))
                                              factor_declined = "s2z_rank1_factor";
    else if (!s2z_direct && !solver.factored())
                                              factor_declined = "factor_unavailable";
    const char* live_factor_declined =
        (factor_declined == nullptr && s2z_direct) ? "s2z_rank1_factor"
                                                   : factor_declined;
    const char* skew_declined =
        (factor_declined == nullptr && hard_probes_declined) ? "hard_constraint"
                                                             : factor_declined;
    const bool skew_factor_valid = (skew_declined == nullptr);
    InnerPinnedSolve pinned_solve;
    if (s2z_direct) {
        pinned_solve = [&](const double* rhs, double* out) {
            return s2z_block_schur(H_builder, H_builder.s2z_rank1, rhs, out,
                                   nullptr, &scratch.s2z_block_schur_cache,
                                   /*field_inertia=*/true);
        };
    }
    const InnerPinnedSolve* probe_solve = s2z_direct ? &pinned_solve : nullptr;

    std::vector<double> pre_center_x(n_x);
    for (int j = 0; j < n_x; j++) pre_center_x[j] = x[j];
    DenseCholeskyScratch unused_dense_chol;  // sparse-only path never reads it
    {   // post-mode probes of the inner layer, timed as one phase
        TULPA_PROFILE_PHASE(PHASE_INNER_DIAG);
        if (compute_skew) {
            std::vector<int> all_idx;
            const std::vector<int>& probe =
                inner_probe_indices(n_x, skew_probe_idx, all_idx);
            if (!skew_factor_valid) {
                inner_probe_decline(result, probe, skew_declined);
            } else if (curvature3_fns) {
                InnerSkewOutcome sk = compute_inner_skew_gamma3_joint(
                    n_x, pre_center_x, unused_dense_chol, solver, /*use_sparse=*/true,
                    compute_eta_joint, x, scratch.etas, scratch.etas_tmp,
                    *curvature3_fns, probe, probe_solve
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
                    eval_objective, x, probe, probe_solve
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
                            result.converged ? live_factor_declined : nullptr,
                            hard_on ? hard : nullptr);

        run_inner_cila(result, n_x, pre_center_x, unused_dense_chol, solver,
                       /*use_sparse=*/true, eval_objective,
                       [&](Rcpp::NumericVector& xv) { center_effects_fn(xv); },
                       x, cila, cila_cell_key, hard_on ? hard : nullptr);
    }

    { TULPA_PROFILE_PHASE(PHASE_LOG_LIK_PRIOR);
      center_effects_fn(x); }
    for (int j = 0; j < n_x; j++) result.mode[j] = x[j];

    // Both read the snapshot taken before the PD-enforced factorize, so the
    // precision the caller gets is the one the scatter built and the block is a
    // covariance of that rather than of a ridge-inflated matrix. The pattern is
    // fit-level and untouched by the escalation, so only the values are held.
    if (want_block) {
        TULPA_PROFILE_PHASE(PHASE_HESSIAN_EXTRACT);
        // The block is conditioned on every constraint the model carries: the
        // caller's sum-to-zero groups and the blocks' hard rows.
        JointFixedBlockRequest req_hard;
        const JointFixedBlockRequest* req = fixed_block;
        if (hard_on) {
            req_hard = *fixed_block;
            append_new_constraint_rows(req_hard.A_cols, hard->rows);
            req = &req_hard;
        }
        extract_joint_fixed_block(
            H_builder.col_ptr.data(), H_builder.row_idx.data(),
            H_values_at_mode.data(), n_x,
            static_cast<int>(H_values_at_mode.size()), *req,
            *scratch.extract_solver,
            result.re_cov_flat, result.re_cov_block_sizes);
    }

    if (want_eta_var) {
        TULPA_PROFILE_PHASE(PHASE_HESSIAN_EXTRACT);
        RowLoadings loadings;
        eta_var->loadings(pre_center_x.data(), loadings);
        result.eta_var.assign(static_cast<std::size_t>(loadings.n_rows()),
                              std::numeric_limits<double>::quiet_NaN());
        // The pins the scatter registered on the side enter as their soft
        // precision and the hard rows as the D^{-1} = 0 limit of the same
        // Woodbury fold.
        const auto& r1 = H_builder.s2z_rank1;
        std::vector<LinearConstraintRow> pin_groups(r1.size());
        for (std::size_t k = 0; k < r1.size(); k++) {
            pin_groups[k].cols.resize(r1[k].n);
            for (int i = 0; i < r1[k].n; i++) pin_groups[k].cols[i] = r1[k].node(i);
        }
        std::vector<double> pin_Dinv;
        double pin_log_det = 0.0;
        if (result.hessian_pd_at_mode &&
            s2z_build_Dinv(r1, H_builder.s2z_coupling, pin_Dinv, pin_log_det)) {
            if (hard_on) {
                const std::size_t K = pin_groups.size();
                const std::size_t KR = K + static_cast<std::size_t>(hard->rank());
                std::vector<double> D(KR * KR, 0.0);
                for (std::size_t a = 0; a < K; a++)
                    for (std::size_t b = 0; b < K; b++)
                        D[a * KR + b] = pin_Dinv[a * K + b];
                pin_Dinv = std::move(D);
                pin_groups.insert(pin_groups.end(), hard->rows.begin(),
                                  hard->rows.end());
            }
            extract_joint_eta_var(
                H_builder.col_ptr.data(), H_builder.row_idx.data(),
                H_values_at_mode.data(), n_x,
                static_cast<int>(H_values_at_mode.size()), loadings,
                pin_groups, pin_Dinv, *scratch.extract_solver, result.eta_var);
        }
    }

    if (store_Q) {
        // Copy the CSC arrays out so the caller doesn't depend on H_builder
        // staying alive.
        result.Q_csc_p = H_builder.col_ptr;
        result.Q_csc_i = H_builder.row_idx;
        result.Q_csc_x = std::move(H_values_at_mode);
        result.Q_csc_n = n_x;
    }

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
    // Curvature each inner step is built from (joint_pd_step.h). The scatter
    // receives the CurvatureMode to assemble at; the final pass is observed.
    StepCurvature step_curvature = StepCurvature::Observed,
    int hessian_refresh = 1,
    // Inner-Laplace skewness diagnostic (inner_laplace_skew.h), opt-in like
    // store_Q. The s2z (sum-to-zero large-field) path solves against H + a
    // rank-1 correction the stored CHOLMOD factor of H_builder alone does not
    // reflect, so its probes solve through the block-Schur fold of that pinned
    // matrix instead (joint_newton_finalize_sparse). The PSD path never
    // populates `solver` at all (its final factorize goes through the dense
    // eigen-clamp branch of joint_pd_step_solve), so it declines BY NAME -- the
    // probed indices come back NaN carrying the reason -- rather than report a
    // wrong gamma_3 or emit nothing at all.
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
    std::uint64_t cila_cell_key = 0,
    // Per-row predictive variance of the linear predictor. Read, like the
    // fixed-effect block, off the snapshot of the precision at the mode with
    // its own factorization; the sum-to-zero pins the scatter registered on the
    // side rather than in the stored H are folded in by Woodbury, so the
    // variance is the one of the precision the log-determinant was taken of.
    const JointEtaVarRequest* eta_var = nullptr,
    // Hard linear constraints the blocks carry (hard_constraints.h); nullptr
    // or empty solves the unconstrained problem.
    HardConstraintState* hard = nullptr
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

    // Under hard constraints every iterate stays on the constraint set, so a
    // warm start from anywhere else is moved to its nearest feasible point.
    const bool hard_on = hard && hard->active();
    if (hard_on) hard->project_point(x.begin());

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
            !H_builder.s2z_rank1.empty() ||  // s2z large-field path: block-Schur every iter, no reuse
            hard_on;                         // the projection needs this iteration's factor

        // Zero, scatter at `cm` and load the uniform upstream base ridge (the
        // numerical hygiene of an already-PD H).
        auto assemble = [&](CurvatureMode cm, bool grad_only) {
            scratch.zero_grad();
            H_builder.zero();
            { TULPA_PROFILE_PHASE(PHASE_SCATTER);
              scatter_joint_sparse(x, scratch.etas, scratch.grad, H_builder,
                                   cm, grad_only); }
            H_builder.add_uniform_ridge(LAPLACE_UNIFORM_RIDGE);
        };

        // PD-enforced factor + solve, at the curvature the policy picks. LM
        // escalates the ridge until CHOLMOD factorizes; PSD eigen-clamps the
        // (small) dense Hessian. Either yields a usable ascent step where a
        // plain factorize of the indefinite mixture Hessian would fail. On
        // reuse iterations the cached factor is re-applied to the refreshed
        // gradient instead (see `reuse_enabled`), and the scatter builds the
        // gradient only.
        bool solve_ok;
        if (do_factor) {
            // Exact block-Schur step on the s2z large-field path (true Newton,
            // quadratic), else LM ridge + Woodbury. s2z_newton_step folds the
            // rank-1 correction itself. Block-Schur does NOT populate the
            // CHOLMOD `solver` factor; reuse is forced off on the s2z path
            // (see do_factor), so have_factor stays false there.
            bool used_block_schur = false;
            solve_ok = inner_step_under_curvature(
                step_curvature,
                [&](CurvatureMode cm) { assemble(cm, /*grad_only=*/false); },
                [&](bool guarded) {
                    TULPA_PROFILE_PHASE(PHASE_FACTORIZE);
                    if (hard_on) {
                        return hard_kkt_step_sparse(
                            H_builder, solver, n_x, *hard, x.begin(),
                            scratch.grad.data(), scratch.delta.data(),
                            used_block_schur, &scratch.s2z_block_schur_cache,
                            guarded);
                    }
                    return s2z_newton_step(H_builder, solver, n_x, pd_mode,
                                           scratch.grad.data(),
                                           scratch.delta.data(),
                                           used_block_schur,
                                           &scratch.s2z_block_schur_cache,
                                           guarded);
                });
            have_factor = solve_ok && !used_block_schur;
        } else {
            assemble(inner_step_plan(step_curvature).first, /*grad_only=*/true);
            TULPA_PROFILE_PHASE(PHASE_FACTORIZE);
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
                           CurvatureMode::Observed, /*grad_only=*/false); }
    joint_newton_finalize_sparse(
        result, n_x, scratch, H_builder, scratch.grad, solver,
        compute_eta_joint, center_effects_fn, compute_log_prior_joint,
        log_lik_fn, eval_objective, store_Q, pd_mode, compute_skew,
        skew_probe_idx, curvature3_fns, fixed_block, debias, cila,
        cila_cell_key, eta_var, hard_on ? hard : nullptr);

    return result;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_JOINT_SPARSE_H
