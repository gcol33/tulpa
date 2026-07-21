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

    void allocate(int n_x, const std::vector<JointArm>& arms) {
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

// PD-enforcement mode for the inner Newton step. The occupancy-mixture arm's
// observed Hessian can be indefinite away from the mode, so a plain CHOLMOD
// factorize fails ("matrix not positive definite") and the step stalls.
enum class JointPDMode { LM = 0, PSD = 1 };

// Cap on n_x for the dense PSD eigen-clamp path. The sparse Newton supports
// fields up to ~10^6 (see header); densifying those would be catastrophic, so
// above this dimension PSD silently falls back to the LM ridge.
inline constexpr int JOINT_PSD_MAX_DIM = 4000;

// Factor + solve for one inner Newton step, enforcing positive-definiteness.
//   LM  : escalate a uniform diagonal ridge until CHOLMOD factorizes, giving a
//         damped-Newton step that interpolates toward gradient ascent. Reduces
//         to plain Newton when H is already PD (one factorize, no escalation).
//   PSD : densify H, symmetric-eigendecompose, clamp eigenvalues to a positive
//         floor, and solve from the clamped spectrum -- PD in one shot.
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
    double* out_log_det = nullptr
) {
    bool use_psd = (pd_mode == JointPDMode::PSD) && (n_x <= JOINT_PSD_MAX_DIM);

    if (use_psd) {
        Eigen::MatrixXd Hd = Eigen::MatrixXd::Zero(n_x, n_x);
        for (int j = 0; j < H.n; ++j) {
            for (int p = H.col_ptr[j]; p < H.col_ptr[j + 1]; ++p) {
                const int i = H.row_idx[p];
                const double v = H.values[p];
                Hd(i, j) = v;
                if (i != j) Hd(j, i) = v;   // CSC stores lower triangle only
            }
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
        if (es.info() != Eigen::Success) return false;
        Eigen::VectorXd ev = es.eigenvalues();
        const double lam_max = ev.cwiseAbs().maxCoeff();
        const double floor = std::max(1e-8 * std::max(lam_max, 1.0), 1e-10);
        double log_det = 0.0;
        for (int i = 0; i < n_x; ++i) {
            if (ev[i] < floor) ev[i] = floor;
            log_det += std::log(ev[i]);
        }
        Eigen::Map<const Eigen::VectorXd> g(grad, n_x);
        Eigen::VectorXd y = es.eigenvectors().transpose() * g;
        for (int i = 0; i < n_x; ++i) y[i] /= ev[i];
        Eigen::VectorXd d = es.eigenvectors() * y;
        for (int i = 0; i < n_x; ++i) {
            if (!std::isfinite(d[i])) return false;
            delta[i] = d[i];
        }
        if (out_log_det) *out_log_det = log_det;
        return true;
    }

    // LM escalating ridge. Base ridge already on the diagonal; escalate the
    // ADDITIONAL diagonal load multiplicatively until CHOLMOD succeeds.
    double added = 0.0;
    for (int t = 0; t < 32; ++t) {
        cholmod_sparse H_cholmod = H.as_cholmod(&solver.common());
        if (!solver.analyzed()) solver.analyze(&H_cholmod);
        if (solver.factorize(&H_cholmod)) {
            solver.solve(grad, delta, n_x);
            bool finite = true;
            for (int i = 0; i < n_x; ++i)
                if (!std::isfinite(delta[i])) { finite = false; break; }
            if (finite) {
                if (out_log_det) *out_log_det = solver.log_determinant();
                return true;
            }
        }
        const double bump = (added == 0.0) ? 1e-6 : added * 9.0;
        H.add_uniform_ridge(bump);
        added += bump;
    }
    return false;
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
    if (ok) apply_s2z_rank1_correction(solver, n_x, H.s2z_rank1, delta,
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
    int hessian_refresh = 1
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
              solver.solve(scratch.grad.data(), scratch.delta.data(), n_x);
              solve_ok = true;
              for (int j = 0; j < n_x; j++)
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
            // Both conditioners failed; take a tiny gradient-ascent step.
            for (int j = 0; j < n_x; j++) {
                if (std::isfinite(scratch.grad[j])) {
                    x[j] += 1e-4 * scratch.grad[j];
                }
            }
            obj_valid = false;
            result.n_iter = iter + 1;
            continue;
        }

        if (!obj_valid) {
            obj_current = eval_objective(x);
            obj_valid = true;
        }

        double step_scale;
        { TULPA_PROFILE_PHASE(PHASE_LINE_SEARCH);
          double slope = newton_decrement(scratch.grad, scratch.delta, n_x);
          step_scale = line_search_backtrack(
              x, scratch.delta, n_x, obj_current, slope, eval_objective,
              obj_current, scratch.x_try
          ); }

        result.n_iter = iter + 1;
        if (newton_converged(scratch.delta, scratch.grad, step_scale, n_x, tol,
                             conv_state)) {
            result.converged = true;
            break;
        }
    }

    // Final pass at the Newton mode (uncentered): rebuild grad + H for
    // log_det evaluation, then center post-hoc. See laplace_newton_joint.h
    // line 207 for the BYM2/ICAR rank-deficiency rationale.
    { TULPA_PROFILE_PHASE(PHASE_ETA);
      compute_eta_joint(x, scratch.etas); }
    scratch.zero_grad();
    H_builder.zero();
    { TULPA_PROFILE_PHASE(PHASE_SCATTER);
      scatter_joint_sparse(x, scratch.etas, scratch.grad, H_builder,
                           /*finalize=*/true, /*grad_only=*/false); }
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
    }
    { TULPA_PROFILE_PHASE(PHASE_FACTORIZE);
      joint_pd_step_solve(H_builder, solver, n_x, pd_mode,
                          scratch.grad.data(), scratch.delta.data(),
                          &result.log_det_Q); }
    // Prefer the cancellation-free direct factor; keep the PD-enforced value only
    // if the direct factor was non-PD (NaN fallback).
    if (s2z_direct && std::isfinite(s2z_log_det)) result.log_det_Q = s2z_log_det;

    double log_lik, log_prior;
    { TULPA_PROFILE_PHASE(PHASE_LOG_LIK_PRIOR);
      log_lik   = log_lik_fn(scratch.etas);
      log_prior = compute_log_prior_joint(x, scratch.etas); }

    result.log_marginal = finalize_log_marginal(log_lik, log_prior,
                                                  result.log_det_Q, n_x);

    { TULPA_PROFILE_PHASE(PHASE_LOG_LIK_PRIOR);
      center_effects_fn(x); }
    for (int j = 0; j < n_x; j++) result.mode[j] = x[j];

    if (store_Q) {
        // Copy the CSC arrays out so the caller doesn't depend on H_builder
        // staying alive.
        result.Q_csc_p = H_builder.col_ptr;
        result.Q_csc_i = H_builder.row_idx;
        result.Q_csc_x = H_builder.values;
        result.Q_csc_n = n_x;
    }

    return result;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_NEWTON_JOINT_SPARSE_H
