// joint_pd_step.h
// Positive-definiteness enforcement for one inner Newton step of a joint
// Laplace solve.
//
// The negative Hessian of a coupled log posterior need not be positive definite
// away from the mode. The occupancy mixture's dark-cell term
// log(psi (1-p)^J + 1 - psi) is not concave in (eta_occ, eta_det), so a cell
// with no detection contributes a negative curvature direction and a plain
// Cholesky of that Hessian has no factor: the raw Newton direction does not
// exist, and a loop that only knows how to take it makes no progress at all.
//
// Two conditioners produce a usable ascent direction there:
//
//   LM   the smallest diagonal load tau for which H + tau I factorizes, found
//        by escalating tau until the factorization succeeds (Nocedal & Wright,
//        Numerical Optimization 2nd ed., Alg. 3.3, "Cholesky with added
//        multiple of the identity"). H + tau I is PD, so
//        grad' (H + tau I)^-1 grad > 0 and the line search has an ascent
//        direction to backtrack along. When H is already PD the first attempt
//        succeeds, tau stays 0, and the step IS the plain Newton step.
//   PSD  eigendecompose the (small, densified) H and clamp its spectrum to a
//        positive floor, which is PD in one shot.
//
// Both policies are shared by the dense and the sparse joint Newton loops.
// Only the factorization backend differs, and it enters as a callback, so the
// escalation schedule and the eigen clamp have one definition each.
//
// A coupled cell can also supply its complete-data EXPECTED information
// (CurvatureMode::Expected), which is PSD by construction, so a third way to an
// ascent direction is to step on it (Fisher scoring) instead of loading the
// observed Hessian's diagonal. `StepCurvature` selects which curvature an inner
// step is built from; `inner_step_under_curvature()` is the one place the
// choice is made, for every joint loop.

#ifndef TULPA_JOINT_PD_STEP_H
#define TULPA_JOINT_PD_STEP_H

#include <RcppEigen.h>
#include <cmath>

#include "tulpa/cell_coupling.h"   // CurvatureMode

namespace tulpa {

// PD-enforcement mode for the inner Newton step.
enum class JointPDMode { LM = 0, PSD = 1 };

// Curvature an inner Newton step is built from.
//   Observed  the observed Hessian under the PD guard.
//   Expected  the coupled cells' expected information under the PD guard
//             (Fisher scoring): PSD, so it factors, but the step contracts
//             linearly towards the mode.
//   Auto      the observed Hessian wherever it factors as it stands, which is
//             the Newton step and converges quadratically; where it does not,
//             the expected information. An indefinite observed Hessian is
//             then never conditioned by the escalating ridge, whose step at a
//             load comparable to the most negative eigenvalue is a short
//             gradient step and costs a factorization per rung of the ladder.
// Without a coupled cell nothing supplies an expected form, so Auto and
// Expected both reduce to Observed (`effective_step_curvature()`).
enum class StepCurvature { Observed = 0, Expected = 1, Auto = 2 };

inline StepCurvature effective_step_curvature(StepCurvature requested,
                                              bool any_coupling) {
    return any_coupling ? requested : StepCurvature::Observed;
}

// Cap on n_x for the dense PSD eigen-clamp path. The sparse Newton supports
// fields up to ~10^6; densifying those would be catastrophic, so above this
// dimension PSD falls back to the LM ridge.
inline constexpr int JOINT_PSD_MAX_DIM = 4000;

// LM escalation schedule: how many factorization attempts, the first added
// load, and the multiplicative growth of the total load between attempts.
inline constexpr int    JOINT_LM_MAX_TRIES    = 32;
inline constexpr double JOINT_LM_RIDGE_INIT   = 1e-6;
inline constexpr double JOINT_LM_RIDGE_GROWTH = 9.0;

// Eigen-clamp floor: the smallest eigenvalue the clamped spectrum may hold,
// as a fraction of the largest absolute eigenvalue, with an absolute lower
// bound for a matrix whose whole spectrum is tiny. The relative form caps the
// clamped matrix's condition number at 1 / JOINT_PSD_FLOOR_REL; the absolute
// one keeps the floor away from zero when lam_max itself is below 1.
inline constexpr double JOINT_PSD_FLOOR_REL = 1e-8;
inline constexpr double JOINT_PSD_FLOOR_ABS = 1e-10;

// Eigen-clamp step: solve H delta = grad from the symmetric eigendecomposition
// of `Hd` with the spectrum clamped to a positive floor. `out_log_det`, when
// non-null, receives the log-determinant of the clamped matrix. `out_modified`
// records whether any eigenvalue had to be clamped, i.e. whether the matrix
// solved against is the one that was handed in.
inline bool pd_eigen_clamp_solve(
    const Eigen::MatrixXd& Hd, int n_x,
    const double* grad, double* delta,
    double* out_log_det = nullptr,
    bool* out_modified = nullptr
) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
    if (es.info() != Eigen::Success) return false;
    Eigen::VectorXd ev = es.eigenvalues();
    const double lam_max = ev.cwiseAbs().maxCoeff();
    const double floor = std::max(JOINT_PSD_FLOOR_REL * std::max(lam_max, 1.0),
                                  JOINT_PSD_FLOOR_ABS);
    double log_det = 0.0;
    bool clamped = false;
    for (int i = 0; i < n_x; ++i) {
        if (ev[i] < floor) { ev[i] = floor; clamped = true; }
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
    if (out_modified) *out_modified = clamped;
    return true;
}

// LM escalating-ridge step.
//   `factor_solve(double* log_det) -> bool` attempts one factorization of the
//       CURRENT Hessian and writes the step; it returns false when the
//       factorization fails or the step is not finite.
//   `add_ridge(double bump)` loads `bump` onto the Hessian diagonal.
// `out_modified`, when non-null, records whether any load had to be added,
// i.e. whether the factorization that succeeded is of the matrix handed in.
// `max_tries = 1` factors the Hessian as handed in: the test of whether it is
// PD as it stands (a failed test leaves one load on the diagonal, and the
// caller reassembles before it steps).
template <typename FactorSolve, typename AddRidge>
inline bool pd_lm_escalate(
    FactorSolve factor_solve, AddRidge add_ridge,
    double* out_log_det = nullptr,
    bool* out_modified = nullptr,
    int max_tries = JOINT_LM_MAX_TRIES
) {
    double added = 0.0;
    for (int t = 0; t < max_tries; ++t) {
        double log_det = 0.0;
        if (factor_solve(&log_det)) {
            if (out_log_det) *out_log_det = log_det;
            if (out_modified) *out_modified = (added > 0.0);
            return true;
        }
        const double bump = (added == 0.0) ? JOINT_LM_RIDGE_INIT
                                           : added * JOINT_LM_RIDGE_GROWTH;
        add_ridge(bump);
        added += bump;
    }
    if (out_modified) *out_modified = true;
    return false;
}

// What one inner step under a curvature policy does: assemble at `first`,
// factor it (`first_guarded` = under the PD guard, else one factorization of
// the matrix as it stands), and, where that unguarded factorization fails,
// assemble the expected information and take the guarded step on it. The
// gradient is the same under either curvature. Every joint loop reads this
// plan, so the rule has one definition whether a loop assembles one problem at
// a time (inner_step_under_curvature) or a batch of species at once.
struct InnerStepPlan {
    CurvatureMode first;
    bool          first_guarded;
};

inline InnerStepPlan inner_step_plan(StepCurvature policy) {
    switch (policy) {
    case StepCurvature::Expected: return {CurvatureMode::Expected, true};
    case StepCurvature::Auto:     return {CurvatureMode::Observed, false};
    default:                      return {CurvatureMode::Observed, true};
    }
}

// One inner step under a curvature policy, for a loop that assembles one
// problem at a time.
//   `assemble(CurvatureMode)` zeroes the gradient and Hessian and scatters
//       them at that curvature, priors and base ridge included, so the step
//       sees the matrix it is to factor.
//   `solve(bool guarded) -> bool` factors that Hessian and writes the step.
template <typename Assemble, typename Solve>
inline bool inner_step_under_curvature(StepCurvature policy, Assemble assemble,
                                       Solve solve) {
    const InnerStepPlan plan = inner_step_plan(policy);
    assemble(plan.first);
    if (solve(plan.first_guarded)) return true;
    if (plan.first_guarded) return false;
    assemble(CurvatureMode::Expected);
    return solve(true);
}

} // namespace tulpa

#endif // TULPA_JOINT_PD_STEP_H
