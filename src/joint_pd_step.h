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

#ifndef TULPA_JOINT_PD_STEP_H
#define TULPA_JOINT_PD_STEP_H

#include <RcppEigen.h>
#include <cmath>

namespace tulpa {

// PD-enforcement mode for the inner Newton step.
enum class JointPDMode { LM = 0, PSD = 1 };

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
template <typename FactorSolve, typename AddRidge>
inline bool pd_lm_escalate(
    FactorSolve factor_solve, AddRidge add_ridge,
    double* out_log_det = nullptr,
    bool* out_modified = nullptr
) {
    double added = 0.0;
    for (int t = 0; t < JOINT_LM_MAX_TRIES; ++t) {
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

} // namespace tulpa

#endif // TULPA_JOINT_PD_STEP_H
