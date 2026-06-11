// spde_logdet.h
// Prior normalizing constant 0.5*log|Q(theta)| for an SPDE Matern precision
// built by SpdeQBuilder.
//
// The Laplace marginal of a latent Gaussian model needs BOTH determinant
// terms of the GMRF prior and the Gaussian-approx posterior:
//
//   log p(y|theta) ~ log p(y|x_hat) - 0.5 x_hat' Q x_hat
//                    + 0.5 log|Q(theta)|        <- prior normalizer (HERE)
//                    - 0.5 log|H(theta)|        <- Laplace/Hessian normalizer
//                    + (n_x/2) log(2 pi).
//
// H = Q + A'WA != Q, so -0.5 log|H| does NOT absorb +0.5 log|Q|. Omitting the
// prior normalizer drops the Occam factor (~ n_mesh*log(tau) ~ -n_mesh*log(sigma))
// that bends the marginal down at large sigma, so the deterministic nested-Laplace
// (range, sigma) marginal becomes monotone in sigma and the optimizer rails sigma
// to the prior boundary. With it, the marginal has an interior maximum at the
// data-supported (range, sigma) -- matching the Tier-1 NUTS-joint integrator,
// which carries the theta-dependent normalizer through log p(x|theta) by
// construction. Every other continuous-hyperparameter block (CAR_proper via
// car_log_det, tgmrf via logdet_Q_per_grid) already injects this term through its
// log_prior; this header is the SPDE counterpart.
//
// Q is proper (kappa^2 > 0) plus the orphan diagonal ridge, hence PD, so the bare
// supernodal Cholesky succeeds. On a numerically non-PD cell half_logdet() returns
// false and the caller treats the cell as infeasible (log_marginal = -inf),
// matching the proper-CAR PD gate.

#ifndef TULPA_SPDE_LOGDET_H
#define TULPA_SPDE_LOGDET_H

#include "spde_qbuilder.h"
#include "sparse_cholesky.h"
#include <vector>
#include <cstddef>

namespace tulpa {

struct SpdeQLogDet {
    std::vector<int>     Lp, Li;   // lower-triangle CSC pattern (built once)
    std::vector<int>     src;      // src[k] = full-Q slot feeding lower entry k
    std::vector<double>  Lx;       // lower-triangle values (refilled per cell)
    SparseCholeskySolver solver;
    bool                 built = false;

    // The SpdeQBuilder stores the FULL symmetric Q in CSC sorted by (col, row).
    // Extract the lower triangle (row >= col) once: the pattern is fixed across
    // outer-grid cells, only the values change with (range, sigma).
    void build_pattern(const SpdeQBuilder& qb) {
        const int n = qb.n_mesh;
        Lp.assign(n + 1, 0);
        Li.clear();
        src.clear();
        for (int col = 0; col < n; ++col) {
            Lp[col] = static_cast<int>(Li.size());
            for (int idx = qb.Q_p[col]; idx < qb.Q_p[col + 1]; ++idx) {
                const int row = qb.Q_i[idx];
                if (row >= col) {
                    Li.push_back(row);
                    src.push_back(idx);
                }
            }
        }
        Lp[n] = static_cast<int>(Li.size());
        Lx.assign(Li.size(), 0.0);
        built = true;
    }

    // On success, sets out = 0.5 * log|Q| and returns true. analyze() runs once
    // (symbolic factor reused across cells); factorize() re-runs per call.
    bool half_logdet(const SpdeQBuilder& qb, double& out) {
        if (!built) build_pattern(qb);
        for (std::size_t k = 0; k < src.size(); ++k) Lx[k] = qb.Q_x[src[k]];

        cholmod_sparse A;
        A.nrow   = static_cast<std::size_t>(qb.n_mesh);
        A.ncol   = static_cast<std::size_t>(qb.n_mesh);
        A.nzmax  = static_cast<std::size_t>(Li.size());
        A.p      = Lp.data();
        A.i      = Li.data();
        A.x      = Lx.data();
        A.z      = nullptr;
        A.stype  = -1;            // symmetric, lower triangle stored
        A.itype  = CHOLMOD_INT;
        A.xtype  = CHOLMOD_REAL;
        A.dtype  = CHOLMOD_DOUBLE;
        A.sorted = 1;
        A.packed = 1;

        if (!solver.analyzed()) solver.analyze(&A);
        if (!solver.factorize(&A)) return false;
        out = 0.5 * solver.log_determinant();
        return true;
    }
};

} // namespace tulpa

#endif // TULPA_SPDE_LOGDET_H
