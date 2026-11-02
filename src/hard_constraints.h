// hard_constraints.h
// Hard linear constraints A x = 0 in the inner Newton solves.
//
// Two kinds of block declare rows of A (LatentBlock::hard_constraints):
//
//   * a block whose intrinsic prior has null directions no fixed effect
//     absorbs (a Knorr-Held interaction: one per time and one per area,
//     aliased with the temporal and spatial main effects);
//   * an intrinsic field whose constant direction aliases a fixed effect (an
//     ICAR / BYM2 structured / RW level against the intercept, or against a
//     covariate's coefficient for a varying coefficient). Its level is the
//     LEVEL ROW (declare_level_constraint): the model puts the level in the
//     coefficient, where that coefficient's own prior acts on it.
//
// The model is the prior CONDITIONED on A x = 0, so the inner problem is an
// equality-constrained maximisation:
//
//   * each Newton step is the KKT step
//       delta_c = delta - W C^{-1} A (x + delta),  W = M^{-1} A',  C = A W,
//     with delta = M^{-1} grad, which keeps every iterate on the constraint
//     set and converges to the constrained mode;
//   * the Laplace approximation integrates over the (n_x - r)-dimensional
//     subspace B spanned by null(A), and
//       log|B' M B| = log|M| + log|C| - log|A A'|
//     is the log-determinant it needs (B orthonormal).
//
// M is the full precision the loop factors (the stored Hessian with every
// sum-to-zero pin, written in or folded in by the solver). A block's prior may
// add any multiple of A'A to make M positive definite along the constrained
// directions: on the subspace it is zero, so neither the step nor
// log|B' M B| sees it. That is what the precision augmentation of an intrinsic
// field does along its level row.
//
// A level row's block carries the AUGMENTED density as its log-prior, a proper
// Gaussian with precision P on the whole block, and P 1 = tau a for the row's
// weights a (the augmentation pins each component's sum at the field
// precision times that component's multiplier). The density of the field
// conditioned on a'x = 0, with respect to Lebesgue measure on the subspace, is
// that density divided by the density of s = a'x at zero and by |a|:
//
//   log p(x | a'x = 0) = log p_aug(x) + 0.5 log(2 pi a' P^{-1} a) - 0.5 log(a'a),
//
// and a' P^{-1} a = (a'1)^2 / (1' P 1) because P^{-1} a = 1 / tau. That
// constant (level_prior_conditioning) is what turns the augmented density into
// the conditioned one in the log-marginal; it carries the field precision, so
// it moves with the outer grid.

#ifndef TULPA_HARD_CONSTRAINTS_H
#define TULPA_HARD_CONSTRAINTS_H

#include "inv_block_extract.h"
#include "latent_block.h"
#include "laplace_newton_loop.h"   // finalize_log_marginal
#include "linear_constraint_rows.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

// The hard constraint rows of a block list, in block order.
inline std::vector<LinearConstraintRow> collect_hard_constraints(
    const std::vector<LatentBlock>& blocks
) {
    std::vector<LinearConstraintRow> rows;
    for (const auto& b : blocks)
        rows.insert(rows.end(), b.hard_constraints.begin(),
                    b.hard_constraints.end());
    return rows;
}

// The log-density conditioning of every level row of a block list at grid cell
// k_grid (see the header): sum over level-constrained blocks of
// 0.5 log(2 pi (a'1)^2 / (1' P 1)) - 0.5 log(a'a). 1' P 1 is read off the
// block's own log-prior as the second difference along the block's constant,
// exact for the Gaussian prior it is. `probe` is caller-owned scratch of at
// least the latent length; its contents are restored on return.
inline double level_prior_conditioning(const std::vector<LatentBlock>& blocks,
                                       int k_grid, Rcpp::NumericVector& probe) {
    double out = 0.0;
    for (const auto& b : blocks) {
        if (!b.level_hard || b.hard_constraints.empty() || !b.log_prior) continue;
        const LinearConstraintRow& a = b.hard_constraints.front();
        double a1 = 0.0, aa = 0.0;
        for (std::size_t e = 0; e < a.cols.size(); e++) {
            a1 += a.weight(e);
            aa += a.weight(e) * a.weight(e);
        }
        std::vector<double> saved(static_cast<std::size_t>(b.size));
        for (int j = 0; j < b.size; j++) saved[j] = probe[b.start + j];
        auto set_level = [&](double v) {
            for (int j = 0; j < b.size; j++) probe[b.start + j] = v;
        };
        set_level(0.0);
        const double lp0 = b.log_prior(probe, k_grid);
        set_level(1.0);
        const double lp_up = b.log_prior(probe, k_grid);
        set_level(-1.0);
        const double lp_dn = b.log_prior(probe, k_grid);
        for (int j = 0; j < b.size; j++) probe[b.start + j] = saved[j];
        const double one_P_one = 2.0 * lp0 - lp_up - lp_dn;
        out += 0.5 * std::log(2.0 * M_PI * a1 * a1 / one_P_one) - 0.5 * std::log(aa);
    }
    return out;
}

// The constraint state a Newton loop carries: A, the Gram factor of A A', the
// columns A' e_g as right-hand sides, and, once the loop has solved them
// against its M, W = M^{-1} A' with the factor of C = A W.
struct HardConstraintState {
    int n_x = 0;
    std::vector<LinearConstraintRow> rows;
    InvBlockConstraint gram;   // W = A', capacitance A A'
    InvBlockConstraint cap;    // W = M^{-1} A', capacitance A M^{-1} A'
    std::vector<double> rhs;   // [g | A' e_1 | ... | A' e_r], each length n_x
    std::vector<double> sol;   // M^{-1} applied to rhs, same layout
    // The level rows' log-density conditioning at the cell being solved
    // (level_prior_conditioning), set by the driver before the solve.
    double prior_conditioning = 0.0;
    // True when every row is a level row. The unconstrained Gaussian of M is
    // then the augmented model's own posterior approximation (the pin carries
    // the field precision along the row), so the skewness and importance-curve
    // probes, which walk coordinate curves of M's factor, stay meaningful; a
    // structural row's A'A term is arbitrary and they do not.
    bool augmented_probes = false;

    bool active() const { return !rows.empty(); }
    int rank() const { return static_cast<int>(rows.size()); }

    // Rows must be linearly independent; a Gram matrix that does not factor
    // says they are not, and the block that declared them is at fault.
    void init(int nx, std::vector<LinearConstraintRow> r) {
        n_x = nx;
        rows = std::move(r);
        prior_conditioning = 0.0;
        augmented_probes = false;
        rhs.clear();
        sol.clear();
        if (rows.empty()) return;
        gram.set_rows(n_x, rows);
        gram.build([nx](const double* in, double* out) {
                       std::copy(in, in + nx, out);
                   }, n_x);
        if (!gram.usable) {
            Rcpp::stop("The hard linear constraints of the latent blocks are "
                       "linearly dependent (A A' is singular); each block has "
                       "to declare an independent set.");
        }
        const std::size_t n = static_cast<std::size_t>(n_x);
        rhs.assign(n * (rows.size() + 1), 0.0);
        sol.assign(rhs.size(), 0.0);
        for (std::size_t g = 0; g < rows.size(); g++) {
            const LinearConstraintRow& row = rows[g];
            for (std::size_t e = 0; e < row.cols.size(); e++)
                rhs[n * (g + 1) + row.cols[e]] += row.weight(e);
        }
    }

    // The rows every block of the list declares, and whether they are all
    // level rows.
    void init(int nx, const std::vector<LatentBlock>& blocks) {
        bool all_level = true;
        for (const auto& b : blocks) {
            const std::size_t n_level = b.level_hard ? 1 : 0;
            if (b.hard_constraints.size() > n_level) all_level = false;
        }
        init(nx, collect_hard_constraints(blocks));
        augmented_probes = all_level;
    }

    double log_det_gram() const { return gram.log_det_fold(); }

    // A v for the r rows.
    void apply_A(const double* v, std::vector<double>& out) const {
        out.resize(rows.size());
        for (std::size_t g = 0; g < rows.size(); g++)
            out[g] = constraint_row_dot(rows[g], v);
    }

    // x <- x - A' (A A')^{-1} A x, the nearest feasible point.
    void project_point(double* x) const {
        if (!active()) return;
        std::vector<double> b, y(rows.size());
        apply_A(x, b);
        gram.solve_cap(b.data(), y.data());
        for (std::size_t g = 0; g < rows.size(); g++) {
            const LinearConstraintRow& r = rows[g];
            for (std::size_t e = 0; e < r.cols.size(); e++)
                x[r.cols[e]] -= y[g] * r.weight(e);
        }
    }

    // max |P g| for P the orthogonal projector onto null(A): the score's
    // component along the constraint set, which is what vanishes at a
    // constrained mode (the rest is the multipliers' A' lambda).
    double max_abs_tangent(const double* g) const {
        std::vector<double> t(g, g + n_x);
        project_point(t.data());
        double m = 0.0;
        for (double v : t) m = std::max(m, std::abs(v));
        return m;
    }

    // Load g as the first right-hand side; the A' columns follow it.
    double* load_rhs(const double* g) {
        std::copy(g, g + n_x, rhs.begin());
        return rhs.data();
    }
    int n_rhs() const { return rank() + 1; }

    // W = M^{-1} A' from the solved right-hand sides `sol` (columns 1..r), and
    // the factor of C = A W.
    bool factor_cap_from_sol() {
        cap.set_rows(n_x, rows);
        cap.build_from_columns(sol.data() + n_x, n_x);
        return cap.usable;
    }

    // W = M^{-1} A' through a per-column solve with M, and the factor of C.
    template <typename SolveFn>
    bool factor_cap(SolveFn solve) {
        cap.build(solve, n_x, rows);
        return cap.usable;
    }

    // The KKT step from a step `delta` = M^{-1} g solved by the loop: W and C
    // through `solve` against the same factor, then the projection below.
    template <typename SolveFn>
    bool kkt_project(const double* x, double* delta, SolveFn solve) {
        if (!factor_cap(solve)) return false;
        project_step(x, delta);
        return true;
    }

    // delta <- delta - W C^{-1} A (x + delta).
    void project_step(const double* x, double* delta) const {
        std::vector<double> xd(static_cast<std::size_t>(n_x));
        for (int j = 0; j < n_x; j++) xd[j] = x[j] + delta[j];
        std::vector<double> b, y(rows.size());
        apply_A(xd.data(), b);
        cap.solve_cap(b.data(), y.data());
        for (std::size_t g = 0; g < rows.size(); g++) {
            const std::vector<double>& wg = cap.W[g];
            for (int j = 0; j < n_x; j++) delta[j] -= wg[j] * y[g];
        }
    }

    // log|B' M B| - log|M| = log|C| - log|A A'|.
    double log_det_subspace_correction() const {
        return cap.log_det_fold() - log_det_gram();
    }

    // The Laplace log-marginal of the constrained model at a feasible mode:
    // `log_det_sub` is log|B' M B|, the expansion runs over n_x - r
    // dimensions, and the level rows' conditioning turns the augmented
    // log-prior into the conditioned one.
    double log_marginal(double log_lik, double log_prior,
                        double log_det_sub) const {
        return finalize_log_marginal(log_lik, log_prior, log_det_sub,
                                     n_x - rank()) + prior_conditioning;
    }
};

}  // namespace tulpa

#endif  // TULPA_HARD_CONSTRAINTS_H
