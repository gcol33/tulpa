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
//   * each Newton step solves the bordered (KKT) system
//       [ M  A' ] [ delta  ]   [ grad   ]
//       [ A  0  ] [ lambda ] = [ -A x   ],
//     which keeps every iterate on the constraint set and converges to the
//     constrained mode;
//   * the Laplace approximation integrates over the (n_x - r)-dimensional
//     subspace B spanned by null(A) (B orthonormal), and log|B' M B| is the
//     log-determinant it needs.
//
// M is the full precision of the model (the stored Hessian with every
// sum-to-zero pin, written in or folded in by the solver). It need be positive
// definite on null(A) only. A level row's block carries curvature along the
// row (the precision augmentation of an intrinsic field: its prior adds a
// multiple of A'A, zero on the subspace), so M is positive definite as it
// stands and the system is solved through the factor of M:
//   delta = M^{-1} grad - W C^{-1} A (x + M^{-1} grad),  W = M^{-1} A',
//   C = A W,  log|B' M B| = log|M| + log|C| - log|A A'|.
// A STRUCTURAL row (a Knorr-Held interaction's null-space direction) carries
// none: the interaction's prior is tau R alone, and an A'A term spanning a
// whole time slice or a whole series would fill the factor of M. Each
// structural row is ANCHORED instead (InvBlockConstraint): one node of its
// support takes a weight c_j on M's diagonal, the loop factors
// M_a = M + E C E', which has M's sparsity, and the same system is solved
// through the anchored fold, with
//   log|B' M B| = log|M_a| + log|Cap| + sum_j log c_j - log|A A'|.
//
// A level-constrained block carries the AUGMENTED density as its log-prior, a
// proper Gaussian with precision P on the whole block. The density of the field
// conditioned on its r level rows A x = 0, with respect to Lebesgue measure on
// the subspace, is that density divided by the density of s = A x at zero and
// by |A A'|^(1/2):
//
//   log p(x | A x = 0) = log p_aug(x) + 0.5 log|2 pi A P^-1 A'| - 0.5 log|A A'|.
//
// The block's probes V (LatentBlock::level_probes) satisfy A' = P V C, so
// P^-1 A' = V C with C = (V' P V)^-1 V' A', and
//
//   A P^-1 A' = (A V) (V' P V)^-1 (A V)',
//
// where V' P V is read off the block's own log-prior along V. For one level
// row and the constant as probe this is (a'1)^2 / (1' P 1): the augmentation
// pins each component's sum at the field precision times that component's
// multiplier, so P 1 = tau a. For p coupled fields (MCAR) the probes are the
// per-field constants and A P^-1 A' is n times the folded fields' block of
// Sigma. That constant (level_prior_conditioning) turns the augmented density
// into the conditioned one in the log-marginal; it carries the field
// precision, so it moves with the outer grid.

#ifndef TULPA_HARD_CONSTRAINTS_H
#define TULPA_HARD_CONSTRAINTS_H

#include "inv_block_extract.h"
#include "latent_block.h"
#include "laplace_newton_loop.h"   // finalize_log_marginal
#include "linear_constraint_rows.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
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

// log|M| of a small symmetric positive-definite r x r row-major matrix, and
// optionally its lower factor; NaN when it does not factor.
inline double small_spd_log_det(const std::vector<double>& M, int r,
                                std::vector<double>* L_out = nullptr) {
    std::vector<double> L(static_cast<std::size_t>(r) * r, 0.0);
    if (!tulpa_linalg::chol_factor_lower<tulpa_linalg::TriLayout::RowMajor>(
            M.data(), L.data(), r, r, 0.0))
        return std::numeric_limits<double>::quiet_NaN();
    double ld = 0.0;
    for (int i = 0; i < r; i++) ld += 2.0 * std::log(L[static_cast<std::size_t>(i) * r + i]);
    if (L_out) *L_out = std::move(L);
    return ld;
}

// The log-density conditioning of every block's level rows at grid cell
// k_grid (see the header): per level-constrained block,
// 0.5 log|2 pi (A V) (V' P V)^-1 (A V)'| - 0.5 log|A A'|. V' P V is the
// polarized second difference of the block's own log-prior along the probes,
// exact for the zero-mean Gaussian prior it is: with q(x) = lp(0) - lp(x),
// (V'PV)_hh = 2 q(v_h) and (V'PV)_hk = q(v_h + v_k) - q(v_h) - q(v_k).
// `probe` is caller-owned scratch of at least the latent length; its contents
// are restored on return.
inline double level_prior_conditioning(const std::vector<LatentBlock>& blocks,
                                       int k_grid, Rcpp::NumericVector& probe) {
    double out = 0.0;
    for (const auto& b : blocks) {
        const int r = b.n_level_rows;
        const int m = static_cast<int>(b.level_probes.size());
        if (r <= 0 || m <= 0 || !b.log_prior) continue;
        std::vector<double> saved(static_cast<std::size_t>(b.size));
        for (int j = 0; j < b.size; j++) saved[j] = probe[b.start + j];
        auto clear = [&]() { for (int j = 0; j < b.size; j++) probe[b.start + j] = 0.0; };
        auto add = [&](const LinearConstraintRow& v) {
            for (std::size_t e = 0; e < v.cols.size(); e++) probe[v.cols[e]] += v.weight(e);
        };
        clear();
        const double lp0 = b.log_prior(probe, k_grid);
        std::vector<double> qh(static_cast<std::size_t>(m));
        for (int h = 0; h < m; h++) {
            clear(); add(b.level_probes[h]);
            qh[h] = lp0 - b.log_prior(probe, k_grid);
        }
        std::vector<double> G(static_cast<std::size_t>(m) * m, 0.0);
        for (int h = 0; h < m; h++) {
            G[static_cast<std::size_t>(h) * m + h] = 2.0 * qh[h];
            for (int k = 0; k < h; k++) {
                clear(); add(b.level_probes[h]); add(b.level_probes[k]);
                const double g = lp0 - b.log_prior(probe, k_grid) - qh[h] - qh[k];
                G[static_cast<std::size_t>(h) * m + k] = g;
                G[static_cast<std::size_t>(k) * m + h] = g;
            }
        }
        // (A V) and A A' over the dense block-local copies of the probes.
        std::vector<double> AV(static_cast<std::size_t>(r) * m, 0.0);
        std::vector<double> AA(static_cast<std::size_t>(r) * r, 0.0);
        for (int h = 0; h < m; h++) {
            clear(); add(b.level_probes[h]);
            for (int g = 0; g < r; g++)
                AV[static_cast<std::size_t>(g) * m + h] =
                    constraint_row_dot(b.hard_constraints[g], probe.begin());
        }
        for (int g = 0; g < r; g++) {
            clear(); add(b.hard_constraints[g]);
            for (int k = 0; k < r; k++)
                AA[static_cast<std::size_t>(k) * r + g] =
                    constraint_row_dot(b.hard_constraints[k], probe.begin());
        }
        for (int j = 0; j < b.size; j++) probe[b.start + j] = saved[j];

        // S = (A V) G^-1 (A V)' through the factor of G.
        std::vector<double> LG;
        if (!std::isfinite(small_spd_log_det(G, m, &LG)))
            return std::numeric_limits<double>::quiet_NaN();
        std::vector<double> Z(static_cast<std::size_t>(r) * m), y(m), z(m);
        for (int g = 0; g < r; g++) {
            tulpa_linalg::tri_solve_lower<tulpa_linalg::TriLayout::RowMajor>(
                LG.data(), m, m, AV.data() + static_cast<std::size_t>(g) * m, y.data());
            for (int h = 0; h < m; h++) Z[static_cast<std::size_t>(g) * m + h] = y[h];
        }
        std::vector<double> S(static_cast<std::size_t>(r) * r, 0.0);
        for (int g = 0; g < r; g++)
            for (int k = 0; k < r; k++) {
                double sgk = 0.0;
                for (int h = 0; h < m; h++)
                    sgk += Z[static_cast<std::size_t>(g) * m + h] *
                           Z[static_cast<std::size_t>(k) * m + h];
                S[static_cast<std::size_t>(g) * r + k] = sgk;
            }
        out += 0.5 * (r * std::log(2.0 * M_PI) + small_spd_log_det(S, r)) -
               0.5 * small_spd_log_det(AA, r);
    }
    return out;
}

// The constraint state a Newton loop carries: A, the Gram factor of A A', the
// anchors of the structural rows, the columns Vb = [A' E] as right-hand sides,
// and, once the loop has solved them against its M_a, W = M_a^{-1} Vb with the
// factored capacitance (`cap`).
//
// A loop that factors M_a loads the anchors onto the matrix it is about to
// factor (load_anchors), every time it factors: the weight is the diagonal
// entry there, so the anchor sits at the curvature scale of M. The fold
// refuses (factor_cap returns false) when anchors exist and were not loaded,
// since its W would then be solved against a matrix without them. Structural
// rows come only from a block with a sparse-only prior
// (constrained_structure_block.h), which routes the fit to the sparse joint
// Newton (blocks_have_sparse_only_prior); the dense loops see level rows only.
struct HardConstraintState {
    int n_x = 0;
    std::vector<LinearConstraintRow> rows;
    InvBlockConstraint gram;   // W = A', capacitance A A'
    InvBlockConstraint cap;    // W = M_a^{-1} Vb, the anchored fold
    std::vector<int> anchor_node;       // one per structural row that has one
    std::vector<double> anchor_weight;  // the weights last loaded
    bool anchors_loaded = false;
    std::vector<double> rhs;   // [g | Vb_1 | ... | Vb_m], each length n_x
    std::vector<double> sol;   // M_a^{-1} applied to rhs, same layout
    // The level rows' log-density conditioning at the cell being solved
    // (level_prior_conditioning), set by the driver before the solve.
    double prior_conditioning = 0.0;
    // True when every row is a level row. The unconstrained Gaussian of M is
    // then the augmented model's own posterior approximation (the pin carries
    // the field precision along the row), so the skewness and importance-curve
    // probes, which walk coordinate curves of M's factor, stay meaningful; a
    // structural row's anchor is arbitrary and they do not.
    bool augmented_probes = false;

    bool active() const { return !rows.empty(); }
    int rank() const { return static_cast<int>(rows.size()); }
    int n_anchor() const { return static_cast<int>(anchor_node.size()); }

    // Rows must be linearly independent; a Gram matrix that does not factor
    // says they are not, and the block that declared them is at fault.
    // `anchored[g]` marks the rows to anchor; each anchors at the first node
    // of its support with a nonzero weight that no earlier row took.
    void init(int nx, std::vector<LinearConstraintRow> r,
              const std::vector<char>& anchored = std::vector<char>()) {
        n_x = nx;
        rows = std::move(r);
        prior_conditioning = 0.0;
        augmented_probes = false;
        anchors_loaded = false;
        anchor_node.clear();
        anchor_weight.clear();
        rhs.clear();
        sol.clear();
        if (rows.empty()) return;
        gram.set_rows(n_x, rows);
        gram.build([nx](const double* in, double* out, int ncol) {
                       std::copy(in, in + static_cast<std::size_t>(nx) * ncol, out);
                   }, n_x);
        if (!gram.usable) {
            Rcpp::stop("The hard linear constraints of the latent blocks are "
                       "linearly dependent (A A' is singular); each block has "
                       "to declare an independent set.");
        }
        std::vector<char> taken(static_cast<std::size_t>(n_x), 0);
        for (std::size_t g = 0; g < rows.size() && g < anchored.size(); g++) {
            if (!anchored[g]) continue;
            const LinearConstraintRow& row = rows[g];
            for (std::size_t e = 0; e < row.cols.size(); e++) {
                const int j = row.cols[e];
                if (j < 0 || j >= n_x || taken[j] || row.weight(e) == 0.0) continue;
                taken[j] = 1;
                anchor_node.push_back(j);
                break;
            }
        }
        anchor_weight.assign(anchor_node.size(), 1.0);
        const std::size_t n = static_cast<std::size_t>(n_x);
        rhs.assign(n * (rows.size() + anchor_node.size() + 1), 0.0);
        sol.assign(rhs.size(), 0.0);
        for (std::size_t g = 0; g < rows.size(); g++) {
            const LinearConstraintRow& row = rows[g];
            for (std::size_t e = 0; e < row.cols.size(); e++)
                rhs[n * (g + 1) + row.cols[e]] += row.weight(e);
        }
        for (std::size_t a = 0; a < anchor_node.size(); a++)
            rhs[n * (rows.size() + a + 1) + anchor_node[a]] = 1.0;
    }

    // The rows every block of the list declares, whether they are all level
    // rows, and the structural ones anchored.
    void init(int nx, const std::vector<LatentBlock>& blocks) {
        bool all_level = true;
        std::vector<char> anchored;
        for (const auto& b : blocks) {
            const std::size_t n_level = static_cast<std::size_t>(b.n_level_rows);
            if (b.hard_constraints.size() > n_level) all_level = false;
            for (std::size_t g = 0; g < b.hard_constraints.size(); g++)
                anchored.push_back(g >= n_level ? 1 : 0);
        }
        init(nx, collect_hard_constraints(blocks), anchored);
        augmented_probes = all_level;
    }

    // Load the anchors onto the matrix about to be factored: `diag(j)` is a
    // reference to its diagonal entry j. The weight is that entry where it is
    // positive and finite, and 1 otherwise; any positive weight gives the same
    // constrained solution, the diagonal only puts it at M's scale.
    template <typename DiagRef>
    void load_anchors(DiagRef diag) {
        for (std::size_t a = 0; a < anchor_node.size(); a++) {
            double& h = diag(anchor_node[a]);
            const double c = (h > 0.0 && std::isfinite(h)) ? h : 1.0;
            anchor_weight[a] = c;
            h += c;
        }
        anchors_loaded = true;
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

    // Load g as the first right-hand side; the Vb columns follow it.
    double* load_rhs(const double* g) {
        std::copy(g, g + n_x, rhs.begin());
        return rhs.data();
    }
    int n_rhs() const { return rank() + n_anchor() + 1; }

    // W = M_a^{-1} Vb from the solved right-hand sides `sol` (columns 1..m),
    // and the factored capacitance.
    bool factor_cap_from_sol() {
        if (!take_anchors()) return false;
        cap.build_from_columns(sol.data() + n_x, n_x);
        return cap.usable;
    }

    // W = M_a^{-1} Vb through one block solve with M_a
    // (`solve(rhs, out, ncol)`), and the factored capacitance.
    template <typename SolveFn>
    bool factor_cap(SolveFn solve) {
        if (!take_anchors()) return false;
        cap.build(solve, n_x);
        return cap.usable;
    }

    // The KKT step from a step `delta` = M_a^{-1} g solved by the loop: W and
    // the capacitance through `solve` against the same factor, then the fold
    // below.
    template <typename SolveFn>
    bool kkt_project(const double* x, double* delta, SolveFn solve) {
        if (!factor_cap(solve)) return false;
        project_step(x, delta);
        return true;
    }

    // delta <- the KKT solution of the header from delta = M_a^{-1} g at the
    // iterate x: the fold with constraint right-hand side -A x.
    void project_step(const double* x, double* delta) const {
        std::vector<double> ax;
        apply_A(x, ax);
        cap.fold(delta, n_x, ax.data());
    }

    // log|B' M B| - log|M_a|.
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

private:
    // The rows and the anchors as loaded for this factorization into `cap`;
    // false when anchors exist and were not loaded onto the factored matrix.
    bool take_anchors() {
        cap.set_rows(n_x, rows);
        if (!anchor_node.empty()) {
            if (!anchors_loaded) return false;
            cap.set_anchors(anchor_node, anchor_weight);
        }
        anchors_loaded = false;
        return true;
    }
};

// The constraints the inner Gaussian of a cell is conditioned on, as a draw
// reads them: the fold whose W was solved against the factor the draw is taken
// through, the number of hard rows (the dimensions the expansion loses), what
// the fold adds to that factor's log-determinant to give log|B' M B| for the
// precision M the log-marginal was read off, and the level rows' log-density
// conditioning. A draw and its density read this one object, so the fold, the
// factor and the determinant cannot come from different matrices.
struct InnerGaussianConstraint {
    const InvBlockConstraint* fold = nullptr;
    int rank = 0;
    double log_det_shift = 0.0;
    double target_shift = 0.0;
    bool active() const { return fold != nullptr; }
};

// The hard rows alone, for a loop whose live factor is M_a itself (no
// sum-to-zero pin left off it): the state's own fold.
inline InnerGaussianConstraint inner_gaussian_constraint(
    const HardConstraintState* hard
) {
    InnerGaussianConstraint c;
    if (!hard || !hard->active()) return c;
    c.fold = &hard->cap;
    c.rank = hard->rank();
    c.log_det_shift = hard->log_det_subspace_correction();
    c.target_shift = hard->prior_conditioning;
    return c;
}

}  // namespace tulpa

#endif  // TULPA_HARD_CONSTRAINTS_H
