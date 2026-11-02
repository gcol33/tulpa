// hard_constraints.h
// Hard linear constraints A x = 0 in the sparse joint inner Newton.
//
// A block whose intrinsic prior has null directions no fixed effect absorbs (a
// Knorr-Held interaction: one per time and one per area, aliased with the
// temporal and spatial main effects) declares them as rows of A
// (LatentBlock::hard_constraints). The model is the prior CONDITIONED on
// A x = 0, so the inner problem is an equality-constrained maximisation:
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
// M is the full precision: the stored Hessian plus the sum-to-zero pins the
// scatter registered on the side (PinnedHessianCopy), so its factor is of the
// matrix the Laplace approximation is built from. The block's prior may add any
// multiple of A'A to make M positive definite along the constrained directions:
// on the subspace it is zero, so neither the step nor log|B' M B| sees it.

#ifndef TULPA_HARD_CONSTRAINTS_H
#define TULPA_HARD_CONSTRAINTS_H

#include "inv_block_extract.h"
#include "latent_block.h"
#include "linear_constraint_rows.h"
#include "sparse_cholesky.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>
#include <utility>
#include <vector>

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

// The stored sparse Hessian with its side-registered sum-to-zero pins written
// in: M = H + U D U', U the pins' indicator columns and D their precision
// (diagonal coef, or the dense coupling). The pattern is H's plus the pins'
// dense blocks, built once per (H pattern, pin layout) and reused.
struct PinnedHessianCopy {
    SparseHessianBuilder M;
    std::vector<int> base_slot;          // H value index -> M value index
    unsigned long long base_generation = 0;
    std::vector<std::vector<int>> pin_layout;
    bool coupled = false;

    void build_pattern(const SparseHessianBuilder& H) {
        const auto& r1 = H.s2z_rank1;
        const int K = static_cast<int>(r1.size());
        std::vector<std::pair<int, int>> pat;
        pat.reserve(static_cast<std::size_t>(H.nnz));
        for (int c = 0; c < H.n; c++)
            for (int p = H.col_ptr[c]; p < H.col_ptr[c + 1]; p++)
                pat.emplace_back(H.row_idx[p], c);
        for (int k1 = 0; k1 < K; k1++)
            for (int k2 = 0; k2 < K; k2++) {
                if (k1 != k2 && !coupled) continue;
                for (int i = 0; i < r1[k1].n; i++)
                    for (int j = 0; j < r1[k2].n; j++) {
                        const int a = r1[k1].node(i), b = r1[k2].node(j);
                        if (a >= b) pat.emplace_back(a, b);
                    }
            }
        M.init(H.n, pat);
        base_slot.assign(static_cast<std::size_t>(H.nnz), -1);
        int t = 0;
        for (int c = 0; c < H.n; c++)
            for (int p = H.col_ptr[c]; p < H.col_ptr[c + 1]; p++, t++)
                base_slot[t] = M.csc_slot(H.row_idx[p], c);
        base_generation = H.pattern_generation;
        pin_layout.assign(K, {});
        for (int k = 0; k < K; k++) pin_layout[k] = s2z_block_nodes(r1[k]);
    }

    // M <- values of H (as stored, with whatever ridge it carries) + U D U'.
    void assemble(const SparseHessianBuilder& H) {
        const bool now_coupled = !H.s2z_coupling.empty();
        if (base_generation != H.pattern_generation ||
            !s2z_layout_matches(pin_layout, H.s2z_rank1) ||
            now_coupled != coupled) {
            coupled = now_coupled;
            build_pattern(H);
        }
        M.zero();
        for (int t = 0; t < H.nnz; t++) M.values[base_slot[t]] += H.values[t];
        const auto& r1 = H.s2z_rank1;
        const int K = static_cast<int>(r1.size());
        for (int k1 = 0; k1 < K; k1++)
            for (int k2 = 0; k2 < K; k2++) {
                const double d = coupled
                    ? H.s2z_coupling[static_cast<std::size_t>(k1) * K + k2]
                    : (k1 == k2 ? r1[k1].coef : 0.0);
                if (d == 0.0) continue;
                for (int i = 0; i < r1[k1].n; i++)
                    for (int j = 0; j < r1[k2].n; j++) {
                        const int a = r1[k1].node(i), b = r1[k2].node(j);
                        if (a >= b) M.values[M.csc_slot(a, b)] += d;
                    }
            }
    }
};

// The constraint state the Newton loop carries: A, the Gram factor of A A',
// and, after factor(), W = M^{-1} A' with the factor of C = A W.
struct HardConstraintState {
    int n_x = 0;
    std::vector<LinearConstraintRow> rows;
    InvBlockConstraint gram;   // W = A', capacitance A A'
    InvBlockConstraint cap;    // W = M^{-1} A', capacitance A M^{-1} A'
    PinnedHessianCopy pinned;
    SparseCholeskySolver solver;

    bool active() const { return !rows.empty(); }
    int rank() const { return static_cast<int>(rows.size()); }

    // Rows must be linearly independent; a Gram matrix that does not factor
    // says they are not, and the block that declared them is at fault.
    void init(int nx, std::vector<LinearConstraintRow> r) {
        n_x = nx;
        rows = std::move(r);
        if (rows.empty()) return;
        gram.build([nx](const double* rhs, double* out) {
                       std::copy(rhs, rhs + nx, out);
                   }, n_x, rows);
        if (!gram.usable) {
            Rcpp::stop("The hard linear constraints of the latent blocks are "
                       "linearly dependent (A A' is singular); each block has "
                       "to declare an independent set.");
        }
    }

    double log_det_gram() const { return gram.log_det_M(); }

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

    // W = M^{-1} A' and the factor of C = A W, against the solver's current
    // factor of M.
    bool factor_cap() {
        cap.build([this](const double* rhs, double* out) {
                      solver.solve(rhs, out, n_x);
                  }, n_x, rows);
        return cap.usable;
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
        return cap.log_det_M() - log_det_gram();
    }
};

}  // namespace tulpa

#endif  // TULPA_HARD_CONSTRAINTS_H
