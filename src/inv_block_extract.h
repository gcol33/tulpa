// inv_block_extract.h
// Selected blocks of the inverse of a factorized precision, read through a
// solve oracle.
//
// Every consumer wants the same thing -- a small diagonal block of H^{-1},
// with the rest of the latent field marginalized out -- and reaches it the
// same way: solve the block's unit columns against a factor that already
// exists and read the block rows back. What varies is WHERE the factor lives
// (the live Newton factor, or a fresh factorization of one stored cell
// precision) and whether a sum-to-zero constraint is imposed on top. Both
// enter here as parameters, so the algebra is written once:
//
//   * `laplace_newton.h`'s `inv_block_layout` extraction (the per-(term,group)
//     random-effect covariance blocks) solves against the LIVE factor it just
//     built for the log-determinant, unconstrained.
//   * `joint_inner_vcov.cpp` extracts the joint tier's per-cell fixed-effect
//     block under the field sum-to-zero constraint, so the covariance it
//     reports is the one the fit's own posterior draws are generated from.
//
// Solve oracle contract: `solve(const double* rhs, double* out)` writes
// H^{-1} rhs into `out`; both buffers have length n_x. The oracle owns whatever
// factor it reads (sparse CHOLMOD factor, dense back-substitution) and is called
// once per extracted column -- never a refactorization.

#ifndef TULPA_INV_BLOCK_EXTRACT_H
#define TULPA_INV_BLOCK_EXTRACT_H

#include "linear_constraint_rows.h"
#include "small_dense_sym.h"  // SmallSymEig
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace tulpa {

// How a solved block is made symmetric before it is stored.
//   Average     -- 0.5 * (B + B'), reading both solved triangles.
//   MirrorLower -- take the lower triangle and mirror it, reading each entry
//                  from one solved column only.
// The two differ at round-off, so each consumer keeps the convention its
// reference output was produced with.
enum class InvBlockSymmetry { Average, MirrorLower };

// Linear-constraint correction for a factorized precision H.
//
// With A the kc x n_x constraint matrix (row g a LinearConstraintRow: a
// sum-to-zero group's indicator, or any weighted row), conditioning by kriging
// gives
//
//   V = H^{-1} - H^{-1} A' (A H^{-1} A')^{-1} A H^{-1},
//
// and a SOFT rank-kc penalty -- a sum-to-zero pin the scatter left off the
// stored H, D the pins' kc x kc precision -- has the Woodbury inverse
//
//   (H + A' D A)^{-1} = H^{-1} - H^{-1} A' (D^{-1} + A H^{-1} A')^{-1} A H^{-1},
//
// of which the hard constraint is the D^{-1} = 0 limit.
//
// Read off H itself, both lose digits wherever H is near singular along a
// direction only the constraint identifies. A folded pin leaves an intrinsic
// field's constant direction to the uniform Laplace ridge, and two intrinsic
// fields loading the same rows (an additive space + time model) leave their
// opposite shift, +c on one field and -c on the other, with no curvature but
// the ridge. H^{-1} is then of order 1 / ridge along it, A H^{-1} A' is that
// large in one direction, and the O(1) part the answer is made of is the
// difference of such terms. The fold is therefore taken against an ANCHORED
// precision
//
//   H_a = H + E C E',
//
// E the unit columns of one anchor node per row and C = diag(c) their
// weights, each the anchor's own diagonal entry H_jj. The directions at issue
// lie in the span of the rows' supports (constant on a group's nodes, or a
// null-space direction of a structure), and each row anchors at a node of its
// own support that no earlier row took, so the anchors identify what the rows
// do, at the curvature scale of H itself. The target is
// H_a plus a symmetric indefinite term of rank m = kc + (number of anchors),
//
//   H + A' D A = H_a + Vb Psi Vb',   Vb = [A' E],   Psi = diag(D, -C),
//
// folded by the same identity with the m x m capacitance
//
//   Cap = Psi^{-1} + Vb' H_a^{-1} Vb,   Psi^{-1} = diag(D^{-1}, -C^{-1}),
//
// D^{-1} = 0 for the hard constraint, now formed from a well-conditioned
// solve. Cap is symmetric indefinite and is factored by its eigendecomposition.
// Haynsworth inertia additivity on [[H_a, Vb], [Vb', -Psi^{-1}]] gives
// neg(H + A' D A) = neg(H_a) + pos(Cap) - kc, so the target is positive
// definite exactly when neg(Cap) = (number of anchors) + neg(H_a); with H_a
// positive definite, when Cap has kc positive eigenvalues. The determinant
// lemma gives |H + A' D A| = |H_a| |D| prod(c) |Cap| in absolute value.
//
// Use: anchor() takes the rows and picks the anchors and their weights; the
// caller factors H_a (H with each anchor weight added to its diagonal entry)
// and hands build() the solve against that factor. Every unconstrained solve
// the caller then makes is against H_a, and correction / quad_correction
// subtract the matching term, so the pair gives V. set_rows() takes the rows
// with no anchors, for a precision that needs none (H_a = H): Cap is then
// D^{-1} + A H^{-1} A' itself, which log_det_fold() and solve_cap() read.
struct InvBlockConstraint {
    int kc = 0;                              // constraint rows
    std::vector<LinearConstraintRow> rows;   // in-range entries of each row
    std::vector<int> anchor_node;            // one per anchored row
    std::vector<double> anchor_weight;       // H_jj at the anchor
    std::vector<std::vector<double>> W;      // m columns of H_a^{-1} Vb, length n_x
    SmallSymEig cap;                         // eigendecomposition of Cap
    bool usable = false;                     // false when the target is not PD

    // Take the rows with no anchors. Entries outside [0, n_x) are dropped; the
    // rest keep the caller's order, so the right-hand side and the Gram entries
    // sum in it (an indicator row built by indicator_row() already holds each
    // index once).
    void set_rows(int n_x, const std::vector<LinearConstraintRow>& A) {
        kc = static_cast<int>(A.size());
        usable = false;
        rows.assign(kc > 0 ? kc : 0, LinearConstraintRow());
        anchor_node.clear();
        anchor_weight.clear();
        W.clear();
        for (int g = 0; g < kc; g++) {
            const LinearConstraintRow& a = A[g];
            LinearConstraintRow& r = rows[g];
            r.cols.reserve(a.cols.size());
            if (!a.w.empty()) r.w.reserve(a.cols.size());
            for (std::size_t e = 0; e < a.cols.size(); e++) {
                const int latent = a.cols[e];
                if (latent < 0 || latent >= n_x) continue;
                r.cols.push_back(latent);
                if (!a.w.empty()) r.w.push_back(a.w[e]);
            }
        }
    }

    // Take the rows and choose the anchors. `diag(j)` returns H_jj, or a value
    // that is not positive and finite where node j has no usable diagonal
    // entry. A row anchors at its first node with a nonzero weight and a
    // positive diagonal that no earlier row anchors at; a row with none carries
    // no anchor.
    template <typename DiagFn>
    void anchor(int n_x, const std::vector<LinearConstraintRow>& A,
                DiagFn diag) {
        set_rows(n_x, A);
        std::vector<char> anchored(n_x > 0 ? n_x : 1, 0);
        for (int g = 0; g < kc; g++) {
            const LinearConstraintRow& r = rows[g];
            for (std::size_t e = 0; e < r.cols.size(); e++) {
                const int latent = r.cols[e];
                if (anchored[latent] || r.weight(e) == 0.0) continue;
                const double c = diag(latent);
                if (!(c > 0.0) || !std::isfinite(c)) continue;
                anchored[latent] = 1;
                anchor_node.push_back(latent);
                anchor_weight.push_back(c);
                break;
            }
        }
    }

    // Solve W = H_a^{-1} Vb and factor Cap. `solve` is the solve against the
    // ANCHORED precision. `Dinv`, when given, is the row-major kc x kc inverse
    // penalty precision (the soft form); nullptr imposes the hard constraint.
    // `neg_base` is the number of negative eigenvalues of H_a (0 when its
    // Cholesky factor exists; an LDL' factor's count otherwise). A target that
    // is not positive definite leaves `usable = false`, and the caller then
    // skips the correction rather than subtracting a wrong one.
    template <typename SolveFn>
    void build(SolveFn solve, int n_x, const std::vector<double>* Dinv = nullptr,
               int neg_base = 0) {
        usable = false;
        W.clear();
        if (kc <= 0) return;
        const int na = static_cast<int>(anchor_node.size());
        const int m = kc + na;

        std::vector<double> e(n_x, 0.0), v(n_x, 0.0);
        W.resize(m);
        for (int i = 0; i < m; i++) {
            std::fill(e.begin(), e.end(), 0.0);
            if (i < kc) {
                const LinearConstraintRow& r = rows[i];
                for (std::size_t t = 0; t < r.cols.size(); t++)
                    e[r.cols[t]] += r.weight(t);
            } else {
                e[anchor_node[i - kc]] = 1.0;
            }
            solve(e.data(), v.data());
            W[i] = v;
        }
        factor_cap(Dinv, neg_base);
    }

    // The same build from the rows' columns W = H^{-1} A' already solved, for
    // rows taken by set_rows() (no anchors): `cols` holds the kc columns back
    // to back, each of length n_x, in row order.
    void build_from_columns(const double* cols, int n_x,
                            const std::vector<double>* Dinv = nullptr) {
        usable = false;
        W.clear();
        if (kc <= 0 || !anchor_node.empty()) return;
        W.resize(kc);
        for (int g = 0; g < kc; g++)
            W[g].assign(cols + static_cast<std::size_t>(g) * n_x,
                        cols + static_cast<std::size_t>(g + 1) * n_x);
        factor_cap(Dinv, 0);
    }

    // Cap = Psi^{-1} + Vb' W from the solved W, and its eigendecomposition.
    void factor_cap(const std::vector<double>* Dinv, int neg_base) {
        const int na = static_cast<int>(anchor_node.size());
        const int m = kc + na;

        // Vb_i' w for column i of Vb.
        auto vb_dot = [&](int i, const std::vector<double>& w) {
            if (i >= kc) return w[anchor_node[i - kc]];
            return constraint_row_dot(rows[i], w.data());
        };
        std::vector<double> C(static_cast<std::size_t>(m) * m, 0.0);
        for (int i = 0; i < m; i++)
            for (int j = 0; j <= i; j++) {
                const double s = 0.5 * (vb_dot(i, W[j]) + vb_dot(j, W[i]));
                C[static_cast<std::size_t>(i) * m + j] = s;
                C[static_cast<std::size_t>(j) * m + i] = s;
            }
        if (Dinv) {
            if (Dinv->size() != static_cast<std::size_t>(kc) * kc) {
                W.clear();
                return;
            }
            for (int i = 0; i < kc; i++)
                for (int j = 0; j < kc; j++)
                    C[static_cast<std::size_t>(i) * m + j] +=
                        (*Dinv)[static_cast<std::size_t>(i) * kc + j];
        }
        for (int a = 0; a < na; a++)
            C[static_cast<std::size_t>(kc + a) * m + (kc + a)] -=
                1.0 / anchor_weight[a];

        cap.factor(C, m);
        usable = cap.ok && cap.n_neg() == na + neg_base;
        if (!usable) W.clear();
    }

    // log|Cap| + sum_a log c_a: what the fold adds to log|H_a| + log|D| to give
    // the log-determinant of the target; NaN when the target is not PD.
    double log_det_fold() const {
        if (!usable) return std::numeric_limits<double>::quiet_NaN();
        double ld = cap.logabsdet();
        for (double c : anchor_weight) ld += std::log(c);
        return ld;
    }

    // out = (H + A' D A)^{-1} y given ainv_y = H_a^{-1} y, both of length n_x:
    // ainv_y - W Cap^{-1} (W' y).
    void apply(const double* y, const double* ainv_y, double* out,
               int n_x) const {
        const int m = static_cast<int>(W.size());
        std::vector<double> wty(m, 0.0), z(m, 0.0);
        for (int i = 0; i < m; i++) {
            double s = 0.0;
            for (int t = 0; t < n_x; t++) s += W[i][t] * y[t];
            wty[i] = s;
        }
        if (usable && m > 0) cap.solve(wty.data(), z.data());
        for (int t = 0; t < n_x; t++) {
            double corr = 0.0;
            for (int i = 0; i < m; i++) corr += W[i][t] * z[i];
            out[t] = ainv_y[t] - corr;
        }
    }

    // Node list of column i of Vb: row i's indices for i < kc, the anchor
    // node otherwise.
    template <typename Fn>
    void for_each_vb_node(int i, Fn fn) const {
        if (i < kc) { for (int latent : rows[i].cols) fn(latent); }
        else fn(anchor_node[i - kc]);
    }

    // y = Cap^{-1} b for an m-vector b (zeros when the target is not PD).
    void solve_cap(const double* b, double* y) const {
        const int m = kc + static_cast<int>(anchor_node.size());
        if (!usable) { std::fill(y, y + m, 0.0); return; }
        cap.solve(b, y);
    }

    // y = Cap^{-1} G_latent, with G_latent[i] = W[i][latent]. `y` is resized to
    // the capacitance's order.
    void solve_y(int latent, std::vector<double>& y) const {
        const int m = static_cast<int>(W.size());
        y.assign(m, 0.0);
        if (!usable) return;
        std::vector<double> g_a(m);
        for (int i = 0; i < m; i++) g_a[i] = W[i][latent];
        cap.solve(g_a.data(), y.data());
    }

    // G_latent' y for a y produced by solve_y at the other index.
    double correction(int latent, const std::vector<double>& y) const {
        if (!usable) return 0.0;
        double corr = 0.0;
        for (std::size_t i = 0; i < W.size(); i++) corr += W[i][latent] * y[i];
        return corr;
    }

    // The correction to the quadratic form a' H_a^{-1} a for a sparse vector
    // a = sum_s w[s] e_{idx[s]} over [first, last): G' Cap^{-1} G with
    // G_i = sum_s w[s] W[i][idx[s]].
    double quad_correction(const int* idx, const double* w,
                           std::size_t first, std::size_t last) const {
        if (!usable || W.empty()) return 0.0;
        const int m = static_cast<int>(W.size());
        std::vector<double> G(m, 0.0), y(m);
        for (int i = 0; i < m; i++) {
            double s = 0.0;
            for (std::size_t e = first; e < last; e++) s += w[e] * W[i][idx[e]];
            G[i] = s;
        }
        cap.solve(G.data(), y.data());
        double corr = 0.0;
        for (int i = 0; i < m; i++) corr += G[i] * y[i];
        return corr;
    }
};

// Diagonal blocks of H^{-1} (or of the constrained V when `constr` is a built,
// usable correction) for the requested (offset, size) ranges. With `constr`,
// `solve` is the solve against the anchored precision it was built on.
//
// Each block costs one solve per column and no refactorization. Blocks are
// appended to `flat_out` column-major, with the side length appended to
// `sizes_out`, so a caller reads block b at the running offset sum of
// sizes_out[0..b-1] squared -- the LaplaceResult re_cov_flat contract.
template <typename SolveFn>
void extract_inv_diag_blocks(SolveFn solve, int n_x,
                             const std::vector<std::pair<int, int>>& layout,
                             const InvBlockConstraint* constr,
                             InvBlockSymmetry sym,
                             std::vector<double>& flat_out,
                             std::vector<int>& sizes_out) {
    if (layout.empty()) return;
    const bool have_corr = constr && constr->usable && constr->kc > 0;

    std::vector<double> rhs(n_x, 0.0), col(n_x, 0.0);

    for (const auto& blk : layout) {
        const int s = blk.first, m = blk.second;
        if (m <= 0 || s < 0 || s + m > n_x) continue;
        std::vector<double> block(static_cast<std::size_t>(m) * m, 0.0);

        for (int c = 0; c < m; c++) {
            std::fill(rhs.begin(), rhs.end(), 0.0);
            rhs[s + c] = 1.0;
            solve(rhs.data(), col.data());
            for (int r = 0; r < m; r++) block[r * m + c] = col[s + r];
        }

        // M^{-1} G_a once per block index, not once per (row, column) pair.
        std::vector<std::vector<double>> yv;
        if (have_corr) {
            yv.resize(m);
            for (int a = 0; a < m; a++) constr->solve_y(s + a, yv[a]);
        }
        auto corr_at = [&](int r, int c) -> double {
            return have_corr ? constr->correction(s + r, yv[c]) : 0.0;
        };

        if (sym == InvBlockSymmetry::Average) {
            for (int cc = 0; cc < m; cc++) {
                for (int r = 0; r < m; r++) {
                    flat_out.push_back(
                        0.5 * (block[r * m + cc] + block[cc * m + r])
                        - corr_at(r, cc));
                }
            }
        } else {
            // Lower triangle from the solved columns, mirrored.
            std::vector<double> out(static_cast<std::size_t>(m) * m, 0.0);
            for (int a = 0; a < m; a++) {
                for (int b = 0; b <= a; b++) {
                    const double val = block[a * m + b] - corr_at(a, b);
                    out[static_cast<std::size_t>(a) + static_cast<std::size_t>(b) * m] = val;
                    out[static_cast<std::size_t>(b) + static_cast<std::size_t>(a) * m] = val;
                }
            }
            flat_out.insert(flat_out.end(), out.begin(), out.end());
        }
        sizes_out.push_back(m);
    }
}

}  // namespace tulpa

#endif  // TULPA_INV_BLOCK_EXTRACT_H
