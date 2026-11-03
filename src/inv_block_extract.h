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
// Solve oracle contract: `solve(const double* rhs, double* out, int ncol)`
// writes H^{-1} rhs into `out`, both column-major [n_x x ncol]. The oracle owns
// whatever factor it reads (sparse CHOLMOD factor, dense back-substitution) and
// is called once per block of columns -- never a refactorization -- so the
// columns of a block reach the factor in one pass.

#ifndef TULPA_INV_BLOCK_EXTRACT_H
#define TULPA_INV_BLOCK_EXTRACT_H

#include "linear_constraint_rows.h"
#include "small_dense_sym.h"  // SmallChol, SmallSymEig
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
// solve. Haynsworth inertia additivity on [[H_a, Vb], [Vb', -Psi^{-1}]] gives
// neg(H + A' D A) = neg(H_a) + pos(Cap) - kc, so the target is positive
// definite exactly when neg(Cap) = (number of anchors) + neg(H_a); with H_a
// positive definite, when Cap has kc positive eigenvalues. The determinant
// lemma gives |H + A' D A| = |H_a| |D| prod(c) |Cap| in absolute value.
//
// Over a positive definite H_a that inertia is a block sign pattern. With
// W = H_a^{-1} Vb split as [W_A W_E],
//
//   Cap = [ P   Q ],   P = D^{-1} + A W_A,  Q = A W_E,  R = E' W_E - C^{-1},
//         [ Q'  R ]
//
// (D^{-1} = 0 on the hard rows). P is positive definite (A has full row rank,
// D^{-1} is positive semi-definite), and the Schur complement
// R - Q' P^{-1} Q = E' V_A E - C^{-1}, V_A the inverse of H_a + A' D A taken
// to the hard limit on its hard rows, is negative definite exactly when the
// target is positive definite (on null of the hard rows, B a basis of it):
// B' (H + A' D A) B = B' (H_a + A' D A) B - B' E C E' B, and by Woodbury its
// inverse exists and is positive definite iff C^{-1} - E' V_A E is. So the
// fold is factored as two Cholesky factors, of P and of N = Q' P^{-1} Q - R,
// and it is usable when both exist; log|Cap| = log|P| + log|N|. An
// indefinite H_a (an LDL' factor, `neg_base` > 0) has no such sign pattern
// and is factored by the eigendecomposition of Cap.
//
// For the hard constraint the fold is the solution operator of the KKT system
//
//   [ H  A' ] [ x      ]   [ y ]
//   [ A  0  ] [ lambda ] = [ b ],
//
// x = H_a^{-1} y - W Cap^{-1} (W' y - [b; 0]): fold() applies it to a column
// already solved against H_a (b = -A x0 moves an infeasible iterate x0 back on
// the constraint set), and with b = 0 it is V = B (B' H B)^{-1} B', the
// constrained covariance. H itself is never factored, so it need not be
// positive definite along a direction only the constraint identifies, and
// the factored matrix H_a has the sparsity of H: the anchors sit on its
// diagonal.
//
// Use: anchor() takes the rows and picks the anchors and their weights; the
// caller factors H_a (H with each anchor weight added to its diagonal entry)
// and hands build() the solve against that factor. Every unconstrained solve
// the caller then makes is against H_a, and correction / quad_correction
// subtract the matching term, so the pair gives V. set_rows() takes the rows
// with no anchors, for a precision that needs none (H_a = H): Cap is then
// D^{-1} + A H^{-1} A' itself, which log_det_fold() and solve_cap() read.
// set_anchors() places anchors chosen elsewhere (a Newton loop that loads them
// onto the matrix it factors).
struct InvBlockConstraint {
    int kc = 0;                              // constraint rows
    std::vector<LinearConstraintRow> rows;   // in-range entries of each row
    std::vector<int> anchor_node;            // one per anchored row
    std::vector<double> anchor_weight;       // H_jj at the anchor
    std::vector<std::vector<double>> W;      // m columns of H_a^{-1} Vb, length n_x
    SmallSymEig cap;                         // eigendecomposition of Cap (eigen form)
    bool block_form = false;                 // Cap factored as P, N
    bool soft = false;                       // built with a finite D
    std::vector<double> soft_Dinv;           // that D^{-1}, kc x kc row-major
    SmallChol cap_P, cap_N;                  // factors of P and N (block form)
    std::vector<double> cap_Q, cap_Z;        // Q and P^{-1} Q, kc x na row-major
    bool usable = false;                     // false when the target is not PD

    int n_anchor() const { return static_cast<int>(anchor_node.size()); }
    int order() const { return kc + n_anchor(); }

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

    // Anchors chosen by the caller, after set_rows(): node j carries weight
    // c_j, which the caller has added to the diagonal of the matrix it factors.
    void set_anchors(const std::vector<int>& nodes,
                     const std::vector<double>& weights) {
        anchor_node = nodes;
        anchor_weight = weights;
    }

    // Solve W = H_a^{-1} Vb and factor Cap. `solve` is the solve against the
    // ANCHORED precision. `Dinv`, when given, is the row-major kc x kc inverse
    // penalty precision (the soft form); nullptr imposes the hard constraint.
    // `neg_base` is the number of negative eigenvalues of H_a (0 when its
    // Cholesky factor exists; an LDL' factor's count otherwise). A target that
    // is not positive definite leaves `usable = false`, and the caller then
    // skips the correction rather than subtracting a wrong one.
    // `solve(rhs, out, ncol)` solves the m columns of Vb in one call.
    template <typename SolveCols>
    void build(SolveCols solve, int n_x, const std::vector<double>* Dinv = nullptr,
               int neg_base = 0) {
        usable = false;
        W.clear();
        if (kc <= 0) return;
        std::vector<double> rhs;
        vb_columns(n_x, rhs);
        std::vector<double> sol(rhs.size(), 0.0);
        solve(rhs.data(), sol.data(), order());
        build_from_columns(sol.data(), n_x, Dinv, neg_base);
    }

    // Vb = [A' E] as m = kc + (number of anchors) columns back to back, each of
    // length n_x, rows first and anchors after, in order.
    void vb_columns(int n_x, std::vector<double>& out) const {
        const int m = order();
        const std::size_t n = static_cast<std::size_t>(n_x);
        out.assign(n * static_cast<std::size_t>(m > 0 ? m : 0), 0.0);
        for (int i = 0; i < m; i++) {
            double* e = out.data() + n * static_cast<std::size_t>(i);
            if (i < kc) {
                const LinearConstraintRow& r = rows[i];
                for (std::size_t t = 0; t < r.cols.size(); t++)
                    e[r.cols[t]] += r.weight(t);
            } else {
                e[anchor_node[i - kc]] = 1.0;
            }
        }
    }

    // The same build from the columns W = H_a^{-1} Vb already solved: `cols`
    // holds them in vb_columns()'s layout.
    void build_from_columns(const double* cols, int n_x,
                            const std::vector<double>* Dinv = nullptr,
                            int neg_base = 0) {
        usable = false;
        W.clear();
        if (kc <= 0) return;
        const int m = order();
        W.resize(m);
        for (int i = 0; i < m; i++)
            W[i].assign(cols + static_cast<std::size_t>(i) * n_x,
                        cols + static_cast<std::size_t>(i + 1) * n_x);
        factor_cap(Dinv, neg_base);
    }

    // Vb_i' w for column i of Vb.
    double vb_dot(int i, const double* w) const {
        if (i >= kc) return w[anchor_node[i - kc]];
        return constraint_row_dot(rows[i], w);
    }

    // Cap = Psi^{-1} + Vb' W from the solved W, and its factorization.
    void factor_cap(const std::vector<double>* Dinv, int neg_base) {
        const int na = n_anchor();
        const int m = kc + na;
        block_form = false;
        soft = (Dinv != nullptr);
        soft_Dinv.clear();

        std::vector<double> C(static_cast<std::size_t>(m) * m, 0.0);
        for (int i = 0; i < m; i++)
            for (int j = 0; j <= i; j++) {
                const double s = 0.5 * (vb_dot(i, W[j].data()) + vb_dot(j, W[i].data()));
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
            soft_Dinv = *Dinv;
        }
        for (int a = 0; a < na; a++)
            C[static_cast<std::size_t>(kc + a) * m + (kc + a)] -=
                1.0 / anchor_weight[a];

        if (neg_base == 0) {
            block_form = true;
            usable = factor_block(C, m);
        } else {
            cap.factor(C, m);
            usable = cap.ok && cap.n_neg() == na + neg_base;
        }
        if (!usable) W.clear();
    }

    // P, Q, R from the row-major m x m Cap, then the factors of P and of N.
    bool factor_block(const std::vector<double>& C, int m) {
        const int na = m - kc;
        std::vector<double> P(static_cast<std::size_t>(kc) * kc);
        for (int i = 0; i < kc; i++)
            for (int j = 0; j < kc; j++)
                P[static_cast<std::size_t>(i) * kc + j] = C[static_cast<std::size_t>(i) * m + j];
        cap_P.factor(P, kc);
        if (!cap_P.ok) return false;
        cap_Q.assign(static_cast<std::size_t>(kc) * na, 0.0);
        cap_Z.assign(static_cast<std::size_t>(kc) * na, 0.0);
        if (na == 0) return true;
        std::vector<double> q(kc), z(kc);
        for (int a = 0; a < na; a++) {
            for (int i = 0; i < kc; i++) q[i] = C[static_cast<std::size_t>(i) * m + kc + a];
            cap_P.solve(q.data(), z.data());
            for (int i = 0; i < kc; i++) {
                cap_Q[static_cast<std::size_t>(i) * na + a] = q[i];
                cap_Z[static_cast<std::size_t>(i) * na + a] = z[i];
            }
        }
        std::vector<double> N(static_cast<std::size_t>(na) * na);
        for (int a = 0; a < na; a++)
            for (int b = 0; b <= a; b++) {
                double s = -C[static_cast<std::size_t>(kc + a) * m + kc + b];
                for (int i = 0; i < kc; i++)
                    s += cap_Q[static_cast<std::size_t>(i) * na + a] *
                         cap_Z[static_cast<std::size_t>(i) * na + b];
                N[static_cast<std::size_t>(a) * na + b] = s;
                N[static_cast<std::size_t>(b) * na + a] = s;
            }
        cap_N.factor(N, na);
        return cap_N.ok;
    }

    // y = Cap^{-1} b for the factored Cap.
    void cap_solve(const double* b, double* y) const {
        if (!block_form) { cap.solve(b, y); return; }
        const int na = n_anchor();
        std::vector<double> t(kc);
        cap_P.solve(b, t.data());
        if (na == 0) { std::copy(t.begin(), t.end(), y); return; }
        // [P Q; Q' R] [u; v] = [f; g]:  N v = Q' P^{-1} f - g,  u = P^{-1} f - Z v.
        std::vector<double> r(na), v(na);
        for (int a = 0; a < na; a++) {
            double s = -b[kc + a];
            for (int i = 0; i < kc; i++) s += cap_Q[static_cast<std::size_t>(i) * na + a] * t[i];
            r[a] = s;
        }
        cap_N.solve(r.data(), v.data());
        for (int i = 0; i < kc; i++) {
            double s = t[i];
            for (int a = 0; a < na; a++) s -= cap_Z[static_cast<std::size_t>(i) * na + a] * v[a];
            y[i] = s;
        }
        for (int a = 0; a < na; a++) y[kc + a] = v[a];
    }

    // log|Cap| + sum_a log c_a: what the fold adds to log|H_a| + log|D| to give
    // the log-determinant of the target; NaN when the target is not PD.
    double log_det_fold() const {
        if (!usable) return std::numeric_limits<double>::quiet_NaN();
        double ld = block_form
            ? cap_P.logdet() + (n_anchor() > 0 ? cap_N.logdet() : 0.0)
            : cap.logabsdet();
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
        if (usable && m > 0) cap_solve(wty.data(), z.data());
        for (int t = 0; t < n_x; t++) {
            double corr = 0.0;
            for (int i = 0; i < m; i++) corr += W[i][t] * z[i];
            out[t] = ainv_y[t] - corr;
        }
    }

    // In place, col <- col - W Cap^{-1} b with b_i = Vb_i' col, plus `shift_A`
    // (length kc) on the row entries when given: for col = H_a^{-1} y this is
    // the KKT solution x of the header with constraint right-hand side
    // -shift_A. Returns false when the fold is not usable.
    bool fold(double* col, int n_x, const double* shift_A = nullptr) const {
        if (!usable) return false;
        const int m = static_cast<int>(W.size());
        if (m == 0) return true;
        std::vector<double> b(m), y(m);
        for (int i = 0; i < m; i++)
            b[i] = vb_dot(i, col) + (shift_A && i < kc ? shift_A[i] : 0.0);
        cap_solve(b.data(), y.data());
        for (int i = 0; i < m; i++) {
            const std::vector<double>& wi = W[i];
            for (int t = 0; t < n_x; t++) col[t] -= wi[t] * y[i];
        }
        return true;
    }

    // The map from a draw z ~ N(0, H_a^{-1}) to a draw of the constrained
    // Gaussian N(0, V), V = B (B' M B)^{-1} B', M = H + A_s' D A_s the target
    // with its soft rows A_s (penalty precision D) and B a basis of null of the
    // hard rows. x = V e has covariance V M V = V for any e ~ N(0, M), and with
    // M = H_a + Vj Psi Vj', Vj = [A_s' E] the soft rows' and the anchors'
    // columns, Psi = diag(D, -C),
    //
    //   e = H_a z + Vj Gam Vj' z,   2 Gam + Gam S Gam = Psi,   S = Vj' H_a^{-1} Vj,
    //
    // has covariance H_a + Vj (2 Gam + Gam S Gam) Vj' = M. With S = L L' and
    // K = L' Psi L = Q diag(lambda) Q', Gam = L^{-T} Q diag(g) Q' L^{-1},
    // g = lambda / (1 + sqrt(1 + lambda)), the root that stays bounded as
    // lambda -> 0. lambda >= -1 is M >= 0 (I + L' Psi L is congruent to the
    // Schur complement of [[H_a, Vj], [Vj', -Psi^{-1}]] that M is), so a target
    // indefinite off the constraint set has no such e and the transform
    // reports `ok = false`. Without soft rows and anchors Gam is empty and x is
    // the conditioning by kriging z - W Cap^{-1} A z.
    struct DrawTransform {
        std::vector<int> cols;      // Vb columns of Vj: soft rows, then anchors
        std::vector<double> Gam;    // nj x nj row-major
        std::vector<double> Psi;    // nj x nj row-major
        bool ok = false;
    };

    // Rows [0, n_soft) carry a finite penalty and the rest none: the leading
    // block of the Dinv the fold was built with is positive definite, every
    // other entry of it is zero. -1 when the rows do not split that way.
    int n_soft_rows() const {
        if (!soft) return 0;
        int ns = 0;
        while (ns < kc && soft_Dinv[static_cast<std::size_t>(ns) * kc + ns] > 0.0) ns++;
        for (int i = 0; i < kc; i++)
            for (int j = 0; j < kc; j++)
                if ((i >= ns || j >= ns) &&
                    soft_Dinv[static_cast<std::size_t>(i) * kc + j] != 0.0)
                    return -1;
        return ns;
    }

    DrawTransform draw_transform() const {
        DrawTransform T;
        if (!usable || !block_form) return T;
        const int ns = n_soft_rows();
        if (ns < 0) return T;
        const int na = n_anchor();
        const int nj = ns + na;
        if (nj == 0) { T.ok = true; return T; }
        T.cols.resize(nj);
        for (int i = 0; i < ns; i++) T.cols[i] = i;
        for (int a = 0; a < na; a++) T.cols[ns + a] = kc + a;

        T.Psi.assign(static_cast<std::size_t>(nj) * nj, 0.0);
        if (ns > 0) {
            std::vector<double> Dinv_s(static_cast<std::size_t>(ns) * ns);
            for (int i = 0; i < ns; i++)
                for (int j = 0; j < ns; j++)
                    Dinv_s[static_cast<std::size_t>(i) * ns + j] =
                        soft_Dinv[static_cast<std::size_t>(i) * kc + j];
            SmallChol cd;
            cd.factor(Dinv_s, ns);
            if (!cd.ok) return T;
            std::vector<double> e(ns), col(ns);
            for (int j = 0; j < ns; j++) {
                std::fill(e.begin(), e.end(), 0.0);
                e[j] = 1.0;
                cd.solve(e.data(), col.data());
                for (int i = 0; i < ns; i++)
                    T.Psi[static_cast<std::size_t>(i) * nj + j] = col[i];
            }
        }
        for (int a = 0; a < na; a++)
            T.Psi[static_cast<std::size_t>(ns + a) * nj + ns + a] = -anchor_weight[a];

        std::vector<double> S(static_cast<std::size_t>(nj) * nj);
        for (int a = 0; a < nj; a++)
            for (int b = 0; b <= a; b++) {
                const int ca = T.cols[a], cb = T.cols[b];
                const double s = 0.5 * (vb_dot(ca, W[cb].data()) + vb_dot(cb, W[ca].data()));
                S[static_cast<std::size_t>(a) * nj + b] = s;
                S[static_cast<std::size_t>(b) * nj + a] = s;
            }
        SmallChol cs;
        cs.factor(S, nj);
        if (!cs.ok) return T;
        const std::vector<double>& L = cs.L;
        // K = L' Psi L.
        std::vector<double> PL(static_cast<std::size_t>(nj) * nj, 0.0);
        for (int i = 0; i < nj; i++)
            for (int j = 0; j < nj; j++) {
                double s = 0.0;
                for (int k = j; k < nj; k++)
                    s += T.Psi[static_cast<std::size_t>(i) * nj + k] *
                         L[static_cast<std::size_t>(k) * nj + j];
                PL[static_cast<std::size_t>(i) * nj + j] = s;
            }
        std::vector<double> K(static_cast<std::size_t>(nj) * nj);
        for (int i = 0; i < nj; i++)
            for (int j = 0; j <= i; j++) {
                double s = 0.0;
                for (int k = i; k < nj; k++)
                    s += L[static_cast<std::size_t>(k) * nj + i] *
                         PL[static_cast<std::size_t>(k) * nj + j];
                K[static_cast<std::size_t>(i) * nj + j] = s;
                K[static_cast<std::size_t>(j) * nj + i] = s;
            }
        SmallSymEig eg;
        eg.factor(K, nj);
        if (!eg.ok) return T;
        std::vector<double> g(nj);
        for (int j = 0; j < nj; j++) {
            const double lam = eg.w[j];
            if (!(lam >= -1.0 - 1e-8) || !std::isfinite(lam)) return T;
            g[j] = lam / (1.0 + std::sqrt(std::max(0.0, 1.0 + lam)));
        }
        // Y = L^{-T} Q, column by column (back substitution against L').
        std::vector<double> Y(static_cast<std::size_t>(nj) * nj, 0.0);
        for (int c = 0; c < nj; c++)
            for (int i = nj - 1; i >= 0; i--) {
                double s = eg.V[static_cast<std::size_t>(i) * nj + c];
                for (int k = i + 1; k < nj; k++)
                    s -= L[static_cast<std::size_t>(k) * nj + i] *
                         Y[static_cast<std::size_t>(k) * nj + c];
                Y[static_cast<std::size_t>(i) * nj + c] =
                    s / L[static_cast<std::size_t>(i) * nj + i];
            }
        T.Gam.assign(static_cast<std::size_t>(nj) * nj, 0.0);
        for (int a = 0; a < nj; a++)
            for (int b = 0; b < nj; b++) {
                double s = 0.0;
                for (int j = 0; j < nj; j++)
                    s += Y[static_cast<std::size_t>(a) * nj + j] * g[j] *
                         Y[static_cast<std::size_t>(b) * nj + j];
                T.Gam[static_cast<std::size_t>(a) * nj + b] = s;
            }
        T.ok = true;
        return T;
    }

    // In place, d (a draw L^{-T} eps against the factor of H_a, so
    // d' H_a d = eps' eps) becomes x ~ N(0, V). Returns eps' eps - x' M x,
    // what the quadratic form of the constrained Gaussian loses against the
    // draw's: b' Cap^{-1} b - u' Psi u with u = Vj' d and b = Vb' H_a^{-1} e,
    // since x' M x = e' V e = e' H_a^{-1} e - b' Cap^{-1} b and
    // e' H_a^{-1} e = d' M d = eps' eps + u' Psi u.
    double condition_draw(double* d, int n_x, const DrawTransform& T) const {
        const int nj = static_cast<int>(T.cols.size());
        double reduce = 0.0;
        if (nj > 0) {
            std::vector<double> u(nj), gu(nj, 0.0);
            for (int a = 0; a < nj; a++) u[a] = vb_dot(T.cols[a], d);
            for (int a = 0; a < nj; a++)
                for (int b = 0; b < nj; b++) {
                    const std::size_t ab = static_cast<std::size_t>(a) * nj + b;
                    gu[a] += T.Gam[ab] * u[b];
                    reduce -= u[a] * T.Psi[ab] * u[b];
                }
            for (int a = 0; a < nj; a++) {
                const std::vector<double>& wa = W[T.cols[a]];
                for (int t = 0; t < n_x; t++) d[t] += wa[t] * gu[a];
            }
        }
        const int m = static_cast<int>(W.size());
        std::vector<double> b(m), y(m);
        for (int i = 0; i < m; i++) b[i] = vb_dot(i, d);
        cap_solve(b.data(), y.data());
        for (int i = 0; i < m; i++) {
            reduce += b[i] * y[i];
            const std::vector<double>& wi = W[i];
            for (int t = 0; t < n_x; t++) d[t] -= wi[t] * y[i];
        }
        return reduce;
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
        const int m = order();
        if (!usable) { std::fill(y, y + m, 0.0); return; }
        cap_solve(b, y);
    }

    // y = Cap^{-1} G_latent, with G_latent[i] = W[i][latent]. `y` is resized to
    // the capacitance's order.
    void solve_y(int latent, std::vector<double>& y) const {
        const int m = static_cast<int>(W.size());
        y.assign(m, 0.0);
        if (!usable) return;
        std::vector<double> g_a(m);
        for (int i = 0; i < m; i++) g_a[i] = W[i][latent];
        cap_solve(g_a.data(), y.data());
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
        cap_solve(G.data(), y.data());
        double corr = 0.0;
        for (int i = 0; i < m; i++) corr += G[i] * y[i];
        return corr;
    }
};

// Diagonal blocks of H^{-1} (or of the constrained V when `constr` is a built,
// usable correction) for the requested (offset, size) ranges. With `constr`,
// `solve` is the solve against the anchored precision it was built on.
//
// Each block costs one block solve and no refactorization. Blocks are
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

    const std::size_t n = static_cast<std::size_t>(n_x);
    for (const auto& blk : layout) {
        const int s = blk.first, m = blk.second;
        if (m <= 0 || s < 0 || s + m > n_x) continue;
        std::vector<double> block(static_cast<std::size_t>(m) * m, 0.0);

        std::vector<double> rhs(n * m, 0.0), col(n * m, 0.0);
        for (int c = 0; c < m; c++) rhs[n * c + s + c] = 1.0;
        solve(rhs.data(), col.data(), m);
        for (int c = 0; c < m; c++)
            for (int r = 0; r < m; r++) block[r * m + c] = col[n * c + s + r];

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
