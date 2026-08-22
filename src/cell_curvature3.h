// cell_curvature3.h
//
// The third-derivative tensor of a CELL-COUPLED log-density,
// contracted against the inner-Laplace probe direction.
//
// A `CellCouplingSpec` replaces a cell's per-observation sum with one
// non-separable `evaluate_cell()` term over several arms at once (tulpaObs's
// occu_cover; the two-arm occupancy mixture in
// src/test_cell_coupling_occupancy_mixture.h). There is no per-eta third
// derivative for the separable formula in inner_laplace_skew.h to read, so such
// an arm used to be excluded from gamma_3 entirely -- correctly, but permanently.
//
// curvature3_contract.h carries the widened contraction: with the cell's ARMS as
// the blocks,
//
//   sum_{a,b,c} T^{abc} u^a u^b u^c = sum_a d/ds [ u' L''(e + s u^(a)) u ]_{s=0},
//
// and each derivative on the right is one central difference of the cross-arm
// Hessian `evaluate_cell()` already writes into `CellDerivs` for the Newton
// solve. So this is ONE finite-difference layer on a quantity the spec computes
// analytically -- the same noise class as the scalar working-weight difference
// in laplace_spec_curvature3.h, not a difference of a difference. Cost is 2K
// extra `evaluate_cell()` calls per cell for K coupled arms; nothing is stored.
//
// The bilinear forms below read exactly the `CellDerivs` layout the joint
// scatter reads (`scatter_cell_coupling_branch_impl`,
// src/nested_laplace_joint_multi.h): the per-arm negative-Hessian diagonal, the
// strict upper triangle of a dense self block, the full (kk, ll) cross block for
// kk < ll, and the rank-1 self-cross descriptor when the spec declares one.
// Blocks the spec does not declare in `dense_cross_pairs()` are absent, which is
// the spec asserting that pair contributes nothing at this cell.
//
// DECLINES with `curvature3_unavailable` -- the vocabulary reason for "this
// likelihood ships no way to reach a third derivative", not the structural
// "coupled_likelihood" -- when there is no spec, no coupled arm, or no cell to
// walk. A cell whose differenced Hessian comes back non-finite takes the whole
// contraction to NaN rather than dropping out of the sum, so a broken difference
// can never read as "no skew" nor as an understated one.

#ifndef TULPA_CELL_CURVATURE3_H
#define TULPA_CELL_CURVATURE3_H

#include "curvature3_contract.h"
#include "inner_laplace_skew.h"   // CellCubic3Fn -- the shape a joint fit stores
#include "tulpa/cell_coupling.h"

#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tulpa {

// The contraction the builder below returns is a CellCubic3Fn
// (inner_laplace_skew.h): `eta0` / `eta1` are the per-arm linear predictors at
// the mode and at the probed point mode + v_i, indexed by the JOINT arm index
// (the `scratch.etas` / `scratch.etas_tmp` layout the skew loop already holds),
// so the direction is u = eta1 - eta0.

// Everything the contraction needs, resolved by the caller once per solve (after
// the grid cell's dispersion sync, so `arm_phi` is the dispersion this solve
// runs at). Copied into the returned closure, so no member has to outlive the
// build call.
struct CellCurvature3Inputs {
    const CellCouplingSpec* spec = nullptr;
    // cell_rows[kk][c] -- 0-based rows of coupled arm kk belonging to cell c.
    const std::vector<std::vector<std::vector<int>>>* cell_rows = nullptr;
    int                        n_cells = 0;
    std::vector<int>           coupled_arms;    // joint arm index per coupled slot
    std::vector<const double*> arm_y;           // full-length, may be null
    std::vector<const int*>    arm_n_trials;    // full-length, may be null
    std::vector<std::string>   arm_family;
    std::vector<double>        arm_phi;
    // Per-arm step scaling (the default) vs one step for the whole cell. The
    // global variant exists so the per-arm policy is measured against it.
    bool                       per_arm_step = true;
};

// Per-cell `CellDerivs` buffers, sized from the cell's row counts and the pairs
// the spec declares dense. Grown monotonically across cells and reused.
struct CellDerivScratch {
    std::vector<std::vector<double>> grad, diag, rank1_vec;
    std::vector<double*>             grad_ptr, diag_ptr, rank1_vec_ptr;
    std::vector<double>              rank1_coef;
    std::vector<std::vector<std::vector<double>>> cross;
    std::vector<std::vector<double*>>             cross_inner;
    std::vector<double* const*>                   cross_outer;
    std::vector<std::vector<char>>                alloc_pair;

    void init(int K, const CellCouplingSpec& spec) {
        grad.assign(K, {}); diag.assign(K, {}); rank1_vec.assign(K, {});
        grad_ptr.assign(K, nullptr); diag_ptr.assign(K, nullptr);
        rank1_vec_ptr.assign(K, nullptr);
        rank1_coef.assign(K, 0.0);
        cross.assign(K, std::vector<std::vector<double>>(K));
        cross_inner.assign(K, std::vector<double*>(K, nullptr));
        cross_outer.assign(K, nullptr);
        for (int k = 0; k < K; k++) cross_outer[k] = cross_inner[k].data();
        alloc_pair.assign(K, std::vector<char>(K, 0));
        for (const auto& pr :
             spec.dense_cross_pairs(K, /*rank1_self_supported=*/true)) {
            const int a = std::min(pr.first, pr.second);
            const int b = std::max(pr.first, pr.second);
            if (a >= 0 && b < K) alloc_pair[a][b] = 1;
        }
    }

    // Zero and resize for one cell's row counts, mirroring the kernel's own
    // per-cell buffer policy so the spec sees the CellDerivs shape it sees under
    // the driver.
    void reset(const std::vector<int>& rc) {
        const int K = static_cast<int>(rc.size());
        for (int k = 0; k < K; k++) {
            const int n = std::max(rc[k], 1);
            if ((int)grad[k].size() < n) {
                grad[k].assign(n, 0.0);
                diag[k].assign(n, 0.0);
                rank1_vec[k].assign(n, 0.0);
            } else {
                std::fill(grad[k].begin(), grad[k].begin() + n, 0.0);
                std::fill(diag[k].begin(), diag[k].begin() + n, 0.0);
                std::fill(rank1_vec[k].begin(), rank1_vec[k].begin() + n, 0.0);
            }
            grad_ptr[k]      = grad[k].data();
            diag_ptr[k]      = diag[k].data();
            rank1_vec_ptr[k] = rank1_vec[k].data();
            rank1_coef[k]    = 0.0;
        }
        for (int k = 0; k < K; k++) {
            for (int l = 0; l < K; l++) {
                if (l < k || !alloc_pair[k][l]) { cross_inner[k][l] = nullptr; continue; }
                const std::size_t n = (std::size_t)rc[k] * (std::size_t)rc[l];
                auto& buf = cross[k][l];
                if (buf.size() < n) buf.assign(std::max<std::size_t>(n, 1), 0.0);
                else                std::fill(buf.begin(), buf.begin() + n, 0.0);
                cross_inner[k][l] = buf.data();
            }
        }
    }

    void bind(CellDerivs& out, const std::vector<int>& rc) {
        out.arm_grad             = grad_ptr.data();
        out.arm_neg_hess_diag    = diag_ptr.data();
        out.arm_cross_hess       = cross_outer.data();
        out.arm_row_count        = rc.data();
        out.n_arms_              = static_cast<int>(rc.size());
        out.curvature            = CurvatureMode::Observed;
        out.grad_only            = false;
        out.arm_cross_rank1_coef = rank1_coef.data();
        out.arm_cross_rank1_vec  = rank1_vec_ptr.data();
    }
};

// Ordered bilinear form of the cell's NEGATIVE Hessian restricted to arm blocks
// (b, c), contracted with u on both slots, so summing over every ordered (b, c)
// reproduces u' NH u. Reads the same entries the joint scatter reads: the
// diagonal plus the strict upper triangle of a dense self block (the kernel
// writes each such entry into both triangles of the joint Hessian, hence the
// factor 2), the rank-1 self-cross when declared, and the full kk < ll cross
// slab.
inline double cell_neg_hess_block_form(const CellDerivScratch& s,
                                       const std::vector<std::vector<double>>& u,
                                       const std::vector<int>& rc,
                                       int b, int c) {
    if (b > c) std::swap(b, c);
    if (b == c) {
        const int n = rc[b];
        double acc = 0.0;
        const double* d = s.diag[b].data();
        for (int j = 0; j < n; j++) acc += d[j] * u[b][j] * u[b][j];
        if (s.rank1_coef[b] != 0.0) {
            double dot = 0.0;
            const double* v = s.rank1_vec[b].data();
            for (int j = 0; j < n; j++) dot += v[j] * u[b][j];
            acc += s.rank1_coef[b] * dot * dot;
        }
        if (const double* ch = s.cross_inner[b][b]) {
            for (int j = 0; j < n; j++) {
                for (int m = j + 1; m < n; m++) {
                    acc += 2.0 * ch[(std::size_t)j * n + m] * u[b][j] * u[b][m];
                }
            }
        }
        return acc;
    }
    const double* ch = s.cross_inner[b][c];
    if (!ch) return 0.0;
    const int nb = rc[b], nc = rc[c];
    double acc = 0.0;
    for (int j = 0; j < nb; j++) {
        for (int m = 0; m < nc; m++) {
            acc += ch[(std::size_t)j * nc + m] * u[b][j] * u[c][m];
        }
    }
    return acc;
}

// State the contraction carries between calls: the resolved inputs plus the
// per-cell scratch. Held by shared_ptr inside the returned std::function, so the
// closure owns everything it reads. Calls on one instance are sequential (one
// per outer-grid thread slot, one probed index at a time), which is what lets
// the scratch be reused instead of reallocated per cell.
struct CellCurvature3Tensor {
    CellCurvature3Inputs in;
    CellDerivScratch     scr;
    std::vector<std::vector<double>> eta_local, u_local;
    std::vector<std::vector<double>> eta_pert;
    std::vector<std::vector<int>>    rows_local;
    std::vector<const double*>       eta_ptr;
    std::vector<const int*>          rows_local_ptr, rows_real_ptr;
    std::vector<const char*>         family_ptr;
    std::vector<int>                 rc;
    std::vector<double>              bf, qp, qm, max_eta, max_u;
    std::vector<char>                have;

    explicit CellCurvature3Tensor(CellCurvature3Inputs inputs)
        : in(std::move(inputs)) {
        const int K = static_cast<int>(in.coupled_arms.size());
        scr.init(K, *in.spec);
        eta_local.assign(K, {}); u_local.assign(K, {}); eta_pert.assign(K, {});
        rows_local.assign(K, {});
        eta_ptr.assign(K, nullptr);
        rows_local_ptr.assign(K, nullptr);
        rows_real_ptr.assign(K, nullptr);
        family_ptr.assign(K, nullptr);
        for (int k = 0; k < K; k++) family_ptr[k] = in.arm_family[k].c_str();
        rc.assign(K, 0);
        bf.assign((std::size_t)K * K * K, 0.0);
        qp.assign((std::size_t)K * K, 0.0);
        qm.assign((std::size_t)K * K, 0.0);
        max_eta.assign(K, 0.0);
        max_u.assign(K, 0.0);
        have.assign(K, 0);
    }

    // Evaluate the spec at the currently staged `eta_pert` and reduce its
    // CellDerivs Hessian to the K x K matrix of ordered bilinear forms.
    // Returns false when any form came back non-finite.
    bool block_forms(int cell, std::vector<double>& q_out) {
        const int K = static_cast<int>(rc.size());
        scr.reset(rc);
        for (int k = 0; k < K; k++) eta_ptr[k] = eta_pert[k].data();

        CellEtas etas_view;
        etas_view.arm_eta_ptr   = eta_ptr.data();
        etas_view.arm_rows      = rows_local_ptr.data();
        etas_view.arm_row_count = rc.data();
        etas_view.n_arms_       = K;

        CellResponse y_view;
        y_view.arm_y         = in.arm_y.data();
        y_view.arm_n_trials  = in.arm_n_trials.data();
        y_view.arm_family    = family_ptr.data();
        y_view.arm_phi       = in.arm_phi.data();
        y_view.arm_rows      = rows_real_ptr.data();
        y_view.arm_row_count = rc.data();
        y_view.n_arms_       = K;

        CellDerivs out;
        scr.bind(out, rc);
        in.spec->evaluate_cell(cell, etas_view, y_view, out);

        for (int b = 0; b < K; b++) {
            for (int c = 0; c < K; c++) {
                const double v = cell_neg_hess_block_form(scr, u_local, rc, b, c);
                if (!std::isfinite(v)) return false;
                q_out[(std::size_t)b * K + c] = v;
            }
        }
        return true;
    }

    double operator()(const std::vector<Rcpp::NumericVector>& eta0,
                      const std::vector<Rcpp::NumericVector>& eta1) {
        const int K = static_cast<int>(in.coupled_arms.size());
        double total = 0.0;
        bool any = false;

        // The two eta vectors are indexed by the arm id the cell-row table
        // carries, and Rcpp::NumericVector::operator[] is unchecked, so a table
        // that disagrees with what was handed in is a read past the allocation
        // rather than an error. An unreadable input is not a small skew.
        const int n_arms_eta = static_cast<int>(eta0.size());
        if (static_cast<int>(eta1.size()) != n_arms_eta) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        for (int c = 0; c < in.n_cells; c++) {
            bool moved = false;
            for (int kk = 0; kk < K; kk++) {
                const int k = in.coupled_arms[kk];
                const std::vector<int>& rows = (*in.cell_rows)[kk][c];
                const int n = static_cast<int>(rows.size());
                rc[kk] = n;
                if (k < 0 || k >= n_arms_eta ||
                    eta1[k].size() != eta0[k].size()) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                const int n_eta = static_cast<int>(eta0[k].size());
                if ((int)eta_local[kk].size() < n) {
                    eta_local[kk].assign(n, 0.0);
                    u_local[kk].assign(n, 0.0);
                    eta_pert[kk].assign(n, 0.0);
                    rows_local[kk].assign(n, 0);
                }
                for (int j = 0; j < n; j++) rows_local[kk][j] = j;
                double me = 0.0, mu = 0.0;
                for (int j = 0; j < n; j++) {
                    const int r = rows[j];
                    if (r < 0 || r >= n_eta) {
                        return std::numeric_limits<double>::quiet_NaN();
                    }
                    const double e0 = eta0[k][r];
                    const double du = eta1[k][r] - e0;
                    eta_local[kk][j] = e0;
                    u_local[kk][j]   = du;
                    me = std::max(me, std::fabs(e0));
                    mu = std::max(mu, std::fabs(du));
                }
                max_eta[kk] = me;
                max_u[kk]   = mu;
                if (mu > 0.0) moved = true;
                rows_local_ptr[kk] = rows_local[kk].data();
                rows_real_ptr[kk]  = rows.data();
            }
            // The probe direction does not reach this cell: its cubic
            // contribution is exactly zero, not "not computable".
            if (!moved) { any = true; continue; }

            const double h_global = in.per_arm_step
                ? 0.0 : curvature3_global_step(max_eta, max_u);

            std::fill(bf.begin(), bf.end(), 0.0);
            std::fill(have.begin(), have.end(), 0);
            for (int a = 0; a < K; a++) {
                // An arm the probe direction does not move contributes exactly
                // zero. A MOVED arm whose step could not be sized -- a
                // non-finite eta or displacement, which is what both step
                // helpers signal by returning 0 -- is unreadable, and dropping
                // it would report a smaller cubic term rather than none.
                if (max_u[a] == 0.0) continue;
                const double h = in.per_arm_step
                    ? curvature3_block_step(max_eta[a], max_u[a]) : h_global;
                if (!(h > 0.0)) {
                    return std::numeric_limits<double>::quiet_NaN();
                }

                for (int kk = 0; kk < K; kk++) {
                    for (int j = 0; j < rc[kk]; j++) eta_pert[kk][j] = eta_local[kk][j];
                }
                for (int j = 0; j < rc[a]; j++) eta_pert[a][j] += h * u_local[a][j];
                if (!block_forms(c, qp)) {
                    return std::numeric_limits<double>::quiet_NaN();
                }

                for (int j = 0; j < rc[a]; j++) {
                    eta_pert[a][j] = eta_local[a][j] - h * u_local[a][j];
                }
                if (!block_forms(c, qm)) {
                    return std::numeric_limits<double>::quiet_NaN();
                }

                const double inv = 1.0 / (2.0 * h);
                for (int b = 0; b < K; b++) {
                    for (int d = 0; d < K; d++) {
                        const std::size_t t = (std::size_t)b * K + d;
                        // L'' = -NH, so the log-density's difference quotient is
                        // the negated difference of the negative Hessian forms.
                        bf[(std::size_t)(a * K + b) * K + d] =
                            -(qp[t] - qm[t]) * inv;
                    }
                }
                have[a] = 1;
            }
            const double cell_cubic = curvature3_symmetrized_sum(bf, have, K);
            // One unreadable cell poisons the whole contraction rather than
            // being dropped: the remaining cells would sum to a number that
            // silently understates the cubic term.
            if (!std::isfinite(cell_cubic)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            total += cell_cubic;
            any = true;
        }
        if (!any) return std::numeric_limits<double>::quiet_NaN();
        return total;
    }
};

// Can a cell tensor be built for this fit at all? The structural half of
// build_cell_curvature3_tensor's own gate, exposed so the per-arm decline
// bookkeeping (build_joint_curvature3_fns' `coupled_scored`, decided before the
// per-solve dispersion is known) and the build itself cannot answer differently.
inline bool cell_curvature3_available(const CellCouplingSpec* spec,
                                      const std::vector<int>& coupled_arms,
                                      int n_cells) {
    return spec != nullptr && !coupled_arms.empty() && n_cells > 0;
}

// Build the contraction. `reason` (optional out-parameter) is set to the
// vocabulary reason when no oracle could be built, "" otherwise.
inline CellCubic3Fn build_cell_curvature3_tensor(CellCurvature3Inputs in,
                                                 const char** reason = nullptr) {
    if (reason) *reason = "";
    const int K = static_cast<int>(in.coupled_arms.size());
    if (!cell_curvature3_available(in.spec, in.coupled_arms, in.n_cells) ||
        !in.cell_rows ||
        (int)in.arm_y.size() != K || (int)in.arm_n_trials.size() != K ||
        (int)in.arm_family.size() != K || (int)in.arm_phi.size() != K ||
        (int)in.cell_rows->size() != K) {
        if (reason) *reason = "curvature3_unavailable";
        return nullptr;
    }
    auto state = std::make_shared<CellCurvature3Tensor>(std::move(in));
    return [state](const std::vector<Rcpp::NumericVector>& eta0,
                   const std::vector<Rcpp::NumericVector>& eta1) -> double {
        return (*state)(eta0, eta1);
    };
}

} // namespace tulpa

#endif // TULPA_CELL_CURVATURE3_H
