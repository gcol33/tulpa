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

#include "linalg_fast.h"
#include <algorithm>
#include <cstddef>
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

// Sum-to-zero constraint correction for a factorized precision H.
//
// With A the k_constr x n_x incidence matrix of the constraint groups (row g is
// the indicator of group g's latent indices), conditioning by kriging gives
//
//   V = H^{-1} - H^{-1} A' (A H^{-1} A')^{-1} A H^{-1}
//
// so the correction at latent indices (a, b) is G_a' M^{-1} G_b with
// W = H^{-1} A' (one solve per group), M = A W, and G_a = W[a, ].
struct InvBlockConstraint {
    int kc = 0;
    std::vector<std::vector<double>> W;   // kc columns of H^{-1} A', length n_x
    std::vector<double> Lm;               // kc x kc row-major lower Cholesky of M
    bool usable = false;                  // false when M is not PD (degenerate)

    // Solve W_g and factor M. `A_cols[g]` holds group g's 0-based latent
    // indices; out-of-range entries are ignored. An M that does not factor
    // leaves `usable = false`, and the caller then skips the correction rather
    // than subtracting a wrong one.
    //
    // Row g of A is the INDICATOR of group g's indices, so a latent index
    // repeated within a group contributes once. Both the right-hand side and the
    // Gram matrix are therefore read off one normalized index list, in the
    // caller's own order with later repeats dropped: the two must use the same A
    // or the factored M is not `A W`, and reordering would move the summation
    // and with it the last bits of the correction.
    template <typename SolveFn>
    void build(SolveFn solve, int n_x,
               const std::vector<std::vector<int>>& A_cols) {
        kc = static_cast<int>(A_cols.size());
        usable = false;
        W.clear();
        Lm.clear();
        if (kc <= 0) return;

        std::vector<std::vector<int>> cols(kc);
        std::vector<char> seen(n_x > 0 ? n_x : 1, 0);
        for (int g = 0; g < kc; g++) {
            cols[g].reserve(A_cols[g].size());
            for (int latent : A_cols[g]) {
                if (latent < 0 || latent >= n_x || seen[latent]) continue;
                seen[latent] = 1;
                cols[g].push_back(latent);
            }
            for (int latent : cols[g]) seen[latent] = 0;
        }

        std::vector<double> e(n_x, 0.0), v(n_x, 0.0);
        W.resize(kc);
        for (int g = 0; g < kc; g++) {
            std::fill(e.begin(), e.end(), 0.0);
            for (int latent : cols[g]) e[latent] = 1.0;
            solve(e.data(), v.data());
            W[g] = v;
        }

        std::vector<double> M(static_cast<std::size_t>(kc) * kc, 0.0);
        for (int g1 = 0; g1 < kc; g1++) {
            for (int g2 = 0; g2 < kc; g2++) {
                double s = 0.0;
                for (int latent : cols[g1]) s += W[g2][latent];
                M[static_cast<std::size_t>(g1) * kc + g2] = s;
            }
        }

        Lm.assign(static_cast<std::size_t>(kc) * kc, 0.0);
        usable = tulpa_linalg::chol_factor_lower<tulpa_linalg::TriLayout::RowMajor>(
            M.data(), Lm.data(), kc, kc, /*nugget=*/0.0);
        if (!usable) Lm.clear();
    }

    // y = M^{-1} G_latent, with G_latent[g] = W[g][latent]. `y` is resized to kc.
    void solve_y(int latent, std::vector<double>& y) const {
        y.assign(kc, 0.0);
        if (!usable) return;
        std::vector<double> g_a(kc), tmp(kc);
        for (int g = 0; g < kc; g++) g_a[g] = W[g][latent];
        tulpa_linalg::tri_solve_lower<tulpa_linalg::TriLayout::RowMajor>(
            Lm.data(), kc, kc, g_a.data(), tmp.data());
        tulpa_linalg::tri_solve_lower_transpose<tulpa_linalg::TriLayout::RowMajor>(
            Lm.data(), kc, kc, tmp.data(), y.data());
    }

    // G_latent' y for a y produced by solve_y at the other index.
    double correction(int latent, const std::vector<double>& y) const {
        if (!usable) return 0.0;
        double corr = 0.0;
        for (int g = 0; g < kc; g++) corr += W[g][latent] * y[g];
        return corr;
    }
};

// Diagonal blocks of H^{-1} (or of the constrained V when `constr` is a built,
// usable correction) for the requested (offset, size) ranges.
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
