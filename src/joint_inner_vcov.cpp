// joint_inner_vcov.cpp
// Per-cell constrained inner-covariance block extraction for the joint
// nested-Laplace post-grid step.
// See joint_inner_vcov.h.

#include "joint_inner_vcov.h"
#include "inv_block_extract.h"  // InvBlockConstraint, extract_inv_diag_blocks
#include "sparse_cholesky.h"
#include "omp_threads.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

namespace {

// Build an owned CHOLMOD lower-triangle symmetric sparse (stype = -1) from a
// CSC (Qp/Qi/Qx). Caller frees via M_cholmod_free_sparse on the same common.
cholmod_sparse* csc_to_cholmod(
    const int* Qp, const int* Qi, const double* Qx, int n_x, int nnz,
    cholmod_common* common
) {
    cholmod_sparse* A = M_cholmod_allocate_sparse(
        n_x, n_x, nnz, /*sorted=*/1, /*packed=*/1, /*stype=*/-1,
        CHOLMOD_REAL, common);
    if (!A) return nullptr;
    int* Ap = static_cast<int*>(A->p);
    int* Ai = static_cast<int*>(A->i);
    double* Ax = static_cast<double*>(A->x);
    for (int j = 0; j <= n_x; j++) Ap[j] = Qp[j];
    for (int e = 0; e < nnz; e++) { Ai[e] = Qi[e]; Ax[e] = Qx[e]; }
    return A;
}

// Factorize Qk, retrying once with a tiny diagonal ridge if the bare precision
// is numerically singular (the field's constant direction can leave Qk singular;
// the sum-to-zero constraint projects that direction out, so the ridge's effect
// on the constrained block is negligible -- the same fallback the draw path
// uses). Leaves `solver` holding the factor on success.
bool factorize_cell(
    SparseCholeskySolver& solver,
    const int* Qp, const int* Qi, const double* Qx, int n_x, int nnz
) {
    solver.reset();
    cholmod_sparse* A = csc_to_cholmod(Qp, Qi, Qx, n_x, nnz, &solver.common());
    if (!A) return false;
    solver.analyze(A);
    bool ok = solver.factorize(A);
    M_cholmod_free_sparse(&A, &solver.common());
    if (ok) return true;

    // Ridge fallback: bump the CSC diagonal (first slot of each lower-tri
    // column, where Qi[Qp[j]] == j) by a relative jitter, then re-factorize. A
    // structurally empty column has no slot to read or bump; on the lower
    // triangle of a factorizable precision there is none, so this leaves the
    // mean over the columns that carry one.
    double diag_mean = 0.0;
    int n_diag = 0;
    for (int j = 0; j < n_x; j++)
        if (Qp[j] < nnz) { diag_mean += Qx[Qp[j]]; n_diag++; }
    diag_mean = (n_diag > 0) ? diag_mean / n_diag : 0.0;
    const double jit = 1e-8 * (std::abs(diag_mean) + 1e-8);
    std::vector<double> Qx_r(Qx, Qx + nnz);
    for (int j = 0; j < n_x; j++)
        if (Qp[j] < nnz) Qx_r[Qp[j]] += jit;

    solver.reset();
    cholmod_sparse* A2 = csc_to_cholmod(Qp, Qi, Qx_r.data(), n_x, nnz,
                                        &solver.common());
    if (!A2) return false;
    solver.analyze(A2);
    ok = solver.factorize(A2);
    M_cholmod_free_sparse(&A2, &solver.common());
    return ok;
}

} // namespace

bool extract_inner_vcov_block_cell(
    const int* Qp, const int* Qi, const double* Qx, int n_x, int nnz,
    const std::vector<int>& idx0, int n_dense,
    const std::vector<std::vector<int>>& A_cols,
    bool field_marginal,
    SparseCholeskySolver& solver,
    std::vector<double>& out_block
) {
    const int p = static_cast<int>(idx0.size());
    out_block.assign(static_cast<std::size_t>(p) * p, 0.0);
    if (p == 0) return true;
    if (n_dense < 0) n_dense = 0;
    if (n_dense > p) n_dense = p;
    const int n_field = p - n_dense;

    if (!factorize_cell(solver, Qp, Qi, Qx, n_x, nnz)) return false;

    // Columns of C = Qk^{-1} to solve: the dense (fixed-effect) block in the
    // cheap recipe; every idx column in full mode. Each solve reuses the factor.
    const int n_solve = field_marginal ? n_dense : p;
    std::vector<std::vector<double>> C_cols(n_solve);
    std::vector<double> e(n_x, 0.0), v(n_x, 0.0);
    for (int t = 0; t < n_solve; t++) {
        const int c = idx0[t];
        std::fill(e.begin(), e.end(), 0.0);
        e[c] = 1.0;
        solver.solve(e.data(), v.data(), n_x);
        C_cols[t] = v;
    }

    // Constraint columns W = C A' (one solve per sum-to-zero group) and the
    // Cholesky of M = A C A'. Shared with the Newton-loop extraction and with
    // laplace_newton.h's diagonal-block path through inv_block_extract.h, so
    // the conditioning-by-kriging algebra is written once.
    auto solve_cell = [&](const double* rhs, double* out) {
        solver.solve(rhs, out, n_x);
    };
    InvBlockConstraint constr;
    constr.build(solve_cell, n_x, A_cols);

    // Field marginal variances via one Takahashi selected-inversion pass (only
    // needed in the cheap recipe; full mode reads field diagonals off C_cols).
    std::vector<double> diag_inv;
    if (field_marginal && n_field > 0) {
        diag_inv = solver.selected_inversion_diagonal();   // length n_x, orig order
        if (static_cast<int>(diag_inv.size()) != n_x) return false;
    }

    // Constraint correction corr(a,b) = G_a' M^{-1} G_b, G_a = W[idx0[a], :].
    // Precompute y_a = M^{-1} G_a for every idx position touched.
    std::vector<std::vector<double>> yvec;
    if (constr.usable) {
        yvec.resize(p);
        for (int a = 0; a < p; a++) constr.solve_y(idx0[a], yvec[a]);
    }
    const bool have_corr = constr.usable;

    // Assemble the constrained block (lower triangle + mirror). field x field
    // off-diagonal in the cheap recipe stays 0.
    for (int a = 0; a < p; a++) {
        for (int b = 0; b <= a; b++) {
            const bool both_field = (a >= n_dense) && (b >= n_dense);
            double cval;
            if (field_marginal && both_field) {
                if (a == b) cval = diag_inv[idx0[a]];
                else continue;   // off-diagonal field x field: leave 0
            } else {
                // Unconstrained C[idx0[a], idx0[b]] from a solved column. `b` is
                // always solved here: without field_marginal every idx column is
                // solved, and with it this arm is reached only when b < n_dense,
                // since b <= a and both_field is false.
                cval = C_cols[b][idx0[a]];
            }
            const double corr =
                have_corr ? constr.correction(idx0[a], yvec[b]) : 0.0;
            const double val = cval - corr;
            out_block[static_cast<std::size_t>(a) + static_cast<std::size_t>(b) * p] = val;
            out_block[static_cast<std::size_t>(b) + static_cast<std::size_t>(a) * p] = val;
        }
    }
    return true;
}

bool extract_joint_fixed_block(
    const int* Qp, const int* Qi, const double* Qx, int n_x, int nnz,
    const JointFixedBlockRequest& req,
    SparseCholeskySolver& solver,
    std::vector<double>& flat_out,
    std::vector<int>& sizes_out
) {
    const int p = req.p;
    if (p <= 0 || p > n_x || nnz <= 0) return false;
    if (!factorize_cell(solver, Qp, Qi, Qx, n_x, nnz)) return false;

    auto solve_cell = [&](const double* rhs, double* out) {
        solver.solve(rhs, out, n_x);
    };
    InvBlockConstraint constr;
    constr.build(solve_cell, n_x, req.A_cols);

    const std::size_t before = sizes_out.size();
    // MirrorLower + the leading (0, p) range reproduce
    // extract_inner_vcov_block_cell's own assembly for idx = [0, p),
    // n_dense = p, field_marginal = false, entry for entry.
    extract_inv_diag_blocks(solve_cell, n_x,
                            std::vector<std::pair<int, int>>{{0, p}},
                            &constr, InvBlockSymmetry::MirrorLower,
                            flat_out, sizes_out);
    return sizes_out.size() > before;
}

} // namespace tulpa

// Rcpp entry: per-cell constrained inner-covariance blocks over the outer grid,
// parallel across cells. Single source for the joint post-grid SD/summary path
// (selected inversion + parallelization).
//
// Q_*_per_grid : per-cell lower-triangle CSC of Qk (Q_i 0-based rows), as the
//                joint kernel stores them; a NULL, empty or malformed cell
//                yields NULL.
// idx          : 1-based latent indices [dense | field], length p. Every entry
//                must lie in [1, n_x]; so must every A_cols_list entry.
// n_dense      : number of leading dense (fixed-effect) indices; the remaining
//                p - n_dense are field. With field_marginal the field block is
//                returned diagonal-only (off-diagonal 0).
// A_cols_list  : list of 1-based latent-index vectors, one per sum-to-zero
//                constraint group (empty list => unconstrained).
// field_marginal : cheap recipe (TRUE, the default) vs full p x p block (FALSE).
// Returns a list of length n_grid: each element NULL or a p x p NumericMatrix.
//
// [[Rcpp::export]]
Rcpp::List cpp_joint_inner_vcov_blocks(
    Rcpp::List Q_p_per_grid, Rcpp::List Q_i_per_grid, Rcpp::List Q_x_per_grid,
    int n_x, Rcpp::IntegerVector idx, int n_dense,
    Rcpp::List A_cols_list, bool field_marginal = true, int n_threads = 1
) {
    const int n_grid = Q_p_per_grid.size();
    const int p = idx.size();

    // Every R-supplied index is checked here, before the parallel region opens:
    // an index outside [1, n_x] is an out-of-bounds WRITE into an n_x-sized
    // solve buffer, and inside an OpenMP team that corrupts a sibling thread's
    // arena rather than faulting.
    if (n_x <= 0)
        Rcpp::stop("n_x (%d) must be positive.", n_x);
    if (Q_i_per_grid.size() != (R_xlen_t)n_grid ||
        Q_x_per_grid.size() != (R_xlen_t)n_grid)
        Rcpp::stop("Q_p_per_grid, Q_i_per_grid and Q_x_per_grid must have the "
                   "same length.");
    if (p > n_x)
        Rcpp::stop("idx holds %d indices, more than the %d latent coordinates.",
                   p, n_x);

    // 1-based R indices -> 0-based POD (outside any parallel region).
    std::vector<int> idx0(p);
    for (int t = 0; t < p; t++) {
        const int v = idx[t];
        if (v == NA_INTEGER || v < 1 || v > n_x)
            Rcpp::stop("idx[%d] is not a latent index in [1, %d].", t + 1, n_x);
        idx0[t] = v - 1;
    }
    const int kc = A_cols_list.size();
    std::vector<std::vector<int>> A_cols(kc);
    for (int g = 0; g < kc; g++) {
        Rcpp::IntegerVector col = A_cols_list[g];
        A_cols[g].reserve(col.size());
        for (int e = 0; e < col.size(); e++) {
            const int v = col[e];
            if (v == NA_INTEGER || v < 1 || v > n_x)
                Rcpp::stop("A_cols_list[[%d]][%d] is not a latent index in "
                           "[1, %d].", g + 1, e + 1, n_x);
            A_cols[g].push_back(v - 1);
        }
    }

    // Pre-extract every cell's CSC into POD (R API is not thread-safe). Empty /
    // NULL cells are marked so the parallel loop skips them, and so is a cell
    // whose CSC triple is not a well-formed n_x x n_x lower triangle: CHOLMOD
    // and the ridge fallback both index Qi / Qx off the pointer array, so a
    // malformed cell reads outside the extraction rather than failing to factor.
    std::vector<std::vector<int>>    Qp_all(n_grid), Qi_all(n_grid);
    std::vector<std::vector<double>> Qx_all(n_grid);
    std::vector<char> has_cell(n_grid, 0);
    for (int k = 0; k < n_grid; k++) {
        if (Rf_isNull(Q_p_per_grid[k]) || Rf_isNull(Q_x_per_grid[k])) continue;
        if (Rf_isNull(Q_i_per_grid[k])) continue;
        Rcpp::IntegerVector Qp = Q_p_per_grid[k];
        Rcpp::IntegerVector Qi = Q_i_per_grid[k];
        Rcpp::NumericVector Qx = Q_x_per_grid[k];
        const int nnz = Qx.size();
        if (nnz == 0) continue;
        if (Qp.size() != (R_xlen_t)n_x + 1 || Qi.size() != (R_xlen_t)nnz) continue;
        if (Qp[0] != 0 || Qp[n_x] != nnz) continue;
        bool well_formed = true;
        for (int j = 0; j < n_x && well_formed; j++)
            if (Qp[j] == NA_INTEGER || Qp[j + 1] == NA_INTEGER ||
                Qp[j] > Qp[j + 1]) well_formed = false;
        for (int e = 0; e < nnz && well_formed; e++)
            if (Qi[e] == NA_INTEGER || Qi[e] < 0 || Qi[e] >= n_x)
                well_formed = false;
        if (!well_formed) continue;
        Qp_all[k].assign(Qp.begin(), Qp.end());
        Qi_all[k].assign(Qi.begin(), Qi.end());
        Qx_all[k].assign(Qx.begin(), Qx.end());
        has_cell[k] = 1;
    }

    // Shared resolver: the requested count clamped by OMP_THREAD_LIMIT, the
    // environment's max threads, the check-farm core cap and the work count.
    const int nthr = tulpa_omp_team_size_req(n_threads, n_grid);

    // One CHOLMOD context per thread (common workspace is not thread-safe).
    std::vector<std::unique_ptr<tulpa::SparseCholeskySolver>> pool(nthr);
    for (int t = 0; t < nthr; t++)
        pool[t] = std::make_unique<tulpa::SparseCholeskySolver>();

    std::vector<std::vector<double>> results(n_grid);
    std::vector<char> ok(n_grid, 0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nthr)
#endif
    for (int k = 0; k < n_grid; k++) {
        if (!has_cell[k]) continue;
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        const int nnz = static_cast<int>(Qx_all[k].size());
        bool good = tulpa::extract_inner_vcov_block_cell(
            Qp_all[k].data(), Qi_all[k].data(), Qx_all[k].data(), n_x, nnz,
            idx0, n_dense, A_cols, field_marginal,
            *pool[tid], results[k]);
        ok[k] = good ? 1 : 0;
    }

    // POD -> Rcpp (single-threaded).
    Rcpp::List out(n_grid);
    for (int k = 0; k < n_grid; k++) {
        if (!ok[k]) { out[k] = R_NilValue; continue; }
        Rcpp::NumericMatrix Mk(p, p);
        std::copy(results[k].begin(), results[k].end(), Mk.begin());
        out[k] = Mk;
    }
    return out;
}
