// sparse_hessian.h
// Sparse Hessian builder for Newton-Raphson in Laplace approximation.
// Eliminates the O(n²) DenseMat bottleneck for large spatial models.
//
// Usage:
//   1. Construct with sparsity pattern (list of (row, col) pairs)
//   2. Each Newton iteration: zero(), scatter entries, factorize+solve via CHOLMOD
//
// The sparsity pattern is fixed across iterations. Only values change.

#ifndef TULPA_SPARSE_HESSIAN_H
#define TULPA_SPARSE_HESSIAN_H

#include "laplace_newton.h"        // NewtonScratch (shared with dense solver)
#include "laplace_newton_loop.h"   // shared eval_*, step_halving_update, finalize_log_marginal
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <vector>
#include <map>
#include <cmath>

namespace tulpa {

class SparseHessianBuilder {
public:
    int n;                          // dimension
    std::vector<int> col_ptr;       // CSC column pointers
    std::vector<int> row_idx;       // CSC row indices
    std::vector<double> values;     // CSC values (zeroed each iteration)
    int nnz;

    // Map from (row, col) to flat index in values array
    // For lower triangle only (symmetric, stype=-1)
    std::map<std::pair<int,int>, int> entry_map;

    // Initialize from a list of (row, col) pairs that define the sparsity pattern.
    // Only lower triangle entries needed (row >= col).
    void init(int dim, const std::vector<std::pair<int,int>>& pattern) {
        n = dim;

        // Deduplicate and sort by (col, row)
        std::set<std::pair<int,int>> unique_entries;
        for (auto& [r, c] : pattern) {
            int lo = std::min(r, c);
            int hi = std::max(r, c);
            unique_entries.insert({hi, lo});  // store as (row, col) with row >= col
        }

        // Build CSC (column-major, lower triangle)
        nnz = static_cast<int>(unique_entries.size());
        col_ptr.assign(n + 1, 0);
        row_idx.resize(nnz);
        values.resize(nnz, 0.0);

        // Sort by column then row
        struct Entry { int row, col; };
        std::vector<Entry> sorted;
        sorted.reserve(nnz);
        for (auto& [r, c] : unique_entries) {
            sorted.push_back({r, c});
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const Entry& a, const Entry& b) {
                      return a.col < b.col || (a.col == b.col && a.row < b.row);
                  });

        int cur_col = 0;
        for (int e = 0; e < nnz; e++) {
            while (cur_col <= sorted[e].col) {
                col_ptr[cur_col] = e;
                cur_col++;
            }
            row_idx[e] = sorted[e].row;
            entry_map[{sorted[e].row, sorted[e].col}] = e;
        }
        while (cur_col <= n) {
            col_ptr[cur_col] = nnz;
            cur_col++;
        }
    }

    // Zero all values (call at start of each Newton iteration)
    void zero() {
        std::fill(values.begin(), values.end(), 0.0);
    }

    // Add `ridge` to every diagonal entry. The CSC lower-triangle stores
    // each column's diagonal as the FIRST entry of the column (smallest
    // row index in the lower triangle is row == col), so the diagonal is
    // at `values[col_ptr[j]]` for column j with `row_idx[col_ptr[j]] == j`.
    // Used to apply the uniform upstream regularization (see
    // `LAPLACE_UNIFORM_RIDGE` in laplace_cholesky.h) after scatter and
    // before factorization.
    void add_uniform_ridge(double ridge) {
        for (int j = 0; j < n; j++) {
            const int idx = col_ptr[j];
            if (idx < col_ptr[j + 1] && row_idx[idx] == j) {
                values[idx] += ridge;
            }
        }
    }

    // Add value to entry (row, col). Handles symmetric storage (lower triangle).
    // If the entry doesn't exist in the pattern, it's silently dropped.
    void add(int row, int col, double val) {
        int lo = std::min(row, col);
        int hi = std::max(row, col);
        auto it = entry_map.find({hi, lo});
        if (it != entry_map.end()) {
            values[it->second] += val;
        }
    }

    // Create a cholmod_sparse view (does NOT allocate — reuses internal arrays).
    // The returned pointer is valid until the next init() or destruction.
    // Caller must NOT free the returned sparse matrix.
    cholmod_sparse as_cholmod(cholmod_common* common) const {
        cholmod_sparse A;
        A.nrow = n;
        A.ncol = n;
        A.nzmax = nnz;
        A.p = const_cast<int*>(col_ptr.data());
        A.i = const_cast<int*>(row_idx.data());
        A.x = const_cast<double*>(values.data());
        A.z = nullptr;
        A.stype = -1;   // lower triangle stored
        A.itype = CHOLMOD_INT;
        A.xtype = CHOLMOD_REAL;
        A.dtype = CHOLMOD_DOUBLE;
        A.sorted = 1;
        A.packed = 1;
        return A;
    }
};

// =====================================================================
// Sparse Newton solver: like laplace_newton_solve but with sparse H
// =====================================================================
//
// ComputeEta: same as before
// ScatterSparse: fn(x, eta, grad, SparseHessianBuilder&) — writes to sparse H
// CenterEffects: same
// ComputeLogPrior: same

// Scratch-aware sparse-Hessian Newton solver. Same scratch contract as the
// dense-Hessian solver in laplace_newton.h: caller owns NewtonScratch
// allocation outside the parallel region.
template<typename ComputeEta, typename ScatterSparse,
         typename CenterEffects, typename ComputeLogPrior>
LaplaceResult laplace_newton_solve_sparse(
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const std::string& family,
    double phi,
    int N, int n_x,
    int max_iter, double tol, int n_threads,
    ComputeEta compute_eta,
    ScatterSparse scatter_sparse,
    CenterEffects center_effects_fn,
    ComputeLogPrior compute_log_prior,
    SparseHessianBuilder& H_builder,
    NewtonScratch& scratch,
    const std::vector<double>& x_init,
    SparseCholeskySolver* shared_solver,
    bool store_Q
) {
    LaplaceResult result;
    result.mode.assign(n_x, 0.0);
    result.converged = false;
    result.n_iter = 0;
    result.log_det_Q = 0.0;
    result.log_marginal = 0.0;

    Rcpp::NumericVector& x = scratch.x;
    if (static_cast<int>(x_init.size()) == n_x) {
        for (int j = 0; j < n_x; j++) x[j] = x_init[j];
    } else {
        for (int j = 0; j < n_x; j++) x[j] = 0.0;
    }

    SparseCholeskySolver local_solver;
    SparseCholeskySolver& solver = shared_solver ? *shared_solver : local_solver;

    for (int iter = 0; iter < max_iter; iter++) {
        compute_eta(x, scratch.eta);

        // Gradient (dense — it's only n_x entries, always manageable)
        DenseVec grad(n_x, 0.0);

        // Hessian (sparse!)
        H_builder.zero();
        scatter_sparse(x, scratch.eta, grad, H_builder);

        // Uniform upstream ridge so the dense pivot-clamp and sparse-dbound
        // hacks aren't needed (see LAPLACE_UNIFORM_RIDGE in laplace_cholesky.h).
        H_builder.add_uniform_ridge(LAPLACE_UNIFORM_RIDGE);

        // Solve H * delta = grad via CHOLMOD
        cholmod_sparse H_cholmod = H_builder.as_cholmod(&solver.common());

        if (!solver.analyzed()) {
            solver.analyze(&H_cholmod);
        }

        std::vector<double> delta(n_x, 0.0);
        bool solve_ok = false;

        if (solver.factorize(&H_cholmod)) {
            solver.solve(grad.data(), delta.data(), n_x);
            solve_ok = true;
            for (int j = 0; j < n_x; j++) {
                if (!std::isfinite(delta[j])) { solve_ok = false; break; }
            }
        }

        if (!solve_ok) {
            for (int j = 0; j < n_x; j++) {
                if (std::isfinite(delta[j])) x[j] += 0.1 * delta[j];
            }
            result.n_iter = iter + 1;
            continue;
        }

        // Step halving on penalized deviance (shared helper).
        auto eval_obj = [&](const Rcpp::NumericVector& xv) -> double {
            return eval_penalized_log_lik(
                xv, y, n_trials, N, family, phi, n_threads,
                compute_eta, compute_log_prior, scratch.eta_tmp
            );
        };

        double obj_old = eval_obj(x);
        double obj_unused = 0.0;
        double step_scale = step_halving_update(
            x, delta, n_x, obj_old, eval_obj, obj_unused, scratch.x_try
        );

        result.n_iter = iter + 1;
        if (max_abs_step(delta, step_scale, n_x) < tol) {
            result.converged = true;
            break;
        }
    }

    center_effects_fn(x);
    for (int j = 0; j < n_x; j++) result.mode[j] = x[j];

    // Finalize: log-det and log-marginal. Reuse scratch.eta as eta_final.
    compute_eta(x, scratch.eta);

    DenseVec grad_final(n_x, 0.0);
    H_builder.zero();
    scatter_sparse(x, scratch.eta, grad_final, H_builder);
    H_builder.add_uniform_ridge(LAPLACE_UNIFORM_RIDGE);

    cholmod_sparse H_final = H_builder.as_cholmod(&solver.common());
    if (!solver.analyzed()) solver.analyze(&H_final);
    if (solver.factorize(&H_final)) {
        result.log_det_Q = solver.log_determinant();
    }

    double log_lik = compute_total_log_lik(y, n_trials, scratch.eta, N, family, phi, n_threads);
    double log_prior = compute_log_prior(x, scratch.eta);

    result.log_marginal = finalize_log_marginal(log_lik, log_prior, result.log_det_Q, n_x);

    if (store_Q) {
        // H_builder already holds Q in CSC lower-triangle form (stype = -1).
        // Copy out the arrays so the caller doesn't need to keep the builder
        // alive past Newton convergence.
        result.Q_csc_p = H_builder.col_ptr;
        result.Q_csc_i = H_builder.row_idx;
        result.Q_csc_x = H_builder.values;
        result.Q_csc_n = n_x;
    }

    return result;
}

// Convenience overload that allocates scratch locally. Use only outside
// parallel regions.
template<typename ComputeEta, typename ScatterSparse,
         typename CenterEffects, typename ComputeLogPrior>
LaplaceResult laplace_newton_solve_sparse(
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const std::string& family,
    double phi,
    int N, int n_x,
    int max_iter, double tol, int n_threads,
    ComputeEta compute_eta,
    ScatterSparse scatter_sparse,
    CenterEffects center_effects_fn,
    ComputeLogPrior compute_log_prior,
    SparseHessianBuilder& H_builder,
    const Rcpp::NumericVector& x_init = Rcpp::NumericVector(),
    SparseCholeskySolver* shared_solver = nullptr,
    bool store_Q = false
) {
    NewtonScratch scratch;
    scratch.allocate(n_x, N);
    std::vector<double> x_init_vec;
    if (x_init.size() == n_x) {
        x_init_vec.assign(x_init.begin(), x_init.end());
    }
    #ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    #endif
    return laplace_newton_solve_sparse(
        y, n_trials, family, phi, N, n_x,
        max_iter, tol, n_threads,
        compute_eta, scatter_sparse, center_effects_fn, compute_log_prior,
        H_builder, scratch, x_init_vec, shared_solver, store_Q
    );
}

} // namespace tulpa

#endif // TULPA_SPARSE_HESSIAN_H
