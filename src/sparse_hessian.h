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
    SparseCholeskySolver* shared_solver = nullptr
) {
    LaplaceResult result;
    result.mode = Rcpp::NumericVector(n_x, 0.0);
    result.converged = false;
    result.n_iter = 0;
    result.log_det_Q = 0.0;
    result.log_marginal = 0.0;

    Rcpp::NumericVector x(n_x, 0.0);
    if (x_init.size() == n_x) {
        for (int j = 0; j < n_x; j++) x[j] = x_init[j];
    }

    SparseCholeskySolver local_solver;
    SparseCholeskySolver& solver = shared_solver ? *shared_solver : local_solver;

    #ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    #endif

    for (int iter = 0; iter < max_iter; iter++) {
        Rcpp::NumericVector eta(N, 0.0);
        compute_eta(x, eta);

        // Gradient (dense — it's only n_x entries, always manageable)
        DenseVec grad(n_x, 0.0);

        // Hessian (sparse!)
        H_builder.zero();
        scatter_sparse(x, eta, grad, H_builder);

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

        // Step halving on penalized deviance
        auto eval_obj = [&](const Rcpp::NumericVector& xv) -> double {
            Rcpp::NumericVector eta_tmp(N, 0.0);
            compute_eta(xv, eta_tmp);
            return compute_total_log_lik(y, n_trials, eta_tmp, N, family, phi, n_threads)
                   + compute_log_prior(xv, eta_tmp);
        };

        double obj_old = eval_obj(x);
        double step_scale = 1.0;
        for (int half = 0; half <= MAX_HALVING; half++) {
            Rcpp::NumericVector x_try(n_x);
            for (int j = 0; j < n_x; j++) x_try[j] = x[j] + step_scale * delta[j];
            double obj_try = eval_obj(x_try);
            if (obj_try >= obj_old - 1e-8 || half == MAX_HALVING) {
                for (int j = 0; j < n_x; j++) x[j] = x_try[j];
                break;
            }
            step_scale *= 0.5;
        }

        double max_delta = 0.0;
        for (int j = 0; j < n_x; j++) {
            max_delta = std::max(max_delta, std::abs(step_scale * delta[j]));
        }

        result.n_iter = iter + 1;
        if (max_delta < tol) {
            result.converged = true;
            break;
        }
    }

    center_effects_fn(x);
    result.mode = x;

    // Finalize: log-det and log-marginal
    Rcpp::NumericVector eta_final(N, 0.0);
    compute_eta(x, eta_final);

    DenseVec grad_final(n_x, 0.0);
    H_builder.zero();
    scatter_sparse(x, eta_final, grad_final, H_builder);

    cholmod_sparse H_final = H_builder.as_cholmod(&solver.common());
    if (!solver.analyzed()) solver.analyze(&H_final);
    if (solver.factorize(&H_final)) {
        result.log_det_Q = solver.log_determinant();
    }

    double log_lik = compute_total_log_lik(y, n_trials, eta_final, N, family, phi, n_threads);
    double log_prior = compute_log_prior(x, eta_final);

    result.log_marginal = log_lik + log_prior
                          - 0.5 * result.log_det_Q
                          + 0.5 * n_x * std::log(2.0 * M_PI);

    return result;
}

} // namespace tulpa

#endif // TULPA_SPARSE_HESSIAN_H
