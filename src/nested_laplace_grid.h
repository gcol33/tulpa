// nested_laplace_grid.h
// Generic outer-grid driver for nested Laplace.
//
// Each backend (ICAR, BYM2, SPDE, RW1, RW2, AR1, ...) provides a single-point
// solve function `solve_at_theta(k, prev_mode) -> LaplaceResult` that fixes
// the hyperparameters at grid index k, runs an inner Newton-Laplace, and
// returns the result. This driver loops over the grid, warm-starts the next
// inner solve from the previous mode, and aggregates log-marginals + modes.
//
// Pulls the outer-grid boilerplate out of the per-backend
// cpp_nested_laplace_* exports so adding a new backend is just a new
// solve_at_theta lambda + a thin Rcpp entry.

#ifndef TULPA_NESTED_LAPLACE_GRID_H
#define TULPA_NESTED_LAPLACE_GRID_H

#include "laplace_core.h"
#include <Rcpp.h>

namespace tulpa {

// Run a nested Laplace outer grid loop.
//
// SolveAtTheta is any callable with signature
//   LaplaceResult(int k, const Rcpp::NumericVector& prev_mode)
// returning the inner Laplace result at grid point k. The driver handles:
//   - allocating per-grid-point output (log_marginal, n_iter, mode row)
//   - threading prev_mode forward as a warm-start
//
// Caller is responsible for any backend-specific output entries (e.g. the
// theta grid itself). This function only emits the universal block:
//   log_marginal[n_grid], n_iter[n_grid], modes[n_grid x n_x], n_grid
template<typename SolveAtTheta>
inline Rcpp::List run_nested_laplace_grid(
    int n_grid, int n_x,
    SolveAtTheta solve_at_theta,
    Rcpp::NumericVector x_init = Rcpp::NumericVector(),
    bool store_modes = true
) {
    Rcpp::NumericVector log_marginals(n_grid);
    Rcpp::IntegerVector n_iters(n_grid);
    int mode_rows = store_modes ? n_grid : 0;
    Rcpp::NumericMatrix all_modes(mode_rows, store_modes ? n_x : 0);

    // Per-grid Q in CSC lower-triangle. Populated only when the per-point
    // solve returns a result with Q_csc_n > 0 (caller opted in via store_Q
    // on its inner laplace_newton_solve call). Each Q_k may have a different
    // nnz pattern, so we keep them as a List of three IntegerVector /
    // NumericVector triples rather than flattening here.
    Rcpp::List Q_p_per_grid(n_grid);
    Rcpp::List Q_i_per_grid(n_grid);
    Rcpp::List Q_x_per_grid(n_grid);
    bool any_Q = false;

    Rcpp::NumericVector prev_mode = x_init;

    for (int k = 0; k < n_grid; k++) {
        LaplaceResult res = solve_at_theta(k, prev_mode);

        log_marginals[k] = res.log_marginal;
        n_iters[k] = res.n_iter;
        if (store_modes) {
            for (int j = 0; j < n_x; j++) all_modes(k, j) = res.mode[j];
        }
        if (res.Q_csc_n > 0) {
            any_Q = true;
            Q_p_per_grid[k] = Rcpp::IntegerVector(
                res.Q_csc_p.begin(), res.Q_csc_p.end());
            Q_i_per_grid[k] = Rcpp::IntegerVector(
                res.Q_csc_i.begin(), res.Q_csc_i.end());
            Q_x_per_grid[k] = Rcpp::NumericVector(
                res.Q_csc_x.begin(), res.Q_csc_x.end());
        }
        prev_mode = res.mode;
    }

    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("log_marginal") = log_marginals,
        Rcpp::Named("n_iter") = n_iters,
        Rcpp::Named("n_grid") = n_grid
    );
    if (store_modes) out["modes"] = all_modes;
    if (any_Q) {
        out["Q_csc_p_per_grid"] = Q_p_per_grid;
        out["Q_csc_i_per_grid"] = Q_i_per_grid;
        out["Q_csc_x_per_grid"] = Q_x_per_grid;
        out["Q_csc_n"] = n_x;
    }
    return out;
}

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_GRID_H
