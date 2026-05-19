// nested_laplace_grid.h
// Generic outer-grid driver for nested Laplace.
//
// Each backend (ICAR, BYM2, SPDE, RW1, RW2, AR1, ...) provides a single-point
// solve function `solve_at_theta(k, prev_mode, solver)` that fixes the
// hyperparameters at grid index k, runs an inner Newton-Laplace using the
// supplied CHOLMOD solver, and returns the result. This driver loops over the
// grid, warm-starts the next inner solve from the previous mode, and
// aggregates log-marginals + modes.
//
// Parallel mode: when `n_threads_outer > 1` the driver
//   1) runs a pilot solve at the grid centre cell (serial),
//   2) allocates one SparseCholeskySolver per outer thread (CHOLMOD common
//      workspace is not thread-safe — each thread needs its own),
//   3) dispatches the remaining cells across an OpenMP parallel for, with
//      each cell warm-started from the pilot mode.
//
// Optional tile pilot (Phase 2): when `tile_ids` and `tile_pilot_cells`
// are provided (both non-empty), the parallel path runs a three-tier warm-
// start:
//   1) global pilot at the centre cell (Tier 1),
//   2) one warm solve per tile from the global pilot (Tier 2, parallel),
//   3) every remaining cell warm-started from its tile pilot mode
//      (Tier 3, parallel).
// Tiles group cells whose donor-arm latent prior + design coincide; for the
// joint copy block under the (sigma, alpha) reparameterisation this is
// every coordinate except alpha, so each tile's cells differ only in the
// copy-arm scaling. The tile pilot's mode is therefore a much better warm-
// start than the global pilot for the inner Newton, especially at boundary
// alpha cells. Caller is responsible for the partition: `tile_ids[k]` is
// the tile id of cell k (0-based, contiguous), and `tile_pilot_cells[t]`
// is the cell index used as tile t's representative (typically the cell
// closest to the median copy coefficient within the tile).
//
// Result post-processing (filling the Rcpp::List) happens single-threaded
// after the parallel region. Per-cell LaplaceResult objects use only
// std::vector storage (laplace_core.h:LaplaceResult after the gcol33/tulpa
// speedup refactor), so they are safe to populate across threads.

#ifndef TULPA_NESTED_LAPLACE_GRID_H
#define TULPA_NESTED_LAPLACE_GRID_H

#include "laplace_core.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <limits>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// Run a nested Laplace outer grid loop.
//
// SolveAtTheta is any callable with signature
//   LaplaceResult(int k, const std::vector<double>& prev_mode,
//                 SparseCholeskySolver* solver)
// returning the inner Laplace result at grid point k using `solver` as the
// CHOLMOD context (caller must not capture its own shared solver; the driver
// allocates one per outer-loop thread and threads the right one in).
//
// Threading
// ---------
// n_threads_outer = 1 (default): serial loop with mode chaining
//   (prev_mode <- res.mode after each cell). Bitwise compatible with the
//   pre-refactor driver.
// n_threads_outer > 1: pilot solve at the centre cell first, then
//   #pragma omp parallel for over all cells warm-started from the pilot mode.
//   Pilot cell's result is reused (no double-solve).
template<typename SolveAtTheta>
inline Rcpp::List run_nested_laplace_grid(
    int n_grid, int n_x,
    SolveAtTheta solve_at_theta,
    Rcpp::NumericVector x_init = Rcpp::NumericVector(),
    bool store_modes = true,
    int n_threads_outer = 1,
    const std::vector<int>& tile_ids = std::vector<int>(),
    const std::vector<int>& tile_pilot_cells = std::vector<int>()
) {
    Rcpp::NumericVector log_marginals(n_grid);
    Rcpp::IntegerVector n_iters(n_grid);
    int mode_rows = store_modes ? n_grid : 0;
    Rcpp::NumericMatrix all_modes(mode_rows, store_modes ? n_x : 0);

    // Per-grid Q in CSC lower-triangle. Populated only when the per-point
    // solve returns a result with Q_csc_n > 0 (caller opted in via store_Q
    // on its inner laplace_newton_solve call). Each Q_k may have a different
    // nnz pattern, so we keep them as a List of three IntegerVector /
    // NumericVector triples rather than flattening here. The Lists are
    // *pre-allocated single-threaded*; threads only set slot k via copies
    // out of the per-cell LaplaceResult after the parallel region.
    Rcpp::List Q_p_per_grid(n_grid);
    Rcpp::List Q_i_per_grid(n_grid);
    Rcpp::List Q_x_per_grid(n_grid);

    if (n_grid <= 0) {
        Rcpp::List out = Rcpp::List::create(
            Rcpp::Named("log_marginal") = log_marginals,
            Rcpp::Named("n_iter") = n_iters,
            Rcpp::Named("n_grid") = n_grid
        );
        if (store_modes) out["modes"] = all_modes;
        return out;
    }

    // Stage results in POD containers; merge to Rcpp slots after parallelism.
    std::vector<LaplaceResult> cell_results(n_grid);

    std::vector<double> x_init_vec;
    if (x_init.size() == n_x) {
        x_init_vec.assign(x_init.begin(), x_init.end());
    }

    if (n_threads_outer <= 1) {
        // Serial path: chained warm-starts as before.
        SparseCholeskySolver solver;
        std::vector<double> prev_mode = x_init_vec;
        for (int k = 0; k < n_grid; k++) {
            cell_results[k] = solve_at_theta(k, prev_mode, &solver);
            prev_mode = cell_results[k].mode;
        }
    } else {
        // Parallel path: pilot solve at the centre cell, then parallel fan-out.
        const int k_pilot = n_grid / 2;
        SparseCholeskySolver pilot_solver;
        cell_results[k_pilot] =
            solve_at_theta(k_pilot, x_init_vec, &pilot_solver);

        // Pilot mode is the warm-start for every other cell.
        std::vector<double> pilot_mode = cell_results[k_pilot].mode;
        // If the pilot itself diverged (rare — e.g. proper-CAR with rho
        // outside the PD interval at the centre cell), fall back to zero
        // warm-start so the parallel pass doesn't inherit garbage.
        if (!std::isfinite(cell_results[k_pilot].log_marginal)) {
            pilot_mode.assign(n_x, 0.0);
        }

        // One CHOLMOD solver per outer thread. CHOLMOD's cholmod_common is
        // *not* thread-safe — each thread must own its own. We use
        // unique_ptr so the solvers RAII-clean on scope exit even if a
        // thread bails on an exception.
        std::vector<std::unique_ptr<SparseCholeskySolver>> solver_pool(
            n_threads_outer);
        for (int t = 0; t < n_threads_outer; t++) {
            solver_pool[t].reset(new SparseCholeskySolver());
        }

        // Decide whether tile metadata is usable. Need both vectors set
        // and `tile_ids` of length n_grid; otherwise fall back to single-
        // tier Phase 1 behaviour.
        const bool use_tiles =
            (static_cast<int>(tile_ids.size()) == n_grid) &&
            (!tile_pilot_cells.empty());

        if (!use_tiles) {
            // Phase 1 path: every cell warm-started from the global pilot.
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1) \
                num_threads(n_threads_outer)
            #endif
            for (int k = 0; k < n_grid; k++) {
                if (k == k_pilot) continue;  // already solved
                #ifdef _OPENMP
                int tid = omp_get_thread_num();
                #else
                int tid = 0;
                #endif
                SparseCholeskySolver* solver = solver_pool[tid].get();
                cell_results[k] = solve_at_theta(k, pilot_mode, solver);
            }
        } else {
            // Phase 2 path: Tier-2 tile pilots (one per tile, warm-started
            // from the global pilot), then Tier-3 all remaining cells
            // warm-started from their tile pilot.
            const int n_tiles = static_cast<int>(tile_pilot_cells.size());

            // Flag tile-pilot cells so Tier-3 can skip them. Also build a
            // tile -> pilot-cell lookup that Tier-3 reads concurrently
            // (read-only after Tier-2 completes).
            std::vector<unsigned char> is_tile_pilot(n_grid, 0);
            for (int t = 0; t < n_tiles; t++) {
                int k = tile_pilot_cells[t];
                if (k >= 0 && k < n_grid) is_tile_pilot[k] = 1;
            }

            std::vector<std::vector<double>> tile_modes(n_tiles);

            // Tier 2: one warm solve per tile, parallel over tiles.
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1) \
                num_threads(n_threads_outer)
            #endif
            for (int t = 0; t < n_tiles; t++) {
                int k = tile_pilot_cells[t];
                if (k < 0 || k >= n_grid) continue;
                if (k == k_pilot) {
                    // Global pilot also serves as this tile's pilot.
                    tile_modes[t] = pilot_mode;
                    continue;
                }
                #ifdef _OPENMP
                int tid = omp_get_thread_num();
                #else
                int tid = 0;
                #endif
                SparseCholeskySolver* solver = solver_pool[tid].get();
                cell_results[k] = solve_at_theta(k, pilot_mode, solver);
                // Tile-pilot mode is the Tier-3 warm-start; if the cell
                // diverged, fall back to the global pilot mode so we don't
                // propagate garbage to the rest of the tile.
                tile_modes[t] = std::isfinite(cell_results[k].log_marginal)
                                ? cell_results[k].mode
                                : pilot_mode;
            }

            // Tier 3: every non-pilot cell warm-started from its tile pilot.
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1) \
                num_threads(n_threads_outer)
            #endif
            for (int k = 0; k < n_grid; k++) {
                if (k == k_pilot) continue;       // global pilot, Tier 1
                if (is_tile_pilot[k]) continue;   // tile pilot, Tier 2
                int t = tile_ids[k];
                if (t < 0 || t >= n_tiles) {
                    // Out-of-range tile id falls back to the global pilot.
                    #ifdef _OPENMP
                    int tid = omp_get_thread_num();
                    #else
                    int tid = 0;
                    #endif
                    cell_results[k] = solve_at_theta(
                        k, pilot_mode, solver_pool[tid].get());
                    continue;
                }
                #ifdef _OPENMP
                int tid = omp_get_thread_num();
                #else
                int tid = 0;
                #endif
                SparseCholeskySolver* solver = solver_pool[tid].get();
                cell_results[k] = solve_at_theta(k, tile_modes[t], solver);
            }
        }
    }

    // -------- Merge POD results into the Rcpp output (single-threaded) --------
    bool any_Q = false;
    for (int k = 0; k < n_grid; k++) {
        const LaplaceResult& res = cell_results[k];
        log_marginals[k] = res.log_marginal;
        n_iters[k] = res.n_iter;
        if (store_modes) {
            int copy_n = std::min(n_x, static_cast<int>(res.mode.size()));
            for (int j = 0; j < copy_n; j++) all_modes(k, j) = res.mode[j];
            // Any remaining slots (only hit if the inner solver returned a
            // shorter mode — should not happen) stay 0.
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
