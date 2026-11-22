// nested_laplace_grid.cpp
// Definitions of the outer-grid result packers declared in nested_laplace_grid.h.

#include <RcppEigen.h>
#include "tulpa_stop.h"
#include "nested_laplace_grid.h"
#include "linear_constraint_rows.h"

namespace tulpa {

Rcpp::List nl_pack_grid_results(
    const std::vector<LaplaceResult>& cell_results, int n_grid, int n_x,
    bool store_modes, int n_threads_outer_realised
) {
    Rcpp::NumericVector log_marginals(n_grid);
    Rcpp::IntegerVector n_iters(n_grid);
    Rcpp::LogicalVector device_cpu_recovered(n_grid);
    // Per-cell solve health. log_det_Q and score_max are the inner solve's own
    // log|H| and achieved joint-score residual at the returned mode; `converged`
    // is its stopping-rule flag. Reported per cell so a grid fit can see which
    // cells settled, and so a fixed-hyperparameter export (a 1-cell grid) can
    // project this result onto the single-fit contract -- see
    // nl_grid_cell_to_result_list below.
    Rcpp::NumericVector log_det_Qs(n_grid);
    Rcpp::NumericVector score_maxs(n_grid);
    Rcpp::LogicalVector convergeds(n_grid);
    // The solve never started: the penalized objective was non-finite at the
    // latent start and no interior point was found. It is a flag rather than a
    // throw because the solve runs inside an OpenMP region, so the R side is
    // what turns it into an error -- and it can only do that if the grid
    // carries it per cell. nl_grid_cell_to_result_list reads this back, which
    // is what gives the one-cell exports (fit_spde, the GP entries) a flag to
    // report instead of a hardcoded false.
    Rcpp::LogicalVector start_infeasibles(n_grid);
    // The sum-to-zero log-determinant fell back to the PD-enforced value at this
    // cell, so its weight in the outer integration was formed from a
    // determinant of H + lambda I rather than of the pinned matrix. Carried per
    // cell for the same reason start_infeasible is: only the R side can report
    // it, and only if the grid says which cells it happened at.
    Rcpp::LogicalVector s2z_fallbacks(n_grid);
    Rcpp::LogicalVector s2z_field_indefs(n_grid);
    Rcpp::LogicalVector pd_conditioneds(n_grid);
    // The Hessian at the cell's returned point was not positive definite: the
    // point is no mode, its log-determinant was read off a PD-enforced factor,
    // and its stored precision cannot be factored. The R side re-solves such a
    // cell and leaves it out of the integration when it stays indefinite.
    Rcpp::LogicalVector hessian_indefinites(n_grid);
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

    // Per-cell inverse block, when the per-point solve was asked for one (the
    // joint tier's fixed-effect retention). O(m^2) per cell
    // and flat in the field size, so this is what a caller keeps instead of a
    // grid's worth of precisions. Every current requester asks for exactly one
    // block, so slot k holds that block as an m x m matrix.
    Rcpp::List cov_block_per_grid(n_grid);
    // The pins each stored Q leaves off: their k x k precision per cell (NULL
    // where the cell's Q holds every pin) and the rows, which the pattern fixes
    // for the whole fit.
    Rcpp::List Q_soft_D_per_grid(n_grid);
    Rcpp::List Q_soft_cols;
    bool any_soft = false;

    bool any_Q = false;
    bool any_cov_block = false;
    int skew_cell = -1;  // first grid cell (if any) that computed inner_skew
    // Subspace-debias draws are PER CELL: the correction enters the reported
    // marginal as a mixture over the outer grid, so unlike the single-cell
    // diagnostics above every integrated cell contributes its own sample.
    int debias_cell = -1;
    Rcpp::List debias_per_grid(n_grid);
    Rcpp::List debias_importance(n_grid);
    Rcpp::List debias_importance_log_weights(n_grid);
    Rcpp::NumericVector debias_accept(n_grid, NA_REAL);
    Rcpp::CharacterVector debias_declined(n_grid, NA_STRING);
    // Corrected integrated Laplace (inner_cila.h) is per cell for the same
    // reason: the corrected marginal is a reweighting of every cell's own
    // particle set, so all of them travel out.
    bool any_cila = false;
    Rcpp::List cila_lw_per_grid(n_grid);
    Rcpp::List cila_fixed_per_grid(n_grid);
    Rcpp::NumericVector cila_log_marginal(n_grid, NA_REAL);
    Rcpp::IntegerVector cila_n_points(n_grid, NA_INTEGER);
    Rcpp::IntegerVector cila_variant(n_grid, NA_INTEGER);
    Rcpp::CharacterVector cila_declined(n_grid, NA_STRING);
    Rcpp::CharacterVector cila_fallback(n_grid, NA_STRING);
    for (int k = 0; k < n_grid; k++) {
        const LaplaceResult& res = cell_results[k];
        log_marginals[k] = res.log_marginal;
        n_iters[k] = res.n_iter;
        device_cpu_recovered[k] = res.device_cpu_recovered;
        log_det_Qs[k] = res.log_det_Q;
        score_maxs[k] = res.score_max;
        convergeds[k] = res.converged;
        start_infeasibles[k] = res.start_infeasible;
        s2z_fallbacks[k] = res.s2z_log_det_fallback;
        s2z_field_indefs[k] = res.s2z_field_indefinite;
        pd_conditioneds[k] = res.pd_conditioned;
        hessian_indefinites[k] = !res.hessian_pd_at_mode;
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
            const int ks = static_cast<int>(res.Q_soft_ptr.size()) - 1;
            if (ks > 0 && res.Q_soft_D.size() == static_cast<std::size_t>(ks) * ks) {
                Rcpp::NumericMatrix Dk(ks, ks);
                for (int i = 0; i < ks; i++)
                    for (int j = 0; j < ks; j++)
                        Dk(i, j) = res.Q_soft_D[static_cast<std::size_t>(i) * ks + j];
                Q_soft_D_per_grid[k] = Dk;
                if (!any_soft) {
                    std::vector<LinearConstraintRow> rows(ks);
                    for (int g = 0; g < ks; g++)
                        rows[g].cols.assign(res.Q_soft_idx.begin() + res.Q_soft_ptr[g],
                                            res.Q_soft_idx.begin() + res.Q_soft_ptr[g + 1]);
                    Q_soft_cols = write_constraint_rows(rows);
                    any_soft = true;
                }
            }
        }
        if (!res.re_cov_block_sizes.empty()) {
            const int m = res.re_cov_block_sizes[0];
            if (m > 0 && static_cast<int>(res.re_cov_flat.size()) >= m * m) {
                Rcpp::NumericMatrix Bk(m, m);
                for (int e = 0; e < m * m; e++) Bk[e] = res.re_cov_flat[e];
                cov_block_per_grid[k] = Bk;
                any_cov_block = true;
            }
        }
        if (skew_cell < 0 && !res.inner_skew_idx.empty()) skew_cell = k;
        if (!res.debias_idx.empty()) {
            if (debias_cell < 0) debias_cell = k;
            debias_accept[k] = res.debias_accept;
            if (!res.debias_importance.empty())
                debias_importance[k] = Rcpp::wrap(res.debias_importance);
            if (!res.debias_importance_log_weights.empty())
                debias_importance_log_weights[k] = Rcpp::wrap(res.debias_importance_log_weights);
            const int q = static_cast<int>(res.debias_idx.size());
            const int S = res.debias_n_kept;
            if (S > 0 && res.debias_draws.size() ==
                             static_cast<std::size_t>(S) * q) {
                Rcpp::NumericMatrix dr(S, q);
                for (std::size_t e = 0; e < res.debias_draws.size(); e++) {
                    dr[e] = res.debias_draws[e];
                }
                debias_per_grid[k] = dr;
            } else {
                debias_declined[k] = res.debias_declined;
            }
        }
        if (res.cila_requested) {
            any_cila = true;
            cila_log_marginal[k] = res.cila_log_marginal;
            cila_n_points[k] = res.cila_n_points;
            cila_variant[k] = res.cila_variant;
            if (!res.cila_declined.empty()) cila_declined[k] = res.cila_declined;
            if (!res.cila_fallback.empty()) cila_fallback[k] = res.cila_fallback;
            if (!res.cila_log_w.empty()) {
                cila_lw_per_grid[k] = Rcpp::NumericVector(
                    res.cila_log_w.begin(), res.cila_log_w.end());
            }
            const int M = res.cila_n_points;
            const int p = res.cila_n_fixed;
            if (M > 0 && p > 0 &&
                res.cila_fixed.size() == static_cast<std::size_t>(M) *
                                         static_cast<std::size_t>(p)) {
                Rcpp::NumericMatrix fx(M, p);
                for (std::size_t e = 0; e < res.cila_fixed.size(); e++) {
                    fx[e] = res.cila_fixed[e];
                }
                cila_fixed_per_grid[k] = fx;
            }
        }
    }

    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("log_marginal") = log_marginals,
        Rcpp::Named("n_iter") = n_iters,
        Rcpp::Named("n_grid") = n_grid
    );
    out["log_det_Q"] = log_det_Qs;
    out["device_cpu_recovered"] = device_cpu_recovered;
    out["score_max"] = score_maxs;
    out["converged"] = convergeds;
    out["start_infeasible"] = start_infeasibles;
    out["s2z_log_det_fallback"] = s2z_fallbacks;
    out["s2z_field_indefinite"] = s2z_field_indefs;
    out["pd_conditioned"] = pd_conditioneds;
    out["hessian_indefinite"] = hessian_indefinites;
    if (n_threads_outer_realised >= 0)
        out["n_threads_outer_realised"] = n_threads_outer_realised;
    if (store_modes) out["modes"] = all_modes;
    if (any_Q) {
        out["Q_csc_p_per_grid"] = Q_p_per_grid;
        out["Q_csc_i_per_grid"] = Q_i_per_grid;
        out["Q_csc_x_per_grid"] = Q_x_per_grid;
        out["Q_csc_n"] = n_x;
        if (any_soft) {
            out["Q_soft_cols"] = Q_soft_cols;
            out["Q_soft_D_per_grid"] = Q_soft_D_per_grid;
        }
    }
    if (any_cov_block) out["cov_block_per_grid"] = cov_block_per_grid;
    auto attach_eta = [&](const char* name, std::vector<double> LaplaceResult::* field) {
        std::size_t n_rows = 0;
        for (const auto& cell : cell_results) {
            if (!(cell.*field).empty()) { n_rows = (cell.*field).size(); break; }
        }
        if (n_rows == 0) return;
        Rcpp::NumericMatrix values(n_grid, n_rows);
        std::fill(values.begin(), values.end(), NA_REAL);
        for (int k = 0; k < n_grid; ++k) {
            const auto& rows = cell_results[k].*field;
            if (rows.size() != n_rows) continue;
            for (std::size_t r = 0; r < n_rows; ++r) values(k, r) = rows[r];
        }
        out[name] = values;
    };
    if (store_modes) attach_eta("fitted_eta", &LaplaceResult::eta_mean);
    attach_eta("fitted_eta_var", &LaplaceResult::eta_var);
    // Inner-Laplace skewness diagnostic (opt-in, computed on the full solve
    // of a single cell -- see the compute_skew doc on
    // run_multi_block_nested_laplace). Emits the FIRST cell that populated
    // it; the intended caller always passes a length-1 grid when requesting
    // this, so there is only ever one cell to report.
    if (skew_cell >= 0) {
        const LaplaceResult& res = cell_results[skew_cell];
        Rcpp::IntegerVector idx_r(res.inner_skew_idx.size());
        for (std::size_t k = 0; k < res.inner_skew_idx.size(); k++) {
            idx_r[k] = res.inner_skew_idx[k] + 1;
        }
        out["inner_skew"] = res.inner_skew;
        out["inner_skew_gamma1"] = res.inner_skew_gamma1;
        out["inner_skew_gamma1_declined"] = res.inner_skew_gamma1_declined;
        out["inner_skew_idx"] = idx_r;
        out["inner_skew_dropped"] = res.inner_skew_dropped;
        out["inner_skew_declined"] = res.inner_skew_declined;
        if (!res.inner_skew_arms_declined.empty()) {
            Rcpp::IntegerVector arms_r(res.inner_skew_arms_declined.size());
            for (std::size_t k = 0; k < res.inner_skew_arms_declined.size(); k++) {
                arms_r[k] = res.inner_skew_arms_declined[k] + 1;
            }
            out["inner_skew_arms_declined"] = arms_r;
        }
        out["inner_skew_cell"] = skew_cell + 1;
        attach_inner_is_fields(out, res);
    }
    if (debias_cell >= 0) {
        const LaplaceResult& res = cell_results[debias_cell];
        Rcpp::IntegerVector idx_r(res.debias_idx.size());
        for (std::size_t k = 0; k < res.debias_idx.size(); k++) {
            idx_r[k] = res.debias_idx[k] + 1;
        }
        out["debias_idx"] = idx_r;
        out["debias_draws_per_grid"] = debias_per_grid;
        out["debias_importance_per_grid"] = debias_importance;
        out["debias_importance_log_weights_per_grid"] = debias_importance_log_weights;
        out["debias_accept"] = debias_accept;
        out["debias_declined"] = debias_declined;
    }
    if (any_cila) {
        out["cila_log_w_per_grid"]  = cila_lw_per_grid;
        out["cila_fixed_per_grid"]  = cila_fixed_per_grid;
        out["cila_log_marginal"]    = cila_log_marginal;
        out["cila_n_points"]        = cila_n_points;
        out["cila_variant"]         = cila_variant;
        out["cila_declined"]        = cila_declined;
        out["cila_fallback"]        = cila_fallback;
    }
    return out;
}

Rcpp::List nl_grid_cell_to_result_list(const Rcpp::List& grid, int k) {
    Rcpp::NumericMatrix modes = grid["modes"];
    if (k < 0 || k >= modes.nrow()) {
        tulpa::stop("nl_grid_cell_to_result_list: cell %d out of range "
                   "[0, %d).", {k, static_cast<int>(modes.nrow())});
    }
    const int n_x = modes.ncol();
    Rcpp::NumericVector mode(n_x);
    for (int j = 0; j < n_x; j++) mode[j] = modes(k, j);

    Rcpp::NumericVector log_marginal = grid["log_marginal"];
    Rcpp::IntegerVector n_iter       = grid["n_iter"];
    Rcpp::NumericVector log_det_Q    = grid["log_det_Q"];
    Rcpp::NumericVector score_max    = grid["score_max"];
    Rcpp::LogicalVector converged    = grid["converged"];

    // A grid that carries the per-cell flag reports it; one that does not (an
    // older driver) reports false, which is what this used to hardcode for
    // every caller. The flag is the only signal an infeasible start leaves, so
    // hardcoding it made the one-cell exports unable to raise the error the R
    // side exists to raise.
    bool cell_infeasible = false;
    if (grid.containsElementNamed("start_infeasible")) {
        Rcpp::LogicalVector si = grid["start_infeasible"];
        if (k < si.size()) cell_infeasible = static_cast<bool>(si[k]);
    }

    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("mode")             = mode,
        Rcpp::Named("log_det_Q")        = log_det_Q[k],
        Rcpp::Named("log_marginal")     = log_marginal[k],
        Rcpp::Named("n_iter")           = n_iter[k],
        Rcpp::Named("converged")        = static_cast<bool>(converged[k]),
        Rcpp::Named("score_max")        = score_max[k],
        Rcpp::Named("start_infeasible") = cell_infeasible
    );
    if (grid.containsElementNamed("device_cpu_recovered")) {
        Rcpp::LogicalVector recovered = grid["device_cpu_recovered"];
        out["device_cpu_recovered"] = static_cast<bool>(recovered[k]);
    }

    // Inner-Laplace skewness diagnostic, under the names the single-fit
    // contract uses (laplace_result_to_list). The grid emits it for the one
    // cell that computed it, which on a fixed-hyperparameter run is this cell.
    if (grid.containsElementNamed("inner_skew")) {
        out["inner_skew"]     = grid["inner_skew"];
        out["inner_skew_idx"] = grid["inner_skew_idx"];
        if (grid.containsElementNamed("inner_skew_dropped")) {
            out["inner_skew_dropped"] = grid["inner_skew_dropped"];
        }
        if (grid.containsElementNamed("inner_skew_declined")) {
            out["inner_skew_declined"] = grid["inner_skew_declined"];
        }
        if (grid.containsElementNamed("inner_skew_arms_declined")) {
            out["inner_skew_arms_declined"] = grid["inner_skew_arms_declined"];
        }
        for (const char* nm : {"inner_skew_gamma1", "inner_skew_gamma1_declined"}) {
            if (grid.containsElementNamed(nm)) out[nm] = grid[nm];
        }
    }
    // Inner-Laplace importance curve, same projection.
    for (const char* nm : {"inner_is_z", "inner_is_sigma", "inner_is_declined",
                           "inner_is_log_joint"}) {
        if (grid.containsElementNamed(nm)) out[nm] = grid[nm];
    }
    return out;
}

Rcpp::List run_nested_laplace_grid(
    int n_grid, int n_x,
    const NlCellSolve& solve_at_theta_fn,
    Rcpp::NumericVector x_init,
    bool store_modes,
    int n_threads_outer,
    const std::vector<int>& tile_ids,
    const std::vector<int>& tile_pilot_cells,
    const NlCheapEval& cheap_eval,
    double prune_tol,
    tulpa_progress::GridProgress* progress,
    GridCheckpoint* ckpt,
    const std::vector<double>& x_init_per_cell,
    int screen_iters,
    const NlResumeRefill& resume_refill,
    const ScreenOffset& screen_log_offset,
    bool screen_only
) {
    if (!screen_log_offset.empty() &&
        static_cast<int>(screen_log_offset.size()) != n_grid) {
        tulpa::stop("screen_log_offset has %d entries for a grid of %d cells.",
                   {static_cast<int>(screen_log_offset.size()), n_grid});
    }
    const auto screen_off = [&](int k) -> double {
        return screen_log_offset.empty() ? 0.0 : screen_log_offset[k];
    };
    // Every full cell solve below -- pilot, serial chain, parallel tiles --
    // goes through this wrapper, so tulpa_profile() counts one outer_grid_cell
    // per solved cell whichever branch ran it. The scope encloses the inner
    // Newton phases the solve itself records, and runs on the worker thread
    // around the whole solve rather than inside any worksharing loop.
    device::BatchScope device_batch(tulpa_omp_team_size_req(n_threads_outer, std::max(1, n_grid)));
    std::vector<unsigned char> cell_cpu_recovery(std::max(0, n_grid), 0);
    auto solve_at_theta = [&solve_at_theta_fn, &device_batch, &cell_cpu_recovery](
        int k, const std::vector<double>& warm, SparseCholeskySolver* solver) {
        TULPA_PROFILE_PHASE(PHASE_OUTER_CELL);
        auto result = device::run_cell(device_batch,
            [&] { return solve_at_theta_fn(k, warm, solver); }, "outer_cell_cpu_recovery");
        result.device_cpu_recovered = result.device_cpu_recovered || cell_cpu_recovery[k];
        cell_cpu_recovery[k] = result.device_cpu_recovered;
        return result;
    };
    auto cheap_at = [&](int k, const std::vector<double>& warm, int steps, int worker) {
        auto result = device::run_cell(device_batch,
            [&] { return cheap_eval(k, warm, steps, worker); }, "outer_screen_cpu_recovery");
        cell_cpu_recovery[k] = cell_cpu_recovery[k] || result.device_cpu_recovered;
        return result;
    };
    // At least one step: a zero-step screen would rank every cell at the
    // warm-start it inherited rather than at its own quasi-mode.
    const int screen_steps = std::max(1, screen_iters);
    // Tile partition, validated once. A cell whose tile id falls outside
    // [0, n_tiles) joins no tile: the cheap screen never reaches it, its
    // cheap_lm stays -inf, and it is pruned out of the posterior with
    // log_marginal = -Inf. The caller owns the partition, so an out-of-range
    // id is its error and is reported instead of deleting the cell.
    if (!tile_ids.empty() && !tile_pilot_cells.empty()) {
        const int n_tiles_declared = static_cast<int>(tile_pilot_cells.size());
        for (std::size_t k = 0; k < tile_ids.size(); k++) {
            if (tile_ids[k] < 0 || tile_ids[k] >= n_tiles_declared) {
                tulpa::stop("tile_ids[%d] = %d is outside the %d tiles declared "
                           "by tile_pilot_cells.",
                           {static_cast<int>(k) + 1, tile_ids[k],
                           n_tiles_declared});
            }
        }
    }

    if (n_grid <= 0) {
        return nl_pack_grid_results(std::vector<LaplaceResult>(), 0, n_x,
                                    store_modes, -1);
    }

    // Clamp the outer width to the environment (OMP_NUM_THREADS via
    // omp_get_max_threads(), OMP_THREAD_LIMIT) and the grid size before it is
    // used. The per-cell block cache (NlCellCache) sizes its slot array from
    // omp_get_max_threads(); a num_threads(n) clause is honored even when n
    // exceeds that (e.g. OMP_NUM_THREADS=1 on a cluster with n_threads_outer >
    // 1), which would map every excess worker onto slot 0 and corrupt the
    // shared CHOLMOD factor. Clamping here keeps every thread id within the
    // cache's slot count; the memory-budget clamp the caller applied still
    // bounds it from above.
    n_threads_outer = tulpa_omp_team_size_req(n_threads_outer, n_grid);

    // The realised outer width is the driver's own property: every caller hands
    // it the concurrency it will actually run at (n_outer, already clamped by
    // the sparse path's memory budget). Stamp it on the reporter here, once, so
    // the ETA projects over the parallel grid rather than the serial pilot rate
    // and the console line shows the active outer-thread count.
    if (progress) progress->set_width(n_threads_outer);
    // The same number on the result, because the progress line is not a record:
    // its thread suffix is printed only above one thread, so a request clamped
    // to one and a serial run by design print the same line. A benchmark
    // recording a per-cell time needs the width the grid actually ran at.
    const int n_threads_outer_realised = n_threads_outer;

    // Stage results in POD containers; merge to Rcpp slots after parallelism.
    std::vector<LaplaceResult> cell_results(n_grid);

    std::vector<double> x_init_vec;
    if (x_init.size() == n_x) {
        x_init_vec.assign(x_init.begin(), x_init.end());
    }

    // Optional PER-CELL warm start: a flat row-major
    // n_grid x n_x matrix where row k warm-starts cell k's inner solve. When
    // present it OVERRIDES the chained / pilot / tile warm start in every path,
    // so each cell starts from its own near-mode (the outer Pareto-k diagnostic
    // passes each importance draw the converged mode of its nearest integration
    // cell -- a far better start than the single broadcast modal mode, and one
    // the parallel pilot-mode path cannot otherwise exploit). Empty (default)
    // reproduces the prior behaviour exactly. `pc_warm(k, fallback)` returns the
    // per-cell row when present, else binds the contextual fallback with no copy.
    const bool has_pc =
        (static_cast<long long>(x_init_per_cell.size()) ==
         static_cast<long long>(n_grid) * n_x);
    auto pc_row = [&](int k) {
        return std::vector<double>(
            x_init_per_cell.begin() + static_cast<size_t>(k) * n_x,
            x_init_per_cell.begin() + static_cast<size_t>(k + 1) * n_x);
    };

    // Checkpoint preload. A cell whose hyperparameter
    // coordinate was completed in a prior pass / killed run is filled straight
    // from disk and flagged done, so every solve site below skips it and the
    // warm-start chain reads its stored mode. `record(k)` appends a freshly
    // solved cell. Both are no-ops when `ckpt == nullptr`.
    std::vector<unsigned char> ckpt_done(n_grid, 0);
    int n_loaded = 0;
    if (ckpt) {
        for (int k = 0; k < n_grid; k++) {
            if (ckpt->has(k)) {
                cell_results[k] = ckpt->get(k);
                ckpt_done[k] = 1;
                n_loaded++;
            }
        }
        // State the payload cannot carry is rebuilt here, after the whole
        // preload so the hook sees a fully populated `cell_results`, and
        // serially against one solver -- the same shape the serial solve path
        // below takes. It costs one factorization at the restored mode per
        // loaded cell, against the full solve resume exists to avoid.
        nl_checkpoint_add_loaded(n_loaded);
        if (n_loaded && resume_refill) {
            SparseCholeskySolver refill_solver;
            for (int k = 0; k < n_grid; k++) {
                if (ckpt_done[k])
                    resume_refill(k, cell_results[k].mode, &refill_solver);
            }
        }
    }
    auto record = [&](int k) {
        if (ckpt && !ckpt_done[k]) ckpt->save(k, cell_results[k]);
    };

    // Exception barrier for the OpenMP cell loops below: an exception
    // escaping a parallel-for body is undefined behaviour (std::terminate
    // takes down the R session), so every worker body runs through
    // guard_cell, which records the first failure message and stamps the
    // cell infeasible; the main thread re-raises after each region.
    std::atomic<bool> cell_failed{false};
    std::string       cell_err;
    auto stamp_failed_cell = [&](int k) {
        LaplaceResult bad;
        bad.log_marginal = -std::numeric_limits<double>::infinity();
        bad.n_iter = 0;
        bad.converged = false;
        bad.log_det_Q = 0.0;
        cell_results[k] = bad;
    };
    auto guard_cell = [&](int k, auto&& body) {
        try {
            body();
        } catch (const std::exception& e) {
            if (!cell_failed.exchange(true)) cell_err = e.what();
            stamp_failed_cell(k);
        } catch (...) {
            cell_failed.exchange(true);
            stamp_failed_cell(k);
        }
    };
    auto raise_if_cell_failed = [&]() {
        if (cell_failed.load()) {
            tulpa::stop("nested-Laplace grid cell failed: %s",
                       {cell_err.empty() ? "unknown error" : cell_err.c_str()});
        }
    };

    // Determine whether we need an explicit pilot pass. Required either by
    // outer parallelism (the parallel branches use the pilot mode as the
    // warm-start for every other cell) or by cheap-pass pruning (the prune
    // step needs the pilot mode to evaluate cheap log-marginals). A caller
    // supplying every cell's own warm start (`x_init_per_cell`) leaves the
    // parallel branch nothing to read off a pilot, so an unscreened batch of
    // that kind fans out at once instead of solving one cell serially first
    // (gcol33/tulpa#920). When the
    // caller leaves cheap_eval empty, prune_tol is
    // ignored — the no-op cheap_eval would prune everything otherwise.
    const bool cheap_eval_supplied = static_cast<bool>(cheap_eval);
    const bool prune_active =
        cheap_eval_supplied && prune_tol > 0.0 && n_grid > 1;
    const bool need_pilot =
        prune_active || (n_threads_outer > 1 && !has_pc);
    const int k_pilot = n_grid / 2;

    // Output flags (filled below). pruned[k]: cheap-pass below threshold,
    // skip full Newton. log_marginal[k] is forced to -inf for pruned cells.
    // n_cells_pruned is reported back to the caller for the R-side toggle.
    std::vector<unsigned char> pruned(n_grid, 0);
    int n_cells_pruned = 0;
    // Cells the kept-set floor put back after the tolerance dropped them.
    int n_floor_restored = 0;
    int n_min_keep = std::min(n_grid, CHEAP_SCREEN_MIN_KEEP);
    // The screen's realised cut, in nats below the best cheap cell, and the
    // spread of the whole screened surface. Both reported, so a caller reads
    // what the tolerance meant on THIS grid instead of inferring it.
    double prune_log_gap_cut = NA_REAL;
    double prune_cheap_spread = NA_REAL;
    std::vector<double> cheap_lm;  // size n_grid when prune_active
    std::vector<double> cheap_dec; // the screen's Newton decrement per cell
    int cheap_argmax = -1;          // cell with the largest cheap log-marginal

    std::vector<double> pilot_mode;
    if (need_pilot) {
        // Tier-1 / global pilot. Always serial: any outer threads would
        // contend on a single CHOLMOD solver and the pilot is one cell. A
        // resumed pilot is read from the checkpoint instead of re-solved.
        if (!ckpt_done[k_pilot]) {
            SparseCholeskySolver pilot_solver;
            std::vector<double> pilot_pc;
            if (has_pc) pilot_pc = pc_row(k_pilot);
            cell_results[k_pilot] =
                solve_at_theta(k_pilot, has_pc ? pilot_pc : x_init_vec,
                               &pilot_solver);
            record(k_pilot);
            if (progress) progress->tick();  // first solved cell
        }
        pilot_mode = cell_results[k_pilot].mode;
        // If the pilot diverged (rare — e.g. proper-CAR with rho outside the
        // PD interval at the centre cell), fall back to zero warm-start so
        // downstream passes don't inherit garbage.
        if (!std::isfinite(cell_results[k_pilot].log_marginal)) {
            pilot_mode.assign(n_x, 0.0);
        }

        // Cheap-pass pruning: a chained sweep over the lattice that keeps
        // every cheap mode near its cell's true mode. Each cell runs a short
        // (screen_steps-step) inner Newton warm-started from the
        // previous screened cell's quasi-mode — lattice-adjacent along the
        // fastest axis of the Cartesian product — instead of one step from a
        // fixed global pilot. The pilot mode seeds the sweep so cell 0 is not
        // mode-starved. All cells use the same screening formula (Laplace
        // log-marginal at the quasi-mode), so the softmax ranking is faithful
        // to the full-solve ranking. The pilot is still never pruned (it is
        // the reference cell, exempted from the threshold below).
        if (prune_active) {
            cheap_lm.assign(n_grid,
                            -std::numeric_limits<double>::infinity());
            cheap_dec.assign(n_grid, NA_REAL);
            // Each cell is screened by exactly one thread, so the two writes
            // below never race.
            auto record_screen = [&](int k, const LaplaceResult& cr) {
                cheap_lm[k]  = screen_log_marginal(cr);
                cheap_dec[k] = cr.newton_decrement;
            };

            // A completed (checkpoint-loaded) cell needs no screening: its
            // true log-marginal ranks it and its stored mode warms the chain,
            // and it is never pruned (handled in the marking loop). Shared by
            // both the serial and per-tile parallel sweeps.
            auto use_loaded_for_chain = [&](int k, std::vector<double>& warm) {
                cheap_lm[k] = cell_results[k].log_marginal;
                if (std::isfinite(cell_results[k].log_marginal) &&
                    static_cast<int>(cell_results[k].mode.size()) == n_x) {
                    warm = cell_results[k].mode;
                }
            };

            // Whether the cheap screen can parallelize: the same tile metadata
            // the full parallel pass uses, plus more than one outer thread. A
            // single global chain is load-bearing for rank faithfulness (a
            // fixed-pilot one-step screen mis-ranks far cells by O(1e5)), so we
            // parallelize only by splitting it into independent per-tile chains
            // -- each tile groups cells whose donor-arm prior + design coincide
            // (only the copy/alpha axis varies), so within-tile chaining is at
            // least as near-neighbour as the global flat chain, while the tiles
            // run concurrently on their own worker slots.
            const bool cheap_use_tiles =
                (n_threads_outer > 1) &&
                (static_cast<int>(tile_ids.size()) == n_grid) &&
                (!tile_pilot_cells.empty());

            if (!cheap_use_tiles) {
                // Serial chained sweep over the whole lattice in flat order
                // (the historical behaviour; worker slot 0).
                std::vector<double> warm = pilot_mode;
                for (int k = 0; k < n_grid; k++) {
                    if (ckpt_done[k]) { use_loaded_for_chain(k, warm); continue; }
                    LaplaceResult cr =
                        cheap_at(k, warm, screen_steps, /*worker=*/0);
                    record_screen(k, cr);
                    // Chain the warm-start only across feasible cells; an
                    // infeasible cell (prep failed, log_marginal = -inf) leaves
                    // `warm` at the last good quasi-mode so the next feasible
                    // cell still starts from a near neighbour.
                    if (std::isfinite(cr.log_marginal) &&
                        static_cast<int>(cr.mode.size()) == n_x) {
                        warm = cr.mode;
                    }
                }
            } else {
                // Parallel cheap screen with the same two-tier warm-start
                // backbone the full parallel pass uses, so the screen ranking
                // stays faithful to the serial chain while parallelizing:
                //   Phase A (serial): chain the TILE PILOTS in tile order from
                //     the global pilot. n_tiles short solves give every tile
                //     root a near-neighbour-chained quasi-mode -- the coarse
                //     backbone (mirrors the full pass's Tier-2 tile pilots).
                //   Phase B (parallel over tiles): chain each tile's remaining
                //     cells in flat order from ITS tile root's quasi-mode, on
                //     the tile's own outer-worker slot. Tiles group cells whose
                //     only differing axis is the copy alpha, so within-tile
                //     chaining from the central tile pilot is a genuine
                // near-neighbour warm-start.
                // The tile partition (.joint_compute_tile_partition) covers
                // every cell exactly once, so each cheap_lm[k] is written by a
                // single thread (no race).
                const int n_tiles = static_cast<int>(tile_pilot_cells.size());
                std::vector<std::vector<int>> tile_cells(n_tiles);
                for (int k = 0; k < n_grid; k++) {
                    tile_cells[tile_ids[k]].push_back(k);
                }

                // Phase A: serial tile-pilot backbone (worker slot 0).
                std::vector<std::vector<double>> tile_root_mode(n_tiles,
                                                               pilot_mode);
                {
                    std::vector<double> warm = pilot_mode;
                    for (int t = 0; t < n_tiles; t++) {
                        int kp = tile_pilot_cells[t];
                        if (kp < 0 || kp >= n_grid) {
                            tile_root_mode[t] = warm;
                            continue;
                        }
                        if (ckpt_done[kp]) {
                            use_loaded_for_chain(kp, warm);
                            tile_root_mode[t] = warm;
                            continue;
                        }
                        LaplaceResult cr =
                            cheap_at(kp, warm, screen_steps, /*worker=*/0);
                        record_screen(kp, cr);
                        if (std::isfinite(cr.log_marginal) &&
                            static_cast<int>(cr.mode.size()) == n_x) {
                            warm = cr.mode;
                        }
                        tile_root_mode[t] = warm;
                    }
                }

                // Phase B: per-tile parallel chains hanging off each tile root.
                #ifdef _OPENMP
                #pragma omp parallel for schedule(dynamic, 1) \
                    num_threads(n_threads_outer)
                #endif
                for (int t = 0; t < n_tiles; t++) {
                    #ifdef _OPENMP
                    int worker = omp_get_thread_num();
                    #else
                    int worker = 0;
                    #endif
                    int kp = tile_pilot_cells[t];
                    int k_guard = (kp >= 0 && kp < n_grid)
                                  ? kp
                                  : (tile_cells[t].empty() ? 0
                                                            : tile_cells[t][0]);
                    guard_cell(k_guard, [&] {
                        std::vector<double> warm = tile_root_mode[t];
                        for (int k : tile_cells[t]) {
                            if (k == kp) continue;  // root screened in Phase A
                            if (ckpt_done[k]) {
                                use_loaded_for_chain(k, warm);
                                continue;
                            }
                            LaplaceResult cr =
                                cheap_at(k, warm, screen_steps, worker);
                            record_screen(k, cr);
                            if (std::isfinite(cr.log_marginal) &&
                                static_cast<int>(cr.mode.size()) == n_x) {
                                warm = cr.mode;
                            }
                        }
                    });
                }
                raise_if_cell_failed();
            }

            // The screened posterior: cheap log-marginal plus the cell's log
            // hyperprior and log measure. A cell whose offset is -Inf carries
            // no prior mass and is dropped with the infeasible ones.
            std::vector<double> screen_lm(n_grid);
            for (int k = 0; k < n_grid; k++) {
                screen_lm[k] = cheap_lm[k] + screen_off(k);
            }
            // Softmax over finite entries. Cells with non-finite cheap log-
            // marginal (block.prep returned infeasible) get weight 0 and are
            // pruned automatically.
            double m = -std::numeric_limits<double>::infinity();
            for (int k = 0; k < n_grid; k++) {
                if (std::isfinite(screen_lm[k]) && screen_lm[k] > m) {
                    m = screen_lm[k];
                    cheap_argmax = k;
                }
            }
            // A call joining a solved grid normalises against that grid's mass
            // as well (ScreenOffset::log_ref), so the cut is a share of the
            // posterior the cells join rather than of the call alone.
            const bool has_ref = screen_log_offset.has_ref();
            if (has_ref && screen_log_offset.log_ref > m) {
                m = screen_log_offset.log_ref;
            }
            double Z = 0.0;
            if (std::isfinite(m)) {
                for (double v : screen_lm) {
                    if (std::isfinite(v)) Z += std::exp(v - m);
                }
                if (has_ref) Z += std::exp(screen_log_offset.log_ref - m);
            }
            // The tolerance is a normalised weight, so the gap in nats it
            // actually cuts at is `-(log(prune_tol) + log(Z))`: a cell is
            // dropped once it falls that far below the best cheap cell. On a
            // log-marginal surface spanning thousands of nats that cut is a
            // sliver, which is what makes the same kept set come back at every
            // tolerance a caller would type. It is reported below so the cut is
            // readable from the fit rather than inferred from the tolerance.
            const double log_tol = std::log(prune_tol);
            const double log_Z   = (Z > 0.0) ? std::log(Z)
                                   : -std::numeric_limits<double>::infinity();
            for (int k = 0; k < n_grid; k++) {
                if (k == k_pilot) continue;
                if (ckpt_done[k]) continue;  // completed cells are never pruned
                double w = (std::isfinite(screen_lm[k]) && Z > 0.0)
                           ? std::exp(screen_lm[k] - m) / Z
                           : 0.0;
                bool drop;
                if (w > 0.0) {
                    drop = (w < prune_tol);
                } else if (std::isfinite(screen_lm[k]) && std::isfinite(m) &&
                           Z > 0.0) {
                    // exp(screen_lm[k] - m) underflowed to zero, which compares
                    // below every positive tolerance and so flattens the knob
                    // past a gap of ~745 nats. The same comparison in log space
                    // is exact there, and agrees with the weight form wherever
                    // the weight is representable, so a tolerance small enough
                    // to reach such a cell reaches it.
                    drop = ((screen_lm[k] - m) < (log_tol + log_Z));
                } else {
                    drop = true;   // infeasible cell: no cheap log-marginal
                }
                if (drop) {
                    pruned[k] = 1;
                    n_cells_pruned++;
                    LaplaceResult r;
                    r.mode = pilot_mode;
                    r.log_marginal =
                        -std::numeric_limits<double>::infinity();
                    r.n_iter = 0;
                    r.converged = false;
                    r.log_det_Q = 0.0;
                    cell_results[k] = r;
                }
            }

            // Kept-set floor. Restore the highest-ranked dropped cells until
            // `min_keep` cells will be solved in full, so the placement pass
            // downstream has more than a point mass to read a curvature off and
            // the full pass has more than the screen's own top cell to
            // disagree with it on. A grid whose tolerance already keeps that
            // many is untouched, so this is a no-op wherever the screen was not
            // collapsing.
            //
            // Only a cell with a finite cheap log-marginal is restorable: an
            // infeasible cell has no mode to solve at and would come back
            // -Inf, so a grid holding too few feasible cells stays below the
            // floor and reports how many it could restore.
            // A call joining a solved grid adds to a surface the placement has
            // already read, so it carries no floor.
            const int min_keep = has_ref ? 0
                                 : std::min(n_grid, CHEAP_SCREEN_MIN_KEEP);
            n_min_keep = min_keep;
            int n_kept_cells = n_grid - n_cells_pruned;
            if (n_kept_cells < min_keep) {
                std::vector<int> restorable;
                for (int k = 0; k < n_grid; k++) {
                    if (pruned[k] && std::isfinite(screen_lm[k])) {
                        restorable.push_back(k);
                    }
                }
                // Rank by the cheap log-marginal, ties by cell index, so the
                // restored set is a function of the screen alone.
                std::sort(restorable.begin(), restorable.end(),
                          [&](int a, int b) {
                              if (screen_lm[a] != screen_lm[b]) {
                                  return screen_lm[a] > screen_lm[b];
                              }
                              return a < b;
                          });
                for (std::size_t i = 0;
                     i < restorable.size() && n_kept_cells < min_keep; i++) {
                    const int k = restorable[i];
                    pruned[k] = 0;
                    n_cells_pruned--;
                    n_kept_cells++;
                    n_floor_restored++;
                }
            }
            prune_log_gap_cut = (std::isfinite(log_tol) && std::isfinite(log_Z))
                                ? -(log_tol + log_Z)
                                : NA_REAL;
            // Spread of the WHOLE screened surface, the scale a cheap-vs-full
            // mis-estimate has to be judged against: the spread of the cells
            // the screen kept is a property of the set the gate is meant to
            // validate, and cannot bound the error on the set it discarded.
            {
                double lo =  std::numeric_limits<double>::infinity();
                double hi = -std::numeric_limits<double>::infinity();
                for (double v : screen_lm) {
                    if (!std::isfinite(v)) continue;
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
                prune_cheap_spread = (hi >= lo) ? (hi - lo) : NA_REAL;
            }
        }
        // Pruned and checkpoint-loaded cells get no full solve, so they are
        // not part of the ETA denominator; revise it down to the survivors
        // actually solved this run.
        if (progress) progress->set_total(n_grid - n_cells_pruned - n_loaded);
    }

    // A screened read: the screen's own log-marginal for every cell it ranked,
    // none pruned, no full solve, so the surface is one quantity throughout.
    // The pilot keeps the mode of the full solve it already carries (the warm
    // start a placement reads) but reports its screened log-marginal; a
    // checkpoint-loaded cell keeps its full solve, which the screen ranked it
    // by.
    const bool screen_read = screen_only && prune_active;
    if (screen_read) {
        for (int k = 0; k < n_grid; k++) {
            if (ckpt_done[k]) continue;
            if (k == k_pilot) {
                if (std::isfinite(cheap_lm[k]))
                    cell_results[k].log_marginal = cheap_lm[k];
                continue;
            }
            LaplaceResult r;
            r.mode = pilot_mode;
            r.log_marginal = cheap_lm[k];
            r.n_iter = 0;
            r.converged = false;
            r.log_det_Q = 0.0;
            cell_results[k] = r;
            pruned[k] = 0;
        }
        n_cells_pruned = 0;
    } else if (n_threads_outer <= 1) {
        // Serial path. Two sub-cases:
        //  - prune off, pilot off: classic mode-chained warm-start across
        //    every cell.
        //  - prune on (need_pilot): pilot already solved; loop over the
        //    remaining cells warm-starting from the previous non-pruned
        //    cell's mode (chained across survivors only).
        SparseCholeskySolver solver;
        if (!need_pilot) {
            // No pilot was solved, so checkpoint-loaded cells were not yet
            // discounted from the progress total; do it here.
            if (progress && n_loaded) progress->set_total(n_grid - n_loaded);
            std::vector<double> prev_mode = x_init_vec;
            std::vector<double> warm_pc;
            for (int k = 0; k < n_grid; k++) {
                // A loaded cell is already in cell_results; chain its stored
                // mode as the next warm-start and skip the solve.
                if (ckpt_done[k]) {
                    prev_mode = cell_results[k].mode;
                    continue;
                }
                if (has_pc) warm_pc = pc_row(k);
                cell_results[k] = solve_at_theta(
                    k, has_pc ? warm_pc : prev_mode, &solver);
                record(k);
                if (!has_pc) prev_mode = cell_results[k].mode;
                if (progress) progress->tick();
            }
        } else {
            std::vector<double> prev_mode = pilot_mode;
            std::vector<double> warm_pc;
            // Iterate in original cell order, starting from cell 0; pilot,
            // pruned, and loaded cells are skipped (already filled above).
            for (int k = 0; k < n_grid; k++) {
                if (k == k_pilot) {
                    prev_mode = cell_results[k_pilot].mode;
                    continue;
                }
                if (pruned[k]) continue;
                if (ckpt_done[k]) {
                    prev_mode = cell_results[k].mode;
                    continue;
                }
                if (has_pc) warm_pc = pc_row(k);
                cell_results[k] = solve_at_theta(
                    k, has_pc ? warm_pc : prev_mode, &solver);
                record(k);
                if (!has_pc) prev_mode = cell_results[k].mode;
                if (progress) progress->tick();
            }
        }
    } else {
        // Parallel path: the pilot, where one was needed, is solved above.
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
        // The tiers exist to hand each cell a warm start; per-cell warm
        // starts already give every cell its own.
        const bool use_tiles = !has_pc &&
            (static_cast<int>(tile_ids.size()) == n_grid) &&
            (!tile_pilot_cells.empty());
        if (!need_pilot && progress && n_loaded) {
            progress->set_total(n_grid - n_loaded);
        }

        if (!use_tiles) {
            // Phase 1 path: every cell warm-started from the global pilot.
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1) \
                num_threads(n_threads_outer)
            #endif
            for (int k = 0; k < n_grid; k++) {
                if (need_pilot && k == k_pilot) continue;  // already solved
                if (pruned[k]) continue;     // cheap-pass pruned
                if (ckpt_done[k]) continue;  // checkpoint-loaded
                #ifdef _OPENMP
                int tid = omp_get_thread_num();
                #else
                int tid = 0;
                #endif
                SparseCholeskySolver* solver = solver_pool[tid].get();
                guard_cell(k, [&] {
                    std::vector<double> warm_pc;
                    if (has_pc) warm_pc = pc_row(k);
                    cell_results[k] = solve_at_theta(
                        k, has_pc ? warm_pc : pilot_mode, solver);
                    record(k);
                });
                if (progress) {
                    #ifdef _OPENMP
                    #pragma omp critical(nl_grid_progress)
                    #endif
                    progress->tick();
                }
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

            // Tier 2: one warm solve per tile, parallel over tiles. Pruned
            // tile-pilots are skipped; their tile_modes entry stays empty
            // and Tier-3 falls back to the global pilot for cells in that
            // tile.
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
                if (pruned[k]) continue;
                if (ckpt_done[k]) {
                    // Loaded tile pilot: its stored mode still seeds Tier 3.
                    tile_modes[t] = std::isfinite(cell_results[k].log_marginal)
                                    ? cell_results[k].mode
                                    : pilot_mode;
                    continue;
                }
                #ifdef _OPENMP
                int tid = omp_get_thread_num();
                #else
                int tid = 0;
                #endif
                SparseCholeskySolver* solver = solver_pool[tid].get();
                guard_cell(k, [&] {
                    std::vector<double> warm_pc;
                    if (has_pc) warm_pc = pc_row(k);
                    cell_results[k] = solve_at_theta(
                        k, has_pc ? warm_pc : pilot_mode, solver);
                    record(k);
                });
                // Tile-pilot mode is the Tier-3 warm-start; if the cell
                // diverged, fall back to the global pilot mode so we don't
                // propagate garbage to the rest of the tile.
                tile_modes[t] = std::isfinite(cell_results[k].log_marginal)
                                ? cell_results[k].mode
                                : pilot_mode;
                if (progress) {
                    #ifdef _OPENMP
                    #pragma omp critical(nl_grid_progress)
                    #endif
                    progress->tick();
                }
            }
            raise_if_cell_failed();

            // Tier 3: every non-pilot cell warm-started from its tile
            // pilot (falling back to the global pilot when the tile pilot
            // is pruned or out of range).
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1) \
                num_threads(n_threads_outer)
            #endif
            for (int k = 0; k < n_grid; k++) {
                if (k == k_pilot) continue;       // global pilot, Tier 1
                if (is_tile_pilot[k]) continue;   // tile pilot, Tier 2
                if (pruned[k]) continue;          // cheap-pass pruned
                if (ckpt_done[k]) continue;       // checkpoint-loaded
                int t = tile_ids[k];
                #ifdef _OPENMP
                int tid = omp_get_thread_num();
                #else
                int tid = 0;
                #endif
                SparseCholeskySolver* solver = solver_pool[tid].get();
                guard_cell(k, [&] {
                    std::vector<double> warm_pc;
                    if (has_pc) warm_pc = pc_row(k);
                    const std::vector<double>& warm =
                        has_pc ? warm_pc
                        : ((t >= 0 && t < n_tiles && !tile_modes[t].empty())
                           ? tile_modes[t]
                           : pilot_mode);
                    cell_results[k] = solve_at_theta(k, warm, solver);
                    record(k);
                });
                if (progress) {
                    #ifdef _OPENMP
                    #pragma omp critical(nl_grid_progress)
                    #endif
                    progress->tick();
                }
            }
        }
        raise_if_cell_failed();
    }

    // Repair a prune whose dropped cells may carry more than
    // CHEAP_SCREEN_GATE_MASS of the posterior. The kept cells' full solves say
    // how far off the screen was; the dropped cells whose screened value, lifted
    // by the worst of that, contributes most to the bound are solved in full,
    // largest first, until what is left of it is half the gate, and the bound is
    // read again off the enlarged kept set. A dropped cell far below the cut
    // never reaches the list, so what the repair pays for is the handful of
    // cells near the cut a large fit's screen was unsure about, where solving
    // the whole grid over again would redo every kept cell as well.
    double prune_mass_bound = NA_REAL;
    int n_repaired = 0;
    if (prune_active && !screen_read) {
        auto kept_rank = [&](int k) {
            return cell_results[k].log_marginal + screen_off(k);
        };
        for (;;) {
            double e_max = -std::numeric_limits<double>::infinity();
            double top   = -std::numeric_limits<double>::infinity();
            for (int k = 0; k < n_grid; k++) {
                if (pruned[k] || !std::isfinite(cell_results[k].log_marginal)) {
                    continue;
                }
                top = std::max(top, kept_rank(k));
                if (std::isfinite(cheap_lm[k])) {
                    e_max = std::max(e_max,
                                     cell_results[k].log_marginal - cheap_lm[k]);
                }
            }
            if (!std::isfinite(top) || !std::isfinite(e_max)) break;
            // The grid the call's cells join is kept mass too.
            const double log_ref = screen_log_offset.log_ref;
            if (screen_log_offset.has_ref() && log_ref > top) top = log_ref;
            double den = screen_log_offset.has_ref() ? std::exp(log_ref - top)
                                                     : 0.0;
            for (int k = 0; k < n_grid; k++) {
                if (!pruned[k] && std::isfinite(cell_results[k].log_marginal)) {
                    den += std::exp(kept_rank(k) - top);
                }
            }
            std::vector<std::pair<double, int>> contrib;
            double num = 0.0;
            for (int k = 0; k < n_grid; k++) {
                if (!pruned[k] || !std::isfinite(cheap_lm[k])) continue;
                const double c =
                    std::exp(cheap_lm[k] + screen_off(k) + e_max - top);
                if (c > 0.0) contrib.emplace_back(c, k);
                num += c;
            }
            prune_mass_bound = (den > 0.0) ? num / den : NA_REAL;
            if (!(den > 0.0) || prune_mass_bound <= CHEAP_SCREEN_GATE_MASS ||
                contrib.empty()) {
                break;
            }
            std::sort(contrib.begin(), contrib.end(),
                      [](const std::pair<double, int>& a,
                         const std::pair<double, int>& b) {
                          if (a.first != b.first) return a.first > b.first;
                          return a.second < b.second;
                      });
            const double target = 0.5 * CHEAP_SCREEN_GATE_MASS * den;
            std::vector<int> fix;
            double rest = num;
            for (const auto& cj : contrib) {
                if (rest <= target) break;
                fix.push_back(cj.second);
                rest -= cj.first;
            }
            for (int k : fix) {
                pruned[k] = 0;
                n_cells_pruned--;
            }
            n_repaired += static_cast<int>(fix.size());
            if (progress) progress->set_total(n_grid - n_cells_pruned - n_loaded);
            const int team =
                tulpa_omp_team_size_req(n_threads_outer,
                                        static_cast<int>(fix.size()));
            std::vector<std::unique_ptr<SparseCholeskySolver>> repair_pool(
                std::max(1, team));
            for (auto& s : repair_pool) s.reset(new SparseCholeskySolver());
            const int n_fix = static_cast<int>(fix.size());
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1) num_threads(team) \
                if (team > 1)
            #endif
            for (int i = 0; i < n_fix; i++) {
                const int k = fix[i];
                #ifdef _OPENMP
                int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
                #else
                int tid = 0;
                #endif
                if (tid < 0 || tid >= static_cast<int>(repair_pool.size())) tid = 0;
                guard_cell(k, [&] {
                    std::vector<double> warm_pc;
                    if (has_pc) warm_pc = pc_row(k);
                    cell_results[k] = solve_at_theta(
                        k, has_pc ? warm_pc : pilot_mode,
                        repair_pool[tid].get());
                    record(k);
                });
                if (progress) {
                    #ifdef _OPENMP
                    #pragma omp critical(nl_grid_progress)
                    #endif
                    progress->tick();
                }
            }
            raise_if_cell_failed();
        }
    }

    if (progress) progress->finish();

    for (int k = 0; k < n_grid; ++k)
        cell_results[k].device_cpu_recovered = cell_results[k].device_cpu_recovered || cell_cpu_recovery[k];
    Rcpp::List out = nl_pack_grid_results(cell_results, n_grid, n_x,
                                          store_modes,
                                          n_threads_outer_realised);
    Rcpp::NumericVector log_marginals = out["log_marginal"];

    if (prune_active) {
        Rcpp::NumericVector cheap_lm_out(n_grid);
        Rcpp::LogicalVector pruned_out(n_grid);
        for (int k = 0; k < n_grid; k++) {
            cheap_lm_out[k] = cheap_lm[k];
            pruned_out[k]   = (pruned[k] != 0);
        }
        out["prune_cheap_log_marginal"] = cheap_lm_out;
        out["prune_screen_decrement"]   = Rcpp::wrap(cheap_dec);
        out["prune_mask"]                = pruned_out;
        out["prune_n_pruned"]           = n_cells_pruned;
        out["prune_tol"]                = prune_tol;
        out["prune_screen_iters"]       = screen_steps;
        if (!screen_log_offset.empty()) {
            out["prune_screen_log_offset"] = Rcpp::wrap(screen_log_offset.off);
        }
        if (screen_log_offset.has_ref()) {
            out["prune_screen_log_ref"] = screen_log_offset.log_ref;
        }
        // What the tolerance cut at (nats below the best cheap cell), what the
        // screened surface spans, the floor that was applied and how many cells
        // it put back. The first two are the pair a caller reads to see whether
        // the tolerance had any resolution on this grid at all.
        out["prune_log_gap_cut"]        = prune_log_gap_cut;
        out["prune_cheap_lm_spread"]    = prune_cheap_spread;
        out["prune_min_keep"]           = n_min_keep;
        out["prune_n_floor_restored"]   = n_floor_restored;
        out["prune_screen_only"]        = screen_read;
        // The dropped-mass bound the kept set closed at, how many dropped cells
        // the repair solved to get it there, and the gate it was held to.
        out["prune_dropped_mass_bound"] = prune_mass_bound;
        out["prune_n_repaired"]         = n_repaired;
        out["prune_gate_mass"]          = CHEAP_SCREEN_GATE_MASS;

        // ---- Ranking diagnostics -----------------------------------------
        // Reported beside the bound; the gate reads the bound alone. The full
        // pass only ran on survivors, so the full-solve argmax is taken over
        // kept cells, and the cheap-vs-full gap at it together with whether
        // the screen's own top cell was dropped say how the screen ranked
        // against the full solve.
        int full_argmax = -1;
        double full_max = -std::numeric_limits<double>::infinity();
        double full_post_max = -std::numeric_limits<double>::infinity();
        for (int k = 0; k < n_grid; k++) {
            if (pruned[k]) continue;
            double v = log_marginals[k] + screen_off(k);
            if (std::isfinite(v) && v > full_post_max) {
                full_post_max = v;
                full_max = log_marginals[k];
                full_argmax = k;
            }
        }
        // Cheap-vs-full gap at the full argmax cell: how far the cheap screen
        // mis-estimated the log-marginal of the cell the full solve favours.
        double cheap_full_gap = NA_REAL;
        if (full_argmax >= 0 &&
            std::isfinite(full_max) &&
            std::isfinite(cheap_lm[full_argmax])) {
            cheap_full_gap = std::abs(full_max - cheap_lm[full_argmax]);
        }
        // Argmax disagreement: the cheap screen's top cell is not the full
        // solve's top cell. A disagreement means the screen would have ranked
        // a different cell first — exactly the failure that prunes the true
        // mode. (cheap_argmax == full_argmax is the safe case.)
        bool argmax_disagree =
            (cheap_argmax >= 0 && full_argmax >= 0 &&
             cheap_argmax != full_argmax);
        // The cheap argmax being pruned is the sharpest tell: the screen's
        // own top-ranked cell never got a full solve.
        bool cheap_argmax_pruned =
            (cheap_argmax >= 0 && pruned[cheap_argmax] != 0);

        out["prune_cheap_argmax"]   = cheap_argmax + 1;   // 1-based for R
        out["prune_full_argmax"]    = full_argmax + 1;    // 1-based for R
        out["prune_cheap_full_gap"] = cheap_full_gap;
        out["prune_argmax_disagree"] =
            (argmax_disagree || cheap_argmax_pruned);
    }
    return out;
}

} // namespace tulpa
