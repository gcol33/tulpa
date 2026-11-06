// nested_laplace_grid.cpp
// Definitions of the outer-grid result packers declared in nested_laplace_grid.h.

#include <RcppEigen.h>
#include "nested_laplace_grid.h"
#include "linear_constraint_rows.h"

namespace tulpa {

Rcpp::List nl_pack_grid_results(
    const std::vector<LaplaceResult>& cell_results, int n_grid, int n_x,
    bool store_modes, int n_threads_outer_realised
) {
    Rcpp::NumericVector log_marginals(n_grid);
    Rcpp::IntegerVector n_iters(n_grid);
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
        Rcpp::stop("nl_grid_cell_to_result_list: cell %d out of range "
                   "[0, %d).", k, static_cast<int>(modes.nrow()));
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

} // namespace tulpa
