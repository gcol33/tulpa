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
//   1) runs a pilot solve at the grid centre cell (serial, on the whole pool:
//      `nl_cell_solve_threads()`),
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
// Optional cheap-pass pruning: when `cheap_eval` is provided and
// `prune_tol > 0`, the driver
//   1) solves the pilot cell (always, regardless of n_threads_outer),
//   2) SWEEPS the outer lattice in flat-index order, warm-starting each
//      cell's short cheap Newton run from the nearest already-screened
//      neighbour's cheap mode (the previous cell in flat order, which is
//      lattice-adjacent along the fastest-varying axis of the Cartesian
//      product), and computes a screening log-marginal at the resulting
//      quasi-mode,
//   3) softmax-normalises the screening log-marginals over the full grid
//      and marks cells with weight < prune_tol as pruned, then restores the
//      highest-ranked of them until at least `CHEAP_SCREEN_MIN_KEEP` cells
//      survive,
//   4) skips the full inner Newton on pruned cells (filling their result
//      with `log_marginal = -inf`, `n_iter = 0`, `mode = pilot_mode`) and
//      runs the full pass only on survivors.
//
// The tolerance is a NORMALISED WEIGHT, so what it cuts at is a gap in nats:
// `-(log(prune_tol) + log(Z))` below the best cheap cell. The whole range a
// caller would type (1e-3 to 1e-12) is 7 to 28 nats of that gap, which on a
// surface spanning thousands of nats is one sliver and returns the same kept
// set at every setting. `prune_log_gap_cut` reports the realised cut and
// `prune_cheap_lm_spread` the surface it was applied to, so the two together
// say whether the tolerance had any resolution on this grid. The kept-set
// floor is what keeps the answer usable where it had none.
//
// The screen only has to RANK the cells: what it does with a cell's screening
// log-marginal is compare it against the others, never report it. Each cell is
// warm-started from its already-screened lattice neighbour rather than from a
// single global pilot, so the quasi-mode it lands on is near its own cell's
// mode after very few Newton steps even where the inner latent mode moves
// substantially across the grid (large spatial fields, wide sigma/rho/alpha
// ranges).
//
// `screen_iters` is therefore a cost-against-ranking-fidelity trade rather than
// a convergence budget. Every step is paid on every cell of the grid, including
// the cells the screen goes on to keep and solve in full, while the full solves
// it avoids are themselves neighbour-warm-started and converge in a handful of
// iterations. A depth above what the ranking needs makes screening cost more
// than the solves it saves.
//
// The screen ranks what the grid integrates. `screen_log_offset` (length
// `n_grid`, or empty) is each cell's log hyperprior plus its log cell measure,
// both fixed before any cell is solved: the driver's `log_marginal` is the
// kernel's likelihood alone, and the posterior weight a cell ends up with is
// `exp(log_marginal + offset)`. Ranking by the likelihood alone would prune a
// cell the prior or its width favours and keep one they suppress. The offset
// enters the ranking, the cut, the floor restore and the gate's full-solve
// argmax; `prune_cheap_log_marginal` and the cheap-vs-full gap stay on the
// kernel's own scale, where the offset cancels.
//
// Repair: after the full pass the kept cells' full solves say how far off the
// screen was, and the driver bounds the posterior mass the dropped cells may
// carry by lifting each by the worst error measured (CHEAP_SCREEN_GATE_MASS).
// Past the gate it solves the dropped cells that contribute most to the bound
// and reads it again, so what reaches the caller is a kept set whose dropped
// cells can carry at most that share (`prune_dropped_mass_bound`,
// `prune_n_repaired`). The R safety gate reads the same bound off the result
// and falls back to the full grid if it still fails. The cheap-screen argmax,
// the full-solve argmax over kept cells and the cheap-vs-full gap at the latter
// are reported beside it (`prune_argmax_disagree`, `prune_cheap_full_gap`,
// `prune_full_argmax`).
//
// The pilot cell is never pruned. Tile pilots whose cells are themselves
// pruned skip their Tier-2 solve; Tier-3 cells in that tile then fall back
// to the global pilot mode as warm-start. The cheap sweep's backbone is
// sequential: each cheap solve is warm-started from the previous screened
// cell's quasi-mode, so the per-cell step budget only has to correct the
// drift between lattice-adjacent cells. Breaking the chain is what costs ranking fidelity
// (a fixed-pilot one-step screen mis-ranks far cells by O(1e5)), so
// parallelism is bought by splitting the chain per tile rather than by
// removing it: given tile metadata and n_threads_outer > 1, the sweep chains
// the tile pilots serially (Phase A) and then runs each tile's own chain
// concurrently (Phase B, an omp parallel for over tiles), for a concurrency
// of min(n_tiles, n_threads_outer). The cheap evaluator therefore runs inside
// a parallel region and must select a per-worker solver + scratch by the
// `worker` slot it is passed -- see the CheapEval contract below.
//
// Result post-processing (filling the Rcpp::List) happens single-threaded
// after the parallel region. Per-cell LaplaceResult objects use only
// std::vector storage (laplace_core.h:LaplaceResult), so they are safe to
// populate across threads.
//
// What a cell is ranked by is not the truncated solve's log-marginal but its
// second-order estimate of the converged one: that value plus half the Newton
// decrement at the point the screen stopped (`screen_log_marginal()`). A
// truncated solve reads lowest where its inner mode has furthest to go, and on
// the full 25 km occu_cover fit of gcol33/tulpa#919 that was the outer mode
// itself -- ~250 inner iterations there against 12 to 21 on the plateau it was
// ranked below -- so a ranking on the raw value pruned the mode and passed its
// own safety gate, which checks the screen only on the cells it kept.

#ifndef TULPA_NESTED_LAPLACE_GRID_H
#define TULPA_NESTED_LAPLACE_GRID_H

#include "laplace_core.h"
#include "tulpa_stop.h"
#include "laplace_profile.h"      // TULPA_PROFILE_PHASE (outer_grid_cell)
#include "nested_laplace_checkpoint.h"
#include "omp_threads.h"          // tulpa_omp_team_size_req (cache-safe clamp)
#include "sparse_cholesky.h"
#include <tulpa/nested_progress.h>
#include <Rcpp.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// Inner threads for one cell solve. A solve inside the outer team holds its
// share of the pool, `in_team`. A solve that runs alone holds `n_threads`, the
// caller's grant for the whole fit (the R side resolves it as the larger of the
// inner request and the outer width, under the performance-core cap): the pilot
// the parallel branches warm-start from, the screen's serial backbone, and
// every cell of a call with the outer loop off. Held to the in-team share, the
// pilot of every parallel grid ran on one thread while the rest of the pool
// waited for it (gcol33/tulpa#924).
inline int nl_cell_solve_threads(int n_threads, int in_team) {
#ifdef _OPENMP
    if (omp_in_parallel()) return std::max(1, in_team);
#endif
    return std::max(1, n_threads);
}

// ---------------------------------------------------------------------
// Outer-grid axis contract
// ---------------------------------------------------------------------
// Every kernel below takes the outer grid as one NumericVector per
// hyperparameter axis, all of them indexed by the same cell number k. A short
// axis is read past its end by the block factories, which index it unchecked,
// so the pairing is checked once here rather than restated at each entry point.
//
// `lead` fixes the cell count; `rest` names the axes that must match it.
inline int nl_grid_axes_length(
    const char* lead_name, const Rcpp::NumericVector& lead,
    std::initializer_list<std::pair<const char*, const Rcpp::NumericVector*>> rest
) {
    const int n_grid = static_cast<int>(lead.size());
    for (const auto& axis : rest) {
        const int len = static_cast<int>(axis.second->size());
        if (len != n_grid) {
            tulpa::stop("`%s` has length %d but `%s` has length %d; every "
                       "outer-grid axis needs one entry per grid cell.",
                       {axis.first, len, lead_name, n_grid});
        }
    }
    return n_grid;
}

// A scale or precision axis is strictly positive. Every density that reads one
// takes its logarithm for the normalizer and divides by it for the field scale,
// so a zero cell gives -Inf and a negative cell NaN -- inside the inner Newton
// solve, as a cell whose marginal is simply not finite, with nothing naming the
// axis. Checked once at the entry point, where the message can.
inline void nl_check_positive(const char* name, double v) {
    if (!(v > 0.0) || !R_finite(v)) {
        tulpa::stop("`%s` is %g; a scale or precision must be finite and "
                   "strictly positive.", {name, v});
    }
}

inline void nl_grid_cell_positive(const char* name, double v, int k) {
    if (!(v > 0.0) || !R_finite(v)) {
        tulpa::stop("`%s[%d]` is %g; a scale or precision axis must be "
                   "finite and strictly positive at every grid cell.",
                   {name, k + 1, v});
    }
}

inline void nl_grid_axes_positive(
    std::initializer_list<std::pair<const char*, const Rcpp::NumericVector*>> axes
) {
    for (const auto& axis : axes) {
        const Rcpp::NumericVector& v = *axis.second;
        const int n = static_cast<int>(v.size());
        for (int k = 0; k < n; k++) nl_grid_cell_positive(axis.first, v[k], k);
    }
}

// The same axis carried as a column of the multi-block `theta_grid`.
inline void nl_grid_axis_positive(const char* name,
                                  const Rcpp::NumericMatrix& grid, int col) {
    const int n = grid.nrow();
    for (int k = 0; k < n; k++) nl_grid_cell_positive(name, grid(k, col), k);
}

// A mixing-weight axis (the BYM2 rho) is a proportion. Outside [0, 1] both
// bym2_sd_* square roots go NaN, which reaches the inner Newton through eta and
// returns a NaN cell rather than an error.
inline bool nl_in_unit_interval(double rho) {
    return rho >= 0.0 && rho <= 1.0;
}

// The same axis supplied as a scalar by a fixed-hyperparameter entry point.
inline void nl_check_unit_interval(const char* name, double rho) {
    if (!nl_in_unit_interval(rho)) {
        tulpa::stop("`%s` is %g; a BYM2 mixing weight must lie in [0, 1].",
                   {name, rho});
    }
}

inline void nl_grid_cell_unit_interval(const char* name, double rho, int k) {
    if (!nl_in_unit_interval(rho)) {
        tulpa::stop("`%s[%d]` is %g; a BYM2 mixing weight must lie in [0, 1].",
                   {name, k + 1, rho});
    }
}

inline void nl_grid_axis_unit_interval(const char* name,
                                       const Rcpp::NumericVector& v) {
    const int n = static_cast<int>(v.size());
    for (int k = 0; k < n; k++) nl_grid_cell_unit_interval(name, v[k], k);
}

// The same axis carried as a column of the multi-block `theta_grid`, whose axes
// are concatenated per block rather than passed one vector at a time.
inline void nl_grid_axis_unit_interval(const char* name,
                                       const Rcpp::NumericMatrix& grid, int col) {
    const int n = grid.nrow();
    for (int k = 0; k < n; k++) nl_grid_cell_unit_interval(name, grid(k, col), k);
}

// Project cell k of a grid-driver result onto the single-fit result shape
// (the laplace_result_to_list contract: mode, log_det_Q, log_marginal, n_iter,
// converged, score_max).
//
// A fixed-hyperparameter export is the same computation as its nested
// integrator run on a one-cell grid, so it builds the same arm + LatentBlock,
// calls the same driver, and reads cell 0 back through here. Every field the
// single-fit contract promises is a per-cell vector on the grid result, so this
// is a projection, not a re-derivation -- there is no second Newton loop to
// drift from the first.
Rcpp::List nl_grid_cell_to_result_list(const Rcpp::List& grid, int k);

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
//   (prev_mode <- res.mode after each cell). Bitwise identical to the
//   serial mode-chained result.
// n_threads_outer > 1: pilot solve at the centre cell first, its result
//   reused (no double-solve), then one of two parallel passes depending on
//   whether the caller supplied tile metadata (`tile_ids` of length n_grid
//   and a non-empty `tile_pilot_cells`):
//     without tiles: a flat #pragma omp parallel for over every cell, each
//       warm-started from the global pilot mode.
//     with tiles: Tier 2 solves one pilot per tile in parallel, warm-started
//       from the global pilot; Tier 3 then solves the remaining cells in
//       parallel, each warm-started from its own tile's pilot mode (falling
//       back to the global pilot where that tile's pilot was pruned or
//       out of range).
// `cheap_eval` (NlCheapEval) has signature
//   LaplaceResult(int k, const std::vector<double>& warm_start, int n_steps,
//                 int worker)
// running a short (`n_steps`-iteration) inner Newton-Laplace at cell k
// warm-started from `warm_start`, and returning the quasi-mode and the
// Laplace screening log-marginal at that quasi-mode. The driver chains the
// warm-start across the lattice (each cell starts from the previous
// screened cell's `.mode`), so `n_steps` need only correct the residual
// drift between adjacent cells, not the full distance from a global pilot.
// `worker` is a 0-based slot id in [0, n_threads_outer): the cheap screen
// may run per-tile chains concurrently across outer threads, so the callable
// must use a per-worker solver + scratch (CHOLMOD's cholmod_common is not
// thread-safe) selected by `worker`. The serial screen always passes 0.
// A returned `log_marginal` of -infinity forces pruning of an infeasible
// cell (e.g. proper-CAR rho outside the PD interval). An empty `cheap_eval`
// disables pruning entirely. The kernels are type-erased: the driver is
// compiled once, and a call through std::function costs nothing beside the
// Laplace solve it makes per cell.
//
// `ckpt` (default nullptr) enables grid-cell checkpoint/resume: a cell whose
// hyperparameter coordinate was completed in a prior pass / killed run is
// loaded from disk in place of its full solve, and every freshly solved cell is
// appended. A loaded cell is still screened, and dropped where the screen drops
// it, so the kept set does not depend on what the file holds. See
// nested_laplace_checkpoint.h.
using NlCellSolve = std::function<LaplaceResult(
    int k, const std::vector<double>& warm, SparseCholeskySolver* solver)>;
using NlCheapEval = std::function<LaplaceResult(
    int k, const std::vector<double>& warm, int n_steps, int worker)>;

// Engine default for the number of inner Newton steps per cell in the cheap
// screening sweep, used when the caller supplies no `screen_iters`. See the
// cost-against-ranking-fidelity note at the top of this file.
static const int CHEAP_SCREEN_ITERS = 2;

// Floor on the number of cells the screen leaves to the full pass, whatever the
// tolerance says. The screen ranks cells; the placement pass that runs after the
// fit reads a finite-difference curvature stencil off the cells that were
// SOLVED, and a kept set of one carries no second difference on any axis, so
// the placement declines for want of curvature and the axis is reported where
// it was laid rather than where the posterior is. That happens on a grid whose
// posterior is concentrated, which is exactly the grid screening is worth doing
// on, so the tolerance alone cannot be what decides it.
//
// A central second difference along one axis needs three collinear nodes.
// Five is the argmax cell plus a two-sided neighbourhood on the two axes an
// outer grid most often carries, and it is the smallest floor under which no
// axis of such a grid can be a point mass. It does not GUARANTEE a full
// stencil -- the floor keeps the highest-ranked cells, not a star of a chosen
// shape -- so a placement can still decline, and the R layer records that a
// decline happened on a screened grid.
//
// Five full solves against the 1-2 a collapsed screen leaves is a cost the
// screening is not measured against: the cells it skips number in the hundreds.
static const int CHEAP_SCREEN_MIN_KEEP = 5;

// The posterior mass a screened grid may, at worst, have dropped before the
// driver solves the dropped cells that could be carrying it (the repair after
// the full pass). With c a cell's screened and f its full log-posterior and
// e_max the worst error f - c measured on a kept cell, the bound is
// sum_dropped exp(c + e_max) / sum_kept exp(f): a dropped cell is taken to be as
// badly screened as the worst kept one. A constant error cancels from it; one
// that varies across the kept cells lifts every dropped cell by the worst of
// it. Mirrors `.NL_SCREEN$gate_mass` in R/settings.R, which the R safety gate
// reads and which a test pins to this value.
static const double CHEAP_SCREEN_GATE_MASS = 0.01;

// What the screen ranks a call's cells with, beyond the kernel's own
// log-marginal: each cell's log hyperprior plus its log cell measure (`off`),
// and the log posterior mass of the grid the call's cells JOIN (`log_ref`).
// A call that is the whole grid carries no reference, and its cells are
// normalised against each other. A refinement round's call is a batch of cells
// added to a grid already solved, so a cell's weight is its screened mass over
// the batch's plus the grid's: without the reference every batch keeps its own
// best cells, however far below the grid's posterior the whole batch sits.
// `log_ref` reaches the kernel as the `log_ref` attribute of the offset vector,
// so every entry that takes the offset takes it.
struct ScreenOffset {
    std::vector<double> off;
    double log_ref = -std::numeric_limits<double>::infinity();
    bool empty() const { return off.empty(); }
    std::size_t size() const { return off.size(); }
    double operator[](std::size_t k) const { return off[k]; }
    bool has_ref() const { return std::isfinite(log_ref); }
};

inline ScreenOffset read_screen_offset(
    const Rcpp::Nullable<Rcpp::NumericVector>& x) {
    ScreenOffset s;
    if (x.isNull()) return s;
    Rcpp::NumericVector v(x);
    s.off.assign(v.begin(), v.end());
    if (v.hasAttribute("log_ref")) {
        Rcpp::NumericVector r = v.attr("log_ref");
        if (r.size() == 1 && std::isfinite(r[0])) s.log_ref = r[0];
    }
    return s;
}

// A screened cell's estimate of its converged log-marginal: the truncated
// solve's value plus the gain of the steps it did not take, half the Newton
// decrement where it stopped to second order, but never more than the next
// step actually delivers (`newton_step_gain`) and never less than nothing. The
// quadratic prediction is unbounded far from the mode, and an overestimated
// cell is what corrupts the screen: it moves the argmax and, through the worst
// kept-cell error, lifts every dropped cell's bound, while an underestimate is
// what that bound already covers. A solve that left no decrement reads as it
// stands.
inline double screen_log_marginal(const LaplaceResult& r) {
    if (!std::isfinite(r.log_marginal)) return r.log_marginal;
    const double d = r.newton_decrement;
    if (!(std::isfinite(d) && d > 0.0)) return r.log_marginal;
    const double g = r.newton_step_gain;
    const double add = std::isfinite(g) ? std::min(0.5 * d, std::max(g, 0.0))
                                        : 0.0;
    return r.log_marginal + add;
}

// Pack per-cell inner-solve results into the outer-grid result list every
// nested-Laplace grid entry returns: per-cell log-marginal and solve health,
// modes, the stored precisions and fixed-effect blocks, and the per-cell
// diagnostics. `n_threads_outer_realised` < 0 omits that field (an empty grid
// reports none). The one packing behind run_nested_laplace_grid and the batched
// joint driver, so a species of a fused solve returns the list a single-species
// grid returns.
Rcpp::List nl_pack_grid_results(
    const std::vector<LaplaceResult>& cell_results, int n_grid, int n_x,
    bool store_modes, int n_threads_outer_realised
);

Rcpp::List run_nested_laplace_grid(
    int n_grid, int n_x,
    const NlCellSolve& solve_at_theta_fn,
    Rcpp::NumericVector x_init = Rcpp::NumericVector(),
    bool store_modes = true,
    int n_threads_outer = 1,
    const std::vector<int>& tile_ids = std::vector<int>(),
    const std::vector<int>& tile_pilot_cells = std::vector<int>(),
    const NlCheapEval& cheap_eval = NlCheapEval(),
    double prune_tol = 0.0,
    tulpa_progress::GridProgress* progress = nullptr,
    GridCheckpoint* ckpt = nullptr,
    const std::vector<double>& x_init_per_cell = std::vector<double>(),
    // Inner Newton steps per cell in the cheap screening sweep. Appended last
    // so every existing positional call site keeps its meaning.
    int screen_iters = CHEAP_SCREEN_ITERS,
    // Per-cell log hyperprior + log cell measure the screen ranks with, and the
    // mass of the grid the cells join (ScreenOffset); empty ranks on the
    // kernel's log-marginal alone.
    const ScreenOffset& screen_log_offset = ScreenOffset(),
    // Return the screened surface instead of solving the kept cells in full:
    // every cell but the pilot carries its cheap log-marginal and the pilot's
    // mode. For a grid read only to DETECT where the posterior sits (a
    // placement pilot), never for one that is integrated. Inert without a
    // screen (prune_tol = 0).
    bool screen_only = false
);

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_GRID_H
