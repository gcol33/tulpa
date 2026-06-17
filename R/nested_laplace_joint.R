#' Joint multi-likelihood nested Laplace approximation
#'
#' @description
#' Outer-grid nested Laplace driver for *joint* models — multiple response
#' arms sharing one latent prior block, parameterized as a per-arm field
#' amplitude (sigma) on a unit-precision latent.
#'
#' Supported priors:
#'  * `"bym2"`       — outer grid over `(sigma, rho [, alpha])`.
#'                     Latent: `phi (n_s) | theta (n_s)` with unit-precision
#'                     ICAR + iid components.
#'  * `"icar"`       — outer grid over `(sigma [, alpha])`.
#'                     Latent: `phi (n_s)` with unit-precision ICAR.
#'  * `"car_proper"` — outer grid over `(sigma, rho_car [, alpha])`.
#'                     Latent: `phi (n_s)` with `Q = D - rho_car * W`.
#'
#' Other backends (NNGP, HSGP, RW1/2, AR1) follow the same interface and
#' land under Phase 3.
#'
#' @section (sigma, alpha) parameterization:
#' Each arm's linear predictor reads
#' \deqn{\eta_{arm} = X_{arm} \beta_{arm} + \sigma_{arm} \cdot z_s,}
#' where \eqn{z_s} is a unit-precision latent (ICAR(tau=1) for ICAR/BYM2,
#' or the BYM2 mix; CAR_proper uses the structure of
#' \eqn{D - \rho_{car} W}). All arms share the donor amplitude
#' \eqn{\sigma} from the outer-grid `sigma_grid` axis. The copy arm's
#' amplitude is \eqn{\sigma_{arm} = \alpha \cdot \sigma}, where \eqn{\alpha}
#' is a direct outer-grid axis taken from `copy$alpha_grid`. The Cartesian
#' product is over `(sigma, [rho/rho_car,] alpha[, phi])`. Direct \eqn{\alpha}
#' as an outer-grid axis (rather than a post-hoc ratio
#' \eqn{\alpha = \sigma_{pos} / \sigma_{occ}}) avoids plug-in bias on the
#' weakly-identified ratio at small `n_pos` and lets a regularizing
#' hyperprior land on \eqn{\alpha} directly.
#'
#' @param responses A named list of arm specs (length >= 1). Each arm:
#'   * `y`           — numeric `[N_arm]` response.
#'   * `n_trials`    — integer `[N_arm]` (use `rep(1L, N_arm)` for non-binomial).
#'   * `X`           — numeric matrix `[N_arm x p_arm]` fixed-effects design.
#'   * `spatial_idx` — integer `[N_arm]`, 1-based map obs -> spatial unit.
#'   * `re_idx`      — optional numeric `[N_arm]` 1-based RE group index;
#'                     defaults to `rep(0, N_arm)` (no RE).
#'   * `n_re_groups` — optional integer (default `0L`).
#'   * `sigma_re`    — optional numeric (default `1`); ignored when
#'                     `n_re_groups == 0`.
#'   * `family`      — one of `"binomial"`, `"gaussian"`, `"poisson"`,
#'                     `"neg_binomial_2"`, `"beta"`, `"lognormal"`,
#'                     `"gamma"`, `"inverse_gaussian"`. For `"lognormal"`,
#'                     `y` is on the natural scale and the linear
#'                     predictor `eta = E[log y]` (identity link on the
#'                     log scale); the `-log(y)` Jacobian is included in
#'                     the kernel's `log_lik`.
#'   * `phi`         — numeric dispersion (gaussian/lognormal residual
#'                     SD, negbin size, beta precision); default `1`.
#'   * `field_coef`  — optional per-arm field coefficient controlling
#'                     this arm's multiplier on the shared latent field's
#'                     amplitude. One of:
#'                     * numeric scalar (default `1`) -- constant
#'                       multiplier. `0` means the arm carries NO field
#'                       at all (the per-row scatter `eta += field_coef *
#'                       sigma * z` is skipped for that arm).
#'                     * character of length 1 -- names an outer-grid
#'                       hyperparam axis (currently `"alpha"`); the
#'                       coefficient varies across the grid.
#'                     * `list(name = , grid = )` -- embedded axis
#'                       declaration, equivalent to declaring the axis
#'                       and naming it on this arm.
#'                     At most one arm may declare a hyperparam-driven
#'                     axis in the first ship (the cover hurdle and
#'                     occu_cover both need only one). Shared axes
#' across multiple arms are deferred. A single-block
#' copy coefficient is declared here, on the arm --
#' not through a separate `copy` argument.
#'
#' @param prior A list describing the shared latent prior block. Required
#'   field `type`. Backend-specific fields:
#'   * **bym2**: `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`,
#'     `n_neighbors`, `scale_factor` (default `1`); optional `sigma_grid`
#'     (donor-arm field amplitude, default 5 log-spaced values in
#'     `[0.1, 3]`), `rho_grid` (default `c(0.2, 0.5, 0.8, 0.95)`).
#'   * **icar**: `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`,
#'     `n_neighbors`; optional `sigma_grid` (default 5 log-spaced values
#'     in `[0.1, 3]`).
#'   * **car_proper**: same as icar plus `rho_car_grid`
#'     (default `c(0.5, 0.8, 0.95, 0.99)`).
#'
#' @param copy Multi-block copy specification (multi-block `prior` only). For a
#' single-block fit there is no `copy` argument: declare the copy coefficient
#' on the arm via `responses[[X]]$field_coef = list(name = "alpha", grid = G)`.
#' On the multi-block path `copy` is an unnamed list of specs --
#' `list(list(arm, block, alpha_grid),...)` -- coupling N distinct shared
#' latent fields, each onto its own arm with its own \eqn{\alpha} axis,
#' integrated over the product outer grid. Each spec must name a distinct
#' block. The copy block may be any of `icar` / `bym2` / `car_proper` / `rw1`
#' / `rw2` / `ar1` / `iid`; blocks with their own per-arm scaling (`lf`,
#'   `hsgp_mo`) or a precomputed precision (`tgmrf`) do not take a copy.
#'
#' @param phi_grid Optional list specifying per-arm dispersion axes on the
#'   outer grid. Accepts either a named list (keys = arm names) or a
#'   positional list of length `n_arms`. Each entry is one of:
#'   * `NULL` or scalar — no axis for that arm; the kernel uses the
#'     parse-time scalar `responses[[k]]$phi`.
#'   * numeric vector of length > 1 — adds a new outer-grid axis
#'     `phi_<arm>` taking those values; the kernel rewrites
#'     `arms[k].phi` at each grid point before the inner Newton solve.
#'
#'   Family-specific interpretation of `arm$phi` (the parse-time scalar
#'   and the grid values):
#'   * `gaussian` — residual SD (variance is `phi^2`). Use `phi_grid` to
#'     estimate the residual SD as a hyperparameter instead of pinning
#'     it pre-fit.
#'   * `lognormal` — residual SD on the log scale; identical kernel
#'     parameterization as `gaussian` plus the `-log(y)` Jacobian.
#'   * `neg_binomial_2` — dispersion (variance is `mu + mu^2/phi`).
#'   * `beta` — precision (variance is `mu(1-mu)/(1+phi)`).
#'   * `gamma`, `inverse_gaussian` — shape / dispersion.
#'   * `binomial`, `poisson` — ignored.
#'
#'   Each `phi_<arm>` axis is appended to the Cartesian product and
#'   varies slowest (within-spatial warm starts hold). The axis appears
#'   as a regular hyperparameter in `theta_grid`, `theta_mean`, and
#'   `theta_sd`, and participates in adaptive-grid refinement.
#'
#' @param prior_sigma,prior_alpha Optional regularizing hyperpriors on the
#'   donor field amplitude \eqn{\sigma} and on the copy coefficient
#'   \eqn{\alpha}. Each is `NULL` (flat, default) or a list of the form
#'   `list(family, params)`:
#'   * `list("pc.prec", c(U, alpha))` -- Penalized Complexity prior,
#'     calibrated by `P(theta > U) = alpha`. Closed-form density
#'     `lambda * exp(-lambda * theta)` with `lambda = -log(alpha)/U`.
#'     Drop-in for the weakly-identified small-`n_pos` regime
#'. Pick `U` at the upper end of plausible values
#'     so the prior shrinks the tail without biasing the modal cell when
#'     the data identifies it: default-friendly choice on \eqn{\sigma} is
#'     `c(U = 1.0, alpha = 0.01)` (donor amplitude); on the dimensionless
#'     copy coefficient \eqn{\alpha} the recommended choice is
#'     `c(U = 2.0, alpha = 0.01)`. Choosing `U` smaller than truth on
#'     \eqn{\alpha} actively shrinks past the modal cell -- e.g. on
#'     a fixture with truth \eqn{\alpha = 1}, `c(U = 1.0, alpha = 0.01)`
#'     puts `P(alpha > 1) = 0.01` and pulls the posterior toward zero.
#'   * `list("half_normal", scale)` -- half-normal with scale `scale > 0`.
#'     Sharper tail decay than PC; use when stronger regularization is
#'     desired and the truth is well inside the prior.
#'   The contribution is added to `log_marginal` cell-by-cell at the
#'   kernel-call boundary, so refinement passes (adaptive grid,
#'   var-of-means consistency) see the regularized posterior. When the
#'   data identifies the parameter (e.g. `n_pos >= ~200`) the prior is
#'   essentially harmless -- the lever is tail-shrinkage at small
#'   `n_pos`. `prior_alpha` only applies when `copy` is active;
#'   `prior_sigma` applies on any `sigma`-named axis.
#'
#' @param cell_coupling Character scalar naming a per-cell coupled likelihood
#'   registered against tulpa's process-global registry (see
#'   `src/cell_coupling_registry.h`). Defaults to `"separable"`, the arm-
#'   separable per-obs sum every existing joint fit uses. Consumer packages
#'   (e.g. tulpaObs) compile a `tulpa::CellCouplingSpec` subclass in their
#'   own `src/` and register it from `R_init_<pkg>` via the
#'   `tulpa_register_cell_coupling` C callable; the R driver validates the
#'   name against the registry and the inner Newton routes the per-cell
#'   contribution through `evaluate_cell()` when the spec couples at least
#' one arm.
#'
#' @param control Optional list of perf/numerical tuning knobs (statistical
#'   arguments stay top-level), following the `control` convention of
#'   [tulpa()]. Recognised elements (defaults in parentheses):
#'   * `max_iter` (`50L`), `tol` (`1e-6`) -- inner Newton iteration budget and
#'     tolerance.
#'   * `n_threads` (`1L`) -- inner-loop OpenMP threads (per-observation
#'     scatter, compute_eta, log-likelihood reduction). For typical joint
#'     workloads (`N` in the hundreds to a few thousand) inner parallelism is
#'     overhead-dominated (see `dev_notes/issue_body_adaptive_grid_cost.md`);
#'     prefer `n_threads_outer`, which stacks better on many-core hardware.
#'     Capped at the physical performance-core count by default (see
#'     `n_threads_scatter`), as the inner per-observation loops oversubscribe a
#'     hybrid CPU's efficiency cores past that point.
#'   * `n_threads_scatter` (performance-core count) -- cap on the inner
#'     per-observation threads. Overrides the default performance-core cap on
#'     `n_threads`; raise it to use all logical cores or lower it to leave
#'     headroom. No effect where the core topology cannot be resolved (off
#'     Windows), where `n_threads` is used as requested.
#'   * `n_threads_outer` (`1L`) -- outer-grid OpenMP threads. When `> 1`, a
#'     pilot Laplace at the centre cell warm-starts the remaining cells, each
#'     dispatched across `n_threads_outer` threads with its own CHOLMOD solver
#'     and NewtonScratch (inner OpenMP auto-disabled). `1L` is serial, chained
#'     warm-starts -- bitwise identical to the pre-speedup driver. Recommended
#'     on multi-core workstations: `parallel::detectCores() - 1L`.
#'   * `tile_warm` (`TRUE`) -- when `n_threads_outer > 1` and a copy block is
#'     present, group outer cells into tiles sharing every axis except the copy
#'     coefficient `alpha`, solve one warm Tier-2 per tile from the centre
#'     pilot, and warm-start the rest from their tile pilot. Falls back to the
#'     single-tier path when no copy block / single tile / `n_threads_outer
#'     <= 1L`. `FALSE` recovers the pre-tiling behaviour (e.g. regression
#'     testing).
#'   * `prune` (`FALSE`), `prune_tol` (`1e-3`) -- opt-in cheap-pass screening.
#'     When `prune = TRUE`, the driver sweeps the outer lattice running a short
#'     inner Newton per cell, each warm-started from the previous screened
#'     cell's quasi-mode (lattice-adjacent), computes a screening Laplace
#'     log-marginal, softmax-normalises, and skips the full inner Newton on
#'     cells whose normalised weight is `< prune_tol`. The neighbour-warm-start
#'     sweep keeps every cheap mode near its cell's true mode, so the cheap
#'     ranking is faithful to the full-solve ranking even when the inner latent
#'     mode moves substantially across the grid. Pruned cells get
#'     `log_marginal = -Inf`, `n_iter = 0`, and inherit the pilot mode; the
#'     pilot cell is never pruned. A safety gate falls back to the full grid
#'     (with a warning) if the cheap-screen argmax disagrees with the
#'     full-solve argmax or the kept posterior collapses onto a cell the screen
#'     badly mis-estimated, so a silently-wrong pruned posterior is impossible.
#'     Stacks with `n_threads_outer`. `prune_tol` must be in `[0, 1)`. Keep it
#'     conservative (`<= 1e-3`); pruning helps most when the grid has many
#'     low-mass tail cells. Default `FALSE` (the full grid is correct).
#'   * `x_init` (`NULL`) -- warm-start for the first grid point's inner solve.
#'   * `verbose` (`FALSE`) -- when `TRUE`, announce the engaged outer integrator
#'     for a multi-block prior in one line at selection time (see `integration`),
#'     and report each CCD decline reason.
#'   * `store_Q` (`FALSE`) -- also return the per-grid joint precision Q (lower
#'     triangle, CSC) as `Q_csc_p_per_grid`, `Q_csc_i_per_grid`,
#'     `Q_csc_x_per_grid`, `Q_csc_n`, letting callers compute INLA-style
#'     total-variance posterior moments (`Var-of-means + Mean-of-Var`) on inner
#'     latent coordinates such as fixed-effect betas.
#'   * `adaptive_grid` (`FALSE`), `adaptive_grid_edge_thresh` (`0.02`),
#'     `adaptive_grid_max_passes` (`1L`) -- when `adaptive_grid = TRUE`, a
#'     mode-tracked 1D refinement pass triggers on any axis whose marginal
#'     boundary weight exceeds `adaptive_grid_edge_thresh`. New points are
#'     appended on that axis (interior densification + outward log-spaced
#'     extension) paired with the boundary cell's modal other-axis values, each
#'     carrying a calibration term so it contributes on the marginal scale --
#'     `O(n_new_points)` kernel solves, not the full cartesian product. The
#'     edge score is `max(marginal_weight_at_boundary, exp(max_log_marginal_at
#'     _boundary - max_log_marginal_overall))`, catching both boundary pile-up
#'     and integrand truncation; `0.02` is ~4 log units of decay.
#'     `adaptive_grid_max_passes` caps the passes (one usually suffices). Fixes
#'     posterior CI under-coverage when truth sits near a grid edge.
#'   * `var_of_means_consistency` (`TRUE`) -- run a post-integration
#'     consistency pass on the variance of the per-arm posterior means and
#'     attach `var_of_means_consistency_info`.
#'   * `force_sparse` (`FALSE`) -- force the sparse linear-algebra backend for
#'     the inner joint solve regardless of the dense/sparse heuristic.
#'   * `integration` (`"auto"`) -- outer-grid node layout for a multi-block
#'     prior. A central composite design (CCD) integrates the hyperparameter
#'     posterior on `1 + 2d + 2^d` nodes oriented by the Cholesky of the
#'     posterior covariance at the joint hyperparameter mode -- far fewer inner
#'     solves than the `d`-dimensional tensor product (`25` vs `81` at `d = 4`).
#'     `"auto"` (the default) uses the CCD only at `>= 4` transformable axes,
#'     where the tensor product's `k^d` blow-up bites hardest, and keeps the
#'     cheaper, more ridge-robust tensor grid at `<= 3` axes; `"ccd"` lowers the
#'     CCD threshold to `>= 3` axes; `"grid"` always forces the full tensor
#'     product. The CCD auto-falls back to the tensor grid for an axis whose
#'     support is not safely transformable (a CAR_proper `rho_car` or a non-BYM2
#'     `rho`), or a flat / ridged / degenerate outer mode-find or Hessian; an
#'     active `phi_grid` rides as a tensor axis crossed on top of the CCD.
#'     Single-block joint priors always use the tensor grid. The CCD mode-find
#'     runs cheap warm-started inner solves and does not write to the checkpoint
#'     file. Under `verbose = TRUE` the engaged integrator is announced in one
#'     line at selection time (e.g. `outer integration: CCD (4 latent axes, 25
#'     nodes)` or `tensor grid (72 cells)`, or `CCD declined -> tensor grid`),
#'     so the auto switch to the CCD at `>= 4` axes is never silent. The
#'     resolved integrator is also returned on the joint result as
#'     `$integration`.
#'   * `inner_refresh` (`1L`) -- inner-Newton Cholesky factor reuse interval
#'     (Shamanskii / chord method). For a non-quadratic positive arm (e.g. a
#'     beta cover arm) the latent Hessian changes every inner iteration, so the
#'     default re-factorizes the sparse Cholesky on each step -- the dominant
#'     per-grid-cell cost. `inner_refresh = m > 1` re-factorizes only every
#'     `m`-th inner step and reuses the cached factor in between (refreshing
#'     early whenever a reused solve fails). The gradient is exact on every step
#'     and each step is line-search safeguarded, so the converged mode is
#'     unchanged and the final mode-pass Hessian (`log_det`, SEs) is always
#'     fresh; only the path to the mode uses a stale curvature, which may cost a
#'     few extra inner iterations. Applies to the sparse joint path with the
#'     default `control$hessian = "lm"` curvature; the dense small-`n_x` path
#'     re-factorizes a cheap Hessian and ignores it. `2L`-`4L` is a good range
#'     for a slow beta arm.
#'   * `diagnose_k` (`TRUE`), `k_samples` (`200L`) -- compute the outer
#'     Pareto-\eqn{\hat{k}} accuracy diagnostic by importance-sampling the
#'     joint hyperparameter posterior against the Gaussian proposal the
#'     integrator fits (mixed per-axis transforms: `log` for positive scales,
#'     logit for the BYM2 mixing weight, identity for the copy coefficient
#'     \eqn{\alpha}). `k_samples` is the number of importance draws, each one
#'     extra inner joint solve; the draw is RNG-restored so the fit's modes /
#'     draws are unchanged. A fit carrying an axis whose support is not safely
#'     known (CAR_proper's `rho_car`) declines to the quadrature-ESS fallback
#'     (`pareto_k = NA`) rather than apply a guessed transform. `FALSE` skips
#'     the diagnostic (`pareto_k = NA`).
#'   * `k_threads` (`NULL`) -- outer-thread width for the diagnostic's
#'     importance batch. The `k_samples` re-solves are independent and run after
#'     the grid (every core free), each solved single-threaded once the batch
#'     saturates the pool, so widening it is a bit-identical wall-clock speedup
#'     (the k-hat is unchanged). `NULL` follows the fit's own thread grant -- the
#'     larger of `n_threads_outer` and the inner `n_threads` -- so a serial fit
#'     keeps a serial diagnostic (no oversubscription when the caller is itself
#'     parallelising per-species fits) while a threaded fit gets a free parallel
#'     diagnostic. `"auto"` uses the physical performance-core count (capped at 2
#'     under R CMD check) for a single serial fit that wants the diagnostic on
#'     every core with one setting; an integer pins the width (`1L` forces
#'     serial). Always capped at `k_samples`.
#'   * `checkpoint` (`NULL`) -- grid-cell checkpoint/resume. Set
#'     `list(path = "fit.ckpt", resume = TRUE)` to make a killed or interrupted
#'     fit resumable: each completed outer-grid cell is appended to `path`, and
#'     a later call with the same responses + grid + control loads the finished
#'     cells and solves only the rest. EVA-scale joint fits run for hours, so a
#'     wrapper teardown, reboot, or OOM near the end otherwise loses the whole
#'     run. `resume = TRUE` (the default when a `path` is given) continues from
#'     an existing file; `resume = FALSE` removes it first and starts over.
#'     Adaptive-grid refinement cells are checkpointed under their own
#'     coordinate keys, so resume covers them too. A file written for different
#'     data or solver settings is rejected (fingerprint mismatch) rather than
#'     resumed onto a stale result.
#'
#' @return A list of class `c("tulpa_nested_laplace_joint",
#'   "tulpa_nested_laplace", "list")` with:
#'   * `theta_grid`, `theta_names` — outer-grid hyperparameter values
#'     (includes the `alpha` axis when `copy` is set).
#'   * `log_marginal`, `weights` — per-grid-point log-marginal and integration
#'      weights (sum to 1).
#'   * `theta_mean`, `theta_sd` — posterior moments per hyperparameter,
#'      including `alpha` when `copy` is set.
#'   * `theta_median`, `theta_ci_lo`, `theta_ci_hi` — weighted-quantile
#'      median and 2.5/97.5 empirical CI per hyperparameter axis (same
#'      names as `theta_mean`). Recommended summary for right-skewed
#'      scale-like axes (alpha at small n_pos, sigma/range/phi near
#'      a boundary), where the posterior mean is pulled by the right
#'      tail away from the bulk and `mean +/- 1.96 sd` mis-states the
#'      uncertainty.
#'   * `modes` — `[n_grid x n_x]` matrix of inner modes.
#'   * `n_iter` — inner Newton iterations per grid point.
#'   * `arm_layout` — list with per-arm `beta_start`, `re_start`,
#'      spatial offset(s) and `n_x` for decoding modes.
#'   * `prior`, `responses`, `copy` — echoed inputs.
#'   * `timing` — named numeric of wall-clock seconds: `total` plus the
#'      `setup` (validation / encoding / grid construction), `grid` (inner
#'      Laplace solves, including adaptive-refinement and consistency passes),
#'      `postproc` (weight / moment / marginal assembly) and `diagnostics`
#'      (outer Pareto-\eqn{\hat{k}}) phases. The `grid` phase is the one that
#'      scales with grid size and core count. Surfaced one-line in `print`.
#'   * `pareto_k`, `pareto_k_is_ess`, `pareto_k_scope` — outer
#'      Pareto-\eqn{\hat{k}} accuracy diagnostic and its importance-sampling
#'      ESS (both `NA` when `control$diagnose_k = FALSE` or the fit declines;
#'      see the `diagnose_k` control knob). `pareto_k < 0.7` indicates the
#'      nested integration is reliable; `>= 0.7` that the (skewed / heavy-
#'      tailed) hyperparameter posterior is misfit by the Gaussian grid and
#'      the fit should escalate to an exact debias.
#'   * `pareto_k_proposal_source` — how the outer importance proposal the
#'      \eqn{\hat{k}} scores was built: `"mode_hessian"` from the Laplace
#'      curvature at the hyperparameter mode (the CCD design's, or a
#'      finite-difference Hessian when a sharp posterior collapses the grid),
#'      or `"grid_moment"` from the grid-weighted covariance. `NA` when the
#'      diagnostic is off or declines. The mode-Hessian source keeps the
#'      \eqn{\hat{k}} meaningful when the grid concentrates on ~1 cell.
#'   * `adaptive_grid_info` — when `adaptive_grid = TRUE`, a list with
#'      `triggered_axes` (character) and `n_points_added` (integer)
#'      describing the refinement passes. NULL otherwise.
#'   * `prune_cheap_log_marginal`, `prune_mask`, `prune_n_pruned`,
#'      `prune_tol` — present only when `prune = TRUE` and the safety gate did
#'      not fall back. Cheap-pass log-marginals at every cell, a logical mask
#'      of pruned cells, the pruned-cell count, and the threshold actually
#'      applied. Pruned cells have `log_marginal = -Inf` so they get zero
#'      weight under `.nl_normalise_weights_safe`.
#'   * `prune_fallback_triggered`, `prune_fallback_reason` — present only when
#'      the safety gate fell back to the full grid. The returned fit is the
#'      full-grid (unpruned) result; the reason string records which gate
#'      condition tripped.
#'
#' @seealso [tulpa_nested_laplace()] for the single-arm engine.
#' @references
#' Rue, Martino & Chopin (2009). Approximate Bayesian inference for latent
#' Gaussian models by using integrated nested Laplace approximations.
#' \emph{JRSS-B} 71(2):319-392.
#' @examples
#' \donttest{
#' set.seed(1)
#' S <- 25L                                   # shared spatial units (chain graph)
#' nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
#' nn <- lengths(nb)
#' field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))
#' mk_arm <- function(m, fam) {
#'   si <- sample(S, m, replace = TRUE); x <- rnorm(m)
#'   lin <- 0.2 + 0.5 * x + 0.8 * field[si]
#'   y <- if (fam == "binomial") rbinom(m, 1L, plogis(lin)) else lin + rnorm(m, 0, 0.5)
#'   list(y = as.numeric(y), n_trials = rep(1L, m), X = cbind(1, x),
#'        spatial_idx = si, family = fam, phi = if (fam == "gaussian") 0.5 else 1)
#' }
#' prior <- list(type = "icar", n_spatial_units = S,
#'               adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
#'               n_neighbors = nn, sigma_grid = c(0.3, 0.7, 1.4))
#' fit <- tulpa_nested_laplace_joint(
#'   responses = list(occ = mk_arm(200L, "binomial"), pos = mk_arm(200L, "gaussian")),
#'   prior = prior)
#' fit$theta_mean        # shared field amplitude, integrated across both arms
#' }
#' @export
tulpa_nested_laplace_joint <- function(responses,
                                       prior,
                                       copy = NULL,
                                       phi_grid = NULL,
                                       prior_sigma = NULL,
                                       prior_alpha = NULL,
                                       cell_coupling = "separable",
                                       control = list()) {
    tm <- .tulpa_timer()                               # gcol33/tulpa#48
    # Resolve and validate the cell-coupling spec name against the C++
    # registry (separable default is auto-registered on first touch). The
    # resolved spec is held on the result but not yet routed through the
    # inner-Newton scatter -- the per-cell branch in
    # scatter_arm_obs_joint_multi[_sparse] lands with Layer B of #32 Change 2.
    # In Layer A the default sentinel ("separable", arm_ids() empty) keeps
    # every existing joint fit on the per-obs path with no behavioural change.
    if (!is.character(cell_coupling) || length(cell_coupling) != 1L ||
        is.na(cell_coupling) || !nzchar(cell_coupling)) {
        stop("`cell_coupling` must be a single non-empty character string ",
             "naming a registered spec (default: \"separable\").",
             call. = FALSE)
    }
    if (!cpp_cell_coupling_registry_has(cell_coupling)) {
        stop("`cell_coupling = \"", cell_coupling, "\"` is not registered. ",
             "Consumer packages register specs from R_init_<pkg> via the ",
             "tulpa_register_cell_coupling C callable; the built-in ",
             "\"separable\" default is always available.",
             call. = FALSE)
    }

    # Perf/numerical knobs live in `control = list()` (matching tulpa()); the
    # top-level signature carries only statistical arguments.
    max_iter                  <- control$max_iter %||% 50L
    tol                       <- control$tol %||% 1e-6
    n_threads                 <- .tulpa_inner_threads(control$n_threads %||% 1L,
                                                       control$n_threads_scatter)
    n_threads_outer           <- control$n_threads_outer %||% 1L
    tile_warm                 <- control$tile_warm %||% TRUE
    prune                     <- control$prune %||% FALSE
    prune_tol                 <- control$prune_tol %||% 1e-3
    x_init                    <- control$x_init
    verbose                   <- control$verbose %||% FALSE
    store_Q                   <- control$store_Q %||% FALSE
    adaptive_grid             <- control$adaptive_grid %||% FALSE
    adaptive_grid_edge_thresh <- control$adaptive_grid_edge_thresh %||% 0.02
    adaptive_grid_max_passes  <- control$adaptive_grid_max_passes %||% 1L
    var_of_means_consistency  <- control$var_of_means_consistency %||% TRUE
    force_sparse              <- control$force_sparse %||% FALSE
    # Outer Pareto-k-hat accuracy diagnostic. `diagnose_k` (default TRUE)
    # importance-samples the joint hyperparameter posterior against the
    # Gaussian proposal the integrator fits; `k_samples` (default 200) is the
    # number of draws, each one extra inner joint solve. RNG-restored, so the
    # fit's draws are unchanged whether or not it runs.
    diagnose_k                <- control$diagnose_k %||% TRUE
    k_samples                 <- control$k_samples %||% 200L
    # Outer-thread width for the diagnostic's importance batch (gcol33/tulpa#117).
    # The `k_samples` re-solves are independent and run after the grid (all cores
    # free), each solved single-threaded, so widening this pool is a bit-identical
    # wall-clock speedup. `NULL` (default) follows the fit's thread grant; "auto"
    # uses the performance cores; an integer pins the width. Resolved once here and
    # threaded into both the single- and multi-block diagnostic attach.
    k_threads                 <- control$k_threads
    pareto_k_threads          <- .tulpa_pareto_k_threads(n_threads_outer,
                                                         n_threads, k_samples,
                                                         k_threads)
    # Outer-grid node layout for the multi-block path (gcol33/tulpa#59). A CCD
    # places a central-composite design around the joint hyperparameter mode --
    # far fewer inner solves than the full tensor product (1 + 2d + 2^d vs k^d).
    # "auto" (default) uses the CCD only where the tensor blow-up bites hardest,
    # >= 4 transformable axes, and the tensor grid at <= 3 axes (the d = 3 grid
    # is cheap and a ridged 3-axis posterior integrates more reliably on it).
    # "ccd" lowers the threshold to >= 3 axes; "grid" always forces the tensor
    # product. The CCD auto-falls back to the tensor grid for an unguessable axis
    # (CAR_proper rho_car / non-BYM2 rho), a flat / ridged outer posterior, or a
    # degenerate mode-find (gcol33/tulpa#62); an active phi grid rides as a
    # tensor axis crossed on top of the CCD (gcol33/tulpa#61). Single-block joint
    # backends always use the tensor grid (CCD applies to the multi-block path).
    integration               <- match.arg(control$integration %||% "auto",
                                            c("auto", "ccd", "grid"))
    # Inner-Newton curvature + PD enforcement for the (possibly indefinite)
    # joint mixture Hessian. "lm" (default) escalates a diagonal ridge until
    # CHOLMOD factorizes the observed Hessian; "psd" eigen-clamps the dense
    # observed Hessian; "fisher" scatters the complete-data expected
    # information (PSD by construction, factorizes first try) for the inner
    # step while keeping the observed Hessian for the final mode-pass log_det
    # and SEs. Two C++ axes: hessian_pd_mode (0 = LM, 1 = PSD) and
    # step_curvature_mode (0 = observed, 1 = Fisher).
    hessian                   <- match.arg(control$hessian %||% "lm",
                                           c("lm", "psd", "fisher"))
    hessian_pd_mode           <- if (identical(hessian, "psd")) 1L else 0L
    step_curvature_mode       <- if (identical(hessian, "fisher")) 1L else 0L
    # Inner-Newton Cholesky factor reuse (Shamanskii / chord method). For a
    # non-quadratic positive arm (e.g. beta cover) the latent Hessian changes
    # every inner iteration, so the plain Newton loop re-factorizes the sparse
    # Cholesky on each step -- the dominant per-grid-cell cost (gcol33/tulpa#46).
    # `inner_refresh = m` re-factorizes only every m-th inner step and reuses
    # the cached factor in between; the gradient stays exact and each step is
    # line-search safeguarded, so the converged mode is unchanged and the final
    # mode-pass Hessian (log_det / SEs) is always fresh. Applies to the sparse
    # joint path (large fields, LM curvature); the dense small-n_x path
    # re-factorizes cheaply and ignores it. Default 1 = re-factorize every step.
    inner_refresh             <- as.integer(control$inner_refresh %||% 1L)
    if (length(inner_refresh) != 1L || is.na(inner_refresh) ||
        inner_refresh < 1L) {
        stop("`control$inner_refresh` must be a single integer >= 1.",
             call. = FALSE)
    }

    # Outer-grid progress (gcol33/tulpa#45). The cpp entry is reached through
    # the polymorphic joint backends (`backend$call_kernel`) and the adaptive-
    # refinement closures, so the four progress knobs travel via a scoped
    # option rather than every backend signature; `.cpp_joint_multi` reads it at
    # the cpp boundary. Restored on exit so the option never leaks past the fit.
    .op_progress <- options(tulpa.nl_progress = .nl_progress_args(control))
    on.exit(options(.op_progress), add = TRUE)

    # Grid-cell checkpoint/resume (gcol33/tulpa#50). Threaded to the cpp
    # boundary via a scoped option, like progress, so every backend / adaptive-
    # refinement kernel call within this fit shares one checkpoint file. On a
    # fresh run (resume = FALSE) any prior file is removed once here, before the
    # first kernel call, so the several within-fit calls all append rather than
    # truncate each other. A resume (the default) keeps the file and the C++
    # layer loads its completed cells.
    .ckpt <- .nl_checkpoint_args(control)
    .op_checkpoint <- options(tulpa.nl_checkpoint = .ckpt)
    on.exit(options(.op_checkpoint), add = TRUE)
    if (nzchar(.ckpt$path) && !isTRUE(.ckpt$resume) &&
        file.exists(.ckpt$path)) {
        file.remove(.ckpt$path)
    }

    if (!is.list(responses) || length(responses) < 1L) {
        stop("`responses` must be a non-empty list of arm specs.", call. = FALSE)
    }
    if (!is.list(prior)) {
        stop("`prior` must be a list with a `type` field, or a list of block specs.",
             call. = FALSE)
    }
    # Parse user-specified regularizing hyperpriors on (sigma, alpha) once,
    # at the entry point. Validation errors raised here surface to the user
    # without going through the multi-block / single-block fork.
    fn_sigma <- .joint_parse_sigma_prior(prior_sigma, "prior_sigma")
    fn_alpha <- .joint_parse_sigma_prior(prior_alpha, "prior_alpha")

    # Detect multi-block prior (list-of-blocks). Routed to a separate
    # dispatch path that builds vector<LatentBlock> on the joint side.
    # Pruning toggle: the kernel takes prune_tol > 0 to mean "screen and
    # prune"; gate it here with the user-facing `prune` boolean so the toggle
    # composes cleanly (prune=FALSE always disables, prune=TRUE uses
    # prune_tol). prune_tol must be in [0, 1) — clamp out-of-range to 0
    # rather than erroring, because a zero tolerance is the safe no-op.
    if (!isTRUE(prune)) {
        prune_tol_eff <- 0.0
    } else {
        prune_tol_eff <- as.numeric(prune_tol)
        if (!is.finite(prune_tol_eff) || prune_tol_eff < 0 ||
            prune_tol_eff >= 1) {
            stop("`prune_tol` must be a finite numeric in [0, 1).",
                 call. = FALSE)
        }
    }

    if (.is_multi_block_prior(prior)) {
        return(.joint_dispatch_multi(
            responses = responses, prior_list = prior, copy = copy,
            phi_grid = phi_grid,
            fn_sigma = fn_sigma, fn_alpha = fn_alpha,
            max_iter = max_iter, tol = tol, n_threads = n_threads,
            n_threads_outer = n_threads_outer,
            tile_warm = tile_warm,
            prune_tol = prune_tol_eff,
            x_init = x_init, verbose = verbose, store_Q = store_Q,
            force_sparse = force_sparse,
            cell_coupling = cell_coupling,
            diagnose_k = diagnose_k, k_samples = k_samples,
            pareto_k_threads = pareto_k_threads,
            inner_refresh = inner_refresh,
            integration = integration,
            timer = tm
        ))
    }
    if (!is.null(copy)) {
        stop("`copy` is not used on the single-block ",
             "tulpa_nested_laplace_joint() path. Declare the copy coefficient on ",
             "the arm itself: field_coef = list(name = \"alpha\", grid = <grid>). ",
             "(`copy` remains a multi-block argument -- a list of block specs, ",
             "each naming a distinct copy block.)", call. = FALSE)
    }
    if (is.null(prior$type)) {
        stop("`prior` must be a list with a `type` field, or a list of block specs.",
             call. = FALSE)
    }
    type <- tolower(prior$type)
    backend <- .joint_backends[[type]]
    if (is.null(backend)) {
        stop("tulpa_nested_laplace_joint(): unsupported prior$type '", type,
             "'. Supported: ",
             paste(shQuote(names(.joint_backends)), collapse = ", "),
             ".", call. = FALSE)
    }

    arms <- lapply(seq_along(responses), function(k) {
        a <- responses[[k]]
        .normalise_joint_arm(a, k)
    })

    # `.resolve_copy` reads the per-arm `field_coef_axis` / `field_coef_const`
    # populated by `.normalise_joint_arm`; pass the normalised `arms` rather
    # than the raw `responses` list.
    cp <- .resolve_copy(arms, prior, type)
    arm_names <- names(responses) %||% paste0("arm", seq_along(responses))
    phi_axes <- .normalise_phi_grid(phi_grid, arm_names)
    grids <- backend$build_grids(prior, cp$has_copy, cp$alpha_grid, phi_axes)
    tm$mark("setup")

    call_kernel_with_tol <- function(tol_prune) {
        backend$call_kernel(arms, prior, cp, grids, max_iter, tol,
                            n_threads, x_init, isTRUE(store_Q),
                            arm_names = arm_names,
                            n_threads_outer = n_threads_outer,
                            tile_warm = tile_warm,
                            prune_tol = tol_prune,
                            force_sparse = force_sparse,
                            cell_coupling = cell_coupling,
                            hessian_pd_mode = hessian_pd_mode,
                            step_curvature_mode = step_curvature_mode,
                            inner_refresh = inner_refresh)
    }
    res <- call_kernel_with_tol(prune_tol_eff)
    # Safety gate: if the cheap-pass ranking is unreliable (the screen's
    # argmax disagrees with the full-solve argmax, or the kept posterior
    # collapses onto a cell the screen badly mis-estimated), warn and fall
    # back to the full grid rather than silently returning a pruned answer
    # (gcol33/tulpa#43). A silently-wrong posterior must be impossible.
    if (prune_tol_eff > 0) {
        res <- .joint_prune_safety_gate(
            res, resolve_full = function() call_kernel_with_tol(0.0))
    }
    tm$mark("grid")

    # Bake the regularizing hyperprior on (sigma, alpha) into log_marginal
    # at the kernel-call boundary. Every cell carries the same prior
    # contribution ratio so refinement decisions (edge scores, modal cell
    # selection, var-of-means thresholds) all read the *regularized*
    # posterior. New cells appended in refinement passes get the prior
    # baked in via the same `hp_fn` closure threaded through the generic
    # `.hyper_apply_axis_refinement`. The generic helper passes the new
    # cells as a numeric matrix, which `.joint_hp_vec_for_grids` already
    # accepts (see `R/nested_laplace_joint_hyperpriors.R`).
    hp_fn <- if (is.null(fn_sigma) && is.null(fn_alpha)) NULL else {
        function(new_cells) {
            .joint_hp_vec_for_grids(new_cells, fn_sigma, fn_alpha)
        }
    }
    if (!is.null(hp_fn)) {
        tg_init <- backend$theta_grid(grids, cp$has_copy)
        hp_init <- hp_fn(tg_init)
        if (!is.null(hp_init)) res$log_marginal <- res$log_marginal + hp_init
    }

    # --- generic-refinement glue (gcol33/tulpa#33 Step 3) -------------------
    # Adaptive grid + var-of-means consistency are now driven by the
    # axis-spec module in `R/hyper_grid_refine.R`. The joint driver builds:
    #   * `specs`     -- one hyper_axis_spec per outer-grid axis (log_scale /
    #                    bounds / refinable / refine_priority encoded once,
    #                    from `.joint_axis_specs`).
    #   * `kernel_fn` -- thin closure around `backend$call_kernel` that
    #                    converts matrix new_cells <-> paired-vector grids and
    #                    packs per-cell modes / n_iter / Q_csc_* into extras.
    #   * `extras_list` -- per-cell side data the kernel returned; refinement
    #                    carries it through, the warm-start chain reads
    #                    `extras[[anchor]]$mode`.
    # After refinement `.joint_glue_extras_to_res` puts the merged extras
    # back into `res$modes` / `res$n_iter` / `res$Q_csc_*_per_grid` so
    # downstream code (`.nl_posterior_moments`, `.nl_refit_axis_sd_laplace`,
    # the modes / Q consumers) reads the refined values unchanged.
    specs <- .joint_axis_specs(grids, cp)
    kernel_fn <- .joint_make_kernel_fn(arms, prior, cp, backend, max_iter,
                                        tol, n_threads, x_init, store_Q,
                                        arm_names,
                                        cell_coupling = cell_coupling,
                                        hessian_pd_mode = hessian_pd_mode,
                                        step_curvature_mode = step_curvature_mode,
                                        inner_refresh = inner_refresh)
    theta_grid_M  <- backend$theta_grid(grids, cp$has_copy)
    log_marginal  <- res$log_marginal
    extras_list   <- .joint_init_extras_from_res(res)
    refining_axis <- rep("", length(log_marginal))

    refine_info <- NULL
    if (isTRUE(adaptive_grid)) {
        refined <- .hyper_adaptive_refine_pass(
            theta_grid    = theta_grid_M,
            log_marginal  = log_marginal,
            extras        = extras_list,
            refining_axis = refining_axis,
            specs         = specs,
            kernel_fn     = kernel_fn,
            edge_thresh   = adaptive_grid_edge_thresh,
            max_passes    = adaptive_grid_max_passes,
            hp_fn         = hp_fn
        )
        theta_grid_M  <- refined$theta_grid
        log_marginal  <- refined$log_marginal
        extras_list   <- refined$extras
        refining_axis <- refined$refining_axis
        refine_info   <- refined$info
    }
    res <- .joint_glue_extras_to_res(res, theta_grid_M, log_marginal,
                                      extras_list, refining_axis)
    tm$mark("grid")                          # adaptive-refinement inner solves

    res$theta_grid  <- theta_grid_M
    res$theta_names <- colnames(res$theta_grid)
    res$weights     <- .nl_normalise_weights_safe(res$log_marginal, "outer grid")
    res             <- .nl_posterior_moments(res, paste0("joint_", type))
    res             <- .joint_recalibrate_axis_moments(res)
    # Replace per-axis var-of-means SDs with Laplace-at-mode SDs where the
    # 3-point parabolic fit at the modal cell succeeds (gcol33/tulpa#20).
    res             <- .nl_refit_axis_sd_laplace(res)
    tm$mark("postproc")

    # Var-of-means consistency pass. Sharply peaked axes (gaussian
    # noise SD, beta phi at high n_pos) collapse joint weight onto a
    # single grid cell, so downstream `sum(w*x^2) - mean^2` underestimates
    # SD even when `theta_sd` (Laplace-at-mode) is correct. Add slice
    # points at `theta_mean +/- k*theta_sd` to repopulate the Laplace
    # support; var-of-means on the merged grid converges to Laplace SD,
    # which lets downstream packages use the legacy `weights * theta_grid`
    # pattern without reaching into `theta_sd` directly (gcol33/tulpa#21).
    if (isTRUE(var_of_means_consistency)) {
        consistency <- .hyper_consistency_pass(
            theta_grid    = theta_grid_M,
            log_marginal  = res$log_marginal,
            extras        = extras_list,
            refining_axis = refining_axis,
            specs         = specs,
            theta_mean    = res$theta_mean,
            theta_sd      = res$theta_sd,
            kernel_fn     = kernel_fn,
            tolerance     = 0.7,
            hp_fn         = hp_fn,
            weights       = res$weights
        )
        if (consistency$n_added > 0L) {
            theta_grid_M  <- consistency$theta_grid
            log_marginal  <- consistency$log_marginal
            extras_list   <- consistency$extras
            refining_axis <- consistency$refining_axis
            res <- .joint_glue_extras_to_res(res, theta_grid_M, log_marginal,
                                              extras_list, refining_axis)
            res$theta_grid  <- theta_grid_M
            res$theta_names <- colnames(res$theta_grid)
            res$weights     <- .nl_normalise_weights_safe(res$log_marginal, "outer grid")
            res             <- .nl_posterior_moments(res, paste0("joint_", type))
            res             <- .joint_recalibrate_axis_moments(res)
            res             <- .nl_refit_axis_sd_laplace(res)
        }
        res$var_of_means_consistency_info <- consistency$info
        tm$mark("grid")                      # consistency-pass inner solves
    }
    res$arm_layout  <- backend$layout(arms, prior)
    res$prior       <- prior
    res$responses   <- responses
    res$copy        <- copy
    res$cell_coupling      <- cell_coupling
    res$adaptive_grid_info <- refine_info
    tm$mark("postproc")
    # Outer Pareto-k-hat: re-evaluate the inner joint marginal at hyperparameters
    # drawn from the integrator's Gaussian proposal (reusing the generic
    # `kernel_fn` + `hp_fn` so no kernel-call machinery is duplicated) and
    # PSIS-smooth. Declines (NA -> quad-ESS) when an axis has unguessable
    # support (e.g. CAR_proper's rho_car).
    res <- .joint_attach_pareto_k_single(res, kernel_fn, hp_fn,
                                         max_iter   = max_iter,
                                         diagnose_k = diagnose_k,
                                         k_samples  = k_samples,
                                         n_threads_outer = pareto_k_threads)
    tm$mark("diagnostics")
    res$timing <- tm$timing()
    .finalize_fit(res, backend = "nested_laplace_joint",
                  extra_class = c("tulpa_nested_laplace_joint",
                                  "tulpa_nested_laplace", "list"))
}

# Thin wrapper over cpp_nested_laplace_joint_multi that injects the outer-grid
# progress knobs (gcol33/tulpa#45) from the scoped `tulpa.nl_progress` option set
# by tulpa_nested_laplace_joint. Every backend / refinement call site routes
# through here, so progress reaches the cpp entry without threading four scalars
# through the polymorphic backend interface. Option unset -> progress = FALSE.
.cpp_joint_multi <- function(...) {
  p <- getOption("tulpa.nl_progress", NULL)
  if (is.null(p)) p <- .nl_progress_args(list(progress = FALSE))
  cp <- getOption("tulpa.nl_checkpoint", NULL)
  checkpoint_path <- if (is.list(cp)) as.character(cp$path) else ""
  do.call(cpp_nested_laplace_joint_multi,
          c(list(...),
            list(progress          = isTRUE(p$progress),
                 progress_every    = as.integer(p$progress_every),
                 progress_throttle = as.numeric(p$progress_throttle),
                 progress_file     = as.character(p$progress_file),
                 checkpoint_path   = checkpoint_path)))
}
