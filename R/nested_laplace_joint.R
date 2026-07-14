#' Joint multi-likelihood nested Laplace approximation
#'
#' @description
#' Outer-grid nested Laplace driver for *joint* models -- multiple response
#' arms sharing one latent prior block, parameterized as a per-arm field
#' amplitude (sigma) on a unit-precision latent.
#'
#' Supported priors:
#'  * `"bym2"`       -- outer grid over `(sigma, rho [, alpha])`.
#'                     Latent: `phi (n_s) | theta (n_s)` with unit-precision
#'                     ICAR + iid components.
#'  * `"icar"`       -- outer grid over `(sigma [, alpha])`.
#'                     Latent: `phi (n_s)` with unit-precision ICAR.
#'  * `"car_proper"` -- outer grid over `(sigma, rho_car [, alpha])`.
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
#'   * `y`           -- numeric `[N_arm]` response.
#'   * `n_trials`    -- integer `[N_arm]` (use `rep(1L, N_arm)` for non-binomial).
#'   * `X`           -- numeric matrix `[N_arm x p_arm]` fixed-effects design.
#'   * `spatial_idx` -- integer `[N_arm]`, 1-based map obs -> spatial unit.
#'   * `re_idx`      -- optional numeric `[N_arm]` 1-based RE group index;
#'                     defaults to `rep(0, N_arm)` (no RE).
#'   * `n_re_groups` -- optional integer (default `0L`).
#'   * `sigma_re`    -- optional numeric (default `1`); ignored when
#'                     `n_re_groups == 0`.
#'   * `family`      -- one of `"binomial"`, `"gaussian"`, `"poisson"`,
#'                     `"neg_binomial_2"`, `"beta"`, `"lognormal"`,
#'                     `"gamma"`, `"inverse_gaussian"`. For `"lognormal"`,
#'                     `y` is on the natural scale and the linear
#'                     predictor `eta = E[log y]` (identity link on the
#'                     log scale); the `-log(y)` Jacobian is included in
#'                     the kernel's `log_lik`.
#'   * `phi`         -- numeric dispersion (gaussian/lognormal residual
#'                     SD, negbin size, beta precision); default `1`.
#'   * `field_coef`  -- optional per-arm field coefficient controlling
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
#'   * `NULL` or scalar -- no axis for that arm; the kernel uses the
#'     parse-time scalar `responses[[k]]$phi`.
#'   * numeric vector of length > 1 -- adds a new outer-grid axis
#'     `phi_<arm>` taking those values; the kernel rewrites
#'     `arms[k].phi` at each grid point before the inner Newton solve.
#'
#'   Family-specific interpretation of `arm$phi` (the parse-time scalar
#'   and the grid values):
#'   * `gaussian` -- residual SD (variance is `phi^2`). Use `phi_grid` to
#'     estimate the residual SD as a hyperparameter instead of pinning
#'     it pre-fit.
#'   * `lognormal` -- residual SD on the log scale; identical kernel
#'     parameterization as `gaussian` plus the `-log(y)` Jacobian.
#'   * `neg_binomial_2` -- dispersion (variance is `mu + mu^2/phi`).
#'   * `beta` -- precision (variance is `mu(1-mu)/(1+phi)`).
#'   * `gamma`, `inverse_gaussian` -- shape / dispersion.
#'   * `binomial`, `poisson` -- ignored.
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
#'     `c(U = 8.0, alpha = 0.01)`. Too small a `U` over-shrinks the copy
#'     coefficient past the modal cell and, through the `alpha * sigma` copy
#'     axis, inflates the coupled donor amplitude `sigma` -- e.g. on a fixture
#'     with truth \eqn{\alpha = 1}, `c(U = 2.0, alpha = 0.01)` pulls the
#'     \eqn{\alpha} posterior below 1 and lifts `sigma` above its truth.
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
#' @param prior_phi Optional regularizing hyperprior on the per-arm
#'   dispersion axes declared through `phi_grid` (e.g. a Beta precision on a
#'   cover arm, a negbin dispersion, a Gaussian residual SD). Same families
#'   as `prior_sigma` -- `NULL` (flat over the phi grid, default),
#'   `list("pc.prec", c(U, alpha))`, or `list("half_normal", scale)`. A
#'   single spec re-weights every `phi_<arm>` axis on the grid, the way
#'   `prior_sigma` re-weights any sigma-named axis; with no `phi_grid` it is
#'   a no-op. Without it the phi grid carries an implicit flat prior over its
#'   bounds. The PC scale is the dispersion's own units (a precision for
#'   `beta`, a size for `neg_binomial_2`), so pick `U` at the upper end of
#'   plausible values.
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
#'     product. `"grid_adaptive"` is the low-dimensional companion to the CCD:
#'     it seeds a coarse subsample of the SAME tensor lattice (latent block axes
#'     and phi axes together), floods outward from the posterior mode on the fine
#'     lattice, and evaluates only the cells within a log-density cutoff of the
#'     peak -- a strict, uniform-weight subset of the dense tensor, so its
#'     posterior matches the dense grid to that cutoff at fewer inner solves when
#'     the hyperparameter posterior concentrates (a sharply-identified field SD /
#'     precision). It declines back to the dense tensor on a diffuse posterior
#'     (kept region would rival the tensor) or a degenerate lattice, so it never
#'     costs accuracy; tune it with `adaptive_grid_cutoff` / `adaptive_grid_stride`
#'     / `adaptive_grid_max_frac`. The CCD auto-falls back to the tensor grid for
#'     an axis whose support is not safely transformable (a CAR_proper `rho_car`
#'     or a non-BYM2 `rho`), or a flat / ridged / degenerate outer mode-find or
#'     Hessian; an active `phi_grid` rides as a tensor axis crossed on top of the
#'     CCD.
#'     Single-block joint priors always use the tensor grid. The CCD mode-find
#'     runs cheap warm-started inner solves and does not write to the checkpoint
#'     file. Under `verbose = TRUE` the engaged integrator is announced in one
#'     line at selection time (e.g. `outer integration: CCD (4 latent axes, 25
#'     nodes)` or `tensor grid (72 cells)`, or `CCD declined -> tensor grid`),
#'     so the auto switch to the CCD at `>= 4` axes is never silent. The
#'     resolved integrator is also returned on the joint result as
#'     `$integration`.
#'   * `local_ccd` (`NULL`) -- local CCD refinement of a multi-block tensor grid.
#'     `TRUE` (defaults) or a `list(max_cells =, f0 =)` refines a few high-weight,
#'     mutually non-adjacent interior cells, replacing each with a small
#'     curvature-aware CCD node cloud so a coarse base grid resolves the
#'     sharply-peaked directions without the `k^d` tensor blow-up. The local
#'     curvature is a diagonal finite difference of the outer log-marginal over
#'     the cell's own grid neighbours (no mode-find; only the off-centre nodes are
#'     new solves), warm-started from the cell's inner mode; each refined cell's
#'     sub-nodes carry partition-of-unity design weights so the total integration
#'     weight is conserved (no double-count). `max_cells` (`8L`) caps the refined
#'     cells; `f0` (`1.1`) is the CCD radius. The design scale is shrunk per cell
#'     so the cloud fits the cell's Voronoi box (the local-Gaussian mass beyond it
#'     belongs to the neighbouring cells). Engages only on the tensor path (the
#'     curvature stencil needs axis neighbours), at `>= 4` transformable latent
#'     axes, with no active `phi_grid`; otherwise it is a no-op. The applied
#'     refinement is summarised on the result as `$local_ccd_info`. Also driven
#'     automatically by `k_refine = "ccd"`.
#'   * `adaptive_grid_cutoff` (`10`), `adaptive_grid_stride` (`2L`),
#'     `adaptive_grid_max_frac` (`0.75`), `adaptive_grid_min_cells` (`48`) --
#'     tuning for `integration = "grid_adaptive"`. `adaptive_grid_cutoff` is the
#'     log-density keep / expand radius from the peak (larger keeps more cells,
#'     closer to the dense tensor); `adaptive_grid_stride` the coarse-seed
#'     subsample stride per axis; `adaptive_grid_max_frac` the kept-fraction
#'     ceiling past which the builder declines back to the dense tensor;
#'     `adaptive_grid_min_cells` the smallest dense tensor worth the adaptive
#'     machinery -- below it the builder declines BEFORE any inner solve (on a
#'     small tensor the coarse seed is already most of the grid, so there is no
#'     tail to skip). Ignored by the other integrators. The kept-cell / dense /
#'     solve counts are returned as `$adaptive_grid_info`.
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
#'   * `k_quality` (`"report"`) -- the reliability intent for the outer
#'     Pareto-\eqn{\hat{k}}, a single statement of how reliable the fit should be.
#'     `"report"` (default) computes the diagnostic and reports the achieved band.
#'     `"ok"` / `"good"` additionally name a TARGET band (the \eqn{\hat{k}}
#'     confidently usable, resp. good) and raise the default `diagnose_draws`
#'     (to `800L` / `2000L`, unless you set it) so the bootstrap CI can resolve it.
#'     `"none"` disables the diagnostic. The fit carries an honest verdict --
#'     `k_quality_requested`, `k_quality_reached`, `k_quality_best`,
#'     `k_quality_reason`, `k_quality_rounds` -- and never silently downgrades: if
#'     the requested band is not confidently met it reports the band actually
#'     reached and why. For `"ok"` / `"good"`, when the first fit does not reach
#'     the band the engine escalates by REFINING THE INTEGRATION GRID, driven by
#'     the bad \eqn{\hat{k}} (see `k_refine`): each round widens / densifies the
#'     grid where the posterior mass escapes its current bounds and re-diagnoses,
#'     up to `k_max_rounds` times. This is the actual fix for a grid-width
#'     deficiency; `diagnose_draws` is the separate knob that sharpens the
#'     \eqn{\hat{k}} ESTIMATE and is not escalated here.
#'   * `k_refine` (`"grid"`) -- the integration-refinement rung for `k_quality`
#'     `"ok"` / `"good"`. `"grid"` (default) re-fits with adaptive grid refinement
#'     (`adaptive_grid`) each escalation round, driven by the bad \eqn{\hat{k}}, so
#'     a too-coarse / too-narrow grid is widened / densified where the importance
#'     weight concentrates until the band is reached or the budget is spent.
#'     `"ccd"` instead refines a few high-weight, mutually non-adjacent interior
#'     cells with local curvature-aware CCD node clouds (see `local_ccd`), the
#'     right rung when the grid is too coarse to resolve a sharply-peaked
#'     direction rather than too narrow; it forces a tensor base grid (the
#'     curvature stencil needs axis neighbours) and engages only on the
#'     multi-block path at >= 4 transformable latent axes. `"none"` disables
#'     refinement: the band is reported but not chased.
#'   * `k_max_rounds` (`2L`) -- the grid-refinement round budget for `k_quality`
#'     `"ok"` / `"good"`: the maximum number of refine-and-re-fit rounds after the
#'     first fit. Each round allows one more refinement pass than the last. `0L`
#'     disables escalation (single-shot, the band is reported but not chased).
#'   * `diagnose_k` (`TRUE`), `diagnose_draws` (`500L`) -- compute the outer
#'     Pareto-\eqn{\hat{k}} accuracy diagnostic by importance-sampling the joint
#'     hyperparameter posterior against the proposal the integrator fits (mixed
#'     per-axis transforms: `log` for positive scales, logit for the BYM2 mixing
#'     weight, identity for the copy coefficient \eqn{\alpha}). `diagnose_draws`
#'     is the number of importance draws, each one an extra inner joint solve, and
#'     is the diagnostic's precision knob: a tighter k-hat needs MORE actual tail
#'     ratios, so increase `diagnose_draws` (not `k_bootstrap`). The draws are
#'     RNG-restored so the fit's modes / draws are unchanged. A fit carrying an
#'     axis whose support is not safely known (CAR_proper's `rho_car`) declines to
#'     the quadrature-ESS fallback (`pareto_k = NA`). `FALSE` skips the diagnostic.
#'     `"by_arm"` additionally computes a k-hat restricted to each arm's
#'     hyperparameter axes (the other arms held at their posterior mean), reported
#'     in `pareto_k_by_arm`, to localise which arm drives a tail-heavy joint k; the
#'     joint k itself is unchanged. Per-arm k is defined for the multi-block layout
#'     with two or more arms and declines for the single-block shared-field layout.
#'     The legacy `k_samples` name is accepted as an alias for `diagnose_draws`.
#'   * `k_threads` (`NULL`) -- outer-thread width for the diagnostic's importance
#'     batch. The `diagnose_draws` re-solves are independent and run after the grid
#'     (every core free), each solved single-threaded once the batch saturates the
#'     pool, so widening it is a bit-identical wall-clock speedup (the k-hat is
#'     unchanged). `NULL` follows the fit's own thread grant -- the larger of
#'     `n_threads_outer` and the inner `n_threads` -- so a serial fit keeps a serial
#'     diagnostic while a threaded fit gets a free parallel one. `"auto"` uses the
#'     physical performance-core count (capped at 2 under R CMD check); an integer
#'     pins the width (`1L` forces serial). Always capped at `diagnose_draws`.
#'   * `k_bootstrap` (`1000L`) -- bootstrap replicates for the outer
#'     Pareto-\eqn{\hat{k}} uncertainty. The k-hat is a single fixed number for a
#'     fit + proposal; its sampling uncertainty GIVEN the proposal is estimated by
#'     resampling the diagnostic's raw importance log-ratios with replacement and
#'     re-fitting the GPD tail `k_bootstrap` times (no new inner solves). Reports
#'     `pareto_k_se_boot` (bootstrap SE), `pareto_k_ci_low` / `pareto_k_ci_high`
#'     (the 2.5\% / 97.5\% bootstrap quantiles), `pareto_k_se_formula` (the
#'     closed-form GPD-shape MLE asymptotic SE \eqn{(1 + k)/\sqrt{M}}, a
#'     cross-check), and `pareto_k_band_confident` (TRUE iff the bootstrap CI lies
#'     within one reliability band). The bootstrap measures how UNSTABLE the
#'     current tail estimate is; it cannot create tail information. Increase
#'     `diagnose_draws`, not `k_bootstrap`, to obtain more tail information. `0L`
#'     skips it (point k-hat only). The per-arm k carries the same fields.
#'   * `k_tail_points` (`NULL`) -- number of upper-tail order statistics for the
#'     GPD fit. `NULL` uses the automatic PSIS rule
#'     \eqn{\lceil\min(0.2 N, 3\sqrt{N})\rceil}. An explicit value is an EXPERT
#'     tail-threshold control, capped at the 20\%-of-draws ceiling (with a warning)
#'     so the fit stays an extreme tail; it is NOT a precision knob (a request that
#'     drags body ratios into the tail lowers variance but biases the k-hat). The
#'     used and requested counts are reported in `pareto_k_tail_points` /
#'     `pareto_k_tail_points_requested`.
#'   * `k_conf_bands` (`NULL`) -- the reliability-band boundaries the bootstrap CI
#'     is tested against for `pareto_k_band_confident`. `NULL` (default) uses the
#'     sample-size-dependent boundaries \eqn{c(0.5, \min(1 - 1/\log_{10} S, 0.7))}
#'     at the realised draw count \eqn{S} (Vehtari et al. 2024): the good cut is
#'     0.5 and the usable cut tightens below 0.7 for small \eqn{S} (about 0.565 at
#'     \eqn{S = 200}, reaching 0.7 only past \eqn{S \approx 2154}). Supply a
#'     strictly-increasing numeric vector to fix the boundaries instead, e.g.
#'     `c(0.5, 0.7)` for the size-independent good / ok / unreliable split.
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
#'   * `theta_grid`, `theta_names` -- outer-grid hyperparameter values
#'     (includes the `alpha` axis when `copy` is set).
#'   * `log_marginal`, `weights` -- per-grid-point log-marginal and integration
#'      weights (sum to 1).
#'   * `theta_mean`, `theta_sd` -- posterior moments per hyperparameter,
#'      including `alpha` when `copy` is set.
#'   * `theta_median`, `theta_ci_lo`, `theta_ci_hi` -- weighted-quantile
#'      median and 2.5/97.5 empirical CI per hyperparameter axis (same
#'      names as `theta_mean`). Recommended summary for right-skewed
#'      scale-like axes (alpha at small n_pos, sigma/range/phi near
#'      a boundary), where the posterior mean is pulled by the right
#'      tail away from the bulk and `mean +/- 1.96 sd` mis-states the
#'      uncertainty.
#'   * `modes` -- `[n_grid x n_x]` matrix of inner modes.
#'   * `n_iter` -- inner Newton iterations per grid point.
#'   * `arm_layout` -- list with per-arm `beta_start`, `re_start`,
#'      spatial offset(s) and `n_x` for decoding modes.
#'   * `prior`, `responses`, `copy` -- echoed inputs.
#'   * `timing` -- named numeric of wall-clock seconds: `total` plus the
#'      `setup` (validation / encoding / grid construction), `grid` (inner
#'      Laplace solves, including adaptive-refinement and consistency passes),
#'      `postproc` (weight / moment / marginal assembly) and `diagnostics`
#'      (outer Pareto-\eqn{\hat{k}}) phases. The `grid` phase is the one that
#'      scales with grid size and core count. Surfaced one-line in `print`.
#'   * `pareto_k`, `pareto_k_is_ess`, `pareto_k_scope` -- outer
#'      Pareto-\eqn{\hat{k}} accuracy diagnostic and its importance-sampling
#'      ESS (both `NA` when `control$diagnose_k = FALSE` or the fit declines;
#'      see the `diagnose_k` control knob). `pareto_k < 0.7` indicates the
#'      nested integration is reliable; `>= 0.7` that the (skewed / heavy-
#'      tailed) hyperparameter posterior is misfit by the Gaussian grid and
#'      the fit should escalate to an exact debias.
#'   * `pareto_k_proposal_source` -- how the outer importance proposal the
#'      \eqn{\hat{k}} scores was built: `"mode_hessian"` from the Laplace
#'      curvature at the hyperparameter mode (the CCD design's, or a
#'      finite-difference Hessian when a sharp posterior collapses the grid),
#'      or `"grid_moment"` from the grid-weighted covariance. `NA` when the
#'      diagnostic is off or declines. The mode-Hessian source keeps the
#'      \eqn{\hat{k}} meaningful when the grid concentrates on ~1 cell.
#'   * `pareto_k_se_boot`, `pareto_k_ci_low`, `pareto_k_ci_high`,
#'      `pareto_k_se_formula`, `pareto_k_tail_points`,
#'      `pareto_k_tail_points_requested`, `pareto_k_band_confident` -- the outer
#'      Pareto-\eqn{\hat{k}} uncertainty (gcol33/tulpa#127), present whenever the
#'      diagnostic ran. `pareto_k_se_boot` is the bootstrap SE of the k-hat and
#'      `pareto_k_ci_low` / `pareto_k_ci_high` its 2.5\% / 97.5\% bootstrap
#'      quantiles -- the estimator's sampling spread GIVEN the proposal, not a
#'      posterior credible interval; `pareto_k_se_formula` is the closed-form
#'      GPD-shape MLE asymptotic SE cross-check. `pareto_k_tail_points` is the GPD
#'      tail size used and `pareto_k_tail_points_requested` the `k_tail_points`
#'      request (`NA` when automatic). `pareto_k_band_confident` is TRUE iff the
#'      bootstrap CI lies within one reliability band, `NA` when the bootstrap was
#'      off or could not fit. `diagnose_draws` and `diagnose_cost_ratio` (the
#'      diagnostic's draw budget and its wall-clock cost relative to the fit) are
#'      attached at the top level.
#'   * `pareto_k_by_arm`, `pareto_k_by_arm_is_ess`, `pareto_k_by_arm_scope` --
#'      present only with `control$diagnose_k = "by_arm"` (gcol33/tulpa#120).
#'      Named (by arm) outer Pareto-\eqn{\hat{k}} restricted to each arm's
#'      hyperparameter axes, the other arms held at their posterior mean, so a
#'      tail-heavy joint k can be localised to one arm. A per-arm entry is `NA`
#'      when that arm carries no varying axis. Each arm carries its own bootstrap
#'      uncertainty: `pareto_k_by_arm_se_boot`, `pareto_k_by_arm_ci_low` /
#'      `pareto_k_by_arm_ci_high`, `pareto_k_by_arm_se_formula`,
#'      `pareto_k_by_arm_tail_points` and `pareto_k_by_arm_band_confident`.
#'   * `k_quality_requested`, `k_quality_reached`, `k_quality_best`,
#'      `k_quality_reason`, `k_quality_rounds` -- the reliability verdict for the
#'      `control$k_quality` intent (gcol33/tulpa#129, #131). `k_quality_requested`
#'      echoes the intent; `k_quality_best` is the band actually achieved
#'      (`"good"` / `"ok"` / `"unreliable"`, or `"uncertain"` when the bootstrap CI
#'      crosses a boundary); `k_quality_reached` is `TRUE`/`FALSE` for an `"ok"` /
#'      `"good"` target (`NA` for `"report"` / `"none"`), never silently
#'      downgraded; `k_quality_reason` records why it stopped; and
#'      `k_quality_rounds` is the number of escalation re-fits performed (`0` when
#'      the first fit sufficed or escalation was off).
#'   * `adaptive_grid_info` -- when `adaptive_grid = TRUE`, a list with
#'      `triggered_axes` (character) and `n_points_added` (integer)
#'      describing the refinement passes. NULL otherwise.
#'   * `local_ccd_info` -- when `local_ccd` engaged (or `k_refine = "ccd"`), a
#'      list with `n_cells_refined`, `n_nodes_added`, the refined `cells`, and the
#'      `n_cells_before` / `n_cells_after` grid sizes. NULL when local CCD was off
#'      or declined (single-block, `< 4` axes, an active `phi_grid`, or no peaked
#'      interior cell).
#'   * `prune_cheap_log_marginal`, `prune_mask`, `prune_n_pruned`,
#'      `prune_tol` -- present only when `prune = TRUE` and the safety gate did
#'      not fall back. Cheap-pass log-marginals at every cell, a logical mask
#'      of pruned cells, the pruned-cell count, and the threshold actually
#'      applied. Pruned cells have `log_marginal = -Inf` so they get zero
#'      weight under `.nl_normalise_weights_safe`.
#'   * `prune_fallback_triggered`, `prune_fallback_reason` -- present only when
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
                                       prior_phi = NULL,
                                       cell_coupling = "separable",
                                       control = list()) {
    # k_quality reliability front door + escalation (gcol33/tulpa#129, #131). The
    # single fit lives in .tulpa_nl_joint_once(); this wrapper resolves the
    # reliability intent, attaches the honest verdict (for BOTH the single- and
    # multi-block paths), and -- for "ok" / "good" -- chases the band by REFINING
    # THE INTEGRATION GRID, driven by the bad outer k (with k_refine = "grid"):
    # each round widens / densifies the grid where the posterior mass escapes its
    # bounds, re-fits + re-diagnoses, until the requested band is confidently
    # reached or the round budget is hit. diagnose_draws is the separate
    # estimate-precision knob, not an escalation lever. Never silently downgrades.
    k_quality <- control$k_quality %||% "report"
    if (!is.character(k_quality) || length(k_quality) != 1L ||
        !k_quality %in% c("none", "report", "ok", "good")) {
        stop("`control$k_quality` must be one of \"none\", \"report\", \"ok\", \"good\".",
             call. = FALSE)
    }
    k_refine <- control$k_refine %||% "grid"
    if (!is.character(k_refine) || length(k_refine) != 1L ||
        !k_refine %in% c("none", "grid", "ccd")) {
        stop("`control$k_refine` must be \"none\", \"grid\", or \"ccd\".", call. = FALSE)
    }
    k_max_rounds <- control$k_max_rounds %||% 2L
    if (length(k_max_rounds) != 1L || is.na(k_max_rounds) ||
        k_max_rounds != round(k_max_rounds) || k_max_rounds < 0L) {
        stop("`control$k_max_rounds` must be a single integer >= 0.", call. = FALSE)
    }
    k_max_rounds <- as.integer(k_max_rounds)
    k_conf_bands <- control$k_conf_bands
    diag_raw     <- control$diagnose_k %||% TRUE
    diagnose_k   <- (identical(diag_raw, "by_arm") || isTRUE(diag_raw)) &&
                    !identical(k_quality, "none")

    attach_q <- function(res)
        .joint_attach_k_quality(res, k_quality, diagnose_k,
                                res$diagnose_draws %||% NA_integer_, k_conf_bands)

    ctrl <- control
    res  <- attach_q(.tulpa_nl_joint_once(responses, prior, copy, phi_grid,
                                          prior_sigma, prior_alpha, prior_phi,
                                          cell_coupling, ctrl))
    res$k_quality_rounds <- 0L

    if (k_quality %in% c("ok", "good") && isTRUE(diagnose_k) &&
        k_max_rounds > 0L && k_refine %in% c("grid", "ccd")) {
        # A bad outer k means the integration grid does not faithfully represent
        # the hyperparameter posterior. Two refinement rungs, both driven by the
        # bad k and re-diagnosed each round until the requested band is confidently
        # reached or the round budget is spent (gcol33/tulpa#130, #131, #64):
        #   * "grid" -- the grid's bounds let posterior mass escape: each round
        #     runs the adaptive boundary-extension / interior-densification pass
        #     (R/hyper_grid_refine.R) where the integrand mass piles at an edge,
        #     one more pass per round (recursive in the bad-k direction).
        #   * "ccd"  -- the grid is too coarse to resolve a sharply-peaked
        #     direction: each round refines more high-weight, mutually
        #     non-adjacent cells with local curvature-aware CCD node clouds
        #     (R/nested_laplace_joint_ccd_local.R). A tensor base is forced (the
        #     finite-difference curvature stencil needs axis neighbours).
        # Raising diagnose_draws would only sharpen the SAME k against the SAME
        # grid, never refine it, so it is NOT the escalation lever -- it stays the
        # separate estimate-precision knob (control$diagnose_draws).
        is_ccd_refine <- identical(k_refine, "ccd")
        refined_field <- if (is_ccd_refine) "local_ccd_info" else "adaptive_grid_info"
        if (is_ccd_refine) {
            ctrl$integration <- "grid"
            lc_base      <- if (is.list(control$local_ccd)) control$local_ccd else list()
            lc_max_cells <- as.integer(lc_base$max_cells %||% 6L)
        } else {
            ctrl$adaptive_grid <- TRUE
        }
        round <- 0L
        while (!isTRUE(res$k_quality_reached) && round < k_max_rounds) {
            round <- round + 1L
            if (is_ccd_refine) {
                ctrl$local_ccd <- utils::modifyList(
                    lc_base, list(max_cells = round * lc_max_cells))
            } else {
                ctrl$adaptive_grid_max_passes <- round
            }
            res <- attach_q(.tulpa_nl_joint_once(responses, prior, copy, phi_grid,
                                                 prior_sigma, prior_alpha, prior_phi,
                                                 cell_coupling, ctrl))
            res$k_quality_rounds <- round
            if (!isTRUE(res$k_quality_reached) && is.null(res[[refined_field]])) {
                # The refiner found nothing to act on, so further rounds cannot
                # move the k: the deficiency the chosen rung addresses is absent.
                res$k_quality_reason <- if (is_ccd_refine) paste0(
                    "local CCD found no peaked interior cell to refine (needs a ",
                    ">= 4-axis multi-block tensor grid); the bad k is not a ",
                    "coarse-grid resolution deficiency") else paste0(
                    "grid refinement found no boundary mass to extend; ",
                    "the bad k is not a grid-width deficiency")
                break
            }
        }
        if (!isTRUE(res$k_quality_reached) && round >= k_max_rounds)
            res$k_quality_reason <- sprintf(
                "requested band not confirmed within the k_max_rounds %s budget",
                if (is_ccd_refine) "local-CCD-refinement" else "grid-refinement")
    }
    res
}

.tulpa_nl_joint_once <- function(responses, prior, copy = NULL, phi_grid = NULL,
                                 prior_sigma = NULL, prior_alpha = NULL,
                                 prior_phi = NULL,
                                 cell_coupling = "separable", control = list()) {
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
    # Local CCD refinement of the multi-block outer grid (gcol33/tulpa#64). A
    # coarse tensor base grid is refined by a small curvature-aware node cloud on
    # a few high-weight, mutually non-adjacent cells, so the base grid can stay
    # coarse at moderate-to-high latent dimension where a uniformly fine tensor is
    # k^d-expensive. `control$local_ccd` is TRUE (defaults) or a list(max_cells=,
    # f0=); NULL (the default) keeps the unrefined grid. Engages only on the
    # multi-block tensor path at >= 4 transformable latent axes with no active phi
    # grid; the single-block path always integrates on the tensor grid unrefined.
    local_ccd                 <- control$local_ccd
    if (!is.null(local_ccd) && !isTRUE(local_ccd) && !is.list(local_ccd)) {
        stop("`control$local_ccd` must be NULL, TRUE, or a list(max_cells=, f0=).",
             call. = FALSE)
    }
    # Adaptive-lattice integrator tuning (integration = "grid_adaptive", the
    # low-dimensional multi-block companion to the CCD; see
    # nested_laplace_joint_adaptive.R). `adaptive_grid_cutoff` is the log-density
    # keep/expand radius from the peak (larger = closer to the dense tensor, fewer
    # cells skipped); `adaptive_grid_stride` the coarse-seed subsample stride;
    # `adaptive_grid_max_frac` the fraction of the dense grid past which the
    # builder declines back to the tensor.
    adaptive_cutoff           <- control$adaptive_grid_cutoff   %||% 10
    adaptive_stride           <- as.integer(control$adaptive_grid_stride %||% 2L)
    adaptive_max_frac         <- control$adaptive_grid_max_frac %||% 0.75
    adaptive_min_cells        <- control$adaptive_grid_min_cells %||% 48
    # Outer Pareto-k-hat accuracy diagnostic. `diagnose_k` (default TRUE)
    # importance-samples the joint hyperparameter posterior against the proposal
    # the integrator fits; `diagnose_draws` (default 500) is the number of draws,
    # each one extra inner joint solve. RNG-restored, so the fit's draws are
    # unchanged whether or not it runs. `diagnose_k = "by_arm"`
    # additionally computes a k-hat restricted to each arm's hyperparameter axes
    # (other arms held at their posterior mean), to localise which arm drives a
    # tail-heavy joint k (gcol33/tulpa#120); the joint k is unchanged and stays
    # the default.
    # k_quality (gcol33/tulpa#129) within ONE fit: "none" disables the diagnostic,
    # and "ok" / "good" raise the default draw budget so the bootstrap CI can
    # resolve the requested band. The verdict quartet, the band classification, and
    # the multi-round escalation (gcol33/tulpa#131) live in the front-door wrapper
    # tulpa_nested_laplace_joint(), which drives this single-fit engine.
    k_quality                 <- control$k_quality %||% "report"
    if (!is.character(k_quality) || length(k_quality) != 1L ||
        !k_quality %in% c("none", "report", "ok", "good")) {
        stop("`control$k_quality` must be one of \"none\", \"report\", \"ok\", \"good\".",
             call. = FALSE)
    }
    diagnose_k_raw            <- control$diagnose_k %||% TRUE
    pareto_k_by_arm           <- identical(diagnose_k_raw, "by_arm")
    diagnose_k                <- (pareto_k_by_arm || isTRUE(diagnose_k_raw)) &&
                                 !identical(k_quality, "none")
    # Diagnostic importance-draw budget (gcol33/tulpa#127). `diagnose_draws` is the
    # single precision knob: the outer Pareto-k is scored ONCE over this many
    # importance draws, and a tighter k needs MORE actual tail ratios, i.e. a larger
    # `diagnose_draws`. The legacy `k_samples` name is accepted as an alias.
    diagnose_draws_user       <- control$diagnose_draws %||% control$k_samples
    diagnose_draws            <- diagnose_draws_user %||% 500L
    if (length(diagnose_draws) != 1L || is.na(diagnose_draws) ||
        diagnose_draws != round(diagnose_draws) || diagnose_draws < 1L) {
        stop("`control$diagnose_draws` must be a single integer >= 1.", call. = FALSE)
    }
    diagnose_draws            <- as.integer(diagnose_draws)
    # k_quality "ok" / "good" raise the default draw budget so the bootstrap CI can
    # resolve the requested band (a tighter CI needs more actual tail ratios); an
    # explicit `diagnose_draws` / `k_samples` is always respected (gcol33/tulpa#129).
    if (is.null(diagnose_draws_user) && k_quality %in% c("ok", "good"))
        diagnose_draws <- if (identical(k_quality, "good")) 2000L else 800L
    # Outer-thread width for the diagnostic's importance batch (gcol33/tulpa#117).
    # The `diagnose_draws` re-solves are independent and run after the grid (all
    # cores free), each solved single-threaded, so widening this pool is a
    # bit-identical wall-clock speedup. `NULL` (default) follows the fit's thread
    # grant; "auto" uses the performance cores; an integer pins the width.
    k_threads                 <- control$k_threads
    pareto_k_threads          <- .tulpa_pareto_k_threads(n_threads_outer,
                                                         n_threads, diagnose_draws,
                                                         k_threads)
    # Bootstrap outer Pareto-k uncertainty (gcol33/tulpa#127). The chosen proposal
    # is scored once; the k-hat's sampling uncertainty is then estimated by
    # bootstrapping its raw importance log-ratios -- `k_bootstrap` replicates, each
    # re-fitting the GPD tail, NO new inner solves. The bootstrap reports how
    # UNSTABLE the current estimate is, not a tighter k (raise `diagnose_draws` for
    # that, NOT `k_bootstrap`). `k_tail_points` (NULL = the automatic PSIS rule) is
    # an expert tail-threshold control, capped at the 20%-of-draws ceiling.
    # `k_conf_bands` are the reliability-band boundaries the bootstrap CI is tested
    # against for `pareto_k_band_confident`; NULL (default) uses the
    # sample-size-dependent boundaries c(0.5, min(1 - 1/log10(S), 0.7)) at the
    # realised draw count S (gcol33/tulpa#128).
    k_bootstrap               <- control$k_bootstrap %||% 1000L
    if (length(k_bootstrap) != 1L || is.na(k_bootstrap) ||
        k_bootstrap != round(k_bootstrap) || k_bootstrap < 0L) {
        stop("`control$k_bootstrap` must be a single integer >= 0.", call. = FALSE)
    }
    k_bootstrap               <- as.integer(k_bootstrap)
    k_tail_points             <- control$k_tail_points
    if (!is.null(k_tail_points)) {
        k_tail_points <- suppressWarnings(as.integer(k_tail_points))
        if (length(k_tail_points) != 1L || is.na(k_tail_points) || k_tail_points < 1L) {
            stop("`control$k_tail_points` must be a single positive integer, or NULL.",
                 call. = FALSE)
        }
    }
    k_conf_bands              <- control$k_conf_bands
    if (!is.null(k_conf_bands) &&
        (!is.numeric(k_conf_bands) || anyNA(k_conf_bands) || length(k_conf_bands) < 1L ||
         is.unsorted(k_conf_bands, strictly = TRUE))) {
        stop("`control$k_conf_bands` must be NULL (the sample-size-dependent ",
             "default) or a strictly increasing numeric vector.", call. = FALSE)
    }
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
                                            c("auto", "ccd", "grid",
                                              "grid_adaptive"))
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
    fn_phi   <- .joint_parse_sigma_prior(prior_phi,   "prior_phi")

    # Detect multi-block prior (list-of-blocks). Routed to a separate
    # dispatch path that builds vector<LatentBlock> on the joint side.
    # Pruning toggle: the kernel takes prune_tol > 0 to mean "screen and
    # prune"; gate it here with the user-facing `prune` boolean so the toggle
    # composes cleanly (prune=FALSE always disables, prune=TRUE uses
    # prune_tol). prune_tol must be in [0, 1) -- clamp out-of-range to 0
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
            fn_sigma = fn_sigma, fn_alpha = fn_alpha, fn_phi = fn_phi,
            max_iter = max_iter, tol = tol, n_threads = n_threads,
            n_threads_outer = n_threads_outer,
            tile_warm = tile_warm,
            prune_tol = prune_tol_eff,
            x_init = x_init, verbose = verbose, store_Q = store_Q,
            force_sparse = force_sparse,
            cell_coupling = cell_coupling,
            diagnose_k = diagnose_k, diagnose_draws = diagnose_draws,
            pareto_k_by_arm = pareto_k_by_arm,
            pareto_k_threads = pareto_k_threads,
            k_bootstrap = k_bootstrap,
            k_tail_points = k_tail_points,
            k_conf_bands = k_conf_bands,
            inner_refresh = inner_refresh,
            integration = integration,
            local_ccd = local_ccd,
            adaptive_cutoff = adaptive_cutoff,
            adaptive_stride = adaptive_stride,
            adaptive_max_frac = adaptive_max_frac,
            adaptive_min_cells = adaptive_min_cells,
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
    hp_fn <- if (is.null(fn_sigma) && is.null(fn_alpha) && is.null(fn_phi)) NULL else {
        function(new_cells) {
            .joint_hp_vec_for_grids(new_cells, fn_sigma, fn_alpha, fn_phi)
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
                                         diagnose_draws = diagnose_draws,
                                         n_threads_outer = pareto_k_threads,
                                         pareto_k_by_arm = pareto_k_by_arm,
                                         k_bootstrap = k_bootstrap,
                                         k_tail_points = k_tail_points,
                                         k_conf_bands = k_conf_bands)
    tm$mark("diagnostics")
    res$timing <- tm$timing()
    res <- .joint_attach_diagnose_cost(res, diagnose_k, diagnose_draws)
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
