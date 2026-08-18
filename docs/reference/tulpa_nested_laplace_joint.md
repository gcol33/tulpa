# Joint multi-likelihood nested Laplace approximation

Outer-grid nested Laplace driver for *joint* models – multiple response
arms sharing one latent prior block, parameterized as a per-arm field
amplitude (sigma) on a unit-precision latent.

Supported priors:

- `"bym2"` – outer grid over `(sigma, rho [, alpha])`. Latent:
  `phi (n_s) | theta (n_s)` with unit-precision ICAR + iid components.

- `"icar"` – outer grid over `(sigma [, alpha])`. Latent: `phi (n_s)`
  with unit-precision ICAR.

- `"car_proper"` – outer grid over `(sigma, rho_car [, alpha])`. Latent:
  `phi (n_s)` with `Q = D - rho_car * W`.

Other backends (NNGP, HSGP, RW1/2, AR1) follow the same interface and
land under Phase 3.

## Usage

``` r
tulpa_nested_laplace_joint(
  responses,
  prior,
  copy = NULL,
  phi_grid = NULL,
  prior_sigma = NULL,
  prior_alpha = NULL,
  prior_phi = NULL,
  cell_coupling = "separable",
  control = list()
)
```

## Arguments

- responses:

  A named list of arm specs (length \>= 1). Each arm:

  - `y` – numeric `[N_arm]` response.

  - `n_trials` – integer `[N_arm]` (use `rep(1L, N_arm)` for
    non-binomial).

  - `X` – numeric matrix `[N_arm x p_arm]` fixed-effects design.

  - `spatial_idx` – integer `[N_arm]`, 1-based map obs -\> spatial unit.

  - `re_idx` – optional numeric `[N_arm]` 1-based RE group index;
    defaults to `rep(0, N_arm)` (no RE).

  - `n_re_groups` – optional integer (default `0L`).

  - `sigma_re` – optional numeric (default `1`); ignored when
    `n_re_groups == 0`.

  - `family` – one of `"binomial"`, `"gaussian"`, `"poisson"`,
    `"neg_binomial_2"`, `"beta"`, `"lognormal"`, `"gamma"`,
    `"inverse_gaussian"`. For `"lognormal"`, `y` is on the natural scale
    and the linear predictor `eta = E[log y]` (identity link on the log
    scale); the `-log(y)` Jacobian is included in the kernel's
    `log_lik`.

  - `phi` – numeric dispersion (gaussian/lognormal residual SD, negbin
    size, beta precision); default `1`.

  - `field_coef` – optional per-arm field coefficient controlling this
    arm's multiplier on the shared latent field's amplitude. One of: \*
    numeric scalar (default `1`) – constant multiplier. `0` means the
    arm carries NO field at all (the per-row scatter
    `eta += field_coef * sigma * z` is skipped for that arm). \*
    character of length 1 – names an outer-grid hyperparam axis
    (currently `"alpha"`); the coefficient varies across the grid. \*
    `list(name = , grid = )` – embedded axis declaration, equivalent to
    declaring the axis and naming it on this arm. At most one arm may
    declare a hyperparam-driven axis (the cover hurdle and occu_cover
    both need only one). Shared axes across multiple arms are deferred.
    A single-block copy coefficient is declared here, on the arm – not
    through a separate `copy` argument.

- prior:

  A list describing the shared latent prior block. Required field
  `type`. Backend-specific fields:

  - **bym2**: `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`,
    `n_neighbors`, `scale_factor` (default `1`); optional `sigma_grid`
    (donor-arm field amplitude, default 5 log-spaced values in
    `[0.1, 3]`), `rho_grid` (default
    `c(0.2, 0.5, 0.8, 0.95, 0.99, 0.999)`).

  - **icar**: `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`,
    `n_neighbors`; optional `sigma_grid` (default 5 log-spaced values in
    `[0.1, 3]`).

  - **car_proper**: same as icar plus `rho_car_grid` (default
    `c(0.5, 0.8, 0.95, 0.99)`).

  `sigma_grid`'s default is a starting axis, not a hard ceiling
  (gcol33/tulpa#289): when the fitted field-SD posterior mode rails the
  top node (`pareto_k_regime = "collapsed_edge"`, see below), the driver
  re-centres the axis on a mode-Hessian and refits (up to two attempts,
  the second adding a light default PC(U=3, alpha=0.01) prior on sigma
  unless `prior_sigma` was pinned – see there), so a sparse or
  strongly-identified species is not silently truncated at 3.0. This
  engages whether or not `control$diagnose_k` computed the full outer
  Pareto-k diagnostic (gcol33/tulpa#292): the mode-Hessian is reused
  from the diagnostic when it ran, or computed on its own (one extra
  batched finite-difference solve, only when the grid actually
  collapsed) when it did not – so `diagnose_k = FALSE`, the default,
  does not leave a railed axis stuck. A `sigma_grid` the caller PINNED
  always wins: auto-recenter engages when the field is left `NULL`, when
  it is marked with
  [`auto_grid()`](https://gillescolling.com/tulpa/reference/auto_grid.md)
  (how a wrapper package declares an axis it defaulted rather than one
  the user chose, gcol33/tulpa#293), or when its nodes are exactly the
  engine's own default axis. Declines gracefully (keeps the fixed-grid
  fit) when another axis in the same grid has unguessable support
  (car_proper's `rho_car`); whichever way it declines,
  `outer_grid_recenter_declined` says which (see below).

- copy:

  Multi-block copy specification (multi-block `prior` only). For a
  single-block fit there is no `copy` argument: declare the copy
  coefficient on the arm via
  `responses[[X]]$field_coef = list(name = "alpha", grid = G)`. On the
  multi-block path `copy` is an unnamed list of specs –
  `list(list(arm, block, alpha_grid),...)` – coupling N distinct shared
  latent fields, each onto its own arm with its own \\\alpha\\ axis,
  integrated over the product outer grid. Each spec must name a distinct
  block. The copy block may be any of `icar` / `bym2` / `car_proper` /
  `rw1` / `rw2` / `ar1` / `iid`; blocks with their own per-arm scaling
  (`lf`, `hsgp_mo`) or a precomputed precision (`tgmrf`) do not take a
  copy. A copy block's own `sigma_grid` (the donor field amplitude, same
  default ceiling as the single-block `prior$sigma_grid` above)
  auto-recenters on `collapsed_edge` the same way, one block per
  attempt.

- phi_grid:

  Optional list specifying per-arm dispersion axes on the outer grid.
  Accepts either a named list (keys = arm names) or a positional list of
  length `n_arms`. Each entry is one of:

  - `NULL` or scalar – no axis for that arm; the kernel uses the
    parse-time scalar `responses[[k]]$phi`.

  - numeric vector of length \> 1 – adds a new outer-grid axis
    `phi_<arm>` taking those values; the kernel rewrites `arms[k].phi`
    at each grid point before the inner Newton solve.

  Family-specific interpretation of `arm$phi` (the parse-time scalar and
  the grid values):

  - `gaussian` – residual SD (variance is `phi^2`). Use `phi_grid` to
    estimate the residual SD as a hyperparameter instead of pinning it
    pre-fit.

  - `lognormal` – residual SD on the log scale; identical kernel
    parameterization as `gaussian` plus the `-log(y)` Jacobian.

  - `neg_binomial_2` – dispersion (variance is `mu + mu^2/phi`).

  - `beta` – precision (variance is `mu(1-mu)/(1+phi)`).

  - `gamma`, `inverse_gaussian` – shape / dispersion.

  - `binomial`, `poisson` – ignored.

  Each `phi_<arm>` axis is appended to the Cartesian product and varies
  slowest (within-spatial warm starts hold). The axis appears as a
  regular hyperparameter in `theta_grid`, `theta_mean`, and `theta_sd`,
  and participates in adaptive-grid refinement.

- prior_sigma, prior_alpha:

  Optional regularizing hyperpriors on the donor field amplitude
  \\\sigma\\ and on the copy coefficient \\\alpha\\. Each is `NULL`
  (flat, default) or a list of the form `list(family, params)`:

  - `list("pc.prec", c(U, alpha))` – Penalized Complexity prior,
    calibrated by `P(theta > U) = alpha`. Closed-form density
    `lambda * exp(-lambda * theta)` with `lambda = -log(alpha)/U`.
    Drop-in for the weakly-identified small-`n_pos` regime . Pick `U` at
    the upper end of plausible values so the prior shrinks the tail
    without biasing the modal cell when the data identifies it:
    default-friendly choice on \\\sigma\\ is `c(U = 1.0, alpha = 0.01)`
    (donor amplitude); on the dimensionless copy coefficient \\\alpha\\
    the recommended choice is `c(U = 8.0, alpha = 0.01)`. Too small a
    `U` over-shrinks the copy coefficient past the modal cell and,
    through the `alpha * sigma` copy axis, inflates the coupled donor
    amplitude `sigma` – e.g. on a fixture with truth \\\alpha = 1\\,
    `c(U = 2.0, alpha = 0.01)` pulls the \\\alpha\\ posterior below 1
    and lifts `sigma` above its truth.

  - `list("half_normal", scale)` – half-normal with scale `scale > 0`.
    Sharper tail decay than PC; use when stronger regularization is
    desired and the truth is well inside the prior. The contribution is
    added to `log_marginal` cell-by-cell at the kernel-call boundary, so
    refinement passes (adaptive grid, var-of-means consistency) see the
    regularized posterior. When the data identifies the parameter (e.g.
    `n_pos >= ~200`) the prior is essentially harmless – the lever is
    tail-shrinkage at small `n_pos`. `prior_alpha` only applies when
    `copy` is active; `prior_sigma` applies on any `sigma`-named axis.

  `prior_sigma` also interacts with the auto-recenter above: its second
  attempt engages the engine's own weakly-informative PC(U = 3, alpha =
  0.01) prior, and a `prior_sigma` the caller PINNED suppresses that
  (the caller's prior stands). Pinning is decided by provenance, not
  presence (gcol33/tulpa#297): a spec marked with
  [`auto_grid()`](https://gillescolling.com/tulpa/reference/auto_grid.md),
  or one equal by value to the engine's own default, is a default and
  does not suppress the escalation. When it does, the fit carries
  `outer_grid_prior_declined = "prior_pinned"`, so a second attempt that
  changed only the grid geometry is legible rather than looking like the
  full escalation.

- prior_phi:

  Optional regularizing hyperprior on the per-arm dispersion axes
  declared through `phi_grid` (e.g. a Beta precision on a cover arm, a
  negbin dispersion, a Gaussian residual SD). Same families as
  `prior_sigma` – `NULL` (flat over the phi grid, default),
  `list("pc.prec", c(U, alpha))`, or `list("half_normal", scale)`. A
  single spec re-weights every `phi_<arm>` axis on the grid, the way
  `prior_sigma` re-weights any sigma-named axis; with no `phi_grid` it
  is a no-op. Without it the phi grid carries an implicit flat prior
  over its bounds. The PC scale is the dispersion's own units (a
  precision for `beta`, a size for `neg_binomial_2`), so pick `U` at the
  upper end of plausible values.

- cell_coupling:

  Character scalar naming a per-cell coupled likelihood registered
  against tulpa's process-global registry (see
  `src/cell_coupling_registry.h`). Defaults to `"separable"`, the arm-
  separable per-obs sum every existing joint fit uses. Consumer packages
  (e.g. tulpaObs) compile a `tulpa::CellCouplingSpec` subclass in their
  own `src/` and register it from `R_init_<pkg>` via the
  `tulpa_register_cell_coupling` C callable; the R driver validates the
  name against the registry and the inner Newton routes the per-cell
  contribution through `evaluate_cell()` when the spec couples at least
  one arm.

- control:

  Optional list of perf/numerical tuning knobs (statistical arguments
  stay top-level), following the `control` convention of
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md).
  Recognised elements (defaults in parentheses):

  - `max_iter` (`50L`), `tol` (`1e-6`) – inner Newton iteration budget
    and tolerance.

  - `n_threads` (`1L`) – inner-loop OpenMP threads (per-observation
    scatter, compute_eta, log-likelihood reduction). For typical joint
    workloads (`N` in the hundreds to a few thousand) inner parallelism
    is overhead-dominated; prefer `n_threads_outer`, which stacks better
    on many-core hardware. Capped at the physical performance-core count
    by default (see `n_threads_scatter`), as the inner per-observation
    loops oversubscribe a hybrid CPU's efficiency cores past that point.
    A fit is reproducible bit for bit at a GIVEN `n_threads`: the
    per-observation sum cuts its range into that many contiguous chunks
    and adds the chunk sums in chunk order, so nothing about the answer
    is left to the OpenMP runtime. Chunking imposes its own association,
    so two different `n_threads` agree only to floating-point tolerance
    (measured at 6e-14 on `log_marginal` over four families).

  - `n_threads_scatter` (performance-core count) – cap on the inner
    per-observation threads. Overrides the default performance-core cap
    on `n_threads`; raise it to use all logical cores or lower it to
    leave headroom. No effect where the core topology cannot be resolved
    (off Windows), where `n_threads` is used as requested.

  - `n_threads_outer` (`1L`) – outer-grid OpenMP threads. When `> 1`, a
    pilot Laplace at the centre cell warm-starts the remaining cells,
    each dispatched across `n_threads_outer` threads with its own
    CHOLMOD solver and NewtonScratch (inner OpenMP auto-disabled). `1L`
    is serial, chained warm-starts – bitwise identical to the
    pre-speedup driver. Recommended on multi-core workstations:
    `parallel::detectCores() - 1L`.

  - `tile_warm` (`TRUE`) – when `n_threads_outer > 1` and a copy block
    is present, group outer cells into tiles sharing every axis except
    the copy coefficient `alpha`, solve one warm Tier-2 per tile from
    the centre pilot, and warm-start the rest from their tile pilot.
    Falls back to the single-tier path when no copy block / single tile
    / `n_threads_outer <= 1L`. `FALSE` recovers the pre-tiling behaviour
    (e.g. regression testing).

  - `prune` (`FALSE`), `prune_tol` (`1e-3`) – opt-in cheap-pass
    screening. When `prune = TRUE`, the driver sweeps the outer lattice
    running a short inner Newton per cell, each warm-started from the
    previous screened cell's quasi-mode (lattice-adjacent), computes a
    screening Laplace log-marginal, softmax-normalises, and skips the
    full inner Newton on cells whose normalised weight is `< prune_tol`.
    The neighbour-warm-start sweep keeps every cheap mode near its
    cell's true mode, so the cheap ranking is faithful to the full-solve
    ranking even when the inner latent mode moves substantially across
    the grid. Pruned cells get `log_marginal = -Inf`, `n_iter = 0`, and
    inherit the pilot mode; the pilot cell is never pruned. A safety
    gate falls back to the full grid (with a warning) if the
    cheap-screen argmax disagrees with the full-solve argmax or the kept
    posterior collapses onto a cell the screen badly mis-estimated, so a
    silently-wrong pruned posterior is impossible. Stacks with
    `n_threads_outer`. `prune_tol` must be in `[0, 1)`. Keep it
    conservative (`<= 1e-3`); pruning helps most when the grid has many
    low-mass tail cells. Default `FALSE` (the full grid is correct).

  - `x_init` (`NULL`) – warm-start for the first grid point's inner
    solve.

  - `verbose` (`FALSE`) – when `TRUE`, announce the engaged outer
    integrator for a multi-block prior in one line at selection time
    (see `integration`), and report each CCD decline reason.

  - `store_Q` (`FALSE`) – also return the per-grid joint precision Q
    (lower triangle, CSC) as `Q_csc_p_per_grid`, `Q_csc_i_per_grid`,
    `Q_csc_x_per_grid`, `Q_csc_n`, letting callers compute INLA-style
    total-variance posterior moments (`Var-of-means + Mean-of-Var`) on
    inner latent coordinates such as fixed-effect betas.

  - `keep_grid_hessians` (`TRUE`) – retain the per-grid-point
    fixed-effect mode and marginal precision as `$grid_modes` /
    `$grid_hessians`, which is what lets
    [`summary.tulpa_fit()`](https://gillescolling.com/tulpa/reference/summary.tulpa_fit.md),
    [`confint.tulpa_fit()`](https://gillescolling.com/tulpa/reference/confint.tulpa_fit.md)
    and
    [`vcov.tulpa_fit()`](https://gillescolling.com/tulpa/reference/vcov.tulpa_fit.md)
    report the grid-marginalized fixed-effect covariance instead of
    `NA`. Memory is `O(n_fixed^2)` per cell. When the retention is not
    available the reason is recorded on `$grid_fixed_declined`.

  - `adaptive_grid` (`FALSE`), `adaptive_grid_edge_thresh` (`0.02`),
    `adaptive_grid_max_passes` (`1L`) – when `adaptive_grid = TRUE`, a
    mode-tracked 1D refinement pass triggers on any axis whose marginal
    boundary weight exceeds `adaptive_grid_edge_thresh`. New points are
    appended on that axis (interior densification + outward log-spaced
    extension) paired with the boundary cell's modal other-axis values,
    each carrying a calibration term so it contributes on the marginal
    scale – `O(n_new_points)` kernel solves, not the full cartesian
    product. The edge score is
    `max(marginal_weight_at_boundary, exp(max_log_marginal_at _boundary - max_log_marginal_overall))`,
    catching both boundary pile-up and integrand truncation; `0.02` is
    ~4 log units of decay. `adaptive_grid_max_passes` caps the passes
    (one usually suffices). Fixes posterior CI under-coverage when truth
    sits near a grid edge.

  - `var_of_means_consistency` (`TRUE`) – run a post-integration
    consistency pass on the variance of the per-arm posterior means and
    attach `var_of_means_consistency_info`.

  - `force_sparse` (`FALSE`) – linear-algebra backend for the inner
    joint solve. `TRUE` / `FALSE` select the sparse or dense path
    outright, regardless of the dense/sparse heuristic. `"auto"` selects
    by the latent dimension the fit will actually build, taking the
    sparse path above `n_x > 1000` and the dense one at or below it,
    where the sparse symbolic-analysis and indirection overhead
    outweighs the fill-in saving. A model whose latent dimension cannot
    be determined resolves to dense.

  - `integration` (`"auto"`) – outer-grid node layout for a multi-block
    prior. A central composite design (CCD) integrates the
    hyperparameter posterior on `1 + 2d + 2^d` nodes oriented by the
    Cholesky of the posterior covariance at the joint hyperparameter
    mode – far fewer inner solves than the `d`-dimensional tensor
    product (`25` vs `81` at `d = 4`). `"auto"` (the default) uses the
    CCD only at `>= 4` transformable axes, where the tensor product's
    `k^d` blow-up bites hardest, and keeps the cheaper, more
    ridge-robust tensor grid at `<= 3` axes; `"ccd"` lowers the CCD
    threshold to `>= 3` axes; `"grid"` always forces the full tensor
    product. `"grid_adaptive"` is the low-dimensional companion to the
    CCD: it seeds a coarse subsample of the SAME tensor lattice (latent
    block axes and phi axes together), floods outward from the posterior
    mode on the fine lattice, and evaluates only the cells within a
    log-density cutoff of the peak – a strict, uniform-weight subset of
    the dense tensor, so its posterior matches the dense grid to that
    cutoff at fewer inner solves when the hyperparameter posterior
    concentrates (a sharply-identified field SD / precision). It
    declines back to the dense tensor on a diffuse posterior (kept
    region would rival the tensor) or a degenerate lattice, so it never
    costs accuracy; tune it with `adaptive_grid_cutoff` /
    `adaptive_grid_stride` / `adaptive_grid_max_frac`. The CCD
    auto-falls back to the tensor grid for an axis whose support is not
    safely transformable (a CAR_proper `rho_car` or a non-BYM2 `rho`),
    or a flat / ridged / degenerate outer mode-find or Hessian; an
    active `phi_grid` rides as a tensor axis crossed on top of the CCD.
    Single-block joint priors always use the tensor grid. The CCD
    mode-find runs cheap warm-started inner solves and does not write to
    the checkpoint file. Under `verbose = TRUE` the engaged integrator
    is announced in one line at selection time (e.g.
    `outer integration: CCD (4 latent axes, 25 nodes)` or
    `tensor grid (72 cells)`, or `CCD declined -> tensor grid`), so the
    auto switch to the CCD at `>= 4` axes is never silent. The resolved
    integrator is also returned on the joint result as `$integration`,
    alongside `$integration_requested` and `$integration_declined` (see
    Value), so a fallback is on the fit itself and not only in a verbose
    message.

  - `local_ccd` (`NULL`) – local CCD refinement of a multi-block tensor
    grid. `TRUE` (defaults) or a `list(max_cells =, f0 =, skew_max =)`
    refines a few high-weight, mutually non-adjacent interior cells,
    replacing each with a small curvature-aware CCD node cloud so a
    coarse base grid resolves the sharply-peaked directions without the
    `k^d` tensor blow-up. The local curvature is a diagonal finite
    difference of the outer log-marginal over the cell's own grid
    neighbours (no mode-find; only the off-centre nodes are new solves),
    warm-started from the cell's inner mode; each refined cell's
    sub-nodes carry partition-of-unity design weights so the total
    integration weight is conserved (no double-count). `max_cells`
    (`8L`) caps the refined cells; `f0` (`1.1`) is the CCD radius. The
    design scale is shrunk per cell so the cloud fits the cell's Voronoi
    box (the local-Gaussian mass beyond it belongs to the neighbouring
    cells). A cell keeps its cloud only while the nodes' own
    log-marginals stay within `skew_max` (the `gamma3_ok` band, `0.5`)
    of the quadratic the cloud was placed from, measured as a
    standardized cubic magnitude; above it the cell is put back as its
    own mass atom, which on a skewed outer target is measurably closer
    than the design (gcol33/tulpa#318). Engages only on the tensor path
    (the curvature stencil needs axis neighbours), at `>= 4`
    transformable latent axes, with no active `phi_grid`; otherwise it
    is a no-op. The applied refinement is summarised on the result as
    `$local_ccd_info`. Also driven automatically by `k_refine = "ccd"`.

  - `adaptive_grid_cutoff` (`10`), `adaptive_grid_stride` (`2L`),
    `adaptive_grid_max_frac` (`0.75`), `adaptive_grid_min_cells` (`48`)
    – tuning for `integration = "grid_adaptive"`. `adaptive_grid_cutoff`
    is the log-density keep / expand radius from the peak (larger keeps
    more cells, closer to the dense tensor); `adaptive_grid_stride` the
    coarse-seed subsample stride per axis; `adaptive_grid_max_frac` the
    kept-fraction ceiling past which the builder declines back to the
    dense tensor; `adaptive_grid_min_cells` the smallest dense tensor
    worth the adaptive machinery – below it the builder declines BEFORE
    any inner solve (on a small tensor the coarse seed is already most
    of the grid, so there is no tail to skip). Ignored by the other
    integrators. The kept-cell / dense / solve counts are returned as
    `$adaptive_grid_info`.

  - `inner_refresh` (`1L`) – inner-Newton Cholesky factor reuse interval
    (Shamanskii / chord method). For a non-quadratic positive arm (e.g.
    a beta cover arm) the latent Hessian changes every inner iteration,
    so the default re-factorizes the sparse Cholesky on each step – the
    dominant per-grid-cell cost. `inner_refresh = m > 1` re-factorizes
    only every `m`-th inner step and reuses the cached factor in between
    (refreshing early whenever a reused solve fails). The gradient is
    exact on every step and each step is line-search safeguarded, so the
    converged mode is unchanged and the final mode-pass Hessian
    (`log_det`, SEs) is always fresh; only the path to the mode uses a
    stale curvature, which may cost a few extra inner iterations.
    Applies to the sparse joint path with the default
    `control$hessian = "lm"` curvature; the dense small-`n_x` path
    re-factorizes a cheap Hessian and ignores it. `2L`-`4L` is a good
    range for a slow beta arm.

  - `k_quality` (`"report"`) – the reliability intent for the outer
    Pareto-\\\hat{k}\\, a single statement of how reliable the fit
    should be. `"report"` (default) computes the diagnostic and reports
    the achieved band. `"ok"` / `"good"` additionally name a TARGET band
    (the \\\hat{k}\\ confidently usable, resp. good) and raise the
    default `k_samples` (to `800L` / `2000L`, unless you set it) so the
    bootstrap CI can resolve it. `"none"` disables the diagnostic. The
    fit carries an honest verdict – `k_quality_requested`,
    `k_quality_reached`, `k_quality_best`, `k_quality_reason`,
    `k_quality_rounds` – and never silently downgrades: if the requested
    band is not confidently met it reports the band actually reached and
    why. For `"ok"` / `"good"`, when the first fit does not reach the
    band the engine escalates by REFINING THE INTEGRATION GRID, driven
    by the bad \\\hat{k}\\ (see `k_refine`): each round widens /
    densifies the grid where the posterior mass escapes its current
    bounds and re-diagnoses, up to `k_max_rounds` times. This is the
    actual fix for a grid-width deficiency; `k_samples` is the separate
    knob that sharpens the \\\hat{k}\\ ESTIMATE and is not escalated
    here.

  - `k_refine` (`"grid"`) – the integration-refinement rung for
    `k_quality` `"ok"` / `"good"`. `"grid"` (default) re-fits with
    adaptive grid refinement (`adaptive_grid`) each escalation round,
    driven by the bad \\\hat{k}\\, so a too-coarse / too-narrow grid is
    widened / densified where the importance weight concentrates until
    the band is reached or the budget is spent. `"ccd"` instead refines
    a few high-weight, mutually non-adjacent interior cells with local
    curvature-aware CCD node clouds (see `local_ccd`), the right rung
    when the grid is too coarse to resolve a sharply-peaked direction
    rather than too narrow; it forces a tensor base grid (the curvature
    stencil needs axis neighbours) and engages only on the multi-block
    path at \>= 4 transformable latent axes. `"none"` disables
    refinement: the band is reported but not chased.

  - `k_max_rounds` (`2L`) – the grid-refinement round budget for
    `k_quality` `"ok"` / `"good"`: the maximum number of
    refine-and-re-fit rounds after the first fit. Each round allows one
    more refinement pass than the last. `0L` disables escalation
    (single-shot, the band is reported but not chased).

  - `diagnose_k` (`TRUE`), `k_samples` (`500L`) – compute the outer
    Pareto-\\\hat{k}\\ accuracy diagnostic by importance-sampling the
    joint hyperparameter posterior against the proposal the integrator
    fits (mixed per-axis transforms: `log` for positive scales, logit
    for the BYM2 mixing weight, identity for the copy coefficient
    \\\alpha\\). `k_samples` is the number of importance draws, each one
    an extra inner joint solve, and is the diagnostic's precision knob:
    a tighter k-hat needs MORE actual tail ratios, so increase
    `k_samples` (not `k_bootstrap`). The draws are RNG-restored so the
    fit's modes / draws are unchanged. A fit carrying an axis whose
    support is not safely known (CAR_proper's `rho_car`) declines to the
    quadrature-ESS fallback (`pareto_k = NA`). `FALSE` skips the
    diagnostic. `"by_arm"` additionally computes a k-hat restricted to
    each arm's hyperparameter axes (the other arms held at their
    posterior mean), reported in `pareto_k_by_arm`, to localise which
    arm drives a tail-heavy joint k; the joint k itself is unchanged.
    Per-arm k is defined for the multi-block layout with two or more
    arms and declines for the single-block shared-field layout. The
    legacy `k_samples` name is accepted as an alias for `k_samples`.

  - `k_threads` (`NULL`) – outer-thread width for the diagnostic's
    importance batch. The `k_samples` re-solves are independent and run
    after the grid (every core free), each solved single-threaded once
    the batch saturates the pool, so widening it is a bit-identical
    wall-clock speedup (the k-hat is unchanged). `NULL` follows the
    fit's own thread grant – the larger of `n_threads_outer` and the
    inner `n_threads` – so a serial fit keeps a serial diagnostic while
    a threaded fit gets a free parallel one. `"auto"` uses the physical
    performance-core count (capped at 2 under R CMD check); an integer
    pins the width (`1L` forces serial). Always capped at `k_samples`.

  - `k_bootstrap` (`1000L`) – bootstrap replicates for the outer
    Pareto-\\\hat{k}\\ uncertainty. The k-hat is a single fixed number
    for a fit + proposal; its sampling uncertainty GIVEN the proposal is
    estimated by resampling the diagnostic's raw importance log-ratios
    with replacement and re-fitting the GPD tail `k_bootstrap` times (no
    new inner solves). Reports `pareto_k_se_boot` (bootstrap SE),
    `pareto_k_ci_low` / `pareto_k_ci_high` (the 2.5\\ closed-form
    GPD-shape MLE asymptotic SE \\(1 + k)/\sqrt{M}\\, a cross-check),
    and `pareto_k_band_confident` (TRUE iff the bootstrap CI lies within
    one reliability band). The bootstrap measures how UNSTABLE the
    current tail estimate is; it cannot create tail information.
    Increase `k_samples`, not `k_bootstrap`, to obtain more tail
    information. `0L` skips it (point k-hat only). The per-arm k carries
    the same fields.

  - `k_tail_points` (`NULL`) – number of upper-tail order statistics for
    the GPD fit. `NULL` uses the automatic PSIS rule \\\lceil\min(0.2 N,
    3\sqrt{N})\rceil\\. An explicit value is an EXPERT tail-threshold
    control, capped at the 20\\ so the fit stays an extreme tail; it is
    NOT a precision knob (a request that drags body ratios into the tail
    lowers variance but biases the k-hat). The used and requested counts
    are reported in `pareto_k_tail_points` /
    `pareto_k_tail_points_requested`.

  - `k_conf_bands` (`NULL`) – the reliability-band boundaries the
    bootstrap CI is tested against for `pareto_k_band_confident`. `NULL`
    (default) uses the sample-size-dependent boundaries \\c(0.5,
    \min(1 - 1/\log\_{10} S, 0.7))\\ at the realised draw count \\S\\
    (Vehtari et al. 2024): the good cut is 0.5 and the usable cut
    tightens below 0.7 for small \\S\\ (about 0.565 at \\S = 200\\,
    reaching 0.7 only past \\S \approx 2154\\). Supply a
    strictly-increasing numeric vector to fix the boundaries instead,
    e.g. `c(0.5, 0.7)` for the size-independent good / ok / unreliable
    split.

  - `diagnose_skew` (`TRUE`), `skew_idx` (`NULL`) – compute the
    inner-Laplace skewness diagnostic (`$inner_skew`, gamma_3, Rue
    Martino & Chopin 2009 Sec 3.2.3) at the fitted MAP grid cell: one
    extra Newton solve via the same `kernel_fn` the outer diagnostic
    reuses, scoring every arm's fixed-effects coefficients by default
    (pass `skew_idx`, 1-based indices in the joint
    `[arm1_beta | arm1_re | arm2_beta | ... | blocks]` latent layout –
    see `$arm_layout` – to probe additional indices). This is the
    complementary layer to `diagnose_k`: that scores the outer
    hyperparameter-grid integration around a FIXED inner Laplace, this
    scores whether that inner Gaussian approximation is itself a good
    fit. A genuinely coupled arm (`cell_coupling != "separable"` on that
    arm, e.g. tulpaObs's `occu_cover`) has no per-obs likelihood for the
    separable formula to score; it is scored instead by the contraction
    of the cell third-derivative tensor (gcol33/tulpa#301), which
    differences the cross-arm Hessian the coupling spec already returns.
    A fit that can carry neither reports `NaN`, not a silently wrong 0,
    and `$inner_skew_declined` says WHY – `"coupled_arm"` marks arms the
    inner layer could not score at all, distinct from a diagnostic that
    was simply switched off (gcol33/tulpa#296) – while
    `$inner_skew_arms_declined` names the arms with no oracle, including
    on a partially scored fit. See
    [`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
    for the combined verdict. Wired for both the single-block backends
    (icar/bym2/car_proper) and the multi-block path (a per-group RE, a
    trend field, or an arm-specific field block).

  - `within_cell` (`"box_uniform"`) – the WITHIN-CELL construction the
    reported per-axis hyperparameter intervals are read with
    (gcol33/tulpa#357). The outer grid's weights say how much mass each
    cell holds; they do not say how it is spread inside the cell, and a
    quantile needs both. `"box_uniform"` puts the cumulative FULL mass
    at each cell EDGE and interpolates between edges; `"chord"` puts the
    cumulative MID-mass at each cell coordinate and interpolates between
    coordinates – the same masses over the same boxes with the knots
    moved half a cell, which measures as a whole order of convergence
    (2.00 against 1.04 on a fixture with a closed-form posterior). THE
    DEFAULT IS `"box_uniform"` since 0.0.188, decided on FIXED-TRUTH
    coverage at the placement the engine ships since gcol33/tulpa#361
    made `auto_recenter = "resolve"` the default. Summed \|coverage -
    nominal\| over nominal 0.95 / 0.80 / 0.50, chord against
    box-uniform: 0.2900 / 0.1233 on the pre-registered fixed-truth
    instrument, 0.2004 / 0.0361 over 4680 truth-swept fits of the same
    fixture, and 0.2467 / 0.1572 over nine (config, axis) rows spanning
    seven families, at 0.69 to 1.08x the width. The conditional-coverage
    swing that held the default back reads 0.110 at the shipped
    placement against 0.415 on the coarse pinned grid it was measured
    on, and at nominal 0.50 it is the same on both reads.
    `outer_grid_h_over_sd` is how wide a cell is on each axis, and
    `theta_within_cell` is what each axis was actually read with. Only a
    `"density"` support admits it – a CCD design, a locally refined grid
    and a posterior sample are not cell partitions that tile – and an
    axis it declines on reports `"chord"` with a reason rather than
    erroring. Nothing else moves: point estimates, moments, draws and
    weights are untouched, and `"chord"` restores the previous report
    exactly.

  - `skew_correct` (`TRUE`) – consume the inner-Laplace expansion
    instead of only grading it (gcol33/tulpa#302, gcol33/tulpa#354):
    report Cornish-Fisher marginal quantiles at each coefficient's own
    `gamma_3`, about the centre `gamma_1 + gamma_3 / 2`, from
    [`summary()`](https://rdrr.io/r/base/summary.html) /
    [`confint()`](https://rdrr.io/r/stats/confint.html) wherever the
    combined inner band (`gamma_3` and the inner importance k-hat) says
    the leading-order expansion is in its regime, and the Gaussian
    quantiles everywhere else. It is post-processing on the reported
    quantiles: draws, modes and weights are untouched, so a fit run with
    it off is bit for bit the fit it was before. The band that bounded
    the relocation itself (`centre_unreliable`, gcol33/tulpa#362) is off
    (`Inf`): every finite cutoff was measured to decline the
    coefficients the correction helps most (gcol33/tulpa#376).
    `$skew_correction` records the per-coefficient `gamma_3`, `gamma_1`
    and the centre they form, the band, the k-hat, the combined band,
    the eligibility and the reason behind it; the `skew_applied`
    attribute on [`summary()`](https://rdrr.io/r/base/summary.html) /
    [`confint()`](https://rdrr.io/r/stats/confint.html) records what was
    actually used at the requested level. A FULLY COUPLED fit declines
    it: the location term's contraction against a covariance block is
    not reachable from the cell third-derivative oracle, and an absent
    `gamma_1` is never read as zero. Such a fit therefore reports what
    it reported before the default moved, to the bit – a declined
    coefficient keeps the grid-mixture read (gcol33/tulpa#386). See
    [`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
    for the measurement behind the default.

  - `subspace_debias` (`FALSE`) – correct only the latent directions the
    inner-layer diagnostics flagged, by exact Metropolis along the
    Gaussian-conditional-mean surface through each cell's mode, and
    leave the rest at their Gaussian conditional (gcol33/tulpa#304,
    extended to both joint paths by gcol33/tulpa#306). Settings and
    semantics are the ones documented on
    [`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md).
    On a fully coupled fit `gamma_3` is `NaN` for every arm, so the
    selector rests on the derivative-free inner Pareto-k-hat
    (gcol33/tulpa#303); where that also bands the coordinate reliable,
    `idx` pins the set explicitly. An EMPTY selection leaves the fit
    bit-for-bit identical to the plain path. A fit whose inner solve
    took the s2z rank-1 or the PSD eigen-clamp path carries no usable
    factor to build the surface from and is left uncorrected, the same
    two paths `diagnose_skew` declines on.

  - `cila` (`FALSE`) – corrected integrated Laplace, the second
    inner-layer debias (gcol33/tulpa#351, after Lai, Margossian and
    Sheldon, arXiv:2605.20345). Where `subspace_debias` selects
    coordinates and runs exact Metropolis on them, this selects nothing:
    at every outer cell it draws `n_points` points from the whole inner
    Gaussian, weights each by the exact joint density it came from, and
    reports the weighted particles. The cell marginals and the latent
    posterior both converge to the exact ones as the effort grows, so
    `n_points` is the only dial. `TRUE` takes the defaults; a list
    overrides `n_points` (`1024L`), `variant` (`"qmc"`, a Sobol net;
    `"is"` for iid draws, `"rqmc"` for the net under `n_shift` random
    shifts), `n_shift` (`8L`), `n_draws` (the reported draw count) and
    `seed` (the auxiliary stream, engine-owned so requesting the
    correction leaves every other posterior draw unchanged). Below 512
    points a cell's particle set is too coarse to be a marginal at all
    and the request is refused. The correction reports DRAWS, so every
    coefficient-facing method reads them instead of the grid's Gaussian
    mixture, and the corrected per-cell masses become the fit's own
    `weights` / `log_marginal` with `weights_source` reporting `"cila"`
    (gcol33/tulpa#367); the pre-correction pair is kept as
    `$cila$laplace` and `$cila$retained_mass` is the original share of
    the cells that produced a usable particle set. `$cila` also carries
    the variant actually run and the PSIS grade (`pareto_k`, `rel_ess`)
    of the correction's own weights. A cell whose inner solve factorized
    SPARSELY draws through the CHOLMOD factor's own triangular and
    permutation solves (gcol33/tulpa#366); an LDL' factor carries no
    square root to draw with and is declined with
    `"sparse_factor_not_ll"`.

  - `auto_recenter` (`TRUE`) – re-centre a default outer grid axis on
    its posterior mode and refit when the fit rails against a boundary
    node (gcol33/tulpa#289 / \#290). `FALSE` integrates over the grid
    exactly as given, whatever it is, and records
    `outer_grid_recenter_declined = "auto_recenter_disabled"`. The joint
    rescues trigger on the whole grid's collapsed-edge regime rather
    than on a per-axis rail, so the per-axis policy names
    [`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
    takes (`"rail"`, `"resolve"`, `"always"`) are refused here with an
    error rather than accepted and ignored.

  - `max_grid_cells` (`2048L`) – cell-count ceiling on a multi-block
    tensor outer grid, refused with an error above it. Each cell is one
    inner Newton solve, so the default catches per-block grids that
    multiplied out to a run nobody asked for; a deliberate converged
    tensor reference grid (4 axes at 7 levels is 2401 cells) raises it
    here, which `integration = "ccd"` cannot serve since a CCD is a
    different integration design.

  - `checkpoint` (`NULL`) – grid-cell checkpoint/resume. Set
    `list(path = "fit.ckpt", resume = TRUE)` to make a killed or
    interrupted fit resumable: each completed outer-grid cell is
    appended to `path`, and a later call with the same responses +
    grid + control loads the finished cells and solves only the rest.
    EVA-scale joint fits run for hours, so a wrapper teardown, reboot,
    or OOM near the end otherwise loses the whole run. `resume = TRUE`
    (the default when a `path` is given) continues from an existing
    file; `resume = FALSE` removes it first and starts over.
    Adaptive-grid refinement cells are checkpointed under their own
    coordinate keys, so resume covers them too. A file written for
    different data or solver settings is rejected (fingerprint mismatch)
    rather than resumed onto a stale result.

## Value

A list of class
`c("tulpa_nested_laplace_joint", "tulpa_nested_laplace", "list")` with:

- `theta_grid`, `theta_names` – outer-grid hyperparameter values
  (includes the `alpha` axis when `copy` is set).

- `log_marginal`, `weights` – per-grid-point log-marginal and
  integration weights (sum to 1).

- `theta_mean`, `theta_sd` – posterior moments per hyperparameter,
  including `alpha` when `copy` is set.

- `theta_median`, `theta_ci_lo`, `theta_ci_hi` – weighted-quantile
  median and 2.5/97.5 empirical CI per hyperparameter axis (same names
  as `theta_mean`). Recommended summary for right-skewed scale-like axes
  (alpha at small n_pos, sigma/range/phi near a boundary), where the
  posterior mean is pulled by the right tail away from the bulk and
  `mean +/- 1.96 sd` mis-states the uncertainty.

- `modes` – `[n_grid x n_x]` matrix of inner modes.

- `n_iter` – inner Newton iterations per grid point.

- `arm_layout` – list with per-arm `beta_start`, `re_start`, spatial
  offset(s) and `n_x` for decoding modes.

- `prior`, `responses`, `copy` – echoed inputs.

- `timing` – named numeric of wall-clock seconds: `total` plus the
  `setup` (validation / encoding / grid construction), `grid` (inner
  Laplace solves, including adaptive-refinement and consistency passes),
  `postproc` (weight / moment / marginal assembly) and `diagnostics`
  (outer Pareto-\\\hat{k}\\) phases. The `grid` phase is the one that
  scales with grid size and core count. Surfaced one-line in `print`.

- `pareto_k`, `pareto_k_is_ess`, `pareto_k_scope` – outer
  Pareto-\\\hat{k}\\ accuracy diagnostic and its importance-sampling ESS
  (both `NA` when `control$diagnose_k = FALSE` or the fit declines; see
  the `diagnose_k` control knob). `pareto_k < 0.7` indicates the nested
  integration is reliable; `>= 0.7` that the (skewed / heavy- tailed)
  hyperparameter posterior is misfit by the Gaussian grid and the fit
  should escalate to an exact debias.

- `pareto_k_declined` – when `pareto_k` is `NA`, WHY (gcol33/tulpa#295):
  `"not_requested"` (`control$diagnose_k = FALSE`; nothing is wrong),
  `"unguessable_axis: <axis>"` (a support the engine will not guess,
  e.g. car_proper's `rho_car` – a PERMANENT limitation of that family,
  so read the quadrature ESS instead), `"draws_too_few"`,
  `"grid_too_small"`, `"no_varying_axis"`, `"degenerate_proposal"` (the
  outer mode curvature came back non-finite – a signal about the fit),
  `"not_applicable"`, or `"internal_inconsistency"` (an engine bug worth
  reporting). `NA` on a fit whose \\\hat{k}\\ WAS computed.

- `pareto_k_proposal_source` – how the outer importance proposal the
  \\\hat{k}\\ scores was built: `"mode_hessian"` from the Laplace
  curvature at the hyperparameter mode (the CCD design's, or a
  finite-difference Hessian when a sharp posterior collapses the grid),
  `"grid_moment"` from the grid-weighted covariance, `"moment_matched"`
  when the moment-matching refinement improved on either,
  `"grid_mixture"` when the faithful mixture-over-grid-cells proposal
  won, or `"skew_normal"` when the skew-normal rescue did. `NA` when the
  diagnostic is off or declines. The mode-Hessian source keeps the
  \\\hat{k}\\ meaningful when the grid concentrates on ~1 cell.

- `pareto_k_regime` – whether the outer grid actually integrated
  hyperparameter uncertainty: `"spread"`, or `"collapsed_interior"` /
  `"collapsed_edge"` when the quadrature effective sample size falls
  below two cells so the integration has degenerated to a point
  evaluation at the modal hyperparameter. An interior collapse is benign
  (the grid bracketed the mode; the fit is empirical Bayes at that mode
  and only the integrated hyperparameter uncertainty is missing); an
  edge collapse means the mode sits at an extreme node, so the grid may
  be too narrow. Attached from the stored weights, so it is present even
  with `control$diagnose_k = FALSE`.

- `pareto_k_grid_edge_axes`, `pareto_k_grid_edge_sides` – for an edge
  collapse, which axes the dominant cell sits against and on which side
  (`"lower"` / `"upper"`); widen those axes and refit to confirm the
  mode is bracketed. Axes the grid pins to one value are excluded
  (pinned, not at a boundary).

- `outer_grid_placement` – `"fixed"` (the default `sigma_grid` axis was
  used as-is) or `"auto_recentered"` when a `collapsed_edge` on `sigma`
  triggered the mode-Hessian recenter-and-refit (gcol33/tulpa#289; see
  the `prior` argument above). `outer_grid_recenter_attempts` (integer)
  and `outer_grid_prior_added` (logical: whether the light default
  PC(U=3, alpha=0.01) sigma prior was applied) are set alongside it,
  plus `outer_grid_prior_declined = "prior_pinned"` when a second
  attempt ran but the caller's pinned `prior_sigma` held that prior back
  (gcol33/tulpa#297).

- `outer_grid_recenter_declined` – on a `"fixed"` placement, why the
  recenter did not run: `"grid_not_collapsed"` (the grid already
  brackets the mode – the common case, no refit needed), `"axis_pinned"`
  (the caller pinned `sigma_grid`; mark it with
  [`auto_grid()`](https://gillescolling.com/tulpa/reference/auto_grid.md)
  if it is a default rather than a choice), or `"no_usable_curvature"`
  (the mode-Hessian the recenter needs was unavailable or degenerate,
  e.g. a car_proper grid whose `rho_car` axis has unguessable support).
  Absent when the fit WAS recentred.

- `outer_grid_recenter_sd_clamp`, `outer_grid_recenter_sd_raw`,
  `outer_grid_recenter_sd_used` – on a recentred fit, one entry per
  moved axis: which mode-SD bound the placement hit (`"none"` /
  `"floor"` / `"ceiling"`), the SD the finite-difference stencil
  MEASURED, and the SD the axis was actually laid from
  (gcol33/tulpa#387). A clamped axis was laid from a substituted spread
  rather than a measured one, and the reported interval is read off that
  span; the two used to be indistinguishable on the fit.

- `pareto_k_outer_skew` – per-axis skewness of the hyperparameter
  marginal in the proposal's whitened coordinate, present only when a
  \\\hat{k}\\ above the good band triggered the skew-normal rescue pass.
  It is the EXPLANATION for an inflated \\\hat{k}\\: a symmetric
  Gaussian proposal against a right-skewed variance-component marginal
  has a heavy importance-ratio tail whatever the integration's quality.
  `NULL` when the Gaussian proposal already fit.

- `pareto_k_se_boot`, `pareto_k_ci_low`, `pareto_k_ci_high`,
  `pareto_k_se_formula`, `pareto_k_tail_points`,
  `pareto_k_tail_points_requested`, `pareto_k_band_confident` – the
  outer Pareto-\\\hat{k}\\ uncertainty, present whenever the diagnostic
  ran. `pareto_k_se_boot` is the bootstrap SE of the k-hat and
  `pareto_k_ci_low` / `pareto_k_ci_high` its 2.5\\ quantiles – the
  estimator's sampling spread GIVEN the proposal, not a posterior
  credible interval; `pareto_k_se_formula` is the closed-form GPD-shape
  MLE asymptotic SE cross-check. `pareto_k_tail_points` is the GPD tail
  size used and `pareto_k_tail_points_requested` the `k_tail_points`
  request (`NA` when automatic). `pareto_k_band_confident` is TRUE iff
  the bootstrap CI lies within one reliability band, `NA` when the
  bootstrap was off or could not fit. `diagnose_draws` and
  `diagnose_cost_ratio` (the diagnostic's draw budget and its wall-clock
  cost relative to the fit) are attached at the top level.

- `pareto_k_by_arm`, `pareto_k_by_arm_is_ess`, `pareto_k_by_arm_scope` –
  present only with `control$diagnose_k = "by_arm"`. Named (by arm)
  outer Pareto-\\\hat{k}\\ restricted to each arm's hyperparameter axes,
  the other arms held at their posterior mean, so a tail-heavy joint k
  can be localised to one arm. A per-arm entry is `NA` when that arm
  carries no varying axis. Each arm carries its own bootstrap
  uncertainty: `pareto_k_by_arm_se_boot`, `pareto_k_by_arm_ci_low` /
  `pareto_k_by_arm_ci_high`, `pareto_k_by_arm_se_formula`,
  `pareto_k_by_arm_tail_points` and `pareto_k_by_arm_band_confident`.

- `k_quality_requested`, `k_quality_reached`, `k_quality_best`,
  `k_quality_reason`, `k_quality_rounds` – the reliability verdict for
  the `control$k_quality` intent. `k_quality_requested` echoes the
  intent; `k_quality_best` is the band actually achieved (`"good"` /
  `"ok"` / `"unreliable"`, or `"uncertain"` when the bootstrap CI
  crosses a boundary); `k_quality_reached` is `TRUE`/`FALSE` for an
  `"ok"` / `"good"` target (`NA` for `"report"` / `"none"`), never
  silently downgraded; `k_quality_reason` records why it stopped; and
  `k_quality_rounds` is the number of escalation re-fits performed (`0`
  when the first fit sufficed or escalation was off).

- `adaptive_grid_info` – when `adaptive_grid = TRUE`, a list with
  `triggered_axes` (character) and `n_points_added` (integer) describing
  the refinement passes. NULL otherwise.

- `local_ccd_info` – when `local_ccd` engaged (or `k_refine = "ccd"`), a
  list with `n_cells_refined`, `n_nodes_added`, the refined `cells`, the
  `n_cells_before` / `n_cells_after` grid sizes, `n_design_nodes` (the
  cells that carry an in-cell design weight rather than their own mass),
  `design_mass` (the share of the integration weight sitting on them)
  and `cell_share` (the share each refined cell held on the BASE grid,
  before any node was placed). The two shares differ: the replacement
  nodes sit nearer the peak than the cell's own coordinate did, so
  refining raises the share the refined region holds, and reading both
  separates how concentrated the base grid already was from how much the
  refinement concentrated it. Three per-cell readings come with it, on
  three orthogonal axes, none of them gating anything. Shape: `misfit`
  (the standardized cubic magnitude of the part of the cell the design
  cannot represent, the one reading `skew_max` gates on, with
  `misfit_declined` / `cells_declined` for the cells put back as their
  own mass atoms). Centring: `offset` (the norm of the whitened
  gradient, i.e. how far the cell's own peak sat from the cell's
  coordinate in units of its marginal spread) and `mode_gain` (the same
  displacement in the cell's own curvature units, `0.5 * g' (-H)^-1 g`
  in nats, read off the same fit's quadratic coefficients and comparable
  across cells whose curvature differs; `NA` where `-H` is not positive
  definite, a cell with no interior peak to be displaced from). Mass:
  `log_mass_ratio` (the cell's refined mass over the coarse atom it
  replaced, `log_mass_refined - log_mass_coarse`, both carried on the
  log scale with the cell's outer design weight in them) and
  `max_node_weight` (the share the single largest node takes of its own
  cell's refined mass). Each carries a `_declined` twin for the cells
  the gate put back, whose nodes were evaluated before the score was
  read. `misfit` certifies that the design can represent the cell's
  shape and nothing more: a cell can be exactly quadratic and still be
  read at a point that is not representative of it (`offset` /
  `mode_gain`), and refining it can still move how much mass it competes
  for against its unrefined neighbours (`log_mass_ratio`, the
  embedded-rule local error indicator of adaptive cubature). NULL when
  local CCD was off or declined (single-block, `< 4` axes, an active
  `phi_grid`, or no peaked interior cell).

- `integration_requested`, `integration_declined` – what `integration`
  asked for, and why the CCD did not run. `$integration` names the
  integrator that RAN, and `.nl_node_support()` keys the interval
  construction off it, so a caller who asked for a moment rule and
  received a density grid reads the reason here rather than inferring
  it. `integration_declined` is `NA_character_` when nothing was
  declined, and otherwise one of `"axis_count"` (fewer transformable
  latent axes than the requested mode's threshold), `"unguessable_axis"`
  (a CAR_proper `rho_car` or a non-BYM2 `rho`), `"degenerate_axis"` (a
  single-valued axis), `"modefind_ridge"` / `"modefind_boundary"` /
  `"modefind_degenerate"` / `"modefind_failed"`, `"hessian_singular"` or
  `"hessian_not_pd"`. Multi-block fits only.

- `weight_kind` – one entry per outer-grid cell, `"mass"` or `"design"`.
  A tensor cell holds the mass of its own cell and a CCD node holds a
  design weight, so a fit integrated by one rule reports one value
  throughout; a locally refined grid carries both, and this says which
  per cell instead of leaving it to be read off `integration`.

- `theta_interval_read`, `theta_interval_design_mass` – what the
  reported per-axis `theta_median` / `theta_ci_lo` / `theta_ci_hi` were
  read off, and how much of the integration weight sits on nodes whose
  cumulative sum is not a CDF. `"density"` is a weighted quantile over
  cells that discretize the posterior (`design_mass` 0), `"moment_rule"`
  an interval from the moments a central-composite design delivers
  (`design_mass` 1), and `"mixed"` the locally CCD-refined grid, which
  carries both kinds at once. A mixed support still reports the weighted
  quantile, which measured best against a converged reference in both
  `design_mass` regimes; the share is the regime variable, since on that
  part of the support the quantile is bounded by the refined cells' own
  grid neighbourhoods rather than by the posterior (gcol33/tulpa#317).
  Every nested path stamps the pair, not only the multi-block driver.

- `within_cell_requested`, `theta_within_cell`,
  `theta_within_cell_declined` – the WITHIN-CELL construction the same
  intervals were read with (gcol33/tulpa#357). The kind above says what
  the integrator left; this says how each cell's mass was spread inside
  its own box when the grid was read back. `"box_uniform"` is the
  default and puts the cumulative full mass at each cell edge; `"chord"`
  (`control$within_cell`) puts the cumulative mid-mass at each cell
  coordinate, the same masses over the same boxes with the knots moved
  half a cell. The construction is recorded per axis, and an axis whose
  cell partition could not be built falls back to `"chord"` on its own
  with the reason in `theta_within_cell_declined` (`"support_<kind>"`,
  `"single_node"`, `"boxes_do_not_tile"`, `"no_usable_node"`).

- `theta_cell_edge_coord`, `theta_cell_edge_declined` – per axis, the
  COORDINATE that axis's outer cell edges were mirrored in, and why the
  support the axis declares did not produce them (gcol33/tulpa#377). A
  declared support is authoritative: its mirrored edge is used when
  finite and inside it, and otherwise the axis reports its extreme grid
  coordinate – which the containment test has already placed inside the
  support – rather than falling through to a coordinate guessed from the
  node values, which is what put a lower bound of 0 on a `positive` axis
  and a bound above 1 on a `unit` one. An axis that declares no support
  at all keeps the guess by design, but its mirrored edge is checked for
  being a representable double that brackets the coordinates, and falls
  back to the extreme grid coordinate with
  `"mirrored_edge_not_representable"` when it is not (gcol33/tulpa#379).
  `theta_cell_edge_declined` is `NA` where nothing was declined.

- `outer_grid_cell_width`, `outer_grid_axis_sd`, `outer_grid_h_over_sd`
  – per axis, the cell width and the posterior SD in that axis's own
  coordinate, and their ratio. A within-cell reconstruction resolves an
  interval endpoint to within one cell, so how much of the reported
  width and of its realized coverage is a property of where the grid
  fell rather than of the posterior is governed by this ratio; below
  `.nl_diag("grid_resolved")` the cells are narrower than the posterior
  they discretize and the two constructions converge to each other. NA
  on an axis whose own marginal is maximal at an endpoint, which is the
  placement question `outer_grid_railed_axes` reports.

- `prune_cheap_log_marginal`, `prune_mask`, `prune_n_pruned`,
  `prune_tol` – present only when `prune = TRUE` and the safety gate did
  not fall back. Cheap-pass log-marginals at every cell, a logical mask
  of pruned cells, the pruned-cell count, and the threshold actually
  applied. Pruned cells have `log_marginal = -Inf` so they get zero
  weight under `.nl_normalise_weights_safe`.

- `prune_fallback_triggered`, `prune_fallback_reason` – present only
  when the safety gate fell back to the full grid. The returned fit is
  the full-grid (unpruned) result; the reason string records which gate
  condition tripped.

## (sigma, alpha) parameterization

Each arm's linear predictor reads \$\$\eta\_{arm} = X\_{arm}
\beta\_{arm} + \sigma\_{arm} \cdot z_s,\$\$ where \\z_s\\ is a
unit-precision latent (ICAR(tau=1) for ICAR/BYM2, or the BYM2 mix;
CAR_proper uses the structure of \\D - \rho\_{car} W\\). All arms share
the donor amplitude \\\sigma\\ from the outer-grid `sigma_grid` axis.
The copy arm's amplitude is \\\sigma\_{arm} = \alpha \cdot \sigma\\,
where \\\alpha\\ is a direct outer-grid axis taken from
`copy$alpha_grid`. The Cartesian product is over
`(sigma, [rho/rho_car,] alpha[, phi])`. Direct \\\alpha\\ as an
outer-grid axis (rather than a post-hoc ratio \\\alpha = \sigma\_{pos} /
\sigma\_{occ}\\) avoids plug-in bias on the weakly-identified ratio at
small `n_pos` and lets a regularizing hyperprior land on \\\alpha\\
directly.

## References

Rue, Martino & Chopin (2009). Approximate Bayesian inference for latent
Gaussian models by using integrated nested Laplace approximations.
*JRSS-B* 71(2):319-392.

## See also

[`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
for the single-arm engine.

## Examples

``` r
# \donttest{
set.seed(1)
S <- 25L                                   # shared spatial units (chain graph)
nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
nn <- lengths(nb)
field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))
mk_arm <- function(m, fam) {
  si <- sample(S, m, replace = TRUE); x <- rnorm(m)
  lin <- 0.2 + 0.5 * x + 0.8 * field[si]
  y <- if (fam == "binomial") rbinom(m, 1L, plogis(lin)) else lin + rnorm(m, 0, 0.5)
  list(y = as.numeric(y), n_trials = rep(1L, m), X = cbind(1, x),
       spatial_idx = si, family = fam, phi = if (fam == "gaussian") 0.5 else 1)
}
prior <- list(type = "icar", n_spatial_units = S,
              adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
              n_neighbors = nn, sigma_grid = c(0.3, 0.7, 1.4))
fit <- tulpa_nested_laplace_joint(
  responses = list(occ = mk_arm(200L, "binomial"), pos = mk_arm(200L, "gaussian")),
  prior = prior)
#> [nested-laplace-joint] 1/3 cells (33%) | elapsed 0s | ETA >=0s | 0.00s/cells
#> [nested-laplace-joint] 3/3 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
fit$theta_mean        # shared field amplitude, integrated across both arms
#>     sigma 
#> 0.3147239 
# }
```
