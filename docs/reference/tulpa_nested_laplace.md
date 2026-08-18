# Nested Laplace approximation for latent Gaussian models

Generic outer-grid nested Laplace driver. Builds a grid over the
hyperparameters of a single latent prior block (spatial or temporal),
runs an inner Laplace at each grid point with warm-starting, and
integrates over the grid to give proper hyperparameter marginals.

Supported priors:

- Spatial (areal): `"icar"` (1D grid on tau), `"bym2"` (2D on (sigma,
  rho)), `"car_proper"` (2D on (tau, rho); rho lives in the eigenvalue
  interval (1/lambda_min, 1/lambda_max) of `D^{-1} W`).

- Spatial (continuous): `"nngp"` (2D on (sigma2, phi_gp)), `"hsgp"` (2D
  on (sigma2, lengthscale)).

- Temporal: `"rw1"`, `"rw2"` (1D grid on tau), `"ar1"` (2D on (tau,
  rho))

- SPDE continuous spatial: see `cpp_nested_laplace_spde()` (separate
  entry, rebuilds Q via SPDE Q-builder).

## Usage

``` r
tulpa_nested_laplace(
  y,
  n_trials,
  X,
  prior = NULL,
  spec = NULL,
  data = NULL,
  re_idx = NULL,
  n_re_groups = 0L,
  sigma_re = 1,
  family = "binomial",
  phi = 1,
  likelihood = NULL,
  control = list()
)
```

## Arguments

- y:

  Response vector.

- n_trials:

  Trial sizes (binomial). Pass `1L`-vector otherwise.

- X:

  Fixed-effects design matrix.

- prior:

  A list describing the latent prior block. Required field `type` one of
  {"icar", "bym2", "car_proper", "rw1", "rw2", "ar1"}. Type-specific
  fields:

  - icar: `spatial_idx`, `n_spatial_units`, `adj_row_ptr`,
    `adj_col_idx`, `n_neighbors` (CSR adjacency, 0-based); optional
    `tau_grid`; optional `svc_weight` (one weight per observation) to
    make it a spatially-varying coefficient whose eta contribution is
    `svc_weight[i] * z[spatial_idx[i]]` rather than `z[spatial_idx[i]]`.

  - bym2: same adjacency; `scale_factor`; optional `sigma_grid`,
    `rho_grid`.

  - car_proper: same adjacency; optional `tau_grid`, `rho_grid`,
    `rho_bounds = c(lower, upper)` (defaults to (0, 1)).

  - rw1/rw2: `temporal_idx` (1-based), `n_times`; optional `tau_grid`,
    `cyclic` (default FALSE).

  - ar1: `temporal_idx`, `n_times`; optional `tau_grid`, `rho_grid`.

  A default grid axis is a starting axis, not a hard ceiling: for `icar`
  (`tau_grid`) and `bym2` (`sigma_grid`) a posterior mode that rails a
  boundary node (`pareto_k_regime = "collapsed_edge"`) triggers one
  mode-Hessian recenter-and-refit (gcol33/tulpa#290), reported through
  `outer_grid_placement` / `outer_grid_recenter_declined` – see
  [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)'s
  return docs. An axis the caller pinned is never moved; mark a grid
  your own code defaulted with
  [`auto_grid()`](https://gillescolling.com/tulpa/reference/auto_grid.md)
  to keep the recenter live on it (gcol33/tulpa#293).

- spec:

  Optional `tulpa_temporal` or `tulpa_spatial` spec object (output of
  [`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
  [`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md),
  [`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md),
  [`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md),
  [`spatial_bym2()`](https://gillescolling.com/tulpa/reference/spatial_bym2.md),
  etc.). When supplied alongside `data`, the `prior` list is built
  automatically via
  [`prior_from_spec()`](https://gillescolling.com/tulpa/reference/prior_from_spec.md)
  – pass either `prior` or `spec`, not both.

- data:

  Data frame used to validate `spec` and resolve time/group/site
  indices. Required when `spec` is supplied.

- re_idx:

  Optional 1-based RE group index per obs (defaults to no RE).

- n_re_groups:

  RE group count (default 0).

- sigma_re:

  RE standard deviation (default 1).

- family:

  `"binomial"`, `"poisson"`, `"neg_binomial_2"`, etc.

- phi:

  Dispersion (negbin/gamma).

- likelihood:

  Optional model-supplied likelihood, replacing the built-in `family`.
  Pass an external pointer to a `tulpa::NestedLikelihood` (built in a
  model package's own C++ from a `LikelihoodSpec`); the inner Laplace
  solve then reads the per-observation score, Fisher weight, and
  log-likelihood from that spec instead of `family`, so `family`/`phi`
  are ignored. Used by model packages to fit a custom response without
  adding a family to tulpa – for example tulpaObs threads its
  marginalized single-season occupancy likelihood (a scaled Bernoulli,
  with the latent occupancy state integrated out) through this.
  Multi-block `prior` only. Default `NULL` (use `family`).

- control:

  Optional list of perf/numerical tuning knobs (statistical arguments
  stay top-level), following the `control` convention of
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md).
  Recognised elements (defaults in parentheses):

  - `max_iter` (`50L`), `tol` (`1e-6`) – inner Newton iteration budget
    and tolerance.

  - `n_threads` (`1L`) – inner-loop OpenMP threads.

  - `x_init` (`NULL`) – warm-start for the first grid point's inner
    solve.

  - `keep_grid_hessians` (`FALSE`) – when `TRUE`, retain per-grid-point
    fixed-effects marginal Hessian \\H\_\beta\\ and mode \\\hat{\beta}\\
    on the return list as `$grid_hessians` (list of dense \\p\times p\\
    matrices) and `$grid_modes` (list of length-\\p\\ vectors). Used
    downstream by simplified-Laplace (SLA) callers to assemble
    skew-aware marginals – see the cumulant pooling in
    [`rubins_pool()`](https://gillescolling.com/tulpa/reference/rubins_pool.md).

  - `diagnose_k` (`TRUE`), `k_samples` (`200L`) – compute the outer
    Pareto-\\\hat{k}\\ accuracy diagnostic (`$pareto_k`) by importance
    sampling the hyperparameter posterior against the Gaussian proposal
    fitted to the grid, drawing `k_samples` extra inner-marginal
    evaluations. Computed for a single-block, single positive-scale-axis
    grid; left `NA` (with the grid's quadrature ESS as the fallback
    diagnostic) for multi-block, multi-axis, or bounded-parameter grids.
    See
    [`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md).

  - `diagnose_skew` (`TRUE`), `skew_idx` (`NULL`) – compute the
    inner-Laplace skewness diagnostic (`$inner_skew`, gamma_3, Rue
    Martino & Chopin 2009 Sec 3.2.3) at the fitted MAP grid cell: one
    extra Newton solve, scoring the `p` fixed-effects latent indices by
    default (pass `skew_idx`, 1-based latent indices, to score
    additional ones, e.g. specific spatial units – the full latent field
    is not scored by default since it costs one linear solve per index).
    This is the complementary layer to `diagnose_k`: the outer
    diagnostic scores the hyperparameter-grid integration around a FIXED
    inner Laplace, this scores whether that inner Gaussian approximation
    is itself a good fit to the latent-field conditional posterior. See
    [`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
    for the combined whole-fit verdict.

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
    `gamma_3`, about the centre `gamma_1 + gamma_3 / 2` that Rue,
    Martino & Chopin (2009) eq. (22) implies, from
    [`summary()`](https://rdrr.io/r/base/summary.html) /
    [`confint()`](https://rdrr.io/r/stats/confint.html) wherever the
    combined inner band says the leading-order expansion is in its
    regime, and the Gaussian quantiles everywhere else. It is
    post-processing on the reported quantiles: draws, modes and weights
    are untouched, so a fit run with it off is bit for bit the fit it
    was before. A coefficient whose location term could not be formed
    declines rather than reading it as zero. The band that bounded the
    relocation itself (`centre_unreliable`, gcol33/tulpa#362) is off
    (`Inf`): scored over seven fixtures with an exact reference, every
    finite cutoff declines the coefficients the correction helps most,
    because a large centre carrying a small `gamma_3` is uniformly weak
    correlation rather than an expansion out of its regime
    (gcol33/tulpa#376). `$skew_correction` records the per-coefficient
    `gamma_3`, `gamma_1` and the centre they form, the band, the inner
    importance k-hat, the combined band, the eligibility and the reason
    behind it; the `skew_applied` attribute on
    [`summary()`](https://rdrr.io/r/base/summary.html) /
    [`confint()`](https://rdrr.io/r/stats/confint.html) records what was
    actually used at the requested level. RMC fit a skew normal here
    instead; the series correction is the same-order alternative, and
    unlike a skew normal its skewness does not saturate inside the band
    it is applied on.

    MEASURED (gcol33/tulpa#346, gcol33/tulpa#354). Against exact
    quadrature quantiles of rare-event binomial-logit posteriors it cuts
    total absolute endpoint error 69.2%, improving both endpoints in
    every case. Scored over the WHOLE marginal – paired CRPS against the
    exact posterior in a 400-replicate prior-predictive experiment – it
    reads -0.01643 against the uncorrected Laplace at t = -1.89,
    essentially all of the -0.01662 the exact posterior itself achieves,
    and its PIT re-enters the simultaneous SBC band. Applied about the
    Laplace mode instead of about `gamma_1 + gamma_3 / 2` the same
    reshaping scored +0.00775 at t = +3.54, a net loss; that centre is
    what gcol33/tulpa#354 supplied.

    IT IS ON BY DEFAULT (gcol33/tulpa#364), so
    [`summary()`](https://rdrr.io/r/base/summary.html) /
    [`confint()`](https://rdrr.io/r/stats/confint.html) on a
    nested-Laplace fit report the corrected quantiles wherever the
    combined inner band admits the coefficient; `skew_correct = FALSE`
    restores the uncorrected report exactly. Scored against the mixture
    read a correction-off fit gives, the flip is t = -1.895 on the
    rare-event intercept and -3.765 / -3.201 on the small-group
    Bernoulli design. Across twelve model classes read off one solve per
    seed, pooled 95% coverage moves 0.9510 -\> 0.9542 at a standard
    error of 0.0070, with every class inside the acceptance the shipped
    gates use. A fit the correction cannot help – a coupled one, whose
    location term is unreachable – reports what it reported before, to
    the bit.

  - `subspace_debias` (`FALSE`) – correct only the latent directions the
    inner-layer diagnostics flagged, by exact Metropolis, and leave the
    rest at their Gaussian conditional (gcol33/tulpa#304, extended to
    this backend by gcol33/tulpa#306). `TRUE` takes every default; a
    list overrides `band` (the inner-reliability floor a coordinate is
    selected at, default `"ok"`), `idx` (pin the corrected set
    explicitly, skipping the selector), `closure` / `closure_max` (grow
    the set by strongly coupled precision-graph neighbours – declined on
    this backend, which retains no joint precision), the sampler budget
    `n_iter` / `warmup` / `thin`, and `n_draws`. The selector reads the
    per-index bands `diagnose_skew` already attached, so it costs no
    extra solve; the correction itself re-runs the settled grid once
    with the sampler on, and the fit then reports `$draws` – the
    per-cell Metropolis sample for the selected coordinates, the rest
    from the Gaussian conditional given them – instead of the
    Gaussian-mixture moments. An EMPTY selection leaves the fit
    bit-for-bit identical to the plain path. `$subspace_debias` records
    what was selected, the bands it was read from, and the per-cell
    acceptance rate. Requesting it turns `keep_grid_hessians` on, since
    the recombination reads exactly those per-cell pieces.

  - `cila` (`FALSE`) – corrected integrated Laplace, the second
    inner-layer debias (gcol33/tulpa#351, after Lai, Margossian and
    Sheldon, arXiv:2605.20345; wired to this backend by
    gcol33/tulpa#368). Where `subspace_debias` selects coordinates and
    runs exact Metropolis on them, this selects nothing: at every outer
    cell it draws `n_points` points from the whole inner Gaussian,
    weights each by the exact joint density it came from, and reports
    the weighted particles. `TRUE` takes the defaults; a list overrides
    `n_points` (`1024L`), `variant` (`"qmc"`, a Sobol net; `"is"` for
    iid draws, `"rqmc"` for the net under `n_shift` random shifts),
    `n_shift` (`8L`), `n_draws` and `seed`. Below 512 points a cell's
    particle set is too coarse to be a marginal at all and the request
    is refused. The corrected per-cell masses become the fit's own
    `weights` / `log_marginal` and `weights_source` reports `"cila"`;
    the pre-correction pair is kept as `$cila$laplace`. A cell whose
    inner solve factorized sparsely draws through the CHOLMOD factor's
    own triangular and permutation solves; an LDL' factor has no square
    root to draw with and is declined with `"sparse_factor_not_ll"`.

  - `auto_recenter` (`TRUE`) – outer-grid placement policy
    (gcol33/tulpa#289 / \#290 / \#361). `TRUE` re-centres the movable
    default axes on the posterior mode and refits when the grid either
    RAILS (an axis's own marginal is maximal at one of its own
    endpoints) or does not RESOLVE its posterior (an axis's node spacing
    exceeds 2 posterior SDs in its own coordinate). Both tests read the
    weights the fit already stored, so a grid that brackets and resolves
    its mode costs nothing beyond them; when the pass does fire it is a
    second full grid solve plus a finite-difference mode/Hessian
    stencil. A recentred axis is `mode +/- 2.5 sd` over 5 nodes, a cell
    width of 1.25 posterior SDs by construction, against a census median
    of 3.9 on the fixed spans.

    Measured over 200 fixed-truth seeds on each of six configurations
    (icar chain / icar lattice / rw1 / bym2 / iid / nngp), the default
    moves mean \|coverage - nominal\| from 0.043 to 0.030 at the 95%
    level, 0.171 to 0.084 at 80% and 0.243 to 0.129 at 50% against the
    rail-only policy, at 0.63 times the 95% interval width and 0.76
    times the median bias, for 1.71 times the wall clock.

    Three other values. `"rail"` is the rail test alone, which is what
    `TRUE` meant before the sizing measurement settled the default.
    `FALSE` integrates over the grid exactly as given, whatever it is,
    and records
    `outer_grid_recenter_declined = "auto_recenter_disabled"`.
    `"always"` re-centres every movable default axis whatever the fit
    did, at 2.04 times the wall clock; it agrees with the default seed
    for seed on five of the six measured configurations and differs on
    the one whose default axes already resolve their posterior, where it
    takes 50% coverage to 0.135 against a nominal 0.5.

    Which families carry movable axes is `.NL_REGISTRY_AXIS_FIELD`:
    icar, rw1, rw2, iid, bym2, nngp, hsgp and spde. car_proper, ar1 and
    hsgp_mo each carry a correlation axis with no guessable coordinate,
    and mcar / miid / tgmrf hold their axes in one matrix field; a fit
    of any of them records which through `outer_grid_recenter_declined`
    rather than passing in silence. The per-axis policy names are the
    standalone registry path only –
    [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)
    and
    [`fit_st_nested()`](https://gillescolling.com/tulpa/reference/fit_st_nested.md)
    refuse them rather than accept them and ignore them.

  - `max_grid_cells` (`2048L`) – cell-count ceiling on a multi-block
    outer grid, refused with an error above it. Each cell is one inner
    Newton solve, so the default catches a per-block grid that
    multiplied out to a run nobody asked for; a deliberate converged
    tensor reference grid (4 axes at 7 levels is 2401 cells) raises it
    here.

## Value

A list with:

- `theta_grid`: matrix or vector of grid hyperparameter values.

- `log_marginal`: log p(y, mode \| theta_k) at each grid point.

- `weights`: integration weights normalising to sum 1.

- `theta_mean`, `theta_sd`: posterior moments per hyperparameter.

- `n_iter`: inner Newton iterations per grid point.

- `modes`: matrix `[n_grid x n_x]` of inner modes, when stored.

- `pareto_k`, `pareto_k_is_ess`: outer Pareto-\\\hat{k}\\ and its
  importance-sampling ESS (`NA` when not computed for the grid; see
  `control$diagnose_k`).

- `inner_skew`, `inner_skew_idx`, `inner_skew_dropped`: the
  inner-Laplace skewness diagnostic (gamma_3) at each scored latent
  index and its 1-based index, plus a count of (index, observation)
  contributions dropped for a non-finite third derivative (see
  `control$diagnose_skew`).

- `timing`: named numeric of wall-clock seconds (`total`, `setup`,
  `grid`, `postproc`, `diagnostics`); the `grid` phase is the inner
  Laplace pass that scales with grid size. Surfaced one-line in `print`.

- `prior`: echoed input.

## References

Rue, Martino & Chopin (2009). Approximate Bayesian inference for latent
Gaussian models by using integrated nested Laplace approximations.
*JRSS-B* 71(2):319-392.

## Examples

``` r
# \donttest{
set.seed(1)
S <- 30L                                   # spatial units arranged in a chain
nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
nn <- lengths(nb)
field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))   # smooth spatial field
idx <- rep(seq_len(S), each = 6L); n <- length(idx); x <- rnorm(n)
y <- rbinom(n, 1L, plogis(-0.2 + 0.6 * x + field[idx]))
prior <- list(type = "icar", n_spatial_units = S, spatial_idx = idx,
              adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
              n_neighbors = nn, tau_grid = c(0.5, 1, 2, 4, 8))
fit <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
                            family = "binomial")
fit$theta_mean        # marginalized ICAR precision
#> [1] 5.742009
# }
```
