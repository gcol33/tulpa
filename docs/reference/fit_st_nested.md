# Fit an additive spatiotemporal GLM by nested Laplace

Fits `y ~ X beta + u_spatial[s] + v_temporal[t]` with a spatial field –
areal (`icar` / `bym2` / `car_proper`) or continuous (`hsgp` / `nngp`) –
and a temporal field (`rw1` / `rw2` / `ar1`), integrating the spatial
hyperparameter(s), temporal precision, and (for `ar1`) the temporal
autocorrelation over a hyperparameter grid via the
`cpp_nested_laplace_st_*` kernels. The fixed-effect posterior is the
grid-marginalised mixture; the spatial and temporal field posterior
means are the grid-weighted latent modes.

## Usage

``` r
fit_st_nested(
  y,
  X,
  spatial_idx,
  adjacency,
  temporal_idx,
  n_times,
  spatial_type = c("icar", "bym2", "car_proper", "hsgp", "nngp"),
  temporal_type = c("ar1", "rw1", "rw2"),
  family = "binomial",
  n_trials = NULL,
  phi = 1,
  cyclic = FALSE,
  re_idx = NULL,
  n_re_groups = 0L,
  sigma_re = 1,
  hyperprior = c("proper", "flat"),
  control = list(),
  coords = NULL,
  nn = 10L,
  cov_type = 2L,
  hsgp_m = 6L,
  hsgp_c = 1.5
)
```

## Arguments

- y:

  Response vector.

- X:

  Fixed-effects design matrix (`nrow(X) == length(y)`).

- spatial_idx:

  Integer per-observation spatial-unit index (1-based). For
  `spatial_type = "icar"/"bym2"/"car_proper"`, an areal unit in
  `[1, nrow(adjacency)]`; for `"nngp"`, a location in
  `[1, nrow(coords)]`. Ignored for `"hsgp"` (the field is evaluated
  directly at each observation's own coordinates); pass any placeholder
  (e.g. `seq_len(N)`).

- adjacency:

  Spatial adjacency (a symmetric 0/1 matrix or `sparseMatrix`). Required
  for `spatial_type = "icar"/"bym2"/"car_proper"`; ignored (pass `NULL`)
  for `"hsgp"`/`"nngp"`, which take `coords` instead.

- temporal_idx:

  Integer per-observation time index (1-based).

- n_times:

  Number of distinct time points.

- spatial_type:

  `"icar"` (default), `"bym2"`, `"car_proper"`, `"hsgp"` (Hilbert-space
  GP basis), or `"nngp"` (nearest-neighbour GP).

- temporal_type:

  `"ar1"` (default), `"rw1"`, or `"rw2"`.

- family:

  Response family (see
  [`family_names()`](https://gillescolling.com/tulpa/reference/family_names.md)).

- n_trials:

  Binomial denominators, or `NULL` (= 1).

- phi:

  Dispersion passed to the family, held fixed. One convention at every
  door: for `gaussian` / `lognormal` this is the residual VARIANCE (the
  SD is `sqrt(phi)`), for `neg_binomial_2` the size, `gamma` the shape,
  `beta` the precision, `t` the scale; `binomial` and `poisson` ignore
  it. The compiled kernels parameterize the two variance families by the
  residual SD and are handed `sqrt(phi)` at the boundary.

  Defaulted, it conditions at 1 and says so with a warning, since for a
  family that reads a dispersion that is a modelling choice rather than
  a neutral value. `fit$phi_estimated` records whether the value on the
  fit was estimated or conditioned on.

- cyclic:

  Logical; wrap the temporal field (seasonal). Default `FALSE`.

- re_idx, n_re_groups, sigma_re:

  Optional single iid random-intercept term alongside the fields
  (conditioned on `sigma_re`); `n_re_groups = 0` (default) is no RE
  term.

- hyperprior:

  The prior an outer hyperparameter axis carries when the call states no
  density for it, `"proper"` (default) or `"flat"`. `"proper"` folds a
  normalised density on the axis's integration coordinate into
  `log_marginal`: the PC prior `P(sigma > 3) = 0.01` on a standard
  deviation, variance or precision axis, the PC range prior at 0.2 times
  the coordinates' bounding-box diagonal on a range or lengthscale axis,
  a uniform on a bounded axis, and the PC + LKJ prior over a
  free-covariance block. An axis with no sourced density is named in
  `log_hyperprior_declined`. `"flat"` folds no density of the engine's
  own, so such an axis is integrated under its cell measure alone, is
  named in `log_hyperprior_declined` as `"flat_hyperprior"`, and the
  fit's `log_evidence` declines with `"improper_hyperprior"`. A density
  the call states – a `prior_*` argument, a block's `rho_prior`,
  `prior_range` or `prior_sigma`, the copy coefficient's slab, a
  [`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md)
  block's own prior – applies under either choice. The same two choices,
  under the same names, are offered by
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md),
  [`tulpa_eb()`](https://gillescolling.com/tulpa/reference/tulpa_eb.md),
  [`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
  and
  [`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md).

- control:

  A list of numerical / grid knobs: `n_grid_spatial`, `n_grid_temporal`
  (default 4 each), `n_grid_rho` (ar1 only, default 3), `tau_lower` /
  `tau_upper` (icar / car_proper precision grid bounds, default 0.25 /
  16), `sigma_lower` / `sigma_upper` (bym2's spatial SD grid bounds in
  place of `tau_lower` / `tau_upper`, default 0.1 / 3 – `n_grid_spatial`
  sizes this axis too; its mixing-weight axis `rho_spatial` is always
  the fixed default node set and is not a `control` knob), `rho_lower` /
  `rho_upper` (ar1 grid, default 0.1 / 0.9), `max_iter`, `tol`,
  `n_threads`, `auto_recenter` (default `TRUE`; `FALSE` holds the grid
  exactly as specified – the per-axis policy names
  [`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
  takes are refused here with an error, since this driver recentres on
  the grid's collapsed-edge regime rather than on a per-axis rail;
  declines outright for `spatial_type = "bym2"`, whose (sigma, rho)
  spatial axes this recenter has no transform for yet, and for
  `"hsgp"`/`"nngp"`, same reason), `rho_spatial` (the proper-CAR mixing
  value the `car_proper` axis is held at, default
  `.NL_ST_GRID$rho_spatial`; unrelated to bym2's own integrated
  `rho_spatial` grid axis) and `within_cell` (`"box_uniform"` /
  `"chord"`, the within-cell construction the reported per-axis
  intervals are read with; defaults to `.NL_DIAG$within_cell`, as on
  every other nested door).

  For `spatial_type = "hsgp"`/`"nngp"`, the spatial axes are the field
  variance (`sigma2`) paired with the lengthscale (`lengthscale`) or
  NNGP range (`phi_gp`), read off the same shared default bounds the
  single-field
  [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
  path uses (`gp_var` / `gp_lengthscale`, `R/settings.R`) – there is no
  separate `sigma2_lower`/`upper` knob here, only `n_grid_spatial`,
  which sizes the pair as it does for the areal families.

  The `(tau_lower, tau_upper)` span (and, for `ar1`,
  `(rho_lower, rho_upper)`) is a starting axis, not a hard ceiling: when
  the fitted precision (or, for `ar1`, autocorrelation) posterior mode
  rails a boundary node (that axis's own marginal is maximal there, or
  the whole grid collapsed onto it:
  `pareto_k_regime = "collapsed_edge"`, see below), the driver fits a
  mode-Hessian via a derivative-free
  [`optim()`](https://rdrr.io/r/stats/optim.html) over the grid and
  refits a grid re-centred on it (one attempt).

  A grid knob PINS the axes it shapes, and a pin always wins – but
  pinning is decided by value, not by presence: a knob set to the
  engine's own default, or marked with
  [`auto_grid()`](https://gillescolling.com/tulpa/reference/auto_grid.md),
  expresses no preference and leaves its axes free. That is what lets a
  wrapper package thread its own `n_grid`-style argument through
  `control` without silently disabling the recenter for every fit it
  makes. Pinning is also per axis: `tau_lower` / `tau_upper` hold the
  two precision axes, `n_grid_spatial` / `n_grid_temporal` one each, and
  `n_grid_rho` / `rho_lower` / `rho_upper` the `ar1` autocorrelation
  axis, so pinning one axis leaves the others free to be recentred. A
  pinned axis keeps its nodes exactly and is named in
  `outer_grid_pinned_axes`; with EVERY axis pinned the recenter declines
  outright and `outer_grid_recenter_declined` records which reason
  applied.

- coords:

  Coordinate matrix for a continuous spatial field, required when
  `spatial_type` is `"hsgp"` or `"nngp"` (ignored otherwise). For
  `"hsgp"`, an `N x 2` matrix (one row per observation, matching
  `spatial_gp(approx = "hsgp")`'s basis convention – the basis is built
  by `cpp_hsgp_basis_2d()`, 2D only). For `"nngp"`, an `n_spatial x d`
  matrix of unique locations that `spatial_idx` indexes into (any `d`,
  matching
  [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)'s
  NNGP convention).

- nn:

  Number of nearest neighbours per location, `spatial_type = "nngp"`
  only. Default 10 (clamped to `nrow(coords) - 1`).

- cov_type:

  Integer NNGP covariance code (`spatial_type = "nngp"` only): 0 =
  exponential, 1 = Matern 3/2, 2 = Matern 5/2 (default), 3 = Gaussian.

- hsgp_m, hsgp_c:

  Hilbert-space GP basis size (per dimension, default 6) and boundary
  factor (default 1.5), `spatial_type = "hsgp"` only – the same
  parameterisation and defaults as `spatial_gp(approx = "hsgp")`.

## Value

A `tulpa_fit` (subclass `tulpa_nested_laplace`) carrying the
fixed-effect posterior (`draws` via the grid mixture),
`spatial_effects`, `temporal_effects`, `log_marginal`, `weights`,
`theta_grid` over `(tau_spatial, tau_temporal, rho)` for an areal
spatial field or `(sigma2, lengthscale, tau_temporal, rho)` /
`(sigma2, phi_gp, tau_temporal, rho)` for `"hsgp"` / `"nngp"`, and
`family` / `n_trials` / `phi` (the R-level convention), which is what
[`fitted()`](https://rdrr.io/r/stats/fitted.values.html),
[`residuals()`](https://rdrr.io/r/stats/residuals.html),
[`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md)
and [`simulate()`](https://rdrr.io/r/stats/simulate.html) read the
response family from. Also carries `pareto_k_regime` (`"spread"` /
`"collapsed_interior"` / `"collapsed_edge"`, see
[`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)'s
return docs for the definition) and `outer_grid_placement` (`"fixed"` or
`"auto_recentered"`) plus, on a `"fixed"` placement,
`outer_grid_recenter_declined` (`"grid_knobs_overridden"` /
`"grid_not_collapsed"` / `"no_usable_curvature"` / `"refit_failed"` /
`"sd_ceiling_unresolved"` / `"sd_floor_unresolved"`). A recentred fit
also carries `outer_grid_pinned_axes`, the axes whose knobs were pinned
and whose nodes were therefore kept, and `outer_grid_recenter_sd_clamp`
/ `_sd_raw` / `_sd_used` – per moved axis, which mode-SD bound the
placement hit, the SD the stencil measured, and the SD the axis was laid
from. A bound-decline is PER AXIS here: the axes the mode-find did
resolve are still re-placed, and `outer_grid_recenter_sd_declined` names
the ones that kept their incoming nodes and on which bound, so a
partially re-placed grid is not read as a fully re-placed one. With
every free axis declined the pass reports the grid as the fixed one it
still is.

## See also

[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) (front
door),
[`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
(single field).

## Examples

``` r
# \donttest{
set.seed(1)
n_s <- 16L; n_t <- 8L; N <- 400L
adj <- matrix(0, n_s, n_s)
for (i in 1:(n_s - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1
s <- sample(n_s, N, TRUE); tt <- sample(n_t, N, TRUE)
us <- as.numeric(scale(cumsum(rnorm(n_s)))); vt <- as.numeric(scale(cumsum(rnorm(n_t))))
x <- rnorm(N)
y <- rbinom(N, 1, plogis(0.2 + 0.5 * x + 0.7 * us[s] + 0.6 * vt[tt]))
fit <- fit_st_nested(y, cbind(1, x), s, adj, tt, n_t, family = "binomial")
# }
```
