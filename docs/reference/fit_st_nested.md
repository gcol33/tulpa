# Fit an additive spatiotemporal GLM by nested Laplace

Fits `y ~ X beta + u_spatial[s] + v_temporal[t]` with an areal spatial
field (`icar` / `bym2` / `car_proper`) and a temporal field (`rw1` /
`rw2` / `ar1`), integrating the spatial precision, temporal precision,
and (for `ar1`) the temporal autocorrelation over a hyperparameter grid
via the `cpp_nested_laplace_st_*` kernels. The fixed-effect posterior is
the grid-marginalised mixture; the spatial and temporal field posterior
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
  spatial_type = c("icar", "bym2", "car_proper"),
  temporal_type = c("ar1", "rw1", "rw2"),
  family = "binomial",
  n_trials = NULL,
  phi = 1,
  cyclic = FALSE,
  re_idx = NULL,
  n_re_groups = 0L,
  sigma_re = 1,
  control = list()
)
```

## Arguments

- y:

  Response vector.

- X:

  Fixed-effects design matrix (`nrow(X) == length(y)`).

- spatial_idx:

  Integer per-observation spatial-unit index (1-based).

- adjacency:

  Spatial adjacency (a symmetric 0/1 matrix or `sparseMatrix`).

- temporal_idx:

  Integer per-observation time index (1-based).

- n_times:

  Number of distinct time points.

- spatial_type:

  `"icar"` (default), `"bym2"`, or `"car_proper"`.

- temporal_type:

  `"ar1"` (default), `"rw1"`, or `"rw2"`.

- family:

  Response family (see
  [`family_names()`](https://gillescolling.com/tulpa/reference/family_names.md)).

- n_trials:

  Binomial denominators, or `NULL` (= 1).

- phi:

  Dispersion passed to the family.

- cyclic:

  Logical; wrap the temporal field (seasonal). Default `FALSE`.

- re_idx, n_re_groups, sigma_re:

  Optional single iid random-intercept term alongside the fields
  (conditioned on `sigma_re`); `n_re_groups = 0` (default) is no RE
  term.

- control:

  A list of numerical / grid knobs: `n_grid_spatial`, `n_grid_temporal`
  (default 4 each), `n_grid_rho` (ar1 only, default 3), `tau_lower` /
  `tau_upper` (precision grid bounds, default 0.25 / 16), `rho_lower` /
  `rho_upper` (ar1 grid, default 0.1 / 0.9), `max_iter`, `tol`,
  `n_threads`, `auto_recenter` (default `TRUE`; `FALSE` holds the grid
  exactly as specified – the per-axis policy names
  [`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
  takes are refused here with an error, since this driver recentres on
  the grid's collapsed-edge regime rather than on a per-axis rail).

  The `(tau_lower, tau_upper)` span (and, for `ar1`,
  `(rho_lower, rho_upper)`) is a starting axis, not a hard ceiling: when
  the fitted precision (or, for `ar1`, autocorrelation) posterior mode
  rails a boundary node (`pareto_k_regime = "collapsed_edge"`, see
  below), the driver fits a mode-Hessian via a derivative-free
  [`optim()`](https://rdrr.io/r/stats/optim.html) over the collapsed
  grid and refits a grid re-centred on it (one attempt).

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

## Value

A `tulpa_fit` (subclass `tulpa_nested_laplace`) carrying the
fixed-effect posterior (`draws` via the grid mixture),
`spatial_effects`, `temporal_effects`, `log_marginal`, `weights`, and
`theta_grid` over `(tau_spatial, tau_temporal, rho)`. Also carries
`pareto_k_regime` (`"spread"` / `"collapsed_interior"` /
`"collapsed_edge"`, see
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
