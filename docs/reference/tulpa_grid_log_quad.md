# Per-cell log quadrature weight of an outer grid

The per-cell log quadrature weight (prior mass) of a nested-Laplace
outer grid: every path that turns a fit's `log_marginal` into posterior
cell weights goes through this one rule, so the prior mass a cell
carries is decided in one place. Rebuilds axis specs from the grid's own
columns when `specs` is not supplied. Intended for reconstructing a
fit's outer-grid posterior weights (with
[`tulpa_theta_matrix()`](https://gillescolling.com/tulpa/reference/tulpa_theta_matrix.md)
and
[`tulpa_normalise_weights_safe()`](https://gillescolling.com/tulpa/reference/tulpa_normalise_weights_safe.md))
when the fit doesn't already carry `fit$log_quad`.

## Usage

``` r
tulpa_grid_log_quad(
  theta_grid,
  specs = NULL,
  copy_slab = "exponential",
  close_domain = TRUE,
  folded_axes = NULL,
  refining = NULL
)
```

## Arguments

- theta_grid:

  A named `[n_cells x n_axes]` matrix (see
  [`tulpa_theta_matrix()`](https://gillescolling.com/tulpa/reference/tulpa_theta_matrix.md)).

- specs:

  Optional pre-built per-axis spec list; `NULL` rebuilds it from
  `theta_grid`'s columns via
  [`tulpa_joint_axis_specs_from_grid()`](https://gillescolling.com/tulpa/reference/tulpa_joint_axis_specs_from_grid.md).

- copy_slab:

  `"exponential"` or `"flat"`; the copy-scale axis's continuum measure
  (see
  [`?tulpa_joint_axis_specs_from_grid`](https://gillescolling.com/tulpa/reference/tulpa_joint_axis_specs_from_grid.md)).

- close_domain:

  Whether an unbounded axis's outer cells are closed at the grid's own
  edge rather than left open to infinity.

- folded_axes:

  Optional names of axes folded onto `[0, Inf)` (e.g. a correlation axis
  reflected at 0).

- refining:

  Optional refinement-slice tag vector (see
  [`tulpa_hyper_slice_home()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_slice_home.md));
  when supplied, specs are rebuilt from the grid's base (non-slice)
  cells only.

## Value

Numeric vector of per-cell log quadrature weights, length
`nrow(theta_grid)`, or `NULL` if `theta_grid` has no axis names.

## See also

[`tulpa_theta_matrix()`](https://gillescolling.com/tulpa/reference/tulpa_theta_matrix.md),
[`tulpa_normalise_weights_safe()`](https://gillescolling.com/tulpa/reference/tulpa_normalise_weights_safe.md),
[`tulpa_joint_axis_specs_from_grid()`](https://gillescolling.com/tulpa/reference/tulpa_joint_axis_specs_from_grid.md)
