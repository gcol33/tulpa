# Per-axis integrated support of an outer grid

The natural-scale support of every axis in `specs` that carries one, as
a named list of intervals – the region each axis's outer-grid measure
actually integrates, accounting for refinement slice cells. Intended for
recovering an axis's integrated span from a settled grid, e.g. so a
sampled-hyperparameter prior can be derived from what the outer
integration used rather than restated alongside it.

## Usage

``` r
tulpa_hyper_grid_supports(theta_grid, specs, refining = NULL)
```

## Arguments

- theta_grid:

  A named `[n_cells x n_axes]` matrix (see
  [`tulpa_theta_matrix()`](https://gillescolling.com/tulpa/reference/tulpa_theta_matrix.md)).

- specs:

  Per-axis spec list, e.g. from
  [`tulpa_joint_axis_specs_from_grid()`](https://gillescolling.com/tulpa/reference/tulpa_joint_axis_specs_from_grid.md).

- refining:

  Optional per-cell refinement-slice tag vector (see
  [`tulpa_hyper_slice_home()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_slice_home.md)).

## Value

A named list of natural-scale `c(lo, hi)` intervals, one per axis in
`specs` that carries a declared coordinate, or `NULL` if `theta_grid` or
`specs` is `NULL`.

## See also

[`tulpa_joint_axis_specs_from_grid()`](https://gillescolling.com/tulpa/reference/tulpa_joint_axis_specs_from_grid.md),
[`tulpa_hyper_slice_home()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_slice_home.md)
