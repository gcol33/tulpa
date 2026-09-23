# Rebuild axis specs from an already-assembled outer grid

Rebuilds each axis's declared spec (support, log/linear coordinate,
prior) from a `theta_grid`'s columns, keyed off column names – the same
metadata the multi-block joint driver recovers rather than carrying a
second description of the axes it already assembled. A column holding
one value across every cell is treated as a fixed setting rather than an
axis (e.g. a fixed NB-size node) and dropped. Intended for a caller that
needs an axis's declared span, coordinate or prior off a settled grid
without restating tulpa's own hyperprior declarations alongside it.

## Usage

``` r
tulpa_joint_axis_specs_from_grid(
  theta_grid,
  copy_slab = "exponential",
  folded_axes = NULL,
  logchol = .hp_logchol_designs(theta_grid)
)
```

## Arguments

- theta_grid:

  A named `[n_cells x n_axes]` matrix (see
  [`tulpa_theta_matrix()`](https://gillescolling.com/tulpa/reference/tulpa_theta_matrix.md)).

- copy_slab:

  `"exponential"` or `"flat"`; the copy-scale axis's continuum measure
  (see
  [`tulpa_hyper_check_copy_slab()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_check_copy_slab.md)).

- folded_axes:

  Optional names of axes folded onto `[0, Inf)`.

- logchol:

  Free-covariance block design(s) as returned by
  `.hp_logchol_designs()`; defaults to the design `theta_grid` itself
  declares.

## Value

A named list of per-axis spec lists, or `NULL` if `theta_grid` has no
axis names.

## See also

[`tulpa_theta_matrix()`](https://gillescolling.com/tulpa/reference/tulpa_theta_matrix.md),
[`tulpa_grid_log_quad()`](https://gillescolling.com/tulpa/reference/tulpa_grid_log_quad.md),
[`tulpa_hyper_grid_supports()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_grid_supports.md)
