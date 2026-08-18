# Describe one outer-grid hyperparameter axis

Builds a single axis spec for
[`tulpa_hyper_grid()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_grid.md).
Carries the candidate values, the optional log-prior, and the metadata
(log-scale, bounds, refinable flag) that the generic refinement /
consistency passes need.

## Usage

``` r
hyper_axis_spec(
  name,
  grid,
  log_prior = NULL,
  log_scale = FALSE,
  bounds = NULL,
  refinable = FALSE
)
```

## Arguments

- name:

  Character. Axis label, used as the column name of the grid matrix and
  in posterior summaries.

- grid:

  Numeric vector of length \>= 1. The per-axis candidate values (the
  outer integration nodes on this axis). The full outer grid is the
  Cartesian product across axes.

- log_prior:

  Optional `function(x)` returning the scalar log prior density at axis
  value `x`. `NULL` (default) is a flat / improper prior (zero log-prior
  contribution).

- log_scale:

  Logical. Does the axis live naturally on a log scale (`sigma`, `tau`,
  `lengthscale`, ...)? Drives geometric vs arithmetic spacing in
  refinement and log-axis quantile fits. Default `FALSE`.

- bounds:

  Numeric vector of length 2 giving the natural support `(lower, upper)`
  of the axis, e.g. `c(0, Inf)` for `sigma`, `c(0, 1)` for a BYM2 mixing
  coefficient. `NULL` (default) is unbounded.

- refinable:

  Logical. When `TRUE`, the axis participates in the adaptive-grid and
  var-of-means consistency passes (when those are enabled at the driver
  level). Spatial prior amplitudes (`sigma`) are typically left at the
  user-specified grid (`refinable = FALSE`); the copy coefficient
  `alpha` and per-arm dispersion `phi` typically opt in. Default
  `FALSE`.

## Value

An object of class `tulpa_hyper_axis_spec` (a validated list with the
six fields above).

## See also

[`tulpa_hyper_grid()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_grid.md).
