# Normalise outer-grid log-marginals to posterior cell weights

The single weight normaliser for every outer hyperparameter grid:
softmax of `lm` (optionally plus a per-cell log quadrature weight) into
weights summing to 1. Non-finite nodes (an inner Newton diverging in a
grid corner) are dropped from the max-shift and zeroed before
renormalising over the finite cells; an all-non-finite grid returns
all-`NA` with a warning rather than `NaN`. Intended for reconstructing a
fit's outer-grid posterior weights (with
[`tulpa_theta_matrix()`](https://gillescolling.com/tulpa/reference/tulpa_theta_matrix.md)
and
[`tulpa_grid_log_quad()`](https://gillescolling.com/tulpa/reference/tulpa_grid_log_quad.md))
when the fit doesn't already carry `fit$weights`.

## Usage

``` r
tulpa_normalise_weights_safe(lm, what = "grids / data", log_quad = NULL)
```

## Arguments

- lm:

  Numeric vector of per-cell log-marginal-likelihood values.

- what:

  Label for the grid, used only in the degenerate-case warning.

- log_quad:

  Optional per-cell log quadrature weight (prior mass) of the outer
  grid, e.g. from
  [`tulpa_grid_log_quad()`](https://gillescolling.com/tulpa/reference/tulpa_grid_log_quad.md);
  `NULL` normalises the likelihood alone.

## Value

Numeric vector of posterior cell weights summing to 1, the same length
as `lm`, or all-`NA` if no cell carries finite mass.

## See also

[`tulpa_theta_matrix()`](https://gillescolling.com/tulpa/reference/tulpa_theta_matrix.md),
[`tulpa_grid_log_quad()`](https://gillescolling.com/tulpa/reference/tulpa_grid_log_quad.md)
