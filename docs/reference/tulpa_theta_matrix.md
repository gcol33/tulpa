# A fit's outer grid as a named matrix

Coerces a nested-Laplace fit's `theta_grid` to matrix form with named
axis columns. A single-axis grid is stored on the fit as a bare vector
named by `theta_names`; every downstream reader that keys an axis by
name needs the named matrix form, so this is the one place that coercion
happens. Intended for reconstructing a fit's outer-grid posterior cell
weights (see
[`tulpa_grid_log_quad()`](https://gillescolling.com/tulpa/reference/tulpa_grid_log_quad.md),
[`tulpa_normalise_weights_safe()`](https://gillescolling.com/tulpa/reference/tulpa_normalise_weights_safe.md))
for a fit or path whose driver doesn't already carry `fit$log_quad`.

## Usage

``` r
tulpa_theta_matrix(res)
```

## Arguments

- res:

  A `tulpa_fit` object (or any list carrying `theta_grid` and
  `theta_names`).

## Value

`res$theta_grid` as an `[n_cells x n_axes]` matrix with axis names as
column names, or `NULL` if the fit has no grid.

## See also

[`tulpa_grid_log_quad()`](https://gillescolling.com/tulpa/reference/tulpa_grid_log_quad.md),
[`tulpa_normalise_weights_safe()`](https://gillescolling.com/tulpa/reference/tulpa_normalise_weights_safe.md)
