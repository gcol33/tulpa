# Read a default nested-Laplace outer-grid axis

Materialises one of tulpa's own default outer-grid axes – the values a
`*_grid` argument (e.g. `sigma_grid`, `range_grid`) takes when a caller
does not supply one. Geometric axes are returned as
`exp(seq(log(lo), log(hi), length.out = n))`; axes declared as explicit
nodes are returned as-is. A default is a decision tulpa makes internally
and can change between releases (gcol33/tulpa#633), so a caller that
needs to know the current default – rather than restate it – should read
it here instead of duplicating tulpa's own grid table.

## Usage

``` r
tulpa_grid_axis(key, n = NULL)
```

## Arguments

- key:

  Name of a default axis, e.g. `"field_sd"`, `"copy_alpha"`,
  `"bym2_rho"`, `"gp_var"`, `"gp_lengthscale"`. Unknown keys error.

- n:

  Optional resolution override: same `lo` / `hi` bounds (and the same
  `prepend` atom, if any) at a different node count. `NULL` uses the
  axis's own declared `n`. Errors for an axis declared as explicit nodes
  (no resolution to vary) or for a data-dependent axis (its shape
  depends on arguments the table does not hold).

## Value

Numeric vector of grid nodes.

## Examples

``` r
tulpa_grid_axis("field_sd")
#> [1] 0.1000000 0.2340347 0.5477226 1.2818610 3.0000000
tulpa_grid_axis("copy_alpha", n = 9)
#>  [1] 0.0000000 0.1000000 0.1529819 0.2340347 0.3580309 0.5477226 0.8379166
#>  [8] 1.2818610 1.9610158 3.0000000
```
