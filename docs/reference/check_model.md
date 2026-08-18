# Diagnostic panel plot

Produces a 2x2 or 1x3 panel of diagnostic plots:

1.  QQ plot of PIT residuals vs Uniform (with KS p-value)

2.  Residuals vs fitted (with lowess smoother)

3.  Dispersion histogram (observed variance vs simulated)

4.  Spatial correlogram via Moran's I (if coords provided)

## Usage

``` r
check_model(object, ...)

# Default S3 method
check_model(object, coords = NULL, nsim = 250L, seed = 123L, ...)
```

## Arguments

- object:

  A fitted model with
  [`simulate()`](https://rdrr.io/r/stats/simulate.html),
  [`fitted()`](https://rdrr.io/r/stats/fitted.values.html),
  [`residuals()`](https://rdrr.io/r/stats/residuals.html)

- ...:

  Passed to methods.

- coords:

  Optional N x 2 coordinate matrix for spatial panel

- nsim:

  Number of simulations (default 250)

- seed:

  Random seed (default 123)

## Value

Invisible list with `ks_p`, `disp_ratio`, `moran` (if spatial)
