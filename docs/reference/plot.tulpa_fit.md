# Plot fixed-effect posteriors

Plot fixed-effect posteriors

## Usage

``` r
# S3 method for class 'tulpa_fit'
plot(x, type = c("density", "trace", "pairs", "smooth"), term = NULL, ...)
```

## Arguments

- x:

  A `tulpa_fit` object.

- type:

  One of `"density"`, `"trace"`, `"pairs"`, `"smooth"`. The Laplace tier
  has no draws, so it always shows the Gaussian densities of the fixed
  effects. `"smooth"` draws the fitted curve of each `s(...)` term and
  requires a fit carrying one.

- term:

  For `type = "smooth"`, which smoother to draw: index or covariate
  name. `NULL` (default) draws every one. Ignored by the other types.

- ...:

  Passed to plotting functions.

## Value

The input `x`, returned invisibly. Called for the side effect of
producing base-graphics plots of the fixed-effect posteriors.
