# Empirical semivariogram of residuals

Computes the empirical semivariogram in distance bins for visual
assessment of remaining spatial structure in residuals.

## Usage

``` r
tulpa_variogram(
  object,
  coords,
  n_bins = 15L,
  max_dist = NULL,
  resid_type = "pearson"
)

# S3 method for class 'tulpa_variogram'
plot(x, ...)

# S3 method for class 'tulpa_variogram'
print(x, ...)
```

## Arguments

- object:

  A fitted model, or a numeric vector of residuals

- coords:

  N x 2 coordinate matrix (required)

- n_bins:

  Number of distance bins (default 15)

- max_dist:

  Maximum distance (default: half the maximum pairwise distance)

- resid_type:

  Residual type if extracting from model (default `"pearson"`)

- x:

  A `tulpa_variogram` table.

- ...:

  Passed to
  [`graphics::plot()`](https://rdrr.io/r/graphics/plot.default.html) by
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html) and to
  [`print()`](https://rdrr.io/r/base/print.html) by
  [`print()`](https://rdrr.io/r/base/print.html).

## Value

A `tulpa_variogram` data.frame with columns `dist`, `gamma`, `n_pairs`

[`plot()`](https://rdrr.io/r/graphics/plot.default.html) draws the
semivariance against distance, point size scaled by the pair count, and
returns `x` invisibly.

## Examples

``` r
set.seed(1)
coords <- cbind(runif(60), runif(60))
tulpa_variogram(rnorm(60), coords, n_bins = 5)
```
