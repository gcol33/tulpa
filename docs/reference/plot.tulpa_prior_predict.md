# Plot method for tulpa_prior_predict

Density overlay of prior predictive draws. Requires bayesplot; falls
back to base graphics matplot of a subset of draws otherwise.

## Usage

``` r
# S3 method for class 'tulpa_prior_predict'
plot(x, process = 1L, max_draws = 50L, ...)
```

## Arguments

- x:

  A tulpa_prior_predict object

- process:

  Process index or name (multi-process families)

- max_draws:

  Maximum draws to overlay. Default 50.

- ...:

  Passed through.

## Value

A `ggplot` object (via bayesplot) when bayesplot is installed; otherwise
`NULL` invisibly, after drawing a base-graphics overlay. Called for the
density overlay of the prior predictive draws.
