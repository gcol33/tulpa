# Credible intervals for the fixed effects

Credible intervals for the fixed effects

## Usage

``` r
# S3 method for class 'tulpa_fit'
confint(object, parm = NULL, level = 0.95, ...)
```

## Arguments

- object:

  A `tulpa_fit` object.

- parm:

  Parameter names or indices (default: all fixed effects).

- level:

  Interval level (default 0.95).

- ...:

  Ignored.

## Value

Matrix with lower and upper columns. A nested-Laplace fit carries
`interval_source` / `interval_declined` (which read produced the bounds
– by default the grid's Gaussian-mixture CDF), `retained_mass` (the
share of the grid weight the bounds are conditional on), and
`skew_applied`, one logical per reported coefficient saying whether its
bounds are the inner-Laplace skew-corrected quantiles or not. See
[`summary.tulpa_fit()`](https://gillescolling.com/tulpa/reference/summary.tulpa_fit.md).
