# Build the summary of a tulpa fit around a coefficient table

The constructor
[`summary.tulpa_fit()`](https://gillescolling.com/tulpa/reference/summary.tulpa_fit.md)
returns through: everything but the fixed-effect table is read off the
fit the same way for every model. A model package whose fit stores its
coefficients in its own layout (several arms, no draws) computes that
table and hands it here, so its
[`summary()`](https://rdrr.io/r/base/summary.html) is the same object,
printed and read the same way.

## Usage

``` r
tulpa_fit_summary(object, coefficients, level = 0.95)
```

## Arguments

- object:

  A `tulpa_fit`.

- coefficients:

  Data frame, one row per coefficient (row names are the coefficient
  names), with columns `estimate`, `std.error` and the two interval
  bounds, in that order.

- level:

  Coverage of the interval columns.

## Value

A `summary.tulpa_fit` object, as described on
[`summary.tulpa_fit()`](https://gillescolling.com/tulpa/reference/summary.tulpa_fit.md).

## See also

[`summary.tulpa_fit()`](https://gillescolling.com/tulpa/reference/summary.tulpa_fit.md)

## Examples

``` r
set.seed(1)
df <- data.frame(x = rnorm(60))
df$y <- rpois(60, exp(0.5 + 0.3 * df$x))
fit <- tulpa(y ~ x, data = df, family = "poisson", mode = "laplace")
tab <- coef(summary(fit))
identical(coef(tulpa_fit_summary(fit, tab)), tab)
```
