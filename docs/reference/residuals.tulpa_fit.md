# Residuals from a tulpa fit

Population-level residuals from the fixed-effect fitted mean:
`"response"` is `y - E[y | eta]` on the response scale (trial-scaled for
binomial, offset included); `"pearson"` additionally scales by the
family standard deviation `sqrt(Var(y | eta))` at the fitted linear
predictor. Random effects are held at zero, matching
[`fitted()`](https://rdrr.io/r/stats/fitted.values.html).

## Usage

``` r
# S3 method for class 'tulpa_fit'
residuals(object, type = c("pearson", "response"), ...)
```

## Arguments

- object:

  A `tulpa_fit` object carrying `$y` and `$model_matrix`.

- type:

  `"pearson"` (default) or `"response"`.

- ...:

  Ignored.

## Value

Numeric vector of length `nobs(object)`.
