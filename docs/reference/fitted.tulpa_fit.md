# Fitted values (population level)

In-sample mean response from the fixed effects and the observation
offset (`E[y] = g^{-1}(X beta + offset)`, trial-scaled for binomial).
Random effects are held at their prior mean of zero; group-level effects
are in [`ranef()`](https://gillescolling.com/tulpa/reference/ranef.md).
`y - fitted(object)` equals `residuals(object, type = "response")`.

## Usage

``` r
# S3 method for class 'tulpa_fit'
fitted(object, ...)
```

## Arguments

- object:

  A `tulpa_fit` object (must carry `$model_matrix`).

- ...:

  Ignored.

## Value

Numeric vector of fitted mean responses, length `nobs`.
