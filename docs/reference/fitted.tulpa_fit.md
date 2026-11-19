# Fitted values

In-sample mean response `E[y] = g^{-1}(eta)` (trial-scaled for binomial)
at a point linear predictor `eta`. With `effects = "conditional"`
(default) `eta` is the posterior mean of the in-sample linear predictor
with every component the fit estimated – fixed effects, the offset,
formula random effects and any spatial, temporal or varying-coefficient
field – read from the same source
[`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md)
and [`simulate()`](https://rdrr.io/r/stats/simulate.html) draw their
replicates from. With `effects = "population"` the random effects and
latent fields are held at their prior mean of zero and
`eta = X beta + offset` at
[`coef()`](https://rdrr.io/r/stats/coef.html); group-level effects are
in [`ranef()`](https://gillescolling.com/tulpa/reference/ranef.md).
`y - fitted(object)` equals `residuals(object, type = "response")` at
the same `effects`. On a zero-inflated fit (`ziformula`) the mean is the
mixture's, `(1 - pi) E[y | eta]` with `pi = plogis(X_zi beta_zi)`; over
a zero-truncated family that is the hurdle mean.

## Usage

``` r
# S3 method for class 'tulpa_fit'
fitted(object, effects = c("conditional", "population"), ...)
```

## Arguments

- object:

  A `tulpa_fit` object (must carry `$model_matrix`).

- effects:

  `"conditional"` (default) or `"population"`, as described above.

- ...:

  Ignored.

## Value

Numeric vector of fitted mean responses, length `nobs`.
