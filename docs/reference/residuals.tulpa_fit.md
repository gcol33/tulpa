# Residuals from a tulpa fit

Residuals from the fitted mean of
[`fitted.tulpa_fit()`](https://gillescolling.com/tulpa/reference/fitted.tulpa_fit.md):
`"response"` is `y - E[y | eta]` on the response scale (trial-scaled for
binomial, offset included); `"pearson"` additionally scales by the
family standard deviation `sqrt(Var(y | eta))` at the same linear
predictor. On a zero-inflated fit both the mean and the variance are the
mixture's.

`effects = "conditional"` (default) reads them against the
posterior-mean linear predictor with every fitted component, random
effects and latent fields included: these are the residuals a structure
diagnostic
([`moran_i()`](https://gillescolling.com/tulpa/reference/moran_i.md),
[`tulpa_variogram()`](https://gillescolling.com/tulpa/reference/tulpa_variogram.md),
[`durbin_watson()`](https://gillescolling.com/tulpa/reference/durbin_watson.md),
[`check_model()`](https://gillescolling.com/tulpa/reference/check_model.md))
has to see, because a spatial or temporal pattern the model's field
absorbed is gone from them and one it left behind is not.
`effects = "population"` holds random effects and fields at zero, so the
residuals carry everything the fixed effects do not explain, including
what the field fitted.

## Usage

``` r
# S3 method for class 'tulpa_fit'
residuals(
  object,
  type = c("pearson", "response"),
  effects = c("conditional", "population"),
  ...
)
```

## Arguments

- object:

  A `tulpa_fit` object carrying `$y` and `$model_matrix`.

- type:

  `"pearson"` (default) or `"response"`.

- effects:

  `"conditional"` (default) or `"population"`.

- ...:

  Ignored.

## Value

Numeric vector of length `nobs(object)`.
