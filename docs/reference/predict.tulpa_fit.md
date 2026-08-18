# Predict at new covariate values (population level)

Prediction of the linear predictor at `newdata`, on the link or response
scale. The fixed-effect part is `X beta` with credible bounds from the
fixed-effect covariance ([`vcov()`](https://rdrr.io/r/stats/vcov.html)).
For a fit carrying a continuous spatial field, the posterior-mean field
is interpolated (kriged) to the `newdata` coordinates and added to the
linear predictor by default, so
[`predict()`](https://rdrr.io/r/stats/predict.html) gives the
conditional (location-specific) prediction. Three continuous field
families are supported: an SPDE Matern field
([`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md)),
projected through the mesh; a Hilbert-space GP field
(`spatial_gp(approx = "hsgp")`), where the Laplacian basis is
re-evaluated at the new coordinates (with the training centring /
boundary); and a GP / NNGP field
([`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)),
interpolated by the NNGP conditional mean at each new location's nearest
training locations. The HSGP and GP/NNGP fields are marginalised over
the hyperparameter grid (not plugged in at the posterior mean). Ordinary
random effects are held at zero (population level); add group effects
from [`ranef()`](https://gillescolling.com/tulpa/reference/ranef.md)
when needed.

## Usage

``` r
# S3 method for class 'tulpa_fit'
predict(
  object,
  newdata = NULL,
  type = c("link", "response"),
  se.fit = FALSE,
  level = 0.95,
  include_field = TRUE,
  ...
)
```

## Arguments

- object:

  A `tulpa_fit` object.

- newdata:

  Data frame of covariates (and, for an SPDE fit, the coordinate columns
  named in the spec's coordinate formula). If `NULL`, predicts at the
  training design (requires `$model_matrix`).

- type:

  `"link"` (linear predictor) or `"response"` (mean scale). For a
  binomial fit the `"response"` scale here is the per-trial success
  probability `g^{-1}(eta)` (there is no `n_trials` at `newdata`); this
  differs from [`fitted()`](https://rdrr.io/r/stats/fitted.values.html),
  which returns the trial-scaled expected count at the training design.

- se.fit:

  If `TRUE`, also return the link-scale standard error and credible
  bounds. With an included SPDE field the SE propagates the joint
  (fixed-effect, field) posterior precision at the fitted
  hyperparameters – including the cross term – conditional on
  `(range, sigma)` (a nested fit's hyperparameter-grid spread is not
  propagated, so the bound is mildly optimistic when that posterior is
  wide). Integer-nu, no-RE SPDE fits only; other layouts decline with an
  explanation.

- level:

  Credible-interval level (default 0.95).

- include_field:

  For a continuous-spatial fit (SPDE, HSGP, or GP/NNGP), add the kriged
  field to the prediction (default `TRUE`). `FALSE` gives the
  fixed-effect (population) prediction. Ignored for fits with no
  continuous field. For an HSGP or GP/NNGP fit the field is added to the
  point prediction but its uncertainty is not yet propagated into
  `se.fit` (the interval reflects the fixed-effect covariance only).

- ...:

  Ignored.

## Value

If `se.fit = FALSE`, a numeric vector. If `se.fit = TRUE`, a data frame
with `fit`, `se.fit` (link scale), `lower`, `upper` on the requested
scale.
