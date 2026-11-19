# Predict at new covariate values

Prediction of the linear predictor at `newdata`, on the link or response
scale. The fixed-effect part is `X beta` with credible bounds from the
fixed-effect covariance ([`vcov()`](https://rdrr.io/r/stats/vcov.html)).

Formula random effects are conditioned on, as lme4 does: a row of a
group the fit saw takes the posterior mean of that group's effect, and a
row of a group it did not see takes the prior mean, zero, with an
interval that carries the between-group spread – its effect is drawn
from the term's covariance, at each posterior draw of it where the fit
samples or integrates the covariance and at its estimate or conditioned
value otherwise. `re_form = NA` gives the population prediction (every
random effect at zero); a one-sided formula such as `~ (1 | site)`
conditions on that term alone. With random effects the interval is read
over 4000 pinned draws of the fixed and random effects jointly (the same
on every call), and `se.fit` is the SD of those draws.

A fit carrying a latent field adds it to the prediction by default
(`include_field = TRUE`), on every backend and at the training design as
at `newdata`. The fit's latent posterior is a mixture of Gaussians in
the coefficients and the fields at their fitted support (the mesh nodes,
the fitted locations, the basis coefficients, the areal units, the
fitted times): one per hyperparameter cell on a nested-Laplace grid, the
one cell of a fit at fixed hyperparameters (`mode = "laplace"`), and one
point per stored draw on a sampler fit (`mode = "hmc"` and its siblings,
`"gibbs"`). The field at a row is its conditional distribution given
that fitted field: the mesh projector there for an SPDE field
([`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md)),
the basis re-evaluated there (on the domain the fit was built on) for an
HSGP field (`spatial_gp(approx = "hsgp")`), the NNGP predictive given
the field at the nearest fitted locations for a GP / NNGP field
([`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)),
the sum of its scales for
[`spatial_multiscale()`](https://gillescolling.com/tulpa/reference/spatial_multiscale.md),
and the value of the row's own unit or time for an areal (ICAR / CAR /
BYM2) or temporal (RW1 / RW2 / AR1) field. `newdata` then carries the
field's unit column (the one named in `spatial(...)`), time column and
group column; a unit the fit was not fitted at is refused, since the
field carries no value there. A temporal walk or GP
([`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
[`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md),
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md),
[`temporal_gp()`](https://gillescolling.com/tulpa/reference/temporal_gp.md))
at a time it was not fitted at is drawn from its conditional given the
fitted path: the recursion run past the fitted times (whole steps of
their equal spacing), the Brownian bridge between them for RW1, the GP
conditional at any time; RW2 / AR1 between fitted times or off the step
lattice are refused. An `s()` smoother is read at the node the fit
binned the row's covariate value into (a value outside the fitted range
is refused), a
[`spatiotemporal()`](https://gillescolling.com/tulpa/reference/spatiotemporal.md)
interaction at the row's area and time, an inline field replicated by a
factor (`spatial(..., by =)`) in the copy of the row's level, and a
[`latent()`](https://gillescolling.com/tulpa/reference/latent.md) block
at the training rows only. A fit carrying a latent term the prediction
cannot be read for is refused, naming the term, rather than predicted
without it. A restricted field
([`spatial_rsr()`](https://gillescolling.com/tulpa/reference/spatial_rsr.md))
is the field less its projection on the restriction design fitted at the
support, `w - x' (X_u' X_u)^-1 X_u' w`, with `x` the design of the
fitted unit or location a row sits at and otherwise the restriction
formula evaluated on `newdata`. `fit` is the exact posterior mean of the
linear predictor, and `se.fit` / `lower` / `upper` are read over 4000
pinned draws of the cell, the coefficients, the field and its
conditional at the rows, so the interval carries the uncertainty of the
coefficients and the field, an NNGP field's conditional (kriging)
variance away from the data, and the spread of the hyperparameters.
These are the draws
[`posterior_linpred()`](https://gillescolling.com/tulpa/reference/posterior_linpred.md)
and
[`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md)
return at the same `newdata`.

A spatially or temporally varying coefficient
([`spatial_svc()`](https://gillescolling.com/tulpa/reference/spatial_svc.md),
[`temporal_tvc()`](https://gillescolling.com/tulpa/reference/temporal_tvc.md))
is predicted at the `newdata` coordinates / times from every posterior
draw: an NNGP field by the NNGP predictive at the nearest fitted
locations, an HSGP field by its basis evaluated there (inside the basis
domain only); a `"gp"` time path by the GP conditional at any time, an
`"rw1"` path by a Brownian bridge between fitted levels and Brownian
steps past them, `"rw2"` and `"ar1"` paths by their recursion at whole
grid steps past the fitted levels. Uncertainty grows with distance from
the data and with the forecast horizon, and the bounds are pointwise. A
time between the levels of an `"rw2"` / `"ar1"` fit, a time off its step
lattice, a time beyond unequally spaced levels on any grid structure, a
factor level the fit did not see, and a group without a fitted path are
refused with the reason. `fit` is then the posterior mean of the linear
predictor and `se.fit` / `lower` / `upper` are its posterior SD and
quantiles.

A
[`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)
fit (no model formula) is predicted for one response `arm` at `newdata`,
a list: `X` is the arm's design at the new rows, with the columns of the
arm's fitted `X` in their order, and each latent term reaching the arm
is placed by the entry it needs – `index`, each row's unit (areal unit,
time, group, location) per block, a list of integer vectors by block
position, by `b<k>` or by the block's `name`; `coords`, the rows'
coordinates, for an `hsgp` block given by its coordinates (a basis
builder: `coords`, `m`, `c`), whose basis is rebuilt there on the fitted
domain; `basis` for an `hsgp` block given by its basis; `weight` for a
block's per-row design weight (`svc_weight`, or one per field for
`field_weight`); `projector` for a projected block; `re_idx`, each row's
group for the arm's random intercepts (`NA` for a group the fit did not
see, whose effect is drawn from its `sigma_re`); and `offset`. Each
block enters at its amplitude on the arm – a copy arm's `alpha` times
the field SD, a `field_coef`, an `iid` block's SD – as the fit's own
linear predictor takes it. Within each outer-grid cell the linear
predictor is Gaussian, with the cell's mode and covariance (read off the
cell's precision, which the fit keeps by default when it carries a
latent term, `control$store_Q`); the prediction is the weighted mixture
of those Gaussians over the cells, so `lower` / `upper` are the
mixture's quantiles. A fit with `store_Q = FALSE` reports the
uncertainty only for rows that load on the fixed effects and
basis-builder blocks alone.

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
  re_form = NULL,
  arm = NULL,
  ...
)
```

## Arguments

- object:

  A `tulpa_fit` object.

- newdata:

  Data frame of covariates (and, for a field fit, the field's coordinate
  columns, unit column or time and group columns; for a
  varying-coefficient fit, its coordinate / time and group columns). If
  `NULL`, predicts at the training design (requires `$model_matrix`).

- type:

  `"link"` (linear predictor) or `"response"` (mean scale, `E[y]`). For
  a binomial fit the `"response"` scale here is the per-trial success
  probability `g^{-1}(eta)` (there is no `n_trials` at `newdata`); this
  differs from [`fitted()`](https://rdrr.io/r/stats/fitted.values.html),
  which returns the trial-scaled expected count at the training design.
  On a zero-inflated fit (`ziformula`) the `"link"` scale is the count
  predictor and the `"response"` scale is the mixture mean
  `(1 - pi) E[y | eta]`, `pi = plogis(X_zi beta_zi)`, with the
  zero-inflation design rebuilt from the fit's `ziformula` at `newdata`.

- se.fit:

  If `TRUE`, also return the link-scale standard error and credible
  bounds. On a zero-inflated fit `se.fit` is the count predictor's, and
  the response-scale bounds are quantiles of the mixture mean over draws
  of the count and zero-inflation coefficients jointly (pinned, so
  repeated calls agree).

- level:

  Credible-interval level (default 0.95).

- include_field:

  For a fit with a latent field (continuous, areal or temporal), add the
  field to the prediction (default `TRUE`), and for a
  varying-coefficient fit the coefficient field at the `newdata`
  coordinates / times. `FALSE` gives the prediction without them.
  Ignored for fits with no field.

- re_form:

  Which formula random-effect terms to condition on, as lme4's
  `re.form`: `NULL` (default) every term, `NA` or `~0` none (the
  population level), or a one-sided formula naming a subset, e.g.
  `~ (1 | site)`. `newdata` then carries the grouping columns of the
  terms kept.

- arm:

  For a
  [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)
  fit, the response arm to predict (its name or index); may be left
  `NULL` on a one-arm fit.

- ...:

  Ignored.

## Value

If `se.fit = FALSE`, a numeric vector. If `se.fit = TRUE`, a data frame
with `fit`, `se.fit` (link scale), `lower`, `upper` on the requested
scale.
