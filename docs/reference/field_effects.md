# Spatial field effects per unit

The posterior of a fit's spatial field, one row per spatial unit: an
areal region (a row of the adjacency matrix) or an SPDE mesh node. The
value reported is the field's contribution to the linear predictor at
that unit, the quantity a covariate is read against.

## Usage

``` r
field_effects(object)
```

## Arguments

- object:

  A `tulpa_fit` carrying a spatial field.

## Value

A data frame with one row per spatial unit: `unit` (the node index),
`label` (the adjacency row name where it has one, else the index),
`estimate`, `sd`, the 2.5% / 97.5% bounds `conf.low` / `conf.high`, and
`source`: `"sampled"` for a posterior draw summary, `"mixture"` for a
nested-Laplace Gaussian mixture, `"mode"` for a point at the mode (`sd`
and the bounds `NA`). The field type is attached as attribute `"field"`.

## Details

Where the posterior comes from follows what the fit computes:

- A ModelData sampler fit (`mode = "hmc"` and its siblings) with an
  ICAR, BYM2 or proper-CAR field: every draw's field contribution is
  evaluated through the engine's own linear-predictor assembly, so an
  intrinsic field is centred as its density centres it and a BYM2 field
  is the combined
  `sigma * (sqrt(rho) * s * phi + sqrt(1 - rho) * theta)` at that draw's
  `sigma` and `rho` (`s` the scaling factor). A unit that no observation
  falls in contributes nothing to the linear predictor and is reported
  as `NA`.

- A Polya-Gamma Gibbs fit (`mode = "gibbs"`) stores the field it samples
  (`phi_spatial[k]`); for BYM2 each draw's two unit-scale components are
  combined into the same
  `sigma * (sqrt(rho) * s * phi + sqrt(1 - rho) * theta)` the sampler
  path reports. The rows summarize those draws.

- A nested-Laplace fit with an ICAR or proper-CAR field: each outer cell
  contributes its conditional mode, and its marginal variance where the
  fit retained one, weighted by the cell's posterior weight. With the
  variances the summary is the exact mean, SD and quantiles of that
  Gaussian mixture (`source = "mixture"`); without them the weighted
  mean of the cell modes, with no spread (`source = "mode"`). A BYM2
  field is reported where the fit stores its grid-weighted combined
  field
  ([`fit_st_nested()`](https://gillescolling.com/tulpa/reference/fit_st_nested.md));
  a [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
  nested BYM2 fit carries its two components with cell-dependent mixing
  and errors.

- A conditional-Laplace areal fit (`mode = "laplace"` / `"eb"`) reports
  the field at its mode, with no spread. For BYM2 this is the combined
  field at the `sigma = 1`, `rho = 0.5` the conditional kernel fixes.

- An SPDE fit
  ([`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md))
  reports the mesh-node field: its draws on the sampler path, its mode
  (the posterior-mean hyperparameters' fit on the nested path)
  otherwise.

A fit whose field cannot be located this way (a continuous GP / NNGP /
HSGP field, a restricted field, an inline
[`spatial()`](https://gillescolling.com/tulpa/reference/spatial.md)
varying-coefficient field) errors naming the field and the backend, as
does a fit with no spatial field. A temporal field is read by
[`temporal()`](https://gillescolling.com/tulpa/reference/temporal.md).

## See also

[`smooth_effects()`](https://gillescolling.com/tulpa/reference/smooth_effects.md)
for an `s(x)` term,
[`ranef()`](https://gillescolling.com/tulpa/reference/ranef.md) for
random effects,
[`temporal()`](https://gillescolling.com/tulpa/reference/temporal.md)
for a temporal field.

## Examples

``` r
# \donttest{
set.seed(1)
K <- 12
W <- matrix(0, K, K)
for (i in 1:K) { j <- if (i < K) i + 1 else 1; W[i, j] <- W[j, i] <- 1 }
region <- sample(1:K, 200, replace = TRUE)
d <- data.frame(region = region, x = rnorm(200))
d$y <- rpois(200, exp(0.3 + 0.5 * d$x + sin(2 * pi * region / K)))
fit <- tulpa(y ~ x + spatial(region), data = d, family = "poisson",
             spatial = list(type = "icar", adjacency = W), mode = "laplace")
fe <- field_effects(fit)
plot(fe$unit, fe$estimate, type = "b")
# }
```
