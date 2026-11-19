# Random-effect summaries

Random-effect summaries

## Usage

``` r
ranef(object, ...)

# S3 method for class 'tulpa_fit'
ranef(object, ...)
```

## Arguments

- object:

  A `tulpa_fit` object.

- ...:

  Ignored.

## Value

Data frame with one row per random-effect coefficient: `term` (the group
level, and the coefficient for a random slope), `estimate`, `sd`, the
2.5% / 97.5% bounds `conf.low` / `conf.high`, and `source` (which
construction the row came from). `sd` and the bounds are `NA` on a
backend that reports a point per group (see Details).

## Details

What each backend reports for a group effect follows what it computes:

- Sampler tier and the RE-covariance Gibbs debias
  ([`tulpa_re_cov_gibbs()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_gibbs.md),
  reached by `tulpa(..., control = list(re_cov = "gibbs"))`) draw the
  random effects jointly with everything else, so `estimate` / `sd` /
  bounds are the empirical posterior summaries.

- The RE-covariance integrator
  ([`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md))
  carries a Gaussian per-group posterior at each `Sigma` node; the
  reported summaries are the exact moments and quantiles of the weighted
  mixture of those, so they carry both the within-node curvature and the
  `Sigma` uncertainty. A group effect the subspace debias selected
  (`control$subspace_debias`) is moved by the Metropolis sampler at
  every node instead, and is reported from those draws.

- The Laplace tier (`mode = "laplace"`) reports the Laplace Gaussian of
  each group effect given the conditioned `sigma_re`: the mode, the
  marginal SD from the diagonal of the inverse joint Hessian (the fixed
  effects and the other groups integrated out), and the Gaussian bounds.
  The interval holds `sigma_re` fixed, so it omits the scale's
  uncertainty. A fit that keeps no such curvature reports the mode with
  `NA` spread.

The `source` column says per row which of these produced it: `"sampled"`
for a posterior draw summary, `"mixture"` for the node mixture,
`"laplace"` for the Laplace Gaussian at a conditioned scale, `"mode"`
for a conditional mode without spread. A fit whose backend never forms a
per-group posterior at all (the adaptive Gauss-Hermite inner marginal
integrates each group out by quadrature) errors with that reason rather
than returning an empty table, which would be indistinguishable from a
model with no random effects. A model that genuinely has none returns a
zero-row data frame.

## See also

[`field_effects()`](https://gillescolling.com/tulpa/reference/field_effects.md)
for a spatial field,
[`smooth_effects()`](https://gillescolling.com/tulpa/reference/smooth_effects.md)
for an `s(x)` term.

## Examples

``` r
# \donttest{
set.seed(1)
df <- data.frame(x = rnorm(100), g = factor(rep(1:10, 10)))
df$y <- rpois(100, exp(0.3 * df$x))
fit <- tulpa(y ~ x + (1 | g), data = df, family = "poisson")
ranef(fit)
#>     term     estimate        sd   conf.low conf.high  source
#> 1   g[1] -0.057396423 0.3008308 -0.6735606 0.5223848 sampled
#> 2   g[2] -0.371631939 0.3566917 -1.1587555 0.2748691 sampled
#> 3   g[3]  0.430592135 0.2902941 -0.1634166 0.9876700 sampled
#> 4   g[4]  0.120593882 0.3243558 -0.5168653 0.7569955 sampled
#> 5   g[5] -0.049037426 0.3270000 -0.7005167 0.5869685 sampled
#> 6   g[6] -0.299676922 0.3521741 -1.0243211 0.3654488 sampled
#> 7   g[7] -0.156148750 0.3576860 -0.8632569 0.5433473 sampled
#> 8   g[8] -0.060701888 0.3289538 -0.7349215 0.5695302 sampled
#> 9   g[9] -0.005503859 0.3175790 -0.6590107 0.6280518 sampled
#> 10 g[10]  0.401565686 0.3193313 -0.1906539 1.0696400 sampled
# }
```
