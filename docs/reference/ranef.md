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

- The Laplace tier reports the conditional mode with no spread (`sd` and
  the bounds are `NA`), which is the only per-group quantity it forms.

The `source` column says per row which of these produced it: `"sampled"`
for a posterior draw summary, `"mixture"` for the node mixture, `"mode"`
for a conditional mode. A fit whose backend never forms a per-group
posterior at all (the adaptive Gauss-Hermite inner marginal integrates
each group out by quadrature) errors with that reason rather than
returning an empty table, which would be indistinguishable from a model
with no random effects. A model that genuinely has none returns a
zero-row data frame.

## Examples

``` r
# \donttest{
set.seed(1)
df <- data.frame(x = rnorm(100), g = factor(rep(1:10, 10)))
df$y <- rpois(100, exp(0.3 * df$x))
fit <- tulpa(y ~ x + (1 | g), data = df, family = "poisson")
ranef(fit)
# }
```
