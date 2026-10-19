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
#> Warning: tulpa(): the 're_cov_gibbs' chain has not converged -- x: Rhat 1.01, bulk-ESS 67. Its estimates and intervals are not yet the posterior. Run longer (control$n_iter / control$warmup), or use mode = 'hmc'; see check_diagnostics().
ranef(fit)
#>     term    estimate        sd   conf.low conf.high  source
#> 1   g[1] -0.04236037 0.3099103 -0.6875981 0.5397354 sampled
#> 2   g[2] -0.31658002 0.3386600 -1.0186997 0.2724267 sampled
#> 3   g[3]  0.41457813 0.2956690 -0.1304489 1.0242238 sampled
#> 4   g[4]  0.10813286 0.3237494 -0.5371499 0.7474092 sampled
#> 5   g[5] -0.04184347 0.3218825 -0.7036788 0.5629522 sampled
#> 6   g[6] -0.26006011 0.3384532 -1.0014842 0.3438895 sampled
#> 7   g[7] -0.18195490 0.3461762 -0.9084533 0.4893878 sampled
#> 8   g[8] -0.04564851 0.3204036 -0.7054439 0.5693173 sampled
#> 9   g[9] -0.01274066 0.3232769 -0.6864370 0.6125156 sampled
#> 10 g[10]  0.39926011 0.2922594 -0.1986759 0.9656433 sampled
# }
```
