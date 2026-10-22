# Widely applicable Bayesian information criterion (WBIC)

Watanabe's (2013) WBIC: the expected negative log-likelihood under the
posterior tempered at inverse temperature `beta = 1 / log(n)`,
\$\$\mathrm{WBIC} = -E\_\beta\left\[\sum\_{i=1}^n \log p(y_i \mid
\theta)\right\], \qquad p\_\beta(\theta) \propto p(\theta)
\prod\_{i=1}^n p(y_i \mid \theta)^\beta.\$\$ It estimates the Bayes free
energy `-log p(y)`, the negative log evidence, and shares its asymptotic
expansion up to `O_p(1)` terms in singular models as well as regular
ones – mixtures, reduced-rank and latent-factor structure, a variance
component at zero – where BIC's `(d / 2) log n` penalty does not
describe the evidence. Lower is better.

## Usage

``` r
wbic(object, ...)

# Default S3 method
wbic(object, ...)

# S3 method for class 'tulpa_fit'
wbic(object, ...)

# S3 method for class 'tulpa_wbic'
print(x, digits = 2, ...)
```

## Arguments

- object:

  A `tulpa_fit` fitted with `mode = "smc"` and
  `control = list(wbic = TRUE)`, or an `[n_draws x n_obs]` pointwise
  log-likelihood matrix (or
  [`tulpa_loglik()`](https://gillescolling.com/tulpa/reference/tulpa_loglik.md))
  at draws from the posterior tempered at `1 / log(n_obs)`.

- ...:

  Unused.

- x:

  A `tulpa_wbic` result.

- digits:

  Decimal places printed.

## Value

A `tulpa_wbic` object: a list with `wbic`, the inverse temperature
`beta`, `n_obs` and `n_draws`.

## Details

No draw from the posterior itself can supply the expectation: it is
taken under a different distribution, and importance-reweighting
posterior draws down to `beta` degenerates as `n` grows. A `tulpa_fit`
therefore carries the tempered draws only when its sampler visited that
distribution, which the SMC path does when asked: fit with
`mode = "smc"` and `control = list(wbic = TRUE)`. The population is then
carried from the reference through `p(theta) L(theta)^beta`, recorded
there, and tempered on to the posterior, so the fit's own `$draws` are
posterior draws as usual. A fit without the tempered draws is refused.

The log-likelihood is the normalized pointwise density
[`pointwise_loglik()`](https://gillescolling.com/tulpa/reference/criteria_doors.md)
returns, the input of
[`loo::waic()`](https://mc-stan.org/loo/reference/waic.html) and
[`loo::loo()`](https://mc-stan.org/loo/reference/loo.html) too, so WBIC
is on the scale of `-log p(y)` and comparable across fits of the same
response. The parameter vector tempered is the one the sampler samples –
fixed effects, latent effects and hyperparameters – with the likelihood
conditional on it, so the free energy estimated is that of the marginal
likelihood integrating all of them.

The default method takes an `[n_draws x n_obs]` pointwise log-likelihood
whose rows are draws from the posterior tempered at `1 / log(n_obs)`. A
model package with its own likelihood draws them through the
`tulpa_smc_fit` C entry with `bridge_end = 1 / log(n)` and evaluates its
density there.

## References

Watanabe, S. (2013). A widely applicable Bayesian information criterion.
*Journal of Machine Learning Research* 14:867-897.

## See also

[`loo::waic()`](https://mc-stan.org/loo/reference/waic.html),
[`dic()`](https://gillescolling.com/tulpa/reference/criteria_doors.md)
and
[`compare_models()`](https://gillescolling.com/tulpa/reference/compare_models.md)
for the other criteria;
[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md).

## Examples

``` r
# \donttest{
set.seed(1)
d <- data.frame(x = rnorm(80))
d$y <- rpois(80, exp(0.3 + 0.5 * d$x))
fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "smc",
             control = list(wbic = TRUE, n_particles = 500L, seed = 1L))
wbic(fit)
# }
```
