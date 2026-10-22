# Fit a Gaussian linear model via tulpa's generic engine

Proof-of-concept function demonstrating the tulpa generic interface: the
Gaussian likelihood is supplied as a `LikelihoodSpec` and fitted by the
production NUTS sampler every other ModelData backend runs
(dual-averaged step size, adapted diagonal mass matrix, reverse-mode AD
gradient checked against finite differences before sampling). Fits
`y ~ Normal(X * beta, sigma)`.

## Usage

``` r
tulpa_gaussian(
  formula,
  data,
  beta_prior = .tulpa_default_beta_prior("gaussian"),
  control = list()
)
```

## Arguments

- formula:

  A formula (e.g., y ~ x1 + x2)

- data:

  A data frame

- beta_prior:

  Fixed-effect prior as `list(mean, sd)`: a mean-zero (`mean = 0`)
  Gaussian on every coefficient with SD `sd` (default the engine
  default, `prior_normal(0, 2.5)`).

- control:

  List of sampler knobs: `iter` (total iterations, default 2000),
  `warmup` (default 1000), `max_treedepth` (default 10), `adapt_delta`
  (target acceptance statistic, default 0.8), `seed` (`NULL` draws from
  the session RNG).

## Value

A `tulpa_fit` list with `draws` (post-warmup draws of `beta[j]` and
`log_sigma`), `means`, `sigma`
([`exp()`](https://rdrr.io/r/base/Log.html) of the posterior mean of
`log_sigma`), `accept_rate` (the mean NUTS acceptance statistic over the
post-warmup iterations), the per-iteration `accept_prob`, `divergent`,
`treedepth` and `log_prob`, the adapted step size `epsilon`, and
`formula`, `N`, `p`.

## Examples

``` r
# \donttest{
set.seed(1)
d <- data.frame(x = rnorm(40))
d$y <- 1 + 2 * d$x + rnorm(40, 0, 0.6)
fit <- tulpa_gaussian(y ~ x, d, control = list(iter = 1000, warmup = 500,
                                              seed = 1))
fit$means
fit$accept_rate
# }
```
