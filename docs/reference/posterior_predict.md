# Posterior predictive replicates

Draw replicated responses from the posterior predictive distribution:
the linear predictor is rebuilt per posterior draw (fixed effects,
formula random effects, offset, and a posterior-mean SPDE field when
present) and pushed through the family's sampling distribution.

Fits carrying posterior draws use them directly (fixed and random
effects jointly per draw). The Laplace tier samples the fixed effects
from the Gaussian approximation `N(coef(fit), vcov(fit))` and holds the
random effects at their posterior mode, so its replicates understate the
RE posterior uncertainty. At `newdata` the prediction is population
level (random effects at zero), matching
[`predict.tulpa_fit()`](https://gillescolling.com/tulpa/reference/predict.tulpa_fit.md).

## Usage

``` r
posterior_predict(object, ...)

# S3 method for class 'tulpa_fit'
posterior_predict(
  object,
  newdata = NULL,
  ndraws = NULL,
  n_trials = NULL,
  seed = NULL,
  ...
)
```

## Arguments

- object:

  A `tulpa_fit` object from
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md).

- ...:

  Passed to methods.

- newdata:

  Optional data frame of covariates to predict at. Population level
  (fixed effects only); `NULL` (default) replicates at the training data
  with random effects and offset included.

- ndraws:

  Number of posterior draws to use. Defaults to all stored draws, or 400
  on the draw-free Laplace tier.

- n_trials:

  Binomial / beta-binomial trial counts for the replicates. Defaults to
  the training trials when `newdata` is `NULL`, else 1.

- seed:

  Optional integer seed (RNG state is restored on exit).

## Value

A `ndraws x n_obs` numeric matrix of replicated responses.

## See also

[`pp_check()`](https://gillescolling.com/tulpa/reference/pp_check.md),
which uses these replicates;
[`simulate.tulpa_fit()`](https://gillescolling.com/tulpa/reference/simulate.tulpa_fit.md).

## Examples

``` r
# \donttest{
set.seed(1)
d <- data.frame(y = rpois(100, 4), x = rnorm(100))
fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
yrep <- posterior_predict(fit, ndraws = 100)
dim(yrep)  # 100 x 100
#> [1] 100 100
# }
```
