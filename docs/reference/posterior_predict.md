# Posterior predictive replicates

Draw replicated responses from the posterior predictive distribution:
the in-sample linear predictor is drawn with every component the fit
estimated – fixed effects, formula random effects, the offset, and any
spatial or temporal field – and pushed through the family's sampling
distribution. The same draws give the pointwise log-likelihood behind
`compare_models(criterion = "waic")` / `"loo"`.

Where the draws come from follows what the fit carries:

- A ModelData sampler fit (`mode = "hmc"` and its siblings) evaluates
  the engine's own linear predictor at each draw, so a field is drawn
  jointly with everything else.

- A nested-Laplace fit draws from its outer grid: a replicate picks a
  cell by its weight and draws each observation from that cell's
  Gaussian for the linear predictor (`fitted_eta`, `fitted_eta_var`).
  The cell's joint covariance across observations is not retained, so
  observations within a replicate are independent given the cell. A fit
  run with `control$fitted_var = FALSE` carries no `fitted_eta_var`, and
  its replicates hold the across-cell spread only.

- Any other fit carries its coefficients rather than its linear
  predictor. Fits with posterior draws use them (fixed and random
  effects jointly per draw); the Laplace tier samples the fixed effects
  from `N(coef(fit), vcov(fit))` and holds the random effects at their
  posterior mode, so its replicates understate the RE posterior
  uncertainty; an SPDE field enters at its posterior mean.

At `newdata` the prediction is population level (random effects at
zero), matching
[`predict.tulpa_fit()`](https://gillescolling.com/tulpa/reference/predict.tulpa_fit.md).

A zero-inflated fit (`ziformula`) draws the structural-zero logit from
the same posterior draw as the count predictor; each replicate
observation is a structural zero with probability `plogis(X_zi beta_zi)`
and otherwise a draw from the family (a zero-truncated family gives the
hurdle model).

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
  with every fitted component included.

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
