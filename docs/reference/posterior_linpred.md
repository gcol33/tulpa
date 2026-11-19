# Posterior of the linear predictor

The posterior of the linear predictor `eta` at the training data with
every component the fit estimated: fixed effects, the offset, formula
random effects, spatial and temporal fields and
[`latent()`](https://gillescolling.com/tulpa/reference/latent.md)
blocks. It is the quantity
[`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md)
pushes through the family, read here on the link scale;
[`fitted.tulpa_fit()`](https://gillescolling.com/tulpa/reference/fitted.tulpa_fit.md)
is its posterior mean at the training data and
[`predict.tulpa_fit()`](https://gillescolling.com/tulpa/reference/predict.tulpa_fit.md)
summarizes it at new data.

On a nested-Laplace fit the posterior of each `eta_i` is the grid
mixture `sum_k w_k N(fitted_eta[k, i], fitted_eta_var[k, i])`, and
`summary = TRUE` reports its exact mean, SD and quantiles. A fit run
with `control$fitted_var = FALSE` carries no within-cell variance, and
its summary reports the mixture mean alone (`source = "mode"`, `sd` and
bounds `NA`). Every other fit summarizes its draws.

At `newdata`, a latent field is drawn at the new rows from its
conditional distribution given the fitted field (within each
hyperparameter cell and mixed over the cells, or given each stored draw
of a sampler): a continuous field at the new locations, an areal or
temporal field at the row's fitted unit or time (see
[`predict.tulpa_fit()`](https://gillescolling.com/tulpa/reference/predict.tulpa_fit.md));
these are the draws
[`predict.tulpa_fit()`](https://gillescolling.com/tulpa/reference/predict.tulpa_fit.md)
summarizes, so the two agree to Monte Carlo error.

## Usage

``` r
posterior_linpred(object, ...)

# S3 method for class 'tulpa_fit'
posterior_linpred(
  object,
  newdata = NULL,
  ndraws = NULL,
  summary = FALSE,
  probs = c(0.025, 0.975),
  seed = NULL,
  re_form = NULL,
  ...
)
```

## Arguments

- object:

  A `tulpa_fit` object.

- ...:

  Passed to methods.

- newdata:

  Optional data frame of covariates, read as in
  [`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md).
  `NULL` (default) reads the training data.

- ndraws:

  Number of draws (`summary = FALSE`, or a draw-based summary).

- summary:

  Logical; return a per-observation summary instead of draws.

- probs:

  The two probabilities of the reported interval.

- seed:

  Optional integer seed (RNG state is restored on exit).

- re_form:

  Which formula random-effect terms to condition on, as in
  [`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md).

## Value

With `summary = FALSE`, an `ndraws x n_obs` matrix of draws. With
`summary = TRUE`, a data frame with one row per observation: `estimate`,
`sd`, `conf.low`, `conf.high` (at `probs`) and `source` (`"mixture"` for
the exact grid mixture, `"mode"` for its mean alone, `"sampled"` for a
draw summary).

## See also

[`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md),
[`predict.tulpa_fit()`](https://gillescolling.com/tulpa/reference/predict.tulpa_fit.md).

## Examples

``` r
# \donttest{
set.seed(1)
d <- data.frame(y = rpois(100, 4), x = rnorm(100), g = gl(10, 10))
fit <- tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "laplace")
#> Warning: tulpa(): `sigma_re` not supplied; conditioning on sigma_re = 1 for each of the 1 RE term(s). Pass `sigma_re` to override.
head(posterior_linpred(fit, summary = TRUE))
#>   estimate        sd conf.low conf.high  source
#> 1 1.487996 0.1528009 1.182013  1.791083 sampled
#> 2 1.449005 0.1550125 1.141495  1.750855 sampled
#> 3 1.485797 0.1523092 1.181792  1.792162 sampled
#> 4 1.429036 0.1646791 1.103815  1.748674 sampled
#> 5 1.527944 0.1731871 1.190870  1.857790 sampled
#> 6 1.549073 0.1910855 1.160851  1.921701 sampled
# }
```
