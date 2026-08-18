# Bayesian R-squared

Per posterior draw `s`,
`R2_s = Var_i(mu_si) / (Var_i(mu_si) + Var_res_s)`, where `mu_si` are
the response-scale fitted means and `Var_res_s` is the family's residual
variance averaged over observations (Gelman et al. 2019). The linear
predictor is rebuilt per draw exactly as in
[`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md)
(fixed effects + random effects + offset at the training data).

## Usage

``` r
bayes_R2(object, ...)

# S3 method for class 'tulpa_fit'
bayes_R2(
  object,
  ndraws = NULL,
  summary = TRUE,
  probs = c(0.025, 0.975),
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

- ndraws:

  Number of posterior draws to use. Defaults to all stored draws, or 400
  on the draw-free Laplace tier.

- summary:

  Summarize the per-draw values (default `TRUE`).

- probs:

  Quantiles reported by the summary (default 2.5% / 97.5%).

- seed:

  Optional integer seed (RNG state is restored on exit), used by the
  Gaussian fixed-effect sampling on draw-free fits.

## Value

With `summary = TRUE` (default) a one-row data frame with `estimate`
(posterior median), `std.error`, and the `probs` quantiles; with
`summary = FALSE` the vector of per-draw R^2 values.

## References

Gelman, Goodrich, Gabry & Vehtari (2019). R-squared for Bayesian
regression models. *The American Statistician* 73(3):307-309.

## Examples

``` r
# \donttest{
set.seed(1)
d <- data.frame(x = rnorm(200))
d$y <- rnorm(200, 2 * d$x, 1)
fit <- tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace", phi = 1)
bayes_R2(fit)
#>     estimate  std.error      2.5%    97.5%
#> R2 0.7710357 0.01427852 0.7395732 0.797705
# }
```
