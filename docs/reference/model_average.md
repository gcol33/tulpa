# Model-averaged predictions

Combine fitted values from several models using native model weights
computed from the pointwise PSIS-LOO (or WAIC) elpd via
[`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md)
– no `loo` package dependency. `"loo"` / `"waic"` give stacking weights
(the simplex-optimal predictive combination); `"pbma"` / `"pbma+"` give
pseudo-BMA(+) weights. Every model must carry an `[n_draws x n_obs]`
pointwise log-likelihood (`fit$draws$log_lik`) over the same
observations.

## Usage

``` r
model_average(
  ...,
  weights = c("loo", "waic", "pbma", "pbma+"),
  fitted_fn = fitted
)
```

## Arguments

- ...:

  Named `tulpa_fit` objects fitted to the same observations.

- weights:

  `"loo"` (stacking, default), `"waic"`, `"pbma"`, or `"pbma+"`.

- fitted_fn:

  Function extracting a length-`n_obs` fitted vector from a fit (default
  [`fitted()`](https://rdrr.io/r/stats/fitted.values.html)).

## Value

A list with `averaged` (the weighted fitted vector), `weights` (the
named model weights), and `comparison` (the
[`compare_models()`](https://gillescolling.com/tulpa/reference/compare_models.md)
table).

## References

Yao, Vehtari, Simpson & Gelman (2018). Using stacking to average
Bayesian predictive distributions. *Bayesian Analysis* 13(3):917-1007.

## See also

[`compare_models()`](https://gillescolling.com/tulpa/reference/compare_models.md).

## Examples

``` r
# \donttest{
set.seed(1)
df <- data.frame(x = rnorm(120))
df$y <- rpois(120, exp(0.4 + 0.5 * df$x))
f1 <- tulpa(y ~ x, data = df, family = "poisson", mode = "hmc",
            control = list(n_iter = 500L, warmup = 250L, seed = 1L))
f2 <- tulpa(y ~ 1, data = df, family = "poisson", mode = "hmc",
            control = list(n_iter = 500L, warmup = 250L, seed = 1L))
ma <- model_average(full = f1, null = f2, weights = "waic")
ma$weights
# }
```
