# Compare models by information criteria

Rank fitted models best-first by an information criterion. `"waic"` and
`"loo"` use the native pointwise log-likelihood layer
([`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
/
[`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md))
– WAIC and PSIS-LOO respectively, computed from each fit's
`[n_draws x n_obs]` pointwise log-likelihood (`fit$draws$log_lik`), with
no `loo` package dependency. `"loglik"` returns the (integrated) joint
log-likelihood with the parameter count. A fit carrying no pointwise
log-likelihood (a deterministic / point approximation) yields `NA`
criterion columns rather than an error, so the table always has one row
per model.

## Usage

``` r
compare_models(..., criterion = c("waic", "loo", "loglik"))
```

## Arguments

- ...:

  Named `tulpa_fit` objects.

- criterion:

  `"waic"` (default), `"loo"`, or `"loglik"`.

## Value

A data frame. For `"loglik"`: `model`, `n_params`, `logLik`. For
`"waic"` / `"loo"` (ranked best-first): `model`, `elpd`, `se_elpd`,
`p_eff`, `ic` (`-2 * elpd`), `delta` (elpd gap to the best model),
`se_diff` (SE of that pointwise elpd difference), and `weight` (the
Akaike-style weight on the criterion).

## See also

[`model_average()`](https://gillescolling.com/tulpa/reference/model_average.md)
for model-averaged predictions,
[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
and
[`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md)
for the native criteria layer.

## Examples

``` r
# \donttest{
set.seed(1)
df <- data.frame(x = rnorm(120))
df$y <- rpois(120, exp(0.4 + 0.5 * df$x))
f1 <- tulpa(y ~ x, data = df, family = "poisson")
f2 <- tulpa(y ~ 1, data = df, family = "poisson")
compare_models(full = f1, null = f2, criterion = "waic")
# }
```
