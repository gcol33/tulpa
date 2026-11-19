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

`"wbic"` ranks by
[`wbic()`](https://gillescolling.com/tulpa/reference/wbic.md), an
estimate of each fit's negative log evidence, and needs every fit to
carry its tempered draws (`mode = "smc"`,
`control = list(wbic = TRUE)`); a fit without them gets an `NA` row.

An evidence (`"wbic"`, and `"loglik"` wherever
[`logLik()`](https://rdrr.io/r/stats/logLik.html) reports a log marginal
likelihood) integrates the likelihood against the prior, so it moves
with the prior's width in a way the posterior hardly does: widening a
prior SD by a factor `c` on a parameter the data pin down lowers the log
evidence by about `log(c)` per parameter. Two model types fitted under
different default priors are therefore compared with their priors, not
only their likelihoods. An ordinal fit, for instance, puts a
`N(0, 10^2)` ridge on its cutpoint parameters (the first cutpoint and
the log gaps, `cut_prior_sd` of
[`tulpa_ordinal()`](https://gillescolling.com/tulpa/reference/tulpa_ordinal.md))
where a multinomial fit's class intercepts carry the fixed-effect
`N(0, 2.5^2)`, so part of the evidence gap between the two is the gap
between those priors. Set the priors to statements you would defend for
both models before ranking by evidence, or rank by `"waic"` / `"loo"`,
which read the posterior predictive and depend on the prior only through
the posterior.

## Usage

``` r
compare_models(..., criterion = c("waic", "loo", "wbic", "loglik"))
```

## Arguments

- ...:

  Named `tulpa_fit` objects.

- criterion:

  `"waic"` (default), `"loo"`, `"wbic"`, or `"loglik"`.

## Value

A data frame. For `"wbic"` (ranked best-first): `model`, `wbic`, `delta`
(WBIC gap to the best model, \>= 0), and `weight` (`exp(-delta)`
normalized: the posterior model probability under equal prior model
weights, with each log evidence read off its WBIC). For `"loglik"`:
`model`, `n_params`, `logLik`, `quantity` and `conditioned_on` (which
log-scale quantity [`logLik()`](https://rdrr.io/r/stats/logLik.html)
reports, and the hyperparameters it holds fixed; fits that disagree on
either are refused rather than ranked). For `"waic"` / `"loo"` (ranked
best-first): `model`, `elpd`, `se_elpd`, `p_eff`, `ic` (`-2 * elpd`),
`delta` (elpd gap to the best model), `se_diff` (SE of that pointwise
elpd difference), and `weight` (the Akaike-style weight on the
criterion).

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
#>   model      elpd  se_elpd    p_eff       ic     delta  se_diff      weight
#> 1  full -188.1304 9.729789 2.187836 376.2607   0.00000       NA 1.00000e+00
#> 2  null -208.3904 8.892827 1.436874 416.7808 -20.26005 7.381431 1.58918e-09
# }
```
