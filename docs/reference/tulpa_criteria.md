# Model criteria from a pointwise log-likelihood

Turn an `[n_draws x n_obs]` pointwise log-likelihood into the standard
Bayesian goodness-of-fit currency: WAIC, DIC, conditional predictive
ordinates (CPO) and their sum LPML, PSIS-LOO, all from one matrix.
PSIS-LOO reuses the native
[`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md)
smoothing, so the CPO and LOO numbers are the same computation
(`CPO_i = exp(elpd_loo_i)`, `LPML = elpd_loo`). The input may be a
matrix or a streaming
[`tulpa_loglik()`](https://gillescolling.com/tulpa/reference/tulpa_loglik.md)
so EVA-scale fits are processed in observation blocks.

## Usage

``` r
tulpa_criteria(
  log_lik,
  criteria = c("waic", "loo", "cpo", "lpml", "dic"),
  loglik_at_mean = NULL,
  group = NULL,
  chunk_size = NULL,
  pointwise = FALSE
)
```

## Arguments

- log_lik:

  An `[n_draws x n_obs]` numeric matrix of pointwise log-likelihoods, or
  a
  [`tulpa_loglik()`](https://gillescolling.com/tulpa/reference/tulpa_loglik.md)
  streaming wrapper.

- criteria:

  Which criteria to compute. Any of `"waic"`, `"loo"`, `"cpo"`,
  `"lpml"`, `"dic"`. `"loo"`, `"cpo"`, and `"lpml"` share the single
  PSIS pass; `"dic"` additionally needs `loglik_at_mean`.

- loglik_at_mean:

  Optional length-`n_obs` vector of pointwise log-likelihoods evaluated
  at the posterior mean of the parameters, supplied by the caller (the
  model package knows the parameterization). Required for DIC's plug-in
  deviance; without it the DIC fields are `NA`.

- group:

  Optional length-`n_obs` grouping (an integer / factor / character
  vector). The LOO unit is **one column of `log_lik`**: with
  `group = NULL` (the default) every column is its own fold
  (leave-one-row-out, e.g. per plot / per visit) and the result is
  byte-identical to the ungrouped call. When supplied, the per-draw
  pointwise log-likelihoods are summed within group to a
  `[n_draws x n_groups]` matrix **before** PSIS, so each fold is a whole
  group (leave-one-group-out cross-validation, LOGO-CV). Use it to
  switch the estimand from per-row to per-group LOO – e.g. on a
  cell-compressed hierarchical fit, leave out a whole cell rather than
  one of its rows. WAIC's variance term, `lppd`, `elpd_loo`, `cpo` and
  `pareto_k` are all computed on the grouped matrix. DIC is a plug-in
  deviance over all observations and is unaffected by `group`.

- chunk_size:

  Number of observation columns to process per block. The default
  streams the whole matrix at once when materialized, else picks a block
  sized to a few million entries.

- pointwise:

  If `TRUE`, also return the per-observation vectors (`elpd_waic`,
  `p_waic`, `elpd_loo`, `pareto_k`, `cpo`) for plotting / stacking.

## Value

A `tulpa_criteria` object: a list with the requested scalar scores (each
estimate paired with its standard error where defined), `n_draws` /
`n_obs`, the PSIS `pareto_k` summary, and – when `pointwise = TRUE` – a
`pointwise` data frame.

## Details

`p_waic` is the well-known positively-biased variance estimator at low
draw counts; the result records `n_draws` and the count of observations
with `p_waic_i > 0.4` (the `loo` heuristic for an unreliable WAIC), and
the PSIS-LOO `elpd_loo` is the more stable figure to report when that
count is non-trivial.

The LOO unit is whatever **one column of `log_lik`** holds. If the
consumer built the matrix with one column per row (plot / visit), the
default is per-row LOO; if a column already carries a whole group's
compressed likelihood, leaving it out drops that group and `pareto_k`
can blow up by construction. The `group` argument makes the unit
explicit: supply it to aggregate columns into folds and report
leave-one-group-out CV (LOGO-CV) instead, a different and deliberate
estimand.

## References

Vehtari, Gelman & Gabry (2017). Practical Bayesian model evaluation
using leave-one-out cross-validation and WAIC. *Statistics and
Computing* 27(5):1413-1432. Watanabe (2010). Spiegelhalter et al.
(2002). Geisser & Eddy (1979).

## See also

[`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md)
for the smoothing core,
[`tulpa_pit()`](https://gillescolling.com/tulpa/reference/tulpa_pit.md)
for the probability-integral-transform companion,
[`compare_models()`](https://gillescolling.com/tulpa/reference/compare_models.md)
for model comparison.

## Examples

``` r
# A draws x observations log-likelihood matrix (here built directly;
# in practice extracted from a fitted model's posterior draws).
set.seed(1)
y  <- rnorm(40)
mu <- matrix(rnorm(200 * 40, sd = 0.2), 200, 40)
ll <- dnorm(matrix(y, 200, 40, byrow = TRUE), mean = mu, log = TRUE)
tulpa_criteria(ll)
#> tulpa model criteria  (200 draws x 40 observations)
#>   WAIC           107.7  (SE 7.3)
#>   elpd_waic      -53.9  (SE 3.7)
#>   p_waic           1.4  (SE 0.3)
#>   LOOIC          107.8  (SE 7.3)
#>   elpd_loo       -53.9  (SE 3.7)
#>   p_loo            1.4
#>   LPML           -53.9
#>   DIC               NA
#>   p_DIC             NA
tulpa_criteria(ll, criteria = "waic", pointwise = TRUE)$pointwise[1:3, ]
#>   obs       lppd  elpd_waic      p_waic
#> 1   1 -1.1272129 -1.1439778 0.016764831
#> 2   2 -0.9525029 -0.9546593 0.002156456
#> 3   3 -1.2680835 -1.3033304 0.035246901
```
