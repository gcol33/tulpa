# Posterior diagnostics for a fitted model

The diagnostic front door for any tulpa fit. What a fit's posterior
draws can be asked depends on how they were produced, so this reads the
draws provenance and returns the diagnostic that applies:

- MCMC chain draws:

  improved Rhat (the maximum of rank-normalized split-Rhat and folded
  split-Rhat), bulk / tail / mean / sd / quantile effective sample size,
  and Monte Carlo standard errors, following Vehtari et al. (2021).
  Split-Rhat is defined for a single chain, so a result is produced for
  any number of chains.

- i.i.d. approximation draws:

  the approximation-reliability table – the PSIS tail-shape `pareto_k`
  scored against the exact inner-Laplace marginal and the outer-grid
  quadrature effective sample size (the OUTER hyperparameter-grid
  integration layer), the inner-Laplace skewness diagnostic `gamma_3`
  when computed (the INNER Gaussian approximation to the latent field, a
  separate layer `pareto_k` does not cover), a combined whole-fit
  verdict naming which layer degrades when one does, and a per-parameter
  posterior summary. See
  [`laplace_diagnostics()`](https://gillescolling.com/tulpa/reference/laplace_diagnostics.md)
  for the full description of this table and its attributes.

- point summaries:

  no sample to diagnose; returns `NULL` with a message naming the
  backend.

Provenance is read from `$draws_kind` (stamped by `tulpa_dispatch`),
falling back to the backend registry's `emits` property and then to an
inner `$joint_fit`. A fit that predates the tag is treated as a chain,
so an older fit is never silently refused.

## Usage

``` r
diagnostics(fit, ...)

# Default S3 method
diagnostics(
  fit,
  pars = NULL,
  measures = c("rhat", "ess_bulk", "ess_tail"),
  probs = c(0.05, 0.95),
  sbc = NULL,
  ...
)

# S3 method for class 'sbc'
diagnostics(fit, ...)
```

## Arguments

- fit:

  A `tulpa_fit` (or subclass) carrying posterior `$draws`. Multiple
  chains are recognised from a 3D `[iter, chain, param]` draws array, a
  `$chain_id` row map, or an `$n_chains` count over chain-major rows.

- ...:

  Passed to the method.

- pars:

  Optional character vector of parameter names to restrict to.

- measures:

  Character vector selecting which diagnostics to compute, in
  output-column order. Available: `"rhat"`, `"rhat_bulk"`,
  `"rhat_fold"`, `"ess_bulk"`, `"ess_tail"`, `"ess_mean"`, `"ess_sd"`,
  `"mcse_mean"`, `"mcse_sd"`, `"ess_quantile"`, `"mcse_quantile"`.
  Defaults to the core set `c("rhat", "ess_bulk", "ess_tail")`. Applies
  to chain fits; the approximation-reliability table has a fixed set of
  columns.

- probs:

  Numeric probabilities for the quantile-based measures
  (`"ess_quantile"`, `"mcse_quantile"`); each expands to one column
  named e.g. `ess_q5`, `ess_q95`. Default `c(0.05, 0.95)`. Chain fits
  only.

- sbc:

  Optional [`sbc()`](https://gillescolling.com/tulpa/reference/sbc.md)
  result for the same model, whose calibration verdict is attached to
  the returned table.

## Value

For a chain fit, a data frame with a `parameter` column followed by one
column per requested measure; entries are `NA` for parameters that are
constant or have too few draws. For an i.i.d. approximation fit, the
`laplace_diagnostics` table (see that function for its attributes). For
a point fit, `NULL`.

## Extending

`diagnostics()` is an S3 generic so a model package can answer for its
own fit class (`diagnostics.ratiod_fit()`, say) rather than shadowing
this export with a same-named function. The default method does the
provenance routing described above and is what a method should delegate
to once it has assembled a draws array.

## Calibration alongside reliability

The tables above score ONE fit's own internal reliability. Whether the
backend's posterior is CALIBRATED is a different question, answered over
many simulated data sets by
[`sbc()`](https://gillescolling.com/tulpa/reference/sbc.md), and the two
disagree in both directions – a fit whose reliability band is clean on
both layers can still fail calibration, and one whose outer k-hat is
well past the escalation threshold can pass. So the band is a screen,
not a verdict. Pass an `sbc` result as `sbc =` and the calibration
verdict is attached to the table and printed underneath it;
`diagnostics()` called on the `sbc` result itself returns its report
table.

## References

Vehtari, Gelman, Simpson, Carpenter & Burkner (2021).
Rank-normalization, folding, and localization: an improved Rhat for
assessing convergence of MCMC. *Bayesian Analysis* 16(2):667-718.

Vehtari, Simpson, Gelman, Yao & Gabry (2024). Pareto smoothed importance
sampling. *JMLR* 25(72):1-58.

## See also

[`sbc()`](https://gillescolling.com/tulpa/reference/sbc.md) for
calibration,
[`laplace_diagnostics()`](https://gillescolling.com/tulpa/reference/laplace_diagnostics.md)
for the approximation-reliability table in full,
[`tulpa_draws_array()`](https://gillescolling.com/tulpa/reference/tulpa_draws_array.md),
[`plot_rhat()`](https://gillescolling.com/tulpa/reference/plot_rhat.md),
[`plot_ess()`](https://gillescolling.com/tulpa/reference/plot_ess.md),
[`diagnostic_summary()`](https://gillescolling.com/tulpa/reference/diagnostic_summary.md),
[`check_diagnostics()`](https://gillescolling.com/tulpa/reference/check_diagnostics.md)

## Examples

``` r
# \donttest{
set.seed(1)
df <- data.frame(x = rnorm(60))
df$y <- rpois(60, exp(0.5 + 0.3 * df$x))

# chain fit -> Rhat / ESS
hmc <- tulpa(y ~ x, data = df, family = "poisson", mode = "hmc",
             control = list(n_iter = 500L, warmup = 250L, n_chains = 2L,
                            seed = 1L))
diagnostics(hmc)

# deterministic fit -> PSIS approximation reliability
smc <- tulpa(y ~ x, data = df, family = "poisson", mode = "smc")
diagnostics(smc)
# }
```
