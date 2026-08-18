# MCMC convergence diagnostics

**\[deprecated\]**

Use
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md),
which reads a fit's draws provenance and returns the diagnostic that
applies – chain mixing for MCMC draws, approximation reliability for
deterministic fits. The name `mcmc_diagnostics()` described only one of
the two branches it already routed between.

## Usage

``` r
mcmc_diagnostics(
  fit,
  pars = NULL,
  measures = c("rhat", "ess_bulk", "ess_tail"),
  probs = c(0.05, 0.95)
)
```

## Arguments

- fit:

  A `tulpa_fit` (or subclass) carrying posterior `$draws`. Multiple
  chains are recognised from a 3D `[iter, chain, param]` draws array, a
  `$chain_id` row map, or an `$n_chains` count over chain-major rows.

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

## Value

The value of
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
for `fit`.
