# Summarize and print a tulpa fit

[`print()`](https://rdrr.io/r/base/print.html) shows a fit in a few
lines: how it was produced (backend and inference tier), the call, the
observation count and the fixed-effect point estimates.
[`summary()`](https://rdrr.io/r/base/summary.html) returns the full
report as a `summary.tulpa_fit` object, which prints as a sectioned
display and whose parts are read by name;
[`coef()`](https://rdrr.io/r/stats/coef.html) on it returns the
fixed-effect table.

## Usage

``` r
# S3 method for class 'tulpa_fit'
print(x, ...)

# S3 method for class 'tulpa_fit'
summary(object, level = 0.95, ...)

# S3 method for class 'summary.tulpa_fit'
coef(object, ...)

# S3 method for class 'summary.tulpa_fit'
print(x, digits = max(3L, getOption("digits") - 3L), ...)
```

## Arguments

- x:

  A `tulpa_fit` object (`print.tulpa_fit()`) or a `summary.tulpa_fit`
  object (`print.summary.tulpa_fit()`).

- ...:

  Ignored.

- object:

  A `tulpa_fit` object.

- level:

  Credible-interval level of the fixed-effect table, a single number in
  `(0, 1)` (default 0.95).

- digits:

  Significant digits in the printed tables.

## Value

`print.tulpa_fit()` and `print.summary.tulpa_fit()` return their input
invisibly.

`summary.tulpa_fit()` returns a list of class `summary.tulpa_fit`:

- `call`, `formula`, `family`:

  as the fit records them, `NULL` where it records none.

- `inference`:

  how the fit was produced: a list of `backend`, `inference_tier`,
  `inference_mode` and `selection_reason` (why `mode = "auto"` chose
  that backend).

- `n_obs`, `n_draws`:

  the observation count and the number of posterior draws the
  fixed-effect summaries read (`NULL` without draws).

- `level`:

  the interval level of `coefficients`.

- `coefficients`:

  data frame, one row per fixed effect named by the term: `estimate`,
  `std.error` and the lower / upper credible bounds, labelled as
  [`confint.tulpa_fit()`](https://gillescolling.com/tulpa/reference/confint.tulpa_fit.md)
  labels them. Carries the `interval_source`, `interval_declined`,
  `retained_mass` and `skew_applied` attributes described in Details.
  Also returned by [`coef()`](https://rdrr.io/r/stats/coef.html) on the
  summary.

- `random`:

  the random-effect standard deviations as
  [`VarCorr()`](https://gillescolling.com/tulpa/reference/VarCorr.md)
  reports them, with their estimated / sampled / conditioned label;
  `NULL` on a fit with no random-effect terms.

- `hyperparameters`:

  data frame of the integrated hyperparameters, one row each:
  `parameter`, `mean`, `sd`, `median` and the 2.5% / 97.5% bounds
  `ci_lo` / `ci_hi`. On a nested-Laplace fit these are the outer-grid
  axes; on the random-effect covariance integrators the derived standard
  deviations and correlations. `NULL` where the fit integrated none.

- `sigma`:

  the residual / dispersion scale a fit reports as one number, else
  `NULL`.

- `smooth_terms`:

  the fit's `s(...)` term metadata, `NULL` without smoothers;
  [`smooth_effects()`](https://gillescolling.com/tulpa/reference/smooth_effects.md)
  returns the fitted curves.

- `reliability`:

  a list: `kind` (`"chain"`, `"iid"` or `"point"`, the draws provenance
  [`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
  routes on); on a chain `rhat_max`, `ess_bulk_min` and `ess_tail_min`
  over the parameters; on i.i.d. draws `band`, the one-row headline
  [`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
  attaches as its `summary` attribute (outer `pareto_k` and its band,
  grid quadrature ESS, inner-Laplace scores, `reliability`), and
  `moment_source`, which read the per-parameter moments came from;
  `verdict`, the whole-fit reading (`"converged"` / `"not converged"` on
  a chain that records its convergence check, the combined `reliability`
  verdict on i.i.d. draws, `NA` otherwise); `convergence`, the fit's own
  convergence record; and the backend-specific quality numbers
  `mean_accept` (Metropolis acceptance rate), `n_divergent` (divergent
  transitions) and `elbo` (the variational objective), each `NULL` where
  the backend has none.

- `axis_fields_dropped`:

  the grid axes the fit's own resolved path could not read, one row per
  dropped field (block, type, field, path, integrates, reason); `NULL`
  whenever every supplied axis was used.
  [`diagnostic_summary()`](https://gillescolling.com/tulpa/reference/diagnostic_summary.md)
  reads the same record in sentences.

- `prior`:

  a list whose `beta` is the Gaussian fixed-effect prior the fit ran
  under, as `list(mean, sd)`: the engine default `prior_normal(0, 2.5)`
  whenever the caller supplied none, and `NULL` on the paths that
  express no Gaussian prior on the fixed effects.

`coef.summary.tulpa_fit()` returns the `coefficients` data frame.

## Details

The fixed-effect table reads the same posterior
[`coef()`](https://rdrr.io/r/stats/coef.html),
[`confint()`](https://rdrr.io/r/stats/confint.html) and
[`vcov()`](https://rdrr.io/r/stats/vcov.html) read. Sampler tiers report
empirical quantiles; the Laplace tier reports the Gaussian
approximation, and so does any fit whose reported posterior is a
closed-form Gaussian (expectation propagation, the multinomial Laplace
fit), whose draws are then samples from that Gaussian and are not what
the summary reads.

On a nested-Laplace fit the estimate and standard error are the
hyperparameter-grid-marginalized moments, and the bounds invert the
Gaussian mixture `sum_k w_k N(mu_kj, V_kjj)` that grid defines, rather
than reading `mu +/- z sigma` off the single Gaussian matching those
moments. The table's `interval_source` attribute records which read
produced them (`"mixture_cdf"`, `"gaussian_moment"`, `"skew_map_cell"`,
or `"skew_map_cell/mixture_cdf"` when the two are in play on different
coefficients) and `interval_declined` says why, whenever the mixture
read did not run. Its `retained_mass` attribute gives the share of the
grid weight whose cells retained a fixed-effect block: 1 on a complete
grid, and below 1 on one that dropped a positive-weight cell, whose
report is then the posterior conditional on the cells that remain.

Where no per-cell block was retained the estimate falls back to the
grid-weighted average of the per-cell modes, restricted to the cells
whose inner solve reached a mode. A fit where none did reports `NA` with
`interval_declined = "not_converged"`, rather than the vector its Newton
started from as an estimate.

With `control$skew_correct = TRUE` a coefficient whose inner-Laplace
`gamma_3` is in the band it is valid on reports Cornish-Fisher quantiles
instead, and the table's `skew_applied` attribute names which
coefficients took the correction. That correction is measured at the MAP
cell, so it is reported on its own and is not composed with the mixture
read; a coefficient it declines keeps the mixture read rather than
falling back further.

The reliability headline is read from
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md),
so it answers the question that applies to how the draws were produced:
Rhat and effective sample size for a chain, the
approximation-reliability band (outer Pareto-k, grid quadrature, inner
Laplace, combined verdict) for i.i.d. draws from a deterministic
approximation, and nothing for a point summary.

## See also

[`coef.tulpa_fit()`](https://gillescolling.com/tulpa/reference/coef.tulpa_fit.md),
[`confint.tulpa_fit()`](https://gillescolling.com/tulpa/reference/confint.tulpa_fit.md),
[`VarCorr()`](https://gillescolling.com/tulpa/reference/VarCorr.md),
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md),
[`tulpa_outer_grid()`](https://gillescolling.com/tulpa/reference/tulpa_outer_grid.md)

## Examples

``` r
set.seed(1)
df <- data.frame(x = rnorm(60))
df$y <- rpois(60, exp(0.5 + 0.3 * df$x))
fit <- tulpa(y ~ x, data = df, family = "poisson", mode = "laplace")
fit
#> tulpa fit  (laplace, tier 2) 
#> Call: tulpa(formula = y ~ x, data = df, family = "poisson", mode = "laplace")
#>   n_obs: 60 
#> 
#> Coefficients:
#> (Intercept)           x 
#>      0.5403      0.3044 
#> 
#> Use summary() for intervals, variance components, hyperparameters and reliability.
s <- summary(fit)
s
#> tulpa fit  (laplace, tier 2) 
#> Call: tulpa(formula = y ~ x, data = df, family = "poisson", mode = "laplace")
#>   inference mode: structured 
#>   selected because: User-specified backend: laplace 
#>   family: poisson 
#>   n_obs: 60 
#> 
#> Fixed effects:
#>             estimate std.error  2.5 % 97.5 %
#> (Intercept)   0.5403    0.1023 0.3398 0.7409
#> x             0.3044    0.1179 0.0733 0.5355
#> 
#> Fixed-effect prior: normal(0, 2.5)
#> 
#> Reliability (iid draws):
coef(s)
#>              estimate std.error      2.5 %    97.5 %
#> (Intercept) 0.5403445 0.1023242 0.33979269 0.7408963
#> x           0.3043774 0.1179012 0.07329522 0.5354596
s$inference$backend
#> [1] "laplace"
```
