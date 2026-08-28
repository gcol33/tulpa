# Posterior summary of the fixed effects

Posterior summary of the fixed effects

## Usage

``` r
# S3 method for class 'tulpa_fit'
summary(object, level = 0.95, ...)
```

## Arguments

- object:

  A `tulpa_fit` object.

- level:

  Credible-interval level (default 0.95).

- ...:

  Ignored.

## Value

Data frame: estimate, std.error, and lower/upper credible bounds, one
row per fixed effect. Sampler tiers report empirical quantiles; the
Laplace tier reports the Gaussian approximation.

On a nested-Laplace fit the estimate and standard error are the
hyperparameter-grid-marginalized moments, and the bounds invert the
Gaussian mixture `sum_k w_k N(mu_kj, V_kjj)` that grid defines, rather
than reading `mu +/- z sigma` off the single Gaussian matching those
moments. An `interval_source` attribute records which read produced them
(`"mixture_cdf"`, `"gaussian_moment"`, `"skew_map_cell"`, or
`"skew_map_cell/mixture_cdf"` when the two are in play on different
coefficients) and `interval_declined` says why, whenever the mixture
read did not run. A `retained_mass` attribute gives the share of the
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
instead, and a `skew_applied` attribute names which coefficients took
the correction. That correction is measured at the MAP cell, so it is
reported on its own and is not composed with the mixture read; a
coefficient it declines keeps the mixture read rather than falling back
further.

An `axis_fields_dropped` attribute carries the grid axes the fit's own
resolved path could not read, one row per dropped field (block, type,
field, path, integrates, reason). It is `NULL` whenever every supplied
axis was used, which is the ordinary case;
[`diagnostic_summary()`](https://gillescolling.com/tulpa/reference/diagnostic_summary.md)
reads the same record in sentences.

A `beta_prior` attribute carries the Gaussian fixed-effect prior the fit
ran under, as `list(mean, sd)`. It is the engine default,
`prior_normal(0, 2.5)`, whenever the caller supplied none, and `NULL` on
the paths that express no Gaussian prior on the fixed effects.
