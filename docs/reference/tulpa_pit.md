# Probability integral transform from a predictive CDF

The generic, family-agnostic half of a PIT residual check: the model
package supplies the posterior-predictive CDF evaluated at each
observation (a `[n_draws x n_obs]` matrix, or a draw-averaged `[n_obs]`
vector), and this returns the PIT value per observation. For a discrete
or mixed response (a hurdle has a point mass at zero) supply the left
limit `cdf_lower` (`P(Y < y)`); the randomized PIT then draws one
uniform per observation and interpolates `F(y^-) + U (F(y) - F(y^-))`,
which is uniform under a correct model. With `cdf_lower = NULL` the
response is treated as continuous and the PIT is the draw-averaged CDF.

## Usage

``` r
tulpa_pit(
  cdf,
  cdf_lower = NULL,
  jitter = TRUE,
  log_lik = NULL,
  tail_points = NULL,
  n_threads = 1L
)
```

## Arguments

- cdf:

  Posterior-predictive CDF at the observed value, `P(Y <= y)`. A
  `[n_draws x n_obs]` matrix (averaged over draws here) or an `[n_obs]`
  vector.

- cdf_lower:

  Optional left-limit CDF `P(Y < y)`, same shape as `cdf`, for the
  randomized PIT of a discrete / mixed response.

- jitter:

  If `TRUE` (default) and `cdf_lower` is `NULL`, add a tiny uniform
  jitter to break ties from a discretized CDF; ignored when `cdf_lower`
  is supplied (the interpolation already randomizes).

- log_lik:

  Optional `[n_draws x n_obs]` matrix of the pointwise log-likelihood at
  each draw. When supplied, the PIT is the **leave-one-out** PIT: `cdf`
  / `cdf_lower` are reweighted by that observation's PSIS leave-one-out
  weights instead of being column-averaged.

- tail_points:

  Optional override for the PSIS tail size used by the leave-one-out
  weighting (see
  [`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md));
  `NULL` uses the automatic rule. Ignored unless `log_lik` is supplied.

- n_threads:

  Number of threads for the leave-one-out weighting (one observation per
  thread). Ignored unless `log_lik` is supplied.

## Value

Numeric vector of length `n_obs` of PIT values in `[0, 1]`.

## Details

Supplying `log_lik` switches to the **leave-one-out** PIT (as in INLA's
`cpo$pit` or `loo::psis_loo()`'s LOO-PIT): each observation's CDF limits
are averaged over draws with PSIS leave-one-out weights (from that
observation's own pointwise log-likelihood) instead of equal weights, so
the PIT does not use the observation to predict itself. A column whose
importance ratio is not all finite falls back to the equal-weight
average for that observation.

## See also

[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md),
[`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md)
