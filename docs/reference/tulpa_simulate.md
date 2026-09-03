# Simulate data from a tulpa model

Generic simulator that dispatches through a `tulpa_family`'s
`simulate_fn`. Given a model spec (formula + family + data) and a
parameter vector or a fitted model, generate one or more synthetic
response datasets.

Used internally by
[`prior_predict()`](https://gillescolling.com/tulpa/reference/prior_predict.md)
and exposed for posterior predictive checks, simulation-based
calibration, and what-if analyses with fixed parameters.

## Usage

``` r
tulpa_simulate(
  formula,
  family,
  data,
  theta = NULL,
  n_sims = 1L,
  priors = NULL,
  seed = NULL,
  ...
)
```

## Arguments

- formula:

  A model formula, or list of formulas keyed by process name.

- family:

  A `tulpa_family` object (see
  [`tulpa_family()`](https://gillescolling.com/tulpa/reference/tulpa_family.md)).

- data:

  Data frame with covariates and grouping factors.

- theta:

  One of:

  - A named list with `beta` (numeric or list per process), `u` (list of
    RE coefficient vectors, one per RE term per process), `extras`
    (named list of family-specific extras like `phi`, `sigma_y`).

  - A `tulpa_fit` object: posterior draws are sampled from `$draws`.

  - `NULL`: equivalent to a single prior draw (shortcut).

- n_sims:

  Number of simulated datasets. When `theta` is a fit, draws are
  subsampled (or recycled) to `n_sims`. Default 1.

- priors:

  Used only if `theta = NULL`; default
  [`tulpa_priors()`](https://gillescolling.com/tulpa/reference/tulpa_priors.md).

- seed:

  Optional integer seed.

- ...:

  Passed to `family$simulate_fn`.

## Value

A `tulpa_simulate` object: list with `y` (length-`n_sims` list of
simulated responses), `theta` (parameters used per sim), `linpred`,
`family`, `n_sims`, `n_obs`.

## Examples

``` r
fam <- tulpa_family(
  name = "gaussian",
  simulate_fn = function(eta, params, n_obs, ...) {
    rnorm(n_obs, eta[[1]], params$sigma_y)
  },
  extra_params = list(sigma_y = prior_half_normal(1))
)
df <- data.frame(y = rep(0, 20), x = rnorm(20))
theta <- list(
  beta = list(y = c(0.5, 1.0)),
  u = list(y = list()),
  extras = list(sigma_y = 0.5)
)
sim <- tulpa_simulate(y ~ x, fam, df, theta = theta, n_sims = 3, seed = 1)
length(sim$y)  # 3
```
