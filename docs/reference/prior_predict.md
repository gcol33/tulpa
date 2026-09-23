# Prior predictive simulation

Draw datasets from the prior predictive distribution: parameters are
sampled from their priors (no data conditioning) and pushed through the
model's linear predictor and the family's simulator.

Useful for checking whether priors imply plausible data ranges before
fitting.

## Usage

``` r
prior_predict(
  formula,
  family,
  data,
  priors = NULL,
  n_draws = 100,
  seed = NULL,
  ...
)
```

## Arguments

- formula:

  A model formula (e.g., `y ~ x + (1 | g)`). For multi-process families,
  a list of formulas keyed by process name.

- family:

  A `tulpa_family` object exposing a `simulate_fn` (see
  [`tulpa_family()`](https://gillescolling.com/tulpa/reference/tulpa_family.md)).
  Model packages (tulpaRatio, tulpaObs) provide families; tests can
  build a minimal one with
  [`tulpa_family()`](https://gillescolling.com/tulpa/reference/tulpa_family.md).

- data:

  Data frame containing covariates and grouping factors. Used for
  dimensions and design matrices; the response column may be absent or
  NA.

- priors:

  Prior specification
  ([`tulpa_priors()`](https://gillescolling.com/tulpa/reference/tulpa_priors.md)).
  If `NULL`, uses defaults.

- n_draws:

  Number of prior parameter draws. Default 100.

- seed:

  Optional integer seed for reproducibility.

- ...:

  Passed to `family$simulate_fn`.

## Value

A `tulpa_prior_predict` object: a list with

- `y`: list of length `n_draws`, each element the simulated response for
  that draw (matching whatever shape `family$simulate_fn` returns).

- `theta`: list of length `n_draws` of parameter draws (`beta`, `sigma`,
  RE coefficients `u`, family-specific extras).

- `linpred`: list of length `n_draws`, each a list of linear predictor
  vectors per process.

- `family`: the family used.

- `n_draws`, `n_obs`.

## Examples

``` r
# Toy Gaussian family for illustration
fam <- tulpa_family(
  name = "gaussian",
  simulate_fn = function(eta, params, n_obs, ...) {
    rnorm(n_obs, eta[[1]], params$sigma_y)
  },
  extra_params = list(sigma_y = prior_half_normal(1))
)
df <- data.frame(y = rep(0, 20), x = rnorm(20))
pp <- prior_predict(y ~ x, fam, df, n_draws = 50, seed = 1)
length(pp$y)  # 50
#> [1] 50
```
