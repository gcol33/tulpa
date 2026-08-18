# Construct a minimal tulpa_family for simulation

Lightweight constructor for a `tulpa_family` object that exposes the
contract required by
[`prior_predict()`](https://gillescolling.com/tulpa/reference/prior_predict.md)
and
[`tulpa_simulate()`](https://gillescolling.com/tulpa/reference/tulpa_simulate.md).
Model packages (tulpaRatio, tulpaObs) register richer families that also
link to C++ likelihoods; this helper is for tests and simple custom
families that only need simulation.

## Usage

``` r
tulpa_family(
  name,
  simulate_fn,
  process_names = "y",
  extra_params = list(),
  link_inv = NULL
)
```

## Arguments

- name:

  Family name (character, length 1).

- simulate_fn:

  `function(eta, params, n_obs, ...)` returning a numeric vector of
  length `n_obs` (single-process) or a list of such vectors keyed by
  `process_names` (multi-process). `eta` is a list of linear predictors,
  one per process.

- process_names:

  Character vector. Defaults to `"y"` (single-process).

- extra_params:

  Named list of `tulpa_prior` objects for likelihood- specific scalar
  parameters (e.g., dispersion `phi`). Drawn at each prior-predictive
  iteration. Defaults to empty.

- link_inv:

  List of inverse-link functions per process; defaults to identity for
  every process. tulpa passes the raw linear predictor to `simulate_fn`,
  so most families implement the link inside `simulate_fn` (e.g.,
  `mu = exp(eta)` for Poisson). The `link_inv` slot exists for families
  that prefer to keep the inverse-link separate.

## Value

A `tulpa_family` object.

## Examples

``` r
fam <- tulpa_family(
  name = "poisson",
  simulate_fn = function(eta, params, n_obs, ...) {
    rpois(n_obs, exp(eta[[1]]))
  }
)
```
