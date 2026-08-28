# Fit a Gaussian linear model via tulpa's generic engine

Proof-of-concept function demonstrating the tulpa generic interface.
Fits y ~ Normal(X \* beta, sigma) with HMC sampling.

## Usage

``` r
tulpa_gaussian(
  formula,
  data,
  beta_prior = .tulpa_default_beta_prior("gaussian"),
  control = list()
)
```

## Arguments

- formula:

  A formula (e.g., y ~ x1 + x2)

- data:

  A data frame

- beta_prior:

  Fixed-effect prior as `list(mean, sd)`: a mean-zero (`mean = 0`)
  Gaussian on every coefficient with SD `sd` (default the engine
  default, `prior_normal(0, 2.5)`).

- control:

  List of numerical / sampler knobs: `iter` (total iterations, default
  2000), `warmup` (default 1000), `step_size` (HMC step size, default
  0.05), `n_leapfrog` (default 10), `seed` (`NULL` draws from the
  session RNG).

## Value

A list with draws matrix, posterior means, and metadata
