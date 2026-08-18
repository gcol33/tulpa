# PIT (Probability Integral Transform) residuals

For each observation, computes the quantile of the observed value within
the posterior predictive distribution. If the model is correct, PIT
residuals follow Uniform(0, 1).

## Usage

``` r
pit_residuals(object, ...)

# Default S3 method
pit_residuals(object, observed = NULL, nsim = 250L, seed = 123L, ...)
```

## Arguments

- object:

  A fitted model with a
  [`simulate()`](https://rdrr.io/r/stats/simulate.html) method, or a
  matrix of simulated values (n_obs x nsim)

- ...:

  Passed to methods.

- observed:

  Observed response vector (required if `object` is a matrix)

- nsim:

  Number of simulations (default 250)

- seed:

  Random seed (default 123)

## Value

Numeric vector of length n_obs with values in `[0, 1]`

## Details

For integer-valued responses, a randomisation step avoids discrete
artefacts: the residual is drawn uniformly between P(sim \< obs) and
P(sim \<= obs).
