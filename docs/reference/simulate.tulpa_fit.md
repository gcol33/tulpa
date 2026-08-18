# Simulate responses from a fitted tulpa model

Base-R alias for
[`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md):
each simulation is one posterior predictive replicate at the training
data.

## Usage

``` r
# S3 method for class 'tulpa_fit'
simulate(object, nsim = 1, seed = NULL, ...)
```

## Arguments

- object:

  A `tulpa_fit` object.

- nsim:

  Number of simulated datasets (default 1).

- seed:

  Optional integer seed (RNG state is restored on exit).

- ...:

  Ignored.

## Value

A data frame with `nsim` columns (`sim_1`, ...), one row per
observation, following the
[`stats::simulate()`](https://rdrr.io/r/stats/simulate.html) convention.
