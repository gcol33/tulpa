# Test for zero inflation

Compares the number of zeros in observed data to the distribution
expected under the fitted model. Equivalent to
[`DHARMa::testZeroInflation()`](https://rdrr.io/pkg/DHARMa/man/testZeroInflation.html).

## Usage

``` r
test_zero_inflation(object, ...)

# Default S3 method
test_zero_inflation(object, observed = NULL, nsim = 250L, seed = 123L, ...)
```

## Arguments

- object:

  A fitted model with
  [`simulate()`](https://rdrr.io/r/stats/simulate.html) method

- ...:

  Passed to methods.

- observed:

  Observed response vector (optional)

- nsim:

  Number of simulations (default 250)

- seed:

  Random seed (default 123)

## Value

An `htest` object with zero-inflation ratio and p-value
