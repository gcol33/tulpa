# Test for over- or underdispersion

Compares the variance of observed data to the variance expected under
the fitted model (via simulation). Ratio \> 1 = overdispersion; \< 1 =
underdispersion. Equivalent to
[`DHARMa::testDispersion()`](https://rdrr.io/pkg/DHARMa/man/testDispersion.html).

## Usage

``` r
test_dispersion(object, ...)

# Default S3 method
test_dispersion(
  object,
  observed = NULL,
  nsim = 250L,
  seed = 123L,
  alternative = c("two.sided", "greater", "less"),
  ...
)
```

## Arguments

- object:

  A fitted model with
  [`simulate()`](https://rdrr.io/r/stats/simulate.html) method

- ...:

  Passed to methods.

- observed:

  Observed response vector (optional – extracted from fit)

- nsim:

  Number of simulations (default 250)

- seed:

  Random seed (default 123)

- alternative:

  `"two.sided"`, `"greater"`, or `"less"`

## Value

An `htest` object with dispersion ratio and p-value
