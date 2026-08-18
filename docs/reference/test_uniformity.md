# Test uniformity of PIT residuals

If the model is correct, PIT residuals should follow Uniform(0, 1).
Applies a Kolmogorov-Smirnov test against that null. Equivalent to
[`DHARMa::testUniformity()`](https://rdrr.io/pkg/DHARMa/man/testUniformity.html).

## Usage

``` r
test_uniformity(
  object,
  observed = NULL,
  nsim = 250L,
  seed = 123L,
  plot = FALSE
)
```

## Arguments

- object:

  A fitted model, a numeric vector of PIT residuals, or a matrix of
  simulations

- observed:

  Observed data (if object is simulations matrix)

- nsim:

  Number of simulations (default 250)

- seed:

  Random seed (default 123)

- plot:

  If TRUE, draws a QQ plot

## Value

An `htest` object (KS test result)
