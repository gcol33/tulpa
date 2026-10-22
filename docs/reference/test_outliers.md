# Test for outliers (simulation envelope)

Counts how many observations fall outside the min-to-max range of all
simulations. Under a correct model, the expected number is approximately
`2 * N / (nsim + 1)`. Equivalent to
[`DHARMa::testOutliers()`](https://rdrr.io/pkg/DHARMa/man/testOutliers.html).

## Usage

``` r
test_outliers(object, ...)

# Default S3 method
test_outliers(object, observed = NULL, nsim = 250L, seed = 123L, ...)
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

An `htest` object (binomial test)

## Examples

``` r
# \donttest{
set.seed(1)
d <- data.frame(x = rnorm(80))
d$y <- rpois(80, exp(0.3 + 0.5 * d$x))
fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
test_outliers(fit, nsim = 50)
# }
```
