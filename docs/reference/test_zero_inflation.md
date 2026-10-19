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

## Examples

``` r
# \donttest{
set.seed(1)
d <- data.frame(x = rnorm(80))
d$y <- rpois(80, exp(0.3 + 0.5 * d$x))
fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
test_zero_inflation(fit, nsim = 50)
#> 
#>  Simulation-based zero-inflation test
#> 
#> data:  fit
#> zero-inflation ratio = 0.90476, observed zeros = 19, expected zeros =
#> 21, p-value = 0.76
#> alternative hypothesis: two.sided
#> 
# }
```
