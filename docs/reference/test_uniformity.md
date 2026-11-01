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

## Examples

``` r
set.seed(1)
test_uniformity(runif(50))   # PIT residuals supplied directly
#> 
#>  Exact one-sample Kolmogorov-Smirnov test
#> 
#> data:  r
#> D = 0.11212, p-value = 0.5196
#> alternative hypothesis: two-sided
#> 
# \donttest{
set.seed(1)
d <- data.frame(x = rnorm(80))
d$y <- rpois(80, exp(0.3 + 0.5 * d$x))
fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
test_uniformity(fit, nsim = 50)
#> 
#>  Exact one-sample Kolmogorov-Smirnov test
#> 
#> data:  r
#> D = 0.044078, p-value = 0.9959
#> alternative hypothesis: two-sided
#> 
# }
```
