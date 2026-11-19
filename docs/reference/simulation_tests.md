# Simulation-based residual tests

Tests of a fitted model against data simulated from it, the
simulation-based residual checks of the DHARMa package:

- `test_uniformity()`: if the model is correct, PIT residuals follow
  Uniform(0, 1). Applies a Kolmogorov-Smirnov test against that null.
  Equivalent to
  [`DHARMa::testUniformity()`](https://rdrr.io/pkg/DHARMa/man/testUniformity.html).

- `test_dispersion()`: compares the variance of the observed data to the
  variance expected under the fitted model (via simulation). Ratio \> 1
  = overdispersion; \< 1 = underdispersion. Equivalent to
  [`DHARMa::testDispersion()`](https://rdrr.io/pkg/DHARMa/man/testDispersion.html).

- `test_outliers()`: counts how many observations fall outside the
  min-to-max range of all simulations. Under a correct model, the
  expected number is approximately `2 * N / (nsim + 1)`. Equivalent to
  [`DHARMa::testOutliers()`](https://rdrr.io/pkg/DHARMa/man/testOutliers.html).

- `test_zero_inflation()`: compares the number of zeros in the observed
  data to the distribution expected under the fitted model. Equivalent
  to
  [`DHARMa::testZeroInflation()`](https://rdrr.io/pkg/DHARMa/man/testZeroInflation.html).

## Usage

``` r
test_uniformity(
  object,
  observed = NULL,
  nsim = 250L,
  seed = 123L,
  plot = FALSE
)

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

test_outliers(object, ...)

# Default S3 method
test_outliers(object, observed = NULL, nsim = 250L, seed = 123L, ...)

test_zero_inflation(object, ...)

# Default S3 method
test_zero_inflation(object, observed = NULL, nsim = 250L, seed = 123L, ...)
```

## Arguments

- object:

  A fitted model with a
  [`simulate()`](https://rdrr.io/r/stats/simulate.html) method. For
  `test_uniformity()` also a numeric vector of PIT residuals, or a
  matrix of simulations.

- observed:

  Observed response vector (optional – extracted from the fit; required
  by `test_uniformity()` when `object` is a simulations matrix).

- nsim:

  Number of simulations (default 250)

- seed:

  Random seed (default 123)

- plot:

  If TRUE, `test_uniformity()` draws a QQ plot

- ...:

  Passed to methods.

- alternative:

  `"two.sided"`, `"greater"`, or `"less"`

## Value

An `htest` object: the KS test (`test_uniformity()`), the dispersion
ratio and its p-value (`test_dispersion()`), the binomial test on the
count outside the envelope (`test_outliers()`), the zero-inflation ratio
and its p-value (`test_zero_inflation()`).

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
test_dispersion(fit, nsim = 50)
#> 
#>  Simulation-based dispersion test
#> 
#> data:  fit
#> dispersion ratio = 0.80214, p-value = 0.32
#> alternative hypothesis: two.sided
#> 
test_outliers(fit, nsim = 50)
#> 
#>  Simulation outlier test (0/80 outside envelope)
#> 
#> data:  n_outliers and N
#> number of successes = 0, number of trials = 80, p-value = 1
#> alternative hypothesis: true probability of success is greater than 0.03921569
#> 95 percent confidence interval:
#>  0 1
#> sample estimates:
#> probability of success 
#>                      0 
#> 
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
