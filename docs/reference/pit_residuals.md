# PIT (probability integral transform) residuals

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

## Examples

``` r
# From a matrix of replicates (n_obs x nsim) and the observed response:
set.seed(1)
y <- rpois(30, 3)
sims <- matrix(rpois(30 * 100, 3), nrow = 30)
r <- pit_residuals(sims, observed = y)
summary(r)
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#> 0.04201 0.28068 0.45159 0.50626 0.71116 0.99745 
# \donttest{
# From a fitted model with a simulate() method:
set.seed(1)
d <- data.frame(x = rnorm(80))
d$y <- rpois(80, exp(0.3 + 0.5 * d$x))
fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
head(pit_residuals(fit, nsim = 50))
#> [1] 0.02656494 0.92452768 0.56489120 0.72762588 0.81118118 0.52185790
# }
```
