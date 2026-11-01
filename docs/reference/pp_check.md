# Posterior predictive check

Generate posterior predictive checks for a fitted tulpa model. Compares
observed data to replicated data from the posterior predictive
distribution.

## Usage

``` r
pp_check(object, ...)

# S3 method for class 'tulpa_fit'
pp_check(
  object,
  type = c("dens_overlay", "scatter", "intervals", "stat"),
  component = NULL,
  stat = mean,
  ndraws = 50,
  ...
)
```

## Arguments

- object:

  A `tulpa_fit` object

- ...:

  Additional arguments passed to bayesplot functions

- type:

  Type of check: "dens_overlay", "scatter", "intervals", "stat"

- component:

  Which component: "numerator", "denominator", or "both"

- stat:

  Function for "stat" type (default: mean)

- ndraws:

  Number of posterior draws to use (default: 50 for plots)

## Value

A ggplot object

## Examples

``` r
# pp_check is a generic; model packages (e.g. tulpaObs, tulpaRatio) provide
# the posterior-predictive method for their fits.
# \donttest{
set.seed(123)
n <- 200L
df <- data.frame(y = rpois(n, 5), x = rnorm(n),
                 site = factor(rep(1:10, each = 20)))
fit <- tulpa(y ~ x + (1 | site), data = df, family = "poisson",
             mode = "hmc", control = list(n_iter = 500, warmup = 250))
# Density overlay (dispatches to the model package's pp_check method).
if (requireNamespace("bayesplot", quietly = TRUE)) {
  pp_check(fit, type = "dens_overlay")
}

# }
```
