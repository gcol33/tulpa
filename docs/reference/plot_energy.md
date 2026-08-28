# Plot Energy Diagnostic (E-BFMI)

Creates overlaid histograms of marginal energy and energy transition,
along with the E-BFMI statistic. Low E-BFMI indicates poor exploration
of the posterior.

## Usage

``` r
plot_energy(fit)
```

## Arguments

- fit:

  A `tulpa_fit` object (HMC backend).

## Value

A ggplot object (if ggplot2 available) or base R plot.

## Details

E-BFMI (Energy Bayesian Fraction of Missing Information) compares the
distribution of energy levels to energy transitions. Values below 0.3
indicate the sampler may not be exploring the full posterior.

## See also

[`diagnostic_summary()`](https://gillescolling.com/tulpa/reference/diagnostic_summary.md),
[`check_diagnostics()`](https://gillescolling.com/tulpa/reference/check_diagnostics.md)

## Examples

``` r
# \donttest{
set.seed(123)
df <- data.frame(x = rnorm(60))
df$y <- rpois(60, exp(0.5 + 0.3 * df$x))
fit <- tulpa(y ~ x, data = df, family = "poisson", mode = "hmc",
             control = list(n_iter = 500L, warmup = 250L, n_chains = 2L,
                            seed = 1L))
plot_energy(fit)
#> Energy values not available in fit object
# }
```
