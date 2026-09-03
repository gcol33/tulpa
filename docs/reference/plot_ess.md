# Plot Effective Sample Size Diagnostic

Creates a visual display of effective sample size (ESS) for all
parameters, expressed as a ratio of ESS to total samples. Low ESS
indicates high autocorrelation.

## Usage

``` r
plot_ess(fit, type = c("bulk", "tail"), threshold = 400, pars = NULL)
```

## Arguments

- fit:

  A `tulpa_fit` object.

- type:

  Type of ESS: "bulk" (default) or "tail".

- threshold:

  Minimum acceptable ESS (default: 400).

- pars:

  Character vector of parameter names to include.

## Value

A ggplot object (if ggplot2 available) or base R plot (invisible).

## Details

ESS/iter ratio interpretation:

- Green: ESS \>= threshold

- Yellow: threshold/2 \<= ESS \< threshold

- Red: ESS \< threshold/2

## See also

[`plot_rhat()`](https://gillescolling.com/tulpa/reference/plot_rhat.md),
[`diagnostic_summary()`](https://gillescolling.com/tulpa/reference/diagnostic_summary.md),
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)

## Examples

``` r
# \donttest{
set.seed(123)
df <- data.frame(x = rnorm(60))
df$y <- rpois(60, exp(0.5 + 0.3 * df$x))
fit <- tulpa(y ~ x, data = df, family = "poisson", mode = "hmc",
             control = list(n_iter = 500L, warmup = 250L, n_chains = 2L,
                            seed = 1L))
plot_ess(fit)
plot_ess(fit, type = "tail")
# }
```
