# Plot Autocorrelation Functions

Creates autocorrelation function (ACF) plots for selected parameters.
High autocorrelation indicates slow mixing and low effective sample
size.

## Usage

``` r
plot_acf(fit, pars = NULL, lags = 25, n_pars = 6)
```

## Arguments

- fit:

  A `tulpa_fit` object.

- pars:

  Character vector of parameter names. If NULL, selects worst-mixing
  parameters based on ESS.

- lags:

  Maximum number of lags to compute (default: 25).

- n_pars:

  Maximum number of parameters to plot (default: 6).

## Value

A ggplot object (if ggplot2 available) or base R plot (invisible).

## Details

Ideal ACF plots show rapid decay to zero. Slow decay indicates high
autocorrelation, which reduces effective sample size and may indicate
poor mixing.

## See also

[`plot_ess()`](https://gillescolling.com/tulpa/reference/plot_ess.md),
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
plot_acf(fit)

plot_acf(fit, lags = 10)

# }
```
