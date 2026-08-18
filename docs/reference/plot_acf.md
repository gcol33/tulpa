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
# See plot_rhat() examples for fitting a model
# plot_acf(fit)
# plot_acf(fit, pars = c("beta_num[1]", "sigma_re"))
```
