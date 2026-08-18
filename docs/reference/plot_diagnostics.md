# Diagnostic Plotting Functions for tulpa Models

Visual diagnostic tools for MCMC convergence assessment. All functions
provide base R fallbacks when ggplot2/bayesplot are unavailable.

Creates a combined diagnostic figure with Rhat, ESS, trace plot, and
energy/ACF panels. Requires the `patchwork` package for layout.

## Usage

``` r
plot_diagnostics(fit, pars = NULL)
```

## Arguments

- fit:

  A `tulpa_fit` object.

- pars:

  Character vector of parameter names for trace plot. If NULL, uses the
  parameter with worst Rhat.

## Value

A combined plot (ggplot + patchwork) or NULL if requirements not met.

## Details

Creates a 2x2 grid:

- Top left: Rhat plot

- Top right: ESS plot

- Bottom left: Trace for worst parameter

- Bottom right: Energy (HMC) or ACF (other)

## See also

[`diagnostic_summary()`](https://gillescolling.com/tulpa/reference/diagnostic_summary.md),
[`plot_rhat()`](https://gillescolling.com/tulpa/reference/plot_rhat.md),
[`plot_ess()`](https://gillescolling.com/tulpa/reference/plot_ess.md)

## Examples

``` r
# See plot_rhat() examples for fitting a model
# plot_diagnostics(fit)
```
