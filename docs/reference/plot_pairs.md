# Plot Bivariate Parameter Posteriors (Pairs Plot)

Creates a pairs plot showing bivariate relationships between parameters.
Divergent transitions (if present) are highlighted to help identify
problematic posterior regions.

## Usage

``` r
plot_pairs(
  fit,
  pars = NULL,
  highlight_divergent = TRUE,
  n_pars = 5,
  alpha = 0.3
)
```

## Arguments

- fit:

  A `tulpa_fit` object.

- pars:

  Character vector of parameter names. If NULL, selects main variance
  parameters.

- highlight_divergent:

  Logical; highlight divergent transitions in red (default: TRUE).

- n_pars:

  Maximum number of parameters (default: 5).

- alpha:

  Point transparency (default: 0.3).

## Value

A ggplot object (if ggplot2/GGally available) or base R plot.

## Details

Pairs plots help identify:

- Strong correlations between parameters (potential non-identifiability)

- Multimodality

- Regions where divergences cluster (indicating problematic geometry)

## See also

[`plot_divergences()`](https://gillescolling.com/tulpa/reference/plot_divergences.md),
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)

## Examples

``` r
# See plot_rhat() examples for fitting a model
# plot_pairs(fit)
# plot_pairs(fit, pars = c("sigma_re", "phi_num", "phi_denom"))
```
