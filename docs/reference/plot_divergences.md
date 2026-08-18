# Plot Divergent Transitions

Creates visualizations to investigate divergent transitions. Parallel
coordinates and scatter plots highlight where in parameter space
divergences occur.

## Usage

``` r
plot_divergences(fit, pars = NULL, type = c("parcoord", "scatter"))
```

## Arguments

- fit:

  A `tulpa_fit` object (HMC backend).

- pars:

  Character vector of parameter names. If NULL, uses variance
  parameters.

- type:

  Plot type: "parcoord" (parallel coordinates) or "scatter".

## Value

A ggplot object or base R plot (invisible).

## Details

Divergent transitions indicate regions of high posterior curvature that
the sampler cannot efficiently explore. Common causes:

- Very narrow funnels (hierarchical models)

- Strong correlations

- Multi-modality

## See also

[`plot_pairs()`](https://gillescolling.com/tulpa/reference/plot_pairs.md),
[`n_divergent()`](https://gillescolling.com/tulpa/reference/n_divergent.md)

## Examples

``` r
# See plot_rhat() examples for fitting a model
# plot_divergences(fit)
# plot_divergences(fit, type = "scatter")
```
