# Plot multiple maps in a grid

Create a multi-panel figure with the fitted-value map and the
prediction-uncertainty map side by side.

## Usage

``` r
plot_map_panel(x, newdata = NULL, ncol = 2, ...)
```

## Arguments

- x:

  A `tulpa_fit` object with spatial structure.

- newdata:

  Optional data frame with prediction locations.

- ncol:

  Number of columns in the grid. Default 2.

- ...:

  Additional arguments passed to
  [`plot_map()`](https://gillescolling.com/tulpa/reference/plot_map.md).

## Value

A patchwork object (if patchwork is installed) or a list of ggplots.

## Examples

``` r
# See plot_map() examples for fitting a spatial model
# plot_map_panel(fit)
# plot_map_panel(fit, ncol = 1)
```
