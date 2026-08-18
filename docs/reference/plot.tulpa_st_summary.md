# Plot method for spatiotemporal effects

Plot method for spatiotemporal effects

## Usage

``` r
# S3 method for class 'tulpa_st_summary'
plot(x, type = "heatmap", ...)
```

## Arguments

- x:

  Spatiotemporal effects object

- type:

  Plot type: `"heatmap"` (default), `"time_series"`, or `"spatial_map"`

- ...:

  Additional arguments passed to plotting functions

## Value

A `ggplot` object when ggplot2 is installed; otherwise `NULL` invisibly,
after drawing a base-graphics plot. Called for the side effect of
visualizing the spatiotemporal interaction effects.
