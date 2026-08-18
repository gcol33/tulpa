# Plot method for tulpa_temporal_posterior

Plot method for tulpa_temporal_posterior

## Usage

``` r
# S3 method for class 'tulpa_temporal_posterior'
plot(x, component = NULL, type = "ribbon", ...)
```

## Arguments

- x:

  A tulpa_temporal_posterior object

- component:

  Which component to plot (for multi-scale). Default: first.

- type:

  Plot type: "ribbon" (default) or "line"

- ...:

  Additional arguments passed to plotting functions

## Value

A `ggplot` object when ggplot2 is installed; otherwise `NULL` invisibly,
after drawing a base-graphics plot. Called for the side effect of
plotting the temporal-effect posterior.
