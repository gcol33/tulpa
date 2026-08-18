# Plot method for tulpa_svc_posterior

Plot method for tulpa_svc_posterior

## Usage

``` r
# S3 method for class 'tulpa_svc_posterior'
plot(x, term = 1, type = "mean", ...)
```

## Arguments

- x:

  A tulpa_svc_posterior object

- term:

  Which term to plot (name or index). Default: first term.

- type:

  Plot type: "mean" (default), "sd", or quantile (e.g., "q50")

- ...:

  Additional arguments passed to plotting functions

## Value

A `ggplot` object when ggplot2 is installed; otherwise `NULL` invisibly,
after drawing a base-graphics map. Called for the side effect of mapping
the selected spatially-varying coefficient.
