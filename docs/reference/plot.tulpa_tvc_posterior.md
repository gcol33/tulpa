# Plot method for tulpa_tvc_posterior

Plot method for tulpa_tvc_posterior

## Usage

``` r
# S3 method for class 'tulpa_tvc_posterior'
plot(x, term = 1, type = "ribbon", ...)
```

## Arguments

- x:

  A tulpa_tvc_posterior object

- term:

  Which term to plot (name or index). Default: first term.

- type:

  Plot type: "ribbon" (default) or "line"

- ...:

  Additional arguments passed to plotting functions

## Value

A `ggplot` object when ggplot2 is installed; otherwise `NULL` invisibly,
after drawing a base-graphics plot. Called for the side effect of
plotting the selected temporally-varying coefficient.
