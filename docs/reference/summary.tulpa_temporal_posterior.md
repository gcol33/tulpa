# Summary method for tulpa_temporal_posterior

Summary method for tulpa_temporal_posterior

## Usage

``` r
# S3 method for class 'tulpa_temporal_posterior'
summary(object, probs = c(0.025, 0.5, 0.975), ...)
```

## Arguments

- object:

  A tulpa_temporal_posterior object

- probs:

  Quantiles to compute

- ...:

  Ignored

## Value

A `tulpa_temporal_summary` data frame with one row per time point (per
component for multi-scale fits), holding the posterior mean, SD, and
requested quantiles of the temporal effect.
