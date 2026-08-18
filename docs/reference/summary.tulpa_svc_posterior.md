# Summary method for tulpa_svc_posterior

Summary method for tulpa_svc_posterior

## Usage

``` r
# S3 method for class 'tulpa_svc_posterior'
summary(object, probs = c(0.025, 0.5, 0.975), ...)
```

## Arguments

- object:

  A tulpa_svc_posterior object

- probs:

  Quantiles to compute

- ...:

  Ignored

## Value

A `tulpa_svc_summary` data frame with one row per location and term,
holding the observation index, term name, coordinates, and the posterior
mean, SD, and requested quantiles of each spatially-varying coefficient.
