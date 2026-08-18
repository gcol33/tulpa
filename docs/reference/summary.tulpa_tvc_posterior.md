# Summary method for tulpa_tvc_posterior

Summary method for tulpa_tvc_posterior

## Usage

``` r
# S3 method for class 'tulpa_tvc_posterior'
summary(object, probs = c(0.025, 0.5, 0.975), ...)
```

## Arguments

- object:

  A tulpa_tvc_posterior object

- probs:

  Quantiles to compute

- ...:

  Ignored

## Value

A `tulpa_tvc_summary` data frame with one row per time point and term,
holding the posterior mean, SD, and requested quantiles of each
temporally-varying coefficient.
