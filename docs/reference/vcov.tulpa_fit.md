# Variance-covariance matrix of the fixed effects

Variance-covariance matrix of the fixed effects

## Usage

``` r
# S3 method for class 'tulpa_fit'
vcov(object, ...)
```

## Arguments

- object:

  A `tulpa_fit` object.

- ...:

  Ignored.

## Value

Fixed-effect variance-covariance matrix (empirical for sampler tiers,
`H_beta^-1` for the Laplace tier).
