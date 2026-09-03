# Gamma prior

Specify a gamma prior for a positive parameter.

## Usage

``` r
prior_gamma(shape = 2, rate = 0.1)
```

## Arguments

- shape:

  Shape parameter (alpha). Must be positive.

- rate:

  Rate parameter (beta). Must be positive.

## Value

A `tulpa_prior` object

## Details

Mean = shape/rate, Variance = shape/rate^2.

## Examples

``` r
prior_gamma(2, 0.1)  # Mean = 20, weakly informative
prior_gamma(1, 1)    # Exponential(1)
```
