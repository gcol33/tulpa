# Exponential prior

Specify an exponential prior for a positive parameter.

## Usage

``` r
prior_exponential(rate = 1)
```

## Arguments

- rate:

  Rate parameter (lambda). Must be positive.

## Value

A `tulpa_prior` object

## Examples

``` r
prior_exponential(1)
#> Exponential(1.00)  [mean = 1.00]
```
