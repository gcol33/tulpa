# Half-Cauchy prior

Specify a half-Cauchy prior for a positive parameter. Has heavier tails
than half-normal, often used for variance parameters.

## Usage

``` r
prior_half_cauchy(scale = 2.5)
```

## Arguments

- scale:

  Scale parameter. Must be positive.

## Value

A `tulpa_prior` object

## Examples

``` r
prior_half_cauchy(2.5)
#> Half-Cauchy(2.50)
```
