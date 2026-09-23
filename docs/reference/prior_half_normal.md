# Half-normal prior

Specify a half-normal (truncated at 0) prior for a positive parameter.

## Usage

``` r
prior_half_normal(sd = 1)
```

## Arguments

- sd:

  Scale parameter. Must be positive.

## Value

A `tulpa_prior` object

## Examples

``` r
prior_half_normal(1)
#> Half-Normal(1.00)
```
