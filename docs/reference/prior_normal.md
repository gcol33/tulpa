# Normal prior

Specify a normal prior for a parameter.

## Usage

``` r
prior_normal(mean = 0, sd = 2.5)
```

## Arguments

- mean:

  Prior mean. Default 0.

- sd:

  Prior standard deviation. Must be positive.

## Value

A `tulpa_prior` object

## Examples

``` r
prior_normal(0, 2.5)
#> Normal(0.00, 2.50)
prior_normal(0, 1)
#> Normal(0.00, 1.00)
```
