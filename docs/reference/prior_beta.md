# Beta prior

Specify a beta prior for a parameter bounded in (0, 1).

## Usage

``` r
prior_beta(alpha = 1, beta = 1)
```

## Arguments

- alpha:

  First shape parameter. Must be positive.

- beta:

  Second shape parameter. Must be positive.

## Value

A `tulpa_prior` object

## Details

- alpha = beta = 1: Uniform(0, 1)

- alpha = beta = 2: Symmetric, peaked at 0.5

- alpha \> beta: Skewed toward 1

- alpha \< beta: Skewed toward 0

## Examples

``` r
prior_beta(1, 1)   # Uniform
#> Beta(1.00, 1.00)  [mean = 0.50]
prior_beta(2, 2)   # Symmetric, centered at 0.5
#> Beta(2.00, 2.00)  [mean = 0.50]
prior_beta(5, 2)   # Skewed toward 1 (for high autocorrelation)
#> Beta(5.00, 2.00)  [mean = 0.71]
```
