# Validate the SPDE smoothness parameter

`nu` must be a single finite positive number. Integer `nu` (1, 2, 3,
...) gives an exact FEM construction; fractional `nu` uses the rational
SPDE approximation (see
[`rational_spde_coefficients()`](https://gillescolling.com/tulpa/reference/rational_spde_coefficients.md)).

## Usage

``` r
.validate_spde_nu(nu)
```

## Arguments

- nu:

  Candidate smoothness parameter.

## Value

Invisibly `TRUE`; raises an error on an invalid `nu`.
