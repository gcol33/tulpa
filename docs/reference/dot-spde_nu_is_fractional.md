# Is an SPDE smoothness `nu` fractional (rational path) or integer (exact path)?

Is an SPDE smoothness `nu` fractional (rational path) or integer (exact
path)?

## Usage

``` r
.spde_nu_is_fractional(nu)
```

## Arguments

- nu:

  Matern smoothness; `alpha = nu + 1` in 2D.

## Value

`TRUE` when `alpha` is non-integer (the rational SPDE path).
