# Dispersion derivatives for a family, or `NULL`

Dispersion derivatives for a family, or `NULL`

## Usage

``` r
.family_dphi(family)
```

## Arguments

- family:

  Family name, as registered in `.FAMILY_OPS`.

## Value

A list with `dloglik`, `dscore` and `dweight`, or `NULL` when the family
has no free dispersion or none has been derived. `NULL` is a refusal to
estimate, never a signal to fall back to a fixed value silently.
