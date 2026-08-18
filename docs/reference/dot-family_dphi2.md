# Second-order dispersion derivatives for a family, or `NULL`

Second-order dispersion derivatives for a family, or `NULL`

## Usage

``` r
.family_dphi2(family)
```

## Arguments

- family:

  Family name, as registered in `.FAMILY_OPS`.

## Value

A list with `dloglik2`, `dscore2`, `dweight2` and `dweight_deta`, or
`NULL` when the family's phi Hessian has not been derived. `NULL` is a
refusal, not a fixed-value fallback: it hands the phi Hessian to the
differencing stencil rather than reporting an inexact closed form.
