# Validate/default a copy-scale slab measure choice

Validates a `copy_slab` argument – `"exponential"` (the default) or
`"flat"`, the two continuum measures tulpa supports for a copy scale's
non-atom mass – defaulting `NULL` to `"exponential"` and erroring on
anything else. Intended so a consumer package's own `copy_slab` argument
stays in sync with tulpa's own accepted choices rather than restating
them.

## Usage

``` r
tulpa_hyper_check_copy_slab(x)
```

## Arguments

- x:

  A `copy_slab` value: `NULL`, `"exponential"`, or `"flat"`.

## Value

`x`, defaulted to `"exponential"` when `NULL`.

## See also

[`tulpa_hyper_copy_slab_density()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_copy_slab_density.md)
