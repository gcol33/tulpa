# The copy-scale axis's slab measure

`tulpa_hyper_copy_slab_density()` returns the proper density tulpa uses
for the copy scale's continuum: an exponential, the penalized-complexity
prior for a scale parameter with its base model at zero (Simpson et al.
2017). The rate is set by putting 5% of the prior mass above `upper`, so
a caller that reads this off a fit's own declared grid gets the exact
rate the outer integration used – rather than restating a PC-prior rate
that could silently drift from it.

`tulpa_hyper_check_copy_slab()` validates a `copy_slab` argument –
`"exponential"` (the default) or `"flat"`, the two continuum measures
tulpa supports for a copy scale's non-atom mass – defaulting `NULL` to
`"exponential"` and erroring on anything else. Intended so a consumer
package's own `copy_slab` argument stays in sync with tulpa's own
accepted choices rather than restating them.

## Usage

``` r
tulpa_hyper_copy_slab_density(upper)

tulpa_hyper_check_copy_slab(x)
```

## Arguments

- upper:

  The largest declared node of the copy-scale axis.

- x:

  A `copy_slab` value: `NULL`, `"exponential"`, or `"flat"`.

## Value

`tulpa_hyper_copy_slab_density()`: a function
`log p(x) = log(lambda) - lambda * x`, or `NULL` if `upper` is not a
finite positive number. `tulpa_hyper_check_copy_slab()`: `x`, defaulted
to `"exponential"` when `NULL`.

## See also

[`tulpa_joint_axis_specs_from_grid()`](https://gillescolling.com/tulpa/reference/tulpa_joint_axis_specs_from_grid.md)

## Examples

``` r
lp <- tulpa_hyper_copy_slab_density(upper = 2)
lp(c(0.1, 1, 2))
#> [1]  0.2542549 -1.0938246 -2.5916908
tulpa_hyper_check_copy_slab("exponential")
#> [1] "exponential"
try(tulpa_hyper_check_copy_slab("uniform"))
#> Error : `copy_slab` must be one of "exponential" or "flat".
```
