# The copy-scale axis's PC-prior density

The proper density tulpa uses for the copy scale's continuum: an
exponential, the penalized-complexity prior for a scale parameter with
its base model at zero (Simpson et al. 2017). The rate is set by putting
5% of the prior mass above `upper`, so a caller that reads this off a
fit's own declared grid gets the exact rate the outer integration used –
rather than restating a PC-prior rate that could silently drift from it.

## Usage

``` r
tulpa_hyper_copy_slab_density(upper)
```

## Arguments

- upper:

  The largest declared node of the copy-scale axis.

## Value

A function `log p(x) = log(lambda) - lambda * x`, or `NULL` if `upper`
is not a finite positive number.

## See also

[`tulpa_hyper_check_copy_slab()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_check_copy_slab.md),
[`tulpa_joint_axis_specs_from_grid()`](https://gillescolling.com/tulpa/reference/tulpa_joint_axis_specs_from_grid.md)
