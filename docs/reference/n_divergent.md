# Number of divergent transitions

Counts divergent transitions recorded by an HMC/NUTS fit, reading
whichever field the backend populated (the top-level `$divergent` flag
vector every sampler in this package writes, or
`$diagnostics$divergent`, `$diagnostics$divergent_idx`,
`$diagnostics$n_divergent`, `$n_divergent`). It reads the same record
[`plot_divergences()`](https://gillescolling.com/tulpa/reference/plot_divergences.md)
locates the divergent rows from, so the two always agree on one fit.

## Usage

``` r
n_divergent(fit)
```

## Arguments

- fit:

  A `tulpa_fit` object.

## Value

Integer count of divergent transitions (0 if none are recorded).

## See also

[`plot_divergences()`](https://gillescolling.com/tulpa/reference/plot_divergences.md),
[`check_diagnostics()`](https://gillescolling.com/tulpa/reference/check_diagnostics.md)
