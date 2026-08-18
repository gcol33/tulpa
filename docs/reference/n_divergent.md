# Number of divergent transitions

Counts divergent transitions recorded by an HMC/NUTS fit, reading
whichever field the backend populated (`$diagnostics$n_divergent`,
`$diagnostics$divergent_idx`, `$diagnostics$divergent`, or the top-level
`$n_divergent` / `$divergent`).

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
