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

## Examples

``` r
# \donttest{
set.seed(1)
d <- data.frame(x = rnorm(60))
d$y <- rpois(60, exp(0.5 + 0.3 * d$x))
fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "hmc",
             control = list(n_iter = 400L, warmup = 200L, n_chains = 2L,
                            seed = 1L))
#> Warning: tulpa(): the 'hmc' chain has not converged -- x: Rhat 1.04, bulk-ESS 72. Its estimates and intervals are not yet the posterior. Run longer (control$n_iter / control$warmup), or use mode = 'hmc'; see check_diagnostics().
n_divergent(fit)
#> [1] 0
# }
```
