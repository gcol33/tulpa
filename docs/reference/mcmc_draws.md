# MCMC chain draws from a fit

Returns a fit's posterior draws only when they form a genuine MCMC chain
(`$draws_kind == "chain"`, or an untagged legacy fit); for an i.i.d. /
approximation fit (nested Laplace, VI, SMC, ...) it returns `NULL`,
because chain diagnostics do not apply. This is the accessor
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
gates on. For the provenance-agnostic posterior sample used by
summaries, see
[`posterior_sample()`](https://gillescolling.com/tulpa/reference/posterior_sample.md).

## Usage

``` r
mcmc_draws(fit)
```

## Arguments

- fit:

  A `tulpa_fit` (or subclass) carrying posterior `$draws`.

## Value

The chain draws matrix/array, or `NULL` for a non-chain fit.

## See also

[`posterior_sample()`](https://gillescolling.com/tulpa/reference/posterior_sample.md),
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
