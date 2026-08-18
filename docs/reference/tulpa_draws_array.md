# Posterior draws as a 3D array

Assembles a fitted model's posterior draws into an
`[iteration, chain, parameter]` array – the layout expected by
`bayesplot` and analogous to
[`posterior::as_draws_array()`](https://mc-stan.org/posterior/reference/draws_array.html).
Multiple chains are recognised from a 3D draws array, a `$chain_id` row
map, or an `$n_chains` count over chain-major rows; a single pooled
chain yields a one-chain array.

## Usage

``` r
tulpa_draws_array(fit)
```

## Arguments

- fit:

  A `tulpa_fit` (or subclass) carrying posterior `$draws`.

## Value

A numeric array with dimensions `[n_iter, n_chain, n_param]` and the
parameter names on the third dimension, or `NULL` if the fit carries no
draws. Chains of unequal length are truncated to the shortest.

## Details

As with
[`posterior_sample()`](https://gillescolling.com/tulpa/reference/posterior_sample.md),
a fit that carries no draws returns `NULL` with a message naming its
backend and the representation it carries instead.

## See also

[`posterior_sample()`](https://gillescolling.com/tulpa/reference/posterior_sample.md),
[`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md),
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
