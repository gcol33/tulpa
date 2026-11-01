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
dim(tulpa_draws_array(fit))   # [iteration, chain, parameter]
#> [1] 200   2   2
# }
```
