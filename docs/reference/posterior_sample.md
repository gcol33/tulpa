# Posterior parameter sample from a fit

Returns a fit's posterior draws for summary purposes (quantiles, derived
quantities, density plots), regardless of how they were produced – an
MCMC chain, a nested-Laplace node mixture, or a variational sample all
answer here. For the chain-only view used by convergence diagnostics,
see
[`mcmc_draws()`](https://gillescolling.com/tulpa/reference/mcmc_draws.md).

## Usage

``` r
posterior_sample(fit)
```

## Arguments

- fit:

  A `tulpa_fit` (or subclass) carrying posterior `$draws`.

## Value

The posterior draws matrix/array, or `NULL` if the fit carries none.

## Details

A fit that carries no draws returns `NULL` with a message naming its
backend, the posterior representation it carries instead, and the
accessor that turns that representation into draws where one exists – an
absent draws matrix is a property of the backend, not a failure of the
accessor, and saying which is the difference between a diagnosable
answer and a bare `NULL`.

## See also

[`mcmc_draws()`](https://gillescolling.com/tulpa/reference/mcmc_draws.md),
[`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md),
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)

## Examples

``` r
# \donttest{
set.seed(1)
df <- data.frame(x = rnorm(80))
df$y <- rpois(80, exp(0.5 + 0.3 * df$x))
fit <- tulpa(y ~ x, data = df, family = "poisson")
dim(posterior_sample(fit))
#> [1] 1000    2
# }
```
