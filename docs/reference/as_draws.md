# Posterior draws in the posterior package's format

Convert a fit's posterior draws to a `posterior` draws object.
`as_draws()` returns the `draws_array` shape; `as_draws_array()`,
`as_draws_matrix()`, `as_draws_df()` and `as_draws_rvars()` return
theirs. When the `posterior` package is installed these are also
registered against its generics, so `posterior::as_draws(fit)` works.

## Usage

``` r
as_draws(x, ...)

# S3 method for class 'tulpa_fit'
as_draws(x, n_draws = NULL, seed = NULL, ...)

as_draws_array(x, ...)

# S3 method for class 'tulpa_fit'
as_draws_array(x, n_draws = NULL, seed = NULL, ...)

as_draws_matrix(x, ...)

# S3 method for class 'tulpa_fit'
as_draws_matrix(x, n_draws = NULL, seed = NULL, ...)

as_draws_df(x, ...)

# S3 method for class 'tulpa_fit'
as_draws_df(x, n_draws = NULL, seed = NULL, ...)

as_draws_rvars(x, ...)

# S3 method for class 'tulpa_fit'
as_draws_rvars(x, n_draws = NULL, seed = NULL, ...)
```

## Arguments

- x:

  A `tulpa_fit` object.

- ...:

  Passed to the corresponding `posterior` converter.

- n_draws:

  Number of draws to synthesize from the Gaussian approximation for a
  fit that carries none. `NULL` (default) errors on such a fit rather
  than silently approximating. Ignored, with a warning, when the fit
  already carries draws.

- seed:

  Optional integer seed for the synthesis. The RNG state is restored
  afterwards.

## Value

A `posterior` draws object of the requested shape.

## Details

Fits differ in whether they carry draws at all. Sampler and
nested-Laplace fits do, and convert directly. A Gaussian-approximation
fit (`mode = "laplace"`, `mode = "eb"`) carries a mode and a precision
instead, and converting it means *drawing from the approximation* –
which is a modelling decision, not a format change, because every
downstream `posterior` summary would then treat a normal approximation
as a posterior sample. So it is opt-in: pass `n_draws` to synthesize
that many draws from `N(coef(object), vcov(object))`, or get an error
naming the alternative. Synthesized draws form a single chain and cover
the fixed effects only.

## See also

[`tulpa_draws_array()`](https://gillescolling.com/tulpa/reference/tulpa_draws_array.md)
for the base R array without the dependency,
[`posterior_sample()`](https://gillescolling.com/tulpa/reference/posterior_sample.md)
for the raw matrix.

## Examples

``` r
# \donttest{
set.seed(1)
df <- data.frame(x = rnorm(80))
df$y <- rpois(80, exp(0.5 + 0.3 * df$x))
fit <- tulpa(y ~ x, data = df, family = "poisson")
if (requireNamespace("posterior", quietly = TRUE)) {
  posterior::summarise_draws(as_draws(fit))
}
# }
```
