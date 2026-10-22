# Validate that a fit used the expected mode

Ensures the fit object was created with the expected inference mode.
Useful for enforcing mode requirements in downstream analysis.

A fit matches a mode when the mode names either the backend that ran
(`fit$backend`, e.g. `"laplace"`, `"mala"`, `"re_cov_gibbs"`) or the
tier it ran in (`fit$inference_mode`: `"exact"`, `"structured"`,
`"optimized"`). So a fit made with `mode = "laplace"` validates against
both `"laplace"` and `"structured"`, and a fit made with `mode = "auto"`
validates against the backend and the tier auto chose. A request that
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) overrode
(see `fit$mode_overridden`) validates against what ran, not what was
asked for.

## Usage

``` r
validate_mode(fit, expected_mode, error = TRUE)
```

## Arguments

- fit:

  A tulpa_fit object

- expected_mode:

  Mode(s) that should have been used: a character vector of backend and
  / or tier names, any one of which is accepted.

- error:

  If TRUE (default), error on mismatch. If FALSE, return logical.

## Value

If error = FALSE, returns TRUE/FALSE. Otherwise errors on mismatch.

## Examples

``` r
# \donttest{
set.seed(1)
d <- data.frame(x = rnorm(60))
d$y <- rpois(60, exp(0.5 + 0.3 * d$x))
fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "hmc",
             control = list(n_iter = 400L, warmup = 200L, n_chains = 2L,
                            seed = 1L))
validate_mode(fit, "exact")                      # the tier hmc runs in
validate_mode(fit, "structured", error = FALSE)  # FALSE
# }
```
