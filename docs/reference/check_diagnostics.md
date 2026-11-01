# Quick convergence check

Runs the core convergence diagnostics on a fit and reports whether Rhat,
bulk-ESS, and divergence thresholds are all met. A terse companion to
[`diagnostic_summary()`](https://gillescolling.com/tulpa/reference/diagnostic_summary.md)
for use in scripts and tests.

## Usage

``` r
check_diagnostics(
  fit,
  rhat_threshold = 1.01,
  ess_threshold = 400,
  quiet = FALSE
)
```

## Arguments

- fit:

  A `tulpa_fit` object.

- rhat_threshold:

  Maximum acceptable Rhat (default 1.01).

- ess_threshold:

  Minimum acceptable bulk-ESS (default 400).

- quiet:

  Logical; if TRUE, suppress messages (default FALSE).

## Value

Invisibly, `TRUE` if all checks pass, `FALSE` if any fail, or `NA` for a
non-chain (approximation) fit where Rhat/ESS do not apply.

## See also

[`diagnostic_summary()`](https://gillescolling.com/tulpa/reference/diagnostic_summary.md),
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md),
[`n_divergent()`](https://gillescolling.com/tulpa/reference/n_divergent.md)

## Examples

``` r
# \donttest{
set.seed(123)
df <- data.frame(x = rnorm(60))
df$y <- rpois(60, exp(0.5 + 0.3 * df$x))
fit <- tulpa(y ~ x, data = df, family = "poisson", mode = "hmc",
             control = list(n_iter = 500L, warmup = 250L, n_chains = 2L,
                            seed = 1L))
check_diagnostics(fit)
#> Convergence warnings:
#>   - 1 parameter(s) with Rhat > 1.01
#>   - 2 parameter(s) with bulk-ESS < 400
# }
```
