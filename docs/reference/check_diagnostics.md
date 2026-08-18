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
# See plot_rhat() examples for fitting a model
# check_diagnostics(fit)
```
