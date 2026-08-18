# Geweke Convergence Test

Performs Geweke's convergence diagnostic, comparing the mean of the
first portion of a chain to the last portion. Useful for single-chain
diagnostics.

## Usage

``` r
geweke_test(fit, frac1 = 0.1, frac2 = 0.5, pars = NULL)
```

## Arguments

- fit:

  A `tulpa_fit` object.

- frac1:

  Fraction of chain for first window (default: 0.1).

- frac2:

  Fraction of chain for second window (default: 0.5).

- pars:

  Character vector of parameter names (default: all main params).

## Value

A data frame with columns: parameter, z_score, p_value.

## Details

The Geweke test computes a z-score comparing the means of early and late
portions of a chain. Large z-scores (\|z\| \> 2) indicate the chain has
not converged.

## See also

[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md),
[`check_diagnostics()`](https://gillescolling.com/tulpa/reference/check_diagnostics.md)

## Examples

``` r
# See plot_rhat() examples for fitting a model
# geweke_test(fit)
```
