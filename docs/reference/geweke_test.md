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
# \donttest{
set.seed(123)
df <- data.frame(x = rnorm(60))
df$y <- rpois(60, exp(0.5 + 0.3 * df$x))
fit <- tulpa(y ~ x, data = df, family = "poisson", mode = "hmc",
             control = list(n_iter = 500L, warmup = 250L, n_chains = 2L,
                            seed = 1L))
geweke_test(fit)
#> Geweke Convergence Diagnostic
#> =============================
#> 
#>    parameter z_score p_value
#>  (Intercept)   1.638  0.1014
#>            x  -3.688  0.0002
#> 
#> Warning: 1 parameter(s) have |z| > 2 (potential non-convergence)
# }
```
