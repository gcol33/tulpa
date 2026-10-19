# Geweke convergence test

Performs Geweke's convergence diagnostic, comparing the mean of the
first portion of a chain to the last portion. It is a within-chain
diagnostic, so on a multi-chain fit it is computed for each chain
separately.

## Usage

``` r
geweke_test(fit, frac1 = 0.1, frac2 = 0.5, pars = NULL)

# S3 method for class 'tulpa_geweke'
print(x, ...)
```

## Arguments

- fit:

  A `tulpa_fit` object.

- frac1:

  Fraction of each chain for the first window (default: 0.1).

- frac2:

  Fraction of each chain for the second window (default: 0.5). Both lie
  in `(0, 1)` and the windows may not overlap: `frac1 + frac2 <= 1`.

- pars:

  Character vector of parameter names (default: all main params).

- x:

  A `tulpa_geweke` result.

- ...:

  Ignored.

## Value

A data frame of class `tulpa_geweke` with columns `chain`, `parameter`,
`z_score` and `p_value`, one row per chain and parameter.

## Details

The Geweke test computes a z-score comparing the means of early and late
portions of a chain. Large z-scores (\|z\| \> 2) indicate the chain has
not converged. The chains are never pooled: concatenated, the "first"
window would come from the first chain and the "last" from another,
which compares two chains rather than a chain's start with its end (use
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
for the between-chain R-hat). A parameter that is constant within a
chain has an undefined z-score, reported as `NA`.

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
#>  chain   parameter z_score p_value
#>      1 (Intercept)   0.787  0.4314
#>      1           x  -2.611  0.0090
#>      2 (Intercept)  -0.727  0.4671
#>      2           x   2.678  0.0074
#> 
#> Warning: 2 chain/parameter pair(s) have |z| > 2 (potential non-convergence)
# }
```
