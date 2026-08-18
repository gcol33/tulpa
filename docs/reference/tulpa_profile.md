# Profile the inner Laplace solve by phase

Times the sparse joint Laplace solver one phase at a time – scatter (the
Hessian and gradient assembly), factorize (numeric Cholesky), eta, line
search, and the rest – and returns the breakdown as a data frame. The
accumulator aggregates across the parallel outer-grid worker threads, so
the reported times cover the whole fit rather than only the calling
thread.

## Usage

``` r
tulpa_profile(expr, sort = TRUE)
```

## Arguments

- expr:

  An expression that runs a fit (for example a call to
  [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)).
  Evaluated once, after the profile counters are reset.

- sort:

  Logical; order rows by descending time. Default `TRUE`.

## Value

A data frame with one row per phase and columns `phase`, `seconds`,
`calls`, `ms_per_call` (mean wall time per phase call), and `share`
(fraction of total timed seconds). The fit result is attached as the
`"value"` attribute.

## Details

Use it to settle where a per-cell solve spends its time, e.g. whether a
slow joint `occu_cover()` fit is bound by the assembly scatter or the
Cholesky factorize:


      p <- tulpa_profile(
        tulpa_nested_laplace_joint(..., control = list(integration = "ccd"))
      )
      print(p)            # rows ordered by time; scatter vs factorize at top
      fit <- attr(p, "value")

## Examples

``` r
# \donttest{
set.seed(1)
n <- 200L; X <- cbind(1, rnorm(n))
y <- rbinom(n, 1, plogis(X %*% c(0, 0.5)))
tulpa_profile(tulpa_laplace(y, rep(1L, n), X, family = "binomial"))
#>            phase seconds calls ms_per_call share
#> 1  pattern_build       0     0           0     0
#> 2           prep       0     0           0     0
#> 3            eta       0     0           0     0
#> 4        scatter       0     0           0     0
#> 5        analyze       0     0           0     0
#> 6      factorize       0     0           0     0
#> 7          solve       0     0           0     0
#> 8    line_search       0     0           0     0
#> 9        log_det       0     0           0     0
#> 10 log_lik_prior       0     0           0     0
# }
```
