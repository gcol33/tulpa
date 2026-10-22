# Pool multiple imputation draws via Rubin's rules

Pools per-imputation block fits via Rubin's rules. When every draw also
carries a per-coefficient skewness vector `$gamma`, the third cumulant
is pooled by the law of total cumulants \$\$\kappa_3 = E\[\kappa_3(X \|
k)\] + 3\\\mathrm{Cov}(\mu_k, \sigma_k^2) + \kappa_3(\mu_k),\$\$ giving
a pooled skewness `$gamma` alongside the usual `$mean` / `$se`. If any
draw is missing `$gamma`, the third-cumulant path is skipped.

`kappa3` is the third cumulant of the equal-weight mixture of the draws,
so `gamma` is standardized by that mixture's variance,
`mean(se^2) + mean((beta - mean(beta))^2)`, not by `V_total`: Rubin's
`(1 + 1/K)` inflation of the between-draw variance is a correction to
the standard error for finite `K`, not a moment of the mixture.

## Usage

``` r
rubins_pool(draws)
```

## Arguments

- draws:

  List of K draws. Each draw is a named list of submodel results, each
  with `beta` (numeric vector), `se` (numeric vector, same length as
  `beta`), and optionally `gamma` (numeric vector, same length as
  `beta`). Every draw of a submodel must carry the same number of
  coefficients. A submodel absent from some draws is pooled over the
  draws that carry it; one carried by a single draw is skipped with a
  warning.

## Value

A named list of pooled submodel summaries (every submodel any draw
carries), each with `mean`, `se`, `V_within`, `V_between`, `V_total`,
`K` (number of draws that contributed), and when applicable `gamma`
(pooled skewness) and `kappa3` (pooled third cumulant).

## Examples

``` r
# Three imputation draws of a two-coefficient submodel.
draws <- list(
  list(m = list(beta = c(1.0, 0.50), se = c(0.2, 0.10))),
  list(m = list(beta = c(1.2, 0.45), se = c(0.2, 0.10))),
  list(m = list(beta = c(0.9, 0.55), se = c(0.2, 0.10)))
)
pooled <- rubins_pool(draws)
pooled$m$mean
pooled$m$se
```
