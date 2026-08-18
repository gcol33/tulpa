# Pool multiple imputation draws via Rubin's rules

Pools per-imputation block fits via Rubin's rules. When every draw also
carries a per-coefficient skewness vector `$gamma`, the third cumulant
is pooled by the law of total cumulants \$\$\kappa_3 = E\[\kappa_3(X \|
k)\] + 3\\\mathrm{Cov}(\mu_k, \sigma_k^2) + \kappa_3(\mu_k),\$\$ giving
a pooled skewness `$gamma` alongside the usual `$mean` / `$se`. If any
draw is missing `$gamma`, the third-cumulant path is skipped.

## Usage

``` r
rubins_pool(draws)
```

## Arguments

- draws:

  List of K draws. Each draw is a named list of submodel results, each
  with `beta` (numeric vector), `se` (numeric vector), and optionally
  `gamma` (numeric vector, same length as `beta`).

## Value

A named list of pooled submodel summaries, each with `mean`, `se`,
`V_within`, `V_between`, `V_total`, `K` (number of draws that
contributed), and when applicable `gamma` (pooled skewness) and `kappa3`
(pooled third cumulant).
