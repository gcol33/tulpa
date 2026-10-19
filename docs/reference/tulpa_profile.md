# Profile a fit by solver phase

Times a fit one solver phase at a time – the inner Newton solve's eta,
scatter (the Hessian and gradient assembly), factorize (Cholesky), line
search and final pass, the nested-Laplace outer grid's cell solves, and
the NUTS sampler's iterations and gradient evaluations – and returns the
breakdown as a data frame. The accumulator aggregates across the
parallel outer-grid worker threads and across concurrently run chains,
so the reported times cover the whole fit rather than only the calling
thread (and can therefore exceed its wall time).

## Usage

``` r
tulpa_profile(expr, sort = TRUE)
```

## Arguments

- expr:

  An expression that runs a fit (for example a call to
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  or [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)).
  Evaluated once, after the profile counters are reset and timing is
  switched on.

- sort:

  Logical; order rows by descending time. Default `TRUE`.

## Value

A data frame with one row per phase and columns `phase`, `seconds`,
`calls`, `ms_per_call` (mean wall time per phase call), and `share`
(fraction of the total timed seconds of the leaf phases; `NA` for an
enclosing phase). The fit result is attached as the `"value"` attribute.

## Details

Use it to settle where a fit spends its time, e.g. whether a slow joint
`occu_cover()` fit is bound by the assembly scatter or the Cholesky
factorize:


      p <- tulpa_profile(
        tulpa_nested_laplace_joint(..., control = list(integration = "ccd"))
      )
      print(p)            # rows ordered by time; scatter vs factorize at top
      fit <- attr(p, "value")

Timing is off outside `tulpa_profile()`: an ordinary fit pays one flag
read per phase scope and nothing else.

## Which fits are timed

- The single-response Laplace solve behind
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md),
  `tulpa(mode = "laplace")` and every per-cell inner solve of
  [`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
  / `tulpa(mode = "nested_laplace")`: phases `eta`, `scatter`,
  `factorize`, `line_search`, `log_det`, `log_lik_prior`,
  `hessian_extract` and `inner_diagnostics`.

- The joint solver behind
  [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md),
  on both its dense and its sparse path (the sparse one adds
  `pattern_build` and `prep`).

- The nested-Laplace outer grid, single-response and joint:
  `outer_grid_cell`, one call per solved cell.

- The samplers: `nuts_warmup` and `nuts_sampling`, one call per NUTS
  iteration per chain, and `gradient`, one call per log-density gradient
  evaluation (NUTS leapfrog steps, static HMC, the step-size search and
  the other samplers that evaluate the gradient through the engine).

An expression that reaches none of these – a Gibbs or quadrature backend
that runs neither an engine Newton solve nor an engine gradient, or no
fit at all – records nothing, and `tulpa_profile()` then warns rather
than returning an all-zero table as if it were a measurement.

## Enclosing phases

`outer_grid_cell`, `nuts_warmup` and `nuts_sampling` ENCLOSE the leaf
phases timed inside them (a cell's inner Newton phases, an iteration's
gradient evaluations), so their seconds overlap the leaves' rather than
adding to them. Their `share` is `NA`; the shares of the leaf phases sum
to one.

## Examples

``` r
# \donttest{
set.seed(1)
n <- 200L; X <- cbind(1, rnorm(n))
y <- rbinom(n, 1, plogis(X %*% c(0, 0.5)))
tulpa_profile(tulpa_laplace(y, rep(1L, n), X, family = "binomial"))
#>                phase  seconds calls  ms_per_call       share
#> 1        line_search 1.14e-05     3 0.0038000000 0.502202643
#> 2            scatter 9.50e-06     4 0.0023750000 0.418502203
#> 3          factorize 5.00e-07     3 0.0001666667 0.022026432
#> 4    hessian_extract 5.00e-07     1 0.0005000000 0.022026432
#> 5                eta 3.00e-07     4 0.0000750000 0.013215859
#> 6      log_lik_prior 2.00e-07     1 0.0002000000 0.008810573
#> 7  inner_diagnostics 2.00e-07     1 0.0002000000 0.008810573
#> 8            log_det 1.00e-07     1 0.0001000000 0.004405286
#> 9      pattern_build 0.00e+00     0 0.0000000000 0.000000000
#> 10              prep 0.00e+00     0 0.0000000000 0.000000000
#> 11           analyze 0.00e+00     0 0.0000000000 0.000000000
#> 12             solve 0.00e+00     0 0.0000000000 0.000000000
#> 13          gradient 0.00e+00     0 0.0000000000 0.000000000
#> 14   outer_grid_cell 0.00e+00     0 0.0000000000          NA
#> 15       nuts_warmup 0.00e+00     0 0.0000000000          NA
#> 16     nuts_sampling 0.00e+00     0 0.0000000000          NA
# }
```
