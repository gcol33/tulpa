# Comprehensive Diagnostic Summary

Provides a comprehensive diagnostic report for a tulpa model, combining
convergence metrics, divergence information, and actionable
recommendations.

## Usage

``` r
diagnostic_summary(fit, quiet = FALSE)
```

## Arguments

- fit:

  A `tulpa_fit` object.

- quiet:

  Logical; if TRUE, suppress printed output (default: FALSE).

## Value

A list with class `tulpa_diagnostic_summary` containing:

- status:

  Overall status: "PASS", "WARN", or "FAIL"

- n_divergent:

  Number of divergent transitions

- divergent_pct:

  Percentage of divergent transitions

- worst_rhat:

  Data frame of parameters with worst Rhat

- worst_ess:

  Data frame of parameters with worst ESS

- e_bfmi:

  E-BFMI value (HMC only)

- pareto_k, quad_ess:

  approximation fits only: the outer PSIS k-hat, or the grid quadrature
  ESS when no k-hat was produced

- pareto_k_declined:

  approximation fits only, and only when there is no k-hat: WHY –
  `"not_requested"` and `"unguessable_axis: <axis>"` are benign or
  permanent, `"degenerate_proposal"` and `"grid_too_small"` are signals
  about the fit, and `"internal_inconsistency"` is an engine bug and
  raises the status to `"WARN"`

- inner_skew_max, inner_skew_declined:

  approximation fits only: the largest scored inner-Laplace `|gamma_3|`,
  or why nothing was scored (`"coupled_arm"` marks arms the inner layer
  could score neither per observation nor through the cell tensor)

- axis_fields_dropped:

  data frame of grid axes the fit's own resolved path could not read and
  dropped as engine defaults: one row per field, with the block, family,
  path and the axis that path integrated instead. Absent whenever every
  supplied axis was used

- recommendations:

  Character vector of recommendations

## See also

[`check_diagnostics()`](https://gillescolling.com/tulpa/reference/check_diagnostics.md),
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md),
[`plot_diagnostics()`](https://gillescolling.com/tulpa/reference/plot_diagnostics.md)

## Examples

``` r
# \donttest{
set.seed(123)
df <- data.frame(x = rnorm(60))
df$y <- rpois(60, exp(0.5 + 0.3 * df$x))
fit <- tulpa(y ~ x, data = df, family = "poisson", mode = "hmc",
             control = list(n_iter = 500L, warmup = 250L, n_chains = 2L,
                            seed = 1L))
ds <- diagnostic_summary(fit)
#> 
#> === tulpa Diagnostic Summary ===
#> 
#> Backend: hmc 
#> Status: WARN
#> 
#> Divergent transitions: 0
#> 
#> Parameters with Rhat > 1.01:
#>    parameter  rhat
#>  (Intercept) 1.015
#> 
#> Parameters with ESS < 400:
#>    parameter ess_bulk ess_tail
#>            x      175      198
#>  (Intercept)      183      174
#> 
#> Recommendations:
#>   - Rhat > 1.01: Run more iterations or chains 
#>   - ESS < 400: Run more iterations or use thinning 
print(ds)
#> 
#> === tulpa Diagnostic Summary ===
#> 
#> Backend: hmc 
#> Status: WARN
#> 
#> Divergent transitions: 0
#> 
#> Parameters with Rhat > 1.01:
#>    parameter  rhat
#>  (Intercept) 1.015
#> 
#> Parameters with ESS < 400:
#>    parameter ess_bulk ess_tail
#>            x      175      198
#>  (Intercept)      183      174
#> 
#> Recommendations:
#>   - Rhat > 1.01: Run more iterations or chains 
#>   - ESS < 400: Run more iterations or use thinning 
# }
```
