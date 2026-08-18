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

  approximation fits only, and only when there is no k-hat: WHY
  (gcol33/tulpa#295) – `"not_requested"` and
  `"unguessable_axis: <axis>"` are benign or permanent,
  `"degenerate_proposal"` and `"grid_too_small"` are signals about the
  fit, and `"internal_inconsistency"` is an engine bug and raises the
  status to `"WARN"`

- inner_skew_max, inner_skew_declined:

  approximation fits only: the largest scored inner-Laplace `|gamma_3|`,
  or why nothing was scored (gcol33/tulpa#296 – `"coupled_arm"` marks
  arms the inner layer could score neither per observation nor through
  the cell tensor)

- axis_fields_dropped:

  data frame of grid axes the fit's own resolved path could not read and
  dropped as engine defaults (gcol33/tulpa#352): one row per field, with
  the block, family, path and the axis that path integrated instead.
  Absent whenever every supplied axis was used

- recommendations:

  Character vector of recommendations

## See also

[`check_diagnostics()`](https://gillescolling.com/tulpa/reference/check_diagnostics.md),
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md),
[`plot_diagnostics()`](https://gillescolling.com/tulpa/reference/plot_diagnostics.md)

## Examples

``` r
# See plot_rhat() examples for fitting a model
# ds <- diagnostic_summary(fit)
# print(ds)
```
