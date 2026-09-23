# Areal spatially varying coefficient field

Declare one or more areal (CAR / Besag) fields over a graph from an
lme4-style bar formula. The bar's left-hand side lists the coefficients
that vary smoothly over the graph; the right-hand side names the
graph-node index. Each coefficient becomes an independent CAR field,
entering the linear predictor scaled by that coefficient's
per-observation design value:

\$\$\eta_i = \ldots + \sum_c X\_{ic}\\ z^{(c)}\_{g_i},\$\$

where \\g_i\\ is the graph node of observation \\i\\, \\z^{(c)}\\ is the
CAR field for design column \\c\\, and \\X\_{ic}\\ is that column's
value at observation \\i\\. The intercept column is all ones, so
`~ 1 || cell` is the ordinary spatial intercept field; a covariate
column (e.g. `time`) gives a spatially varying slope on that covariate
(a per-region trend).

Use it inline in a
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) model
formula, the same way a random-effect bar is written:


    y ~ time + spatial(graph = adj, formula = ~ 1 + time || cell) + (1 | site)

## Usage

``` r
spatial(graph, formula, proper = FALSE, shared = NULL, by = NULL)
```

## Arguments

- graph:

  Symmetric adjacency matrix of the spatial graph (`[n_node x n_node]`,
  dense or sparse). One CAR field is defined over its nodes per
  coefficient.

- formula:

  One-sided formula carrying a grouping bar, e.g. `~ 1 + time || cell`.
  The left-hand side is expanded with
  [`stats::model.matrix()`](https://rdrr.io/r/stats/model.matrix.html)
  into one CAR field per column (`1` = intercept field; a covariate =
  spatially varying slope on it; `0 +` drops the intercept). The
  right-hand side must be a single bare column naming the graph node.
  The double bar `||` builds independent fields (each its own
  precision); a single bar `|` builds correlated fields – a separable
  multivariate CAR where the per-cell coefficient vector shares a
  cross-covariance `Sigma` (covariance `Sigma (x) Q^-1`), so the
  intercept-slope correlation `rho` is shared across the graph.

- proper:

  Logical; `FALSE` (default) builds intrinsic CAR (ICAR / Besag) fields
  with the sum-to-zero constraint (`rho` fixed at 1). `TRUE` builds
  proper CAR fields, each with its own precision `Q = D - rho_car W` and
  the spatial autocorrelation `rho_car` estimated from the data (one
  `(sigma, rho_car)` pair per field).
  [`summary()`](https://rdrr.io/r/base/summary.html) and
  [`print()`](https://rdrr.io/r/base/print.html) report the per-field
  `rho_car`. Independent (`||`) only; correlated proper CAR (a single
  `|` with `proper = TRUE`) is a separate model.

- shared:

  Optional shared-effect handle, passed through to the field blocks (see
  the model docs). Default `NULL` (shared).

- by:

  Optional replicated-CAR factor: a bare column name (or a string)
  naming a factor in the model data. With `L` levels it builds one
  independent copy of the whole field per level – the field over the
  block-diagonal Kronecker graph `I_L (x) Q` (`L` disjoint copies of the
  graph) – with the hyperparameters shared across levels (one `Sigma`
  for `|`; one `(sigma[, rho_car])` for `||`). This is `INLA`'s
  `replicate =` / `mgcv`'s `s(cell, by = ...)` generalised to the
  varying-coefficient bar, and is orthogonal to the bar character: `|` /
  `||` sets the covariance among the coefficient columns within a field,
  while `by` sets how many independent replicates of the whole field
  exist. Default `NULL` (one field). Correlated proper CAR (`|` with
  `proper = TRUE`) stays out of scope with or without `by`.

## Value

A `tulpa_spatial_field` object describing the field(s). It is expanded
into one CAR block per design column at fit time, when the data is
available.

## Details

Each field is independent (its own precision), matching `INLA`'s two
separate `f(cell, model = "besag")` and
`f(cell.slope, time, model = "besag")` fields. Nesting (`a / b`),
interaction, or expression grouping is rejected: the grouping must be a
single graph node. Add ordinary nested random effects, e.g.
`(1 | site)`, as separate terms.

## See also

[`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md)
for a single areal field passed via the `spatial =` argument,
[`spatial_svc()`](https://gillescolling.com/tulpa/reference/spatial_svc.md)
for the coordinate-based (Gaussian-process) spatially varying
coefficient.

## Examples

``` r
# Chain graph over 10 cells
adj <- matrix(0, 10, 10)
for (i in 1:9) adj[i, i + 1] <- adj[i + 1, i] <- 1

# Spatial intercept plus a spatially varying time slope
f <- spatial(graph = adj, formula = ~ 1 + time || cell)
print(f)
#> tulpa areal varying-coefficient field
#> =====================================
#> 
#> Structure: ICAR (Besag) 
#> Graph nodes: 10 
#> Graph node index: cell 
#> Fields: independent (|| -> separate precision per coefficient)
#> Expands to 2 CAR field(s) (one per design-matrix column):
#>   cell.Intercept
#>   cell.time
```
