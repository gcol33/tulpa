# Rebuild axis specs from an already-assembled outer grid

Rebuilds each axis's declared spec (support, log/linear coordinate,
prior) from a `theta_grid`'s columns, keyed off column names – the same
metadata the multi-block joint driver recovers rather than carrying a
second description of the axes it already assembled. A column holding
one value across every cell is treated as a fixed setting rather than an
axis (e.g. a fixed NB-size node) and dropped. Intended for a caller that
needs an axis's declared span, coordinate or prior off a settled grid
without restating tulpa's own hyperprior declarations alongside it.

## Usage

``` r
tulpa_joint_axis_specs_from_grid(
  theta_grid,
  copy_slab = "exponential",
  folded_axes = NULL,
  logchol = .hp_logchol_designs(theta_grid)
)
```

## Arguments

- theta_grid:

  A named `[n_cells x n_axes]` matrix (see
  [`tulpa_theta_matrix()`](https://gillescolling.com/tulpa/reference/tulpa_theta_matrix.md)).

- copy_slab:

  `"exponential"` or `"flat"`; the copy-scale axis's continuum measure
  (see
  [`tulpa_hyper_check_copy_slab()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_check_copy_slab.md)).

- folded_axes:

  Optional names of axes folded onto `[0, Inf)`.

- logchol:

  Free-covariance block design(s) as returned by
  `.hp_logchol_designs()`; defaults to the design `theta_grid` itself
  declares.

## Value

A named list of per-axis spec lists, or `NULL` if `theta_grid` has no
axis names.

## See also

[`tulpa_theta_matrix()`](https://gillescolling.com/tulpa/reference/tulpa_theta_matrix.md),
[`tulpa_grid_log_quad()`](https://gillescolling.com/tulpa/reference/tulpa_grid_log_quad.md),
[`tulpa_hyper_grid_supports()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_grid_supports.md)

## Examples

``` r
# \donttest{
set.seed(1)
S <- 30L                                   # spatial units in a chain
nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
nn <- lengths(nb)
field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))
idx <- rep(seq_len(S), each = 6L); n <- length(idx); x <- rnorm(n)
y <- rbinom(n, 1L, plogis(-0.2 + 0.6 * x + field[idx]))
prior <- list(type = "icar", n_spatial_units = S, spatial_idx = idx,
              adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
              n_neighbors = nn, tau_grid = c(0.5, 1, 2, 4, 8))
fit <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
                            family = "binomial",
                            control = list(progress = FALSE))
specs <- tulpa_joint_axis_specs_from_grid(tulpa_theta_matrix(fit))
specs[[1]]$log_scale
#> [1] TRUE
# }
```
