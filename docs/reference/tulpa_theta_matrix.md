# A fit's outer grid as a named matrix

Coerces a nested-Laplace fit's `theta_grid` to matrix form with named
axis columns. A single-axis grid is stored on the fit as a bare vector
named by `theta_names`; every downstream reader that keys an axis by
name needs the named matrix form, so this is the one place that coercion
happens. Intended for reconstructing a fit's outer-grid posterior cell
weights (see
[`tulpa_grid_log_quad()`](https://gillescolling.com/tulpa/reference/tulpa_grid_log_quad.md),
[`tulpa_normalise_weights_safe()`](https://gillescolling.com/tulpa/reference/tulpa_normalise_weights_safe.md))
for a fit or path whose driver doesn't already carry `fit$log_quad`.

## Usage

``` r
tulpa_theta_matrix(res)
```

## Arguments

- res:

  A `tulpa_fit` object (or any list carrying `theta_grid` and
  `theta_names`).

## Value

`res$theta_grid` as an `[n_cells x n_axes]` matrix with axis names as
column names, or `NULL` if the fit has no grid.

## See also

[`tulpa_grid_log_quad()`](https://gillescolling.com/tulpa/reference/tulpa_grid_log_quad.md),
[`tulpa_normalise_weights_safe()`](https://gillescolling.com/tulpa/reference/tulpa_normalise_weights_safe.md)

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
tulpa_theta_matrix(fit)
#>      tau
#> [1,] 0.5
#> [2,] 1.0
#> [3,] 2.0
#> [4,] 4.0
#> [5,] 8.0
# }
```
