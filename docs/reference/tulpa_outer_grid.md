# The outer hyperparameter grid of a nested-Laplace fit

A nested-Laplace fit integrates its hyperparameters on an outer grid:
each cell is one hyperparameter value, carries an integration weight,
and holds the inner Laplace solve at that value. This returns that grid,
cell by cell, so the integration can be inspected or reused (a discrete
hyperparameter posterior for
[`sbc_discrete()`](https://gillescolling.com/tulpa/reference/sbc_predictive.md),
a grid-marginalized latent field).

The per-cell latent modes average to the grid-marginalized posterior
mean of any latent coordinate: `colSums(g$weights * g$modes)`. Columns
`seq_len(g$n_fixed)` are the fixed effects and the rest the latent field
and random effects, in the order the fit's latent vector stores them.

## Usage

``` r
tulpa_outer_grid(fit)
```

## Arguments

- fit:

  A nested-Laplace fit:
  [`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md),
  [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md),
  or a [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
  fit routed to that tier.

## Value

A list:

- `nodes`:

  numeric matrix `[n_cells x n_axes]`, the hyperparameter value of each
  cell, with the axis names as column names.

- `weights`:

  the cells' integration weights, normalized to sum to one.

- `log_marginal`:

  each cell's log marginal likelihood, hyperprior density included; with
  the cell's quadrature volume it gives the weight. `NULL` when the fit
  does not keep it.

- `support`:

  named list of natural-scale `c(lo, hi)` intervals, the span each
  axis's outer measure integrates (including levels a refinement pass
  added); `NULL` when the fit records none.

- `modes`:

  numeric matrix `[n_cells x n_latent]`, each cell's inner Laplace mode,
  or `NULL` when the fit keeps none.

- `n_fixed`:

  how many leading columns of `modes` are fixed effects.

## See also

[`tulpa_hyper_draws()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_draws.md)
for continuous draws of the hyperparameters,
[`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md)
for draws of the latent field, and `summary(fit)$hyperparameters` for
their marginal summaries.

## Examples

``` r
# \donttest{
set.seed(1)
S <- 30L
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
g <- tulpa_outer_grid(fit)
cbind(g$nodes, weight = g$weights)
field_hat <- colSums(g$weights * g$modes)[-seq_len(g$n_fixed)]
# }
```
