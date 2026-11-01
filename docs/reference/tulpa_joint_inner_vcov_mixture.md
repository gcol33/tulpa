# Outer-grid posterior covariance of a joint fit's latent coordinates

The law-of-total-covariance mixture over the outer grid, \$\$V = \sum_k
w_k \[C_k + (m_k - \bar m)(m_k - \bar m)^T\],\$\$ with `C_k` the same
per-cell constrained block
[`tulpa_joint_inner_vcov_blocks()`](https://gillescolling.com/tulpa/reference/tulpa_joint_inner_vcov_blocks.md)
returns, accumulated inside the engine without holding one block per
cell. Memory is one `length(idx)` square result whatever the grid size,
so this is the door for an `idx` that carries a field.

## Usage

``` r
tulpa_joint_inner_vcov_mixture(
  Q_p_per_grid,
  Q_i_per_grid,
  Q_x_per_grid,
  n_x,
  idx,
  n_dense,
  A_cols_list,
  weights,
  modes = NULL,
  field_marginal = TRUE,
  n_threads = 1L
)
```

## Arguments

- Q_p_per_grid, Q_i_per_grid, Q_x_per_grid:

  Per-outer-grid-cell CSC storage of the latent precision `Q`
  (`fit$Q_csc_p_per_grid` / `_i_per_grid` / `_x_per_grid`), one list
  element per cell.

- n_x:

  Integer, the latent dimension `Q` is built over (`fit$Q_csc_n`).

- idx:

  Integer vector (1-based) of latent coordinates to extract the
  sub-block for.

- n_dense:

  Integer, the number of leading `idx` entries that are dense fixed
  effects rather than field coordinates (see Details).

- A_cols_list:

  A list of integer vectors, one per sum-to-zero constraint group, each
  the (1-based) latent coordinates that group averages to zero.

- weights:

  Numeric, one non-negative weight per outer-grid cell (normalized
  here). A cell of zero weight is never extracted.

- modes:

  Optional `n_grid x length(idx)` matrix of the cell modes on `idx`.
  `NULL` returns the within-cell term alone.

- field_marginal:

  Logical; if `TRUE` (default) and `n_dense < length(idx)`, only the
  field coordinates' marginal variances are computed (a single Takahashi
  pass) rather than their full dense sub-block.

- n_threads:

  Integer, grid cells processed concurrently.

## Value

A list: `vcov`, the mixture covariance; `mean`, the weighted mean of
`modes` (`NULL` without them); `dense_blocks`, one `n_dense x n_dense`
block per cell (`NULL` where the cell was not extracted); `failed`, the
cells of positive weight whose block is missing (no stored precision, or
its Cholesky failed). Such a cell keeps its weight and contributes its
between-cell term only. With `field_marginal = TRUE` the field x field
part of the within term is its diagonal.

## Examples

``` r
# Two outer-grid cells of a latent precision over two fixed effects and a
# ten-node random-walk field (sum-to-zero), each stored as the lower-triangle
# CSC a joint fit keeps per cell, with the cells' modes and weights.
make_Q <- function(tau) {
  Q <- matrix(0, 12, 12)
  Q[1:2, 1:2] <- diag(2, 2) + 0.1
  Q[3:12, 3:12] <- tau * crossprod(diff(diag(10))) + diag(0.3, 10)
  Q[1, 3:12] <- Q[3:12, 1] <- 0.1
  L <- Matrix::tril(methods::as(methods::as(Q, "generalMatrix"),
                                "CsparseMatrix"))
  list(p = L@p, i = L@i, x = L@x)
}
cells <- list(make_Q(1), make_Q(4))
set.seed(1)
modes <- matrix(rnorm(2 * 12, sd = 0.2), 2, 12)
mix <- tulpa_joint_inner_vcov_mixture(
  lapply(cells, `[[`, "p"), lapply(cells, `[[`, "i"), lapply(cells, `[[`, "x"),
  n_x = 12L, idx = 1:12, n_dense = 2L, A_cols_list = list(3:12),
  weights = c(0.7, 0.3), modes = modes)
mix$mean[1:2]
#> [1] -0.07668493 -0.02127116
mix$vcov[1:2, 1:2]
#>              [,1]         [,2]
#> [1,]  0.482785289 -0.006185382
#> [2,] -0.006185382  0.526911020
```
