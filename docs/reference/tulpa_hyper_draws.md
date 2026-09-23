# Hyperparameter draws from a nested-Laplace fit

The hyperparameter half of a draw from the outer-grid mixture: one row
per draw, one column per outer-grid axis. A draw's coordinate on each
axis is sampled from the within-cell density its cell carries under the
fit's own within-cell read, so the columns are a continuous marginal
rather than the handful of grid-node values `fit$theta_grid[cells, ]`
returns.

An axis carrying a declared POINT MASS – the copy scale's `alpha = 0`,
the "no coupling" model whose prior probability is stated rather than
read off a node count – keeps it: draws in that cell are exactly the
level, in the proportion the fit reports as
`fit$copy_atom$posterior_mass`, and the continuum above it is
continuized on its own partition. The level is a model and not a cell
representative, so spreading it over a box would both put mass where the
axis has none and leave none on the level itself.

Reading the node coordinate directly is what
[`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md)
consumers used to do, and on a 5- to 15-node axis it makes every draw an
atom: a truth between two nodes, or past the outermost one, has no draw
that can land near it. The continuization is the cell-conditional of the
SAME construction the fit reports its interval from (`box_uniform` by
default, `chord` when the fit asked for it or its support does not admit
the box read), so the draws reproduce the fit's own `theta_ci_lo` /
`theta_median` / `theta_ci_hi` to Monte Carlo error.

## Usage

``` r
tulpa_hyper_draws(fit, cells = NULL, n = 1000, within = NULL)
```

## Arguments

- fit:

  A nested-Laplace fit
  ([`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
  or
  [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)).

- cells:

  Integer vector of outer-grid cell indices, one per draw – the
  `"cells"` attribute of a
  [`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md)
  matrix, so the hyperparameter row and the latent row of a draw come
  from the same cell. `NULL` (default) allocates `n` fresh draws across
  the cells by weight.

- n:

  Number of draws when `cells` is `NULL` (default 1000); ignored
  otherwise.

- within:

  Within-cell construction, `"box_uniform"` or `"chord"`. `NULL`
  (default) takes the one the fit was read with
  (`fit$within_cell_requested`).

## Value

A numeric matrix `[n x n_axes]` with the outer grid's axis names as
columns, or `NULL` when the fit carries no outer-grid axis. Carries
`attr(., "cells")` – the cell each row's coordinates were drawn in –
plus `attr(., "within_cell")` and `attr(., "within_cell_declined")`, the
per-axis construction that ran and why a requested one did not, and
`attr(., "within_cell_copula")`, the Gaussian-copula correlation matrix
that ties the axes' within-cell draws together so the draws keep the
grid's own correlation between axes (the identity where the grid's axes
are uncorrelated). Each axis's marginal is the same whatever the copula.

## See also

[`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md),
[`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
