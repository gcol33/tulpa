# Per-cell constrained inverse-precision sub-block for a joint fit

Runs the same per-outer-grid-cell selected-inversion the joint
nested-Laplace driver's own fixed-effect retention uses internally
(`extract_inner_vcov_block_cell()`, see the note above
`.joint_attach_grid_fixed()`), for a caller-chosen index set. A consumer
package computing its own standard-error correction on a subset of a
joint fit's latent coordinates – a different `idx` than whatever the fit
itself retained on `$cov_block_per_grid` – reruns the extraction through
this door instead of reimplementing the sparse selected inversion.

## Usage

``` r
tulpa_joint_inner_vcov_blocks(
  Q_p_per_grid,
  Q_i_per_grid,
  Q_x_per_grid,
  n_x,
  idx,
  n_dense,
  A_cols_list,
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

- field_marginal:

  Logical; if `TRUE` (default) and `n_dense < length(idx)`, only the
  field coordinates' marginal variances are computed (a single Takahashi
  pass) rather than their full dense sub-block.

- n_threads:

  Integer, grid cells processed concurrently.

## Value

A list of length `n_grid` (one per outer-grid cell), each element either
`NULL` (the cell stored no `Q`, or its sparse Cholesky failed) or a
dense `length(idx) x length(idx)` matrix: the constrained covariance
sub-block for `idx`, with the sum-to-zero constraints in `A_cols_list`
conditioned out by the same kriging correction the fit's own posterior
draws use.

## Details

When a field is present (`n_dense < length(idx)`), the extraction takes
a cheap selected-inversion path: the dense (fixed-effect) block and the
fixed-effect x field cross terms are exact; the field x field block is
either its marginal diagonal (`field_marginal = TRUE`) or left at zero
(`FALSE`) – never the full field x field covariance.

## Examples

``` r
if (FALSE) { # \dontrun{
tulpa_joint_inner_vcov_blocks(
  fit$Q_csc_p_per_grid, fit$Q_csc_i_per_grid, fit$Q_csc_x_per_grid,
  n_x = fit$Q_csc_n, idx = 1:2, n_dense = 2L, A_cols_list = list()
)
} # }
```
