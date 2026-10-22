# Per-cell refinement tag of an outer grid

The axis whose refinement pass added each cell of a nested-Laplace outer
grid, `""` for a cell of the grid as declared. A pass adds levels to an
axis, laid in every row of the others that holds the posterior; the tag
tells the declared levels apart wherever a reader needs them alone, e.g.
rebuilding axis specs from a grid's declared nodes (see
[`tulpa_joint_axis_specs_from_grid()`](https://gillescolling.com/tulpa/reference/tulpa_joint_axis_specs_from_grid.md)),
whose prior must not move with the nodes a pass added.

## Usage

``` r
tulpa_hyper_slice_home(refining, n)
```

## Arguments

- refining:

  The `refining` tag vector stored on a fit (`NULL` for a grid with no
  refinement).

- n:

  Number of grid cells; `refining`, if not `NULL`, must have this
  length.

## Value

A character vector of length `n`: `""` for a declared cell, the axis
name for a cell a refinement pass added.

## See also

[`tulpa_hyper_grid_supports()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_grid_supports.md),
[`tulpa_joint_axis_specs_from_grid()`](https://gillescolling.com/tulpa/reference/tulpa_joint_axis_specs_from_grid.md)

## Examples

``` r
tulpa_hyper_slice_home(NULL, 3)
tulpa_hyper_slice_home(c("", "", "sigma"), 3)
```
