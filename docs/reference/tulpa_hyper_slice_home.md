# Per-cell refinement-slice tag of an outer grid

The axis each cell of a nested-Laplace outer grid was placed on by a
refinement pass, `""` for a base (unrefined) tensor cell. Used to tell a
base grid apart from its refinement slices wherever a reader needs to
restrict to one or the other, e.g. rebuilding axis specs from a grid's
declared nodes only (see
[`tulpa_joint_axis_specs_from_grid()`](https://gillescolling.com/tulpa/reference/tulpa_joint_axis_specs_from_grid.md)).

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

A character vector of length `n`: `""` for a base cell, the axis name
for a refinement-slice cell.

## See also

[`tulpa_hyper_grid_supports()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_grid_supports.md),
[`tulpa_joint_axis_specs_from_grid()`](https://gillescolling.com/tulpa/reference/tulpa_joint_axis_specs_from_grid.md)
