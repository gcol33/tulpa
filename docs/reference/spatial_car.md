# CAR / ICAR spatial structure

Constructs a conditional autoregressive spatial random effect from an
adjacency matrix. With `proper = FALSE` (the default) this is the
improper CAR / ICAR (`type = "car"`); with `proper = TRUE` it returns
the same object as
[`spatial_car_proper()`](https://gillescolling.com/tulpa/reference/spatial_car_proper.md).

## Usage

``` r
spatial_car(
  adjacency,
  level = c("group", "obs"),
  group_var = NULL,
  proper = FALSE,
  shared = NULL,
  parameterization = c("standard", "collapsed")
)
```

## Arguments

- adjacency:

  Symmetric adjacency matrix (`[n_units x n_units]`).

- level:

  Either `"group"` (one effect per level of `group_var`) or `"obs"` (one
  effect per row of the data; `nrow(data)` must equal
  `nrow(adjacency)`).

- group_var:

  Name of the grouping variable in the data; required when
  `level = "group"`.

- proper:

  If `TRUE`, use proper CAR (`type = "car_proper"`); else ICAR
  (`type = "car"`).

- shared:

  Optional shared-effect handle (see model docs).

- parameterization:

  `"standard"` (default) or `"collapsed"` (deprecated).

## Value

A `tulpa_spatial` object with `type = "car"` (or `"car_proper"` when
`proper = TRUE`).

## See also

[`spatial_car_proper()`](https://gillescolling.com/tulpa/reference/spatial_car_proper.md),
[`spatial_bym2()`](https://gillescolling.com/tulpa/reference/spatial_bym2.md).
