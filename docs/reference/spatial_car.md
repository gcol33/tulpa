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
  parameterization = c("auto", "standard", "collapsed"),
  prior_sigma = NULL
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

  `"auto"` scales field amplitudes using prior and observation
  information; `"standard"` samples effects directly. For ICAR and BYM2,
  `"collapsed"` analytically integrates Gaussian spatial effects under
  NUTS/HMC and returns exact conditional field draws. Non-Gaussian
  collapsing is refused; use `"auto"` for full-field NUTS.

- prior_sigma:

  Optional PC prior on the field's standard deviation, `c(U, alpha)`
  with \\P(\sigma \> U) = \alpha\\, read by the nested-Laplace and
  sampler paths, carried to the precision for CAR and ICAR fields.
  `NULL` uses the engine's default anchor \\P(\sigma \> 3) = 0.01\\.

## Value

A `tulpa_spatial` object with `type = "car"` (or `"car_proper"` when
`proper = TRUE`).

## See also

[`spatial_car_proper()`](https://gillescolling.com/tulpa/reference/spatial_car_proper.md),
[`spatial_bym2()`](https://gillescolling.com/tulpa/reference/spatial_bym2.md).

## Examples

``` r
adj <- matrix(0, 10, 10)
for (i in 1:9) adj[i, i + 1] <- adj[i + 1, i] <- 1
spatial_car(adj, level = "group", group_var = "region")
#> tulpa spatial specification
#> ===========================
#> 
#> Type: ICAR (Intrinsic CAR) 
#> Level: group 
#> Spatial units: 10 
#> Shared: Yes (enters both processes) 
#> Group variable: region 
#>   (rho fixed at 1, sum-to-zero constraint applied)
```
