# Proper CAR spatial structure

Convenience wrapper for `spatial_car(..., proper = TRUE)`. Creates a
proper conditional autoregressive (CAR) spatial random effect with the
autocorrelation parameter rho estimated from the data.

Use this when you want spatial autocorrelation to be a parameter of the
model rather than fixed at 1 (as in ICAR). rho ~= 0 collapses to IID,
rho ~= 1 approaches ICAR.

## Usage

``` r
spatial_car_proper(
  adjacency,
  level = c("group", "obs"),
  group_var = NULL,
  shared = NULL
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

- shared:

  Optional shared-effect handle (see model docs).

## Value

A `tulpa_spatial` object with `type = "car_proper"`.

## See also

[`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md)
for ICAR (rho fixed at 1),
[`spatial_bym2()`](https://gillescolling.com/tulpa/reference/spatial_bym2.md)
for the BYM2 decomposition.

## Examples

``` r
adj <- matrix(0, 10, 10)
for (i in 1:9) adj[i, i+1] <- adj[i+1, i] <- 1
spec <- spatial_car_proper(adj, level = "group", group_var = "site")
print(spec)
```
