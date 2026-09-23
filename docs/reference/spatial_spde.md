# SPDE Spatial Field (Matern via Triangular Mesh)

Specify a continuous Matern spatial field using the SPDE approach
(Lindgren, Rue & Lindstrom 2011). Builds a triangular mesh from
observation coordinates, computes FEM matrices, and passes them to
tulpa's SPDE Laplace engine with CHOLMOD sparse solver.

## Usage

``` r
spatial_spde(
  coords,
  data = NULL,
  mesh = NULL,
  boundary = NULL,
  max_edge = NULL,
  cutoff = 0,
  nu = 1,
  prior_range = NULL,
  prior_sigma = NULL
)
```

## Arguments

- coords:

  A formula `~ x + y` or a two-column matrix of coordinates.

- data:

  Optional data.frame for formula evaluation.

- mesh:

  A pre-built `tulpa_mesh` object. If NULL (default), a mesh is built
  automatically from `coords`.

- boundary:

  Optional boundary: a two-column matrix of polygon vertices, an sf
  polygon, or NULL for convex hull with extension.

- max_edge:

  Maximum edge length for mesh refinement. A single value or
  `c(inner, outer)`.

- cutoff:

  Minimum distance between mesh vertices. Default 0.

- nu:

  Matern smoothness parameter. A positive number. Integer `nu` (1, 2, 3,
  ...) gives an exact FEM construction (operator order
  `alpha = nu + 1`). Fractional `nu` (e.g. 0.5, 1.5) uses the
  operator-based rational SPDE approximation with BRASIL best-rational
  coefficients (Bolin & Kirchner 2020; Hofreither 2021); supported by
  the Laplace fitter
  [`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md)
  (single-point and nested over range/sigma). NUTS and analytic marginal
  SEs remain integer-only. Default 1.

- prior_range:

  PC prior on the spatial range, `c(U, alpha)` with P(range \< U) =
  alpha. `NULL` (the default) anchors it on the data: `U` is a fifth of
  the diagonal of the coordinates' bounding box and `alpha = 0.5`, so
  `U` is the prior median.

- prior_sigma:

  PC prior on the marginal standard deviation, `c(U, alpha)` with
  P(sigma \> U) = alpha. `NULL` (the default) is `c(3, 0.01)`, the
  engine's prior on every field scale.

## Value

A `tulpa_spatial` object with type `"spde"`.

## Examples

``` r
set.seed(42)
coords <- cbind(runif(50), runif(50))
spec <- spatial_spde(coords)
print(spec)
#> tulpa_spatial: SPDE (Matern, nu = 1 )
#>   Mesh nodes: 63 
#>   Triangles:  111 
#>   Observations: 50 
#>   Prior range: P(range < 0.2782498 ) = 0.5 
#>   Prior sigma: P(sigma > 3 ) = 0.01 
```
