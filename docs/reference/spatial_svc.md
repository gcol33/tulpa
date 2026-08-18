# Spatially varying coefficient structure

Specify a spatially varying coefficient (SVC): one or more fixed-effect
coefficients are allowed to vary smoothly over space, with the variation
governed by a Gaussian process (NNGP or HSGP approximation).

## Usage

``` r
spatial_svc(
  coords,
  terms = 1,
  cov = c("exponential", "matern"),
  nn = 15,
  shared = NULL,
  scale_coords = TRUE,
  approx = c("nngp", "hsgp"),
  m = 6,
  c_boundary = 1.5,
  parameterization = c("noncentered", "centered")
)
```

## Arguments

- coords:

  A formula (`~ lon + lat`) or character vector of length 2 naming the
  two coordinate variables in the data.

- terms:

  Which coefficients vary over space. A formula, an integer vector of
  design-matrix column indices, or a character vector of term names.
  Default `1` (the intercept).

- cov:

  Covariance function. One of `"exponential"` or `"matern"`.

- nn:

  Number of nearest neighbours used in the NNGP approximation
  (`approx = "nngp"`).

- shared:

  Whether the effect is shared across processes in a multi-process
  model. `NULL` (default) shares it; `FALSE` fits process-specific
  effects and emits a warning.

- scale_coords:

  Logical. Standardize coordinates before fitting (default `TRUE`).

- approx:

  Spatial approximation. `"nngp"` (nearest-neighbour GP) or `"hsgp"`
  (Hilbert-space GP).

- m:

  Number of basis functions per dimension for the HSGP approximation
  (`approx = "hsgp"`).

- c_boundary:

  Boundary-extension factor for the HSGP domain (`approx = "hsgp"`).

- parameterization:

  Latent parameterization for the exact-NUTS field (`approx = "nngp"`
  only; HSGP is already non-centered by construction). `"noncentered"`
  (default) samples `z_j ~ N(0, I)` per term and reconstructs each field
  as `w_j = f(z_j, sigma2_j, phi_j)`, removing the field/hyperparameter
  funnel that otherwise attenuates the field's amplitude when it is
  weakly identified. `"centered"` places the NNGP density on each term's
  field directly; it is marginally cheaper on a well-identified
  response, but on a weakly identified one it recovers only about a
  third of the field's spread.

## Value

A `tulpa_svc` object (also of class `tulpa_spatial`).

## See also

[`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
for a spatial random effect (rather than a varying coefficient).

## Examples

``` r
# Intercept that varies smoothly over space
spatial_svc(~ lon + lat)
#> tulpa spatially-varying coefficients
#> =====================================
#> 
#> Coordinates: lon, lat 
#> Covariance: exponential 
#> Neighbors (NNGP): 15 
#> Shared: Yes (enters both processes) 
#> 
#> Terms: columns  1 
```
