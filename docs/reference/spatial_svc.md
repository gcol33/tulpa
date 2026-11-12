# Spatially varying coefficient structure

Specify a spatially varying coefficient (SVC): one or more fixed-effect
coefficients are allowed to vary smoothly over space, with the variation
governed by a Gaussian process (NNGP or HSGP approximation).

## Usage

``` r
spatial_svc(
  coords,
  terms = 1,
  cov = NULL,
  nu = 1.5,
  nn = 15,
  shared = NULL,
  scale_coords = TRUE,
  approx = c("nngp", "hsgp"),
  m = NULL,
  c_boundary = NULL,
  parameterization = c("auto", "noncentered", "centered"),
  sigma_prior_U = 1,
  sigma_prior_alpha = 0.01
)

# S3 method for class 'tulpa_svc'
print(x, ...)
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

  Covariance function, read as
  [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
  reads it: with `approx = "nngp"`, `"exponential"` (default) or `"matern"`;
  with `approx = "hsgp"`, `"matern"` (default, at `nu = 1.5`) or
  `"squared_exponential"`.

- nu:

  Matern smoothness, `1.5` (default) or `2.5`, used when `cov = "matern"`.

- nn:

  Number of nearest neighbours used in the NNGP approximation
  (`approx = "nngp"`).

- shared:

  Whether the effect is shared across processes in a multi-process
  model. `NULL` (default) shares it; `FALSE` fits process-specific
  effects and emits a warning.

- scale_coords:

  Logical. Standardize coordinates before fitting (default `TRUE`), by
  one common factor as in
  [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md).
  Ranges and the coordinates
  [`svc()`](https://gillescolling.com/tulpa/reference/svc.md) reports
  are in the original units either way.

- approx:

  Spatial approximation. `"nngp"` (nearest-neighbour GP) or `"hsgp"`
  (Hilbert-space GP).

- m:

  Number of basis functions per dimension for the HSGP approximation
  (`approx = "hsgp"`), between 3 and 50. `NULL` (default) chooses it from
  the data by the rule
  [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
  describes, read at the smallest lengthscale of the varying coefficients,
  which share one basis.

- c_boundary:

  Boundary-extension factor for the HSGP domain (`approx = "hsgp"`), `>= 1`.
  `NULL` (default) chooses it from the data, at the largest lengthscale of
  the varying coefficients.

- parameterization:

  Latent parameterization for the exact-NUTS field (`approx = "nngp"`
  only; HSGP is already non-centered by construction). `"auto"` and
  `"noncentered"` sample independent `z_j ~ N(0, I)` innovations per term
  and reconstruct each field as `w_j = f(z_j, sigma2_j, phi_j)`. The
  innovations' prior is independent of the sampled scale and range.
  `"centered"` places the NNGP density on each term's field directly.

- sigma_prior_U, sigma_prior_alpha:

  Penalized-complexity prior on each coefficient field's marginal standard
  deviation, calibrated so that `P(sigma > sigma_prior_U) = sigma_prior_alpha`.
  Defaults to `P(sigma > 1) = 0.01`, as for HSGP fields and
  [`temporal_tvc()`](https://gillescolling.com/tulpa/reference/temporal_tvc.md).
  Applies to both approximations. `sigma_prior_U` must be positive and
  `sigma_prior_alpha` must lie in `(0, 1)`.

- x:

  A `tulpa_svc` specification.

- ...:

  Ignored.

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
