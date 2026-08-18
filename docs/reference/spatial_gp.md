# Gaussian process spatial structure (NNGP)

Specify a Gaussian-process spatial random effect, approximated with a
nearest-neighbour GP (NNGP) for scalability. Captures smooth spatial
variation from point-referenced coordinates.

## Usage

``` r
spatial_gp(
  coords,
  approx = c("nngp", "hsgp"),
  cov = c("exponential", "matern"),
  nu = 1.5,
  nn = 15,
  m = 6,
  c = 1.5,
  solver = c("auto", "cholesky", "cg", "pcg", "gpu"),
  cg_tol = 1e-06,
  cg_maxiter = 100,
  shared = NULL,
  scale_coords = TRUE,
  parameterization = c("noncentered", "centered", "collapsed")
)
```

## Arguments

- coords:

  A formula (`~ lon + lat`) or character vector naming the coordinate
  variables in the data. With `approx = "nngp"` the coordinate DIMENSION
  is however many are named: two for a map, one for a transect or a
  depth profile, three for a depth-resolved domain. The neighbour graph
  and the neighbour covariance both read every column (gcol33/tulpa#389,
  gcol33/tulpa#391). `approx = "hsgp"` takes exactly two, and so does
  any sampler mode, because both store coordinates at a fixed 2-D
  stride.

- approx:

  GP approximation: `"nngp"` (default, a nearest-neighbour GP with the
  `cov` / `nu` / `nn` / `solver` arguments) or `"hsgp"` (a Hilbert-space
  basis GP with `m` functions per dimension and boundary factor `c`).

- cov:

  Covariance function (NNGP only). One of `"exponential"` or `"matern"`.

- nu:

  Matern smoothness parameter, one of `1.5` or `2.5`. Used only when
  `cov = "matern"` (`nu = 0.5` is `cov = "exponential"`).

- nn:

  Number of nearest neighbours used in the NNGP approximation.

- m:

  Number of HSGP basis functions per dimension (`approx = "hsgp"`).

- c:

  HSGP boundary factor, `>= 1` (`approx = "hsgp"`).

- solver:

  Linear solver for the GP. One of `"auto"`, `"cholesky"`, `"cg"`,
  `"pcg"`, or `"gpu"`. `"gpu"` falls back to `"pcg"` when CUDA support
  is unavailable.

- cg_tol:

  Convergence tolerance for the (preconditioned) CG solver.

- cg_maxiter:

  Maximum number of (preconditioned) CG iterations.

- shared:

  Whether the spatial effect is shared across processes in a
  multi-process model. `NULL` (default) shares the effect; `FALSE` fits
  process-specific effects and emits a warning.

- scale_coords:

  Logical. Standardize coordinates before fitting (default `TRUE`).

- parameterization:

  Latent parameterization for the exact-NUTS field. One of
  `"noncentered"` (default; samples `z ~ N(0, I)` and reconstructs the
  field as `w = f(z, sigma2, phi)`, avoiding the field/hyperparameter
  funnel), `"centered"` (places the NNGP density on the field directly),
  or `"collapsed"` (deprecated).

## Value

A `tulpa_gp` object (also of class `tulpa_spatial`).

## See also

[`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md),
[`spatial_bym2()`](https://gillescolling.com/tulpa/reference/spatial_bym2.md)
for areal spatial effects.

## Examples

``` r
# GP spatial specification from coordinate columns
spatial_gp(~ lon + lat)
#> tulpa Gaussian Process spatial specification
#> =============================================
#> 
#> Coordinates: lon, lat 
#> Covariance: exponential 
#> Neighbors (NNGP): 15 
#> Solver: auto (Cholesky<2k, PCG<5k, GPU/CG for larger) 
#> Shared: Yes (enters both processes) 
spatial_gp(~ lon + lat, cov = "matern", nu = 1.5)
#> tulpa Gaussian Process spatial specification
#> =============================================
#> 
#> Coordinates: lon, lat 
#> Covariance: matern (nu = 1.5) 
#> Neighbors (NNGP): 15 
#> Solver: auto (Cholesky<2k, PCG<5k, GPU/CG for larger) 
#> Shared: Yes (enters both processes) 
```
