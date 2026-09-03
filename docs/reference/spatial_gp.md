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
  sigma_prior_U = 1,
  sigma_prior_alpha = 0.01,
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
  and the neighbour covariance both read every column. `approx = "hsgp"`
  takes exactly two, and so does any sampler mode, because both store
  coordinates at a fixed 2-D stride.

- approx:

  GP approximation: `"nngp"` (default, a nearest-neighbour GP with the
  `cov` / `nu` / `nn` arguments) or `"hsgp"` (a Hilbert-space basis GP
  with `m` functions per dimension and boundary factor `c`).

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

- sigma_prior_U, sigma_prior_alpha:

  Penalized-complexity prior on the field's marginal standard deviation
  (`approx = "hsgp"`), calibrated so that
  `P(sigma > sigma_prior_U) = sigma_prior_alpha`. Defaults to
  `P(sigma > 1) = 0.01`. `sigma_prior_U` must be positive and
  `sigma_prior_alpha` must lie in `(0, 1)`.

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
spatial_gp(~ lon + lat, cov = "matern", nu = 1.5)
```
