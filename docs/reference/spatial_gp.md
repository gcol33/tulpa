# Gaussian process spatial structure (NNGP)

Specify a Gaussian-process spatial random effect, approximated with a
nearest-neighbour GP (NNGP) for scalability. Captures smooth spatial
variation from point-referenced coordinates.

## Usage

``` r
spatial_gp(
  coords,
  approx = c("nngp", "hsgp"),
  cov = NULL,
  nu = 1.5,
  nn = 15,
  m = NULL,
  c = NULL,
  sigma_prior_U = 1,
  sigma_prior_alpha = 0.01,
  shared = NULL,
  scale_coords = TRUE,
  parameterization = c("auto", "noncentered", "centered", "collapsed")
)

# S3 method for class 'tulpa_gp'
print(x, ...)

# S3 method for class 'tulpa_hsgp'
print(x, ...)
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
  with `m` functions per dimension and boundary factor `c`, which
  approximates the kernel `cov` / `nu` name).

- cov:

  Covariance function. With `approx = "nngp"`, `"exponential"` (the
  default) or `"matern"`. With `approx = "hsgp"`, `"matern"` (the
  default, at `nu = 1.5`) or `"squared_exponential"`; the exponential
  kernel has no HSGP basis rule. `NULL` takes the approximation's
  default.

- nu:

  Matern smoothness parameter, one of `1.5` or `2.5`. Used only when
  `cov = "matern"` (`nu = 0.5` is `cov = "exponential"`).

- nn:

  Number of nearest neighbours used in the NNGP approximation.

- m:

  Number of HSGP basis functions per dimension (`approx = "hsgp"`),
  between 3 and 50. `NULL` (default) chooses it from the data; see
  Details.

- c:

  HSGP boundary factor, `>= 1` (`approx = "hsgp"`): the basis lives on
  the coordinates' bounding box widened by `c`. `NULL` (default) chooses
  it from the data; see Details.

- sigma_prior_U, sigma_prior_alpha:

  Penalized-complexity prior on the field's marginal standard deviation
  (`approx = "hsgp"`), calibrated so that
  `P(sigma > sigma_prior_U) = sigma_prior_alpha`. Defaults to
  `P(sigma > 1) = 0.01`. `sigma_prior_U` must be positive and
  `sigma_prior_alpha` must lie in `(0, 1)`. Sampler and nested-Laplace
  routes use these anchors. Under `hyperprior = "flat"`, nested Laplace
  retains explicitly supplied anchors.

- shared:

  Whether the spatial effect is shared across processes in a
  multi-process model. `NULL` (default) shares the effect; `FALSE` fits
  process-specific effects and emits a warning.

- scale_coords:

  Logical. Standardize coordinates before fitting (default `TRUE`): each
  column is centred and all are divided by one common factor, so
  distances keep their proportions and an isotropic kernel stays
  isotropic.
  [`spatial_range()`](https://gillescolling.com/tulpa/reference/spatial_range.md)
  reports the range in the original coordinate units either way.

- parameterization:

  Latent parameterization for the exact-NUTS field. `"auto"` scales
  covariance-whitened coordinates using prior and observation
  information. `"noncentered"` samples `z ~ N(0, I)` and reconstructs
  the field as `w = f(z, sigma2, phi)`. `"centered"` samples field
  values. `"collapsed"` analytically integrates a Gaussian-likelihood
  NNGP field under NUTS/HMC and returns exact conditional field draws.
  Non-Gaussian collapsing is refused; use `"auto"` for full-field NUTS.

- x:

  A `tulpa_gp` or `tulpa_hsgp` specification.

- ...:

  Ignored.

## Value

A `tulpa_gp` object (also of class `tulpa_spatial`).

## Details

**HSGP kernel.** The basis approximates a Matern kernel of smoothness
3/2 (the default) or 5/2,
`sigma^2 (1 + sqrt(3) d / ell) exp(-sqrt(3) d / ell)` and
`sigma^2 (1 + sqrt(5) d / ell + 5 d^2 / (3 ell^2)) exp(-sqrt(5) d / ell)`
at distance `d`, or the squared-exponential kernel
`sigma^2 exp(-d^2 / (2 ell^2))`, through each kernel's spectral density
(Riutort-Mayol et al. 2023, Eqs. 1-3). The squared-exponential kernel
draws fields smoother than any Matern one, and a field rougher than the
kernel assumes is smoothed and its intervals narrowed.

For nested Laplace and Hamiltonian inference, the default PC (penalised
complexity) prior places half its probability below an effective range
of one fifth of the coordinate bounding-box diagonal. The effective
range is `rho = 2 * ell`, where `ell` is the kernel lengthscale (`phi`
in the NNGP kernels). Both approximations use this anchor.
[`spatial_multiscale()`](https://gillescolling.com/tulpa/reference/spatial_multiscale.md)
takes each scale's anchor from its declared range interval.

**HSGP basis size.** A Hilbert-space basis of `m` functions per
dimension on a domain widened by `c` has a lower lengthscale cutoff
`a c max(S) / m`, where `S` contains the coordinate half-widths and
`c >= 1.2`. The coefficient `a` retains 99% of the joint two-dimensional
spectral variance using the tensor expansion and cumulative-variance
recommendation of Riutort-Mayol et al. (2023, Eqs. 9-12, Section 4.2).
It is at least their one-dimensional covariance-error coefficient (Eqs.
19, 21, 23). The boundary factor follows Eqs. 20, 22, 24 and bounds the
variance removed by the Dirichlet boundary at the domain's corners by
1%, using the kernel correlations at twice the distance to each face. A
rough field asks for a short lengthscale, and a basis too small for it
smooths the field and narrows its intervals. Left unset, `m` and `c`
follow the iterative procedure of the paper's Section 4.5.1. In its
Phase A the first fit takes them from the prior median of the
lengthscale, and while the posterior median lengthscale falls below the
smallest one the fitted basis represents (the paper's diagnostic, with
its slack of 0.01 on the lengthscale divided by `S`), the field is
fitted again with the `m` and `c` that estimate asks for. In its Phase B
five functions per dimension are added and the field fitted again until
the refit passes the diagnostic and is stable against the fit before it:
the paper names the lengthscale estimate and the predictive accuracy as
what has to be stable and gives no threshold, and stable is read here as
the refitted posterior mean of the lengthscale, and of the linear
predictor at every observation, lying inside the previous fit's 95%
interval. `m` never shrinks below a size already fitted and stops at 50.
A supplied `m` or `c` is kept as given, and a supplied `m` ends the
procedure after Phase A.
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
reports when the posterior median lengthscale lies outside the window of
the fitted basis at either end; above it, the field is smoother than the
boundary factor allows for, and a larger `c` covers it.

## References

Riutort-Mayol, G., Buerkner, P.-C., Andersen, M. R., Solin, A. and
Vehtari, A. (2023). Practical Hilbert space approximate Bayesian
Gaussian processes for probabilistic programming. *Statistics and
Computing* 33, 17.
[doi:10.1007/s11222-022-10167-2](https://doi.org/10.1007/s11222-022-10167-2)

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
#> Shared: Yes (enters both processes) 
spatial_gp(~ lon + lat, cov = "matern", nu = 1.5)
#> tulpa Gaussian Process spatial specification
#> =============================================
#> 
#> Coordinates: lon, lat 
#> Covariance: matern (nu = 1.5) 
#> Neighbors (NNGP): 15 
#> Shared: Yes (enters both processes) 
spatial_gp(~ lon + lat, approx = "hsgp", cov = "squared_exponential")
#> tulpa Hilbert Space GP (HSGP) spatial specification
#> =====================================================
#> 
#> Coordinates: lon, lat 
#> Kernel: squared exponential 
#> Basis functions: chosen from the data
#> Boundary factor: chosen from the data
#> Shared: Yes (enters both processes) 
```
