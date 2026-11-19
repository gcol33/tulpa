# Multi-scale Gaussian process spatial structure

Specify a multi-scale spatial random effect that decomposes spatial
variation into local (fine-scale) and regional (broad-scale) components.
Each scale has its own range and variance parameters.

## Usage

``` r
spatial_multiscale(
  coords,
  scales = c("local", "regional"),
  approx = c("nngp", "hsgp"),
  m = NULL,
  c_boundary = NULL,
  range_local = NULL,
  range_regional = c(1, 10),
  cov = NULL,
  nu = 1.5,
  nn_local = 10,
  nn_regional = 30,
  shared = NULL,
  scale_coords = TRUE,
  sampler = c("auto", "noncentered", "centered")
)

# S3 method for class 'tulpa_multiscale'
print(x, ...)
```

## Arguments

- coords:

  A one-sided formula specifying coordinate columns (e.g.,
  `~ lon + lat`), or a character vector of length 2 with column names.

- scales:

  Character vector specifying scale names. Default:
  `c("local", "regional")`.

- approx:

  Approximation method: `"nngp"` (default) for Nearest Neighbor GP;
  `"hsgp"` for a Hilbert-space basis shared by both scales, which is
  fitted by the nested-Laplace path only (`mode = "auto"` or
  `"nested_laplace"`); the samplers carry the NNGP form.

- m:

  Number of HSGP basis functions per dimension, between 3 and 50; only
  used when `approx = "hsgp"`, where each scale carries its own basis of
  m^2 functions. A value given applies to both scales. `NULL` (default)
  chooses each scale's from the data by the rule
  [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
  describes: first at the median of that scale's range prior, then at
  its posterior median lengthscale while the basis does not represent
  it.

- c_boundary:

  Boundary factor for HSGP domain extension, `>= 1`; only used when
  `approx = "hsgp"`. A value given applies to both scales; `NULL`
  (default) chooses each scale's from the data.

- range_local:

  Plausible range interval for the local scale as `c(lower, upper)` in
  coordinate units. `NULL` (default) takes the upper bound 1 (after
  scaling) and the lower bound from the data: the median, over the
  distinct locations, of the distance to the nearest other location. A
  range below the spacing of the locations is one the data do not
  inform, and with `approx = "hsgp"` a lower bound there asks the basis
  rule for more functions than the data can use. This is not a hard box:
  `lower` anchors a PC prior on that scale's range
  (`P(range < lower) = 0.05`, the same prior
  [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
  uses), on every route that fits the field; a sampler starts the range
  at the pair's geometric mean. The range itself is free on `(0, Inf)`.
  The range is the Matern range `rho = 2 * ell`, matching
  [`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md),
  on both approximations. Here `ell` is the kernel lengthscale (`phi` in
  the NNGP kernels). The squared-exponential kernel uses the same
  effective-range convention.

- range_regional:

  Plausible range interval for the regional scale, read the same way.
  Default: `c(1, 10)` (after scaling). Keeping the two intervals
  separated is what identifies the scales against each other.

- cov:

  Covariance function, read as
  [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
  reads it: with `approx = "nngp"`, `"exponential"` (default) or
  `"matern"`; with `approx = "hsgp"`, `"matern"` (default, at
  `nu = 1.5`) or `"squared_exponential"`.

- nu:

  Smoothness parameter for Matern covariance, one of `1.5` or `2.5`.

- nn_local:

  Number of nearest neighbors for local scale. Default 10.

- nn_regional:

  Number of nearest neighbors for regional scale. Default 30.

- shared:

  Logical; if TRUE (default), spatial effects enter both all processes.

- scale_coords:

  Logical; if TRUE (default), coordinates are centred and divided by one
  common factor before computing distances (see
  [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)).

- sampler:

  Latent parameterization for the exact-NUTS field. `"auto"` (default)
  and `"noncentered"` sample `z ~ N(0, I)` per scale and reconstruct
  each field as `w = f(z, sigma2, phi)`, avoiding the
  field/hyperparameter funnel; `"centered"` places the NNGP density on
  each field directly.

- x:

  A `tulpa_multiscale` specification.

- ...:

  Ignored.

## Value

A `tulpa_multiscale` object

## Details

The multi-scale model decomposes spatial variation additively:

\$\$\eta(s) = X\beta + w\_{local}(s) + w\_{regional}(s)\$\$

where each component follows an independent Gaussian process:
\$\$w\_{local}(s) \sim GP(0, \sigma^2\_{local} C(\phi\_{local}))\$\$
\$\$w\_{regional}(s) \sim GP(0, \sigma^2\_{regional}
C(\phi\_{regional}))\$\$

**Fitting.** `mode = "auto"`, `"structured"` and `"nested_laplace"` fit
the field by nested Laplace, as two field blocks of the multi-block
driver (two NNGP blocks, each on its own neighbour structure, or two
HSGP blocks on one basis). Each scale carries its own SD and range axes
on the outer grid, placed and refined like any other field axis, so both
scales' variances and ranges are integrated. The exact samplers
(`mode = "exact"` and its siblings) sample the NNGP form with the same
priors.
[`spatial_range()`](https://gillescolling.com/tulpa/reference/spatial_range.md)
reports each scale's `sigma` and `range` (`3 * phi`), tagged `[local]` /
`[regional]`;
[`field_effects()`](https://gillescolling.com/tulpa/reference/field_effects.md)
reports the summed field per distinct location, or one scale with
`component =`.

**Identifiability.** Both fields are proper, so neither needs a level
constraint; the constant they share with the intercept is held by their
priors. What tells the scales apart is the range priors: the regional
field's PC prior puts 5% of its mass below `range_regional[1]`, and the
local field's below `range_local[1]`, while neither range is bounded
above. Where the data do not separate the scales, the local range can
therefore reach into the regional interval and the two scales share the
variance, which widens both intervals rather than switching the labels.
On simulated Poisson data with 300 locations on a 10 x 10 square (local
SD 0.63 and range 0.5, regional SD 1 and range 3,
`range_local = c(0.2, 1.5)`, `range_regional = c(1.5, 10)`), the nested
95% intervals of all four hyperparameters covered the truth on each of 5
data sets, with posterior means 0.66 / 1.1 (local SD / range) and 1.03 /
7.3 (regional); the component maps correlated with the simulated fields
at 0.49 (local), 0.79 (regional) and 0.85 (sum).

**Computational cost.** On those data sets the nested fit took 97-125 s
(100 locations: 29 s). Exact NUTS on 100 locations took 453 s for 4
chains of 1000 iterations and had not converged (R-hat 1.14 and bulk ESS
13 on the local log-variance), its chains disagreeing on how the
variance splits between the scales.

## See also

[`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
for single-scale GP,
[`temporal_multiscale()`](https://gillescolling.com/tulpa/reference/temporal_multiscale.md)
for multi-scale temporal effects

## Examples

``` r
# Create multi-scale spatial structure
ms <- spatial_multiscale(
  ~ lon + lat,
  range_local = c(0.1, 0.5),
  range_regional = c(1, 5)
)
print(ms)
#> tulpa Multi-Scale spatial specification
#> ========================================
#> 
#> Coordinates: lon, lat 
#> Scales: local + regional 
#> 
#> Local scale:
#>   Range prior: [0.1, 0.5]
#>   Neighbors: 10 
#> 
#> Regional scale:
#>   Range prior: [1, 5]
#>   Neighbors: 30 
#> 
#> Covariance: exponential 
#> Shared: Yes (enters both processes) 

# \donttest{
set.seed(101)
n <- 25
df <- data.frame(
  lon = runif(n, 0, 10),
  lat = runif(n, 0, 10),
  depth = rnorm(n),
  temp = rnorm(n)
)
df$count <- rpois(n, exp(1 + 0.2 * df$depth))

# Both scales' variances and ranges are integrated by nested Laplace.
fit <- tulpa(
  count ~ depth + temp,
  data = df,
  family = "poisson",
  spatial = spatial_multiscale(
    ~ lon + lat,
    range_local = c(0.1, 0.5),
    range_regional = c(1, 5),
    nn_local = 5L,
    nn_regional = 8L
  )
)
#> [nested-laplace] preparing joint Hessian sparsity pattern
#> [nested-laplace] 1/625 cells (0%) | elapsed 0s | ETA >=1s | 0.00s/cells
#> [nested-laplace] 191/191 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
#> [nested-laplace] preparing joint Hessian sparsity pattern
#> [nested-laplace] 1/625 cells (0%) | elapsed 0s | ETA >=1s | 0.00s/cells
#> [nested-laplace] 275/275 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
spatial_range(fit)
#>                      mean        sd       q2.5       q97.5
#> sigma[local]    0.2588812 0.2016770 0.01019120   0.7182364
#> range[local]          Inf       Inf 0.35671721  77.2081895
#> sigma[regional] 0.3550972 0.3128667 0.01230064   1.1595261
#> range[regional]       Inf       Inf 3.63862499 688.0435731
head(field_effects(fit, component = "regional"))
#>   unit label    estimate        sd   conf.low conf.high  source
#> 1    1     1 -0.08598745 0.4267037 -1.0333570 0.7551617 mixture
#> 2    2     2 -0.04477110 0.4142071 -0.9309312 0.8060006 mixture
#> 3    3     3  0.08382016 0.4155629 -0.6874523 1.0392009 mixture
#> 4    4     4  0.12629010 0.4258823 -0.6257884 1.1258412 mixture
#> 5    5     5 -0.12286536 0.4330018 -1.1080479 0.6903297 mixture
#> 6    6     6 -0.10042866 0.4319503 -1.0704365 0.7365485 mixture
# }
```
