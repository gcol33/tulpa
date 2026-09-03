# Multi-Scale Gaussian Process spatial structure

Specify a multi-scale spatial random effect that decomposes spatial
variation into local (fine-scale) and regional (broad-scale) components.
Each scale has its own range and variance parameters.

This is particularly useful for large datasets (\>100k observations)
where spatial patterns exist at multiple scales.

## Usage

``` r
spatial_multiscale(
  coords,
  scales = c("local", "regional"),
  approx = c("nngp", "hsgp"),
  m = 6L,
  c_boundary = 1.5,
  range_local = c(0.01, 1),
  range_regional = c(1, 10),
  cov = c("exponential", "matern"),
  nu = 1.5,
  nn_local = 10,
  nn_regional = 30,
  shared = NULL,
  scale_coords = TRUE,
  sampler = c("auto", "noncentered", "centered", "interweaved")
)
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
  `"hsgp"` for Hilbert Space GP (faster for smooth fields).

- m:

  Number of HSGP basis functions per dimension (default 6). Only used
  when `approx = "hsgp"`. Total basis functions will be m^2.

- c_boundary:

  Boundary factor for HSGP domain extension (default 1.5). Only used
  when `approx = "hsgp"`.

- range_local:

  Plausible range interval for the local scale as `c(lower, upper)` in
  coordinate units. Default: `c(0.01, 1)` (after scaling). Under exact
  NUTS this is not a hard box: `lower` anchors a PC prior on that
  scale's range (`P(range < lower) = 0.05`, the same prior
  [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
  uses), and the pair places the sampler's starting range at their
  geometric mean. The range itself is free on `(0, Inf)`.

- range_regional:

  Plausible range interval for the regional scale, read the same way.
  Default: `c(1, 10)` (after scaling). Keeping the two intervals
  separated is what identifies the scales against each other.

- cov:

  Covariance function: `"exponential"` (default) or `"matern"`.

- nu:

  Smoothness parameter for Matern covariance, one of `1.5` or `2.5`.

- nn_local:

  Number of nearest neighbors for local scale. Default 10.

- nn_regional:

  Number of nearest neighbors for regional scale. Default 30.

- shared:

  Logical; if TRUE (default), spatial effects enter both all processes.

- scale_coords:

  Logical; if TRUE (default), coordinates are scaled to unit variance
  before computing distances.

- sampler:

  Latent parameterization for the exact-NUTS field. `"auto"` (default)
  and `"noncentered"` sample `z ~ N(0, I)` per scale and reconstruct
  each field as `w = f(z, sigma2, phi)`, avoiding the
  field/hyperparameter funnel; `"centered"` places the NNGP density on
  each field directly. `"interweaved"` alternates between
  parameterizations and is not implemented on the exact-NUTS path.

## Value

A `tulpa_multiscale` object

## Details

The multi-scale model decomposes spatial variation additively:

\$\$\eta(s) = X\beta + w\_{local}(s) + w\_{regional}(s)\$\$

where each component follows an independent Gaussian process:
\$\$w\_{local}(s) \sim GP(0, \sigma^2\_{local} C(\phi\_{local}))\$\$
\$\$w\_{regional}(s) \sim GP(0, \sigma^2\_{regional}
C(\phi\_{regional}))\$\$

**Identifiability**: With sufficient data (\>500 locations), the two
scales are typically well-identified when prior ranges are
non-overlapping. PC priors on variance components help prevent
overfitting.

**Computational cost**: Approximately 1.5-2x the cost of single-scale
GP, as two NNGP likelihoods must be evaluated.

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

# Both scales are sampled by exact NUTS (mode = "exact"); the field is
# returned as gp_local[i] / gp_regional[i] draws.
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
  ),
  mode = "exact",
  control = list(n_iter = 60L, n_warmup = 30L, seed = 1L)
)
summary(fit)
# }
```
