# Gaussian Process temporal structure

Specify a Gaussian Process (GP) temporal random effect for
irregularly-spaced or continuous time points. Unlike RW1/RW2/AR1 which
assume equally-spaced observations, GP temporal effects model
correlation as a function of time distance.

This is particularly useful for:

- Irregularly-spaced time series

- Continuous time (e.g., exact timestamps)

- Smooth temporal trends with uncertainty

## Usage

``` r
temporal_gp(
  time_var,
  cov = c("exponential", "matern", "gaussian", "periodic"),
  nu = 1.5,
  period = NULL,
  group_var = NULL,
  shared = NULL,
  scale_coords = TRUE,
  parameterization = c("noncentered", "centered")
)
```

## Arguments

- time_var:

  Name of the time variable in data. Can be a formula (e.g., `~ year`)
  or a character string (e.g., `"year"`). Should be numeric (continuous
  time) or convertible to numeric.

- cov:

  Covariance function: `"exponential"` (default, rough), `"matern"`
  (tunable smoothness), `"gaussian"` (very smooth), or `"periodic"` (for
  seasonal patterns).

- nu:

  Smoothness parameter for Matern covariance, one of:

  - 0.5: equivalent to exponential (rough)

  - 1.5: once differentiable (moderate smoothness)

  - 2.5: twice differentiable (smooth) These are the smoothnesses with a
    closed form; anything between them needs a Bessel function of the
    second kind and is rejected. Ignored for non-Matern covariance
    functions.

- period:

  Period for periodic covariance (e.g., 12 for monthly, 365 for daily
  data with annual cycle). Only used when `cov = "periodic"`.

- group_var:

  Optional name of grouping variable for panel data. If provided,
  separate GPs are estimated for each group.

- shared:

  Logical; if TRUE (default), temporal effect enters both all processes.

- scale_coords:

  Logical; if TRUE (default), time values are scaled to unit variance
  before computing distances.

- parameterization:

  Parameterization for GP effects: `"noncentered"` (default) stores z ~
  N(0,1) and scales by covariance (better for weakly-informed effects);
  `"centered"` stores effects directly (better for strongly-informed
  effects).

## Value

A `tulpa_temporal_gp` object

## Details

The GP temporal model adds a time-correlated random effect:

\$\$\eta(t) = X\beta + f(t)\$\$

where \\f(t)\\ follows a Gaussian process: \$\$f(t) \sim GP(0, \sigma^2
C(\|t - t'\|; \phi))\$\$

The correlation function \\C(d; \phi)\\ depends on time distance \\d\\:

- **Exponential**: \\C(d) = \exp(-d/\phi)\\ - continuous but not
  differentiable

- **Matern**: Smooth with tunable roughness via \\\nu\\

- **Gaussian**: \\C(d) = \exp(-(d/\phi)^2)\\ - infinitely differentiable

- **Periodic**: \\C(d) = \exp(-2\sin^2(\pi d/p)/\phi^2)\\ - for seasonal
  data

**Implementation**: the exponential kernel (equivalently Matern with
`nu = 0.5`) is an Ornstein-Uhlenbeck process, so its joint density
factorizes into a first-order Markov chain and evaluates in O(T) with no
matrix. The other kernels have no finite-dimensional state-space form
and are evaluated from a dense T x T Cholesky, where T is the number of
distinct time points.

The field is fit on the sampler path: its two hyperparameters
(`log_sigma2_temporal_gp`, `logit_phi_temporal_gp`) are sampled jointly
with the field and the fixed effects, so pass a sampler `mode` such as
`mode = "hmc"`. There is no nested-Laplace kernel for it.

## See also

[`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)
for equally-spaced temporal effects,
[`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
for spatial GP effects

## Examples

``` r
# Create GP temporal specification
temporal_gp("timestamp")
temporal_gp("day", cov = "matern", nu = 1.5)
temporal_gp("month", cov = "periodic", period = 12)

# \donttest{
# Irregularly-spaced time series
set.seed(140)
times <- sort(runif(40, 0, 10))
df <- data.frame(
  time = times,
  x = rnorm(40),
  y = rbinom(40, 1, 0.5)
)

fit <- tulpa(
  y ~ x,
  data = df,
  family = "binomial",
  temporal = temporal_gp("time"),
  mode = "hmc",
  control = list(n_iter = 200, warmup = 100, n_chains = 1)
)
# }
```
