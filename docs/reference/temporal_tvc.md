# Time-varying coefficient structure

Specify a time-varying coefficient (TVC): one or more fixed-effect
coefficients are allowed to evolve over time, with the evolution
governed by a temporal prior (`rw1`, `rw2`, `ar1` or `gp`).

`rw1`, `rw2` and `ar1` read the time index as a position on a grid, so
consecutive instants are one step apart whatever the data says. `gp` is
the continuous-time structure: the coefficient is a Gaussian process
over the distinct time VALUES, which is what irregular spacing needs. It
is distinct from
[`temporal_gp()`](https://gillescolling.com/tulpa/reference/temporal_gp.md),
a GP over time entering the linear predictor additively
(`eta_i += f(t_i)`); here the GP IS a coefficient
(`eta_i += x_i w(t_i)`).

## Usage

``` r
temporal_tvc(
  time_var,
  terms = 1,
  structure = c("rw1", "rw2", "ar1", "gp"),
  cov = c("exponential", "matern", "gaussian", "periodic"),
  nu = 1.5,
  period = NULL,
  group_var = NULL,
  shared = NULL,
  sigma_prior_U = 1,
  sigma_prior_alpha = 0.01,
  scale_coords = TRUE
)
```

## Arguments

- time_var:

  Single character string naming the time variable in the data.
  `structure = "gp"` needs it numeric: a factor states an ordering with
  no spacing for the kernel to measure.

- terms:

  Which coefficients vary over time. A formula, an integer vector of
  design-matrix column indices, or a character vector of term names.
  Default `1` (the intercept).

- structure:

  Temporal prior governing how the coefficients evolve. One of `"rw1"`,
  `"rw2"`, `"ar1"` or `"gp"`.

- cov, nu, period:

  Covariance kernel for `structure = "gp"`, ignored otherwise. `cov` is
  one of `"exponential"` (the default), `"matern"`, `"gaussian"` or
  `"periodic"`; `nu` is the Matern smoothness, closed-form at `0.5`,
  `1.5` and `2.5` only (`0.5` IS the exponential kernel); `period` is
  the periodic kernel's period. Exponential and Matern `nu = 0.5`
  evaluate in `O(T)` through the Ornstein-Uhlenbeck factorization; the
  rest take a dense `T x T` Cholesky per coefficient per gradient
  evaluation.

- group_var:

  Optional character string naming a grouping variable for
  group-specific time-varying coefficients.

- shared:

  Whether the effect is shared across processes in a multi-process
  model. `NULL` (default) shares it; `FALSE` fits process-specific
  effects and emits a warning.

- sigma_prior_U, sigma_prior_alpha:

  Penalized-complexity prior on each varying coefficient's marginal
  standard deviation, calibrated so that
  `P(sigma > sigma_prior_U) = sigma_prior_alpha`. Defaults to
  `P(sigma > 1) = 0.01`. `sigma_prior_U` must be positive and
  `sigma_prior_alpha` must lie in `(0, 1)`. Read on every structure: it
  is the same anchor pair whether the field samples a log-precision
  (`rw1` / `rw2` / `ar1`) or a log-variance (`gp`).

- scale_coords:

  Logical, `structure = "gp"` only: standardize the time values before
  fitting (default `TRUE`), which puts the lengthscale on the same
  universal support
  [`temporal_gp()`](https://gillescolling.com/tulpa/reference/temporal_gp.md)
  uses. `period` is stated in the raw time units and makes the same
  trip.

## Value

A `tulpa_tvc` object.

## See also

[`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
[`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md),
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)
for the underlying temporal priors;
[`temporal_gp()`](https://gillescolling.com/tulpa/reference/temporal_gp.md)
for a GP over time that is not a varying coefficient.

## Examples

``` r
# Intercept that drifts as a first-order random walk over year
temporal_tvc("year", structure = "rw1")
#> tulpa temporally-varying coefficients
#> ======================================
#> 
#> Time variable: year 
#> Structure: RW1 (first-order random walk) 
#> Shared: Yes (enters both processes) 
#> 
#> Terms: columns  1 

# A slope evolving as a continuous-time GP over irregularly-spaced visits
temporal_tvc("day", terms = ~ x - 1, structure = "gp", cov = "matern")
#> tulpa temporally-varying coefficients
#> ======================================
#> 
#> Time variable: day 
#> Structure: GP (Gaussian process over the distinct times) 
#> Covariance: matern (nu = 1.5) 
#> Shared: Yes (enters both processes) 
#> 
#> Terms: ~x - 1 
```
