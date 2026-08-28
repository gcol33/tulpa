# Sample an SPDE GLM via NUTS, optionally jointly over Matern hypers

Bayesian counterpart to
[`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md) /
[`laplace_spde_at()`](https://gillescolling.com/tulpa/reference/laplace_spde_at.md).
Routes the SPDE-augmented GLM through tulpa's full NUTS backend via the
generic `LikelihoodSpec` interface. Two modes:

- **Legacy / fixed-hyper** (`joint = FALSE`, the default): conditions on
  the supplied `(range, sigma)`. Samples the latent block
  `(beta, w_mesh, log_phi)` jointly. Useful as the inner step of an
  outer CCD / nested-Laplace integration over hypers.

- **Joint hypers** (`joint = TRUE`): samples
  `(beta, z_mesh, log_kappa, log_tau, log_phi)` jointly via the
  non-centered transform `w = L(theta)^{-T} z`. A PC prior (Fuglstad et
  al. 2019 JASA) on `(range, sigma)` is taken from `prior_range`,
  `prior_sigma`. Returns additional `w_draws`, `range_draws`,
  `sigma_draws`, `kappa_draws`, `tau_draws` columns/vectors on top of
  the raw parameter draws.

## Usage

``` r
tulpa_nuts_spde(
  y,
  X,
  spatial,
  family = c("gaussian", "poisson", "binomial", "gamma", "neg_binomial_2", "beta"),
  n_trials = NULL,
  joint = FALSE,
  range = NULL,
  sigma = NULL,
  prior_range = NULL,
  prior_sigma = NULL,
  log_kappa_init = NULL,
  log_tau_init = NULL,
  beta_prior = .tulpa_default_beta_prior("spde_nuts"),
  log_phi_prior_sd = 3,
  log_phi_init = 0,
  control = list()
)
```

## Arguments

- y:

  Response vector. Family-specific:

  - `gaussian`: any real

  - `poisson`, `neg_binomial_2`: non-negative integers

  - `binomial`: non-negative integers in `[0, n_trials]`

  - `gamma`: strictly positive reals

  - `beta`: strictly in `(0, 1)`

- X:

  Fixed-effects design matrix.

- spatial:

  A `tulpa_spatial` object from
  [`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md)
  /
  [`spatial_spde_custom()`](https://gillescolling.com/tulpa/reference/spatial_spde_custom.md)
  – supplies the FEM matrices (C0, G1) and projection (A) plus the
  smoothness `nu`.

- family:

  One of `"gaussian"`, `"poisson"`, `"binomial"`, `"gamma"`,
  `"neg_binomial_2"`, `"beta"`.

- n_trials:

  Integer vector for `family = "binomial"` (else ignored).

- joint:

  Logical. `FALSE` (default) conditions on fixed `(range, sigma)`.
  `TRUE` activates joint sampling of `(log_kappa, log_tau)` with the PC
  prior from `prior_range`, `prior_sigma`.

- range, sigma:

  Fixed-hyper mode only. Matern range and marginal SD on the field.
  Default to the SPDE prior medians (`spatial$prior_range[1]`,
  `spatial$prior_sigma[1]`).

- prior_range, prior_sigma:

  Joint mode only. PC prior anchors as `c(value, alpha)` pairs:

  - `prior_range = c(r0, a_r)` encodes `P(range < r0) = a_r`

  - `prior_sigma = c(s0, a_s)` encodes `P(sigma > s0) = a_s` Default to
    the spec's anchors (`spatial$prior_range`, `spatial$prior_sigma`)
    when those are length-2 PC anchors.

- log_kappa_init, log_tau_init:

  Joint mode only. Initial values for the hyper slots. Default to the
  value implied by the PC anchor's `(r0, s0)` pair via
  `kappa = sqrt(8 nu) / r0`, `tau = 1 / (sqrt(4 pi) * kappa * s0)`.

- beta_prior:

  Fixed-effect prior as `list(mean, sd)`: a mean-zero (`mean = 0`)
  Gaussian on each coefficient with SD `sd` (default the engine default,
  `prior_normal(0, 2.5)`).

- log_phi_prior_sd:

  Prior SD on `log(phi)`. Role of `phi` is family-specific:

  - `gaussian`: `phi` is the residual SD (sampled jointly)

  - `gamma`: `phi` is the Gamma shape (sampled jointly)

  - `neg_binomial_2`: `phi` is the NB size `r` (sampled jointly)

  - `beta`: `phi` is the Beta precision (sampled jointly)

  - `poisson`, `binomial`: `log_phi` is held tight and ignored
    downstream.

- log_phi_init:

  Starting value for `log(phi)`.

- control:

  A named list of numerical / sampler knobs (statistical arguments stay
  in the signature): `n_iter` (default 2000), `n_warmup` (default 1000),
  `max_treedepth` (default 10), `adapt_delta` (default 0.8), `seed`
  (`NULL` draws from the session RNG), `verbose` (default FALSE),
  `noncenter` (fixed-hyper only, default TRUE: sample the mesh field in
  a non-centered `v = L^{-T} z` parameterisation – the same target
  density, a priori isotropised; ignored when `joint = TRUE`), and
  `mass_matrix` (NUTS metric: `"auto"` (default) picks a dense metric at
  `<= 200` parameters and diagonal above, capturing the fixed-effect /
  field cross-curvature that corrects the intercept marginal; `"diag"`,
  `"dense"`, `"block_diag"` force the choice).

## Value

A list with `draws` (matrix `n_samples x n_params`), `means`,
`phi_summary` (where applicable), `accept_prob`, `divergent`,
`treedepth`, `epsilon`, `joint_hypers`, plus the supplied `spatial`
spec. In `joint = TRUE` mode additionally: `w_draws` (transformed
`z -> w` per draw), `range_draws`, `sigma_draws`, `kappa_draws`,
`tau_draws`, and `range_summary`, `sigma_summary` (mean/median/5%/95%
quantiles).

## See also

[`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md)
for the Laplace counterpart and the nested-Laplace path over (range,
sigma).

## Examples

``` r
if (FALSE) { # \dontrun{
# SPDE-field NUTS. `spatial` is an SPDE spec built from a mesh, e.g.
# spatial_spde(~ x + y, df). See fit_spde() for the Laplace analogue.
fit <- tulpa_nuts_spde(y, X, spatial = spatial_spde(~ x + y, df))
} # }
```
