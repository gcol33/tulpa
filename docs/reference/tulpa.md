# Fit a tulpa model

Single entry point for fitting a Bayesian hierarchical model. `tulpa()`
parses the formula, builds the model matrices, selects an inference
backend through the tier/mode system (see
[`inference_mode_info()`](https://gillescolling.com/tulpa/reference/inference_mode_info.md)),
assembles the arguments that backend needs, and dispatches.

The fit conditions on the random-effect standard deviations `sigma_re`
(and, for non-Gaussian dispersion, `phi`): both the Laplace (Tier 2) and
the sampler (Tier 1) paths target the posterior given these. Integrating
over the hyperparameters is the role of the nested-Laplace / EM layer.

## Usage

``` r
tulpa(
  formula,
  data,
  family = "gaussian",
  mode = "auto",
  sigma_re = NULL,
  n_trials = NULL,
  weights = NULL,
  phi = 1,
  estimate_phi = FALSE,
  phi2 = NULL,
  beta_prior = NULL,
  re_prior = NULL,
  ziformula = NULL,
  zi_prior = NULL,
  warm_start = NULL,
  spatial = NULL,
  temporal = NULL,
  control = list(),
  ...
)
```

## Arguments

- formula:

  A model formula. Fixed effects, `(1 | g)` / `(1 + x | g)` random
  effects, and `offset(...)` terms are recognised.

- data:

  A data frame.

- family:

  Character family name: one of
  [`family_names()`](https://gillescolling.com/tulpa/reference/family_names.md)
  (`"binomial"`, `"poisson"`, `"neg_binomial_2"`, `"gaussian"`,
  `"beta"`, ...), or a categorical response family – `"multinomial"`
  (baseline-category logit via
  [`tulpa_multinomial()`](https://gillescolling.com/tulpa/reference/tulpa_multinomial.md)),
  `"ordinal"` (cumulative logit via
  [`tulpa_ordinal()`](https://gillescolling.com/tulpa/reference/tulpa_ordinal.md)),
  or `"ordinal_probit"` (cumulative probit). Categorical families take
  fixed-effect models only.

- mode:

  Inference mode or backend. `"auto"` (default) picks the most reliable
  Tier 1/Tier 2 method expected to finish; a tier (`"exact"`,
  `"structured"`) or a backend name (`"laplace"`, `"mala"`, ...) forces
  it. `"eb"` estimates the random-effect covariance(s) by empirical
  Bayes instead of conditioning on `sigma_re` (see
  [`tulpa_eb()`](https://gillescolling.com/tulpa/reference/tulpa_eb.md));
  it is opt-in by name, because its intervals are conditional on that
  estimate rather than marginal over it.

- sigma_re:

  Random-effect SDs to condition on: length 1 (recycled) or one per RE
  term. Defaults to 1 per term with a message. Ignored by the backends
  that determine the covariance themselves (`"eb"`, `re_cov_nested`,
  `re_cov_gibbs`, `gibbs`, `agq`), which warn if it is supplied anyway.

- n_trials:

  Binomial denominators (length `nrow(data)`), or `NULL`.

- weights:

  Optional observation weights (non-negative numeric vector, length
  `nrow(data)`): each observation's log-likelihood contribution is
  scaled by its weight (prior / frequency weights, e.g. survey weights
  or aggregated-data counts – a weight of 2 is equivalent to duplicating
  the row). Supported on the non-spatial Laplace path
  (`mode = "laplace"`) and the log-posterior samplers (`mala`,
  `imh_laplace`, `pathfinder`); other backends reject weights loudly.

- phi:

  Dispersion/precision passed to the family (residual variance for
  gaussian and lognormal, size for neg_binomial_2, precision for beta,
  scale for t). The variance convention holds across every backend; the
  SD-parameterized compiled kernels receive `sqrt(phi)` at the boundary.

- estimate_phi:

  Estimate the dispersion from the data instead of conditioning on
  `phi`, which then supplies the starting value. `log(phi)` joins the
  empirical-Bayes maximization as one further coordinate carrying the
  exact derivative of the Laplace log-marginal, so the estimate is
  ML-II: the hyperprior covers the random-effect covariances only and
  the dispersion enters unpenalized. `fit$phi` is the estimate and
  `fit$phi_estimated` distinguishes it from a conditioned value.

  Available under `mode = "eb"`, and for the families whose dispersion
  derivative is registered (see
  [`tulpa_eb()`](https://gillescolling.com/tulpa/reference/tulpa_eb.md)).
  Any other mode errors rather than fitting at the starting value under
  a name that says otherwise.

- phi2:

  Optional second dispersion: the Student-t degrees of freedom
  (`family = "t"`; default 4 when `NULL`) or the Tweedie variance power
  (`family = "tweedie"`, required – a defaulted power would be a
  statistical decision the caller never made). Supported on the
  non-spatial Laplace path, the random-effect covariance paths
  (`mode = "eb"` and the nested `Sigma` integrator, which thread it into
  their inner Laplace solve), the log-posterior samplers, and the
  ModelData samplers. Backends without a `phi2` channel refuse it rather
  than fit at the family's default. `estimate_phi` covers `phi` alone;
  `phi2` is always conditioned on.

- beta_prior:

  Optional `list(mean, sd)` Gaussian prior on the fixed effects. `NULL`
  takes the engine default, `prior_normal(0, 2.5)`, on every backend
  that carries a fixed-effect prior – the prior is a modelling
  statement, so the backend `mode = "auto"` selects does not change it.
  The nested-Laplace and SPDE paths hold their own field-conditional
  prior and reject a supplied `beta_prior`. The resolved prior is
  reported on the fit as `$beta_prior`.

- re_prior:

  Optional [`list()`](https://rdrr.io/r/base/list.html) of random-effect
  / variance-component hyperpriors (statistical, so they live in the
  signature rather than in `control`). Recognised entries, each consumed
  by the backend that needs it: `hyperprior` (`"flat"` default or
  `"pc_lkj"`, `mode = "laplace"` random slopes and `mode = "eb"` – see
  [`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)),
  `prior_sigma` (PC-prior anchor `c(U, alpha)` on a free RE covariance
  SD, used when `hyperprior = "pc_lkj"`), `eta` (LKJ concentration for a
  correlated RE covariance, same condition), `prior_df` / `prior_scale`
  (inverse-Wishart on the RE covariance, `control$re_cov = "gibbs"`),
  `prior_sigma_scale` (half-Cauchy scale on the RE SD for
  `mode = "gibbs"`), and `sigma_re_scale` (half-Cauchy scale on the RE /
  BYM2 SD for the ModelData samplers).

- ziformula:

  Optional one-sided formula for the zero-inflation probability, e.g.
  `~ 1` for a constant structural-zero rate or `~ x` to model it. The
  response becomes a mixture: with probability `plogis(X_zi beta_zi)`
  the observation is a structural zero, otherwise it is drawn from
  `family`. Available for the count families with a compiled
  zero-inflated kernel; paired with `truncated_poisson` or
  `truncated_neg_binomial_2` it is the hurdle model, since the base
  `P(Y = 0)` is then 0 and the mixture degenerates to the two-part
  likelihood. Backends that do not carry the mixture refuse it rather
  than fit the model without it.

- zi_prior:

  Optional `list(sd)` Gaussian prior on the zero-inflation coefficients,
  `beta_zi ~ N(0, sd^2)`; `NULL` (default) uses 2.5. One scalar SD
  applies to the whole block, and the mean is fixed at 0, because that
  is what the compiled kernels carry. The prior is what identifies the
  logit where a level contributes no zeros – there the likelihood is
  monotone in that coefficient and alone would send it to `-Inf`.
  `sd = Inf` removes the penalty. Ignored without `ziformula`.

- warm_start:

  Optional starting point for the NUTS sampler, from a cheaper fit of
  the same model: `"eb"` or `"laplace"` fits one first, or pass an
  existing fit from either mode. The sampler then starts at that mode
  with an inverse mass read off its curvature, instead of at the origin
  with a structural one. Chains after the first are dispersed around the
  mode at the fit's own scale, so between-chain spread – which `rhat()`
  compares against – is not collapsed by the shared starting point. Only
  the NUTS/HMC backends take one; the rest error rather than ignore it.
  Not available under a spatial, temporal or GP field, whose
  hyperparameters neither source fit estimates.

  The variance-component slots take an adapting mass by default, because
  a plug-in fit estimates no curvature for them. Passing a fit from
  [`tulpa_eb()`](https://gillescolling.com/tulpa/reference/tulpa_eb.md)
  with `marginal = TRUE` supplies one: its `theta_cov` gives each
  `log_sigma_re` slot a posterior variance to start from. This applies
  to uncorrelated terms, whose hyperparameter coordinates are the log
  standard deviations the sampler holds; a correlated term stays
  adapting, since its log-Cholesky coordinates are not the sampler's.

- spatial:

  Optional spatial-field spec. How it is addressed depends on the field
  family:

  - **Areal** (`"icar"`, `"car"`, `"bym2"`, `"car_proper"`): a list with
    `type` and `adjacency`, paired with a `spatial(col)` term in
    `formula` naming the per-observation unit column. Term and spec must
    be supplied together.

  - **Continuous** (`spatial_gp(~ lon + lat)` for an NNGP field,
    `spatial_gp(~ lon + lat, approx = 'hsgp')` for a Hilbert-space GP,
    `spatial_spde(~ lon + lat, data)` for a Matern SPDE field): the spec
    object carries the coordinate columns (the SPDE spec also carries
    the mesh + FEM matrices), so **no** `spatial(col)` term is used –
    observations are mapped to locations from their coordinates.

  The mode selects how the spatial hyperparameter is handled:

  - `mode = "nested_laplace"`, `"structured"`, and `"auto"` (when not
    the binomial Gibbs case below) **integrate** the hyperparameter –
    the designed Tier 2 path, mirroring `latent(...)` blocks. Areal
    `icar`/`car`/ `bym2`/`car_proper` and continuous `gp`/`nngp`/`hsgp`
    go through
    [`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md);
    SPDE is redirected to
    [`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md),
    which integrates `(range, sigma)` with its own CCD / grid design.

  - `mode = "laplace"` **conditions** on a fixed hyperparameter via
    [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
    (the cheap explicit fit).

  - `mode = "gibbs"` routes the areal `icar`/`bym2` cases through the
    binomial Polya-Gamma samplers (Tier 1 exact); `mode = "auto"` picks
    this for a binomial `icar`/`bym2` field.

- temporal:

  Optional temporal field spec
  ([`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
  [`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md),
  or
  [`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)),
  integrated by nested Laplace. A plain field routes the single-block
  temporal kernel; a `group_var` panel spec fits a separate walk per
  group sharing one hyperparameter; combined with an areal `spatial`
  field it forms an additive space-time joint prior.

- control:

  Optional list of backend tuning arguments (e.g. `n_iter`, `warmup`,
  `epsilon` for `mala`; `n_draws` for `pathfinder`).

- ...:

  Reserved for future statistical arguments. Nothing is read from it
  today, so any entry errors: a stray name here is a misspelled argument
  or a tuning knob that belongs in `control`.

## Value

A `tulpa_fit` object carrying the backend's output plus
`inference_mode`, `inference_tier`, `backend`, `selection_reason`,
`formula`, and `family`. Two field-name conventions to know when
reaching into the object directly (the generic accessors handle both):
on nested-Laplace fits `$weights` is the hyperparameter GRID weights;
user observation weights are stored as `$obs_weights`. `$draws` is a
draws matrix on engine fits, while model-package fits may carry a list
(`$y_rep`, `$log_lik`) under the same name.

## Coverage

- **No random effects** and **random intercepts** (`(1 | g)`) are
  supported on the design path (`mode = "laplace"`) and the sampler path
  (`mode = "mala"`, `"pathfinder"`, `"imh_laplace"`, and the ModelData
  kernels `"hmc"` / `"sghmc"` / `"sgld"` / `"mclmc"` / `"smc"` / `"vi"`
  / `"ess"`).

- **Random slopes** are supported on the Laplace (Tier 2) path: there is
  no scalar `sigma_re` to condition on, so the RE covariance `Sigma` is
  integrated rather than fixed. This covers correlated terms
  (`(1 + x | g)`, a full `Sigma`), uncorrelated terms (`(1 + x || g)`, a
  diagonal `Sigma`), and several terms together
  (`(1 + x | g) + (1 | h)`) – each term becomes a covariance block, and
  any accompanying `(1 | g)` term is integrated as a 1x1 block (nothing
  is silently conditioned at `sigma_re = 1`). `mode = "laplace"` routes
  to the nested-Laplace `Sigma` integrator
  ([`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md),
  CCD design, flat-in-log hyperprior by default – see
  `re_prior$hyperprior`); `control$re_cov = "gibbs"` switches to the
  exact Metropolis-within-Gibbs debias
  ([`tulpa_re_cov_gibbs()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_gibbs.md)),
  and `control$re_cov = "aghq"` keeps the nested integrator but replaces
  the inner joint-Laplace marginal with adaptive Gauss-Hermite
  quadrature (`control$n_quad`, default 9 there; see `n_quad` in
  [`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)).
  Both also run on the sampler path (`mode = "mala"` / `"pathfinder"`).

- `mode = "gibbs"` (Polya-Gamma) fits a single random-intercept model
  for `family = "binomial"` or `"neg_binomial_2"`, and **samples** the
  RE sd rather than conditioning on `sigma_re`; tune it via
  `re_prior$prior_sigma_scale` and a mean-zero `beta_prior`.

- **Latent prior blocks** (`latent(tgmrf(...))`) route to the
  nested-Laplace path (Tier 2), which integrates over the block
  hyperparameters. `mode = "auto"` and `"structured"` select it
  automatically when latent blocks are present;
  `mode = "nested_laplace"` forces it. At most one random-intercept
  `(1 | g)` term may accompany the blocks (model richer grouping as an
  `iid` block). Joint multi-arm nested models cannot be expressed by a
  single-response formula – call
  [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)
  directly.

- The ModelData sampler kernels (`"hmc"`, `"ess"`, `"sghmc"`, `"sgld"`,
  `"mclmc"`, `"smc"`, `"vi"`) thread the full latent vector – fixed
  effects, random effects (all forms), areal spatial (`icar` / `bym2`),
  and temporal (`rw1` / `rw2` / `ar1`) – through one ModelData builder
  and sample the variance components jointly with the field. `ess`
  carries random effects but declines a structured spatial / temporal
  block (its isotropic Gaussian-prior block cannot represent the graph
  precision); continuous-coordinate fields (`gp` / `nngp` / `hsgp` /
  `spde`), `car_proper`, and exotic latent blocks stay on the dedicated
  nested-Laplace / SPDE / Polya-Gamma paths.

## References

Rue, Martino & Chopin (2009). Approximate Bayesian inference for latent
Gaussian models by using integrated nested Laplace approximations.
*JRSS-B* 71(2):319-392. Hoffman & Gelman (2014). The No-U-Turn Sampler:
adaptively setting path lengths in Hamiltonian Monte Carlo. *JMLR*
15(47):1593-1623.

## See also

[`inference_mode_info()`](https://gillescolling.com/tulpa/reference/inference_mode_info.md),
[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md),
[`mala()`](https://gillescolling.com/tulpa/reference/mala.md),
[`pathfinder()`](https://gillescolling.com/tulpa/reference/pathfinder.md)

## Examples

``` r
# \donttest{
set.seed(1)
n <- 200L
g <- sample(letters[1:12], n, replace = TRUE)
d <- data.frame(
  y = rbinom(n, 1, plogis(-0.3 + 0.6 * rnorm(n))),
  x = rnorm(n),
  g = g
)
# Random-intercept logistic GLMM, Laplace tier.
fit <- tulpa(y ~ x + (1 | g), data = d, family = "binomial", mode = "laplace")
#> Warning: tulpa(): `sigma_re` not supplied; conditioning on sigma_re = 1 for each of the 1 RE term(s). Pass `sigma_re` to override.
coef(fit)
#> (Intercept)           x 
#>  -0.4031451   0.1952674 
summary(fit)
#>               estimate std.error        2.5%     97.5%
#> (Intercept) -0.4031451 0.3251204 -1.04036939 0.2340792
#> x            0.1952674 0.1382947 -0.07578526 0.4663201
# }
```
