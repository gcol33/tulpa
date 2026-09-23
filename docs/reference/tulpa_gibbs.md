# Fit via Polya-Gamma Gibbs sampler

Public API for PG Gibbs sampling. Used by model packages for binomial
and negative binomial GLMMs.

## Usage

``` r
tulpa_gibbs(
  y,
  n_trials,
  X,
  group,
  n_groups,
  family = "binomial",
  beta_prior = .tulpa_default_beta_prior("gibbs"),
  prior_sigma_scale = 2.5,
  spatial = NULL,
  temporal = NULL,
  control = list()
)
```

## Arguments

- y:

  Response vector

- n_trials:

  Trial sizes (binomial)

- X:

  Design matrix

- group:

  Integer vector of group indices (1-based)

- n_groups:

  Number of groups

- family:

  Character: "binomial" or "neg_binomial_2"

- beta_prior:

  Fixed-effect prior as `list(mean, sd)`: a mean-zero (`mean = 0`)
  Gaussian on every coefficient with SD `sd` (default the engine
  default, `prior_normal(0, 2.5)`). The Polya-Gamma sampler uses a
  mean-zero prior, so a non-zero `mean` errors.

- prior_sigma_scale:

  Prior scale for RE sigma (statistical; default 2.5).

- spatial:

  Optional spatial spec. When supplied the fit routes to the matching
  spatial Polya-Gamma Gibbs sampler via
  [`dispatch_gibbs_spatial()`](https://gillescolling.com/tulpa/reference/dispatch_gibbs_spatial.md);
  `group`/`n_groups` are the iid random-effect block carried alongside
  the field. The full areal + continuous family is available for
  `family = "binomial"`; `family = "neg_binomial_2"` is backed by the
  areal ICAR negbin sampler only. Supported `type`s:

  - areal – `"icar"`, `"bym2"`, `"rsr"`: a list with `type`, `adjacency`
    and a 1-based `spatial_idx` per observation (e.g.
    `list(type = "icar", adjacency = W, spatial_idx = unit)`). `"rsr"`
    reuses `spatial$rsr_projection` if present, else builds the
    unit-level projector from the design.

  - continuous – `"gp"`/`"nngp"` (a validated
    [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
    spec) and `"multiscale_gp"` (a validated
    [`spatial_multiscale()`](https://gillescolling.com/tulpa/reference/spatial_multiscale.md)
    spec). These samplers carry no observation-\>location map, so they
    require one observation per unique location in coordinate order.

- temporal:

  Optional temporal spec: a validated
  [`temporal_multiscale()`](https://gillescolling.com/tulpa/reference/temporal_multiscale.md)
  object. Routes to the multiscale temporal Polya-Gamma sampler via
  [`dispatch_gibbs_temporal()`](https://gillescolling.com/tulpa/reference/dispatch_gibbs_temporal.md)
  (binomial only; RW1 trend + cyclic seasonal + AR1/IID short-term).
  Cannot be combined with `spatial`.

- control:

  A named list of numerical / tuning knobs (statistical arguments stay
  in the signature above): `n_iter` (default 2000), `warmup` (default
  1000), `thin` (default 1, applied on every route including the spatial
  and temporal ones; the run keeps `ceiling((n_iter - warmup) / thin)`
  draws), `seed` (`NULL` draws from the session RNG; the Polya-Gamma
  kernels use R's RNG, so a seed makes the fit reproducible), `verbose`
  (default FALSE), `n_threads` (default 1).

## Value

A `tulpa_fit` holding one MCMC chain: `draws`, the `[n_saved x n_param]`
matrix of retained draws, with `chain_id`, `n_chains`, `means` (column
means) and `param_names`. Columns follow the parameter naming of the
ModelData samplers: the fixed effects (the column names of `X`, else
`beta[j]`), `log_sigma_re` and `re[g]` for the random-intercept block,
then the route's own blocks – `log_phi` (the negative-binomial size);
`log_tau_spatial` and `phi_spatial[u]` (ICAR, and RSR's projected
field); `log_sigma_spatial`, `logit_rho_bym2`, `phi_spatial[u]` and
`theta_spatial[u]` (BYM2); `log_sigma2_gp`, `log_phi_gp` and `gp_w[u]`
(GP); the `_local` / `_regional` counterparts with `gp_local[u]` /
`gp_regional[u]` (multiscale GP); `log_sigma2_trend`, `trend[t]`,
`log_sigma2_seasonal`, `seasonal[s]`, `log_sigma2_short`,
`short_term[t]` and, for an AR1 short-term component, `logit_rho_short`
(temporal). Scales are stored on the log scale and correlations on the
logit scale, as the samplers store them; a component the model does not
carry has no column.

## Details

For `family = "neg_binomial_2"` the Polya-Gamma weights are drawn at the
exact real shape `PG(y + r, eta)` and the dispersion `r` is updated by a
random-walk Metropolis-Hastings step on `log(r)` whose stationary
support is bounded to `r` in `[0.1, 500]`; data favouring a dispersion
outside that range pile up at the boundary.

Every sampler that moves a latent effect's level through the first
coefficient – the negative-binomial kernels, and the binomial kernels
carrying a `spatial` or `temporal` field – leaves `eta` unchanged only
when the first column of `X` is an all-ones intercept, and those routes
error on a design without one. An intrinsic field (ICAR, the structured
BYM2 part, RW1) is reported centred with its level in the intercept, and
its sweep carries the intercept's prior through the field mean; a proper
one (the negative-binomial iid block, an NNGP field) has the level it
shares with the intercept drawn from its full conditional, which both
priors define.

## Examples

``` r
set.seed(1)
G <- 20L; npg <- 15L; n <- G * npg
grp <- rep(seq_len(G), each = npg)
X <- cbind(1, rnorm(n))
b <- rnorm(G, 0, 0.6)
y <- rbinom(n, 1, plogis(X %*% c(-0.2, 0.5) + b[grp]))
# \donttest{
fit <- tulpa_gibbs(y, rep(1L, n), X, grp, G, family = "binomial",
                   control = list(n_iter = 500L, warmup = 250L))
coef(fit)
#>    beta[1]    beta[2] 
#> 0.05113498 0.47895299 
diagnostics(fit)
#>       parameter      rhat  ess_bulk  ess_tail
#> 1       beta[1] 0.9994892 174.67795 171.50405
#> 2       beta[2] 1.0079899 195.29629 222.99091
#> 3  log_sigma_re 1.0189955  30.08361  44.55205
#> 4         re[1] 0.9981450 248.67134 205.65889
#> 5         re[2] 1.0103747 159.72752 169.70804
#> 6         re[3] 1.0014289  62.42346 180.58399
#> 7         re[4] 0.9999902  86.58332 183.94880
#> 8         re[5] 1.0055948 258.88968  98.48576
#> 9         re[6] 1.0006344  78.93758 157.76480
#> 10        re[7] 1.0067015 160.72260 190.56408
#> 11        re[8] 1.0035467 217.12126 240.08551
#> 12        re[9] 1.0114679 129.51291 138.50106
#> 13       re[10] 1.0020908 155.25306 216.15668
#> 14       re[11] 1.0002456 230.14582 232.21690
#> 15       re[12] 1.0077990 261.31527 187.34137
#> 16       re[13] 1.0043439 208.37927 234.64435
#> 17       re[14] 0.9994984 217.48683 260.27161
#> 18       re[15] 0.9997324 228.78774 208.60393
#> 19       re[16] 0.9967272 171.25306 165.00215
#> 20       re[17] 0.9971519 187.42189 181.37760
#> 21       re[18] 1.0030375 228.04636 182.03745
#> 22       re[19] 1.0121462 135.01159 202.01066
#> 23       re[20] 1.0050720 233.24839 190.52853
# }
```
