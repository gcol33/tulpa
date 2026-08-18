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
  beta_prior = list(mean = 0, sd = 10),
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
  Gaussian on every coefficient with SD `sd` (default
  `list(mean = 0, sd = 10)`). The Polya-Gamma sampler uses a mean-zero
  prior, so a non-zero `mean` errors.

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
  1000), `thin` (default 1), `seed` (`NULL` draws from the session RNG;
  the Polya-Gamma kernels use R's RNG, so a seed makes the fit
  reproducible), `verbose` (default FALSE), `n_threads` (default 1).

## Value

List with beta draws, RE draws, sigma_re draws (plus the spatial field
draws when `spatial` is supplied)

## Details

For `family = "neg_binomial_2"` the Polya-Gamma weights are drawn at the
exact real shape `PG(y + r, eta)` and the dispersion `r` is updated by a
random-walk Metropolis-Hastings step on `log(r)` whose stationary
support is bounded to `r` in `[0.1, 500]`; data favouring a dispersion
outside that range pile up at the boundary.

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
colMeans(fit$beta)
#> [1] 0.05346681 0.52102042
# }
```
