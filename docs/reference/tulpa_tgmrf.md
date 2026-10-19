# Fit a custom tgmrf latent block

One front door for inference over a user-defined
[`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md) latent
block's hyperparameter vector `theta`. `mode` selects the inference
engine, all of which share the same Laplace body for
`(beta, z) | theta`:

- `"imh"` (default) – Tier-1 exact MCMC: the Laplace body plus an
  independence-Metropolis bias correction over `theta` (the "Laplace +
  MH debias" composition).

- `"nuts"` – Tier-1 exact MCMC: NUTS over the marginal `theta`
  posterior.

- `"vi"` – Tier-2 structured: a single-path Pathfinder Gaussian fit of
  the `theta` posterior (no bias correction).

- `"nuts_joint"` – Tier-1 exact MCMC sampling the FULL joint
  `(beta, z, theta)` rather than the Laplace-marginalized `theta`
  (requires a C++-backend block, see
  [`tgmrf_cpp()`](https://gillescolling.com/tulpa/reference/tgmrf_cpp.md)).

The inference method is an argument, not a parallel verb.

## Usage

``` r
tulpa_tgmrf(
  y,
  n_trials,
  X,
  block,
  family = "binomial",
  phi = 1,
  re_idx = NULL,
  n_re_groups = 0L,
  sigma_re = 1,
  mode = c("imh", "nuts", "vi", "nuts_joint"),
  control = list(),
  ...
)
```

## Arguments

- y, n_trials, X:

  Response, binomial trial counts (or `NULL`), and the fixed-effect
  design matrix.

- block:

  A [`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md)
  latent block.

- family, phi:

  Observation family and its dispersion.

- re_idx, n_re_groups, sigma_re:

  Optional scalar random-intercept structure.

- mode:

  Inference engine: `"imh"`, `"nuts"`, `"vi"`, or `"nuts_joint"`.

- control:

  A list of numerical / tuning knobs for the chosen `mode` (e.g.
  `n_iter`, `warmup`, `thin`, `scale` for `"imh"`; `epsilon`,
  `max_depth`, `target_accept` for the NUTS modes; `n_draws`,
  `max_lbfgs`, `lbfgs_tol` for `"vi"`; plus the shared
  `pilot_axis_points`, `max_iter`, `tol`, `n_threads`, `verbose`). An
  unknown knob for the chosen `mode` errors rather than being silently
  ignored.

- ...:

  Individual `control` knobs may also be passed directly by name; they
  are merged into `control` (a named argument here wins over the same
  name inside `control`).

## Value

A `tulpa_fit` object; `$backend` and `$mode` record the engine used.

## See also

[`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md) for the
block,
[`tgmrf_cpp()`](https://gillescolling.com/tulpa/reference/tgmrf_cpp.md)
for the compiled-block form.

## Examples

``` r
# \donttest{
# An iid random intercept written as a tgmrf block over 15 groups; the
# inference method is an argument, not a parallel verb. See
# vignette("tgmrf").
set.seed(3)
J <- 15L; g <- rep(seq_len(J), each = 6L); n <- length(g); x <- rnorm(n)
y <- rpois(n, exp(0.3 + 0.5 * x + rnorm(J, 0, 0.6)[g]))
blk <- tgmrf(
  Q = function(theta) Matrix::Diagonal(J, exp(-2 * theta[1])),
  prior = function(theta) stats::dnorm(theta[1], 0, 1, log = TRUE),
  init = c(log_sigma = 0), obs_idx = g)
fit <- tulpa_tgmrf(y, rep(1L, n), cbind(1, x), block = blk,
                   family = "poisson", mode = "imh",
                   control = list(n_iter = 300L, warmup = 150L))
mean(exp(fit$draws[, "log_sigma"]))   # posterior mean group SD
#> [1] 0.6861869
# }
```
