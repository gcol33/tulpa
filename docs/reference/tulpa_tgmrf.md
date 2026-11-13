# Fit a custom tgmrf latent block

One front door for inference over a user-defined
[`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md) latent
block's hyperparameter vector `theta`. `mode` selects the inference
target and correction:

- `"imh"` (default) – Tier-1 pseudo-marginal MCMC against an unbiased
  importance estimate of `p(y | theta)`, with jointly resampled latent
  draws.

- `"nuts"` – NUTS on the Laplace-approximate theta marginal, followed by
  PSIS importance correction to the joint posterior. Tier 2: finite
  weighted approximation. The correction is applied only when its
  Pareto-k passes the sample-size-dependent reliability threshold.

- `"vi"` – Tier-2 structured: a single-path Pathfinder Gaussian fit of
  the Laplace-approximate theta marginal.

- `"nuts_joint"` – Tier-1 exact MCMC sampling the FULL joint
  `(beta, z, theta)` against its true density (requires a C++-backend
  block, see
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
Every mode reports the fixed effects: `$draws` holds the `beta` columns
first, any retained random effects and latent `z`, then `theta`, so
[`coef()`](https://rdrr.io/r/stats/coef.html),
[`summary()`](https://rdrr.io/r/base/summary.html) and
[`posterior_sample()`](https://gillescolling.com/tulpa/reference/posterior_sample.md)
read them as on any other fit. `"imh"` returns the selected IS particle
retained by pseudo-marginal MH; `"nuts_joint"` samples the full joint
density. Corrected `"nuts"` resamples whole joint rows with PSIS
weights; `$correction_applied`, `$correction_declined`, `$pareto_k`,
`$pareto_k_is_ess` and `$target` describe the result. With
`control$is_correction = FALSE`, too few draws, or a k above the gate,
NUTS returns its uncorrected Laplace theta chain with Gaussian
conditional beta draws, `inference_mode = "structured"`,
`inference_tier = 2L`. Corrected NUTS also remains Tier 2 and carries
`correction = "psis"`; its resampled rows have `draws_kind = "iid"` and
their dependence through the source chain is recorded in `$source_index`
and `$source_theta_draws`. `"vi"` returns a Gaussian theta approximation
and conditional Gaussian beta draws. `$draws_beta`, `$draws_z` (when
available) and `$draws_theta` split the joint rows by block.

The inner IS budget is calibrated at the outer Laplace mode, using
repeated independent likelihood estimates. The package setting
`.TGMRF_IS` sets the target log-estimate SD (1.2), initial budget (16),
calibration batches (16), ceiling (65536), and t proposal df (7).
`$is_particles` and `$log_estimate_sd` report the chosen budget and
measured SD. Each estimate costs that many exact joint-density
evaluations plus one inner mode/precision solve. `$inner_solves` counts
adapter evaluations after the pilot; `$is_likelihood_evaluations` also
includes calibration.

## References

Andrieu and Roberts (2009), The pseudo-marginal approach for efficient
Monte Carlo computations, Annals of Statistics 37:697-725,
[doi:10.1214/07-AOS574](https://doi.org/10.1214/07-AOS574) . Each MH
proposal pays for an unbiased inner estimate and retains that estimate
on rejection.

Doucet, Pitt, Deligiannidis and Kohn (2015), Efficient implementation of
Markov chain Monte Carlo when using an unbiased likelihood estimator,
Biometrika 102:295-313,
[doi:10.1093/biomet/asu075](https://doi.org/10.1093/biomet/asu075) .
Repeated pilot estimates calibrate the particle budget against
log-estimate variability.

Vehtari, Simpson, Gelman, Yao and Gabry (2024), Pareto smoothed
importance sampling, JMLR 25(72):1-58,
<https://www.jmlr.org/papers/v25/19-556.html>. NUTS correction pays one
IS estimate per retained theta and a tail fit.

Gomez-Rubio and Rue (2018), Markov chain Monte Carlo with the integrated
nested Laplace approximation, Statistics and Computing 28:1033-1051,
[doi:10.1007/s11222-017-9778-y](https://doi.org/10.1007/s11222-017-9778-y)
. MCMC with an INLA marginal retains its deterministic approximation
error.

Rue, Martino and Chopin (2009), Approximate Bayesian inference for
latent Gaussian models by using integrated nested Laplace
approximations, JRSS B 71:319-392,
[doi:10.1111/j.1467-9868.2008.00700.x](https://doi.org/10.1111/j.1467-9868.2008.00700.x)
. The inner mode and sparse precision solve build the proposal at each
theta.

## See also

[`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md) for the
block,
[`tgmrf_cpp()`](https://gillescolling.com/tulpa/reference/tgmrf_cpp.md)
for the compiled-block form.

## Examples

``` r
# \donttest{
# An iid random intercept written as a tgmrf block over 15 groups; the
# inference method is an argument, not a parallel verb. See the
# custom-process article at https://gillescolling.com/tulpa/articles/tgmrf.html.
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
# }
```
