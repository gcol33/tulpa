# Adaptive Gauss-Hermite quadrature for one-RE GLMMs

Marginal-likelihood approximation for a generalised linear mixed model
with one cluster-level intercept random effect. Adaptive Gauss-Hermite
quadrature (AGQ) generalises Laplace by replacing the single-point
Gaussian integral around the cluster's posterior mode with
`n_quad`-point quadrature. AGQ at `n_quad = 1` recovers the Laplace
approximation; higher `n_quad` reduces approximation error, especially
for clusters with few observations or non-Gaussian likelihoods.

This is an engine block, not a front door: its tuning knobs (`max_iter`,
`tol`, `n_quad`) sit in the signature rather than in a `control` list.

Scoped to the `lme4::glmer(..., nAGQ = N)` use case: one intercept- only
RE term, families `binomial`, `poisson`, or `gaussian`. For multi-RE or
random-slope models, use
[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
(Laplace at the joint mode) or HMC.

## Usage

``` r
agq_fit(
  y,
  X,
  group,
  n_groups = max(group),
  family = c("binomial", "poisson", "gaussian"),
  n_trials = NULL,
  sigma_eps = 1,
  n_quad = 7L,
  beta_init = NULL,
  sigma_init = 1,
  max_iter = 200L,
  tol = 1e-06,
  verbose = FALSE
)
```

## Arguments

- y:

  Response vector.

- X:

  Fixed-effects design matrix (`n_obs x p`).

- group:

  Integer cluster labels (1-based, length `n_obs`).

- n_groups:

  Number of clusters (default `max(group)`).

- family:

  One of `"binomial"`, `"poisson"`, `"gaussian"`.

- n_trials:

  Trial sizes (binomial only; default `rep(1, n_obs)`).

- sigma_eps:

  Residual SD (gaussian only; default `1`). Held **fixed** at this value
  – it is not profiled or estimated, so for a gaussian response set it
  to a known / pre-estimated residual SD rather than the default.

- n_quad:

  Number of Gauss-Hermite quadrature nodes per cluster. `1` recovers
  Laplace; common choices are `5` or `7`. Default `7`.

- beta_init:

  Initial fixed-effects (default zeros).

- sigma_init:

  Initial RE SD (default `1`).

- max_iter:

  Optimiser iteration cap (default `200`).

- tol:

  Convergence tolerance (default `1e-6`).

- verbose:

  Print optimiser summary (default `FALSE`).

## Value

A list with class `tulpa_fit` carrying:

- `means`: named vector `c(beta, log_sigma_re)`.

- `mode`: same as `means` (Gaussian fit at the optimum).

- `cov`: covariance matrix of `(beta, log_sigma_re)` from inverse
  observed information.

- `sigma_re`: estimated RE SD.

- `log_marginal`: AGQ-approximated log marginal likelihood at the fitted
  values.

- `n_quad`: quadrature order used.

- `n_iter`, `converged`: optimiser diagnostics.

- `inference_mode`: `"structured"`.

- `inference_tier`: `2L`.

- `backend`: `"agq"`.

## Tier

Tier 2 (Structured). AGQ is exact in the limit `n_quad -> infinity` but
at finite `n_quad` it is a controlled approximation – same epistemic
class as Laplace.

## References

Pinheiro, J. C., & Bates, D. M. (1995). Approximations to the
log-likelihood function in the nonlinear mixed-effects model. *Journal
of Computational and Graphical Statistics*, 4(1), 12-35.

## See also

[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
for the joint-mode Laplace path (handles multi-RE and spatial
structure).

## Examples

``` r
set.seed(1)
n_g <- 20; n_per <- 5; n <- n_g * n_per
group <- rep(seq_len(n_g), each = n_per)
x <- rnorm(n)
X <- cbind(1, x)
u <- rnorm(n_g, 0, 0.8)
eta <- 0.3 + 0.7 * x + u[group]
y <- rbinom(n, 1, plogis(eta))

# Compare Laplace (n_quad = 1) and AGQ-7.
fit_lap <- agq_fit(y, X, group, family = "binomial", n_quad = 1)
fit_agq <- agq_fit(y, X, group, family = "binomial", n_quad = 7)
c(fit_lap$log_marginal, fit_agq$log_marginal)
#> [1] -58.97319 -58.96228
```
