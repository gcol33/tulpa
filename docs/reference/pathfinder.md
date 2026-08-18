# Pathfinder: variational warm-start via L-BFGS + ELBO scoring

Single-path Pathfinder (Zhang, Carpenter, Gelman, Vehtari 2022): run
L-BFGS toward the posterior mode, fit a Gaussian at the optimum using
the inverse-Hessian estimate, and report draws plus the ELBO. Cheap,
derivative-only, embarrassingly parallel – meant as an HMC warm-start,
an initialiser for
[`imh_laplace()`](https://gillescolling.com/tulpa/reference/imh_laplace.md),
or a quick sanity check on the Laplace approximation.

This implementation is **single-path only**. The full multi-path
Pathfinder (K parallel L-BFGS runs + mixture proposal + Pareto- smoothed
importance reweighting) is a follow-on. The single-path version is what
most users want as a Laplace-equivalent diagnostic.

## Usage

``` r
pathfinder(
  log_posterior,
  init,
  grad_log_posterior = NULL,
  n_draws = 1000L,
  max_iter = 100L,
  tol = 1e-06,
  verbose = FALSE
)
```

## Arguments

- log_posterior:

  Function `function(theta) -> numeric` returning the *unnormalized* log
  posterior at `theta`.

- init:

  Numeric vector: initial point for L-BFGS. Should be in the support of
  the posterior (finite log_posterior).

- grad_log_posterior:

  Optional function returning the gradient of `log_posterior` at
  `theta`. If `NULL`, gradients are computed numerically via
  [stats::optim](https://rdrr.io/r/stats/optim.html)'s built-in finite
  differences.

- n_draws:

  Number of draws from the fitted Gaussian (default 1000).

- max_iter:

  L-BFGS iteration cap (default 100).

- tol:

  Gradient-norm tolerance for L-BFGS convergence (default 1e-6).

- verbose:

  Print L-BFGS / ELBO summary at end (default FALSE).

## Value

A list with class `tulpa_fit` carrying:

- `draws`: `n_draws x d` matrix of Gaussian draws at the mode.

- `means`: posterior means (= mode for a Gaussian fit).

- `mode`: the L-BFGS mode.

- `cov`: the proposal covariance (`solve(-hessian)`).

- `elbo`: Monte-Carlo estimate of `E_q[log p - log q]`.

- `n_iter`: L-BFGS iterations.

- `converged`: logical.

- `inference_mode`: `"structured"`.

- `inference_tier`: `2L`.

- `backend`: `"pathfinder"`.

## Tier

Tier 2 (Structured). The output is a Gaussian approximation, not samples
from the exact posterior – same epistemic class as
[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md).
Pair with
[`imh_laplace()`](https://gillescolling.com/tulpa/reference/imh_laplace.md)
for an exact-tier upgrade.

## References

Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022). Pathfinder:
parallel quasi-Newton variational inference. *Journal of Machine
Learning Research*, 23(306), 1-49.

## See also

[`imh_laplace()`](https://gillescolling.com/tulpa/reference/imh_laplace.md)
for an exact-tier MH using the Pathfinder Gaussian as proposal;
[`bridge_sampling()`](https://gillescolling.com/tulpa/reference/bridge_sampling.md)
for marginal-likelihood estimation on the resulting draws.

## Examples

``` r
# Toy: 2-D conjugate normal.
y <- c(0.5, -0.7)
log_post <- function(t) {
  sum(dnorm(y, t, 1, log = TRUE)) +
    sum(dnorm(t, 0, sqrt(10), log = TRUE))
}
pf <- pathfinder(log_post, init = c(0, 0), n_draws = 2000)
pf$mode      # near c(0.45, -0.64)
#> [1]  0.4545455 -0.6363636
pf$elbo
#> [1] -4.269409
```
