# Metropolis-Adjusted Langevin Algorithm (MALA)

Sample from a posterior using MALA: a Langevin proposal driven by the
gradient of the log posterior, corrected by a Metropolis- Hastings
accept/reject step. Each iteration is one log-posterior + one gradient
evaluation. The natural stepping stone between random-walk MH and HMC:
cheaper per iteration than HMC (no leapfrog integration), better-mixing
than RWMH because the drift term moves toward higher density.

Step size `epsilon` is adapted via dual averaging during warmup to
target an acceptance rate of 0.574 (Roberts & Rosenthal 1998 optimal for
high-dimensional Gaussians).

## Usage

``` r
mala(
  log_posterior,
  grad_log_posterior,
  init,
  n_iter = 2000L,
  warmup = n_iter%/%2L,
  epsilon = 0.1,
  target_accept = 0.574,
  mass_diag = NULL,
  thin = 1L,
  verbose = FALSE
)
```

## Arguments

- log_posterior:

  Function `function(theta) -> numeric` returning the *unnormalized* log
  posterior.

- grad_log_posterior:

  Function `function(theta) -> numeric vector` returning the gradient at
  `theta`.

- init:

  Numeric vector: initial state (must give finite log_posterior).

- n_iter:

  Total iterations including warmup (default 2000).

- warmup:

  Warmup iterations (step-size adaptation + discarded; default
  `n_iter / 2`).

- epsilon:

  Initial step size (default 0.1). Adapted during warmup.

- target_accept:

  Target acceptance during warmup adaptation (default 0.574, the Roberts
  & Rosenthal 1998 optimum).

- mass_diag:

  Optional preconditioner: a length-`d` vector of per-dimension
  **variances**, used as the diagonal inverse-mass `M^-1`. The proposal
  is `N(theta + (eps^2/2) * M^-1 * grad, eps^2 * M^-1)`, so entry `j`
  scales the proposal variance along dimension `j` and the noise
  standard deviation is `eps * sqrt(mass_diag[j])`. Default `rep(1, d)`.
  For posteriors with very different scales across dimensions, set this
  to (an estimate of) the posterior variances, i.e. the squared
  posterior SDs.

- thin:

  Keep every `thin`-th post-warmup sample (default 1).

- verbose:

  Print acceptance + step-size summary at end (default FALSE).

## Value

A list with class `tulpa_fit` carrying:

- `draws`: post-warmup draws.

- `means`, `n_params`, `n_samples`, `log_prob`.

- `accept_prob`: per-iteration accepted indicators (post-warmup).

- `mean_accept`: post-warmup acceptance rate.

- `epsilon`: final adapted step size.

- `inference_mode`: `"exact"`.

- `inference_tier`: `1L`.

- `backend`: `"mala"`.

## Tier

Tier 1 (Exact). The MH step makes the chain asymptotically correct.
Mixing depends on whether the gradient gives useful local geometry –
poor for posteriors with very different scales across dimensions (use a
preconditioner or HMC instead).

## References

Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of
Langevin distributions and their discrete approximations. *Bernoulli*,
2(4), 341-363.

Roberts, G. O., & Rosenthal, J. S. (1998). Optimal scaling of discrete
approximations to Langevin diffusions. *JRSS B*, 60(1), 255-268.

## Examples

``` r
log_post <- function(t) -0.5 * sum((t - c(1, 2))^2)
grad <- function(t) -(t - c(1, 2))
fit <- mala(log_post, grad, init = c(0, 0), n_iter = 1000)
colMeans(fit$draws)  # near c(1, 2)
#> [1] 1.086552 2.030870
fit$mean_accept      # should adapt toward 0.574
#> [1] 0.604
```
