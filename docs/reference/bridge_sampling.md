# Bridge sampling for marginal likelihood

Estimate the log marginal likelihood \\\log Z = \log p(y)\\ from
posterior draws via the Meng-Wong / Gronau bridge-sampling identity.
Useful for Bayes factors, model comparison, and as a sanity check on
Laplace-approximated marginal likelihoods.

The classical iterative scheme (Meng & Wong 1996; Gronau et al. 2017) is
used with a multivariate-normal proposal fit to half of the posterior
draws – the other half is used for the bridge ratio so the proposal is
independent of the samples that score it. All computations are done in
log-space via `logsumexp` for numerical stability.

## Usage

``` r
bridge_sampling(
  draws,
  log_posterior,
  n_proposal = nrow(draws),
  max_iter = 1000L,
  tol = 1e-10,
  split = TRUE,
  verbose = FALSE
)
```

## Arguments

- draws:

  Numeric matrix of posterior draws, one parameter per column. Typically
  `fit$draws` from a tulpa Tier-1 fit.

- log_posterior:

  Function `function(theta) -> numeric` returning the *unnormalized* log
  posterior density `log p(theta, y)` at a single parameter vector. Must
  accept a numeric vector of length `ncol(draws)` and return a finite
  scalar.

- n_proposal:

  Number of proposal draws. Default `nrow(draws)`.

- max_iter:

  Maximum bridge iterations (default 1000).

- tol:

  Convergence tolerance on `|log r_{t+1} - log r_t|` (default 1e-10).

- split:

  Logical: split the posterior draws in half so the proposal is fit on
  one half and scored on the other (default `TRUE`, recommended). Set
  `FALSE` only for diagnostic comparison.

- verbose:

  Print iteration history (default `FALSE`).

## Value

A list with:

- `log_marginal`: the bridge-sampling estimate of \\\log Z\\.

- `n_iter`: bridge iterations to convergence.

- `converged`: logical.

- `re_sd`: relative MSE estimate (Fruehwirth-Schnatter 2004); a rough
  quality check, smaller is better.

- `proposal`: the fitted proposal `list(mean, cov)`.

## Tier

Bridge sampling is a *post-hoc* marginal-likelihood estimator, not a
sampling backend, so it does not appear in the
[INFERENCE_TIERS](https://gillescolling.com/tulpa/reference/INFERENCE_TIERS.md)
registry. It operates on draws from any Tier-1 (Exact) backend.

## References

Meng, X.-L., & Wong, W. H. (1996). Simulating ratios of normalizing
constants via a simple identity: a theoretical exploration. *Statistica
Sinica*, 6, 831-860.

Gronau, Q. F., Sarafoglou, A., Matzke, D., Ly, A., Boehm, U., Marsman,
M., ... & Steingroever, H. (2017). A tutorial on bridge sampling.
*Journal of Mathematical Psychology*, 81, 80-97.

## Examples

``` r
# Toy: marginal likelihood of N(theta | 0, 1) under Y = theta + eps,
# eps ~ N(0, 1), prior theta ~ N(0, 10). Closed-form available.
y <- 1.5
log_post <- function(theta) {
  dnorm(y, theta, 1, log = TRUE) + dnorm(theta, 0, 10, log = TRUE)
}
draws <- matrix(rnorm(2000, mean = y * 100 / 101, sd = sqrt(100 / 101)),
                ncol = 1)
bs <- bridge_sampling(draws, log_post)
bs$log_marginal  # should match dnorm(y, 0, sqrt(101), log = TRUE)
#> [1] -3.238743
```
