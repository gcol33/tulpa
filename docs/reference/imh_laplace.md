# Independence Metropolis-Hastings with Laplace proposal

Sample from a posterior using independence Metropolis-Hastings with a
multivariate-t proposal centred at the Laplace mode, with the inverse
Hessian as its scale matrix. The t's polynomial tails dominate any
target whose tails are no heavier than its own, which is the condition
under which an independence sampler is uniformly ergodic; a Gaussian
proposal lighter-tailed than the target under-visits the target's tails
and the chain sticks there. The degrees of freedom are one package
setting. For posteriors that are well- approximated by a Gaussian near
the mode this is dramatically cheaper than HMC: each iteration is one
log-posterior evaluation plus one accept/reject step. Embarrassingly
parallel across chains.

Use cases:

- Quick draws when Laplace is "almost right" but you want
  asymptotically-correct samples.

- Cross-validation, simulation studies, or other repeated-fit workflows
  where HMC startup cost dominates.

- Sanity check on the Laplace approximation: low IMH acceptance means
  the posterior is far from Gaussian and Laplace is biased.

## Usage

``` r
imh_laplace(
  log_posterior,
  mode,
  hessian,
  n_iter = 2000L,
  warmup = n_iter%/%2L,
  scale = 1,
  init = NULL,
  thin = 1L,
  seed = NULL,
  verbose = FALSE
)
```

## Arguments

- log_posterior:

  Function `function(theta) -> numeric` returning the *unnormalized* log
  posterior at a single parameter vector.

- mode:

  Numeric vector of length `d`: the Laplace mode (i.e.,
  `tulpa_laplace(...)$mode[seq_len(d)]` for the fixed-effects block, or
  any other mode you trust).

- hessian:

  Symmetric positive-definite `d x d` matrix: the negative Hessian of
  `log_posterior` at `mode` (or `H_beta` from `tulpa_laplace`).

- n_iter:

  Total iterations including warmup (default 2000).

- warmup:

  Warmup iterations to discard (default `n_iter / 2`).

- scale:

  Optional inflation factor on the proposal scale matrix (default 1.0);
  the scale matrix is `scale^2 * hessian^{-1}`.

- init:

  Optional starting parameter vector (default = `mode`).

- thin:

  Keep every `thin`-th post-warmup sample (default 1).

- seed:

  Optional integer RNG seed. Scoped to this call: the caller's
  `.Random.seed` is restored on exit, so two calls with the same `seed`
  give identical draws regardless of the surrounding RNG stream.

- verbose:

  Print acceptance rate at end (default `FALSE`).

## Value

A list with class `tulpa_fit` carrying:

- `draws`: `(n_iter - warmup) / thin` x `d` matrix of draws.

- `means`: posterior means.

- `accept_prob`: per-iteration acceptance indicators.

- `mean_accept`: overall post-warmup acceptance rate.

- `log_prob`: per-draw log posterior values.

- `inference_mode`: `"exact"`.

- `inference_tier`: `1L`.

- `backend`: `"imh_laplace"`.

- `proposal`: `list(mode, cov, scale, df)`, the proposal's location,
  scale matrix, inflation and degrees of freedom.

## Tier

Tier 1 (Exact). The MH accept/reject step makes the chain asymptotically
correct for the supplied `log_posterior` under the standard MH
conditions. Posterior exactness requires that function to evaluate the
true density. Supplying a Laplace, EP or VI target leaves that target's
approximation error intact. The proposal affects efficiency.

## See also

[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
for the mode + Hessian,
[`bridge_sampling()`](https://gillescolling.com/tulpa/reference/bridge_sampling.md)
for marginal-likelihood estimation on the resulting draws.

## Examples

``` r
# Toy: Bernoulli logistic with one covariate.
set.seed(1)
n <- 100
x <- rnorm(n)
eta <- 0.3 + 1.2 * x
y <- rbinom(n, 1, plogis(eta))
X <- cbind(1, x)

lap <- tulpa_laplace(y, n_trials = rep(1L, n), X = X,
                     family = "binomial")

log_post <- function(beta) {
  eta <- as.numeric(X %*% beta)
  sum(y * eta - log1p(exp(eta))) +
    sum(dnorm(beta, 0, 10, log = TRUE))
}

fit <- imh_laplace(log_post, mode = lap$mode[1:2],
                   hessian = lap$H_beta, n_iter = 1000)
fit$mean_accept
colMeans(fit$draws)
```
