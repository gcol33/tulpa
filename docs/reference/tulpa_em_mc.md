# Generic Monte-Carlo EM driver

Same M-step plumbing as
[`tulpa_em_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_em_laplace.md):
every iteration calls `m_step_encode(weights, ...)` to assemble
per-submodel blocks and fits each block via
[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md).
The difference is the E-step: instead of computing closed-form weights,
an `e_step_sample` callback returns `n_mc` weight draws per iteration.
Each draw is run through the M-step independently and the resulting
parameter estimates are pooled via
[`rubins_pool()`](https://gillescolling.com/tulpa/reference/rubins_pool.md).

Convergence criterion is the max relative change in *pooled* M-step
parameter estimates between iterations. To increase Monte-Carlo accuracy
as iterations progress (Booth-Hobert ascent-based MCEM), set
`n_mc_growth > 1`.

This is an engine block, not a front door: model packages call it
programmatically, so its tuning knobs (`max_iter`, `tol`, `n_mc`) sit in
the signature rather than in a `control` list.

## Usage

``` r
tulpa_em_mc(
  e_step_sample,
  m_step_encode,
  n_mc = 10L,
  n_mc_growth = 1,
  n_mc_max = 200L,
  max_iter = 30L,
  tol = 0.001,
  verbose = TRUE,
  ...
)
```

## Arguments

- e_step_sample:

  Function `function(fits, n_mc, ...) -> list`. Must return a list of
  length `n_mc`, each element a weights object of the same shape
  `m_step_encode` consumes (typically a numeric vector of length
  `n_obs`, or a matrix `n_obs x K` for K-class latent variables). On the
  first iteration `fits` is [`list()`](https://rdrr.io/r/base/list.html)
  – the callback should return draws from the prior.

- m_step_encode:

  Function `function(weights, ...) -> list of blocks`. Identical to the
  contract used by
  [`tulpa_em_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_em_laplace.md):
  each block is a list with required fields `y`, `X`, `family`, and
  optional `n_trials`, `offset`, `phi`, `re_list`, `spatial`, `weights`.
  See
  [`?tulpa_em_laplace`](https://gillescolling.com/tulpa/reference/tulpa_em_laplace.md)
  for the full spec.

- n_mc:

  Initial number of Monte-Carlo draws per iteration (default `10L`).

- n_mc_growth:

  Multiplicative growth of `n_mc` per iteration (default `1.0` =
  constant). Set `> 1` for ascent-based MCEM that ramps up MC accuracy
  near convergence.

- n_mc_max:

  Cap on `n_mc` (default `200L`).

- max_iter:

  Maximum EM iterations (default `30L`).

- tol:

  Convergence tolerance on max relative change in pooled parameter
  estimates (default `1e-3`). Looser than `tulpa_em_laplace` default
  because Monte-Carlo noise floors the achievable precision.

- verbose:

  Print per-iteration progress (default `TRUE`).

- ...:

  Forwarded to `e_step_sample` and `m_step_encode`.

## Value

A list with:

- `pooled` – named list of pooled per-submodel summaries (`mean`, `se`,
  `V_within`, `V_between`, `V_total`); see
  [`rubins_pool()`](https://gillescolling.com/tulpa/reference/rubins_pool.md).

- `fits` – list of per-MC-draw fit lists from the *final* iteration,
  each indexed by submodel name.

- `n_iter` – iterations actually run.

- `n_mc_final` – `n_mc` value used in the last iteration.

- `converged` – logical.

- `history` – `data.frame(iter, n_mc, delta)`.

## Tier

Inherits the tier of the inner M-step (Laplace =\> Tier 2). The
Monte-Carlo E-step *itself* is exact in the limit `n_mc -> infinity`, so
as a *full pipeline* MCEM is asymptotically Tier 1 if `e_step_sample` is
exact.

## References

Wei, G. C. G., & Tanner, M. A. (1990). A Monte Carlo implementation of
the EM algorithm and the poor man's data augmentation algorithms.
*Journal of the American Statistical Association*, 85(411), 699-704.

Booth, J. G., & Hobert, J. P. (1999). Maximizing generalized linear
mixed model likelihoods with an automated Monte Carlo EM algorithm.
*JRSS B*, 61(1), 265-285.

## See also

[`tulpa_em_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_em_laplace.md)
for the closed-form-weights variant,
[`rubins_pool()`](https://gillescolling.com/tulpa/reference/rubins_pool.md)
for the pooling rule.

## Examples

``` r
# \donttest{
# Two-component Gaussian mixture with unit variances: the latent class z is
# sampled in the E-step, and each class mean is an intercept-only Gaussian
# block in the M-step.
set.seed(1)
z <- rbinom(100, 1, 0.4)
y <- ifelse(z == 0, rnorm(100, -1), rnorm(100, 2))

e_step_sample <- function(fits, n_mc, ...) {
  p1 <- if (length(fits) == 0L) {
    ifelse(y > median(y), 0.8, 0.2)
  } else {
    mu0 <- fits[[1]]$comp0$mode[1]
    mu1 <- fits[[1]]$comp1$mode[1]
    l0 <- dnorm(y, mu0, log = TRUE) + log(0.6)
    l1 <- dnorm(y, mu1, log = TRUE) + log(0.4)
    1 / (1 + exp(l0 - l1))
  }
  lapply(seq_len(n_mc), function(s) rbinom(length(y), 1, p1))
}
m_step_encode <- function(weights, ...) {
  list(comp0 = list(y = y[weights == 0], X = matrix(1, sum(weights == 0), 1),
                    family = "gaussian"),
       comp1 = list(y = y[weights == 1], X = matrix(1, sum(weights == 1), 1),
                    family = "gaussian"))
}

fit <- tulpa_em_mc(e_step_sample, m_step_encode, n_mc = 5L, max_iter = 5L,
                   tol = 5e-3, verbose = FALSE)
sapply(fit$pooled, function(p) p$mean)
#>     comp0     comp1 
#> -1.062815  1.984884 
fit$history
#>   iter n_mc       delta
#> 1    1    5         Inf
#> 2    2    5 0.633027460
#> 3    3    5 0.201654017
#> 4    4    5 0.003172351
# }
```
