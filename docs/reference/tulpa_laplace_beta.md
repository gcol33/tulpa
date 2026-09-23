# Fit a beta-regression model via Laplace, estimating the precision

Thin wrapper around
[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
for `family = "beta"`. The Laplace engine treats `phi` as fixed per fit
(same contract as gamma and neg_binomial_2); this wrapper does an outer
1-D optimisation of the Laplace-approximated log-marginal over `phi`,
then refits at the optimum to return betas and Hessian.

The mean-precision parameterisation is
`y ~ Beta(mu * phi, (1 - mu) * phi)` with default logit link; `y` must
be strictly in `(0, 1)`.

## Usage

``` r
tulpa_laplace_beta(
  y,
  X,
  re_list = list(),
  spatial = NULL,
  weights = NULL,
  offset = NULL,
  max_iter = 100L,
  tol = 1e-06,
  n_threads = 1L,
  beta_prior = NULL,
  phi_init = NULL,
  phi_bounds = c(0.1, 10000),
  outer_tol = 1e-04,
  mode = c("laplace", "nuts"),
  control = list()
)
```

## Arguments

- y:

  Response in `(0, 1)`.

- X:

  Fixed-effects design matrix.

- re_list, spatial, weights, offset, max_iter, tol, n_threads,
  beta_prior:

  Passed to
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  verbatim. `beta_prior` places a Gaussian penalty on the fixed effects
  (a list with `sd`, optional `mean`; see
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)).
  It is included in the Laplace log-marginal that the precision `phi` is
  optimised against, so the penalised model is fit consistently across
  the outer `phi` search. Not supported with `spatial` (the spatial
  solver carries its own prior).

- phi_init:

  Optional starting value for the precision. If `NULL`, a
  method-of-moments warm start is used.

- phi_bounds:

  Numeric length-2 vector with lower/upper bounds on `phi` for the outer
  optimisation. Default `c(0.1, 1e4)`.

- outer_tol:

  Tolerance for the outer optimisation. Default 1e-4.

- mode:

  Inference method (the method is an argument, not a parallel verb):
  `"laplace"` (default) is the Laplace + Brent-over-`phi` point fit
  documented here; `"nuts"` delegates to
  [`tulpa_nuts_beta()`](https://gillescolling.com/tulpa/reference/tulpa_nuts_beta.md),
  which samples `phi` jointly with the coefficients via NUTS. In
  `"nuts"` mode the Laplace-only arguments (`re_list`, `spatial`,
  `weights`, `offset`, `phi_init`, `phi_bounds`, `outer_tol`) are not
  used, and NUTS knobs are passed via `control` (see
  [`tulpa_nuts_beta()`](https://gillescolling.com/tulpa/reference/tulpa_nuts_beta.md)).

- control:

  Passed to
  [`tulpa_nuts_beta()`](https://gillescolling.com/tulpa/reference/tulpa_nuts_beta.md)
  when `mode = "nuts"` (ignored for `mode = "laplace"`).

## Value

For `mode = "laplace"`, the list returned by
[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
at the optimum, augmented with `phi` (the optimised precision) and
`phi_log_marginal` (the optimisation trace). For `mode = "nuts"`, the
draws object returned by
[`tulpa_nuts_beta()`](https://gillescolling.com/tulpa/reference/tulpa_nuts_beta.md).

## Examples

``` r
set.seed(1)
n <- 200L
X <- cbind(1, rnorm(n))
mu <- plogis(X %*% c(0.2, 0.7)); phi <- 8
y <- rbeta(n, mu * phi, (1 - mu) * phi)
fit <- tulpa_laplace_beta(y, X)
fit$mode
#> [1] 0.1156379 0.6452995
```
