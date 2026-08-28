# Gibbs estimation of random-effect covariances (exact-target debias)

For one or more random-effects terms (e.g. `(1 + x | g)`,
`(1 + x || g)`, or several terms together), estimate the random-effect
covariances `Sigma` by sampling the exact joint posterior
`p(beta, {b}, {Sigma} | y)` rather than fixing each `Sigma` at the
Laplace mode. This removes the Laplace / PQL "approximation" bias that
shrinks variance components low for binary and low-count responses with
small groups.

## Usage

``` r
tulpa_re_cov_gibbs(
  y,
  n_trials = NULL,
  X,
  re_terms,
  family = "binomial",
  phi = 1,
  prior_df = NULL,
  prior_scale = NULL,
  beta_prior = .tulpa_default_beta_prior("re_cov_gibbs"),
  control = list()
)
```

## Arguments

- y, n_trials, X, family, phi:

  Passed to the likelihood and to
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  for the pilot solve. `n_trials = NULL` defaults to 1.

- re_terms:

  Either a single random-effect term or a list of them. Each term is a
  list with `idx` (1-based group index per observation), `n_groups`,
  `n_coefs` (`c`), `Z` (the `n_obs x c` RE design; only required when
  `c > 1`), and `correlated` (`TRUE` for a full `Sigma`, `FALSE` for a
  diagonal one; defaults to `TRUE`). An optional `label` / `group_var`
  names the block. Any supplied `L` / `cov` / `sigma` is ignored –
  `Sigma` is what this function samples.

- prior_df:

  Inverse-Wishart prior degrees of freedom. Applied to every correlated
  block (default `n_coefs + 1`, the minimal proper choice) and as the
  scalar inverse-gamma shape for every diagonal block (default 2). Must
  leave each block's prior proper – unlike
  [`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
  /
  [`tulpa_eb()`](https://gillescolling.com/tulpa/reference/tulpa_eb.md)
  (`hyperprior = "flat"` by default), the `Sigma_m | b_m` conjugate draw
  here needs a proper Inverse-Wishart to sample from, so an improper
  flat prior is not an option; the minimal-`df` default is the closest
  analogue this sampler can offer.

- prior_scale:

  Inverse-Wishart prior scale matrix. Used for a block when its
  dimension matches (default `diag(n_coefs)`); otherwise the per-block
  default is used.

- beta_prior:

  Gaussian fixed-effect prior as `list(mean, sd)` (default the engine
  default, `prior_normal(0, 2.5)`). Scalar `mean` / `sd` are recycled to
  `ncol(X)`; a length-`ncol(X)` vector sets a per-coefficient prior.

- control:

  A named list of numerical / tuning knobs (statistical arguments stay
  in the signature above). Recognized entries:

  - `n_iter`: recorded post-warmup sweeps (default 2000).

  - `warmup`: warmup (burn-in) sweeps, used for proposal-scale
    adaptation (default 1000).

  - `thin`: keep every `thin`-th recorded sweep (default 1).

  - `seed`: optional integer seed for reproducibility.

  - `max_iter`, `tol`, `n_threads`: pilot-solve controls (see
    [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)).

## Value

A list with:

- `posterior`: data frame with one row per parameter (`sigma_i`,
  `rho_ij`, `Sigma_ij`; prefixed by block label when there are several
  terms; diagonal blocks report no `rho`) and columns `mean`, `sd`,
  `median`, `ci_lo`, `ci_hi`.

- `Sigma_mean`: the posterior mean of `Sigma` (a matrix for one block, a
  named list of matrices for several).

- `Sigma_draws`: list of recorded draws; each element is the per-block
  list of `Sigma` matrices for that sweep.

- `beta_draws` / `draws`: matrix of recorded `beta` draws; `draws` (with
  `means`, `param_names`, `process_info`) drives the generic `tulpa_fit`
  methods.

- `re`: matrix of recorded random-effect draws, `n_kept` rows by one
  column per (block, group, coefficient) in that order – the exact
  per-group posterior the sweep samples. Row-aligned with `beta_draws`,
  so a draw is a joint `(beta, b)` state. This is what
  [`ranef()`](https://gillescolling.com/tulpa/reference/ranef.md)
  summarizes and what
  [`posterior_predict()`](https://gillescolling.com/tulpa/reference/posterior_predict.md)
  adds to the linear predictor.

- `accept`: list with `beta` and `b` acceptance rates over recorded
  sweeps.

- `n_kept`, `n_coefs` (vector of per-block `c`), `prior`: bookkeeping.

## Details

Metropolis-within-Gibbs targeting `p(beta, {b_m}, {Sigma_m} | y)`:

- **`b_{m,g} | beta, Sigma, y`** – random-walk Metropolis per (term,
  group) (groups are conditionally independent given `beta` and the
  covariances). The proposal *shape* is the Laplace per-group posterior
  covariance block (`return_re_cov` from
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md));
  the Metropolis acceptance is computed against the exact group
  log-likelihood plus the Gaussian RE log-prior, which corrects the
  non-Gaussianity the Laplace approximation misses. The linear predictor
  holds every other term's contribution fixed.

- **`beta | b, Sigma, y`** – random-walk Metropolis, proposal shape from
  the Laplace fixed-effect Hessian block `H_beta`.

- **`Sigma_m | b_m`** – exact conjugate draw. A correlated block draws
  the full matrix from
  `IW(prior_df + G_m, prior_scale + sum_g b_{m,g} b_{m,g}')`; an
  uncorrelated (diagonal) block draws each variance from its scalar
  conjugate (inverse-gamma). No linearization enters this step.

A single Laplace solve provides the starting values (`beta`, `b`) and
the proposal shapes; the random-walk scales for the `beta` block and the
per-term `b` blocks are adapted toward their target acceptance during
burn-in (Robbins-Monro), then held fixed for the recorded sweeps. The
covariance summary marginalizes the derived scale / correlation
parameters over the posterior draws via the same machinery as
[`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md).

For `family = "gaussian"` the response is already conditionally
Gaussian, so there is no Laplace bias to remove; `phi` (the residual
variance) is treated as known. The sampler still runs and is useful as a
reference / for `Sigma` uncertainty.

## References

Lewandowski, Kurowicka & Joe (2009). Generating random correlation
matrices based on vines and extended onion method. *Journal of
Multivariate Analysis* 100(9):1989-2001.

## See also

[`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
for the grid-integration (summary-bias) fix;
[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
for the pilot solve and per-group covariance blocks.

## Examples

``` r
# \donttest{
set.seed(1)
G <- 20L; per <- 12L; n <- G * per
grp <- rep(seq_len(G), each = per); x <- rnorm(n)
b <- cbind(rnorm(G, 0, 0.7), rnorm(G, 0, 0.5))     # random intercept + slope
eta <- -0.2 + 0.5 * x + b[grp, 1] + b[grp, 2] * x
y <- rbinom(n, 1L, plogis(eta))
re_term <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = cbind(1, x),
                correlated = TRUE)
fit <- tulpa_re_cov_gibbs(y, rep(1L, n), cbind(1, x), re_term,
                          family = "binomial",
                          control = list(n_iter = 300L, warmup = 150L))
fit$Sigma_mean        # exact-debias RE covariance posterior mean
#>             [,1]        [,2]
#> [1,]  0.48897898 -0.08635451
#> [2,] -0.08635451  0.42945640
# }
```
