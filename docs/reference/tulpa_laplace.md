# Fit a model via Laplace approximation

General-purpose Laplace approximation for latent Gaussian models. Finds
the mode of the latent field (beta + random effects) and returns the
Laplace-approximated marginal likelihood.

This is the public API for model packages (tulpaGlmm, tulpaObs, etc.) to
call tulpa's Laplace engine. As the low-level engine entry point (not a
front-door fitter), it keeps its numerical controls (`max_iter`, `tol`,
`n_threads`, `return_hessian`) inline in the signature rather than in a
`control` list, so callers assembling many solves pass them
positionally.

## Usage

``` r
tulpa_laplace(
  y,
  n_trials,
  X,
  re_list = list(),
  family = "binomial",
  phi = 1,
  phi2 = NULL,
  spatial = NULL,
  weights = NULL,
  offset = NULL,
  max_iter = 100L,
  tol = 1e-06,
  n_threads = 1L,
  return_hessian = TRUE,
  beta_prior = NULL,
  return_re_cov = FALSE,
  X_zi = NULL,
  zi_prior_sd = 2.5,
  return_joint_hessian = FALSE,
  compute_skew = FALSE,
  skew_idx = NULL,
  debias = NULL
)
```

## Arguments

- y:

  Response vector (integer for binomial/poisson/negbin, numeric for
  gaussian)

- n_trials:

  Trial sizes (integer vector, used for binomial only)

- X:

  Fixed-effects design matrix

- re_list:

  List of RE specifications. Each element is a list with:

  - `idx`: integer vector of group indices (1-based)

  - `n_groups`: number of groups

  - `n_coefs`: coefficients per group (1 = intercept-only, \>1 = random
    slopes)

  - `sigma`: per-coefficient RE standard deviation(s), a diagonal
    covariance (uncorrelated, lme4 `(x || g)`). A scalar is recycled to
    `n_coefs`.

  - `Z`: slope design matrix (n_obs x n_coefs) when `n_coefs > 1`;
    `NULL` means intercept-only.

  - `L` / `cov`: optional `n_coefs x n_coefs` covariance for a
    *correlated* term (lme4 `(1 + x | g)`) – supply either a
    lower-triangular Cholesky factor `L` (covariance = `L L'`) or the
    covariance matrix `cov`. When present these take precedence over
    `sigma`; the off-diagonal enters both the joint Hessian (mode
    finding) and the marginal fixed-effect SE.

- family:

  Character: `"binomial"`, `"poisson"`, `"neg_binomial_2"`, `"gaussian"`

- phi:

  Dispersion parameter. For `gaussian` / `lognormal` this is the
  residual VARIANCE (matching the R-side family registry and
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)); the
  SD-parameterized compiled kernels receive `sqrt(phi)` internally. For
  `neg_binomial_2` the size, `beta` the precision, `t` the scale.

- phi2:

  Optional second dispersion: the Student-t degrees of freedom
  (`family = "t"`; default 4 when `NULL`). Non-spatial path only.

- spatial:

  Optional spatial specification (tulpa_spatial object)

- weights:

  Optional observation weights (numeric vector, length `length(y)`).
  Scales each observation's log-density, score and Fisher curvature by
  the same `w_i`, on the spatial route as well as the non-spatial one,
  so the mode and the marginal precision `H_beta` describe one weighted
  model. `NULL` (default) uses 1.

- offset:

  Optional observation-level offset on the linear predictor (numeric
  vector, length `length(y)`). `NULL` (default) uses 0.

- max_iter:

  Maximum Newton iterations (default 100)

- tol:

  Convergence tolerance (default 1e-6)

- n_threads:

  Number of threads (default 1)

- return_hessian:

  Logical: return the fixed-effect Hessian block? (default TRUE)

- beta_prior:

  Optional Gaussian prior on the fixed effects. `NULL` (default) keeps
  the weak built-in prior `beta ~ N(0, 100^2)`. Otherwise a list with
  element `sd` (prior standard deviation, required) and optional `mean`
  (prior mean, default 0). Each may be a scalar (applied to every
  coefficient) or a length-`ncol(X)` vector. Adds
  `sum((beta - mean)^2 / (2 * sd^2))` to the negative log-posterior, so
  the mode is the penalized (MAP) estimate. A coefficient's `sd` may be
  `+Inf`, which sets its precision to 0 (no penalty on that
  coefficient). Not supported on the spatial path.

- return_re_cov:

  If `TRUE`, additionally return per-group marginal posterior covariance
  blocks `Cov(u_g | y, Sigma)` – one `n_coefs x n_coefs` matrix per (RE
  term, group), with the fixed effects and other groups marginalized out
  (each block is a diagonal block of the full inverse Hessian, not the
  inverse of a diagonal block). Used by the EM M-step for a full
  random-effect covariance. Non-spatial multi-RE path only.

- X_zi:

  Optional zero-inflation design matrix (`length(y)` rows). When
  supplied the latent fixed-effect block becomes
  `[beta_count | beta_zi]` and the family's compiled zero-inflated
  kernel is used. Non-spatial path only.

- zi_prior_sd:

  Prior SD on the zero-inflation coefficients,
  `beta_zi ~ N(0, zi_prior_sd^2)` (default 2.5, matching the samplers'
  `ModelData::zi_prior_sd`). It is what keeps the logit identified when
  a level contributes no zeros, where the likelihood alone drives
  `beta_zi` to `-Inf`. `+Inf` removes the penalty. Ignored when `X_zi`
  is `NULL`.

- return_joint_hessian:

  If `TRUE`, additionally return `H_joint`: the full joint posterior
  precision of the latent field `[beta | random effects]` at the mode,
  as a symmetric sparse matrix. This is the matrix the Laplace
  approximation takes the determinant of, so it is what an exact
  derivative of the log-marginal has to differentiate through; `H_beta`
  is only its fixed-effect Schur complement. Costs one extra copy of the
  Hessian, so it is off by default. Non-spatial multi-RE path only.

- compute_skew:

  If `TRUE`, additionally return the inner-Laplace reliability material
  at `skew_idx`: `inner_skew` (gamma_3, Rue Martino & Chopin 2009's
  cubic term), and the importance curve `inner_is_z` /
  `inner_is_log_joint` the inner Pareto-k-hat is fitted from. Costs one
  linear solve plus a fixed batch of objective evaluations per probed
  index. Non-spatial path only.

- skew_idx:

  1-based latent indices to probe (in the `[beta | random effects]`
  layout of `mode`). `NULL` with `compute_skew = TRUE` probes every
  latent index.

- debias:

  Subspace debias (gcol33/tulpa#304): a list with `idx` (1-based latent
  indices to correct by Metropolis along the Gaussian-conditional-mean
  surface through the mode) and optional `n_iter` / `warmup` / `thin`.
  The result then carries `debias_draws` (`n_kept x length(idx)`, the
  sampled `x_S - mode_S`), `debias_sigma_ss` (the inner Laplace's own
  marginal covariance of `x_S`), `debias_accept` and `debias_idx`.
  `NULL` (default) or an empty `idx` leaves the solve bit-for-bit as it
  was and consumes no random number. Non-spatial path only.

## Value

A list with:

- `mode`: full mode vector (beta, then RE values per term)

- `log_marginal`: Laplace-approximated log-marginal likelihood

- `n_iter`: number of Newton iterations

- `converged`: logical, whether the stopping rule was met

- `score_max`: largest absolute component of the joint penalized score
  at the returned mode – the residual the solve actually achieved, which
  is a different question from `converged` and the one anything
  differentiating through the mode depends on

- `log_det_Q`: log-determinant of the Hessian

- `H_beta`: fixed-effect block of the Hessian (if return_hessian = TRUE)

- `cov_blocks`: list of per-group posterior covariance matrices, one per
  (RE term, group) in term-major then group order (if return_re_cov =
  TRUE)

## Examples

``` r
set.seed(1)
n <- 200L
X <- cbind(1, rnorm(n))
eta <- X %*% c(-0.3, 0.8)
y <- rbinom(n, 1, plogis(eta))
fit <- tulpa_laplace(y, rep(1L, n), X, family = "binomial")
fit$mode          # posterior mode of the fixed effects
#> [1] -0.4406277  0.4822831
```
