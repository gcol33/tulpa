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

## Usage

``` r
tulpa_em_mc(
  e_step_sample,
  m_step_encode,
  n_mc = 10L,
  n_mc_growth = 4/3,
  n_mc_max = 200L,
  max_iter = 30L,
  tol = 0.1,
  alpha = 0.25,
  n_consecutive = 3L,
  verbose = TRUE,
  correction = c("none", "gibbs"),
  n_gibbs = 2000L,
  log_complete = NULL,
  correction_control = list(),
  beta_prior = NULL,
  ...
)
```

## Arguments

- e_step_sample:

  Function `function(fits, n_mc, ...) -> list`. Must return a list of
  length `n_mc`, each element a weights object of the same shape
  `m_step_encode` consumes (typically a numeric vector of length
  `n_obs`, or a matrix `n_obs x K` for K-class latent variables). `fits`
  has the shape the `e_step` of
  [`tulpa_em_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_em_laplace.md)
  receives: a named list with one entry per submodel, each carrying
  `mode` (the mean of the per-draw mode vectors of the previous M-step),
  `beta` and `se` (the pooled fixed effects and their standard errors).
  On the first iteration `fits` is
  [`list()`](https://rdrr.io/r/base/list.html) – the callback should
  return draws from the prior or a starting guess.

- m_step_encode:

  Function `function(weights, ...) -> list of blocks`. Identical to the
  contract used by
  [`tulpa_em_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_em_laplace.md):
  each block is a list with required fields `y`, `X`, `family`, and
  optional `n_trials`, `offset`, `phi`, `re_list`, `spatial`, `weights`.
  See
  [`?tulpa_em_laplace`](https://gillescolling.com/tulpa/reference/tulpa_em_laplace.md)
  for the full spec. A submodel has to keep one latent layout (the
  length of its mode vector) across draws.

- n_mc:

  Initial number of Monte-Carlo draws per iteration (default `10L`).

- n_mc_growth:

  Factor by which `n_mc` grows after an iteration whose step is swamped
  by Monte Carlo error (default `4 / 3`, Booth and Hobert's `1 + 1/r`
  with `r` set to 3). `1` keeps `n_mc` constant.

- n_mc_max:

  Cap on `n_mc` (default `200L`).

- max_iter:

  Maximum EM iterations (default `30L`).

- tol:

  Convergence tolerance on the largest change between iterations in
  units of each coefficient's pooled standard error (default `0.1`).

- alpha:

  Level of the confidence ellipsoid that decides whether a step is
  swamped by Monte Carlo error (default `0.25`, Booth and Hobert's
  choice).

- n_consecutive:

  Number of consecutive iterations `delta < tol` has to hold for
  (default `3L`).

- verbose:

  Print per-iteration progress (default `TRUE`).

- correction:

  `"none"` (default) returns the Tier 2 Rubin estimator. `"gibbs"` adds
  true-density MH-within-Gibbs, returning paired parameter and latent
  draws from the joint posterior.

- n_gibbs:

  Total Gibbs sweeps including warmup (default `2000L`). Used when
  `correction = "gibbs"`.

- log_complete:

  Optional function `function(fits, z, ...) -> numeric` returning
  `log p(y, z, theta)`, with every complete-data likelihood normalizer,
  parameter prior and transformation Jacobian. Each block of `fits`
  carries the evaluated parameter vector in `mode` and its fixed effects
  in `beta`. The posterior evidence `p(y)` is unnecessary. `NULL` uses
  the engine density assembled from the complete-data GLMM blocks. The
  callback must be deterministic and return one finite value or `-Inf`.
  Used only with `correction = "gibbs"`.

- correction_control:

  List of Gibbs settings: `warmup` (default `n_gibbs %/% 2L`), `thin`
  (default `1L`), `mh_steps` (conditional MH transitions per sweep,
  default `1L`), and `scale` (t proposal scale inflation, default `1`).
  Unknown entries are errors.

- beta_prior:

  Optional Gaussian prior on the fixed effects, applied to every block
  fit via
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  (i.e. blocks without a `prior` field). `NULL` (default) takes the
  engine default `N(0, 2.5^2)`. Otherwise a list with `sd` (required)
  and optional `mean`; see
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md).
  The same prior flows into the MI / Gibbs correction refits, so
  penalized corrections come for free. A block may override the default
  by setting its own `beta_prior` field in `m_step_encode` (e.g.
  different priors for the occupancy and detection submodels). Use
  scalar `mean` / `sd` here when blocks differ in width; per-coefficient
  vectors belong on the block.

- ...:

  Forwarded to `e_step_sample` and `m_step_encode`.

## Value

A list with:

- `pooled` – named list of pooled per-submodel summaries (`mean`, `se`,
  `V_within`, `V_between`, `V_total`); see
  [`rubins_pool()`](https://gillescolling.com/tulpa/reference/rubins_pool.md).

- `fits` – the point estimate in the shape `e_step_sample` receives: per
  submodel, `mode`, `beta`, `se`.

- `draws` – with `correction = "none"`, per-MC-draw fit lists from the
  final iteration, indexed by submodel. With `"gibbs"`, a matrix of full
  parameter draws paired by row with `latent_draws`; `log_prob` stores
  the corresponding complete-data log density and `final_fits` the last
  sampled state in E-step form.

- `em_draws`, `pooled_target` – with `"gibbs"`, the original final MCEM
  refits and the Tier 2 target of the retained `pooled` summaries.

- `n_iter` – iterations actually run.

- `n_mc_final` – `n_mc` used in the last iteration.

- `converged` – logical.

- `inference_mode`, `inference_tier`, `target` – `"structured"`, `2L`,
  and `"Rubin-pooled complete-data Laplace estimates"` for `"none"`.
  Gibbs reports `"exact"`, `1L`, and
  `"joint posterior p(theta, Z | y), conditional on fixed block hyperparameters"`.

- `accept_prob`, `mean_accept`, `draws_kind` – with `"gibbs"`,
  conditional MH acceptance indicators on retained sweeps, their mean,
  and `"chain"`.

- `means`, `sds`, `vcov`, `se` – with `"gibbs"`, posterior means, SDs,
  full parameter covariance and per-block fixed-effect SDs from the
  chain.

- `history` – `data.frame(iter, n_mc, delta, mcse, swamped)`: `delta`
  the standardized step, `mcse` the largest Monte Carlo standard error
  as a fraction of the pooled standard error, `swamped` whether the step
  lay inside the Monte Carlo confidence ellipsoid.

## Convergence and the Monte Carlo sample size

An MCEM update carries Monte Carlo error, so a rule on the raw change
between iterations stalls at the noise floor of the draws however long
it runs, and a change relative to `|theta|` asks for unbounded precision
on a coefficient near zero. The rule here follows Booth and Hobert
(1999):

- **Step against noise.** The Monte Carlo standard error of the pooled
  update is `mcse_j = sqrt(V_between_j / n_mc)`. When the previous
  estimate lies inside the `1 - alpha` confidence ellipsoid of the
  update,
  `sum_j (theta_j - theta_prev_j)^2 / mcse_j^2 < qchisq(1 - alpha, P)`,
  the EM step is swamped by Monte Carlo error and the next iteration
  draws `n_mc * n_mc_growth` (capped at `n_mc_max`). A step clearly
  larger than the noise keeps the draw count.

- **Stopping on the posterior scale.** The change is measured against
  the pooled standard error `se_j` of each coefficient,
  `delta = max_j |theta_j - theta_prev_j| / se_j`, and the fit stops
  once `delta < tol` on `n_consecutive` consecutive iterations. A step
  of `tol` posterior SDs is small whatever the coefficient's magnitude.

The noise in `delta` is about `sqrt(2) mcse_j / se_j`, so `tol` is also
the precision the draws have to reach; `history$mcse` reports
`max_j mcse_j / se_j` per iteration. A run that ends with `n_mc` at
`n_mc_max` and `delta` still at that noise level warns: the cap, not the
EM, is what stopped it.

Each M-step fits every draw on its own and averages the per-draw
estimates. The fixed point is therefore the mean of the complete-data
estimators under `p(z | y, theta)`, which agrees with the EM fixed point
(the maximizer of the averaged complete-data log-likelihood) to first
order in `1/n`.

This is an engine block, not a front door: model packages call it
programmatically, so its tuning knobs (`max_iter`, `tol`, `n_mc`) sit in
the signature rather than in a `control` list.

## Tier

Tier 2 (structured): complete-data Laplace fits pooled over latent
draws. Increasing `n_mc` reduces Monte Carlo error in the E-step. The
fitted modes, Gaussian conditional approximations and Rubin pooling
remain the inferential approximation. An exact E-step sampler alone
supplies no posterior correction for those approximations.

With `correction = "gibbs"`, the Rubin fit initializes a
data-augmentation chain. Each sweep calls `e_step_sample(fits, 1L, ...)`
for one exact latent draw and applies true-density MH to the full
complete-data parameter vector with a Laplace t proposal.
`fits[[k]]$mode` and `$beta` then contain the current sampled parameter
values. The target is `p(theta, Z | y)`, conditional on fixed block
dispersion and random-effect covariance, and the result is Tier 1. See
[`tulpa_em_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_em_laplace.md)
for the density contract, supported blocks, parameter layout,
computational cost and convergence requirements.

## References

Wei, G. C. G., & Tanner, M. A. (1990). A Monte Carlo implementation of
the EM algorithm and the poor man's data augmentation algorithms.
*Journal of the American Statistical Association*, 85(411), 699-704.
[doi:10.1080/01621459.1990.10474930](https://doi.org/10.1080/01621459.1990.10474930)
.

Tanner, M. A., and Wong, W. H. (1987). The calculation of posterior
distributions by data augmentation. *JASA*, 82(398), 528-540.
[doi:10.1080/01621459.1987.10478458](https://doi.org/10.1080/01621459.1987.10478458)
.

Robert, C. P., and Casella, G. (1999). *Monte Carlo Statistical
Methods*. Springer, sections 7.3.2-7.3.3.
[doi:10.1007/978-1-4757-3071-5](https://doi.org/10.1007/978-1-4757-3071-5)
.

Andrieu, C., and Roberts, G. O. (2009). The pseudo-marginal approach for
efficient Monte Carlo computations. *Annals of Statistics*, 37(2),
697-725. [doi:10.1214/07-AOS574](https://doi.org/10.1214/07-AOS574) .

Booth, J. G., & Hobert, J. P. (1999). Maximizing generalized linear
mixed model likelihoods with an automated Monte Carlo EM algorithm.
*JRSS B*, 61(1), 265-285.

Caffo, B. S., Jank, W., & Jones, G. L. (2005). Ascent-based Monte Carlo
expectation-maximization. *JRSS B*, 67(2), 235-251.

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
    l0 <- dnorm(y, fits$comp0$mode[1], log = TRUE) + log(0.6)
    l1 <- dnorm(y, fits$comp1$mode[1], log = TRUE) + log(0.4)
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

fit <- tulpa_em_mc(e_step_sample, m_step_encode, n_mc = 5L, max_iter = 8L,
                   verbose = FALSE)
sapply(fit$pooled, function(p) p$mean)
#>      comp0      comp1 
#> -0.9868029  2.0396015 
fit$history
#>   iter n_mc      delta      mcse swamped
#> 1    1    5        Inf 0.2545194      NA
#> 2    2    5 3.72095902 0.2674963   FALSE
#> 3    3    5 0.57057074 0.1697500   FALSE
#> 4    4    5 0.23660514 0.2233103    TRUE
#> 5    5    7 0.07377956 0.1868036    TRUE
#> 6    6   10 0.28729681 0.1028114   FALSE
#> 7    7   10 0.21766243 0.1472689    TRUE
#> 8    8   14 0.22858400 0.1046445   FALSE
# }
```
