# Fit a latent-variable model via EM + Laplace approximation

Generic EM engine. Model packages provide callbacks for the E-step
(latent variable posterior) and the M-step encoding (how to assemble
submodel data from weights). Each M-step submodel block is fit
independently via
[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md);
the engine reads `family` and `offset` per block so heterogeneous-family
mixtures (e.g. a binomial zero submodel + poisson positive submodel for
a hurdle model) work without engine changes.

## Usage

``` r
tulpa_em_laplace(
  e_step,
  m_step_encode,
  spatial = NULL,
  re_list = list(),
  max_iter = 50L,
  tol = 1e-04,
  damping = 0.3,
  correction = c("auto", "mi", "gibbs", "none"),
  n_imputations = 20L,
  n_gibbs = 2000L,
  n_louis = 1000L,
  draw_z = NULL,
  m_step_extra = NULL,
  beta_prior = NULL,
  verbose = TRUE,
  log_complete = NULL,
  correction_control = list(),
  ...
)
```

## Arguments

- e_step:

  Callback: `function(fits, ...) -> list(weights = numeric, ...)`.
  Called once per EM iteration with the current per-submodel
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  fits. Must return a list whose `weights` element is the
  latent-variable posterior used by the next M-step.

- m_step_encode:

  Callback: `function(weights, ...) -> list of blocks`. Each block is
  itself a list with the following fields:

  - `y` (numeric, required) – response.

  - `X` (matrix, required) – fixed-effects design with
    `nrow(X) == length(y)`.

  - `family` (character scalar, required) – one of `"binomial"`,
    `"poisson"`, `"gaussian"`, `"negbin"` / `"neg_binomial_2"`,
    `"gamma"`, `"beta"`. Forwarded to
    [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
    for that block.

  - `n_trials` (numeric or `NULL`, optional; absent or `NULL` defaults
    to 1) – binomial trial counts.

  - `offset` (numeric or `NULL`, optional) – observation-level offset,
    length-matched to `y` when non-`NULL`.

  - `phi` (numeric scalar, optional) – dispersion forwarded to
    [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
    (used by `negbin`, `gamma`).

  - `weights` (numeric, optional; length `length(y)`) – per-observation
    likelihood weight, scaling that row's log-density, score and Fisher
    information. This is the channel a soft (fractional) latent label
    travels on; see the section below.

  - `re_list`, `spatial` (optional) – forwarded as-is.

  - `control` (list, optional) – on a block with a `prior`, entries
    merged over the driver's own
    [`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
    control (`n_threads`, `keep_grid_hessians = TRUE`,
    `diagnose_k = FALSE`); an M-step fit is one iterate of the EM
    objective, so its outer Pareto-k diagnostic is off unless the block
    sets `diagnose_k = TRUE`.

- spatial, re_list:

  Not consumed at the driver level – latent structure is per-submodel,
  set as a `spatial` / `re_list` field on each block `m_step_encode()`
  returns. Supplying either here is an error (rather than a silent
  no-op).

- max_iter:

  Maximum EM iterations.

- tol:

  Convergence tolerance on max relative parameter change.

- damping:

  EM damping factor in `[0, 1)`. With `damping = d`, the E-step weights
  are smoothed between iterations as `(1 - d) * new + d * prev` (`d = 0`
  is no damping); the M-step then refits on the smoothed weights, so the
  parameter update is damped indirectly through the weights rather than
  by mixing successive parameter vectors.

- correction:

  Post-EM correction. `"none"` returns the EM point estimate only.
  `"mi"` draws `n_imputations` independent hard `z` from the converged
  posterior weights P(z\|y, theta_hat), refits each block on the hard z,
  and pools via
  [`rubins_pool()`](https://gillescolling.com/tulpa/reference/rubins_pool.md).
  `"gibbs"` runs true-density MH-within-Gibbs for `n_gibbs` sweeps
  starting from the EM fits, returning joint posterior draws. `"auto"`
  resolves to `"none"`.

- n_imputations:

  Number of MI draws (default `20L`). Used when `correction = "mi"`.

- n_gibbs:

  Total Gibbs sweeps including warmup (default `2000L`). Used when
  `correction = "gibbs"`.

- n_louis:

  Number of hard latent draws for the Monte Carlo estimate of the
  missing information in Louis' observed-information standard errors
  (default `1000L`; `0` skips them). Each draw costs one
  `m_step_encode()` call and a score evaluation, no refit. See the
  section below.

- draw_z:

  Optional function `function(weights) -> hard_z` that turns the
  E-step's continuous weights into a hard latent draw. Used by
  `correction %in% c("mi", "gibbs")` and by the Louis standard errors
  (`n_louis`). The default treats `weights` as a numeric vector of
  Bernoulli probabilities and draws per-observation. Multi-class /
  matrix-valued latent structures must supply their own callback.

- m_step_extra:

  Optional `function(fits, weights, ...) -> fits`. Fired once per M-step
  in the EM iterations and MI draws. Receives the freshly assembled list
  of
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  results (`fits`), the continuous E-step weights P(z\|y, theta) (NOT
  the hard z draw used by MI to encode the block), and any extra
  arguments forwarded through `...`. Returns a list with the same length
  and names as the input, possibly with mutated dispersion / shape /
  precision fields (e.g. `fits[[k]]$phi`). Use this to update non-eta
  parameters that fall out of the Laplace M-step (NB overdispersion,
  Gamma shape, Beta precision, Gaussian sigma). Gibbs correction
  requires `NULL` and fixed block hyperparameters.

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

- verbose:

  Print per-iteration progress.

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

- ...:

  Forwarded to `e_step`, `m_step_encode`, and `m_step_extra`.

## Value

A list with:

- `fits` – named list of
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  results, one per block, each with `beta` and its complete-data `se`
  attached.

- `vcov`, `se`, `louis` – for EM and MI, observed-information covariance
  and standard errors by Louis' method. For Gibbs, `vcov` is the full
  sampled parameter covariance and `se` the per-block fixed-effect
  posterior SDs; the EM information is in `em_information`.

- `weights` – final E-step weights.

- `n_iter` – number of EM iterations actually run.

- `converged` – logical.

- `history` – `data.frame(iter, delta)` of max relative parameter change
  per iteration.

- `correction` – the resolved correction mode (`"none"`, `"mi"`, or
  `"gibbs"`).

- `pooled` – present when `correction = "mi"`. Named list of pooled
  per-submodel summaries from
  [`rubins_pool()`](https://gillescolling.com/tulpa/reference/rubins_pool.md).

- `draws` – for MI, per-draw fit lists with `beta` / `se`. For Gibbs, a
  matrix of retained full parameter draws; column names identify the
  block, coefficient and random-effect group.

- `latent_draws`, `log_prob`, `final_fits` – for Gibbs, latent states
  paired by row with `draws`, their complete-data log densities, and the
  final sampled parameters in E-step form. `fits` retains the EM
  initialization; `converged` describes that EM loop.

- `means`, `sds` – Gibbs posterior means and SDs over retained draws.

- `accept_prob`, `mean_accept`, `draws_kind` – for Gibbs, acceptance
  indicators for the conditional MH steps on retained sweeps, their
  mean, and `"chain"`.

- `inference_mode`, `inference_tier`, `target` – the actual inference
  class and target. Gibbs reports `"exact"`, `1L`, and
  `"joint posterior p(theta, Z | y), conditional on fixed block hyperparameters"`.

## Details

This is an engine block, not a front door: model packages call it
programmatically, so its tuning knobs (`max_iter`, `tol`, `damping`) sit
in the signature rather than in a `control` list.

## Soft latent labels go in `weights`, not in `y`

The M-step maximizes the expected complete-data log-likelihood. For a
Bernoulli latent `z_i` carrying E-step posterior weight `w_i` that is
\$\$Q = \sum_i \[\\ w_i \log p_i + (1 - w_i) \log (1 - p_i) \\\],\$\$ a
**weighted Bernoulli** log-likelihood, which carries no binomial
coefficient. Encode it as two rows per unit – `y = 1` at weight `w_i`
and `y = 0` at weight `1 - w_i`:

    list(y       = rep(c(1, 0), each = n),
         X       = rbind(X, X),
         weights = c(w, 1 - w),
         family  = "binomial")

A fractional `y` on a binomial block is refused, and is not the same
objective: it asks for the exact binomial density at a non-integer
response, whose normalizer `lchoose(n, y)` is not zero there and depends
on `w`. On the weighted encoding the block's `log_marginal` is the
Laplace marginal of `Q`, an EM objective that increases across
iterations.

## Standard errors

Each entry of `fits` carries `beta` (the fixed effects) and `se`, read
off that block's M-step curvature `H_beta`. That curvature is the
complete-data information at the converged weights: it treats the latent
labels as known, so these `se` understate the uncertainty by the
information the latent variable withholds.

The observed-data information follows from Louis' (1982) identity,
\$\$I\_{obs}(\theta) = E\[-\nabla^2 \ell_c(\theta; z) \mid y\] -
\mathrm{Var}\[\nabla \ell_c(\theta; z) \mid y\],\$\$ evaluated at the EM
estimate. The first term is the blocks' `H_beta` (the M-step objective
is the expected complete-data log-likelihood when the weights enter
linearly, as in the weighted encoding below); the second, the missing
information, is estimated from `n_louis` hard draws
`z ~ draw_z(weights)`, each encoded by `m_step_encode()` and scored at
the estimate. The score covariance couples the blocks that share the
latent variable, so `vcov` is one joint matrix across blocks. The result
carries `vcov` (named `"<block>:<coef>"`), `se` (a named list per block)
and `louis` (`n_draws`, and `missing_fraction`, the share of each
coefficient's variance owed to the latent variable,
`1 - se_complete^2 / se^2`). It needs blocks without latent structure
(`re_list`, `spatial`, `prior`), whose fixed effects are the whole
parameter; otherwise, or when the estimated observed information is not
positive definite, `vcov` and `se` are `NULL` and `louis$unavailable`
says why. Dispersions set on a block (`phi`) are held at their values.
The draws consume the random-number stream.

`correction = "mi"` targets the same quantity by refitting each draw and
pooling with Rubin's rules. Its draws are taken at the fixed estimate,
an improper imputation that leaves out the estimate's own uncertainty,
so its pooled SE can run below the observed-information one.

## Posterior sampling

`correction = "gibbs"` alternates an exact draw from `p(Z | theta, y)`
with independence Metropolis-Hastings for `p(theta | Z, y)`. Here
`theta` contains every block's fixed effects and random-effect values.
The complete-data Laplace mode and full latent precision define a
multivariate-t proposal. The acceptance ratio uses the true
complete-data density, including priors, divided by the proposal density
at both parameter states. The current density is evaluated again after
each latent update. The resulting chain has invariant joint posterior
`p(theta, Z | y)` under the usual support and ergodicity conditions
(Tanner and Wong, 1987; Robert and Casella, 1999). Finite chains require
convergence and Monte Carlo diagnostics.

The engine density evaluates the native family likelihood, retaining
normalizing constants, and normalized Gaussian coefficient and
random-effect priors. It reads the hard complete-data blocks from
`m_step_encode(z, ...)`. These must encode every factor of
`p(y, z | theta)`, including the latent label model. Observation weights
must be 0 or 1 on this path. Supply `log_complete` when the model needs
additional factors or a different joint density. The callback replaces
the entire density, including priors.

Complete-data blocks currently support fixed effects and any number of
Gaussian random-effect terms with fixed covariance, including correlated
slopes. Dispersion is fixed. Nested, spatial and custom-likelihood
blocks are refused because their integrated fits supply no proposal for
the full conditional parameter vector. Deterministic `m_step_extra`
updates are refused on the Gibbs path. The parameter layout, coefficient
order and random-effect group identities must remain fixed across latent
draws.

During sampling the E-step receives per-block `mode` and `beta`
containing the current parameter draw. It must use those values to
compute the exact latent conditional of the same joint model as the
density. A finite MCMC approximation to that conditional does not meet
this contract. A separate unbiased-likelihood extended-state
construction is one alternative when exact latent draws are unavailable
(Andrieu and Roberts, 2009).

Each sweep costs one E-step, one complete-data Laplace solve per block,
and `1 + correction_control$mh_steps` joint-density evaluations. The
proposal factorization covers the full parameter vector. MI costs one
refit per block per imputation and retains the Tier 2 approximation.

## Tier

`"none"` and `"mi"` are Tier 2. `"gibbs"` is Tier 1 for the declared
joint posterior, conditional on the fixed dispersion and covariance.

## References

Louis, T. A. (1982). Finding the observed information matrix when using
the EM algorithm. *JRSS B*, 44(2), 226-233.

Tanner, M. A., and Wong, W. H. (1987). The calculation of posterior
distributions by data augmentation. *JASA*, 82(398), 528-540.
[doi:10.1080/01621459.1987.10478458](https://doi.org/10.1080/01621459.1987.10478458)
.

Wei, G. C. G., and Tanner, M. A. (1990). A Monte Carlo implementation of
the EM algorithm and the poor man's data augmentation algorithms.
*JASA*, 85(411), 699-704.
[doi:10.1080/01621459.1990.10474930](https://doi.org/10.1080/01621459.1990.10474930)
.

Robert, C. P., and Casella, G. (1999). *Monte Carlo Statistical
Methods*. Springer, sections 7.3.2-7.3.3.
[doi:10.1007/978-1-4757-3071-5](https://doi.org/10.1007/978-1-4757-3071-5)
.

Andrieu, C., and Roberts, G. O. (2009). The pseudo-marginal approach for
efficient Monte Carlo computations. *Annals of Statistics*, 37(2),
697-725. [doi:10.1214/07-AOS574](https://doi.org/10.1214/07-AOS574) .

## Examples

``` r
# \donttest{
# Zero-inflated Poisson via EM: the E-step scores the posterior probability
# that each zero is non-structural. Those are soft labels, so the occupancy
# arm is the weighted Bernoulli above -- two rows per unit -- and the
# abundance arm weights each count by the same w.
set.seed(1)
n <- 200
z <- rbinom(n, 1, 0.7)
y <- rpois(n, 4) * z
X <- cbind(1, rnorm(n))

e_step <- function(fits, ...) {
  if (!length(fits)) return(list(weights = pmax(as.numeric(y > 0), 0.5)))
  psi <- plogis(drop(X %*% fits$occ$mode))
  lam <- exp(drop(X %*% fits$abund$mode))
  w <- ifelse(y > 0, 1, psi * exp(-lam) / (psi * exp(-lam) + (1 - psi)))
  list(weights = w)
}
m_step_encode <- function(weights, ...) {
  list(
    occ   = list(y = rep(c(1, 0), each = n), X = rbind(X, X),
                 weights = c(weights, 1 - weights), family = "binomial"),
    abund = list(y = y, X = X, family = "poisson", weights = weights)
  )
}

res <- tulpa_em_laplace(e_step, m_step_encode, verbose = FALSE)
res$converged
#> [1] TRUE
plogis(res$fits$occ$mode[1])    # P(non-structural), truth 0.7
#> [1] 0.7034743
exp(res$fits$abund$mode[1])     # abundance mean, truth 4
#> [1] 3.871112
# }
```
