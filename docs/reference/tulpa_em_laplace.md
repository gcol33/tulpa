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
  n_gibbs = 10L,
  draw_z = NULL,
  m_step_extra = NULL,
  beta_prior = NULL,
  verbose = TRUE,
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
  `"gibbs"` runs a warm-started `z|theta -> theta|z` Markov chain of
  length `n_gibbs` starting from the EM fits, also pooled via
  [`rubins_pool()`](https://gillescolling.com/tulpa/reference/rubins_pool.md).
  `"auto"` resolves to `"none"`.

- n_imputations:

  Number of MI draws (default `20L`). Used when `correction = "mi"`.

- n_gibbs:

  Length of the Gibbs chain (default `10L`). Used when
  `correction = "gibbs"`.

- draw_z:

  Optional function `function(weights) -> hard_z` that turns the
  E-step's continuous weights into a hard latent draw. Used only by
  `correction %in% c("mi", "gibbs")`. The default treats `weights` as a
  numeric vector of Bernoulli probabilities and draws per-observation.
  Multi-class / matrix-valued latent structures must supply their own
  callback.

- m_step_extra:

  Optional `function(fits, weights, ...) -> fits`. Fired once per M-step
  in every phase (EM iterations, MI draws, Gibbs steps). Receives the
  freshly assembled list of
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  results (`fits`), the continuous E-step weights P(z\|y, theta) (NOT
  the hard z draw used by MI/Gibbs to encode the block), and any extra
  arguments forwarded through `...`. Returns a list with the same length
  and names as the input, possibly with mutated dispersion / shape /
  precision fields (e.g. `fits[[k]]$phi`). Use this to update non-eta
  parameters that fall out of the Laplace M-step (NB overdispersion,
  Gamma shape, Beta precision, Gaussian sigma). When `NULL` (default),
  behavior is unchanged.

- beta_prior:

  Optional Gaussian prior on the fixed effects, applied to every block
  fit via
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  (i.e. blocks without a `prior` field). `NULL` (default) keeps the weak
  built-in prior. Otherwise a list with `sd` (required) and optional
  `mean`; see
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md).
  The same prior flows into the MI / Gibbs correction refits, so
  penalized corrections come for free. A block may override the default
  by setting its own `beta_prior` field in `m_step_encode` (e.g.
  different priors for the occupancy and detection submodels). Use
  scalar `mean` / `sd` here when blocks differ in width; per-coefficient
  vectors belong on the block.

- verbose:

  Print per-iteration progress.

- ...:

  Forwarded to `e_step`, `m_step_encode`, and `m_step_extra`.

## Value

A list with:

- `fits` – named list of
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  results, one per block.

- `weights` – final E-step weights.

- `n_iter` – number of EM iterations actually run.

- `converged` – logical.

- `history` – `data.frame(iter, delta)` of max relative parameter change
  per iteration.

- `correction` – the resolved correction mode (`"none"`, `"mi"`, or
  `"gibbs"`).

- `pooled` – present when `correction %in% c("mi", "gibbs")`. Named list
  of pooled per-submodel summaries from
  [`rubins_pool()`](https://gillescolling.com/tulpa/reference/rubins_pool.md).

- `draws` – present when `correction %in% c("mi", "gibbs")`. List of
  per-draw fits with `beta` / `se` attached.

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
#> [1] 0.7041959
exp(res$fits$abund$mode[1])     # abundance mean, truth 4
#> [1] 3.872501
# }
```
