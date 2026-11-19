# Block-coordinate ascent for a grouped model with a shared field and latent factors

Fits a grouped (community) model whose linear predictor on one arm
carries a shared Gaussian Markov random field, latent factors, or both:
\$\$\eta\_{s,i} = X_i (\mu + b_s) + \sum_k W\_{ik} (M^\top F)\_{ik} +
\lambda_s^\top \zeta_i,\$\$ with \\b_s \sim N(0, \Sigma)\\ per-group
coefficient deviations, \\F\\ the field(s) on the geometry's nodes and
\\\zeta_i \sim N(0, I_Q)\\ per-site factors with per-group loadings
\\\lambda_s\\. Each outer pass runs (a) the caller's coefficient EM with
the combined latent structure as a per-cell offset, (b) a Newton field
solve with a closed-form precision M-step, and (c) the loading update.
The loadings are estimated by marginal maximum likelihood: an EM on the
joint site marginal with the factors integrated out on an adaptive
Gauss-Hermite grid, initialised on the first pass from several candidate
directions (a joint-mode ascent from the supplied start, a
principal-factor direction, and fixed-seed random restarts), each scaled
by a bracketed 1-D magnitude search and selected on the converged
marginal. The offset the coefficient EM conditions on matches the
integrated score per cell. After the loop the field's precision
hyperparameters are profiled over `geometry$hyper_grid` by the Laplace
field marginal, through
[`tulpa_hyper_grid()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_grid.md).

## Usage

``` r
tulpa_latent_ascent(
  make_oracle,
  em_fit,
  offset_of,
  n_sites,
  n_species,
  geometry = NULL,
  factors = NULL,
  control = list()
)
```

## Arguments

- make_oracle:

  `function(em)` returning the family oracle at the current
  coefficients: a list with `working(eta)` (a list of `score` and
  `curv`, each `[n_sites x n_species]`: the score and curvature of the
  marginal log-likelihood with respect to an additive offset on the
  arm's predictor), `data_ll(eta)` (the summed data log-likelihood),
  `ll_cell(eta, idx = NULL)` (per-cell marginal log-likelihoods;
  required when `factors` is given; may honour `idx` by returning only
  those site rows) and optionally `cov_curv(eta)` (the curvature used
  for the field's Laplace covariance).

- em_fit:

  `function(site_off, fac_off, em_prev)` running the caller's
  coefficient EM with the field offset `site_off` (length `n_sites`) and
  the factor offset `fac_off` (`[n_sites x n_species]`), warm-started
  from `em_prev` (`NULL` on the first pass); returns the EM state.

- offset_of:

  `function(em)` returning the arm's `[n_sites x n_species]` coefficient
  predictor at that EM state.

- n_sites, n_species:

  Dimensions of the cell grid.

- geometry:

  `NULL` (no field) or a list describing the field:

  - `type` – a label returned with the result; `"bym2"` selects the BYM2
    field \\\phi = a v + b u\\ (structured \\v\\, unstructured \\u \sim
    N(0, I)\\, \\a = \sigma\sqrt{\rho / scale}\\, \\b = \sigma\sqrt{1 -
    \rho}\\), any other value a field with prior \\\tau_k Q\\.

  - `Q` – `function(hyper)` returning the unit-scale sparse precision at
    the named numeric hyperparameters `hyper` (the structured precision
    for BYM2).

  - `hyper` – named numeric hyperparameters used during the ascent
    (BYM2: `sigma`, `rho`); `numeric(0)` when `Q` has none.

  - `hyper_grid` – named list of axis grids profiled after the ascent,
    the first axis varying fastest (`NULL` or empty: none).

  - `components` – list of node index sets, each held at sum zero (one
    per null direction of an intrinsic precision); `NULL` leaves the
    field unconstrained. Its length is the rank deficiency the precision
    M-step and the profile log-determinant discount.

  - `M` – the `[n_nodes x n_sites]` node-from-site map (a group
    incidence or a projector transpose), or `NULL` for one node per
    site.

  - `W` – `[n_sites x K]` per-field weights (ones for an intercept
    field).

  - `n_nodes` – the number of field nodes.

  - `scale` – BYM2 only: the generalised variance the structured part is
    divided by.

- factors:

  `NULL` (no factors) or `list(zeta, lambda)`: starting values,
  `[n_sites x Q]` and `[n_species x Q]`.

- control:

  List of

  - `tol` (`1e-4`) – convergence tolerance on the largest change of the
    field or factor offset between passes.

  - `max_outer` (`NULL`) – outer-pass budget; `NULL` takes
    `factor_outer` when factors are present, else 400 (a field run stops
    at `tol`, so this is a cap; how many passes it needs depends on how
    strongly the field overlaps the coefficient block).

  - `factor_outer` (`25`) – the budget a factor model takes by default
    (the factor block alternates with the coefficient block along a slow
    mode and rarely meets `tol`).

  - `factor_starts` (`8`) – candidate starting directions on the first
    pass.

  - `n_quad` (`5`) – Gauss-Hermite nodes per factor dimension.

  - `verbose` (`FALSE`), `label` (`"latent"`) – per-pass progress
    message and its prefix.

## Value

A list with

- `em` – the last coefficient EM state.

- `geometry` – the geometry as supplied.

- `settings` – the resolved `max_outer`, `tol`, `factor_outer`,
  `factor_starts`, `n_quad` (the factor knobs `NA` without factors).

- `field` (`NULL` without a field) – `Fm` (`[n_nodes x K]`), `tau`,
  `type`, `hyper` (the selected hyperparameters), `site_off` (the
  per-site field contribution) and, for BYM2,
  `bym2 = list(sigma, rho, scale, phi)`.

- `factor` (`NULL` without factors) – `n_factors`, `zeta`, `lambda`,
  `offset` (the score-matched per-cell offset the coefficients were
  fitted against), `magnitude_saturated`, `loading_em_converged` and
  `marginal_loglik` (the joint marginal at the returned loadings).

## See also

[`tulpa_re_em()`](https://gillescolling.com/tulpa/reference/tulpa_re_em.md)
for a coefficient EM `em_fit` can wrap,
[`tulpa_hyper_grid()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_grid.md),
[`tulpa_gauss_hermite()`](https://gillescolling.com/tulpa/reference/tulpa_gauss_hermite.md),
[`tulpa_car_precision_Q()`](https://gillescolling.com/tulpa/reference/tulpa_car_precision_Q.md),
[`tulpa_spde_precision_Q()`](https://gillescolling.com/tulpa/reference/tulpa_spde_precision_Q.md)

## Examples

``` r
# Poisson counts with a shared latent factor and one free intercept per
# species, the intercepts updated by a closed-form Newton step.
set.seed(1)
Ns <- 40L; S <- 5L
lam <- matrix(rnorm(S, 0, 0.7), S, 1L)
y <- matrix(rpois(Ns * S, exp(0.5 + tcrossprod(rnorm(Ns), lam))), Ns, S)
oracle <- list(
  working = function(eta) list(score = y - exp(eta), curv = exp(eta)),
  ll_cell = function(eta, idx = NULL) {
    if (!is.null(idx)) { eta <- eta[idx, , drop = FALSE]; yy <- y[idx, , drop = FALSE] }
    else yy <- y
    yy * eta - exp(eta)
  },
  data_ll = function(eta) sum(y * eta - exp(eta)))
em_fit <- function(site_off, fac_off, em_prev) {
  b <- if (is.null(em_prev)) rep(0, S) else em_prev
  for (it in 1:20) b <- b + colSums(y - exp(outer(rep(1, Ns), b) + fac_off)) /
    colSums(exp(outer(rep(1, Ns), b) + fac_off))
  b
}
res <- tulpa_latent_ascent(
  make_oracle = function(em) oracle, em_fit = em_fit,
  offset_of = function(b) outer(rep(1, Ns), b), n_sites = Ns, n_species = S,
  factors = list(zeta = matrix(scale(cos(seq_len(Ns)))[, 1], Ns, 1L),
                 lambda = matrix(0.1, S, 1L)),
  control = list(max_outer = 3L, factor_starts = 2L))
cor(res$factor$lambda, lam)
#>            [,1]
#> [1,] -0.9759935
```
