# Adaptive Gauss-Hermite refinement of a grouped random-effect covariance

Refines a generalized linear mixed model's fixed effects and
random-effect covariance by replacing the per-group Laplace integral
with `n_quad`-point adaptive Gauss-Hermite quadrature (AGHQ). At
`n_quad = 1` this is the joint Laplace (glmer `nAGQ = 1`); higher
`n_quad` reduces the small-cluster attenuation of the variance
components for binary / count data. Unlike
[`agq_fit()`](https://gillescolling.com/tulpa/reference/agq_fit.md)
(intercept-only RE, built-in `binomial`/`poisson`/`gaussian`
likelihoods), this engine is **callback-driven**: the caller supplies
the per-group conditional likelihood, so a custom marginal (e.g. a
latent-state-integrated occupancy / detection likelihood, or the
latent-abundance-integrated N-mixture marginal) refines through the same
quadrature.

This is an engine block, not a front door: model packages call it
programmatically, so its tuning knobs (`max_iter`, `n_quad`, `keep`) sit
in the signature rather than in a `control` list.

The engine is **structure-agnostic**. It integrates the per-group
marginal \$\$M_g = \int \exp\\\ell_g(b_g)\\\\ N(b_g; 0, \Sigma)\\
db_g,\$\$ where \\b_g\\ is the group's random-effect vector (dimension
\\\sum_m c_m\\ over the RE terms) and \\\ell_g(b_g)\\ is the group's
conditional log-likelihood when its linear predictors are perturbed by
\\b_g\\. How \\b_g\\ enters the likelihood – through one linear
predictor, or through several coupled arms at different observation
granularities (e.g. a per-site abundance arm and a per-visit detection
arm sharing a species grouping) – lives entirely in the callback. The
engine only needs, per group, the value / gradient / Hessian of
\\\ell_g\\ in \\b_g\\ (for the mode) and \\\ell_g\\ at the quadrature
nodes (for the sum). The fixed parameters `theta` and the log-Cholesky
coordinates of \\\Sigma\\ are optimized jointly on \\\sum_g \log M_g\\;
standard errors come from the exact-marginal Hessian.

Two callback forms select the structure (supply exactly one):

- **`make_site`** – the common **single-arm, per-row-separable** case
  (\\\ell_g(b_g) = \sum\_{i \in g} \log f_i(\eta_i + Z_i b_g)\\ for one
  linear predictor \\\eta\\). The engine builds the oracle from the
  per-row marginal and the RE design `Z` itself.

- **`make_group`** – the **general / multi-arm** case. The caller
  supplies the per-group `b`-space oracle directly, so non-separable
  units (e.g. the visits of an N-mixture site coupled through the shared
  latent count) and random effects on several arms at once are handled
  with no engine change.

Scope: one shared grouping factor across all RE terms (the per-group
integral factorizes). The total RE dimension per group should be small
(the quadrature grid is `n_quad^dim`).

## Usage

``` r
tulpa_re_aghq(
  theta0,
  re_terms,
  Sigma0,
  make_site = NULL,
  make_group = NULL,
  oracle = NULL,
  n_obs = NULL,
  keep = NULL,
  n_quad = 9L,
  lkj_eta = 1,
  theta_prior_sd = Inf,
  sigma_prior = NULL,
  gradient = c("fd", "analytic"),
  max_iter = 200L
)
```

## Arguments

- theta0:

  Initial fixed-parameter vector. The engine optimizes these jointly
  with the RE covariance; the callback interprets them.

- re_terms:

  A list of RE term specs (or one spec), each defining a covariance
  block: `n_coefs` (block dimension \\c_m\\), optional `correlated`
  (default `TRUE` for `c_m > 1`; `FALSE` gives a diagonal block), and
  `n_groups` (shared across terms). For the `make_site` path each term
  also carries `idx` (1-based group index, length `n_obs`) and, for a
  slope block, `Z` (the `n_obs x n_coefs` design). For the `make_group`
  path the per-observation `idx` / `Z` are optional – the callback owns
  them – and the term needs only `n_coefs` / `correlated` / `n_groups`.

- Sigma0:

  List of initial per-term covariance matrices (the EM estimate).

- make_site:

  `function(theta)` for the single-arm separable case, returning a list
  with: `eta_re` (length `n_obs`, the RE-arm fixed predictor),
  `deriv = function(rows, eta)` returning `list(logL, d1, d2)` (per-row
  marginal log-likelihood and its first/second derivatives w.r.t. the
  RE-arm predictor `eta`, used for the per-group mode), and
  `lmat = function(rows, ETA)` returning a `length(rows) x ncol(ETA)`
  matrix of per-observation log-likelihoods over the quadrature node
  columns. Supply this or `make_group`, not both.

- make_group:

  `function(theta)` for the general / multi-arm case, returning a list
  with two per-group closures (let `d = sum(n_coefs)` be the group RE
  dimension):

  - `grad_hess(g, b)` – for group `g` at RE value `b` (length `d`), the
    list `list(logL, grad, negH)`: the group conditional log-likelihood
    \\\ell_g(b)\\, its gradient \\\partial \ell_g/\partial b\\ (length
    `d`), and the **data-only** observed information \\-\partial^2
    \ell_g/\partial b^2\\ (`d x d`; the engine adds the \\\Sigma^{-1}\\
    prior curvature).

  - `node_ll(g, B)` – for group `g`, a numeric vector of length
    `nrow(B)` giving \\\ell_g\\ at each quadrature node (rows of the
    `nrow x d` matrix `B` are candidate `b` vectors). The callback owns
    all arm / design / clamping bookkeeping. Supply this or `make_site`,
    not both.

- oracle:

  Optional prebuilt native (compiled) oracle, an external pointer to a
  `REGroupOracle` (constructed in a consumer package's src/ via
  `LinkingTo: tulpa` against `<tulpa/aghq_oracle.h>`). When supplied the
  engine drives it directly, with no per-group / per-node round trip
  into R, and neither `make_site` nor `make_group` is needed;
  `re_terms`, `theta0` and `Sigma0` must still describe the same layout
  the oracle exposes. The integration core is identical to the R-closure
  path.

- n_obs:

  Number of observations (length of each term's `idx`). Required for the
  `make_site` path; ignored for `make_group`.

- keep:

  Optional logical/integer mask of observations to include (default all;
  `make_site` path only). Rows outside `keep` are dropped from every
  group.

- n_quad:

  Quadrature nodes per RE dimension. Either a single integer (default 9;
  `1` = Laplace) broadcast to every covariance block, or an integer
  vector of length `length(re_terms)` giving a per-block node count. The
  tensor grid then uses `n_quad[b]` nodes along every dimension of block
  `b`, for `prod_b n_quad[b]^(dim_b)` total nodes; a scalar reproduces
  the uniform grid exactly. Per-block orders let a heterogeneous stack
  spend fewer nodes on cheap scalar nuisance blocks than on the
  correlated coefficient blocks (e.g. `c(3, 3, 2, 2)` on blocks of
  dimension `2, 2, 1, 1` gives `3^2 * 3^2 * 2 * 2 = 324` nodes rather
  than `3^6 = 729`).

- lkj_eta:

  LKJ shape for an optional correlation penalty on each *correlated*
  block (log-density `(eta - 1) log det R`, maximized at independence).
  `1` disables it; `> 1` regularizes a weakly-identified correlation off
  the boundary without touching the marginal SDs. The marginal SDs are
  otherwise unpenalized (pure ML), so the refinement debiases them
  rather than shrinking them.

- theta_prior_sd:

  Optional Gaussian ridge SD on the fixed parameters `theta` (a
  mean-zero `N(0, theta_prior_sd^2)` prior, added to the optimized
  objective and hence the marginal Hessian). `Inf` (default) is pure ML
  on `theta`; a large finite value (e.g. 100) is a weak ridge that
  stabilizes a weakly-identified fixed effect without materially
  shifting the estimate.

- sigma_prior:

  Optional Penalized-Complexity prior on the marginal standard
  deviations of one or more RE covariance blocks, added to the objective
  (and hence the marginal Hessian). `NULL` (default) is pure ML on the
  covariances – the refinement debiases the SDs rather than shrinking
  them. Otherwise a `c(U, alpha)` pair (`P(sigma_i > U) = alpha`, the
  same convention as
  [`re_cov_pc_lkj_prior()`](https://gillescolling.com/tulpa/reference/re_cov_pc_lkj_prior.md))
  applied to every block, or a list
  `list(blocks = <integer indices>, prior_sigma = c(U, alpha))` applied
  to the named blocks only. Reuses the exact PC log-prior + Jacobian of
  [`re_cov_pc_lkj_prior()`](https://gillescolling.com/tulpa/reference/re_cov_pc_lkj_prior.md).
  A weakly-identified variance component (e.g. a scalar dispersion /
  zero-inflation random effect at few groups) can drift to the boundary
  and flatten the marginal Hessian; a weak PC prior adds curvature there
  (the `+ log sigma` Jacobian repels `sigma -> 0`, the `- lambda sigma`
  term caps inflation), keeping the joint optimum non-singular without
  materially shifting an identified fit.

- gradient:

  How [`stats::optim`](https://rdrr.io/r/stats/optim.html) gets the
  gradient of the AGHQ objective. `"fd"` (default) lets `optim`
  finite-difference the objective – correct at every `n_quad` and the
  only option for the R-closure (`make_site` / `make_group`) paths.
  `"analytic"` supplies the Fisher-identity gradient (posterior-weighted
  theta-score plus the `Sigma` moment-matching residual), which avoids
  the per-coordinate objective re-solve and so is far cheaper for the
  quadrature debias. It requires a prebuilt native `oracle` (the only
  one exposing the theta-score) and `n_quad > 1`: being the gradient of
  the true marginal it omits the node-placement terms (`O` of the AGHQ
  truncation), so it agrees with the objective only as `n_quad` grows.

- max_iter:

  Optimizer iteration cap (default 200).

## Value

A list with: `theta` (refined fixed parameters), `Sigma_list` (refined
per-term covariance), `blup` / `blup_var` (per-term `n_groups x n_coefs`
posterior mean / variance of the RE), `group_ok` (logical, length
`n_groups`: `FALSE` where that group's mode search or precision
factorization failed, which is what the `NA` rows of `blup`, `blup_var`,
`blup_cov_g` and `blup_cross_g` mean – a caller conditions its per-group
reads on this rather than on trapping the accompanying warning),
`blup_cross` (per-term `n_groups x n_theta x n_coefs` array: the
mode/theta cross-Hessian block `Bf`, `-d^2 ell_g / d theta db` at each
group's mode, in the same negative-Hessian sign convention as the
posterior precision underlying `blup_var` – so a joint draw of `theta`
and a group's RE `b_g` uses
`b_g | theta ~ N(blup_g - Cinv_g %*% t(Bf_g) %*% (theta_draw - theta), Cinv_g)`
with `Cinv_g` the group's `n_coefs x n_coefs` posterior covariance
block. `NA` throughout when `blup_cross_available` is `FALSE`: the
cross-Hessian needs the oracle's analytic `theta_score`, which the
R-closure bridge (`make_site` / `make_group`) does not supply – only a
prebuilt native `oracle` carries it), `blup_cov_g` / `blup_cross_g`
(per-group lists, length `n_groups`, of the FULL joint posterior
covariance (`d x d`, `d` = every RE term's width combined) and
mode/theta cross-Hessian (`n_theta x d`) across ALL RE terms sharing
that group – the superset `blup_var`/`blup_cross` reduce to a per-term
diagonal block of when a group carries more than one term, since a
group's terms are found jointly and can carry real posterior covariance
BETWEEN terms (e.g. an abundance-arm and a detection-arm term sharing
one grouping factor); same `NA`-when-unavailable rule as `blup_cross`),
`theta_cov` / `theta_se` (fixed-parameter covariance / SE from the
marginal Hessian), `re_par` / `re_par_cov` / `re_par_se` (the
RE-covariance coordinates the optimizer carried – log-Cholesky for a
full block, log-SD for a diagonal one – with their block of the same
inverse Hessian, so `SE(log sigma)` is available for a boundary test on
a weakly-identified variance component), `re_par_layout` (per block:
`label`, `nc`, `full`, the `index` range into `re_par` and the `coord`
names, so a caller does not reconstruct the packing), `joint_cov` (the
whole `(n_theta + n_chol)` inverse Hessian), `log_marginal` (the AGHQ
marginal log-likelihood at the optimum, excluding any ridge), `n_quad`,
`lkj_eta`, `converged`, and `counts`
([`stats::optim`](https://rdrr.io/r/stats/optim.html)'s own `function` /
`gradient` evaluation counts, so a caller reporting how much work the
fit took has a number to report rather than `NA`). RE terms that do not
share one grouping factor are an input error and stop. Three conditions
warn and return `NULL` (caller keeps its prior fit): a singular /
non-finite optimum, an objective that is already undefined at the
starting parameters (some group's solve fails there, so there is nothing
to descend), and an optimum whose objective is the failure sentinel
rather than an attained marginal likelihood – the last two report which
groups failed.

## References

Pinheiro & Bates (1995). Approximations to the log-likelihood function
in the nonlinear mixed-effects model. *Journal of Computational and
Graphical Statistics* 4(1):12-35. Lewandowski, Kurowicka & Joe (2009).
Generating random correlation matrices based on vines and extended onion
method. *Journal of Multivariate Analysis* 100(9):1989-2001.

## Examples

``` r
# \donttest{
# A per-row-separable binomial GLMM marginal supplied through `make_site`.
l1pe <- function(x) ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))
make_binom_site <- function(X, y, nt) function(theta) {
  eta_fixed <- as.numeric(X %*% theta)
  list(eta_re = eta_fixed,
       deriv = function(rows, eta) {
         p <- plogis(eta)
         list(logL = y[rows] * eta - nt[rows] * l1pe(eta),
              d1 = y[rows] - nt[rows] * p, d2 = -nt[rows] * p * (1 - p))
       },
       lmat = function(rows, ETA) y[rows] * ETA - nt[rows] * l1pe(ETA))
}
set.seed(1)
ng <- 30L; npg <- 8L; n <- ng * npg
g <- rep(seq_len(ng), each = npg); x <- rnorm(n)
X <- cbind(1, x); nt <- rep(3L, n); u <- rnorm(ng, 0, 0.9)
y <- rbinom(n, nt, plogis(0.3 + 0.7 * x + u[g]))
fit <- tulpa_re_aghq(theta0 = c(0, 0),
                     re_terms = list(list(idx = g, n_groups = ng, n_coefs = 1L)),
                     Sigma0 = list(matrix(0.25, 1, 1)),
                     make_site = make_binom_site(X, y, nt), n_obs = n, n_quad = 5L)
sqrt(fit$Sigma_list[[1]][1, 1])     # adaptive-GHQ RE standard deviation
#> [1] 0.796036
# }
```
