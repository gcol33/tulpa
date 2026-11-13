# Grouped random effects by Laplace-EM on a per-group oracle

The `n_quad = 1` sibling of
[`tulpa_re_aghq()`](https://gillescolling.com/tulpa/reference/tulpa_re_aghq.md)
on the same per-group oracle: an expectation-conditional-maximisation
(ECM) fit of a model whose conditional likelihood factorises over the
levels of one grouping factor. Each iteration

1.  solves every group's posterior mode `b_g` and covariance
    `C_g = Cov(b_g | y)` at the current parameters (the E-step, the same
    per-group solve the AGHQ engine reports as its BLUPs);

2.  updates each random-effect covariance in closed form,
    `Sigma_k = mean_g(b_gk b_gk' + C_gk)`, projected to its diagonal for
    an uncorrelated term and floored at `sigma_floor^2` on its
    eigenvalues;

3.  moves the fixed parameters `theta` (see `theta_update`).

It stops when the Laplace log-marginal changes by less than `tol`
relative to its size.

## Usage

``` r
tulpa_re_em(
  theta0,
  re_terms,
  Sigma0,
  make_site = NULL,
  make_group = NULL,
  oracle = NULL,
  n_obs = NULL,
  keep = NULL,
  theta_prior_sd = Inf,
  sigma_floor = 1e-04,
  tol = 1e-08,
  max_iter = 200L,
  theta_iter = 3L,
  n_threads = 1L,
  theta_update = c("marginal", "latent"),
  b0 = NULL,
  newton_max = 30L,
  progress_label = "re-em",
  aghq_debias = 0L
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

  As in
  [`tulpa_re_aghq()`](https://gillescolling.com/tulpa/reference/tulpa_re_aghq.md);
  the list it returns may also carry `joint(g, b)`, returning
  `list(logL, grad, negH, theta_grad, theta_info, cross)`: the
  `grad_hess` blocks plus the data theta score \\\partial \ell_g /
  \partial\theta\\, the theta observed information \\-\partial^2 \ell_g
  / \partial\theta^2\\ (`n_theta x n_theta`) and the cross block
  \\-\partial^2 \ell_g / \partial\theta \partial b\\ (`n_theta x d`).
  `theta_update = "latent"` requires it.

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

- theta_prior_sd:

  Prior standard deviation of `theta`, one value or one per coordinate;
  `Inf` leaves a coordinate unpenalized.

- sigma_floor:

  Floor on every random-effect standard deviation, applied to the
  eigenvalues of each covariance update.

- tol:

  Relative tolerance on the change in the Laplace log-marginal.

- max_iter:

  Maximum EM iterations.

- theta_iter:

  Maximum BFGS steps on `theta` per EM iteration.

- n_threads:

  Threads for the per-group loops, used only when the oracle declares
  itself thread-safe. The result does not depend on it.

- theta_update:

  `"marginal"` (default) or `"latent"`; see Details.

- b0:

  Optional `n_groups x d` matrix of starting group modes (`d` the total
  random-effect dimension, terms in `re_terms` order). Absent, the
  latent rule starts every group at zero and the marginal rule at each
  group's own solve.

- newton_max:

  Maximum joint Newton steps per iteration (`theta_update = "latent"`).

- progress_label:

  Label of the progress line and heartbeat (the `tulpa.nl_progress`
  option sets whether they are written).

- aghq_debias:

  Latent rule only: a per-axis node count `> 1` adds one adaptive
  Gauss-Hermite moment step at the converged mode. Each group's
  posterior second moment is integrated on the AGHQ engine's tensor grid
  (centred at the mode, scaled by the group's posterior covariance) and
  the covariance update takes their mean, which removes the Laplace
  small-cluster attenuation of the variance components; the group
  covariances and the `theta` covariance are then recomputed at the new
  covariance through the likelihood curvature. `0` (default) skips it.

## Value

A list with `theta`, `theta_cov`, `theta_hessian` (the `theta` curvature
`theta_cov` inverts; `theta_cov` is `NULL` when it is not positive
definite), `theta_se`, `Sigma_list`, `log_marginal` (Laplace), `blup`,
`blup_var`, `blup_cov_g` (per-group posterior covariances),
`blup_cross_g` (mode/theta cross-Hessian blocks, the `Bf` of
[`tulpa_re_aghq()`](https://gillescolling.com/tulpa/reference/tulpa_re_aghq.md)),
`group_ok`, `converged`, `n_iter`, `blup_cross_available`,
`log_marginal_trace`, `stuck` (latent rule: the final joint Newton
accepted no step at any damping, so `converged` is `FALSE`),
`aghq_debiased` (whether the `aghq_debias` step ran; it leaves the
Laplace fit in place when a factorization or quadrature sum fails);
`NULL` with a warning when the per-group solve fails at the start.

## Details

`theta_update = "marginal"` moves `theta` by BFGS on the Laplace
marginal at the current covariance, with its exact (envelope) gradient,
and reports the inverse curvature of that marginal as the `theta`
covariance. Its gradient differentiates each group's Hessian in `theta`,
which the engine does by central differences.

`theta_update = "latent"` treats `theta` as a latent Gaussian variable
with prior `N(0, theta_prior_sd^2)`, as INLA treats fixed effects:
`theta` and every group's mode are solved jointly by a damped Newton on
the penalized joint log-density (the Hessian is an arrowhead, solved
through the Schur complement of the random-effect blocks), and the
`theta` covariance is the inverse of that Schur complement. It needs
only first and second derivatives of the group log-likelihood, including
its `theta` score: a native oracle supplies it, and a `make_group`
callback through its optional `joint` closure (below). With a Gaussian
conditional and `theta` entering the linear predictor, the two rules
reach the same `theta`.

## See also

[`tulpa_re_aghq()`](https://gillescolling.com/tulpa/reference/tulpa_re_aghq.md)
for adaptive Gauss-Hermite marginal fit on the same oracle.
