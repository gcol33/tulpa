# Empirical-Bayes random-effect covariances

Estimate one or more random-effect covariances `Sigma` by maximizing the
Laplace marginal likelihood over them (plus the hyperprior), then report
the fixed effects conditional on the maximizer. This is the plug-in
("ML-II" / empirical-Bayes) counterpart of
[`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md),
which integrates over `Sigma` instead of fixing it at the maximizer.

## Usage

``` r
tulpa_eb(
  y,
  n_trials = NULL,
  X,
  re_terms,
  family = "binomial",
  phi = 1,
  phi2 = NULL,
  prior_sigma = c(3, 0.05),
  eta = 2,
  hyperprior = c("flat", "pc_lkj"),
  log_prior_theta = NULL,
  beta_prior = NULL,
  offset = NULL,
  n_quad = 1L,
  marginal = FALSE,
  estimate_phi = FALSE,
  X_zi = NULL,
  zi_prior_sd = 2.5,
  control = list()
)
```

## Arguments

- y, n_trials, X, family, phi:

  Passed to
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  for the inner solve. `n_trials = NULL` defaults to 1 (binary /
  single-trial).

- re_terms:

  Either a single random-effect term or a list of them; see
  [`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
  for the per-term fields.

- phi2:

  Optional second dispersion, threaded into every inner
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  solve: the Student-t degrees of freedom (`family = "t"`, default 4
  when `NULL`) or the Tweedie variance power (`family = "tweedie"`,
  required – a defaulted power would be a statistical decision the
  caller never made). A `phi2` supplied for any other family errors
  rather than being ignored. It is conditioned on, never estimated:
  `estimate_phi` covers `phi` alone.

- prior_sigma, eta:

  Hyperparameters of the PC + LKJ prior used when
  `hyperprior = "pc_lkj"` (see
  [`re_cov_pc_lkj_prior()`](https://gillescolling.com/tulpa/reference/re_cov_pc_lkj_prior.md)).
  Ignored when `hyperprior = "flat"` or `log_prior_theta` is supplied.
  When active, the prior is part of the maximized objective, so it
  regularizes the estimate: with few groups it is what keeps a block off
  the `sigma = 0` boundary.

- hyperprior:

  `"flat"` (default) or `"pc_lkj"`. `"flat"` maximizes with
  `log_prior_theta` the zero function – an unpenalized maximum-marginal-
  likelihood estimate, which can reach the `sigma = 0` boundary on small
  designs (see the `"lower end of the search bracket"` warning).
  `"pc_lkj"` builds the PC + LKJ prior from `prior_sigma` / `eta`,
  regularizing the estimate away from that boundary. Ignored when
  `log_prior_theta` is supplied. Must match `hyperprior` on the paired
  [`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
  call for the two to share `theta_hat`.

- log_prior_theta:

  Optional `function(theta)` returning a scalar log prior density on the
  full stacked parameter vector, overriding `hyperprior` entirely.
  Default `NULL`, which defers to `hyperprior`.

- beta_prior:

  Optional Gaussian prior on the fixed effects, threaded into every
  inner
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  solve (`list(mean, sd)`).

- offset:

  Optional observation-level offset on the linear predictor (length
  `length(y)`), e.g. `log(exposure)` for a rate model. Not supported
  with `n_quad > 1`, which errors rather than dropping it.

- n_quad:

  Quadrature order for the inner marginal. `1` (default) uses the
  joint-field Laplace inner solve. `> 1` refines it with `n_quad`-point
  adaptive Gauss-Hermite quadrature, which requires a single shared
  grouping factor; see
  [`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md).

- marginal:

  Report fixed-effect intervals that carry the hyperparameter
  uncertainty, instead of the intervals conditional on `theta_hat`. The
  posterior for `theta` is taken as Gaussian around the maximizer with
  covariance `solve(H_theta)`, the inner mode is linearized in `theta`,
  and the law of total variance adds `J solve(H_theta) J'` to the
  conditional covariance, where `J = d mode / d theta`. Both `H_theta`
  and `J` come from one central-difference stencil over the outer
  objective, costing `1 + 2k^2` further inner solves for `k`
  hyperparameter coordinates (`k` is `1` for a scalar `(1 | g)` block
  and `3` for a correlated `(1 + x | g)` one). Default `FALSE`. Widens
  intervals; never narrows them. When the variance components themselves
  are the target, or the correction's two approximations look strained
  (a strongly skewed variance-component marginal), integrate with
  [`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
  instead.

- estimate_phi:

  Estimate the family's dispersion alongside the random-effect
  covariances, instead of conditioning on `phi`. When `TRUE` the
  supplied `phi` is the starting value and `fit$phi` is the estimate,
  with `fit$phi_estimated` distinguishing the two cases. `log(phi)`
  joins the maximization as one further coordinate, carrying the exact
  derivative of the Laplace log-marginal with respect to it, so the cost
  is one more coordinate for BFGS and not a second optimization.

  The dispersion enters unpenalized – the hyperprior covers the
  covariance coordinates only – so this is the ML-II estimate of `phi`,
  not a MAP under an undeclared prior.

  Available for every family carrying a dispersion, which is every
  front-door family except `poisson`, `binomial` and `truncated_poisson`
  – those have no free dispersion at all, so estimating one is a
  category error rather than a missing feature, and it is refused. Needs
  `n_quad = 1`.

  Alongside `X_zi` both mixture kinds are covered. A hurdle (a
  zero-truncated base) has zero branch `log(pi)`, which carries no
  dispersion, so the base family's registered derivative is already the
  mixture's. Genuine zero inflation has zero branch
  `log(pi + (1 - pi) P(Y = 0))`, which depends on `phi` through
  `P(Y = 0)` and couples it to both linear predictors; that branch is
  supplied for `neg_binomial_2`, the only untruncated mixture family
  here with a free dispersion. Other untruncated bases are refused
  rather than handed the base derivative under a model it does not
  describe.

- X_zi:

  Optional zero-inflation design matrix (`length(y)` rows), making the
  model a two-process mixture: each observation is a structural zero
  with probability `plogis(X_zi beta_zi)` and otherwise follows
  `family`. Paired with a zero-truncated family it is the hurdle model.
  The random effects enter the count predictor only, and the
  maximization is over the same covariance coordinates – the mixture
  changes the inner solve, not the outer objective's parameters. The ZI
  coefficients are reported alongside the count ones in
  [`coef()`](https://rdrr.io/r/stats/coef.html) /
  [`vcov()`](https://rdrr.io/r/stats/vcov.html), so the fixed block is
  `ncol(X) + ncol(X_zi)` wide. Needs `n_quad = 1`: the adaptive
  Gauss-Hermite inner marginal runs through a single-predictor oracle.

- zi_prior_sd:

  Prior SD on `beta_zi`, keeping the logit identified where a level
  carries no zeros (the likelihood alone would send it to `-Inf`).
  Ignored when `X_zi` is `NULL`.

- control:

  A named list of numerical knobs: `max_iter`, `tol`, `n_threads`
  (inner-solve controls, see
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)),
  and `outer_maxit` (iteration budget for the maximization over `Sigma`,
  default 500; applies to the Nelder-Mead simplex used from two
  parameters up, since the one-parameter case is bracketed by Brent).
  Exhausting the budget warns. `outer_reltol` sets that maximization's
  convergence tolerance (default `1e-10` for the gradient-driven
  methods, `1e-8` for the simplex, which cannot resolve as finely); it
  is converted to L-BFGS-B's `factr` on the bounded path, so one request
  means the same thing whichever method runs. `sigma_init` supplies the
  starting random-effect SD – a scalar, or one per coefficient across
  all blocks – replacing the method-of-moments guess taken from a pilot
  fit at `Sigma = I`. Worth setting when the true scale is far from 1,
  where that pilot starts the search on a flat stretch, and when a run
  should be reproducible from its inputs rather than from a pilot fit.
  It is diagonal: it sets each coefficient's scale and leaves any
  correlation to be fitted. Two further knobs tune `marginal = TRUE` and
  are inert without it: `marginal_step` (the stencil step in `theta`
  space, default `1e-3`) and `marginal_richardson` (default `FALSE`;
  evaluate the stencil at `step` and `step / 2` and extrapolate, turning
  the `O(step^2)` truncation error into `O(step^4)` at twice the solves
  – worth it only when the inner solver's own noise sits well below the
  truncation error, i.e. a tight `tol`).

## Value

A `tulpa_fit` with:

- `mode`, `H_beta`: the fixed-effect (and, on the `n_quad = 1` path,
  random- effect) mode and the fixed-effect precision at `theta_hat`,
  driving `coef`/`confint`/`vcov`/`summary`.

- `map`: the `Sigma` / `sigma` / `rho` summary at `theta_hat` (a single
  list for one block, a named list of them for several).

- `Sigma`: the estimated covariance (a matrix for one block, a named
  list of matrices for several).

- `theta_hat`, `log_marginal`, `layout`, `n_blocks`, `n_coefs`.

- `converged`: whether the inner Newton solve at `theta_hat` converged.

- `outer_convergence`: `optim`'s code for the maximization over `Sigma`
  (`0` on success). A non-zero value also warns.

- With `marginal = TRUE` and a correction that formed: `cov_marginal`
  (the widened fixed-effect covariance, which `vcov` / `summary` /
  `confint` then report), `cov_conditional` (the `solve(H_beta)` they
  would otherwise have reported, kept so the two are comparable on one
  fit), `H_theta` and `theta_cov` (the outer Hessian at `theta_hat` and
  its inverse, named by `theta_names`), and the `marginal_step` /
  `marginal_richardson` actually used. All absent when the correction
  was not requested or could not be formed, so
  `is.null(fit$cov_marginal)` tests whether the reported intervals are
  marginal. A requested correction that fails warns and leaves the
  conditional covariance in place.

## Details

Blocks, coordinates and the default hyperprior are exactly those of
[`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
– a correlated `(1 + x | g)` term is a full `Sigma = L L'` in
log-Cholesky coordinates, an uncorrelated `(1 + x || g)` term is
diagonal in log-SD coordinates, and a scalar `(1 | g)` term is the
degenerate one-coefficient block. Both functions call the same outer
objective and the same optimizer, so `tulpa_eb()$theta_hat` and
`tulpa_re_cov_nested()$theta_hat` are the same estimate on the same data
– which requires `hyperprior` to default the same way on both: `"flat"`,
the zero function, matching the nested-Laplace convention on every other
scale axis in the engine (icar / rw1 / rw2 / ar1's tau / iid all lack a
hyperprior on their scale too; see
[`vignette("priors")`](https://gillescolling.com/tulpa/articles/priors.md)).
Set `hyperprior = "pc_lkj"` for the weakly-informative PC + LKJ prior
instead (see
[`re_cov_pc_lkj_prior()`](https://gillescolling.com/tulpa/reference/re_cov_pc_lkj_prior.md))
– the regularizer that, at small G, keeps a block off the `sigma = 0`
boundary this maximizer would otherwise reach.

The reported fixed-effect covariance is the conditional one at
`theta_hat` (`solve(H_beta)`). It does not include the hyperparameter
uncertainty that
[`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
integrates over, so EB intervals are narrower – increasingly so as the
number of groups falls. Use the nested integrator when the variance
components themselves, or calibrated fixed-effect intervals, are the
target; use EB when the point estimate is, or as a fast starting fit.

## References

Casella (1985). An introduction to empirical Bayes data analysis. *The
American Statistician* 39(2):83-87. Rue, Martino & Chopin (2009).
Approximate Bayesian inference for latent Gaussian models by using
integrated nested Laplace approximations. *JRSS-B* 71(2):319-392.

## See also

[`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
to integrate over `Sigma` rather than fix it;
[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
for the inner solve.

## Examples

``` r
# \donttest{
set.seed(1)
G <- 30L; per <- 10L; n <- G * per
grp <- rep(seq_len(G), each = per); x <- rnorm(n)
b <- rnorm(G, 0, 0.8)
y <- rpois(n, exp(0.3 + 0.5 * x + b[grp]))
re_term <- list(idx = grp, n_groups = G, n_coefs = 1L)
fit <- tulpa_eb(y, NULL, cbind(1, x), re_term, family = "poisson")
fit$map$sigma          # empirical-Bayes RE standard deviation
#> [1] 0.7681137
# }
```
