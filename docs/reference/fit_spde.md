# Fit a Spatial Model using SPDE Laplace Approximation

Fits a GLM with a Matern spatial field via the SPDE approach. Uses
CHOLMOD sparse solver with optional nested Laplace for hyperparameter
integration.

## Usage

``` r
fit_spde(
  y,
  X,
  spatial,
  family = "binomial",
  n_trials = NULL,
  range = NULL,
  sigma = NULL,
  nested_laplace = is.null(range) || is.null(sigma),
  phi = 1,
  offset = NULL,
  re_idx = NULL,
  n_re_groups = 0L,
  sigma_re = 1,
  mode = c("laplace", "nuts"),
  control = list()
)
```

## Arguments

- y:

  Integer response vector.

- X:

  Design matrix.

- spatial:

  A `tulpa_spatial` object from
  [`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md)
  or
  [`spatial_spde_custom()`](https://gillescolling.com/tulpa/reference/spatial_spde_custom.md).

- family:

  Distribution family: `"binomial"`, `"poisson"`, `"neg_binomial_2"`, or
  `"gaussian"` (continuous-field geostatistics, with `phi` the
  observation-noise standard deviation).

- n_trials:

  Integer vector of trial sizes (binomial only).

- range:

  Spatial range parameter. If NULL, uses nested Laplace to integrate
  over range and sigma.

- sigma:

  Marginal standard deviation. If NULL, uses nested Laplace.

- nested_laplace:

  Logical. If TRUE (default when range/sigma are NULL), use nested
  Laplace approximation over hyperparameters.

- phi:

  Dispersion passed to the family, held fixed. One convention at every
  door: for `gaussian` / `lognormal` this is the residual VARIANCE (the
  SD is `sqrt(phi)`), for `neg_binomial_2` the size, `gamma` the shape,
  `beta` the precision, `t` the scale; `binomial` and `poisson` ignore
  it. The compiled kernels parameterize the two variance families by the
  residual SD and are handed `sqrt(phi)` at the boundary.

- offset:

  Optional fixed additive term on the linear predictor
  (`eta = offset + X beta + A w`), length `length(y)`; `NULL` -\> no
  offset.

- re_idx, n_re_groups, sigma_re:

  Optional single iid random-intercept `(1 | g)` term alongside the
  Matern field: `re_idx` is a length-`length(y)` 1-based group index,
  `n_re_groups` the number of groups, and `sigma_re` the (conditioned)
  random-effect SD. The field and the RE block are Laplace- marginalised
  jointly. `n_re_groups = 0` (default) is no RE term. Not supported for
  a fractional-nu field.

- mode:

  Inference method (the method is an argument, not a parallel verb):
  `"laplace"` (default) is the nested-Laplace integration over
  `(range, sigma)` documented here; `"nuts"` delegates to
  [`tulpa_nuts_spde()`](https://gillescolling.com/tulpa/reference/tulpa_nuts_spde.md)
  for exact HMC over the field (and, unless both `range` and `sigma` are
  fixed, the Matern hyperparameters). `mode = "nuts"` does not support
  an `offset` or a random-effect term, and its sampler knobs pass via
  `control` (see
  [`tulpa_nuts_spde()`](https://gillescolling.com/tulpa/reference/tulpa_nuts_spde.md));
  it returns that sampler's draws object.

- control:

  A named list of numerical / tuning knobs (statistical arguments stay
  in the signature above). Recognized entries:

  - `method`: hyperparameter integration backend when nested Laplace is
    active. `"ccd"` (default) uses a central composite design centered
    on the joint posterior mode of `(range, sigma)`, oriented by the
    local Hessian (9 design points instead of `n_grid^2`), folding the
    PC priors from `spatial$prior_range` / `spatial$prior_sigma` into
    the integrated marginal and falling back to `"grid"` if the surface
    is too flat for a Hessian-based design. `"grid"` uses a rectangular
    grid in `log(range) x log(sigma)` around the prior modes.

  - `n_grid`: grid points per hyperparameter dimension for
    `method = "grid"` (ignored under `"ccd"`). Default 5.

  - `diagnose_k`: if TRUE (default), compute the outer
    Pareto-\\\hat{k}\\ accuracy diagnostic (`$pareto_k`) by importance
    sampling the joint `(range, sigma)` posterior on the log scale
    against the Gaussian proposal that orients the integration. See
    [`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md).

  - `k_samples`: importance draws for `diagnose_k`. Default 500, each
    one extra batched SPDE marginal evaluation. It is a precision knob:
    the GPD tail size is held at the fraction the default budget
    implies, so a larger budget sharpens the same k-hat rather than
    moving it to a deeper quantile of the weight distribution
    (gcol33/tulpa#631).

  - `k_tail_points`: expert override for that tail size, in upper-tail
    order statistics. Silently capped at 20% of the draws, beyond which
    body ratios enter the tail and bias the shape.

  - `mode_find`: tuning for the outer `(range, sigma)` mode-find under
    `method = "ccd"`, as `list(factr =, ndeps =, maxit =)`; supply any
    subset. `ndeps` is the central-difference step for
    [`optim()`](https://rdrr.io/r/stats/optim.html)'s numerical gradient
    on the log scale (default 1e-2): it must clear the inner solver's
    convergence tolerance, and a step wide enough that its truncation
    error exceeds the reduction the line search chases near a flat
    optimum leaves L-BFGS-B aborting at the mode it just reached, in
    which case the CCD design declines to the rectangular grid. `factr`
    is the relative-reduction stop in units of `.Machine$double.eps`
    (default 1e5); `maxit` the iteration cap (default 300).

  - `max_iter`: maximum Newton iterations. Default 100.

  - `tol`: Newton convergence tolerance. Default 1e-6.

  - `n_threads`: OpenMP threads. Default 1.

  - `checkpoint`: grid-cell checkpoint/resume spec
    `list(path =, resume =)`. Each solved `(range, sigma)` cell is
    appended to `path`; a `resume = TRUE` run loads the finished cells
    and re-solves only the rest, so a killed or rebooted fit resumes
    instead of restarting. `resume = FALSE` starts a fresh file. Default
    `NULL` (off).

## Value

A list with:

- `mode`: mode of the latent field (beta + mesh node effects)

- `log_marginal`: log marginal likelihood

- `converged`: convergence flag

- `spatial`: the spatial specification (for prediction)

- `pareto_k`, `pareto_k_is_ess`: outer Pareto-\\\hat{k}\\ and its
  importance-sampling ESS (`NA` when `diagnose_k = FALSE`)

- `pareto_k_proposal_source`, `pareto_k_first_pass`: which proposal
  family the reported k-hat came from (several candidates are scored and
  the best kept), and the k-hat of the first pass – the proposal exactly
  as placed, before refinement. A large gap says the placement is poor
  even where the verdict is fine.

- `nested`: nested Laplace results (if used)

## References

Lindgren, Rue & Lindstrom (2011). An explicit link between Gaussian
fields and Gaussian Markov random fields: the stochastic partial
differential equation approach. *JRSS-B* 73(4):423-498. Rue, Martino &
Chopin (2009). Approximate Bayesian inference for latent Gaussian models
by using integrated nested Laplace approximations. *JRSS-B*
71(2):319-392.

## Examples

``` r
# \donttest{
if (requireNamespace("fmesher", quietly = TRUE)) {
  set.seed(1)
  n <- 200L
  coords <- cbind(runif(n), runif(n))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.15, 0.4), cutoff = 0.05)
  fem  <- fmesher::fm_fem(mesh)
  A    <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  spec <- spatial_spde_custom(C = fem$c0, G = fem$g1, A = A, nu = 1,
                              prior_range = c(0.3, 0.5), prior_sigma = c(0.6, 0.05))
  w <- as.numeric(rnorm(spec$n_mesh, 0, 0.6)); w <- w - mean(w)
  x <- rnorm(n)
  y <- rpois(n, exp(2.0 + 0.5 * x + as.numeric(spec$A %*% w)))
  fit <- fit_spde(y = y, X = cbind(1, x), spatial = spec, family = "poisson")
  fit$nested$range_mean
}
# }
```
