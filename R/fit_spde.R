#' Fit a spatial model using SPDE Laplace approximation
#'
#' Fits a GLM with a Matern spatial field via the SPDE approach.
#' Uses CHOLMOD sparse solver with optional nested Laplace for
#' hyperparameter integration.
#'
#' @param y Integer response vector.
#' @param X Design matrix.
#' @param spatial A `tulpa_spatial` object from [spatial_spde()] or
#'   [spatial_spde_custom()].
#' @param family Distribution family: `"binomial"`, `"poisson"`,
#'   `"neg_binomial_2"`, or `"gaussian"` (continuous-field geostatistics, with
#'   `phi` the observation-noise standard deviation).
#' @param n_trials Integer vector of trial sizes (binomial only).
#' @param range Spatial range parameter. If NULL, uses nested Laplace to
#'   integrate over range and sigma.
#' @param sigma Marginal standard deviation. If NULL, uses nested Laplace.
#' @param nested_laplace Logical. If TRUE (default when range/sigma are NULL),
#'   use nested Laplace approximation over hyperparameters.
#' @template phi
#' @param offset Optional fixed additive term on the linear predictor
#'   (`eta = offset + X beta + A w`), length `length(y)`; `NULL` -> no offset.
#' @param beta_prior Optional Gaussian prior on the fixed effects, as
#'   `list(mean, sd)` with each entry a scalar or a length-`ncol(X)` vector
#'   (`sd = Inf` leaves a coefficient unpenalized). `NULL` (default) is the
#'   engine default `beta ~ N(0, 2.5^2)`. Every inner solve carries it, so the
#'   nested marginal over `(range, sigma)` is the marginal under this prior.
#'   Under `mode = "nuts"` it must be mean-zero with one scalar `sd`.
#' @param re_idx,n_re_groups,sigma_re Optional single iid random-intercept
#'   `(1 | g)` term alongside the Matern field: `re_idx` is a length-`length(y)`
#'   1-based group index, `n_re_groups` the number of groups, and `sigma_re` the
#'   (conditioned) random-effect SD. The field and the RE block are Laplace-
#'   marginalised jointly. `n_re_groups = 0` (default) is no RE term. Not
#'   supported for a fractional-nu field.
#' @param mode Inference method (the method is an argument, not a parallel
#'   verb): `"laplace"` (default) is the nested-Laplace integration over
#'   `(range, sigma)` documented here; `"nuts"` delegates to [tulpa_nuts_spde()]
#'   for exact HMC over the field (and, unless both `range` and `sigma` are
#'   fixed, the Matern hyperparameters). `mode = "nuts"` does not support an
#'   `offset` or a random-effect term, and its sampler knobs pass via `control`
#'   (see [tulpa_nuts_spde()]); it returns that sampler's draws object.
#' @param hyperprior `"proper"` (default) or `"flat"`, the prior the nested
#'   integration puts on `(range, sigma)`. `"proper"` carries the spec's PC
#'   range and PC sigma priors. `"flat"` keeps only the anchors the spec was
#'   given explicitly (`prior_range` / `prior_sigma` passed to
#'   [spatial_spde()]); an axis whose anchor was defaulted carries no density,
#'   is named in `log_hyperprior_declined` as `"flat_hyperprior"`, and the
#'   evidence declines. Not read by `mode = "nuts"`, where `"flat"` errors.
#' @param control A named list of numerical / tuning knobs (statistical
#'   arguments stay in the signature above). When `(range, sigma)` is
#'   integrated, an integer-nu field is fitted as an `spde` block of
#'   [tulpa_nested_laplace_joint()] and `control` is that door's control: its
#'   outer grid is placed at the outer mode of `(log range, log sigma)`, refined,
#'   screened and scored by the outer Pareto-\eqn{\hat{k}} as on every joint
#'   fit (`diagnose_k`, `k_samples`, `k_tail_points`, `prune`, `checkpoint`,
#'   `progress`, ...). The inner-solve knobs `max_iter` (default 100), `tol`
#'   (default 1e-6) and `n_threads` (default 1) also apply at fixed
#'   hyperparameters. A fractional-nu field integrates through
#'   [tulpa_hyper_grid()] on a mode-centred CCD, or on the tensor of its axes
#'   with `integration = "grid"`.
#'
#' @return A list with:
#'   \itemize{
#'     \item `mode`: mode of the latent field (beta + mesh node effects)
#'     \item `log_marginal`: log marginal likelihood
#'     \item `converged`: convergence flag
#'     \item `spatial`: the spatial specification (for prediction)
#'     \item `pareto_k`, `pareto_k_is_ess`: outer Pareto-\eqn{\hat{k}} and its
#'       importance-sampling ESS (`NA` when `diagnose_k = FALSE`)
#'     \item `pareto_k_proposal_source`, `pareto_k_first_pass`: which proposal
#'       family the reported k-hat came from (several candidates are scored and
#'       the best kept), and the k-hat of the first pass -- the proposal exactly
#'       as placed, before refinement. A large gap says the placement is poor
#'       even where the verdict is fine.
#'     \item `nested`: the outer-grid fit (if used), with the `(range, sigma)`
#'       summaries `range_mean`, `sigma_mean` and the cell of highest posterior
#'       density (`range_best`, `sigma_best`), whose inner mode is `mode`
#'   }
#'
#' @references
#' Lindgren, Rue & Lindstrom (2011). An explicit link between Gaussian fields
#' and Gaussian Markov random fields: the stochastic partial differential
#' equation approach. \emph{JRSS-B} 73(4):423-498.
#' Rue, Martino & Chopin (2009). Approximate Bayesian inference for latent
#' Gaussian models by using integrated nested Laplace approximations.
#' \emph{JRSS-B} 71(2):319-392.
#' @examples
#' \donttest{
#' if (requireNamespace("fmesher", quietly = TRUE)) {
#'   set.seed(1)
#'   n <- 200L
#'   coords <- cbind(runif(n), runif(n))
#'   mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.15, 0.4), cutoff = 0.05)
#'   fem  <- fmesher::fm_fem(mesh)
#'   A    <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
#'   spec <- spatial_spde_custom(C = fem$c0, G = fem$g1, A = A, nu = 1,
#'                               prior_range = c(0.3, 0.5), prior_sigma = c(0.6, 0.05))
#'   w <- as.numeric(rnorm(spec$n_mesh, 0, 0.6)); w <- w - mean(w)
#'   x <- rnorm(n)
#'   y <- rpois(n, exp(2.0 + 0.5 * x + as.numeric(spec$A %*% w)))
#'   fit <- fit_spde(y = y, X = cbind(1, x), spatial = spec, family = "poisson")
#'   fit$nested$range_mean
#' }
#' }
#' @export
fit_spde <- function(y, X, spatial,
                     family = "binomial", n_trials = NULL,
                     range = NULL, sigma = NULL,
                     nested_laplace = is.null(range) || is.null(sigma),
                     phi = 1.0, offset = NULL, beta_prior = NULL,
                     re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                     mode = c("laplace", "nuts"),
                     hyperprior = c("proper", "flat"),
                     control = list()) {

  mode <- match.arg(mode)
  family <- .canonical_family(family)
  .family_or_stop(family)
  hyperprior <- .hp_choice(match.arg(hyperprior))
  if (mode == "nuts" && identical(hyperprior, "flat")) {
    stop("`hyperprior = \"flat\"` is not read by mode = 'nuts', which samples ",
         "(range, sigma) under the spec's PC priors.", call. = FALSE)
  }
  if (!inherits(spatial, "tulpa_spatial") || spatial$type != "spde") {
    stop("spatial must be an SPDE tulpa_spatial object", call. = FALSE)
  }
  .check_spde_rows(spatial, length(y), "fit_spde()")
  if (mode == "nuts") {
    # The inference method is an argument, not a parallel verb: delegate to the
    # NUTS engine (tulpa_nuts_spde). The Laplace-path-only structure (an offset,
    # a random-effect block) is unsupported there.
    if (!is.null(offset)) {
      stop("`offset` is not supported on mode = 'nuts' (Laplace-path only).",
           call. = FALSE)
    }
    if (!is.null(re_idx) && n_re_groups > 0L) {
      stop("A random-effect term is not supported on mode = 'nuts' for an SPDE ",
           "field; use mode = 'laplace'.", call. = FALSE)
    }
    return(tulpa_nuts_spde(
      y = as.numeric(y), X = as.matrix(X), spatial = spatial, family = family,
      n_trials = n_trials,
      joint = is.null(range) || is.null(sigma),
      range = range, sigma = sigma,
      beta_prior = beta_prior %||% .tulpa_default_beta_prior("spde_nuts"),
      control = control))
  }
  tulpa_check_control(control, .CONTROL_KEYS$spde, "fit_spde")
  .device_session <- .device_start(control)
  on.exit(cpp_device_end(.device_session), add = TRUE)

  # Perf/numerical knobs live in `control = list()` (matching tulpa() /
  # tulpa_nested_laplace()); the signature carries only statistical arguments.
  # The nested integration is the joint door's, so `control` is its control.
  max_iter   <- as.integer(control$max_iter %||% 100L)
  tol        <- control$tol %||% 1e-6
  n_threads  <- as.integer(control$n_threads %||% 1L)

  if (!is.null(offset)) {
    offset <- as.numeric(offset)
    if (length(offset) != length(y)) {
      stop(sprintf("length(offset) (%d) must equal length(y) (%d).",
                   length(offset), length(y)), call. = FALSE)
    }
  }

  y <- as.numeric(y)
  n_obs <- length(y)
  X <- as.matrix(X)
  n_trials <- .validate_glm_design(y, X, n_trials, "fit_spde()",
                             missing_response = TRUE)$n_trials
  # A stated prior equal to the engine default is the kernels' own default.
  bp <- .normalize_beta_prior(beta_prior, ncol(X))
  if (!is.null(bp) && all(bp$mean == 0) && all(bp$sd == .tulpa_prior_sd())) {
    bp <- NULL
  }

  sp <- spatial  # shorthand

  # v10 nested-Laplace ABI requires re_idx / n_re_groups / sigma_re even when
  # there is no formula-side RE term. A single iid random-intercept `(1 | g)`
  # can ride alongside the Matern field, conditioned on `sigma_re` (like the
  # areal spatial Laplace path); the whole latent block -- field + RE -- is
  # Laplace-marginalised jointly inside the kernel. When there is no RE term
  # these pin to the no-RE defaults so every nested-Laplace call is consistent.
  has_re <- !is.null(re_idx) && n_re_groups > 0L
  if (has_re) {
    re_idx <- as.integer(re_idx)
    if (length(re_idx) != n_obs) {
      stop(sprintf("length(re_idx) (%d) must equal length(y) (%d).",
                   length(re_idx), n_obs), call. = FALSE)
    }
    if (anyNA(re_idx) || min(re_idx) < 1L || max(re_idx) > n_re_groups) {
      stop(sprintf(paste0("`re_idx` must be 1-based integers in ",
                          "[1, n_re_groups = %d]; got range [%d, %d]."),
                   n_re_groups, min(re_idx), max(re_idx)), call. = FALSE)
    }
    if (!is.numeric(sigma_re) || length(sigma_re) != 1L ||
        !is.finite(sigma_re) || sigma_re <= 0) {
      stop("`sigma_re` must be a positive scalar.", call. = FALSE)
    }
  }
  no_re_idx       <- if (has_re) re_idx      else rep(0L, n_obs)
  no_re_n_groups  <- if (has_re) as.integer(n_re_groups) else 0L
  no_re_sigma     <- if (has_re) as.numeric(sigma_re)    else 1.0
  # The inputs of an inner solve the fit does not otherwise carry, kept on the
  # fit so a cell can be solved again at prediction (`.field_cell_solve()`).
  solve <- list(re_idx = no_re_idx, n_re_groups = no_re_n_groups,
                sigma_re = no_re_sigma, max_iter = max_iter, tol = tol,
                beta_prior = bp)

  is_frac <- .spde_nu_is_fractional(sp$nu)
  if (is_frac && has_re) {
    stop("A random-effect term alongside a fractional-nu SPDE field is not ",
         "supported yet; use an integer nu, or drop the (1 | g) term.",
         call. = FALSE)
  }
  single_at <- function(r, s) {
    laplace_spde_at(
      y = y, n_trials = n_trials, X = X, spatial = sp,
      family = family, phi = phi, range = r, sigma = s,
      max_iter = max_iter, tol = tol, n_threads = n_threads,
      offset = offset, beta_prior = bp,
      re_idx = no_re_idx, n_re_groups = no_re_n_groups, sigma_re = no_re_sigma)
  }

  fit <- if (nested_laplace && (is.null(range) || is.null(sigma))) {
    nested <- if (is_frac) {
      .spde_nested_fractional(sp, y = y, X = X, family = family, phi = phi,
                              n_trials = n_trials, offset = offset,
                              beta_prior = bp, hyperprior = hyperprior,
                              control = control)
    } else {
      .spde_nested_joint(sp, y = y, X = X, family = family, phi = phi,
                         n_trials = n_trials, offset = offset, beta_prior = bp,
                         re = if (has_re) solve[c("re_idx", "n_re_groups",
                                                  "sigma_re")],
                         hyperprior = hyperprior, control = control)
    }
    .spde_fit_from_nested(nested, sp, spatial, n_fixed = ncol(X),
                          refit = single_at)
  } else {
    # --- Single-point Laplace at fixed hyperparameters ---
    # Delegate to the shared helper used by dispatch_laplace_spatial so the
    # SPDE Laplace call site stays singular.
    result <- single_at(range, sigma)
    list(
      mode = result$mode,
      beta = result$beta,
      spatial_effects = result$spatial_effects,
      log_det_Q = result$log_det_Q,
      log_marginal = result$log_marginal,
      n_iter = result$n_iter,
      converged = result$converged,
      spatial = spatial,
      range = result$range,
      sigma = result$sigma,
      nested = NULL
    )
  }

  # Prediction support (predict.tulpa_fit se.fit): the family and the
  # dispersion, in the one engine convention (gaussian / lognormal: the
  # residual variance), the same `$phi` a front-door tulpa() fit carries.
  fit$family <- fit$family %||% family
  fit$phi    <- phi
  # The record a cell of the fit is solved again from when its latent
  # posterior is drawn (`.field_cell_solve()`): a nested fit draws from the
  # precisions its outer grid kept, a fit at fixed hyperparameters solves its
  # one cell again. The fractional-nu operator is assembled per cell in R and
  # keeps no precision, so it has none.
  if (!is_frac) {
    fit$field_solve <- if (is.list(fit$nested$Q_csc_p_per_grid)) {
      list(door = "stored")
    } else {
      .nl_field_solve_record("spde", solve)
    }
  }

  # Fixed-effect marginal curvature so the generic accessors (coef / summary /
  # confint / vcov / predict) work on an SPDE fit: the field-marginal Schur
  # complement (.marginal_H_beta_spde) at the hyperparameters the mode was
  # solved at, the same block mode = "laplace" attaches.
  fit$H_beta <- fit$H_beta %||% tryCatch(
    .marginal_H_beta_spde(
      mode = fit$mode, X = X, spatial = sp,
      family = family, phi = phi,
      n_trials = n_trials, offset = offset,
      range_val = fit$range, sigma_val = fit$sigma,
      re_idx = no_re_idx, n_re_groups = no_re_n_groups, sigma_re = no_re_sigma,
      y = y
    ),
    error = function(e) NULL
  )
  fit$beta_prior <- .normalize_beta_prior(beta_prior, ncol(X))
  .finalize_fit(fit, backend = "spde",
                n_fixed = ncol(X), fixed_names = colnames(X),
                data = list(y = y, n_trials = n_trials, model_matrix = X,
                           family = family, offset = offset, phi = phi))
}


# The SPDE projector A maps observations to mesh nodes, one row per
# observation, and every SPDE kernel indexes it by the observation index with no
# bound of its own. A longer response than A has rows used to be accepted, the
# extra rows reaching the likelihood with no field term; a shorter one reached
# the kernel's CSC check as "A: row index 10 at slot 11 is outside [0, 10)"
# (gcol33/tulpa#908). One check, at every R door into an SPDE fit.
#' @keywords internal
.check_spde_rows <- function(spatial, n_obs, where) {
  n_a <- tryCatch(nrow(spatial$A), error = function(e) NULL)
  if (is.null(n_a) || n_a != n_obs) {
    stop(where, ": the SPDE projector matrix A has ", n_a %||% "?",
         " row(s) but the response has ", n_obs, " observation(s). Build the ",
         "SPDE spec from the same data (spatial_spde(~ lon + lat, data = ",
         "<data>)), one row per observation.", call. = FALSE)
  }
  invisible(TRUE)
}


# One call of the integer-nu SPDE nested-Laplace kernel at the cells
# (`range_vec`, `sigma_vec`). `solve` holds the inputs of an inner solve a fit
# does not otherwise carry (the random-intercept block, the Newton tuning and
# the fixed-effect prior), so the fit's own grid and a later solve of some of
# its cells (`.field_cell_solve()`) run one call; `extra` carries the per-call
# kernel requests (screening, checkpoint, corrections).
#' @keywords internal
.spde_grid_solve <- function(sp, y, n_trials, X, family, phi, offset, solve,
                             range_vec, sigma_vec, n_threads = 1L,
                             store_Q = FALSE, extra = list()) {
  do.call(cpp_nested_laplace_spde, c(list(
    y = y, n_trials = n_trials, X = X,
    re_idx = solve$re_idx, n_re_groups = solve$n_re_groups,
    sigma_re = solve$sigma_re,
    A_x = sp$A_x, A_i = sp$A_i, A_p = sp$A_p,
    n_obs = length(y), n_mesh = sp$n_mesh,
    C0_diag = sp$C0_diag,
    G1_x = sp$G1_x, G1_i = sp$G1_i, G1_p = sp$G1_p,
    range_grid = range_vec, sigma_grid = sigma_vec,
    nu = sp$nu,
    family = family, phi = .phi_to_kernel(family, phi),
    max_iter = solve$max_iter, tol = solve$tol, n_threads = n_threads,
    store_Q = store_Q, offset_nullable = offset,
    beta_prior = solve$beta_prior), extra))
}
