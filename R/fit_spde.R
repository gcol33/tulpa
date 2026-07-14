#' Fit a Spatial Model using SPDE Laplace Approximation
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
#' @param phi Dispersion parameter (negbin only).
#' @param offset Optional fixed additive term on the linear predictor
#'   (`eta = offset + X beta + A w`), length `length(y)`; `NULL` -> no offset.
#' @param control A named list of numerical / tuning knobs (statistical
#'   arguments stay in the signature above). Recognized entries:
#'   \itemize{
#'     \item `method`: hyperparameter integration backend when nested Laplace is
#'       active. `"ccd"` (default) uses a central composite design centered on
#'       the joint posterior mode of `(range, sigma)`, oriented by the local
#'       Hessian (9 design points instead of `n_grid^2`), folding the PC priors
#'       from `spatial$prior_range` / `spatial$prior_sigma` into the integrated
#'       marginal and falling back to `"grid"` if the surface is too flat for a
#'       Hessian-based design. `"grid"` uses a rectangular grid in
#'       `log(range) x log(sigma)` around the prior modes.
#'     \item `n_grid`: grid points per hyperparameter dimension for
#'       `method = "grid"` (ignored under `"ccd"`). Default 5.
#'     \item `diagnose_k`: if TRUE (default), compute the outer
#'       Pareto-\eqn{\hat{k}} accuracy diagnostic (`$pareto_k`) by importance
#'       sampling the joint `(range, sigma)` posterior on the log scale against
#'       the Gaussian proposal that orients the integration. See [tulpa_psis()].
#'     \item `k_samples`: importance draws for `diagnose_k`. Default 200, each
#'       one extra batched SPDE marginal evaluation.
#'     \item `max_iter`: maximum Newton iterations. Default 100.
#'     \item `tol`: Newton convergence tolerance. Default 1e-6.
#'     \item `n_threads`: OpenMP threads. Default 1.
#'     \item `checkpoint`: grid-cell checkpoint/resume spec
#'       `list(path =, resume =)`. Each solved `(range, sigma)` cell is appended
#'       to `path`; a `resume = TRUE` run loads the finished cells and re-solves
#'       only the rest, so a killed or rebooted fit resumes instead of
#'       restarting. `resume = FALSE` starts a fresh file. Default `NULL` (off).
#'   }
#'
#' @return A list with:
#'   \itemize{
#'     \item `mode`: mode of the latent field (beta + mesh node effects)
#'     \item `log_marginal`: log marginal likelihood
#'     \item `converged`: convergence flag
#'     \item `spatial`: the spatial specification (for prediction)
#'     \item `pareto_k`, `pareto_k_is_ess`: outer Pareto-\eqn{\hat{k}} and its
#'       importance-sampling ESS (`NA` when `diagnose_k = FALSE`)
#'     \item `nested`: nested Laplace results (if used)
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
                     phi = 1.0, offset = NULL,
                     control = list()) {

  .check_control(control, .CONTROL_KEYS$spde, "fit_spde")
  if (!inherits(spatial, "tulpa_spatial") || spatial$type != "spde") {
    stop("spatial must be an SPDE tulpa_spatial object", call. = FALSE)
  }

  # Perf/numerical knobs live in `control = list()` (matching tulpa() /
  # tulpa_nested_laplace()); the signature carries only statistical arguments.
  method     <- match.arg(control$method %||% "ccd", c("ccd", "grid"))
  n_grid     <- as.integer(control$n_grid %||% 5L)
  diagnose_k <- isTRUE(control$diagnose_k %||% TRUE)
  k_samples  <- as.integer(control$k_samples %||% 200L)
  max_iter   <- as.integer(control$max_iter %||% 100L)
  tol        <- control$tol %||% 1e-6
  n_threads  <- as.integer(control$n_threads %||% 1L)
  checkpoint <- control$checkpoint

  if (!is.null(offset)) {
    offset <- as.numeric(offset)
    if (length(offset) != length(y)) {
      stop(sprintf("length(offset) (%d) must equal length(y) (%d).",
                   length(offset), length(y)), call. = FALSE)
    }
  }

  # Grid-cell checkpoint/resume (gcol33/tulpa#50). `checkpoint = list(path =,
  # resume =)` appends each solved (range, sigma) cell to `path`; a resume loads
  # finished cells and re-solves only the rest. The outer integrator
  # (CCD / grid) calls spde_log_marginal() many times across mode search and the
  # final grid -- every cell is keyed by its (range, sigma) coordinate, so all
  # those calls share one file and any previously solved coordinate is reused.
  .ckpt <- .nl_checkpoint_args(list(checkpoint = checkpoint))
  if (nzchar(.ckpt$path) && !isTRUE(.ckpt$resume) && file.exists(.ckpt$path)) {
    file.remove(.ckpt$path)
  }

  y <- as.numeric(y)
  n_obs <- length(y)
  if (is.null(n_trials)) n_trials <- rep(1L, n_obs)
  n_trials <- as.integer(n_trials)
  X <- as.matrix(X)

  sp <- spatial  # shorthand

  # v10 nested-Laplace ABI requires re_idx / n_re_groups / sigma_re even when
  # there is no formula-side RE term. Pin them here so every nested-Laplace
  # call inside this function is consistent.
  no_re_idx       <- rep(0L, n_obs)
  no_re_n_groups  <- 0L
  no_re_sigma     <- 1.0

  # Fractional nu integrates the operator-based rational SPDE (gcol33/tulpa#71):
  # each (range, sigma) cell assembles its own (Q, A_eff) via the validated R
  # oracle and is solved by the precomputed C++ fit, with 0.5 log|Q| folded into
  # the per-cell marginal (done inside .spde_laplace_fractional_at). The outer
  # CCD / grid integrators are reused unchanged -- they read only $log_marginal
  # and $n_iter. The cpp grid-cell checkpoint is integer-path only.
  is_frac   <- .spde_nu_is_fractional(sp$nu)
  order_rat <- sp$rational_order %||% 2L
  if (is_frac && nzchar(.ckpt$path)) {
    warning("Grid-cell checkpointing is not supported for fractional nu; ",
            "the fractional SPDE grid runs without it.", call. = FALSE)
  }

  spde_log_marginal <- function(range_vec, sigma_vec) {
    if (is_frac) {
      K  <- length(range_vec)
      lm <- numeric(K); ni <- integer(K)
      for (k in seq_len(K)) {
        f <- .spde_nested_logmarginal_at(
          spatial = sp, range = range_vec[k], sigma = sigma_vec[k],
          y = y, X = X, family = family, phi = phi, n_trials = n_trials,
          order = order_rat, max_iter = max_iter, tol = tol,
          n_threads = n_threads, offset = offset
        )
        lm[k] <- f$log_marginal
        ni[k] <- f$n_iter
      }
      return(list(log_marginal = lm, n_iter = ni))
    }
    res <- cpp_nested_laplace_spde(
      y = y, n_trials = n_trials, X = X,
      re_idx = no_re_idx, n_re_groups = no_re_n_groups,
      sigma_re = no_re_sigma,
      A_x = sp$A_x, A_i = sp$A_i, A_p = sp$A_p,
      n_obs = n_obs, n_mesh = sp$n_mesh,
      C0_diag = sp$C0_diag,
      G1_x = sp$G1_x, G1_i = sp$G1_i, G1_p = sp$G1_p,
      range_grid = range_vec, sigma_grid = sigma_vec,
      nu = sp$nu,
      family = family, phi = phi,
      max_iter = max_iter, tol = tol, n_threads = n_threads,
      checkpoint_path = .ckpt$path,
      offset_nullable = offset
    )
    res
  }

  fit <- if (nested_laplace && (is.null(range) || is.null(sigma))) {
    if (method == "grid") {
      fit_spde_nested_grid(spde_log_marginal, sp, n_grid, spatial,
                           diagnose_k = diagnose_k, k_samples = k_samples)
    } else {
      fit_spde_nested_ccd(spde_log_marginal,
                                 fit_spde_single = function(r, s) {
                                   laplace_spde_at(
                                     y = y, n_trials = n_trials, X = X,
                                     spatial = sp, family = family, phi = phi,
                                     range = r, sigma = s,
                                     max_iter = max_iter, tol = tol,
                                     n_threads = n_threads, offset = offset
                                   )
                                 },
                                 sp = sp, spatial = spatial,
                          diagnose_k = diagnose_k, k_samples = k_samples)
    }
  } else {
    # --- Single-point Laplace at fixed hyperparameters ---
    # Delegate to the shared helper used by dispatch_laplace_spatial so the
    # SPDE Laplace call site stays singular.
    result <- laplace_spde_at(
      y = y, n_trials = n_trials, X = X, spatial = sp,
      family = family, phi = phi,
      range = range, sigma = sigma,
      max_iter = max_iter, tol = tol, n_threads = n_threads,
      offset = offset
    )

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
  # kernel-convention dispersion (gaussian/lognormal: residual SD -- the
  # front-door tulpa() fit additionally carries $phi as the variance).
  fit$family     <- fit$family %||% family
  fit$phi_kernel <- phi

  # Fixed-effect mode + marginal curvature so the generic accessors (coef /
  # summary / confint / vcov / predict) work on a direct or auto-mode SPDE
  # fit. The nested grid path stores no per-cell modes, so anchor one refit
  # at the posterior-mean hyperparameters; H_beta is the field-marginal Schur
  # complement (.marginal_H_beta_spde), the same block mode = "laplace"
  # attaches.
  if (is.null(fit$mode) && !is.null(fit$nested)) {
    range_hy <- fit$nested$range_mean %||% fit$nested$range_best
    sigma_hy <- fit$nested$sigma_mean %||% fit$nested$sigma_best
    anchor <- laplace_spde_at(
      y = y, n_trials = n_trials, X = X, spatial = sp,
      family = family, phi = phi,
      range = range_hy, sigma = sigma_hy,
      max_iter = max_iter, tol = tol, n_threads = n_threads,
      offset = offset
    )
    fit$mode            <- anchor$mode
    fit$beta            <- fit$beta %||% anchor$beta
    fit$spatial_effects <- fit$spatial_effects %||% anchor$spatial_effects
    fit$range           <- fit$range %||% range_hy
    fit$sigma           <- fit$sigma %||% sigma_hy
  }
  if (!is.null(fit$mode)) {
    # glmm_weights follows the R-registry dispersion convention (variance
    # for gaussian / lognormal); this door's `phi` is the kernel SD.
    phi_w <- if (family %in% c("gaussian", "lognormal")) phi^2 else phi
    range_hy <- fit$range %||% fit$nested$range_mean %||% fit$nested$range_best
    sigma_hy <- fit$sigma %||% fit$nested$sigma_mean %||% fit$nested$sigma_best
    fit$H_beta <- fit$H_beta %||% tryCatch(
      .marginal_H_beta_spde(
        mode = fit$mode, X = X, spatial = sp,
        family = family, phi = phi_w,
        n_trials = n_trials, offset = offset,
        range_val = range_hy, sigma_val = sigma_hy
      ),
      error = function(e) NULL
    )
  }

  .finalize_fit(fit, backend = "spde",
                n_fixed = ncol(X), fixed_names = colnames(X))
}
