#' Fit a Spatial Model using SPDE Laplace Approximation
#'
#' Fits a GLM with a Matérn spatial field via the SPDE approach.
#' Uses CHOLMOD sparse solver with optional nested Laplace for
#' hyperparameter integration.
#'
#' @param y Integer response vector.
#' @param X Design matrix.
#' @param spatial A `tulpa_spatial` object from [spatial_spde()] or
#'   [spatial_spde_custom()].
#' @param family Distribution family: `"binomial"`, `"poisson"`, or `"neg_binomial_2"`.
#' @param n_trials Integer vector of trial sizes (binomial only).
#' @param range Spatial range parameter. If NULL, uses nested Laplace to
#'   integrate over range and sigma.
#' @param sigma Marginal standard deviation. If NULL, uses nested Laplace.
#' @param nested_laplace Logical. If TRUE (default when range/sigma are NULL),
#'   use nested Laplace approximation over hyperparameters.
#' @param method Hyperparameter integration backend when nested Laplace is
#'   active. One of:
#'   \itemize{
#'     \item `"ccd"` (default): central composite design centered on the
#'       joint posterior mode of (range, sigma), oriented by the local
#'       Hessian. Uses 9 design points instead of `n_grid^2`. Folds the
#'       PC priors from `spatial$prior_range` / `spatial$prior_sigma` into
#'       the integrated marginal posterior. Falls back to `"grid"` if the
#'       posterior surface is too flat for a Hessian-based design.
#'     \item `"grid"`: rectangular grid in `log(range) x log(sigma)`
#'       around the prior modes (`n_grid` points per axis).
#'   }
#' @param n_grid Number of grid points per hyperparameter dimension for
#'   `method = "grid"`. Ignored under `method = "ccd"`. Default 5.
#' @param phi Dispersion parameter (negbin only).
#' @param diagnose_k Logical. If TRUE (default), compute the outer
#'   Pareto-\eqn{\hat{k}} accuracy diagnostic (`$pareto_k`) by importance
#'   sampling the joint `(range, sigma)` posterior on the log scale against the
#'   Gaussian proposal that orients the integration. See [tulpa_psis()].
#' @param k_samples Number of importance draws for `diagnose_k`. Default 200,
#'   each one extra batched SPDE marginal evaluation.
#' @param max_iter Maximum Newton iterations. Default 100.
#' @param tol Newton convergence tolerance. Default 1e-6.
#' @param n_threads OpenMP threads. Default 1.
#' @param checkpoint Optional grid-cell checkpoint/resume spec
#'   (gcol33/tulpa#50), `list(path = , resume = )`. Each solved `(range, sigma)`
#'   cell is appended to `path`; a `resume = TRUE` run loads the finished cells
#'   and re-solves only the rest, so a killed or rebooted fit resumes instead of
#'   restarting. `resume = FALSE` starts a fresh file. Default `NULL` (off).
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
#' @export
fit_spde <- function(y, X, spatial,
                     family = "binomial", n_trials = NULL,
                     range = NULL, sigma = NULL,
                     nested_laplace = is.null(range) || is.null(sigma),
                     method = c("ccd", "grid"),
                     n_grid = 5L, phi = 1.0,
                     diagnose_k = TRUE, k_samples = 200L,
                     max_iter = 100L, tol = 1e-6, n_threads = 1L,
                     checkpoint = NULL) {

  if (!inherits(spatial, "tulpa_spatial") || spatial$type != "spde") {
    stop("spatial must be an SPDE tulpa_spatial object", call. = FALSE)
  }

  method <- match.arg(method)

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

  spde_log_marginal <- function(range_vec, sigma_vec) {
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
      checkpoint_path = .ckpt$path
    )
    res
  }

  if (nested_laplace && (is.null(range) || is.null(sigma))) {
    if (method == "grid") {
      return(fit_spde_nested_grid(spde_log_marginal, sp, n_grid, spatial,
                                  diagnose_k = diagnose_k, k_samples = k_samples))
    } else {
      return(fit_spde_nested_ccd(spde_log_marginal,
                                 fit_spde_single = function(r, s) {
                                   laplace_spde_at(
                                     y = y, n_trials = n_trials, X = X,
                                     spatial = sp, family = family, phi = phi,
                                     range = r, sigma = s,
                                     max_iter = max_iter, tol = tol,
                                     n_threads = n_threads
                                   )
                                 },
                                 sp = sp, spatial = spatial,
                                 diagnose_k = diagnose_k, k_samples = k_samples))
    }
  } else {
    # --- Single-point Laplace at fixed hyperparameters ---
    # Delegate to the shared helper used by dispatch_laplace_spatial so the
    # SPDE Laplace call site stays singular.
    result <- laplace_spde_at(
      y = y, n_trials = n_trials, X = X, spatial = sp,
      family = family, phi = phi,
      range = range, sigma = sigma,
      max_iter = max_iter, tol = tol, n_threads = n_threads
    )

    p <- ncol(X)
    list(
      mode = result$mode,
      beta = result$mode[1:p],
      spatial_effects = result$mode[(p + 1):(p + sp$n_mesh)],
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
}
