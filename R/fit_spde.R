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
#' @param n_grid Number of grid points per hyperparameter dimension for
#'   nested Laplace. Default 5.
#' @param phi Dispersion parameter (negbin only).
#' @param max_iter Maximum Newton iterations. Default 100.
#' @param tol Newton convergence tolerance. Default 1e-6.
#' @param n_threads OpenMP threads. Default 1.
#'
#' @return A list with:
#'   \itemize{
#'     \item `mode`: mode of the latent field (beta + mesh node effects)
#'     \item `log_marginal`: log marginal likelihood
#'     \item `converged`: convergence flag
#'     \item `spatial`: the spatial specification (for prediction)
#'     \item `nested`: nested Laplace results (if used)
#'   }
#'
#' @export
fit_spde <- function(y, X, spatial,
                     family = "binomial", n_trials = NULL,
                     range = NULL, sigma = NULL,
                     nested_laplace = is.null(range) || is.null(sigma),
                     n_grid = 5L, phi = 1.0,
                     max_iter = 100L, tol = 1e-6, n_threads = 1L) {

  if (!inherits(spatial, "tulpa_spatial") || spatial$type != "spde") {
    stop("spatial must be an SPDE tulpa_spatial object", call. = FALSE)
  }

  y <- as.numeric(y)
  n_obs <- length(y)
  if (is.null(n_trials)) n_trials <- rep(1L, n_obs)
  n_trials <- as.integer(n_trials)
  X <- as.matrix(X)

  sp <- spatial  # shorthand

  if (nested_laplace && (is.null(range) || is.null(sigma))) {
    # --- Nested Laplace: grid over (range, sigma) ---
    # Build grid around prior modes
    range_mode <- sp$prior_range[1]
    sigma_mode <- sp$prior_sigma[1]

    range_grid <- exp(seq(log(range_mode * 0.3), log(range_mode * 3),
                          length.out = n_grid))
    sigma_grid <- exp(seq(log(sigma_mode * 0.3), log(sigma_mode * 3),
                          length.out = n_grid))

    # Full grid (all combinations)
    grid <- expand.grid(range = range_grid, sigma = sigma_grid)

    result <- cpp_nested_laplace_spde(
      y = y, n_trials = n_trials, X = X,
      A_x = sp$A_x, A_i = sp$A_i, A_p = sp$A_p,
      n_obs = n_obs, n_mesh = sp$n_mesh,
      C0_diag = sp$C0_diag,
      G1_x = sp$G1_x, G1_i = sp$G1_i, G1_p = sp$G1_p,
      range_grid = grid$range, sigma_grid = grid$sigma,
      nu = sp$nu,
      family = family, phi = phi,
      max_iter = max_iter, tol = tol, n_threads = n_threads
    )

    # Find best grid point
    best <- which.max(result$log_marginal)

    # Numerical integration for hyperparameter posterior
    log_max <- max(result$log_marginal)
    weights <- exp(result$log_marginal - log_max)
    weights <- weights / sum(weights)

    list(
      mode = NULL,  # would need to re-fit at best point
      log_marginal = result$log_marginal,
      converged = all(result$n_iter > 0),
      spatial = spatial,
      nested = list(
        range_grid = grid$range,
        sigma_grid = grid$sigma,
        weights = weights,
        n_iter = result$n_iter,
        best_idx = best,
        range_mean = sum(weights * grid$range),
        sigma_mean = sum(weights * grid$sigma),
        range_best = grid$range[best],
        sigma_best = grid$sigma[best]
      )
    )
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
