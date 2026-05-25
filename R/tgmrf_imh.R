#' IMH-Laplace MCMC over a tgmrf block's hyperparameters
#'
#' @description
#' Composes the Laplace adapter (`tulpa_nested_laplace()`) with the
#' generic [imh_laplace()] independence-Metropolis sampler to obtain
#' asymptotically-correct posterior draws for the tgmrf block's
#' hyperparameter vector `theta`.
#'
#' The pipeline is the canonical "Laplace body + MH bias correction" used
#' throughout tulpa's debias philosophy:
#'   1. Pilot grid fit to get a coarse `log_marginal(theta_k)` landscape.
#'   2. Localise the mode in theta-space (grid argmax) and estimate a
#'      finite-difference Hessian around it.
#'   3. Run IMH-Laplace with proposals
#'      `theta' ~ N(mode, scale^2 * H^{-1})` and exact MH accept/reject
#'      against the target
#'      `log p(theta | y) = log_marginal(theta) + log p(theta)`
#'      (the user prior is already embedded in `log_marginal` via the
#'      `prior` closure attached to the block).
#'   4. Each proposal `theta'` is evaluated by a single-point inner
#'      Laplace solve through `cpp_nested_laplace_multi` -- `n_iter`
#'      inner solves total.
#'
#' Because the body (Laplace approximation of `(beta, z) | theta`) is
#' Gaussian and the bias is corrected by MH, the resulting draws are
#' exact in the limit. For posteriors that are nearly Laplace-Gaussian
#' the acceptance rate is high and the sampler is dramatically cheaper
#' than full NUTS over the joint `(beta, z, theta)`.
#'
#' @param y,n_trials,X Response, trials per obs (1L vector for non-binomial),
#'   and fixed-effects design matrix -- same arguments as
#'   [tulpa_nested_laplace()].
#' @param block A `tgmrf` object.
#' @param family,phi Likelihood family and dispersion.
#' @param re_idx,n_re_groups,sigma_re Optional RE wiring (forwarded to
#'   the inner Laplace).
#' @param n_iter,warmup,thin,scale MCMC arguments forwarded to
#'   [imh_laplace()].
#' @param pilot_axis_points Per-axis grid resolution for the pilot fit
#'   (default 5).
#' @param fd_step Step size for the finite-difference Hessian on
#'   `log_marginal(theta)` (default 0.05; in the block's theta units).
#' @param max_iter,tol Inner-Newton budget per proposal.
#' @param verbose Forwarded to [imh_laplace()].
#' @param n_threads OpenMP threads inside each inner Laplace solve.
#'
#' @return A list with class `c("tulpa_tgmrf_imh", "tulpa_fit")`:
#'   * `draws`: posterior draws of `theta` (rows = post-warmup samples,
#'     cols = `theta_dim`).
#'   * `means`, `sds`: posterior moments.
#'   * `mean_accept`: post-warmup acceptance rate.
#'   * `pilot`: the pilot `tulpa_nested_laplace` object (for diagnostics).
#'   * `mode_theta`, `hessian_theta`: pilot-derived mode + FD Hessian.
#'   * `inference_mode`: `"exact"`.
#'   * `inference_tier`: `1L` (MH-corrected).
#'   * `backend`: `"tgmrf_imh"`.
#'
#' @seealso [tulpa_nested_laplace()] for the grid-only Laplace,
#'   [imh_laplace()] for the generic MH wrapper.
#' @export
tulpa_tgmrf_imh <- function(y, n_trials, X, block,
                            family = "binomial",
                            phi = 1.0,
                            re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                            n_iter = 2000L, warmup = n_iter %/% 2L,
                            thin = 1L, scale = 1.0,
                            pilot_axis_points = 5L,
                            fd_step = 0.05,
                            max_iter = 50L, tol = 1e-7,
                            n_threads = 1L,
                            verbose = FALSE) {

  if (!inherits(block, "tgmrf")) {
    stop("`block` must be a tgmrf object.", call. = FALSE)
  }

  d <- block$theta_dim
  N <- length(y)
  obs_idx <- block$obs_idx %||% seq_len(N)
  if (length(obs_idx) != N) {
    stop("obs_idx length (", length(obs_idx),
         ") does not match N = ", N, ".", call. = FALSE)
  }

  # --- 1. Pilot grid fit ----------------------------------------------------
  pilot_block <- block
  pilot_block$obs_idx <- obs_idx
  # Override the default axis-points count if requested.
  if (!is.null(pilot_axis_points) && pilot_axis_points != 5L) {
    axes <- vector("list", d)
    for (j in seq_len(d)) {
      lo <- if (!is.null(block$bounds)) block$bounds$lower[j] else block$init[j] - 2
      hi <- if (!is.null(block$bounds)) block$bounds$upper[j] else block$init[j] + 2
      axes[[j]] <- seq(lo, hi, length.out = pilot_axis_points)
    }
    names(axes) <- block$theta_names
    pilot_block$theta_grid_built <- as.matrix(do.call(expand.grid, axes))
  }

  pilot <- tulpa_nested_laplace(
    y = y, n_trials = n_trials, X = X,
    prior = pilot_block,
    re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
    family = family, phi = phi,
    control = list(max_iter = max_iter, tol = tol, n_threads = n_threads)
  )

  # Strip block-prefix from joint axis names for the d-dim theta vector.
  grid <- pilot$theta_grid
  if (ncol(grid) != d) {
    stop("Pilot grid has ", ncol(grid), " columns but block has ",
         d, " hyperparameters.", call. = FALSE)
  }
  k_star <- which.max(pilot$log_marginal)
  theta_mode <- as.numeric(grid[k_star, ])
  names(theta_mode) <- block$theta_names

  # --- 2. FD Hessian on log_marginal(theta) ---------------------------------
  # Evaluate log_marginal at theta_mode +/- fd_step * e_j and cross-terms.
  # log_marginal at a single theta is one inner Laplace solve.
  log_marginal_at <- function(theta_vec) {
    blk <- block
    blk$obs_idx <- obs_idx
    blk$theta_grid_built <- matrix(theta_vec, nrow = 1L,
                                   dimnames = list(NULL, block$theta_names))
    out <- tryCatch(
      tulpa_nested_laplace(
        y = y, n_trials = n_trials, X = X,
        prior = blk,
        re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
        family = family, phi = phi,
        control = list(max_iter = max_iter, tol = tol, n_threads = n_threads)
      ),
      error = function(e) NULL
    )
    if (is.null(out)) return(-Inf)
    as.numeric(out$log_marginal[1])
  }

  H_theta <- matrix(0, nrow = d, ncol = d,
                    dimnames = list(block$theta_names, block$theta_names))
  lm_centre <- log_marginal_at(theta_mode)
  if (!is.finite(lm_centre)) {
    stop("Pilot log_marginal at the grid argmax is not finite -- ",
         "tighten `bounds` or check the user `Q()`/`prior()`.", call. = FALSE)
  }

  for (j in seq_len(d)) {
    e_j <- numeric(d); e_j[j] <- fd_step
    lm_p <- log_marginal_at(theta_mode + e_j)
    lm_m <- log_marginal_at(theta_mode - e_j)
    # -log_marginal'' = -(lm_p - 2 lm_centre + lm_m) / h^2
    H_theta[j, j] <- -(lm_p - 2 * lm_centre + lm_m) / fd_step^2
  }
  for (j in seq_len(d - 1L)) {
    for (k in (j + 1L):d) {
      e_j <- numeric(d); e_j[j] <- fd_step
      e_k <- numeric(d); e_k[k] <- fd_step
      lm_pp <- log_marginal_at(theta_mode + e_j + e_k)
      lm_pm <- log_marginal_at(theta_mode + e_j - e_k)
      lm_mp <- log_marginal_at(theta_mode - e_j + e_k)
      lm_mm <- log_marginal_at(theta_mode - e_j - e_k)
      # Mixed second derivative.
      mixed <- (lm_pp - lm_pm - lm_mp + lm_mm) / (4 * fd_step^2)
      H_theta[j, k] <- -mixed
      H_theta[k, j] <- -mixed
    }
  }
  H_theta <- 0.5 * (H_theta + t(H_theta))   # numerically symmetrise

  # If the FD Hessian is not PD, ridge-regularise. Small positive jitter on
  # the diagonal; for tgmrf the typical issue is grid coarseness, not the
  # Hessian itself being non-existent.
  ev <- eigen(H_theta, symmetric = TRUE, only.values = TRUE)$values
  if (min(ev) < 1e-6) {
    H_theta <- H_theta + diag(max(1e-3, 1e-3 - min(ev)), d)
  }

  # --- 3. IMH-Laplace over theta -------------------------------------------
  mh <- imh_laplace(
    log_posterior = log_marginal_at,
    mode    = theta_mode,
    hessian = H_theta,
    n_iter  = n_iter,
    warmup  = warmup,
    thin    = thin,
    scale   = scale,
    verbose = verbose
  )

  draws <- mh$draws
  colnames(draws) <- block$theta_names

  fit <- list(
    draws          = draws,
    means          = colMeans(draws),
    sds            = apply(draws, 2L, stats::sd),
    n_params       = d,
    n_samples      = nrow(draws),
    mean_accept    = mh$mean_accept,
    log_prob       = mh$log_prob,
    pilot          = pilot,
    mode_theta     = theta_mode,
    hessian_theta  = H_theta,
    inference_mode = "exact",
    inference_tier = 1L,
    backend        = "tgmrf_imh"
  )
  class(fit) <- c("tulpa_tgmrf_imh", "tulpa_fit")
  fit
}
