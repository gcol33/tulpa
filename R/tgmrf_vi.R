#' Variational inference over a tgmrf block's hyperparameters
#'
#' @description
#' Single-path Pathfinder VI (Zhang, Carpenter, Gelman, Vehtari 2022)
#' over the user-defined hyperparameter vector `theta` of a [tgmrf()]
#' block. L-BFGS finds the mode of `log_marginal(theta)`, the optimum's
#' inverse Hessian becomes the variational covariance, and the resulting
#' Gaussian is the variational posterior `q(theta)`. ELBO is computed by
#' MC averaging over draws from `q`.
#'
#' This is the Tier-2 (structured) sibling of the tgmrf IMH and NUTS
#' fitters: same Laplace body for `(beta, z) | theta`, but the outer
#' integration over `theta` is the L-BFGS Gaussian fit rather than MH or
#' NUTS. No bias correction -- the Gaussian fit *is* the approximation.
#' Pair with the tgmrf IMH fitter for a Tier-1 upgrade (the VI fit is a
#' perfect IMH proposal).
#'
#' The implementation simply composes:
#'
#' 1. Pilot grid Laplace -> init `theta` (grid argmax).
#' 2. [pathfinder()] on `log_marginal(theta)` with FD-based L-BFGS
#'    (gradients from `stats::optim`'s internal finite differences).
#' 3. Wrap the result with tgmrf-specific metadata + block hyperparameter
#'    names.
#'
#' @inheritParams.tgmrf_fit_imh
#' @param n_draws Number of variational draws to return (default 1000).
#' @param max_lbfgs L-BFGS iteration cap (default 100).
#' @param lbfgs_tol L-BFGS gradient-norm tolerance (default 1e-6).
#'
#' @return A list with class `c("tulpa_tgmrf_vi", "tulpa_fit")`:
#'   * `draws` -- `n_draws x theta_dim` matrix of variational draws.
#'   * `means`, `sds` -- variational moments.
#'   * `mode_theta` -- L-BFGS mode in `theta`.
#'   * `cov` -- variational covariance (`solve(-hessian)`).
#'   * `elbo` -- MC estimate of `E_q[log_marginal(theta) - log q(theta)]`.
#'   * `pilot` -- the pilot [tulpa_nested_laplace()] object.
#'   * `converged` -- logical, L-BFGS convergence flag.
#'   * `inference_mode`, `inference_tier`, `backend` -- `"structured"`,
#'     `2L`, `"tgmrf_vi"`.
#'
#' @references Zhang, Carpenter, Gelman, Vehtari (2022). Pathfinder:
#'   parallel quasi-Newton variational inference. JMLR 23(306):1-49.
#' @seealso the tgmrf IMH (Tier 1 MH) and NUTS (Tier 1 NUTS) fitters, and
#'   [pathfinder()] (underlying VI engine).
#' @noRd
.tgmrf_fit_vi <- function(y, n_trials, X, block,
                           family = "binomial",
                           phi = 1.0,
                           re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                           n_draws = 1000L,
                           max_lbfgs = 100L,
                           lbfgs_tol = 1e-6,
                           pilot_axis_points = 5L,
                           max_iter = 50L, tol = 1e-7,
                           n_threads = 1L,
                           verbose = FALSE) {

  if (!inherits(block, "tgmrf")) {
    stop("`block` must be a tgmrf object.", call. = FALSE)
  }
  if (n_draws < 2L) stop("`n_draws` must be >= 2.", call. = FALSE)

  d <- block$theta_dim
  N <- length(y)
  obs_idx <- block$obs_idx %||% seq_len(N)

  # --- 1. Pilot Laplace for init ------------------------------------------
  pilot_block <- block
  pilot_block$obs_idx <- obs_idx
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
  k_star <- which.max(pilot$log_marginal)
  theta_init <- as.numeric(pilot$theta_grid[k_star, ])
  names(theta_init) <- block$theta_names

  # --- 2. log_marginal(theta) closure -------------------------------------
  # Shared builder (see .tgmrf_make_log_marginal): hard-wall bounds, suppress
  # CHOLMOD non-PD warnings inside the L-BFGS line search, return -Inf on any
  # failure so L-BFGS-B retreats from the infeasible region.
  bounds_lo <- if (!is.null(block$bounds)) block$bounds$lower else NULL
  bounds_hi <- if (!is.null(block$bounds)) block$bounds$upper else NULL
  .lm <- .tgmrf_make_log_marginal(
    y = y, n_trials = n_trials, X = X, block = block, obs_idx = obs_idx,
    re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
    family = family, phi = phi,
    max_iter = max_iter, tol = tol, n_threads = n_threads,
    bounds_lo = bounds_lo, bounds_hi = bounds_hi
  )
  log_marginal_at <- .lm$eval

  # Eager structural check at the feasible pilot mode (the grid argmax, where
  # the inner Laplace already succeeded): a failure here is a bug, not
  # numerical infeasibility -- surface its real message rather than letting
  # pathfinder's finite-init guard resurface it as a cryptic "not finite".
  init_lm <- .lm$raw(theta_init)
  if (!is.finite(init_lm)) {
    stop("Inner nested-Laplace returned a non-finite log-marginal at the ",
         "pilot mode; cannot initialise VI. Tighten `bounds` or check the ",
         "user `Q()` / `prior()`.", call. = FALSE)
  }

  # --- 3. Pathfinder on log_marginal(theta) -------------------------------
  pf <- pathfinder(
    log_posterior      = log_marginal_at,
    init               = theta_init,
    grad_log_posterior = NULL,           # FD via optim
    n_draws            = n_draws,
    max_iter           = max_lbfgs,
    tol                = lbfgs_tol,
    verbose            = verbose
  )

  draws <- pf$draws
  colnames(draws) <- block$theta_names
  mode_theta <- pf$mode
  names(mode_theta) <- block$theta_names

  fit <- list(
    draws          = draws,
    means          = colMeans(draws),
    sds            = apply(draws, 2L, stats::sd),
    n_params       = d,
    n_samples      = nrow(draws),
    mode_theta     = mode_theta,
    cov            = pf$cov,
    hessian        = pf$hessian,
    elbo           = pf$elbo,
    log_prob       = pf$log_prob,
    n_lbfgs_iter   = pf$n_iter,
    converged      = pf$converged,
    pilot          = pilot,
    inference_mode = "structured",
    inference_tier = 2L,
    backend        = "tgmrf_vi"
  )
  .finalize_fit(fit, draws_kind = "iid", extra_class = "tulpa_tgmrf")
}
