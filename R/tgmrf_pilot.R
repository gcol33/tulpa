# Shared pilot-Laplace warm start for the tgmrf hyperparameter adapters
# (tulpa_tgmrf(mode = "imh" / "vi" / "nuts" / "nuts_joint")).
#
# Every adapter starts the same way: resolve the observation -> latent index
# map, build a pilot block (optionally overriding the per-axis grid density),
# run one grid nested-Laplace over it, and read the argmax cell for the theta
# init. Only what happens AFTER that differs (FD Hessian + IMH, Pathfinder
# L-BFGS, marginal NUTS, joint NUTS), so the shared prefix lives here once --
# the same reason `.tgmrf_make_log_marginal` exists next door.


# Resolve `block$obs_idx` against the response length. Every adapter routes
# through this, so a mismatched index map is a clear error on all of them
# rather than a silent mis-index on the ones that skipped the check.
.tgmrf_obs_idx <- function(block, N) {
  obs_idx <- block$obs_idx %||% seq_len(N)
  if (length(obs_idx) != N) {
    stop("obs_idx length (", length(obs_idx),
         ") does not match N = ", N, ".", call. = FALSE)
  }
  obs_idx
}


# The pilot grid Laplace the adapters warm-start from.
#
# `pilot_axis_points` other than the built-in 5 replaces the block's own grid
# with an even `d`-dimensional tensor product; the per-axis span comes from
# `block$bounds` when the block declares them and falls back to +/- 2 around
# `block$init` when it does not.
#
# Returns `list(fit, k_star, theta_init, block)`:
#   * `fit`        the nested-Laplace result (callers reading `$modes` or
#                  `$theta_sd` take them from here).
#   * `k_star`     index of the grid cell with the largest log-marginal.
#   * `theta_init` that cell's hyperparameter vector, named by
#                  `block$theta_names`.
#   * `block`      the pilot block actually fitted, obs_idx attached.
.tgmrf_pilot <- function(y, n_trials, X, block, obs_idx,
                         pilot_axis_points = 5L,
                         re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                         family = "binomial", phi = 1.0,
                         max_iter = 50L, tol = 1e-7, n_threads = 1L) {
  d <- block$theta_dim

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

  fit <- tulpa_nested_laplace(
    y = y, n_trials = n_trials, X = X,
    prior = pilot_block,
    re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
    family = family, phi = phi,
    # An inner solve on the adapter's behalf, silent like the per-theta ones
    # in `.tgmrf_make_log_marginal` (gcol33/tulpa#890).
    control = list(max_iter = max_iter, tol = tol, n_threads = n_threads,
                   progress = FALSE)
  )

  # A joint grid carries block-prefixed axis names, so its width is the
  # contract the d-dim theta vector is read against.
  grid <- fit$theta_grid
  if (ncol(grid) != d) {
    stop("Pilot grid has ", ncol(grid), " columns but block has ",
         d, " hyperparameters.", call. = FALSE)
  }

  k_star <- which.max(fit$log_marginal)
  theta_init <- as.numeric(grid[k_star, ])
  names(theta_init) <- block$theta_names

  list(fit = fit, k_star = k_star, theta_init = theta_init,
       block = pilot_block)
}


# The outer mode of `log p(theta | y)` and the curvature there, found from the
# pilot.
#
# The pilot's argmax is a node of a coarse lattice: on a 5-point axis it can sit
# a node spacing from the mode, and the curvature read there is the curvature
# of the wrong point. The mode is found instead by the outer mode-find the
# nested doors place their axes with (`.nl_placement_mode()`), started at the
# pilot's heaviest cell and searching a box padded past the pilot's span. The
# block's `theta` is already the unconstrained coordinate its `prior()` is
# written in, so every axis is searched on the identity and the density
# maximised is `lm$eval`, the one the adapters sample.
#
# `pl` is `.tgmrf_pilot()`'s result, `lm` `.tgmrf_make_log_marginal()`'s.
# Returns `list(mode, cov, status, evals)`: `mode` named by `block$theta_names`,
# `cov` the inverse negative Hessian of the log-posterior at `mode`, `status`
# as `.nl_placement_mode()` reports it. Errors when the mode-find finds no
# usable curvature: a Gaussian centred anywhere else is not a proposal for
# this posterior.
.tgmrf_outer_mode <- function(pl, lm, block) {
  d  <- block$theta_dim
  tg <- matrix(pl$fit$theta_grid, ncol = d,
               dimnames = list(NULL, block$theta_names))
  eval_logpost <- function(theta_mat) {
    vapply(seq_len(nrow(theta_mat)),
           function(i) lm$eval(as.numeric(theta_mat[i, ])), numeric(1))
  }
  tags <- rep("identity", d)
  pm <- .nl_placement_mode(tg, pl$fit$weights, tags, eval_logpost,
                           coord_tags = tags)
  if (!is.null(pm$declined) || length(pm$mode_u) != d ||
      any(diag(pm$cov_u) <= 0)) {
    stop("The outer mode-find over theta found no usable curvature (",
         pm$declined %||% "an axis held fixed", "); widen the block's ",
         "`bounds` or raise `pilot_axis_points` so the pilot spans the ",
         "posterior.", call. = FALSE)
  }
  list(mode = stats::setNames(pm$mode_u, block$theta_names),
       cov = matrix(pm$cov_u, d, d,
                    dimnames = list(block$theta_names, block$theta_names)),
       status = pm$status, evals = pm$evals)
}
