# Shared `log_marginal(theta)` builder for the tgmrf hyperparameter adapters
# (tulpa_tgmrf(mode = "imh" / "vi" / "nuts")).
#
# All three integrate the same Laplace body for (beta, z) | theta and differ
# only in how they explore theta-space (FD Hessian + IMH, Pathfinder L-BFGS,
# NUTS). The per-theta evaluation is identical: pack theta into a one-row
# grid, run one inner nested-Laplace solve, read its scalar log-marginal. It
# lives here once so the inner-call argument contract has a single source of
# truth. (It used to be copy-pasted into all three; the Phase-2 migration of
# the nested-Laplace knobs into `control = list()` then desynced two copies,
# which silently caught the resulting "unused argument" error and returned
# -Inf -- the init guards in vi/nuts fired with a cryptic "not finite".)
#
# Returns `list(eval, raw, fixed)`:
#   * `raw(theta)`  one inner solve, NO error swallowing. A failure here is a
#                   structural bug (e.g. an engine-API drift), not numerical
#                   infeasibility, so it surfaces with its real message. Use
#                   for the eager check at the feasible pilot mode.
#   * `eval(theta)` the optimizer/sampler-facing closure: optional hard-wall
#                   bounds, then `raw` with errors/warnings from an infeasible
#                   theta mapped to -Inf so the caller retreats from the
#                   infeasible region.
#   * `fixed(theta_mat)` the conditional fixed-effect posterior
#                   `beta | theta, y ~ N(mean_k, cov_k)` of the same Laplace
#                   body at every row of `theta_mat`, from one grid solve.
#
# `bounds_lo` / `bounds_hi` NULL  -> no wall (local FD use in mode = "imh",
# whose +/- fd_step steps stay near the interior mode).
.tgmrf_make_log_marginal <- function(y, n_trials, X, block, obs_idx,
                                     re_idx, n_re_groups, sigma_re,
                                     family, phi,
                                     max_iter, tol, n_threads,
                                     bounds_lo = NULL, bounds_hi = NULL) {
  solve_grid <- function(theta_mat, keep_fixed) {
    blk <- block
    blk$obs_idx <- obs_idx
    blk$theta_grid_built <- matrix(theta_mat,
                                   ncol = length(block$theta_names),
                                   dimnames = list(NULL, block$theta_names))
    # One solve made on the sampler's behalf: its outer-grid progress line
    # would print once per theta evaluation, whatever the caller's `verbose`
    # (gcol33/tulpa#890).
    ctrl <- list(max_iter = max_iter, tol = tol, n_threads = n_threads,
                 progress = FALSE)
    if (keep_fixed) {
      # The rows are the sampler's theta draws, each one a point the fixed
      # effects are read at: no row may be screened out and none added.
      ctrl <- c(ctrl, list(keep_grid_hessians = TRUE, prune = FALSE,
                           axis_refine = "none"))
    }
    tulpa_nested_laplace(
      y = y, n_trials = n_trials, X = X, prior = blk,
      re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re,
      family = family, phi = phi, control = ctrl
    )
  }

  raw <- function(theta_vec) {
    out <- solve_grid(matrix(theta_vec, nrow = 1L), keep_fixed = FALSE)
    as.numeric(out$log_marginal[1])
  }

  eval <- function(theta_vec) {
    if (!is.null(bounds_lo) &&
        (any(theta_vec < bounds_lo) || any(theta_vec > bounds_hi))) {
      return(-Inf)
    }
    lm <- tryCatch(suppressWarnings(raw(theta_vec)),
                   error = function(e) NA_real_,
                   warning = function(w) NA_real_)
    if (!is.finite(lm)) return(-Inf)
    lm
  }

  fixed <- function(theta_mat) {
    theta_mat <- matrix(theta_mat, ncol = length(block$theta_names))
    out <- solve_grid(theta_mat, keep_fixed = TRUE)
    p <- ncol(X)
    grid <- matrix(out$theta_grid, ncol = ncol(theta_mat))
    if (nrow(grid) != nrow(theta_mat) ||
        !isTRUE(all.equal(unname(grid), unname(theta_mat))) ||
        length(out$grid_modes) != nrow(theta_mat) ||
        length(out$grid_hessians) != nrow(theta_mat)) {
      stop("The fixed-effect solve did not return one cell per theta draw.",
           call. = FALSE)
    }
    mean <- t(vapply(out$grid_modes, function(m) as.numeric(m)[seq_len(p)],
                     numeric(p)))
    if (p == 1L) mean <- matrix(mean, ncol = 1L)
    cov <- lapply(out$grid_hessians, .nested_cell_fixed_cov, p = p)
    list(mean = mean, cov = cov)
  }

  list(eval = eval, raw = raw, fixed = fixed)
}


# Fixed-effect column names of a tgmrf fit, shared by every adapter.
.tgmrf_beta_names <- function(X) {
  nm <- colnames(X)
  if (is.null(nm)) nm <- paste0("beta_", seq_len(ncol(X)))
  nm
}


# Joint (beta, theta) draws for the adapters that sample theta alone. The
# fixed effects are Gaussian given theta in the Laplace body, so a draw of the
# marginal posterior of beta is one conditional draw beta_s ~ N(mean(theta_s),
# cov(theta_s)) per theta draw: the mixture over the sampled theta. Each
# distinct theta row is solved once (an IMH chain repeats its rejected states).
.tgmrf_joint_draws <- function(theta_draws, lm, X) {
  p <- ncol(X)
  key <- apply(theta_draws, 1L, function(r) paste(format(r, digits = 17L),
                                                   collapse = "|"))
  first <- !duplicated(key)
  cond <- lm$fixed(theta_draws[first, , drop = FALSE])
  row_of <- match(key, key[first])
  beta <- matrix(NA_real_, nrow(theta_draws), p)
  for (u in seq_len(sum(first))) {
    rows <- which(row_of == u)
    V <- cond$cov[[u]]
    R <- tryCatch(chol(V), error = function(e) NULL)
    if (is.null(R)) {
      stop("The conditional fixed-effect covariance at a theta draw is not ",
           "positive definite.", call. = FALSE)
    }
    Z <- matrix(stats::rnorm(length(rows) * p), length(rows), p)
    beta[rows, ] <- sweep(Z %*% R, 2L, cond$mean[u, ], `+`)
  }
  colnames(beta) <- .tgmrf_beta_names(X)
  beta
}


# Lays the joint draws onto an adapter's fit: fixed effects first, as every
# chain producer reports them to the generic accessors, with the per-block
# matrices and moments beside them (the shape mode = "nuts_joint" carries).
.tgmrf_with_fixed <- function(fit, theta_draws, lm, X) {
  beta_draws <- .tgmrf_joint_draws(theta_draws, lm, X)
  draws <- cbind(beta_draws, theta_draws)
  fit$draws       <- draws
  fit$means       <- colMeans(draws)
  fit$sds         <- apply(draws, 2L, stats::sd)
  fit$param_names <- colnames(draws)
  fit$n_fixed     <- ncol(beta_draws)
  fit$fixed_names <- colnames(beta_draws)
  fit$n_params    <- ncol(draws)
  fit$n_samples   <- nrow(draws)
  fit$draws_beta  <- beta_draws
  fit$draws_theta <- theta_draws
  fit$means_beta  <- colMeans(beta_draws)
  fit$means_theta <- colMeans(theta_draws)
  fit$sds_beta    <- apply(beta_draws, 2L, stats::sd)
  fit$sds_theta   <- apply(theta_draws, 2L, stats::sd)
  fit
}
