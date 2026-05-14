# PC prior log-density on (range, sigma) for a 2D Matern SPDE field.
# Convention (matching spatial_spde): prior_range = c(U_r, alpha_r) with
# P(range < U_r) = alpha_r; prior_sigma = c(U_s, alpha_s) with
# P(sigma > U_s) = alpha_s. Closed forms from Fuglstad et al. (2019):
#   range:  exponential on r^{-d/2}, d = 2 -> rate lambda_r,
#           p(r) = lambda_r * r^{-2} * exp(-lambda_r / r),
#           lambda_r = -U_r * log(1 - alpha_r).
#   sigma:  exponential, p(s) = lambda_s * exp(-lambda_s * s),
#           lambda_s = -log(alpha_s) / U_s.
# C++ cpp_nested_laplace_spde returns only the marginal likelihood; we add
# this prior here so the joint surface integrated by CCD / grid is a
# proper posterior with an interior mode in well-identified problems.
pc_prior_log_density <- function(range, sigma, prior_range, prior_sigma) {
  U_r <- prior_range[1]; alpha_r <- prior_range[2]
  U_s <- prior_sigma[1]; alpha_s <- prior_sigma[2]
  lambda_r <- -U_r * log(1 - alpha_r)
  lambda_s <- -log(alpha_s) / U_s
  log(lambda_r) - 2 * log(range) - lambda_r / range +
    log(lambda_s) - lambda_s * sigma
}

# Nested-Laplace integration backends for fit_spde().
#
# Two methods are supported:
#   * "grid" — rectangular grid in log(range) x log(sigma) around the
#     prior modes. Cheap, defensible, but burns most of its budget on
#     points that the data have already ruled out.
#   * "ccd"  — Central composite design centered on the joint posterior
#     mode of (range, sigma), oriented by the local Hessian. Mirrors
#     R-INLA's hyperparameter integration recipe; for k = 2 it uses
#     1 + 2*2 + 2^2 = 9 design points instead of n_grid^2 = 25 (default).
#
# Both backends return the same shape so the user-facing fit_spde()
# return contract is unchanged across methods.

# ---------------------------------------------------------------------
# Method = "grid": legacy rectangular grid, repaired to use the v10
# nested-Laplace ABI (re_idx / n_re_groups / sigma_re).
# ---------------------------------------------------------------------
fit_spde_nested_grid <- function(spde_log_marginal, sp, n_grid, spatial) {
  range_mode <- sp$prior_range[1]
  sigma_mode <- sp$prior_sigma[1]

  range_grid <- exp(seq(log(range_mode * 0.3), log(range_mode * 3),
                        length.out = n_grid))
  sigma_grid <- exp(seq(log(sigma_mode * 0.3), log(sigma_mode * 3),
                        length.out = n_grid))

  grid <- expand.grid(range = range_grid, sigma = sigma_grid)
  result <- spde_log_marginal(grid$range, grid$sigma)

  log_post <- result$log_marginal +
    pc_prior_log_density(grid$range, grid$sigma,
                         sp$prior_range, sp$prior_sigma)

  best <- which.max(log_post)
  log_max <- max(log_post)
  weights <- exp(log_post - log_max)
  weights <- weights / sum(weights)

  list(
    mode = NULL,
    log_marginal = result$log_marginal,
    converged = all(result$n_iter > 0),
    spatial = spatial,
    nested = list(
      method     = "grid",
      range_grid = grid$range,
      sigma_grid = grid$sigma,
      weights    = weights,
      n_iter     = result$n_iter,
      best_idx   = best,
      range_mean = sum(weights * grid$range),
      sigma_mean = sum(weights * grid$sigma),
      range_best = grid$range[best],
      sigma_best = grid$sigma[best]
    )
  )
}

# ---------------------------------------------------------------------
# Method = "ccd": find the joint posterior mode of (log_range, log_sigma),
# build a CCD grid oriented by the local negative-Hessian inverse, and
# integrate. Refits at the best grid point so `mode` is non-NULL.
# ---------------------------------------------------------------------
fit_spde_nested_ccd <- function(spde_log_marginal,
                                fit_spde_single,
                                sp, spatial) {
  range_mode <- sp$prior_range[1]
  sigma_mode <- sp$prior_sigma[1]

  # Outer mode-find on log scale, box-constrained around the prior modes
  # (log range / sigma in [prior * 1e-2, prior * 1e2]). Nelder-Mead drifted
  # to (~0, +Inf) on weakly-identified data; L-BFGS-B with bounds keeps the
  # optimiser inside the prior support and lets the post-fit sanity check
  # below catch the rare cases where the mode still lands at a corner.
  obj <- function(theta) {
    r <- exp(theta[1])
    s <- exp(theta[2])
    lm <- tryCatch(
      spde_log_marginal(r, s)$log_marginal,
      error = function(e) NA_real_
    )
    if (!is.finite(lm)) return(1e10)
    lp <- pc_prior_log_density(r, s, sp$prior_range, sp$prior_sigma)
    if (!is.finite(lp)) return(1e10)
    -(lm + lp)
  }
  init  <- c(log(range_mode), log(sigma_mode))
  lower <- init - log(100)
  upper <- init + log(100)
  # factr default (1e7) accepted the prior mode unchanged on weakly informative
  # problems; tightening to 1e5 forces a real descent. ndeps widened to 5e-2 so
  # the finite-difference gradient sees enough signal at log-scale.
  op <- tryCatch(
    stats::optim(par = init, fn = obj,
                 method = "L-BFGS-B", lower = lower, upper = upper,
                 control = list(factr = 1e5, maxit = 300,
                                ndeps = rep(5e-2, length(init)))),
    error = function(e) NULL
  )

  bad_mode <- function(op) {
    if (is.null(op)) return(TRUE)
    if (op$convergence != 0) return(TRUE)
    if (any(!is.finite(op$par))) return(TRUE)
    # Mode pinned to the optimisation box -> data are uninformative for
    # this hyper; fall back to the rectangular grid.
    eps <- 1e-3
    any(abs(op$par - lower) < eps) || any(abs(op$par - upper) < eps)
  }

  if (bad_mode(op)) {
    warning("nested-Laplace mode-find did not produce a usable mode ",
            "(degenerate or hit prior bounds); falling back to the ",
            "rectangular grid.")
    return(fit_spde_nested_grid(spde_log_marginal, sp, n_grid = 5L, spatial))
  }
  theta_hat <- op$par
  range_hat <- exp(theta_hat[1])
  sigma_hat <- exp(theta_hat[2])

  # Local Hessian via stats::optimHess (central differences). On the
  # log-scale, -H gives the precision of the Gaussian approximation to
  # p(theta | y); we want its Cholesky to scale the CCD design.
  H <- stats::optimHess(par = theta_hat, fn = obj)
  neg_H <- -H
  # Force symmetry, then ensure PD by ridging up if needed.
  neg_H <- 0.5 * (neg_H + t(neg_H))
  ev <- eigen(neg_H, symmetric = TRUE, only.values = TRUE)$values
  # Reject degenerate Hessians: if the smallest eigenvalue is essentially
  # zero relative to the largest, the posterior is too flat for a CCD
  # design to wrap. Quadratic-rule integration would put nodes at extreme
  # log-scale values where the SPDE Newton fails.
  if (any(!is.finite(ev)) ||
      max(abs(ev)) <= 0 ||
      min(ev) <= 1e-6 * max(abs(ev))) {
    warning("nested-Laplace Hessian is degenerate at the mode ",
            "(condition unfit for CCD); falling back to the rectangular grid.")
    return(fit_spde_nested_grid(spde_log_marginal, sp, n_grid = 5L, spatial))
  }
  sigma_post <- tryCatch(solve(neg_H), error = function(e) NULL)
  if (is.null(sigma_post)) {
    warning("nested-Laplace Hessian inverse failed; ",
            "falling back to the rectangular grid.")
    return(fit_spde_nested_grid(spde_log_marginal, sp, n_grid = 5L, spatial))
  }
  L <- t(chol(sigma_post))   # sigma_post = L L^T

  # CCD design (k = 2 -> 1 + 4 + 4 = 9 points). Map z to physical
  # (log_range, log_sigma), then exponentiate. Cap log-scale excursions
  # at the optimisation box so a CCD axial that exceeds the prior support
  # snaps back to the boundary instead of overflowing to 1e+200.
  ccd <- ccd_grid(k = 2L, f_0 = sqrt(2))
  theta_grid <- ccd_to_theta(ccd$z, theta_hat, L, log_scale = FALSE)
  theta_grid[, 1] <- pmin(pmax(theta_grid[, 1], lower[1]), upper[1])
  theta_grid[, 2] <- pmin(pmax(theta_grid[, 2], lower[2]), upper[2])
  range_grid <- exp(theta_grid[, 1])
  sigma_grid <- exp(theta_grid[, 2])

  result <- spde_log_marginal(range_grid, sigma_grid)
  log_post <- result$log_marginal +
    pc_prior_log_density(range_grid, sigma_grid,
                         sp$prior_range, sp$prior_sigma)
  log_max <- max(log_post)
  weights <- exp(log_post - log_max)
  weights <- weights / sum(weights)

  best <- which.max(log_post)

  # Refit at the joint posterior mode so the result has a usable mode
  # (the Laplace posterior at the marginal-mode hypers).
  fit_at_mode <- fit_spde_single(range_hat, sigma_hat)

  list(
    mode             = fit_at_mode$mode,
    beta             = fit_at_mode$beta,
    spatial_effects  = fit_at_mode$spatial_effects,
    log_det_Q        = fit_at_mode$log_det_Q,
    log_marginal     = result$log_marginal,
    converged        = all(result$n_iter > 0),
    spatial          = spatial,
    range            = range_hat,
    sigma            = sigma_hat,
    nested = list(
      method        = "ccd",
      range_grid    = range_grid,
      sigma_grid    = sigma_grid,
      weights       = weights,
      n_iter        = result$n_iter,
      kind          = ccd$kind,
      best_idx      = best,
      range_mode    = range_hat,
      sigma_mode    = sigma_hat,
      range_mean    = sum(weights * range_grid),
      sigma_mean    = sum(weights * sigma_grid),
      range_best    = range_grid[best],
      sigma_best    = sigma_grid[best],
      hessian_logsc = H,
      f_0           = ccd$f_0,
      n_points      = ccd$n_points
    )
  )
}
