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

# Outer Pareto-k-hat for the SPDE (range, sigma) integration: importance-sample
# the joint hyperparameter posterior on the log scale against the Gaussian
# proposal (theta_hat, L_scale, both in (log_range, log_sigma) space) and
# PSIS-smooth. The target is the SAME log_marginal + PC-prior the quadrature
# weights use, so no extra Jacobian is needed -- the SPDE integrator already
# works on the log scale. Both axes are positive, so the transform is
# unambiguous. Runs with the RNG restored so the fit is unperturbed.
.spde_pareto_k <- function(theta_hat, L_scale, spde_log_marginal, sp, n_samples) {
  lt <- function(U) {
    r <- exp(U[, 1L]); s <- exp(U[, 2L])
    lm <- tryCatch(spde_log_marginal(r, s)$log_marginal,
                   error = function(e) rep(-Inf, length(r)))
    if (length(lm) != length(r)) return(rep(-Inf, length(r)))
    lm + pc_prior_log_density(r, s, sp$prior_range, sp$prior_sigma)
  }
  has_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (has_seed) get(".Random.seed", envir = .GlobalEnv) else NULL
  kd <- tryCatch(.nested_is_pareto_k(theta_hat, L_scale, lt, n_samples),
                 error = function(e) NULL)
  if (!is.null(old_seed)) assign(".Random.seed", old_seed, envir = .GlobalEnv)
  if (is.null(kd)) list(pareto_k = NA_real_, is_ess = NA_real_) else kd
}

# ---------------------------------------------------------------------
# Method = "grid": legacy rectangular grid, repaired to use the v10
# nested-Laplace ABI (re_idx / n_re_groups / sigma_re).
# ---------------------------------------------------------------------
fit_spde_nested_grid <- function(spde_log_marginal, sp, n_grid, spatial,
                                 diagnose_k = TRUE, k_samples = 200L) {
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

  # Outer k-hat: no mode Hessian on the grid path, so fit the Gaussian proposal
  # to the grid posterior moments in (log_range, log_sigma).
  kd <- list(pareto_k = NA_real_, is_ess = NA_real_)
  if (isTRUE(diagnose_k)) {
    u_grid <- cbind(log(grid$range), log(grid$sigma))
    u_hat  <- as.numeric(crossprod(weights, u_grid))
    cen    <- sweep(u_grid, 2L, u_hat)
    Su     <- crossprod(cen * weights, cen); Su <- (Su + t(Su)) / 2
    Lk <- tryCatch(t(chol(Su)), error = function(e) NULL)
    if (!is.null(Lk)) kd <- .spde_pareto_k(u_hat, Lk, spde_log_marginal, sp, k_samples)
  }

  list(
    mode = NULL,
    log_marginal = result$log_marginal,
    converged = all(result$n_iter > 0),
    spatial = spatial,
    pareto_k = kd$pareto_k,
    pareto_k_is_ess = kd$is_ess,
    pareto_k_scope = "outer (range, sigma) Gaussian proposal",
    nested = list(
      method     = "grid",
      range_grid = grid$range,
      sigma_grid = grid$sigma,
      weights    = weights,
      pareto_k   = kd$pareto_k,
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
                                sp, spatial,
                                diagnose_k = TRUE, k_samples = 200L) {
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
    return(fit_spde_nested_grid(spde_log_marginal, sp, n_grid = 5L, spatial,
                                diagnose_k = diagnose_k, k_samples = k_samples))
  }
  theta_hat <- op$par
  range_hat <- exp(theta_hat[1])
  sigma_hat <- exp(theta_hat[2])

  # Local Hessian via stats::optimHess (central differences). obj is the
  # negative log-posterior, so its Hessian at the mode is the precision of
  # the Gaussian approximation to p(theta | y); the Cholesky of its inverse
  # scales the CCD design.
  H <- stats::optimHess(par = theta_hat, fn = obj)
  prec <- 0.5 * (H + t(H))   # symmetrise before the conditioning check
  ev <- eigen(prec, symmetric = TRUE, only.values = TRUE)$values
  # Reject degenerate Hessians: if the smallest eigenvalue is essentially
  # zero relative to the largest, the posterior is too flat for a CCD
  # design to wrap. Quadratic-rule integration would put nodes at extreme
  # log-scale values where the SPDE Newton fails.
  if (any(!is.finite(ev)) ||
      max(abs(ev)) <= 0 ||
      min(ev) <= 1e-6 * max(abs(ev))) {
    warning("nested-Laplace Hessian is degenerate at the mode ",
            "(condition unfit for CCD); falling back to the rectangular grid.")
    return(fit_spde_nested_grid(spde_log_marginal, sp, n_grid = 5L, spatial,
                                diagnose_k = diagnose_k, k_samples = k_samples))
  }
  sigma_post <- tryCatch(solve(prec), error = function(e) NULL)
  if (is.null(sigma_post)) {
    warning("nested-Laplace Hessian inverse failed; ",
            "falling back to the rectangular grid.")
    return(fit_spde_nested_grid(spde_log_marginal, sp, n_grid = 5L, spatial,
                                diagnose_k = diagnose_k, k_samples = k_samples))
  }
  L <- t(chol(sigma_post))   # sigma_post = L L^T

  # CCD design (k = 2 -> 1 + 4 + 4 = 9 points). Sphere radius sqrt(2)*1.1
  # places the factorial corners at +/- 1.1 per axis -- INLA's standardized
  # scaling (f0 = 1.1) -- and `ccd_weights()` gives the matching corrected
  # R-INLA design weights (single source of truth with re_cov; see ccd_grid.R).
  # Map z to physical (log_range, log_sigma), then exponentiate. Cap log-scale
  # excursions at the optimisation box so a CCD axial that exceeds the prior
  # support snaps back to the boundary instead of overflowing to 1e+200.
  ccd <- ccd_grid(k = 2L, f_0 = sqrt(2) * 1.1)
  dnode <- ccd_weights(ccd)
  theta_grid <- ccd_to_theta(ccd$z, theta_hat, L, log_scale = FALSE)
  theta_grid[, 1] <- pmin(pmax(theta_grid[, 1], lower[1]), upper[1])
  theta_grid[, 2] <- pmin(pmax(theta_grid[, 2], lower[2]), upper[2])
  range_grid <- exp(theta_grid[, 1])
  sigma_grid <- exp(theta_grid[, 2])

  result <- spde_log_marginal(range_grid, sigma_grid)
  log_post <- result$log_marginal +
    pc_prior_log_density(range_grid, sigma_grid,
                         sp$prior_range, sp$prior_sigma)
  # CCD quadrature: node weight = design weight (Delta_k) times exp(log_post),
  # the INLA convention int ~ sum_k Delta_k pi(theta_k).
  log_max <- max(log_post)
  weights <- dnode * exp(log_post - log_max)
  weights <- weights / sum(weights)

  best <- which.max(log_post)

  # Refit at the joint posterior mode so the result has a usable mode
  # (the Laplace posterior at the marginal-mode hypers).
  fit_at_mode <- fit_spde_single(range_hat, sigma_hat)

  # Outer k-hat reuses the mode-find's own Gaussian (theta_hat on the log scale,
  # L = chol of the posterior covariance) as the importance proposal -- the same
  # Gaussian the CCD design is oriented by.
  kd <- if (isTRUE(diagnose_k)) {
    .spde_pareto_k(theta_hat, L, spde_log_marginal, sp, k_samples)
  } else list(pareto_k = NA_real_, is_ess = NA_real_)

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
    pareto_k         = kd$pareto_k,
    pareto_k_is_ess  = kd$is_ess,
    pareto_k_scope   = "outer (range, sigma) Gaussian proposal",
    nested = list(
      method        = "ccd",
      range_grid    = range_grid,
      sigma_grid    = sigma_grid,
      weights       = weights,
      pareto_k      = kd$pareto_k,
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
