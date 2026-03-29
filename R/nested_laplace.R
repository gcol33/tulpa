#' Nested Laplace Approximation for ICAR Spatial Models
#'
#' Phase 3a proof of concept: R-level outer loop over a 1D grid of
#' tau_spatial values, with inner Laplace at each grid point via
#' cpp_laplace_fit_spatial. Uses warm-starting (hot start) from the
#' previous grid point's mode for 5-10x fewer Newton iterations.
#'
#' @param y Integer response vector
#' @param n_trials Integer vector of trial sizes (binomial)
#' @param X Design matrix
#' @param spatial_idx Integer vector of site assignments (1-based)
#' @param n_spatial_units Number of spatial units
#' @param adj_row_ptr CSR row pointers for adjacency (0-based)
#' @param adj_col_idx CSR column indices (0-based)
#' @param n_neighbors Integer vector of neighbor counts
#' @param family Distribution family ("binomial", "poisson", "neg_binomial_2")
#' @param phi Dispersion parameter (negbin only)
#' @param tau_mode Initial mode for tau_spatial (positive)
#' @param n_grid Number of grid points (odd; default 9)
#' @param grid_width Width of grid in log-tau scale (default 4 SD)
#' @param max_iter Max Newton iterations per grid point
#' @param tol Newton convergence tolerance
#' @param n_threads OpenMP threads
#' @param verbose Print progress
#'
#' @return List with:
#'   - tau_grid: grid of tau values
#'   - log_tau_grid: grid on log scale
#'   - log_marginal: log p(y | tau) at each grid point
#'   - weights: normalized integration weights
#'   - n_iter: Newton iterations at each grid point
#'   - tau_mean: posterior mean of tau
#'   - tau_sd: posterior SD of tau
#'   - mode_at_tau_mode: mode vector at the modal tau
#'
#' @keywords internal
nested_laplace_icar <- function(
    y, n_trials, X, spatial_idx, n_spatial_units,
    adj_row_ptr, adj_col_idx, n_neighbors,
    family = "binomial", phi = 1.0,
    tau_mode = NULL, n_grid = 9L, grid_width = 4.0,
    max_iter = 50L, tol = 1e-6, n_threads = 1L,
    verbose = FALSE
) {
  N <- length(y)
  re_idx <- rep(0L, N)
  n_re_groups <- 0L
  sigma_re <- 1.0

  # --- Step 1: Find modal tau via grid search if not provided ---
  if (is.null(tau_mode)) {
    # Coarse search over log(tau) in [log(0.1), log(100)]
    log_tau_search <- seq(log(0.1), log(100), length.out = 15)
    lml_search <- numeric(length(log_tau_search))

    for (i in seq_along(log_tau_search)) {
      tau_i <- exp(log_tau_search[i])
      res <- cpp_laplace_fit_spatial(
        y = y, n = n_trials, X = X, re_idx = re_idx,
        n_re_groups = n_re_groups, sigma_re = sigma_re,
        spatial_idx = spatial_idx, n_spatial_units = n_spatial_units,
        adj_row_ptr = adj_row_ptr, adj_col_idx = adj_col_idx,
        n_neighbors = n_neighbors, tau_spatial = tau_i,
        family = family, phi = phi,
        max_iter = max_iter, tol = tol, n_threads = n_threads
      )
      # Add log-prior for tau: Gamma(1, 0.01) => log_prior = -0.01 * tau
      lml_search[i] <- res$log_marginal - 0.01 * tau_i
    }

    best <- which.max(lml_search)
    tau_mode <- exp(log_tau_search[best])
    if (verbose) message("Modal tau found: ", round(tau_mode, 3))
  }

  log_tau_mode <- log(tau_mode)

  # --- Step 2: Build grid around mode ---
  # Approximate curvature from 3 points around the mode
  delta <- 0.2
  lml_center <- {
    res <- cpp_laplace_fit_spatial(
      y = y, n = n_trials, X = X, re_idx = re_idx,
      n_re_groups = n_re_groups, sigma_re = sigma_re,
      spatial_idx = spatial_idx, n_spatial_units = n_spatial_units,
      adj_row_ptr = adj_row_ptr, adj_col_idx = adj_col_idx,
      n_neighbors = n_neighbors, tau_spatial = tau_mode,
      family = family, phi = phi,
      max_iter = max_iter, tol = tol, n_threads = n_threads
    )
    res$log_marginal - 0.01 * tau_mode
  }
  lml_left <- {
    res <- cpp_laplace_fit_spatial(
      y = y, n = n_trials, X = X, re_idx = re_idx,
      n_re_groups = n_re_groups, sigma_re = sigma_re,
      spatial_idx = spatial_idx, n_spatial_units = n_spatial_units,
      adj_row_ptr = adj_row_ptr, adj_col_idx = adj_col_idx,
      n_neighbors = n_neighbors, tau_spatial = exp(log_tau_mode - delta),
      family = family, phi = phi,
      max_iter = max_iter, tol = tol, n_threads = n_threads
    )
    res$log_marginal - 0.01 * exp(log_tau_mode - delta)
  }
  lml_right <- {
    res <- cpp_laplace_fit_spatial(
      y = y, n = n_trials, X = X, re_idx = re_idx,
      n_re_groups = n_re_groups, sigma_re = sigma_re,
      spatial_idx = spatial_idx, n_spatial_units = n_spatial_units,
      adj_row_ptr = adj_row_ptr, adj_col_idx = adj_col_idx,
      n_neighbors = n_neighbors, tau_spatial = exp(log_tau_mode + delta),
      family = family, phi = phi,
      max_iter = max_iter, tol = tol, n_threads = n_threads
    )
    res$log_marginal - 0.01 * exp(log_tau_mode + delta)
  }

  # Approximate second derivative (curvature) on log scale
  d2 <- (lml_left - 2 * lml_center + lml_right) / (delta^2)
  log_tau_sd <- if (d2 < -1e-6) 1 / sqrt(-d2) else 1.0

  # Grid: n_grid points spanning grid_width SDs
  half_width <- grid_width / 2 * log_tau_sd
  log_tau_grid <- seq(log_tau_mode - half_width, log_tau_mode + half_width,
                      length.out = n_grid)
  tau_grid <- exp(log_tau_grid)

  if (verbose) {
    message("Grid: log(tau) in [", round(min(log_tau_grid), 2), ", ",
            round(max(log_tau_grid), 2), "], SD = ", round(log_tau_sd, 3))
  }

  # --- Step 3: Inner Laplace at each grid point with warm-starting ---
  log_marginals <- numeric(n_grid)
  n_iters <- integer(n_grid)
  prev_mode <- NULL  # warm-start chain

  for (k in seq_len(n_grid)) {
    res <- cpp_laplace_fit_spatial(
      y = y, n = n_trials, X = X, re_idx = re_idx,
      n_re_groups = n_re_groups, sigma_re = sigma_re,
      spatial_idx = spatial_idx, n_spatial_units = n_spatial_units,
      adj_row_ptr = adj_row_ptr, adj_col_idx = adj_col_idx,
      n_neighbors = n_neighbors, tau_spatial = tau_grid[k],
      family = family, phi = phi,
      max_iter = max_iter, tol = tol, n_threads = n_threads,
      x_init = if (!is.null(prev_mode)) prev_mode else NULL
    )

    # Log marginal + log prior for tau (Gamma(1, 0.01))
    log_marginals[k] <- res$log_marginal - 0.01 * tau_grid[k]
    n_iters[k] <- res$n_iter
    prev_mode <- res$mode  # warm-start next grid point

    if (verbose) {
      message("  Grid point ", k, "/", n_grid,
              ": tau=", round(tau_grid[k], 3),
              " log_marg=", round(log_marginals[k], 2),
              " iters=", n_iters[k])
    }
  }

  # Save mode at the modal tau (closest grid point to mode)
  mode_idx <- which.min(abs(log_tau_grid - log_tau_mode))

  # --- Step 4: Numerical integration (trapezoidal on log scale) ---
  # Normalize log-marginals for numerical stability
  log_max <- max(log_marginals)
  log_weights <- log_marginals - log_max

  # Trapezoidal rule on log-tau scale
  d_log_tau <- diff(log_tau_grid)
  # Trapezoidal weights: (f[i] + f[i+1]) / 2 * h
  unnorm_weights <- exp(log_weights)
  trap_integral <- sum((unnorm_weights[-n_grid] + unnorm_weights[-1]) / 2 * d_log_tau)

  # Normalized weights
  weights <- unnorm_weights / trap_integral
  # Adjust for non-uniform spacing if needed
  weights <- weights / sum(weights)

  # Posterior moments for tau
  tau_mean <- sum(weights * tau_grid)
  tau_var <- sum(weights * (tau_grid - tau_mean)^2)
  tau_sd <- sqrt(max(0, tau_var))

  # Re-fit at modal tau to get the mode vector
  res_mode <- cpp_laplace_fit_spatial(
    y = y, n = n_trials, X = X, re_idx = re_idx,
    n_re_groups = n_re_groups, sigma_re = sigma_re,
    spatial_idx = spatial_idx, n_spatial_units = n_spatial_units,
    adj_row_ptr = adj_row_ptr, adj_col_idx = adj_col_idx,
    n_neighbors = n_neighbors, tau_spatial = tau_mode,
    family = family, phi = phi,
    max_iter = max_iter, tol = tol, n_threads = n_threads
  )

  list(
    tau_grid = tau_grid,
    log_tau_grid = log_tau_grid,
    log_marginal = log_marginals,
    weights = weights,
    n_iter = n_iters,
    tau_mean = tau_mean,
    tau_sd = tau_sd,
    tau_mode = tau_mode,
    log_tau_sd = log_tau_sd,
    mode_at_tau_mode = res_mode$mode
  )
}
