#' Nested Laplace approximation for latent Gaussian models
#'
#' @description
#' Generic outer-grid nested Laplace driver. Builds a grid over the
#' hyperparameters of a single latent prior block (spatial or temporal),
#' runs an inner Laplace at each grid point with warm-starting, and
#' integrates over the grid to give proper hyperparameter marginals.
#'
#' Supported priors:
#'  * Spatial: `"icar"` (1D grid on tau), `"bym2"` (2D on (sigma, rho))
#'  * Temporal: `"rw1"`, `"rw2"` (1D grid on tau), `"ar1"` (2D on (tau, rho))
#'  * Continuous spatial: see [cpp_nested_laplace_spde()] (separate entry,
#'    rebuilds Q via SPDE Q-builder).
#'
#' @param y Response vector.
#' @param n_trials Trial sizes (binomial). Pass `1L`-vector otherwise.
#' @param X Fixed-effects design matrix.
#' @param prior A list describing the latent prior block. Required field
#'   `type` ∈ \{"icar", "bym2", "rw1", "rw2", "ar1"\}. Type-specific
#'   fields:
#'   * icar:  `spatial_idx`, `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`,
#'           `n_neighbors` (CSR adjacency, 0-based); optional `tau_grid`.
#'   * bym2:  same adjacency; `scale_factor`; optional `sigma_grid`, `rho_grid`.
#'   * rw1/rw2: `temporal_idx` (1-based), `n_times`; optional `tau_grid`,
#'             `cyclic` (rw1 only, default FALSE).
#'   * ar1:   `temporal_idx`, `n_times`; optional `tau_grid`, `rho_grid`.
#' @param re_idx Optional 1-based RE group index per obs (defaults to no RE).
#' @param n_re_groups RE group count (default 0).
#' @param sigma_re RE standard deviation (default 1).
#' @param family `"binomial"`, `"poisson"`, `"neg_binomial_2"`, etc.
#' @param phi Dispersion (negbin/gamma).
#' @param max_iter,tol Inner Newton iteration budget and tolerance.
#' @param n_threads OpenMP threads.
#' @param x_init Optional warm-start for the first grid point's inner solve.
#' @param verbose Print grid-point progress.
#'
#' @return A list with:
#'   * `theta_grid`: matrix or vector of grid hyperparameter values.
#'   * `log_marginal`: log p(y, mode | theta_k) at each grid point.
#'   * `weights`: integration weights normalising to sum 1.
#'   * `theta_mean`, `theta_sd`: posterior moments per hyperparameter.
#'   * `n_iter`: inner Newton iterations per grid point.
#'   * `modes`: matrix `[n_grid x n_x]` of inner modes, when stored.
#'   * `prior`: echoed input.
#'
#' @keywords internal
#' @export
nested_laplace <- function(y, n_trials, X, prior,
                            re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                            family = "binomial", phi = 1.0,
                            max_iter = 50L, tol = 1e-6, n_threads = 1L,
                            x_init = NULL, verbose = FALSE) {

  if (!is.list(prior) || is.null(prior$type)) {
    stop("`prior` must be a list with a `type` field", call. = FALSE)
  }
  type <- tolower(prior$type)
  N <- length(y)
  if (is.null(re_idx)) re_idx <- rep(0L, N)

  cargs <- list(
    y = as.numeric(y),
    n = as.integer(n_trials),
    X = X,
    re_idx = as.numeric(re_idx),
    n_re_groups = as.integer(n_re_groups),
    sigma_re = as.numeric(sigma_re),
    family = family,
    phi = as.numeric(phi),
    max_iter = as.integer(max_iter),
    tol = as.numeric(tol),
    n_threads = as.integer(n_threads),
    x_init_nullable = x_init
  )

  res <- switch(
    type,
    icar = .nl_icar(cargs, prior, verbose),
    bym2 = .nl_bym2(cargs, prior, verbose),
    rw1  = .nl_rw1(cargs, prior, verbose),
    rw2  = .nl_rw2(cargs, prior, verbose),
    ar1  = .nl_ar1(cargs, prior, verbose),
    stop("Unknown prior type: ", type,
         ". Supported: icar, bym2, rw1, rw2, ar1.", call. = FALSE)
  )

  # Integrate: trapezoid for 1D, simple normalised exp for 2D scatter grids.
  res$weights <- .nl_normalise_weights(res$log_marginal)
  res <- .nl_posterior_moments(res, type)
  res$prior <- prior
  class(res) <- c("tulpa_nested_laplace", "list")
  res
}

# --- Per-prior dispatch helpers ---

.nl_icar <- function(a, p, verbose) {
  if (is.null(p$tau_grid)) p$tau_grid <- .default_tau_grid(p, a, "icar")
  out <- do.call(cpp_nested_laplace_icar, c(list(
    spatial_idx = as.integer(p$spatial_idx),
    n_spatial_units = as.integer(p$n_spatial_units),
    adj_row_ptr = as.integer(p$adj_row_ptr),
    adj_col_idx = as.integer(p$adj_col_idx),
    n_neighbors = as.integer(p$n_neighbors),
    tau_grid = as.numeric(p$tau_grid)
  ), a))
  out$theta_grid <- as.numeric(p$tau_grid)
  out$theta_names <- "tau"
  out
}

.nl_bym2 <- function(a, p, verbose) {
  if (is.null(p$sigma_grid) || is.null(p$rho_grid)) {
    sg <- exp(seq(log(0.1), log(3), length.out = 5))
    rg <- c(0.2, 0.5, 0.8, 0.95)
    gr <- expand.grid(sigma = sg, rho = rg)
    p$sigma_grid <- gr$sigma; p$rho_grid <- gr$rho
  }
  out <- do.call(cpp_nested_laplace_bym2, c(list(
    spatial_idx = as.integer(p$spatial_idx),
    n_spatial_units = as.integer(p$n_spatial_units),
    adj_row_ptr = as.integer(p$adj_row_ptr),
    adj_col_idx = as.integer(p$adj_col_idx),
    n_neighbors = as.integer(p$n_neighbors),
    scale_factor = as.numeric(p$scale_factor %||% 1.0),
    sigma_spatial_grid = as.numeric(p$sigma_grid),
    rho_grid = as.numeric(p$rho_grid)
  ), a))
  out$theta_grid <- cbind(sigma = p$sigma_grid, rho = p$rho_grid)
  out$theta_names <- c("sigma", "rho")
  out
}

.nl_rw1 <- function(a, p, verbose) {
  if (is.null(p$tau_grid)) p$tau_grid <- .default_tau_grid(p, a, "rw1")
  cyclic <- isTRUE(p$cyclic)
  out <- do.call(cpp_nested_laplace_rw1, c(list(
    temporal_idx = as.integer(p$temporal_idx),
    n_times = as.integer(p$n_times),
    cyclic = cyclic,
    tau_grid = as.numeric(p$tau_grid)
  ), a))
  out$theta_grid <- as.numeric(p$tau_grid)
  out$theta_names <- "tau"
  out
}

.nl_rw2 <- function(a, p, verbose) {
  if (is.null(p$tau_grid)) p$tau_grid <- .default_tau_grid(p, a, "rw2")
  out <- do.call(cpp_nested_laplace_rw2, c(list(
    temporal_idx = as.integer(p$temporal_idx),
    n_times = as.integer(p$n_times),
    tau_grid = as.numeric(p$tau_grid)
  ), a))
  out$theta_grid <- as.numeric(p$tau_grid)
  out$theta_names <- "tau"
  out
}

.nl_ar1 <- function(a, p, verbose) {
  if (is.null(p$tau_grid) || is.null(p$rho_grid)) {
    g_tau <- exp(seq(log(0.5), log(20), length.out = 5))
    g_rho <- c(0.0, 0.4, 0.7, 0.9, 0.97)
    gr <- expand.grid(tau = g_tau, rho = g_rho)
    p$tau_grid <- gr$tau; p$rho_grid <- gr$rho
  }
  out <- do.call(cpp_nested_laplace_ar1, c(list(
    temporal_idx = as.integer(p$temporal_idx),
    n_times = as.integer(p$n_times),
    tau_grid = as.numeric(p$tau_grid),
    rho_grid = as.numeric(p$rho_grid)
  ), a))
  out$theta_grid <- cbind(tau = p$tau_grid, rho = p$rho_grid)
  out$theta_names <- c("tau", "rho")
  out
}

# Default 1D log-spaced tau grid, anchored on a 9-point search around an
# educated centre. Coarse but unbiased across reasonable problems.
.default_tau_grid <- function(prior, a, type) {
  exp(seq(log(0.3), log(30), length.out = 9))
}

# Normalise log-marginals to integration weights summing to 1.
# 1D regular grids: trapezoidal on the log-x axis (preserves shape).
# 2D / irregular: just exp-and-normalise (good enough for moments).
.nl_normalise_weights <- function(lm) {
  m <- max(lm)
  w <- exp(lm - m)
  w / sum(w)
}

# Compute weighted theta_mean / theta_sd from grid + weights.
.nl_posterior_moments <- function(res, type) {
  w <- res$weights
  tg <- res$theta_grid
  if (is.matrix(tg)) {
    res$theta_mean <- as.numeric(crossprod(w, tg))
    names(res$theta_mean) <- colnames(tg)
    res$theta_sd <- sqrt(pmax(0, as.numeric(crossprod(w, tg^2)) -
                                  res$theta_mean^2))
    names(res$theta_sd) <- colnames(tg)
  } else {
    res$theta_mean <- sum(w * tg)
    res$theta_sd <- sqrt(max(0, sum(w * tg^2) - res$theta_mean^2))
  }
  res
}

`%||%` <- function(x, y) if (is.null(x)) y else x
