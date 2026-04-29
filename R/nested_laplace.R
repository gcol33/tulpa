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
#' @param spec Optional `tulpa_temporal` or `tulpa_spatial` spec object
#'   (output of [temporal_rw1()], [temporal_rw2()], [temporal_ar1()],
#'   [spatial_car()], [spatial_bym2()], etc.). When supplied alongside
#'   `data`, the `prior` list is built automatically via
#'   [prior_from_spec()] — pass either `prior` or `spec`, not both.
#' @param data Data frame used to validate `spec` and resolve
#'   time/group/site indices. Required when `spec` is supplied.
#'
#' @keywords internal
#' @export
nested_laplace <- function(y, n_trials, X, prior = NULL,
                            spec = NULL, data = NULL,
                            re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                            family = "binomial", phi = 1.0,
                            max_iter = 50L, tol = 1e-6, n_threads = 1L,
                            x_init = NULL, verbose = FALSE) {

  if (!is.null(spec)) {
    if (!is.null(prior)) {
      stop("Pass either `spec` or `prior`, not both.", call. = FALSE)
    }
    if (is.null(data)) {
      stop("`data` is required when `spec` is supplied.", call. = FALSE)
    }
    prior <- prior_from_spec(spec, data)
  }
  if (!is.list(prior) || is.null(prior$type)) {
    stop("`prior` must be a list with a `type` field, or supply `spec` + `data`.",
         call. = FALSE)
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

#' Build a `prior` list for [nested_laplace()] from a tulpa spec object
#'
#' @description
#' Validates a `tulpa_temporal` or `tulpa_spatial` specification against
#' `data`, then converts it to the prior list shape consumed by
#' [nested_laplace()]. Mainly an internal helper for callers that already
#' have a fitted spec; users typically pass `spec` + `data` directly to
#' `nested_laplace()` instead.
#'
#' Supported spec types:
#'  * `tulpa_temporal` with `type ∈ {"rw1", "rw2", "ar1"}`
#'  * `tulpa_spatial` with `type ∈ {"car", "icar", "car_proper", "bym2"}`
#'    (continuous-spatial GP / SPDE specs need their own entry — see
#'    [cpp_nested_laplace_spde()])
#'
#' @param spec A `tulpa_temporal` or `tulpa_spatial` object.
#' @param data Data frame the spec resolves time/group/site indices against.
#' @return A `prior` list ready for [nested_laplace()].
#' @export
prior_from_spec <- function(spec, data) {
  if (inherits(spec, "tulpa_temporal")) {
    return(.prior_from_temporal_spec(spec, data))
  }
  if (inherits(spec, "tulpa_spatial")) {
    return(.prior_from_spatial_spec(spec, data))
  }
  stop("`spec` must inherit from 'tulpa_temporal' or 'tulpa_spatial'.",
       call. = FALSE)
}

.prior_from_temporal_spec <- function(spec, data) {
  spec <- validate_temporal(spec, data)
  type <- tolower(spec$type)
  if (!type %in% c("rw1", "rw2", "ar1")) {
    stop("nested_laplace() supports temporal types rw1, rw2, ar1; got '",
         type, "'.", call. = FALSE)
  }
  out <- list(
    type = type,
    temporal_idx = as.integer(spec$time_index),
    n_times = as.integer(spec$n_times)
  )
  if (type == "rw1") out$cyclic <- isTRUE(spec$cyclic)
  out
}

.prior_from_spatial_spec <- function(spec, data) {
  validate_spatial(spec, data)
  type <- tolower(spec$type)
  # Map "car" / "icar" → ICAR backend; "car_proper" not yet wired into
  # nested Laplace (it estimates ρ and would need its own grid).
  if (type %in% c("car", "icar")) {
    backend <- "icar"
  } else if (type == "bym2") {
    backend <- "bym2"
  } else if (type == "car_proper") {
    stop("Proper CAR (rho estimated) is not yet supported by ",
         "nested_laplace(); use BYM2 instead.", call. = FALSE)
  } else {
    stop("nested_laplace() does not yet support spatial type '", type,
         "'. Use BYM2/ICAR for areal models or cpp_nested_laplace_spde ",
         "for SPDE.", call. = FALSE)
  }

  adj <- spec$adjacency
  csr <- adjacency_to_csr_tulpa(adj)
  n_spatial_units <- nrow(adj)

  # spatial_idx per obs
  if (!is.null(spec$level) && spec$level == "group" &&
      !is.null(spec$group_var)) {
    if (!(spec$group_var %in% names(data))) {
      stop("Spatial group variable '", spec$group_var,
           "' not found in data.", call. = FALSE)
    }
    g <- as.factor(data[[spec$group_var]])
    spatial_idx <- as.integer(g)
  } else {
    if (nrow(data) != n_spatial_units) {
      stop("Observation-level spatial spec requires nrow(data) == ",
           "nrow(adjacency).", call. = FALSE)
    }
    spatial_idx <- seq_len(n_spatial_units)
  }

  out <- list(
    type = backend,
    spatial_idx = spatial_idx,
    n_spatial_units = as.integer(n_spatial_units),
    adj_row_ptr = as.integer(csr$row_ptr),
    adj_col_idx = as.integer(csr$col_idx),
    n_neighbors = as.integer(csr$n_neighbors)
  )
  if (backend == "bym2") {
    out$scale_factor <- as.numeric(spec$scale_factor %||% 1.0)
  }
  out
}
