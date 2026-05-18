#' Nested Laplace approximation for latent Gaussian models
#'
#' @description
#' Generic outer-grid nested Laplace driver. Builds a grid over the
#' hyperparameters of a single latent prior block (spatial or temporal),
#' runs an inner Laplace at each grid point with warm-starting, and
#' integrates over the grid to give proper hyperparameter marginals.
#'
#' Supported priors:
#'  * Spatial (areal): `"icar"` (1D grid on tau), `"bym2"`
#'    (2D on (sigma, rho)), `"car_proper"` (2D on (tau, rho); rho lives
#'    in the eigenvalue interval (1/lambda_min, 1/lambda_max) of
#'    D^{-1}W).
#'  * Spatial (continuous): `"nngp"` (2D on (sigma2, phi_gp)),
#'    `"hsgp"` (2D on (sigma2, lengthscale)).
#'  * Temporal: `"rw1"`, `"rw2"` (1D grid on tau), `"ar1"` (2D on (tau, rho))
#'  * SPDE continuous spatial: see [cpp_nested_laplace_spde()] (separate
#'    entry, rebuilds Q via SPDE Q-builder).
#'
#' @param y Response vector.
#' @param n_trials Trial sizes (binomial). Pass `1L`-vector otherwise.
#' @param X Fixed-effects design matrix.
#' @param prior A list describing the latent prior block. Required field
#'   `type` ∈ \{"icar", "bym2", "car_proper", "rw1", "rw2", "ar1"\}.
#'   Type-specific fields:
#'   * icar:  `spatial_idx`, `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`,
#'           `n_neighbors` (CSR adjacency, 0-based); optional `tau_grid`.
#'   * bym2:  same adjacency; `scale_factor`; optional `sigma_grid`, `rho_grid`.
#'   * car_proper: same adjacency; optional `tau_grid`, `rho_grid`,
#'           `rho_bounds = c(lower, upper)` (defaults to (0, 1)).
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
#' @param keep_grid_hessians If `TRUE`, retain per-grid-point fixed-effects
#'   marginal Hessian \eqn{H_\beta} and mode \eqn{\hat{\beta}} on the return
#'   list as `$grid_hessians` (list of dense \eqn{p\times p} matrices) and
#'   `$grid_modes` (list of length-\eqn{p} vectors). Default `FALSE`.
#'   Used downstream by simplified-Laplace (SLA) callers to assemble
#'   skew-aware marginals — see the cumulant pooling in [rubins_pool()].
#'
#' @keywords internal
#' @export
tulpa_nested_laplace <- function(y, n_trials, X, prior = NULL,
                            spec = NULL, data = NULL,
                            re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                            family = "binomial", phi = 1.0,
                            max_iter = 50L, tol = 1e-6, n_threads = 1L,
                            x_init = NULL, verbose = FALSE,
                            keep_grid_hessians = FALSE) {

  if (!is.null(spec)) {
    if (!is.null(prior)) {
      stop("Pass either `spec` or `prior`, not both.", call. = FALSE)
    }
    if (is.null(data)) {
      stop("`data` is required when `spec` is supplied.", call. = FALSE)
    }
    prior <- prior_from_spec(spec, data)
  }
  # Allow passing a single tgmrf S3 object directly: wrap it in a length-1
  # block list so the multi-block dispatch picks it up. This keeps the
  # ergonomic API close to the formula sketch in generic-todo.md (one block,
  # one call) while still routing through the validated multi-block path.
  if (inherits(prior, "tulpa_latent_block")) {
    prior <- list(prior)
  }
  if (!is.list(prior)) {
    stop("`prior` must be a list (single block) or list-of-lists (multi-block).",
         call. = FALSE)
  }
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
    x_init_nullable = x_init,
    store_Q = isTRUE(keep_grid_hessians)
  )

  p_fixed <- ncol(X)

  if (.is_multi_block_prior(prior)) {
    res <- .nl_dispatch_multi(cargs, prior)
    if (isTRUE(keep_grid_hessians)) {
      res <- .nl_attach_grid_hessians(res, p_fixed)
    }
    res$prior <- prior
    class(res) <- c("tulpa_nested_laplace", "list")
    return(res)
  }

  if (is.null(prior$type)) {
    stop("Single-block `prior` must have a `type` field, ",
         "or pass a list of blocks for multi-block.", call. = FALSE)
  }
  type <- tolower(prior$type)
  res <- .nl_dispatch(type, cargs, prior)

  # Integrate: trapezoid for 1D, simple normalised exp for 2D scatter grids.
  res$weights <- .nl_normalise_weights(res$log_marginal)
  res <- .nl_posterior_moments(res, type)
  if (isTRUE(keep_grid_hessians)) {
    res <- .nl_attach_grid_hessians(res, p_fixed)
  }
  res$prior <- prior
  class(res) <- c("tulpa_nested_laplace", "list")
  res
}

#' @rdname tulpa_nested_laplace
#' @description
#' `nested_laplace()` is a deprecated alias retained for backward compatibility.
#' Use [tulpa_nested_laplace()] instead. Will be removed in a future release.
#' @export
nested_laplace <- function(...) {
  .Deprecated("tulpa_nested_laplace", package = "tulpa")
  tulpa_nested_laplace(...)
}

# --- Per-prior registry & dispatcher ---
#
# Each entry describes one supported prior `type`. To add a new prior:
#   1. Add a row here with cpp_fn, defaults, pack, theta.
#   2. Make sure cpp_nested_laplace_<variant> exists in RcppExports.
# No changes needed to .nl_dispatch() or the public driver.

# Areal CSR adjacency block — shared by icar/bym2/car_proper.
.nl_adj_args <- function(p) list(
  spatial_idx     = as.integer(p$spatial_idx),
  n_spatial_units = as.integer(p$n_spatial_units),
  adj_row_ptr     = as.integer(p$adj_row_ptr),
  adj_col_idx     = as.integer(p$adj_col_idx),
  n_neighbors     = as.integer(p$n_neighbors)
)

.NL_REGISTRY <- list(
  icar = list(
    cpp_fn = "cpp_nested_laplace_icar",
    defaults = function(p, a) {
      if (is.null(p$tau_grid)) p$tau_grid <- .default_tau_grid(p, a, "icar")
      p
    },
    pack = function(p) c(.nl_adj_args(p), list(
      tau_grid = as.numeric(p$tau_grid)
    )),
    theta = function(p) list(grid = as.numeric(p$tau_grid), names = "tau")
  ),

  bym2 = list(
    cpp_fn = "cpp_nested_laplace_bym2",
    defaults = function(p, a) {
      if (is.null(p$sigma_grid) || is.null(p$rho_grid)) {
        sg <- exp(seq(log(0.1), log(3), length.out = 5))
        rg <- c(0.2, 0.5, 0.8, 0.95)
        gr <- expand.grid(sigma = sg, rho = rg)
        p$sigma_grid <- gr$sigma; p$rho_grid <- gr$rho
      }
      p
    },
    pack = function(p) c(.nl_adj_args(p), list(
      scale_factor       = as.numeric(p$scale_factor %||% 1.0),
      sigma_spatial_grid = as.numeric(p$sigma_grid),
      rho_grid           = as.numeric(p$rho_grid)
    )),
    theta = function(p) list(
      grid  = cbind(sigma = p$sigma_grid, rho = p$rho_grid),
      names = c("sigma", "rho")
    )
  ),

  car_proper = list(
    cpp_fn = "cpp_nested_laplace_car_proper",
    defaults = function(p, a) {
      if (is.null(p$tau_grid) || is.null(p$rho_grid)) {
        rb <- p$rho_bounds %||% c(0, 1)
        # Margin in from the eigenvalue endpoints — Q goes singular at the
        # boundary, so anchoring the grid in (lower + eps, upper - eps) avoids
        # NaN log-determinants from the per-grid-point Cholesky.
        eps <- 0.05 * (rb[2] - rb[1])
        lo <- rb[1] + eps
        hi <- rb[2] - eps
        g_tau <- exp(seq(log(0.3), log(30), length.out = 5))
        g_rho <- seq(lo, hi, length.out = 5)
        gr <- expand.grid(tau = g_tau, rho = g_rho)
        p$tau_grid <- gr$tau; p$rho_grid <- gr$rho
      }
      p
    },
    pack = function(p) c(.nl_adj_args(p), list(
      tau_grid = as.numeric(p$tau_grid),
      rho_grid = as.numeric(p$rho_grid)
    )),
    theta = function(p) list(
      grid  = cbind(tau = p$tau_grid, rho = p$rho_grid),
      names = c("tau", "rho")
    )
  ),

  nngp = list(
    cpp_fn = "cpp_nested_laplace_nngp",
    defaults = function(p, a) {
      if (is.null(p$sigma2_grid) || is.null(p$phi_gp_grid)) {
        s2 <- exp(seq(log(0.05), log(2), length.out = 5))
        pg <- exp(seq(log(0.05), log(1.5), length.out = 5))
        gr <- expand.grid(sigma2 = s2, phi_gp = pg)
        p$sigma2_grid <- gr$sigma2
        p$phi_gp_grid <- gr$phi_gp
      }
      p
    },
    pack = function(p) {
      # nn_order in cpp_nested_laplace_nngp expects 0-based indices, matching
      # the convention in cpp_laplace_fit_gp (see R/fit_laplace.R:433).
      n_spatial <- as.integer(p$n_spatial)
      # spatial_idx (1-based, length N) maps obs -> spatial unit. Default to
      # 1..n_spatial when N == n_spatial (the legacy one-obs-per-location case).
      spatial_idx <- if (!is.null(p$spatial_idx)) {
        as.integer(p$spatial_idx)
      } else {
        seq_len(n_spatial)
      }
      list(
        spatial_idx = spatial_idx,
        coords      = as.matrix(p$coords),
        nn_idx      = as.matrix(p$nn_idx),
        nn_dist     = as.matrix(p$nn_dist),
        nn_order    = as.integer((p$nn_order %||% seq_len(n_spatial)) - 1L),
        n_spatial   = n_spatial,
        nn          = as.integer(p$nn),
        sigma2_grid = as.numeric(p$sigma2_grid),
        phi_gp_grid = as.numeric(p$phi_gp_grid),
        cov_type    = as.integer(p$cov_type %||% 2L)  # default Matern-5/2
      )
    },
    theta = function(p) list(
      grid  = cbind(sigma2 = p$sigma2_grid, phi_gp = p$phi_gp_grid),
      names = c("sigma2", "phi_gp")
    )
  ),

  hsgp = list(
    cpp_fn = "cpp_nested_laplace_hsgp",
    defaults = function(p, a) {
      if (is.null(p$sigma2_grid) || is.null(p$lengthscale_grid)) {
        s2 <- exp(seq(log(0.05), log(2), length.out = 5))
        ls <- exp(seq(log(0.05), log(1.5), length.out = 5))
        gr <- expand.grid(sigma2 = s2, ell = ls)
        p$sigma2_grid <- gr$sigma2
        p$lengthscale_grid <- gr$ell
      }
      p
    },
    pack = function(p) list(
      phi_basis        = as.matrix(p$phi_basis),
      lambda_eig       = as.numeric(p$lambda_eig),
      sigma2_grid      = as.numeric(p$sigma2_grid),
      lengthscale_grid = as.numeric(p$lengthscale_grid)
    ),
    theta = function(p) list(
      grid  = cbind(sigma2 = p$sigma2_grid, lengthscale = p$lengthscale_grid),
      names = c("sigma2", "lengthscale")
    )
  ),

  rw1 = list(
    cpp_fn = "cpp_nested_laplace_rw1",
    defaults = function(p, a) {
      if (is.null(p$tau_grid)) p$tau_grid <- .default_tau_grid(p, a, "rw1")
      p
    },
    pack = function(p) list(
      temporal_idx = as.integer(p$temporal_idx),
      n_times      = as.integer(p$n_times),
      cyclic       = isTRUE(p$cyclic),
      tau_grid     = as.numeric(p$tau_grid)
    ),
    theta = function(p) list(grid = as.numeric(p$tau_grid), names = "tau")
  ),

  rw2 = list(
    cpp_fn = "cpp_nested_laplace_rw2",
    defaults = function(p, a) {
      if (is.null(p$tau_grid)) p$tau_grid <- .default_tau_grid(p, a, "rw2")
      p
    },
    pack = function(p) list(
      temporal_idx = as.integer(p$temporal_idx),
      n_times      = as.integer(p$n_times),
      tau_grid     = as.numeric(p$tau_grid)
    ),
    theta = function(p) list(grid = as.numeric(p$tau_grid), names = "tau")
  ),

  ar1 = list(
    cpp_fn = "cpp_nested_laplace_ar1",
    defaults = function(p, a) {
      if (is.null(p$tau_grid) || is.null(p$rho_grid)) {
        g_tau <- exp(seq(log(0.5), log(20), length.out = 5))
        g_rho <- c(0.0, 0.4, 0.7, 0.9, 0.97)
        gr <- expand.grid(tau = g_tau, rho = g_rho)
        p$tau_grid <- gr$tau; p$rho_grid <- gr$rho
      }
      p
    },
    pack = function(p) list(
      temporal_idx = as.integer(p$temporal_idx),
      n_times      = as.integer(p$n_times),
      tau_grid     = as.numeric(p$tau_grid),
      rho_grid     = as.numeric(p$rho_grid)
    ),
    theta = function(p) list(
      grid  = cbind(tau = p$tau_grid, rho = p$rho_grid),
      names = c("tau", "rho")
    )
  ),

  iid = list(
    # Multi-block-only entry. No standalone `cpp_fn` -- IID with no other
    # latent structure is just a Gaussian RE, served by the `n_re_groups`
    # path of `tulpa_nested_laplace()`. This registry entry exists so that
    # IID can appear as a block inside a multi-block prior (e.g. BYM2 +
    # AR1 + IID-observer).
    cpp_fn = NULL,
    defaults = function(p, a) {
      if (is.null(p$sigma_grid)) {
        p$sigma_grid <- exp(seq(log(0.1), log(3), length.out = 5))
      }
      p
    },
    pack = function(p) stop(
      "IID is only supported inside a multi-block prior. ",
      "For a standalone Gaussian RE, use the `re_idx`/`n_re_groups` args ",
      "of `tulpa_nested_laplace()`.", call. = FALSE
    ),
    theta = function(p) list(
      grid = as.numeric(p$sigma_grid), names = "sigma"
    )
  ),

  tgmrf = list(
    # User-defined GMRF block (see ?tgmrf). Only available inside a multi-
    # block prior. The outer theta-grid is the Cartesian product of
    # per-axis grids; per-axis defaults are 5 equispaced points between
    # block$bounds$lower[j] and block$bounds$upper[j], or block$init[j] +/-
    # 2 if no bounds are supplied. Precomputed Q_k / logdet / log p(theta_k)
    # at every grid row are packed into the C++ block spec by
    # `.nl_block_spec_for_cpp`.
    cpp_fn = NULL,
    defaults = function(p, a) {
      if (is.null(p$theta_grid_built)) {
        d <- p$theta_dim
        axes <- vector("list", d)
        for (j in seq_len(d)) {
          if (!is.null(p$bounds)) {
            lo <- p$bounds$lower[j]
            hi <- p$bounds$upper[j]
          } else {
            lo <- p$init[j] - 2
            hi <- p$init[j] + 2
          }
          axes[[j]] <- seq(lo, hi, length.out = 5L)
        }
        names(axes) <- p$theta_names
        gr <- do.call(expand.grid, axes)
        p$theta_grid_built <- as.matrix(gr)
      }
      p
    },
    pack = function(p) stop(
      "tgmrf is only supported inside a multi-block prior. ",
      "Wrap the block in list(): tulpa_nested_laplace(prior = list(blk), ...).",
      call. = FALSE
    ),
    theta = function(p) list(
      grid  = p$theta_grid_built,
      names = p$theta_names
    )
  )
)

.nl_dispatch <- function(type, a, p) {
  spec <- .NL_REGISTRY[[type]]
  if (is.null(spec)) {
    stop("Unknown prior type: ", type,
         ". Supported: ", paste(names(.NL_REGISTRY), collapse = ", "), ".",
         call. = FALSE)
  }
  p   <- spec$defaults(p, a)
  out <- do.call(spec$cpp_fn, c(spec$pack(p), a))
  th  <- spec$theta(p)
  out$theta_grid  <- th$grid
  out$theta_names <- th$names
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

# Marginal log-density along a single hyperparameter axis (logsumexp over
# the other-axis cells at each unique value). `vals` and `log_marg` are
# length n_cells; `keep` is an optional logical mask (cartesian + same-
# axis slice cells). Returns sorted unique axis values and the matching
# marginal log-density.
.nl_axis_marginal_logdensity <- function(vals, log_marg, keep = NULL) {
  if (is.null(keep)) keep <- rep(TRUE, length(vals))
  v <- vals[keep]; l <- log_marg[keep]
  uv <- sort(unique(v))
  if (length(uv) == 0L) return(list(vals = uv, log_marg = numeric(0)))
  lm <- vapply(uv, function(u) {
    li <- l[v == u]
    if (length(li) == 0L) return(-Inf)
    m <- max(li)
    if (!is.finite(m)) return(-Inf)
    m + log(sum(exp(li - m)))
  }, numeric(1))
  list(vals = uv, log_marg = lm)
}

# Laplace-at-mode SD on a single axis. Fits a 3-point parabola to the
# marginal log-density at the modal cell and one neighbour on each side,
# returns sqrt(-1 / (2 a)) from the quadratic coefficient. When the
# axis values are positive (sigma / phi / tau) the parabola is fit on
# log(theta), and the SD is mapped back to the linear axis via the
# delta method (sigma_theta = theta_mode * sigma_log_theta). Returns NA
# when the mode sits at an axis edge or the parabola is concave up.
#
# `return_log_sd = TRUE` skips the delta back-map and returns sd(log theta)
# directly. Only meaningful with `log_axis = TRUE` -- returns NA otherwise.
# Used by `.joint_attach_alpha_moments` to combine per-axis sigma SDs into
# SD(alpha) via the delta method (Var(log alpha) = Var(log sigma_pos) +
# Var(log sigma_occ) under independence).
.nl_laplace_at_mode_sd_axis <- function(vals, log_marg, log_axis = NULL,
                                        return_log_sd = FALSE) {
  if (length(vals) < 3L) return(NA_real_)
  ix <- which.max(log_marg)
  if (ix == 1L || ix == length(vals)) return(NA_real_)
  if (is.null(log_axis)) log_axis <- all(is.finite(vals)) && all(vals > 0)
  u <- if (log_axis) log(vals) else vals
  dm <- u[ix - 1L] - u[ix]
  dp <- u[ix + 1L] - u[ix]
  det <- dm * dp * (dm - dp)
  if (!is.finite(det) || abs(det) < .Machine$double.eps) return(NA_real_)
  lm_m <- log_marg[ix - 1L] - log_marg[ix]
  lm_p <- log_marg[ix + 1L] - log_marg[ix]
  a <- (lm_m * dp - lm_p * dm) / det
  if (!is.finite(a) || a >= 0) return(NA_real_)
  sd_u <- sqrt(-1 / (2 * a))
  if (return_log_sd) {
    if (log_axis) sd_u else NA_real_
  } else {
    if (log_axis) vals[ix] * sd_u else sd_u
  }
}

# Laplace-at-mode (mu, sd) on a single positive-valued axis, both on the
# log scale. Fits the same 3-point parabola as `.nl_laplace_at_mode_sd_axis`
# (centred on the modal cell, log-transformed axis) and in addition to the
# curvature-derived SD = sqrt(-1 / (2a)) returns the parabola vertex
# u_v = u[ix] - b/(2a) as the continuous MAP / Gaussian-Laplace mean on the
# log axis. Returns list(mu = NA, sd = NA) when the modal cell sits at an
# axis edge or the parabola is concave up (no Laplace approximation).
#
# Used by `.joint_alpha_log_params` to compute closed-form Lognormal
# moments of derived `alpha = sigma_pos / sigma_occ` from per-axis Laplace
# fits, replacing the discrete weighted sum `sum(weights * alpha_grid)` that
# was upward-biased by Jensen's inequality on `1/sigma_occ` on coarse sigma
# grids (gcol33/tulpa#21 follow-up).
.nl_laplace_at_mode_log_params_axis <- function(vals, log_marg) {
  na_pair <- list(mu = NA_real_, sd = NA_real_)
  if (length(vals) < 3L) return(na_pair)
  if (!all(is.finite(vals)) || !all(vals > 0)) return(na_pair)
  ix <- which.max(log_marg)
  if (ix == 1L || ix == length(vals)) return(na_pair)
  u <- log(vals)
  dm <- u[ix - 1L] - u[ix]
  dp <- u[ix + 1L] - u[ix]
  det <- dm * dp * (dm - dp)
  if (!is.finite(det) || abs(det) < .Machine$double.eps) return(na_pair)
  lm_m <- log_marg[ix - 1L] - log_marg[ix]
  lm_p <- log_marg[ix + 1L] - log_marg[ix]
  a <- (lm_m * dp - lm_p * dm) / det
  b <- (lm_p * dm^2 - lm_m * dp^2) / det
  if (!is.finite(a) || !is.finite(b) || a >= 0) return(na_pair)
  list(mu = u[ix] - b / (2 * a),
       sd = sqrt(-1 / (2 * a)))
}

# Replace `theta_sd` (and `block_moments[[b]]$sd` when present) entries
# with the Laplace-at-mode SD wherever the 3-point fit succeeds. Axes
# with the mode at an edge or wrong-signed curvature keep their var-of-
# means SD. Grid-spacing-independent: fixes the symptom where a sharply
# peaked marginal log-likelihood on a coarse grid collapses var-of-
# means to ~0 and undercovers (gcol33/tulpa#20).
.nl_refit_axis_sd_laplace <- function(res, refining = NULL) {
  if (is.null(res$theta_grid) || is.null(res$log_marginal)) return(res)
  tg <- res$theta_grid
  if (!is.matrix(tg)) {
    marg <- .nl_axis_marginal_logdensity(as.numeric(tg), res$log_marginal)
    sd_lam <- .nl_laplace_at_mode_sd_axis(marg$vals, marg$log_marg)
    if (is.finite(sd_lam)) res$theta_sd <- sd_lam
    return(res)
  }
  if (is.null(refining)) refining <- res$refining_axis
  if (is.null(refining)) refining <- rep("", nrow(tg))
  col_names <- colnames(tg)
  if (!is.null(col_names) && !is.null(res$theta_sd)) {
    for (col in col_names) {
      if (identical(col, "alpha")) next  # derived axis; not a grid column
      if (!col %in% names(res$theta_sd)) next
      keep <- refining == "" | refining == col |
              refining == paste0("consistency_", col)
      marg <- .nl_axis_marginal_logdensity(tg[, col], res$log_marginal, keep)
      sd_lam <- .nl_laplace_at_mode_sd_axis(marg$vals, marg$log_marg)
      if (is.finite(sd_lam)) res$theta_sd[[col]] <- sd_lam
    }
  }
  if (!is.null(res$block_moments)) {
    for (b in seq_along(res$block_moments)) {
      bm <- res$block_moments[[b]]
      axis_cols <- bm$axis_cols
      if (is.null(axis_cols) || length(axis_cols) == 0L) next
      bare <- names(bm$sd)
      for (j in seq_along(axis_cols)) {
        col_ix <- axis_cols[j]
        col_name <- if (!is.null(col_names)) col_names[col_ix] else ""
        keep <- refining == "" | refining == col_name |
                refining == paste0("consistency_", col_name)
        marg <- .nl_axis_marginal_logdensity(tg[, col_ix], res$log_marginal,
                                              keep)
        sd_lam <- .nl_laplace_at_mode_sd_axis(marg$vals, marg$log_marg)
        if (is.finite(sd_lam)) res$block_moments[[b]]$sd[[j]] <- sd_lam
      }
    }
  }
  res
}

`%||%` <- function(x, y) if (is.null(x)) y else x

# ----------------------------------------------------------------------------
# Per-grid-point fixed-effects Hessian extraction.
#
# When tulpa_nested_laplace() is called with keep_grid_hessians = TRUE the
# inner Laplace solves are run with store_Q = TRUE, so each grid point's full
# posterior precision (over the joint latent vector x = (beta, RE, latent))
# is returned in CSC lower-triangle form. Here we
#
#   1. Reconstruct sparse symmetric Q_k from the CSC triple at grid point k.
#   2. Compute Sigma_{beta beta} = (Q_k^{-1})[1:p, 1:p] via a single sparse
#      solve with the first p columns of the identity as RHS.
#   3. Invert that small p x p block to get the marginal fixed-effects
#      Hessian H_beta = Sigma_{beta beta}^{-1}.
#
# Per-grid memory cost: O(p^2). The dominant compute is the sparse solve,
# already required for the integrated log-marginal — so adding it here is
# essentially free.
# ----------------------------------------------------------------------------
.nl_attach_grid_hessians <- function(res, p_fixed) {
  Q_p <- res$Q_csc_p_per_grid
  Q_i <- res$Q_csc_i_per_grid
  Q_x <- res$Q_csc_x_per_grid
  n_x <- res$Q_csc_n

  if (is.null(Q_p) || is.null(Q_i) || is.null(Q_x) || is.null(n_x)) {
    warning("keep_grid_hessians = TRUE but the inner solve did not return Q. ",
            "This is a tulpa bug; please report.",
            call. = FALSE)
    return(res)
  }

  n_grid <- length(Q_p)
  grid_hessians <- vector("list", n_grid)
  grid_modes    <- vector("list", n_grid)

  E <- matrix(0, nrow = n_x, ncol = p_fixed)
  for (j in seq_len(p_fixed)) E[j, j] <- 1

  modes_mat <- res$modes  # n_grid x n_x

  for (k in seq_len(n_grid)) {
    L <- Matrix::sparseMatrix(
      i = Q_i[[k]], p = Q_p[[k]], x = Q_x[[k]],
      dims = c(n_x, n_x), index1 = FALSE
    )
    diag_L <- Matrix::diag(L)
    Qk <- L + Matrix::t(L) - Matrix::Diagonal(n_x, diag_L)

    V <- as.matrix(Matrix::solve(Qk, E))
    Sigma_bb <- V[seq_len(p_fixed), , drop = FALSE]
    grid_hessians[[k]] <- solve(Sigma_bb)

    grid_modes[[k]] <- if (!is.null(modes_mat)) {
      as.numeric(modes_mat[k, seq_len(p_fixed)])
    } else {
      rep(NA_real_, p_fixed)
    }
  }

  res$grid_hessians <- grid_hessians
  res$grid_modes    <- grid_modes
  # Strip the verbose CSC scratch fields once Hessians are assembled.
  res$Q_csc_p_per_grid <- NULL
  res$Q_csc_i_per_grid <- NULL
  res$Q_csc_x_per_grid <- NULL
  res$Q_csc_n          <- NULL
  res
}

# --- Multi-block dispatch ---
#
# A multi-block prior is `list(block1, block2, ...)` where each blockN is a
# list with a `type` field (same shape as a single-block prior). At each
# outer-grid point all blocks share a Newton solve; the joint grid is the
# Cartesian product of per-block axes.

.is_multi_block_prior <- function(p) {
  is.list(p) && is.null(p$type) && length(p) > 0 &&
    all(vapply(p, function(x) is.list(x) && !is.null(x$type), logical(1)))
}

# Structural spec for one block sent to cpp_nested_laplace_multi. Grid values
# go in theta_grid separately; this function returns only the type-specific
# data needed to build a LatentBlock (idx vector, sizes, adjacency, etc.).
#
# `block_joint_grid` is the joint-grid sub-matrix (n_joint x block$theta_dim)
# already realised from the Cartesian product. tgmrf needs it to precompute
# Q(theta_k) at every joint-grid row; built-in types ignore the argument
# because they read theta values directly from theta_grid in the C++ side.
.nl_block_spec_for_cpp <- function(p, block_joint_grid = NULL) {
  type <- tolower(p$type)
  if (type %in% c("icar", "bym2", "car_proper")) {
    out <- list(
      type            = type,
      spatial_idx     = as.integer(p$spatial_idx),
      n_spatial_units = as.integer(p$n_spatial_units),
      adj_row_ptr     = as.integer(p$adj_row_ptr),
      adj_col_idx     = as.integer(p$adj_col_idx),
      n_neighbors     = as.integer(p$n_neighbors)
    )
    if (type == "bym2") {
      out$scale_factor <- as.numeric(p$scale_factor %||% 1.0)
    }
    out
  } else if (type %in% c("rw1", "rw2", "ar1")) {
    out <- list(
      type         = type,
      temporal_idx = as.integer(p$temporal_idx),
      n_times      = as.integer(p$n_times)
    )
    if (type == "rw1") out$cyclic <- isTRUE(p$cyclic)
    out
  } else if (type == "iid") {
    list(
      type    = "iid",
      obs_idx = as.integer(p$obs_idx),
      n_units = as.integer(p$n_units)
    )
  } else if (type == "tgmrf") {
    # Precompute Q(theta_k) for every joint-grid row and pack CSC triples +
    # logdet + log p(theta_k). The C++ side reads them directly at grid
    # point k - no R callback during the inner Newton loop. Iterating over
    # block_joint_grid (rather than the block's own grid) is what lets
    # tgmrf compose with other blocks in a multi-block prior: when other
    # blocks vary across joint rows but tgmrf does not, the duplicated Q
    # matrices land at the right joint index.
    #
    # The Q / log_prior(theta) evaluation either goes through the user's R
    # closure (`backend == "r"`, the default) or through the registered C++
    # spec (`backend == "cpp"`). Wire format downstream is identical in both
    # cases.
    if (is.null(block_joint_grid)) {
      stop("Internal error: tgmrf block_spec called without block_joint_grid.",
           call. = FALSE)
    }
    tg <- as.matrix(block_joint_grid)
    n_cells   <- nrow(tg)
    n_latent  <- p$n_latent
    Q_csc_p   <- vector("list", n_cells)
    Q_csc_i   <- vector("list", n_cells)
    Q_csc_x   <- vector("list", n_cells)
    logdet_Q  <- numeric(n_cells)
    log_pi_th <- numeric(n_cells)
    use_cpp <- isTRUE(identical(p$backend, "cpp"))
    if (use_cpp) {
      if (is.null(p$cpp_id) || !nzchar(p$cpp_id)) {
        stop("Internal error: tgmrf block has backend = 'cpp' ",
             "but no cpp_id.", call. = FALSE)
      }
      if (!cpp_tgmrf_registry_has(p$cpp_id)) {
        stop("tgmrf C++ spec '", p$cpp_id, "' is not registered in the ",
             "current R session. Was the user .cpp file sourceCpp'd? ",
             "(Re-load via tgmrf_cpp(cpp_file = ..., id = '", p$cpp_id,
             "').)", call. = FALSE)
      }
    }
    for (k in seq_len(n_cells)) {
      th <- as.numeric(tg[k, ])
      names(th) <- p$theta_names
      if (use_cpp) {
        evk <- cpp_tgmrf_eval(p$cpp_id, th)
        if (as.integer(evk$n) != n_latent) {
          stop("Registered tgmrf_cpp spec '", p$cpp_id,
               "' returned Q of size ", evk$n, " at grid row ", k,
               " but the block was registered with n_latent = ", n_latent,
               ".", call. = FALSE)
        }
        Q_csc_p[[k]] <- as.integer(evk$p)
        Q_csc_i[[k]] <- as.integer(evk$i)
        Q_csc_x[[k]] <- as.numeric(evk$x)
        Qk <- Matrix::sparseMatrix(
          i = Q_csc_i[[k]], p = Q_csc_p[[k]], x = Q_csc_x[[k]],
          dims = c(n_latent, n_latent), index1 = FALSE
        )
        lp <- as.numeric(evk$log_prior)
      } else {
        Qk <- p$Q(th)
        Qk <- methods::as(methods::as(Qk, "generalMatrix"), "CsparseMatrix")
        if (nrow(Qk) != n_latent || ncol(Qk) != n_latent) {
          stop("`Q(theta)` returned ", nrow(Qk), " x ", ncol(Qk),
               " at grid row ", k,
               " but the block was registered with n_latent = ", n_latent, ".",
               call. = FALSE)
        }
        Q_csc_p[[k]] <- as.integer(Qk@p)
        Q_csc_i[[k]] <- as.integer(Qk@i)
        Q_csc_x[[k]] <- as.numeric(Qk@x)
        lp <- as.numeric(p$prior(th))
      }
      # sparse Cholesky log-determinant; falls back to dense det for tiny
      # blocks where Matrix::Cholesky balks.
      ld <- tryCatch({
        ch <- Matrix::Cholesky(methods::as(Qk, "symmetricMatrix"), LDL = FALSE)
        # log|Q| = 2 * sum(log(diag(L))) for the LL' factor.
        2 * Matrix::determinant(ch, sqrt = TRUE, logarithm = TRUE)$modulus
      }, error = function(e) {
        as.numeric(determinant(Qk, logarithm = TRUE)$modulus)
      })
      logdet_Q[k] <- as.numeric(ld)
      if (!is.finite(lp)) {
        stop("`prior(theta)` returned a non-finite value at grid row ", k,
             " (theta = ", paste(format(th, digits = 4), collapse = ", "),
             ").", call. = FALSE)
      }
      log_pi_th[k] <- lp
    }
    list(
      type                     = "tgmrf",
      n_latent                 = as.integer(n_latent),
      obs_idx                  = as.integer(p$obs_idx),
      Q_csc_p_per_grid         = Q_csc_p,
      Q_csc_i_per_grid         = Q_csc_i,
      Q_csc_x_per_grid         = Q_csc_x,
      logdet_Q_per_grid        = as.numeric(logdet_Q),
      log_prior_theta_per_grid = as.numeric(log_pi_th)
    )
  } else {
    stop("Block type '", type, "' is not supported in multi-block priors.",
         call. = FALSE)
  }
}

# Per-block axis grid as a matrix `[n_block_cells x n_axes_for_block]`.
# Single-axis types (icar / rw1 / rw2 / iid) get a 1-column matrix; 2-axis
# types (bym2 / car_proper / ar1) get a 2-column matrix.
.nl_block_axis_grid <- function(p) {
  spec <- .NL_REGISTRY[[tolower(p$type)]]
  if (is.null(spec)) {
    stop("Unknown block type: ", p$type, call. = FALSE)
  }
  p <- spec$defaults(p, list())
  th <- spec$theta(p)
  if (is.matrix(th$grid)) {
    grid <- th$grid
    if (is.null(colnames(grid))) colnames(grid) <- th$names
  } else {
    grid <- matrix(as.numeric(th$grid), ncol = 1)
    colnames(grid) <- th$names
  }
  list(grid = grid, names = colnames(grid), prepared = p)
}

# Soft cap on joint-grid cell count. CCD integration around a pilot mode is
# the standard fix for k >= 3 blocks (see R/ccd_grid.R); plumbing the pilot
# search into the multi-block driver is a follow-up. Until then, warn and
# proceed -- the Cartesian outer integration is still correct, just slower.
.NL_MULTI_GRID_WARN <- 50L
.NL_MULTI_GRID_HARD_CAP <- 2048L

.nl_dispatch_multi <- function(cargs, prior_list) {
  # Inject a default obs_idx for any tgmrf block that didn't supply one.
  # The C++ scatter needs obs_idx[i] -> latent slot for each observation;
  # the canonical "one obs per latent slot" case has N == n_latent and the
  # identity mapping. Heterogeneous (N != n_latent) layouts must supply
  # obs_idx explicitly at tgmrf() construction.
  N_obs <- length(cargs$y)
  for (b in seq_along(prior_list)) {
    blk <- prior_list[[b]]
    if (identical(tolower(blk$type %||% ""), "tgmrf") && is.null(blk$obs_idx)) {
      if (blk$n_latent != N_obs) {
        stop("tgmrf block has n_latent = ", blk$n_latent,
             " but data has N = ", N_obs, " observations. Provide an explicit ",
             "`obs_idx` argument to tgmrf() mapping observations to latent slots.",
             call. = FALSE)
      }
      prior_list[[b]]$obs_idx <- seq_len(N_obs)
    }
  }

  per_block <- lapply(prior_list, .nl_block_axis_grid)
  prepared  <- lapply(per_block, function(x) x$prepared)
  block_grids <- lapply(per_block, function(x) x$grid)

  # Cartesian product of per-block row indices.
  row_counts <- vapply(block_grids, nrow, integer(1))
  idx <- do.call(expand.grid, lapply(row_counts, seq_len))
  n_cells <- nrow(idx)
  if (n_cells > .NL_MULTI_GRID_HARD_CAP) {
    stop(sprintf(
      "Joint multi-block grid has %d cells (hard cap %d). Reduce per-block grid sizes or wait for CCD integration support.",
      n_cells, .NL_MULTI_GRID_HARD_CAP
    ), call. = FALSE)
  }
  if (n_cells > .NL_MULTI_GRID_WARN) {
    warning(sprintf(
      "Joint multi-block grid has %d cells (>%d). Each cell costs one inner Newton solve; reduce per-block grid sizes if this is slow. CCD integration is a follow-up.",
      n_cells, .NL_MULTI_GRID_WARN
    ), call. = FALSE)
  }

  # Concatenate per-block axis grids into the joint theta_grid.
  joint_grid <- do.call(cbind, lapply(seq_along(block_grids), function(b) {
    block_grids[[b]][idx[[b]], , drop = FALSE]
  }))
  axis_counts  <- vapply(block_grids, ncol, integer(1))
  axis_offsets <- as.integer(c(0L, cumsum(axis_counts)))
  # Block-prefix the axis names so they're unambiguous (e.g. block1.tau).
  axis_names <- unlist(lapply(seq_along(block_grids), function(b) {
    paste0("b", b, ".", colnames(block_grids[[b]]))
  }))
  colnames(joint_grid) <- axis_names

  blocks_spec <- lapply(seq_along(prepared), function(b) {
    cols <- (axis_offsets[b] + 1L):axis_offsets[b + 1L]
    block_joint <- joint_grid[, cols, drop = FALSE]
    .nl_block_spec_for_cpp(prepared[[b]], block_joint)
  })

  out <- cpp_nested_laplace_multi(
    y           = cargs$y,
    n           = cargs$n,
    X           = cargs$X,
    re_idx      = cargs$re_idx,
    n_re_groups = cargs$n_re_groups,
    sigma_re    = cargs$sigma_re,
    blocks_spec = blocks_spec,
    theta_grid  = joint_grid,
    axis_offsets = axis_offsets,
    family      = cargs$family,
    phi         = cargs$phi,
    max_iter    = cargs$max_iter,
    tol         = cargs$tol,
    n_threads   = cargs$n_threads,
    x_init_nullable = cargs$x_init_nullable,
    store_Q     = isTRUE(cargs$store_Q)
  )

  out$theta_grid   <- joint_grid
  out$theta_names  <- axis_names
  out$axis_offsets <- axis_offsets
  out$weights      <- .nl_normalise_weights(out$log_marginal)
  out <- .nl_posterior_moments_multi(out, prepared, axis_offsets, joint_grid)
  out$blocks       <- prepared
  out
}

# Posterior moments for multi-block grids. Two flavours:
#  * joint moments: across all axes (same as single-block 2D scatter).
#  * per-block marginal moments: integrate out the other blocks' axes.
.nl_posterior_moments_multi <- function(out, prepared, axis_offsets, joint_grid) {
  w  <- out$weights
  tg <- joint_grid
  out$theta_mean <- as.numeric(crossprod(w, tg))
  names(out$theta_mean) <- colnames(tg)
  out$theta_sd <- sqrt(pmax(0, as.numeric(crossprod(w, tg^2)) -
                              out$theta_mean^2))
  names(out$theta_sd) <- colnames(tg)

  # Per-block marginals: for each block, sum weights over rows that share the
  # same per-block axis values, then take weighted mean/sd within those rows.
  # Because the joint grid is a Cartesian product, the "rows that share this
  # block's values" form n_block_rows groups indexed by idx[, b]. The marginal
  # weight for group g is sum of joint weights over its rows.
  B <- length(prepared)
  per_block_moments <- vector("list", B)
  for (b in seq_len(B)) {
    cols <- (axis_offsets[b] + 1L):axis_offsets[b + 1L]
    sub <- joint_grid[, cols, drop = FALSE]
    block_mean <- as.numeric(crossprod(w, sub))
    block_sd   <- sqrt(pmax(0, as.numeric(crossprod(w, sub^2)) - block_mean^2))
    # Use bare per-block axis names (e.g. "tau", "rho") instead of the
    # joint-grid-prefixed "b<N>.tau" — the block index is already implicit
    # in the list position.
    bare_names <- .nl_block_axis_grid(prepared[[b]])$names
    names(block_mean) <- bare_names
    names(block_sd)   <- bare_names
    per_block_moments[[b]] <- list(
      type      = tolower(prepared[[b]]$type),
      mean      = block_mean,
      sd        = block_sd,
      axis_cols = cols
    )
  }
  out$block_moments <- per_block_moments
  out
}

#' Build a `prior` list for [tulpa_nested_laplace()] from a tulpa spec object
#'
#' @description
#' Validates a `tulpa_temporal` or `tulpa_spatial` specification against
#' `data`, then converts it to the prior list shape consumed by
#' [tulpa_nested_laplace()]. Mainly an internal helper for callers that already
#' have a fitted spec; users typically pass `spec` + `data` directly to
#' `tulpa_nested_laplace()` instead.
#'
#' Supported spec types:
#'  * `tulpa_temporal` with `type ∈ {"rw1", "rw2", "ar1"}`
#'  * `tulpa_spatial` with `type ∈ {"car", "icar", "car_proper", "bym2"}`.
#'    Proper CAR (rho estimated) is converted to a 2D grid over (tau, rho)
#'    using the spec's eigenvalue-derived `rho_bounds`.
#'    Continuous-spatial GP / SPDE specs need their own entry — see
#'    [cpp_nested_laplace_spde()].
#'
#' @param spec A `tulpa_temporal` or `tulpa_spatial` object.
#' @param data Data frame the spec resolves time/group/site indices against.
#' @return A `prior` list ready for [tulpa_nested_laplace()].
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
    stop("tulpa_nested_laplace() supports temporal types rw1, rw2, ar1; got '",
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
  if (type %in% c("car", "icar")) {
    backend <- "icar"
  } else if (type == "bym2") {
    backend <- "bym2"
  } else if (type == "car_proper") {
    backend <- "car_proper"
  } else {
    stop("tulpa_nested_laplace() does not yet support spatial type '", type,
         "'. Use BYM2/ICAR/proper CAR for areal models or ",
         "cpp_nested_laplace_spde for SPDE.", call. = FALSE)
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
  if (backend == "car_proper") {
    rb <- spec$rho_bounds
    if (is.null(rb)) rb <- c(0, 1)
    out$rho_bounds <- as.numeric(rb)
  }
  out
}
