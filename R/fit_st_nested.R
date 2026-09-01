# fit_st_nested.R
# ------------------------------------------------------------------------------
# Front-door driver for the additive spatiotemporal nested-Laplace kernels
# (cpp_nested_laplace_st_{icar,bym2,car_proper}). Each kernel fits a GLM with an
# additive areal spatial field + a temporal field (rw1 / rw2 / ar1), integrating
# jointly over the spatial precision, the temporal precision, and (for ar1) the
# temporal autocorrelation on a hyperparameter grid. The kernels return the
# per-cell log-marginal + latent modes + the per-cell precision (Q_csc), the same
# output shape the areal single-block path emits, so this driver reuses the
# shared nested-Laplace post-processing (weight normalisation, grid-Hessian
# extraction for the fixed-effect marginal SE) and the generic tulpa_fit
# accessors. Previously these kernels were reachable only from consumer packages

# ------------------------------------------------------------------------------

# Log-spaced positive grid, floored at 2 points.
.st_log_grid <- function(lo, hi, n) {
  n <- max(2L, as.integer(n))
  exp(seq(log(lo), log(hi), length.out = n))
}

# Integration coordinates of the spatiotemporal outer grid, declared where the
# grid is built. `.st_log_grid()` lays the two precisions out log-spaced and the
# AR1 correlation evenly, so those are the coordinates their cell widths are
# measured on and an unrefined grid carries equal weights. A recentred grid
# keeps the same coordinates and only moves the nodes.
.nl_st_axis_specs <- function(theta_grid) {
    if (is.null(theta_grid) || is.null(colnames(theta_grid))) return(NULL)
    tg <- as.matrix(theta_grid)
    lapply(colnames(tg), function(a) {
        lv <- sort(unique(as.numeric(tg[, a])))
        log_scale <- startsWith(a, "tau")
        spec <- hyper_axis_spec(name = a, grid = lv, log_scale = log_scale,
                                bounds = if (log_scale) c(0, Inf) else NULL,
                                refinable = FALSE)
        pos <- lv[lv > 0]
        if (log_scale && length(pos) >= 2L)
            spec$slab_bounds <- exp(.hyper_default_coord_bounds(log(pos)))
        spec
    })
}


#' Fit an additive spatiotemporal GLM by nested Laplace
#'
#' @description
#' Fits `y ~ X beta + u_spatial[s] + v_temporal[t]` with an areal spatial field
#' (`icar` / `bym2` / `car_proper`) and a temporal field (`rw1` / `rw2` / `ar1`),
#' integrating the spatial precision, temporal precision, and (for `ar1`) the
#' temporal autocorrelation over a hyperparameter grid via the
#' `cpp_nested_laplace_st_*` kernels. The fixed-effect posterior is the
#' grid-marginalised mixture; the spatial and temporal field posterior means are
#' the grid-weighted latent modes.
#'
#' @param y Response vector.
#' @param X Fixed-effects design matrix (`nrow(X) == length(y)`).
#' @param spatial_idx Integer per-observation spatial-unit index (1-based).
#' @param adjacency Spatial adjacency (a symmetric 0/1 matrix or `sparseMatrix`).
#' @param temporal_idx Integer per-observation time index (1-based).
#' @param n_times Number of distinct time points.
#' @param spatial_type `"icar"` (default), `"bym2"`, or `"car_proper"`.
#' @param temporal_type `"ar1"` (default), `"rw1"`, or `"rw2"`.
#' @param family Response family (see [family_names()]).
#' @param n_trials Binomial denominators, or `NULL` (= 1).
#' @param phi Dispersion passed to the family, in the compiled-kernel
#'   convention: for `gaussian` / `lognormal` this is the residual SD (the
#'   variance is `phi^2`), for `neg_binomial_2` the size, `gamma` the shape,
#'   `beta` the precision, `t` the scale; `binomial` / `poisson` ignore it.
#'   The [tulpa()] and [tulpa_laplace()] front doors take the residual
#'   VARIANCE instead and convert at the kernel boundary, so one model is
#'   `phi` there and `sqrt(phi)` here.
#' @param cyclic Logical; wrap the temporal field (seasonal). Default `FALSE`.
#' @param re_idx,n_re_groups,sigma_re Optional single iid random-intercept term
#'   alongside the fields (conditioned on `sigma_re`); `n_re_groups = 0` (default)
#'   is no RE term.
#' @param control A list of numerical / grid knobs: `n_grid_spatial`,
#'   `n_grid_temporal` (default 4 each), `n_grid_rho` (ar1 only, default 3),
#'   `tau_lower` / `tau_upper` (precision grid bounds, default 0.25 / 16),
#'   `rho_lower` / `rho_upper` (ar1 grid, default 0.1 / 0.9), `max_iter`, `tol`,
#'   `n_threads`, `auto_recenter` (default `TRUE`; `FALSE` holds the grid
#'   exactly as specified -- the per-axis policy names
#'   [tulpa_nested_laplace()] takes are refused here with an error, since this
#'   driver recentres on the grid's collapsed-edge regime rather than on a
#'   per-axis rail).
#'
#'   The `(tau_lower, tau_upper)` span (and, for `ar1`, `(rho_lower,
#'   rho_upper)`) is a starting axis, not a hard ceiling:
#'   when the fitted precision (or, for `ar1`, autocorrelation) posterior
#'   mode rails a boundary node (`pareto_k_regime = "collapsed_edge"`, see
#'   below), the driver fits a mode-Hessian via a derivative-free `optim()`
#'   over the collapsed grid and refits a grid re-centred on it (one
#'   attempt).
#'
#'   A grid knob PINS the axes it shapes, and a pin always wins -- but
#'   pinning is decided by value, not by presence: a knob
#'   set to the engine's own default, or marked with [auto_grid()], expresses
#'   no preference and leaves its axes free. That is what lets a wrapper
#'   package thread its own `n_grid`-style argument through `control` without
#'   silently disabling the recenter for every fit it makes. Pinning is also
#'   per axis: `tau_lower` / `tau_upper` hold the two precision axes,
#'   `n_grid_spatial` / `n_grid_temporal` one each, and `n_grid_rho` /
#'   `rho_lower` / `rho_upper` the `ar1` autocorrelation axis, so pinning one
#'   axis leaves the others free to be recentred. A pinned axis keeps its
#'   nodes exactly and is named in `outer_grid_pinned_axes`; with EVERY axis
#'   pinned the recenter declines outright and
#'   `outer_grid_recenter_declined` records which reason applied.
#'
#' @return A `tulpa_fit` (subclass `tulpa_nested_laplace`) carrying the
#'   fixed-effect posterior (`draws` via the grid mixture), `spatial_effects`,
#'   `temporal_effects`, `log_marginal`, `weights`, and `theta_grid` over
#'   `(tau_spatial, tau_temporal, rho)`. Also carries `pareto_k_regime`
#'   (`"spread"` / `"collapsed_interior"` / `"collapsed_edge"`, see
#'   [tulpa_nested_laplace_joint()]'s return docs for the definition) and
#'   `outer_grid_placement` (`"fixed"` or `"auto_recentered"`) plus, on a
#'   `"fixed"` placement, `outer_grid_recenter_declined`
#'   (`"grid_knobs_overridden"` / `"grid_not_collapsed"` /
#'   `"no_usable_curvature"` / `"refit_failed"` / `"sd_ceiling_unresolved"` /
#'   `"sd_floor_unresolved"`). A recentred fit also carries
#'   `outer_grid_pinned_axes`, the axes whose knobs were pinned and whose
#'   nodes were therefore kept, and `outer_grid_recenter_sd_clamp` /
#'   `_sd_raw` / `_sd_used` -- per moved axis, which mode-SD bound the
#'   placement hit, the SD the stencil measured, and the SD the axis was laid
#'   from. A bound-decline is PER AXIS here: the axes the
#'   mode-find did resolve are still re-placed, and
#'   `outer_grid_recenter_sd_declined` names the ones that kept their incoming
#'   nodes and on which bound, so a partially re-placed grid is not read as a
#'   fully re-placed one. With every free axis declined the pass reports the
#'   grid as the fixed one it still is.
#'
#' @seealso [tulpa()] (front door), [tulpa_nested_laplace()] (single field).
#' @examples
#' \donttest{
#' set.seed(1)
#' n_s <- 16L; n_t <- 8L; N <- 400L
#' adj <- matrix(0, n_s, n_s)
#' for (i in 1:(n_s - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1
#' s <- sample(n_s, N, TRUE); tt <- sample(n_t, N, TRUE)
#' us <- as.numeric(scale(cumsum(rnorm(n_s)))); vt <- as.numeric(scale(cumsum(rnorm(n_t))))
#' x <- rnorm(N)
#' y <- rbinom(N, 1, plogis(0.2 + 0.5 * x + 0.7 * us[s] + 0.6 * vt[tt]))
#' fit <- fit_st_nested(y, cbind(1, x), s, adj, tt, n_t, family = "binomial")
#' }
#' @export
fit_st_nested <- function(y, X, spatial_idx, adjacency, temporal_idx, n_times,
                          spatial_type = c("icar", "bym2", "car_proper"),
                          temporal_type = c("ar1", "rw1", "rw2"),
                          family = "binomial", n_trials = NULL, phi = 1.0,
                          cyclic = FALSE,
                          re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                          control = list()) {
  spatial_type  <- match.arg(spatial_type)
  temporal_type <- match.arg(temporal_type)
  X <- as.matrix(X)
  vd <- .validate_glm_design(y, X, n_trials, "fit_st_nested")
  N  <- vd$N
  n_trials <- vd$n_trials
  y <- as.numeric(y)

  if (length(spatial_idx) != N || length(temporal_idx) != N) {
    stop("`spatial_idx` / `temporal_idx` must have length length(y).",
         call. = FALSE)
  }
  csr <- adjacency_to_csr_tulpa(adjacency)
  n_s <- nrow(as.matrix(adjacency))
  if (max(spatial_idx) > n_s || min(spatial_idx) < 1L) {
    stop("`spatial_idx` must be 1-based indices in [1, nrow(adjacency)].",
         call. = FALSE)
  }
  if (max(temporal_idx) > n_times || min(temporal_idx) < 1L) {
    stop("`temporal_idx` must be 1-based indices in [1, n_times].", call. = FALSE)
  }
  if (is.null(re_idx)) re_idx <- rep(0, N)

  # Hyperparameter grid: spatial precision x temporal precision x (ar1) rho.
  n_gs   <- as.integer(control$n_grid_spatial  %||% .nl_st_default("n_spatial"))
  n_gt   <- as.integer(control$n_grid_temporal %||% .nl_st_default("n_temporal"))
  n_grho <- if (temporal_type == "ar1") as.integer(control$n_grid_rho %||% .nl_st_default("n_rho")) else 1L
  tau_lo <- control$tau_lower %||% .nl_st_default("tau_lower")
  tau_hi <- control$tau_upper %||% .nl_st_default("tau_upper")
  ts_axis  <- .st_log_grid(tau_lo, tau_hi, n_gs)
  tt_axis  <- .st_log_grid(tau_lo, tau_hi, n_gt)
  rho_axis <- if (temporal_type == "ar1") {
    seq(control$rho_lower %||% .nl_st_default("rho_lower"),
        control$rho_upper %||% .nl_st_default("rho_upper"), length.out = n_grho)
  } else 0.0
  grid <- expand.grid(tau_spatial = ts_axis, tau_temporal = tt_axis, rho = rho_axis)

  kernel <- switch(spatial_type,
                   icar        = cpp_nested_laplace_st_icar,
                   bym2        = cpp_nested_laplace_st_bym2,
                   car_proper  = cpp_nested_laplace_st_car_proper)

  kargs <- list(
    y = y, n = n_trials, X = X,
    re_idx = as.numeric(re_idx), n_re_groups = as.integer(n_re_groups),
    sigma_re = as.numeric(sigma_re),
    spatial_idx = as.integer(spatial_idx), n_spatial_units = as.integer(n_s),
    adj_row_ptr = as.integer(csr$row_ptr), adj_col_idx = as.integer(csr$col_idx),
    n_neighbors = as.integer(csr$n_neighbors),
    temporal_idx = as.integer(temporal_idx), n_times = as.integer(n_times),
    tau_spatial_grid = as.numeric(grid$tau_spatial),
    temporal_type = temporal_type,
    tau_temporal_grid = as.numeric(grid$tau_temporal),
    rho_temporal_grid = as.numeric(grid$rho),
    cyclic = isTRUE(cyclic), family = family, phi = as.numeric(phi),
    max_iter = as.integer(control$max_iter %||% 50L),
    tol = as.numeric(control$tol %||% 1e-6),
    n_threads = as.integer(control$n_threads %||% 1L),
    store_Q = TRUE
  )
  # bym2 needs a scale_factor; car_proper an rho_spatial_grid. Keep to the
  # kernels' documented defaults for those extra axes here (icar is the base).
  rho_spatial_val <- control$rho_spatial %||% 0.9
  if (spatial_type == "car_proper") {
    kargs$rho_spatial_grid <- rep(rho_spatial_val, nrow(grid))
  }
  out <- do.call(kernel, kargs)

  out$theta_grid  <- as.matrix(grid)
  out$theta_names <- colnames(grid)
  st_specs <- .nl_st_axis_specs(out$theta_grid)
  out$log_quad     <- .hyper_log_quad_weights(out$theta_grid, st_specs)
  out$axis_support <- .hyper_grid_supports(out$theta_grid, st_specs)
  out$weights <- .nl_normalise_weights_safe(out$log_marginal,
                                            "spatiotemporal grid",
                                            log_quad = out$log_quad)
  # Outer-grid collapse visibility + recenter:
  # tau_lower/tau_upper's default [0.25, 16] span (and, for ar1, the default
  # rho_lower/rho_upper) is a starting axis, not a hard ceiling, the same
  # contract every other nested-Laplace family's default grid carries. A
  # from-scratch mode-Hessian recenter-and-refit engages when the grid
  # collapses onto a boundary (`pareto_k_regime = "collapsed_edge"`) and no
  # grid knob was explicitly overridden; see `.st_auto_grid_rescue()`
  # (R/fit_st_nested_auto_grid.R) for the optim()-based mode-find it uses.
  out <- .joint_attach_pareto_k_regime(out)
  out <- .st_auto_grid_rescue(out, kernel, kargs, spatial_type, temporal_type,
                              n_gs, n_gt, n_grho, tau_lo, tau_hi, control,
                              rho_spatial_val = rho_spatial_val)
  out$outer_grid_placement <- out$outer_grid_placement %||% "fixed"
  # Within-cell construction for the reported per-axis intervals; the default
  # is `.nl_diag("within_cell")`.
  within_cell <- .nl_within_cell_mode(control$within_cell)
  out <- .nl_posterior_moments(out, "st", within = within_cell)
  out <- .nl_attach_grid_hessians(out, ncol(X))

  # Grid-marginalised field posterior means: the latent block after the fixed
  # effects is [spatial (n_s), temporal (n_times)].
  p <- ncol(X)
  w <- out$weights
  sp_cols <- p + seq_len(n_s)
  te_cols <- p + n_s + seq_len(n_times)
  out$spatial_effects  <- as.numeric(crossprod(w, out$modes[, sp_cols, drop = FALSE]))
  out$temporal_effects <- as.numeric(crossprod(w, out$modes[, te_cols, drop = FALSE]))
  out$spatial_type  <- spatial_type
  out$temporal_type <- temporal_type
  out$N <- N
  out$y <- y
  out$model_matrix <- X

  .finalize_fit(out, backend = "nested_laplace",
                n_fixed = p, fixed_names = colnames(X),
                extra_class = c("tulpa_st_nested", "tulpa_nested_laplace", "list"))
}
