# fit_st_nested.R
# ------------------------------------------------------------------------------
# Front-door driver for the additive spatiotemporal nested-Laplace kernels
# (cpp_nested_laplace_st_{icar,bym2,car_proper,hsgp,nngp}). Each kernel fits a
# GLM with an additive spatial field (areal: icar/bym2/car_proper; continuous:
# hsgp/nngp) + a temporal field (rw1 / rw2 / ar1), integrating jointly over the
# spatial hyperparameter(s), the temporal precision, and (for ar1) the temporal
# autocorrelation on a hyperparameter grid. The kernels return the
# per-cell log-marginal + latent modes + the per-cell precision (Q_csc), the same
# output shape the areal single-block path emits, so this driver reuses the
# shared nested-Laplace post-processing (weight normalisation, grid-Hessian
# extraction for the fixed-effect marginal SE) and the generic tulpa_fit
# accessors. Previously these kernels were reachable only from consumer packages
# (gcol33/tulpa#807 wired hsgp / nngp through this same door).

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
.nl_st_axis_specs <- function(theta_grid, folded_axes = NULL) {
    if (is.null(theta_grid) || is.null(colnames(theta_grid))) return(NULL)
    tg <- as.matrix(theta_grid)
    lapply(colnames(tg), function(a) {
        lv <- sort(unique(as.numeric(tg[, a])))
        # bym2's spatial axis is `sigma_spatial` (an SD), not a precision, but
        # it is the same positive log-scale coordinate as `tau_spatial` /
        # `tau_temporal` (gcol33/tulpa#776). hsgp's `lengthscale` and nngp's
        # `phi_gp` are the same positive log-scale coordinate too (matching the
        # single-field registry's `range` / `phi_gp` / `lengthscale` axes,
        # gcol33/tulpa#807); `sigma2` (hsgp/nngp's field variance) is covered by
        # the `sigma` prefix already.
        log_scale <- startsWith(a, "tau") || startsWith(a, "sigma") ||
          a %in% c("lengthscale", "phi_gp")
        spec <- hyper_axis_spec(name = a, grid = lv, log_scale = log_scale,
                                bounds = if (log_scale) c(0, Inf) else NULL,
                                refinable = FALSE)
        pos <- lv[lv > 0]
        if (log_scale && length(pos) >= 2L && !a %in% folded_axes)
            spec$slab_bounds <- exp(.hyper_default_coord_bounds(log(pos)))
        spec
    })
}

# The spatiotemporal grid's hyperprior (`R/hyperprior_default.R`): under
# `"proper"` the PC prior on both precisions and, for ar1, the uniform on the
# autocorrelation's domain; under `"flat"` none of them. `rho` is always the
# TEMPORAL ar1 autocorrelation (see `fit_st_nested()`'s grid columns); bym2's
# spatial mixing weight is a distinct column, `rho_spatial`, routed to the
# same `bym2:rho` Uniform(0, 1) density `tulpa_nested_laplace(type = "bym2")`
# gives it (gcol33/tulpa#776).
#
# `block` carries the coordinate context a continuous (hsgp/nngp) field's
# `lengthscale` / `phi_gp` range axis needs to anchor its PC prior
# (`.hp_range_anchor()` reads `block$coords`, matching what
# `.spatial_spec_to_nl_prior()` attaches to the single-field prior block);
# areal fields pass none and get the same `range_extent_unknown` decline any
# other anchor-less range axis would.
.st_log_hyperprior <- function(theta_grid, axes, hyperprior = "proper", block = list()) {
    tg <- as.matrix(theta_grid)
    .hp_collect(tg, function(a) {
        if (identical(a, "rho")) {
            .hp_axis_prior("rho", list(type = "ar1"), hyperprior = hyperprior)
        } else if (identical(a, "rho_spatial")) {
            .hp_axis_prior("rho", list(type = "bym2"), hyperprior = hyperprior)
        } else .hp_axis_prior(a, block, hyperprior = hyperprior)
    }, axes = axes)
}

# Fold the hyperprior into a spatiotemporal kernel result and attach its cell
# measure, weights and evidence. The one tail behind the first solve and the
# placement refit.
.st_attach_outer_integration <- function(out, theta_grid, hyperprior = "proper",
                                         block = list()) {
    out$theta_grid  <- as.matrix(theta_grid)
    out$theta_names <- colnames(out$theta_grid)
    out <- .nl_fold_hyperprior(
        out, list(.st_log_hyperprior(out$theta_grid,
                                     axes = .hp_integrated_axes(out$theta_grid),
                                     hyperprior = hyperprior, block = block)))
    st_specs <- .nl_st_axis_specs(out$theta_grid,
                                  folded_axes = out$log_hyperprior_axes)
    out$log_quad     <- .hyper_log_quad_weights(out$theta_grid, st_specs)
    out$axis_support <- .hyper_grid_supports(out$theta_grid, st_specs)
    out$weights <- .nl_normalise_weights_safe(out$log_marginal,
                                              "spatiotemporal grid",
                                              log_quad = out$log_quad)
    .nl_attach_evidence(out, out$theta_grid, st_specs)
}


#' Fit an additive spatiotemporal GLM by nested Laplace
#'
#' @description
#' Fits `y ~ X beta + u_spatial[s] + v_temporal[t]` with a spatial field --
#' areal (`icar` / `bym2` / `car_proper`) or continuous (`hsgp` / `nngp`) --
#' and a temporal field (`rw1` / `rw2` / `ar1`), integrating the spatial
#' hyperparameter(s), temporal precision, and (for `ar1`) the temporal
#' autocorrelation over a hyperparameter grid via the
#' `cpp_nested_laplace_st_*` kernels. The fixed-effect posterior is the
#' grid-marginalised mixture; the spatial and temporal field posterior means are
#' the grid-weighted latent modes.
#'
#' @param y Response vector.
#' @param X Fixed-effects design matrix (`nrow(X) == length(y)`).
#' @param spatial_idx Integer per-observation spatial-unit index (1-based).
#'   For `spatial_type = "icar"/"bym2"/"car_proper"`, an areal unit in
#'   `[1, nrow(adjacency)]`; for `"nngp"`, a location in `[1, nrow(coords)]`.
#'   Ignored for `"hsgp"` (the field is evaluated directly at each
#'   observation's own coordinates); pass any placeholder (e.g. `seq_len(N)`).
#' @param adjacency Spatial adjacency (a symmetric 0/1 matrix or `sparseMatrix`).
#'   Required for `spatial_type = "icar"/"bym2"/"car_proper"`; ignored (pass
#'   `NULL`) for `"hsgp"`/`"nngp"`, which take `coords` instead.
#' @param temporal_idx Integer per-observation time index (1-based).
#' @param n_times Number of distinct time points.
#' @param spatial_type `"icar"` (default), `"bym2"`, `"car_proper"`, `"hsgp"`
#'   (Hilbert-space GP basis), or `"nngp"` (nearest-neighbour GP).
#' @param temporal_type `"ar1"` (default), `"rw1"`, or `"rw2"`.
#' @param family Response family (see [family_names()]).
#' @param n_trials Binomial denominators, or `NULL` (= 1).
#' @template phi
#' @param cyclic Logical; wrap the temporal field (seasonal). Default `FALSE`.
#' @param re_idx,n_re_groups,sigma_re Optional single iid random-intercept term
#'   alongside the fields (conditioned on `sigma_re`); `n_re_groups = 0` (default)
#'   is no RE term.
#' @template hyperprior
#' @param coords Coordinate matrix for a continuous spatial field, required
#'   when `spatial_type` is `"hsgp"` or `"nngp"` (ignored otherwise). For
#'   `"hsgp"`, an `N x 2` matrix (one row per observation, matching
#'   `spatial_gp(approx = "hsgp")`'s basis convention -- the basis is built by
#'   `cpp_hsgp_basis_2d()`, 2D only). For `"nngp"`, an `n_spatial x d` matrix
#'   of unique locations that `spatial_idx` indexes into (any `d`, matching
#'   [spatial_gp()]'s NNGP convention).
#' @param nn Number of nearest neighbours per location, `spatial_type = "nngp"`
#'   only. Default 10 (clamped to `nrow(coords) - 1`).
#' @param cov_type Integer NNGP covariance code (`spatial_type = "nngp"` only):
#'   0 = exponential, 1 = Matern 3/2, 2 = Matern 5/2 (default), 3 = Gaussian.
#' @param hsgp_m,hsgp_c Hilbert-space GP basis size (per dimension, default 6)
#'   and boundary factor (default 1.5), `spatial_type = "hsgp"` only -- the
#'   same parameterisation and defaults as `spatial_gp(approx = "hsgp")`.
#' @param control A list of numerical / grid knobs: `n_grid_spatial`,
#'   `n_grid_temporal` (default 4 each), `n_grid_rho` (ar1 only, default 3),
#'   `tau_lower` / `tau_upper` (icar / car_proper precision grid bounds,
#'   default 0.25 / 16), `sigma_lower` / `sigma_upper` (bym2's spatial SD grid
#'   bounds in place of `tau_lower` / `tau_upper`, default 0.1 / 3 --
#'   `n_grid_spatial` sizes this axis too; its mixing-weight axis
#'   `rho_spatial` is always the fixed default node set and is not a `control`
#'   knob), `rho_lower` / `rho_upper` (ar1 grid, default 0.1 / 0.9), `max_iter`,
#'   `tol`, `n_threads`, `auto_recenter` (default `TRUE`; `FALSE` holds the
#'   grid exactly as specified -- the per-axis policy names
#'   [tulpa_nested_laplace()] takes are refused here with an error, since this
#'   driver recentres on the grid's collapsed-edge regime rather than on a
#'   per-axis rail; declines outright for `spatial_type = "bym2"`, whose
#'   (sigma, rho) spatial axes this recenter has no transform for yet, and for
#'   `"hsgp"`/`"nngp"`, same reason), `rho_spatial` (the proper-CAR mixing
#'   value the `car_proper` axis is held
#'   at, default `.NL_ST_GRID$rho_spatial`; unrelated to bym2's own integrated
#'   `rho_spatial` grid axis) and `within_cell` (`"box_uniform"` / `"chord"`,
#'   the within-cell construction the reported per-axis intervals are read
#'   with; defaults to
#'   `.NL_DIAG$within_cell`, as on every other nested door).
#'
#'   For `spatial_type = "hsgp"`/`"nngp"`, the spatial axes are the field
#'   variance (`sigma2`) paired with the lengthscale (`lengthscale`) or NNGP
#'   range (`phi_gp`), read off the same shared default bounds the
#'   single-field `spatial_gp()` path uses (`gp_var` / `gp_lengthscale`,
#'   `R/settings.R`) -- there is no separate `sigma2_lower`/`upper` knob here,
#'   only `n_grid_spatial`, which sizes the pair as it does for the areal
#'   families.
#'
#'   The `(tau_lower, tau_upper)` span (and, for `ar1`, `(rho_lower,
#'   rho_upper)`) is a starting axis, not a hard ceiling:
#'   when the fitted precision (or, for `ar1`, autocorrelation) posterior
#'   mode rails a boundary node (that axis's own marginal is maximal there,
#'   or the whole grid collapsed onto it: `pareto_k_regime =
#'   "collapsed_edge"`, see below), the driver fits a mode-Hessian via a
#'   derivative-free `optim()` over the grid and refits a grid re-centred on
#'   it (one attempt).
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
#'   `temporal_effects`, `log_marginal`, `weights`, `theta_grid` over
#'   `(tau_spatial, tau_temporal, rho)` for an areal spatial field or
#'   `(sigma2, lengthscale, tau_temporal, rho)` / `(sigma2, phi_gp,
#'   tau_temporal, rho)` for `"hsgp"` / `"nngp"`, and `family` / `n_trials` /
#'   `phi` (the
#'   R-level convention), which is what [fitted()], [residuals()],
#'   [posterior_predict()] and [simulate()] read the response family from.
#'   Also carries `pareto_k_regime`
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
                          spatial_type = c("icar", "bym2", "car_proper",
                                          "hsgp", "nngp"),
                          temporal_type = c("ar1", "rw1", "rw2"),
                          family = "binomial", n_trials = NULL, phi = 1.0,
                          cyclic = FALSE,
                          re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
                          hyperprior = c("proper", "flat"),
                          control = list(),
                          coords = NULL, nn = 10L, cov_type = 2L,
                          hsgp_m = 6L, hsgp_c = 1.5) {
  # Every other nested door checks its control keys; this one did not, so a
  # misspelling (`rho_spatail`, `n_thread`) was accepted in silence and the fit
  # ran at the default (gcol33/tulpa#673).
  tulpa_check_control(control, .CONTROL_KEYS$st_nested, "fit_st_nested")
  spatial_type  <- match.arg(spatial_type)
  temporal_type <- match.arg(temporal_type)
  hyperprior    <- .hp_choice(match.arg(hyperprior))
  family <- .canonical_family(family)
  .family_or_stop(family)
  X <- as.matrix(X)
  vd <- .validate_glm_design(y, X, n_trials, "fit_st_nested")
  N  <- vd$N
  n_trials <- vd$n_trials
  y <- as.numeric(y)

  # Areal (icar/bym2/car_proper) is addressed by adjacency + a per-obs unit
  # index; nngp by a coordinate matrix of unique locations + the same kind of
  # per-obs unit index; hsgp evaluates its basis directly at each
  # observation's own coordinates, so it carries no unit index at all
  # (gcol33/tulpa#807).
  is_areal <- spatial_type %in% c("icar", "bym2", "car_proper")

  if (length(temporal_idx) != N) {
    stop("`temporal_idx` must have length length(y).", call. = FALSE)
  }
  if (max(temporal_idx) > n_times || min(temporal_idx) < 1L) {
    stop("`temporal_idx` must be 1-based indices in [1, n_times].", call. = FALSE)
  }
  if (is.null(re_idx)) re_idx <- rep(0, N)

  basis <- NULL
  if (is_areal) {
    if (length(spatial_idx) != N) {
      stop("`spatial_idx` must have length length(y).", call. = FALSE)
    }
    csr <- adjacency_to_csr_tulpa(adjacency)
    n_s <- nrow(as.matrix(adjacency))
    if (max(spatial_idx) > n_s || min(spatial_idx) < 1L) {
      stop("`spatial_idx` must be 1-based indices in [1, nrow(adjacency)].",
           call. = FALSE)
    }
  } else if (spatial_type == "nngp") {
    if (is.null(coords)) {
      stop("`coords` is required for spatial_type = 'nngp' (an n_spatial x d ",
           "matrix of unique locations `spatial_idx` indexes into).",
           call. = FALSE)
    }
    if (length(spatial_idx) != N) {
      stop("`spatial_idx` must have length length(y).", call. = FALSE)
    }
    coords <- as.matrix(coords)
    storage.mode(coords) <- "double"
    n_s <- nrow(coords)
    if (max(spatial_idx) > n_s || min(spatial_idx) < 1L) {
      stop("`spatial_idx` must be 1-based indices in [1, nrow(coords)].",
           call. = FALSE)
    }
    nn <- min(as.integer(nn), n_s - 1L)
    ni <- compute_nngp_neighbors(coords, nn)
  } else {
    # hsgp
    if (is.null(coords)) {
      stop("`coords` is required for spatial_type = 'hsgp' (an N x 2 matrix, ",
           "one row per observation).", call. = FALSE)
    }
    coords <- .coords_2col(as.matrix(coords),
                           "fit_st_nested(spatial_type = 'hsgp')")
    if (nrow(coords) != N) {
      stop("`coords` must have one row per observation (", N, ") for ",
           "spatial_type = 'hsgp'; got ", nrow(coords), " row(s).",
           call. = FALSE)
    }
    # Laplacian basis built by cpp_hsgp_basis_2d (setup_hsgp_2d, the single
    # source of truth) -- the same call `.spatial_spec_to_nl_prior()` makes for
    # the single-field HSGP path.
    basis <- cpp_hsgp_basis_2d(coords, as.integer(hsgp_m), as.numeric(hsgp_c))
  }

  # Hyperparameter grid: spatial (family-specific) x temporal precision x
  # (ar1) rho.
  n_gs   <- as.integer(control$n_grid_spatial  %||% .nl_st_default("n_spatial"))
  n_gt   <- as.integer(control$n_grid_temporal %||% .nl_st_default("n_temporal"))
  n_grho <- if (temporal_type == "ar1") as.integer(control$n_grid_rho %||% .nl_st_default("n_rho")) else 1L
  tau_lo <- control$tau_lower %||% .nl_st_default("tau_lower")
  tau_hi <- control$tau_upper %||% .nl_st_default("tau_upper")
  tt_axis  <- .st_log_grid(tau_lo, tau_hi, n_gt)
  rho_axis <- if (temporal_type == "ar1") {
    seq(control$rho_lower %||% .nl_st_default("rho_lower"),
        control$rho_upper %||% .nl_st_default("rho_upper"), length.out = n_grho)
  } else 0.0

  # bym2's kernel (cpp_nested_laplace_st_bym2) takes `scale_factor` +
  # `sigma_spatial_grid` + `rho_spatial_grid`, not `tau_spatial_grid` -- the
  # same (sigma, rho) reparameterisation the single-field bym2 nested-Laplace
  # path integrates, not a precision (gcol33/tulpa#776). car_proper's spatial
  # mixing weight stays a single PINNED value (`rho_spatial_grid` repeated
  # across the grid), unchanged from before. hsgp / nngp pair the field
  # variance with the lengthscale / NNGP range over the SAME shared default
  # bounds (`gp_var` / `gp_lengthscale`) the single-field `spatial_gp()` path
  # reads, matching the paired (not crossed) convention the ST kernels expect
  # (one theta_grid row per grid row, as for every other spatial axis pair
  # here).
  if (spatial_type == "bym2") {
    sigma_lo <- control$sigma_lower %||% .nl_st_default("sigma_lower")
    sigma_hi <- control$sigma_upper %||% .nl_st_default("sigma_upper")
    sigma_axis <- .st_log_grid(sigma_lo, sigma_hi, n_gs)
    rho_spatial_axis <- .nl_grid_axis("bym2_rho")
    grid <- expand.grid(sigma_spatial = sigma_axis, rho_spatial = rho_spatial_axis,
                        tau_temporal = tt_axis, rho = rho_axis)
  } else if (spatial_type == "hsgp") {
    sg_axis <- .nl_grid_axis("gp_var", n = n_gs)
    ls_axis <- .nl_grid_axis("gp_lengthscale", n = n_gs)
    grid <- expand.grid(sigma2 = sg_axis, lengthscale = ls_axis,
                        tau_temporal = tt_axis, rho = rho_axis)
  } else if (spatial_type == "nngp") {
    sg_axis <- .nl_grid_axis("gp_var", n = n_gs)
    pg_axis <- .nl_grid_axis("gp_lengthscale", n = n_gs)
    grid <- expand.grid(sigma2 = sg_axis, phi_gp = pg_axis,
                        tau_temporal = tt_axis, rho = rho_axis)
  } else {
    ts_axis <- .st_log_grid(tau_lo, tau_hi, n_gs)
    grid <- expand.grid(tau_spatial = ts_axis, tau_temporal = tt_axis, rho = rho_axis)
  }

  kernel <- switch(spatial_type,
                   icar        = cpp_nested_laplace_st_icar,
                   bym2        = cpp_nested_laplace_st_bym2,
                   car_proper  = cpp_nested_laplace_st_car_proper,
                   hsgp        = cpp_nested_laplace_st_hsgp,
                   nngp        = cpp_nested_laplace_st_nngp)

  kargs <- list(
    y = y, n = n_trials, X = X,
    re_idx = as.numeric(re_idx), n_re_groups = as.integer(n_re_groups),
    sigma_re = as.numeric(sigma_re),
    temporal_idx = as.integer(temporal_idx), n_times = as.integer(n_times),
    temporal_type = temporal_type,
    tau_temporal_grid = as.numeric(grid$tau_temporal),
    rho_temporal_grid = as.numeric(grid$rho),
    cyclic = isTRUE(cyclic), family = family,
    phi = .phi_to_kernel(family, as.numeric(phi)),
    max_iter = as.integer(control$max_iter %||% 50L),
    tol = as.numeric(control$tol %||% 1e-6),
    n_threads = as.integer(control$n_threads %||% 1L),
    store_Q = TRUE
  )
  # bym2 needs a scale_factor; car_proper an rho_spatial_grid. Keep to the
  # kernels' documented defaults for those extra axes here (icar is the base).
  # The default is one registry entry, not a literal here: a selector gets one
  # default, in one place (gcol33/tulpa#673).
  rho_spatial_val <- control$rho_spatial %||% .nl_st_default("rho_spatial")
  # A continuous field's range/lengthscale axis needs the coordinate extent to
  # anchor its PC prior (`.hp_range_anchor()` reads `block$coords`); an areal
  # field passes none, matching every other anchor-less family here.
  hp_block <- list()
  if (is_areal) {
    kargs$spatial_idx <- as.integer(spatial_idx)
    kargs$n_spatial_units <- as.integer(n_s)
    kargs$adj_row_ptr <- as.integer(csr$row_ptr)
    kargs$adj_col_idx <- as.integer(csr$col_idx)
    kargs$n_neighbors <- as.integer(csr$n_neighbors)
    if (spatial_type == "bym2") {
      sc <- .bym2_component_scaling(adjacency)
      kargs$scale_factor       <- sc$scale_factor
      kargs$node_prec          <- sc$node_prec
      kargs$sigma_spatial_grid <- as.numeric(grid$sigma_spatial)
      kargs$rho_spatial_grid   <- as.numeric(grid$rho_spatial)
    } else {
      kargs$tau_spatial_grid <- as.numeric(grid$tau_spatial)
      if (spatial_type == "car_proper") {
        kargs$rho_spatial_grid <- rep(rho_spatial_val, nrow(grid))
      }
    }
  } else if (spatial_type == "hsgp") {
    kargs$phi_basis  <- as.matrix(basis$phi_basis)
    kargs$lambda_eig <- as.numeric(basis$lambda_eig)
    kargs$sigma2_spatial_grid      <- as.numeric(grid$sigma2)
    kargs$lengthscale_spatial_grid <- as.numeric(grid$lengthscale)
    hp_block <- list(coords = coords)
  } else {
    # nngp: coords in ORIGINAL unique-location order, nn_idx/nn_dist as
    # compute_nngp_neighbors() returns them (1-based ordering positions), and
    # a 0-based nn_order -- the exact convention
    # `.spatial_spec_to_nl_prior()` + the single-field `nngp` registry pack()
    # already ship for `cpp_nested_laplace_nngp` (batch_nngp_scatter reads
    # `coords(nn_order[i])`).
    kargs$spatial_idx <- as.integer(spatial_idx)
    kargs$n_spatial    <- as.integer(n_s)
    kargs$coords       <- coords
    kargs$nn_idx       <- as.matrix(ni$nn_idx)
    kargs$nn_dist      <- as.matrix(ni$nn_dist)
    kargs$nn_order     <- as.integer(ni$nn_order) - 1L
    kargs$nn           <- as.integer(ncol(ni$nn_idx))
    kargs$cov_type     <- as.integer(cov_type)
    kargs$sigma2_spatial_grid <- as.numeric(grid$sigma2)
    kargs$phi_gp_spatial_grid <- as.numeric(grid$phi_gp)
    hp_block <- list(coords = coords)
  }
  out <- .st_attach_outer_integration(do.call(kernel, kargs), grid, hyperprior,
                                      block = hp_block)
  # Outer-grid collapse visibility + recenter:
  # tau_lower/tau_upper's default [0.25, 16] span (and, for ar1, the default
  # rho_lower/rho_upper) is a starting axis, not a hard ceiling, the same
  # contract every other nested-Laplace family's default grid carries. A
  # from-scratch mode-Hessian recenter-and-refit engages when a free axis
  # rails (`.nl_axis_railed()`) and no grid knob was explicitly overridden; see `.st_auto_grid_rescue()`
  # (R/fit_st_nested_auto_grid.R) for the optim()-based mode-find it uses.
  # (hsgp / nngp decline this rescue outright -- see the bym2-style guard at
  # its top -- since their axes are not the tau_spatial/tau_temporal/rho pair
  # it knows how to transform.)
  out <- .joint_attach_pareto_k_regime(out)
  out <- .st_auto_grid_rescue(out, kernel, kargs, spatial_type, temporal_type,
                              n_gs, n_gt, n_grho, tau_lo, tau_hi, control,
                              rho_spatial_val = rho_spatial_val,
                              hyperprior = hyperprior)
  out$outer_grid_placement <- out$outer_grid_placement %||% "fixed"
  # Within-cell construction for the reported per-axis intervals; the default
  # is `.nl_diag("within_cell")`.
  within_cell <- .nl_within_cell_mode(control$within_cell)
  out <- .nl_posterior_moments(out, "st", within = within_cell)
  out <- .nl_attach_grid_hessians(out, ncol(X))

  # Grid-marginalised field posterior means: the latent block after the fixed
  # effects is [spatial (n_s), temporal (n_times)] for icar / car_proper /
  # nngp, [structured (n_s), unstructured (n_s), temporal (n_times)] for
  # bym2 -- its spatial contribution to eta is the sigma/rho-weighted MIX of
  # the two raw (standardized) sub-fields, not either one alone
  # (gcol33/tulpa#776; see bym2_mixing.h for the same `sigma * (sqrt(rho) *
  # scale_factor * phi + sqrt(1 - rho) * theta)` combination the kernel's own
  # d_fac applies) -- or [basis weights (M), temporal (n_times)] for hsgp,
  # whose field has no per-location layout: the reported `spatial_effects` is
  # the per-observation field `phi_basis %*% beta_bar`, projecting the
  # weight-averaged basis coefficients through the same basis every
  # observation shares.
  p <- ncol(X)
  w <- out$weights
  if (spatial_type == "bym2") {
    phi_cols   <- p + seq_len(n_s)
    theta_cols <- p + n_s + seq_len(n_s)
    te_cols    <- p + 2L * n_s + seq_len(n_times)
    sigma_k <- out$theta_grid[, "sigma_spatial"]
    rho_k   <- out$theta_grid[, "rho_spatial"]
    combined <- out$modes[, phi_cols, drop = FALSE] *
      (sigma_k * sqrt(rho_k) * kargs$scale_factor) +
      out$modes[, theta_cols, drop = FALSE] * (sigma_k * sqrt(1 - rho_k))
    out$spatial_effects <- as.numeric(crossprod(w, combined))
  } else if (spatial_type == "hsgp") {
    m_basis <- ncol(basis$phi_basis)
    sp_cols <- p + seq_len(m_basis)
    te_cols <- p + m_basis + seq_len(n_times)
    beta_bar <- as.numeric(crossprod(w, out$modes[, sp_cols, drop = FALSE]))
    out$spatial_effects <- as.numeric(basis$phi_basis %*% beta_bar)
  } else {
    sp_cols <- p + seq_len(n_s)
    te_cols <- p + n_s + seq_len(n_times)
    out$spatial_effects <- as.numeric(crossprod(w, out$modes[, sp_cols, drop = FALSE]))
  }
  out$temporal_effects <- as.numeric(crossprod(w, out$modes[, te_cols, drop = FALSE]))
  out$spatial_type  <- spatial_type
  out$temporal_type <- temporal_type
  out$N <- N
  out$y <- y
  out$model_matrix <- X
  # Observation-level accessors (fitted/residuals/posterior_predict/simulate,
  # the simulation-based tests) read these off the fit the way tulpa()'s own
  # fits carry them; `phi` is the R-level convention, not the kernel's
  # (gcol33/tulpa#777).
  out$family   <- family
  out$n_trials <- n_trials
  out$phi      <- phi

  .finalize_fit(out, backend = "nested_laplace",
                n_fixed = p, fixed_names = colnames(X),
                extra_class = c("tulpa_st_nested", "tulpa_nested_laplace", "list"))
}
