# Generic spec-driven outer-grid refinement (Step 2).
#
# Port of the joint nested-Laplace refinement machinery in
# `R/nested_laplace_joint_refine.R`, generalised so the axis metadata (log-
# scale, bounds, refinable, refinement priority) lives in
# `hyper_axis_spec` objects instead of hardcoded by-axis-name lookups. Driven
# only through a user-supplied `kernel_fn(new_cells, warm_start, store_extras)`
# callback, so the SAME refinement engine serves:
#   * `tulpa_hyper_grid()` (kernel_fn wraps the user's inner_fit; no per-cell
#     warm-start chain), and
#   * `tulpa_nested_laplace_joint()` (kernel_fn wraps the backend's joint C++
#     kernel call; warm_start is the anchor cell's mode vector).
#
# Math ground (porting note: the joint driver's file-level math note in
# `R/nested_laplace_joint_refine.R` applies verbatim here -- one source of
# truth, the formulas didn't move).
#
# The on-disk canonical representation throughout is:
#   * `theta_grid`     -- numeric matrix `[n_cells x n_axes]` of outer-grid
#                         hyperparameter values; columns named after axes.
#   * `log_marginal`   -- numeric `[n_cells]`; per-cell log integrand
#                         (kernel log marginal + log prior already baked in).
#   * `extras`         -- list of length `n_cells` or `NULL`; opaque per-cell
#                         side data the kernel returned (e.g. the joint
#                         driver's mode + Q_csc; hyper_grid passes NULL).
#                         Refinement carries it through by concatenation.
#   * `refining_axis`  -- character `[n_cells]`; per-cell tag identifying
#                         which refinement pass produced the cell (`""` for
#                         the initial Cartesian grid, `<axis>` for slice
#                         cells refining that axis, `consistency_<axis>`
#                         for var-of-means consistency slices).
#
# kernel_fn signature:
#   function(new_cells, warm_start = NULL, store_extras = FALSE) -> list(
#     log_marginal = numeric[nrow(new_cells)],
#     extras       = list[nrow(new_cells)] or NULL
#   )
# `warm_start` is opaque to refinement: refinement passes whatever the
# anchor cell's extras carry, and the kernel decides how to use it (e.g.
# read $mode from joint extras to warm-start the slice kernel call).

# ============================================================================
# Spec lookups -- the single point where axis metadata is read.
# ============================================================================

# Axis names that opt in to refinement passes.
.hyper_refinable_axes <- function(specs) {
  vapply(specs, function(s)
    if (isTRUE(s$refinable)) s$name else NA_character_,
    character(1))
}
.hyper_refinable_names <- function(specs) {
  out <- .hyper_refinable_axes(specs)
  out[!is.na(out)]
}

# Per-spec helpers.
.hyper_spec_by_name <- function(specs, name) {
  for (s in specs) if (identical(s$name, name)) return(s)
  stop("Unknown axis '", name, "'.", call. = FALSE)
}
.hyper_axis_is_log_scale <- function(spec) isTRUE(spec$log_scale)
.hyper_axis_bounds <- function(spec) spec$bounds

# Refinement-order priority. Defaults to declaration order; callers may
# attach `refine_priority` (integer, smaller = earlier) to a spec to override.
.hyper_axis_refinement_order <- function(axes, specs) {
  prio <- vapply(axes, function(a) {
    s <- .hyper_spec_by_name(specs, a)
    as.integer(s$refine_priority %||% 100L)
  }, integer(1))
  axes[order(prio, seq_along(axes))]
}

# ============================================================================
# Edge scores per axis (boundary mass + integrand-density-at-boundary).
# ============================================================================
.hyper_axis_edge_scores <- function(theta_grid, log_marginal, specs, axes) {
  if (length(log_marginal) == 0L) return(list())
  # Non-finite cells (inner Newton non-convergent on consumer-package
  # joint fitters, e.g. occu_cover_joint_coupled at degenerate sigma+alpha
  # hyperpoints) must not poison the edge-score statistics. Treat them as
  # zero-mass: drop from the global max-shift, use the safe weight
  # normaliser, and skip them inside per-level max/sum reductions.
  finite_lm <- log_marginal[is.finite(log_marginal)]
  if (length(finite_lm) == 0L) return(list())
  lm_max_total <- max(finite_lm)
  weights      <- .nl_normalise_weights_safe(log_marginal,
                                              what = "adaptive_grid edge scores")
  if (all(is.na(weights))) return(list())
  out <- list()
  per_level_max <- function(lm_at_lev) {
    finite_lev <- lm_at_lev[is.finite(lm_at_lev)]
    if (length(finite_lev) == 0L) -Inf else max(finite_lev)
  }
  for (a in axes) {
    v   <- as.numeric(theta_grid[, a])
    lev <- sort(unique(v))
    if (length(lev) < 2L) next
    w_lo <- sum(weights[v == lev[1L]],          na.rm = TRUE)
    w_hi <- sum(weights[v == lev[length(lev)]], na.rm = TRUE)
    d_lo <- exp(per_level_max(log_marginal[v == lev[1L]])          - lm_max_total)
    d_hi <- exp(per_level_max(log_marginal[v == lev[length(lev)]]) - lm_max_total)
    out[[a]] <- list(
      levels    = lev,
      min_frac  = w_lo, max_frac = w_hi,
      min_dens  = d_lo, max_dens = d_hi,
      min_score = max(w_lo, d_lo),
      max_score = max(w_hi, d_hi)
    )
  }
  out
}

# ============================================================================
# Per-axis point proposals (boundary extension + interior densification).
# Identical to the joint driver's `.propose_axis_extension` /
# `.propose_interior_densification` but reading log_scale / bounds from `spec`.
# ============================================================================
.hyper_propose_axis_extension <- function(spec, lev, side, extend_ok = TRUE) {
  log_scale <- .hyper_axis_is_log_scale(spec)
  bounds    <- .hyper_axis_bounds(spec)
  mid <- if (log_scale)
    function(a, b) exp(0.5 * (log(a) + log(b))) else
    function(a, b) 0.5 * (a + b)
  if (side == "max") {
    edge      <- lev[length(lev)]
    neighbour <- lev[length(lev) - 1L]
    densify   <- mid(neighbour, edge)
    extend1   <- if (log_scale) edge * (edge / neighbour)
                 else            edge + (edge - neighbour)
    extend2   <- if (log_scale) extend1 * (edge / neighbour)
                 else            extend1 + (edge - neighbour)
  } else {
    edge      <- lev[1L]
    neighbour <- lev[2L]
    densify   <- mid(edge, neighbour)
    extend1   <- if (log_scale) edge * (edge / neighbour)
                 else            edge - (neighbour - edge)
    extend2   <- if (log_scale) extend1 * (edge / neighbour)
                 else            extend1 - (neighbour - edge)
  }
  pts <- if (isTRUE(extend_ok)) c(densify, extend1, extend2) else densify
  if (!is.null(bounds)) {
    pts <- pts[pts > bounds[1L] & pts < bounds[2L]]
  }
  keep <- vapply(pts, function(p) {
    all(abs(lev - p) > 1e-8 * max(1, abs(p)))
  }, logical(1))
  pts[keep]
}

.hyper_propose_interior_densification <- function(spec, lev, mode_idx,
                                                  do_left = FALSE,
                                                  do_right = FALSE) {
  log_scale <- .hyper_axis_is_log_scale(spec)
  mid <- if (log_scale)
    function(a, b) exp(0.5 * (log(a) + log(b))) else
    function(a, b) 0.5 * (a + b)
  pts <- numeric(0)
  if (do_left  && mode_idx > 1L)              pts <- c(pts, mid(lev[mode_idx - 1L], lev[mode_idx]))
  if (do_right && mode_idx < length(lev))     pts <- c(pts, mid(lev[mode_idx], lev[mode_idx + 1L]))
  if (length(pts) == 0L) return(pts)
  bounds <- .hyper_axis_bounds(spec)
  if (!is.null(bounds)) {
    pts <- pts[pts > bounds[1L] & pts < bounds[2L]]
  }
  keep <- vapply(pts, function(p) {
    all(abs(lev - p) > 1e-8 * max(1, abs(p)))
  }, logical(1))
  pts[keep]
}

.hyper_propose_consistency_points <- function(spec, mu, sd, lev) {
  if (!is.finite(mu) || !is.finite(sd) || sd <= 0) return(numeric(0))
  is_log <- .hyper_axis_is_log_scale(spec)
  if (is_log) {
    if (mu <= 0) return(numeric(0))
    log_mu <- log(mu); log_sd <- sd / mu
    if (!is.finite(log_sd) || log_sd <= 0) return(numeric(0))
    pts <- exp(log_mu + c(-1.5, -0.7, 0.7, 1.5) * log_sd)
  } else {
    pts <- mu + c(-1.5, -0.7, 0.7, 1.5) * sd
  }
  bounds <- .hyper_axis_bounds(spec)
  if (!is.null(bounds)) {
    pts <- pts[pts > bounds[1L] & pts < bounds[2L]]
  }
  if (length(pts) == 0L) return(numeric(0))
  keep <- vapply(pts, function(p) {
    if (is_log) !any(abs(log(lev) - log(p)) < 0.05)
    else        !any(abs(lev - p) < 0.05 * max(abs(lev), 1))
  }, logical(1))
  pts[keep]
}

# ============================================================================
# Slice-triple builder. For each new axis value the helper produces ONE row
# at (axis = pt, other_axes = modal_at_anchor). Returns the matrix of new
# cells plus the marginal-scale calibration term (log S_a, see math note).
# ============================================================================
.hyper_new_mode_tracked_triples <- function(theta_grid, log_marginal, specs,
                                             axis_name, new_pts, anchor_lev) {
  if (length(new_pts) == 0L) return(NULL)
  v <- as.numeric(theta_grid[, axis_name])
  mask <- abs(v - anchor_lev) < 1e-12 * max(1, abs(anchor_lev))
  if (!any(mask)) return(NULL)
  anchor_lm   <- log_marginal[mask]
  k_map_local <- which.max(anchor_lm)
  idx_global  <- which(mask)[k_map_local]
  L_map       <- anchor_lm[k_map_local]
  calibration <- log(sum(exp(anchor_lm - L_map)))

  axis_names <- colnames(theta_grid)
  n_new <- length(new_pts)
  new_cells <- matrix(NA_real_, n_new, length(axis_names))
  colnames(new_cells) <- axis_names
  for (a in axis_names) {
    new_cells[, a] <- if (a == axis_name) as.numeric(new_pts)
                      else rep(as.numeric(theta_grid[idx_global, a]), n_new)
  }
  list(new_cells      = new_cells,
       calibration    = rep(calibration, n_new),
       warm_start_idx = idx_global)
}

# Stitch slice triples from multiple (axis, side) packs into one matrix +
# calibration vector. Drops rows whose cell already appears in `theta_grid`
# at numerical tolerance via a stringified key.
.hyper_concat_slice_triples <- function(triple_packs, theta_grid) {
  if (length(triple_packs) == 0L) return(NULL)
  axis_names <- colnames(theta_grid)
  parts <- lapply(triple_packs, `[[`, "new_cells")
  if (length(parts) == 0L) return(NULL)
  new_cells <- do.call(rbind, parts)
  calib     <- as.numeric(unlist(lapply(triple_packs, `[[`, "calibration"),
                                  use.names = FALSE))
  if (nrow(new_cells) == 0L) return(NULL)
  fmt <- function(m) {
    cols <- lapply(axis_names, function(a) sprintf("%.10g", m[, a]))
    do.call(paste, c(cols, sep = ":"))
  }
  new_keys <- fmt(new_cells)
  old_keys <- fmt(theta_grid)
  keep <- !new_keys %in% old_keys
  if (!any(keep)) return(NULL)
  list(new_cells   = new_cells[keep, , drop = FALSE],
       calibration = calib[keep])
}

# ============================================================================
# Detect refinement triggers on one axis.
# Boundary (peak-at-edge or tail-mass-at-edge) and interior (peak between
# levels with wide spacing) checks. Returns NULL if no trigger fires.
# ============================================================================
.hyper_detect_axis_refinement <- function(theta_grid, log_marginal, edge_info,
                                          axis_name, spec, edge_thresh) {
  ei  <- edge_info
  lev <- ei$levels
  v   <- as.numeric(theta_grid[, axis_name])
  lm_max_at_lev <- vapply(lev, function(lv) {
    lm_lv <- log_marginal[v == lv]
    finite_lv <- lm_lv[is.finite(lm_lv)]
    if (length(finite_lv) == 0L) -Inf else max(finite_lv)
  }, numeric(1))
  mode_idx <- which.max(lm_max_at_lev)
  n_lev    <- length(lev)

  tr_min <- ei$min_score >= edge_thresh &&
            (mode_idx == 1L || ei$min_dens >= edge_thresh)
  tr_max <- ei$max_score >= edge_thresh &&
            (mode_idx == n_lev || ei$max_dens >= edge_thresh)

  interior_log_step <- 0.55
  interior_lin_step <- 0.25
  wide_left <- wide_right <- FALSE
  if (mode_idx > 1L && mode_idx < n_lev) {
    if (.hyper_axis_is_log_scale(spec)) {
      wide_left  <- (log(lev[mode_idx])     - log(lev[mode_idx - 1L])) >= interior_log_step
      wide_right <- (log(lev[mode_idx + 1L]) - log(lev[mode_idx]))     >= interior_log_step
    } else {
      span <- diff(range(lev))
      if (span > 0) {
        wide_left  <- (lev[mode_idx]     - lev[mode_idx - 1L]) / span >= interior_lin_step
        wide_right <- (lev[mode_idx + 1L] - lev[mode_idx])     / span >= interior_lin_step
      }
    }
  }

  if (!tr_min && !tr_max && !wide_left && !wide_right) return(NULL)
  packs <- list()
  if (tr_max) {
    pts <- .hyper_propose_axis_extension(spec, lev, "max", extend_ok = TRUE)
    pk  <- .hyper_new_mode_tracked_triples(theta_grid, log_marginal, NULL,
                                            axis_name, pts,
                                            anchor_lev = lev[n_lev])
    if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
  }
  if (tr_min) {
    pts <- .hyper_propose_axis_extension(spec, lev, "min", extend_ok = TRUE)
    pk  <- .hyper_new_mode_tracked_triples(theta_grid, log_marginal, NULL,
                                            axis_name, pts,
                                            anchor_lev = lev[1L])
    if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
  }
  if (wide_left || wide_right) {
    pts <- .hyper_propose_interior_densification(spec, lev, mode_idx,
                                                  wide_left, wide_right)
    pk  <- .hyper_new_mode_tracked_triples(theta_grid, log_marginal, NULL,
                                            axis_name, pts,
                                            anchor_lev = lev[mode_idx])
    if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
  }
  if (length(packs) == 0L) return(NULL)
  packs
}

# ============================================================================
# Apply refinement for one axis: build the slice cells, call kernel_fn at
# them with the anchor's warm-start material, merge into the existing grid
# / log_marginal / extras / refining_axis.
# ============================================================================
.hyper_apply_axis_refinement <- function(theta_grid, log_marginal, extras,
                                          refining_axis, triple_packs,
                                          axis_name, specs, kernel_fn,
                                          hp_fn = NULL,
                                          consistency_tag = FALSE) {
  merged <- .hyper_concat_slice_triples(triple_packs, theta_grid)
  if (is.null(merged)) {
    return(list(theta_grid = theta_grid, log_marginal = log_marginal,
                extras = extras, refining_axis = refining_axis, n_new = 0L))
  }
  new_cells   <- merged$new_cells
  calibration <- merged$calibration
  n_new <- nrow(new_cells)

  warm_start <- NULL
  if (!is.null(extras)) {
    idx0 <- triple_packs[[1L]]$warm_start_idx
    if (length(idx0) == 1L && idx0 >= 1L && idx0 <= length(extras)) {
      warm_start <- extras[[idx0]]
    }
  }

  fit_out <- kernel_fn(new_cells, warm_start = warm_start,
                        store_extras = !is.null(extras))
  new_lm  <- fit_out$log_marginal + calibration
  if (!is.null(hp_fn)) {
    hp_new <- hp_fn(new_cells)
    if (!is.null(hp_new) && length(hp_new) == n_new) {
      new_lm <- new_lm + hp_new
    }
  }
  tag <- if (isTRUE(consistency_tag)) paste0("consistency_", axis_name)
         else axis_name

  theta_grid_out   <- rbind(theta_grid, new_cells)
  log_marginal_out <- c(log_marginal, new_lm)
  refining_out     <- c(refining_axis, rep(tag, n_new))
  extras_out       <- extras
  if (!is.null(extras_out)) {
    extras_out <- c(extras_out, fit_out$extras %||% vector("list", n_new))
  }
  list(theta_grid = theta_grid_out, log_marginal = log_marginal_out,
       extras = extras_out, refining_axis = refining_out, n_new = n_new)
}

# ============================================================================
# Main entries: adaptive boundary/interior pass and var-of-means consistency.
# ============================================================================

.hyper_adaptive_refine_pass <- function(theta_grid, log_marginal, extras,
                                        refining_axis, specs, kernel_fn,
                                        edge_thresh = 0.02, max_passes = 1L,
                                        hp_fn = NULL) {
  info <- list(triggered_axes = character(0),
               n_points_added = integer(0))
  if (max_passes < 1L) {
    return(list(theta_grid = theta_grid, log_marginal = log_marginal,
                extras = extras, refining_axis = refining_axis, info = NULL))
  }
  for (pass in seq_len(max_passes)) {
    axes <- .hyper_axis_refinement_order(.hyper_refinable_names(specs), specs)
    if (length(axes) == 0L) break
    any_triggered <- FALSE
    triggered_this_pass <- character(0)
    for (a in axes) {
      edge_info <- .hyper_axis_edge_scores(theta_grid, log_marginal, specs, a)
      ei <- edge_info[[a]]
      if (is.null(ei)) next
      spec <- .hyper_spec_by_name(specs, a)
      packs <- .hyper_detect_axis_refinement(theta_grid, log_marginal, ei, a,
                                              spec, edge_thresh)
      if (is.null(packs)) next
      step <- .hyper_apply_axis_refinement(theta_grid, log_marginal, extras,
                                            refining_axis, packs, a, specs,
                                            kernel_fn, hp_fn = hp_fn)
      if (step$n_new == 0L) next
      theta_grid   <- step$theta_grid
      log_marginal <- step$log_marginal
      extras       <- step$extras
      refining_axis <- step$refining_axis
      triggered_this_pass <- c(triggered_this_pass, a)
      info$n_points_added <- c(info$n_points_added, step$n_new)
      any_triggered <- TRUE
    }
    if (!any_triggered) break
    info$triggered_axes <- c(info$triggered_axes,
                              paste(triggered_this_pass, collapse = ","))
  }
  if (length(info$triggered_axes) == 0L) info <- NULL
  list(theta_grid = theta_grid, log_marginal = log_marginal,
       extras = extras, refining_axis = refining_axis, info = info)
}

.hyper_consistency_pass <- function(theta_grid, log_marginal, extras,
                                    refining_axis, specs, theta_mean,
                                    theta_sd, kernel_fn,
                                    tolerance = 0.7, hp_fn = NULL,
                                    weights = NULL) {
  refinable <- .hyper_refinable_names(specs)
  info <- list(axes = character(0), n_added = integer(0),
               vom_before = numeric(0), sd_laplace = numeric(0))
  n_added_total <- 0L
  if (length(refinable) == 0L) {
    return(list(theta_grid = theta_grid, log_marginal = log_marginal,
                extras = extras, refining_axis = refining_axis,
                info = NULL, n_added = 0L))
  }
  for (axis in refinable) {
    sd_lap <- theta_sd[[axis]] %||% NA_real_
    mu     <- theta_mean[[axis]] %||% NA_real_
    if (!is.finite(sd_lap) || !is.finite(mu) || sd_lap <= 0) next
    # Refinement on a previous axis grows theta_grid / log_marginal, so the
    # weights and the mode index are recomputed each iteration to stay aligned
    # with the current grid rows.
    iter_weights <- weights
    if (is.null(iter_weights) || length(iter_weights) != length(log_marginal)) {
      iter_weights <- .nl_normalise_weights_safe(log_marginal, "refinement grid")
    }
    overall_mode_idx <- which.max(log_marginal)
    axis_vals <- as.numeric(theta_grid[, axis])
    vom_sd    <- .nl_wtd_mean_sd(axis_vals, iter_weights)$sd
    if (vom_sd >= tolerance * sd_lap) next

    lev  <- sort(unique(as.numeric(theta_grid[, axis])))
    spec <- .hyper_spec_by_name(specs, axis)
    new_pts <- .hyper_propose_consistency_points(spec, mu, sd_lap, lev)
    if (length(new_pts) == 0L) next

    anchor_lev <- as.numeric(theta_grid[overall_mode_idx, axis])
    pack <- .hyper_new_mode_tracked_triples(theta_grid, log_marginal, NULL,
                                             axis, new_pts, anchor_lev)
    if (is.null(pack)) next
    step <- .hyper_apply_axis_refinement(theta_grid, log_marginal, extras,
                                          refining_axis, list(pack), axis,
                                          specs, kernel_fn, hp_fn = hp_fn,
                                          consistency_tag = TRUE)
    if (step$n_new == 0L) next
    theta_grid    <- step$theta_grid
    log_marginal  <- step$log_marginal
    extras        <- step$extras
    refining_axis <- step$refining_axis
    info$axes        <- c(info$axes, axis)
    info$n_added     <- c(info$n_added, step$n_new)
    info$vom_before  <- c(info$vom_before, vom_sd)
    info$sd_laplace  <- c(info$sd_laplace, sd_lap)
    n_added_total    <- n_added_total + step$n_new
  }
  if (length(info$axes) == 0L) info <- NULL
  list(theta_grid = theta_grid, log_marginal = log_marginal,
       extras = extras, refining_axis = refining_axis, info = info,
       n_added = n_added_total)
}

# ============================================================================
# Per-axis moment recalibration when slice cells are present. Weighted
# mean / SD on each axis over its own refinement slice (own-axis and
# consistency cells), skipping axes with no foreign-slice contamination.
# ============================================================================
.hyper_recalibrate_axis_moments <- function(theta_grid, log_marginal,
                                            refining_axis, theta_mean,
                                            theta_sd) {
  if (is.null(refining_axis) || all(refining_axis == "")) {
    return(list(theta_mean = theta_mean, theta_sd = theta_sd))
  }
  cols <- colnames(theta_grid)
  for (col in cols) {
    keep <- refining_axis == "" | refining_axis == col |
            refining_axis == paste0("consistency_", col)
    if (all(keep)) next
    lm_k <- log_marginal[keep]
    m    <- max(lm_k)
    if (!is.finite(m)) next
    w    <- exp(lm_k - m); w <- w / sum(w)
    vals <- theta_grid[keep, col]
    ms   <- .nl_wtd_mean_sd(vals, w)
    theta_mean[[col]] <- ms$mean
    theta_sd[[col]]   <- ms$sd
  }
  list(theta_mean = theta_mean, theta_sd = theta_sd)
}
