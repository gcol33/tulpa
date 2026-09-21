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

# May refinement place a node PAST the outermost declared node on this axis?
# A spec that predates the field says nothing about it and keeps the historical
# answer.
.hyper_axis_may_extend <- function(spec) !identical(spec$extend, FALSE)

# The interval new nodes are confined to on an axis that may not be extended:
# the span the axis's DECLARED nodes cover, on the natural scale. `spec$grid`
# holds those nodes -- refinement appends to `theta_grid`, never to the spec --
# so this is the range the caller wrote down however many passes have run.
# A log-scale axis's zero level is its point mass rather than a point of the
# continuum, so the span is taken over the positive nodes.
# NULL where the axis may be extended, and where fewer than two continuum nodes
# leave no interior to place anything in.
.hyper_axis_node_limit <- function(spec) {
  if (.hyper_axis_may_extend(spec)) return(NULL)
  g <- as.numeric(spec$grid)
  g <- g[is.finite(g)]
  if (.hyper_axis_is_log_scale(spec)) g <- g[g > 0]
  if (length(g) < 2L) return(c(NA_real_, NA_real_))
  range(g)
}

# Drop proposed nodes that fall outside the declared span. A no-op on an axis
# that may be extended; on one that may not, this is the whole of what makes a
# stated axis a bound, so every proposal path goes through it rather than each
# restating the comparison.
.hyper_clip_to_node_limit <- function(pts, spec) {
  lim <- .hyper_axis_node_limit(spec)
  if (is.null(lim)) return(pts)
  if (anyNA(lim)) return(pts[0])
  pts[pts >= lim[1L] & pts <= lim[2L]]
}

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
.hyper_axis_edge_scores <- function(theta_grid, log_marginal, specs, axes,
                                    refining_axis = NULL) {
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
                                              what = "adaptive_grid edge scores",
                                              log_quad = .hyper_log_quad_weights(
                                                  theta_grid, specs,
                                                  refining = refining_axis))
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
.hyper_propose_axis_extension <- function(spec, lev, side,
                                          extend_ok = .hyper_axis_may_extend(spec)) {
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
  pts <- .hyper_clip_to_node_limit(pts, spec)
  if (!is.null(bounds)) {
    pts <- pts[pts > bounds[1L] & pts < bounds[2L]]
  }
  # The declared prior support is fixed, so a node outside it would carry zero
  # weight and only cost an inner solve.
  slab <- spec$slab_bounds
  if (!is.null(slab)) {
    pts <- pts[pts >= slab[1L] & pts <= slab[2L]]
  }
  keep <- vapply(pts, function(p) {
    all(abs(lev - p) > 1e-8 * max(1, abs(p)))
  }, logical(1))
  pts[keep]
}

# Every point here is a midpoint between two existing levels, so it lies inside
# the span whatever the axis's extension setting; no clip is needed.
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

# Points bisecting the gaps an axis marginal's mass sits across.
#
# `vals` / `log_mass` are the axis's levels and their log masses, the quadrature
# weight already folded in. Only the continuum is bisected: a declared point
# mass is a level of the model rather than a node of the rule, so it neither
# bounds a gap nor enters the shares. A gap between adjacent continuum levels is
# bisected on the axis's integration coordinate when the two levels bounding it
# together carry at least `1 / min_ess` of the continuum's mass, the share one
# level holds on an axis spread evenly at the ESS floor. The heaviest gap is
# always bisected, so a round under the floor never proposes nothing.
#
# The points are placed by where the mass IS rather than at a multiple of a
# modal SD: a marginal whose mass sits on two adjacent nodes has its modal
# parabola read the grid's spacing, and points placed at a fraction of that
# around the mean land inside the gap and leave both heavy nodes' outer sides
# as coarse as before (gcol33/tulpa#858).
#
# Nothing is proposed where the density peaks on an outermost continuum level.
#
# Every point is a midpoint between two existing levels, so it lies inside the
# span, the bounds and the slab whatever the axis declares; no clip is needed.
# Points come back heaviest gap first, so a caller spending a node budget
# spends it where the mass is.
.hyper_propose_mass_bisection <- function(spec, vals, log_mass,
                                          min_ess = .nl_diag("axis_sd_ess")) {
  vals <- as.numeric(vals)
  keep <- is.finite(vals) & !.hyper_is_atom_level(vals, spec)
  if (.hyper_axis_is_log_scale(spec)) keep <- keep & vals > 0
  v  <- vals[keep]
  lm <- as.numeric(log_mass)[keep]
  o  <- order(v); v <- v[o]; lm <- lm[o]
  if (length(v) < 2L) return(numeric(0))
  m <- max(lm)
  if (!is.finite(m)) return(numeric(0))
  p <- exp(lm - m)
  p[!is.finite(p)] <- 0
  p <- p / sum(p)
  u <- .hyper_axis_coord(v, spec)
  # A marginal whose density peaks on an outermost level is truncated by the
  # span, not under-resolved inside it: bisecting towards that level only
  # shrinks its box, and the extension that would serve it belongs to the
  # adaptive pass. The density is the mass less the level's own box width, so
  # a level made heavy by the width it owns does not read as the mode.
  lw <- .nl_level_log_width(u)
  ld <- if (is.null(lw)) lm else lm - lw
  if (which.max(ld) %in% c(1L, length(ld))) return(numeric(0))
  gap <- p[-1L] + p[-length(p)]
  width <- diff(u)
  ok <- is.finite(width) & width > 1e-8 * pmax(1, abs(u[-1L]))
  if (!any(ok)) return(numeric(0))
  pick <- ok & gap >= 1 / min_ess
  pick[which(ok)[which.max(gap[ok])]] <- TRUE
  idx <- which(pick)
  idx <- idx[order(-gap[idx])]
  u_mid <- 0.5 * (u[idx] + u[idx + 1L])
  if (.hyper_axis_is_log_scale(spec)) exp(u_mid) else u_mid
}

# ============================================================================
# Slice-cell builder. For each new axis value the helper produces ONE cell at
# (axis = pt, other_axes = the modal cell at the anchor level). The cell is a
# point evaluation like any other and is measured by the box it owns in its
# row (`.hyper_refined_log_quad()`), so it carries no term standing in for the
# rest of its level.
#
# The anchor is chosen among base-tensor cells and slice cells on the SAME axis,
# never a slice cell placed on another one, so every slice cell sits on base
# levels off its own axis and its row is a row of the base tensor.
# ============================================================================
.hyper_new_mode_tracked_triples <- function(theta_grid, log_marginal, specs,
                                             axis_name, new_pts, anchor_lev,
                                             refining_axis = NULL) {
  if (length(new_pts) == 0L) return(NULL)
  v <- as.numeric(theta_grid[, axis_name])
  mask <- abs(v - anchor_lev) < 1e-12 * max(1, abs(anchor_lev)) &
    .hyper_slice_anchor_ok(refining_axis, axis_name, length(v))
  if (!any(mask)) return(NULL)
  anchor_lm   <- log_marginal[mask]
  if (!any(is.finite(anchor_lm))) return(NULL)
  k_map_local <- which.max(anchor_lm)
  idx_global  <- which(mask)[k_map_local]

  axis_names <- colnames(theta_grid)
  n_new <- length(new_pts)
  new_cells <- matrix(NA_real_, n_new, length(axis_names))
  colnames(new_cells) <- axis_names
  for (a in axis_names) {
    new_cells[, a] <- if (a == axis_name) as.numeric(new_pts)
                      else rep(as.numeric(theta_grid[idx_global, a]), n_new)
  }
  list(new_cells      = new_cells,
       warm_start_idx = idx_global)
}

# Stitch slice cells from multiple (axis, side) packs into one matrix. Drops
# rows whose cell already appears in `theta_grid` at numerical tolerance via a
# stringified key.
.hyper_concat_slice_triples <- function(triple_packs, theta_grid) {
  if (length(triple_packs) == 0L) return(NULL)
  axis_names <- colnames(theta_grid)
  parts <- lapply(triple_packs, `[[`, "new_cells")
  if (length(parts) == 0L) return(NULL)
  new_cells <- do.call(rbind, parts)
  if (nrow(new_cells) == 0L) return(NULL)
  fmt <- function(m) {
    cols <- lapply(axis_names, function(a) sprintf("%.10g", m[, a]))
    do.call(paste, c(cols, sep = ":"))
  }
  new_keys <- fmt(new_cells)
  old_keys <- fmt(theta_grid)
  keep <- !new_keys %in% old_keys & !duplicated(new_keys)
  if (!any(keep)) return(NULL)
  list(new_cells = new_cells[keep, , drop = FALSE])
}

# ============================================================================
# Detect refinement triggers on one axis.
# Boundary (peak-at-edge or tail-mass-at-edge) and interior (peak between
# levels with wide spacing) checks. Returns NULL if no trigger fires.
# ============================================================================
.hyper_detect_axis_refinement <- function(theta_grid, log_marginal, edge_info,
                                          axis_name, spec, edge_thresh,
                                          refining_axis = NULL) {
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
    pts <- .hyper_propose_axis_extension(spec, lev, "max")
    pk  <- .hyper_new_mode_tracked_triples(theta_grid, log_marginal, NULL,
                                            axis_name, pts,
                                            anchor_lev = lev[n_lev],
                                            refining_axis = refining_axis)
    if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
  }
  if (tr_min) {
    pts <- .hyper_propose_axis_extension(spec, lev, "min")
    pk  <- .hyper_new_mode_tracked_triples(theta_grid, log_marginal, NULL,
                                            axis_name, pts,
                                            anchor_lev = lev[1L],
                                            refining_axis = refining_axis)
    if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
  }
  if (wide_left || wide_right) {
    pts <- .hyper_propose_interior_densification(spec, lev, mode_idx,
                                                  wide_left, wide_right)
    pk  <- .hyper_new_mode_tracked_triples(theta_grid, log_marginal, NULL,
                                            axis_name, pts,
                                            anchor_lev = lev[mode_idx],
                                            refining_axis = refining_axis)
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
  new_cells <- merged$new_cells
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
  new_lm  <- fit_out$log_marginal
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
      edge_info <- .hyper_axis_edge_scores(theta_grid, log_marginal, specs, a,
                                           refining_axis = refining_axis)
      ei <- edge_info[[a]]
      if (is.null(ei)) next
      spec <- .hyper_spec_by_name(specs, a)
      packs <- .hyper_detect_axis_refinement(theta_grid, log_marginal, ei, a,
                                              spec, edge_thresh,
                                              refining_axis = refining_axis)
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

# Repopulate an axis whose marginal has collapsed onto too few nodes to carry a
# spread, by bisecting the gaps its mass sits across
# (`.hyper_propose_mass_bisection()`) with slice points in the modal cell's row.
#
# The trigger is the axis's own quadrature effective sample size, read off the
# weights the fit integrates with, over the axis's continuum: a declared point
# mass is part of the model, and no node placed in the continuum changes the
# share it holds. It used to be the weighted SD compared against the parabola at
# the modal node, which is one SD estimator judging the other: with the reported
# SD now the weighted one wherever the axis is resolved (gcol33/tulpa#621), that
# comparison would have been the estimator against itself and the pass would
# never fire. The ESS answers the question the pass is actually asking -- how
# many nodes the marginal spreads over.
#
# The pass re-reads the ESS after every round and bisects again until the axis
# reaches `min_ess` or has taken `max_nodes` new nodes. A single round cannot
# certify what it produced: bisecting two heavy nodes can leave the mass on the
# new midpoint and one of them, and a marginal is only resolved once the ESS it
# ends on says so (gcol33/tulpa#858).
.hyper_consistency_pass <- function(theta_grid, log_marginal, extras,
                                    refining_axis, specs, kernel_fn,
                                    min_ess = .nl_diag("axis_sd_ess"),
                                    max_nodes = .nl_diag("axis_refine_nodes"),
                                    hp_fn = NULL) {
  refinable <- .hyper_refinable_names(specs)
  info <- list(axes = character(0), n_added = integer(0),
               ess_before = numeric(0), ess_after = numeric(0))
  n_added_total <- 0L
  if (length(refinable) == 0L) {
    return(list(theta_grid = theta_grid, log_marginal = log_marginal,
                extras = extras, refining_axis = refining_axis,
                info = NULL, n_added = 0L))
  }
  axis_ess <- function(axis, spec) {
    # Refinement grows theta_grid / log_marginal, so the log quadrature weights
    # are recomputed on every read to stay aligned with the current grid rows.
    log_quad <- .hyper_log_quad_weights(theta_grid, specs,
                                        refining = refining_axis)
    lm_eff <- log_marginal
    if (!is.null(log_quad) && length(log_quad) == length(lm_eff)) {
      lm_eff <- lm_eff + log_quad
      lm_eff[is.na(lm_eff)] <- -Inf
    }
    marg <- .nl_axis_marginal_logdensity(as.numeric(theta_grid[, axis]), lm_eff)
    cont <- !.hyper_is_atom_level(marg$vals, spec)
    list(marg = marg, ess = .nl_axis_quad_ess(marg$log_marg[cont]))
  }
  for (axis in refinable) {
    spec <- .hyper_spec_by_name(specs, axis)
    rd <- axis_ess(axis, spec)
    ess_before <- rd$ess
    if (!is.finite(ess_before) || ess_before >= min_ess) next
    # Every round anchors in the same row, the modal cell's, so the slice
    # points of one axis re-tile one fibre rather than scattering across rows.
    anchor_lev <- as.numeric(theta_grid[which.max(log_marginal), axis])
    added <- 0L
    while (added < max_nodes && is.finite(rd$ess) && rd$ess < min_ess) {
      new_pts <- .hyper_propose_mass_bisection(spec, rd$marg$vals,
                                               rd$marg$log_marg, min_ess)
      if (length(new_pts) == 0L) break
      new_pts <- utils::head(new_pts, max_nodes - added)
      pack <- .hyper_new_mode_tracked_triples(theta_grid, log_marginal, NULL,
                                               axis, new_pts, anchor_lev,
                                               refining_axis = refining_axis)
      if (is.null(pack)) break
      step <- .hyper_apply_axis_refinement(theta_grid, log_marginal, extras,
                                            refining_axis, list(pack), axis,
                                            specs, kernel_fn, hp_fn = hp_fn,
                                            consistency_tag = TRUE)
      if (step$n_new == 0L) break
      theta_grid    <- step$theta_grid
      log_marginal  <- step$log_marginal
      extras        <- step$extras
      refining_axis <- step$refining_axis
      added <- added + step$n_new
      rd <- axis_ess(axis, spec)
    }
    if (added == 0L) next
    info$axes       <- c(info$axes, axis)
    info$n_added    <- c(info$n_added, added)
    info$ess_before <- c(info$ess_before, ess_before)
    info$ess_after  <- c(info$ess_after, rd$ess)
    n_added_total   <- n_added_total + added
  }
  if (length(info$axes) == 0L) info <- NULL
  list(theta_grid = theta_grid, log_marginal = log_marginal,
       extras = extras, refining_axis = refining_axis, info = info,
       n_added = n_added_total)
}
