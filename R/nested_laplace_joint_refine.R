# Adaptive-grid refinement + var-of-means consistency passes.
#
# Part of the joint nested-Laplace driver; the public entry point
# tulpa_nested_laplace_joint() lives in nested_laplace_joint.R.

# --- adaptive grid refinement -----------------------------------------------
#
# Math ground.
#
# The joint marginal is approximated by
#   p(y) ~ sum_k w_k * exp(L_k),  L_k = log p(y, x_hat_k | theta_k),
# with normalised weights w_k. Posterior moments for any hyperparameter
# theta_j read
#   E[theta_j]   = sum_k w_k * theta_jk
#   Var[theta_j] = sum_k w_k * theta_jk^2 - E[theta_j]^2.
# When the user's grid for axis j is {t_1, ..., t_m} and the true posterior
# concentrates beyond the endpoints, the discretisation truncates the
# integrand. The resulting moments are biased toward the grid centre and
# their variance is bounded by the span of the grid points carrying weight.
#
# Refinement strategy: mode-tracked 1D slice (gcol33/tulpa#19).
#   1. Normalise log_marginal -> weights, project onto each refinable axis
#      by summing weight over the other axes (marginal weight per level).
#   2. For each refinable axis, check whether the lowest or highest level
#      carries marginal weight (or peak integrand density) above
#      `edge_thresh`. If so, propose:
#        a) one densification point at the (geometric, for log-spaced)
#           midpoint between the boundary and its inward neighbour,
#        b) two outward extension points past the boundary, mirroring the
#           spacing from the boundary to its neighbour.
#   3. Locate the boundary MAP cell: argmax of `log_marginal` over rows
#      whose refining-axis value equals the boundary level. Read off the
#      modal values of every *other* axis (sigma, alpha, rho/rho_car/tau,
#      phi_<arm>, ...) from that single cell.
#   4. Evaluate the kernel at each new axis point paired with the modal
#      other-axis values — one Newton-Laplace solve per new point, *not*
#      one per `(new_point) x (other_cartesian)`. For an outer grid with
#      M other-axis cells this is an M-fold cost cut vs. the legacy
#      cartesian refinement (often M = 100-400 in production).
#   5. Calibrate the slice cells onto the *marginal* scale before merging:
#      add `log S_b = logsumexp_{k: axis == boundary} (L_k - L_b_modal)`
#      to each slice cell's log_marginal. Without this, slice cells sit
#      at the conditional-MAP density and are systematically under-weighted
#      by a factor of S_b in the joint softmax. With it, slice cells
#      contribute the integrated marginal mass the cartesian refinement
#      would have produced (under the assumption that the conditional
#      posterior in the other axes is locally stable across the refining
#      axis — true at the tail because the other axes are anchored by the
#      rest of the grid, not the refining axis).
#   6. Append the slice cells to the joint grid. Modes and per-grid Q come
#      from the real kernel evaluation, so downstream total-variance and
#      field-decoding code reads correct values at the new cells.
#
# Single helper handles every refinable axis (`alpha`, `phi_<arm>`); the
# legacy cartesian path is gone — one code path, no fixed-vs-refined
# branching.

# Decide which axes are eligible for refinement.
#
# `alpha` (copy coefficient): at small effective sample size the
#   cover-arm likelihood barely identifies alpha and the posterior tail
#   can extend past the user's grid endpoint. Boundary refinement
#   catches the truncation; interior densification handles the case
#   where the posterior peak sits between grid levels.
#
# `phi_<arm>` (per-arm dispersion, e.g. beta concentration): the joint
#   engine integrates dispersion over an outer log-spaced grid; coarsening
#   that grid is the main lever on baseline wall time, but the posterior
#   peak can sit between grid levels and a discretisation bias creeps in.
#   Boundary + interior densification (mode-tracked) lets callers ship a
#   smaller default grid without regressing phi-recovery.
#
# Donor sigma is the spatial prior amplitude and is treated as a
# deliberate prior choice; extending it requires explicit user opt-in
# (separate feature). Spatial mixing axes (rho, rho_car, tau) are
# similarly fixed for now — they have small grids and the integrand
# tends to span the user range.
.refinable_axes <- function(grids, cp_has_copy) {
    out <- character(0)
    if (cp_has_copy) {
        lev <- sort(unique(as.numeric(grids$alpha)))
        if (length(lev) >= 2L) out <- c(out, "alpha")
    }
    phi_cols <- grep("^phi_", names(grids), value = TRUE)
    for (col in phi_cols) {
        lev <- sort(unique(as.numeric(grids[[col]])))
        if (length(lev) >= 2L) out <- c(out, col)
    }
    out
}

# Per-axis edge diagnostics. Marginal weight gives the integrated posterior
# share on the boundary level — useful when the integrand has clearly
# decayed but the user's grid is just wide enough that the boundary
# *cell* still contributes. Integrand density at the boundary
# (`max_lm[boundary] - max_lm[overall]`, exponentiated) catches the
# complementary failure: when the boundary level is the heaviest *point-
# wise* mass but the user's grid is so narrow that very little is left
# inside it, so the boundary's *integrated* weight stays small while the
# truncated tail beyond the grid still carries posterior mass.
#
# We trigger refinement when EITHER criterion exceeds the threshold,
# which catches both failure modes with a single helper.
#
# Returns a named list keyed by axis with sorted unique `levels` and the
# scalar `min_score`, `max_score` to compare against `edge_thresh`.
.axis_edge_scores <- function(grids, log_marginal, axes) {
    if (length(log_marginal) == 0L) return(list())
    lm_max_total <- max(log_marginal)
    weights      <- .nl_normalise_weights(log_marginal)
    out <- list()
    for (a in axes) {
        v   <- as.numeric(grids[[a]])
        lev <- sort(unique(v))
        if (length(lev) < 2L) next
        # Integrated marginal weight at each boundary.
        w_lo <- sum(weights[v == lev[1L]])
        w_hi <- sum(weights[v == lev[length(lev)]])
        # Peak integrand density at each boundary (normalised by overall
        # peak): exp(max_lm@boundary - max_lm@overall).
        d_lo <- exp(max(log_marginal[v == lev[1L]])               - lm_max_total)
        d_hi <- exp(max(log_marginal[v == lev[length(lev)]])      - lm_max_total)
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

# Axis-aware spacing: sigma / alpha / tau live on a log scale; rho and
# rho_car live on a linear scale. Per-arm phi axes (`phi_<arm>`) follow
# the GP-lengthscale convention — strictly positive, log-spaced.
.axis_is_log_scale <- function(axis_name) {
    axis_name %in% c("sigma", "alpha", "tau", "sigma2",
                      "phi_gp", "lengthscale") ||
        startsWith(axis_name, "phi_")
}

# Natural domain clamp for bounded axes. NULL means unbounded (so just
# log-scale-positive for sigma/tau axes, which the log midpoint handles).
.axis_bounds <- function(axis_name) {
    if (axis_name == "sigma")     return(c(0, Inf))   # field amplitude >= 0
    if (axis_name == "alpha")     return(c(0, Inf))   # copy coefficient >= 0
    if (axis_name == "rho")       return(c(0, 1))     # BYM2/AR1 mixing fraction
    if (axis_name == "rho_car")   return(c(-Inf, 1))  # proper-CAR (eigenvalue gated upstream)
    if (startsWith(axis_name, "phi_")) return(c(0, Inf))  # dispersion > 0
    NULL
}

# Propose new levels on one axis. Always adds one densification point
# between the boundary and its inward neighbour. Adds an outward extension
# only when the boundary is genuinely in the integrand's *tail*. When the
# boundary is the local mode, extension is suppressed because the data has
# not signalled that the true peak lies further out — densification alone
# safely tightens the integration in the heavy region without expanding
# the prior support beyond what the user requested.
.propose_axis_extension <- function(axis_name, lev, side,
                                     extend_ok = TRUE) {
    log_scale <- .axis_is_log_scale(axis_name)
    bounds    <- .axis_bounds(axis_name)
    mid <- if (log_scale) {
        function(a, b) exp(0.5 * (log(a) + log(b)))
    } else {
        function(a, b) 0.5 * (a + b)
    }
    if (side == "max") {
        edge      <- lev[length(lev)]
        neighbour <- lev[length(lev) - 1L]
        densify   <- mid(neighbour, edge)
        extend1   <- if (log_scale) edge * (edge / neighbour)
                     else            edge + (edge - neighbour)
        # Second extension point catches seeds where the integrand is
        # still appreciable one step past the boundary; without it we
        # under-cover when the true peak sits two steps beyond the user
        # grid.
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
        pts <- pts[pts >= bounds[1L] & pts <= bounds[2L]]
    }
    # De-duplicate vs the existing levels at numerical tolerance.
    keep <- vapply(pts, function(p) {
        all(abs(lev - p) > 1e-8 * max(1, abs(p)))
    }, logical(1))
    pts[keep]
}

# Propose midpoint(s) on one axis to densify around an interior peak.
# `mode_idx` is the level index of the peak (1 < mode_idx < length(lev));
# `do_left` adds the midpoint between (mode_idx - 1, mode_idx),
# `do_right` adds the midpoint between (mode_idx, mode_idx + 1).
# Spacing follows the axis scale (geometric mean on log-spaced axes,
# arithmetic on linear axes). De-duplicated against the existing grid.
.propose_interior_densification <- function(axis_name, lev, mode_idx,
                                             do_left = FALSE, do_right = FALSE) {
    log_scale <- .axis_is_log_scale(axis_name)
    mid <- if (log_scale) {
        function(a, b) exp(0.5 * (log(a) + log(b)))
    } else {
        function(a, b) 0.5 * (a + b)
    }
    pts <- numeric(0)
    if (do_left  && mode_idx > 1L)              pts <- c(pts, mid(lev[mode_idx - 1L], lev[mode_idx]))
    if (do_right && mode_idx < length(lev))     pts <- c(pts, mid(lev[mode_idx], lev[mode_idx + 1L]))
    if (length(pts) == 0L) return(pts)
    bounds <- .axis_bounds(axis_name)
    if (!is.null(bounds)) {
        pts <- pts[pts >= bounds[1L] & pts <= bounds[2L]]
    }
    keep <- vapply(pts, function(p) {
        all(abs(lev - p) > 1e-8 * max(1, abs(p)))
    }, logical(1))
    pts[keep]
}

# Build mode-tracked slice triples for one (axis, anchor-level) pair plus
# the matching marginal-scale calibration. For each new axis point we
# produce *one* triple at (axis = p, others = modal_at_anchor), not a
# full cartesian — this is the M-fold cost cut.
#
# `anchor_lev` is the existing axis level whose modal cell we use as the
# anchor for the conditional posterior on the other axes:
#   * boundary refinement (peak at/near edge): anchor is the boundary
#     level (outer for max-side, inner for min-side).
#   * interior densification (peak between grid levels): anchor is the
#     peak level; new points sit on either side. The conditional posterior
#     is locally stable across one grid step in either direction.
#
# Returns NULL when no new points were given. Otherwise:
#   triples       : named list of equal-length vectors, kernel-input shaped
#                   (alpha is present iff `cp_has_copy`).
#   calibration   : numeric, one entry per new point, added to that cell's
#                   raw log_marginal before it is merged into the joint
#                   grid. `log S_b` defined in the file-level math note.
#   warm_start_idx: index into the joint grid of the anchor MAP cell.
.new_mode_tracked_triples <- function(grids, log_marginal, axis_name,
                                       new_pts, anchor_lev, cp_has_copy) {
    if (length(new_pts) == 0L) return(NULL)
    v <- as.numeric(grids[[axis_name]])
    mask <- abs(v - anchor_lev) < 1e-12 * max(1, abs(anchor_lev))
    if (!any(mask)) return(NULL)
    anchor_lm   <- log_marginal[mask]
    k_map_local <- which.max(anchor_lm)
    idx_global  <- which(mask)[k_map_local]
    L_map       <- anchor_lm[k_map_local]
    # log S_a = log sum_{k @ anchor} exp(L_k - L_anchor_modal). >= 0.
    # Equals 0 when the conditional posterior collapses to a single cell,
    # log(M) at most (M = number of other-axis cells) when it is flat.
    calibration <- log(sum(exp(anchor_lm - L_map)))

    # Modal values of every *other* axis at the boundary MAP cell. Empty
    # axes (alpha when has_copy = FALSE) stay empty.
    full_axes <- if (cp_has_copy) names(grids) else setdiff(names(grids), "alpha")
    other_axes <- setdiff(full_axes, axis_name)
    triples <- vector("list", length(names(grids)))
    names(triples) <- names(grids)
    for (a in names(grids)) {
        if (a == axis_name) {
            triples[[a]] <- as.numeric(new_pts)
        } else if (a %in% other_axes && length(grids[[a]]) > 0L) {
            triples[[a]] <- rep(grids[[a]][idx_global], length(new_pts))
        } else {
            triples[[a]] <- numeric(0)
        }
    }
    list(triples       = triples,
         calibration   = rep(calibration, length(new_pts)),
         # Index into the original joint grid of the boundary MAP cell
         # whose modes are the best available warm-start for the slice
         # kernel call (first slice triple reuses it; later triples
         # chain prev_mode within the kernel).
         warm_start_idx = idx_global)
}

# Stitch the slice triples from multiple (axis, side) pairs into a single
# kernel-input shape, alongside one calibration vector. Drops any rows
# whose all-axis key already appears in `grids` (numerical tolerance via
# the same `%.10g` key format the legacy path used).
.concat_slice_triples <- function(triple_packs, grids, cp_has_copy) {
    if (length(triple_packs) == 0L) return(NULL)
    axes <- names(grids)
    combined <- vector("list", length(axes))
    names(combined) <- axes
    for (a in axes) {
        parts <- lapply(triple_packs, function(p) p$triples[[a]] %||% numeric(0))
        combined[[a]] <- as.numeric(unlist(parts, use.names = FALSE))
    }
    calib <- as.numeric(unlist(lapply(triple_packs, `[[`, "calibration"),
                                use.names = FALSE))
    n_new <- length(combined[[axes[1L]]])
    if (n_new == 0L) return(NULL)

    # De-duplicate against the existing grid on the active (non-empty)
    # axes. alpha when has_copy = FALSE has length 0 and is ignored.
    active <- vapply(axes, function(a) length(grids[[a]]) > 0L, logical(1))
    active_axes <- axes[active]
    fmt <- function(lst) {
        if (length(lst[[active_axes[1L]]]) == 0L) return(character(0))
        cols <- lapply(active_axes, function(a) sprintf("%.10g", lst[[a]]))
        do.call(paste, c(cols, sep = ":"))
    }
    new_keys <- fmt(combined)
    old_keys <- fmt(grids)
    keep <- !new_keys %in% old_keys
    if (!any(keep)) return(NULL)
    for (a in axes) {
        if (length(combined[[a]]) > 0L) combined[[a]] <- combined[[a]][keep]
    }
    calib <- calib[keep]
    if (!cp_has_copy) combined$alpha <- numeric(0)
    list(triples = combined, calibration = calib)
}

# Concatenate two kernel results row-wise, matching the run_nested_laplace_grid
# output layout (log_marginal, modes, n_iter, optional Q_csc_* lists).
# Also carries the per-cell `refining_axis` tag so per-axis moment
# recalibration downstream can drop slice cells that don't vary the axis
# whose marginal is being computed. Cells without an explicit tag (e.g.
# the initial cartesian pass) default to "" (= varies on every axis).
.concat_kernel_results <- function(a, b) {
    out <- a
    out$log_marginal <- c(a$log_marginal, b$log_marginal)
    out$n_iter       <- c(a$n_iter,       b$n_iter)
    out$n_grid       <- a$n_grid + b$n_grid
    if (!is.null(a$modes) && !is.null(b$modes)) {
        out$modes <- rbind(a$modes, b$modes)
    }
    tag_a <- a$refining_axis %||% rep("", length(a$log_marginal))
    tag_b <- b$refining_axis %||% rep("", length(b$log_marginal))
    out$refining_axis <- c(tag_a, tag_b)
    if (!is.null(a$Q_csc_p_per_grid) || !is.null(b$Q_csc_p_per_grid)) {
        out$Q_csc_p_per_grid <- c(a$Q_csc_p_per_grid %||% vector("list", a$n_grid),
                                  b$Q_csc_p_per_grid %||% vector("list", b$n_grid))
        out$Q_csc_i_per_grid <- c(a$Q_csc_i_per_grid %||% vector("list", a$n_grid),
                                  b$Q_csc_i_per_grid %||% vector("list", b$n_grid))
        out$Q_csc_x_per_grid <- c(a$Q_csc_x_per_grid %||% vector("list", a$n_grid),
                                  b$Q_csc_x_per_grid %||% vector("list", b$n_grid))
        out$Q_csc_n <- a$Q_csc_n %||% b$Q_csc_n
    }
    out
}

# Merge new axis triples into the kernel-input `grids` representation
# (paired vectors, alpha empty when has_copy = FALSE).
.merge_grids <- function(grids, new_triples, cp_has_copy) {
    full_axes <- if (cp_has_copy) names(grids) else setdiff(names(grids), "alpha")
    out <- grids
    for (a in full_axes) {
        out[[a]] <- c(grids[[a]], new_triples[[a]])
    }
    if (!cp_has_copy) out$alpha <- numeric(0)
    out
}

# Detect refinement triggers on one axis and return the slice triple_pack
# (or NULL if nothing triggers). Pulled out so the outer pass can apply
# axes sequentially and let later axes see the updated grid/posterior.
.detect_axis_refinement <- function(grids, res, edge_info_a, axis_name,
                                     edge_thresh, cp_has_copy) {
    ei  <- edge_info_a
    lev <- ei$levels
    v   <- as.numeric(grids[[axis_name]])
    lm_max_at_lev <- vapply(lev, function(lv) {
        max(res$log_marginal[v == lv])
    }, numeric(1))
    mode_idx <- which.max(lm_max_at_lev)
    n_lev    <- length(lev)

    # Boundary triggers (peak-at-edge or tail-mass-at-edge).
    tr_min <- ei$min_score >= edge_thresh &&
              (mode_idx == 1L || ei$min_dens >= edge_thresh)
    tr_max <- ei$max_score >= edge_thresh &&
              (mode_idx == n_lev || ei$max_dens >= edge_thresh)

    # Interior densification trigger. When the posterior on the refining
    # axis is narrower than the grid spacing, weights collapse to a
    # single grid point and the posterior mean snaps to that grid value
    # — no continuous interpolation. Adjacent-level density rounds to
    # zero in that regime, so a density-based trigger cannot catch the
    # discretisation bias. Trigger on *grid coarseness* instead: when
    # the log-axis neighbour ratio exceeds `interior_log_step`, an
    # interior peak is suspected of sub-grid mismatch — densify between
    # peak and neighbour. Thresholds tuned to "always fire on a 7-point
    # log grid over [2, 300] (ratio ~2.4), never fire on a 25-point
    # grid (ratio ~1.25)".
    interior_log_step <- 0.55
    interior_lin_step <- 0.25
    wide_left <- wide_right <- FALSE
    if (mode_idx > 1L && mode_idx < n_lev) {
        if (.axis_is_log_scale(axis_name)) {
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
        pts <- .propose_axis_extension(axis_name, lev, "max", extend_ok = TRUE)
        pk  <- .new_mode_tracked_triples(grids, res$log_marginal, axis_name,
                                          pts,
                                          anchor_lev   = lev[n_lev],
                                          cp_has_copy  = cp_has_copy)
        if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
    }
    if (tr_min) {
        pts <- .propose_axis_extension(axis_name, lev, "min", extend_ok = TRUE)
        pk  <- .new_mode_tracked_triples(grids, res$log_marginal, axis_name,
                                          pts,
                                          anchor_lev   = lev[1L],
                                          cp_has_copy  = cp_has_copy)
        if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
    }
    if (wide_left || wide_right) {
        pts <- .propose_interior_densification(axis_name, lev, mode_idx,
                                                wide_left, wide_right)
        pk  <- .new_mode_tracked_triples(grids, res$log_marginal, axis_name,
                                          pts,
                                          anchor_lev   = lev[mode_idx],
                                          cp_has_copy  = cp_has_copy)
        if (!is.null(pk)) packs[[length(packs) + 1L]] <- pk
    }
    if (length(packs) == 0L) return(NULL)
    packs
}

# Run the slice kernel for one axis's triple_packs and merge the result
# into (res, grids). Warm-start from the first pack's anchor MAP cell.
# Tags each new cell with `axis_name` so per-axis moment recalibration
# downstream can exclude cells that fix this axis at a non-varying value
# from OTHER axes' marginals.
.apply_axis_refinement <- function(grids, res, triple_packs, axis_name,
                                    backend, arms, prior, cp,
                                    max_iter, tol, n_threads, x_init,
                                    store_Q, arm_names, hp_fn = NULL) {
    merged <- .concat_slice_triples(triple_packs, grids, cp$has_copy)
    if (is.null(merged)) return(list(grids = grids, res = res, n_new = 0L))
    new_triples <- merged$triples
    calibration <- merged$calibration

    # Warm-start the slice kernel call from the first pack's anchor MAP
    # mode. The kernel chains prev_mode forward within a call, so this
    # skips a cold Newton on triple 1; subsequent triples benefit
    # transitively. Slice cells live close in theta-space to the
    # anchor, so the converged anchor mode is the best initial guess.
    slice_x_init <- x_init
    if (!is.null(res$modes) && is.matrix(res$modes)) {
        idx0 <- triple_packs[[1L]]$warm_start_idx
        if (length(idx0) == 1L && idx0 >= 1L && idx0 <= nrow(res$modes)) {
            slice_x_init <- as.numeric(res$modes[idx0, ])
        }
    }

    res_extra <- backend$call_kernel(arms, prior, cp, new_triples,
                                      max_iter, tol, n_threads,
                                      slice_x_init, isTRUE(store_Q),
                                      arm_names = arm_names)
    # Lift slice cells onto the marginal scale before merging. Same
    # softmax invariant the cartesian path used to give us for free.
    res_extra$log_marginal  <- res_extra$log_marginal + calibration
    # Bake the regularizing hyperprior into the new cells' log_marginal
    # before merging, so the appended cells join an invariant where
    # every cell carries its prior contribution (gcol33/tulpa#22).
    if (!is.null(hp_fn)) {
        hp_new <- hp_fn(new_triples)
        if (!is.null(hp_new) && length(hp_new) == length(res_extra$log_marginal)) {
            res_extra$log_marginal <- res_extra$log_marginal + hp_new
        }
    }
    res_extra$refining_axis <- rep(axis_name, length(res_extra$log_marginal))

    res   <- .concat_kernel_results(res, res_extra)
    grids <- .merge_grids(grids, new_triples, cp$has_copy)
    n_new <- length(new_triples[[names(grids)[which(vapply(grids, length,
                                                            integer(1)) > 0L)[1L]]]])
    list(grids = grids, res = res, n_new = n_new)
}

# Priority order for refining axes. `alpha` (boundary truncation on the
# copy coefficient) is refined first; any `phi_<arm>` axes (interior
# densification — sharpens marginal recovery without shifting joint
# structure) are refined second so they read the post-`alpha`-extension
# modal. Other axes fall in declaration order.
.axis_refinement_order <- function(axes) {
    priority <- function(a) {
        if (a == "alpha") 1L
        else if (startsWith(a, "phi_")) 2L
        else 3L
    }
    axes[order(vapply(axes, priority, integer(1)), seq_along(axes))]
}

# Main entry: run up to `max_passes` of refinement. Each pass processes
# refinable axes *sequentially* (one kernel call per axis), so each axis
# sees the grid/posterior updates from any earlier axis in the same pass.
# Re-uses the backend kernel — one code path, no fixed-grid fallback.
# Stops early when no axis triggers.
.adaptive_refine_pass <- function(grids, res, backend, arms, prior, cp,
                                  max_iter, tol, n_threads, x_init, store_Q,
                                  edge_thresh, max_passes,
                                  arm_names = NULL, hp_fn = NULL) {
    info <- list(triggered_axes = character(0),
                 n_points_added = integer(0))
    if (max_passes < 1L) {
        return(list(grids = grids, res = res, info = NULL))
    }
    for (pass in seq_len(max_passes)) {
        axes <- .axis_refinement_order(.refinable_axes(grids, cp$has_copy))
        if (length(axes) == 0L) break
        any_triggered <- FALSE
        triggered_this_pass <- character(0)
        for (a in axes) {
            edge_info <- .axis_edge_scores(grids, res$log_marginal, a)
            ei <- edge_info[[a]]
            if (is.null(ei)) next
            packs <- .detect_axis_refinement(grids, res, ei, a,
                                              edge_thresh, cp$has_copy)
            if (is.null(packs)) next
            step <- .apply_axis_refinement(grids, res, packs, a, backend,
                                            arms, prior, cp, max_iter, tol,
                                            n_threads, x_init, store_Q,
                                            arm_names, hp_fn = hp_fn)
            if (step$n_new == 0L) next
            grids <- step$grids
            res   <- step$res
            triggered_this_pass <- c(triggered_this_pass, a)
            info$n_points_added <- c(info$n_points_added, step$n_new)
            any_triggered <- TRUE
        }
        if (!any_triggered) break
        info$triggered_axes <- c(info$triggered_axes,
                                  paste(triggered_this_pass, collapse = ","))
    }
    if (length(info$triggered_axes) == 0L) info <- NULL
    list(grids = grids, res = res, info = info)
}


# Propose new slice points on `axis_name` covering the Laplace-implied
# support `mu +/- k*sd` (k = 0.7, 1.5; symmetric, four points). Log-scale
# axes get points on the log scale via delta-method (log_sd = sd / mu).
# Points falling outside the axis's natural bounds, or within ~5% spacing
# of an existing level, are dropped. Returns numeric() if no usable points
# remain.
.propose_consistency_points <- function(axis_name, mu, sd, lev) {
    if (!is.finite(mu) || !is.finite(sd) || sd <= 0) return(numeric(0))
    is_log <- .axis_is_log_scale(axis_name)
    if (is_log) {
        if (mu <= 0) return(numeric(0))
        log_mu <- log(mu)
        log_sd <- sd / mu
        if (!is.finite(log_sd) || log_sd <= 0) return(numeric(0))
        pts <- exp(log_mu + c(-1.5, -0.7, 0.7, 1.5) * log_sd)
    } else {
        pts <- mu + c(-1.5, -0.7, 0.7, 1.5) * sd
    }
    bounds <- .axis_bounds(axis_name)
    if (!is.null(bounds)) {
        pts <- pts[pts > bounds[1L] & pts < bounds[2L]]
    }
    if (length(pts) == 0L) return(numeric(0))
    keep <- vapply(pts, function(p) {
        if (is_log) {
            !any(abs(log(lev) - log(p)) < 0.05)
        } else {
            !any(abs(lev - p) < 0.05 * max(abs(lev), 1))
        }
    }, logical(1))
    pts[keep]
}

# Var-of-means consistency pass. For each refinable axis with a finite
# Laplace SD whose joint-grid var-of-means falls short of that SD by more
# than `tolerance`, add four Laplace-guided slice points anchored at the
# overall modal cell (other axes pinned). One kernel call per axis;
# re-uses `.apply_axis_refinement` for the merge.
#
# Triggers ONLY when the discrepancy exceeds `tolerance` (default 0.7),
# so axes whose default grid already resolves the marginal pay no
# extra kernel time. See gcol33/tulpa#21.
.nl_var_of_means_consistency_pass <- function(grids, res, backend, arms, prior,
                                               cp, max_iter, tol, n_threads,
                                               x_init, store_Q,
                                               arm_names = NULL,
                                               tolerance = 0.7,
                                               hp_fn = NULL) {
    refinable <- .refinable_axes(grids, cp$has_copy)
    info <- list(axes = character(0), n_added = integer(0),
                 vom_before = numeric(0), sd_laplace = numeric(0))
    n_added_total <- 0L
    if (length(refinable) == 0L) {
        return(list(grids = grids, res = res, info = NULL, n_added = 0L))
    }
    overall_mode_idx <- which.max(res$log_marginal)
    for (axis in refinable) {
        sd_lap <- res$theta_sd[[axis]] %||% NA_real_
        mu     <- res$theta_mean[[axis]] %||% NA_real_
        if (!is.finite(sd_lap) || !is.finite(mu) || sd_lap <= 0) next

        axis_vals <- as.numeric(res$theta_grid[, axis])
        vom_mean  <- sum(res$weights * axis_vals)
        vom_sd    <- sqrt(max(0, sum(res$weights * axis_vals^2) - vom_mean^2))
        if (vom_sd >= tolerance * sd_lap) next

        lev <- sort(unique(as.numeric(grids[[axis]])))
        new_pts <- .propose_consistency_points(axis, mu, sd_lap, lev)
        if (length(new_pts) == 0L) next

        # Anchor at the overall modal cell's value on `axis`. The slice
        # builder pins other axes at the cells where `axis == anchor_lev`
        # has the maximum log_marginal, which is the overall mode here.
        anchor_lev <- as.numeric(grids[[axis]])[overall_mode_idx]
        pack <- .new_mode_tracked_triples(grids, res$log_marginal, axis,
                                           new_pts, anchor_lev, cp$has_copy)
        if (is.null(pack)) next

        n_before <- length(res$log_marginal)
        step <- .apply_axis_refinement(grids, res, list(pack), axis, backend,
                                        arms, prior, cp, max_iter, tol,
                                        n_threads, x_init, store_Q,
                                        arm_names, hp_fn = hp_fn)
        if (step$n_new == 0L) next
        grids <- step$grids
        res   <- step$res
        # Re-tag the appended cells with a `consistency_<axis>` prefix.
        # `.apply_axis_refinement` tags them as `<axis>` (its standard slice
        # tag), but the consistency pass pins other axes at the modal cell
        # and these slices would distort posterior moments on axes the
        # consistency pass didn't refine. The prefix lets
        # `.joint_recalibrate_axis_moments` still pick them up for the
        # refined axis's own marginal (via the `consistency_<axis>` keep
        # check) while derived-quantity moments naturally skip them.
        n_after <- length(res$log_marginal)
        if (n_after > n_before && !is.null(res$refining_axis)) {
            new_idx <- (n_before + 1L):n_after
            res$refining_axis[new_idx] <- paste0("consistency_", axis)
        }
        info$axes        <- c(info$axes, axis)
        info$n_added     <- c(info$n_added, step$n_new)
        info$vom_before  <- c(info$vom_before, vom_sd)
        info$sd_laplace  <- c(info$sd_laplace, sd_lap)
        n_added_total    <- n_added_total + step$n_new
    }
    # Note (2026-05-19): after the (sigma, alpha) reparameterization,
    # alpha is a regular refinable axis -- the loop above already picks
    # it up through `.refinable_axes`. The previous bespoke alpha-axis
    # extension (via the delta-method-on-log-sigma composition) is gone.
    if (length(info$axes) == 0L) info <- NULL
    list(grids = grids, res = res, info = info, n_added = n_added_total)
}
