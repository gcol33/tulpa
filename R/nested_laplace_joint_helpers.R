# Grid / arm / copy / layout helpers.
#
# Part of the joint nested-Laplace driver; the public entry point
# tulpa_nested_laplace_joint() lives in nested_laplace_joint.R.

# --- helpers -----------------------------------------------------------------

# Cartesian product over a named list of spatial axes plus an optional
# alpha axis (copy coefficient) and optional per-arm phi axes.
# `phi_axes` is a list keyed by arm name; entries are either NULL/empty (no
# axis for that arm) or numeric vectors that become a new outer-grid axis
# named `phi_<arm>`. Returns a named list of paired vectors of identical
# length, ready to feed the C++ kernel. Phi axes vary slowest (added last)
# so within-spatial-block warm-starts stay good.
.joint_cartesian <- function(axes, has_copy, alpha_axis, phi_axes = NULL) {
    full <- if (has_copy) c(axes, list(alpha = alpha_axis)) else axes
    if (!is.null(phi_axes)) {
        active <- phi_axes[vapply(phi_axes, length, integer(1)) > 0L]
        if (length(active) > 0L) {
            names(active) <- paste0("phi_", names(active))
            full <- c(full, active)
        }
    }
    gr <- do.call(expand.grid,
                  c(full, list(KEEP.OUT.ATTRS = FALSE,
                                stringsAsFactors = FALSE)))
    out <- as.list(gr)
    if (!has_copy) out$alpha <- numeric(0)
    out
}

# Append any `phi_<arm>` columns from `grids` onto a backend's spatial
# theta_grid matrix so downstream posterior-moment helpers see phi as a
# regular hyperparameter axis.
.append_phi_columns <- function(base, grids) {
    phi_cols <- grep("^phi_", names(grids), value = TRUE)
    if (length(phi_cols) == 0L) return(base)
    extra <- do.call(cbind, lapply(phi_cols, function(c) {
        out <- as.numeric(grids[[c]]); attr(out, "name") <- c; out
    }))
    colnames(extra) <- phi_cols
    cbind(base, extra)
}

# Build the `phi_grid_per_arm` argument for the C++ kernels from a
# Cartesian-product `grids` list and arm names. Returns a list of length
# `n_arms`: entry k is either `NULL` (no phi axis for that arm — kernel
# uses the parse-time scalar phi) or a NumericVector of length n_grid
# matching the flat outer-grid size. Phi columns in `grids` follow the
# `phi_<arm_name>` convention produced by `.joint_cartesian`.
.joint_phi_grid_per_arm <- function(grids, arm_names) {
    out <- vector("list", length(arm_names))
    any_active <- FALSE
    for (k in seq_along(arm_names)) {
        col <- paste0("phi_", arm_names[k])
        if (!is.null(grids[[col]])) {
            out[[k]] <- as.numeric(grids[[col]])
            any_active <- TRUE
        }
    }
    if (!any_active) NULL else out
}

# Compute a tile partition for the outer-grid loop's three-tier warm-start
# (Phase 2 of the speedup plan, dev_notes/speedup.md). A tile groups all
# outer-grid cells that share every hyperparameter coordinate except the
# copy coefficient alpha — for the joint copy block under the (sigma, alpha)
# reparam, the shared latent prior Q and the donor-arm linear predictor are
# tile-constant, so the joint mode varies smoothly across the alpha axis
# within a tile. Using the tile's median-alpha cell as a warm-start for the
# remaining alpha cells saves 1-2 Newton iters each.
#
# Arguments:
#   non_alpha_matrix - numeric matrix [n_grid x n_axes_excl_alpha]; row k
#                      is the non-alpha coordinates of cell k.
#   alpha_vec        - numeric [n_grid]; alpha value for each cell.
#   n_grid           - integer; number of outer-grid cells (= nrow).
#
# Returns NULL when the partition has no useful structure (every cell its
# own tile, or only one tile total); otherwise a list with 0-based
# integer fields:
#   tile_ids[k]        - tile membership of cell k.
#   tile_pilot_cells[t]- cell index used as tile t's representative pilot.
#                        When the global pilot cell (n_grid %/% 2) falls
#                        into tile t, it is reused so the Tier-2 pass
#                        does not re-solve that tile.
.joint_compute_tile_partition <- function(non_alpha_matrix, alpha_vec, n_grid) {
    if (is.null(non_alpha_matrix) || ncol(non_alpha_matrix) == 0L) return(NULL)
    if (nrow(non_alpha_matrix) != n_grid || length(alpha_vec) != n_grid) {
        return(NULL)
    }
    keys <- do.call(paste, c(
        lapply(seq_len(ncol(non_alpha_matrix)), function(j) {
            formatC(non_alpha_matrix[, j], digits = 15L, format = "g")
        }),
        list(sep = "\r")
    ))
    uniq_keys <- unique(keys)
    tile_ids <- match(keys, uniq_keys) - 1L  # 0-based
    n_tiles <- length(uniq_keys)
    if (n_tiles <= 1L || n_tiles >= n_grid) return(NULL)

    # Tier-1 global pilot cell matches the C++ driver's `n_grid / 2`.
    k_global_pilot <- as.integer(n_grid %/% 2L)
    tile_of_global <- tile_ids[k_global_pilot + 1L]

    tile_pilot_cells <- integer(n_tiles)
    for (t in seq_len(n_tiles) - 1L) {
        if (t == tile_of_global) {
            # Global pilot doubles as this tile's pilot — no Tier-2 solve.
            tile_pilot_cells[t + 1L] <- k_global_pilot
            next
        }
        cells_1based <- which(tile_ids == t)
        med <- stats::median(alpha_vec[cells_1based])
        best <- cells_1based[which.min(abs(alpha_vec[cells_1based] - med))][1L]
        tile_pilot_cells[t + 1L] <- as.integer(best - 1L)  # 0-based
    }
    list(tile_ids        = as.integer(tile_ids),
         tile_pilot_cells = as.integer(tile_pilot_cells))
}

# Normalise the user-facing `phi_grid` argument into a list keyed by arm
# name, with NULL entries for arms without a phi axis. Accepts either a
# named list (subset of arm names) or a positional list of length n_arms.
# Single-element entries are treated as no-axis (the parse-time scalar phi
# already serves as that arm's dispersion).
.normalise_phi_grid <- function(phi_grid, arm_names) {
    if (is.null(phi_grid)) return(NULL)
    if (!is.list(phi_grid)) {
        stop("`phi_grid` must be a list (named by arm or positional).",
             call. = FALSE)
    }
    out <- vector("list", length(arm_names))
    names(out) <- arm_names
    if (!is.null(names(phi_grid))) {
        unknown <- setdiff(names(phi_grid), arm_names)
        if (length(unknown) > 0L) {
            stop("`phi_grid` names not in `responses`: ",
                 paste(shQuote(unknown), collapse = ", "), ".", call. = FALSE)
        }
        for (nm in names(phi_grid)) {
            v <- phi_grid[[nm]]
            if (!is.null(v) && length(v) > 1L) out[[nm]] <- as.numeric(v)
        }
    } else {
        if (length(phi_grid) != length(arm_names)) {
            stop("positional `phi_grid` must have length n_arms (",
                 length(arm_names), ").", call. = FALSE)
        }
        for (k in seq_along(phi_grid)) {
            v <- phi_grid[[k]]
            if (!is.null(v) && length(v) > 1L) out[[k]] <- as.numeric(v)
        }
    }
    out
}

# Validate one arm spec and fill in defaults.
.normalise_joint_arm <- function(a, k) {
    if (!is.list(a)) {
        stop("Arm ", k, ": expected a list of arm spec fields.", call. = FALSE)
    }
    must_have <- c("y", "X", "spatial_idx", "family")
    missing <- setdiff(must_have, names(a))
    if (length(missing)) {
        stop("Arm ", k, ": missing fields ", paste(shQuote(missing), collapse = ", "),
             ".", call. = FALSE)
    }
    N <- length(a$y)
    a$y <- as.numeric(a$y)
    if (!is.matrix(a$X)) {
        stop("Arm ", k, ": `X` must be a numeric matrix.", call. = FALSE)
    }
    if (nrow(a$X) != N) {
        stop("Arm ", k, ": nrow(X) (", nrow(a$X), ") must equal length(y) (",
             N, ").", call. = FALSE)
    }
    if (length(a$spatial_idx) != N) {
        stop("Arm ", k, ": length(spatial_idx) (", length(a$spatial_idx),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$spatial_idx <- as.integer(a$spatial_idx)
    a$n_trials <- if (is.null(a$n_trials)) rep(1L, N) else as.integer(a$n_trials)
    if (length(a$n_trials) != N) {
        stop("Arm ", k, ": length(n_trials) (", length(a$n_trials),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$re_idx <- if (is.null(a$re_idx)) rep(0, N) else as.numeric(a$re_idx)
    if (length(a$re_idx) != N) {
        stop("Arm ", k, ": length(re_idx) (", length(a$re_idx),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    a$n_re_groups <- as.integer(a$n_re_groups %||% 0L)
    a$sigma_re    <- as.numeric(a$sigma_re %||% 1.0)
    a$family      <- as.character(a$family)
    a$phi         <- as.numeric(a$phi %||% 1.0)
    a
}

# Decide which arm (if any) is the copy arm and what the alpha grid is.
# Single-block path; `copy$alpha_grid` is the outer-grid axis on the copy
# coefficient.
.resolve_copy <- function(copy, responses, prior, type) {
    if (is.null(copy)) {
        return(list(has_copy = FALSE, copy_arm_zero = -1L,
                    alpha_grid = numeric(0)))
    }
    arm_id <- copy$arm
    if (is.null(arm_id)) {
        stop("`copy$arm` must be a name or 1-based index.", call. = FALSE)
    }
    if (is.character(arm_id)) {
        nm <- names(responses)
        if (is.null(nm) || !arm_id %in% nm) {
            stop("`copy$arm` = '", arm_id, "' not found in names(responses).",
                 call. = FALSE)
        }
        arm_zero <- match(arm_id, nm) - 1L
    } else {
        arm_zero <- as.integer(arm_id) - 1L
        if (arm_zero < 0L || arm_zero >= length(responses)) {
            stop("`copy$arm` index out of range.", call. = FALSE)
        }
    }
    if (!is.null(copy$alpha_grid)) {
        alpha_axis <- as.numeric(copy$alpha_grid)
    } else {
        # Default: a small log-spaced grid in [~0.1, ~3], with 0 included so
        # the alpha=0 base-model atom carries posterior mass when the data
        # supports "no copy".
        alpha_axis <- c(0, exp(seq(log(0.1), log(3), length.out = 5)))
    }
    if (length(alpha_axis) == 0L) {
        stop("`copy$alpha_grid` must have at least one non-negative value.",
             call. = FALSE)
    }
    if (any(alpha_axis < 0)) {
        stop("`copy$alpha_grid` values must be non-negative.",
             call. = FALSE)
    }
    list(has_copy = TRUE, copy_arm_zero = arm_zero,
         alpha_grid = alpha_axis)
}

# Recalibrate per-axis posterior moments after the joint pass. Slice
# cells from mode-tracked refinement on axis Y are pinned at modal
# (non-Y) values; including them in axis X's marginal (X != Y) collapses
# X to a point and shrinks Sd(X). Recompute mean/Sd for each column of
# `theta_grid` using only cells that vary that column — cartesian cells
# (`refining_axis == ""`) plus same-axis slice cells.
#
# Joint theta_mean / theta_sd come from `.nl_posterior_moments` and are
# left in place for axes with no foreign slice cells (the recompute is a
# no-op there). The original `theta_mean` / `theta_sd` are overwritten
# in place rather than augmented — downstream callers should read the
# axis marginal, not the cartesian-only joint moment.
.joint_recalibrate_axis_moments <- function(res) {
    refining <- res$refining_axis
    if (is.null(refining) || all(refining == "")) return(res)
    if (is.null(res$theta_grid) || !is.matrix(res$theta_grid)) return(res)
    cols  <- colnames(res$theta_grid)
    lm    <- res$log_marginal
    for (col in cols) {
        # Include `consistency_<col>` cells too: they vary on `col` only
        # (other axes pinned at mode) and ARE part of col's own slice. The
        # only cells we exclude are foreign-axis slices that fix `col` at
        # a single non-varying value.
        keep <- refining == "" | refining == col |
                refining == paste0("consistency_", col)
        if (all(keep)) next
        lm_k <- lm[keep]
        m    <- max(lm_k)
        w    <- exp(lm_k - m)
        w    <- w / sum(w)
        vals <- res$theta_grid[keep, col]
        mu   <- sum(w * vals)
        sd_v <- sqrt(max(0, sum(w * vals^2) - mu^2))
        res$theta_mean[[col]] <- mu
        res$theta_sd[[col]]   <- sd_v
    }
    res
}

# Weighted-quantile median + 2.5/97.5 empirical CI for every hyperparameter
# axis are produced generically by `.nl_axis_quantiles` (defined in
# nested_laplace.R) and attached by `.nl_posterior_moments`. No alpha-
# specific helper is needed: median/CI are the recommended summary for
# right-skewed scale-like axes (alpha = sigma_pos/sigma_occ, sigma,
# range, phi at small n), while mean +/- SD remains available on the
# same axis names via `theta_mean / theta_sd`.

# Alpha refinement piggybacks on the generic consistency pass: alpha
# appears in `.refinable_axes`, and `.nl_var_of_means_consistency_pass`
# fires whenever the joint-grid alpha SD falls short of the Laplace-at-
# mode alpha SD on its own axis. No bespoke helper.


# Compute per-arm latent offsets so callers can decode `modes` back into
# per-arm (beta, re) blocks plus the shared spatial block(s). For BYM2 the
# spatial block is two sub-blocks (phi, theta); for ICAR/CAR_proper it's
# just phi.
.joint_layout <- function(arms, n_spatial_units, n_spatial_blocks,
                          spatial_block_names) {
    n_arms <- length(arms)
    p_arm  <- vapply(arms, function(a) ncol(a$X),     integer(1))
    n_re   <- vapply(arms, function(a) a$n_re_groups, integer(1))

    beta_start <- integer(n_arms)
    cur <- 0L
    for (k in seq_len(n_arms)) {
        beta_start[k] <- cur
        cur <- cur + p_arm[k]
    }
    re_start <- integer(n_arms)
    for (k in seq_len(n_arms)) {
        re_start[k] <- cur
        cur <- cur + n_re[k]
    }
    spatial_starts <- integer(n_spatial_blocks)
    for (b in seq_len(n_spatial_blocks)) {
        spatial_starts[b] <- cur
        cur <- cur + n_spatial_units
    }
    n_x <- cur

    out <- list(
        n_arms     = n_arms,
        p          = p_arm,
        n_re       = n_re,
        beta_start = beta_start,
        re_start   = re_start,
        n_x        = n_x
    )
    for (b in seq_len(n_spatial_blocks)) {
        out[[spatial_block_names[b]]] <- spatial_starts[b]
    }
    out
}
