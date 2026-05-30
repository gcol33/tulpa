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
    a <- .normalise_arm_field_coef(a, k)
    a <- .normalise_arm_cell_coupling(a, k, N)
    a
}

# Parse per-arm `coupled` / `cell_obs_map`. When `coupled` is TRUE, the
# inner Newton routes this arm's per-cell contribution through the
# registered CellCouplingSpec; `cell_obs_map[i]` is the 1-based cell id
# for row i of this arm.
.normalise_arm_cell_coupling <- function(a, k, N) {
    coupled <- isTRUE(a$coupled)
    a$coupled <- coupled
    if (!coupled) {
        a$cell_obs_map <- integer(0)
        return(a)
    }
    m <- a$cell_obs_map
    if (is.null(m)) {
        stop("Arm ", k, ": coupled = TRUE requires `cell_obs_map`.",
             call. = FALSE)
    }
    m <- as.integer(m)
    if (length(m) != N) {
        stop("Arm ", k, ": length(cell_obs_map) (", length(m),
             ") must equal length(y) (", N, ").", call. = FALSE)
    }
    if (any(is.na(m)) || any(m < 1L)) {
        stop("Arm ", k, ": `cell_obs_map` entries must be positive integers.",
             call. = FALSE)
    }
    a$cell_obs_map <- m
    a
}

# Parse the per-arm `field_coef` spec into resolved fields. Each arm carries
# one of: numeric scalar (constant), character (named outer-grid hyperparam
# axis), or list(name = , grid = ) (embedded axis declaration). The default
# is `field_coef = 1` (donor behaviour -- existing non-copy arms).
#
# Resolves to two fields on the arm spec:
#   * `field_coef_const` numeric scalar (default 1) -- the constant per-arm
#     multiplier always applied to that arm's field amplitude. For arms with
#     a hyperparam-driven coefficient this is 1 (the hyperparam carries the
#     coefficient).
#   * `field_coef_axis` NULL or list(name = , grid = ) -- when set, this arm
#     declares a hyperparam-driven field coefficient. The driver maps that
#     declaration to the existing `copy = list(arm, alpha_grid)` plumbing
#     (at most one such axis is supported in the first ship).
.normalise_arm_field_coef <- function(a, k) {
    fc <- a$field_coef
    if (is.null(fc)) {
        a$field_coef_const <- 1.0
        a$field_coef_axis  <- NULL
        return(a)
    }
    if (is.numeric(fc) && length(fc) == 1L) {
        if (!is.finite(fc) || fc < 0) {
            stop("Arm ", k, ": `field_coef` numeric scalar must be a finite ",
                 "non-negative number (got ", fc, ").", call. = FALSE)
        }
        a$field_coef_const <- as.numeric(fc)
        a$field_coef_axis  <- NULL
        return(a)
    }
    if (is.character(fc) && length(fc) == 1L) {
        a$field_coef_const <- 1.0
        a$field_coef_axis  <- list(name = fc, grid = NULL)
        return(a)
    }
    if (is.list(fc)) {
        nm <- fc$name
        if (is.null(nm) || !is.character(nm) || length(nm) != 1L) {
            stop("Arm ", k, ": `field_coef` list must carry a single character ",
                 "`name`.", call. = FALSE)
        }
        gr <- fc$grid
        if (!is.null(gr)) {
            gr <- as.numeric(gr)
            if (length(gr) == 0L || any(!is.finite(gr)) || any(gr < 0)) {
                stop("Arm ", k, ": `field_coef$grid` must be a non-empty ",
                     "numeric vector of finite non-negative values.",
                     call. = FALSE)
            }
        }
        a$field_coef_const <- 1.0
        a$field_coef_axis  <- list(name = as.character(nm), grid = gr)
        return(a)
    }
    stop("Arm ", k, ": `field_coef` must be NULL, a numeric scalar, a single ",
         "character name, or `list(name = , grid = )`.", call. = FALSE)
}

# Apply the back-compat `copy = list(arm, alpha_grid)` shim by rewriting
# `responses[[X]]$field_coef = list(name = "alpha", grid = alpha_grid)`
# before any further processing. Returns the (possibly modified) responses
# list. Errors when both `copy` and an explicit `field_coef` on the same
# arm are present (ambiguous).
#
# Non-destructive on `copy`: the legacy `copy` argument is still echoed onto
# the result for callers that read it back, and the multi-block path's
# `.resolve_copy_multi()` continues to read `copy$arm` / `copy$block` for
# block targeting. The desugaring only ensures the chosen arm carries the
# matching `field_coef` so the kernel boundary reads a single source of truth.
.desugar_copy_to_field_coef <- function(responses, copy) {
    if (is.null(copy)) return(responses)
    arm_id <- copy$arm
    if (is.null(arm_id)) {
        stop("`copy$arm` must be a name or 1-based index.", call. = FALSE)
    }
    nm <- names(responses)
    if (is.character(arm_id)) {
        if (is.null(nm) || !arm_id %in% nm) {
            stop("`copy$arm` = '", arm_id, "' not found in names(responses).",
                 call. = FALSE)
        }
        k <- match(arm_id, nm)
    } else {
        k <- as.integer(arm_id)
        if (k < 1L || k > length(responses)) {
            stop("`copy$arm` index out of range.", call. = FALSE)
        }
    }
    if (!is.null(responses[[k]]$field_coef)) {
        stop("Arm ", k, ": both `copy` and `field_coef` are set. `copy` is a ",
             "back-compat shim for `field_coef = list(name = \"alpha\", grid = )`; ",
             "use only one.", call. = FALSE)
    }
    alpha_grid <- if (!is.null(copy$alpha_grid)) {
        as.numeric(copy$alpha_grid)
    } else {
        c(0, exp(seq(log(0.1), log(3), length.out = 5)))
    }
    responses[[k]]$field_coef <- list(name = "alpha", grid = alpha_grid)
    responses
}

# Resolve the per-arm field_coef structure on a normalised arms list into
# the kernel-facing pieces used by the joint dispatcher:
#   * has_axis        logical -- is there any hyperparam axis declared?
#   * axis_arm_zero   0-based arm index of the axis-carrying arm, -1 if none
#   * axis_name       character (default "alpha")
#   * axis_grid       numeric grid (NULL if the axis was named-only and
#                     the user must supply it via another mechanism)
#   * field_coef_const numeric [n_arms] -- per-arm constant multiplier
#   * any_nontrivial logical -- any arm has field_coef_const != 1?
#
# First-ship validation: at most ONE arm may declare a hyperparam axis
# (multi-arm shared axes are deferred to v5; the cover hurdle / occu_cover
# only need one).
.resolve_arm_field_coefs <- function(arms) {
    n_arms <- length(arms)
    consts <- vapply(arms, function(a) a$field_coef_const %||% 1.0,
                     numeric(1))
    axes <- lapply(arms, function(a) a$field_coef_axis)
    has_axis_per_arm <- !vapply(axes, is.null, logical(1))
    if (sum(has_axis_per_arm) > 1L) {
        stop("At most one arm may declare a hyperparam `field_coef` axis ",
             "(first ship). Got ", sum(has_axis_per_arm),
             ". Multi-arm shared axes are deferred.", call. = FALSE)
    }
    if (any(has_axis_per_arm)) {
        k_axis  <- which(has_axis_per_arm)
        spec    <- axes[[k_axis]]
        list(
            has_axis        = TRUE,
            axis_arm_zero   = as.integer(k_axis - 1L),
            axis_name       = spec$name %||% "alpha",
            axis_grid       = spec$grid,
            field_coef_const = consts,
            any_nontrivial  = any(abs(consts - 1.0) > 0)
        )
    } else {
        list(
            has_axis        = FALSE,
            axis_arm_zero   = -1L,
            axis_name       = NA_character_,
            axis_grid       = NULL,
            field_coef_const = consts,
            any_nontrivial  = any(abs(consts - 1.0) > 0)
        )
    }
}

# Decide which arm (if any) is the copy arm and what the alpha grid is.
# Single-block path; `copy$alpha_grid` is the outer-grid axis on the copy
# coefficient.
#
# Source of truth: `arms` already carries the resolved per-arm `field_coef_axis`
# / `field_coef_const` after `.normalise_joint_arm` + `.desugar_copy_to_field_coef`.
# When any arm declared a hyperparam axis, the resolved axis arm + grid are
# returned as the kernel-facing copy spec. The legacy `copy` argument is no
# longer consulted here -- the entry-point shim rewrote it onto the responses
# list before this is reached.
#
# `field_coef_const` carries the per-arm constant multipliers (default 1 for
# every arm). Arms with `field_coef = 0` get const 0; arms with `field_coef =
# c` get c. These multipliers ride alongside any hyperparam axis on the same
# arm (axis multiplied by const), so they are returned to the caller and
# threaded down to the kernel via `arms[k].field_coef` (see the C++ side).
.resolve_copy <- function(copy, responses, prior, type) {
    n_arms <- length(responses)
    consts <- vapply(responses, function(a) {
        if (is.null(a$field_coef_const)) 1.0 else as.numeric(a$field_coef_const)
    }, numeric(1))
    axes <- lapply(responses, function(a) a$field_coef_axis)
    has_axis_per_arm <- !vapply(axes, is.null, logical(1))
    if (sum(has_axis_per_arm) > 1L) {
        stop("At most one arm may declare a hyperparam `field_coef` axis ",
             "(first ship). Got ", sum(has_axis_per_arm), ".", call. = FALSE)
    }
    if (!any(has_axis_per_arm)) {
        return(list(has_copy = FALSE, copy_arm_zero = -1L,
                    alpha_grid = numeric(0),
                    field_coef_const = consts))
    }
    k_axis <- which(has_axis_per_arm)
    spec   <- axes[[k_axis]]
    alpha_axis <- spec$grid
    if (is.null(alpha_axis)) {
        alpha_axis <- c(0, exp(seq(log(0.1), log(3), length.out = 5)))
    }
    if (length(alpha_axis) == 0L) {
        stop("Arm ", k_axis, ": `field_coef$grid` must have at least one ",
             "non-negative value.", call. = FALSE)
    }
    if (any(alpha_axis < 0)) {
        stop("Arm ", k_axis, ": `field_coef$grid` values must be non-negative.",
             call. = FALSE)
    }
    list(has_copy = TRUE, copy_arm_zero = as.integer(k_axis - 1L),
         alpha_grid = as.numeric(alpha_axis),
         field_coef_const = consts)
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
    if (is.null(res$refining_axis) || all(res$refining_axis == "")) return(res)
    if (is.null(res$theta_grid) || !is.matrix(res$theta_grid)) return(res)
    moments <- .hyper_recalibrate_axis_moments(
        theta_grid    = res$theta_grid,
        log_marginal  = res$log_marginal,
        refining_axis = res$refining_axis,
        theta_mean    = res$theta_mean,
        theta_sd      = res$theta_sd
    )
    res$theta_mean <- moments$theta_mean
    res$theta_sd   <- moments$theta_sd
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
# appears in `.hyper_refinable_names`, and `.hyper_consistency_pass`
# fires whenever the joint-grid alpha SD falls short of the Laplace-at-
# mode alpha SD on its own axis. No bespoke helper.


# Generic axis-spec adapter (gcol33/tulpa#33 Step 3).
#
# Builds the list of `hyper_axis_spec` objects the generic refinement / consistency
# helpers (`R/hyper_grid_refine.R`) consume from the joint driver's paired-vector
# `grids` + copy state. The spec's `grid` field is informational (refinement reads
# the per-cell axis values from `theta_grid[, axis]` directly), but populating it
# with the sorted unique levels keeps the spec object self-describing.
#
# Hardcoded metadata captures what the legacy joint helpers `.axis_is_log_scale`,
# `.axis_bounds`, `.refinable_axes`, `.axis_refinement_order` encoded by name --
# the same set, in one place.
.joint_axis_specs <- function(grids, cp) {
    has_alpha <- cp$has_copy && length(grids$alpha) > 0L
    axes <- names(grids)
    if (!has_alpha) axes <- setdiff(axes, "alpha")
    lapply(axes, function(a) {
        log_scale <- a %in% c("sigma", "alpha", "tau", "sigma2",
                                "phi_gp", "lengthscale") ||
                     startsWith(a, "phi_")
        bounds <- if (a == "sigma")            c(0, Inf)
                  else if (a == "alpha")        c(0, Inf)
                  else if (a == "rho")          c(0, 1)
                  else if (a == "rho_car")      c(-Inf, 1)
                  else if (startsWith(a, "phi_")) c(0, Inf)
                  else NULL
        refinable <- a == "alpha" || startsWith(a, "phi_")
        refine_priority <- if (a == "alpha") 1L
                           else if (startsWith(a, "phi_")) 2L
                           else 100L
        spec <- hyper_axis_spec(
            name      = a,
            grid      = sort(unique(as.numeric(grids[[a]]))),
            log_scale = log_scale,
            bounds    = bounds,
            refinable = refinable
        )
        spec$refine_priority <- refine_priority
        spec
    })
}

# Convert a generic `new_cells` matrix [n_new x n_axes] back to the joint
# kernel's paired-vector `grids` representation. When `cp$has_copy = FALSE`
# the alpha entry stays `numeric(0)` (the no-copy contract the backend expects).
.joint_grids_from_cells <- function(new_cells, cp) {
    axes <- colnames(new_cells)
    out <- stats::setNames(lapply(axes, function(a)
        as.numeric(new_cells[, a])), axes)
    if (!cp$has_copy) out$alpha <- numeric(0)
    out
}

# Build the generic `kernel_fn(new_cells, warm_start, store_extras)` closure
# refinement passes around `backend$call_kernel`. Packs the joint kernel's
# per-cell modes + n_iter + Q_csc_* into a list of per-cell `extras` so the
# generic helpers can carry them along across refinement appends (and the
# warm-start chain still reads `extras[[idx0]]$mode`).
.joint_make_kernel_fn <- function(arms, prior, cp, backend, max_iter, tol,
                                  n_threads, x_init_default, store_Q,
                                  arm_names, cell_coupling = "separable",
                                  hessian_pd_mode = 0L) {
    function(new_cells, warm_start = NULL, store_extras = FALSE) {
        new_grids <- .joint_grids_from_cells(new_cells, cp)
        slice_x_init <- if (!is.null(warm_start) && !is.null(warm_start$mode))
                        as.numeric(warm_start$mode) else x_init_default
        res_x <- backend$call_kernel(arms, prior, cp, new_grids,
                                      max_iter, tol, n_threads,
                                      slice_x_init, isTRUE(store_Q),
                                      arm_names = arm_names,
                                      cell_coupling = cell_coupling,
                                      hessian_pd_mode = hessian_pd_mode)
        extras <- NULL
        if (isTRUE(store_extras)) {
            n <- nrow(new_cells)
            extras <- vector("list", n)
            modes_mat  <- res_x$modes
            n_iter_vec <- res_x$n_iter
            Qp <- res_x$Q_csc_p_per_grid
            Qi <- res_x$Q_csc_i_per_grid
            Qx <- res_x$Q_csc_x_per_grid
            for (k in seq_len(n)) {
                e <- list()
                if (!is.null(modes_mat))  e$mode   <- as.numeric(modes_mat[k, ])
                if (!is.null(n_iter_vec)) e$n_iter <- as.integer(n_iter_vec[k])
                if (!is.null(Qp))         e$Q_csc_p <- Qp[[k]]
                if (!is.null(Qi))         e$Q_csc_i <- Qi[[k]]
                if (!is.null(Qx))         e$Q_csc_x <- Qx[[k]]
                extras[[k]] <- e
            }
        }
        list(log_marginal = res_x$log_marginal, extras = extras)
    }
}

# Build the initial per-cell extras list from the initial joint kernel result,
# matching what `.joint_make_kernel_fn` would have produced for the cartesian
# pass. Refinement extends this list; `.joint_glue_extras_to_res` puts the
# refined extras back into `res` once integration is done.
.joint_init_extras_from_res <- function(res) {
    n <- length(res$log_marginal)
    if (n == 0L) return(vector("list", 0L))
    modes_mat  <- res$modes
    n_iter_vec <- res$n_iter
    Qp <- res$Q_csc_p_per_grid
    Qi <- res$Q_csc_i_per_grid
    Qx <- res$Q_csc_x_per_grid
    lapply(seq_len(n), function(k) {
        e <- list()
        if (!is.null(modes_mat))  e$mode   <- as.numeric(modes_mat[k, ])
        if (!is.null(n_iter_vec)) e$n_iter <- as.integer(n_iter_vec[k])
        if (!is.null(Qp))         e$Q_csc_p <- Qp[[k]]
        if (!is.null(Qi))         e$Q_csc_i <- Qi[[k]]
        if (!is.null(Qx))         e$Q_csc_x <- Qx[[k]]
        e
    })
}

# Glue refined extras + log_marginal + refining_axis back into the joint
# kernel result. Downstream `.joint_recalibrate_axis_moments` /
# `.nl_posterior_moments` / `.nl_refit_axis_sd_laplace` read `res$modes`,
# `res$log_marginal`, `res$refining_axis` directly; this keeps them in sync
# after refinement without touching their implementations.
.joint_glue_extras_to_res <- function(res, theta_grid_matrix, log_marginal,
                                      extras, refining_axis) {
    res$log_marginal  <- log_marginal
    res$n_grid        <- nrow(theta_grid_matrix)
    res$refining_axis <- refining_axis
    if (is.null(extras) || length(extras) == 0L) return(res)
    if (!is.null(extras[[1L]]$mode)) {
        n_x <- length(extras[[1L]]$mode)
        res$modes <- do.call(rbind, lapply(extras, function(e) {
            if (is.null(e$mode)) rep(NA_real_, n_x) else as.numeric(e$mode)
        }))
    }
    if (!is.null(extras[[1L]]$n_iter)) {
        res$n_iter <- vapply(extras, function(e) {
            if (is.null(e$n_iter)) NA_integer_ else as.integer(e$n_iter)
        }, integer(1))
    }
    if (!is.null(extras[[1L]]$Q_csc_p)) {
        res$Q_csc_p_per_grid <- lapply(extras, `[[`, "Q_csc_p")
        res$Q_csc_i_per_grid <- lapply(extras, `[[`, "Q_csc_i")
        res$Q_csc_x_per_grid <- lapply(extras, `[[`, "Q_csc_x")
    }
    res
}

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
