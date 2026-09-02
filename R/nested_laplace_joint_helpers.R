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
# `n_arms`: entry k is either `NULL` (no phi axis for that arm -- kernel
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

# Compute a tile partition for the outer-grid loop's three-tier warm-start.
# A tile groups all
# outer-grid cells that share every hyperparameter coordinate except the
# copy coefficient alpha -- for the joint copy block under the (sigma, alpha)
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
            # Global pilot doubles as this tile's pilot -- no Tier-2 solve.
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

# Validate one arm spec and fill in defaults. `spatial_idx` selects the
# policy for the arm-level spatial index: "required" (single-block path:
# a mandatory field) or "optional" (multi-block path: per-arm idx vectors
# live inside each block spec, so a missing spatial_idx gets a length-N
# placeholder of zeros -- it satisfies parse_joint_arms' length check
# without contributing to eta).
.normalise_joint_arm_core <- function(a, k,
                                      spatial_idx = c("required", "optional")) {
    spatial_idx <- match.arg(spatial_idx)
    if (!is.list(a)) {
        stop("Arm ", k, ": expected a list of arm spec fields.", call. = FALSE)
    }
    must_have <- if (spatial_idx == "required") c("y", "X", "spatial_idx", "family")
                 else c("y", "X", "family")
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
    if (spatial_idx == "optional" && is.null(a$spatial_idx)) {
        a$spatial_idx <- rep(0L, N)
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
    # Optional grouped beta sufficient statistics: when
    # present, n_trials is the per-row group count and (slog_y, slog_1my) the
    # within-group sum of log(y) / log(1-y); the built-in beta spec reads them.
    if (!is.null(a$slog_y)) {
        a$slog_y   <- as.numeric(a$slog_y)
        a$slog_1my <- as.numeric(a$slog_1my)
        if (length(a$slog_y) != N || length(a$slog_1my) != N) {
            stop("Arm ", k, ": length(slog_y)/length(slog_1my) must equal ",
                 "length(y) (", N, ").", call. = FALSE)
        }
    }
    # Optional interval-censored Gaussian bounds (family == "interval_gaussian"):
    # row i records the latent value fell in (lower[i], upper[i]] on the predictor
    # scale, with -Inf / +Inf the open outer classes. The built-in interval spec
    # reads (lower, upper) in place of the point response y.
    if (!is.null(a$lower) || !is.null(a$upper)) {
        if (is.null(a$lower) || is.null(a$upper)) {
            stop("Arm ", k, ": `lower` and `upper` must be supplied together.",
                 call. = FALSE)
        }
        a$lower <- as.numeric(a$lower)
        a$upper <- as.numeric(a$upper)
        if (length(a$lower) != N || length(a$upper) != N) {
            stop("Arm ", k, ": length(lower)/length(upper) must equal ",
                 "length(y) (", N, ").", call. = FALSE)
        }
        if (any(is.na(a$lower)) || any(is.na(a$upper)) || any(a$lower >= a$upper)) {
            stop("Arm ", k, ": each `lower` must be finite-or-(-Inf), non-NA, ",
                 "and strictly below its `upper`.", call. = FALSE)
        }
    }
    # Optional upper-truncated Gaussian ceiling (family == "truncated_gaussian"):
    # row i's latent log-response is truncated to <= trunc_upper[i] on the predictor
    # scale; +Inf => no truncation on that row. The built-in truncated spec reads
    # (y, trunc_upper).
    if (!is.null(a$trunc_upper)) {
        a$trunc_upper <- as.numeric(a$trunc_upper)
        if (length(a$trunc_upper) != N) {
            stop("Arm ", k, ": length(trunc_upper) (", length(a$trunc_upper),
                 ") must equal length(y) (", N, ").", call. = FALSE)
        }
        if (any(is.na(a$trunc_upper))) {
            stop("Arm ", k, ": `trunc_upper` must be non-NA (use +Inf for an ",
                 "untruncated row).", call. = FALSE)
        }
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

# Validate one arm spec and fill in defaults (single-block path:
# arm-level `spatial_idx` is required).
.normalise_joint_arm <- function(a, k) {
    .normalise_joint_arm_core(a, k, spatial_idx = "required")
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
#   * `field_coef_axis` NULL or list(name = , grid = , n = ) -- when set, this
#     arm declares a hyperparam-driven field coefficient. The driver maps that
#     declaration to the existing `copy = list(arm, alpha_grid)` plumbing
#     (at most one such axis is supported). `grid` states the axis's nodes; `n`
#     re-reads the engine's own axis at a higher resolution instead
#     (gcol33/tulpa#633), keeping its atom at 0 and its slab bounds.
#     `grid_auto` records whether a stated `grid` carried the `auto_grid()`
#     marker, which is the provenance the refinement rule reads.
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
        a$field_coef_axis  <- list(name = fc, grid = NULL, alpha_n = NULL,
                                   grid_auto = FALSE)
        return(a)
    }
    if (is.list(fc)) {
        nm <- fc$name
        if (is.null(nm) || !is.character(nm) || length(nm) != 1L) {
            stop("Arm ", k, ": `field_coef` list must carry a single character ",
                 "`name`.", call. = FALSE)
        }
        gr <- fc$grid
        # `as.numeric()` drops attributes, so the `auto_grid()` marker -- a
        # wrapper package declaring these nodes a default of its own rather than
        # a user's choice -- is read off the value before it is coerced and
        # carried on the resolved axis, where the refinement provenance rule
        # reads it.
        gr_auto <- is_auto_grid(gr)
        if (!is.null(gr)) {
            gr <- as.numeric(gr)
            if (length(gr) == 0L || any(!is.finite(gr)) || any(gr < 0)) {
                stop("Arm ", k, ": `field_coef$grid` must be a non-empty ",
                     "numeric vector of finite non-negative values.",
                     call. = FALSE)
            }
        }
        # `[[` and not `$`: `$` PARTIAL-matches on a list, so `fc$n` resolves
        # to `fc$name` on every spec that names its coefficient and feeds a
        # character into the integer check below.
        an <- fc[["n"]]
        if (!is.null(an)) {
            an <- suppressWarnings(as.integer(an))
            if (length(an) != 1L || is.na(an) || an < 1L) {
                stop("Arm ", k, ": `field_coef$n` must be a single integer ",
                     ">= 1.", call. = FALSE)
            }
        }
        a$field_coef_const <- 1.0
        a$field_coef_axis  <- list(name = as.character(nm), grid = gr,
                                   alpha_n = an, grid_auto = gr_auto)
        return(a)
    }
    stop("Arm ", k, ": `field_coef` must be NULL, a numeric scalar, a single ",
         "character name, or `list(name = , grid = )`.", call. = FALSE)
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
        stop("At most one arm may declare a hyperparam `field_coef` axis. Got ",
             sum(has_axis_per_arm),
             ". Multi-arm shared axes are not supported.", call. = FALSE)
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
# Single-block path: the copy coefficient's outer-grid axis lives on the arm
# that declares it.
#
# Source of truth: `arms` already carries the resolved per-arm `field_coef_axis`
# / `field_coef_const` from `.normalise_joint_arm`. When any arm declared a
# hyperparam axis, the resolved axis arm + grid are returned as the kernel-facing
# copy spec.
#
# `field_coef_const` carries the per-arm constant multipliers (default 1 for
# every arm). Arms with `field_coef = 0` get const 0; arms with `field_coef =
# c` get c. These multipliers ride alongside any hyperparam axis on the same
# arm (axis multiplied by const), so they are returned to the caller and
# threaded down to the kernel via `arms[k].field_coef` (see the C++ side).
.resolve_copy <- function(responses, prior, type) {
    n_arms <- length(responses)
    consts <- vapply(responses, function(a) {
        if (is.null(a$field_coef_const)) 1.0 else as.numeric(a$field_coef_const)
    }, numeric(1))
    axes <- lapply(responses, function(a) a$field_coef_axis)
    has_axis_per_arm <- !vapply(axes, is.null, logical(1))
    if (sum(has_axis_per_arm) > 1L) {
        stop("At most one arm may declare a hyperparam `field_coef` axis. Got ",
             sum(has_axis_per_arm), ".", call. = FALSE)
    }
    if (!any(has_axis_per_arm)) {
        return(list(has_copy = FALSE, copy_arm_zero = -1L,
                    alpha_grid = numeric(0),
                    field_coef_const = consts))
    }
    k_axis <- which(has_axis_per_arm)
    spec   <- axes[[k_axis]]
    alpha_axis <- .nl_copy_alpha_axis(spec[["grid"]], spec[["alpha_n"]],
                                      what = paste0("Arm ", k_axis))
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
# `theta_grid` using only cells that vary that column -- cartesian cells
# (`refining_axis == ""`) plus same-axis slice cells.
#
# Joint theta_mean comes from `.nl_posterior_moments` and is left in place for
# axes with no foreign slice cells (the recompute is a no-op there). It is
# overwritten in place rather than augmented -- downstream callers should read
# the axis marginal, not the cartesian-only joint moment. The SD over the same
# mask is `.nl_attach_axis_sd()`'s, inside `.nl_posterior_moments()`.
.joint_recalibrate_axis_mean <- function(res) {
    if (is.null(res$refining_axis) || all(res$refining_axis == "")) return(res)
    if (is.null(res$theta_grid) || !is.matrix(res$theta_grid)) return(res)
    lm_eff <- res$log_marginal
    if (!is.null(res$log_quad) && length(res$log_quad) == length(lm_eff)) {
        lm_eff <- lm_eff + res$log_quad
        lm_eff[is.na(lm_eff)] <- -Inf
    }
    res$theta_mean <- .hyper_recalibrate_axis_mean(
        theta_grid    = res$theta_grid,
        log_marginal  = lm_eff,
        refining_axis = res$refining_axis,
        theta_mean    = res$theta_mean
    )
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
# fires whenever the alpha marginal spreads over too few nodes to carry a
# spread. No bespoke helper.


# --- which axes the joint grid carries, and how far each may be refined -------
#
# The axes one `grids` list produces specs for. The alpha column is present in
# the paired-vector representation whether or not a copy arm was declared, so a
# fit with no copy drops it.
.joint_spec_axis_names <- function(grids, cp) {
    axes <- names(grids)
    has_alpha <- isTRUE(cp$has_copy) && length(grids$alpha) > 0L
    if (!has_alpha) axes <- setdiff(axes, "alpha")
    axes
}

# Which axes the joint refinement machinery can place nodes on at all. This is
# a property of the DRIVER -- the copy coefficient and the per-arm dispersions
# are the axes its passes know how to propose on -- not of where the nodes came
# from, which is the separate question `.joint_axis_is_stated()` answers.
.joint_axis_refine_eligible <- function(axis) {
    identical(axis, "alpha") || startsWith(axis, "phi_")
}

# One entry of the user-facing `phi_grid` argument, by arm name. The argument is
# either named by arm or positional over `arm_names`, and the axis it produces
# is `phi_<arm name>` either way.
.joint_phi_grid_entry <- function(phi_grid, arm, arm_names = NULL) {
    if (!is.list(phi_grid) || !length(phi_grid)) return(NULL)
    if (!is.null(names(phi_grid))) {
        if (!arm %in% names(phi_grid)) return(NULL)
        return(phi_grid[[arm]])
    }
    k <- match(arm, arm_names %||% character(0))
    if (is.na(k) || k > length(phi_grid)) return(NULL)
    phi_grid[[k]]
}

# Did the CALLER write this axis's nodes down?
#
# A stated axis is a statement about where the fit integrates, so refinement
# resolves it more finely and does not leave it; an axis the engine placed
# carries no such statement. Three things count as engine-placed, matching the
# provenance vocabulary the recentring pass already uses
# (`.nl_axis_is_pinned()`): no nodes given at all (the copy axis resolved from
# `alpha_n`, or from the engine default), nodes marked with `auto_grid()` by a
# wrapper package that computed a default of its own, and nodes that ARE the
# engine's own default axis, which carry nothing a statement would add.
.joint_axis_is_stated <- function(axis, arms, phi_grid = NULL,
                                  arm_names = NULL) {
    if (identical(axis, "alpha")) {
        for (a in arms) {
            fc <- a$field_coef_axis
            if (is.null(fc)) next
            g <- fc$grid
            if (is.null(g) || !length(g)) return(FALSE)
            if (isTRUE(fc$grid_auto)) return(FALSE)
            return(!.nl_axis_matches_default(g, "alpha_grid", ".copy"))
        }
        return(FALSE)
    }
    if (startsWith(axis, "phi_")) {
        v <- .joint_phi_grid_entry(phi_grid, sub("^phi_", "", axis), arm_names)
        if (is.null(v) || length(v) < 2L) return(FALSE)
        return(!is_auto_grid(v))
    }
    FALSE
}

# The refinement mode of every axis of one joint grid, as a named character
# vector over `.NL_AXIS_REFINE_MODES`. Eligible axes take the provenance
# default; everything else takes "none". `user` -- the caller's
# `control$axis_refine`, already validated -- overrides per axis.
.joint_axis_refine_modes <- function(grids, cp, arms, phi_grid = NULL,
                                     arm_names = NULL, user = NULL) {
    axes <- .joint_spec_axis_names(grids, cp)
    out  <- stats::setNames(rep("none", length(axes)), axes)
    for (a in axes) {
        if (!.joint_axis_refine_eligible(a)) next
        out[[a]] <- if (.joint_axis_is_stated(a, arms, phi_grid, arm_names))
            .nl_axis_refine("stated") else .nl_axis_refine("placed")
    }
    if (!is.null(user)) {
        for (a in intersect(names(user), axes)) out[[a]] <- user[[a]]
    }
    out
}

# Validate `control$axis_refine` against the axes this fit actually has.
#
# Accepts a named character vector (or list) keyed by axis name, or one unnamed
# value applying to every eligible axis. An unknown axis name is an error rather
# than a silent no-op, and so is asking for nodes on an axis the driver's passes
# cannot place any on: a knob that quietly does nothing is how a caller comes to
# believe a range was honoured when it was not.
.joint_check_axis_refine <- function(x, axis_names) {
    if (is.null(x)) return(NULL)
    if (is.list(x)) x <- unlist(x, use.names = TRUE)
    if (!is.character(x) || !length(x) || anyNA(x)) {
        stop("`control$axis_refine` must be a character vector of ",
             paste(sprintf('"%s"', .NL_AXIS_REFINE_MODES), collapse = " / "),
             ", named by outer-grid axis.", call. = FALSE)
    }
    bad <- setdiff(unique(unname(x)), .NL_AXIS_REFINE_MODES)
    if (length(bad)) {
        stop("`control$axis_refine`: unknown mode ",
             paste(shQuote(bad), collapse = ", "), ". Use one of ",
             paste(sprintf('"%s"', .NL_AXIS_REFINE_MODES), collapse = ", "),
             ".", call. = FALSE)
    }
    eligible <- axis_names[vapply(axis_names, .joint_axis_refine_eligible,
                                  logical(1))]
    nm <- names(x)
    if (is.null(nm) || !all(nzchar(nm))) {
        if (length(x) != 1L) {
            stop("`control$axis_refine` must be named by axis, or a single ",
                 "mode applying to every refinable axis.", call. = FALSE)
        }
        if (!length(eligible)) return(NULL)
        return(stats::setNames(rep(unname(x), length(eligible)), eligible))
    }
    unknown <- setdiff(nm, axis_names)
    if (length(unknown)) {
        stop("`control$axis_refine` names axes this fit does not have: ",
             paste(shQuote(unknown), collapse = ", "), ". Its axes are ",
             paste(shQuote(axis_names), collapse = ", "), ".", call. = FALSE)
    }
    wrong <- nm[!nm %in% eligible & x != "none"]
    if (length(wrong)) {
        stop("`control$axis_refine`: ",
             paste(shQuote(wrong), collapse = ", "),
             " cannot take refinement nodes on this driver, so only \"none\" ",
             "applies there. The axes it refines are ",
             if (length(eligible)) paste(shQuote(eligible), collapse = ", ")
             else "none of this fit's", ".", call. = FALSE)
    }
    x
}

# The mode one axis ends up at. Absent provenance -- a spec list rebuilt from an
# assembled grid, which no refinement pass reads -- keeps the driver's
# eligibility as the answer.
.joint_axis_refine_mode <- function(axis_refine, axis) {
    if (!is.null(axis_refine) && axis %in% names(axis_refine)) {
        return(as.character(axis_refine[[axis]]))
    }
    if (.joint_axis_refine_eligible(axis)) .nl_axis_refine("placed") else "none"
}

# Generic axis-spec adapter (Step 3).
#
# Builds the list of `hyper_axis_spec` objects the generic refinement / consistency
# helpers (`R/hyper_grid_refine.R`) consume from the joint driver's paired-vector
# `grids` + copy state. The spec's `grid` field is informational (refinement reads
# the per-cell axis values from `theta_grid[, axis]` directly), but populating it
# with the sorted unique levels keeps the spec object self-describing.
#
# Hardcoded metadata captures what the legacy joint helpers `.axis_is_log_scale`,
# `.axis_bounds`, `.axis_refinement_order` encoded by name -- the same set, in
# one place. `axis_refine` carries the one piece of metadata that is NOT a
# property of the axis's name: how far each axis may be refined, which follows
# from where its nodes came from and so is resolved by
# `.joint_axis_refine_modes()` at the front door.
# `user_priors` carries the caller's regularizing hyperpriors, named by the axis
# each applies to (`sigma`, `alpha`, `phi` -- the last matching every `phi_<arm>`
# axis), matching how `.joint_hp_vec_for_grids()` folds them into
# `log_marginal`. An axis with one keeps the flat-over-declared-span measure,
# because applying the engine's declared density as well would put two priors on
# one axis, and carries the density on the spec so the axis quadrature knows
# what the fold will apply -- which is what lets a declared point mass keep its
# prior probability (`.hyper_atom_fold_scale()`).
.joint_axis_specs <- function(grids, cp, user_priors = NULL,
                              copy_atom_mass = .TULPA_COPY_ATOM_MASS,
                              copy_slab = "exponential",
                              axis_refine = NULL) {
    copy_slab <- .hyper_check_copy_slab(copy_slab)
    axes <- .joint_spec_axis_names(grids, cp)
    lapply(axes, function(a) {
        # A multi-block grid prefixes its axes (`b1.sigma`), so the scale, the
        # domain and the point-mass metadata are resolved on the bare axis name:
        # an axis's support does not change because the block it is laid on is
        # the second one. Which axes refine, and in what order, stays keyed on
        # the full name.
        bare <- sub("^b[0-9]+[.]", "", a)
        scale_known <- .hyper_axis_scale(bare)
        log_scale <- isTRUE(scale_known)
        bounds <- .hyper_spec_bounds(bare)
        # How far refinement may move this axis: whether it participates at all,
        # and whether it may place a node past the nodes it was given. Both come
        # from `axis_refine`, resolved from the caller's provenance one level up
        # -- deciding by axis NAME would opt a caller's stated nodes into being
        # extended (gcol33/tulpa#658).
        mode <- .joint_axis_refine_mode(axis_refine, a)
        refine_priority <- if (a == "alpha") 1L
                           else if (startsWith(a, "phi_")) 2L
                           else 100L
        spec <- hyper_axis_spec(
            name      = a,
            grid      = sort(unique(as.numeric(grids[[a]]))),
            log_scale = log_scale,
            bounds    = bounds,
            refinable = !identical(mode, "none"),
            extend    = identical(mode, "extend")
        )
        spec$refine_priority <- refine_priority
        # An axis this table does not classify keeps equal node weights, which
        # is what the engine integrated before any coordinate was declared.
        if (is.na(scale_known)) spec$unweighted <- TRUE
        # The copy scale carries an explicit zero level ("no coupling"), which
        # is a point mass rather than part of the log continuum, so it needs a
        # declared prior probability. Fixing it here keeps it independent of how
        # many continuum nodes the grid ends up with.
        if (identical(bare, "alpha")) spec$atom_mass <- copy_atom_mass
        # A flat measure on a log axis is improper, so the support is a prior
        # choice. The incoming grid is what the user declared, so the support is
        # its span widened by half a node step at each end: that is the region
        # the initial nodes already tile, so an unrefined grid integrates what it
        # always did, and fixing it here, ahead of any refinement, is what keeps
        # the measure independent of the data.
        pos <- sort(unique(spec$grid[spec$grid > 0]))
        user_fn <- if (identical(a, "sigma")) user_priors$sigma
                   else if (identical(a, "alpha")) user_priors$alpha
                   else if (startsWith(a, "phi_")) user_priors$phi
                   else NULL
        user_prior <- !is.null(user_fn)
        if (user_prior) {
            # The joint driver's hyperprior families are densities on the axis's
            # natural scale.
            spec$log_prior <- user_fn
            spec$log_prior_coord <- "natural"
        }
        if (log_scale && length(pos) >= 2L) {
            copy_exp_slab <- identical(bare, "alpha") && !user_prior &&
                             identical(copy_slab, "exponential")
            if (copy_exp_slab) {
                # The copy scale is the axis a user cannot be expected to bracket
                # in advance, and the one carrying the point mass the continuum
                # is weighed against. It therefore gets a proper density over the
                # whole positive line rather than a span: refinement can then
                # follow the posterior out as far as it needs to, and still be
                # integrating the measure declared here, before the fit.
                #
                # `copy_slab = "flat"` takes the other branch instead, so the
                # copy scale is flat in log alpha over the span its declared
                # nodes tile, the same measure sigma and phi carry.
                spec$slab_log_density <- .hyper_copy_slab_density(max(pos))
            } else {
                bd <- .hyper_default_coord_bounds(.hyper_axis_coord(pos, spec))
                spec$slab_bounds <- exp(bd)
            }
        }
        spec
    })
}

# Axis specs for a grid that is already assembled, keyed off its column names.
# The multi-block driver builds its grid before any spec list exists, so it
# recovers the same metadata from the columns rather than carrying a second
# description of the same axes.
.joint_axis_specs_from_grid <- function(theta_grid,
                                        copy_slab = "exponential") {
    if (is.null(theta_grid) || is.null(colnames(theta_grid))) return(NULL)
    theta_grid <- as.matrix(theta_grid)
    grids <- stats::setNames(
        lapply(colnames(theta_grid),
               function(a) sort(unique(as.numeric(theta_grid[, a])))),
        colnames(theta_grid))
    # A column holding one value across every cell is a fixed setting rather
    # than an axis of the grid: it contributes the same factor to every cell,
    # which cancels when the weights are normalised. Some are not quantities to
    # integrate at all, such as the `r = Inf` node a Poisson count grid carries
    # to pin the negative-binomial size, so they are dropped before an axis spec
    # is built for them.
    grids <- grids[vapply(grids, function(g) length(g) > 1L, logical(1))]
    if (length(grids) == 0L) return(NULL)
    .joint_axis_specs(grids, list(has_copy = TRUE), copy_slab = copy_slab)
}

# The continuum levels of one axis: its distinct finite values, with the zero
# level dropped on a log-scale axis, where it is the point mass rather than a
# point of the continuum. The same convention `.hyper_axis_support()` applies
# before it measures a span.
.joint_axis_continuum_levels <- function(v, spec) {
    v <- sort(unique(as.numeric(v)))
    v <- v[is.finite(v)]
    if (isTRUE(spec$log_scale)) v <- v[v > 0]
    v
}

# The span each outer axis was integrated over, and the two terms that separate
# it from the nodes the caller wrote down.
#
# `axis_support` reports the interval the quadrature finally reached, which is
# the number a reader needs and not one they can decompose: it carries both the
# half node step the outermost cells own -- `k / (k - 1)` times the node range on
# the axis's integration coordinate, for `k` equally spaced nodes, so 2x at two
# nodes and 1.125x at nine -- and whatever refinement added past the end nodes.
# Reported side by side the two terms are readable off the fit:
#
#   nodes       range of the axis's continuum nodes, before any refinement pass
#   declared    support of that initial node set (the nodes plus the half step)
#   integrated  support after refinement; identical to `axis_support`
#   refine      the mode that governed how far refinement could move the axis
#   n_nodes     continuum node count, initial and final
#
# An axis with fewer than two continuum levels has no span to report and is left
# out, matching what `.hyper_grid_supports()` does with it.
.joint_axis_span <- function(theta_grid_init, theta_grid_final, specs) {
    if (is.null(theta_grid_init) || is.null(specs)) return(NULL)
    theta_grid_init  <- as.matrix(theta_grid_init)
    theta_grid_final <- if (is.null(theta_grid_final)) theta_grid_init
                        else as.matrix(theta_grid_final)
    out <- list()
    for (spec in specs) {
        a <- spec$name
        if (!a %in% colnames(theta_grid_init)) next
        lev0 <- .joint_axis_continuum_levels(theta_grid_init[, a], spec)
        if (length(lev0) < 2L) next
        lev1 <- if (a %in% colnames(theta_grid_final))
            .joint_axis_continuum_levels(theta_grid_final[, a], spec) else lev0
        mode <- if (!isTRUE(spec$refinable)) "none"
                else if (.hyper_axis_may_extend(spec)) "extend" else "densify"
        out[[a]] <- list(
            nodes      = range(lev0),
            declared   = .hyper_axis_support(theta_grid_init[, a], spec),
            integrated = .hyper_axis_support(theta_grid_final[, a], spec),
            refine     = mode,
            n_nodes    = c(initial = length(lev0), final = length(lev1))
        )
    }
    if (!length(out)) return(NULL)
    out
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
                                  hessian_pd_mode = 0L,
                                  step_curvature_mode = 0L,
                                  inner_refresh = 1L,
                                  fixed_block_p = 0L,
                                  fixed_block_constraints = NULL) {
    function(new_cells, warm_start = NULL, store_extras = FALSE,
             max_iter_override = NULL, n_threads_outer = 1L,
             inner_refresh_override = NULL, tol_override = NULL,
             x_init_per_cell = NULL,
             # Inner-Laplace skewness diagnostic, opt-in
             # like `store_extras`. Off by default so every existing caller
             # (adaptive-grid refinement, the outer Pareto-k re-evaluation)
             # is unaffected; .nlj_inner_skew_at_theta() is the only caller
             # that sets it, at a single (MAP) row of `new_cells`.
             compute_skew = FALSE, skew_idx = NULL,
             # Subspace debias: the kernel-facing request
             # list. Off by default, so every existing caller re-solves exactly
             # what it did; `.nl_subspace_debias_attach()` is the only caller
             # that sets it, over the fit's whole settled grid.
             debias = NULL,
             # Corrected integrated Laplace: the
             # kernel-facing request list, off by default for the same reason.
             cila = NULL) {
        new_grids <- .joint_grids_from_cells(new_cells, cp)
        slice_x_init <- if (!is.null(warm_start) && !is.null(warm_start$mode))
                        as.numeric(warm_start$mode) else x_init_default
        mi <- if (is.null(max_iter_override)) max_iter
              else as.integer(max_iter_override)
        # Per-call Shamanskii reuse override for the outer Pareto-k diagnostic
        #: the diagnostic re-solves only need the converged
        # log-marginal, so they run with factor reuse (grad-only scatter on the
        # off-factor steps) even when the fit itself keeps refresh = 1.
        ir <- if (is.null(inner_refresh_override)) inner_refresh
              else as.integer(inner_refresh_override)
        # Per-call inner-tol override for the diagnostic:
        # loosened to .K_DIAG_TOL since the Laplace log-marginal error is
        # O(tol^2). Never tighter than the fit's own tol.
        tl <- if (is.null(tol_override)) tol else max(as.numeric(tol_override), tol)
        # The refinement / consistency passes call serially (n_threads_outer
        # left at 1) and chain warm-starts cell-to-cell; the outer Pareto-k
        # re-evaluation passes its whole importance batch in one call with
        # n_threads_outer > 1 so the independent re-solves run concurrently,
        # each warm-started from the broadcast modal mode. Tiling is left off:
        # the IS batch is not a per-axis alpha lattice, so there is no tile
        # structure for the three-tier warm-start to exploit.
        # Both the precision and the per-cell fixed-effect block leave this
        # closure only inside `extras`, so a call that asks for no extras
        # requests neither. That keeps the outer Pareto-k batch -- hundreds of
        # cells, `store_extras` off -- from paying for either.
        res_x <- backend$call_kernel(arms, prior, cp, new_grids,
                                      mi, tl, n_threads,
                                      slice_x_init,
                                      isTRUE(store_Q) && isTRUE(store_extras),
                                      fixed_block_p = if (isTRUE(store_extras))
                                          as.integer(fixed_block_p) else 0L,
                                      fixed_block_constraints =
                                          fixed_block_constraints,
                                      arm_names = arm_names,
                                      n_threads_outer = as.integer(n_threads_outer),
                                      tile_warm = FALSE,
                                      cell_coupling = cell_coupling,
                                      hessian_pd_mode = hessian_pd_mode,
                                      step_curvature_mode = step_curvature_mode,
                                      inner_refresh = ir,
                                      x_init_per_cell = x_init_per_cell,
                                      compute_skew = compute_skew,
                                      skew_idx = skew_idx,
                                      debias = debias,
                                      cila = cila)
        extras <- NULL
        if (isTRUE(store_extras)) {
            n <- nrow(new_cells)
            extras <- vector("list", n)
            modes_mat  <- res_x$modes
            n_iter_vec <- res_x$n_iter
            Qp <- res_x$Q_csc_p_per_grid
            Qi <- res_x$Q_csc_i_per_grid
            Qx <- res_x$Q_csc_x_per_grid
            CB <- res_x$cov_block_per_grid
            for (k in seq_len(n)) {
                e <- list()
                if (!is.null(modes_mat))  e$mode   <- as.numeric(modes_mat[k, ])
                if (!is.null(n_iter_vec)) e$n_iter <- as.integer(n_iter_vec[k])
                if (!is.null(Qp))         e$Q_csc_p <- Qp[[k]]
                if (!is.null(Qi))         e$Q_csc_i <- Qi[[k]]
                if (!is.null(Qx))         e$Q_csc_x <- Qx[[k]]
                if (!is.null(CB))         e$cov_block <- CB[[k]]
                extras[[k]] <- e
            }
        }
        list(log_marginal = res_x$log_marginal, extras = extras,
             inner_skew = res_x$inner_skew,
             inner_skew_gamma1 = res_x$inner_skew_gamma1,
             inner_skew_gamma1_declined = res_x$inner_skew_gamma1_declined,
             inner_skew_idx = res_x$inner_skew_idx,
             inner_skew_dropped = res_x$inner_skew_dropped,
             inner_skew_declined = res_x$inner_skew_declined,
             inner_skew_arms_declined = res_x$inner_skew_arms_declined,
             inner_is_z = res_x$inner_is_z,
             inner_is_sigma = res_x$inner_is_sigma,
             inner_is_log_joint = res_x$inner_is_log_joint,
             inner_is_declined = res_x$inner_is_declined,
             debias_draws_per_grid = res_x$debias_draws_per_grid,
             debias_idx = res_x$debias_idx,
             debias_accept = res_x$debias_accept,
             debias_declined = res_x$debias_declined,
             cila_log_w_per_grid = res_x$cila_log_w_per_grid,
             cila_fixed_per_grid = res_x$cila_fixed_per_grid,
             cila_log_marginal = res_x$cila_log_marginal,
             cila_variant = res_x$cila_variant,
             cila_declined = res_x$cila_declined,
             cila_fallback = res_x$cila_fallback)
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
    CB <- res$cov_block_per_grid
    lapply(seq_len(n), function(k) {
        e <- list()
        if (!is.null(modes_mat))  e$mode   <- as.numeric(modes_mat[k, ])
        if (!is.null(n_iter_vec)) e$n_iter <- as.integer(n_iter_vec[k])
        if (!is.null(Qp))         e$Q_csc_p <- Qp[[k]]
        if (!is.null(Qi))         e$Q_csc_i <- Qi[[k]]
        if (!is.null(Qx))         e$Q_csc_x <- Qx[[k]]
        if (!is.null(CB))         e$cov_block <- CB[[k]]
        e
    })
}

# The first cell carrying `field`, or 0 when none does.
#
# Which per-cell side data a fit holds is a property of the FIT (`store_Q`, the
# kernel's own outputs), but which cells hold it is not: a cheap-pass pruned
# cell is never solved, so it has no mode, no precision and no covariance
# block, whatever the fit stored. Reading cell 1 to decide for the whole list
# reads a pruned cell whenever the screen dropped the grid's first corner.
.joint_extras_first_with <- function(extras, field) {
    for (k in seq_along(extras)) {
        if (!is.null(extras[[k]][[field]])) return(k)
    }
    0L
}

# Glue refined extras + log_marginal + refining_axis back into the joint
# kernel result. Downstream `.joint_recalibrate_axis_mean` /
# `.nl_posterior_moments` / `.nl_attach_axis_sd` read `res$modes`,
# `res$log_marginal`, `res$refining_axis` directly; this keeps them in sync
# after refinement without touching their implementations.
#
# Every per-cell list is rewritten from `extras`, which the refinement passes
# extend cell for cell with the grid. A list left behind indexes the grid the
# fit had BEFORE refinement, and the per-cell precisions are what
# `tulpa_posterior_draws()` builds its mixture from, so a stale one stops
# prediction on a fit that otherwise converged.
.joint_glue_extras_to_res <- function(res, theta_grid_matrix, log_marginal,
                                      extras, refining_axis) {
    res$log_marginal  <- log_marginal
    res$n_grid        <- nrow(theta_grid_matrix)
    res$refining_axis <- refining_axis
    if (is.null(extras) || length(extras) == 0L) return(res)
    k_mode <- .joint_extras_first_with(extras, "mode")
    if (k_mode > 0L) {
        n_x <- length(extras[[k_mode]]$mode)
        res$modes <- do.call(rbind, lapply(extras, function(e) {
            if (is.null(e$mode)) rep(NA_real_, n_x) else as.numeric(e$mode)
        }))
    }
    if (.joint_extras_first_with(extras, "n_iter") > 0L) {
        res$n_iter <- vapply(extras, function(e) {
            if (is.null(e$n_iter)) NA_integer_ else as.integer(e$n_iter)
        }, integer(1))
    }
    if (.joint_extras_first_with(extras, "Q_csc_p") > 0L) {
        res$Q_csc_p_per_grid <- lapply(extras, `[[`, "Q_csc_p")
        res$Q_csc_i_per_grid <- lapply(extras, `[[`, "Q_csc_i")
        res$Q_csc_x_per_grid <- lapply(extras, `[[`, "Q_csc_x")
    }
    if (.joint_extras_first_with(extras, "cov_block") > 0L) {
        res$cov_block_per_grid <- lapply(extras, `[[`, "cov_block")
    }
    res
}

# ----------------------------------------------------------------------------
# Per-cell fixed-effect mode + precision for a joint fit.
#
# `.nested_fixed_moments()` (R/methods_generic.R) marginalizes the fixed effects
# over the outer grid by the law of total variance, and reads one representation
# to do it: `$grid_modes[[k]]` and `$grid_hessians[[k]]`, the cell-k fixed-effect
# mode and marginal precision. `tulpa_nested_laplace()` fills that pair from the
# stored per-cell precision; without it every reported interval is NA. This is
# the joint counterpart, filling the SAME pair so both tiers reach the one
# marginalizer -- and it is called from both the single-block and the multi-block
# joint driver, which is why it lives here rather than in either of them.
#
# Both joint layouts stack every arm's coefficients as a contiguous prefix of the
# latent vector (`.joint_layout()` / `.joint_multi_layout()` both start
# `beta_start` at 0 and run the arms consecutively), so the fixed-effect block is
# latent indices `1:n_fixed` -- the same span `.joint_fixed_layout()` names.
#
# The per-cell block arrives on `$cov_block_per_grid`, extracted by the inner
# Newton loop inside each cell's own solve through the same
# `extract_inner_vcov_block_cell()` the R-level `cpp_joint_inner_vcov_blocks()`
# drives, so the block is the constrained covariance the fit's own posterior
# draws are generated from -- the field sum-to-zero groups go in with the kernel
# call. Nothing here holds a precision: the whole grid's worth of them is never
# resident, and what the fit carries home is O(n_fixed^2) per cell.
#
# A decline is recorded on `$grid_fixed_declined` rather than left as an absent
# field, so a fit that reports NA intervals says why it does.
.joint_attach_grid_fixed <- function(res, n_fixed) {
    decline <- function(reason) {
        res$grid_modes <- NULL
        res$grid_hessians <- NULL
        res$grid_fixed_declined <- reason
        res
    }
    p <- as.integer(n_fixed %||% 0L)
    if (length(p) != 1L || is.na(p) || p < 1L) return(decline("no_fixed_effects"))

    # Read before the retention checks below, because it is upstream of them: a
    # cell whose inner solve never reached a mode has no fixed-effect
    # covariance to hand over, so the retention would report the absence it
    # trips over first and name a step downstream of the real cause.
    if (!.nested_any_weighted_converged(res)) return(decline("not_converged"))

    V <- res$cov_block_per_grid
    if (is.null(V)) return(decline("block_not_extracted"))

    modes <- res$modes
    w     <- res$weights
    n_grid <- length(V)
    # Every per-cell array must describe the SAME grid. A refinement pass that
    # rewrote the grid without carrying the blocks along would otherwise be
    # marginalized against stale cells.
    if (!is.matrix(modes) || nrow(modes) != n_grid || ncol(modes) < p ||
        is.null(w) || length(w) != n_grid) {
        return(decline("grid_misaligned"))
    }

    grid_hessians <- vector("list", n_grid)
    grid_modes    <- vector("list", n_grid)
    for (k in seq_len(n_grid)) {
        Vk <- V[[k]]
        ok <- is.matrix(Vk) && nrow(Vk) == p && ncol(Vk) == p
        Hk <- if (!ok) NULL else tryCatch(solve(Vk), error = function(e) NULL)
        # A cell carrying no usable block is fatal only if it carries weight;
        # a zero-weight cell contributes nothing and its slot is left empty for
        # `.nested_fixed_moments()` to skip. The slot is SKIPPED rather than
        # assigned NULL: `l[[k]] <- NULL` removes the element, which on a
        # trailing empty cell shortens the list below `n_grid` and takes the
        # whole marginalization down on the length check.
        if (is.null(Hk)) {
            if (is.finite(w[k]) && w[k] > 0) {
                return(decline("cell_block_unavailable"))
            }
            next
        }
        grid_hessians[[k]] <- Hk
        grid_modes[[k]]    <- as.numeric(modes[k, seq_len(p)])
    }

    # A retention holding no cell the weights put mass on cannot produce a
    # coefficient table, so it is a decline with a reason rather than a pair of
    # lists `.nested_fixed_moments()` silently returns NULL on.
    filled <- !vapply(grid_hessians, is.null, logical(1))
    if (!any(filled & is.finite(w) & w > 0)) {
        return(decline("no_weighted_cell_block"))
    }

    res$grid_hessians <- grid_hessians
    res$grid_modes    <- grid_modes
    res$grid_fixed_declined <- NA_character_
    res
}

# The one place either joint driver settles what it retains for the fixed-effect
# marginalization. The per-cell blocks are already on the result; this turns
# them into the `$grid_modes` / `$grid_hessians` pair the marginalizer reads,
# and drops the raw blocks afterwards.
.joint_finalize_grid_fixed <- function(res, n_fixed, keep_grid_hessians) {
    res <- if (isTRUE(keep_grid_hessians)) {
        .joint_attach_grid_fixed(res, n_fixed)
    } else {
        res$grid_fixed_declined <- "not_requested"
        res
    }
    res$cov_block_per_grid <- NULL
    res
}

# Latent dimension above which `force_sparse = "auto"` selects the sparse
# backend. Below it the dense factorization is the cheaper of the two: the
# sparse path pays a symbolic-analysis and indirection overhead that only earns
# its keep once the field is large enough for the fill-in saving to dominate.
.FORCE_SPARSE_AUTO_NX <- 1000L

# Resolve `control$force_sparse` to the boolean the kernels take. TRUE / FALSE
# pass through; "auto" compares the threshold against the latent dimension the
# kernel will actually build, supplied by the caller as a thunk because the
# single-block and multi-block paths reach it through different layouts
# (`.joint_layout` vs `.joint_multi_layout`) at different points. A thunk that
# fails or yields no finite n_x falls back to dense: the sparse path is the
# specialized one, so an unknown size should not silently opt into it.
.resolve_force_sparse <- function(force_sparse, n_x_fn) {
  if (is.logical(force_sparse) && length(force_sparse) == 1L &&
      !is.na(force_sparse)) {
    return(force_sparse)
  }
  if (!identical(force_sparse, "auto")) {
    stop("`control$force_sparse` must be TRUE, FALSE, or \"auto\".",
         call. = FALSE)
  }
  n_x <- tryCatch(n_x_fn(), error = function(e) NULL)
  if (is.null(n_x) || length(n_x) != 1L || !is.finite(n_x)) return(FALSE)
  n_x > .FORCE_SPARSE_AUTO_NX
}

# Which factorization the DENSE inner joint Newton uses on the Hessian it has
# already assembled densely. `force_sparse` above chooses the driver -- dense
# assembly or sparse assembly; this chooses the solver inside the dense one, and
# the two are independent. "auto" is the latent-dimension threshold, and the two
# explicit settings are what drive one problem through both backends.
.INNER_FACTORIZATION <- c("auto", "sparse", "dense")

.resolve_inner_factorization <- function(x) {
  if (is.null(x)) return(0L)
  if (!is.character(x) || length(x) != 1L || !(x %in% .INNER_FACTORIZATION)) {
    stop("`control$inner_factorization` must be one of ",
         paste(dQuote(.INNER_FACTORIZATION, FALSE), collapse = ", "), ".",
         call. = FALSE)
  }
  switch(x, auto = 0L, sparse = 1L, dense = -1L)
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

# The one R -> compiled-kernel dispersion boundary for the joint tier. Every
# R-level `phi` in the package is the engine convention (gaussian / lognormal:
# the residual VARIANCE); the kernels parameterize those two families by the
# residual SD. An arm carries its own family, so the arm scalar, the per-arm
# outer-grid override (`phi_grid_per_arm`) and the per-species batch matrix
# (`phi_batch`, [n_arms x n_batch]) all convert against the same family here
# rather than at each of the joint entries.
#' @keywords internal
.joint_phi_args_to_kernel <- function(args) {
    arms <- args$arms_list
    if (is.null(arms)) return(args)
    fams <- vapply(arms, function(a) as.character(a$family %||% ""), character(1))
    for (k in seq_along(arms)) {
        arms[[k]]$phi <- .phi_to_kernel(fams[k], as.numeric(arms[[k]]$phi %||% 1.0))
    }
    args$arms_list <- arms
    g <- args$phi_grid_per_arm
    if (!is.null(g)) {
        for (k in seq_along(g)) {
            if (!is.null(g[[k]])) {
                g[[k]] <- .phi_to_kernel(fams[k], as.numeric(g[[k]]))
            }
        }
        args$phi_grid_per_arm <- g
    }
    pb <- args$phi_batch
    if (!is.null(pb)) {
        pb <- as.matrix(pb)
        for (k in seq_len(nrow(pb))) {
            pb[k, ] <- .phi_to_kernel(fams[k], as.numeric(pb[k, ]))
        }
        args$phi_batch <- pb
    }
    args
}
