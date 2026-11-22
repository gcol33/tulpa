# Placement pilot for the outer-grid auto-recenter (gcol33/tulpa#636).
#
# The placement pass answers ONE question -- where should the outer grid go --
# and needs two numbers to answer it: the argmax cell of the outer weights, and
# the FD curvature stencil at that cell. Both are read off `log_marginal`.
# Neither is read off the INTEGRATION the pass paid for, so when a placement
# fires, every inner Newton solve of the grid it detected on is discarded. On a
# 4435-cell `occu_cover` joint fit that is 120 cells at 25.2 s/cell thrown away
# before the placed grid's own 120 cells are solved.
#
# A pilot answers the placement question on a COARSER grid over the SAME spans,
# and hands the answer to the same rescue. The rescues need no change for this:
# each detects on the fit it is handed and writes onto the prior it is handed,
# so `(pilot fit, full prior)` places the FULL grid from a cheap detection,
# while `(full fit, full prior)` -- what every caller passed before -- is what it
# always was. What the CALLER has to add is the other half: a pilot grid is too
# coarse to integrate, so a placement that DECLINES must be followed by the full
# fit the pilot stood in for.
#
# COST. With `F` full cells, `P` pilot cells, `S` stencil cells and a firing
# rate `p`:
#
#   without a pilot   F + p (S + F)
#   with a pilot      P + p S + F
#
# so the pilot is cheaper exactly when `P < p F`. `p` is a property of the
# WORKLOAD and not of the engine -- every species of the 78-species run
# gcol33/tulpa#636 reports fires, while a fit whose default axis already
# brackets its mode never does -- which is why this is a knob with an off
# default rather than a policy the engine picks. `control$recenter_pilot = TRUE`
# turns it on at the default resolution; an integer names the resolution.
#
# SCOPE: the joint front door (`tulpa_nested_laplace_joint()`), where the cost
# was measured. The standalone registry path stores a family's axes PRE-PAIRED
# (`.nl_fill_family_axes()` crosses them into one row per tuple), so coarsening
# them per axis there would re-pair a grid rather than thin it; `.NL_PATH_CROSSES`
# is the property that says which paths cross their own axes, and a block whose
# path does not is left at full resolution and recorded.

# One axis, thinned to at most `n` nodes with BOTH endpoints kept. The endpoints
# are what make the pilot a coarser read of the SAME span rather than a
# different one: the rail test asks whether the posterior's mode sits at a
# boundary node, so a pilot that moved the boundary would be answering about a
# grid the fit will never integrate. Returns NULL when the axis is already at or
# below the pilot resolution -- there is nothing to thin, and a caller reads NULL
# as "left alone" rather than as an error.
.nl_pilot_axis <- function(x, n) {
    if (!is.numeric(x) || is.matrix(x)) return(NULL)
    v <- as.numeric(x)
    if (length(v) <= n) return(NULL)
    idx <- unique(as.integer(round(seq(1, length(v), length.out = n))))
    if (length(idx) >= length(v)) return(NULL)
    v[idx]
}

# The pilot resolution `control$recenter_pilot` asks for: NA when the pilot is
# off. TRUE takes `.NL_RECENTER$pilot_n`; an integer names its own. Two nodes is
# the floor -- an axis thinned below that carries no interior, so the rail test
# reads every node as a boundary.
.nl_pilot_n <- function(control) {
    x <- control$recenter_pilot
    if (is.null(x) || isFALSE(x)) return(NA_integer_)
    if (isTRUE(x)) return(as.integer(.nl_recenter("pilot_n")))
    n <- suppressWarnings(as.integer(x))
    if (length(n) != 1L || is.na(n) || n < 2L) {
        stop("control$recenter_pilot must be TRUE, FALSE, or a single integer ",
             ">= 2 (the pilot grid's per-axis node count); got ",
             paste(utils::capture.output(utils::str(x)), collapse = " "),
             call. = FALSE)
    }
    n
}

# One prior block at pilot resolution, plus the record of what moved and what
# did not. An axis the caller left absent is materialised at the pilot
# resolution from the engine's own default for that (path, family, field), so
# the axis the driver would have expanded to full resolution is thinned too --
# writing it onto the PILOT block only, which no rescue reads for provenance.
# A data-dependent default (spde, tgmrf) has no axis to materialise here and is
# left alone.
.nl_pilot_block <- function(block, path, n) {
    type   <- tolower(block$type %||% "")
    fields <- .nl_path_axis_fields(type, path)
    # `alpha_grid` is a copy SPEC field, not a block field -- the copy
    # coefficient reaches the driver through `copy` / `field_coef`, and is
    # thinned there by its own resolution knob.
    fields <- setdiff(fields, "alpha_grid")
    if (!length(fields)) {
        return(list(block = block, moved = character(0), kept = character(0)))
    }
    if (!.nl_path_crosses(path) && length(fields) > 1L) {
        return(list(block = block, moved = character(0), kept = fields))
    }
    out   <- block
    moved <- character(0)
    kept  <- character(0)
    for (f in fields) {
        g <- block[[f]]
        if (is.null(g)) {
            key <- .nl_path_axis_key(type, path, f)
            if (is.null(key)) { kept <- c(kept, f); next }
            g <- tryCatch(.nl_grid_axis(key), error = function(e) NULL)
            if (is.null(g)) { kept <- c(kept, f); next }
        }
        ax <- .nl_pilot_axis(g, n)
        if (is.null(ax)) { kept <- c(kept, f); next }
        out[[f]] <- ax
        moved    <- c(moved, f)
    }
    list(block = out, moved = moved, kept = kept)
}

# Thin a copy coefficient's axis by RESOLUTION, never by subsampling. The axis
# carries prior structure -- an atom at 0 against a log-spaced slab -- so
# dropping nodes from a stated one states a different axis, while `alpha_n`
# re-reads the engine's own axis at a lower resolution and keeps the atom and
# the slab bounds (gcol33/tulpa#633). A spec that STATES its nodes therefore
# keeps them, and says so.
#
# A grid a wrapper MARKED as its own default (`auto_grid()`) is not a statement
# the pilot has to keep: the placement pass may move it, so the pilot thins it
# the way it thins every other default axis, with the atom and both ends of the
# slab kept (`.nl_pilot_alpha_grid()`).
.nl_pilot_alpha_grid <- function(g, n) {
    if (!is_auto_grid(g) || !auto_grid_place(g)) return(NULL)
    v <- as.numeric(g)
    slab <- .nl_pilot_axis(v[v > 0], n)
    if (is.null(slab)) return(NULL)
    auto_grid(c(v[v <= 0], slab))
}

# The one thinning rule over a copy coefficient's axis, wherever its spec
# lives: an explicit `grid` is thinned by `.nl_pilot_alpha_grid()` (a stated
# one is kept), otherwise the resolution `res` is lowered to `n` unless it is
# already within it, or is unset and the engine's default `copy_alpha` axis is.
# NULL when nothing moves; otherwise `list(grid =)` or `list(res =)`, the one
# entry to write back.
.nl_pilot_copy_axis <- function(grid, res, n) {
    if (!is.null(grid) && length(grid) > 0L) {
        thin <- .nl_pilot_alpha_grid(grid, n)
        if (is.null(thin)) return(NULL)
        return(list(grid = thin))
    }
    if (!is.null(res) && length(res) > 0L && as.integer(res) <= n) {
        return(NULL)
    }
    if (is.null(res) && length(.nl_grid_axis("copy_alpha")) <= n) {
        return(NULL)
    }
    list(res = as.integer(n))
}

.nl_pilot_alpha <- function(spec, n) {
    if (!is.list(spec)) return(list(spec = spec, moved = FALSE))
    th <- .nl_pilot_copy_axis(spec$alpha_grid, spec$alpha_n, n)
    if (is.null(th)) return(list(spec = spec, moved = FALSE))
    if (!is.null(th[["grid"]])) spec$alpha_grid <- th[["grid"]]
    else spec$alpha_n <- th[["res"]]
    list(spec = spec, moved = TRUE)
}

# The whole pilot: the coarsened call arguments, and the record of what it
# thinned. `active = FALSE` means the fit runs exactly as it does with the knob
# off -- either the knob is off, the placement pass is off (nothing would read
# the pilot), or every axis is already at or below the pilot resolution, in
# which case a pilot would be a second copy of the same grid.
#
# `copy_blocks` are the 1-based block indices the multi-block path resolved as
# copy blocks; a block outside that set is on the registry path and reads the
# registry's paired-axis convention.
.nl_recenter_pilot <- function(prior, phi_grid, copy, responses,
                               copy_blocks = integer(0),
                               n = NA_integer_, enabled = TRUE) {
    off <- list(active = FALSE, prior = prior, phi_grid = phi_grid,
                copy = copy, responses = responses,
                axes = character(0), kept = character(0), n = n)
    if (is.na(n) || !isTRUE(enabled)) return(off)

    moved <- character(0)
    kept  <- character(0)

    for (b in seq_along(prior)) {
        path <- if (b %in% copy_blocks) "copy" else "registry"
        r <- .nl_pilot_block(prior[[b]], path, n)
        prior[[b]] <- r$block
        if (length(r$moved)) moved <- c(moved, .axis_label(b, r$moved))
        if (length(r$kept))  kept  <- c(kept,  .axis_label(b, r$kept))
    }

    if (is.list(phi_grid)) {
        for (k in seq_along(phi_grid)) {
            ax <- .nl_pilot_axis(phi_grid[[k]], n)
            nm <- names(phi_grid)[k] %||% as.character(k)
            if (is.null(ax)) {
                if (!is.null(phi_grid[[k]]) && length(phi_grid[[k]]) > 1L) {
                    kept <- c(kept, paste0("phi_", nm))
                }
                next
            }
            phi_grid[[k]] <- ax
            moved <- c(moved, paste0("phi_", nm))
        }
    }

    if (is.list(copy) && length(copy)) {
        specs <- if (is.null(copy$arm)) copy else list(copy)
        one   <- is.null(copy$arm)
        for (i in seq_along(specs)) {
            r <- .nl_pilot_alpha(specs[[i]], n)
            specs[[i]] <- r$spec
            if (isTRUE(r$moved)) moved <- c(moved, paste0("alpha", i))
            else kept <- c(kept, paste0("alpha", i))
        }
        copy <- if (one) specs else specs[[1L]]
    }

    if (!length(moved)) return(off)
    list(active = TRUE, prior = prior, phi_grid = phi_grid, copy = copy,
         responses = responses, axes = moved, kept = unique(kept), n = n)
}

# The control a pilot fit runs under. Everything the pilot's two outputs -- the
# argmax cell and the FD curvature at it -- do not read is switched off, because
# a pilot's cells are discarded whichever way the placement goes: adaptive
# refinement and the var-of-means consistency pass would add cells to a grid
# that is not integrated, and the inner-skew / correction / debias / CILA layers
# are per-cell payloads nothing downstream of a pilot reads. The outer k-hat
# diagnostic goes with them; the placement mode-find
# (`.joint_attach_placement()`) supplies the mode and Hessian without it, which
# is the same route a `diagnose_k = FALSE` fit already takes.
#
# `pilot_screen` goes one step further on a screened fit: the pilot's grid is
# read off the cheap screen alone (`screen_only` in run_nested_laplace_grid),
# with no full pass over the kept cells. The rail test, the heaviest cell and
# the weights the placement seeds from are all readings of where the posterior
# sits, which is what the screen ranks, and whichever way the placement goes
# the pilot's cells are discarded (gcol33/tulpa#922). An unscreened pilot
# (prune off) is unaffected.
.nl_pilot_control <- function(control) {
    utils::modifyList(control, list(
        pilot_screen             = TRUE,
        adaptive_grid            = FALSE,
        var_of_means_consistency = FALSE,
        keep_grid_hessians       = FALSE,
        diagnose_k               = FALSE,
        k_quality                = "none",
        diagnose_skew            = FALSE,
        skew_correct             = FALSE,
        subspace_debias          = NULL,
        cila                     = NULL,
        checkpoint               = NULL
    ))
}

# Record the pilot on the fit it placed (or failed to place). A two-pass fit is
# then visible in the OBJECT rather than only in the progress log, which is what
# a caller reading a saved fit has.
.nl_pilot_attach <- function(res, pilot, detect, declined = NULL) {
    if (!isTRUE(pilot$active)) return(res)
    res$outer_grid_pilot <- list(
        n_pilot   = as.integer(pilot$n),
        cells     = as.integer(detect$cells %||% NA_integer_),
        axes      = pilot$axes,
        axes_kept = pilot$kept,
        # What the placement was DECIDED from. The pilot's grid is not the one
        # the fit reports, so without these a fit placed from a pilot cannot say
        # what read triggered it -- and the trigger is exactly the thing a coarse
        # detector answers differently.
        regime    = detect$regime %||% NA_character_,
        ess_grid  = detect$ess_grid %||% NA_real_,
        edge_axes = detect$edge_axes %||% character(0),
        # TRUE when the detecting grid was read off the cheap screen alone,
        # with no full pass over its cells.
        screened  = isTRUE(detect$screened)
    )
    res$outer_grid_pilot_declined <- declined
    res
}
