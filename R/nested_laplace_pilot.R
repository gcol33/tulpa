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
.nl_pilot_alpha <- function(spec, n) {
    if (!is.list(spec)) return(list(spec = spec, moved = FALSE))
    if (!is.null(spec$alpha_grid) && length(spec$alpha_grid) > 0L) {
        return(list(spec = spec, moved = FALSE))
    }
    cur <- spec$alpha_n
    if (!is.null(cur) && length(cur) > 0L && as.integer(cur) <= n) {
        return(list(spec = spec, moved = FALSE))
    }
    if (is.null(cur) &&
        length(.nl_grid_axis("copy_alpha")) <= n) {
        return(list(spec = spec, moved = FALSE))
    }
    spec$alpha_n <- as.integer(n)
    list(spec = spec, moved = TRUE)
}

# The per-arm `field_coef` counterpart, for the single-block path's inline copy
# declaration. Same rule: a stated `grid` is kept, a resolution is lowered.
.nl_pilot_field_coef <- function(arm, n) {
    fc <- arm$field_coef
    if (is.null(fc)) return(list(arm = arm, moved = FALSE))
    if (is.numeric(fc)) return(list(arm = arm, moved = FALSE))
    if (is.character(fc) && length(fc) == 1L) fc <- list(name = fc)
    if (!is.list(fc)) return(list(arm = arm, moved = FALSE))
    if (!is.null(fc$grid) && length(fc$grid) > 0L) {
        return(list(arm = arm, moved = FALSE))
    }
    cur <- fc[["n"]]
    if (!is.null(cur) && length(cur) > 0L && as.integer(cur) <= n) {
        return(list(arm = arm, moved = FALSE))
    }
    if (is.null(cur) && length(.nl_grid_axis("copy_alpha")) <= n) {
        return(list(arm = arm, moved = FALSE))
    }
    fc[["n"]] <- as.integer(n)
    arm$field_coef <- fc
    list(arm = arm, moved = TRUE)
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

    multi <- .is_multi_block_prior(prior)
    if (multi) {
        for (b in seq_along(prior)) {
            path <- if (b %in% copy_blocks) "copy" else "registry"
            r <- .nl_pilot_block(prior[[b]], path, n)
            prior[[b]] <- r$block
            if (length(r$moved)) moved <- c(moved, paste0("b", b, ".", r$moved))
            if (length(r$kept))  kept  <- c(kept,  paste0("b", b, ".", r$kept))
        }
    } else if (is.list(prior) && !is.null(prior$type)) {
        r <- .nl_pilot_block(prior, "joint_single", n)
        prior <- r$block
        moved <- c(moved, r$moved)
        kept  <- c(kept,  r$kept)
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
    } else if (is.list(responses)) {
        for (k in seq_along(responses)) {
            r <- .nl_pilot_field_coef(responses[[k]], n)
            responses[[k]] <- r$arm
            if (isTRUE(r$moved)) {
                moved <- c(moved, paste0("alpha.", names(responses)[k] %||% k))
            }
        }
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
# diagnostic goes with them; the placement path
# (`.joint_attach_pareto_k_placement()`) supplies the mode and Hessian without
# it, which is the same route a `diagnose_k = FALSE` fit already takes.
.nl_pilot_control <- function(control) {
    utils::modifyList(control, list(
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
        edge_axes = detect$edge_axes %||% character(0)
    )
    res$outer_grid_pilot_declined <- declined
    res
}
