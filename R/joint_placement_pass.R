# The joint driver's outer-grid placement pass.
#
# Every axis `tulpa_nested_laplace_joint()` may move -- each copy block's field
# SD, every registry axis of a non-copy block, the per-arm dispersions, the
# copy scale, a field's mixing weight and a free covariance's coordinates -- is
# laid from the ONE outer mode the detecting fit's mode-find reached (`.joint_attach_placement()`), and the grid is refit once per attempt
# with every placement applied. The mode-find varies every axis the grid lays
# more than one node on, so one run already holds what each axis needs. Placing
# the families one refit at a time would pay a full grid solve per family, and
# each of those refits would run a mode-find of its own for whichever axis it
# left unresolved (gcol33/tulpa#925).
#
# A FAMILY is one entry of `.joint_placement_families()`: how to find its axes
# on a fit, whether the caller left each one movable, when one needs placing,
# and where its new nodes are written.
#
#   slots(res, st)          the family's axes. Each slot carries `label` (its
#                           name on the fit's placement records), `axis` and
#                           `block` (where the placement mode is read), `ref`
#                           (its incoming nodes on a fit).
#   hold(slot, st)          NULL when the pass may move the axis, else the
#                           decline reason, in `.nl_axis_hold()`'s vocabulary.
#   fires(res, slot)        does the axis need placing on this fit.
#   write(st, slot, nodes)  the call state with the axis laid on `nodes`.
#   together                one firing axis moves every movable axis of the
#                           family. The dispersions: each is crossed onto the
#                           tensor independently of the field, and moving them
#                           as a set costs one refit.
#   escalate                a second attempt engages the default SD prior, for a
#                           near-separation mode that keeps running on a field
#                           SD whatever the geometry.
#   idle                    the reason an axis records when it never needed
#                           placing.
#
# `st` is the call state the refit reads: `prior`, `prior_sigma`, `phi_grid`,
# `responses`, `copy`, and `cp`, the resolved multi-block copy spec.
#
# Decline reasons, per axis (`outer_grid_axis_declined`) and for the fit
# (`outer_grid_recenter_declined`, `.nl_decline_recenter()`), keep the
# vocabulary documented beside `.NL_AXIS_SCOPED_DECLINE`: an axis the caller
# held records its hold; one whose placement could not be built records why
# (`"no_usable_curvature"` and the clamp reasons); one still firing when the
# attempts ran out records `"attempts_exhausted"`; the rest record the family's
# `idle` reason.

.joint_placement_families <- function(auto = logical(0), phi_auto = NULL) {
    list(
        copy_sigma = list(
            slots = function(res, st) {
                if (is.null(st$cp) || !isTRUE(st$cp$has_copy)) return(list())
                n_b <- .nl_fit_n_blocks(res)
                lapply(st$cp$copy_blocks_zero + 1L, function(b) list(
                    label = .nl_axis_alias("sigma", b, n_b)[1L],
                    axis = "sigma", block = b,
                    ref = function(r) .nl_axis_ref_nodes(
                        r, "sigma", b, .nl_fit_n_blocks(r))))
            },
            hold = function(s, st)
                .nl_axis_hold(st$prior[[s$block]], "sigma_grid",
                              .nl_auto_fields_at(auto, s$block), type = ".copy"),
            fires = function(res, s) .nl_axis_placement_fires(res, s$label, "log"),
            write = function(st, s, nodes) {
                st$prior[[s$block]]$sigma_grid <- auto_grid(nodes)
                st
            },
            together = FALSE, escalate = TRUE, idle = "grid_not_collapsed"),
        phi = list(
            slots = function(res, st)
                lapply(.nl_phi_axis_slots(res, st$phi_grid), function(p) list(
                    label = p$axis, axis = p$axis, block = NULL, arm = p$arm,
                    ref = function(r) .nl_rescue_axis_nodes(r, p$axis))),
            hold = function(s, st) .nl_phi_axis_hold(s$arm, phi_auto),
            fires = function(res, s) .nl_axis_placement_fires(res, s$axis, "log"),
            write = function(st, s, nodes) {
                st$phi_grid[[s$arm]] <- nodes
                st
            },
            together = TRUE, escalate = FALSE, idle = "grid_resolves_posterior"),
        alpha = list(
            slots = function(res, st) .joint_alpha_slots(res, st),
            hold = function(s, st)
                .nl_axis_hold(list(alpha_grid = .joint_alpha_declared(st, s)),
                              "alpha_grid", type = ".copy"),
            fires = function(res, s) .nl_axis_placement_fires(res, s$label, "log"),
            write = function(st, s, nodes) .joint_alpha_write(st, s, nodes),
            together = FALSE, escalate = FALSE, idle = "grid_resolves_posterior"),
        mixing = list(
            slots = function(res, st) .joint_mixing_slots(res, st),
            hold = function(s, st)
                .nl_axis_hold(st$prior[[s$block]], s$field,
                              .nl_auto_fields_at(auto, s$block), type = s$type),
            fires = function(res, s) .nl_axis_placement_fires(res, s$label, s$tag),
            write = function(st, s, nodes) {
                st$prior[[s$block]][[s$field]] <- auto_grid(nodes)
                st
            },
            together = FALSE, escalate = FALSE, idle = "grid_resolves_posterior"),
        field = list(
            slots = function(res, st) .joint_field_slots(res, st),
            hold = function(s, st)
                .nl_slot_hold(st$prior, s$slot, .nl_auto_fields_at(auto, s$block)),
            fires = function(res, s) .nl_axis_placement_fires(res, s$label, s$tag),
            write = function(st, s, nodes) .joint_field_write(st, s, nodes, auto),
            together = FALSE, escalate = FALSE, idle = "grid_resolves_posterior"),
        logchol = list(
            slots = function(res, st) .joint_logchol_slots(res, st),
            hold = function(s, st)
                .nl_axis_hold(st$prior[[s$block]], "logchol_grid",
                              .nl_auto_fields_at(auto, s$block),
                              type = tolower(st$prior[[s$block]]$type %||% "")),
            fires = function(res, s) .joint_logchol_fires(res, s$label),
            write = function(st, s, nodes) .joint_logchol_write(st, s, nodes),
            together = TRUE, escalate = FALSE, idle = "grid_resolves_posterior")
    )
}

# --- a free-covariance block's log-Cholesky axes -----------------------------
#
# An `mcar` / `miid` block integrates its Sigma = L L' on the p(p + 1) / 2
# log-Cholesky columns `b<k>.L<i><j>` (`.mcar_logchol_names()`), and is placed
# in its CPC coordinates -- the marginal log SDs and the atanh canonical partial
# correlations, column `L<i><i>` holding log sigma_i and `L<i><j>` atanh z_ij
# (`.hp_logchol_coords()`). The placement mode-find searches the block there
# (`.nl_placement_mode()`, tag `"cpc"`), the family reads each axis's rail and
# resolution there, and it writes the placed axes back as a TENSOR in those
# coordinates -- the "cpc" design (`.hp_logchol_design()`), whose prior and
# cell measure are the PC + LKJ density and the product of widths in the same
# coordinates. The axes are coupled through the one `logchol_grid` matrix the
# block carries, so the family moves them together.
.joint_logchol_slots <- function(res, st) {
    cn <- colnames(res$theta_grid) %||% character(0)
    out <- list()
    for (b in seq_along(st$prior)) {
        if (!tolower(st$prior[[b]]$type %||% "") %in% c("mcar", "miid")) next
        cols <- .hp_logchol_block_cols(cn, .axis_prefix(b))
        for (lbl in cols) {
            out[[length(out) + 1L]] <- list(
                label = lbl, axis = .axis_bare(lbl), block = b,
                ref = .joint_cpc_axis_ref(lbl))
        }
    }
    out
}

# A fit read in its free-covariance blocks' CPC coordinates: the same cells,
# weights and log densities, with each block's columns carried by
# `.hp_grid_design_coords()`. What the family's rail and resolution reads take.
.joint_fit_in_cpc <- function(res) {
    tg <- res$theta_grid
    if (is.null(tg) || !is.matrix(tg)) return(res)
    blocks <- .hp_cpc_blocks(colnames(tg) %||% character(0))
    if (length(blocks)) {
        res$theta_grid <- .hp_grid_design_coords(tg, lapply(blocks, `[[`, "design"))
    }
    res
}

# Does a free-covariance column need placing? Its rail and resolution, read in
# its block's CPC coordinates, the coordinates it is placed in.
.joint_logchol_fires <- function(res, label) {
    .nl_axis_placement_fires(.joint_fit_in_cpc(res), label, "cpc")
}

# An axis's incoming nodes in its CPC coordinate, off the fit's own cells.
.joint_cpc_axis_ref <- function(label) {
    force(label)
    function(r) {
        tg <- .joint_fit_in_cpc(r)$theta_grid
        if (is.null(tg) || !label %in% colnames(tg)) return(numeric(0))
        sort(unique(signif(as.numeric(tg[, label]), 12L)))
    }
}

# The block's grid with one axis laid on `nodes` (CPC coordinates), as the
# tensor in those coordinates over every axis's levels, mapped to the
# log-Cholesky columns. An axis not written keeps the levels the block's current
# grid gives it when that grid is a "cpc" tensor; on any other grid its
# coordinate takes no level set, so it keeps its range at the placement's node
# count. The family moves its axes together, so within one attempt every placed
# axis is written before the refit reads the grid.
.joint_logchol_write <- function(st, s, nodes) {
    blk <- st$prior[[s$block]]
    G <- .nl_strip_auto(.mcar_logchol_grid(blk))
    p <- as.integer(blk$n_fields)
    design <- list(design = "cpc", p = p)
    C <- signif(.hp_logchol_coords(G, design), 12L)
    is_cpc <- identical(.hp_logchol_design(G)$design, "cpc")
    lev <- lapply(colnames(C), function(c) {
        v <- sort(unique(C[, c]))
        if (is_cpc) v else seq(min(v), max(v), length.out = .nl_recenter("n_pts"))
    })
    names(lev) <- colnames(C)
    lev[[s$axis]] <- sort(unique(as.numeric(nodes)))
    out <- .hp_logchol_from_coords(
        as.matrix(expand.grid(lev, KEEP.OUT.ATTRS = FALSE)), design)
    attr(out, "log_jac") <- NULL
    colnames(out) <- colnames(G)
    st$prior[[s$block]]$logchol_grid <- auto_grid(out)
    st
}

# --- a field's correlation on (0, 1) -----------------------------------------
#
# A BYM2 mixing weight `rho` and a proper-CAR correlation `rho_car` live on
# (0, 1) and are placed on their logit, the coordinate the transform registry
# tags them with, from the same outer mode as the field's SD. A default axis is
# a fixed ladder graded towards 1, so a posterior reaching below its lowest
# node is cut at that node's box edge until the axis is laid where the
# mode-find found it. This family places a copied block's correlation
# (`b<k>.rho`, `b<k>.rho_car`), whose copy grid crosses each axis it carries.
# A non-copy block's correlation is one of its registry axes, paired with its
# SD, and the `field` family re-crosses the two. A correlation whose block
# declares a support the registry gives no coordinate on is not placed.
.JOINT_MIXING_AXES <- c(bym2 = "rho", car_proper = "rho_car")

.joint_mixing_slots <- function(res, st) {
    cn <- colnames(res$theta_grid) %||% character(0)
    slot <- function(b) {
        blk  <- st$prior[[b]]
        type <- tolower(blk$type %||% "")
        axis <- .JOINT_MIXING_AXES[type]
        if (is.na(axis)) return(NULL)
        label <- .axis_label(b, axis)
        if (!label %in% cn) return(NULL)
        tag <- .joint_pareto_block_tags(type, axis, blk)
        if (is.na(tag)) return(NULL)
        list(label = label, axis = unname(axis), block = b, type = type,
             field = paste0(axis, "_grid"), tag = unname(tag),
             ref = .joint_declared_axis_ref(label))
    }
    Filter(Negate(is.null), lapply(.joint_copied_blocks(st), slot))
}

# --- a block's registry field axes ------------------------------------------
#
# Every axis a non-copy block lays on a grid field of its own, as the registry
# door binds it (`.NL_REGISTRY_AXIS_FIELD`): the block's own scale (an iid
# random effect's SD `b<k>.sigma`, an areal or temporal field's precision
# `b<k>.tau`), a Gaussian-process field's variance and lengthscale
# (`b<k>.sigma2`, `b<k>.lengthscale`, `b<k>.phi_gp`), an AR(1) or proper-CAR
# block's correlation. Each is laid from the outer mode in the coordinate the
# transform registry gives it (`tag`); an axis the registry gives none is not
# placed. A block that is copied is the `copy_sigma` family's, and a BYM2
# mixing weight the `mixing` family's (`.joint_placement_slots()`).
#
# A block's vector fields are paired cells, so one axis is never written alone:
# `.joint_field_write()` re-crosses the block as the tensor of every axis's
# levels. A matrix field (a tgmrf block's `theta_grid_built`) holds every axis
# of its block as one column each, and is written whole the same way.
.joint_field_slots <- function(res, st) {
    cn <- colnames(res$theta_grid) %||% character(0)
    out <- list()
    for (sl in .joint_field_registry_slots(st)) {
        if (!sl$label %in% cn) next
        out[[length(out) + 1L]] <- list(
            label = sl$label, axis = sl$axis, block = sl$block, slot = sl,
            tag = sl$tag, ref = .joint_declared_axis_ref(sl$label),
            cross = .joint_block_declared_nodes(res, st, sl$block))
    }
    out
}

# The registry slots (`.nl_registry_axis_slots()`) the field family walks, read
# off the call state alone: the vector-field axes of every non-copy block that
# the transform registry gives a coordinate (`tag`), each labelled as the grid
# names it.
.joint_field_registry_slots <- function(st) {
    copied <- .joint_copied_blocks(st)
    out <- list()
    for (sl in .nl_registry_axis_slots(st$prior)) {
        if (sl$block %in% copied) next
        tag <- tryCatch(.joint_pareto_block_tags(sl$type, sl$axis,
                                                 st$prior[[sl$block]]),
                        error = function(e) NA_character_)
        if (length(tag) != 1L || is.na(tag)) next
        sl$tag <- tag
        sl$label <- .axis_label(sl$block, sl$axis)
        out[[length(out) + 1L]] <- sl
    }
    out
}

# The blocks of a multi-block prior a copy targets, 1-based.
.joint_copied_blocks <- function(st) {
    if (!is.null(st$cp) && isTRUE(st$cp$has_copy))
        as.integer(st$cp$copy_blocks_zero) + 1L else integer(0)
}

# The nodes each vector-field axis of block `b` was declared on in `res`, named
# by axis: what a placement re-crosses an axis it does not move with.
.joint_block_declared_nodes <- function(res, st, b) {
    out <- list()
    for (sl in .nl_registry_axis_slots(st$prior[b])) {
        out[[sl$axis]] <- .joint_declared_axis_ref(.axis_label(b, sl$axis))(res)
    }
    out
}

# The call state with one registry field axis laid on `nodes`. The block's
# vector fields are paired cells (the i-th entries of each form one cell), so
# the block is re-crossed as the tensor of every axis's levels: the placed one
# on its new nodes, the rest on the levels the block's field carries, or, for a
# field the block leaves to its default, on the nodes the fit declared it on.
# A field the caller did not state is written marked (`auto_grid()`): a default
# the engine placed, which refinement may follow past its ends.
.joint_field_write <- function(st, s, nodes, auto = logical(0)) {
    b <- s$block
    blk <- st$prior[[b]]
    bs <- .nl_registry_axis_slots(list(blk))
    lev <- lapply(bs, function(x) {
        if (identical(x$axis, s$axis)) return(sort(unique(as.numeric(nodes))))
        g <- .nl_strip_auto(blk[[x$field]])
        if (!is.null(g) && !is.null(x$column)) g <- as.matrix(g)[, x$column]
        if (!is.null(g)) sort(unique(signif(as.numeric(g), 12L)))
        else s$cross[[x$axis]]
    })
    if (any(vapply(lev, function(v) !length(v), logical(1)))) return(st)
    gr <- expand.grid(lev, KEEP.OUT.ATTRS = FALSE)
    out <- .nl_registry_write_slots(list(blk), bs, unname(as.list(gr)))[[1L]]
    for (x in bs) {
        if (is.null(.nl_axis_hold(blk, x$field, .nl_auto_fields_at(auto, b),
                                  type = x$type))) {
            out[[x$field]] <- auto_grid(out[[x$field]])
        }
    }
    st$prior[[b]] <- out
    st
}

# An axis's incoming nodes on a fit: the ones its kept cells carry together
# with the ones it was declared on, so an axis whose kept cells hold a single
# node still reports the span it was laid over.
.joint_declared_axis_ref <- function(label) {
    force(label)
    function(r) sort(unique(c(.nl_rescue_axis_nodes(r, label),
                              as.numeric(r$outer_axis_nodes[[label]]))))
}

# --- the copy scale ----------------------------------------------------------
#
# The copy scale is an atom at `alpha = 0` -- the "no coupling" model, carrying
# its own declared prior mass -- beside a log continuum. Placement moves the
# continuum and keeps the atom: the new axis is the zero level, when the grid
# had one, plus the continuum laid at the mode. It is written back MARKED
# (`auto_grid()`), which is what the axis was: a default the engine placed,
# which refinement may follow past its ends (`.joint_multi_axis_is_stated()`).
#
# The axis is a copy spec's `alpha_grid` and the fit calls it `b<k>.alpha`, for
# the copy block `k` the spec targets.

.joint_alpha_slots <- function(res, st) {
    cn <- colnames(res$theta_grid) %||% character(0)
    out <- lapply(.joint_copied_blocks(st), function(b) {
        lbl <- .axis_label(b, "alpha")
        if (!lbl %in% cn) return(NULL)
        list(label = lbl, axis = "alpha", block = b,
             ref = function(r) .nl_rescue_axis_nodes(r, lbl))
    })
    Filter(Negate(is.null), out)
}

# The copy spec whose target block is `b`, as an index into the spec list, and
# the spec list itself: `copy` is one spec or a list of them.
.joint_copy_specs <- function(copy) {
    if (is.list(copy) && !is.null(copy$arm)) list(copy) else copy
}
.joint_copy_spec_for_block <- function(copy, b) {
    specs <- .joint_copy_specs(copy)
    for (i in seq_along(specs)) {
        if (isTRUE(as.integer(specs[[i]]$block) == as.integer(b))) return(i)
    }
    NA_integer_
}

# The nodes the caller declared for the copy scale, NULL when it declared none
# (the engine then lays its own default axis at the declared resolution).
.joint_alpha_declared <- function(st, s) {
    i <- .joint_copy_spec_for_block(st$copy, s$block)
    if (is.na(i)) return(NULL)
    .joint_copy_specs(st$copy)[[i]]$alpha_grid
}

.joint_alpha_write <- function(st, s, nodes) {
    grid  <- auto_grid(c(if (isTRUE(s$atom)) 0, nodes))
    one   <- is.list(st$copy) && !is.null(st$copy$arm)
    specs <- .joint_copy_specs(st$copy)
    i <- .joint_copy_spec_for_block(st$copy, s$block)
    specs[[i]]$alpha_grid <- grid
    specs[[i]]$alpha_n    <- NULL
    st$copy <- if (one) specs[[1L]] else specs
    st
}

# The axes the pass may move beyond a copied field's SD, named as the grid
# names them, read off the call arguments before any fit: the mode-find's own
# trigger (`.joint_attach_placement()`) is handed them, so it runs on a grid
# that concentrated without railing. They are the dispersions, the copy scales,
# a copied field's SD, every free-covariance coordinate, every registry axis of
# a non-copy block (`.joint_movable_field_axes()`, its SD or precision among
# them), and every copied field's correlation, since each of those fires on an
# unresolved posterior as well as on a rail.
.joint_movable_extra_axes <- function(st, phi_auto, auto = logical(0)) {
    phi <- paste0("phi_", intersect(names(which(phi_auto)),
                                    names(st$phi_grid) %||% character(0)))
    alpha <- character(0)
    scale <- character(0)
    copied <- .joint_copied_blocks(st)
    for (b in copied) {
        if (is.null(.nl_axis_hold(
                list(alpha_grid = .joint_alpha_declared(st, list(block = b))),
                "alpha_grid", type = ".copy"))) {
            alpha <- c(alpha, .axis_label(b, "alpha"))
        }
    }
    for (b in seq_along(st$prior)) {
        blk <- st$prior[[b]]
        if (b %in% copied &&
            is.null(.nl_axis_hold(blk, "sigma_grid", .nl_auto_fields_at(auto, b),
                                  type = ".copy"))) {
            scale <- c(scale, .axis_label(b, "sigma"))
        }
        type <- tolower(blk$type %||% "")
        if (type %in% c("mcar", "miid") &&
            is.null(.nl_axis_hold(blk, "logchol_grid",
                                  .nl_auto_fields_at(auto, b), type = type))) {
            scale <- c(scale, .axis_label(b,
                                     .mcar_logchol_names(as.integer(blk$n_fields))))
        }
    }
    unique(c(phi, alpha, scale, .joint_movable_mixing_axes(st, auto),
             .joint_movable_field_axes(st, auto)))
}

# The registry field axes the pass may move (`.joint_field_registry_slots()`),
# named as the grid names them: every vector-field axis of a non-copy block,
# unless the caller stated its nodes.
.joint_movable_field_axes <- function(st, auto = logical(0)) {
    out <- character(0)
    for (sl in .joint_field_registry_slots(st)) {
        if (is.null(.nl_slot_hold(st$prior, sl,
                                  .nl_auto_fields_at(auto, sl$block)))) {
            out <- c(out, sl$label)
        }
    }
    out
}

# The (0, 1) correlations the pass may move, named as the grid names them:
# `b<k>.rho` for every BYM2 block and `b<k>.rho_car` for a copied proper-CAR
# block, each unless the caller stated its nodes.
.joint_movable_mixing_axes <- function(st, auto = logical(0)) {
    axis_of <- function(blk, copied) {
        type <- tolower(blk$type %||% "")
        if (identical(type, "car_proper") && !copied) return(NA_character_)
        unname(.JOINT_MIXING_AXES[type])
    }
    movable <- function(blk, b, a) {
        !is.na(a) && is.null(.nl_axis_hold(blk, paste0(a, "_grid"),
                                           .nl_auto_fields_at(auto, b),
                                           type = tolower(blk$type)))
    }
    copied <- .joint_copied_blocks(st)
    out <- character(0)
    for (b in seq_along(st$prior)) {
        a <- axis_of(st$prior[[b]], b %in% copied)
        if (movable(st$prior[[b]], b, a)) out <- c(out, .axis_label(b, a))
    }
    out
}

# --- the pass ----------------------------------------------------------------

# Every family's slots, in table order. An axis belongs to the first family
# that lists it, so a family that walks a whole binding table (`field`) leaves
# the axes an earlier family places to that family.
.joint_placement_slots <- function(res, st, families) {
    out <- list()
    seen <- character(0)
    for (f in names(families)) {
        for (s in families[[f]]$slots(res, st)) {
            if (s$label %in% seen) next
            seen <- c(seen, s$label)
            s$family <- f
            out[[length(out) + 1L]] <- s
        }
    }
    out
}

# Which of `slots` fire on `res`, each family's `together` applied: one firing
# axis of such a family fires every axis of it. `skip` marks a slot read as
# not firing whatever its family says.
.joint_placement_fire <- function(res, slots, families,
                                  skip = logical(length(slots))) {
    fam_of <- vapply(slots, `[[`, character(1), "family")
    fire <- vapply(seq_along(slots), function(i) !skip[[i]] &&
        isTRUE(families[[fam_of[i]]]$fires(res, slots[[i]])), logical(1))
    for (f in unique(fam_of)) {
        idx <- fam_of == f
        if (isTRUE(families[[f]]$together) && any(fire[idx])) fire[idx] <- TRUE
    }
    fire
}

# Would the placement pass move an axis of `res` at call state `st`: does a
# slot the caller left movable fire? Read before a fit refines its grid, so a
# fit that will be laid again skips the refinement its refit replaces
# (`tulpa.nl_defer_refine`).
.joint_placement_moves <- function(res, st, families) {
    slots <- .joint_placement_slots(res, st, families)
    if (!length(slots)) return(FALSE)
    movable <- vapply(slots, function(s)
        is.null(families[[s$family]]$hold(s, st)), logical(1))
    any(.joint_placement_fire(res, slots, families) & movable)
}

# `refit(st, from)` reruns the fit at the call state `st` and returns its
# result (carrying its own placement mode when it still needs one); `from` is
# the fit the placement was read off. Returns `list(res =, st =)`: the
# possibly-refit result and the state that produced it, so a caller refining
# further continues from the placed axes.
.joint_place_axes <- function(res, st, refit, families, enabled = TRUE,
                              max_attempts = .nl_recenter("max_attempts_joint")) {
    out   <- list(res = res, st = st)
    slots <- .joint_placement_slots(res, st, families)
    if (!length(slots)) return(out)
    labels <- vapply(slots, `[[`, character(1), "label")
    fam_of <- vapply(slots, `[[`, character(1), "family")

    if (!isTRUE(enabled)) {
        r <- res
        for (lbl in labels) r <- .nl_decline_axis(r, lbl, "auto_recenter_disabled")
        out$res <- .nl_decline_recenter(r, "auto_recenter_disabled")
        return(out)
    }

    held    <- lapply(slots, function(s) families[[s$family]]$hold(s, st))
    movable <- vapply(held, is.null, logical(1))
    # Whether each copy-scale axis carried its "no coupling" level is read off
    # the grid the pass started from; it rides the slot to the write.
    for (i in which(fam_of == "alpha")) {
        slots[[i]]$atom <- any(slots[[i]]$ref(res) <= 0)
    }
    prior_pinned <- .nl_prior_sigma_is_pinned(st$prior_sigma)

    attempt  <- 0L
    placed   <- character(0)
    failed   <- stats::setNames(rep(NA_character_, length(slots)), labels)
    fired    <- stats::setNames(logical(length(slots)), labels)
    clamp    <- character(0)
    used     <- numeric(0)
    raw      <- numeric(0)
    escalated_on <- 0L
    exhausted <- FALSE
    # An axis whose last placement was laid at the SD floor is as fine as a
    # placement will lay it; what is left of its resolution is the consistency
    # pass's to take, at the mode (`.hyper_propose_at_mode()`). It fires again
    # only on a rail, which says the mode left the span it was laid on.
    floored <- stats::setNames(logical(length(slots)), labels)
    held_floor <- function() vapply(seq_along(slots), function(i)
        floored[[i]] && !.nl_axis_railed(out$res, slots[[i]]$label), logical(1))
    fires_now <- function(i) {
        s <- slots[[i]]
        if (floored[[i]] && !.nl_axis_railed(out$res, s$label)) return(FALSE)
        isTRUE(families[[s$family]]$fires(out$res, s))
    }
    repeat {
        fire <- .joint_placement_fire(out$res, slots, families, skip = held_floor())
        fired <- fired | fire
        move <- fire & movable
        if (!any(move)) break
        if (attempt >= max_attempts) {
            exhausted <- TRUE
            break
        }
        n_b <- .nl_fit_n_blocks(out$res)
        laid <- list()
        for (i in which(move)) {
            s  <- slots[[i]]
            rc <- .nl_axis_recenter_from_fit_full(
                out$res$outer_mode_u, out$res$outer_mode_cov_u,
                out$res$outer_mode_axis_tags, out$res$outer_mode_axis_names,
                s$axis, block_index = s$block, n_blocks = n_b,
                ref_nodes = s$ref(out$res))
            # The RAW SD is recorded even for an axis this attempt could not
            # place: that reading is what says whether declining was right.
            if (!is.null(rc$sd_raw)) raw[s$label] <- rc$sd_raw
            if (!is.null(rc$sd_clamp)) clamp[s$label] <- rc$sd_clamp
            if (is.null(rc$nodes)) {
                failed[i] <- rc$reason
                next
            }
            used[s$label] <- rc$sd_used
            floored[i] <- identical(rc$sd_clamp, "floor")
            laid[[as.character(i)]] <- rc$nodes
        }
        if (!length(laid)) break

        attempt <- attempt + 1L
        nxt <- out$st
        moved_now <- as.integer(names(laid))
        for (i in moved_now) {
            nxt <- families[[fam_of[i]]]$write(nxt, slots[[i]], laid[[as.character(i)]])
            failed[i] <- NA_character_
        }
        escalating <- any(vapply(fam_of[moved_now], function(f)
            isTRUE(families[[f]]$escalate), logical(1)))
        if (escalating) {
            nxt$prior_sigma <- .nl_strip_auto(nxt$prior_sigma)
            if (attempt >= 2L && !prior_pinned) {
                nxt$prior_sigma <- .nl_recenter("sigma_pc_prior")
            }
            escalated_on <- attempt
        }
        placed <- union(placed, labels[moved_now])
        new_res <- refit(nxt, out$res)
        new_res$outer_grid_placement         <- "auto_recentered"
        new_res$outer_grid_recenter_attempts <- attempt
        new_res$outer_grid_recenter_axes     <- labels[labels %in% placed]
        new_res$outer_grid_recenter_sd_clamp <- clamp
        new_res$outer_grid_recenter_sd_used  <- used
        new_res$outer_grid_recenter_sd_raw   <- raw
        if (escalated_on > 0L) {
            new_res$outer_grid_prior_added    <- escalated_on >= 2L && !prior_pinned
            new_res$outer_grid_prior_declined <-
                if (escalated_on >= 2L && prior_pinned) "prior_pinned" else NULL
        }
        out <- list(res = new_res, st = nxt)
    }

    # Per axis: a placed axis speaks only when it still rails; every other axis
    # says why it stayed.
    for (i in seq_along(slots)) {
        s <- slots[[i]]
        idle <- families[[s$family]]$idle
        still <- fires_now(i)
        why <- if (s$label %in% placed) {
            if (!.nl_axis_railed(out$res, s$label)) NULL
            else if (attempt >= max_attempts) "attempts_exhausted"
            else if (is.na(failed[[i]])) idle
            else failed[[i]]
        } else if (!movable[i]) {
            held[[i]]
        } else if (!is.na(failed[[i]])) {
            failed[[i]]
        } else if (exhausted && still) {
            "attempts_exhausted"
        } else {
            idle
        }
        if (!is.null(why)) out$res <- .nl_decline_axis(out$res, s$label, why)
    }
    # For the fit, one reason per family in table order, reduced by
    # `.nl_decline_recenter()` (a no-op once any axis was placed).
    for (f in unique(fam_of)) {
        idx <- which(fam_of == f)
        why <- if (all(!movable[idx])) {
            .nl_reduce_decline(held[idx])
        } else if (any(!is.na(failed[idx]))) {
            .nl_reduce_failed(failed[idx])
        } else if (any(fired[idx] & !movable[idx])) {
            .nl_reduce_decline(held[idx][fired[idx] & !movable[idx]])
        } else {
            families[[f]]$idle
        }
        out$res <- .nl_decline_recenter(out$res, why)
    }
    out
}

# One reason for a family whose placements could not be built: the reason, when
# they agree, and `"no_usable_curvature"` when they do not.
.nl_reduce_failed <- function(failed) {
    u <- unique(stats::na.omit(unlist(failed, use.names = FALSE)))
    if (length(u) == 1L) u else "no_usable_curvature"
}
