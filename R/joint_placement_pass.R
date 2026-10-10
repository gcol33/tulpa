# The joint driver's outer-grid placement pass.
#
# Every axis `tulpa_nested_laplace_joint()` may move -- the single-block field
# SD, each multi-block copy block's field SD, the per-arm dispersions and the
# copy scale -- is laid from the ONE outer mode the detecting fit's mode-find
# reached (`.joint_attach_placement()`), and the grid is refit once per attempt
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
    field_sd <- function(fires, hold, slots, write) list(
        slots = slots, hold = hold, fires = fires, write = write,
        together = FALSE, escalate = TRUE, idle = "grid_not_collapsed")
    list(
        sigma = field_sd(
            slots = function(res, st) {
                type <- tolower(st$prior$type %||% "")
                if (.is_multi_block_prior(st$prior) ||
                    !type %in% c("bym2", "icar", "car_proper")) return(list())
                list(list(label = "sigma", axis = "sigma", block = NULL,
                          ref = function(r) .nl_axis_ref_nodes(r, "sigma")))
            },
            hold = function(s, st)
                .nl_axis_hold(st$prior, "sigma_grid", .nl_auto_fields_at(auto),
                              type = ".joint_areal"),
            fires = function(res, s) .nl_axis_railed(res, "sigma"),
            write = function(st, s, nodes) {
                st$prior$sigma_grid <- nodes
                st
            }),
        copy_sigma = field_sd(
            slots = function(res, st) {
                if (!.is_multi_block_prior(st$prior) || is.null(st$cp) ||
                    !isTRUE(st$cp$has_copy)) return(list())
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
                st$prior[[s$block]]$sigma_grid <- nodes
                st
            }),
        block_scale = list(
            slots = function(res, st) .joint_block_scale_slots(res, st),
            hold = function(s, st)
                .nl_axis_hold(st$prior[[s$block]], s$field,
                              .nl_auto_fields_at(auto, s$block),
                              type = tolower(st$prior[[s$block]]$type %||% "")),
            fires = function(res, s) .nl_axis_placement_fires(res, s$label, "log"),
            write = function(st, s, nodes) {
                st$prior[[s$block]][[s$field]] <- nodes
                st
            },
            together = FALSE, escalate = FALSE, idle = "grid_resolves_posterior"),
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
            together = FALSE, escalate = FALSE, idle = "grid_resolves_posterior")
    )
}

# --- a non-copy block's own scale --------------------------------------------
#
# A multi-block prior block that carries no copy integrates its own scale on the
# axis its registry path reads: an iid random effect's SD (`b<k>.sigma`), an
# areal field's precision (`b<k>.tau`, an arm-specific field the copy does not
# reach), a temporal block's precision. Each is laid from the same outer mode as
# the copied field's SD, in its own log coordinate, and written back onto the
# block's own grid field. A block that is copied is the `copy_sigma` family's.
.JOINT_BLOCK_SCALE_AXES <- c(sigma = "sigma_grid", tau = "tau_grid")

.joint_block_scale_slots <- function(res, st) {
    if (!.is_multi_block_prior(st$prior)) return(list())
    cn <- colnames(res$theta_grid) %||% character(0)
    copied <- if (!is.null(st$cp) && isTRUE(st$cp$has_copy))
        as.integer(st$cp$copy_blocks_zero) + 1L else integer(0)
    out <- list()
    for (b in setdiff(seq_along(st$prior), copied)) {
        for (a in names(.JOINT_BLOCK_SCALE_AXES)) {
            lbl <- paste0("b", b, ".", a)
            if (!lbl %in% cn) next
            out[[length(out) + 1L]] <- list(
                label = lbl, axis = a, block = b,
                field = .JOINT_BLOCK_SCALE_AXES[[a]],
                ref = .joint_declared_axis_ref(lbl))
        }
    }
    out
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
# which refinement may follow past its ends (`.joint_axis_is_stated()`).
#
# Single-block, the axis is the copy arm's `field_coef` and the fit calls it
# `alpha`; multi-block it is a copy spec's `alpha_grid` and the fit calls it
# `b<k>.alpha`, for the copy block `k` the spec targets.

.joint_alpha_slots <- function(res, st) {
    cn <- colnames(res$theta_grid) %||% character(0)
    if (.is_multi_block_prior(st$prior)) {
        if (is.null(st$cp) || !isTRUE(st$cp$has_copy)) return(list())
        n_b <- .nl_fit_n_blocks(res)
        out <- lapply(st$cp$copy_blocks_zero + 1L, function(b) {
            lbl <- .nl_axis_alias("alpha", b, n_b)[1L]
            if (!lbl %in% cn) return(NULL)
            list(label = lbl, axis = "alpha", block = b,
                 ref = function(r) .nl_rescue_axis_nodes(r, lbl))
        })
        return(Filter(Negate(is.null), out))
    }
    if (!"alpha" %in% cn) return(list())
    k <- .joint_copy_arm_index(st$responses)
    if (is.na(k)) return(list())
    list(list(label = "alpha", axis = "alpha", block = NULL, arm = k,
              ref = function(r) .nl_rescue_axis_nodes(r, "alpha")))
}

# The single-block arm that declares the copy scale as an axis.
.joint_copy_arm_index <- function(responses) {
    for (k in seq_along(responses)) {
        fc <- responses[[k]]$field_coef
        if (is.list(fc) && !is.null(fc$name)) return(k)
        if (is.character(fc) && length(fc) == 1L) return(k)
    }
    NA_integer_
}

# The copy spec (multi-block) whose target block is `b`, as an index into the
# spec list, and the spec list itself: `copy` is one spec or a list of them.
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
    if (is.null(s$block)) {
        fc <- st$responses[[s$arm]]$field_coef
        return(if (is.list(fc)) fc$grid else NULL)
    }
    i <- .joint_copy_spec_for_block(st$copy, s$block)
    if (is.na(i)) return(NULL)
    .joint_copy_specs(st$copy)[[i]]$alpha_grid
}

.joint_alpha_write <- function(st, s, nodes) {
    grid <- auto_grid(c(if (isTRUE(s$atom)) 0, nodes))
    if (is.null(s$block)) {
        fc <- st$responses[[s$arm]]$field_coef
        if (!is.list(fc)) fc <- list(name = as.character(fc))
        fc$grid <- grid
        fc[["n"]] <- NULL
        st$responses[[s$arm]]$field_coef <- fc
        return(st)
    }
    one   <- is.list(st$copy) && !is.null(st$copy$arm)
    specs <- .joint_copy_specs(st$copy)
    i <- .joint_copy_spec_for_block(st$copy, s$block)
    specs[[i]]$alpha_grid <- grid
    specs[[i]]$alpha_n    <- NULL
    st$copy <- if (one) specs[[1L]] else specs
    st
}

# The axes beyond the single-block field SD the pass may move, named as the
# grid names them, read off the call arguments before any fit: the mode-find's
# own trigger (`.joint_attach_placement()`) is handed them, so it runs on a grid
# that concentrated without railing. They are the dispersions, the copy scales,
# and on a multi-block prior every block's own scale -- a copied field's SD as
# much as a non-copy block's SD or precision -- since each of those fires on an
# unresolved posterior as well as on a rail.
.joint_movable_extra_axes <- function(st, phi_auto, auto = logical(0)) {
    phi <- paste0("phi_", intersect(names(which(phi_auto)),
                                    names(st$phi_grid) %||% character(0)))
    movable <- function(s) is.null(.nl_axis_hold(
        list(alpha_grid = .joint_alpha_declared(st, s)), "alpha_grid",
        type = ".copy"))
    alpha <- character(0)
    scale <- character(0)
    if (.is_multi_block_prior(st$prior)) {
        copied <- integer(0)
        if (isTRUE(st$cp$has_copy)) {
            copied <- as.integer(st$cp$copy_blocks_zero) + 1L
            for (b in copied) {
                if (movable(list(block = b))) {
                    alpha <- c(alpha, .nl_axis_alias("alpha", b,
                                                     length(st$prior))[1L])
                }
            }
        }
        for (b in seq_along(st$prior)) {
            blk <- st$prior[[b]]
            fields <- if (b %in% copied) c(sigma = "sigma_grid")
                      else .JOINT_BLOCK_SCALE_AXES
            for (a in names(fields)) {
                hold <- .nl_axis_hold(
                    blk, fields[[a]], .nl_auto_fields_at(auto, b),
                    type = if (b %in% copied) ".copy"
                           else tolower(blk$type %||% ""))
                if (is.null(hold)) scale <- c(scale, paste0("b", b, ".", a))
            }
        }
    } else {
        k <- .joint_copy_arm_index(st$responses)
        if (!is.na(k) && movable(list(block = NULL, arm = k))) alpha <- "alpha"
    }
    c(phi, alpha, scale)
}

# --- the pass ----------------------------------------------------------------

.joint_placement_slots <- function(res, st, families) {
    out <- list()
    for (f in names(families)) {
        for (s in families[[f]]$slots(res, st)) {
            s$family <- f
            out[[length(out) + 1L]] <- s
        }
    }
    out
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
    fires_now <- function(i) {
        s <- slots[[i]]
        if (floored[[i]] && !.nl_axis_railed(out$res, s$label)) return(FALSE)
        isTRUE(families[[s$family]]$fires(out$res, s))
    }
    repeat {
        fire <- vapply(seq_along(slots), fires_now, logical(1))
        for (f in unique(fam_of)) {
            idx <- fam_of == f
            if (isTRUE(families[[f]]$together) && any(fire[idx])) fire[idx] <- TRUE
        }
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
