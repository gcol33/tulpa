# Outer-grid placement mode.
#
# A placement rescue -- the joint driver's `.joint_sigma_grid_rescue()`,
# `.joint_multi_sigma_grid_rescue()` and `.joint_phi_grid_rescue()`, and the
# standalone `.nl_registry_grid_rescue()` (R/nested_laplace_auto_grid.R) --
# lays a railed or under-resolved axis at `mode +/- span * sd`. What it needs
# from the fit it detected on is the outer MODE and the curvature there: a
# property of the log-posterior surface, which the grid it detected on cannot
# supply on exactly the fits that need placing. A railed grid's argmax is its
# boundary node, where the mode is not; an axis whose weight collapsed onto one
# node has no spread to read.
#
# Reading the argmax as the mode and a moment of the weights as the spread moved
# a railed axis by at most `span` floored SDs per attempt, and a weighted
# variance of zero declined the placement outright whenever some OTHER axis
# still carried spread, since the finite-difference curvature was only computed
# when no axis did. On the full 25 km occu_cover fit of gcol33/tulpa#919 the
# first left the field SD on a recentred axis's edge after both attempts, and
# the second left it on the 200-cell tensor's ceiling.
#
# So the mode is FOUND: the damped-Newton outer mode-find the CCD integrator
# places its design with (`.joint_ccd_modefind()`), started at the grid's
# heaviest cell and run over every axis the grid lays more than one node on,
# reading the log-posterior through full inner solves at the fit's own
# tolerances. The curvature at the mode it reaches is the placement spread.
#
# The search runs in the coordinate each axis is laid in -- a positive scale in
# log, a mixing weight in logit, the copy scale on its log continuum
# (`.joint_ccd_coord_tags()`) -- and maximises the grid's own log-density, the
# one the rail test reads (`.nl_axis_rail()`, `measure = "inner"`), carrying no
# change-of-variables term. The axis a placement lays therefore contains the
# mode that test looks for; with the logit Jacobian a mixing weight would be
# centred on a different density and read as railed again at its refit. A copy
# scale whose heaviest cell is its "no coupling" point mass is held there: the
# continuum a Newton step moves along does not reach it.

# Search box and step, per axis, in the placement coordinate: the axis's own
# node range, padded the way the CCD pads its box (so a mode past a coarse
# default axis is reachable), and a per-round trust radius of about that range.
.nl_placement_box <- function(u_vals) {
    lower0 <- vapply(u_vals, min, numeric(1))
    upper0 <- vapply(u_vals, max, numeric(1))
    span   <- pmax(upper0 - lower0, 1e-3)
    pad    <- pmax(1.5 * span, 0.5)
    list(lower = lower0 - pad, upper = upper0 + pad, span = span,
         trust = pmax(span, 0.5))
}

# The outer mode of a grid's log-posterior surface. `tg` is the outer grid as a
# named `[n_cells x d]` matrix, `w` its integration weights, `tags` one
# transform tag per column (`.joint_pareto_fwd()`'s vocabulary).
# `eval_logpost(theta_mat)` maps a physical `[S x d]` matrix (columns named as
# `tg`'s) to the outer log-posterior at the fit's own inner solver settings,
# carrying each row's inner mode as a `"modes"` attribute where the kernel
# returns them; `set_warm(mode)` advances the inner warm start to an accepted
# point's mode.
#
# Returns `list(mode_u, cov_u, tags, names, status, rounds, evals, value)`, the
# mode and covariance over every axis in the placement coordinate (an axis held
# fixed has zero rows and columns), or `list(declined = <reason>)`. `status` is
# `"ok"`, `"not_converged"` (the round cap was reached first), or `"boundary"`
# (the mode ran to the search box, which is past the axis's own range by 1.5
# spans: a mode the data do not bound).
.nl_placement_mode <- function(tg, w, tags, eval_logpost, set_warm = NULL) {
    if (is.null(tg) || !is.matrix(tg) || is.null(w) || length(w) != nrow(tg) ||
        length(tags) != ncol(tg) || anyNA(tags) ||
        !any(is.finite(w) & w > 0)) {
        return(list(declined = "no_usable_curvature"))
    }
    cn <- colnames(tg)
    d  <- ncol(tg)
    w[!is.finite(w)] <- 0
    x0 <- as.numeric(tg[which.max(w), ])
    ptag <- .joint_ccd_coord_tags(cn, tags)

    cont <- lapply(seq_len(d), function(j) {
        v <- sort(unique(as.numeric(tg[, j])))
        if (identical(ptag[j], "log")) v[v > 0] else v
    })
    on_cont <- vapply(seq_len(d), function(j)
        !identical(ptag[j], "log") || x0[j] > 0, logical(1))
    vary <- which(lengths(cont) >= 2L & on_cont)
    if (!length(vary)) return(list(declined = "no_usable_curvature"))

    u_vals <- lapply(vary, function(j) .joint_pareto_fwd(ptag[j], cont[[j]]))
    if (any(!is.finite(unlist(u_vals)))) {
        return(list(declined = "no_usable_curvature"))
    }
    box <- .nl_placement_box(u_vals)
    u0  <- vapply(seq_along(vary), function(i)
        .joint_pareto_fwd(ptag[vary[i]], x0[vary[i]]), numeric(1))

    meter <- .ccd_meter_new(Inf, label = "outer placement")
    eval1 <- function(U) {
        .ccd_meter_spend(meter, nrow(U))
        theta <- matrix(x0, nrow(U), d, byrow = TRUE,
                        dimnames = list(NULL, cn))
        for (i in seq_along(vary)) {
            theta[, vary[i]] <- .joint_pareto_inv(ptag[vary[i]], U[, i])$theta
        }
        out <- eval_logpost(theta)
        lp  <- as.numeric(out)
        lp[!is.finite(lp)] <- -1e10
        md <- attr(out, "modes")
        if (!is.null(md)) attr(lp, "modes") <- md
        lp
    }
    find <- function(h) .joint_ccd_modefind(
        u0, eval1, box$lower, box$upper, h, trust = box$trust,
        on_accept = set_warm, meter = meter, ridge_check = FALSE,
        max_rounds = .ccd_placement("max_rounds"),
        max_halve = .ccd_placement("max_halve"))
    mf <- find(rep(0.1, length(vary)))
    if (!identical(mf$status, "ok")) {
        h_cal <- .joint_ccd_calibrate_step(u0, eval1, box$span, meter = meter)
        mf <- find(h_cal)
    }
    if (!identical(mf$status, "ok") || is.null(mf$hess) ||
        any(!is.finite(mf$hess))) {
        return(list(declined = "no_usable_curvature"))
    }
    cov_v <- tryCatch(solve(-.joint_ccd_neg_def(mf$hess)),
                      error = function(e) NULL)
    if (is.null(cov_v) || any(!is.finite(cov_v))) {
        return(list(declined = "no_usable_curvature"))
    }

    out_tag <- tags
    out_tag[vary] <- ptag[vary]
    mode_u <- vapply(seq_len(d), function(j)
        .joint_pareto_fwd(out_tag[j], x0[j]), numeric(1))
    mode_u[vary] <- mf$par
    cov_u <- matrix(0, d, d)
    cov_u[vary, vary] <- (cov_v + t(cov_v)) / 2

    eps <- 1e-3 * pmax(box$upper - box$lower, 1)
    at_box <- abs(mf$par - box$lower) < eps | abs(mf$par - box$upper) < eps
    status <- if (any(at_box)) "boundary" else
        if (isTRUE(mf$converged)) "ok" else "not_converged"
    list(mode_u = mode_u, cov_u = cov_u, tags = out_tag, names = cn,
         status = status, rounds = as.integer(meter$rounds),
         evals = as.numeric(meter$evals), value = mf$value)
}

# The joint fit's placement mode: its grid, weights and per-axis tags
# (`.joint_pareto_axis_tags()`, which declines on an axis whose support the
# engine will not guess) handed to `.nl_placement_mode()`.
.joint_placement_mode <- function(res, eval_logpost, set_warm = NULL) {
    tags <- .joint_pareto_axis_tags(res)
    if (.k_is_decline(tags)) return(list(declined = .k_decline_label(tags)))
    .nl_placement_mode(res$theta_grid, res$weights, tags, eval_logpost,
                       set_warm = set_warm)
}

# Should this fit carry a placement mode? The rescues fire on a field SD that
# railed -- the whole grid collapsed onto a boundary cell, or the axis's own
# marginal maximal at an endpoint -- and on a movable dispersion axis that
# rails or does not resolve its own posterior (`extra_axes`); anything else
# never reads one, so it costs nothing.
.joint_placement_wanted <- function(res, extra_axes = character(0)) {
    identical(res$pareto_k_regime, "collapsed_edge") ||
        .nl_sigma_axis_railed(res) ||
        .nl_placement_axis_wanted(res, extra_axes)
}

# Attach the placement mode the rescues read (`outer_mode_*`), or the reason
# there is none (`outer_mode_declined`). A no-op on a fit no rescue would place.
.joint_attach_placement <- function(res, eval_logpost, set_warm = NULL,
                                    extra_axes = character(0)) {
    if (!.joint_placement_wanted(res, extra_axes)) return(res)
    pm <- .joint_placement_mode(res, eval_logpost, set_warm = set_warm)
    if (!is.null(pm$declined)) {
        res$outer_mode_declined <- pm$declined
        return(res)
    }
    res$outer_mode_u          <- pm$mode_u
    res$outer_mode_cov_u      <- pm$cov_u
    res$outer_mode_axis_tags  <- pm$tags
    res$outer_mode_axis_names <- pm$names
    res$outer_mode_status     <- pm$status
    res$outer_mode_rounds     <- pm$rounds
    res$outer_mode_evals      <- pm$evals
    res
}
