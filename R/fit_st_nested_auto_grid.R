# fit_st_nested_auto_grid.R
# ------------------------------------------------------------------------------
# Auto mode-Hessian outer-grid recenter for fit_st_nested() (gcol33/tulpa#291).
#
# fit_st_nested()'s tensor grid (tau_spatial x tau_temporal [x rho]) is a
# fixed default axis, not a hard ceiling -- the SAME "starting axis, not a
# ceiling" contract every other nested-Laplace family's default grid carries
# (gcol33/tulpa#289, #290). Unlike those families, this driver had NO
# mode-find machinery at all to reuse (no CCD, no registry k-hat FD-Hessian,
# no SPDE method), so this builds one from scratch: a derivative-free
# optim(..., hessian = TRUE) over the unconstrained per-axis coordinate,
# mirroring the no-analytic-gradient fallback branch of .re_cov_theta_fit()
# (R/nested_laplace_re_cov.R) -- the closest existing precedent in this
# codebase for a from-scratch outer mode-find.
# ------------------------------------------------------------------------------

# Which axes are free to move: the two precision axes always are; the
# temporal autocorrelation only for ar1 (rw1/rw2 carry no rho axis -- it is
# pinned at 0 and excluded, the same "single grid value -> pinned, not a
# search axis" exclusion `.joint_pareto_grid_vary_axes()` makes elsewhere).
.st_grid_axes <- function(temporal_type) {
    c("tau_spatial", "tau_temporal", if (identical(temporal_type, "ar1")) "rho")
}

# Unconstrained per-axis transform: log for the two precision axes (unbounded
# positive support, matching `.st_log_grid()`'s own log-uniform default
# axis), `qlogis((rho + 1) / 2)` for ar1's temporal autocorrelation (support
# `(-1, 1)`, the model's stationarity bound -- see
# `.nl_apply_ar1_rho_prior()`'s default Uniform(-1, 1) prior).
.st_axis_fwd <- function(axis, x) if (identical(axis, "rho")) stats::qlogis((x + 1) / 2) else log(x)
.st_axis_inv <- function(axis, u) if (identical(axis, "rho")) 2 * stats::plogis(u) - 1 else exp(u)

# --- knob provenance ---------------------------------------------------------
#
# The other three rescues guard on `.nl_axis_is_pinned()`, which asks whether a
# grid VECTOR carries information a pin would add. This driver has no grid
# vector to override -- its axes are BUILT from scalar `control` knobs -- so the
# same question is asked one level down, per knob: a knob absent, marked with
# `auto_grid()`, or set to the engine's own default (`.nl_st_default()`) carries
# no preference, and only anything else is a pin. Presence alone is not
# provenance: a wrapper that exposes its own `n_grid` argument defaulted to the
# engine's value threads that value into `control` on every fit, and reading
# that as an override made the recenter inert for every fit it makes
# (gcol33/tulpa#294, the #293 shape one level down).
#
# Each knob is mapped to the axis or axes it SHAPES, so a pin is per-axis rather
# than all-or-nothing: `tau_lower` / `tau_upper` are the shared bounds of both
# precision axes (`fit_st_nested()` builds `ts_axis` and `tt_axis` from the same
# pair), `n_grid_spatial` / `n_grid_temporal` shape one each, and the three rho
# knobs shape the ar1 autocorrelation axis alone.
.ST_GRID_KNOBS <- list(
    tau_lower       = list(default = "tau_lower",  axes = c("tau_spatial", "tau_temporal")),
    tau_upper       = list(default = "tau_upper",  axes = c("tau_spatial", "tau_temporal")),
    n_grid_spatial  = list(default = "n_spatial",  axes = "tau_spatial"),
    n_grid_temporal = list(default = "n_temporal", axes = "tau_temporal"),
    n_grid_rho      = list(default = "n_rho",      axes = "rho"),
    rho_lower       = list(default = "rho_lower",  axes = "rho"),
    rho_upper       = list(default = "rho_upper",  axes = "rho")
)

# Is one grid knob a PIN? The scalar counterpart of `.nl_axis_is_pinned()`.
.st_knob_is_pinned <- function(control, knob) {
    v <- control[[knob]]
    if (is.null(v)) return(FALSE)
    if (is_auto_grid(v)) return(FALSE)
    if (length(v) != 1L) return(TRUE)
    v <- suppressWarnings(as.numeric(v))
    if (!is.finite(v)) return(TRUE)
    d <- as.numeric(.nl_st_default(.ST_GRID_KNOBS[[knob]]$default))
    !isTRUE(all.equal(v, d))
}

# The grid axes the caller genuinely pinned. An axis absent from this set is
# free to be recentred even when another axis was pinned.
.st_pinned_axes <- function(control) {
    hit <- vapply(names(.ST_GRID_KNOBS),
                  function(nm) .st_knob_is_pinned(control, nm), logical(1))
    as.character(unique(unlist(lapply(.ST_GRID_KNOBS[hit], `[[`, "axes"),
                               use.names = FALSE)))
}

# Set the three (or two) grid-defining kernel args to a single trial cell (or
# an arbitrary multi-row grid), threading car_proper's `rho_spatial_grid`
# along -- it is a per-CELL vector (`rep(rho_spatial, nrow(grid))` in
# `fit_st_nested()`), so it must be resized to match whatever grid replaces
# it or the kernel call errors on a length mismatch.
.st_set_grid_args <- function(kargs, ts, tt, rho, spatial_type, rho_spatial_val) {
    kargs$tau_spatial_grid  <- ts
    kargs$tau_temporal_grid <- tt
    kargs$rho_temporal_grid <- rho
    if (identical(spatial_type, "car_proper")) {
        kargs$rho_spatial_grid <- rep(rho_spatial_val, length(ts))
    }
    kargs
}

# One-cell re-evaluation of the ST kernel's log-marginal at an arbitrary
# (tau_spatial, tau_temporal[, rho]) point, reusing the SAME kernel + fixed
# arguments (`kargs`) the main fit ran -- no probe-specific kernel-call
# machinery. `store_Q` is off (a probe point's precision is never read).
# Returns `-Inf` (never `NA` / an error) on a non-finite or failed solve, so
# the optimizer treats an implausible trial point as arbitrarily unattractive
# rather than aborting.
.st_probe_log_marginal <- function(kernel, kargs, spatial_type, rho_spatial_val,
                                   ts, tt, rho) {
    a <- .st_set_grid_args(kargs, ts, tt, rho, spatial_type, rho_spatial_val)
    a$store_Q <- FALSE
    lm <- tryCatch(do.call(kernel, a)$log_marginal, error = function(e) NA_real_)
    if (length(lm) != 1L || !is.finite(lm)) -Inf else lm
}

# Mode-Hessian recenter-and-refit for a collapsed spatiotemporal outer grid.
# `out` is the just-completed fit (already carrying `pareto_k_regime` via
# `.joint_attach_pareto_k_regime()`); `kernel` / `kargs` reproduce the
# ORIGINAL `do.call(kernel, kargs)` call `fit_st_nested()` made, so a probe or
# a refit reuses every fixed argument (y, X, family, spatial/temporal
# structure, ...) unchanged. `n_gs` / `n_gt` / `n_grho` are the ALREADY
# RESOLVED per-axis node counts, so the recentered grid keeps the same
# density per axis, just a different span.
#
# One attempt only (unlike the joint path's two): `fit_st_nested()` is a
# standalone driver with no donor/copy coupling to push a mode toward
# near-separation, the same reasoning `.nl_registry_grid_rescue()` gives for
# its own single attempt. Seeds the search at the collapsed grid's own
# highest-weight cell (already close to the true mode -- that is exactly what
# "collapsed onto a boundary" means), optimizes the box-constrained objective
# with L-BFGS-B (no analytic gradient is available from this compiled kernel;
# see the box-constraint note below) + `hessian = TRUE`, and lays a new
# per-axis grid centred on the mode: log-spaced for the two precision axes (matching
# `.st_log_grid()`'s own convention), natural-spaced for rho (matching the
# original `seq(rho_lower, rho_upper, ...)` convention, just re-centred and
# no longer bounded by the retired default range).
#
# Declines (returns `out` unchanged, carrying the reason in
# `outer_grid_recenter_declined`) when `control$auto_recenter = FALSE`, EVERY
# axis the grid has was pinned by a genuinely non-default knob, the regime never
# collapsed onto an edge, or the mode-find / Hessian is unusable (non-finite,
# non-positive-definite, a degenerate refit) -- the same guard-rather-than-guess
# stance every other family's rescue takes.
#
# A pin is PER AXIS (`.st_pinned_axes()`): a pinned axis keeps its original
# nodes, the rest are recentred, and `outer_grid_pinned_axes` records which were
# held. The joint mode-find still runs over every axis -- the free axes are
# placed at the joint mode's coordinates, not at a mode conditioned on the
# pinned axes' node sets, which are a range rather than a value.
#
# The mode-find is BOX-CONSTRAINED (L-BFGS-B, numerical gradient -- no
# analytic gradient is available from this compiled kernel, and L-BFGS-B
# falls back to a finite-difference one when `gr` is omitted): a precision
# axis with no real evidence against it (the response carries no signal on
# that axis at all) has a log-marginal that FLATTENS rather than turning
# over as tau grows, so an unbounded search runs away along it -- the exact
# failure mode `.re_cov_theta_fit()` brackets its dispersion coordinate
# against (see its `phi_lo` / `phi_hi` comment). `tau_lo` / `tau_hi` are the
# resolved default axis bounds (0.25 / 16); the search box extends 6 nats
# (~2.6 decades) past each side -- generous enough to escape the retired
# ceiling, finite enough that a truly flat direction hits the box rather than
# wandering to a numerically meaningless extreme. rho's own transform already
# saturates its box (`(-1, 1)`), so its bracket is a wide numerical safety
# net rather than a substantive constraint.
.st_auto_grid_rescue <- function(out, kernel, kargs, spatial_type, temporal_type,
                                 n_gs, n_gt, n_grho, tau_lo, tau_hi, control,
                                 rho_spatial_val = 0.9) {
    if (is.character(control$auto_recenter)) {
        stop("control$auto_recenter = \"",
             paste(control$auto_recenter, collapse = "\", \""),
             "\" names a per-axis placement policy implemented on the ",
             "standalone tulpa_nested_laplace() registry path only. Use TRUE ",
             "or FALSE here.", call. = FALSE)
    }
    if (isFALSE(control$auto_recenter)) {
        return(.nl_decline_recenter(out, "auto_recenter_disabled"))
    }
    axes   <- .st_grid_axes(temporal_type)
    pinned <- intersect(.st_pinned_axes(control), axes)
    free   <- setdiff(axes, pinned)
    if (!length(free)) {
        return(.nl_decline_recenter(out, "grid_knobs_overridden"))
    }
    if (!identical(out$pareto_k_regime, "collapsed_edge")) {
        return(.nl_decline_recenter(out, "grid_not_collapsed"))
    }

    tg <- out$theta_grid
    w  <- out$weights
    if (is.null(tg) || is.null(w) || length(w) != nrow(tg) ||
        !all(axes %in% colnames(tg))) {
        return(.nl_decline_recenter(out, "no_usable_curvature"))
    }
    seed_row <- tg[which.max(w), , drop = TRUE]
    u0 <- vapply(axes, function(a) .st_axis_fwd(a, seed_row[[a]]), numeric(1))
    lo <- vapply(axes, function(a) if (identical(a, "rho")) -15 else log(tau_lo) - 6, numeric(1))
    hi <- vapply(axes, function(a) if (identical(a, "rho")) 15 else log(tau_hi) + 6, numeric(1))
    u0 <- pmin(pmax(u0, lo), hi)

    objective <- function(u) {
        vals <- stats::setNames(
            vapply(seq_along(axes), function(j) .st_axis_inv(axes[j], u[j]), numeric(1)),
            axes)
        rho <- if ("rho" %in% axes) vals[["rho"]] else 0.0
        lm  <- .st_probe_log_marginal(kernel, kargs, spatial_type, rho_spatial_val,
                                      vals[["tau_spatial"]], vals[["tau_temporal"]], rho)
        if (!is.finite(lm)) .Machine$double.xmax else -lm
    }

    opt <- tryCatch(
        stats::optim(u0, objective, method = "L-BFGS-B", lower = lo, upper = hi,
                     hessian = TRUE, control = list(maxit = 300L, factr = 1e7)),
        error = function(e) NULL)
    if (is.null(opt) || !all(is.finite(opt$par)) || is.null(opt$hessian) ||
        !all(is.finite(opt$hessian))) {
        return(.nl_decline_recenter(out, "no_usable_curvature"))
    }
    H     <- (opt$hessian + t(opt$hessian)) / 2
    cov_u <- tryCatch(solve(H), error = function(e) NULL)
    if (is.null(cov_u) || any(!is.finite(cov_u)) || any(diag(cov_u) <= 0)) {
        return(.nl_decline_recenter(out, "no_usable_curvature"))
    }
    u_hat <- stats::setNames(opt$par, axes)
    sd_u  <- stats::setNames(sqrt(pmax(diag(cov_u), 0)), axes)

    # `.nl_recenter_log_axis()`'s own SD clamp bounds the SPREAD, not the
    # RANGE: `optim(hessian = TRUE)` finite-differences around `par` without
    # regard to `lower` / `upper`, so a direction the search box caught (no
    # real curvature past it) can still report a small but nonzero FD
    # curvature there, and `u_hat +/- span * sd_u` (span up to 2.5, sd_u up
    # to the 3-nat ceiling) can then reach well past the same box the
    # mode-find was held to. Clamping the recentred axis to that box keeps
    # "escape the ceiling by a sane margin" true of the OUTPUT, not just the
    # search.
    clamp_axis <- function(vals, axis) sort(unique(pmin(pmax(vals, exp(lo[[axis]])), exp(hi[[axis]]))))
    # A pinned axis keeps exactly the nodes the caller's knobs built, read back
    # off the kernel arguments the fit ran with rather than rebuilt from the
    # knobs (one source, and it already carries `.st_log_grid()`'s 2-node floor).
    kept <- function(field) sort(unique(as.numeric(kargs[[field]])))

    if ("tau_spatial" %in% free) {
        new_ts <- .nl_recenter_log_axis(u_hat[["tau_spatial"]], sd_u[["tau_spatial"]],
                                        n_pts = n_gs)
        if (is.null(new_ts)) return(.nl_decline_recenter(out, "no_usable_curvature"))
        new_ts <- clamp_axis(new_ts, "tau_spatial")
        if (length(new_ts) < 2L) return(.nl_decline_recenter(out, "no_usable_curvature"))
    } else {
        new_ts <- kept("tau_spatial_grid")
    }
    if ("tau_temporal" %in% free) {
        new_tt <- .nl_recenter_log_axis(u_hat[["tau_temporal"]], sd_u[["tau_temporal"]],
                                        n_pts = n_gt)
        if (is.null(new_tt)) return(.nl_decline_recenter(out, "no_usable_curvature"))
        new_tt <- clamp_axis(new_tt, "tau_temporal")
        if (length(new_tt) < 2L) return(.nl_decline_recenter(out, "no_usable_curvature"))
    } else {
        new_tt <- kept("tau_temporal_grid")
    }

    if (!("rho" %in% axes)) {
        new_rho <- 0.0
    } else if ("rho" %in% free) {
        span  <- 2.5
        su    <- min(max(sd_u[["rho"]], 0.15), 3)
        u_seq <- seq(u_hat[["rho"]] - span * su, u_hat[["rho"]] + span * su,
                     length.out = n_grho)
        # (-1, 1) is the model's own stationarity bound (see .st_axis_inv()),
        # not the retired rho_lower/rho_upper default -- the whole point of
        # the recenter is to no longer be confined to that default range.
        new_rho <- sort(unique(pmin(pmax(2 * stats::plogis(u_seq) - 1, -0.999), 0.999)))
        if (length(new_rho) < 2L) {
            return(.nl_decline_recenter(out, "no_usable_curvature"))
        }
    } else {
        new_rho <- kept("rho_temporal_grid")
    }

    new_grid <- expand.grid(tau_spatial = new_ts, tau_temporal = new_tt, rho = new_rho)
    a <- .st_set_grid_args(kargs, as.numeric(new_grid$tau_spatial),
                           as.numeric(new_grid$tau_temporal),
                           as.numeric(new_grid$rho), spatial_type, rho_spatial_val)
    a$store_Q <- TRUE
    refit <- tryCatch(do.call(kernel, a), error = function(e) NULL)
    if (is.null(refit) || is.null(refit$log_marginal) ||
        !all(is.finite(refit$log_marginal))) {
        return(.nl_decline_recenter(out, "refit_failed"))
    }

    refit$theta_grid  <- as.matrix(new_grid)
    refit$theta_names <- colnames(new_grid)
    refit$weights <- .nl_normalise_weights_safe(refit$log_marginal, "spatiotemporal grid")
    refit <- .joint_attach_pareto_k_regime(refit)
    refit$outer_grid_placement         <- "auto_recentered"
    refit$outer_grid_recenter_attempts <- 1L
    refit$outer_grid_pinned_axes       <- pinned
    refit
}
