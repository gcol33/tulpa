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

# Has the caller explicitly set any grid-construction knob? An explicit
# choice always wins -- mirrors every other family's "an explicit sigma_grid
# always wins" contract, generalized to the knobs that BUILD this grid since
# fit_st_nested() has no single grid-vector argument to override.
.st_grid_overridden <- function(control) {
    knobs <- c("tau_lower", "tau_upper", "n_grid_spatial", "n_grid_temporal",
              "n_grid_rho", "rho_lower", "rho_upper")
    any(!vapply(knobs, function(nm) is.null(control[[nm]]), logical(1)))
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
# Declines (returns `out` unchanged) when the grid knobs were explicitly
# overridden, the regime never collapsed onto an edge, or the mode-find /
# Hessian is unusable (non-finite, non-positive-definite, a degenerate
# refit) -- the same guard-rather-than-guess stance every other family's
# rescue takes.
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
    if (.st_grid_overridden(control)) return(out)
    if (!identical(out$pareto_k_regime, "collapsed_edge")) return(out)

    axes <- .st_grid_axes(temporal_type)
    tg <- out$theta_grid
    w  <- out$weights
    if (is.null(tg) || is.null(w) || length(w) != nrow(tg) ||
        !all(axes %in% colnames(tg))) {
        return(out)
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
        return(out)
    }
    H     <- (opt$hessian + t(opt$hessian)) / 2
    cov_u <- tryCatch(solve(H), error = function(e) NULL)
    if (is.null(cov_u) || any(!is.finite(cov_u)) || any(diag(cov_u) <= 0)) {
        return(out)
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
    new_ts <- .nl_recenter_log_axis(u_hat[["tau_spatial"]],  sd_u[["tau_spatial"]],  n_pts = n_gs)
    new_tt <- .nl_recenter_log_axis(u_hat[["tau_temporal"]], sd_u[["tau_temporal"]], n_pts = n_gt)
    if (is.null(new_ts) || is.null(new_tt)) return(out)
    new_ts <- clamp_axis(new_ts, "tau_spatial")
    new_tt <- clamp_axis(new_tt, "tau_temporal")
    if (length(new_ts) < 2L || length(new_tt) < 2L) return(out)

    if ("rho" %in% axes) {
        span  <- 2.5
        su    <- min(max(sd_u[["rho"]], 0.15), 3)
        u_seq <- seq(u_hat[["rho"]] - span * su, u_hat[["rho"]] + span * su,
                     length.out = n_grho)
        # (-1, 1) is the model's own stationarity bound (see .st_axis_inv()),
        # not the retired rho_lower/rho_upper default -- the whole point of
        # the recenter is to no longer be confined to that default range.
        new_rho <- sort(unique(pmin(pmax(2 * stats::plogis(u_seq) - 1, -0.999), 0.999)))
        if (length(new_rho) < 2L) return(out)
    } else {
        new_rho <- 0.0
    }

    new_grid <- expand.grid(tau_spatial = new_ts, tau_temporal = new_tt, rho = new_rho)
    a <- .st_set_grid_args(kargs, as.numeric(new_grid$tau_spatial),
                           as.numeric(new_grid$tau_temporal),
                           as.numeric(new_grid$rho), spatial_type, rho_spatial_val)
    a$store_Q <- TRUE
    refit <- tryCatch(do.call(kernel, a), error = function(e) NULL)
    if (is.null(refit) || is.null(refit$log_marginal) ||
        !all(is.finite(refit$log_marginal))) {
        return(out)
    }

    refit$theta_grid  <- as.matrix(new_grid)
    refit$theta_names <- colnames(new_grid)
    refit$weights <- .nl_normalise_weights_safe(refit$log_marginal, "spatiotemporal grid")
    refit <- .joint_attach_pareto_k_regime(refit)
    refit$outer_grid_placement         <- "auto_recentered"
    refit$outer_grid_recenter_attempts <- 1L
    refit
}
