# Central-composite-design (CCD) outer integration for the joint multi-block
# nested-Laplace path.
#
# Part of the joint nested-Laplace driver; the public entry point
# tulpa_nested_laplace_joint() lives in nested_laplace_joint.R, the tensor-grid
# dispatch in nested_laplace_joint_multi.R.
#
# The tensor product over the per-block hyperparameter axes is k^d outer cells
# (d axes, k points each); each cell is one inner Newton solve. A CCD places
# 1 + 2d + 2^d nodes around the joint hyperparameter posterior mode, oriented by
# the Cholesky of its posterior covariance, and integrates with the corrected
# R-INLA design weights -- the same ccd_grid() / ccd_weights() machinery the
# SPDE (fit_spde_nested.R) and RE-covariance (nested_laplace_re_cov.R) paths
# use, here generalised to the joint path's mixed-support axes via the per-axis
# unconstraining transforms already defined for the outer Pareto-k diagnostic
# (nested_laplace_joint_pareto_k.R). For d = 4 that is 25 nodes vs 81 tensor
# cells; the gap widens with d.

# Resolve per-axis unconstraining transform tags for the joint multi-block
# LATENT grid (the b<N>.<axis> columns, phi dispersion axes excluded), reusing
# the Pareto-k registry so "which axes are safely transformable" lives in one
# place. Returns one tag ("log" / "logit01" / "identity") per latent axis, or
# NULL to DECLINE CCD when any axis has unguessable support (CAR_proper's
# rho_car, a non-BYM2 rho) -- the caller then falls back to the tensor grid.
# Does the CCD engage at this transformable-axis count for this integration
# mode? "grid" never engages; "ccd" lowers the threshold to
# >= 3 axes (explicit opt-in); "auto" (the default) engages only at >= 4 axes,
# where the tensor product's k^d blow-up bites hardest, and keeps the cheaper,
# more ridge-robust tensor grid at <= 3 axes.
.joint_ccd_engage <- function(integration, d_axes) {
    if (identical(integration, "grid")) return(FALSE)
    min_axes <- if (identical(integration, "ccd")) 3L else 4L
    d_axes >= min_axes
}

.joint_ccd_axis_tags <- function(axis_names, axis_offsets, prepared) {
    if (length(axis_names) == 0L) return(NULL)
    pseudo <- list(
        theta_grid   = matrix(0, 1L, length(axis_names),
                              dimnames = list(NULL, axis_names)),
        axis_offsets = axis_offsets,
        blocks       = prepared
    )
    # This caller only needs "can every axis be unconstrained" -- a decline of
    # any kind (unguessable support, an inconsistent layout) means no, so it
    # collapses back to NULL rather than carrying the reason (which is for the
    # FIT's k-hat field, not for the CCD design).
    tags <- .joint_pareto_axis_tags(pseudo)
    if (.k_is_decline(tags)) NULL else tags
}

# --- placement cost: the budget, the running meter, the heartbeat ------------
#
# Placing a design is not free: every probe the mode-find issues is one full
# inner Newton solve of the joint model, the same solve the outer grid pays for
# once per cell. The round caps bound how many rounds a placement takes and know
# nothing about what a round costs, so this section states the cost in inner
# solves, projects it before any is paid, meters it while it is being paid, and
# reports it afterwards. The numbers behind the projection live in
# `.CCD_PLACEMENT` (R/settings.R).

# Points one full finite-difference stencil evaluates at `d` axes: the centre,
# the 2d axials and the 4 * C(d, 2) mixed corners.
.ccd_stencil_evals <- function(d) 1 + 2 * d + 4 * choose(d, 2)

# Points one mode-find round evaluates: the stencil plus the batched line search
# over `max_halve + 1` candidates.
.ccd_round_evals <- function(d, cfg) {
    .ccd_stencil_evals(d) + cfg$max_halve + 1
}

# The cheapest placement over `d` free axes that can still produce a design: the
# mode-find's opening centre evaluation, one round, and the closing stencil that
# supplies the design's curvature. A budget below this cannot buy a design at
# all, which is what the up-front decline tests.
.ccd_min_evals <- function(d, cfg) {
    if (d < 1L) return(0)
    1 + 2 * .ccd_stencil_evals(d) + (cfg$max_halve + 1)
}

# The designed first attempt: grid-median seed, fixed step, `first_rounds`
# rounds, closing stencil. Its cost is set by `d` alone and does not grow with
# the grid.
.ccd_first_evals <- function(d, cfg) {
    if (d < 1L) return(0)
    1 + cfg$first_rounds * .ccd_round_evals(d, cfg) + .ccd_stencil_evals(d)
}

# The escalation the first attempt falls through to: the joint-grid seed, the
# step calibration, and the full-budget rescue mode-find. `n_seed` is what the
# seed pass will evaluate (below).
.ccd_rescue_evals <- function(d, n_seed, cfg) {
    if (d < 1L) return(0)
    n_seed + cfg$calibrate_rounds * .ccd_stencil_evals(d) +
        1 + cfg$max_rounds * .ccd_round_evals(d, cfg) + .ccd_stencil_evals(d)
}

# What the seed pass will evaluate for a component whose free axes carry the
# level counts `n_levels`: the joint Cartesian grid when it fits under the cap,
# otherwise the coordinate-ascent sweep, which visits each axis carrying at
# least two levels.
.ccd_seed_evals <- function(n_levels, cfg) {
    if (length(n_levels) == 0L) return(0)
    n_pts <- prod(as.numeric(n_levels))
    if (n_pts >= 2 && n_pts <= cfg$seed_max_pts) return(n_pts)
    sum(as.numeric(n_levels[n_levels >= 2L]))
}

# Nodes the design itself will carry over `d` free axes, read off the design
# builder rather than restated, so the two cannot drift. A component with no
# free axis is the single cell at the origin of the copy coordinates.
.ccd_design_nodes <- function(d) {
    if (d < 1L) return(1)
    as.numeric(ccd_grid(d)$n_points)
}

# The placement ceiling, in inner solves. `n_tensor` is the latent tensor grid
# the design replaces and `n_nodes` the design that replaces it, so at
# `evals_per_cell = 1` a placement plus its design costs no more than the
# integration it avoids. The floor exempts the designed first attempt, whose
# cost is a function of the axis count and not of the grid: without it a small
# tensor grid declines every placement on arithmetic that was meant to bound the
# escalation.
.ccd_budget <- function(n_tensor, n_nodes, free_counts, cfg) {
    if (!is.finite(cfg$evals_per_cell)) return(Inf)
    cap <- ceiling(cfg$evals_per_cell * max(n_tensor - n_nodes, 0))
    flr <- if (isTRUE(cfg$budget_floor))
        sum(vapply(free_counts, .ccd_first_evals, numeric(1), cfg = cfg)) else 0
    max(cap, flr)
}

# Wall-clock seconds, for the heartbeat's elapsed / per-eval fields. The outer
# grid runs its cells across threads, so wall clock is the quantity a user
# projects a phase from.
.ccd_now <- function() as.numeric(Sys.time())

# Human-readable duration, mirroring `tulpa_progress::format_secs()` in
# inst/include/tulpa/nested_progress.h so a mode-find line and a grid line read
# the same. ASCII only (logs, redirects).
.ccd_format_secs <- function(s) {
    if (!is.finite(s) || s < 0) return("?")
    if (s < 90)   return(sprintf("%.0fs", s))
    if (s < 5400) return(sprintf("%.1fmin", s / 60))
    sprintf("%.1fh", s / 3600)
}

# The heartbeat's gate is the fit's own progress record, the same
# `tulpa.nl_progress` the outer-grid reporter reads and the joint front door
# sets once per fit, so the placement phase and the grid loop are on one switch.
# Absent (a direct call outside a fit) is silent.
.ccd_progress_opts <- function() {
    p <- getOption("tulpa.nl_progress", NULL)
    if (!is.list(p)) return(list(on = FALSE, file = ""))
    list(on   = isTRUE(p$progress),
         file = as.character(p$progress_file %||% ""))
}

# The placement knobs this fit runs under: the registry defaults, overridden
# where the caller set a `control` key. `.joint_ccd_grid()` is reached through
# the multi-block dispatch rather than from the front door's own frame, so the
# resolved record travels on a scoped option, exactly as progress, the
# checkpoint and the screening depth do.
.ccd_placement_args <- function(control) {
    cfg <- as.list(.CCD_PLACEMENT)
    if (!is.null(control$ccd_budget)) {
        v <- suppressWarnings(as.numeric(control$ccd_budget))
        if (length(v) != 1L || is.na(v) || v < 0) {
            stop("`control$ccd_budget` must be a single non-negative number ",
                 "(a multiple of the tensor grid the design replaces), or Inf ",
                 "to place without a ceiling.", call. = FALSE)
        }
        cfg$evals_per_cell <- v
    }
    if (!is.null(control$ccd_budget_floor)) {
        cfg$budget_floor <- isTRUE(control$ccd_budget_floor)
    }
    if (!is.null(control$ccd_stencil_reuse)) {
        cfg$stencil_reuse <- isTRUE(control$ccd_stencil_reuse)
    }
    if (!is.null(control$ccd_refresh_every)) {
        v <- suppressWarnings(as.integer(control$ccd_refresh_every))
        if (length(v) != 1L || is.na(v) || v < 1L) {
            stop("`control$ccd_refresh_every` must be a single integer >= 1.",
                 call. = FALSE)
        }
        cfg$refresh_every <- v
    }
    cfg
}

.ccd_placement_cfg <- function() {
    o <- getOption("tulpa.ccd_placement", NULL)
    if (is.list(o)) return(o)
    as.list(.CCD_PLACEMENT)
}

# The running meter: one per `.joint_ccd_grid()` call, shared by every mixture
# component, so the ceiling bounds the whole placement rather than each design
# separately. It is also the heartbeat's state (elapsed, rounds, spend).
.ccd_meter_new <- function(budget, progress = .ccd_progress_opts()) {
    e <- new.env(parent = emptyenv())
    e$evals    <- 0
    e$rounds   <- 0L
    e$budget   <- budget
    e$t0       <- .ccd_now()
    e$progress <- progress
    e
}

# Charge a batched evaluation BEFORE it is issued, so a batch that would cross
# the ceiling is never paid for. Crossing signals `tulpa_ccd_budget`, which
# unwinds to `.joint_ccd_grid()`: it declines to the tensor grid with the spend
# recorded, and the only state the abort leaves behind is an advanced inner warm
# start, which the tensor path does not read.
.ccd_meter_spend <- function(meter, n) {
    if (is.null(meter)) return(invisible(NULL))
    if (is.finite(meter$budget) && meter$evals + n > meter$budget) {
        stop(structure(
            class = c("tulpa_ccd_budget", "error", "condition"),
            list(message = sprintf(
                     paste0("CCD placement budget exhausted: %.0f inner solves ",
                            "spent, %.0f more requested, ceiling %.0f."),
                     meter$evals, n, meter$budget),
                 call = NULL)))
    }
    meter$evals <- meter$evals + n
    invisible(NULL)
}

# One completed mode-find round, across every component of the placement.
.ccd_meter_round <- function(meter) {
    if (!is.null(meter)) meter$rounds <- meter$rounds + 1L
    invisible(NULL)
}

# Re-raise a placement-budget signal from inside a `tryCatch(error =)` that
# exists to absorb a failed stencil. A budget stop is not a numerical failure to
# absorb: it has to unwind to `.joint_ccd_grid()` so the fit declines under its
# own reason instead of being reported as a mode-find failure, and so the loop
# it aborted does not go on to issue further probes.
.ccd_rethrow_budget <- function(e) {
    if (inherits(e, "tulpa_ccd_budget")) stop(e)
    NULL
}

# One heartbeat line. The console channel is cat() + flush.console(), which is
# what the outer-grid reporter's console line uses, so a placement line and a
# grid line read the same and neither arrives as a condition a caller has to
# handle. When a heartbeat file is configured it is overwritten with that
# reporter's own four-number wire format ("<done> <total> <elapsed_s> <eta_s>"),
# so a detached reader parsing the file sees liveness through the placement
# phase instead of an empty file until the grid loop starts.
.ccd_meter_beat <- function(meter, phase, detail = "") {
    if (is.null(meter) || !isTRUE(meter$progress$on)) return(invisible(NULL))
    elapsed <- .ccd_now() - meter$t0
    per     <- if (meter$evals > 0) elapsed / meter$evals else 0
    spent   <- if (is.finite(meter$budget))
        sprintf("%.0f/%.0f evals", meter$evals, meter$budget) else
        sprintf("%.0f evals", meter$evals)
    line <- sprintf("[ccd mode-find] %s | %s%s | elapsed %s | %s/eval",
                    phase, spent,
                    if (nzchar(detail)) paste0(" | ", detail) else "",
                    .ccd_format_secs(elapsed), .ccd_format_secs(per))
    cat(line, "\n", sep = "")
    utils::flush.console()
    f <- meter$progress$file
    if (length(f) == 1L && nzchar(f)) {
        total <- if (is.finite(meter$budget)) meter$budget else 0
        eta   <- if (is.finite(meter$budget) && meter$evals > 0)
            max(0, meter$budget - meter$evals) * per else 0
        tryCatch(
            cat(sprintf("%.0f %.0f %.1f %.1f\n", meter$evals, total,
                        elapsed, eta), file = f),
            error = function(e) NULL, warning = function(w) NULL)
    }
    invisible(NULL)
}

# The axial half of a stencil: the centre and the 2d points at +/- h_j e_j, with
# the keys the derivative readers index them by. Shared by the full stencil and
# by the axial-only stencil the curvature-reuse path takes, so the point layout
# and the key convention have one definition.
.ccd_axial_rows <- function(u, h) {
    d <- length(u)
    rows <- list(u)                    # 1: centre
    key  <- c("0")
    for (j in seq_len(d)) {            # axials
        up <- u; up[j] <- u[j] + h[j]; rows[[length(rows) + 1L]] <- up
        dn <- u; dn[j] <- u[j] - h[j]; rows[[length(rows) + 1L]] <- dn
        key <- c(key, paste0("+", j), paste0("-", j))
    }
    list(rows = rows, key = key)
}

# The mixed corners: the 4 * C(d, 2) points at (+/- h_i e_i) + (+/- h_j e_j),
# which carry the off-diagonal second differences.
.ccd_corner_rows <- function(u, h) {
    d <- length(u)
    rows <- list()
    key  <- character(0)
    if (d >= 2L) for (i in seq_len(d - 1L)) for (j in (i + 1L):d) {
        for (si in c(1, -1)) for (sj in c(1, -1)) {
            v <- u; v[i] <- u[i] + si * h[i]; v[j] <- u[j] + sj * h[j]
            rows[[length(rows) + 1L]] <- v
            key <- c(key, paste0(if (si > 0) "+" else "-", i,
                                 if (sj > 0) "+" else "-", j))
        }
    }
    list(rows = rows, key = key)
}

# Value, gradient and diagonal curvature from an axial-keyed value vector.
#   g_i  = (f(+e_i) - f(-e_i)) / (2 h_i)
#   H_ii = (f(+e_i) - 2 f(0) + f(-e_i)) / h_i^2
.ccd_axial_derivs <- function(f, h, d) {
    f0 <- f[["0"]]
    g  <- numeric(d)
    hd <- numeric(d)
    for (j in seq_len(d)) {
        g[j]  <- (f[[paste0("+", j)]] - f[[paste0("-", j)]]) / (2 * h[j])
        hd[j] <- (f[[paste0("+", j)]] - 2 * f0 + f[[paste0("-", j)]]) / (h[j]^2)
    }
    list(f0 = f0, grad = g, hdiag = hd)
}

# Central-difference value / gradient / Hessian of an objective `eval1` (a
# closure mapping an [S x d] u-space matrix to a length-S vector of values) at
# point u. The centre, the 2d axial points (+/- h e_j) and the 4 * C(d, 2)
# mixed corners are evaluated in ONE `eval1` call, so every inner solve in a
# round fans out across the outer-grid threads.
#   H_ij = (f(+i+j) - f(+i-j) - f(-i+j) + f(-i-j)) / (4 h_i h_j)
.joint_ccd_fd_stencil <- function(u, eval1, h) {
    d  <- length(u)
    ax <- .ccd_axial_rows(u, h)
    co <- .ccd_corner_rows(u, h)
    U <- do.call(rbind, c(ax$rows, co$rows))
    f <- eval1(U)
    names(f) <- c(ax$key, co$key)
    der <- .ccd_axial_derivs(f, h, d)
    H <- matrix(0, d, d)
    diag(H) <- der$hdiag
    if (d >= 2L) for (i in seq_len(d - 1L)) for (j in (i + 1L):d) {
        hij <- (f[[paste0("+", i, "+", j)]] - f[[paste0("+", i, "-", j)]] -
                f[[paste0("-", i, "+", j)]] + f[[paste0("-", i, "-", j)]]) /
               (4 * h[i] * h[j])
        H[i, j] <- hij
        H[j, i] <- hij
    }
    list(f0 = as.numeric(der$f0), grad = der$grad, hess = H)
}

# The axial-only stencil of the curvature-reuse path: 1 + 2d evaluations, the
# value, the gradient and the measured diagonal curvature. The off-diagonal
# block it does not measure is carried by the secant update below.
.joint_ccd_axial_stencil <- function(u, eval1, h) {
    d  <- length(u)
    ax <- .ccd_axial_rows(u, h)
    U <- do.call(rbind, ax$rows)
    f <- eval1(U)
    names(f) <- ax$key
    der <- .ccd_axial_derivs(f, h, d)
    list(f0 = as.numeric(der$f0), grad = der$grad, hdiag = der$hdiag)
}

# Symmetric rank-1 secant update of the outer curvature, used only by the
# curvature-reuse path. BFGS does not apply here: it maintains a definite
# approximation and requires `s'y > 0`, which a log-posterior's negative-definite
# curvature does not give, and the mode-find has to keep working on the
# indefinite curvature a ridge produces. SR1 carries no such requirement and is
# the standard quasi-Newton update for an indefinite Hessian.
#
# The DIAGONAL is not inferred: it is overwritten with the round's own axial
# second differences. Only the off-diagonal block is carried by the secant, so
# the reuse trades the 4 * C(d, 2) mixed corners for an update and leaves every
# measured quantity measured.
#
# Returns NULL when the standard SR1 safeguard refuses the update -- the
# denominator small relative to the vectors it is formed from, where the rank-1
# correction is numerically meaningless. The caller then pays for a full
# stencil rather than carrying a stale model.
.joint_ccd_sr1_offdiag <- function(H, s, y, hdiag, safeguard = 1e-8) {
    if (is.null(H) || any(!is.finite(H)) || any(!is.finite(s)) ||
        any(!is.finite(y)) || any(!is.finite(hdiag))) return(NULL)
    r  <- y - as.numeric(H %*% s)
    nr <- sqrt(sum(r^2))
    ns <- sqrt(sum(s^2))
    if (nr <= .Machine$double.eps * max(1, sqrt(sum(y^2)))) {
        # The carried curvature already predicts this round's gradient change:
        # there is no correction to make, and forming one would be 0 / 0.
        Hn <- H
    } else {
        den <- sum(r * s)
        if (!is.finite(den) || den == 0 ||
            abs(den) < safeguard * nr * ns) return(NULL)
        Hn <- H + outer(r, r) / den
        Hn <- 0.5 * (Hn + t(Hn))
    }
    if (any(!is.finite(Hn))) return(NULL)
    diag(Hn) <- hdiag
    Hn
}

# Per-axis finite-difference step calibrated to each axis's local curvature, so
# the stencil spans ~1 posterior sd on every axis. An ill-conditioned outer
# posterior mixes sharp axes (a narrow field SD: a coarse step straddles the peak
# and reads low curvature) with wide / weakly-curved axes (a fine step's second
# difference is swamped by the inner marginal's small non-smoothness); one fixed
# step cannot serve both. From `h0`, each round sets h_j to ~1 sd where the
# curvature is concave and widens it toward `h_max` where it is not (the signal
# sits below the noise floor), settling in a few rounds. Returns the per-axis
# step. `span` is the per-axis box width (the supplied grid's u-range).
.joint_ccd_calibrate_step <- function(u, eval1, span, h0 = 0.1,
                                      h_min = 1e-3,
                                      rounds = .ccd_placement("calibrate_rounds"),
                                      meter = NULL) {
    d     <- length(u)
    h_max <- pmin(0.6, pmax(0.5 * span, h0))
    h     <- rep(h0, d)
    for (it in seq_len(rounds)) {
        st <- tryCatch(.joint_ccd_fd_stencil(u, eval1, h),
                       error = .ccd_rethrow_budget)
        .ccd_meter_beat(meter, sprintf("step calibration %d/%d", it, as.integer(rounds)))
        if (is.null(st) || any(!is.finite(diag(st$hess)))) break
        dH    <- diag(st$hess)
        neg   <- dH < -1e-8
        h_new <- h
        h_new[neg]  <- 1 / sqrt(-dH[neg])               # concave: span ~1 sd
        h_new[!neg] <- pmin(2 * h[!neg], h_max[!neg])   # below noise floor: widen
        h_new <- pmin(pmax(h_new, h_min), h_max)
        if (max(abs(h_new - h)) < 0.05 * h0) { h <- h_new; break }
        h <- h_new
    }
    h
}

# Coordinate-ascent pilot seed: one sweep that jumps each axis to its
# highest-log-posterior supplied grid value (the others held at the running
# best). A sharply-peaked outer posterior whose mode sits at a grid edge needs a
# seed near the peak: from the grid median a heavy-tailed inner marginal is
# gently sloped and its curvature reads as a false ridge. An axis the
# log-posterior barely depends on (an unidentified copy amplitude on a
# sigma-alpha ridge) keeps the median -- moving it would seed onto the ridge --
# so a jump is taken only when the axis's log-posterior range exceeds `min_gain`.
# `u_vals` is the per-axis grid in u-space; `eval1` maps a [S x d] u-matrix to
# log-posterior values.
.joint_ccd_pilot_seed <- function(u0, u_vals, eval1, min_gain = 1,
                                  meter = NULL) {
    u <- u0
    for (j in seq_along(u)) {
        gv <- u_vals[[j]]
        if (length(gv) < 2L) next
        cand <- matrix(u, nrow = length(gv), ncol = length(u), byrow = TRUE)
        cand[, j] <- gv
        lp  <- eval1(cand)
        fin <- is.finite(lp)
        if (any(fin) && (max(lp[fin]) - min(lp[fin]) > min_gain))
            u[j] <- gv[which.max(lp)]
        .ccd_meter_beat(meter, sprintf("coordinate seed, axis %d/%d",
                                       j, length(u)))
    }
    u
}

# Joint-grid pilot seed: evaluate the full latent Cartesian grid in ONE batched
# (parallel) call and return its joint argmax. This is a better mode-find seed
# than the per-axis coordinate sweep when the axes are correlated -- it lands at
# the actual joint peak, so the subsequent mode-find converges in a couple of
# rounds rather than crawling. Capped at `max_pts` so it never rebuilds the dense
# tensor CCD exists to avoid; above the cap (or on an all-failed eval) it falls
# back to the coordinate-ascent sweep.
.joint_ccd_grid_seed <- function(u0, u_vals, eval1,
                                 max_pts = .ccd_placement("seed_max_pts"),
                                 meter = NULL) {
    n_pts <- prod(vapply(u_vals, length, integer(1)))
    if (n_pts < 2L || n_pts > max_pts)
        return(.joint_ccd_pilot_seed(u0, u_vals, eval1, meter = meter))
    .ccd_meter_beat(meter, sprintf("grid seed (%d points)", as.integer(n_pts)))
    grid <- as.matrix(expand.grid(u_vals))
    dimnames(grid) <- NULL
    lp <- eval1(grid)
    if (!any(is.finite(lp)))
        return(.joint_ccd_pilot_seed(u0, u_vals, eval1, meter = meter))
    .ccd_meter_beat(meter, "grid seed done")
    as.numeric(grid[which.max(lp), ])
}

# Negative-definite regularisation of a symmetric matrix: eigen-clip every
# eigenvalue to <= -ridge so the log-posterior curvature gives an ascent
# direction (and a usable Gaussian precision -H).
.joint_ccd_neg_def <- function(H, ridge_rel = 1e-6) {
    Hs <- 0.5 * (H + t(H))
    eg <- eigen(Hs, symmetric = TRUE)
    cap <- -ridge_rel * max(abs(eg$values), 1)
    ev  <- pmin(eg$values, cap)
    eg$vectors %*% (ev * t(eg$vectors))
}

# Is the outer log-posterior curvature well conditioned for a Gaussian CCD? The
# CCD orients its nodes by the Cholesky of -H^-1, so -H (the precision) must be
# positive-definite with no near-zero eigenvalue. A flat / indefinite curvature
# (the sigma-alpha ridge, where only the per-arm sigma*alpha
# product is identified) makes -H^-1 huge along the ridge, so the design nodes
# land at extreme hyperparameters where the inner Newton fails. Single source of
# truth for the centre pre-check (decline fast, before the line search) and the
# post-mode-find guard.
.joint_ccd_outer_hess_ok <- function(H, rel_tol = 1e-6) {
    if (is.null(H) || any(!is.finite(H))) return(FALSE)
    neg_H <- -0.5 * (H + t(H))
    ev <- tryCatch(eigen(neg_H, symmetric = TRUE, only.values = TRUE)$values,
                   error = function(e) NULL)
    if (is.null(ev) || any(!is.finite(ev))) return(FALSE)
    mx <- max(abs(ev))
    if (mx <= 0) return(FALSE)
    min(ev) > rel_tol * mx
}

# Pull the inner latent mode of row `k` of an `eval1` result out of its "modes"
# attribute (attached by the caller's eval_logpost), for warm-starting the next
# probe. NULL when absent or non-finite. `k` selects the accepted candidate when
# several were evaluated in one batched call.
.joint_ccd_eval_mode <- function(f, k = 1L) {
    m <- attr(f, "modes")
    if (is.null(m) || !is.matrix(m) || nrow(m) < k) return(NULL)
    v <- as.numeric(m[k, ])
    if (any(!is.finite(v))) return(NULL)
    v
}

# Damped-Newton mode-find for the joint hyperparameter log-posterior on the
# unconstrained scale. Each round fits a local quadratic via one batched
# stencil call (gradient + Hessian), regularises the Hessian negative-definite,
# and takes a trust-clamped, backtracked, box-clamped Newton step uphill. Robust
# to the inner log-marginal's small non-smoothness (a least-curvature quadratic
# model rather than a gradient line search, which a finite-difference gradient
# breaks on).
#
# Three guards keep an ill-conditioned outer posterior (the sigma-alpha ridge of
#) from burning hours of full-field inner solves in the line
# search:
#   * the centre FD-Hessian is checked BEFORE the first line search; a ridge /
#     flat curvature declines immediately (status "ridge") -- the same verdict
#     the post-mode-find guard would reach, minus the line search,
#   * the Newton step is trust-clamped per coordinate so a near-singular Hessian
#     cannot fling a candidate to an extreme hyperparameter, and
#   * the backtracking line search is capped at `max_halve` halvings, all
#     evaluated in one batched call so they run across the outer-grid threads.
# An optional `on_accept(mode)` hook receives the inner latent mode at each
# accepted point so the caller can advance its inner warm start (so probes near
# the current iterate solve in a few Newton steps instead of cold from the
# centre).
#
# Returns list(par, hess, value, converged, status) with status in
# {"ok", "ridge", "fail"}; only "ok" yields a usable CCD scale.
.joint_ccd_modefind <- function(u0, eval1, lower, upper, h,
                                max_rounds = .ccd_placement("max_rounds"),
                                tol = 1e-3,
                                max_halve = .ccd_placement("max_halve"),
                                trust = NULL,
                                on_accept = NULL, meter = NULL,
                                reuse = FALSE,
                                refresh_every = .ccd_placement("refresh_every")) {
    d <- length(u0)
    if (is.null(trust)) trust <- rep(Inf, d)
    fail <- function(u, H, f) {
        .ccd_meter_beat(meter, "declined", "mode-find failed")
        list(par = u, hess = H, value = f, converged = FALSE, status = "fail")
    }
    u   <- pmin(pmax(u0, lower), upper)
    f0e <- eval1(matrix(u, nrow = 1L))
    f_u <- as.numeric(f0e)
    if (!is.finite(f_u)) return(fail(u, NULL, f_u))
    if (!is.null(on_accept)) {
        m0 <- .joint_ccd_eval_mode(f0e)
        if (!is.null(m0)) on_accept(m0)
    }
    H <- NULL
    converged <- FALSE
    # Curvature-reuse state (the `reuse` path only): the previous round's point
    # and gradient feed the secant update, `since_full` counts rounds since the
    # last measured off-diagonal block, and `force_full` demands a measured one
    # on the next round.
    u_prev     <- NULL
    g_prev     <- NULL
    since_full <- 0L
    force_full <- FALSE
    for (iter in seq_len(max_rounds)) {
        want_full <- !isTRUE(reuse) || force_full || is.null(H) ||
                     is.null(g_prev) || since_full >= refresh_every
        force_full <- FALSE
        st <- tryCatch(
            if (want_full) .joint_ccd_fd_stencil(u, eval1, h)
            else .joint_ccd_axial_stencil(u, eval1, h),
            error = .ccd_rethrow_budget)
        if (is.null(st) || any(!is.finite(st$grad)))
            return(fail(u, H, f_u))
        if (want_full) {
            if (any(!is.finite(st$hess))) return(fail(u, H, f_u))
            H <- st$hess
            since_full <- 0L
        } else {
            if (any(!is.finite(st$hdiag))) return(fail(u, H, f_u))
            H_up <- .joint_ccd_sr1_offdiag(H, u - u_prev, st$grad - g_prev,
                                           st$hdiag)
            if (is.null(H_up)) {
                # The secant refused: measure the curvature rather than carry a
                # stale off-diagonal block into the step.
                st <- tryCatch(.joint_ccd_fd_stencil(u, eval1, h),
                               error = .ccd_rethrow_budget)
                if (is.null(st) || any(!is.finite(st$grad)) ||
                    any(!is.finite(st$hess))) return(fail(u, H, f_u))
                H <- st$hess
                since_full <- 0L
                want_full  <- TRUE
            } else {
                H <- H_up
                since_full <- since_full + 1L
            }
        }
        u_prev <- u
        g_prev <- st$grad
        # Fast decline on a ridge / flat curvature: the centre Hessian already
        # tells us the Gaussian CCD scale is ill-defined, so bail BEFORE the
        # expensive backtracking line search. Read on the first round, which
        # always measures a full stencil.
        if (iter == 1L && !.joint_ccd_outer_hess_ok(H)) {
            .ccd_meter_beat(meter, "declined",
                            "outer curvature flat / ridged")
            return(list(par = u, hess = H, value = f_u,
                        converged = FALSE, status = "ridge"))
        }
        H_reg <- .joint_ccd_neg_def(H)
        step  <- tryCatch(as.numeric(-solve(H_reg, st$grad)),
                          error = function(e) NULL)
        if (is.null(step) || any(!is.finite(step))) return(fail(u, H, f_u))
        # Trust-region clamp: bound each coordinate's step so a near-singular
        # Hessian cannot send a candidate to an extreme hyperparameter where the
        # inner Newton needs many iterations.
        step <- pmax(pmin(step, trust), -trust)
        # Backtracking line search over `max_halve` halvings, all evaluated in ONE
        # batched call so the candidates fan out across the outer-grid threads:
        # a serial probe per halving does not parallelise, and on an expensive
        # inner solve (a full-field Laplace) the line search would otherwise
        # dominate the mode-find. Accept the largest step that improves -- the
        # same point a sequential backtrack would take.
        t_steps <- 0.5 ^ (seq_len(max_halve + 1L) - 1L)
        cands   <- t(vapply(t_steps, function(ts)
                       pmin(pmax(u + ts * step, lower), upper), numeric(d)))
        fce     <- eval1(cands)
        f_cands <- as.numeric(fce)
        u_try <- u; f_try <- f_u; mode_try <- NULL
        acc <- which(is.finite(f_cands) & f_cands >= f_u - 1e-8)
        if (length(acc)) {
            k        <- acc[1L]                 # t_steps is descending: largest step
            u_try    <- cands[k, ]
            f_try    <- f_cands[k]
            mode_try <- .joint_ccd_eval_mode(fce, k)
        }
        delta <- max(abs(u_try - u))
        u <- u_try; f_u <- f_try
        if (!is.null(on_accept) && !is.null(mode_try)) on_accept(mode_try)
        .ccd_meter_round(meter)
        .ccd_meter_beat(meter, sprintf("round %d/%d", iter, as.integer(max_rounds)),
                        sprintf("|step| %.3g | logpost %.7g", delta, f_u))
        if (delta < tol) {
            # A reused curvature model that stops producing a step has not shown
            # a mode: its off-diagonal block was inferred, so the next round
            # measures one and convergence is decided on measured curvature.
            if (!want_full) { force_full <- TRUE; next }
            converged <- TRUE
            break
        }
    }
    # Clean Hessian at the final point for the CCD scale. Always the full
    # stencil: this is the curvature the design is oriented by.
    st_fin <- tryCatch(.joint_ccd_fd_stencil(u, eval1, h),
                       error = .ccd_rethrow_budget)
    if (!is.null(st_fin) && all(is.finite(st_fin$hess))) H <- st_fin$hess
    .ccd_meter_beat(meter, "mode-find done",
                    sprintf("logpost %.7g", f_u))
    list(par = u, hess = H, value = f_u, converged = converged, status = "ok")
}

# Why a requested CCD fell back to the tensor grid. One entry per `return`
# below, so the caller records a named reason on the fit instead of leaving a
# consumer to read the choice off the absence of a field (the established
# convention):
#
#   "axis_count"          fewer transformable latent axes than the mode's
#                         threshold -- decided by .joint_ccd_engage(), so it is
#                         stamped by the caller rather than here.
#   "unguessable_axis"    an axis whose support the engine will not guess
#                         (CAR_proper's rho_car, a non-BYM2 rho).
#   "degenerate_axis"     an axis carrying a single value, or a non-finite
#                         u-space box: no curvature to orient a design across.
#   "modefind_ridge"      the outer log-posterior is flat / ridged.
#   "modefind_boundary"   the outer mode pinned to the (wide) axis box.
#   "modefind_degenerate" the outer Hessian is degenerate at the mode.
#   "modefind_failed"     the mode-find did not converge.
#   "hessian_singular"    the outer Hessian would not invert.
#   "hessian_not_pd"      its inverse has no Cholesky factor.
#   "copy_atom_components" more copy atoms than the component cap: the mixture
#                         below expands into 2^n designs, and past the cap the
#                         tensor rule is the cheaper one.
#   "copy_atom_mass"      the declared atom mass is not a probability.
#   "placement_budget"    placing the design costs more inner solves than
#                         integrating the tensor grid it replaces. The only
#                         reason here about whether a definable design is WORTH
#                         placing rather than whether one exists: it fires
#                         up-front when even the cheapest placement cannot fit
#                         the ceiling, and mid-placement when the running spend
#                         reaches it.
.CCD_DECLINE_REASONS <- c("axis_count", "unguessable_axis", "degenerate_axis",
                          "modefind_ridge", "modefind_boundary",
                          "modefind_degenerate", "modefind_failed",
                          "hessian_singular", "hessian_not_pd",
                          "copy_atom_components", "copy_atom_mass",
                          "placement_budget")

# The coordinate each CCD axis is DESIGNED on.
#
# `.joint_pareto_block_tags()` answers the neighbouring question for the outer
# Pareto-k proposal, where a copy `alpha` is carried as an unconstrained real:
# there the nodes already exist and the tag only says how to reweight them. A
# CCD places nodes instead, so it needs the coordinate the axis's declared
# measure lives on. `.joint_axis_specs()` declares the copy scale a point mass
# at 0 plus a log continuum on (0, Inf), so its continuum is designed in
# log alpha; an affine design on the identity coordinate puts nodes at negative
# alpha, which is outside the model's support.
.joint_ccd_coord_tags <- function(axis_names, tags) {
    tags[sub("^b[0-9]+\\.", "", axis_names) == "alpha"] <- "log"
    tags
}

# The copy-scale axes whose supplied levels carry the "no coupling" atom.
.joint_ccd_atom_axes <- function(axis_names, axis_values) {
    bare <- sub("^b[0-9]+\\.", "", axis_names)
    has_zero <- vapply(axis_values,
                       function(v) any(as.numeric(v) == 0), logical(1))
    unname(which(bare == "alpha" & has_zero))
}

# Cap on the mixture components an atom-carrying grid expands into (2^n_atom).
.JOINT_CCD_MAX_COMPONENTS <- 8L

# One mixture component: a CCD over the `free` axes with the `pinned` copy
# axes held at alpha = 0. Returns the component's [n_node x d] node matrix in
# FULL width (pinned columns zero) with its design weights, or
# `list(declined = <reason>)`.
.joint_ccd_component <- function(free, pinned, axis_names, tags, axis_values,
                                 eval_logpost, verbose = FALSE,
                                 set_warm = NULL, meter = NULL,
                                 cfg = .ccd_placement_cfg()) {
    d  <- length(axis_names)
    dk <- length(free)
    declined <- function(reason) list(declined = reason)

    # Widen a design over the free axes back to the full axis set. The pinned
    # copy axes are exactly zero here, which is the "no coupling" node the
    # tensor rule carries as a level.
    fill <- function(M_free, n) {
        M <- matrix(0, n, d, dimnames = list(NULL, axis_names))
        if (dk > 0L) M[, free] <- M_free
        M
    }
    # Every copy axis switched off: the component is the single cell at the
    # origin of the copy coordinates, carrying its whole prior mass.
    if (dk == 0L) {
        return(list(grid = fill(NULL, 1L), dnode = 1,
                    u_hat = numeric(0), L_scale = matrix(0, 0L, 0L)))
    }

    # u-space search box, generously wider than the supplied per-axis grid
    # range so the mode-find can reach a posterior mode that sits beyond the
    # user's coarse grid (the common case: the grid is a net, not a support
    # bound). The CCD axials are clamped to this box so a design point never
    # runs to a numerically-infeasible hyperparameter; the box also bounds the
    # mode-find, and a mode pinned to its (wide) edge is treated as a runaway /
    # boundary fit and declined to the tensor grid.
    u_vals <- lapply(free, function(j)
        .joint_pareto_fwd(tags[j], as.numeric(axis_values[[j]])))
    lower0 <- vapply(u_vals, min, numeric(1))
    upper0 <- vapply(u_vals, max, numeric(1))
    u0     <- vapply(u_vals, stats::median, numeric(1))
    span   <- pmax(upper0 - lower0, 1e-3)
    pad    <- pmax(1.5 * span, 0.5)
    lower  <- lower0 - pad
    upper  <- upper0 + pad
    # Per-coordinate trust radius for the mode-find step: roughly the user's
    # supplied grid width, so the mode can traverse the box over a few rounds but
    # a single Newton step never leaps to an extreme hyperparameter where the
    # inner Newton stalls.
    trust  <- pmax(span, 0.5)
    # Degenerate axis (a single supplied value) carries no curvature; CCD
    # cannot orient a design across it -- fall back to the tensor grid.
    if (any(!is.finite(c(lower, upper, u0))) || any(upper - lower <= 0)) {
        return(declined("degenerate_axis"))
    }

    # Map a u-space matrix to physical theta (per-axis inverse transform).
    u_to_theta <- function(U) {
        M <- matrix(0, nrow(U), dk)
        for (j in seq_len(dk)) {
            M[, j] <- .joint_pareto_inv(tags[free[j]], U[, j])$theta
        }
        fill(M, nrow(U))
    }
    eval1 <- function(U) {
        # Charged before the solves are issued, so a batch that would cross the
        # placement ceiling is declined rather than paid for.
        .ccd_meter_spend(meter, nrow(U))
        out <- eval_logpost(u_to_theta(U))
        md  <- attr(out, "modes")
        lp  <- as.numeric(out)
        lp[!is.finite(lp)] <- -1e10
        # Carry the inner latent modes through (single-row probes only) so the
        # mode-find can advance the inner warm start.
        if (!is.null(md)) attr(lp, "modes") <- md
        lp
    }

    on_accept <- if (is.null(set_warm)) NULL else function(mode) set_warm(mode)

    # Mode-find from one seed / step, returning the validated Gaussian centre and
    # curvature (u_hat, H) or a `reason` for declining: the mode bailed / ridged,
    # ran to the (wide) box edge (a boundary-supported hyperparameter), or sits on
    # a flat / indefinite curvature -- none of which define a usable Gaussian CCD
    # scale (its nodes would land at extreme hyperparameters where the inner
    # Newton fails).
    try_modefind <- function(u_start, h_step, max_rounds = cfg$max_rounds) {
        mf <- .joint_ccd_modefind(u_start, eval1, lower, upper, h_step,
                                  trust = trust, on_accept = on_accept,
                                  max_rounds = max_rounds, meter = meter,
                                  max_halve = cfg$max_halve,
                                  reuse = isTRUE(cfg$stencil_reuse),
                                  refresh_every = cfg$refresh_every)
        if (is.null(mf) || !identical(mf$status, "ok"))
            return(list(reason = if (!is.null(mf) && identical(mf$status, "ridge"))
                                 "ridge" else "fail"))
        u_hat   <- mf$par
        eps_box <- 1e-3 * pmax(upper - lower, 1)
        if (any(abs(u_hat - lower) < eps_box) || any(abs(u_hat - upper) < eps_box))
            return(list(reason = "boundary"))
        if (!.joint_ccd_outer_hess_ok(mf$hess))
            return(list(reason = "degenerate"))
        list(u_hat = u_hat, H = mf$hess)
    }

    # Default: grid-median seed with a fixed step. A well-conditioned outer
    # posterior reaches a usable Gaussian centre in a handful of Newton rounds,
    # and a sigma-alpha copy ridge converges at once (a flat direction has ~0
    # gradient, so its step is ~0): both engage here, unperturbed. The round cap
    # escalates to the rescue when the median seed does not converge, rather than
    # crawling the full budget across a gently-sloped tail.
    #
    # On a decline, the posterior is sharply peaked and ill-conditioned (a narrow
    # field-SD axis at a grid edge beside a wide axis), which the grid median
    # cannot characterise. The rescue warm-starts at the best latent grid cell
    # (one batched, parallel evaluation -> joint argmax, already near the peak) and
    # uses a per-axis step calibrated to the local curvature (a single fixed step
    # cannot resolve a sharp field-SD axis beside a wide / weakly-curved one), then
    # runs the mode-find to convergence from there.
    got <- try_modefind(u0, rep(0.1, dk), max_rounds = cfg$first_rounds)
    if (is.null(got$u_hat)) {
        u_seed <- .joint_ccd_grid_seed(u0, u_vals, eval1,
                                       max_pts = cfg$seed_max_pts,
                                       meter = meter)
        h_cal  <- .joint_ccd_calibrate_step(u_seed, eval1, span,
                                            rounds = cfg$calibrate_rounds,
                                            meter = meter)
        got    <- try_modefind(u_seed, h_cal)
    }
    if (is.null(got$u_hat)) {
        why <- got$reason %||% "fail"
        if (verbose)
            message("tulpa CCD: ", switch(why,
                ridge      = "outer log-posterior is flat / ridged (curvature ill-conditioned)",
                boundary   = "outer mode pinned to the axis box (boundary-supported hyperparameter)",
                degenerate = "outer Hessian degenerate at the mode",
                             "outer mode-find failed"),
                "; using the tensor grid.")
        return(declined(switch(why,
            ridge      = "modefind_ridge",
            boundary   = "modefind_boundary",
            degenerate = "modefind_degenerate",
                          "modefind_failed")))
    }
    u_hat <- got$u_hat
    H     <- got$H
    neg_H <- -0.5 * (H + t(H))
    post_cov <- tryCatch(solve(neg_H), error = function(e) NULL)
    if (is.null(post_cov)) return(declined("hessian_singular"))
    L_scale <- tryCatch(t(chol(post_cov)), error = function(e) NULL)
    if (is.null(L_scale)) return(declined("hessian_not_pd"))

    # CCD design: factorial corners at +/- 1.1 per whitened axis (INLA's f0),
    # with the matching corrected design weights (single source of truth with
    # the SPDE / RE-cov paths; see ccd_grid.R). Map z -> u_hat + L z, clamp to
    # the box, inverse-transform to physical theta.
    ccd   <- ccd_grid(dk, f_0 = sqrt(dk) * 1.1)
    dnode <- ccd_weights(ccd)
    u_grid <- ccd_to_theta(ccd$z, u_hat, L_scale)
    for (j in seq_len(dk)) {
        u_grid[, j] <- pmin(pmax(u_grid[, j], lower[j]), upper[j])
    }
    list(grid = u_to_theta(u_grid), dnode = dnode, u_hat = u_hat,
         L_scale = L_scale)
}

# Build the CCD node grid for the joint multi-block path. `eval_logpost` maps a
# user-facing [S x d] theta matrix (columns = `axis_names`) to a length-S
# log-posterior vector (inner log-marginal + baked hyperprior). `axis_values`
# is a length-d list of the per-axis grid values the user supplied (used to set
# the mode-find box and the initial point). Returns
#   list(grid, dnode, u_hat, L_scale, tags)
# with `grid` the [n_node x d] physical node matrix (colnames = axis_names) and
# `dnode` the corrected CCD design weights, or `list(declined = <reason>)` to
# fall back to the tensor grid. A caller tests `is.null(x$grid)`. Either shape
# also carries what the placement COST -- `modefind_evals` / `modefind_budget` /
# `modefind_rounds` / `modefind_seconds` -- which is what the fit reports as
# `ccd_modefind_*`, so a decline says how much was paid before it fired.
#
# A copy `alpha` carries a point mass at 0 beside its log continuum
# (`.joint_axis_specs()`), so the outer posterior over a grid with `n_atom` such
# axes is a MIXTURE of 2^n_atom components -- one per subset of copy couplings
# switched off. An affine CCD is a single Gaussian and represents one of them,
# so the design is built per component and the components are combined by the
# prior mass their configuration declares. Without the split the atom is
# unreachable: no design node lands exactly at alpha = 0, so "no coupling"
# carries no posterior weight at all, while the tensor rule integrates it as a
# level.
.joint_ccd_grid <- function(axis_names, axis_offsets, prepared, axis_values,
                            eval_logpost, verbose = FALSE, set_warm = NULL,
                            atom_mass = .TULPA_COPY_ATOM_MASS,
                            cfg = .ccd_placement_cfg(),
                            n_tensor_cells = NULL) {
    d <- length(axis_names)
    # The placement meter exists from the moment the design's shape is known;
    # `finish()` stamps whatever it holds onto every return, so a decline
    # reports the cost it had already paid rather than leaving the caller to
    # read it off the absence of a field.
    meter <- NULL
    finish <- function(x) {
        if (is.null(meter)) return(x)
        x$modefind_evals   <- as.numeric(meter$evals)
        x$modefind_budget  <- as.numeric(meter$budget)
        x$modefind_rounds  <- as.integer(meter$rounds)
        x$modefind_seconds <- .ccd_now() - meter$t0
        x
    }
    declined <- function(reason) finish(list(declined = reason))
    tags <- .joint_ccd_axis_tags(axis_names, axis_offsets, prepared)
    if (is.null(tags)) return(declined("unguessable_axis"))
    tags <- .joint_ccd_coord_tags(axis_names, tags)

    atom   <- .joint_ccd_atom_axes(axis_names, axis_values)
    n_atom <- length(atom)
    if (bitwShiftL(1L, n_atom) > .JOINT_CCD_MAX_COMPONENTS) {
        if (verbose)
            message("tulpa CCD: ", n_atom, " copy atoms expand into ",
                    bitwShiftL(1L, n_atom), " designs; using the tensor grid.")
        return(declined("copy_atom_components"))
    }

    a <- as.numeric(atom_mass)
    if (n_atom > 0L && (length(a) != 1L || !is.finite(a) || a < 0 || a >= 1)) {
        return(declined("copy_atom_mass"))
    }

    # The design coordinate reaches the continuum only, so the box is set from
    # the positive levels; the atom enters as its own component.
    cont_values <- axis_values
    for (j in atom) {
        v <- as.numeric(axis_values[[j]])
        cont_values[[j]] <- v[v > 0]
    }
    if (any(vapply(cont_values, length, integer(1)) == 0L)) {
        return(declined("degenerate_axis"))
    }

    # Subsets of the copy axes, as bit patterns: component `m` switches off
    # every copy axis whose bit is set.
    offs <- lapply(seq_len(bitwShiftL(1L, n_atom)) - 1L, function(m)
        if (n_atom == 0L) integer(0)
        else atom[bitwAnd(bitwShiftR(m, seq_len(n_atom) - 1L), 1L) == 1L])

    # What the placement costs, against what it replaces. `n_tensor` is the
    # latent tensor grid the design stands in for -- the product of the supplied
    # per-axis level counts, which is what the fallback below expands -- and
    # `n_nodes` the design that replaces it. A caller who knows the replaced
    # cell count exactly (a block grid that is not a full cross of its own axes)
    # supplies it rather than having it inferred.
    free_counts <- vapply(offs, function(off) length(setdiff(seq_len(d), off)),
                          integer(1))
    n_tensor <- if (is.null(n_tensor_cells))
        prod(vapply(axis_values, function(v) length(unique(as.numeric(v))),
                    integer(1))) else as.numeric(n_tensor_cells)
    n_nodes  <- sum(vapply(free_counts, .ccd_design_nodes, numeric(1)))
    budget   <- .ccd_budget(n_tensor, n_nodes, free_counts, cfg)
    meter    <- .ccd_meter_new(budget)

    # Up-front decline: the cheapest placement that could still produce a design
    # over these axes, against the ceiling. Above it there is nothing to buy, so
    # the fit falls back to the tensor grid without paying for a single inner
    # solve.
    min_evals <- sum(vapply(free_counts, .ccd_min_evals, numeric(1), cfg = cfg))
    if (min_evals > budget) {
        if (verbose)
            message("tulpa CCD: placing the design needs at least ", min_evals,
                    " inner solves against a ceiling of ", budget,
                    " (tensor grid ", n_tensor, " cells); using the tensor grid.")
        return(declined("placement_budget"))
    }
    max_evals <- sum(vapply(seq_along(offs), function(i) {
        dk <- free_counts[[i]]
        if (dk < 1L) return(0)
        n_lv <- vapply(cont_values[setdiff(seq_len(d), offs[[i]])],
                       function(v) length(unique(as.numeric(v))), integer(1))
        .ccd_first_evals(dk, cfg) +
            .ccd_rescue_evals(dk, .ccd_seed_evals(n_lv, cfg), cfg)
    }, numeric(1)))
    .ccd_meter_beat(meter, "placement start",
                    sprintf("tensor %.0f cells | design %.0f nodes | worst case %.0f evals",
                            n_tensor, n_nodes, max_evals))

    grids  <- vector("list", length(offs))
    dnodes <- vector("list", length(offs))
    full   <- NULL
    # A budget reached mid-placement unwinds here. Nothing the abort leaves
    # behind is read by the tensor fallback: the components already designed are
    # dropped with the design, and the only other state advanced is the inner
    # warm start, which the fallback does not consume.
    why <- tryCatch({
        reason <- NULL
        for (i in seq_along(offs)) {
            off  <- offs[[i]]
            free <- setdiff(seq_len(d), off)
            cmp  <- .joint_ccd_component(free, off, axis_names, tags,
                                         cont_values, eval_logpost,
                                         verbose = verbose,
                                         set_warm = set_warm, meter = meter,
                                         cfg = cfg)
            # A component carries declared prior mass, so one that cannot be
            # designed cannot be dropped: the whole CCD declines to the tensor
            # rule, which integrates every component as levels.
            if (is.null(cmp$grid)) {
                reason <- as.character(cmp$declined %||% "modefind_failed")
                break
            }
            mass <- if (n_atom == 0L) 1
                    else prod(ifelse(atom %in% off, a, 1 - a))
            grids[[i]]  <- cmp$grid
            dnodes[[i]] <- cmp$dnode * mass
            if (length(off) == 0L) full <- cmp
        }
        reason
    }, tulpa_ccd_budget = function(e) {
        if (verbose)
            message("tulpa CCD: ", conditionMessage(e),
                    " Using the tensor grid.")
        .ccd_meter_beat(meter, "declined", "placement budget exhausted")
        "placement_budget"
    })
    if (!is.null(why)) return(declined(why))

    # `u_hat` / `L_scale` describe the all-continuum component, the one defined
    # on all d axes. `atom_split` says whether there are others beside it, which
    # is what decides if a single Gaussian describes the design at all.
    finish(list(grid = do.call(rbind, grids), dnode = unlist(dnodes),
                u_hat = full$u_hat, L_scale = full$L_scale, tags = tags,
                atom_split = n_atom > 0L))
}

# Announce the engaged outer integrator under verbose, at selection time
#. One authoritative line tied to the resolved
# `integration_used`, so a consumer who omitted `integration` from the control
# sees which integrator actually ran -- whether the CCD engaged, the tensor grid
# was kept, or (on a ridged posterior) the CCD declined back to the tensor --
# rather than having to read `fit$...$integration` after the fact. `n_latent` is
# the latent node / cell count, `n_phi` the crossed phi-tensor cell count,
# `n_total` their product; `declined` flags a CCD that engaged by axis count but
# fell back (the specific reason is messaged by .joint_ccd_grid just before).
.joint_announce_integration <- function(integration_used, d_axes,
                                        n_latent, n_phi, n_total,
                                        declined) {
    if (identical(integration_used, "ccd")) {
        phi_part <- if (n_phi > 1L)
            sprintf(" x %d phi = %d cells", n_phi, n_total) else ""
        message(sprintf(
            "tulpa joint: outer integration: CCD (%d latent axes, %d nodes%s)",
            d_axes, n_latent, phi_part))
    } else if (isTRUE(declined)) {
        message(sprintf(
            "tulpa joint: outer integration: CCD declined -> tensor grid (%d cells)",
            n_total))
    } else {
        message(sprintf(
            "tulpa joint: outer integration: tensor grid (%d cells)", n_total))
    }
}

# Integration weights for an outer grid carrying CCD design weights `dnode`:
# w_k proportional to dnode_k * exp(log_marginal_k). With dnode == NULL this is
# the plain softmax (uniform tensor-cell weight). Both branches drop non-finite
# log-marginal nodes from the max-shift and zero their weight: a non-converged
# inner Newton returns NaN log_marginal, and an unguarded max() would propagate
# that NaN to every weight, poisoning the whole vector (the dense tensor path is
# the one tulpa_posterior_draws / theta_mean read directly).
# `log_quad` is the tensor grid's per-cell prior mass (R/hyper_quadrature.R). It
# applies only where the nodes ARE the tensor rule; a CCD design carries its own
# integration weights in `dnode`, which already encode the volume each node
# stands for, so folding cell widths in on top would count it twice.
.joint_integration_weights <- function(log_marginal, dnode = NULL,
                                       log_quad = NULL) {
    if (is.null(dnode)) {
        return(.nl_normalise_weights_safe(log_marginal, "outer grid",
                                          log_quad = log_quad))
    }
    fin <- is.finite(log_marginal)
    if (!any(fin)) return(rep(NA_real_, length(log_marginal)))
    w <- dnode * exp(log_marginal - max(log_marginal[fin]))
    w[!is.finite(w)] <- 0
    w[w < 0] <- 0
    s <- sum(w)
    if (s <= 0) return(rep(NA_real_, length(log_marginal)))
    w / s
}
