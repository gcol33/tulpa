# Outer Pareto-k-hat for the joint nested-Laplace backend.
#
# Part of the joint nested-Laplace driver; the public entry point
# tulpa_nested_laplace_joint() lives in nested_laplace_joint.R.
#
# The re-cov / generic-single-axis / SPDE paths importance-sample a single
# positive-scale axis on the log scale (see R/psis.R, R/fit_spde_nested.R).
# The joint backend's hyperparameter space is heterogeneous -- positive
# scales (sigma, tau, phi_*, ...), a bounded BYM2 mixing weight (rho), an
# unbounded copy coefficient (alpha), and a CAR_proper correlation
# (rho_car) whose support is the adjacency's eigenvalue interval and is not
# safely guessable. Each axis therefore carries its own unconstraining
# transform + log-Jacobian; a fit carrying any axis whose support is not
# safely known DECLINES to the quadrature-ESS fallback rather than apply a
# guessed transform (never a wrong k-hat). This mirrors the generic path's
# decline on a bounded `rho_grid` (gcol33/tulpa#42).

# Capped inner-Newton budget for the outer Pareto-k diagnostic's re-evaluation
# solves (gcol33/tulpa#51). Warm-started from the modal latent mode, a draw at a
# plausible hyperparameter converges in a few steps; a draw at an implausible
# one (where the cold Newton would stall to the fit's full `max_iter`) carries
# negligible importance weight, so capping it bounds the diagnostic cost without
# moving the k-hat. The effective cap is `min(max_iter, .K_DIAG_MAX_ITER)`.
.K_DIAG_MAX_ITER <- 25L

# Maximum moment-matching refinement passes for the outer Pareto-k proposal
# (gcol33/tulpa#119). Each pass re-estimates the proposal from the PSIS-weighted
# moments of its own draws and re-scores; iteration stops early once the k-hat
# reaches the usable band, so a proposal that already fits pays a single pass.
.K_DIAG_MM_MAX <- 5L

# Shamanskii (chord) factor-reuse interval for the diagnostic re-solves
# (gcol33/tulpa#118). Profiling the joint occu_cover diagnostic showed the
# dominant cost is NOT the sparse Cholesky factorize (~8-12%, flat ~0.5 ms up to
# ~1100 cells) but the per-Newton-iteration Hessian/gradient SCATTER (73-83%) --
# the beta cover arm's per-observation digamma/trigamma curvature fill, paid on
# every step of every importance draw. With `inner_refresh = m > 1` a reuse
# iteration re-applies the cached factor to a refreshed gradient AND scatters
# `grad_only` (the solver skips the curvature fill), so reuse attacks the
# scatter cost, not just the factorize. The final mode-pass always re-factorizes
# with the true Hessian, so the log-marginal -- and thus the k-hat -- is
# unchanged; only the path to the mode uses a stale curvature. The diagnostic
# only needs the converged log-marginal (no per-draw SEs), so the stale-curvature
# path is harmless here even where the fit itself keeps refresh = 1.
.K_DIAG_REFRESH <- 4L

# Loosened inner-Newton convergence tolerance for the diagnostic re-solves
# (gcol33/tulpa#118). Profiling showed a large share of the per-draw Newton
# steps is intrinsic convergence to the FIT's tol (~1e-6), not warm-start drift
# -- and the diagnostic does not need that accuracy. The Laplace log-marginal
# error from stopping at gradient norm ~ t is O(t^2) (the mode sits at a
# near-stationary point, so the objective is flat there), immaterial to the
# tail-shape k-hat the diagnostic reports. A 1e-4 inner tol therefore cuts the
# step count with no measurable k-hat shift (verified vs the 1e-6 path), and
# composes with the near-neighbour warm start + Shamanskii reuse. Never TIGHTER
# than the fit's own tol (a fit run looser than this keeps its own).
.K_DIAG_TOL <- 1e-4

# Resolve the diagnostic's speed knobs (gcol33/tulpa#118). The fast values are
# the defaults; each is overridable via an option so a power user (or a test)
# can request the byte-for-byte "exact" diagnostic -- refresh = 1, the fit's own
# tol, no batch re-order -- and confirm the fast path's k-hat matches it.
#   tulpa.kdiag.refresh : Shamanskii reuse interval (default .K_DIAG_REFRESH; 1 disables)
#   tulpa.kdiag.tol     : inner-Newton tol floor    (default .K_DIAG_TOL)
#   tulpa.kdiag.reorder : near-neighbour chain order (default TRUE)
.kdiag_knobs <- function() {
    list(
        refresh = as.integer(getOption("tulpa.kdiag.refresh", .K_DIAG_REFRESH)),
        tol     = as.numeric(getOption("tulpa.kdiag.tol",     .K_DIAG_TOL)),
        reorder = isTRUE(getOption("tulpa.kdiag.reorder", TRUE)),
        # Per-cell warm start: each importance draw's inner re-solve starts from
        # the converged latent mode of its NEAREST integration cell instead of
        # the single broadcast modal mode (gcol33/tulpa#118). Strictly better
        # than the chain re-order AND works in the parallel pilot-mode path, so
        # it supersedes the re-order when stored grid modes are available.
        percell = isTRUE(getOption("tulpa.kdiag.percell", TRUE))
    )
}

# Per-draw nearest-grid-mode warm start (gcol33/tulpa#118). For each row of
# `theta_mat` (importance draws in user-facing theta space), find the nearest
# integration-grid cell in STANDARDISED theta space and return that cell's
# stored converged latent mode. The `[nrow(theta_mat) x n_x]` result is consumed
# by the kernel row-aligned as the per-cell warm start, so a draw starts from a
# near-mode rather than the single broadcast modal mode -- and the parallel
# pilot-mode path benefits too. Returns NULL when modes / grid are unavailable or
# shaped wrong (caller falls back to the broadcast mode + chain re-order).
.joint_nearest_grid_mode <- function(theta_mat, res) {
    tg    <- res$theta_grid
    modes <- res$modes
    if (is.null(tg) || !is.matrix(tg) || is.null(modes) || !is.matrix(modes)) {
        return(NULL)
    }
    if (nrow(modes) != nrow(tg)) return(NULL)
    d <- ncol(tg)
    if (is.null(dim(theta_mat)) || ncol(theta_mat) != d) return(NULL)
    sds <- apply(tg, 2L, stats::sd)
    sds[!is.finite(sds) | sds <= 0] <- 1
    Gt <- t(sweep(tg, 2L, sds, `/`))                # d x K standardised grid
    Q  <- sweep(theta_mat, 2L, sds, `/`)            # S x d standardised draws
    idx <- vapply(seq_len(nrow(Q)), function(s) {
        which.min(colSums((Gt - Q[s, ])^2))
    }, integer(1))
    modes[idx, , drop = FALSE]
}

# Are per-cell grid modes available to drive the nearest-grid-mode warm start?
.joint_has_grid_modes <- function(res) {
    tg <- res$theta_grid
    m  <- res$modes
    is.matrix(tg) && is.matrix(m) && nrow(m) == nrow(tg) && nrow(m) >= 1L
}

# Build the diagnostic re-evaluation closure (gcol33/tulpa#118), choosing the
# warm-start strategy by knob + availability, single source for both the single-
# and multi-block paths. `solve_fn(theta_mat, x_init_per_cell = NULL)` round-trips
# a batch through the kernel (baked hyperprior included).
#   1. per-cell  : each draw warm-started from its nearest stored grid mode
#                  (best; serial AND parallel). Default when modes are stored.
#   2. re-order  : near-neighbour chain so the serial driver's previous-cell
#                  warm start is a near neighbour (no per-cell modes).
#   3. plain     : single broadcast modal mode (both knobs off).
.joint_make_diag_refit <- function(res, solve_fn, modal_theta, knobs) {
    if (isTRUE(knobs$percell) && .joint_has_grid_modes(res)) {
        function(theta_mat) {
            wpc <- .joint_nearest_grid_mode(theta_mat, res)
            solve_fn(theta_mat, x_init_per_cell = wpc)   # NULL -> broadcast
        }
    } else if (isTRUE(knobs$reorder)) {
        function(theta_mat) .joint_is_solve_reordered(theta_mat, modal_theta, solve_fn)
    } else {
        function(theta_mat) solve_fn(theta_mat)
    }
}

# Greedy nearest-neighbour visiting order for the importance batch
# (gcol33/tulpa#118). The diagnostic's `k_samples` draws come off the Gaussian
# proposal in random order; the serial outer-grid driver warm-starts each cell
# from the PREVIOUS cell's converged mode, so a random order means every draw
# starts from a random-neighbour mode and the inner Newton pays many steps
# (measured 8-16/draw). Re-ordering the batch into a near-neighbour chain (each
# draw adjacent to a close one in hyperparameter space) makes the chained
# warm-start a genuine near-neighbour, so the inner Newton corrects only the
# small drift between adjacent draws -- the same principle the lattice grid
# already uses (flat-order chaining along its fastest axis). Columns are
# standardised so axes of different scale weigh equally; the chain is seeded at
# the draw nearest `center` (the modal grid cell, where the broadcast warm mode
# is the converged mode). O(S^2 d) -- trivial at S ~ 200. Returns a permutation
# of `seq_len(nrow(theta_mat))`.
.joint_is_chain_order <- function(theta_mat, center = NULL) {
    S <- nrow(theta_mat)
    d <- ncol(theta_mat)
    if (S <= 2L) return(seq_len(S))
    sds <- apply(theta_mat, 2L, stats::sd)
    sds[!is.finite(sds) | sds <= 0] <- 1
    Z <- sweep(theta_mat, 2L, sds, `/`)              # S x d, standardised
    tZ <- t(Z)                                        # d x S for column ops
    c0 <- if (!is.null(center) && length(center) == d) as.numeric(center) / sds
          else colMeans(Z)
    start <- which.min(colSums((tZ - c0)^2))
    visited <- logical(S)
    ord <- integer(S)
    cur <- start
    for (i in seq_len(S)) {
        ord[i] <- cur
        visited[cur] <- TRUE
        if (i == S) break
        dvec <- colSums((tZ - Z[cur, ])^2)
        dvec[visited] <- Inf
        cur <- which.min(dvec)
    }
    ord
}

# Run an importance-batch solve through the near-neighbour chain order, then
# restore the caller's row order (gcol33/tulpa#118). `solve_fn(theta_mat)`
# returns one log-marginal per row of its argument (the existing refit closure,
# baked hyperprior included); this wrapper feeds it the re-ordered batch and
# un-permutes the result, so the chained warm-start sees near neighbours while
# the PSIS layer above is unaffected. A length mismatch (kernel failure) is
# passed through verbatim for the caller's own guard.
.joint_is_solve_reordered <- function(theta_mat, center, solve_fn) {
    S <- nrow(theta_mat)
    if (S <= 2L) return(solve_fn(theta_mat))
    ord    <- .joint_is_chain_order(theta_mat, center)
    lm_ord <- solve_fn(theta_mat[ord, , drop = FALSE])
    if (length(lm_ord) != S) return(lm_ord)
    out <- numeric(S)
    out[ord] <- lm_ord
    out
}

# Modal grid cell in user-facing theta space, the chain seed for
# .joint_is_solve_reordered: the same highest-weight cell whose converged latent
# mode .joint_modal_mode broadcasts as the warm start, so the re-ordered batch's
# first cell sits where that warm mode is exact. Returns NULL when the grid /
# weights are unusable (caller falls back to centroid seeding).
.joint_modal_theta <- function(res) {
    tg <- res$theta_grid
    w  <- res$weights
    if (is.null(tg) || !is.matrix(tg) || is.null(w) ||
        length(w) != nrow(tg) || !any(is.finite(w))) {
        return(NULL)
    }
    as.numeric(tg[which.max(w), ])
}

# Modal-cell latent mode for warm-starting the diagnostic solves: the converged
# inner mode at the highest-weight grid cell. Broadcast as the `x_init` for
# every re-evaluation draw (the bulk of the importance weight sits near the
# posterior mode, so those draws warm-start well). Returns NULL when modes were
# not stored, so the caller falls back to the kernel's cold default.
.joint_modal_mode <- function(res) {
    modes <- res$modes
    w     <- res$weights
    if (is.null(modes) || !is.matrix(modes) || is.null(w) ||
        length(w) != nrow(modes) || !any(is.finite(w))) {
        return(NULL)
    }
    m <- as.numeric(modes[which.max(w), ])
    if (length(m) == 0L || any(!is.finite(m))) return(NULL)
    m
}

# Outer-thread width for the diagnostic's importance batch.
#
# The batch is `k_samples` INDEPENDENT inner re-solves run after the grid
# integration has finished, so every core the fit used is free. Each point is
# solved single-threaded: once the batch saturates the outer pool the inner
# reduction collapses to one thread (joint_inner_thread_budget, n_grid >=
# n_outer), so the per-point log-marginal -- and thus the k-hat -- is identical
# to the serial path regardless of how many run concurrently. Widening the outer
# pool is therefore a pure wall-clock speedup with an unchanged diagnostic.
#
# `k_threads` resolves the width:
#   * NULL (default) -- follow the fit's own thread grant: the larger of
#     `n_threads_outer` and the inner `n_threads`. A serial fit keeps a serial
#     diagnostic (no surprise oversubscription when the caller is itself forking
#     per-species fits across cores), while a threaded fit gets a free parallel
#     diagnostic. The inner threads are re-purposed here because the batch's
#     independent points parallelise better across the outer loop than the
#     per-observation reduction does.
#   * "auto" -- the physical performance-core count (.tulpa_perf_cores(),
#     hybrid-CPU aware), floored at the thread grant. For a single serial fit
#     that wants the diagnostic on every core with one setting. Capped at 2 under
#     R CMD check so examples / tests honour the CRAN core limit.
#   * an integer >= 1 -- that exact width (1 forces serial).
# Always capped at `k_samples` (no point holding more threads than draws).
.tulpa_pareto_k_threads <- function(n_threads_outer, n_threads, k_samples,
                                    k_threads = NULL) {
    ks         <- max(1L, as.integer(k_samples))
    auto_grant <- max(1L, as.integer(n_threads_outer), as.integer(n_threads))

    if (is.null(k_threads)) {
        w <- auto_grant
    } else if (is.character(k_threads) && length(k_threads) == 1L &&
               identical(tolower(k_threads), "auto")) {
        cores <- .tulpa_perf_cores()
        if (is.na(cores) || cores < 1L) cores <- auto_grant
        chk <- Sys.getenv("_R_CHECK_LIMIT_CORES_", "")
        if (nzchar(chk) && !(tolower(chk) %in% c("false", "0")))
            cores <- min(cores, 2L)
        w <- max(auto_grant, cores)
    } else {
        kt <- suppressWarnings(as.integer(k_threads))
        if (length(kt) != 1L || is.na(kt) || kt < 1L) {
            stop("`control$k_threads` must be a single integer >= 1, or \"auto\".",
                 call. = FALSE)
        }
        w <- kt
    }
    as.integer(min(w, ks))
}

# Positive-scale axes integrated on the log scale (theta = exp(u), Jacobian
# d theta / d u = theta, log-Jacobian u). Per-arm dispersion axes carry the
# `phi_` prefix and join this set by name.
.JOINT_POS_AXES <- c("sigma", "tau", "sigma2", "phi_gp", "ell", "lengthscale",
                     "range", "sigma_spatial", "sigma_occ", "sigma_pos",
                     "sigma_1", "sigma_2")

# Per-axis transform tag for one block. `type` is the (lower-case) block type,
# `axes` the bare axis names (block prefix already stripped). Returns one tag
# per axis ("log" / "logit01" / "identity") or NULL to DECLINE the whole fit
# when any axis has a support that cannot be safely transformed:
#   * `rho` is the BYM2 mixing weight in (0, 1) -> logit, but for CAR_proper /
#     AR1 / multi-output HSGP the same name denotes an eigenvalue- or
#     autocorrelation-bounded parameter whose support is not (0, 1); decline.
#   * `rho_car` is the proper-CAR correlation on the adjacency eigenvalue
#     interval; decline.
#   * `alpha` is the unbounded copy coefficient; identity.
#   * `mcar` axes are the log-Cholesky coordinates of Sigma = L L' (log L_ii on
#     the diagonal, raw strict-lower L_ij), already unconstrained on all of R,
#     so every axis is identity (zero Jacobian). This is what lets the joint CCD
#     mode-centre the Sigma grid (and the outer Pareto-k score it) rather than
#     decline to the fixed log-Cholesky tensor.
.joint_pareto_block_tags <- function(type, axes) {
    # mcar / miid axes are the log-Cholesky coordinates of Sigma = L L' (log L_ii
    # on the diagonal, raw strict-lower L_ij), unconstrained on all of R, so
    # every axis is identity (zero Jacobian).
    if (type %in% c("mcar", "miid")) return(rep("identity", length(axes)))
    tag_one <- function(a) {
        if (a == "alpha") return("identity")
        if (a == "rho") return(if (identical(type, "bym2")) "logit01" else NA_character_)
        if (a == "rho_car") return(NA_character_)
        if (a %in% .JOINT_POS_AXES || startsWith(a, "phi_")) return("log")
        NA_character_
    }
    tags <- vapply(axes, tag_one, character(1))
    if (anyNA(tags)) return(NULL)
    tags
}

# Resolve the per-column unconstraining transforms for a joint result.
# Walks the result's block layout (multi-block: `axis_offsets` + `blocks`;
# single-block: the lone `prior$type`) so the `rho` ambiguity above is
# resolved by the block that owns the axis, not by name alone. Trailing
# `phi_<arm>` dispersion columns (appended after the latent-block axes) are
# positive-scale (log). Returns a character vector of tags, one per
# `theta_grid` column, or NULL to decline.
.joint_pareto_axis_tags <- function(res) {
    tg <- res$theta_grid
    if (is.null(tg) || !is.matrix(tg) || ncol(tg) == 0L) return(NULL)
    cn <- colnames(tg)
    d  <- ncol(tg)
    tags <- rep(NA_character_, d)

    bare <- function(idx) sub("^b[0-9]+\\.", "", cn[idx])

    if (!is.null(res$axis_offsets) && !is.null(res$blocks)) {
        ao <- as.integer(res$axis_offsets)
        B  <- length(res$blocks)
        for (b in seq_len(B)) {
            if (ao[b + 1L] <= ao[b]) next                  # block carries no axis
            cols <- (ao[b] + 1L):ao[b + 1L]
            t_b <- .joint_pareto_block_tags(tolower(res$blocks[[b]]$type %||% ""),
                                            bare(cols))
            if (is.null(t_b)) return(NULL)
            tags[cols] <- t_b
        }
        # Columns past the last block axis are per-arm phi dispersion axes.
        if (ao[B + 1L] < d) {
            extra <- (ao[B + 1L] + 1L):d
            if (!all(startsWith(cn[extra], "phi_"))) return(NULL)
            tags[extra] <- "log"
        }
    } else {
        # Single-block: every non-phi column belongs to the lone prior block;
        # phi_<arm> columns are positive-scale dispersion axes.
        type <- tolower(res$prior$type %||% "")
        is_phi <- startsWith(cn, "phi_")
        if (any(!is_phi)) {
            t_blk <- .joint_pareto_block_tags(type, bare(which(!is_phi)))
            if (is.null(t_blk)) return(NULL)
            tags[!is_phi] <- t_blk
        }
        tags[is_phi] <- "log"
    }
    if (anyNA(tags)) return(NULL)
    tags
}

# Forward (constrained -> unconstrained) transform for one axis.
.joint_pareto_fwd <- function(tag, theta) {
    switch(tag,
        log      = log(theta),
        logit01  = stats::qlogis(theta),
        identity = theta)
}

# Inverse (unconstrained -> constrained) transform plus log|d theta / d u|.
.joint_pareto_inv <- function(tag, u) {
    switch(tag,
        log = list(theta = exp(u), log_jac = u),
        logit01 = {
            p <- stats::plogis(u)
            # log p + log(1 - p), stable form via plogis(., log.p = TRUE).
            list(theta = p,
                 log_jac = stats::plogis(u, log.p = TRUE) +
                           stats::plogis(-u, log.p = TRUE))
        },
        identity = list(theta = u, log_jac = rep(0, length(u))))
}

# Axes carrying posterior spread under a hyperparameter covariance: those with
# non-negligible variance. The importance proposal is built only over these and
# holds the zero-variance axes fixed at their mean -- a copy `alpha` pinned at 0
# or a one-point dispersion grid leaves the covariance rank deficient, so the
# proposal Cholesky would otherwise be undefined (gcol33/tulpa#114). `cov` is a
# hyperparameter covariance in the unconstrained coordinate. Returns the varying
# indices.
.joint_pareto_vary_axes <- function(cov) {
    ax_var  <- diag(cov)
    var_tol <- 1e-10 * max(ax_var, 0)
    which(ax_var > var_tol)
}

# Axes the integration GRID offers more than one value along -- the axes that
# CAN carry posterior spread, regardless of how the weight concentrates
# (gcol33/tulpa#117). The collapsed-grid mode-Hessian fallback differences the
# outer target only over these, holding the rest fixed. Unlike
# .joint_pareto_vary_axes, which reads a covariance, this reads the grid layout:
# a covariance cannot tell a genuinely pinned axis (one grid value) from a
# multi-valued one merely collapsed by a sharp posterior (a delta weight zeros
# the weighted variance of every axis, including ones whose FD curvature the
# rescue must still recover). An axis with a single distinct grid value is
# pinned (a copy alpha fixed at 0, a one-point dispersion grid); its FD curvature
# is zero, so a full-axis stencil would be singular. Reads the spread of the raw
# grid values (unweighted), so it is invariant to the weight concentration that
# triggers the fallback in the first place. Returns the varying indices.
.joint_pareto_grid_vary_axes <- function(theta_grid) {
    ax_var <- apply(theta_grid, 2L, stats::var)
    ax_var[!is.finite(ax_var)] <- 0
    var_tol <- 1e-10 * max(ax_var, 0)
    which(ax_var > var_tol)
}

# Laplace-at-mode covariance of the joint hyperparameter posterior, from a
# finite-difference Hessian of the outer target at the modal grid cell
# (gcol33/tulpa#116). The importance proposal for the outer Pareto-k is normally
# the grid-weighted covariance; when the posterior is sharp the (tensor) grid
# concentrates on too few cells to estimate it, and the residual far-cell weight
# yields a degenerate covariance and a spurious k-hat. The CCD integrator
# sidesteps this by carrying its own mode-Hessian proposal, but the tensor path
# has none -- so reconstruct it here by differencing the same inner re-solve the
# k-hat already calls. Reuses the CCD stencil / conditioning helpers (single
# source of truth with `.joint_ccd_grid`). `u_center` is the modal cell in the
# unconstrained coordinate; `col_names` labels the physical theta columns the
# re-solve expects.
#
# `vary` (gcol33/tulpa#117) restricts the FD stencil to the axes carrying
# posterior spread, holding the pinned axes (single grid value: a copy `alpha`
# fixed at 0, a one-point dispersion grid) fixed at `u_center`. A
# full-`d` stencil is singular along a pinned axis, so without this the Hessian
# is rejected and the fallback the rescue targets never engages. Excluding a
# pinned axis from the curvature is exact (it carries no posterior spread), not
# an approximation; the returned covariance is the inverse over `vary` embedded
# block-diagonally into a full `d x d` with zeros on the pinned rows / cols, so
# the caller's downstream varying-axis logic recovers the same set. Returns
# NULL when the varying-axis curvature is degenerate (the caller then keeps the
# grid-weighted estimate) or when every axis is pinned.
.joint_pareto_mode_cov <- function(u_center, tags, col_names,
                                   refit_log_marginal, d, vary = NULL,
                                   h = 0.1) {
    if (is.null(vary)) vary <- seq_len(d)
    if (length(vary) == 0L) return(NULL)

    # Full-d outer target: back-transform + summed per-axis log-Jacobian.
    target <- function(U) {
        theta_mat <- matrix(0, nrow(U), d)
        log_jac   <- numeric(nrow(U))
        for (j in seq_len(d)) {
            iv <- .joint_pareto_inv(tags[j], U[, j])
            theta_mat[, j] <- iv$theta
            log_jac <- log_jac + iv$log_jac
        }
        colnames(theta_mat) <- col_names
        lm <- refit_log_marginal(theta_mat)
        if (length(lm) != nrow(U)) return(rep(-Inf, nrow(U)))
        lm + log_jac
    }
    # Reduced target over the varying axes: embed an [S x |vary|] block into the
    # full d-space at u_center (pinned axes held fixed), then call the full
    # target. The stencil thus differences only the varying subspace.
    target_v <- function(Uv) {
        U <- matrix(u_center, nrow(Uv), d, byrow = TRUE)
        U[, vary] <- Uv
        target(U)
    }
    st <- tryCatch(.joint_ccd_fd_stencil(u_center[vary], target_v,
                                         rep(h, length(vary))),
                   error = function(e) NULL)
    if (is.null(st) || any(!is.finite(st$hess))) return(NULL)
    if (!.joint_ccd_outer_hess_ok(st$hess)) return(NULL)
    neg_H <- -0.5 * (st$hess + t(st$hess))                # precision at the mode
    cov_v <- tryCatch(solve(neg_H), error = function(e) NULL)
    if (is.null(cov_v) || any(!is.finite(cov_v))) return(NULL)

    cov <- matrix(0, d, d)
    cov[vary, vary] <- (cov_v + t(cov_v)) / 2             # pinned rows/cols zero
    cov
}

# Shared preparation for the outer Pareto-k-hat of a joint nested-Laplace
# result. Fits a Gaussian proposal to the joint hyperparameter posterior in the
# per-axis unconstrained coordinate `u` (weighted mean `u_hat` + covariance `Su`
# over the integration grid, the analogue of `.nested_grid_pareto_k` generalised
# to mixed support), splices the CCD mode-Hessian `proposal` (gcol33/tulpa#116)
# over the axes it spans, and engages the delta-collapse FD rescue
# (gcol33/tulpa#116, #117, #119). Returns the proposal summary the scorer draws
# from, or NULL to DECLINE (an axis with unguessable support, an unusable grid /
# weight vector, a sub-floor sample budget). The joint k and the opt-in per-arm
# k (gcol33/tulpa#120) score this SAME (u_hat, Su) summary, differing only in
# which axes are allowed to vary -- so the grid build, proposal splice and rescue
# are computed once here.
.joint_pareto_prepare <- function(res, refit_log_marginal, n_samples, proposal) {
    tags <- .joint_pareto_axis_tags(res)
    if (is.null(tags)) return(NULL)

    # Decline before any inner solve when the sample budget cannot reach the
    # GPD-fit floor (gcol33/tulpa#51): a sub-floor `n_samples` would run every
    # one of its solves and then discard the result as NA. The shared core
    # short-circuits, but catching it here too keeps the costly forward/inverse
    # transform + proposal fit off the table.
    if (as.integer(n_samples) < .PSIS_MIN_EVAL) return(NULL)

    tg <- res$theta_grid
    w  <- res$weights
    if (is.null(w) || length(w) != nrow(tg) || !is.finite(sum(w)) || sum(w) <= 0) {
        return(NULL)
    }
    cn <- colnames(tg)
    d  <- ncol(tg)

    u_grid <- matrix(0, nrow(tg), d)
    for (j in seq_len(d)) {
        u_grid[, j] <- .joint_pareto_fwd(tags[j], as.numeric(tg[, j]))
    }
    if (any(!is.finite(u_grid))) return(NULL)

    u_hat <- as.numeric(crossprod(w, u_grid))
    cen   <- sweep(u_grid, 2L, u_hat)
    Su    <- crossprod(cen * w, cen)
    Su    <- (Su + t(Su)) / 2

    # Splice the CCD mode-Hessian proposal over the latent axes it spans
    # (gcol33/tulpa#116). The grid-weighted `Su` above is the spread of the
    # integration nodes; a sharp hyperparameter posterior concentrates the grid
    # on ~1 cell, collapsing `Su` toward 0 and leaving the proposal degenerate
    # even though the fit is fine. The CCD integrator already built a Gaussian
    # from the analytic curvature at the outer mode (and placed its design with
    # it); reuse that covariance here. The block is independent of the
    # tensor-crossed phi axes, so the override is block-diagonal: Hessian
    # covariance on the CCD axes, grid-weighted covariance retained on the rest.
    proposal_source   <- "grid_moment"
    used_mode_hessian <- FALSE
    if (is.list(proposal) && is.numeric(proposal$u_hat) &&
        is.matrix(proposal$L_scale) &&
        length(proposal$u_hat) == nrow(proposal$L_scale) &&
        nrow(proposal$L_scale) == ncol(proposal$L_scale)) {
        cols <- proposal$cols %||% seq_along(proposal$u_hat)
        consistent <- length(cols) == length(proposal$u_hat) &&
            all(is.finite(cols)) && min(cols) >= 1L && max(cols) <= d &&
            (is.null(proposal$tags) || identical(tags[cols], proposal$tags)) &&
            all(is.finite(proposal$u_hat)) && all(is.finite(proposal$L_scale))
        if (consistent) {
            Sig <- proposal$L_scale %*% t(proposal$L_scale)
            u_hat[cols]    <- proposal$u_hat
            Su[cols, ]     <- 0
            Su[, cols]     <- 0
            Su[cols, cols] <- Sig
            Su             <- (Su + t(Su)) / 2
            used_mode_hessian <- TRUE
        }
    }

    # Delta-collapse fallback (gcol33/tulpa#116, #117, #119). When the grid
    # weight concentrates on a single cell the grid-weighted `Su` is exactly zero
    # on every axis: no axis carries posterior spread, so the proposal covariance
    # cannot be estimated from the nodes. Reconstruct a Laplace-at-mode covariance
    # from a finite-difference Hessian of the outer target at the modal cell,
    # restricted to the grid-layout-varying axes (a single-value axis -- a copy
    # `alpha` fixed at 0, a one-point dispersion grid -- has zero FD curvature and
    # would make the full-axis stencil singular). This engages ONLY at genuine
    # degeneracy (no grid-weighted spread on any axis). When ANY axis still
    # carries weighted spread, the grid-weighted `Su` is the actual posterior
    # spread and is kept: the FD mode curvature is the LOCAL curvature at the
    # mode, which on a non-Gaussian outer marginal (flat-topped, sharper
    # drop-off) over-widens the proposal and scatters importance draws to extreme
    # hyperparameters where the inner Laplace log-marginal inflates, so a single
    # draw dominates the weights (k-hat -> large, IS-ESS -> 1) even though the
    # integration is well resolved on that axis.
    if (!used_mode_hessian && length(.joint_pareto_vary_axes(Su)) == 0L) {
        vary_g <- .joint_pareto_grid_vary_axes(tg)
        u_mode <- as.numeric(u_grid[which.max(w), ])
        cov_h  <- .joint_pareto_mode_cov(u_mode, tags, cn,
                                         refit_log_marginal, d, vary = vary_g)
        if (!is.null(cov_h)) {
            u_hat <- u_mode
            Su    <- cov_h
            proposal_source   <- "mode_hessian"
            used_mode_hessian <- TRUE
        }
    } else if (used_mode_hessian) {
        proposal_source <- "mode_hessian"
    }

    list(tags = tags, u_grid = u_grid, u_hat = u_hat, Su = Su, cn = cn, d = d,
         proposal_source = proposal_source)
}

# Score the outer Pareto-k over a chosen set of varying axes `vary`, holding the
# rest fixed at `u_hat`. The shared scorer behind both the joint k (all
# genuinely-varying axes) and the per-arm k (one arm's axes; gcol33/tulpa#120).
# `prep` is the `.joint_pareto_prepare` summary, `refit_log_marginal` the inner
# re-solve closure (round-trips an `S x d` user-facing theta matrix through the
# SAME joint kernel the integrator used, returning the per-cell log-marginal WITH
# the baked hyperprior). The proposal's normalizing constant is common to every
# draw and drops under PSIS, so only the quadratic enters (handled inside
# `.nested_is_pareto_k`). Does NOT manage the RNG -- the driver saves / restores
# it once around all scoring so the fit's draws are bit-for-bit unchanged.
#
# Importance-sampling k-hat with moment-matching refinement (gcol33/tulpa#119,
# after Paananen, Piironen, Burkner & Vehtari 2021, Stat. Comput. 31:16). The
# initial proposal is the integration-node covariance (or the mode-Hessian / CCD
# curvature). When the grid is sharply concentrated that covariance is estimated
# from few effective cells and can mis-scale the proposal -- too wide scatters
# draws to extreme hyperparameters where the inner Laplace log-marginal inflates,
# too narrow leaves the target tail uncovered -- so the k-hat reads unreliable
# even though the fit is fine. Re-estimate the proposal from the PSIS-smoothed
# importance-weighted moments of its own draws and re-score, keeping the
# lowest-k-hat proposal; the smoothed weights bound any single draw's influence,
# so a sharp posterior is matched in a couple of passes. Iteration stops once the
# k-hat reaches the usable band (<= 0.7); a proposal that already fits pays a
# single pass. The per-pass radius cap (gcol33/tulpa#94) is recomputed for the
# current proposal so the grid-coverage envelope follows it. Returns
# list(pareto_k, is_ess, refined) or NULL (empty / rank-deficient `vary`, no
# finite importance draw).
.joint_pareto_score <- function(prep, vary, refit_log_marginal, n_samples) {
    if (length(vary) == 0L) return(NULL)
    Su <- prep$Su; u_hat <- prep$u_hat; u_grid <- prep$u_grid
    tags <- prep$tags; cn <- prep$cn; d <- prep$d

    Su_v <- Su[vary, vary, drop = FALSE]
    L_v  <- tryCatch(t(chol(Su_v)), error = function(e) NULL)
    if (is.null(L_v)) return(NULL)
    u_hat_v <- u_hat[vary]

    lt <- function(U_v) {
        S <- nrow(U_v)
        U <- matrix(u_hat, S, d, byrow = TRUE)   # axes outside `vary` at u_hat
        U[, vary] <- U_v
        theta_mat <- matrix(0, S, d)
        log_jac   <- numeric(S)
        for (j in seq_len(d)) {
            iv <- .joint_pareto_inv(tags[j], U[, j])
            theta_mat[, j] <- iv$theta
            log_jac <- log_jac + iv$log_jac
        }
        colnames(theta_mat) <- cn
        lm <- refit_log_marginal(theta_mat)
        if (length(lm) != S) return(rep(-Inf, S))
        lm + log_jac
    }

    u_grid_v <- u_grid[, vary, drop = FALSE]
    prop_u  <- u_hat_v
    prop_L  <- L_v
    best    <- NULL
    refined <- FALSE
    for (iter in seq_len(.K_DIAG_MM_MAX)) {
        rc <- .nested_grid_radius_cap(u_grid_v, prop_u, prop_L)
        kd <- tryCatch(.nested_is_pareto_k(prop_u, prop_L, lt, n_samples,
                                           radius_cap = rc, return_draws = TRUE),
                       error = function(e) NULL)
        if (is.null(kd) || !is.finite(kd$pareto_k)) break
        if (is.null(best) || kd$pareto_k < best$pareto_k)
            best <- list(pareto_k = kd$pareto_k, is_ess = kd$is_ess, refined = refined)
        if (kd$pareto_k <= 0.7 || iter == .K_DIAG_MM_MAX) break
        wts <- exp(kd$log_weights); sw <- sum(wts)
        if (!is.finite(sw) || sw <= 0 || nrow(kd$U) < length(prop_u) + 1L) break
        wts   <- wts / sw
        mu_w  <- as.numeric(crossprod(wts, kd$U))
        cen   <- sweep(kd$U, 2L, mu_w)
        Sig_w <- crossprod(cen * wts, cen); Sig_w <- (Sig_w + t(Sig_w)) / 2
        Lw    <- tryCatch(t(chol(Sig_w)), error = function(e) NULL)
        if (is.null(Lw)) break
        prop_u <- mu_w; prop_L <- Lw; refined <- TRUE
    }
    best
}

# Shared outer Pareto-k-hat driver for a joint nested-Laplace result. Prepares
# the proposal summary once, scores the joint k over every genuinely-varying
# axis, and -- when `arm_axes` is supplied (gcol33/tulpa#120) -- additionally
# scores a k restricted to each arm's hyperparameter axes (the rest held at the
# posterior mean `u_hat`). Runs all scoring with the RNG restored at the end so
# the fit's draws are bit-for-bit unchanged.
#
# Returns list(pareto_k, is_ess, proposal_source): all NA / NA-source when the
# fit declines (an axis with unguessable support, a degenerate proposal
# covariance, or too few finite importance draws), in which case the diagnostic
# layer reports quad-ESS. With `arm_axes`, also `by_arm_k` / `by_arm_is_ess`
# (named numeric over the arms) -- a per-arm k of NA means that arm carries no
# genuinely-varying axis.
.joint_pareto_k <- function(res, refit_log_marginal, n_samples = 200L,
                            proposal = NULL, arm_axes = NULL) {
    na_out <- list(pareto_k = NA_real_, is_ess = NA_real_,
                   proposal_source = NA_character_)
    prep <- .joint_pareto_prepare(res, refit_log_marginal, n_samples, proposal)
    if (is.null(prep)) return(na_out)

    has_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    old_seed <- if (has_seed) get(".Random.seed", envir = .GlobalEnv) else NULL
    on.exit(if (!is.null(old_seed))
                assign(".Random.seed", old_seed, envir = .GlobalEnv))

    # Axes the grid pins to a single value (e.g. a copy `alpha` fixed at 0, or a
    # one-point dispersion axis) carry zero weighted variance, leaving Su rank
    # deficient and its Cholesky undefined. The importance proposal holds them
    # fixed at u_hat and is built only on the varying axes (gcol33/tulpa#114).
    vary  <- .joint_pareto_vary_axes(prep$Su)
    joint <- .joint_pareto_score(prep, vary, refit_log_marginal, n_samples)
    if (is.null(joint)) return(na_out)
    src <- if (isTRUE(joint$refined)) "moment_matched" else prep$proposal_source
    out <- list(pareto_k = joint$pareto_k, is_ess = joint$is_ess,
                proposal_source = src)

    if (!is.null(arm_axes) && length(arm_axes) > 0L) {
        pa <- lapply(arm_axes, function(ax) {
            v <- intersect(vary, as.integer(ax))
            s <- if (length(v) == 0L) NULL else
                .joint_pareto_score(prep, v, refit_log_marginal, n_samples)
            if (is.null(s)) c(NA_real_, NA_real_)
            else c(s$pareto_k, s$is_ess)
        })
        out$by_arm_k      <- vapply(pa, function(z) z[[1L]], numeric(1))
        out$by_arm_is_ess <- vapply(pa, function(z) z[[2L]], numeric(1))
        names(out$by_arm_k)      <- names(arm_axes)
        names(out$by_arm_is_ess) <- names(arm_axes)
    }
    out
}

# Which arms does a latent block load on? A block enters arm `a`'s linear
# predictor iff its per-arm index for `a` is non-empty. The per-arm index lives
# under one of `obs_idx` / `spatial_idx` / `temporal_idx` on the (prepared)
# block spec -- a length-n_arms list (one integer vector per arm; an empty entry
# means the block does not load that arm) or a single vector replicated across
# every arm. A block with no per-arm index field loads on all arms. Returns a
# length-n_arms logical (gcol33/tulpa#120).
.joint_block_arms <- function(block, n_arms) {
    for (fld in c("spatial_idx", "temporal_idx", "obs_idx")) {
        idx <- block[[fld]]
        if (is.null(idx)) next
        if (is.list(idx) && length(idx) == n_arms) {
            return(vapply(idx, function(v) length(v) > 0L, logical(1)))
        }
        return(rep(TRUE, n_arms))            # single vector -> shared by all arms
    }
    rep(TRUE, n_arms)
}

# Map each joint hyperparameter axis (theta_grid column) to the arm(s) whose
# linear predictor it enters, for the opt-in per-arm outer Pareto-k
# (gcol33/tulpa#120). Returns a named list arm_name -> integer column indices, or
# NULL to DECLINE (single-block layout, < 2 arms, or fewer than two arms with any
# axis) so the per-arm diagnostic is simply withheld rather than reporting a
# mis-attributed k -- the same decline-rather-than-guess stance the joint axis
# tags take.
#
# Axis attribution:
#   * a latent block's axes -> the arms the block loads on (`.joint_block_arms`),
#     so a field shared across arms contributes to each arm it enters;
#   * a copy block's `alpha` (the cross-arm copy coefficient) -> the recipient
#     arm only, since it measures how strongly that arm reads the donor field;
#   * a trailing `phi_<arm>` dispersion column -> that arm, by name suffix.
.joint_pareto_arm_axes <- function(res) {
    tg <- res$theta_grid
    if (is.null(tg) || !is.matrix(tg) || ncol(tg) == 0L) return(NULL)
    # Per-arm decomposition is defined for the multi-block layout (b<b>. axis
    # names + axis_offsets + blocks). The single-block joint shares one field
    # across arms, so decline rather than mis-attribute its axes.
    if (is.null(res$axis_offsets) || is.null(res$blocks)) return(NULL)
    cn <- colnames(tg)
    n_arms <- res$arm_layout$n_arms %||% length(res$responses)
    if (is.null(n_arms) || n_arms < 2L) return(NULL)
    arm_names <- names(res$responses)
    if (is.null(arm_names) || length(arm_names) != n_arms) {
        arm_names <- paste0("arm", seq_len(n_arms))
    }

    ao <- as.integer(res$axis_offsets)
    B  <- length(res$blocks)
    if (length(ao) < B + 1L) return(NULL)

    # Resolve copy (donor block -> recipient arm) to attribute each copy `alpha`
    # to its recipient arm. Decline-safe: an unresolvable copy just leaves alpha
    # attributed to the arms its donor block loads on.
    cp <- tryCatch(.resolve_copy_multi(res$copy, res$responses, res$prior),
                   error = function(e) NULL)
    copy_blocks <- if (!is.null(cp) && isTRUE(cp$has_copy))
        as.integer(cp$copy_blocks_zero) + 1L else integer(0)
    copy_arms   <- if (!is.null(cp) && isTRUE(cp$has_copy))
        as.integer(cp$copy_arms_zero)   + 1L else integer(0)

    arm_cols <- replicate(n_arms, integer(0), simplify = FALSE)
    names(arm_cols) <- arm_names
    bare <- function(idx) sub("^b[0-9]+\\.", "", cn[idx])

    for (b in seq_len(B)) {
        if (ao[b + 1L] <= ao[b]) next                     # block carries no axis
        cols  <- (ao[b] + 1L):ao[b + 1L]
        loads <- .joint_block_arms(res$blocks[[b]], n_arms)
        pos   <- match(b, copy_blocks)
        recip <- if (!is.na(pos)) copy_arms[pos] else NA_integer_
        if (!is.na(recip)) loads[recip] <- TRUE           # recipient reads the field
        bc <- bare(cols)
        for (k in seq_along(cols)) {
            if (identical(bc[k], "alpha") && !is.na(recip)) {
                arm_cols[[recip]] <- c(arm_cols[[recip]], cols[k])
            } else {
                for (a in which(loads)) arm_cols[[a]] <- c(arm_cols[[a]], cols[k])
            }
        }
    }
    # Trailing per-arm dispersion columns by name suffix.
    for (a in seq_len(n_arms)) {
        pc <- which(cn == paste0("phi_", arm_names[a]))
        if (length(pc)) arm_cols[[a]] <- c(arm_cols[[a]], pc)
    }
    arm_cols <- lapply(arm_cols, function(v) sort(unique(v)))
    keep <- vapply(arm_cols, length, integer(1)) > 0L
    if (sum(keep) < 2L) return(NULL)
    arm_cols[keep]
}

# Attach the opt-in per-arm outer Pareto-k to a joint result (gcol33/tulpa#120).
# No-op when the driver computed no per-arm k (diagnostic off, single-block
# layout, or a declined per-arm map). Single source for both attach paths.
.joint_attach_by_arm_k <- function(res, kd) {
    if (is.null(kd$by_arm_k)) return(res)
    res$pareto_k_by_arm        <- kd$by_arm_k
    res$pareto_k_by_arm_is_ess <- kd$by_arm_is_ess
    res$pareto_k_by_arm_scope  <-
        "per-arm hyperparameter axes (other arms fixed at posterior mean)"
    res
}

# Single-block joint wrapper. Reuses the driver's already-built generic
# `kernel_fn` (the closure refinement passes drive, which round-trips a
# user-facing cell matrix through `backend$call_kernel`) and `hp_fn` (the
# baked-hyperprior closure) as the re-evaluation path, so no kernel-call
# machinery is duplicated. The diagnostic solves are warm-started from the
# modal latent mode and capped at `.K_DIAG_MAX_ITER` (gcol33/tulpa#51), the
# same cost bound as the multi-block path. The whole importance batch is
# re-solved in one `kernel_fn` call with `n_threads_outer` so the independent
# re-solves run concurrently across cores rather than one-at-a-time using all
# inner threads. Attaches `pareto_k` / `pareto_k_is_ess` / `pareto_k_scope`;
# with `diagnose_k = FALSE` the fields are present but NA.
.joint_attach_pareto_k_single <- function(res, kernel_fn, hp_fn,
                                          max_iter = 50L,
                                          diagnose_k = TRUE, k_samples = 200L,
                                          n_threads_outer = 1L,
                                          pareto_k_by_arm = FALSE) {
    res$pareto_k        <- NA_real_
    res$pareto_k_is_ess <- NA_real_
    res$pareto_k_scope  <- "outer (hyperparameter) Gaussian proposal"
    res$pareto_k_proposal_source <- NA_character_
    if (!isTRUE(diagnose_k)) return(res)

    warm        <- .joint_modal_mode(res)
    warm_arg    <- if (is.null(warm)) NULL else list(mode = warm)
    modal_theta <- .joint_modal_theta(res)
    k_max_iter  <- min(as.integer(max_iter), .K_DIAG_MAX_ITER)
    n_to        <- as.integer(n_threads_outer)
    knobs       <- .kdiag_knobs()

    # One kernel call over the importance batch, with Shamanskii factor reuse
    # (off-factor steps scatter grad-only) + per-cell warm start, both attacking
    # the dominant per-iteration scatter cost on the beta cover arm
    # (gcol33/tulpa#118). `x_init_per_cell` is an [S x n_x] warm matrix or NULL.
    solve_fn <- function(theta_mat, x_init_per_cell = NULL) {
        r  <- kernel_fn(theta_mat, warm_start = warm_arg,
                        max_iter_override = k_max_iter,
                        n_threads_outer = n_to,
                        inner_refresh_override = knobs$refresh,
                        tol_override = knobs$tol,
                        x_init_per_cell = x_init_per_cell)
        lm <- r$log_marginal
        if (!is.null(hp_fn)) {
            hp <- hp_fn(theta_mat)
            if (!is.null(hp) && length(hp) == length(lm)) lm <- lm + hp
        }
        lm
    }
    # Per-cell warm start (each draw from its nearest stored grid mode) is the
    # best start and works in serial AND parallel; it supersedes the chain
    # re-order. Fall back to the near-neighbour chain re-order when modes are
    # unavailable, else the plain broadcast warm (gcol33/tulpa#118).
    refit <- .joint_make_diag_refit(res, solve_fn, modal_theta, knobs)
    arm_axes <- if (isTRUE(pareto_k_by_arm)) .joint_pareto_arm_axes(res) else NULL
    kd <- .joint_pareto_k(res, refit, k_samples, arm_axes = arm_axes)
    res$pareto_k        <- kd$pareto_k
    res$pareto_k_is_ess <- kd$is_ess
    res$pareto_k_proposal_source <- kd$proposal_source
    res <- .joint_attach_by_arm_k(res, kd)
    res
}
