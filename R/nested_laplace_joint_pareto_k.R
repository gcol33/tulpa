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
# decline on a bounded `rho_grid`.

# Capped inner-Newton budget for the outer Pareto-k diagnostic's re-evaluation
# solves. Warm-started from the modal latent mode, a draw at a
# plausible hyperparameter converges in a few steps; a draw at an implausible
# one (where the cold Newton would stall to the fit's full `max_iter`) carries
# negligible importance weight, so capping it bounds the diagnostic cost without
# moving the k-hat. The effective cap is `min(max_iter, .K_DIAG_MAX_ITER)`.
.K_DIAG_MAX_ITER <- 25L

# Maximum moment-matching refinement passes for the outer Pareto-k proposal
#. Each pass re-estimates the proposal from the PSIS-weighted
# moments of its own draws and re-scores, keeping the lowest-k-hat proposal.
# Proposal refinement is a separate step from the bare diagnostic and is NOT under
# the diagnostic's cost target, so the cap is generous: a backstop against a
# runaway loop, not a cost throttle (the earlier cap of 3 was a cost
# throttle, lifted here). The loop self-limits well below this on a typical fit --
# it stops as soon as the k-hat reaches the usable band (a proposal that already
# fits pays a single pass) OR a refined pass fails to improve on the one it was
# estimated from -- so the extra budget is spent only on a stubborn k still above
# the usable band and still falling, which is exactly where more passes help.
.K_DIAG_MM_MAX <- 8L

# Internal proposal-loop threshold for the moment-matching early-stop: a proposal
# whose k-hat is at or below this is good enough to stop refining (Vehtari,
# Simpson, Gelman, Yao & Gabry 2024 usable band). This is a fixed loop control,
# distinct from the REPORTED reliability bands, which are sample-size dependent
# (`.ps_conf_bands` in R/psis.R).
.K_DIAG_USABLE <- 0.7

# Internal proposal-loop threshold for the grid-mixture skip: a
# single-Gaussian k-hat below this is already good, so the mixture rescue (which
# can only LOWER an inflated k) is skipped. A fixed loop control, distinct from
# the reported sample-size-dependent bands (`.ps_conf_bands` in R/psis.R).
.K_DIAG_GOOD <- 0.5

# Grid-mixture (basin) proposal for the outer Pareto-k on a spread tensor grid
#. The nested-Laplace engine represents the hyperparameter
# posterior as the WEIGHTED INTEGRATION GRID and draws hyperparameters from it (a
# grid cell ~ its weight, then that cell's latent Laplace), never from one
# continuous Gaussian. Scoring the outer Pareto-k against a single grid-moment
# Gaussian therefore validates a proposal the engine does not sample: on a skewed
# or multi-node hyperparameter posterior (which the grid covers through its
# nodes) the symmetric Gaussian underweights the off-mode mass, importance draws
# that land there carry runaway weights, and the k-hat reads unreliable even
# though the grid representation is fine. The faithful proposal is a defensive
# mixture of local Gaussian bumps centred at the grid cells and mixed by the grid
# weights, the smoothed form of what the integrator actually represents; it
# covers the skew by construction, so the weights stay bounded. Each bump's
# per-axis SD is `.K_DIAG_MIX_BW` times the largest adjacent grid gap on that
# axis (the scale over which that axis is resolved); cells below
# `.K_DIAG_MIX_FLOOR` of the peak weight are dropped. The bandwidth is
# overridable via `getOption("tulpa.kdiag.mix_bw")`.
.K_DIAG_MIX_BW    <- 0.5
.K_DIAG_MIX_FLOOR <- 1e-3

# Grid-coverage tolerance for adopting the grid-mixture over the single Gaussian
#. The mixture is confined to the grid's coordinate hull, so it
# cannot detect a target tail BEYOND the grid; the single Gaussian, whose tails
# extend past the grid, can. The dispatcher therefore adopts the mixture only when
# the single Gaussian's importance weight is essentially all INSIDE the grid hull
# (the grid covers the posterior, so the high single-Gaussian k-hat is a within-
# grid shape mismatch the mixture corrects): at most this fraction of the weight
# may sit outside. Above it the grid is too narrow, and the single Gaussian's
# higher k-hat is kept so that grid-width deficiency is still flagged. The "hull"
# is the kept-cell node range EXPANDED by `.K_DIAG_HULL_PAD` bump SDs per axis,
# i.e. the mixture's actual coverage: its edge bumps reach ~3 SD past the outer
# node, so a draw that far out is still covered (a near-edge dip below the lowest
# node is not "beyond the grid"); only mass past the bumps' reach counts as
# uncovered.
.K_DIAG_HULL_TOL <- 0.02
.K_DIAG_HULL_PAD <- 3

# Shamanskii (chord) factor-reuse interval for the diagnostic re-solves
#. Profiling the joint occu_cover diagnostic showed the
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
#. Profiling showed a large share of the per-draw Newton
# steps is intrinsic convergence to the FIT's tol (~1e-6), not warm-start drift
# -- and the diagnostic does not need that accuracy. The Laplace log-marginal
# error from stopping at gradient norm ~ t is O(t^2) (the mode sits at a
# near-stationary point, so the objective is flat there), immaterial to the
# tail-shape k-hat the diagnostic reports. A 1e-4 inner tol therefore cuts the
# step count with no measurable k-hat shift (verified vs the 1e-6 path), and
# composes with the near-neighbour warm start + Shamanskii reuse. Never TIGHTER
# than the fit's own tol (a fit run looser than this keeps its own).
.K_DIAG_TOL <- 1e-4

# Resolve the diagnostic's speed knobs. The fast values are
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
        # the single broadcast modal mode. Strictly better
        # than the chain re-order AND works in the parallel pilot-mode path, so
        # it supersedes the re-order when stored grid modes are available.
        percell = isTRUE(getOption("tulpa.kdiag.percell", TRUE))
    )
}

# Per-draw nearest-grid-mode warm start. For each row of
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

# Build the diagnostic re-evaluation closure, choosing the
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
#. The diagnostic's `k_samples` draws come off the Gaussian
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
# restore the caller's row order. `solve_fn(theta_mat)`
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
# proposal Cholesky would otherwise be undefined. `cov` is a
# hyperparameter covariance in the unconstrained coordinate. Returns the varying
# indices.
.joint_pareto_vary_axes <- function(cov) {
    ax_var  <- diag(cov)
    var_tol <- 1e-10 * max(ax_var, 0)
    which(ax_var > var_tol)
}

# Axes the integration GRID offers more than one value along -- the axes that
# CAN carry posterior spread, regardless of how the weight concentrates
#. The collapsed-grid mode-Hessian fallback differences the
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
#. The importance proposal for the outer Pareto-k is normally
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
# `vary` restricts the FD stencil to the axes carrying
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
# to mixed support), splices the CCD mode-Hessian `proposal`
# over the axes it spans, and engages the delta-collapse FD rescue
#. Returns the proposal summary the scorer draws
# from, or NULL to DECLINE (an axis with unguessable support, an unusable grid /
# weight vector, a sub-floor sample budget). The joint k and the opt-in per-arm
# k score this SAME (u_hat, Su) summary, differing only in
# which axes are allowed to vary -- so the grid build, proposal splice and rescue
# are computed once here.
.joint_pareto_prepare <- function(res, refit_log_marginal, n_samples, proposal) {
    tags <- .joint_pareto_axis_tags(res)
    if (is.null(tags)) return(NULL)

    # Decline before any inner solve when the sample budget cannot reach the
    # GPD-fit floor: a sub-floor `n_samples` would run every
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
    #. The grid-weighted `Su` above is the spread of the
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

    # Delta-collapse fallback. When the grid
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
         w = w, proposal_source = proposal_source)
}

# The outer target on the varying subspace `vary`: embed an [S x |vary|] block
# into the full d-space at `u_hat` (axes outside `vary` held at their posterior
# mean), back-transform each axis to user-facing theta with its log-Jacobian,
# and call the inner re-solve. Shared by the single-Gaussian scorer
# (.joint_pareto_score) and the grid-mixture scorer (.joint_pareto_score_mixture)
# so both evaluate the identical target.
.joint_pareto_make_lt <- function(prep, vary, refit_log_marginal) {
    u_hat <- prep$u_hat; tags <- prep$tags; cn <- prep$cn; d <- prep$d
    function(U_v) {
        S <- nrow(U_v)
        U <- matrix(u_hat, S, d, byrow = TRUE)
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
}

# Score the outer Pareto-k over a chosen set of varying axes `vary`, holding the
# rest fixed at `u_hat`. The shared scorer behind both the joint k (all
# genuinely-varying axes) and the per-arm k (one arm's axes).
# `prep` is the `.joint_pareto_prepare` summary, `refit_log_marginal` the inner
# re-solve closure (round-trips an `S x d` user-facing theta matrix through the
# SAME joint kernel the integrator used, returning the per-cell log-marginal WITH
# the baked hyperprior). The proposal's normalizing constant is common to every
# draw and drops under PSIS, so only the quadratic enters (handled inside
# `.nested_is_pareto_k`). Does NOT manage the RNG -- the driver saves / restores
# it once around all scoring so the fit's draws are bit-for-bit unchanged.
#
# Importance-sampling k-hat with moment-matching refinement (
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
# single pass. The per-pass radius cap is recomputed for the
# current proposal so the grid-coverage envelope follows it. Returns
# list(pareto_k, is_ess, refined) or NULL (empty / rank-deficient `vary`, no
# finite importance draw).
.joint_pareto_score <- function(prep, vary, refit_log_marginal, n_samples,
                                tail_points = NULL) {
    if (length(vary) == 0L) return(NULL)
    Su <- prep$Su; u_hat <- prep$u_hat; u_grid <- prep$u_grid
    tags <- prep$tags; cn <- prep$cn; d <- prep$d

    Su_v <- Su[vary, vary, drop = FALSE]
    L_v  <- tryCatch(t(chol(Su_v)), error = function(e) NULL)
    if (is.null(L_v)) return(NULL)
    u_hat_v <- u_hat[vary]

    lt <- .joint_pareto_make_lt(prep, vary, refit_log_marginal)

    u_grid_v <- u_grid[, vary, drop = FALSE]
    prop_u  <- u_hat_v
    prop_L  <- L_v
    best    <- NULL
    gm_full <- NULL                      # first-pass (grid-moment) result
    refined <- FALSE
    prev_k  <- Inf                       # k-hat of the proposal each pass refines from
    for (iter in seq_len(.K_DIAG_MM_MAX)) {
        rc <- .nested_grid_radius_cap(u_grid_v, prop_u, prop_L)
        kd <- tryCatch(.nested_is_pareto_k(prop_u, prop_L, lt, n_samples,
                                           radius_cap = rc, return_draws = TRUE,
                                           tail_points = tail_points),
                       error = function(e) NULL)
        if (is.null(kd) || !is.finite(kd$pareto_k)) break
        cand <- list(pareto_k = kd$pareto_k, is_ess = kd$is_ess, refined = refined,
                     U = kd$U, log_weights = kd$log_weights, lr = kd$lr,
                     prop_u = prop_u, prop_L = prop_L)
        # The first pass is the canonical grid-moment Gaussian -- its draws span the
        # actual posterior spread (not the moment-matching-widened proposal, which
        # scatters seed-dependently). The dispatcher reads it for the grid-coverage
        # check and the escaped-grid guard, so the decision is stable across RNG
        # seeds.
        if (iter == 1L) gm_full <- cand
        # `prop_u` / `prop_L` are the proposal that PRODUCED this `kd`, so the stored
        # proposal stays consistent with `best$pareto_k` and the bootstrap re-fits
        # the GPD on this same converged proposal's ratios.
        if (is.null(best) || kd$pareto_k < best$pareto_k) best <- cand
        if (kd$pareto_k <= .K_DIAG_USABLE || iter == .K_DIAG_MM_MAX) break
        # Stop refining once a pass no longer improves on the proposal it was
        # estimated from: moment matching has converged (or started to drift on the
        # seed-dependent widening), so further passes only spend budget without
        # lowering k. This keeps the generous .K_DIAG_MM_MAX a safe backstop -- a
        # stubborn k that is still falling keeps refining; a plateaued one stops.
        if (kd$pareto_k >= prev_k) break
        prev_k <- kd$pareto_k
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
    if (!is.null(best)) {
        best$gm    <- gm_full
        best$gm_U  <- if (!is.null(gm_full)) gm_full$U else NULL
        best$gm_lw <- if (!is.null(gm_full)) gm_full$log_weights else NULL
    }
    best
}

# Grid-mixture (basin) outer Pareto-k scorer. Scores the
# diagnostic against the proposal the engine ACTUALLY samples hyperparameters
# from: a defensive mixture q(u) = sum_k w_k N(u_k, diag(s^2)) of local Gaussian
# bumps at the integration-grid cells `u_k` (projected onto `vary`), mixed by the
# grid weights `w_k`. This covers a skewed or multi-node hyperparameter posterior
# by construction (the single grid-moment Gaussian cannot), so the importance
# weights stay bounded and the k-hat reflects the grid representation's fidelity
# rather than a Gaussian's mismatch to a skewed marginal. Per-axis bump SD is
# `bw * (largest adjacent grid gap on that axis)`, falling back to the
# grid-weighted SD where an axis keeps a single distinct value. Draws `n_samples`
# (= `diagnose_draws`) points, the single user-facing precision knob: the
# mixture's tail-shape k-hat needs a long enough tail to be stable, so raise
# `diagnose_draws` for a tighter estimate. The draws stay near the grid cells, so
# no inner Newton stalls. Returns
# list(pareto_k, is_ess, refined = FALSE, lr, s, lo, hi) -- `lr` the raw finite
# importance log-ratios (for the k bootstrap), `s` the per-axis bump SD and
# `lo` / `hi` the kept-cell node range on `vary`, which the dispatcher uses to
# bound the mixture's coverage -- or NULL to fall back to the single-Gaussian
# scorer (fewer than two distinct weighted cells -- not actually spread -- or too
# few finite importance draws). Like .joint_pareto_score it does NOT manage the
# RNG (the driver saves / restores it once around all scoring).
.joint_pareto_score_mixture <- function(prep, vary, refit_log_marginal, n_samples,
                                        tail_points = NULL) {
    if (length(vary) == 0L) return(NULL)
    w <- prep$w
    if (is.null(w) || !any(is.finite(w)) || sum(w) <= 0) return(NULL)
    cu   <- prep$u_grid[, vary, drop = FALSE]
    keep <- is.finite(w) & w >= .K_DIAG_MIX_FLOOR * max(w)
    cu   <- cu[keep, , drop = FALSE]
    wk   <- w[keep]
    if (nrow(cu) < 2L) return(NULL)

    # Collapse cells that coincide on the `vary` axes (a pinned axis dropped here
    # can repeat a varying-axis point across its values): sum their weights so
    # each mixture component is a distinct varying-subspace cell.
    key <- apply(round(cu, 10L), 1L, paste, collapse = "\r")
    if (anyDuplicated(key)) {
        first <- !duplicated(key)
        wk    <- as.numeric(tapply(wk, factor(key, levels = key[first]), sum))
        cu    <- cu[first, , drop = FALSE]
    }
    K <- nrow(cu); p <- ncol(cu)
    if (K < 2L) return(NULL)
    wk <- wk / sum(wk)

    bw   <- as.numeric(getOption("tulpa.kdiag.mix_bw", .K_DIAG_MIX_BW))
    sdax <- sqrt(pmax(diag(prep$Su[vary, vary, drop = FALSE]), 0))
    # Bump SD from the FULL grid's node spacing on each axis (the grid's actual
    # resolution), NOT the floor-surviving cells: when the posterior weight
    # concentrates on one node of an axis only that cell survives the floor, and a
    # gap from the kept cells alone would collapse to a degenerate near-zero width
    # (a spike the single-Gaussian's near-mode draws then read as "uncovered").
    full_v <- prep$u_grid[, vary, drop = FALSE]
    gap  <- vapply(seq_len(p), function(j) {
        u <- sort(unique(full_v[, j])); if (length(u) < 2L) NA_real_ else max(diff(u))
    }, numeric(1))
    s   <- bw * gap
    bad <- !is.finite(s) | s <= 0
    s[bad] <- (bw * sdax)[bad]
    s[!is.finite(s) | s <= 0] <- 1e-3

    n_mix <- as.integer(n_samples)
    lt   <- .joint_pareto_make_lt(prep, vary, refit_log_marginal)
    comp <- sample.int(K, n_mix, replace = TRUE, prob = wk)
    U_v  <- cu[comp, , drop = FALSE] +
            sweep(matrix(stats::rnorm(n_mix * p), n_mix, p), 2L, s, `*`)

    # log q(u) up to the per-component-common diagonal normalizer (the SAME `s`
    # for every component, so it cancels under PSIS): logsumexp over components of
    # log w_k - 0.5 sum_j ((u_j - u_kj) / s_j)^2.
    tcu  <- t(cu)
    logq <- vapply(seq_len(n_mix), function(i) {
        q <- log(wk) - 0.5 * colSums(((tcu - U_v[i, ]) / s)^2)
        m <- max(q); if (!is.finite(m)) return(-Inf)
        m + log(sum(exp(q - m)))
    }, numeric(1))

    ltv <- lt(U_v)
    if (length(ltv) != n_mix) return(NULL)
    lr  <- ltv - logq
    fin <- is.finite(lr)
    if (sum(fin) < .PSIS_MIN_EVAL) return(NULL)
    ps <- tulpa_psis(lr[fin], tail_points = tail_points)
    if (!is.finite(ps$pareto_k)) return(NULL)
    list(pareto_k = ps$pareto_k, is_ess = ps$is_ess, refined = FALSE,
         lr = lr[fin], s = s, lo = apply(cu, 2L, min), hi = apply(cu, 2L, max))
}

# Fraction of the single-Gaussian importance weight that falls OUTSIDE the
# mixture's coverage hull. `U` are the proposal's importance
# draws on the varying subspace, `log_weights` their PSIS-smoothed (normalized)
# log weights, `lo` / `hi` the per-axis coverage bounds (the kept-cell node range
# expanded by a few bump SDs -- the mixture's actual reach). A draw is outside if
# any axis falls beyond [lo, hi]. The WEIGHT (not count) outside measures how much
# target mass sits beyond what the mixture covers: small => the grid covers the
# posterior (a high single-Gaussian k-hat is then a within-grid shape mismatch the
# mixture can fix); large => the grid is too narrow and the single Gaussian's
# k-hat must stand. Returns NA when draws / weights are unavailable.
.joint_pareto_hull_weight <- function(U, log_weights, lo, hi) {
    if (is.null(U) || is.null(log_weights) || !is.matrix(U) || nrow(U) == 0L ||
        length(log_weights) != nrow(U)) return(NA_real_)
    outside <- logical(nrow(U))
    for (j in seq_len(ncol(U))) outside <- outside | (U[, j] < lo[j]) | (U[, j] > hi[j])
    w <- exp(log_weights - max(log_weights))
    sw <- sum(w)
    if (!is.finite(sw) || sw <= 0) return(NA_real_)
    sum(w[outside]) / sw
}

# Dispatch the outer Pareto-k scoring to the right proposal.
# Always scores the single-Gaussian proposal (the grid-moment / mode-Hessian / CCD
# Gaussian with moment-matching refinement), whose tails extend beyond the grid so
# it detects a target heavier than the grid represents (the grid-width signal). On
# a spread-tensor `grid_moment` grid it ALSO scores the grid-mixture proposal --
# the faithful representation of what the engine samples (a local bump at each grid
# cell, mixed by the grid weights), which covers a skewed / multi-node posterior
# the single Gaussian underweights WITHIN the grid. The mixture covers only the
# grid (its bumps reach a few SDs past the outer nodes), so it is adopted only when
# (1) the single Gaussian's importance weight is essentially all inside that
# coverage hull -- the grid covers the posterior, so a high single-Gaussian k-hat
# is a within-grid shape mismatch -- AND (2) the mixture gives a lower k-hat.
# Otherwise the single Gaussian stands: a target tail beyond the grid keeps its
# (higher) k-hat so the grid-width deficiency is flagged, and a near-collapsed grid
# (where the mixture's few bumps cover worse) keeps the moment-matched Gaussian. A
# true delta collapse and a supplied CCD proposal are not `grid_moment`, so only
# the single Gaussian is scored. Returns list(best, source) or NULL. Shared by the
# joint k and the per-arm k.
.joint_pareto_score_dispatch <- function(prep, vary, refit_log_marginal, n_samples,
                                         tail_points = NULL) {
    g     <- .joint_pareto_score(prep, vary, refit_log_marginal, n_samples,
                                 tail_points = tail_points)
    g_src <- if (!is.null(g) && isTRUE(g$refined)) "moment_matched"
             else prep$proposal_source
    g_out <- if (is.null(g)) NULL else list(best = g, source = g_src)

    if (!identical(prep$proposal_source, "grid_moment")) return(g_out)
    gm     <- g$gm %||% g
    gm_k   <- if (is.list(gm)) gm$pareto_k %||% NA_real_ else NA_real_
    gm_out <- if (is.list(gm)) list(best = gm, source = "grid_moment") else g_out

    # The skip is judged on the GRID-MOMENT k (what the engine samples), not the
    # moment-matching-widened single Gaussian: when the
    # grid-moment proposal is already good the verdict cannot improve, so skip the
    # mixture's `diagnose_draws`-draw evaluation. The rescue still runs for any
    # grid-moment k in the ok / unreliable band.
    if (is.finite(gm_k) && gm_k < .K_DIAG_GOOD) return(g_out)

    mix <- .joint_pareto_score_mixture(prep, vary, refit_log_marginal, n_samples,
                                       tail_points = tail_points)
    if (is.null(mix)) return(gm_out)
    if (is.null(g))   return(list(best = mix, source = "grid_mixture"))

    # The mixture's coverage hull: the kept-cell node range expanded by a few bump
    # SDs. The check reads the first-pass GRID-MOMENT draws (`gm_U`), whose spread
    # is the posterior's, so the decision does not depend on how moment matching
    # happened to widen the single Gaussian on a given seed.
    gm_U  <- g$gm_U %||% g$U
    gm_lw <- g$gm_lw %||% g$log_weights
    out_frac <- .joint_pareto_hull_weight(
        gm_U, gm_lw,
        mix$lo - .K_DIAG_HULL_PAD * mix$s, mix$hi + .K_DIAG_HULL_PAD * mix$s)
    covered  <- is.finite(out_frac) && out_frac <= .K_DIAG_HULL_TOL

    # Compare the mixture to the GRID-MOMENT single Gaussian (`gm_k`), not the
    # moment-matching-refined one: a refined k that dropped below
    # the mixture only by widening the single Gaussian past the grid is not a
    # faithful within-grid reading, so it must not win. Adopt the mixture (the
    # faithful within-grid proposal) when the grid covers the posterior AND the
    # mixture IMPROVES on the grid-moment proposal. A mixture that does NOT improve
    # on it is degenerate -- a near-collapsed grid where the few
    # bumps cover worse than the moment-matched Gaussian -- so the moment-matched
    # single Gaussian stands. For a target heavier / wider than the grid the
    # mixture is confined to the grid and reads unreliable, correctly flagging the
    # grid-width deficiency the escaped single Gaussian masked.
    improves <- is.finite(mix$pareto_k) && is.finite(gm_k) && mix$pareto_k < gm_k
    if (covered && improves) list(best = mix, source = "grid_mixture") else g_out
}

# Bootstrap + closed-form uncertainty of a CHOSEN proposal's outer Pareto-k-hat
#. `best` is the proposal the dispatcher selected, carrying its
# raw finite importance log-ratios `lr`. The k-hat is a single fixed number for
# this fit + proposal; its sampling uncertainty GIVEN the proposal is estimated by
# resampling those SAME ratios with replacement and re-fitting the GPD tail at the
# same `tail_points` (`k_bootstrap` replicates), which is free -- no new inner
# solves -- and estimator-agnostic. Bootstrap measures how UNSTABLE the current
# tail estimate is; it cannot create tail information. A tighter k needs more
# ACTUAL tail ratios, i.e. a larger `diagnose_draws`, NOT a larger `k_bootstrap`.
# Returns the point k, IS-ESS, tail size used, bootstrap SE / 95% CI, closed-form
# (GPD-shape MLE asymptotic) SE cross-check, and the band-confidence flag (the
# bootstrap CI within one reliability band). Falls back to the scoring-pass point
# k with NA uncertainty when no ratios were captured.
.joint_pareto_uncertainty <- function(best, tail_points, n_boot, conf_bands) {
    if (is.null(best) || is.null(best$lr) || !length(best$lr)) {
        return(list(pareto_k = if (is.null(best)) NA_real_ else best$pareto_k %||% NA_real_,
                    is_ess = if (is.null(best)) NA_real_ else best$is_ess %||% NA_real_,
                    tail_points = NA_integer_, se_boot = NA_real_,
                    ci_low = NA_real_, ci_high = NA_real_,
                    se_formula = NA_real_, band_confident = NA, conf_bands = NULL))
    }
    .tulpa_psis_k_uncertainty(best$lr, tail_points = tail_points,
                              n_boot = n_boot, conf_bands = conf_bands)
}

# Shared outer Pareto-k-hat driver for a joint nested-Laplace result. Prepares
# the proposal summary once, scores the joint k over every genuinely-varying
# axis, and -- when `arm_axes` is supplied -- additionally
# scores a k restricted to each arm's hyperparameter axes (the rest held at the
# posterior mean `u_hat`). Runs all scoring with the RNG restored at the end so
# the fit's draws are bit-for-bit unchanged.
#
# Uncertainty. The canonical pass scores the chosen proposal
# ONCE over `n_samples` (= `diagnose_draws`) importance draws and returns its raw
# finite log-ratios; the k-hat's sampling uncertainty is then estimated by
# bootstrapping those ratios (`k_bootstrap` replicates, re-fitting the GPD tail at
# the resolved `tail_points`), which adds NO inner solves. `diagnose_draws` is the
# precision knob (more actual tail ratios => tighter k); `k_bootstrap` only
# quantifies the current estimate's instability and cannot create tail
# information. `k_tail_points` (NULL = the automatic PSIS rule) is an expert
# tail-threshold control, capped at the 20%-of-draws ceiling with one warning. The
# per-arm pass scores + bootstraps each arm the same way.
#
# Returns list(pareto_k, is_ess, proposal_source): all NA / NA-source when the fit
# declines (an axis with unguessable support, a degenerate proposal covariance, or
# too few finite importance draws), in which case the diagnostic layer reports
# quad-ESS. Otherwise also `pareto_k_se_boot` / `pareto_k_ci_low` /
# `pareto_k_ci_high` / `pareto_k_se_formula` / `pareto_k_tail_points` /
# `pareto_k_tail_points_requested` / `pareto_k_band_confident`. With `arm_axes`,
# also `by_arm_k` / `by_arm_is_ess` (named over the arms; NA where an arm carries
# no genuinely-varying axis) plus per-arm `by_arm_se_boot` / `by_arm_ci_low` /
# `by_arm_ci_high` / `by_arm_se_formula` / `by_arm_tail_points` /
# `by_arm_band_confident`.
.joint_pareto_k <- function(res, refit_log_marginal, n_samples = 500L,
                            proposal = NULL, arm_axes = NULL,
                            k_bootstrap = 1000L, k_tail_points = NULL,
                            k_conf_bands = NULL) {
    na_out <- list(pareto_k = NA_real_, is_ess = NA_real_,
                   proposal_source = NA_character_)
    prep <- .joint_pareto_prepare(res, refit_log_marginal, n_samples, proposal)
    if (is.null(prep)) return(na_out)

    # Resolve the GPD tail-size request once: an explicit
    # `k_tail_points` beyond the 20%-of-draws ceiling is capped, with a single
    # user-facing warning HERE so the per-replicate bootstrap re-fits stay silent.
    tp_req <- if (is.null(k_tail_points)) NA_integer_ else as.integer(k_tail_points)
    if (is.finite(tp_req)) {
        cap <- as.integer(floor(0.2 * as.integer(n_samples)))
        if (tp_req > cap) {
            warning(sprintf(paste0(
                "k_tail_points = %d exceeds the 20%% PSIS tail cap; using %d ",
                "instead. Increase diagnose_draws, not k_bootstrap, to obtain ",
                "more tail information."), tp_req, cap), call. = FALSE)
        }
    }

    .preserve_seed_in_frame()

    n_arm <- if (!is.null(arm_axes)) length(arm_axes) else 0L

    # Axes the grid pins to a single value (e.g. a copy `alpha` fixed at 0, or a
    # one-point dispersion axis) carry zero weighted variance, leaving Su rank
    # deficient and its Cholesky undefined. The importance proposal holds them
    # fixed at u_hat and is built only on the varying axes.
    vary  <- .joint_pareto_vary_axes(prep$Su)
    joint <- .joint_pareto_score_dispatch(prep, vary, refit_log_marginal,
                                          n_samples, tail_points = k_tail_points)
    if (is.null(joint)) return(na_out)

    ju  <- .joint_pareto_uncertainty(joint$best, k_tail_points, k_bootstrap,
                                     k_conf_bands)
    out <- list(pareto_k = ju$pareto_k, is_ess = ju$is_ess,
                proposal_source = joint$source,
                pareto_k_se_boot               = ju$se_boot,
                pareto_k_ci_low                = ju$ci_low,
                pareto_k_ci_high               = ju$ci_high,
                pareto_k_se_formula            = ju$se_formula,
                pareto_k_tail_points           = ju$tail_points,
                pareto_k_tail_points_requested = tp_req,
                pareto_k_band_confident        = ju$band_confident,
                pareto_k_conf_bands            = ju$conf_bands)

    if (n_arm > 0L) {
        pa <- lapply(seq_len(n_arm), function(a) {
            v <- intersect(vary, as.integer(arm_axes[[a]]))
            none <- list(k = NA_real_, ess = NA_real_, se = NA_real_,
                         lo = NA_real_, hi = NA_real_, sef = NA_real_,
                         tp = NA_integer_, conf = NA)
            if (length(v) == 0L) return(none)
            s <- .joint_pareto_score_dispatch(prep, v, refit_log_marginal,
                                              n_samples, tail_points = k_tail_points)
            if (is.null(s)) return(none)
            u <- .joint_pareto_uncertainty(s$best, k_tail_points, k_bootstrap,
                                           k_conf_bands)
            list(k = u$pareto_k, ess = u$is_ess, se = u$se_boot, lo = u$ci_low,
                 hi = u$ci_high, sef = u$se_formula, tp = u$tail_points,
                 conf = u$band_confident)
        })
        nm <- names(arm_axes)
        out$by_arm_k           <- stats::setNames(vapply(pa, function(z) z$k,   numeric(1)), nm)
        out$by_arm_is_ess      <- stats::setNames(vapply(pa, function(z) z$ess, numeric(1)), nm)
        out$by_arm_se_boot     <- stats::setNames(vapply(pa, function(z) z$se,  numeric(1)), nm)
        out$by_arm_ci_low      <- stats::setNames(vapply(pa, function(z) z$lo,  numeric(1)), nm)
        out$by_arm_ci_high     <- stats::setNames(vapply(pa, function(z) z$hi,  numeric(1)), nm)
        out$by_arm_se_formula  <- stats::setNames(vapply(pa, function(z) z$sef, numeric(1)), nm)
        out$by_arm_tail_points <- stats::setNames(vapply(pa, function(z) as.integer(z$tp), integer(1)), nm)
        out$by_arm_band_confident <- stats::setNames(vapply(pa, function(z) z$conf, logical(1)), nm)
    }
    out
}

# Which arms does a latent block load on? A block enters arm `a`'s linear
# predictor iff its per-arm index for `a` is non-empty. The per-arm index lives
# under one of `obs_idx` / `spatial_idx` / `temporal_idx` on the (prepared)
# block spec -- a length-n_arms list (one integer vector per arm; an empty entry
# means the block does not load that arm) or a single vector replicated across
# every arm. A block with no per-arm index field loads on all arms. Returns a
# length-n_arms logical.
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
#. Returns a named list arm_name -> integer column indices, or
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

# Attach the opt-in per-arm outer Pareto-k to a joint result.
# No-op when the driver computed no per-arm k (diagnostic off, single-block
# layout, or a declined per-arm map). Single source for both attach paths. Each
# arm carries its own bootstrap SE (`*_se_boot`), 95% CI (`*_ci_low` / `*_ci_high`),
# closed-form SE (`*_se_formula`), tail size (`*_tail_points`) and band-confidence
# flag (`*_band_confident`), the per-arm analogue of the joint uncertainty

.joint_attach_by_arm_k <- function(res, kd) {
    if (is.null(kd$by_arm_k)) return(res)
    res$pareto_k_by_arm        <- kd$by_arm_k
    res$pareto_k_by_arm_is_ess <- kd$by_arm_is_ess
    res$pareto_k_by_arm_scope  <-
        "per-arm hyperparameter axes (other arms fixed at posterior mean)"
    res$pareto_k_by_arm_se_boot        <- kd$by_arm_se_boot
    res$pareto_k_by_arm_ci_low         <- kd$by_arm_ci_low
    res$pareto_k_by_arm_ci_high        <- kd$by_arm_ci_high
    res$pareto_k_by_arm_se_formula     <- kd$by_arm_se_formula
    res$pareto_k_by_arm_tail_points    <- kd$by_arm_tail_points
    res$pareto_k_by_arm_band_confident <- kd$by_arm_band_confident
    res
}

# Attach the bootstrap + closed-form outer Pareto-k uncertainty to a joint result
#. `pareto_k_se_boot` / `pareto_k_ci_low` / `pareto_k_ci_high`
# are the bootstrap SE and 95% CI of the k-hat -- its sampling uncertainty GIVEN
# the proposal, NOT a posterior CI; `pareto_k_se_formula` the GPD-shape MLE
# asymptotic SE cross-check; `pareto_k_tail_points` the tail size used (the request
# in `pareto_k_tail_points_requested`); `pareto_k_band_confident` whether the
# bootstrap CI lies within one reliability band. No-op when the diagnostic declined
# (the uncertainty fields are absent from `kd`).
.joint_attach_pareto_k_uncertainty <- function(res, kd) {
    if (is.null(kd$pareto_k_tail_points) && is.null(kd$pareto_k_se_boot)) return(res)
    res$pareto_k_se_boot               <- kd$pareto_k_se_boot
    res$pareto_k_ci_low                <- kd$pareto_k_ci_low
    res$pareto_k_ci_high               <- kd$pareto_k_ci_high
    res$pareto_k_se_formula            <- kd$pareto_k_se_formula
    res$pareto_k_tail_points           <- kd$pareto_k_tail_points
    res$pareto_k_tail_points_requested <- kd$pareto_k_tail_points_requested
    res$pareto_k_band_confident        <- kd$pareto_k_band_confident
    res$pareto_k_conf_bands            <- kd$pareto_k_conf_bands
    res
}

# Attach the diagnostic's draw budget and wall-clock cost ratio.
# `diagnose_cost_ratio` = diagnostic seconds / fit seconds (the latter excluding the
# diagnostic), read from the fit timer's "diagnostics" bucket vs the rest, so a
# caller can see how expensive the outer Pareto-k diagnostic was relative to the fit
# it certifies (the design target is roughly 1-3x). `diagnose_draws` records the
# importance-draw budget actually used. Both NA when the diagnostic was off or the
# timing is unavailable. Called by both the single- and multi-block drivers after
# `res$timing` is set.
.joint_attach_diagnose_cost <- function(res, diagnose_k, diagnose_draws) {
    res$diagnose_draws <- if (isTRUE(diagnose_k)) as.integer(diagnose_draws)
                          else NA_integer_
    ratio <- NA_real_
    tmg <- res$timing
    if (isTRUE(diagnose_k) && is.numeric(tmg) && !is.null(names(tmg)) &&
        all(c("total", "diagnostics") %in% names(tmg))) {
        diag_s <- tmg[["diagnostics"]]
        fit_s  <- tmg[["total"]] - diag_s
        if (is.finite(diag_s) && is.finite(fit_s) && fit_s > 0) ratio <- diag_s / fit_s
    }
    res$diagnose_cost_ratio <- ratio
    res
}

# Attach the k_quality reliability verdict. Reads the outer
# Pareto-k point estimate, its bootstrap band-confidence flag, and the reliability
# bands the diagnostic used, and reports an honest reached / best / reason quartet
# against the requested quality intent. NEVER silently downgrades: when the fit
# cannot confidently meet the requested band it returns the band it did reach plus
# the reason it fell short. `conf_bands` is an explicit override; when NULL it uses
# the bands the uncertainty pass recorded (`res$pareto_k_conf_bands`, at the
# realised finite-draw count, matching the band-confidence flag), else the
# sample-size-dependent default at `diagnose_draws`. Called for both the single-
# and multi-block paths after the diagnostic fields are attached.
.joint_attach_k_quality <- function(res, k_quality, diagnose_k, diagnose_draws,
                                    conf_bands = NULL) {
    res$k_quality_requested <- k_quality
    res$k_quality_reached   <- NA
    res$k_quality_best      <- NA_character_
    res$k_quality_reason    <- NA_character_

    if (identical(k_quality, "none") || !isTRUE(diagnose_k)) {
        res$k_quality_reason <- if (identical(k_quality, "none")) "diagnostic disabled"
                                else "diagnostic not run"
        return(res)
    }
    k <- res$pareto_k
    if (!is.finite(k)) {
        res$k_quality_reason <- "k-hat unavailable (diagnostic declined)"
        return(res)
    }
    # Prefer the bands the uncertainty pass actually used (at the realised
    # finite-draw count), so the band index and the band-confidence flag it is
    # read against share one boundary set; fall back to the draw-budget default.
    bands  <- if (!is.null(conf_bands)) conf_bands
              else res$pareto_k_conf_bands %||% .ps_conf_bands(as.integer(diagnose_draws))
    labels <- if (length(bands) >= 2L) c("good", "ok", "unreliable")
              else c("reliable", "unreliable")
    conf   <- isTRUE(res$pareto_k_band_confident)
    bi     <- .k_band_b(k, bands)                       # 0 = best band, increasing
    res$k_quality_best <- if (conf) labels[min(bi + 1L, length(labels))] else "uncertain"

    if (identical(k_quality, "report")) {
        res$k_quality_reason <- "report only (no target band requested)"
        return(res)
    }
    # "good" requires the confident band to be the best (index 0); "ok" allows the
    # best or the next (index <= 1, the usable band).
    target  <- if (identical(k_quality, "good")) 0L else 1L
    reached <- conf && bi <= target
    res$k_quality_reached <- reached
    res$k_quality_reason  <-
        if (reached) "requested band reached"
        else if (!conf)
            "k-hat interval crosses a band boundary; raise diagnose_draws or refine (gcol33/tulpa#131)"
        else "k-hat confidently outside the requested band; the integration is genuinely less reliable"
    res
}

# Single-block joint wrapper. Reuses the driver's already-built generic
# `kernel_fn` (the closure refinement passes drive, which round-trips a
# user-facing cell matrix through `backend$call_kernel`) and `hp_fn` (the
# baked-hyperprior closure) as the re-evaluation path, so no kernel-call
# machinery is duplicated. The diagnostic solves are warm-started from the
# modal latent mode and capped at `.K_DIAG_MAX_ITER`, the
# same cost bound as the multi-block path. The whole importance batch is
# re-solved in one `kernel_fn` call with `n_threads_outer` so the independent
# re-solves run concurrently across cores rather than one-at-a-time using all
# inner threads. Attaches `pareto_k` / `pareto_k_is_ess` / `pareto_k_scope`;
# with `diagnose_k = FALSE` the fields are present but NA.
.joint_attach_pareto_k_single <- function(res, kernel_fn, hp_fn,
                                          max_iter = 50L,
                                          diagnose_k = TRUE, diagnose_draws = 500L,
                                          n_threads_outer = 1L,
                                          pareto_k_by_arm = FALSE,
                                          k_bootstrap = 1000L,
                                          k_tail_points = NULL,
                                          k_conf_bands = NULL) {
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
    #. `x_init_per_cell` is an [S x n_x] warm matrix or NULL.
    # The diagnostic re-solve runs with its own cheaper knobs (k_max_iter,
    # knobs$tol / refresh), so its checkpoint fingerprint would differ from the
    # main grid's; run it checkpoint-free so it neither collides with nor
    # appends to the fit's checkpoint file.
    solve_fn <- function(theta_mat, x_init_per_cell = NULL) {
        r  <- .joint_with_quiet_opts(kernel_fn(theta_mat, warm_start = warm_arg,
                        max_iter_override = k_max_iter,
                        n_threads_outer = n_to,
                        inner_refresh_override = knobs$refresh,
                        tol_override = knobs$tol,
                        x_init_per_cell = x_init_per_cell))
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
    # unavailable, else the plain broadcast warm.
    refit <- .joint_make_diag_refit(res, solve_fn, modal_theta, knobs)
    arm_axes <- if (isTRUE(pareto_k_by_arm)) .joint_pareto_arm_axes(res) else NULL
    kd <- .joint_pareto_k(res, refit, diagnose_draws, arm_axes = arm_axes,
                          k_bootstrap = k_bootstrap, k_tail_points = k_tail_points,
                          k_conf_bands = k_conf_bands)
    res$pareto_k        <- kd$pareto_k
    res$pareto_k_is_ess <- kd$is_ess
    res$pareto_k_proposal_source <- kd$proposal_source
    res <- .joint_attach_pareto_k_uncertainty(res, kd)
    res <- .joint_attach_by_arm_k(res, kd)
    res
}
