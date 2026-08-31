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
# Each pass re-estimates the proposal from the PSIS-weighted
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
# (the number itself lives in `.NL_DIAG`, R/settings.R -- read at use time,
# since a top-level value here would evaluate before that file is collated).

# Internal proposal-loop threshold for the grid-mixture skip: a
# single-Gaussian k-hat below this is already good, so the mixture rescue (which
# can only LOWER an inflated k) is skipped. A fixed loop control, distinct from
# the reported sample-size-dependent bands (`.ps_conf_bands` in R/psis.R).
.K_DIAG_GOOD <- 0.5

# Grid-mixture (basin) proposal for the outer Pareto-k on a spread tensor grid
# The nested-Laplace engine represents the hyperparameter
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

# Skew-normal rescue for the outer Pareto-k
# A hyperparameter marginal on a variance component stays
# right-skewed even after the log unconstraining transform, and a SYMMETRIC
# Gaussian proposal against a skewed target has a heavy importance-ratio tail
# whatever the integration's quality -- so the k-hat reads unreliable for a
# reason that lives in the proposal. The grid-mixture rescue above covers that
# skew where the grid is SPREAD (its bumps trace the posterior's shape), but a
# sharp posterior collapses the grid onto ~1 cell, the mixture's few bumps then
# cover worse than the Gaussian, and nothing corrects the mismatch: the k-hat
# tracks the collapse rather than the fit. The skew-normal proposal is the
# rescue for that regime -- and, being scored last, for any regime where the
# chosen proposal is still above the good band.
#
# Below this |skewness| the target is "approximately symmetric" on the usual
# magnitude convention (the same Bulmer 1979 scale `.tulpa_gamma3_band` bands
# the inner gamma_3 on, whose first cut is 0.5), so a skew-normal proposal is
# numerically the Gaussian and cannot move the tail. Skip it rather than pay a
# scoring pass that can only reproduce the k it started from.
.K_DIAG_SKEW_MIN <- 0.2

# ... but a fixed floor alone is not enough, because the skewness is ESTIMATED.
# A sample skewness of a normal variable has standard error
# `sqrt(6n(n-1) / ((n-2)(n+1)(n+3)))` -- about 0.17 at n = 200 -- so a bare 0.2
# floor is barely one standard error and fires on noise. The rescue therefore
# also requires the skewness to be significantly non-zero at `.K_DIAG_SKEW_Z`
# standard errors, evaluated at the weights' EFFECTIVE sample size rather than
# the raw draw count, since these are importance-weighted moments.
#
# The multiplier is set by measurement, not convention. On a GAUSSIAN outer
# target at 200 draws (so the true skewness is 0 and every adoption is noise),
# the share of RNG states in which the skew proposal was adopted ran: no
# significance screen 18%, a two-SE screen 5%, a three-SE screen 0%. Because a
# batch consumer bins fits on the REPORTED number, a proposal source that flaps
# with the RNG seed is itself a defect, so the screen is set where spurious
# adoption vanishes. Sensitivity is
# given up only at small draw counts, where the k-hat being rescued is too noisy
# to act on anyway.
.K_DIAG_SKEW_Z <- 3

# Standard error of a sample skewness at effective sample size `n`, under the
# null that the variable is normal (Cramer 1946; the finite-sample form behind
# the usual D'Agostino skewness test).
.skew_se <- function(n) {
    if (!is.finite(n) || n < 4) return(Inf)
    sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
}

# Grid-coverage tolerance for adopting the grid-mixture over the single Gaussian
# The mixture is confined to the grid's coordinate hull, so it
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
# Profiling the joint occu_cover diagnostic showed the
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
# Profiling showed a large share of the per-draw Newton
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
# The diagnostic's `k_samples` draws come off the Gaussian
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
    # NA marks an axis whose support is not safely guessable. Returned rather
    # than collapsed to NULL so a caller can NAME the offending axis in its
    # decline reason instead of reporting a bare NA k-hat.
    vapply(axes, tag_one, character(1))
}

# Resolve the per-column unconstraining transforms for a joint result.
# Walks the result's block layout (multi-block: `axis_offsets` + `blocks`;
# single-block: the lone `prior$type`) so the `rho` ambiguity above is
# resolved by the block that owns the axis, not by name alone. Trailing
# `phi_<arm>` dispersion columns (appended after the latent-block axes) are
# positive-scale (log).
#
# Returns one tag per `theta_grid` column, NA where the axis support is not
# safely guessable, or a `.k_decline()` for a structural fault (no grid at all,
# a layout inconsistency). The per-axis NAs are kept here rather than collapsed
# so a per-axis consumer can act on the axes it CAN transform;
# `.joint_pareto_axis_tags()` below is the whole-fit view that declines on any
# of them.
.joint_axis_tags_raw <- function(res) {
    tg <- res$theta_grid
    if (is.null(tg) || !is.matrix(tg) || ncol(tg) == 0L) {
        return(.k_decline("not_applicable", "no outer hyperparameter grid"))
    }
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
            tags[cols] <- .joint_pareto_block_tags(
                tolower(res$blocks[[b]]$type %||% ""), bare(cols))
        }
        # Columns past the last block axis are per-arm phi dispersion axes.
        if (ao[B + 1L] < d) {
            extra <- (ao[B + 1L] + 1L):d
            named <- startsWith(cn[extra], "phi_")
            if (!all(named)) {
                return(.k_decline("internal_inconsistency",
                                  paste0("axis past the last block is not a phi axis: ",
                                         paste(cn[extra][!named], collapse = ", "))))
            }
            tags[extra] <- "log"
        }
    } else {
        # Single-block: every non-phi column belongs to the lone prior block;
        # phi_<arm> columns are positive-scale dispersion axes.
        type <- tolower(res$prior$type %||% "")
        is_phi <- startsWith(cn, "phi_")
        if (any(!is_phi)) {
            tags[!is_phi] <- .joint_pareto_block_tags(type, bare(which(!is_phi)))
        }
        tags[is_phi] <- "log"
    }
    tags
}

# Whole-fit view of the same walk: every axis must be transformable, so a fit
# carrying an unguessable one (car_proper's `rho_car`, a non-BYM2 `rho`)
# DECLINES, naming the axes that stopped it.
.joint_pareto_axis_tags <- function(res) {
    tags <- .joint_axis_tags_raw(res)
    if (.k_is_decline(tags)) return(tags)
    if (anyNA(tags)) {
        cn <- colnames(res$theta_grid)
        return(.k_decline("unguessable_axis", cn[is.na(tags)]))
    }
    tags
}

# The domain each joint-grid axis lives on, for the moment-matched interval a
# quadrature-design node set is summarized with (`.nl_summary_quantile`). Reads
# the SAME per-axis registry the outer Pareto-k unconstrains with, so "what
# coordinate is this axis Gaussian on" has one definition: a positive scale
# (sigma / tau / range / phi_*) is `positive`, the BYM2 mixing weight is `unit`,
# an unconstrained coordinate (a copy `alpha`, an MCAR log-Cholesky entry) is
# `unbounded`. An axis the registry will not guess carries NA, and its summary
# is withheld rather than guessed.
.JOINT_AXIS_DOMAIN <- c(log = "positive", logit01 = "unit", identity = "unbounded")

.joint_axis_domains <- function(res) {
    d <- if (is.matrix(res$theta_grid)) ncol(res$theta_grid) else 0L
    tags <- .joint_axis_tags_raw(res)
    if (.k_is_decline(tags)) return(rep(NA_character_, d))
    unname(.JOINT_AXIS_DOMAIN[tags])
}

# Forward (constrained -> unconstrained) transform for one axis.
.joint_pareto_fwd <- function(tag, theta) {
    switch(tag,
        log      = log(theta),
        logit01  = stats::qlogis(theta),
        identity = theta)
}

# Inverse (unconstrained -> constrained) transform plus the log-Jacobian that
# maps the integrator's grid coordinate to the proposal's unconstrained `u`.
# exp(log_marginal) is the density in whatever coordinate the grid is
# uniform in, weighted by plain softmax with NO volume element. A `log` axis
# is geometric -- uniform in
# u = log(theta) -- so log_marginal is already the u-space density and the
# Jacobian is 0, matching the single-block .nested_grid_pareto_k and the SPDE
# path (confirmed empirically: with the log-axis Jacobian the outer k-hat
# overstates the grid-node ground truth by ~0.25-0.55, enough to flip a verdict
# near 0.7). A `logit01` (correlation) axis has a grid uniform in the natural
# rho, so mapping rho -> logit(rho) DOES carry the logit Jacobian
# log(rho (1 - rho)). `identity` (a copy alpha) is uniform in u already.
.joint_pareto_inv <- function(tag, u) {
    switch(tag,
        log = list(theta = exp(u), log_jac = rep(0, length(u))),
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
# The collapsed-grid mode-Hessian fallback differences the
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

# Regime of the outer integration grid: does the grid
# actually INTEGRATE hyperparameter uncertainty, and if not, is its dominant
# cell interior to the grid or against a boundary?
#
# The quadrature effective sample size `ess_grid = 1 / sum(w_k^2)` counts the
# cells the integration effectively averages over. Below two effective cells no
# axis can carry resolved spread (a second moment along any direction needs two
# points), so the outer integration has degenerated to a point evaluation --
# empirical Bayes at the modal hyperparameter -- and the outer Pareto-k-hat is
# then scoring how well a Gaussian at that mode stands in for the hyperparameter
# marginal, NOT how well a grid integrated it. That distinction matters: a
# bare k-hat threshold reads the two as the same failure.
#
# Where the grid HAS collapsed, the dominant cell's position matters and is a
# one-liner on the stored nodes:
#   * interior -- the grid bracketed the mode, so the collapse is benign: the
#     estimate is empirical-Bayes-at-the-mode and only the integrated
#     hyperparameter uncertainty is missing.
#   * boundary -- the mode sits at the extreme node of some axis, so the grid may
#     simply be too narrow and the true mode may lie outside it. Actionable:
#     widen that axis and refit. Reported per axis with its side, since a scale
#     axis pinned at its floor (field effectively flat) and one pinned at its
#     ceiling (field truncated) call for different reading.
# An axis the grid gives a single value (a copy `alpha` fixed at 0, a one-point
# dispersion axis) is PINNED, not at a boundary, and is excluded.
#
# Returns list(regime, ess_grid, n_grid, max_weight, edge_axes, edge_sides), or
# NULL when the grid / weights are unusable.
.K_DIAG_COLLAPSE_ESS <- 2

.joint_pareto_grid_regime <- function(res) {
    tg <- res$theta_grid
    w  <- res$weights
    if (is.null(tg) || is.null(w)) return(NULL)
    # The generic single-axis nested path stores its grid as a bare vector; the
    # joint paths store a named matrix. One axis is still an axis, so coerce
    # rather than decline.
    if (!is.matrix(tg)) tg <- matrix(as.numeric(tg), ncol = 1L,
                                     dimnames = list(NULL, "theta"))
    if (length(w) != nrow(tg)) return(NULL)
    ok <- is.finite(w) & w > 0
    if (!any(ok) || sum(w[ok]) <= 0) return(NULL)
    wn <- w; wn[!ok] <- 0; wn <- wn / sum(wn)
    ess <- 1 / sum(wn^2)
    out <- list(ess_grid = ess, n_grid = nrow(tg), max_weight = max(wn),
                edge_axes = character(0), edge_sides = character(0))
    # The boundary-MASS read is taken in every regime, before the collapse
    # short-circuit. A well-spread grid still truncates whatever its outermost
    # node's marginal does not reach, so gating this on `ess_grid` reported such
    # a grid clean (gcol33/tulpa#622). `edge_axes` below keeps its own, stronger
    # and collapse-specific meaning: the dominant CELL sits at a boundary.
    em <- .nl_edge_mass_axes(res)
    parts <- strsplit(em, ":", fixed = TRUE)
    out$edge_mass_axes  <- vapply(parts, `[[`, character(1), 1L)
    out$edge_mass_sides <- vapply(parts, `[[`, character(1), 2L)
    if (ess >= .K_DIAG_COLLAPSE_ESS) {
        out$regime <- "spread"
        return(out)
    }

    map <- which.max(wn)
    cn  <- colnames(tg) %||% paste0("axis", seq_len(ncol(tg)))
    for (j in seq_len(ncol(tg))) {
        v <- sort(unique(as.numeric(tg[, j])))
        if (length(v) < 2L) next                       # pinned axis, not an edge
        x <- as.numeric(tg[map, j])
        side <- if (isTRUE(all.equal(x, v[1L]))) "lower"
                else if (isTRUE(all.equal(x, v[length(v)]))) "upper"
                else NA_character_
        if (!is.na(side)) {
            out$edge_axes  <- c(out$edge_axes, cn[j])
            out$edge_sides <- c(out$edge_sides, side)
        }
    }
    out$regime <- if (length(out$edge_axes)) "collapsed_edge" else "collapsed_interior"
    out
}

# Laplace-at-mode covariance of the joint hyperparameter posterior, from a
# finite-difference Hessian of the outer target at the modal grid cell
# The importance proposal for the outer Pareto-k is normally
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

# Grid + per-axis unconstrained transform: theta_grid / weights / column names
# / dimension, plus `u_grid` (the grid rows forward-transformed to the
# unconstrained coordinate `tags` describes). Single source for
# `.joint_pareto_prepare()`'s weighted-moment proposal and the
# diagnose_k-independent placement path (`.joint_attach_pareto_k_placement()`),
# both of which need the same u-space grid before they
# diverge on what they do with it. Declines (a `.k_decline()`)
# when the grid / weights are unusable or a forward transform produces a
# non-finite value, distinguishing a fit whose weights carry no mass
# (`grid_too_small`) from a layout fault (`internal_inconsistency`).
.joint_pareto_grid_u <- function(res, tags) {
    tg <- res$theta_grid
    w  <- res$weights
    if (is.null(tg) || is.null(w)) {
        return(.k_decline("not_applicable", "no outer grid or weights"))
    }
    if (length(w) != nrow(tg)) {
        return(.k_decline("internal_inconsistency",
                          "weights do not match theta_grid rows"))
    }
    if (!is.finite(sum(w)) || sum(w) <= 0) {
        return(.k_decline("grid_too_small", "no positive integration weight"))
    }
    cn <- colnames(tg)
    d  <- ncol(tg)
    u_grid <- matrix(0, nrow(tg), d)
    for (j in seq_len(d)) u_grid[, j] <- .joint_pareto_fwd(tags[j], as.numeric(tg[, j]))
    if (any(!is.finite(u_grid))) {
        bad <- cn[apply(!is.finite(u_grid), 2L, any)]
        return(.k_decline("internal_inconsistency",
                          paste0("node outside its axis support on ",
                                 paste(bad, collapse = ", "))))
    }
    list(tg = tg, w = w, u_grid = u_grid, cn = cn, d = d)
}

# Delta-collapse mode-Hessian: reconstruct a Laplace-at-mode covariance from a
# finite-difference Hessian of the outer target at the grid's highest-weight
# cell, restricted to the axes the GRID LAYOUT offers spread along (a pinned
# axis -- a copy alpha fixed at 0, a one-point dispersion grid -- has zero FD
# curvature and would make a full-axis stencil singular; see
# `.joint_pareto_grid_vary_axes()`). Single source for
# `.joint_pareto_prepare()`'s degenerate-grid-weight fallback (engaged during
# the full outer-k diagnostic) and the diagnose_k-independent placement path
# (`.joint_attach_pareto_k_placement()`) that recenters a collapsed axis
# WITHOUT running the diagnostic. `refit_log_marginal` as in
# `.joint_pareto_mode_cov()`. Returns `list(u_mode=, cov=)` or NULL when the FD
# curvature is unusable.
.joint_pareto_grid_mode_cov <- function(tg, w, u_grid, tags, cn, d,
                                        refit_log_marginal) {
    vary_g <- .joint_pareto_grid_vary_axes(tg)
    u_mode <- as.numeric(u_grid[which.max(w), ])
    cov_h  <- .joint_pareto_mode_cov(u_mode, tags, cn, refit_log_marginal, d,
                                     vary = vary_g)
    if (is.null(cov_h)) return(NULL)
    list(u_mode = u_mode, cov = cov_h)
}

# Shared preparation for the outer Pareto-k-hat of a joint nested-Laplace
# result. Fits a Gaussian proposal to the joint hyperparameter posterior in the
# per-axis unconstrained coordinate `u` (weighted mean `u_hat` + covariance `Su`
# over the integration grid, the analogue of `.nested_grid_pareto_k` generalised
# to mixed support), splices the CCD mode-Hessian `proposal`
# over the axes it spans, and engages the delta-collapse FD rescue
# Returns the proposal summary the scorer draws
# from, or a `.k_decline()` (an axis with unguessable support, an unusable grid /
# weight vector, a sub-floor sample budget -- each named).
# The joint k and the opt-in per-arm
# k score this SAME (u_hat, Su) summary, differing only in
# which axes are allowed to vary -- so the grid build, proposal splice and rescue
# are computed once here.
.joint_pareto_prepare <- function(res, refit_log_marginal, n_samples, proposal) {
    tags <- .joint_pareto_axis_tags(res)
    if (.k_is_decline(tags)) return(tags)

    # Decline before any inner solve when the sample budget cannot reach the
    # GPD-fit floor: a sub-floor `n_samples` would run every
    # one of its solves and then discard the result as NA. The shared core
    # short-circuits, but catching it here too keeps the costly forward/inverse
    # transform + proposal fit off the table.
    if (as.integer(n_samples) < .PSIS_MIN_EVAL) return(.k_decline("draws_too_few"))

    gu <- .joint_pareto_grid_u(res, tags)
    if (.k_is_decline(gu)) return(gu)
    tg <- gu$tg; w <- gu$w; u_grid <- gu$u_grid; cn <- gu$cn; d <- gu$d

    u_hat <- as.numeric(crossprod(w, u_grid))
    cen   <- sweep(u_grid, 2L, u_hat)
    Su    <- crossprod(cen * w, cen)
    Su    <- (Su + t(Su)) / 2

    # Splice the CCD mode-Hessian proposal over the latent axes it spans
    # The grid-weighted `Su` above is the spread of the
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
        gmc <- .joint_pareto_grid_mode_cov(tg, w, u_grid, tags, cn, d,
                                           refit_log_marginal)
        if (!is.null(gmc)) {
            u_hat <- gmc$u_mode
            Su    <- gmc$cov
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
# and call the inner re-solve. It is the joint path's `spec$lt`, so every
# candidate in `R/outer_pareto_candidates.R` evaluates the identical target.
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

# The joint path's adapter onto the shared candidate contract
# (`.k_cand_spec`, R/outer_pareto_candidates.R): restrict the proposal, the
# nodes and the weights to `vary` -- the axes carrying posterior spread, the
# rest held at `u_hat` -- and hand the transform-aware target as the spec's own
# `lt`. Everything joint-specific (the per-axis support registry, the pinned
# axes, the mode-Hessian splice) is resolved before this point, which is what
# lets the four backends share one dispatch.
.joint_cand_spec <- function(prep, vary, refit_log_marginal) {
    if (length(vary) == 0L) return(NULL)
    .k_cand_spec(lt = .joint_pareto_make_lt(prep, vary, refit_log_marginal),
                 u_hat = prep$u_hat[vary],
                 Su = prep$Su[vary, vary, drop = FALSE],
                 u_grid = prep$u_grid[, vary, drop = FALSE],
                 w = prep$w,
                 proposal_source = prep$proposal_source,
                 names = prep$cn[vary])
}


# Bootstrap + closed-form uncertainty of a CHOSEN proposal's outer Pareto-k-hat
# `best` is the proposal the dispatcher selected, carrying its
# raw finite importance log-ratios `lr`. The k-hat is a single fixed number for
# this fit + proposal; its sampling uncertainty GIVEN the proposal is estimated by
# resampling those SAME ratios with replacement and re-fitting the GPD tail at the
# same `tail_points` (`k_bootstrap` replicates), which is free -- no new inner
# solves -- and estimator-agnostic. Bootstrap measures how UNSTABLE the current
# tail estimate is; it cannot create tail information. A tighter k needs more
# ACTUAL tail ratios, i.e. a larger `control$k_samples`, NOT a larger
# `control$k_bootstrap`.
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
# ONCE over `n_samples` importance draws -- the resolved `control$k_samples`,
# reported as `diagnose_draws` -- and returns its raw
# finite log-ratios; the k-hat's sampling uncertainty is then estimated by
# bootstrapping those ratios (`k_bootstrap` replicates, re-fitting the GPD tail at
# the resolved `tail_points`), which adds NO inner solves. `control$k_samples` is
# where more tail information comes from (more actual tail ratios); `k_bootstrap`
# only quantifies the current estimate's instability and cannot create tail
# information. `k_samples` is NOT a pure precision knob at the automatic tail
# rule: `min(S/5, 3 sqrt(S))` fits a fraction that shrinks as `3 / sqrt(S)`, so a
# larger budget reads a deeper quantile of the weight distribution and can move
# the k-hat by several units (gcol33/tulpa#631). Holding `tail_points` in
# proportion to `k_samples` holds the estimand and makes the extra draws pure
# precision, which is what the k_quality draw rung does. `k_tail_points` (NULL = the automatic PSIS rule) is an expert
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
.joint_pareto_k <- function(res, refit_log_marginal,
                            n_samples = .nl_diag("k_samples"),
                            proposal = NULL, arm_axes = NULL,
                            k_bootstrap = .nl_diag("k_bootstrap"),
                            k_tail_points = NULL,
                            k_conf_bands = NULL) {
    na_out <- function(x) list(pareto_k = NA_real_, is_ess = NA_real_,
                               proposal_source = NA_character_,
                               declined = .k_reason_of(x))
    prep <- .joint_pareto_prepare(res, refit_log_marginal, n_samples, proposal)
    if (.k_is_decline(prep)) return(na_out(prep))

    # Resolve the GPD tail-size request once: an explicit
    # `k_tail_points` beyond the 20%-of-draws ceiling is capped, with a single
    # user-facing warning HERE so the per-replicate bootstrap re-fits stay silent.
    tp_req <- if (is.null(k_tail_points)) NA_integer_ else as.integer(k_tail_points)
    if (is.finite(tp_req)) {
        cap <- as.integer(floor(0.2 * as.integer(n_samples)))
        if (tp_req > cap) {
            warning(sprintf(paste0(
                "k_tail_points = %d exceeds the 20%% PSIS tail cap; using %d ",
                "instead. Increase control$k_samples, not control$k_bootstrap, ",
                "to obtain more tail information."), tp_req, cap), call. = FALSE)
        }
    }

    .preserve_seed_in_frame()

    n_arm <- if (!is.null(arm_axes)) length(arm_axes) else 0L

    # Axes the grid pins to a single value (e.g. a copy `alpha` fixed at 0, or a
    # one-point dispersion axis) carry zero weighted variance, leaving Su rank
    # deficient and its Cholesky undefined. The importance proposal holds them
    # fixed at u_hat and is built only on the varying axes.
    vary  <- .joint_pareto_vary_axes(prep$Su)
    joint <- .k_dispatch(.joint_cand_spec(prep, vary, refit_log_marginal),
                         n_samples, tail_points = k_tail_points)
    if (.k_is_decline(joint)) return(na_out(joint))

    ju  <- .joint_pareto_uncertainty(joint$best, k_tail_points, k_bootstrap,
                                     k_conf_bands)
    # The aperture publishes the SELECTED proposal's ratios -- the ones `ju`
    # just fitted the reported shape on. The dispatch above scored several
    # candidates (each moment-matching pass, the grid mixture, the skew-normal
    # rescue) and kept one; the per-arm passes below score more. Writing here,
    # after the choice and before those, is what makes the aperture reproduce
    # the number the fit reports.
    .kdiag_capture(joint$best$lr, tail_points = k_tail_points,
                   scope = paste0("joint nested (", joint$source, ")"))
    out <- list(pareto_k = ju$pareto_k, is_ess = ju$is_ess,
                proposal_source = joint$source,
                pareto_k_first_pass = joint$first_pass_k %||% NA_real_,
                declined = NA_character_,
                outer_skew = joint$outer_skew,
                pareto_k_se_boot               = ju$se_boot,
                pareto_k_ci_low                = ju$ci_low,
                pareto_k_ci_high               = ju$ci_high,
                pareto_k_se_formula            = ju$se_formula,
                pareto_k_tail_points           = ju$tail_points,
                pareto_k_tail_points_requested = tp_req,
                pareto_k_band_confident        = ju$band_confident,
                pareto_k_conf_bands            = ju$conf_bands,
                # The proposal's (mode, covariance) in the per-axis
                # unconstrained coordinate, exactly as fit to score this k-hat
                # (grid-weighted moments, or the mode-Hessian / delta-collapse
                # FD rescue -- see .joint_pareto_prepare()). Exposed so an
                # outer-grid rescue can re-center a
                # collapsed axis from the SAME (mode, H) rather than
                # re-optimizing.
                mode_u     = prep$u_hat,
                cov_u      = prep$Su,
                axis_tags  = prep$tags,
                axis_names = prep$cn)

    if (n_arm > 0L) {
        pa <- lapply(seq_len(n_arm), function(a) {
            v <- intersect(vary, as.integer(arm_axes[[a]]))
            none <- list(k = NA_real_, ess = NA_real_, se = NA_real_,
                         lo = NA_real_, hi = NA_real_, sef = NA_real_,
                         tp = NA_integer_, conf = NA)
            if (length(v) == 0L) return(none)
            s <- .k_dispatch(.joint_cand_spec(prep, v, refit_log_marginal),
                             n_samples, tail_points = k_tail_points)
            if (.k_is_decline(s)) return(none)
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
# Returns a named list arm_name -> integer column indices, or
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
# `pareto_k_se_boot` / `pareto_k_ci_low` / `pareto_k_ci_high`
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

# Attach the outer-integration REGIME and the outer hyperparameter skewness --
# the context that keeps a bare `pareto_k` threshold from being the whole
# story. Single source for both joint attach paths, and called whether or not
# the k-hat diagnostic ran: the regime is read off the stored grid weights, so
# it costs nothing and is available even with `control$diagnose_k = FALSE`.
#
#   * `pareto_k_regime` -- "spread" / "collapsed_interior" / "collapsed_edge".
#   * `pareto_k_grid_edge_axes` / `pareto_k_grid_edge_sides` -- which axes the
#     collapsed mode sits against, and on which side.
#   * `pareto_k_grid_edge_mass_axes` / `_sides` -- which axes hold material
#     weight on one of their own boundary nodes, and on which side. Read in
#     every regime, since a well-spread grid truncates its own marginal just as
#     a collapsed one does.
#   * `pareto_k_outer_skew` -- per-axis skewness of the hyperparameter marginal
#     in the proposal's whitened coordinate, estimated only when the rescue pass
#     ran (a high k-hat), so `NULL` on a fit whose Gaussian proposal already fit.
.joint_attach_pareto_k_regime <- function(res, kd = NULL) {
    rg <- .joint_pareto_grid_regime(res)
    res$pareto_k_regime          <- if (is.null(rg)) NA_character_ else rg$regime
    res$pareto_k_grid_edge_axes  <- if (is.null(rg)) character(0) else rg$edge_axes
    res$pareto_k_grid_edge_sides <- if (is.null(rg)) character(0) else rg$edge_sides
    res$pareto_k_grid_edge_mass_axes  <-
        if (is.null(rg)) character(0) else rg$edge_mass_axes
    res$pareto_k_grid_edge_mass_sides <-
        if (is.null(rg)) character(0) else rg$edge_mass_sides
    res$pareto_k_outer_skew      <- if (is.null(kd)) NULL else kd$outer_skew
    res
}

# Placement-only mode-Hessian for a collapsed outer grid, computed
# INDEPENDENTLY of whether the full outer Pareto-k diagnostic ran. At
# `control$diagnose_k = FALSE` (the default) the full diagnostic in
# `.joint_pareto_k()` never executes, so `res$pareto_k_mode_u` / `cov_u` /
# `axis_tags` / `axis_names` -- what the auto-recenter rescues
# (`.joint_sigma_grid_rescue()` / `.joint_multi_sigma_grid_rescue()` in
# `R/nested_laplace_auto_grid.R`) consume to recentre a railed axis -- are
# never attached, so a fit that collapses onto a field-SD ceiling stays railed
# even though `SIGMA_GRID = "auto"` was requested.
#
# Calls the SAME `.joint_pareto_prepare()` the full diagnostic scores its
# proposal from -- not a re-derived subset -- so the (mode, covariance) this
# attaches is exactly what the diagnostic would have attached, whichever of
# its three sources applies: the grid-weighted moment (pure arithmetic over
# the stored grid, no extra solve, when `collapsed_edge` still leaves SOME
# axis with weighted spread -- `ess_grid` in `[1, 2)`), the CCD `proposal`
# splice (already built at grid-construction time, independent of
# `diagnose_k`), or the delta-collapse finite-difference Hessian at the modal
# cell (one batched stencil call, only when the grid weight has concentrated
# on essentially one cell). Threading `proposal` through matters: without it
# a CCD-gridded fit would fall back to the (potentially still-informative)
# grid moment instead of the sharper mode-Hessian the CCD integrator already
# has. `n_samples` only gates `.joint_pareto_prepare()`'s sample-floor decline
# (irrelevant here -- no importance draws are taken), so a fixed floor value
# is enough. A no-op (returns `res` unchanged) unless the grid has actually
# collapsed onto a boundary (`pareto_k_regime == "collapsed_edge"`, already
# attached by `.joint_attach_pareto_k_regime()` regardless of `diagnose_k`) --
# so this is zero extra cost for the common fit whose grid already brackets
# the mode.
.joint_attach_pareto_k_placement <- function(res, refit_log_marginal,
                                             proposal = NULL) {
    if (!identical(res$pareto_k_regime, "collapsed_edge")) return(res)
    prep <- .joint_pareto_prepare(res, refit_log_marginal, .PSIS_MIN_EVAL, proposal)
    if (.k_is_decline(prep)) return(res)
    res$pareto_k_mode_u     <- prep$u_hat
    res$pareto_k_cov_u      <- prep$Su
    res$pareto_k_axis_tags  <- prep$tags
    res$pareto_k_axis_names <- prep$cn
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
# the reason it fell short.
#
# A miss is CLASSIFIED, not just reported, because the two causes take different
# levers and the escalation loop reads which one it is (`k_quality_miss`):
#   * `"resolution"` -- the k-hat sits CONFIDENTLY outside the requested band.
#     The Gaussian proposal the grid places itself with does not fit the
#     hyperparameter posterior, which is a property of the integration grid, so
#     refining the grid is the lever.
#   * `"precision"` -- the bootstrap CI STRADDLES a band boundary. The point
#     estimate may already be inside the requested band; what prevents
#     confirming it is the width of the interval, i.e. the variance of the
#     k-hat ESTIMATOR. That is reduced by fitting the GPD tail on more actual
#     tail ratios, which no grid refinement touches, so it is the miss that
#     still has a lever once refinement is exhausted.
# `NA_character_` when there is no miss to classify (the band was reached, the
# intent carried no target band, or the diagnostic did not run).
#
# `conf_bands` is an explicit override; when NULL it uses
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
    res$k_quality_miss      <- NA_character_

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
    res$k_quality_miss    <- if (reached) NA_character_
                             else if (!conf) "precision" else "resolution"
    res$k_quality_reason  <-
        if (reached) "requested band reached"
        else if (!conf)
            "k-hat interval crosses a band boundary; raise control$k_samples or refine the grid"
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
# with `diagnose_k = FALSE` the fields are present but NA -- but
# `pareto_k_mode_u` / `cov_u` / `axis_tags` / `axis_names` are still populated
# on a collapsed-edge grid via the diagnose_k-independent placement path, so
# the auto-recenter rescue in `R/nested_laplace_auto_grid.R` engages
# regardless of `diagnose_k`.
.joint_attach_pareto_k_single <- function(res, kernel_fn, hp_fn,
                                          max_iter = 50L,
                                          diagnose_k = TRUE,
                                          diagnose_draws = .nl_diag("k_samples"),
                                          n_threads_outer = 1L,
                                          pareto_k_by_arm = FALSE,
                                          k_bootstrap = .nl_diag("k_bootstrap"),
                                          k_tail_points = NULL,
                                          k_conf_bands = NULL) {
    res$pareto_k        <- NA_real_
    res$pareto_k_is_ess <- NA_real_
    res$pareto_k_scope  <- "outer (hyperparameter) Gaussian proposal"
    res$pareto_k_proposal_source <- NA_character_
    res$pareto_k_declined <- NA_character_
    # The grid regime is read off stored weights, so it is attached even when the
    # k-hat diagnostic is off -- a collapsed or boundary-pinned grid is worth
    # knowing about regardless.
    res <- .joint_attach_pareto_k_regime(res)

    warm        <- .joint_modal_mode(res)
    warm_arg    <- if (is.null(warm)) NULL else list(mode = warm)
    modal_theta <- .joint_modal_theta(res)
    k_max_iter  <- min(as.integer(max_iter), .K_DIAG_MAX_ITER)
    n_to        <- as.integer(n_threads_outer)
    knobs       <- .kdiag_knobs()

    # One kernel call over the importance batch, with Shamanskii factor reuse
    # (off-factor steps scatter grad-only) + per-cell warm start, both attacking
    # the dominant per-iteration scatter cost on the beta cover arm
    # `x_init_per_cell` is an [S x n_x] warm matrix or NULL.
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

    if (!isTRUE(diagnose_k)) {
        # Placement-only recenter curvature: cheap (one
        # batched FD-stencil solve, only when the grid actually collapsed on a
        # boundary) even though the full diagnostic below never runs.
        res <- .k_attach_declined(res, .k_decline("not_requested"))
        return(.joint_attach_pareto_k_placement(res, solve_fn))
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
    res <- .k_attach_declined(res, kd)
    res$pareto_k_mode_u     <- kd$mode_u
    res$pareto_k_cov_u      <- kd$cov_u
    res$pareto_k_axis_tags  <- kd$axis_tags
    res$pareto_k_axis_names <- kd$axis_names
    res <- .joint_attach_pareto_k_regime(res, kd)
    res <- .joint_attach_pareto_k_uncertainty(res, kd)
    res <- .joint_attach_by_arm_k(res, kd)
    res
}
