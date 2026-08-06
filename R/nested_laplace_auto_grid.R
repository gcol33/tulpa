# Shared mode-Hessian outer-axis recentering (gcol33/tulpa#289, #290).
#
# Every nested-Laplace family builds its outer hyperparameter grid from a
# FIXED default axis in original coordinates (e.g. bym2/icar/car_proper's
# `sigma_grid = exp(seq(log(0.1), log(3), length.out = 5))`). A dataset whose
# field-SD posterior mode sits above the top node rails onto that ceiling:
# every outer weight collapses onto the boundary node
# (`pareto_k_regime = "collapsed_edge"`, edge side `"upper"`), silently,
# because the fixed grid never had a chance to bracket the true mode.
#
# `.nl_recenter_log_axis()` is the shared node-generator: given a mode and SD
# already computed on the log-transformed axis -- reusing whatever
# mode-Hessian a fit already computed for its outer Pareto-k diagnostic (see
# `R/nested_laplace_joint_pareto_k.R`) rather than re-optimizing -- it lays a
# new log-spaced grid centred at the mode. This is placement, not a second
# optimizer: callers detect the collapse (the `pareto_k_regime` diagnostic
# every family already attaches, gcol33/tulpa#276), recentre once, and
# refit; a second attempt composes a light default PC(U, alpha) prior
# (`.NL_DEFAULT_SIGMA_PC_PRIOR`) for genuinely unidentified (near-separation)
# cases where the mode itself keeps running rather than settling on finite
# curvature.

# `mode_u` / `sd_u` are on the log scale (u = log(theta)). Returns a sorted
# numeric vector of `n_pts` positive theta values spanning
# `mode_u +/- span * sd_u`, or NULL when the curvature is not usable
# (non-finite / non-positive SD) -- the caller then leaves the existing grid
# untouched rather than centre on a meaningless spread.
# `sd_u` is clamped to `[min_sd_u, max_sd_u]`: a floor so a razor-sharp local
# curvature does not collapse the new grid to near-duplicate nodes (the
# purpose of the retry is to bracket the mode with actual spread), and a
# ceiling so a near-flat direction does not fling nodes to implausible
# extremes.
.nl_recenter_log_axis <- function(mode_u, sd_u, n_pts = 5L, span = 2.5,
                                   min_sd_u = 0.15, max_sd_u = 3) {
    if (length(mode_u) != 1L || length(sd_u) != 1L) return(NULL)
    if (!is.finite(mode_u) || !is.finite(sd_u) || sd_u <= 0) return(NULL)
    sd_u  <- min(max(sd_u, min_sd_u), max_sd_u)
    u_seq <- seq(mode_u - span * sd_u, mode_u + span * sd_u,
                 length.out = as.integer(n_pts))
    sort(exp(u_seq))
}

# Build the recentered axis for `axis_name` from the (mode, covariance) a
# fit's outer Pareto-k diagnostic already attached -- `mode_u` / `cov_u` /
# `axis_tags` / `axis_names` are the `res$pareto_k_mode_u` /
# `res$pareto_k_cov_u` / `res$pareto_k_axis_tags` / `res$pareto_k_axis_names`
# fields (`R/nested_laplace_joint_pareto_k.R`, `R/nested_laplace.R`). Only
# recentres a positive-scale ("log"-tagged) axis; declines (returns NULL) for
# an axis absent from the grid, an axis on a different transform (e.g. a
# BYM2 `rho` or a CAR_proper `rho_car`), or when the diagnostic that would
# supply the curvature did not run or itself declined (an unguessable axis
# elsewhere in the same grid, such as `rho_car`, makes the whole proposal
# decline -- see `.joint_pareto_axis_tags()` -- so a fit's `sigma` mode is
# only ever recentred when EVERY axis in that fit's grid is guessable).
.nl_axis_recenter_from_fit <- function(mode_u, cov_u, axis_tags, axis_names,
                                       axis_name, n_pts = 5L, span = 2.5) {
    if (is.null(mode_u) || is.null(cov_u) || is.null(axis_names)) return(NULL)
    j <- match(axis_name, axis_names)
    if (is.na(j)) return(NULL)
    if (is.null(axis_tags) || !identical(axis_tags[j], "log")) return(NULL)
    if (!is.matrix(cov_u) || nrow(cov_u) < j || ncol(cov_u) < j) return(NULL)
    sd_u <- suppressWarnings(sqrt(cov_u[j, j]))
    .nl_recenter_log_axis(mode_u[j], sd_u, n_pts = n_pts, span = span)
}

# Weakly-informative default for the donor field-SD axis, engaged only on the
# SECOND auto-recenter attempt (a genuinely runaway mode -- see
# `.nl_recenter_log_axis()`'s guards) and only when the user supplied no
# `prior_sigma` of their own. `U = 3` matches the retired fixed-grid ceiling,
# so the shrinkage is only felt past where the old default axis already
# stopped; `P(sigma > 3) = 0.01` is weak enough to leave a data-identified
# mode essentially untouched (gcol33/tulpa#289).
.NL_DEFAULT_SIGMA_PC_PRIOR <- list("pc.prec", c(U = 3, alpha = 0.01))

# Single-block joint auto-recenter rescue (gcol33/tulpa#289). `res` is the
# just-completed single-block fit (bym2 / icar / car_proper); `refit(prior_i,
# prior_sigma_i)` reruns the SAME fit with a modified prior / prior_sigma and
# returns the new result (already carrying its own `pareto_k_regime`, since
# it re-enters the normal driver). Declines immediately (returns `res`,
# `prior`, `prior_sigma` unchanged) when the user supplied their own
# `sigma_grid` (an explicit override always wins), the prior has no `type` in
# `c("bym2", "icar", "car_proper")` (multi-block priors have no top-level
# `$type` and fall through here harmlessly), or the grid never collapsed onto
# the sigma axis in the first place -- so this is a zero-cost, byte-stable
# no-op for every fit that did not need it.
#
# Two attempts, both reusing the mode/Hessian the outer Pareto-k diagnostic
# already computed rather than a fresh optimization (see
# `.nl_axis_recenter_from_fit()`):
#   1. Recentre `sigma_grid` alone.
#   2. If STILL `collapsed_edge` (a genuinely unidentified / near-separation
#      case whose mode has no finite curvature to settle on), additionally
#      apply the light default PC(U=3, alpha=0.01) prior (only if the user
#      set no `prior_sigma` of their own) and recentre once more.
# Gives up (keeps the last, still-improved fit) rather than looping when a
# recenter attempt cannot be built (e.g. the diagnostic declined because
# another axis in the same grid is unguessable, such as CAR_proper's
# `rho_car` -- see `.nl_axis_recenter_from_fit()`).
#
# Returns `list(res=, prior=, prior_sigma=)`: the possibly-refit result, and
# the prior / prior_sigma that produced it, so a caller chaining further
# refinement (e.g. the k_quality escalation loop) continues from the
# recentered grid rather than the original one.
.joint_sigma_grid_rescue <- function(res, prior, prior_sigma, refit,
                                     max_attempts = 2L) {
    out <- list(res = res, prior = prior, prior_sigma = prior_sigma)
    if (!is.null(prior$sigma_grid)) return(out)
    type <- tolower(prior$type %||% "")
    if (!type %in% c("bym2", "icar", "car_proper")) return(out)

    cur_prior       <- prior
    cur_prior_sigma <- prior_sigma
    attempt <- 0L
    while (attempt < max_attempts &&
           identical(res$pareto_k_regime, "collapsed_edge") &&
           "sigma" %in% (res$pareto_k_grid_edge_axes %||% character(0))) {
        attempt <- attempt + 1L
        new_axis <- .nl_axis_recenter_from_fit(
            res$pareto_k_mode_u, res$pareto_k_cov_u,
            res$pareto_k_axis_tags, res$pareto_k_axis_names, "sigma")
        if (is.null(new_axis)) break
        cur_prior$sigma_grid <- new_axis
        if (attempt >= 2L && is.null(cur_prior_sigma)) {
            cur_prior_sigma <- .NL_DEFAULT_SIGMA_PC_PRIOR
        }
        res <- refit(cur_prior, cur_prior_sigma)
        res$outer_grid_placement           <- "auto_recentered"
        res$outer_grid_recenter_attempts   <- attempt
        res$outer_grid_prior_added         <- attempt >= 2L && is.null(prior_sigma)
        out <- list(res = res, prior = cur_prior, prior_sigma = cur_prior_sigma)
    }
    out
}

# Mode + FD-Hessian covariance of a single-block REGISTRY fit's outer grid
# (gcol33/tulpa#290) -- the standalone `tulpa_nested_laplace()` counterpart
# of the joint path's `.joint_pareto_prepare()` delta-collapse rescue,
# reusing the SAME generic tagging (`.joint_pareto_block_tags()`) and
# FD-Hessian machinery (`.joint_pareto_mode_cov()`) rather than a fresh
# implementation. `type` is the (lower-case) registry family name;
# `refit_log_marginal(theta_mat)` re-evaluates the inner marginal at an
# arbitrary `[S x d]` theta matrix (columns named per `res$theta_names`)
# through the SAME single-block kernel the fit used. Declines (NULL) when
# any axis in the grid has unguessable support under
# `.joint_pareto_block_tags()` -- e.g. car_proper's `rho`, the identical
# limitation the joint path already has for that family.
.nl_registry_axis_mode_cov <- function(res, type, refit_log_marginal) {
    cn <- res$theta_names
    tg <- res$theta_grid
    if (is.null(cn) || is.null(tg)) return(NULL)
    if (!is.matrix(tg)) tg <- matrix(as.numeric(tg), ncol = 1L)
    colnames(tg) <- cn
    w <- res$weights
    if (is.null(w) || length(w) != nrow(tg)) return(NULL)

    tags <- .joint_pareto_block_tags(type, cn)
    if (is.null(tags)) return(NULL)
    d <- ncol(tg)
    u_grid <- matrix(0, nrow(tg), d)
    for (j in seq_len(d)) u_grid[, j] <- .joint_pareto_fwd(tags[j], as.numeric(tg[, j]))
    if (any(!is.finite(u_grid))) return(NULL)
    u_mode <- as.numeric(u_grid[which.max(w), ])

    refit_lm <- function(theta_mat) {
        colnames(theta_mat) <- cn
        refit_log_marginal(theta_mat)
    }
    cov_h <- .joint_pareto_mode_cov(u_mode, tags, cn, refit_lm, d, vary = seq_len(d))
    if (is.null(cov_h)) return(NULL)
    list(u_mode = u_mode, cov = cov_h, tags = tags, col_names = cn)
}

# Multi-block joint auto-recenter rescue (gcol33/tulpa#289/#290), the
# multi-block counterpart of `.joint_sigma_grid_rescue()`. Scope: a COPY
# block's own scalar `sigma` axis (icar / bym2 / car_proper / rw1 / rw2 /
# ar1 / iid copy blocks all build it via the identical
# `p$sigma_grid %||% exp(seq(log(0.1), log(3), length.out = 5))` default --
# see `.joint_block_axis_grid()`, `R/nested_laplace_joint_multi.R`), the
# donor field amplitude a copy coefficient scales -- the exact axis role
# gcol33/tulpa#289's driver (`occu_cover`) hits. A non-copy block's axis
# (RW1/RW2 tau, MCAR's log-Cholesky Sigma, ...) reuses
# `.NL_REGISTRY`/`.nl_block_axis_grid()` and is out of scope here (a
# materially larger, per-block-type surface than the single shared "donor
# sigma" convention every copy block shares).
#
# `res$blocks`/`res$axis_offsets` (attached whenever a fit carries >= 1
# block) name each block's columns `b<index>.<axis>` (1-based); `cp` is the
# resolved copy spec (`.resolve_copy_multi()`) telling which block indices
# are copy blocks. Fixes at most one collapsed copy block per attempt (the
# documented case is a single copy block; a fit with several SIMULTANEOUSLY
# collapsed copy blocks partially improves within `max_attempts` rather than
# looping without bound).
.joint_multi_sigma_grid_rescue <- function(res, prior, copy, cp, prior_sigma,
                                           refit, max_attempts = 2L) {
    out <- list(res = res, prior = prior, prior_sigma = prior_sigma)
    if (!.is_multi_block_prior(prior) || is.null(cp) || !isTRUE(cp$has_copy)) {
        return(out)
    }

    cur_prior       <- prior
    cur_prior_sigma <- prior_sigma
    attempt <- 0L
    while (attempt < max_attempts &&
           identical(res$pareto_k_regime, "collapsed_edge")) {
        edge_axes <- res$pareto_k_grid_edge_axes %||% character(0)
        target_b <- NULL
        for (b0 in cp$copy_blocks_zero) {
            b <- b0 + 1L
            axis_name <- paste0("b", b, ".sigma")
            if (axis_name %in% edge_axes && is.null(cur_prior[[b]]$sigma_grid)) {
                target_b <- b
                break
            }
        }
        if (is.null(target_b)) break
        attempt <- attempt + 1L
        new_axis <- .nl_axis_recenter_from_fit(
            res$pareto_k_mode_u, res$pareto_k_cov_u,
            res$pareto_k_axis_tags, res$pareto_k_axis_names,
            paste0("b", target_b, ".sigma"))
        if (is.null(new_axis)) break
        cur_prior[[target_b]]$sigma_grid <- new_axis
        if (attempt >= 2L && is.null(cur_prior_sigma)) {
            cur_prior_sigma <- .NL_DEFAULT_SIGMA_PC_PRIOR
        }
        res <- refit(cur_prior, cur_prior_sigma)
        res$outer_grid_placement           <- "auto_recentered"
        res$outer_grid_recenter_attempts   <- attempt
        res$outer_grid_prior_added         <- attempt >= 2L && is.null(prior_sigma)
        out <- list(res = res, prior = cur_prior, prior_sigma = cur_prior_sigma)
    }
    out
}

# Per-family value/other axis field names for the standalone registry
# rescue below. `value` is the prior-list field the positive-scale axis
# lives on (recentred on collapse); `other` (when present) is the paired
# bounded axis's field, re-crossed with the new value axis via the same
# `expand.grid()` a family's own `defaults()` uses; `other_axis` is that
# axis's column name in `res$theta_grid`. `col_name` is the axis's REAL name
# in `res$theta_names` (what `.joint_pareto_block_tags()` / the FD-Hessian
# stencil key on); `value_axis_name` is separately what
# `pareto_k_grid_edge_axes` calls it -- for icar these differ, because
# `.joint_pareto_grid_regime()` coerces a bare single-axis vector grid
# (icar's `theta_grid` is a plain numeric vector, not a named matrix) to a
# 1-column matrix generically named `"theta"`, while `theta_names` still
# says `"tau"`.
.NL_REGISTRY_AXIS_FIELD <- list(
    icar = list(value = "tau_grid",   value_axis_name = "theta", col_name = "tau"),
    bym2 = list(value = "sigma_grid", value_axis_name = "sigma", col_name = "sigma",
               other = "rho_grid", other_axis = "rho")
)

# Standalone (non-joint) `tulpa_nested_laplace()` single-block registry
# rescue (gcol33/tulpa#290) -- the registry generalization of
# `.joint_sigma_grid_rescue()`. Scope: icar's `tau_grid` and bym2's
# `sigma_grid`, the two fixed-ceiling defaults #290 confirms
# (`exp(seq(log(0.1|0.3), log(3|30), length.out = 5|9))`). car_proper
# declines (its `rho` axis is unguessable under
# `.joint_pareto_block_tags()`, the same limitation the joint path already
# has); MCAR's log-Cholesky axis geometry is a materially different
# recentering problem and is out of scope here.
#
# One recenter attempt (not the joint path's two): the "runaway mode needs
# a regularizing prior" pathology `.joint_sigma_grid_rescue()`'s second
# attempt targets is specific to a donor/copy-coupled fit pushing toward
# near-separation (gcol33/tulpa#289's actual driver); a standalone
# single-response fit has no such coupling, so a geometry-only recenter is
# the proportionate fix here.
#
# `refit(prior_i)` reruns the full single-block pipeline (dispatch, weights,
# moments, pareto-k) at a modified prior; `refit_log_marginal(prior_i,
# theta_mat)` re-evaluates just the inner marginal at an arbitrary theta
# matrix (used by the FD-Hessian stencil, many more calls, none of them a
# full fit).
.nl_registry_grid_rescue <- function(res, type, prior, refit, refit_log_marginal,
                                     max_attempts = 1L) {
    out <- list(res = res, prior = prior)
    fields <- .NL_REGISTRY_AXIS_FIELD[[type]]
    if (is.null(fields) || !is.null(prior[[fields$value]])) return(out)

    cur_prior <- prior
    attempt <- 0L
    while (attempt < max_attempts &&
           identical(res$pareto_k_regime, "collapsed_edge") &&
           fields$value_axis_name %in% (res$pareto_k_grid_edge_axes %||% character(0))) {

        mc <- .nl_registry_axis_mode_cov(
            res, type, function(theta_mat) refit_log_marginal(cur_prior, theta_mat))
        if (is.null(mc)) break

        j <- match(fields$col_name, mc$col_names)
        if (is.na(j)) break
        new_axis <- .nl_recenter_log_axis(mc$u_mode[j], sqrt(mc$cov[j, j]))
        if (is.null(new_axis)) break

        attempt <- attempt + 1L
        if (is.null(fields$other)) {
            cur_prior[[fields$value]] <- new_axis
        } else {
            tg <- res$theta_grid
            other_vals <- sort(unique(as.numeric(tg[, fields$other_axis])))
            gr <- expand.grid(value = new_axis, other = other_vals,
                              KEEP.OUT.ATTRS = FALSE)
            cur_prior[[fields$value]] <- gr$value
            cur_prior[[fields$other]] <- gr$other
        }
        res <- refit(cur_prior)
        res$outer_grid_placement         <- "auto_recentered"
        res$outer_grid_recenter_attempts <- attempt
        out <- list(res = res, prior = cur_prior)
    }
    out
}
