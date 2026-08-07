# Shared mode-Hessian outer-axis recentering (gcol33/tulpa#289, #290, #293).
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
#
# Three cross-cutting concerns live here because all four rescues (joint
# single-block, joint multi-block copy, standalone registry, spatiotemporal)
# share them:
#
#   * AXIS PROVENANCE -- whether an axis the incoming prior carries is a USER
#     PIN (always wins, never recentred) or a DEFAULT some layer wrote for
#     convenience (`.nl_axis_is_pinned()`, and the `auto_grid()` declaration a
#     caller marks its own default with).
#   * AXIS NAMING -- one axis is named three ways depending on the grid it
#     landed in (`sigma`, `b<k>.sigma`, `theta`); `.nl_axis_alias()` resolves
#     all three so a rescue matches the axis it is looking for
#     (`.nl_edge_axis_hit()` / `.nl_axis_index()`).
#   * DECLINE REASONS -- a rescue that does not run says why
#     (`res$outer_grid_recenter_declined`), so an inert auto-recenter is
#     visible in the fit instead of indistinguishable from one that was never
#     needed (`.nl_decline_recenter()`).

# The default field-SD axis for the areal joint backends: 5 log-spaced nodes
# over [0.1, 3]. Single source of truth -- the three single-block
# `build_grids` closures (`R/nested_laplace_joint_backends.R`), the
# multi-block copy-block axis builder (`.joint_block_axis_grid()`,
# `R/nested_laplace_joint_multi.R`), the bym2 registry default (`.NL_REGISTRY`,
# `R/nested_laplace.R`) and `.nl_axis_matches_default()` below all read it
# from here.
.nl_default_sigma_axis <- function() exp(seq(log(0.1), log(3), length.out = 5))

# --- axis provenance ---------------------------------------------------------
#
# A rescue must recentre a DEFAULT axis and leave a USER PIN alone, so it needs
# to tell the two apart. Field presence does not answer that question: a
# consumer package that computes the engine's own default itself (because it
# also derives a second axis from it, or feeds the same vector to several
# blocks) writes a non-NULL `prior$sigma_grid` on a fit where the user named no
# grid at all, and `!is.null()` reads that as an override
# (gcol33/tulpa#293: every `occu_cover()` fit's auto-recenter was inert for
# exactly this reason). Provenance is only known to the layer that CHOSE the
# values, so `auto_grid()` lets that layer say so.

#' Mark an outer hyperparameter grid as a default rather than a pin
#'
#' @description
#' Declares that an outer-grid axis (`sigma_grid`, `tau_grid`, ...) on a
#' nested-Laplace `prior` block carries a *default* the caller computed, not a
#' choice the user made. The auto-recenter pass
#' (\code{outer_grid_placement}, gcol33/tulpa#289) leaves a user-pinned axis
#' exactly where it is, and re-centres a marked one on the posterior mode when
#' the fit rails against its ceiling.
#'
#' Wrapper packages are the intended caller: one that builds a default axis of
#' its own -- because it derives a second axis from it, or hands the same
#' vector to several blocks -- would otherwise be indistinguishable from a user
#' who pinned that grid deliberately. Mark it and the recenter stays live.
#' A grid whose node set is exactly the engine's own default axis is recognised
#' without a mark; anything else needs one.
#'
#' The mark is an attribute, so it is dropped by `sort()`, `[`, `c()` and
#' `as.numeric()`: build the axis first, mark it last.
#'
#' @param x Numeric vector of grid nodes.
#' @return `x` as a numeric vector carrying the marker attribute.
#' @seealso [is_auto_grid()], [tulpa_nested_laplace_joint()]
#' @examples
#' prior <- list(type = "icar", sigma_grid = auto_grid(c(0.1, 0.5, 1, 2, 3)))
#' is_auto_grid(prior$sigma_grid)
#' @export
auto_grid <- function(x) {
    x <- as.numeric(x)
    if (!length(x) || anyNA(x)) {
        stop("`auto_grid()` takes a non-empty numeric grid with no NA.",
             call. = FALSE)
    }
    attr(x, "tulpa_auto_grid") <- TRUE
    x
}

#' Is an outer hyperparameter grid marked as a default?
#'
#' @param x Any object.
#' @return `TRUE` when `x` carries the [auto_grid()] marker.
#' @seealso [auto_grid()]
#' @examples
#' is_auto_grid(auto_grid(c(0.5, 1, 2)))
#' is_auto_grid(c(0.5, 1, 2))
#' @export
is_auto_grid <- function(x) isTRUE(attr(x, "tulpa_auto_grid", exact = TRUE))

# Per-field candidate default axes. A grid equal (as a node SET, so a
# Cartesian expansion against a second axis still matches) to one of these is
# the engine's own default coming back in through a caller's prior, and carries
# no information a pin would add -- treated as a default, not a pin. Keyed by
# grid field; the values are every default the engine itself lays on that field
# (`tau_grid` has two: the icar registry's 9-node axis and car_proper's 5-node
# one).
.NL_DEFAULT_AXIS_CANDIDATES <- list(
    sigma_grid = function() list(.nl_default_sigma_axis()),
    tau_grid   = function() list(.default_tau_grid(),
                                 exp(seq(log(0.3), log(30), length.out = 5)))
)

.nl_axis_matches_default <- function(value, field) {
    gen <- .NL_DEFAULT_AXIS_CANDIDATES[[field]]
    if (is.null(gen)) return(FALSE)
    u <- sort(unique(as.numeric(value)))
    if (!length(u)) return(FALSE)
    for (d in gen()) {
        du <- sort(unique(as.numeric(d)))
        if (length(du) == length(u) && isTRUE(all.equal(du, u))) return(TRUE)
    }
    FALSE
}

# Names of the grid fields on ONE block that carry the `auto_grid()` marker.
.nl_block_auto_fields <- function(block) {
    if (!is.list(block) || !length(block)) return(character(0))
    nm <- names(block) %||% character(0)
    if (!length(nm)) return(character(0))
    marked <- vapply(block, is_auto_grid, logical(1))
    nm[marked & nzchar(nm)]
}

.nl_block_strip_auto <- function(block) {
    for (f in .nl_block_auto_fields(block)) {
        attr(block[[f]], "tulpa_auto_grid") <- NULL
    }
    block
}

# Record which axes a prior declared as defaults, and hand back the prior with
# the markers removed, so nothing downstream of the front door ever sees an
# attributed numeric (grid values reach C++, `expand.grid()` and `cbind()`
# unchanged). `auto` is a character vector for a single-block prior and a
# per-block list for a multi-block one -- read it back with
# `.nl_auto_fields_at()`.
.nl_grid_provenance <- function(prior) {
    if (.is_multi_block_prior(prior)) {
        auto <- lapply(prior, .nl_block_auto_fields)
        prior <- lapply(prior, .nl_block_strip_auto)
        return(list(prior = prior, auto = auto))
    }
    if (!is.list(prior)) return(list(prior = prior, auto = character(0)))
    list(prior = .nl_block_strip_auto(prior), auto = .nl_block_auto_fields(prior))
}

.nl_auto_fields_at <- function(auto, block_index = NULL) {
    if (is.null(auto)) return(character(0))
    if (is.null(block_index)) {
        if (is.list(auto)) return(character(0))
        return(as.character(auto))
    }
    if (!is.list(auto) || block_index > length(auto)) return(character(0))
    as.character(auto[[block_index]] %||% character(0))
}

# THE provenance predicate every rescue guards on. `block` is the prior block
# carrying the axis, `field` its grid field, `auto_fields` the marker record
# `.nl_grid_provenance()` took for that block. An absent axis, a marked one,
# and one whose nodes are the engine's own default are all defaults; anything
# else is a pin the rescue must leave alone.
.nl_axis_is_pinned <- function(block, field, auto_fields = character(0)) {
    g <- if (is.list(block)) block[[field]] else NULL
    if (is.null(g)) return(FALSE)
    if (field %in% auto_fields) return(FALSE)
    if (is_auto_grid(g)) return(FALSE)
    !.nl_axis_matches_default(g, field)
}

# --- axis naming -------------------------------------------------------------
#
# The same physical axis is named three ways depending on the grid it landed
# in: bare (`"sigma"`) in a single-block joint grid, block-prefixed
# (`"b2.sigma"`) in a multi-block one, and `"theta"` when a single-axis grid
# stored as a bare vector is coerced to a 1-column matrix by
# `.joint_pareto_grid_regime()` (icar's registry path). A rescue that hard-codes
# one spelling silently misses the axis under the other two.
#
# `block_index` (1-based) is supplied by a multi-block caller. The bare /
# `"theta"` spellings are only accepted for a single-block caller or a fit that
# carries at most one block -- with several blocks every axis is prefixed, so an
# unprefixed match there would be attributing another block's axis.
.nl_axis_alias <- function(axis, block_index = NULL, n_blocks = 0L) {
    c(if (!is.null(block_index)) paste0("b", block_index, ".", axis),
      if (is.null(block_index) || n_blocks <= 1L) c(axis, "theta"))
}

.nl_fit_n_blocks <- function(res) length(res$blocks %||% list())

# Did the collapsed grid rail on `axis`? Reads the `pareto_k_grid_edge_axes`
# every family attaches regardless of `diagnose_k` (gcol33/tulpa#276, #292).
.nl_edge_axis_hit <- function(res, axis, block_index = NULL) {
    ea <- res$pareto_k_grid_edge_axes %||% character(0)
    if (!length(ea)) return(FALSE)
    any(.nl_axis_alias(axis, block_index, .nl_fit_n_blocks(res)) %in% ea)
}

# Column index of `axis` in a fit's axis-name vector. Falls back to the single
# column of a one-axis grid whatever it is named (that axis IS the family's
# scale axis) -- but only on a positive-scale ("log"-tagged) axis, so a lone
# bounded axis is declined rather than recentred on a guessed support.
.nl_axis_index <- function(axis_names, aliases, axis_tags = NULL) {
    if (is.null(axis_names) || !length(axis_names)) return(NA_integer_)
    j <- which(axis_names %in% aliases)
    if (length(j)) return(j[1L])
    if (length(axis_names) == 1L &&
        (is.null(axis_tags) || identical(axis_tags[1L], "log"))) return(1L)
    NA_integer_
}

# --- decline reasons ---------------------------------------------------------
#
# `res$outer_grid_recenter_declined` records why an applicable auto-recenter did
# not run: `"axis_pinned"` (the caller pinned the axis), `"grid_not_collapsed"`
# (the grid already brackets the mode, the common no-op),
# `"no_usable_curvature"` (the mode-Hessian the recenter needs was unavailable
# or degenerate), `"auto_recenter_disabled"` (`control$auto_recenter = FALSE`,
# the way to hold ANY grid -- the engine's own default axis included -- exactly
# where it is), `"grid_knobs_overridden"` (the spatiotemporal driver's
# grid-construction knobs were set explicitly), `"refit_failed"` (the recentred
# grid did not solve). Absent on a fit that WAS recentred, and never stamped by
# a rescue whose prior shape it does not apply to -- a fit carries the reason
# from the one rescue that could have run, not a tally of the others declining.
.nl_decline_recenter <- function(res, reason) {
    if (identical(res$outer_grid_placement, "auto_recentered")) return(res)
    res$outer_grid_recenter_declined <- reason
    res
}

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

# Build the recentered axis for `axis` from the (mode, covariance) a fit's
# outer Pareto-k diagnostic already attached -- `mode_u` / `cov_u` /
# `axis_tags` / `axis_names` are the `res$pareto_k_mode_u` /
# `res$pareto_k_cov_u` / `res$pareto_k_axis_tags` / `res$pareto_k_axis_names`
# fields (`R/nested_laplace_joint_pareto_k.R`, `R/nested_laplace.R`). `axis` is
# the bare axis name; `block_index` / `n_blocks` resolve it against the fit's
# own spelling (see `.nl_axis_alias()`). Only recentres a positive-scale
# ("log"-tagged) axis; declines (returns NULL) for an axis absent from the
# grid, an axis on a different transform (e.g. a BYM2 `rho` or a CAR_proper
# `rho_car`), or when the diagnostic that would supply the curvature did not
# run or itself declined (an unguessable axis elsewhere in the same grid, such
# as `rho_car`, makes the whole proposal decline -- see
# `.joint_pareto_axis_tags()` -- so a fit's `sigma` mode is only ever recentred
# when EVERY axis in that fit's grid is guessable).
.nl_axis_recenter_from_fit <- function(mode_u, cov_u, axis_tags, axis_names,
                                       axis, n_pts = 5L, span = 2.5,
                                       block_index = NULL, n_blocks = 0L) {
    if (is.null(mode_u) || is.null(cov_u) || is.null(axis_names)) return(NULL)
    aliases <- .nl_axis_alias(axis, block_index, n_blocks)
    j <- .nl_axis_index(axis_names, aliases, axis_tags)
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
# it re-enters the normal driver). `auto` is the provenance record
# `.nl_grid_provenance()` took at the front door.
#
# Declines (returns `res`, `prior`, `prior_sigma` unchanged) when the prior has
# no `type` in `c("bym2", "icar", "car_proper")` (multi-block priors have no
# top-level `$type` and fall through here harmlessly, unstamped), when the
# `sigma_grid` axis is PINNED (`.nl_axis_is_pinned()`: named by the caller and
# neither marked with `auto_grid()` nor equal to the engine's own default
# axis), or when the grid never collapsed onto the sigma axis in the first
# place -- so this is a zero-cost, byte-stable no-op for every fit that did not
# need it, and the reason is recorded in
# `res$outer_grid_recenter_declined`.
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
                                     auto = character(0), enabled = TRUE,
                                     max_attempts = 2L) {
    out <- list(res = res, prior = prior, prior_sigma = prior_sigma)
    type <- tolower(prior$type %||% "")
    if (!type %in% c("bym2", "icar", "car_proper")) return(out)
    if (!isTRUE(enabled)) {
        out$res <- .nl_decline_recenter(res, "auto_recenter_disabled")
        return(out)
    }
    if (.nl_axis_is_pinned(prior, "sigma_grid", .nl_auto_fields_at(auto))) {
        out$res <- .nl_decline_recenter(res, "axis_pinned")
        return(out)
    }

    cur_prior       <- prior
    cur_prior_sigma <- prior_sigma
    attempt <- 0L
    reason  <- "grid_not_collapsed"
    while (attempt < max_attempts &&
           identical(res$pareto_k_regime, "collapsed_edge") &&
           .nl_edge_axis_hit(res, "sigma")) {
        attempt <- attempt + 1L
        new_axis <- .nl_axis_recenter_from_fit(
            res$pareto_k_mode_u, res$pareto_k_cov_u,
            res$pareto_k_axis_tags, res$pareto_k_axis_names, "sigma")
        if (is.null(new_axis)) {
            reason <- "no_usable_curvature"
            break
        }
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
    out$res <- .nl_decline_recenter(out$res, reason)
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
# `p$sigma_grid %||% .nl_default_sigma_axis()` default -- see
# `.joint_block_axis_grid()`, `R/nested_laplace_joint_multi.R`), the
# donor field amplitude a copy coefficient scales -- the exact axis role
# gcol33/tulpa#289's driver (`occu_cover`) hits. A non-copy block's axis
# (RW1/RW2 tau, MCAR's log-Cholesky Sigma, ...) reuses
# `.NL_REGISTRY`/`.nl_block_axis_grid()` and is out of scope here (a
# materially larger, per-block-type surface than the single shared "donor
# sigma" convention every copy block shares).
#
# `res$blocks`/`res$axis_offsets` (attached whenever a fit carries >= 1
# block) name each block's columns `b<index>.<axis>` (1-based), which
# `.nl_axis_alias()` resolves along with the bare and coerced spellings; `cp` is
# the resolved copy spec (`.resolve_copy_multi()`) telling which block indices
# are copy blocks, and `auto` the per-block provenance record. Fixes at most one
# collapsed copy block per attempt (the documented case is a single copy block;
# a fit with several SIMULTANEOUSLY collapsed copy blocks partially improves
# within `max_attempts` rather than looping without bound).
.joint_multi_sigma_grid_rescue <- function(res, prior, copy, cp, prior_sigma,
                                           refit, auto = list(), enabled = TRUE,
                                           max_attempts = 2L) {
    out <- list(res = res, prior = prior, prior_sigma = prior_sigma)
    if (!.is_multi_block_prior(prior) || is.null(cp) || !isTRUE(cp$has_copy)) {
        return(out)
    }
    if (!isTRUE(enabled)) {
        out$res <- .nl_decline_recenter(res, "auto_recenter_disabled")
        return(out)
    }

    cur_prior       <- prior
    cur_prior_sigma <- prior_sigma
    attempt <- 0L
    reason  <- "grid_not_collapsed"
    while (attempt < max_attempts &&
           identical(res$pareto_k_regime, "collapsed_edge")) {
        target_b <- NULL
        for (b0 in cp$copy_blocks_zero) {
            b <- b0 + 1L
            if (!.nl_edge_axis_hit(res, "sigma", b)) next
            if (.nl_axis_is_pinned(cur_prior[[b]], "sigma_grid",
                                   .nl_auto_fields_at(auto, b))) {
                reason <- "axis_pinned"
                next
            }
            target_b <- b
            break
        }
        if (is.null(target_b)) break
        attempt <- attempt + 1L
        new_axis <- .nl_axis_recenter_from_fit(
            res$pareto_k_mode_u, res$pareto_k_cov_u,
            res$pareto_k_axis_tags, res$pareto_k_axis_names, "sigma",
            block_index = target_b, n_blocks = .nl_fit_n_blocks(res))
        if (is.null(new_axis)) {
            reason <- "no_usable_curvature"
            break
        }
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
    out$res <- .nl_decline_recenter(out$res, reason)
    out
}

# Per-family value/other axis field names for the standalone registry
# rescue below. `value` is the prior-list field the positive-scale axis
# lives on (recentred on collapse); `other` (when present) is the paired
# bounded axis's field, re-crossed with the new value axis via the same
# `expand.grid()` a family's own `defaults()` uses; `other_axis` is that
# axis's column name in `res$theta_grid`. `col_name` is the axis's bare name
# in `res$theta_names` -- `.nl_axis_alias()` resolves it against whatever the
# fit calls it (icar's `theta_grid` is a plain numeric vector, so
# `.joint_pareto_grid_regime()` coerces it to a 1-column matrix generically
# named `"theta"` while `theta_names` still says `"tau"`).
.NL_REGISTRY_AXIS_FIELD <- list(
    icar = list(value = "tau_grid",   col_name = "tau"),
    bym2 = list(value = "sigma_grid", col_name = "sigma",
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
# full fit). `auto` is the front door's provenance record.
.nl_registry_grid_rescue <- function(res, type, prior, refit, refit_log_marginal,
                                     auto = character(0), enabled = TRUE,
                                     max_attempts = 1L) {
    out <- list(res = res, prior = prior)
    fields <- .NL_REGISTRY_AXIS_FIELD[[type]]
    if (is.null(fields)) return(out)
    if (!isTRUE(enabled)) {
        out$res <- .nl_decline_recenter(res, "auto_recenter_disabled")
        return(out)
    }
    if (.nl_axis_is_pinned(prior, fields$value, .nl_auto_fields_at(auto))) {
        out$res <- .nl_decline_recenter(res, "axis_pinned")
        return(out)
    }

    cur_prior <- prior
    attempt <- 0L
    reason  <- "grid_not_collapsed"
    while (attempt < max_attempts &&
           identical(res$pareto_k_regime, "collapsed_edge") &&
           .nl_edge_axis_hit(res, fields$col_name)) {

        mc <- .nl_registry_axis_mode_cov(
            res, type, function(theta_mat) refit_log_marginal(cur_prior, theta_mat))
        if (is.null(mc)) {
            reason <- "no_usable_curvature"
            break
        }

        j <- .nl_axis_index(mc$col_names, .nl_axis_alias(fields$col_name), mc$tags)
        if (is.na(j)) {
            reason <- "no_usable_curvature"
            break
        }
        new_axis <- .nl_recenter_log_axis(mc$u_mode[j], sqrt(mc$cov[j, j]))
        if (is.null(new_axis)) {
            reason <- "no_usable_curvature"
            break
        }

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
    out$res <- .nl_decline_recenter(out$res, reason)
    out
}
