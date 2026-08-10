# Shared mode-Hessian outer-axis recentering (gcol33/tulpa#289, #290, #293).
#
# Every nested-Laplace family builds its outer hyperparameter grid from a
# FIXED default axis in original coordinates (`.NL_GRID` / `.NL_FAMILY_AXES`,
# R/settings.R -- e.g. the areal families' `field_sd`). A dataset whose
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
# (`.NL_RECENTER$sigma_pc_prior`, R/settings.R) for genuinely unidentified
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

# --- placement policy --------------------------------------------------------
#
# `control$auto_recenter` selects the outer-grid PLACEMENT policy. Four values
# on one knob rather than a second knob beside it, so a fit's placement has one
# spelling:
#
#   TRUE (default)  "resolve" -- the movable default axes are recentred on the
#                   posterior mode when the grid either RAILS (some axis's own
#                   marginal is maximal at one of its own endpoints,
#                   `.nl_axis_rail()`) or does not RESOLVE its own posterior
#                   (some axis's `h / sd` exceeds `.NL_RECENTER$resolve_mult`,
#                   `.nl_axis_h_over_sd()`). Both tests read the weights the fit
#                   already stored, so a grid that brackets and resolves its
#                   mode costs nothing beyond them.
#   "rail"          the rail test alone, which is what TRUE meant before
#                   gcol33/tulpa#361 settled the sizing half.
#   FALSE           the grid is integrated exactly as given, whatever it is.
#   "always"        every movable default axis is recentred whatever the fit
#                   did.
#
# A recentred axis is `mode +/- span * sd` over `n_pts` nodes, so it is
# `h / sd = 1.25` BY CONSTRUCTION, against a census median of 3.9 on the fixed
# spans (gcol33/tulpa#357, #361).
#
# The default is "resolve" and not "always" for COST, and the two are closer
# than the coverage table alone reads. They agree seed for seed on five of the
# six measured configurations; they differ on the one whose default axes
# already resolve their own posterior (NNGP, median `h / sd` 1.81 and 1.50),
# where "resolve" fires on 39.5% of seeds against 97.5% and covers 0.530 /
# 0.500 at the 50% level against 0.135 / 0.385. Coverage is what arbitrates a
# placement rule (gcol33/tulpa#331), so that row favours "resolve" -- but the
# reference read on a dense pinned axis that CONTAINS the posterior says
# "always" is the nearest read of it there (log-scale distance 0.208 / 0.243
# against 0.394 / 0.452), and that the fixture's posterior itself sits 0.62 /
# 0.73 above its own truth. A narrower span pulls the reported median back
# toward the truth by cancellation, so on that row the arm approximating the
# posterior worst covers best. What separates the two policies cleanly is
# 1.71 against 2.04 times the wall clock. The threshold's own measurement is on
# `.NL_RECENTER$resolve_mult` (`R/settings.R`).
#
# What "resolve" buys against "rail", over 200 fixed-truth seeds on each of six
# configurations (icar chain / icar lattice / rw1 / bym2 / iid / nngp, eight
# (configuration, axis) rows): mean |coverage - nominal| goes 0.043 -> 0.030 at
# the 95% level, 0.171 -> 0.084 at 80% and 0.243 -> 0.129 at 50%, at 0.63 times
# the 95% width and 0.76 times the median bias, for 1.71 times the wall clock
# (a second full grid solve plus the FD mode/Hessian stencil on the fits that
# fire). "rail" is over-covering across the board -- it holds 1.000 at nominal
# 0.95 on four of the eight rows, and 0.75 to 0.98 at nominal 0.50.
.nl_recenter_mode <- function(x) {
    if (is.null(x) || isTRUE(x)) return("resolve")
    if (isFALSE(x)) return("off")
    if (is.character(x) && length(x) == 1L &&
        x %in% c("rail", "resolve", "always")) {
        return(x)
    }
    stop("control$auto_recenter must be TRUE, FALSE, \"rail\", \"resolve\" or ",
         "\"always\"; got ",
         paste(utils::capture.output(utils::str(x)), collapse = " "),
         call. = FALSE)
}

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

#' Mark an outer-grid setting as a default rather than a pin
#'
#' @description
#' Declares that a setting shaping the outer hyperparameter grid carries a
#' *default* the caller computed, not a choice the user made. The auto-recenter
#' pass (\code{outer_grid_placement}, gcol33/tulpa#289) leaves a user-pinned
#' setting exactly as given, and re-centres (or, for a prior, engages its own
#' regularizer over) a marked one when the fit rails against its ceiling.
#'
#' Three kinds of setting take the mark:
#' \itemize{
#'   \item a grid axis on a nested-Laplace `prior` block (`sigma_grid`,
#'     `tau_grid`, ...) -- a numeric vector of nodes, or the `[n_cells x k]`
#'     matrix of pre-paired coordinates the families whose axis is a matrix
#'     take (`mcar` / `miid`'s `logchol_grid`, `tgmrf`'s `theta_grid_built`);
#'   \item a scalar grid-construction knob in `control`, for a driver that
#'     builds its axes rather than taking them (`fit_st_nested()`'s
#'     `n_grid_spatial`, `tau_upper`, ..., gcol33/tulpa#294);
#'   \item a `prior_sigma` hyperprior specification -- a list, e.g.
#'     `list("pc.prec", c(U = 3, alpha = 0.01))` (gcol33/tulpa#297).
#' }
#'
#' Wrapper packages are the intended caller: one that builds a default of its
#' own -- because it derives a second axis from it, hands the same vector to
#' several blocks, or exposes its own argument with a default -- would
#' otherwise be indistinguishable from a user who pinned that setting
#' deliberately. Mark it and the rescue stays live. A setting whose value is
#' exactly the engine's own default is recognised without a mark; anything else
#' needs one.
#'
#' The mark is an attribute, so it is dropped by `sort()`, `[`, `c()` and
#' `as.numeric()`: build the value first, mark it last.
#'
#' @param x Numeric vector or matrix of grid nodes, a numeric scalar knob, or a
#'   prior-specification list.
#' @return `x` carrying the marker attribute. Numeric input is coerced to
#'   double IN PLACE, so everything else it carries -- `dim()` and `dimnames()`
#'   above all -- survives the mark; a list is returned unchanged apart from
#'   the attribute.
#' @seealso [is_auto_grid()], [tulpa_nested_laplace_joint()], [fit_st_nested()]
#' @examples
#' prior <- list(type = "icar", sigma_grid = auto_grid(c(0.1, 0.5, 1, 2, 3)))
#' is_auto_grid(prior$sigma_grid)
#' is_auto_grid(auto_grid(list("pc.prec", c(U = 3, alpha = 0.01))))
#' @export
auto_grid <- function(x) {
    if (is.list(x)) {
        if (!length(x)) {
            stop("`auto_grid()` takes a non-empty prior specification.",
                 call. = FALSE)
        }
    } else {
        # In place, not `as.numeric()`: two families store their axis as a
        # matrix of pre-paired coordinates (`mcar` / `miid`'s `logchol_grid`,
        # `tgmrf`'s `theta_grid_built`), and flattening one destroys the axis
        # the caller is declaring (gcol33/tulpa#360).
        storage.mode(x) <- "double"
        if (!length(x) || anyNA(x)) {
            stop("`auto_grid()` takes a non-empty numeric grid with no NA.",
                 call. = FALSE)
        }
    }
    attr(x, "tulpa_auto_grid") <- TRUE
    x
}

#' Is an outer-grid setting marked as a default?
#'
#' @param x Any object.
#' @return `TRUE` when `x` carries the [auto_grid()] marker.
#' @seealso [auto_grid()]
#' @examples
#' is_auto_grid(auto_grid(c(0.5, 1, 2)))
#' is_auto_grid(c(0.5, 1, 2))
#' @export
is_auto_grid <- function(x) isTRUE(attr(x, "tulpa_auto_grid", exact = TRUE))

# Is a supplied `prior_sigma` a PIN? The prior-spec counterpart of
# `.nl_axis_is_pinned()` (gcol33/tulpa#297). The second recenter attempt exists
# to engage the weakly-informative PC prior on a mode with no finite curvature
# to settle on, and it must not be suppressed by a wrapper package that stamps a
# `prior_sigma` of its own -- the same presence-is-not-provenance mistake #293
# fixed one field over. Absent, marked with `auto_grid()`, or equal by value to
# the engine's own `.NL_RECENTER$sigma_pc_prior` are all defaults; anything else
# is a deliberate choice the rescue leaves alone.
.nl_prior_sigma_is_pinned <- function(prior_sigma) {
    if (is.null(prior_sigma)) return(FALSE)
    if (is_auto_grid(prior_sigma)) return(FALSE)
    d <- .nl_recenter("sigma_pc_prior")
    p <- prior_sigma
    attr(p, "tulpa_auto_grid") <- NULL
    !isTRUE(all.equal(d, p, check.attributes = FALSE))
}

# Drop the marker so nothing downstream of the rescue sees an attributed
# object (a prior spec is passed on to `.joint_parse_sigma_prior()`).
.nl_strip_auto <- function(x) {
    attr(x, "tulpa_auto_grid") <- NULL
    x
}

# Is `value` a grid the ENGINE would have laid on `field` itself? Such a grid
# carries no information a pin would add, so it counts as a default.
#
# Compared as a node SET (sorted, de-duplicated), because a family stores its
# axes pre-paired: bym2's default `sigma_grid` is the 5-node field-SD axis
# repeated across the 4 rho nodes, and that 20-long vector must still be
# recognised as the default axis it was expanded from.
#
# Candidates come from `.NL_FAMILY_AXES` (`R/settings.R`), so every family the
# engine defaults an axis for is covered by construction -- the hand-maintained
# two-field list this replaced could only see `sigma_grid` and `tau_grid`.
# `type` narrows the comparison to the axis THAT family defaults, which is the
# precise question; without it (a call site that does not know the block type)
# every axis any family binds to the field is a candidate. Data-dependent axes
# (`car_rho`, `spde_*`, `tgmrf_axis`) cannot be materialised without the data
# that shapes them, so they never match -- the safe direction, since a
# non-matching axis is treated as a pin and left alone.
.nl_axis_matches_default <- function(value, field, type = NULL) {
    keys <- if (!is.null(type)) .nl_family_axis_key(type, field) else
        .nl_field_axis_keys(field)
    if (!length(keys)) return(FALSE)
    u <- sort(unique(as.numeric(value)))
    if (!length(u)) return(FALSE)
    for (k in keys) {
        if (isTRUE(.NL_GRID[[k]]$data_dependent)) next
        du <- sort(unique(.nl_grid_axis(k)))
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
#
# `type` narrows the default comparison to the axis that ONE path-and-family
# lays on the field, and must be passed EXPLICITLY -- it is deliberately not
# inferred from `block$type`, because the block's family is not the same thing
# as the path that defaulted the axis: a joint areal fit carries
# `type = "icar"` on a block whose `sigma_grid` default comes from
# `.joint_areal`, while the icar REGISTRY entry defaults a precision axis and no
# `sigma_grid` at all. Inferring would silently answer "pinned" there, which is
# gcol33/tulpa#293 again one layer down. Unnarrowed (`NULL`) compares against
# every family's binding for the field, which errs toward recognising a default.
.nl_axis_is_pinned <- function(block, field, auto_fields = character(0),
                               type = NULL) {
    g <- if (is.list(block)) block[[field]] else NULL
    if (is.null(g)) return(FALSE)
    if (field %in% auto_fields) return(FALSE)
    if (is_auto_grid(g)) return(FALSE)
    !.nl_axis_matches_default(g, field, type)
}

# --- axis consumption (gcol33/tulpa#352) --------------------------------------
#
# A grid field the resolved path does not read must not pass in silence. Which
# fields a path reads is `.NL_PATH_AXES` (`R/settings.R`), so this is one check
# over the whole binding table rather than a rule per family: any family whose
# drivers parameterize it differently -- icar (precision on the registry path,
# field SD on the joint areal backends), car_proper, every copy block -- is
# covered by its table entry.
#
# The verdict splits on PROVENANCE, the same question every rescue above asks.
# A PINNED axis (named by the caller, neither marked with `auto_grid()` nor
# equal to the engine's own default nodes) is a choice the path cannot honour,
# so it is REFUSED, naming the field, the block, the path, the axis that path
# integrates, and -- where the engine itself establishes the conversion
# (`.NL_AXIS_EQUIV`) -- how to write the same grid in the axis that path reads.
# An axis that IS an engine default carries nothing a pin would add, so refusing
# it would be a false alarm; it is dropped, and the drop is RECORDED on the fit
# (`$axis_fields_dropped`) rather than left invisible, per gcol33/tulpa#293.

.NL_AXIS_PATH_LABEL <- c(
    registry     = paste0("the nested-Laplace registry path (`tulpa_nested_laplace()`, ",
                          "and every non-copy block of `tulpa_nested_laplace_joint()`)"),
    joint_single = "the single-block joint areal backend",
    copy         = "a copy block on the joint multi-block path"
)

# The conversion sentence for `field` into whichever consumed axis the engine
# converts it to, or NULL when no such conversion is established.
.nl_axis_equiv_hint <- function(type, field, consumed) {
    eq <- .NL_AXIS_EQUIV[[tolower(type %||% "")]][[field]]
    if (is.null(eq)) return(NULL)
    hit <- intersect(names(eq), consumed)
    if (!length(hit)) return(NULL)
    unname(eq[[hit[1L]]])
}

.nl_axis_refusal <- function(type, path, block_index, field, consumed) {
    lab <- .NL_AXIS_PATH_LABEL[[path]] %||% path
    hint <- .nl_axis_equiv_hint(type, field, consumed)
    paste0(
        "prior block ", if (is.null(block_index)) "" else paste0(block_index, " "),
        "'", type, "': `", field, "` is not an axis ", lab, " reads. ",
        "That path integrates ",
        paste0("`", consumed, "`", collapse = ", "), ". ",
        if (!is.null(hint)) paste0("Write the same grid as ", hint, ", ") else
            "Pin the axis that path reads, ",
        "or drop the field."
    )
}

# One block. Returns a list of drop records (possibly empty); refuses a pinned
# unread axis with an error.
.nl_check_block_axis_fields <- function(blk, path, block_index = NULL,
                                        auto_fields = character(0)) {
    if (!is.list(blk)) return(list())
    type <- tolower(blk$type %||% "")
    consumed <- .nl_path_axis_fields(type, path)
    if (!length(consumed)) return(list())
    present <- intersect(names(blk) %||% character(0), .nl_known_axis_fields())
    present <- present[vapply(present, function(f)
        length(blk[[f]]) > 0L && is.numeric(blk[[f]]), logical(1))]
    unread <- setdiff(present, consumed)
    if (!length(unread)) return(list())
    out <- list()
    for (f in unread) {
        # `type = NULL`: the field is not this path's, so the question is
        # whether the value is an engine default under ANY binding for it. That
        # errs toward recognising a default, which is the direction that errs
        # away from refusing a fit.
        if (.nl_axis_is_pinned(blk, f, auto_fields, type = NULL)) {
            stop(.nl_axis_refusal(type, path, block_index, f, consumed),
                 call. = FALSE)
        }
        out[[length(out) + 1L]] <- data.frame(
            block      = if (is.null(block_index)) NA_integer_ else
                             as.integer(block_index),
            type       = type,
            field      = f,
            path       = path,
            integrates = paste(consumed, collapse = ", "),
            reason     = "default_axis_not_read_by_this_path",
            stringsAsFactors = FALSE
        )
    }
    out
}

# The fields `.resolve_one_copy_spec()` (`R/nested_laplace_joint_multi.R`)
# reads off ONE copy spec. A copy spec is a three-field object rather than a
# payload carrier like a prior block -- it names an arm, a block, and the copy
# coefficient's axis -- so anything numeric beyond these is a grid the driver
# cannot act on, and gets the same provenance-split verdict the block check
# above gives an unread axis (gcol33/tulpaObs#192: a `sigma_pos_grid` from the
# retired (sigma_occ, sigma_pos) parameterization reached this spec and was
# neither read nor reported, so a pinned amplitude axis fell back to the
# engine's own default with a bit-identical `log_marginal`).
.NL_COPY_SPEC_FIELDS <- c("arm", "block", "alpha_grid")

.nl_copy_spec_refusal <- function(spec_index, field, type) {
    paste0(
        "copy spec ", if (is.null(spec_index)) "" else paste0(spec_index, " "),
        "(block ", type, "): `", field, "` is not a field the copy resolver ",
        "reads. A copy spec is resolved from ",
        paste0("`", .NL_COPY_SPEC_FIELDS, "`", collapse = ", "), " only, and ",
        "the copy arm's field amplitude is `alpha * sigma` -- `alpha` from the ",
        "spec's `alpha_grid`, `sigma` from the donor block's own `sigma_grid`. ",
        "Write the grid on one of those, or drop the field."
    )
}

# One copy spec. Returns a list of drop records (possibly empty); refuses a
# pinned unread field with an error.
.nl_check_one_copy_spec <- function(spec, type, spec_index = NULL) {
    if (!is.list(spec)) return(list())
    nm <- setdiff(names(spec) %||% character(0), .NL_COPY_SPEC_FIELDS)
    nm <- nm[nzchar(nm)]
    nm <- nm[vapply(nm, function(f)
        length(spec[[f]]) > 0L && is.numeric(spec[[f]]), logical(1))]
    out <- list()
    for (f in nm) {
        # `.copy` is the path pseudo-type whose binding names the axes a copy
        # block defaults, so a field carrying one of those axes' own default
        # nodes still counts as a default here.
        if (.nl_axis_is_pinned(spec, f, character(0), type = ".copy")) {
            stop(.nl_copy_spec_refusal(spec_index, f, type), call. = FALSE)
        }
        out[[length(out) + 1L]] <- data.frame(
            block      = if (is.null(spec$block)) NA_integer_ else
                             as.integer(spec$block),
            type       = type,
            field      = f,
            path       = "copy",
            integrates = "alpha_grid",
            reason     = "field_not_read_by_the_copy_spec_resolver",
            stringsAsFactors = FALSE
        )
    }
    out
}

.nl_check_copy_specs <- function(copy, prior) {
    if (is.null(copy) || !is.list(copy)) return(list())
    specs <- if (.is_copy_spec_list(copy)) copy else list(copy)
    multi <- length(specs) > 1L || .is_copy_spec_list(copy)
    rec <- list()
    for (i in seq_along(specs)) {
        s <- specs[[i]]
        b <- suppressWarnings(as.integer(s$block %||% NA_integer_))
        type <- if (!is.na(b) && b >= 1L && b <= length(prior))
            tolower(prior[[b]]$type %||% "") else ""
        rec <- c(rec, .nl_check_one_copy_spec(s, type,
                                              if (multi) i else NULL))
    }
    rec
}

# THE front-door check. `path` is `"registry"` (`tulpa_nested_laplace()`) or
# `"joint"` (`tulpa_nested_laplace_joint()`, which resolves per block: the
# single-block areal backend, a copy block, or the registry path). `auto` is the
# provenance record `.nl_grid_provenance()` just took. Returns the drop record
# (a data.frame) or NULL, and errors on a pinned unread axis.
.nl_check_axis_fields <- function(prior, path = "registry", auto = NULL,
                                  copy = NULL, responses = NULL) {
    if (!is.list(prior) || !length(prior)) return(NULL)
    multi  <- .is_multi_block_prior(prior)
    blocks <- if (multi) prior else list(prior)
    copy_at <- integer(0)
    if (identical(path, "joint") && multi && !is.null(copy)) {
        cp <- tryCatch(.resolve_copy_multi(copy, responses, prior),
                       error = function(e) NULL)
        # An unresolvable copy spec leaves every block's path unknown, and a
        # guess there refuses the wrong block with the wrong reason. Decline,
        # and let the driver raise the error the spec actually has.
        if (is.null(cp)) return(NULL)
        if (isTRUE(cp$has_copy)) {
            copy_at <- as.integer(cp$copy_blocks_zero) + 1L
        }
    }
    rec <- if (identical(path, "joint") && multi)
        .nl_check_copy_specs(copy, blocks) else list()
    for (b in seq_along(blocks)) {
        blk <- blocks[[b]]
        if (!is.list(blk) || is.null(blk$type)) next
        bpath <- if (!identical(path, "joint")) "registry"
                 else if (!multi) "joint_single"
                 else if (b %in% copy_at) "copy"
                 else "registry"
        bi <- if (multi) b else NULL
        rec <- c(rec, .nl_check_block_axis_fields(
            blk, bpath, bi, .nl_auto_fields_at(auto, bi)))
    }
    if (!length(rec)) return(NULL)
    do.call(rbind, rec)
}

# Publish the drop record for the duration of one fit, the way every front door
# publishes a fit-scoped setting (`tulpa.nl_max_grid_cells`, `tulpa.nl_progress`).
# `.finalize_fit()` reads it onto `$axis_fields_dropped`, so every fit the front
# door produces -- the first solve and any rescue refit -- carries it, and
# nothing outside the scope sees it. Call as
# `on.exit(options(.nl_publish_axis_dropped(rec)), add = TRUE)`-style: it
# returns the previous option list, exactly like `options()`.
.nl_publish_axis_dropped <- function(rec) {
    options(tulpa.nl_axis_dropped = rec)
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

# --- axis rails (gcol33/tulpa#361) --------------------------------------------
#
# `.nl_edge_axis_hit()` above asks the #276 question: did the WHOLE grid
# collapse onto one cell, and does that cell sit on a node. That is a joint
# quantity over the tensor -- `ess_grid = 1 / sum(w^2)` across every cell -- so
# on a crossed grid a second axis carrying spread lifts it past the collapse
# threshold while one axis is hard against its own boundary. The gcol33/tulpa#357
# census's 100-region BYM2 fixture measures `ess_grid = 1.928` against a
# threshold of 2: whether its railed `rho` axis is seen at all rests on 0.072 of
# an effective cell, and on a number the `rho` marginal barely enters.
#
# The per-axis question is exact and costs nothing beyond the weights already
# stored. Marginalize the fit's own `log_marginal` onto one axis -- the SAME
# marginal `.nl_axis_quantiles()` reports that axis's median and interval off --
# and ask whether the weight is maximal at an endpoint. For a unimodal marginal
# that happens exactly when the mode is AT or BEYOND that endpoint, so the span
# does not contain it and the axis integrates a tail at any spacing.

# One axis's marginal over its own sorted distinct nodes, normalized.
.nl_axis_marginal_w <- function(res, axis) {
    tg <- res$theta_grid
    lm <- res$log_marginal
    if (is.null(tg) || is.null(lm)) return(NULL)
    cn <- if (is.matrix(tg)) colnames(tg) else (res$theta_names %||% "theta")
    if (!is.matrix(tg)) tg <- matrix(as.numeric(tg), ncol = 1L)
    if (is.null(cn) || length(cn) != ncol(tg)) {
        cn <- paste0("axis", seq_len(ncol(tg)))
    }
    j <- match(axis, cn)
    if (is.na(j) && ncol(tg) == 1L) j <- 1L
    if (is.na(j) || length(lm) != nrow(tg)) return(NULL)
    m <- .nl_axis_marginal_logdensity(as.numeric(tg[, j]), lm)
    if (length(m$vals) < 2L) return(NULL)
    top <- max(m$log_marg)
    if (!is.finite(top)) return(NULL)
    w <- exp(m$log_marg - top)
    s <- sum(w)
    if (!is.finite(s) || s <= 0) return(NULL)
    list(vals = m$vals, w = w / s, col = j)
}

# Is `axis` railed against one of its own endpoints? Returns
# `list(side, mass, lift, node)` or NULL.
#
# Two clauses. The first IS the statement -- a marginal maximal at a boundary
# node has its mode at or beyond that boundary. The second is the materiality
# guard: what keeps a marginal that is merely uneven, or flat to within the
# weights' own noise (where the argmax is arbitrary), from being read as a rail
# and moved onto curvature it does not have -- a near-flat direction returns a
# huge mode SD, and a grid laid over it is coarser than the one it replaced.
#
# The guard reads the boundary node's weight against what a FLAT marginal would
# put there, `1 / m`, rather than against a fixed share (gcol33/tulpa#375). The
# two differ by exactly the node count, and that is the whole defect: an axis's
# weights are a distribution over its OWN nodes, so the same posterior read at
# more nodes carries less on any one of them and a fixed share turns a longer
# axis into a weaker detector. `lift = m * w[k]` is 1 for a flat marginal at any
# node count and any spacing, so the threshold means the same thing wherever it
# is applied. Measured on a BYM2 `rho` posterior held past a fixed span, the
# lift of a railed fit is flat in the node count (2.70 / 2.81 / 2.87 / 2.94 /
# 3.02 at m = 4 / 5 / 6 / 8 / 12) where its share collapses (0.68 / 0.56 / 0.48
# / 0.37 / 0.25).
.nl_axis_rail <- function(res, axis, edge_mult = .nl_recenter("edge_mass_mult")) {
    mw <- .nl_axis_marginal_w(res, axis)
    if (is.null(mw)) return(NULL)
    m <- length(mw$w)
    k <- which.max(mw$w)
    if (k != 1L && k != m) return(NULL)
    lift <- m * mw$w[k]
    if (!is.finite(lift) || lift < edge_mult) return(NULL)
    list(side = if (k == 1L) "lower" else "upper",
         mass = mw$w[k], lift = lift, node = mw$vals[k])
}

# Does the axis's own grid RESOLVE its own marginal? `h / sd` is the median node
# spacing in the axis's unconstraining coordinate (`tag`, the same coordinate a
# recentred axis is laid in) over that marginal's SD in the same coordinate.
# Reads the stored weights, so it costs nothing beyond the fit.
#
# A marginal whose weight sits entirely on one node has SD 0 and returns `Inf`
# -- the grid resolved nothing at all, which is the coarsest case there is, not
# a missing measurement. `NA` is reserved for an axis the read cannot be taken
# on (fewer than two nodes, a node outside its own support), and the caller
# treats that as unresolved too rather than silently reporting a grid it could
# not score.
.nl_axis_h_over_sd <- function(res, axis, tag) {
    if (length(tag) != 1L || is.na(tag)) return(NA_real_)
    mw <- .nl_axis_marginal_w(res, axis)
    if (is.null(mw) || length(mw$vals) < 2L) return(NA_real_)
    u <- as.numeric(.joint_pareto_fwd(tag, mw$vals))
    if (any(!is.finite(u))) return(NA_real_)
    mu <- sum(mw$w * u)
    sd <- sqrt(max(0, sum(mw$w * u^2) - mu^2))
    if (!is.finite(sd)) return(NA_real_)
    if (sd <= 0) return(Inf)
    stats::median(diff(sort(u))) / sd
}

# Every axis of a fit that is railed, as `axis:side`, whether or not any rescue
# covers it. Recorded on the fit (`$outer_grid_railed_axes`) so a span that does
# not contain its own posterior mode is visible instead of silently integrating
# a tail -- the gcol33/tulpa#293 rule that a placement the engine leaves alone
# has to say so.
.nl_railed_axes <- function(res) {
    tg <- res$theta_grid
    if (is.null(tg) || is.null(res$log_marginal)) return(character(0))
    cn <- if (is.matrix(tg)) colnames(tg) else (res$theta_names %||% "theta")
    if (is.null(cn)) return(character(0))
    hits <- character(0)
    for (a in cn) {
        r <- .nl_axis_rail(res, a)
        if (!is.null(r)) hits <- c(hits, paste0(a, ":", r$side))
    }
    hits
}

.nl_attach_railed_axes <- function(res) {
    res$outer_grid_railed_axes <- .nl_railed_axes(res)
    res
}

# Why the registry rescue covers no axis of `type` (gcol33/tulpa#370). Two
# distinguishable answers, and returning unstamped conflated them with a fit the
# rescue never applied to:
#
#   * `"unguessable_axis: <names>"` -- the family HAS a positive-scale axis the
#     rescue could place, but some other axis of the same grid has a support the
#     transform registry will not guess (car_proper's `rho_car` on the adjacency
#     eigenvalue interval), and `.nl_registry_axis_mode_cov()` declines for the
#     WHOLE fit rather than per axis. Naming the blocking axis is #295's
#     convention and is what tells a caller that pinning it themselves unblocks
#     the rest.
#   * `"family_out_of_scope"` -- nothing about this fit's axis geometry is in
#     the rescue's scope at all (mcar / miid's log-Cholesky coordinates, a
#     `tgmrf` block's user-declared axes, a `lf` block carrying no axis).
#
# `tags` is the per-column transform registry read for this fit
# (`.nl_registry_axis_tags()`), so the two answers are decided from the same
# object the rescue itself gates on rather than from a second read.
.nl_out_of_scope_reason <- function(res, tags) {
    cn <- names(tags) %||% res$theta_names %||%
        colnames(res$theta_grid) %||% character(0)
    if (!is.null(tags) && length(cn) == length(tags) && anyNA(tags)) {
        return(paste0("unguessable_axis: ",
                      paste(cn[is.na(tags)], collapse = ", ")))
    }
    "family_out_of_scope"
}

# --- decline reasons ---------------------------------------------------------
#
# `res$outer_grid_recenter_declined` records why an applicable auto-recenter did
# not run: `"axis_pinned"` (the caller pinned the axis), `"grid_not_collapsed"`
# (the grid already brackets the mode, the common no-op, on the rescues whose
# trigger is the whole grid's collapse), `"no_axis_railed"` (its per-axis
# counterpart on the registry rescue under `control$auto_recenter = "rail"`: no
# axis of the family is maximal at one of its own endpoints),
# `"grid_resolves_posterior"` (the same under the default `"resolve"` policy:
# no axis rails AND each is resolved to within `.NL_RECENTER$resolve_mult`
# posterior SDs per node, so re-placing would buy nothing),
# `"no_usable_curvature"` (the mode-Hessian the recenter needs was unavailable
# or degenerate), `"sd_ceiling_unresolved"` / `"sd_floor_unresolved"` (the
# stencil returned a curvature past one of the mode-SD bounds, so an axis laid
# from it would be laid from a substituted spread rather than a measured one --
# gcol33/tulpa#387; only under the declining clamp policies),
# `"auto_recenter_disabled"` (`control$auto_recenter = FALSE`,
# the way to hold ANY grid -- the engine's own default axis included -- exactly
# where it is), `"grid_knobs_overridden"` (the spatiotemporal driver's
# grid-construction knobs were set explicitly), `"refit_failed"` (the recentred
# grid did not solve), `"unguessable_axis: <names>"` / `"family_out_of_scope"`
# (the registry rescue covers no axis of this family -- gcol33/tulpa#370, which
# is what returning UNSTAMPED used to look like). Absent on a fit that WAS
# recentred, and never stamped by
# a rescue whose prior shape it does not apply to -- a fit carries the reason
# from the one rescue that could have run, not a tally of the others declining.
.nl_decline_recenter <- function(res, reason) {
    if (identical(res$outer_grid_placement, "auto_recentered")) return(res)
    res$outer_grid_recenter_declined <- reason
    res
}

# `mode_u` / `sd_u` are in the axis's own unconstrained coordinate, the one
# `.joint_pareto_fwd()` / `.joint_pareto_inv()` define per `tag`: `log` for a
# positive scale, `logit01` for a proportion, `identity` for an axis already on
# all of R. Returns a sorted numeric vector of `n_pts` nodes spanning
# `mode_u +/- span * sd_u`, mapped back to the axis's own support, or NULL when
# the curvature is not usable (non-finite / non-positive SD, an unguessable
# tag) -- the caller then leaves the existing grid untouched rather than centre
# on a meaningless spread.
# `sd_u` is clamped to `[min_sd_u, max_sd_u]`: a floor so a razor-sharp local
# curvature does not collapse the new grid to near-duplicate nodes (the
# purpose of the retry is to bracket the mode with actual spread), and a
# ceiling so a near-flat direction does not fling nodes to implausible
# extremes.
#
# A `logit01` axis maps back into the OPEN interval, and both endpoints are
# singular for the families that carry one (a BYM2 `rho` of exactly 0 or 1 is a
# degenerate mixture), so a node that saturates to a boundary in double
# precision is dropped rather than laid down; too few survivors declines.
#
# THE CLAMP IS NOT A MEASUREMENT (gcol33/tulpa#387). Whenever a bound binds, the
# axis is laid from a number the engine substituted for a curvature the stencil
# could not read, and the two cases are otherwise indistinguishable on the fit.
# `.nl_recenter_sd_clamp()` is the one place either bound is applied, so what
# the pass does about it is a policy (`.NL_RECENTER$sd_clamp_policy` /
# `$sd_floor_policy`, `R/settings.R`) rather than a constant buried in a node
# generator, and the state it returns is recorded on the fit.
#
# `ref_span_u` is the INCOMING axis's own span in the same coordinate, supplied
# by callers that have it; the `"relative"` ceiling caps the re-placed span by
# it, and falls back to the absolute ceiling where a caller has none.
.nl_recenter_sd_clamp <- function(sd_u,
                                  min_sd_u   = .nl_recenter("min_sd_u"),
                                  max_sd_u   = .nl_recenter("max_sd_u"),
                                  span       = .nl_recenter("span"),
                                  ref_span_u = NULL,
                                  ceiling    = .nl_recenter("sd_clamp_policy"),
                                  floor      = .nl_recenter("sd_floor_policy")) {
    if (length(sd_u) != 1L || !is.finite(sd_u) || sd_u <= 0) {
        return(list(sd = NA_real_, sd_raw = NA_real_, clamp = NA_character_,
                    reason = "no_usable_curvature"))
    }
    if (sd_u < min_sd_u) {
        return(list(sd = if (identical(floor, "decline")) NA_real_ else min_sd_u,
                    sd_raw = sd_u, clamp = "floor",
                    reason = if (identical(floor, "decline"))
                        "sd_floor_unresolved" else NULL))
    }
    if (sd_u <= max_sd_u) {
        return(list(sd = sd_u, sd_raw = sd_u, clamp = "none", reason = NULL))
    }
    cap <- max_sd_u
    if (identical(ceiling, "relative") && length(ref_span_u) == 1L &&
        is.finite(ref_span_u) && ref_span_u > 0) {
        cap <- min(max_sd_u, ref_span_u / (2 * span))
    }
    list(sd = if (identical(ceiling, "decline")) NA_real_ else max(cap, min_sd_u),
         sd_raw = sd_u, clamp = "ceiling",
         reason = if (identical(ceiling, "decline"))
             "sd_ceiling_unresolved" else NULL)
}

# The node generator, reporting what it was laid from. `nodes` is NULL exactly
# when the axis was not built, and `reason` then says why in the decline
# vocabulary `.nl_decline_recenter()` records -- so a caller stamps the clamp's
# own reason rather than folding it into `"no_usable_curvature"`, which is what
# made a substituted spread and a measured one read alike.
.nl_recenter_axis_full <- function(tag, mode_u, sd_u,
                                   n_pts      = .nl_recenter("n_pts"),
                                   span       = .nl_recenter("span"),
                                   min_sd_u   = .nl_recenter("min_sd_u"),
                                   max_sd_u   = .nl_recenter("max_sd_u"),
                                   ref_span_u = NULL) {
    bad <- function(reason, clamp = NA_character_, raw = NA_real_) {
        list(nodes = NULL, sd_clamp = clamp, sd_used = NA_real_, sd_raw = raw,
             reason = reason)
    }
    if (length(tag) != 1L || is.na(tag) ||
        !tag %in% c("log", "logit01", "identity")) {
        return(bad("no_usable_curvature"))
    }
    if (length(mode_u) != 1L || length(sd_u) != 1L) return(bad("no_usable_curvature"))
    if (!is.finite(mode_u)) return(bad("no_usable_curvature"))
    cl <- .nl_recenter_sd_clamp(sd_u, min_sd_u = min_sd_u, max_sd_u = max_sd_u,
                                span = span, ref_span_u = ref_span_u)
    if (!is.null(cl$reason)) return(bad(cl$reason, cl$clamp, cl$sd_raw))
    u_seq <- seq(mode_u - span * cl$sd, mode_u + span * cl$sd,
                 length.out = as.integer(n_pts))
    nodes <- .joint_pareto_inv(tag, u_seq)$theta
    keep  <- is.finite(nodes)
    if (identical(tag, "logit01")) keep <- keep & nodes > 0 & nodes < 1
    nodes <- sort(unique(nodes[keep]))
    if (length(nodes) < .nl_recenter("min_nodes")) {
        return(bad("no_usable_curvature", cl$clamp, cl$sd_raw))
    }
    list(nodes = nodes, sd_clamp = cl$clamp, sd_used = cl$sd,
         sd_raw = cl$sd_raw, reason = NULL)
}

.nl_recenter_axis <- function(tag, mode_u, sd_u,
                              n_pts    = .nl_recenter("n_pts"),
                              span     = .nl_recenter("span"),
                              min_sd_u = .nl_recenter("min_sd_u"),
                              max_sd_u = .nl_recenter("max_sd_u")) {
    .nl_recenter_axis_full(tag, mode_u, sd_u, n_pts = n_pts, span = span,
                           min_sd_u = min_sd_u, max_sd_u = max_sd_u)$nodes
}

# The positive-scale case, kept as its own name because every existing caller
# (both joint rescues) recentres a field amplitude.
.nl_recenter_log_axis <- function(mode_u, sd_u,
                                   n_pts    = .nl_recenter("n_pts"),
                                   span     = .nl_recenter("span"),
                                   min_sd_u = .nl_recenter("min_sd_u"),
                                   max_sd_u = .nl_recenter("max_sd_u")) {
    .nl_recenter_axis("log", mode_u, sd_u, n_pts = n_pts, span = span,
                      min_sd_u = min_sd_u, max_sd_u = max_sd_u)
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
# `ref_nodes` is the axis's incoming nodes in its OWN coordinates, when the
# caller has them: the `"relative"` ceiling reads its span from them.
.nl_axis_recenter_from_fit_full <- function(mode_u, cov_u, axis_tags, axis_names,
                                            axis, n_pts = .nl_recenter("n_pts"),
                                            span = .nl_recenter("span"),
                                            block_index = NULL, n_blocks = 0L,
                                            ref_nodes = NULL) {
    bad <- function(reason = "no_usable_curvature") {
        list(nodes = NULL, sd_clamp = NA_character_, sd_used = NA_real_,
             sd_raw = NA_real_, reason = reason)
    }
    if (is.null(mode_u) || is.null(cov_u) || is.null(axis_names)) return(bad())
    aliases <- .nl_axis_alias(axis, block_index, n_blocks)
    j <- .nl_axis_index(axis_names, aliases, axis_tags)
    if (is.na(j)) return(bad())
    if (is.null(axis_tags) || !identical(axis_tags[j], "log")) return(bad())
    if (!is.matrix(cov_u) || nrow(cov_u) < j || ncol(cov_u) < j) return(bad())
    sd_u <- suppressWarnings(sqrt(cov_u[j, j]))
    .nl_recenter_axis_full("log", mode_u[j], sd_u, n_pts = n_pts, span = span,
                           ref_span_u = .nl_axis_span_u(ref_nodes, "log"))
}

.nl_axis_recenter_from_fit <- function(mode_u, cov_u, axis_tags, axis_names,
                                       axis, n_pts = .nl_recenter("n_pts"),
                                       span = .nl_recenter("span"),
                                       block_index = NULL, n_blocks = 0L) {
    .nl_axis_recenter_from_fit_full(mode_u, cov_u, axis_tags, axis_names, axis,
                                    n_pts = n_pts, span = span,
                                    block_index = block_index,
                                    n_blocks = n_blocks)$nodes
}

# The span an existing set of nodes covers in the unconstraining coordinate,
# for the `"relative"` ceiling. NULL / unusable nodes give NA, which that
# policy reads as "no reference span" and falls back to the absolute ceiling.
.nl_axis_span_u <- function(nodes, tag) {
    if (is.null(nodes) || length(tag) != 1L || is.na(tag)) return(NA_real_)
    v <- suppressWarnings(as.numeric(nodes))
    v <- v[is.finite(v)]
    if (length(v) < 2L) return(NA_real_)
    u <- suppressWarnings(as.numeric(.joint_pareto_fwd(tag, v)))
    u <- u[is.finite(u)]
    if (length(u) < 2L) return(NA_real_)
    diff(range(u))
}


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
#      apply the light default PC(U=3, alpha=0.01) prior (only if the caller
#      PINNED no `prior_sigma` of their own -- `.nl_prior_sigma_is_pinned()`,
#      the same provenance question the axis asks, so a wrapper stamping a
#      default prior does not silently disable the escalation; the suppression
#      is recorded in `res$outer_grid_prior_declined`) and recentre once more.
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
                                     max_attempts = .nl_recenter("max_attempts_joint")) {
    out <- list(res = res, prior = prior, prior_sigma = prior_sigma)
    type <- tolower(prior$type %||% "")
    if (!type %in% c("bym2", "icar", "car_proper")) return(out)
    if (!isTRUE(enabled)) {
        out$res <- .nl_decline_recenter(res, "auto_recenter_disabled")
        return(out)
    }
    if (.nl_axis_is_pinned(prior, "sigma_grid", .nl_auto_fields_at(auto),
                           type = ".joint_areal")) {
        out$res <- .nl_decline_recenter(res, "axis_pinned")
        return(out)
    }

    prior_pinned    <- .nl_prior_sigma_is_pinned(prior_sigma)
    cur_prior       <- prior
    cur_prior_sigma <- .nl_strip_auto(prior_sigma)
    attempt <- 0L
    reason  <- "grid_not_collapsed"
    while (attempt < max_attempts &&
           identical(res$pareto_k_regime, "collapsed_edge") &&
           .nl_edge_axis_hit(res, "sigma")) {
        attempt <- attempt + 1L
        rc <- .nl_axis_recenter_from_fit_full(
            res$pareto_k_mode_u, res$pareto_k_cov_u,
            res$pareto_k_axis_tags, res$pareto_k_axis_names, "sigma",
            ref_nodes = .nl_axis_ref_nodes(res, "sigma"))
        if (is.null(rc$nodes)) {
            reason <- rc$reason
            break
        }
        cur_prior$sigma_grid <- rc$nodes
        if (attempt >= 2L && !prior_pinned) {
            cur_prior_sigma <- .nl_recenter("sigma_pc_prior")
        }
        res <- refit(cur_prior, cur_prior_sigma)
        res$outer_grid_placement           <- "auto_recentered"
        res$outer_grid_recenter_attempts   <- attempt
        res$outer_grid_recenter_sd_clamp   <- stats::setNames(rc$sd_clamp, "sigma")
        res$outer_grid_recenter_sd_used    <- stats::setNames(rc$sd_used, "sigma")
        res$outer_grid_recenter_sd_raw     <- stats::setNames(rc$sd_raw, "sigma")
        res$outer_grid_prior_added         <- attempt >= 2L && !prior_pinned
        res$outer_grid_prior_declined      <-
            if (attempt >= 2L && prior_pinned) "prior_pinned" else NULL
        out <- list(res = res, prior = cur_prior, prior_sigma = cur_prior_sigma)
    }
    out$res <- .nl_decline_recenter(out$res, reason)
    out
}

# Mode + FD-Hessian covariance of a REGISTRY fit's outer grid
# (gcol33/tulpa#290) -- the standalone `tulpa_nested_laplace()` counterpart
# of the joint path's `.joint_pareto_prepare()` delta-collapse rescue,
# reusing the SAME generic tagging (`.joint_pareto_block_tags()`, read through
# `.nl_registry_axis_tags()`) and FD-Hessian machinery
# (`.joint_pareto_mode_cov()`) rather than a fresh implementation. `tags` is
# one transform tag per grid column; `refit_log_marginal(theta_mat)`
# re-evaluates the inner marginal at an arbitrary `[S x d]` theta matrix
# (columns named per `res$theta_names`) through the SAME kernel the fit used.
# Declines (NULL) when any axis in the grid has unguessable support -- e.g.
# car_proper's `rho`, the identical limitation the joint path already has for
# that family.
.nl_registry_axis_mode_cov <- function(res, tags, refit_log_marginal) {
    cn <- res$theta_names
    tg <- res$theta_grid
    if (is.null(cn) || is.null(tg)) return(NULL)
    if (!is.matrix(tg)) tg <- matrix(as.numeric(tg), ncol = 1L)
    colnames(tg) <- cn
    w <- res$weights
    if (is.null(w) || length(w) != nrow(tg)) return(NULL)

    if (is.null(tags) || length(tags) != ncol(tg) || anyNA(tags)) return(NULL)
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
# `p$sigma_grid %||% .nl_grid_axis("field_sd")` default -- see
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
                                           max_attempts = .nl_recenter("max_attempts_joint")) {
    out <- list(res = res, prior = prior, prior_sigma = prior_sigma)
    if (!.is_multi_block_prior(prior) || is.null(cp) || !isTRUE(cp$has_copy)) {
        return(out)
    }
    if (!isTRUE(enabled)) {
        out$res <- .nl_decline_recenter(res, "auto_recenter_disabled")
        return(out)
    }

    prior_pinned    <- .nl_prior_sigma_is_pinned(prior_sigma)
    cur_prior       <- prior
    cur_prior_sigma <- .nl_strip_auto(prior_sigma)
    attempt <- 0L
    reason  <- "grid_not_collapsed"
    while (attempt < max_attempts &&
           identical(res$pareto_k_regime, "collapsed_edge")) {
        target_b <- NULL
        for (b0 in cp$copy_blocks_zero) {
            b <- b0 + 1L
            if (!.nl_edge_axis_hit(res, "sigma", b)) next
            if (.nl_axis_is_pinned(cur_prior[[b]], "sigma_grid",
                                   .nl_auto_fields_at(auto, b),
                                   type = ".copy")) {
                reason <- "axis_pinned"
                next
            }
            target_b <- b
            break
        }
        if (is.null(target_b)) break
        attempt <- attempt + 1L
        n_b <- .nl_fit_n_blocks(res)
        rc <- .nl_axis_recenter_from_fit_full(
            res$pareto_k_mode_u, res$pareto_k_cov_u,
            res$pareto_k_axis_tags, res$pareto_k_axis_names, "sigma",
            block_index = target_b, n_blocks = n_b,
            ref_nodes = .nl_axis_ref_nodes(res, "sigma", target_b, n_b))
        if (is.null(rc$nodes)) {
            reason <- rc$reason
            break
        }
        cur_prior[[target_b]]$sigma_grid <- rc$nodes
        if (attempt >= 2L && !prior_pinned) {
            cur_prior_sigma <- .nl_recenter("sigma_pc_prior")
        }
        res <- refit(cur_prior, cur_prior_sigma)
        res$outer_grid_placement           <- "auto_recentered"
        res$outer_grid_recenter_attempts   <- attempt
        res$outer_grid_recenter_sd_clamp   <- stats::setNames(
            rc$sd_clamp, .nl_axis_alias("sigma", target_b, n_b)[1L])
        res$outer_grid_recenter_sd_used    <- stats::setNames(
            rc$sd_used, .nl_axis_alias("sigma", target_b, n_b)[1L])
        res$outer_grid_recenter_sd_raw     <- stats::setNames(
            rc$sd_raw, .nl_axis_alias("sigma", target_b, n_b)[1L])
        res$outer_grid_prior_added         <- attempt >= 2L && !prior_pinned
        res$outer_grid_prior_declined      <-
            if (attempt >= 2L && prior_pinned) "prior_pinned" else NULL
        out <- list(res = res, prior = cur_prior, prior_sigma = cur_prior_sigma)
    }
    out$res <- .nl_decline_recenter(out$res, reason)
    out
}

# Which axes the registry rescue below can move, per family: the prior-list
# FIELD each axis lives on, mapped to that axis's bare name in
# `res$theta_names`. `.nl_axis_alias()` resolves the name against whatever the
# fit calls it (icar's `theta_grid` is a plain numeric vector, so
# `.joint_pareto_grid_regime()` coerces it to a 1-column matrix generically
# named `"theta"` while `theta_names` still says `"tau"`; a multi-block grid
# prefixes every axis `b<k>.`).
#
# Every axis a family lists is movable on its own. Before gcol33/tulpa#361 the
# entry named ONE recentrable axis and carried the family's other axis as a
# passenger, re-crossed unchanged, so BYM2's `rho_grid` was detected against its
# 0.95 ceiling (`pareto_k_grid_edge_axes` names it) and then left there -- with
# the fit recording `grid_not_collapsed`, which is not what happened. Field
# order follows `.NL_FAMILY_AXES` (`R/settings.R`), the same order
# `.nl_fill_family_axes()` crosses a family's defaults in.
#
# MEMBERSHIP is decided by one question, not by hand: does
# `.joint_pareto_block_tags()` name a coordinate for EVERY axis of the family's
# grid? A recentred axis is laid in that coordinate (`.nl_recenter_axis()`) and
# the FD stencil differences the whole grid in it (`.joint_pareto_mode_cov()`),
# so one unguessable axis takes the fit's curvature with it. The five registry
# families absent below are absent for a stated reason, and each records it on
# the fit rather than passing in silence (gcol33/tulpa#370):
#
#   * car_proper (`rho` on the adjacency eigenvalue interval), ar1 (`rho` an
#     autocorrelation on (-1, 1)) and hsgp_mo (`rho` a cross-output
#     correlation on (-1, 1)) each carry ONE axis the transform registry
#     declines, so their scale axes are blocked with it. The fit says
#     `unguessable_axis: rho`, which is also what tells a caller that pinning
#     that axis themselves unblocks the rest.
#   * mcar / miid hold their p(p+1)/2 log-Cholesky coordinates in a SINGLE
#     matrix field (`logchol_grid`), so the field-to-axis binding this table is
#     built on does not describe them at all, and the re-cross below would
#     write a `data.frame` where a matrix belongs. Their tags are all
#     `identity`, so the block is the field shape, not the coordinate.
#   * tgmrf declares its own axis names and its own bounds box
#     (`theta_grid_built`, again one matrix field), and the transform registry
#     names no coordinate for a user's axis.
#   * lf carries no outer axis at all (a 1 x 0 grid), so there is nothing to
#     place.
.NL_REGISTRY_AXIS_FIELD <- list(
    icar = c(tau_grid = "tau"),
    rw1  = c(tau_grid = "tau"),
    rw2  = c(tau_grid = "tau"),
    iid  = c(sigma_grid = "sigma"),
    bym2 = c(sigma_grid = "sigma", rho_grid = "rho"),
    nngp = c(sigma2_grid = "sigma2", phi_gp_grid = "phi_gp"),
    hsgp = c(sigma2_grid = "sigma2", lengthscale_grid = "lengthscale"),
    spde = c(range_grid = "range", sigma_grid = "sigma")
)

# The per-axis movable slots a prior offers: one entry per (block, field) the
# table above binds, in block then table order. A single-block prior is the
# length-1 case, so the rescue has one representation to walk whether it was
# handed a block or a list of them.
.nl_registry_axis_slots <- function(blocks) {
    slots <- list()
    for (b in seq_along(blocks)) {
        fl <- .NL_REGISTRY_AXIS_FIELD[[tolower(blocks[[b]]$type %||% "")]]
        if (is.null(fl)) next
        for (f in names(fl)) {
            slots[[length(slots) + 1L]] <- list(
                block = b, type = tolower(blocks[[b]]$type),
                field = f, axis = unname(fl[[f]]))
        }
    }
    slots
}

# One transform tag per grid column of a registry fit. Single-block: the lone
# family's own read. Multi-block: the per-block walk `.joint_axis_tags_raw()`
# does, driven off `res$axis_offsets` and the block types the CALLER holds --
# the prior list is the authority on which family owns which columns, and it is
# available here where `res$prior` is not yet attached.
.nl_registry_axis_tags <- function(res, blocks, multi) {
    cn <- res$theta_names %||% colnames(res$theta_grid)
    if (is.null(cn) || !length(cn)) return(NULL)
    tag1 <- function(type, axes) tryCatch(
        .joint_pareto_block_tags(tolower(type %||% ""), axes),
        error = function(e) rep(NA_character_, length(axes)))
    if (!isTRUE(multi)) {
        tags <- tag1(blocks[[1L]]$type, cn)
        names(tags) <- cn
        return(tags)
    }
    ao <- as.integer(res$axis_offsets %||% integer(0))
    if (length(ao) != length(blocks) + 1L || ao[length(ao)] != length(cn)) {
        return(NULL)
    }
    tags <- rep(NA_character_, length(cn))
    bare <- sub("^b[0-9]+\\.", "", cn)
    for (b in seq_along(blocks)) {
        if (ao[b + 1L] <= ao[b]) next
        cols <- (ao[b] + 1L):ao[b + 1L]
        tags[cols] <- tag1(blocks[[b]]$type, bare[cols])
    }
    names(tags) <- cn
    tags
}

# The grid column `axis` landed in, under whichever of its spellings the fit
# carries. NA when the fit has no such column.
.nl_axis_colname <- function(res, aliases) {
    cn <- if (is.matrix(res$theta_grid)) colnames(res$theta_grid) else
        (res$theta_names %||% "theta")
    if (is.null(cn) || !length(cn)) return(NA_character_)
    hit <- cn[cn %in% aliases]
    if (length(hit)) return(hit[1L])
    if (length(cn) == 1L) return(cn[1L])
    NA_character_
}

# Write an `[S x d]` theta matrix back onto the prior's own grid fields, so the
# FD stencil's re-evaluation runs the kernel at exactly those coordinates. One
# generic write over `.NL_REGISTRY_AXIS_FIELD` -- before gcol33/tulpa#361 the
# call site branched on `type == "icar"` / `"bym2"` by hand and silently
# ignored `theta_mat` for every other family, which returned the fit's OWN
# log-marginal at the wrong length and made the curvature unusable.
.nl_registry_write_theta <- function(blocks, theta_mat, cn, multi = FALSE) {
    n_blocks <- if (isTRUE(multi)) length(blocks) else 0L
    for (s in .nl_registry_axis_slots(blocks)) {
        aliases <- .nl_axis_alias(s$axis, if (isTRUE(multi)) s$block else NULL,
                                  n_blocks)
        j <- .nl_axis_index(cn, aliases)
        if (is.na(j)) next
        blocks[[s$block]][[s$field]] <- as.numeric(theta_mat[, j])
    }
    blocks
}

# The nodes an axis was laid on for the fit just completed -- the INCOMING span
# the `"relative"` ceiling caps a re-placement against (`.nl_axis_span_u()`).
# Read off the fit rather than the prior, so an axis the caller left to the
# engine's own default has a reference span like any other.
.nl_axis_ref_nodes <- function(res, axis, block_index = NULL, n_blocks = 0L) {
    .nl_rescue_axis_nodes(
        res, .nl_axis_colname(res, .nl_axis_alias(axis, block_index, n_blocks)))
}

# The nodes a fit carried on one axis, for an axis the rescue re-crosses
# unchanged.
.nl_rescue_axis_nodes <- function(res, axis) {
    tg <- res$theta_grid
    if (is.null(tg)) return(NULL)
    if (!is.matrix(tg)) return(sort(unique(as.numeric(tg))))
    if (is.na(axis)) return(NULL)
    j <- match(axis, colnames(tg) %||% character(0))
    if (is.na(j)) return(NULL)
    sort(unique(as.numeric(tg[, j])))
}

# Standalone (non-joint) `tulpa_nested_laplace()` registry rescue
# (gcol33/tulpa#290, gcol33/tulpa#361) -- the registry generalization of
# `.joint_sigma_grid_rescue()`. Scope: every axis `.NL_REGISTRY_AXIS_FIELD`
# lists, on every block of the prior, each moved on its own rail in whichever
# coordinate the engine's transform registry gives it (`log` for a scale,
# `logit01` for the BYM2 mixing weight). The families it does NOT cover, and
# why each declines rather than passing in silence, are written out on that
# table.
#
# `multi = TRUE` says `prior` is a BLOCK LIST rather than one block, which is
# the only difference between the two `tulpa_nested_laplace()` paths here: the
# axes are then block-prefixed (`b<k>.tau`), provenance is read per block, and
# a moved block is re-crossed on its own fields while its neighbours keep
# theirs. A single-block prior is the length-1 case of the same walk -- there is
# one rescue, not one per path.
#
# One recenter attempt (not the joint path's two): the "runaway mode needs
# a regularizing prior" pathology `.joint_sigma_grid_rescue()`'s second
# attempt targets is specific to a donor/copy-coupled fit pushing toward
# near-separation (gcol33/tulpa#289's actual driver); a standalone
# single-response fit has no such coupling, so a geometry-only recenter is
# the proportionate fix here.
#
# `refit(prior_i)` reruns the full pipeline (dispatch, weights, moments,
# pareto-k) at a modified prior; `refit_log_marginal(prior_i, theta_mat)`
# re-evaluates just the inner marginal at an arbitrary theta matrix (used by
# the FD-Hessian stencil, many more calls, none of them a full fit). `auto` is
# the front door's provenance record.
.nl_registry_grid_rescue <- function(res, type, prior, refit, refit_log_marginal,
                                     auto = character(0),
                                     policy = .nl_recenter_mode(NULL),
                                     multi = FALSE,
                                     max_attempts = .nl_recenter("max_attempts_registry")) {
    out <- list(res = res, prior = prior)
    # The rail REPORT is taken before anything can decline, so a fit says which
    # of its axes do not contain their own mode whether or not this rescue is
    # allowed to, able to, or built to move them (gcol33/tulpa#370). It reads
    # stored weights and needs neither curvature nor a scope entry.
    out$res <- .nl_attach_railed_axes(out$res)

    multi    <- isTRUE(multi)
    blocks   <- if (multi) prior else {
        b <- if (is.list(prior)) prior else list()
        if (is.null(b$type)) b$type <- type
        list(b)
    }
    n_blocks <- if (multi) length(blocks) else 0L
    unwrap   <- function(b) if (multi) b else b[[1L]]
    bidx     <- function(s) if (multi) s$block else NULL
    alias_of <- function(s) .nl_axis_alias(s$axis, bidx(s), n_blocks)

    slots <- .nl_registry_axis_slots(blocks)
    tags  <- .nl_registry_axis_tags(out$res, blocks, multi)
    if (!length(slots) || is.null(tags) || anyNA(tags)) {
        out$res <- .nl_decline_recenter(out$res,
                                        .nl_out_of_scope_reason(out$res, tags))
        return(out)
    }
    if (identical(policy, "off")) {
        out$res <- .nl_decline_recenter(out$res, "auto_recenter_disabled")
        return(out)
    }

    # A prior that pins EVERY axis the table lists leaves the rescue nothing to
    # move whatever the fit did, which is the answer #290 gave and the one a
    # caller holding their own grid expects.
    pinned <- vapply(slots, function(s) .nl_axis_is_pinned(
        blocks[[s$block]], s$field, .nl_auto_fields_at(auto, bidx(s)),
        type = s$type), logical(1))
    if (all(pinned)) {
        out$res <- .nl_decline_recenter(out$res, "axis_pinned")
        return(out)
    }

    cur <- blocks
    attempt <- 0L
    reason  <- if (identical(policy, "resolve")) "grid_resolves_posterior" else
        "no_axis_railed"
    while (attempt < max_attempts) {
        # Every policy shares the FD mode/Hessian stencil, the `mode +/- span *
        # sd` node layout, the provenance gate and the attempt budget; they
        # differ only in WHEN the pass fires and, once it does, in HOW MANY of
        # the family's axes it re-places.
        #
        #   "rail"     fire on a railed axis, and move the railed axes alone --
        #              the placement half of gcol33/tulpa#361 as it shipped.
        #   "resolve"  fire on a railed OR an under-resolved axis, and move all
        #              of them: the stencil and the refit are paid once per fit,
        #              not once per axis, so the question the trigger asks is
        #              whether the FIT's grid is worth re-placing at all.
        #   "always"   fire unconditionally, and move all of them.
        railed <- vapply(slots, function(s) {
            .nl_edge_axis_hit(res, s$axis, bidx(s)) ||
                !is.null(.nl_axis_rail(res, .nl_axis_colname(res, alias_of(s))))
        }, logical(1))
        coarse <- if (!identical(policy, "resolve")) rep(FALSE, length(slots)) else
            vapply(slots, function(s) {
                nm <- .nl_axis_colname(res, alias_of(s))
                j  <- .nl_axis_index(names(tags), alias_of(s), tags)
                hs <- if (is.na(j)) NA_real_ else
                    .nl_axis_h_over_sd(res, nm, tags[[j]])
                !is.finite(hs) || hs > .nl_recenter("resolve_mult")
            }, logical(1))
        fire <- switch(policy,
                       always  = TRUE,
                       resolve = any(railed) || any(coarse),
                       any(railed))
        if (!fire) break
        sel <- if (identical(policy, "rail")) which(railed) else seq_along(slots)
        movable <- sel[!pinned[sel]]
        if (!length(movable)) {
            reason <- "axis_pinned"
            break
        }

        mc <- .nl_registry_axis_mode_cov(
            res, tags, function(theta_mat) refit_log_marginal(unwrap(cur), theta_mat))
        if (is.null(mc)) {
            reason <- "no_usable_curvature"
            break
        }

        moved    <- list()
        clamps   <- character(0)
        sd_used  <- numeric(0)
        sd_raw   <- numeric(0)
        declines <- character(0)
        for (i in movable) {
            s <- slots[[i]]
            j <- .nl_axis_index(mc$col_names, alias_of(s), mc$tags)
            if (is.na(j)) next
            rc <- .nl_recenter_axis_full(
                mc$tags[j], mc$u_mode[j], sqrt(mc$cov[j, j]),
                ref_span_u = .nl_axis_span_u(
                    .nl_rescue_axis_nodes(res, .nl_axis_colname(res, alias_of(s))),
                    mc$tags[j]))
            # The RAW SD is recorded even for an axis the policy declined -- that
            # is the reading which says whether declining was right.
            sd_raw[alias_of(s)[1L]] <- rc$sd_raw
            if (is.null(rc$nodes)) {
                clamps[alias_of(s)[1L]] <- rc$sd_clamp
                declines <- c(declines, rc$reason)
                next
            }
            clamps[alias_of(s)[1L]]  <- rc$sd_clamp
            sd_used[alias_of(s)[1L]] <- rc$sd_used
            moved[[paste(s$block, s$field, sep = ".")]] <- rc$nodes
        }
        # An axis the clamp policy declined says so; a mixed set of reasons
        # reports the generic one rather than picking a winner among them.
        if (!length(moved)) {
            reason <- if (length(unique(declines)) == 1L) unique(declines) else
                "no_usable_curvature"
            break
        }

        # Re-cross every axis of a block that moved -- the moved ones on their
        # new nodes, the rest on the nodes the fit already carried -- so the
        # prior goes back pre-paired the way the family's own `defaults()`
        # builds it. A block with nothing moved keeps its grid untouched.
        nxt <- cur
        ok  <- TRUE
        for (b in unique(vapply(slots[movable], function(s) s$block, integer(1)))) {
            bs <- Filter(function(s) identical(s$block, b), slots)
            axes <- lapply(bs, function(s)
                moved[[paste(b, s$field, sep = ".")]] %||%
                    .nl_rescue_axis_nodes(res, .nl_axis_colname(res, alias_of(s))))
            names(axes) <- vapply(bs, function(s) s$field, character(1))
            if (any(vapply(axes, is.null, logical(1)))) { ok <- FALSE; break }
            gr <- expand.grid(axes, KEEP.OUT.ATTRS = FALSE)
            for (f in names(axes)) nxt[[b]][[f]] <- as.numeric(gr[[f]])
        }
        if (!ok) {
            reason <- "no_usable_curvature"
            break
        }
        attempt <- attempt + 1L
        cur <- nxt

        res <- refit(unwrap(cur))
        res <- .nl_attach_railed_axes(res)
        res$outer_grid_placement         <- "auto_recentered"
        res$outer_grid_recenter_attempts <- attempt
        res$outer_grid_recenter_axes     <- vapply(
            slots[movable], function(s) .nl_axis_alias(s$axis, bidx(s),
                                                       n_blocks)[1L], character(1))
        res$outer_grid_recenter_sd_clamp <- clamps
        res$outer_grid_recenter_sd_used  <- sd_used
        res$outer_grid_recenter_sd_raw   <- sd_raw
        out <- list(res = res, prior = unwrap(cur))
    }
    out$res <- .nl_decline_recenter(out$res, reason)
    out
}
