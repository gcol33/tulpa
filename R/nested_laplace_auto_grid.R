# Shared mode-Hessian outer-axis recentering.
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
# optimizer: callers detect the rail (`.nl_axis_railed()`: the collapsed
# grid's dominant cell on the axis's node, or the axis's own marginal maximal at
# an endpoint), recentre once, and refit; a second attempt composes a light default PC(U, alpha) prior
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
#     (`.nl_axis_railed()` / `.nl_axis_index()`).
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
#   "rail"          the rail test alone, without the sizing half.
#   FALSE           the grid is integrated exactly as given, whatever it is.
#   "always"        every movable default axis is recentred whatever the fit
#                   did.
#
# A recentred axis is `mode +/- span * sd_used` over `n_pts` nodes, so its
# spacing is `1.25` PLACEMENT SDs, against a census median of 3.9 on the fixed
# spans.
#
# THAT IS NOT WHAT `.nl_axis_h_over_sd()` COMPUTES, and the two were read as one
# number (gcol33/tulpa#636). The trigger's ratio divides the spacing by the
# grid-WEIGHTED posterior SD the placed grid realizes, so on a recentred axis it
# is
#
#   h / sd_realized = 1.25 * sd_used / sd_realized,
#
# which is 1.25 only where the placement SD is the one the weights then realize.
# Wherever `.nl_recenter_sd_clamp()` SUBSTITUTED a bound it is not: the floor
# `min_sd_u` exists precisely to widen an axis whose measured curvature is
# sharper than it, so a floor-clamped placement reports
# `1.25 * min_sd_u / sd_raw` and is LARGER than 1.25 by exactly the factor the
# floor widened by. Measured on a (sigma, alpha, phi) donor + copy fixture, 18
# placed fits, the floor binding on all 18: `sd_raw` median 0.0543 against the
# substituted 0.15, realized weighted SD 0.0489, reported ratio 3.83 -- and
# `1.25 * 0.15 / 0.0489 = 3.83`. A reported 6.03 is `sd_raw = 0.031`. The
# placement is still doing its work there: the same axis un-recentred reads a
# median 55.2 on the same fixture. `outer_grid_recenter_sd_clamp` / `_sd_raw` /
# `_sd_used` are what separate a substituted spread from a measured one on the
# fit, and a recentred axis's reported ratio cannot be read without them.
#
# The FIT'S OWN `outer_grid_h_over_sd` is a third quantity again, and the two
# are not interchangeable (gcol33/tulpa#660). That field is filled by
# `.nl_axis_resolution()` (`R/nested_laplace_moments.R`), whose denominator is
# the three-point Laplace-at-mode SD of the UNWEIGHTED marginal -- a local
# curvature, which is finite on an axis whose weights have collapsed onto one
# node where the trigger's moment spread is 0 and the ratio `Inf`, and which
# declines `mode_at_edge` on a railed axis the trigger scores a number for.
# Measured on twelve axes of BYM2 and ICAR fits (`dev_notes/issue660/
# probe660b.R`): both finite on eight, equal on none, field-over-trigger median
# 1.0012 and range 0.0000 to 1.3793, six orders apart on a collapsed BYM2
# `sigma`. The two are kept because they answer differently on purpose -- the
# trigger has to fire on a collapsed axis, and the report has to decline what it
# cannot score -- so read the derivation above against `.nl_axis_h_over_sd()`,
# which is what `test-recenter-pilot.R` asserts it on, and not against the
# field.
#
# An axis NO placement moves -- a caller-pinned copy `alpha` or per-arm
# dispersion axis -- carries no such relation at all: its ratio is whatever its own fixed
# nodes and its own posterior make it (median 1605 and 1.95 on the same
# fixture), and reading a large one there as a placement failure mistakes a
# sharp posterior on a fixed axis for a mis-sized one.
#
# A DECLARED per-arm dispersion axis or copy scale is not in that set
# (`.joint_place_axes()`, gcol33/tulpa#663, gcol33/tulpa#925): marked with
# `auto_grid()` it is placed like a field SD, and left unmarked it is a pin whose ratio means
# what the paragraph above says it means. Which of the two a given axis is, is
# recorded on the fit per axis (`outer_grid_axis_declined`), because the
# whole-fit `outer_grid_recenter_declined` slot holds one reason for the fit
# and says nothing about an axis the pass left alone beside one it moved.
#
# The default is "resolve" and not "always" for COST, and the two are closer
# than the coverage table alone reads. They agree seed for seed on five of the
# six measured configurations; they differ on the one whose default axes
# already resolve their own posterior (NNGP, median `h / sd` 1.81 and 1.50),
# where "resolve" fires on 39.5% of seeds against 97.5% and covers 0.530 /
# 0.500 at the 50% level against 0.135 / 0.385. Coverage is what arbitrates a
# placement rule, so that row favours "resolve" -- but the
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
# (every `occu_cover()` fit's auto-recenter was inert for
# exactly this reason). Provenance is only known to the layer that CHOSE the
# values, so `auto_grid()` lets that layer say so.

#' Mark an outer-grid setting as a default rather than a pin
#'
#' @description
#' Declares that a setting shaping the outer hyperparameter grid carries a
#' *default* the caller computed, not a choice the user made. The auto-recenter
#' pass (\code{outer_grid_placement}) leaves a user-pinned
#' setting exactly as given, and re-centres (or, for a prior, engages its own
#' regularizer over) a marked one when the fit rails against its ceiling.
#'
#' Four kinds of setting take the mark:
#' \itemize{
#'   \item a grid axis on a nested-Laplace `prior` block (`sigma_grid`,
#'     `tau_grid`, ...) -- a numeric vector of nodes, or the `[n_cells x k]`
#'     matrix of pre-paired coordinates the families whose axis is a matrix
#'     take (`mcar` / `miid`'s `logchol_grid`, `tgmrf`'s `theta_grid_built`);
#'   \item an entry of [tulpa_nested_laplace_joint()]'s `phi_grid` -- one arm's
#'     dispersion axis;
#'   \item a scalar grid-construction knob in `control`, for a driver that
#'     builds its axes rather than taking them (`fit_st_nested()`'s
#'     `n_grid_spatial`, `tau_upper`, ...);
#'   \item a `prior_sigma` hyperprior specification -- a list, e.g.
#'     `list("pc.prec", c(U = 3, alpha = 0.01))`.
#' }
#'
#' Wrapper packages are the intended caller: one that builds a default of its
#' own -- because it derives a second axis from it, hands the same vector to
#' several blocks, or exposes its own argument with a default -- would
#' otherwise be indistinguishable from a user who pinned that setting
#' deliberately. Mark it and the rescue stays live. A setting whose value is
#' exactly the engine's own default is recognised without a mark; anything else
#' needs one -- and a dispersion axis ALWAYS needs one, since the engine has no
#' default of its own to recognise it against.
#'
#' The mark is an attribute, so it is dropped by `sort()`, `[`, `c()` and
#' `as.numeric()`: build the value first, mark it last.
#'
#' @section Declaring a default the engine must integrate as written:
#' `place = FALSE` separates the two questions the mark otherwise answers at
#' once. Provenance -- whose choice these nodes are -- and placement policy --
#' whether the pass may move them -- are different questions, and a package that
#' measured its own default as the one to integrate has an answer to the second
#' that is not "the user pinned it". Such an axis is treated exactly as a pin
#' everywhere the engine ACTS on it (the placement pass leaves it, refinement
#' densifies within its span rather than following the posterior past the end
#' nodes, and no curvature is computed for it), and differs only in what the fit
#' REPORTS: `"default_axis_pinned"` rather than `"axis_pinned"`, so a reader is
#' not told they pinned an axis they never wrote down. Leaving the mark off
#' instead buys the same integration and says the wrong thing about it.
#'
#' `place` is read wherever the engine would otherwise RE-PLACE the setting: a
#' grid axis, and a scalar grid-construction knob. A `prior_sigma`
#' specification is not re-placed but replaced, by the engine's own
#' regularizer, and carries no `place`.
#'
#' @param x Numeric vector or matrix of grid nodes, a numeric scalar knob, or a
#'   prior-specification list.
#' @param place May the auto-placement pass move this axis onto its own
#'   posterior? `TRUE` (default) declares a default the engine may re-place;
#'   `FALSE` declares one it must integrate as written.
#' @return `x` carrying the marker attribute. Numeric input is coerced to
#'   double IN PLACE, so everything else it carries -- `dim()` and `dimnames()`
#'   above all -- survives the mark; a list is returned unchanged apart from
#'   the attribute.
#' @seealso [is_auto_grid()], [auto_grid_place()],
#'   [tulpa_nested_laplace_joint()], [fit_st_nested()]
#' @examples
#' prior <- list(type = "icar", sigma_grid = auto_grid(c(0.1, 0.5, 1, 2, 3)))
#' is_auto_grid(prior$sigma_grid)
#' is_auto_grid(auto_grid(list("pc.prec", c(U = 3, alpha = 0.01))))
#' auto_grid_place(auto_grid(c(0.5, 1, 2), place = FALSE))
#' @export
auto_grid <- function(x, place = TRUE) {
    if (!is.logical(place) || length(place) != 1L || is.na(place)) {
        stop("`auto_grid(place = )` takes TRUE or FALSE.", call. = FALSE)
    }
    if (is.list(x)) {
        if (!length(x)) {
            stop("`auto_grid()` takes a non-empty prior specification.",
                 call. = FALSE)
        }
    } else {
        # In place, not `as.numeric()`: two families store their axis as a
        # matrix of pre-paired coordinates (`mcar` / `miid`'s `logchol_grid`,
        # `tgmrf`'s `theta_grid_built`), and flattening one destroys the axis
        # the caller is declaring.
        storage.mode(x) <- "double"
        if (!length(x) || anyNA(x)) {
            stop("`auto_grid()` takes a non-empty numeric grid with no NA.",
                 call. = FALSE)
        }
    }
    attr(x, "tulpa_auto_grid") <- TRUE
    # Only the non-default state is carried, so a placeable mark is the byte it
    # always was and `auto_grid_place()` answers for an unmarked value too.
    attr(x, "tulpa_auto_place") <- if (place) NULL else FALSE
    x
}

#' Is an outer-grid setting marked as a default?
#'
#' @param x Any object.
#' @return `TRUE` when `x` carries the [auto_grid()] marker. This is the
#'   PROVENANCE question -- whose choice the nodes are -- and is `TRUE` whether
#'   or not the mark also asked for them to be integrated as written; that is
#'   [auto_grid_place()].
#' @seealso [auto_grid()], [auto_grid_place()]
#' @examples
#' is_auto_grid(auto_grid(c(0.5, 1, 2)))
#' is_auto_grid(c(0.5, 1, 2))
#' @export
is_auto_grid <- function(x) isTRUE(attr(x, "tulpa_auto_grid", exact = TRUE))

#' May the placement pass move a marked outer-grid axis?
#'
#' Reads back what [auto_grid()]'s `place` argument recorded, so a wrapper that
#' rebuilds a value (`as.numeric()` drops every attribute) can re-apply both
#' halves of the declaration rather than only the provenance half.
#'
#' @param x Any object.
#' @return `FALSE` when `x` was marked `auto_grid(place = FALSE)`, `TRUE`
#'   otherwise -- including for a value carrying no mark at all, which the
#'   engine holds because it reads as a pin rather than because it asked to be
#'   held.
#' @seealso [auto_grid()], [is_auto_grid()]
#' @examples
#' auto_grid_place(auto_grid(c(0.5, 1, 2)))
#' auto_grid_place(auto_grid(c(0.5, 1, 2), place = FALSE))
#' @export
auto_grid_place <- function(x)
    !isFALSE(attr(x, "tulpa_auto_place", exact = TRUE))

# Is a supplied `prior_sigma` a PIN? The prior-spec counterpart of
# `.nl_axis_is_pinned()`. The second recenter attempt exists
# to engage the weakly-informative PC prior on a mode with no finite curvature
# to settle on, and it must not be suppressed by a wrapper package that stamps a
# `prior_sigma` of its own -- the same presence-is-not-provenance mistake one
# field over. Absent, marked with `auto_grid()`, or equal by value to
# the engine's own `.NL_RECENTER$sigma_pc_prior` are all defaults; anything else
# is a deliberate choice the rescue leaves alone.
.nl_prior_sigma_is_pinned <- function(prior_sigma) {
    if (is.null(prior_sigma)) return(FALSE)
    if (is_auto_grid(prior_sigma)) return(FALSE)
    d <- .nl_recenter("sigma_pc_prior")
    !isTRUE(all.equal(d, .nl_strip_auto(prior_sigma), check.attributes = FALSE))
}

# Drop the marker so nothing downstream of the rescue sees an attributed
# object (a prior spec is passed on to `.joint_parse_sigma_prior()`). Both
# halves of the declaration go: `auto_grid()` writes two attributes and a value
# that kept one of them would still reach `expand.grid()` / `cbind()` / C++
# attributed.
.nl_strip_auto <- function(x) {
    attr(x, "tulpa_auto_grid")  <- NULL
    attr(x, "tulpa_auto_place") <- NULL
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

# The grid fields on ONE block that carry the `auto_grid()` marker, as a NAMED
# LOGICAL: the names are the declared fields, and each value is whether that
# declaration also lets the placement pass move the axis
# (`auto_grid(place = )`). One record carries both halves, so no call site can
# read the provenance half and miss the policy half.
.nl_block_auto_fields <- function(block) {
    if (!is.list(block) || !length(block)) return(logical(0))
    nm <- names(block) %||% character(0)
    if (!length(nm)) return(logical(0))
    keep <- vapply(block, is_auto_grid, logical(1)) & nzchar(nm)
    if (!any(keep)) return(logical(0))
    stats::setNames(vapply(block[keep], auto_grid_place, logical(1)), nm[keep])
}

.nl_block_strip_auto <- function(block) {
    for (f in names(.nl_block_auto_fields(block))) {
        block[[f]] <- .nl_strip_auto(block[[f]])
    }
    block
}

# Record which axes a prior declared as defaults, and hand back the prior with
# the markers removed, so nothing downstream of the front door ever sees an
# attributed numeric (grid values reach C++, `expand.grid()` and `cbind()`
# unchanged). `auto` is a named logical for a single-block prior and a
# per-block list of them for a multi-block one -- read it back with
# `.nl_auto_fields_at()`.
.nl_grid_provenance <- function(prior) {
    if (.is_multi_block_prior(prior)) {
        auto <- lapply(prior, .nl_block_auto_fields)
        prior <- lapply(prior, .nl_block_strip_auto)
        return(list(prior = prior, auto = auto))
    }
    if (!is.list(prior)) return(list(prior = prior, auto = logical(0)))
    list(prior = .nl_block_strip_auto(prior), auto = .nl_block_auto_fields(prior))
}

.nl_auto_fields_at <- function(auto, block_index = NULL) {
    if (is.null(auto)) return(logical(0))
    at <- if (is.null(block_index)) {
        if (is.list(auto)) return(logical(0))
        auto
    } else {
        if (!is.list(auto) || block_index > length(auto)) return(logical(0))
        auto[[block_index]] %||% logical(0)
    }
    # A plain character vector names the declared fields and says nothing about
    # placement, which is the placeable default.
    if (is.character(at)) return(stats::setNames(rep(TRUE, length(at)), at))
    at
}

# THE provenance predicate every rescue guards on: why must the placement pass
# leave this axis exactly as declared, and NULL when it may move it. `block` is
# the prior block carrying the axis, `field` its grid field, `auto_fields` the
# marker record `.nl_grid_provenance()` took for that block. An absent axis, a
# marked one, and one whose nodes are the engine's own default are all defaults;
# anything else is a pin.
#
# Two answers hold the axis, and the engine ACTS identically on both -- the pass
# leaves it, refinement densifies within its span (`.NL_AXIS_REFINE`), and no
# curvature is computed for it. They differ in WHOSE declaration it was, which
# is the whole of what a reader can act on: `"axis_pinned"` is the caller's own
# nodes, `"default_axis_pinned"` a default the package that built the fit asked
# to have integrated as written (`auto_grid(place = FALSE)`). Reporting the
# second as the first tells a user they pinned an axis they never wrote down.
#
# `type` narrows the default comparison to the axis that ONE path-and-family
# lays on the field, and must be passed EXPLICITLY -- it is deliberately not
# inferred from `block$type`, because the block's family is not the same thing
# as the path that defaulted the axis: a joint areal fit carries
# `type = "icar"` on a block whose `sigma_grid` default comes from
# `.joint_areal`, while the icar REGISTRY entry defaults a precision axis and no
# `sigma_grid` at all. Inferring would silently answer "pinned" there, reading
# an engine default as a user pin. Unnarrowed (`NULL`) compares against
# every family's binding for the field, which errs toward recognising a default.
.nl_axis_hold <- function(block, field, auto_fields = logical(0), type = NULL) {
    g <- if (is.list(block)) block[[field]] else NULL
    if (is.null(g)) return(NULL)
    auto_fields <- .nl_auto_fields_at(auto_fields)
    i <- match(field, names(auto_fields) %||% character(0))
    # The record is taken with the markers stripped; a call site holding the
    # value before that reads the same declaration off the value itself.
    declared <- !is.na(i) || is_auto_grid(g)
    if (declared) {
        place <- if (!is.na(i)) isTRUE(auto_fields[[i]]) else auto_grid_place(g)
        return(if (place) NULL else "default_axis_pinned")
    }
    if (.nl_axis_matches_default(g, field, type)) return(NULL)
    "axis_pinned"
}

.nl_axis_is_pinned <- function(block, field, auto_fields = logical(0),
                               type = NULL) {
    !is.null(.nl_axis_hold(block, field, auto_fields, type = type))
}

# --- axis consumption --------------------------------------
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
# (`$axis_fields_dropped`) rather than left invisible.

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
# above gives an unread axis (a `sigma_pos_grid` from the
# retired (sigma_occ, sigma_pos) parameterization reached this spec and was
# neither read nor reported, so a pinned amplitude axis fell back to the
# engine's own default with a bit-identical `log_marginal`).
.NL_COPY_SPEC_FIELDS <- c("arm", "block", "alpha_grid", "alpha_n")

# Resolve a copy spec's alpha axis. `alpha_grid` REPLACES the axis (the caller
# states the nodes, and with them the prior structure the axis carries);
# `alpha_n` re-reads the engine's declared axis at a higher RESOLUTION, keeping
# the atom at 0 and the slab bounds. They answer different questions, so
# supplying both is refused rather than silently ranked (gcol33/tulpa#633).
#
# Why the resolution knob has to exist: the alpha axis is the one outer axis a
# copy fit cannot raise. Measured engine-side on an ICAR chain with a gaussian
# copy arm, raising the donor `sigma_grid` from 13 to 29 nodes leaves the alpha
# axis at the 6 nodes it was declared at in that probe, at every setting and the grid ESS at 1.7 / 3.1 / 4.3,
# while supplying the alpha nodes explicitly takes it to 2.5 / 7.6 / 14.4. The
# saturation is in the PLACEMENT, not in the prune: `prune = TRUE` reproduces
# the same node counts and the same ESS to the digit
# (`dev_notes/issue633/probe_alpha_engine.R`).
.nl_copy_alpha_axis <- function(alpha_grid, alpha_n, what = "copy") {
    has_grid <- !is.null(alpha_grid) && length(alpha_grid) > 0L
    has_n    <- !is.null(alpha_n) && length(alpha_n) > 0L
    if (has_grid && has_n) {
        stop(what, ": give `alpha_grid` OR `alpha_n`, not both -- `alpha_grid` ",
             "states the axis's nodes, `alpha_n` re-reads the engine's own axis ",
             "at a higher resolution, and the two are different requests.",
             call. = FALSE)
    }
    if (has_grid) return(as.numeric(alpha_grid))
    .nl_grid_axis("copy_alpha", n = if (has_n) alpha_n else NULL)
}

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
# every family attaches regardless of `diagnose_k`.
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

# --- axis rails --------------------------------------------
#
# `.nl_edge_axis_hit()` above asks the collapse question: did the WHOLE grid
# collapse onto one cell, and does that cell sit on a node. That is a joint
# quantity over the tensor -- `ess_grid = 1 / sum(w^2)` across every cell -- so
# on a crossed grid a second axis carrying spread lifts it past the collapse
# threshold while one axis is hard against its own boundary. The support
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

# The three measures one axis's marginal can be read against. They differ only
# in which cell widths the nodes are weighed by, and each answers a different
# question (gcol33/tulpa#660):
#
#   "posterior"  the cells' own quadrature weights, closed inside the axis's
#                declared domain -- the measure the fit integrates and the one
#                every reported mean, interval and spread has to carry.
#   "span"       the same widths with the outermost cell left on its naive
#                half-step mirror. Where the support closes is prior-side
#                bookkeeping; a detector asking how much of the marginal the
#                span leaves out must not weaken because the boundary it is
#                looking at is the boundary the parameter stops at.
#   "inner"      no measure at all: the likelihood the grid measured. What an
#                ARGMAX question reads, since a mode is a property of the
#                density and not of the cells it is tiled with.
.NL_AXIS_MEASURE <- c("posterior", "span", "inner")

# The cells' log quadrature weights under `measure`, or NULL for none.
#
# The "span" weights are rebuilt from the grid rather than stored, and the
# rebuild is the SAME call every producer fills `log_quad` with
# (`.nl_grid_log_quad()` on the grid alone, no caller specs), with the domain
# closure switched off -- so the two differ in the outermost cell and in
# nothing else.
.nl_axis_measure_quad <- function(res, measure) {
    switch(measure,
           posterior = res$log_quad,
           inner     = NULL,
           span      = if (is.null(res$log_quad)) NULL else
                           .nl_grid_log_quad(.nl_theta_matrix(res),
                                             close_domain = FALSE,
                                             refining = res$refining_axis))
}

# One axis's marginal over its own sorted distinct nodes, normalized.
.nl_axis_marginal_w <- function(res, axis,
                                measure = c("posterior", "span", "inner")) {
    measure <- match.arg(measure)
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
    lq <- .nl_axis_measure_quad(res, measure)
    measured <- !is.null(lq) && length(lq) == length(lm)
    if (measured) {
        lm <- lm + lq
        lm[is.na(lm)] <- -Inf
    }
    m <- .nl_axis_marginal_logdensity(
        as.numeric(tg[, j]), lm,
        .nl_axis_read_cells(res$refining_axis, nrow(tg), measured = measured))
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
# Both clauses read the INNER marginal, the likelihood the grid measured, with
# the cells' quadrature weights left out. A rail asks whether the span stops
# short of where the data are still pulling the marginal, and the answer must
# not move when the outer prior's cells are re-apportioned: a cell width is a
# property of the node spacing and of where the axis's own domain closes the
# outermost cell (`.hyper_domain_clamp()`), and the boundary cell is precisely
# the one both of those shorten. Folding the widths in makes the detector
# weakest at the boundary it is being asked about -- on the 100-region BYM2
# `rho` fixture the same railed configuration reads 2.511 on the inner marginal
# against 2.099 with the widths folded in, and the shortened boundary cell of a
# clamped support takes the latter to 1.696, below a threshold calibrated at
# 2.511. The posterior summaries of the axis keep the weighted marginal, which
# is what a reported mean or interval has to carry.
#
# The guard reads the boundary node's weight against what a FLAT marginal would
# put there, `1 / m`, rather than against a fixed share. The
# two differ by exactly the node count, and that is the whole defect: an axis's
# weights are a distribution over its OWN nodes, so the same posterior read at
# more nodes carries less on any one of them and a fixed share turns a longer
# axis into a weaker detector. `lift = m * w[k]` is 1 for a flat marginal at any
# node count, so the threshold means the same thing wherever it is applied.
# Measured on a BYM2 `rho` posterior held past a fixed span, the
# lift of a railed fit is flat in the node count (2.70 / 2.81 / 2.87 / 2.94 /
# 3.02 at m = 4 / 5 / 6 / 8 / 12) where its share collapses (0.68 / 0.56 / 0.48
# / 0.37 / 0.25).
#
# A declared point mass is not an endpoint. The copy scale's zero level on its
# log axis (`.hyper_axis_scale()`'s one rule) is the "no coupling" model: a
# marginal heaviest there has its mass ON the model the atom states, not past
# the span, so the axis is not railed; otherwise the rail is read on the
# continuum alone.
.nl_axis_rail <- function(res, axis, edge_mult = .nl_recenter("edge_mass_mult")) {
    mw <- .nl_axis_marginal_w(res, axis, measure = "inner")
    if (is.null(mw)) return(NULL)
    if (isTRUE(.hyper_axis_scale(sub("^b[0-9]+[.]", "", axis))) &&
        any(mw$vals <= 0)) {
        if (mw$vals[which.max(mw$w)] <= 0) return(NULL)
        mw <- .nl_axis_continuum(mw, "log")
        if (is.null(mw) || length(mw$w) < 2L) return(NULL)
    }
    m <- length(mw$w)
    k <- which.max(mw$w)
    if (k != 1L && k != m) return(NULL)
    lift <- m * mw$w[k]
    if (!is.finite(lift) || lift < edge_mult) return(NULL)
    list(side = if (k == 1L) "lower" else "upper",
         mass = mw$w[k], lift = lift, node = mw$vals[k])
}

# Is `axis` railed on this fit? The one trigger every placement rescue reads: the
# collapsed grid's dominant cell on the axis's node (`.nl_edge_axis_hit()`), or
# the axis's own marginal maximal at an endpoint (`.nl_axis_rail()`). The second
# is what keeps the answer from depending on OTHER axes: a grid whose weight is
# spread along a dispersion axis the consistency pass resolved has an `ess_grid`
# past the collapse threshold while its field SD sits on its ceiling
# (gcol33/tulpa#858).
.nl_axis_railed <- function(res, axis, block_index = NULL,
                            n_blocks = .nl_fit_n_blocks(res)) {
    if (.nl_edge_axis_hit(res, axis, block_index)) return(TRUE)
    nm <- .nl_axis_colname(res, .nl_axis_alias(axis, block_index, n_blocks))
    !is.na(nm) && !is.null(.nl_axis_rail(res, nm))
}

# Is a field-SD axis one of the two sigma rescues moves railed? Every block's,
# under the spelling its grid gives it.
.nl_sigma_axis_railed <- function(res) {
    n_b <- .nl_fit_n_blocks(res)
    if (n_b <= 1L && .nl_axis_railed(res, "sigma")) return(TRUE)
    for (b in seq_len(n_b)) if (.nl_axis_railed(res, "sigma", b)) return(TRUE)
    FALSE
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
#
# It reads the "posterior" measure, and unlike the two labels above that is not
# a close call: the denominator is a SPREAD, the spread of the distribution the
# fit reports its intervals from, and that distribution is the one weighed by
# the cells (gcol33/tulpa#660). The trigger's threshold
# `.NL_RECENTER$resolve_mult` was nevertheless calibrated at cc9ef82
# (2026-08-10), twenty days before `53a2ef9` folded those weights in, so what it
# was sized against is the unweighted read. Re-measured across 240 axis-
# configurations spanning both -- ICAR precision axes at eight node counts and
# BYM2 (sigma, rho) grids at four ceilings x four node counts
# (`dev_notes/issue660/probe660[bc].R`) -- the two reads flip the fire decision
# on NONE of them, and on a single-axis log-spaced grid they are equal to the
# bit, because a uniform coordinate spacing gives every cell the same width and
# a constant shifts no softmax. The ratio spans 0.78 to 1.34 where they differ,
# which is inside the 1.6 headroom the threshold was chosen with.
#
# A declared point mass is not part of the resolution: the copy scale's zero
# level on its log axis (`.hyper_axis_scale()`'s one rule) is a separate model
# the continuum's spacing says nothing about, so the read is taken over the
# continuum alone.
.nl_axis_h_over_sd <- function(res, axis, tag) {
    if (length(tag) != 1L || is.na(tag)) return(NA_real_)
    mw <- .nl_axis_continuum(.nl_axis_marginal_w(res, axis, measure = "posterior"),
                             tag)
    if (is.null(mw) || length(mw$vals) < 2L) return(NA_real_)
    u <- as.numeric(.joint_pareto_fwd(tag, mw$vals))
    if (any(!is.finite(u))) return(NA_real_)
    mu <- sum(mw$w * u)
    sd <- sqrt(max(0, sum(mw$w * u^2) - mu^2))
    if (!is.finite(sd)) return(NA_real_)
    if (sd <= 0) return(Inf)
    stats::median(diff(sort(u))) / sd
}

# One axis's marginal read (`.nl_axis_marginal_w()`) with any declared point
# mass dropped and the rest renormalized: the zero level of a log-tagged axis.
# An axis with no such level is returned as it came.
.nl_axis_continuum <- function(mw, tag) {
    if (is.null(mw) || !identical(tag, "log")) return(mw)
    keep <- mw$vals > 0
    if (all(keep)) return(mw)
    s <- sum(mw$w[keep])
    if (!any(keep) || !is.finite(s) || s <= 0) return(NULL)
    list(vals = mw$vals[keep], w = mw$w[keep] / s, col = mw$col)
}

# Does a MOVABLE axis need placing on this fit? It does when it rails, or when
# its own grid does not resolve its own posterior (`h / sd` past
# `.NL_RECENTER$resolve_mult`, or unreadable). An axis whose heaviest level is a
# declared point mass does not: the mode-find holds the point where it is
# (`.nl_placement_mode()`), so a placement asked for there would pay a
# mode-find and lay nothing. The one predicate every family of the joint
# placement pass and the mode-find's own trigger read.
.nl_axis_placement_fires <- function(res, axis, tag = "log") {
    mw <- .nl_axis_marginal_w(res, axis, measure = "posterior")
    if (!is.null(mw) && identical(tag, "log") &&
        mw$vals[which.max(mw$w)] <= 0) return(FALSE)
    if (.nl_axis_railed(res, axis)) return(TRUE)
    hs <- .nl_axis_h_over_sd(res, axis, tag)
    !is.finite(hs) || hs > .nl_recenter("resolve_mult")
}

# Every axis of a fit that is railed, as `axis:side`, whether or not any rescue
# covers it. Recorded on the fit (`$outer_grid_railed_axes`) so a span that does
# not contain its own posterior mode is visible instead of silently integrating
# a tail: a placement the engine leaves alone has to say so.
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

# --- boundary MASS ------------------------------------------------------------
#
# The rail above asks whether the span contains the axis's own MODE. An axis can
# hold a large share of its marginal on a boundary node with the mode one node
# in, and that is the same truncation: whatever the marginal does past the outer
# node is unrepresented either way, and the mode's position does not say how much
# of it there is. Read on its own, the rail reports such a grid clean
# (gcol33/tulpa#622).
#
# So this is a SECOND label rather than a widening of the first: `railed` keeps
# its stronger statement ("the span does not contain its own mode") and
# `edge_mass` names an axis whose boundary node carries material weight whatever
# the argmax does. Both ends are tested, since a marginal can press on either.
#
# The currency is the rail's own `lift = m * w_edge`, the boundary node's weight
# against what a flat marginal would put there, for the reason the rail carries
# it: a fixed share makes a longer axis a weaker detector of the same posterior.
# `.nl_diag("edge_mass_lift")` carries the threshold and what it was read off.
#
# It reads the "span" measure, and which half of the measure that drops is the
# whole of the choice (gcol33/tulpa#660). This is a MASS question, so the cell
# widths belong in it: an outer node owning a wide cell holds mass its node
# count does not see, and reading the bare node weights would delete that. What
# does NOT belong is `.hyper_domain_clamp()`, which shortens the outermost cell
# when the naive mirror would reach past the axis's declared support -- a
# statement about where the parameter stops existing, applied to the one cell
# the detector is asking about, and moved by a knob it is not asking about.
#
# Measured over 192 BYM2 (sigma, rho) axis-configurations, 6 data sets x 4 rho
# ceilings x 4 node counts (`dev_notes/issue660/probe660d.R`), lift ratios to
# the bare-node read:
#
#   with the clamp (as shipped 53a2ef9 .. 0.2.14)   median 0.7826   min 0.5100
#   with the mirror left standing                   1.0000 on every row
#
# So on that family the clamp is the ENTIRE difference, and it is one-sided:
# thirteen axes lost the label under it and none gained one. `1/0.51` is the
# half-width an open domain's midpoint rule leaves the outermost cell, which is
# the largest a single clamp can be.
#
# Returns a list of `list(side, mass, lift, node)`, empty when neither end
# qualifies.
.nl_axis_edge_mass <- function(res, axis,
                               lift_mult = .nl_diag("edge_mass_lift")) {
    mw <- .nl_axis_marginal_w(res, axis, measure = "span")
    if (is.null(mw)) return(list())
    m <- length(mw$w)
    out <- list()
    for (k in c(1L, m)) {
        lift <- m * mw$w[k]
        if (!is.finite(lift) || lift < lift_mult) next
        out[[length(out) + 1L]] <- list(
            side = if (k == 1L) "lower" else "upper",
            mass = mw$w[k], lift = lift, node = mw$vals[k])
    }
    out
}

# Every axis of a fit holding material weight on one of its own boundary nodes,
# as `axis:side`. Recorded on the fit (`$outer_grid_edge_mass_axes`) beside the
# railed axes, so a span that truncates its own marginal is named whether or not
# it also fails to contain its mode.
.nl_edge_mass_axes <- function(res) {
    tg <- res$theta_grid
    if (is.null(tg) || is.null(res$log_marginal)) return(character(0))
    cn <- if (is.matrix(tg)) colnames(tg) else (res$theta_names %||% "theta")
    if (is.null(cn)) return(character(0))
    hits <- character(0)
    for (a in cn) {
        for (e in .nl_axis_edge_mass(res, a)) {
            hits <- c(hits, paste0(a, ":", e$side))
        }
    }
    hits
}

# Why the registry rescue covers no axis of `type`. Two
# distinguishable answers, and returning unstamped conflated them with a fit the
# rescue never applied to:
#
#   * `"unguessable_axis: <names>"` -- the family HAS a positive-scale axis the
#     rescue could place, but some other axis of the same grid has a support the
#     transform registry will not guess (car_proper's `rho_car` on the adjacency
#     eigenvalue interval), and `.nl_registry_axis_mode_cov()` declines for the
#     WHOLE fit rather than per axis. Naming the blocking axis is the decline
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
# not run: `"axis_pinned"` (the caller pinned the axis),
# `"default_axis_pinned"` (a wrapper package declared the nodes and asked for
# them as written, `auto_grid(place = FALSE)`), `"grid_not_collapsed"`
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
# from it would be laid from a substituted spread rather than a measured one;
# only under the declining clamp policies),
# `"attempts_exhausted"` (per axis only: the axis was still railed when
# `max_attempts` ran out on a sibling), `"auto_recenter_disabled"`
# (`control$auto_recenter = FALSE`,
# the way to hold ANY grid -- the engine's own default axis included -- exactly
# where it is), `"grid_knobs_overridden"` (the spatiotemporal driver's
# grid-construction knobs were set explicitly), `"refit_failed"` (the recentred
# grid did not solve), `"unguessable_axis: <names>"` / `"family_out_of_scope"`
# (the registry rescue covers no axis of this family, which is what returning
# UNSTAMPED used to look like). Absent on a fit that WAS
# recentred, and never stamped by
# a rescue whose prior shape it does not apply to.
#
# SEVERAL rescues can speak on one fit -- a joint fit's field SD and its per-arm
# dispersion are placed by different passes over the same grid -- so the slot is
# a REDUCTION over what they said, not the last one to say it. The two
# axis-scoped reasons are properties of ONE axis's declaration rather than of
# the fit's grid, its curvature or a control knob, so a pass whose every axis
# was declared has not answered the question the slot asks; it yields to any
# pass that had an axis it could have moved and did not need to. Without that,
# a fit whose defaulted field SD simply needed no placement reported
# `"axis_pinned"` because a dispersion axis beside it was declared -- which
# reads as the caller having pinned the axis they did not pin
# (gcol33/tulpaObs#361). The per-axis record answers per axis either way.
#
# `outer_grid_recenter_declined_pruned` says whether the fit the decline was
# read off had been cheap-pass screened. The two are different events with the
# same reason string: `"no_usable_curvature"` on a full grid is a posterior the
# stencil could not read, while the same reason on a screened grid can be the
# screen having removed the cells the stencil reads -- pruning pays off exactly
# when the posterior is concentrated, and concentration is what collapses the
# kept set. Recorded on every decline, TRUE or FALSE, so a reader tells them
# apart from the fit rather than from the absence of a field.
.NL_AXIS_SCOPED_DECLINE <- c("axis_pinned", "default_axis_pinned")

.nl_decline_is_axis_scoped <- function(reason)
    length(reason) == 1L && !is.na(reason) &&
        reason %in% .NL_AXIS_SCOPED_DECLINE

# One reason for a set of per-axis holds. A user's own pin is the statement a
# reader can act on, so it stands for the set whenever one is in it.
.nl_reduce_decline <- function(held) {
    held <- unlist(held, use.names = FALSE)
    if (!length(held)) return("axis_pinned")
    if ("axis_pinned" %in% held) "axis_pinned" else held[[1L]]
}

.nl_decline_recenter <- function(res, reason) {
    if (identical(res$outer_grid_placement, "auto_recentered")) return(res)
    prev <- res$outer_grid_recenter_declined
    if (!is.null(prev) && .nl_decline_is_axis_scoped(reason)) {
        # A pass whose every axis was DECLARED has not answered the question
        # the slot asks, so it yields to one that had an axis it could have
        # moved; against another axis-scoped answer the two reduce by the same
        # rule the per-pass one does.
        if (!.nl_decline_is_axis_scoped(prev)) return(res)
        reason <- .nl_reduce_decline(list(prev, reason))
    }
    res$outer_grid_recenter_declined <- reason
    res$outer_grid_recenter_declined_pruned <-
        isTRUE(any(as.logical(res$prune_mask), na.rm = TRUE))
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
# THE CLAMP IS NOT A MEASUREMENT. Whenever a bound binds, the
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

# Build the recentered axis for `axis` from the (mode, covariance) a fit
# carries -- `mode_u` / `cov_u` / `axis_tags` / `axis_names` are the joint
# fit's `res$outer_mode_u` / `res$outer_mode_cov_u` / `res$outer_mode_axis_tags`
# / `res$outer_mode_axis_names`, the outer mode its placement mode-find reached
# (`.joint_attach_placement()`, R/nested_laplace_placement.R). `axis` is
# the bare axis name; `block_index` / `n_blocks` resolve it against the fit's
# own spelling (see `.nl_axis_alias()`). Only recentres a positive-scale
# ("log"-tagged) axis; declines (returns NULL) for an axis absent from the
# grid, an axis on a different transform (e.g. a BYM2 `rho` or a CAR_proper
# `rho_car`), or when no mode was attached (an unguessable axis elsewhere in
# the same grid, such as `rho_car`, declines the whole mode-find -- see
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


# Outer mode + covariance of a REGISTRY fit's grid -- the standalone
# `tulpa_nested_laplace()` counterpart of the joint path's placement mode, found
# by the same mode-find (`.nl_placement_mode()`) over the same generic tagging
# (`.joint_pareto_block_tags()`, read through `.nl_registry_axis_tags()`).
# `tags` is one transform tag per grid column; `refit_log_marginal(theta_mat)`
# re-evaluates the inner marginal at an arbitrary `[S x d]` theta matrix
# (columns named per `res$theta_names`) through the SAME kernel the fit used.
# Declines (NULL) when any axis in the grid has unguessable support -- e.g.
# car_proper's `rho`, the identical limitation the joint path already has for
# that family -- or when the mode-find reaches no usable curvature.
.nl_registry_axis_mode_cov <- function(res, tags, refit_log_marginal) {
    cn <- res$theta_names
    tg <- res$theta_grid
    if (is.null(cn) || is.null(tg)) return(NULL)
    if (!is.matrix(tg)) tg <- matrix(as.numeric(tg), ncol = 1L)
    colnames(tg) <- cn
    pm <- .nl_placement_mode(tg, res$weights, tags, function(theta_mat) {
        colnames(theta_mat) <- cn
        refit_log_marginal(theta_mat)
    })
    if (!is.null(pm$declined)) return(NULL)
    list(u_mode = pm$mode_u, cov = pm$cov_u, tags = pm$tags, col_names = cn)
}

# --- per-arm dispersion axes -------------------------------------------------
#
# A `phi_grid` axis is a hyperparameter of an ARM, not of a prior block, so it
# is in no entry of `.NL_REGISTRY_AXIS_FIELD` and the two sigma rescues above
# walk past it. Everything else it needs is already here: the transform registry
# tags a `phi_<arm>` column `"log"` (`.joint_pareto_block_tags()`), the outer
# mode/Hessian stencil varies it like any other column (`.joint_grids_from_cells()`
# hands `grids$phi_<arm>` straight back to `.joint_phi_grid_per_arm()`), and the
# node layout, clamp policy and decline vocabulary are the shared helpers above.
# So what this adds is a slot source and a write target, not a second placement
# machine (gcol33/tulpa#663).
#
# The axes are read off the FIT rather than off the argument: `phi_grid` accepts
# a named or a positional list and treats a length-1 entry as no axis at all, so
# the `phi_<arm>` columns the grid actually carries are the authority on which
# arms have one. The driver normalises the argument to the named form before the
# first fit, which is what lets the write target be `phi_grid[[arm]]`.
.nl_phi_axis_slots <- function(res, phi_grid) {
    cn <- colnames(res$theta_grid) %||% character(0)
    nm <- cn[startsWith(cn, "phi_")]
    if (!length(nm) || !is.list(phi_grid)) return(list())
    arms <- sub("^phi_", "", nm)
    keep <- arms %in% (names(phi_grid) %||% character(0))
    lapply(which(keep), function(i)
        list(arm = arms[i], axis = nm[i]))
}

# The provenance predicate for a dispersion axis, in the vocabulary
# `.nl_axis_hold()` answers in. That helper's third branch -- nodes equal to the
# engine's own default read as a default -- has no counterpart here ON PURPOSE:
# the engine has no default dispersion axis. An arm with no `phi_grid` entry
# carries the parse-time scalar `phi` and no axis at all, so every axis that
# exists was written by a caller, and the ONLY thing separating a wrapper's
# computed default from a user's pin is whether that caller said so with
# `auto_grid()`.
.nl_phi_axis_hold <- function(arm, auto_arms) {
    auto_arms <- .nl_auto_fields_at(auto_arms)
    i <- match(arm, names(auto_arms) %||% character(0))
    if (is.na(i)) return("axis_pinned")
    if (isTRUE(auto_arms[[i]])) return(NULL)
    "default_axis_pinned"
}

# Which arms of a `phi_grid` argument carry the `auto_grid()` marker and whether
# each let the pass place its axis, plus the argument with the markers removed.
# Read at the front door, BEFORE `.normalise_phi_grid()` -- that helper coerces
# each entry with `as.numeric()`, which drops the attributes the marker lives
# in. A positional list is keyed through `arm_names` so the record is by arm
# either way.
.nl_phi_provenance <- function(phi_grid, arm_names) {
    if (!is.list(phi_grid) || !length(phi_grid)) {
        return(list(phi_grid = phi_grid, auto = logical(0)))
    }
    nm <- names(phi_grid)
    keys <- if (!is.null(nm)) nm else
        arm_names[seq_len(min(length(phi_grid), length(arm_names)))]
    auto <- logical(0)
    for (k in seq_along(phi_grid)) {
        v <- phi_grid[[k]]
        if (is.null(v)) next            # `attr<-`(NULL, ...) DELETES the element
        if (k <= length(keys) && is_auto_grid(v)) {
            auto[keys[k]] <- auto_grid_place(v)
        }
        phi_grid[[k]] <- .nl_strip_auto(v)
    }
    list(phi_grid = phi_grid, auto = auto)
}

# Would a placement pass on `axes` fire on this fit? The `"resolve"` trigger,
# read off the weights the fit already stored, so asking costs nothing.
#
# The placement mode-find (`.joint_attach_placement()`) exists for the two
# sigma rescues, whose own trigger is a railed field SD -- so it only computes
# a mode and Hessian on such a grid. A dispersion axis is crossed onto the
# tensor independently of the field's geometry and fires on its OWN sizing, and
# `collapsed_interior` (weight concentrated, but the modal cell interior on every
# axis) is precisely the regime the reported case sat in: the field SD axis had
# been placed, the dispersion axis was 55 posterior SDs per cell, and no
# curvature had been computed for either. This is what the placement path asks
# to decide whether to compute one anyway.
.nl_placement_axis_wanted <- function(res, axes) {
    if (!length(axes)) return(FALSE)
    cn <- colnames(res$theta_grid) %||% character(0)
    for (a in intersect(axes, cn)) {
        if (.nl_axis_placement_fires(res, a, "log")) return(TRUE)
    }
    FALSE
}

# A default axis the placement pass could not bring off a boundary of its grid
# is reported off that endpoint, not off a mode. Said once per fit, naming each
# such axis, its side and the reason it stayed (gcol33/tulpa#919). An axis the
# caller pinned, declared as written, or held by switching the pass off is the
# caller's own statement; it stays on the fit (`outer_grid_railed_axes`,
# `outer_grid_axis_declined`) without a warning.
.NL_RAIL_CALLER_HOLDS <- c("axis_pinned", "default_axis_pinned",
                           "auto_recenter_disabled")

.nl_warn_unplaced_rail <- function(res, fn) {
    railed <- res$outer_grid_railed_axes
    dec    <- res$outer_grid_axis_declined
    if (!length(railed) || !length(dec)) return(invisible(res))
    parts <- strsplit(railed, ":", fixed = TRUE)
    hits  <- character(0)
    for (p in parts) {
        why <- dec[p[1L]]
        if (is.na(why) || why %in% .NL_RAIL_CALLER_HOLDS) next
        hits <- c(hits, sprintf("`%s` (%s edge: %s)", p[1L], p[2L], why))
    }
    if (length(hits)) {
        warning(sprintf(paste0(
            "%s: the outer posterior mode lies on the boundary of %s, which ",
            "placement could not move it off; the reported value is that grid ",
            "endpoint, not an estimate. See `outer_grid_axis_declined`."),
            fn, paste(hits, collapse = ", ")), call. = FALSE)
    }
    invisible(res)
}

# PER-AXIS decline record, beside the whole-fit `outer_grid_recenter_declined`.
# The whole-fit slot holds the reason from the ONE rescue that could have run
# and is written only while the fit is unplaced, so on a fit where the field SD
# axis moved and a dispersion axis did not it says `auto_recentered` and nothing
# about the axis that stayed. That is the fit whose own `grid_coarsest_axis`
# names the unmoved axis, so the reason it did not move has to survive the
# placement of a different one.
.nl_decline_axis <- function(res, axis, reason) {
    rec <- res$outer_grid_axis_declined %||% character(0)
    rec[axis] <- reason
    res$outer_grid_axis_declined <- rec
    res
}

# Which axes the registry rescue below can move, per family: the prior-list
# FIELD each axis lives on, mapped to that axis's bare name in
# `res$theta_names`. `.nl_axis_alias()` resolves the name against whatever the
# fit calls it (icar's `theta_grid` is a plain numeric vector, so
# `.joint_pareto_grid_regime()` coerces it to a 1-column matrix generically
# named `"theta"` while `theta_names` still says `"tau"`; a multi-block grid
# prefixes every axis `b<k>.`).
#
# Every axis a family lists is movable on its own. Naming ONE recentrable axis
# and carrying the family's other axis as a passenger, re-crossed unchanged,
# left BYM2's `rho_grid` detected against its
# 0.95 ceiling (`pareto_k_grid_edge_axes` names it) and then left there -- with
# the fit recording `grid_not_collapsed`, which is not what happened. Field
# order follows `.NL_FAMILY_AXES` (`R/settings.R`), the same order
# `.nl_fill_family_axes()` crosses a family's defaults in.
#
# MEMBERSHIP is decided by one question, not by hand: does
# `.joint_pareto_block_tags()` name a coordinate for EVERY axis of the family's
# grid? A recentred axis is laid in that coordinate (`.nl_recenter_axis()`) and
# the placement mode-find searches the whole grid in it (`.nl_placement_mode()`),
# so one unguessable axis takes the fit's curvature with it. The five registry
# families absent below are absent for a stated reason, and each records it on
# the fit rather than passing in silence:
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
# generic write over `.NL_REGISTRY_AXIS_FIELD`. A call site branching on
# `type == "icar"` / `"bym2"` by hand silently
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

# Standalone (non-joint) `tulpa_nested_laplace()` registry rescue -- the
# registry counterpart of the joint placement pass (`.joint_place_axes()`). Scope: every axis
# `.NL_REGISTRY_AXIS_FIELD` lists, on every block of the prior, each moved on
# its own rail in whichever coordinate the engine's transform registry gives it
# (`log` for a scale, `logit01` for the BYM2 mixing weight). The families it
# does NOT cover, and why each declines rather than passing in silence, are
# written out on that table.
#
# `multi = TRUE` says `prior` is a BLOCK LIST rather than one block, which is
# the only difference between the two `tulpa_nested_laplace()` paths here: the
# axes are then block-prefixed (`b<k>.tau`), provenance is read per block, and
# a moved block is re-crossed on its own fields while its neighbours keep
# theirs. A single-block prior is the length-1 case of the same walk -- there is
# one rescue, not one per path.
#
# One recenter attempt (not the joint path's two): the "runaway mode needs
# a regularizing prior" pathology the joint pass's second attempt on a field SD
# targets is specific to a donor/copy-coupled fit pushing toward
# near-separation; a standalone
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
    # allowed to, able to, or built to move them. It reads
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
    # move whatever the fit did, which is the answer the joint path gives and
    # the one a caller holding their own grid expects.
    held <- lapply(slots, function(s) .nl_axis_hold(
        blocks[[s$block]], s$field, .nl_auto_fields_at(auto, bidx(s)),
        type = s$type))
    pinned <- !vapply(held, is.null, logical(1))
    if (all(pinned)) {
        out$res <- .nl_decline_recenter(out$res, .nl_reduce_decline(held))
        return(out)
    }

    cur <- blocks
    attempt <- 0L
    reason  <- if (identical(policy, "resolve")) "grid_resolves_posterior" else
        "no_axis_railed"
    while (attempt < max_attempts) {
        # Every policy shares the placement mode-find, the `mode +/- span *
        # sd` node layout, the provenance gate and the attempt budget; they
        # differ only in WHEN the pass fires and, once it does, in HOW MANY of
        # the family's axes it re-places.
        #
        #   "rail"     fire on a railed axis, and move the railed axes alone --
        #              the placement half alone.
        #   "resolve"  fire on a railed OR an under-resolved axis, and move all
        #              of them: the stencil and the refit are paid once per fit,
        #              not once per axis, so the question the trigger asks is
        #              whether the FIT's grid is worth re-placing at all.
        #   "always"   fire unconditionally, and move all of them.
        railed <- vapply(slots, function(s) {
            .nl_axis_railed(res, s$axis, bidx(s), n_blocks)
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
            reason <- .nl_reduce_decline(held[sel])
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
