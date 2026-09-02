# Engine settings: the single source of truth for every default the
# nested-Laplace machinery lays down on its own.
#
# WHY THIS FILE EXISTS. A default outer hyperparameter axis used to be written
# where it was consumed, so the same numbers appeared in several places: the
# field-SD axis `exp(seq(log(0.1), log(3), length.out = 5))` in five (three
# joint backends, the multi-block copy-block builder, the bym2/iid registry
# entries), the copy-coefficient axis `c(0, exp(seq(log(0.1), log(3), ...)))`
# verbatim in two, the wide precision axis in three, `k_samples = 200L` in five,
# the reported Pareto-k usable threshold `0.7` in seven. Copy-pasted defaults
# drift: one site gets tuned and the others silently do not, and nothing in the
# package can then state what its own default IS -- which is precisely how
# a consumer-stamped engine default stayed invisible (the auto-recenter had to
# recognise the engine's own default axis coming back in through a consumer's
# prior, and the only way to recognise it was to have ONE place that defines
# it).
#
# The outer-k draw budget is the worked example of the drift outrunning the
# fix (gcol33/tulpa#632). It was consolidated here as `k_samples = 200L` and
# three of the four backends were swept onto the accessor, but the joint path
# had renamed its own variable to `diagnose_draws` and raised the value to
# `500L` (gcol33/tulpa#127), so it matched neither the name nor the number the
# sweep and its lint were looking for and kept a second default for four
# releases -- one that banded fits differently, since the budget fixes the PSIS
# tail fraction. A source lint keyed to a LITERAL only catches the copies that
# have not been renamed yet; `test-settings.R` therefore keys this family on
# the concept and separately evaluates each entry point's default against the
# table.
#
# THE RULE. A number that answers "what does the engine do when the user says
# nothing?" lives here and nowhere else. Consumers of a default call the
# accessor; they never restate the number. Everything here is a plain value or a
# pure function of the table, so `.nl_grid_axis("field_sd")` is the same vector
# at every call site by construction rather than by review.
#
# WHAT IS *NOT* HERE. Numbers that are not defaults: mathematical constants,
# convergence tolerances local to one algorithm, and the internal loop-control
# thresholds of the Pareto-k proposal search (`.K_DIAG_MM_MAX`,
# `.K_DIAG_GOOD`, `.K_DIAG_MIX_BW`, ... in `R/nested_laplace_joint_pareto_k.R`),
# which are documented at length where the loop that reads them lives and are
# each consumed in exactly one file. The REPORTED reliability threshold is the
# exception: it is user-facing and was duplicated across the diagnostics,
# plotting, k-fold and PSIS-LOO surfaces as well as that loop's early stop, so
# every one of them now reads `.nl_diag("k_usable")` from here.

# --- outer hyperparameter grid axes ------------------------------------------
#
# One entry per DISTINCT default axis, keyed by a name that says what the axis
# measures rather than which family happens to use it (`field_sd`, not
# `bym2_sigma`) -- families that share an axis share the entry, so tuning it
# moves every one of them together, which is the intent whenever two families
# integrate the same quantity over the same support.
#
# Each entry is one of:
#   * `list(lo =, hi =, n =)`      geometric (log-spaced) axis over [lo, hi].
#   * `list(nodes =)`              explicit nodes (a bounded axis: correlations,
#                                  mixing weights -- placed at interpretable
#                                  values, not mechanically spaced).
#   * `list(..., prepend =)`       nodes prefixed to the generated axis (the
#                                  copy coefficient's exact 0, so the "no copy"
#                                  base model carries posterior mass).
#   * `list(data_dependent = TRUE, ...)`  an axis the engine can only build once
#                                  it has seen the data or the user's prior
#                                  (a PC-prior mode, an adjacency eigenvalue
#                                  interval, a `tgmrf` bounds box). The shape
#                                  parameters live here; the materialisation
#                                  needs arguments, so `.nl_grid_axis()` refuses
#                                  these and the caller reads the fields it
#                                  needs (`.nl_grid_par()`).
.NL_GRID <- list(
    # Field amplitude (SD) of an areal / iid latent block. The engine's most
    # widely shared axis: bym2 and iid in the registry, all three single-block
    # joint areal backends, and every multi-block copy block's donor axis.
    field_sd       = list(lo = 0.1,  hi = 3,    n = 5L),

    # Intrinsic-GMRF precision (icar / rw1 / rw2). Wide (two decades) and
    # 9-node because a precision posterior on an intrinsic field is poorly
    # located a priori -- there is no scale in the prior to anchor it.
    gmrf_tau       = list(lo = 0.3,  hi = 30,   n = 9L),

    # Proper-CAR precision: the same decade span at 5 nodes, because the axis
    # is crossed with a correlation axis (a 9 x 5 tensor is 45 inner solves).
    car_tau        = list(lo = 0.3,  hi = 30,   n = 5L),

    # AR1 precision: narrower than the intrinsic axis -- a stationary AR1 field
    # has a finite marginal variance, so the plausible precision range is
    # tighter -- and crossed with `ar1_rho`.
    ar1_tau        = list(lo = 0.5,  hi = 20,   n = 5L),

    # GP marginal variance and lengthscale, shared by nngp and hsgp (the same
    # Matern covariance under two approximations, so the same default support).
    gp_var         = list(lo = 0.05, hi = 2,    n = 5L),
    gp_lengthscale = list(lo = 0.05, hi = 1.5,  n = 5L),

    # Multi-output HSGP: 4 axes cross into one tensor, so each is coarse (3
    # nodes) and narrow -- 3^4 = 81 cells at these settings already.
    mo_field_sd    = list(lo = 0.3,  hi = 1.5,  n = 3L),
    mo_lengthscale = list(lo = 0.1,  hi = 1.0,  n = 3L),

    # Copy coefficient (the scale at which one arm's field is transferred to
    # another). Exact 0 is prepended so the no-transfer base model is IN the
    # grid rather than a limit of it.
    copy_alpha     = list(lo = 0.1,  hi = 3,    n = 5L, prepend = 0),

    # BYM2 mixing weight: proportion of the field variance that is spatially
    # structured. Nodes placed at interpretable proportions, weighted toward
    # the structured end where areal data usually sits.
    #
    # The span reaches 0.999 because a BOUNDED axis's SPAN and its RESOLUTION
    # are not interchangeable: the `outside = "extend"` read
    # mirrors the outer cell edge in the axis's own logit coordinate, so an axis
    # topping out at 0.95 cannot report an upper bound above
    # plogis(logit(0.95) + 0.5 (logit(0.95) - logit(0.8))) = 0.97642 whatever
    # the data say, and no node count moves that. Measured over 800 fits per
    # candidate set, four fixed truths x 200 seeds: at rho = 0.99 the 4-node
    # axis covers 0.555 at nominal 0.95 and only because the placement rescue
    # moved 55.5% of them, against 0.955 here with the rescue never firing.
    # Mean |coverage - nominal| over the four truths goes 0.140 -> 0.011 at the
    # 95% level, 0.189 -> 0.088 at 80% and 0.081 -> 0.099 at 50%, mean |median
    # bias| 0.058 -> 0.041, at a mean 95% width of 0.494 against 0.487 -- so the
    # span is not bought with wider intervals.
    bym2_rho       = list(nodes = c(0.2, 0.5, 0.8, 0.95, 0.99, 0.999)),

    # AR1 autocorrelation: nodes concentrated near 1, where the temporal
    # likelihood changes fastest.
    ar1_rho        = list(nodes = c(0.0, 0.4, 0.7, 0.9, 0.97)),

    # Multi-output cross-field correlation: symmetric about 0 (no sign is
    # expected a priori), coarse for the same tensor-size reason as above.
    mo_rho         = list(nodes = c(-0.4, 0.0, 0.4)),

    # MCAR / miid free Sigma, p = 2: nodes are placed in the interpretable
    # (sigma_1, sigma_2, rho) space and converted to log-Cholesky, so the
    # correlation axis is covered evenly (see `.mcar_default_logchol_grid()`).
    mcar_sd        = list(nodes = c(0.4, 0.7, 1.1, 1.7)),
    mcar_rho       = list(nodes = c(-0.8, -0.4, 0.0, 0.4, 0.7, 0.9)),

    # MCAR / miid free Sigma, p > 2: a raw log-Cholesky tensor (the individual
    # off-diagonal coordinates are not separately interpretable at p > 2).
    # `mcar_logchol_diag` nodes are SDs, logged by the builder; the
    # off-diagonal nodes are already on the log-Cholesky scale.
    mcar_logchol_diag = list(nodes = c(0.4, 0.8, 1.5, 2.5)),
    mcar_logchol_off  = list(nodes = c(-1.2, 0.0, 1.2)),

    # Proper-CAR correlation. Support is the adjacency eigenvalue interval, so
    # the axis is built from `rho_bounds`: `margin` is the fraction of the
    # interval width held back from each endpoint (Q is singular AT the
    # boundary, so a node there returns a NaN log-determinant).
    car_rho = list(data_dependent = TRUE, n = 5L, margin = 0.05,
                   bounds = c(0, 1)),

    # The same correlation on the JOINT drivers, which do not compute
    # `rho_bounds` and so lay fixed nodes on (0, 1) instead of the eigenvalue
    # interval. A separate entry rather than a shape of `car_rho`, because the
    # two axes are built from different information; it is here so the single-
    # block and multi-block joint backends read one binding rather than each
    # restating the nodes. Deliberately NOT bound to a
    # `.NL_FAMILY_AXES` field: `.nl_axis_matches_default()` reads that table,
    # and binding it would silently reclassify a caller's identical nodes from
    # a pin to a default.
    joint_car_rho = list(nodes = c(0.5, 0.8, 0.95, 0.99)),

    # SPDE range / marginal SD on the registry (multi-block) path. Centred on
    # the PC-prior mode and deliberately TIGHT (`mode / span` .. `mode * span`):
    # a wide rectangular Cartesian grid with no mode-find runs into a
    # small-range / large-sigma corner where a binary-occupancy field over-fits,
    # and that corner -- carrying the highest inner likelihood -- then dominates
    # the weighted field even though the PC prior disfavours it.
    spde_registry = list(data_dependent = TRUE, n = 5L, span = 1.4,
                         prior_range = c(1, 0.5), prior_sigma = c(1, 0.5)),

    # SPDE range / sigma on the `fit_spde(method = "grid")` path, which scores
    # the PC prior explicitly at every cell and so can afford the wide window
    # (`mode * lo` .. `mode * hi`) the registry path avoids.
    spde_direct = list(data_dependent = TRUE, lo_mult = 0.3, hi_mult = 3),

    # User-defined GMRF (`tgmrf`): per-axis nodes over the declared bounds
    # box, or `init +/- half_width` when the block declares no bounds.
    tgmrf_axis = list(data_dependent = TRUE, n = 5L, half_width = 2)
)

# Materialise a default axis. Geometric when `lo`/`hi` are present, explicit
# nodes otherwise, with `prepend` nodes in front. Errors on a data-dependent
# entry -- those need arguments the table cannot hold, so their callers read
# the shape parameters with `.nl_grid_par()` and build the axis themselves.
# `n` re-reads the declared axis at a different RESOLUTION: same `lo` / `hi`
# and the same `prepend` atom, more nodes between them. It is the knob that
# separates "integrate this more accurately" from "integrate something else"
# (gcol33/tulpa#633), which matters wherever an axis carries prior structure a
# caller must not displace -- `copy_alpha` is the atom at 0 plus the slab over
# [0.1, 3], and replacing it with a raw numeric grid changes what is being
# integrated rather than how well. `n` counts the SLAB nodes, so the returned
# length is `n + length(prepend)`. An axis declared as explicit `nodes` has no
# resolution to vary and refuses.
.nl_grid_axis <- function(key, n = NULL) {
    spec <- .NL_GRID[[key]]
    if (is.null(spec)) {
        stop("Unknown default grid axis '", key, "'. Known: ",
             paste(names(.NL_GRID), collapse = ", "), ".", call. = FALSE)
    }
    if (isTRUE(spec$data_dependent)) {
        stop("Default grid axis '", key, "' is data-dependent; read its shape ",
             "with .nl_grid_par() and build the axis at the call site.",
             call. = FALSE)
    }
    if (!is.null(n)) {
        n <- suppressWarnings(as.integer(n))
        if (length(n) != 1L || is.na(n) || n < 1L) {
            stop("Grid axis resolution `n` must be a single integer >= 1.",
                 call. = FALSE)
        }
        if (!is.null(spec$nodes)) {
            stop("Default grid axis '", key, "' is declared as explicit nodes, ",
                 "so it has no resolution to raise.", call. = FALSE)
        }
    }
    ax <- if (!is.null(spec$nodes)) {
        as.numeric(spec$nodes)
    } else {
        exp(seq(log(spec$lo), log(spec$hi),
                length.out = as.integer(n %||% spec$n)))
    }
    if (!is.null(spec$prepend)) ax <- c(as.numeric(spec$prepend), ax)
    ax
}

# One shape parameter of a default axis (`n`, `span`, `margin`, ...). The read
# path for a data-dependent axis, and for a caller that needs the node count
# without the nodes.
.nl_grid_par <- function(key, par, default = NULL) {
    spec <- .NL_GRID[[key]]
    if (is.null(spec)) {
        stop("Unknown default grid axis '", key, "'.", call. = FALSE)
    }
    spec[[par]] %||% default
}

# --- which axis each family defaults ------------------------------------------
#
# The binding from a prior block's TYPE and grid FIELD to the axis above that
# the engine lays on it when the field is absent. Read by two layers that must
# never disagree:
#
#   * `.NL_REGISTRY`'s per-family `defaults()` closures (`R/nested_laplace.R`),
#     which fill the field in;
#   * axis PROVENANCE (`.nl_axis_matches_default()`,
#     `R/nested_laplace_auto_grid.R`), which has to recognise the engine's own
#     default coming back in through a caller's prior and treat it as a default
#     rather than a user pin.
#
# Before this table the second layer carried a hand-maintained list of two
# fields (`sigma_grid`, `tau_grid`) and could not see any of the others; now
# both layers read the same binding, so a new family is one entry here and is
# covered by both.
.NL_FAMILY_AXES <- list(
    icar       = list(tau_grid = "gmrf_tau"),
    rw1        = list(tau_grid = "gmrf_tau"),
    rw2        = list(tau_grid = "gmrf_tau"),
    car_proper = list(tau_grid = "car_tau", rho_grid = "car_rho"),
    bym2       = list(sigma_grid = "field_sd", rho_grid = "bym2_rho"),
    ar1        = list(tau_grid = "ar1_tau", rho_grid = "ar1_rho"),
    iid        = list(sigma_grid = "field_sd"),
    nngp       = list(sigma2_grid = "gp_var", phi_gp_grid = "gp_lengthscale"),
    hsgp       = list(sigma2_grid = "gp_var",
                      lengthscale_grid = "gp_lengthscale"),
    hsgp_mo    = list(sigma_1_grid = "mo_field_sd",
                      sigma_2_grid = "mo_field_sd",
                      rho_grid = "mo_rho",
                      lengthscale_grid = "mo_lengthscale"),
    mcar       = list(logchol_grid = "mcar_logchol_diag"),
    miid       = list(logchol_grid = "mcar_logchol_diag"),
    spde       = list(range_grid = "spde_registry",
                      sigma_grid = "spde_registry"),
    tgmrf      = list(theta_grid_built = "tgmrf_axis"),

    # PATH pseudo-types (leading dot, so they cannot collide with a prior
    # `type`). The same family reaches the outer grid through more than one
    # driver, and the drivers do not always parameterize it the same way, so the
    # binding is per PATH-and-family rather than per family alone:
    #
    #   * `.joint_areal` -- the single-block joint backends
    #     (`R/nested_laplace_joint_backends.R`) integrate icar / bym2 /
    #     car_proper over the field SD, where the registry integrates icar and
    #     car_proper over the PRECISION. A rescue on the joint path must compare
    #     against `field_sd`, not against that family's registry axis.
    #   * `.copy` -- the donor SD axis and copy coefficient the multi-block
    #     driver defaults on a copy block (`.joint_block_axis_grid()`), which is
    #     the same convention whatever the block's own family is.
    .joint_areal = list(sigma_grid = "field_sd"),
    .copy        = list(sigma_grid = "field_sd", alpha_grid = "copy_alpha")
)

# --- which grid fields each PATH reads ----------------------------------------
#
# The binding above says which axis the engine LAYS DOWN on a field. This says
# which fields a resolved path READS, which is the other half of the same
# question and the half a caller gets wrong: a family whose paths parameterize
# it differently accepts one spelling on one driver and ignores it on another
# (a `sigma_grid` on an icar block reached the multi-block
# driver, which integrates `tau_grid`, and neither took effect nor said so).
#
# `.NL_PATH_AXES[[path]][[family]]` is the COMPLETE set of grid fields that
# (path, family) reads. A leading-dot token expands to the field names of a
# binding above: `.registry` is the family's own entry, anything else is the
# path pseudo-type of the same name. An absent entry means the path reads
# exactly the family's registry binding, so a family whose paths agree needs no
# entry here at all and a family whose paths differ is one line.
#
# Paths:
#   registry     `.nl_dispatch()` / `.nl_block_axis_grid()` -- every
#                `tulpa_nested_laplace()` fit and every non-copy block of a
#                multi-block joint fit.
#   joint_single the single-block joint areal backends
#                (`R/nested_laplace_joint_backends.R`), which read the field SD.
#   copy         a copy block on the joint multi-block path
#                (`.joint_block_axis_grid()`), which leads with (sigma, alpha).
.NL_PATH_AXES <- list(
    registry = list(
        # car_proper's `defaults()` accepts the joint-API spelling of the
        # correlation axis as an alias, so the registry path reads both.
        car_proper = c(".registry", "rho_car_grid")
    ),
    joint_single = list(
        icar       = ".joint_areal",
        bym2       = c(".joint_areal", "rho_grid"),
        car_proper = c(".joint_areal", "rho_car_grid")
    ),
    copy = list(
        icar       = ".copy",
        rw1        = ".copy",
        rw2        = ".copy",
        iid        = ".copy",
        bym2       = c(".copy", "rho_grid"),
        car_proper = c(".copy", "rho_car_grid"),
        ar1        = c(".copy", "rho_grid"),
        # A copied correlated field keeps its own log-Cholesky axes and appends
        # the copy coefficient; there is no scalar sigma on this one.
        mcar       = c(".registry", "alpha_grid"),
        miid       = c(".registry", "alpha_grid")
    )
)

# Whether a path CROSSES the axis fields above itself, or takes them already
# paired. The registry stores a family's axes PRE-PAIRED -- `.nl_fill_family_axes()`
# expands them into one row per tuple and every field then has length n_cells --
# while the joint backends (`.joint_cartesian()`) and the copy block
# (`.joint_block_axis_grid()`'s `expand.grid`) take one axis per field and cross
# them. The difference is invisible in the field values: two length-20 vectors
# are a 20-cell paired grid on one path and a 400-cell crossed one on another.
# Anything rewriting a field per axis has to know which (`.nl_pilot_block()`),
# so the property is declared beside the fields rather than inferred at each
# call site.
.NL_PATH_CROSSES <- c(registry = FALSE, joint_single = TRUE, copy = TRUE)

.nl_path_crosses <- function(path) {
    path <- as.character(path %||% "")
    if (!path %in% names(.NL_PATH_CROSSES)) return(FALSE)
    isTRUE(unname(.NL_PATH_CROSSES[[path]]))
}

# The default-axis KEY a (path, family) lays on `field` -- the read half of
# `.nl_path_axis_fields()`, which names the fields but not where their defaults
# come from. Path pseudo-types are consulted in the order the path declares
# them, then the family's own binding, so a field a path re-parameterizes
# (`.joint_areal`'s `sigma_grid`) resolves to the path's axis and not the
# family's. NULL when no table binds the field.
.nl_path_axis_key <- function(type, path = "registry", field) {
    type <- tolower(type %||% "")
    spec <- .NL_PATH_AXES[[path]][[type]]
    toks <- if (is.null(spec)) character(0) else
        vapply(spec, function(tok) if (identical(tok, ".registry")) type else tok,
               character(1))
    for (tk in unique(c(toks, type))) {
        if (!tk %in% names(.NL_FAMILY_AXES)) next
        k <- .nl_family_axis_key(tk, field)
        if (!is.null(k)) return(k)
    }
    NULL
}

# Exact conversions between two spellings of ONE axis, for the paths that
# parameterize a family differently. Only pairs the ENGINE ITSELF converts are
# listed: `.joint_call_kernel_via_multi()`
# (`R/nested_laplace_joint_backends.R`) hands the single-block joint icar
# kernel `b1.tau = 1 / sigma^2` from its own `sigma` axis, so the two spellings
# name the same set of physical grids and either can be written as the other.
# A family whose conversion is not established in engine code is absent, and
# its unread axis is refused with the consumed axis named and no conversion
# offered -- a guessed relation would be worse than none.
.NL_AXIS_EQUIV <- list(
    icar = list(
        sigma_grid = c(tau_grid   = "tau_grid = 1 / sigma_grid^2"),
        tau_grid   = c(sigma_grid = "sigma_grid = 1 / sqrt(tau_grid)")
    )
)

# The complete field set `path` reads on a block of `type`.
.nl_path_axis_fields <- function(type, path = "registry") {
    type <- tolower(type %||% "")
    base <- names(.NL_FAMILY_AXES[[type]]) %||% character(0)
    spec <- .NL_PATH_AXES[[path]][[type]]
    if (is.null(spec)) return(base)
    out <- lapply(spec, function(tok) {
        if (!startsWith(tok, ".")) return(tok)
        if (identical(tok, ".registry")) return(base)
        names(.NL_FAMILY_AXES[[tok]]) %||% character(0)
    })
    unique(unlist(out, use.names = FALSE))
}

# Every field name any family or path binds -- the set a stray `*_grid` on a
# block is checked against, so a field the tables know nothing about (a
# materialised per-grid payload, a consumer's own bookkeeping) is left alone.
.nl_known_axis_fields <- function() {
    fromtab <- unlist(lapply(.NL_FAMILY_AXES, names), use.names = FALSE)
    frompath <- unlist(.NL_PATH_AXES, use.names = FALSE)
    frompath <- frompath[!startsWith(frompath, ".")]
    unique(c(fromtab, frompath))
}

# Fill in a prior block's absent default axes from the family binding. When any
# of `fields` is missing, ALL of them are rebuilt and crossed as a Cartesian
# product -- the registry's per-family convention: a family's axes are stored
# pre-paired (one row of `theta_grid` per tuple), so a partially-supplied set
# cannot be honoured by pairing a user axis of length 4 against a default of
# length 5. `fields` defaults to every field the family binds, in table order.
#
# This is the materialisation half of `.NL_FAMILY_AXES`: with it, a registry
# family that defaults a plain Cartesian grid is one line in its `defaults()`
# closure and one entry in the binding, so adding a family cannot leave the
# provenance layer behind.
.nl_fill_family_axes <- function(p, type, fields = NULL) {
    fam <- .NL_FAMILY_AXES[[tolower(type %||% "")]]
    if (is.null(fam)) return(p)
    if (is.null(fields)) fields <- names(fam)
    if (!any(vapply(fields, function(f) is.null(p[[f]]), logical(1)))) return(p)
    axes <- lapply(fields, function(f) .nl_grid_axis(fam[[f]]))
    names(axes) <- fields
    gr <- expand.grid(axes, KEEP.OUT.ATTRS = FALSE)
    for (f in fields) p[[f]] <- as.numeric(gr[[f]])
    p
}

# The axis key a family defaults on a field, or NULL when it defaults none.
.nl_family_axis_key <- function(type, field) {
    fam <- .NL_FAMILY_AXES[[tolower(type %||% "")]]
    if (is.null(fam)) return(NULL)
    fam[[field]]
}

# Every axis key the engine can default onto `field`, across all families.
# The fallback for provenance when the block's type is not known at the call
# site.
.nl_field_axis_keys <- function(field) {
    keys <- unlist(lapply(.NL_FAMILY_AXES, function(fam) fam[[field]]),
                   use.names = FALSE)
    unique(keys)
}

# --- outer-grid auto-recentering ---------------------------------------------
#
# Policy for the mode-Hessian recenter of a railed default axis
# (see `R/nested_laplace_auto_grid.R`).
.NL_RECENTER <- list(
    # Nodes in a recentred axis, and how many mode-SDs it spans either side.
    n_pts     = 5L,
    span      = 2.5,

    # The mode SD is clamped into [min_sd_u, max_sd_u] on the log axis: a floor
    # so a razor-sharp local curvature does not collapse the new grid to
    # near-duplicate nodes (the point of recentring is to BRACKET the mode with
    # real spread), a ceiling so a near-flat direction does not fling nodes to
    # implausible extremes.
    #
    # Both values were swept and both are KEPT. The ceiling
    # is close to inert: over 268 axis reads it binds on 2, on neither of the
    # well-identified families, and across a ladder spanning a factor of 15
    # (0.4 to 6) the summed coverage deviation moves 0.2843 to 0.3071 and the
    # mean 95% width 1.986 to 2.120 -- 3 is the best rung on calibration.
    # Fixtures built to REACH it (an `iid` design shrunk until 27.5% of raw mode
    # SDs pass 3) return a NON-MONOTONE coverage response, and the reason is not
    # the cap: the reported bound lies outside the node range on ~89% of those
    # fits, identically at 0.8 / 1.5 / 2 / 3, so what moves across that ladder
    # is where the outer-cell extrapolation lands.
    #
    # SETTLED, and the fixture did not have to be rebuilt --
    # the LEVEL was the problem. Over 48 (cap, span, n_pts, clamp policy) rungs
    # on two ceiling-reaching fixtures at 200 seeds, the reported bound leaves
    # the node range on 56-90% of fits at nominal 0.95 at EVERY setting, and on
    # 0% at nominal 0.50. So 0.50 is the level whose bound the design actually
    # supports, and it is the one to score this on; 0.95 measures the `extend`
    # rule. At 0.50:
    #
    #   * at `n_pts = 9` the cap is EXACTLY inert -- 1.5, 3 and 6 give identical
    #     coverage to three decimals in 7 of 8 (fixture, policy, span) cells;
    #   * at the shipped `n_pts = 5`, 3 is nearer nominal than 1.5 at
    #     `span = 4` (0.450 / 0.415 against 0.370) and equal at `span = 2.5`,
    #     while 6 over-covers on a doubled width (0.520 at width 1.264 against
    #     3's 0.450 at 0.790);
    #   * under the SHIPPED `sd_clamp_policy = "decline"` a lower ceiling is not
    #     free: dropping to 1.5 abandons the placement on 32-39% of fits
    #     (recentred 0.875 -> 0.680 and 0.835 -> 0.615) and buys nothing at the
    #     level that measures it.
    #
    # So 3 is kept on evidence rather than inertia, and `span` / `n_pts` are
    # kept with it. What the sweep DID find is that the outer-cell `"extend"`
    # read, not this cap, is what a diffuse axis's 95% bound is governed by --
    # recorded per axis on every fit as `theta_ci_outside_nodes`.
    # Evidence: `dev_notes/issue390/RESULTS390.md`.
    #
    # The earlier reading that the ceiling produces 95% widths in the hundreds
    # came from the one row that reaches it, `nngp_120`, whose fits are not
    # reproducible (72 of 120 differ between two passes of the
    # same seeds in one process). The floor's own ladder is on
    # `sd_floor_policy` below.
    min_sd_u  = 0.15,
    max_sd_u  = 3,

    # What the pass DOES when that ceiling binds.
    #
    # A clamp is not a spread the stencil measured -- it is the stencil failing
    # to resolve a direction, with a number substituted for what it could not
    # read. Laying an axis from the substitute states a spread the fit does not
    # have: on a `log` axis `mode +/- span * max_sd_u` is `exp(+/- 7.5)`, a
    # factor of 1808 either side of the mode, and the reported interval is read
    # off that span.
    #
    #   "clamp"    lay the axis from the clamped SD (what shipped through
    #              0.0.186).
    #   "decline"  keep the incoming span and record
    #              `outer_grid_recenter_declined = "sd_ceiling_unresolved"` --
    #              a placement the engine declines to make has to say so
    #              rather than be indistinguishable from one that was not
    #              needed.
    #   "relative" cap the recentred span by the INCOMING axis's own span in the
    #              same coordinate, so a direction the stencil could not resolve
    #              re-places within the range the caller's own grid already
    #              covered instead of past it.
    #
    # MEASURED (`dev_notes/issue387/analyse_policy387.R`), 200
    # fixed-truth seeds on each of six configurations x two placement policies,
    # arms paired seed by seed and differing only in this setting. Summed
    # |coverage - nominal| over nominal 0.95 / 0.80 / 0.50 at the shipped
    # placement, and the paired 95%-level win/loss against "clamp":
    #
    #   ceiling policy   summed dev   changed   won   lost   width ratio
    #   clamp                0.1464         -     -      -            -
    #   decline              0.1393        35     7      0       1.0000
    #   relative             0.1536        35     1      7       0.9983
    #
    # "decline" is the default because it never loses a trial: of the 35 it
    # changes it improves 7 and worsens none (sign test p = 0.0078) at the same
    # width. It is also the only arm that does not report a substituted spread
    # as though the stencil had measured it.
    sd_clamp_policy = "decline",

    # The floor is the OPPOSITE answer, and it is the same table that says so:
    #
    #   floor            summed dev   won   lost
    #   0.02 / 0.05          0.7600     0    374
    #   0.15 (this)          0.1464     -      -
    #   0.30                 0.2621     0     30
    #   0.50                 0.4893    39     47
    #   decline              0.3186     9     22
    #
    # A clamped floor WIDENS a too-narrow axis, which is the direction that
    # cannot rail, so substituting there is the right move and declining costs
    # 22 trials against 9. 0.15 is a minimum of the ladder in both directions --
    # dropping it to 0.05 loses 374 trials and wins none -- and the floor is the
    # bound that actually binds: it engages on 3 of 7 rows and on every fit of
    # those rows, where the ceiling reaches 2 of 268 axis reads.
    sd_floor_policy = "clamp",

    # A recentred axis must survive the map back onto its own support with at
    # least this many distinct nodes; a `logit01` axis whose nodes saturate to a
    # boundary in double precision loses some, and a two-node axis carries no
    # interior cell for the mode to sit in.
    min_nodes = 3L,

    # The `h / sd` above which an axis counts as UNDER-RESOLVED and is worth
    # re-placing even though it contains its own mode -- the trigger of the
    # default placement policy (`.nl_recenter_mode()` `"resolve"`).
    # `h / sd` is the median node spacing in the axis's own
    # unconstraining coordinate over the marginal SD in that coordinate, read
    # off the weights the fit already stored, so the test itself costs nothing.
    #
    # 2, and it is measured rather than picked. A recentred axis is `mode +/-
    # span * sd` over `n_pts` nodes, so its own `h / sd` is 1.25 by
    # construction: the threshold is the factor by which re-placing has to
    # improve the resolution before it is worth a second grid solve, and 2 /
    # 1.25 = 1.6. Over 200 fixed-truth seeds on each of six configurations
    # (`dev_notes/issue361/ext361.R`, 2400 fits per arm) a family of thresholds
    # was scored against the shipped rail-only policy and against
    # unconditional re-placement, mean |coverage - nominal| over the eight
    # (configuration, axis) rows:
    #
    #   threshold   fires   95%     80%     50%    width   |bias|   cost
    #   rail only   0.020   0.0425  0.1713  0.2431  1.000   1.000   1.00
    #   1.5         0.969   0.0294  0.0750  0.1713  0.629   0.787   1.95
    #   2           0.895   0.0300  0.0844  0.1294  0.630   0.763   1.71
    #   2.5         0.670   0.0338  0.1025  0.1525  0.730   0.798   1.43
    #   4           0.505   0.0338  0.1213  0.1656  0.801   0.839   1.29
    #   always      0.996   0.0294  0.0725  0.1856  0.725   0.849   2.04
    #
    # 2 minimizes the 50% deviation and the median bias, sits 0.0006 off the
    # best 95% deviation, and costs 1.71x against unconditional re-placement's
    # 2.04x. Width and bias are means of per-row ratios to the rail-only arm,
    # so both are BELOW 1: the policy narrows intervals rather than buying
    # coverage with them. The spread across thresholds is one configuration --
    # the only one whose default axes already resolve their posterior -- so the
    # table separates "fire on a coarse grid" from "fire always" on cost more
    # cleanly than on calibration; see `.nl_recenter_mode()`.
    resolve_mult = 2,

    # How far above a FLAT marginal the boundary node's own weight has to sit
    # before the axis counts as railed against that boundary
    # (`.nl_axis_rail()`). A marginal maximal at a boundary node has its mode
    # at or beyond it, which is the statement; this is the materiality guard that
    # keeps a merely uneven -- or numerically flat -- marginal from being moved
    # onto curvature it does not have.
    #
    # The comparison is against `1 / m` for an `m`-node axis, not against a
    # constant share. A share is not comparable across node counts: the same
    # posterior read at more nodes spreads its weight over more of them, so a
    # fixed share makes a LONGER axis a WEAKER detector: the shipped 0.5 share
    # fires on 12 / 10 / 6 / 3 / 0 of the same 20 railed fits as the span's
    # node count goes 4 / 5 / 6 / 8 / 12. Relative
    # to uniform the same reads are 12 / 11 / 10 / 10 / 10, against 12 / 11 / 10
    # / 10 / 10 fits whose marginal is maximal at the top node at all.
    #
    # `2` is the retired 0.5 share at the four nodes it was tuned on, so the
    # calibration is transported rather than re-chosen, and both
    # railed configurations still clear it (4.000 on the 144-cell ICAR lattice's
    # `tau`, 2.511 on the 100-region BYM2's `rho`).
    edge_mass_mult = 2,

    # Recenter attempts. The joint paths take two (the second adds the light PC
    # prior below, for a genuinely unidentified near-separation mode that keeps
    # running); the standalone registry path takes one -- that pathology is
    # specific to a donor/copy-coupled fit, and a single-response fit has no
    # such coupling, so a geometry-only recenter is proportionate there.
    max_attempts_joint    = 2L,
    max_attempts_registry = 1L,

    # Weakly-informative PC(U, alpha) prior engaged only on a second attempt,
    # and only when the user set no `prior_sigma` of their own. `U = 3` is the
    # retired fixed-grid ceiling, so the shrinkage is felt only PAST where the
    # old default axis already stopped; `P(sigma > 3) = 0.01` leaves a
    # data-identified mode essentially untouched.
    sigma_pc_prior = list("pc.prec", c(U = 3, alpha = 0.01)),

    # Per-axis node count of a PLACEMENT PILOT (`control$recenter_pilot = TRUE`,
    # `R/nested_laplace_pilot.R`). Placement reads two things off a grid -- the
    # argmax cell and the FD curvature at it -- and neither needs the resolution
    # the INTEGRATION needs, so a pilot answers the placement question on a
    # thinned grid over the same spans and the full grid is solved once, at the
    # placed axes. Three is the floor that still carries an interior node on
    # every axis, which is what the rail test reads; a coarser pilot would make
    # every node a boundary.
    pilot_n = 3L
)

# `options(tulpa.recenter.<par> = )` overrides one entry, the same seam the
# `tulpa.kdiag.*` diagnostics knobs use. It exists so a placement policy can be
# SWEPT -- one build, several arms paired on the same seeds -- rather than
# measured across builds that differ in more than the setting under test.
.nl_recenter <- function(par) {
    if (!par %in% names(.NL_RECENTER)) {
        stop("Unknown recenter setting '", par, "'.", call. = FALSE)
    }
    ov <- getOption(paste0("tulpa.recenter.", par), NULL)
    if (!is.null(ov)) return(ov)
    .NL_RECENTER[[par]]
}

# --- spatiotemporal driver grid ----------------------------------------------
#
# `fit_st_nested()`'s own default grid (its `control` knobs override each
# entry). Coarser per axis than the single-field defaults above because the
# spatial precision, temporal precision and (ar1) autocorrelation cross into
# ONE tensor: 4 x 4 x 3 = 48 inner solves at these settings.
.NL_ST_GRID <- list(
    n_spatial  = 4L,
    n_temporal = 4L,
    n_rho      = 3L,
    tau_lower  = 0.25,
    tau_upper  = 16,
    rho_lower  = 0.1,
    rho_upper  = 0.9
)

.nl_st_default <- function(par) {
    if (!par %in% names(.NL_ST_GRID)) {
        stop("Unknown spatiotemporal grid setting '", par, "'.", call. = FALSE)
    }
    .NL_ST_GRID[[par]]
}

# --- diagnostics --------------------------------------------------------------
#
# Defaults of the approximation-reliability layer. `k_usable` is the REPORTED
# Pareto-k threshold (Vehtari, Simpson, Gelman, Yao & Gabry 2024): below it the
# nested integration is reliable, at or above it the hyperparameter posterior is
# misfit by the Gaussian proposal the grid is placed with and the fit should
# escalate to the Gibbs debias. It is one number across every surface that
# reports it (`diagnostic_summary()`, `print.laplace_diagnostics()`,
# `plot_pareto_k()`, `tulpa_reloo()`, the PSIS-LOO warning, the timing verdict)
# and the Pareto-k proposal loop's own early stop.
#
# `k_samples` is the outer Pareto-k's importance-draw budget, one value across
# all four backends that report the diagnostic. It has to be one value because
# the k-hat is read against the FIXED bands `k_usable` and `gamma3_*` above:
# under gcol33/tulpa#631 the budget sets the PSIS tail FRACTION as well as the
# precision -- `.psis_tail_len(S) = min(S/5, 3 sqrt(S))` gives 20.0% at 200 and
# 13.6% at 500 -- so two backends scoring the same hyperparameter posterior at
# two budgets characterise it at two different quantiles of the weight
# distribution and can report different bands for it (gcol33/tulpa#632).
#
# 500 rather than 200, on the record rather than on preference. The joint path
# was raised from 200 to 500 at gcol33/tulpa#127 (609f262) when outer scoring
# stopped being adaptive-batched: with the proposal scored ONCE, the single
# budget carries the whole estimate instead of a batch of it. Every backend
# scores single-batch now, so that reason covers all four. 13.6% is also the
# fraction gcol33/tulpa#631 measured the k-hat to be stable at, and the one
# every shipped outer-k number -- the whole gcol33/tulpa#629 and #630 corpus,
# including #630's bit-identity baseline -- was read at. At 200 the `S/5` cap
# binds, which is the published rule's cap rather than the rule itself.
#
# What it costs: the grid, SPDE and RE-covariance paths go from 200 to 500
# inner Laplace solves when `diagnose_k` is on, since one draw is one off-grid
# re-solve. `control$k_samples` sets it per fit on every front door.
#
# `k_samples_ok` / `k_samples_good` are the same budget raised at the entry to a
# `k_quality` run, where the bootstrap CI has to resolve a named band before the
# escalation ladder starts; `k_precision_growth` below is what the ladder then
# multiplies by. They live here for the reason the base value does -- they were
# written inline at the joint front door, which is how the base value drifted.
#
# `gamma3_ok` / `gamma3_unreliable` band the inner-Laplace skewness on the
# usual skewness-magnitude convention (Bulmer 1979) -- a general reading of
# "moderate" / "substantial" skew, NOT a Rue-Martino-Chopin cutoff.
#
# `gamma3_ok` is also the local-CCD refinement's engagement gate: a refined
# outer cell keeps its node cloud only while the standardized cubic magnitude
# of its own log-marginal, read off the design's own nodes, stays below it.
# That is the same convention on the same kind of quantity, one layer out -- a
# standardized third-order departure from the Gaussian the approximation was
# placed from -- so it is one number, not two. Where it belongs is measured,
# not inherited: across an eight-family ladder of analytic outer targets (an
# equicorrelated Gaussian and Gaussian copulas with Gamma(1 .. 64) marginals,
# 48 configurations each, scored as absolute endpoint error against closed-form
# axis quantiles), 0.5 is the only threshold on the ladder 0.01 .. 2 that
# improves or ties EVERY family against refining unconditionally. Lower values
# score better pooled (0.175 gives 337.41 against 340.14) by regressing on the
# two least skewed families; higher ones regress on the moderately skewed. See
# the settings note beside the gate.
#
# `inner_k_material_ess` is the materiality floor for the INNER Pareto-k. A
# Pareto shape index is scale-free: it describes the SHAPE of the
# importance-weight tail and says nothing about its size, so where the inner
# Gaussian already reproduces the conditional posterior over the sampled region
# the weights are uniform, there is no tail for the generalized Pareto to
# describe, and the shape returned is fitted to the residual wiggle. Measured
# on the engine's own fixtures at 256 draws: a gaussian-family coefficient,
# where the inner Laplace is EXACT and gamma_3 is exactly 0, reads k-hat 0.19 /
# 0.26 with realized IS efficiency 1.000; a balanced binomial intercept (N =
# 500, S = 230, gamma_3 = -0.007) reads 0.640 at efficiency 0.99998. Both are
# noise on a proposal that needs no correction, and banding them would flag
# healthy fits, the failure these diagnostics exist to stop. The k-hat is
# therefore banded only on probed indices whose realized IS efficiency `is_ess
# / n_draws` falls BELOW this floor, i.e. where correcting the proposal costs
# at least half a percent of the sample; the raw shape is reported either way.
#
# `k_precision_growth` is the factor the `k_quality` escalation multiplies the
# importance-draw budget by once grid refinement is EXHAUSTED and the miss that
# remains is a precision miss. A `k_quality` miss has two causes: the k-hat can
# sit CONFIDENTLY outside the requested band (the grid does not represent the
# hyperparameter posterior), or its bootstrap CI can STRADDLE a band boundary
# (the point estimate may already be inside the band and only the interval's
# width prevents confirming it). Refinement is tried first on BOTH, because it
# lowers the k-hat itself rather than the noise around it and so can move an
# ambiguous k INTO the band; the draw budget is what is left when the grid has
# nothing more to give and the estimate is still ambiguous. The standard error
# of a GPD shape estimate falls as `1 / sqrt(S)`, so a constant factor buys a
# constant proportional narrowing per round: at 2 the CI narrows by ~29% per
# round, which crosses a band boundary for an estimate not already sitting on
# it, while keeping the round's cost -- one inner Laplace solve per draw -- to
# one doubling.
#
# `skew_correct` decides whether the inner-Laplace marginal quantiles are
# corrected rather than only graded: a Cornish-Fisher
# reshaping at each coordinate's own gamma_3 about the centre gamma_1 +
# gamma_3 / 2, gated to the `good` / `ok` bands of the
# COMBINED inner band (gamma_3 and the importance k-hat) by
# `gamma3_unreliable` and `k_usable` above.
#
# MEASURED. Against exact quadrature quantiles of rare-event binomial-logit
# posteriors it cuts total absolute endpoint error from 2.4931 to 0.7687
# (69.2%), improving both endpoints in every case
# (test-inner-skew-correction.R). Scored over the WHOLE marginal -- paired CRPS
# against the exact posterior in a 400-replicate prior-predictive experiment on
# the same family of fits -- it reads delta -0.01643 against the uncorrected
# Laplace at t = -1.89, essentially all of the -0.01662 the exact posterior
# itself achieves, with SBC uniformity moving 0.0833 -> 0.0329 against the exact
# reference's 0.0290 and the PIT re-entering the simultaneous band at p = 0.089
# (the section-4 gate in test-inner-skew-correction.R).
#
# THE CENTRE IS WHAT MADE THE DIFFERENCE. Reshaping about the Laplace mode
# places a mean-zero standardized variate there. RMC eq. (22) does not have
# mean zero: expanding it gives
# E[z] = gamma_1 + gamma_3 / 2, so placing the reshaped variate at mu_i asserts
# gamma_1 = -gamma_3 / 2 rather than an absent location term. On the same 400
# replicates that read scored +0.00775 at t = +3.54, a NET LOSS, and it is kept
# as a control arm in the gate. At a SYMMETRIC level pair the reshaping term
# sigma (gamma_3 / 6) (z_p^2 - 1) takes the same value at both ends, because
# z_p^2 = z_{1-p}^2, so a two-point symmetric metric could not see the defect at
# all -- only the whole-CDF score could.
#
# WHAT REMAINS UNCORRECTED. gamma_3 is a LOWER bound on the true skewness (0.875
# to 0.943 of it on the cases above), so the reshaping moves part of the way;
# and a coordinate whose location term cannot be formed (a coupled or
# multi-process unit, a field past the eta-variance solve budget) DECLINES the
# whole correction rather than reading the absent gamma_1 as zero.
#
# IT IS ON BY DEFAULT; `control$skew_correct = FALSE` restores
# the uncorrected report per fit, exactly. Three things had to hold, each
# measured on the fixtures above with the centre band removed and a declined
# coefficient back on the mixture read
# (dev_notes/issue364/RESULTS.md).
#
# THE FLIP SURVIVES THE SHIPPED GATE. Scored against the read a default-OFF fit
# gives -- the grid mixture -- on 400 prior-predictive
# replicates: the rare-event intercept t = -1.895, and the small-group Bernoulli
# design's two coefficients t = -3.765 and t = -3.201, against +3.54 / +6.12 /
# +4.64 for the earlier read that had no location term.
#
# COVERAGE HOLDS ACROSS MODEL CLASSES. Twelve configurations -- the six built-in
# families on the single-block driver, a rare-event small-group binomial, a
# small-group Poisson, the same data on the joint driver, and three crossed
# groupings at outer dimension 3 -- read off ONE solve per seed by the shipped
# `recov_sweep()`, so the corrected and uncorrected arms are paired and differ
# only in the marginal read. Pooled over 960 trials at nominal 0.95: mixture
# 0.9510, corrected 0.9542, standard error 0.0070. Every configuration is inside
# the 3-standard-error acceptance the shipped gates use, and gaussian is
# identical to the bit (its gamma_3 and gamma_1 are exactly 0).
#
# Two SMALL-SAMPLE classes move, in opposite directions, and are the whole of
# the movement (200 seeds, three levels, 400 trials per cell). The correction
# takes the small-group Poisson design from 0.8950 / 0.7050 / 0.4200 to 0.9400 /
# 0.7950 / 0.4650 at nominal 0.95 / 0.80 / 0.50, and the rare-event binomial
# from 0.9650 / 0.8050 / 0.4900 to 0.9175 / 0.7550 / 0.4700. Summed distance
# from nominal over the nine cells: 0.295 uncorrected, 0.175 corrected.
#
# THE RARE-EVENT DROP IS THE EXACT ANSWER, not a regression, and coverage at a
# FIXED truth is what cannot say so on its own -- a credible interval attains its
# nominal rate averaged over the prior, not at one parameter value. Fixture A's
# posterior is EXACT by one-dimensional quadrature, so it can be run at fixed
# truths with the exact posterior as an arm (five truths x 400 seeds): pooled,
# exact 0.9470 / 0.8650 / 0.5630 against the corrected 0.9290 / 0.8650 / 0.5630
# and the Gaussian 0.9625 / 0.8210 / 0.4165. The corrected interval reproduces
# what the EXACT posterior does at 0.80 and 0.50 and is 0.018 from it at 0.95,
# where the Gaussian is 0.044 and 0.147 away at the two lower levels. At
# beta = -2 and level 0.50 the Gaussian interval contains the truth on 0 of 400
# replicates, the exact posterior's on 367, and the corrected one on 367.
#
# THE DECLINE PATHS ARE NO-OPS. A coupled fit (every arm `multi_eta_unit`, so no
# location term), a coefficient the importance k-hat flags, a coefficient past
# the shape band and a non-nested fit all report bounds identical to the
# correction-off fit, to 0.000e+00, while an eligible coefficient on the same fit
# moves by 0.397. That takes the per-row composition against the mixture read;
# without it every one of those classes moved.
#
# `centre_unreliable` is the CENTRE band's cutoff, the counterpart of
# `gamma3_unreliable` on the other term of the same expansion.
# The reported quantile is mu_i + sigma_i {m_i + w(z_p; gamma_3)} with
# m_i = gamma_1 + gamma_3 / 2, so the correction RELOCATES the marginal by m_i
# standard errors and a band on |gamma_3| alone bounds only the reshaping. Past
# the cutoff a coefficient reports the Gaussian quantiles and records
# `centre_unreliable`.
#
# IT IS `Inf`: THE BAND IS OFF. It shipped at 1.20, chosen as
# the smallest cutoff that declined nothing the correction was MEASURED to help,
# on four fixtures none of which could reach it. Three fixtures that do reach it
# were then built, on two sampling designs, and on every one of them the band
# costs. The machinery is retained rather than deleted -- one predicate
# (`cornish_fisher_in_band()`) behind both the eligibility record and the
# quantile path, the reason in `.SKEW_CORRECT_REASONS`, its place in the
# precedence -- so a finite value here restores the band on every path at once.
#
# MEASURED, on seven fixtures with an exact reference -- 6220 coefficient-seeds,
# 3600 of them admitted -- every one gated by the SHIPPED combined inner band
# (dev_notes/issue362, dev_notes/issue376): the rare-event binomial-logit
# intercept, the small-group Bernoulli RE fit with a real outer grid, two
# rare-event binomial-logit designs carrying
# a SLOPE, and three small-group POISSON RE designs which are
# the ones that reach past 1.20 at all. Each candidate cutoff is scored as the
# PAIRED CRPS difference between the banded and the unbanded correction on the
# same fits, so the number is what the band COSTS:
#
#   cutoff  1.00    1.15    1.20    1.50    2.00    3.00    4.00    5.00
#   cost   +0.302  +0.274  +0.270  +0.252  +0.191  +0.089  +0.015   0.000
#
# monotone in the cutoff and zero only past the largest admitted centre measured
# (4.62). The correction's advantage GROWS with |m| across that whole range,
# without turning: binned by |m| the paired gain over the Gaussian runs -0.036
# at (1.2, 1.5], -0.087 at (1.5, 2], -0.150 at (2, 3] and -0.249 at (3, 6], at
# t = -4.5, -9.1, -9.9 and -8.0, with the share of seeds the correction hurts
# falling from 0.34 to 0.04 as |m| rises.
#
# NO CUTOFF IS PROTECTIVE ANYWHERE, and the band can only help where the
# correction HURTS, since converting a corrected coefficient back to the
# Gaussian is the whole of what it does. All 13 fixture-coefficients score a
# negative paired gain (t = -1.65 to -9.83), recovering 0.80 to 1.07 of what the
# exact posterior itself achieves. Binned by |m|, by |gamma_3| and by the
# effective correlation below, no bin is positive at even t = +1.3, and the
# single most positive cell anywhere is 13 seeds at t = +1.01.
#
# WHY |m| IS THE WRONG AXIS FOR A BOUND. With rho_ij the Gaussian correlation
# between eta_j and the probed coordinate and c_j = l_j''' s_j^{3/2},
#
#   m_i = (1/2) sum_j c_j rho_ij,      gamma_3(i) = sum_j c_j rho_ij^3,
#
# the same weighted sum at the first and third powers, so for sign-coherent
# terms |gamma_3| <= max_j(rho_ij^2) 2|m| and
#
#   rho_eff := sqrt(|gamma_3| / (2 |m|))  <=  max_j |rho_ij|
#
# bounds the strongest single correlation from below, out of the two numbers the
# engine already reports. It is exactly 1 on an intercept-only fit, where every
# eta reads the one latent coordinate and gamma_1 vanishes. A LARGE |m| with a
# SMALL |gamma_3| is therefore uniformly WEAK correlation, not a strong direction
# being extrapolated: measured, rho_eff has median 0.72 to 0.84 on the fixtures
# that stay under 1.20 and 0.086 to 0.116 on the three that reach past it. The
# band is anti-correlated with the pathology it was imagined for, which is why
# every cutoff costs.
#
# WHAT BOUNDS THE DISPLACEMENT NOW. `gamma1_not_computable` and the shape band,
# which is what the two terms' own reliability is read at. The centre band bound
# the displacement ARITHMETICALLY, on a quantity whose size is not evidence that
# the expansion has left its regime -- that is what the rho_eff derivation says
# and what the ladder measures. A regime in which a large |m| does signal a bad
# correction would be a reason to restore a finite cutoff here; nothing in the
# seven fixtures produces one, and a bound kept against a regime nothing has
# produced is paid for at the rate above on the regimes that do occur.
#
# `debias_select_band` is the floor the SUBSPACE DEBIAS selector reads the inner
# bands at: a probed coordinate whose combined inner band is
# at or above it is sampled exactly, the rest stay at their Gaussian
# conditional. It is "ok", i.e. one step BELOW the `unreliable` band the
# reporting layer flags on, and the reason is a measured property of the
# selector's input rather than caution. gamma_3 is a LOWER bound on the true
# skewness, not a two-sided estimate: against exact quadrature it reads 0.564,
# 0.766, 0.859 and 0.929 of the truth on the engine's own fixtures
# (test-inner-skew.R). A coordinate reporting |gamma_3| = 0.5 is therefore
# consistent with a true skewness up to 0.5 / 0.564 = 0.89, and one reporting
# 1.0 with up to 1.77, so selecting at the reported `unreliable` boundary would
# leave genuinely unreliable coordinates uncorrected. Only the SELECTOR is
# widened; the reported bands are untouched.
#
# `debias_closure_pcor` is the partial-correlation threshold the optional
# S-closure grows the selected set by, and `debias_closure_max` caps how many
# coordinates it may add. The closure exists because a coordinate left OUT of S
# is carried at its Gaussian conditional mean, a LINEAR function of x_S, so a
# neighbour strongly coupled to a member of S is being carried by exactly the
# approximation the correction is trying to remove.
#
# It is OFF by default, and that is a measurement rather than caution. On the
# engine's own fixtures the partial correlations between a fixed effect and the
# random effects run about 0.09 to 0.25, so at this threshold the closure adds
# almost nothing and changes nothing: against the exact quadrature marginal it
# moves the total endpoint error 0.5923 -> 0.5651, a difference of 0.027 against
# a combined seed standard error of 0.038, and across a 400-seed whole-fit sweep
# it leaves coverage identical on 1572 of 1600 seed-coefficient-levels. Lowering
# it far enough to matter does help -- but only by growing S to nearly the whole
# latent field (at 0.05 on a 14-coordinate fixture, |S| reaches 13.4), which is
# the FULL debias wearing a different name rather than a subspace one. So the
# honest reading is that conditioning x_{-S} on the Gaussian leaves a residual
# error the closure cannot remove cheaply, not that the coupling is absent.
# `debias_closure_max` caps the growth for the same reason.
#
# `debias_n_draws` is how many fixed-effect draws a corrected grid fit reports.
# A corrected coordinate no longer has a Gaussian-mixture summary, so the fit
# reports draws instead of moments and this is their count -- Monte Carlo error
# on a reported quantile, not a property of the correction.
# `within_cell` is the default WITHIN-CELL construction for the reported
# per-axis hyperparameter intervals, and `grid_resolved` is
# the cell-width-to-posterior-SD ratio below which the choice stops mattering.
#
# The outer grid's cell weights say how much mass each cell holds; they do not
# say how it is spread INSIDE the cell, and the reported quantile needs both.
# `box_uniform` places the cumulative full mass at each cell EDGE and
# interpolates between edges; `chord` places the cumulative mid-mass at each
# cell COORDINATE and interpolates between coordinates. The same masses over the
# same boxes with the knots moved half a cell, and that one difference is a
# whole order of convergence.
#
# MEASURED, and box-uniform is ahead on every prior-average instrument. Against
# the closed-form posterior of a gaussian-LMM fixture the two converge at order
# 1.04 and 2.00 (`dev_notes/issue353/RESULTS.md` 2.3); over a twelve-rung ladder
# spanning `h / sd` 2.82 to 27.34 -- built by sharpening the posterior at
# realistic cell counts, which is how a fit reaches a high ratio in practice --
# the paired CRPS favours box-uniform at 12 of 12 rungs, the folded PIT at 12 of
# 12 and the 95% coverage is closer to nominal at 11 of 12 -- the twelfth an
# exact tie, 0.9933 and 0.9067 both 0.0433 from nominal -- at 0.46 to 0.92x the
# width (`dev_notes/issue357/RESULTS.md` sections 4 and 6.6).
#
# THE DEFAULT IS `box_uniform` (0.0.188), decided on
# FIXED-TRUTH coverage -- the pre-registered instrument for this choice -- at
# the placement the engine ships, which is what changed. Before the placement
# pass the default axes were laid without reference to the posterior, and every
# earlier measurement of this choice was taken on a grid pinned coarser than any
# a user now gets. Three fixed-truth sweeps on current main
# (`dev_notes/issue357/RESULTS357C.md`), summed |coverage - nominal| over
# nominal 0.95 / 0.80 / 0.50, chord against box-uniform:
#
#   the instrument, truth 0.7, 300 seeds, engine placement     0.2900   0.1233
#   the same fixture truth-swept, 4680 fits the axis contained  0.2004   0.0361
#   nine (config, axis) rows over seven families, 200 seeds ea. 0.2467   0.1572
#
# Box-uniform is nearer nominal on 6 of those 9 rows and at all three levels of
# the other two, at 0.69 to 1.08x the width. The one arrangement it still loses
# on is the five-level pinned grid the failure was recorded on, where that
# fixture's truth of 0.7 falls at fraction 0.9870 of its cell -- the worst
# position in the box sweep. The four-level grid is coarser and ties, the
# seven- and nine-level grids are finer and box-uniform wins, so what fails there
# is a box POSITION and not a resolution.
#
# The position sensitivity is what held the default back, and it is what the
# placement change shrank. A within-cell reconstruction resolves an endpoint to
# within one cell, so realized coverage depends on where in its cell the unknown
# truth fell. On the pinned five-level grid box-uniform's conditional 95%
# coverage runs 0.585 to 1.000; at the shipped placement, 25 truth positions x
# 200 seeds binned on the realized position, it runs 0.868 to 0.979 -- a swing of
# 0.110 against the chord read's own 0.067 -- and at nominal 0.50 the two swings
# are 0.238 and 0.231, i.e. the same. The dependence belongs to any within-cell
# reconstruction, which is what the chord read's own numbers always said; what is
# new is that it no longer separates the two.
#
# A RESOLUTION-CONDITIONAL default was scored rather than assumed, since the two
# reads converge as `h / sd` falls and a rule keyed on it is expressible. Reading
# box-uniform only below a threshold and the chord read above it is DOMINATED by
# reading box-uniform always: summed |coverage - nominal| over the nine rows is
# 0.2517 / 0.2578 / 0.2133 / 0.2322 / 0.1733 at thresholds 1 / 1.25 / 1.5 / 2 / 3
# against 0.1572 for box-uniform everywhere. The threshold that scores best is
# the one that fires almost always, which is the fixed rule.
#
# `control$within_cell = "chord"` restores the previous report per fit, exactly.
#
# `grid_resolved = 1` is `h / sd`, both in the axis's own coordinate. It is not
# a tuning cutoff: at `h / sd` below 1 the cell is narrower than the posterior
# it discretizes, the two constructions converge to each other
# (`dev_notes/issue353/RESULTS.md` 2.3) and the position sensitivity is bounded
# by a fraction of an SD. Above it they part, and the 34-configuration census of
# the engine's own default axes puts every one of them above it -- minimum 1.01,
# median 4.25, maximum 18.06 -- so an unresolved axis is the ordinary case and
# is worth reporting rather than warning about.

# --- outer hyperparameter mode-find ------------------------------------------
#
# Box-constrained L-BFGS-B over a hyperparameter vector. Both consumers
# optimise a marginal that comes back from a compiled kernel with no analytic
# gradient, so both rely on `optim()`'s central-difference gradient and both
# are governed by its step size.
#
# `ndeps` is that step, on the axis's own (log) scale. It sets two competing
# error terms: the truncation error of the central difference grows as the
# step squared, while the step has to stay wide enough to clear the inner
# solver's own convergence tolerance. Too wide and the gradient near a flat
# optimum is dominated by truncation, leaving L-BFGS-B's line search no descent
# direction to find; it then aborts AT the mode and reports a nonzero
# convergence code, which every caller here reads as an unusable mode.
#
# `factr` is the relative-reduction stop, in units of `.Machine$double.eps`.
#
# The two consumers carry different values because they were tuned against
# different objectives, so they are recorded separately rather than averaged
# into one number that serves neither:
#
#  * `spde` -- `fit_spde_nested_ccd()` over (log range, log sigma). `factr`
#    1e5 rather than the 1e7 default, which accepted the prior mode unchanged
#    on weakly informative problems. `ndeps` 1e-2 measured across 1e-4 to 5e-2
#    on both the analytic fixture and a real inner-Laplace marginal: every step
#    up to 2.5e-2 returns convergence 0 on Linux and Windows alike and agrees
#    bit for bit, and on the real marginal 1e-2 reaches the same mode in the
#    same evaluation count at a lower objective.
#
#  * `st` -- `fit_st_nested()`'s auto-grid over (tau_spatial, tau_temporal,
#    rho). `ndeps` 1e-3 is `optim()`'s own default, written out so the step
#    this path depends on is stated rather than inherited.
#  * `pathfinder` -- `tulpa_pathfinder()`'s L-BFGS mode-find. It does NOT go
#    through `.nl_lbfgsb_mode_find()`: it is unbounded, takes an analytic
#    gradient when the caller supplies one, and carries `maxit` / `pgtol` as
#    its own arguments, so `factr` is the only value it needs from here. It is
#    recorded in this table anyway so every L-BFGS-B stop tolerance in the
#    package is set in one place.
.NL_MODE_FIND <- list(
    spde       = list(factr = 1e5, ndeps = 1e-2, maxit = 300L),
    st         = list(factr = 1e7, ndeps = 1e-3, maxit = 300L),
    pathfinder = list(factr = 1e7)
)

.nl_mode_find <- function(consumer, par) {
    if (!consumer %in% names(.NL_MODE_FIND)) {
        stop("Unknown mode-find consumer '", consumer, "'.", call. = FALSE)
    }
    tune <- .NL_MODE_FIND[[consumer]]
    if (!par %in% names(tune)) {
        stop("Unknown mode-find setting '", par, "'.", call. = FALSE)
    }
    ov <- getOption(paste0("tulpa.mode_find.", consumer, ".", par), NULL)
    if (!is.null(ov)) return(ov)
    tune[[par]]
}

# `axis_sd_ess` is the quadrature effective sample size an outer axis has to
# reach on its own marginal before its reported SD is read off the WEIGHTS
# rather than off a parabola at the modal node.
#
# The two estimators answer different questions. The weighted SD integrates the
# axis marginal against the measure the nodes carry, so it is consistent as the
# grid refines and it is the spread of the posterior the fit actually holds. The
# 3-point parabola reads the curvature at the mode, so it is a Gaussian summary
# and it moves with the spacing of the three nodes it reads. Neither is right
# everywhere: on a grid whose mass has collapsed onto one node the weighted read
# is a floor at zero, which is the case the parabola was added for.
#
# The separating statistic has to come off the WEIGHTS, so that the choice is
# not one estimator judging the other. `ess = 1 / sum(p^2)` over the axis's own
# level shares is that statistic: it counts the nodes the marginal actually
# spreads over, so it is low exactly where a discrete spread is not a spread.
#
# MEASURED (`dev_notes/issue621/probe_ess_threshold.R`), sweeping node spacing
# `h / sd` 0.25 to 4 and the mode's offset inside its cell 0 to 0.9, on a
# Gaussian axis marginal -- where the parabola is exact by construction, so it
# says what the weighted read COSTS -- and on a skewed one, where the parabola
# targets a different number. Worst relative error over every arrangement
# clearing each threshold:
#
#   threshold   weighted (gaussian)  weighted (skew)   parabola (skew)
#     1.5             1.00                0.526             0.92
#     2.5             8.5e-04             0.303             0.92
#     3               9.8e-06             0.074             0.92
#     4               7.5e-11             0.025             0.92
#
# Below 2.5 the weighted read is up to 100 % wrong on a Gaussian and the
# parabola is exact, so that is the regime the parabola serves; at 3 the
# weighted read is exact on a Gaussian to 1e-05 and within 7.4 % on a skewed
# marginal, against the parabola's 92 %. Across grids of ONE skewed density at
# `ess >= 3` the weighted read spans 0.974 to 1.074 of the truth while the
# parabola spans 0.080 to 0.642, which is the grid dependence gcol33/tulpa#621
# reports as a factor of two on a copy axis.
# `edge_mass_lift` is how far above a FLAT marginal an outer axis's boundary
# node has to sit before the axis is NAMED as holding boundary mass
# (`.nl_axis_edge_mass()`, `$outer_grid_edge_mass_axes`). Same currency as the
# rail's `.NL_RECENTER$edge_mass_mult`, `lift = m * w_edge`, and for the same
# reason: a fixed share of the marginal makes a longer axis a weaker detector of
# the same posterior. The two thresholds are separate because the labels are:
# the rail is a rescue TRIGGER on an axis whose span misses its own mode, and
# this is a report on an axis that truncates its own marginal whatever the
# argmax does.
#
# `1` is the point at which the boundary node carries what a flat marginal would
# put there, and it is read off the mass such a grid leaves OUTSIDE its span
# (`dev_notes/issue622/probe_edge_mass_lift.R`: 1400 arrangements over four
# marginal shapes, five node counts and 35 spans, the reference being the mass
# beyond the outer CELL EDGE, which is what a reported interval extends to):
#
#   lift >= 0.75   410 arrangements   least truncated 0.0074   median 0.258
#   lift >= 1.00   338               least truncated 0.0074   median 0.540
#   lift >= 1.25   287               least truncated 0.0304   median 0.726
#
# Against "more than 5 % of the marginal left outside", a lift of 1 catches 60 %
# of the truncating arrangements at a false-alarm rate of 0.011, and the worst
# arrangement it names falsely leaves 4.6 % outside -- i.e. just under the
# definition. It is a one-sided read: a high lift means the span truncates, a low
# one does not certify that it does not, which is why it is reported rather than
# acted on. On gcol33/tulpa#622's own cases the lifts are 0.62 (the observed
# fit), 1.80, 3.06 and 3.96, so the two the issue asks to see named are named and
# the near-flat one is not.
.NL_DIAG <- list(
    within_cell          = "box_uniform",
    grid_resolved        = 1,
    axis_sd_ess          = 3,
    edge_mass_lift       = 1,
    k_usable             = 0.7,
    k_samples            = 500L,
    k_samples_ok         = 800L,
    k_samples_good       = 2000L,
    k_bootstrap          = 1000L,
    gamma3_ok            = 0.5,
    gamma3_unreliable    = 1.0,
    centre_unreliable    = Inf,
    inner_k_material_ess = 0.995,
    k_precision_growth   = 2,
    skew_correct         = TRUE,
    debias_select_band   = "ok",
    debias_closure_pcor  = 0.5,
    debias_closure_max   = 200L,
    debias_n_iter        = 2000L,
    debias_warmup        = 1000L,
    debias_thin          = 1L,
    debias_n_draws       = 4000L
)

.nl_diag <- function(par) {
    if (!par %in% names(.NL_DIAG)) {
        stop("Unknown diagnostic setting '", par, "'.", call. = FALSE)
    }
    .NL_DIAG[[par]]
}



# --- cheap-pass grid screening -----------------------------------------------
#
# The outer driver can rank every cell with a short warm-started Newton run and
# skip the full inner solve wherever the screening weight is negligible. Two
# numbers set that: the weight below which a cell is dropped, and how many
# Newton steps the screen takes per cell.
#
# `iters` is a cost-vs-fidelity trade, not a convergence budget: the screen only
# has to RANK the cells, and every cell it ranks is warm-started from its
# already-screened lattice neighbour, so the quasi-mode it lands on is close to
# the cell's own mode after very few steps. Each extra step is paid on every
# cell of the grid, including the ones the screen goes on to keep, so a depth
# above what the ranking needs makes screening cost more than the solves it
# avoids.
#
# `min_keep` is the floor on how many cells the screen leaves to the full pass,
# whatever the tolerance says. The tolerance is a normalised weight, so what it
# cuts at is a gap in nats, `-(log(prune_tol) + log(Z))`; the whole range a
# caller would type is a few tens of nats, and on a log-marginal surface
# spanning thousands the screen keeps one cell at every setting. A kept set of
# one is a point mass: the placement pass reads a curvature stencil off the
# cells that were solved, finds none, and the axis is reported where it was laid
# rather than where the posterior is. A central second difference along one axis
# needs three collinear nodes, so five is the argmax cell plus a two-sided
# neighbourhood on the two axes an outer grid most often carries.
#
# The value is mirrored by `CHEAP_SCREEN_MIN_KEEP` in `src/nested_laplace_grid.h`
# -- the driver applies it, and a C++ constant cannot read this registry -- and
# the two are pinned together by test.
.NL_SCREEN <- list(
    prune_tol = 1e-3,
    iters     = 2L,
    min_keep  = 5L
)

.nl_screen <- function(par) {
    if (!par %in% names(.NL_SCREEN)) {
        stop("Unknown screening setting '", par, "'.", call. = FALSE)
    }
    .NL_SCREEN[[par]]
}

# --- default fixed-effect prior ----------------------------------------------
#
# The Gaussian prior a fitter puts on the fixed effects when the caller supplies
# no `beta_prior`. ONE value, because the prior is a modelling statement and a
# backend is a computational choice: `mode = "auto"` routes a plain fixed-effect
# model and the same model plus a random-effect term to different backends, so a
# per-backend default makes adding `(1 | g)` move the posterior for a reason the
# fit does not report.
#
# The value is `tulpa_priors()`'s documented fixed-effect default,
# `prior_normal(0, 2.5)`, which is also the scale the zero-inflation coefficient
# block already carries.
.TULPA_PRIOR <- list(
    beta_sd = 2.5
)

# Which setting each front door reads. A fitter names itself here rather than
# restating a number, so a backend that genuinely needs a different scale
# becomes a visible entry in this table instead of a literal in its own
# argument list.
.PRIOR_CONSUMERS <- list(
    tulpa         = "beta_sd",
    ridge         = "beta_sd",
    laplace       = "beta_sd",
    gaussian      = "beta_sd",
    gibbs         = "beta_sd",
    ep            = "beta_sd",
    beta_nuts     = "beta_sd",
    spde_nuts     = "beta_sd",
    multinomial   = "beta_sd",
    ordinal       = "beta_sd",
    re_cov_gibbs  = "beta_sd",
    glmm_logpost  = "beta_sd",
    sample_glmm   = "beta_sd"
)

.tulpa_prior_sd <- function(consumer = "tulpa") {
    if (!consumer %in% names(.PRIOR_CONSUMERS)) {
        stop("Unknown fixed-effect prior consumer '", consumer, "'.",
             call. = FALSE)
    }
    .TULPA_PRIOR[[.PRIOR_CONSUMERS[[consumer]]]]
}

# The engine's fixed-effect prior object, in the `list(mean, sd)` shape every
# fitter's `beta_prior` argument takes.
.tulpa_default_beta_prior <- function(consumer = "tulpa") {
    list(mean = 0, sd = .tulpa_prior_sd(consumer))
}

# --- how far refinement may move an outer axis --------------------------------
#
# The adaptive-grid and var-of-means passes add nodes to an outer axis after the
# first grid has been solved. How far they may move it is a three-rung ladder:
#
#   "none"     the axis takes no refinement nodes at all.
#   "densify"  nodes may be added strictly inside the span the axis's declared
#              nodes already cover; none may be placed past either end node.
#   "extend"   nodes may also be placed beyond the end nodes, one and two
#              declared steps out (`.hyper_propose_axis_extension()`).
#
# Which rung an axis starts on is a question of PROVENANCE, not of axis name. A
# caller who wrote the nodes down stated where the fit integrates, so refinement
# resolves that range more finely and does not leave it; a mode outside it shows
# up as mass at the edge. An axis the engine placed carries no such statement, so
# refinement may follow the posterior out. `auto_grid()` is how a wrapper package
# that computed a default of its own says which side its axis is on -- the same
# marker the recentring pass reads, so one declaration answers both questions.
.NL_AXIS_REFINE_MODES <- c("none", "densify", "extend")

.NL_AXIS_REFINE <- list(
    stated = "densify",
    placed = "extend"
)

.nl_axis_refine <- function(par) {
    if (!par %in% names(.NL_AXIS_REFINE)) {
        stop("Unknown axis-refinement setting '", par, "'.", call. = FALSE)
    }
    .NL_AXIS_REFINE[[par]]
}

# --- natural domain of a bounded outer axis ----------------------------------
#
# The set of values an axis's parameter is DEFINED on. This is a fact of the
# parameterisation, not of any grid: a proper-CAR precision `Q = D - rho W` is
# positive definite only for `rho < 1 / lambda_max`, and `lambda_max(D^-1 W)` is
# 1 on any connected graph, so 1 bounds that axis above whatever the adjacency
# is. A quadrature rule may place its nodes anywhere inside such a domain, but
# the cell EDGES it closes the outermost nodes with are an extrapolation half a
# node step past them, and an extrapolation has no reason of its own to stay
# inside. Declaring the domain is what lets `.hyper_axis_support()` refuse to
# report a support outside it (gcol33/tulpa#657), so the class of bug is
# unreachable rather than repaired one axis at a time.
#
# Keyed on the BARE axis name -- a multi-block grid prefixes its columns with
# the block that owns them (`b1.rho_car`) -- so every claim here has to hold for
# that name on every path that uses it. `rho` names four different parameters
# across the families: the BYM2 mixing weight on (0, 1), the proper-CAR
# correlation on the adjacency eigenvalue interval, the AR1 autocorrelation and
# a multi-output cross-field correlation on (-1, 1). Their upper end is 1 in all
# four; their lower ends disagree (0, `1 / lambda_min`, -1), and `1 / lambda_min`
# can sit below -1, so only the upper end is stated here. A spec built by a
# block that knows which of the four it holds declares the tighter interval
# itself through `hyper_axis_spec(bounds = )`, and the two are intersected.
#
# `open` says whether the endpoint is a member of the domain. Every endpoint
# here is open: it is the value at which the parameterisation degenerates -- a
# singular `Q` at `rho = 1`, a zero scale -- so a support reaching it is one no
# sampler can be handed. A closed endpoint would be one the model can be
# evaluated AT; the field exists so such an axis is expressible in this table
# rather than as a branch in the support rule.
#
# `.positive` is not an axis name (the leading dot marks a pseudo-entry, as in
# `.NL_FAMILY_AXES`): it is the domain of every axis integrated in log.
# `.hyper_axis_scale()` already holds which names those are, and a quantity
# integrated in log is a positive one, so membership is read off that function
# rather than from a second name list drifting against it.
.NL_AXIS_DOMAIN <- list(
    rho       = list(bounds = c(-Inf, 1),  open = c(TRUE, TRUE)),
    rho_car   = list(bounds = c(-Inf, 1),  open = c(TRUE, TRUE)),
    .positive = list(bounds = c(0, Inf),   open = c(TRUE, TRUE))
)

.nl_axis_domain_entry <- function(key) {
    if (!key %in% names(.NL_AXIS_DOMAIN)) {
        stop("Unknown axis domain '", key, "'.", call. = FALSE)
    }
    .NL_AXIS_DOMAIN[[key]]
}

# --- joint CCD placement cost ------------------------------------------------
#
# What PLACING a central composite design costs, in the only unit that matters
# on an expensive model: inner Newton solves. The mode-find behind
# `.joint_ccd_grid()` reads the outer log-posterior through the same solve the
# outer grid pays for once per cell, so a placement round is not bookkeeping --
# it is `1 + 2d + 4 C(d, 2)` full inner solves, 33 of them at `d = 4`, and the
# round caps below bound how many rounds a placement takes without knowing what
# a round costs. Left at the caps alone, placement can spend roughly 1976 solves
# at `d = 4` to place a 25-node design.
#
# `evals_per_cell` states the ceiling in the currency the caller already has:
# the tensor grid the design replaces. At `1` a placement may not spend more
# inner solves than integrating that tensor grid would have, so a placement that
# succeeds never costs more than the integration it avoids and one that runs out
# caps the wasted spend at the same number rather than at the round caps'
# worst case.
#
# `budget_floor` exempts the DESIGNED first attempt (`first_rounds` rounds from
# the grid-median seed, plus its closing stencil). That attempt is the path the
# CCD was built for and its cost does not grow with the grid, so a small tensor
# grid would otherwise decline every placement on arithmetic that only bounds
# the ESCALATION -- the joint-grid seed, the step calibration and the
# `max_rounds` rescue mode-find, which is where the unbounded cost lives. Set it
# `FALSE` to make `evals_per_cell` a hard cap on the whole placement.
#
# The five loop caps live here rather than as literal argument defaults because
# the budget is PROJECTED from them: the up-front decline computes the cheapest
# placement that could still produce a design, and it has to read the same
# numbers the loops run on.
#
# `stencil_reuse` is the curvature-reuse path: an axial-only stencil plus a
# symmetric rank-1 secant update of the off-diagonal block between full
# refreshes, `1 + 2d` evaluations a round instead of `1 + 2d + 4 C(d, 2)`. It is
# OFF because its effect on the walk has not been measured; the default is the
# full stencil every round.
.CCD_PLACEMENT <- list(
    evals_per_cell   = 1,
    budget_floor     = TRUE,
    first_rounds     = 8L,
    max_rounds       = 30L,
    calibrate_rounds = 4L,
    seed_max_pts     = 256L,
    max_halve        = 6L,
    stencil_reuse    = FALSE,
    refresh_every    = 4L
)

.ccd_placement <- function(par) {
    if (!par %in% names(.CCD_PLACEMENT)) {
        stop("Unknown CCD placement setting '", par, "'.", call. = FALSE)
    }
    .CCD_PLACEMENT[[par]]
}
