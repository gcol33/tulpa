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
# gcol33/tulpa#293 stayed invisible (the auto-recenter had to recognise the
# engine's own default axis coming back in through a consumer's prior, and the
# only way to recognise it was to have ONE place that defines it).
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
    # are not interchangeable (gcol33/tulpa#361): the `outside = "extend"` read
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
    # restating the nodes (gcol33/tulpa#361). Deliberately NOT bound to a
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
.nl_grid_axis <- function(key) {
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
    ax <- if (!is.null(spec$nodes)) {
        as.numeric(spec$nodes)
    } else {
        exp(seq(log(spec$lo), log(spec$hi), length.out = as.integer(spec$n)))
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
#     rather than a user pin (gcol33/tulpa#293).
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
# (gcol33/tulpa#352 -- a `sigma_grid` on an icar block reached the multi-block
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
# Policy for the mode-Hessian recenter of a railed default axis (gcol33/tulpa
# #289 / #290 / #291 / #293, see `R/nested_laplace_auto_grid.R`).
.NL_RECENTER <- list(
    # Nodes in a recentred axis, and how many mode-SDs it spans either side.
    n_pts     = 5L,
    span      = 2.5,

    # The mode SD is clamped into [min_sd_u, max_sd_u] on the log axis: a floor
    # so a razor-sharp local curvature does not collapse the new grid to
    # near-duplicate nodes (the point of recentring is to BRACKET the mode with
    # real spread), a ceiling so a near-flat direction does not fling nodes to
    # implausible extremes.
    min_sd_u  = 0.15,
    max_sd_u  = 3,

    # A recentred axis must survive the map back onto its own support with at
    # least this many distinct nodes; a `logit01` axis whose nodes saturate to a
    # boundary in double precision loses some, and a two-node axis carries no
    # interior cell for the mode to sit in.
    min_nodes = 3L,

    # How far above a FLAT marginal the boundary node's own weight has to sit
    # before the axis counts as railed against that boundary (`.nl_axis_rail()`,
    # gcol33/tulpa#361, #375). A marginal maximal at a boundary node has its mode
    # at or beyond it, which is the statement; this is the materiality guard that
    # keeps a merely uneven -- or numerically flat -- marginal from being moved
    # onto curvature it does not have.
    #
    # The comparison is against `1 / m` for an `m`-node axis, not against a
    # constant share. A share is not comparable across node counts: the same
    # posterior read at more nodes spreads its weight over more of them, so a
    # fixed share makes a LONGER axis a WEAKER detector (gcol33/tulpa#375
    # measures the shipped 0.5 share firing on 12 / 10 / 6 / 3 / 0 of the same
    # 20 railed fits as the span's node count goes 4 / 5 / 6 / 8 / 12). Relative
    # to uniform the same reads are 12 / 11 / 10 / 10 / 10, against 12 / 11 / 10
    # / 10 / 10 fits whose marginal is maximal at the top node at all.
    #
    # `2` is the retired 0.5 share at the four nodes it was tuned on, so the
    # calibration is transported rather than re-chosen, and both of #357's two
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
    sigma_pc_prior = list("pc.prec", c(U = 3, alpha = 0.01))
)

.nl_recenter <- function(par) {
    if (!par %in% names(.NL_RECENTER)) {
        stop("Unknown recenter setting '", par, "'.", call. = FALSE)
    }
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
# `gamma3_ok` / `gamma3_unreliable` band the inner-Laplace skewness on the
# usual skewness-magnitude convention (Bulmer 1979) -- a general reading of
# "moderate" / "substantial" skew, NOT a Rue-Martino-Chopin cutoff.
#
# `gamma3_ok` is also the local-CCD refinement's engagement gate
# (gcol33/tulpa#318): a refined outer cell keeps its node cloud only while the
# standardized cubic magnitude of its own log-marginal, read off the design's own
# nodes, stays below it. That is the same convention on the same kind of
# quantity, one layer out -- a standardized third-order departure from the
# Gaussian the approximation was placed from -- so it is one number, not two.
# Where it belongs is measured, not inherited: across an eight-family ladder of
# analytic outer targets (an equicorrelated Gaussian and Gaussian copulas with
# Gamma(1 .. 64) marginals, 48 configurations each, scored as absolute endpoint
# error against closed-form axis quantiles), 0.5 is the only threshold on the
# ladder 0.01 .. 2 that improves or ties EVERY family against refining
# unconditionally. Lower values score better pooled (0.175 gives 337.41 against
# 340.14) by regressing on the two least skewed families; higher ones regress on
# the moderately skewed. See the settings note beside the gate.
#
# `inner_k_material_ess` is the materiality floor for the INNER Pareto-k
# (gcol33/tulpa#303). A Pareto shape index is scale-free: it describes the SHAPE
# of the importance-weight tail and says nothing about its size, so where the
# inner Gaussian already reproduces the conditional posterior over the sampled
# region the weights are uniform, there is no tail for the generalized Pareto to
# describe, and the shape returned is fitted to the residual wiggle. Measured on
# the engine's own fixtures at 256 draws: a gaussian-family coefficient, where
# the inner Laplace is EXACT and gamma_3 is exactly 0, reads k-hat 0.19 / 0.26
# with realized IS efficiency 1.000; a balanced binomial intercept (N = 500,
# S = 230, gamma_3 = -0.007) reads 0.640 at efficiency 0.99998. Both are noise
# on a proposal that needs no correction, and banding them would flag healthy
# fits -- the failure gcol33/tulpa#272 exists to stop. The k-hat is therefore
# banded only on probed indices whose realized IS efficiency `is_ess / n_draws`
# falls BELOW this floor, i.e. where correcting the proposal costs at least half
# a percent of the sample; the raw shape is reported either way.
#
# `skew_correct` decides whether the inner-Laplace marginal quantiles are
# corrected (gcol33/tulpa#302) rather than only graded: a Cornish-Fisher
# reshaping at each coordinate's own gamma_3 about the centre gamma_1 +
# gamma_3 / 2 (gcol33/tulpa#354), gated to the `good` / `ok` bands of the
# COMBINED inner band (gamma_3 and the importance k-hat, gcol33/tulpa#346) by
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
# (gcol33/tulpa#346, the section-4 gate in test-inner-skew-correction.R).
#
# THE CENTRE IS WHAT MADE THE DIFFERENCE. Until gcol33/tulpa#354 the reshaping
# was applied about the Laplace mode, i.e. about a mean-zero standardized
# variate. RMC eq. (22) does not have mean zero: expanding it gives
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
# `control$skew_correct = TRUE` turns it on per fit; the default stays FALSE.
#
# `centre_unreliable` is the CENTRE band's cutoff, the counterpart of
# `gamma3_unreliable` on the other term of the same expansion (gcol33/tulpa#362).
# The reported quantile is mu_i + sigma_i {m_i + w(z_p; gamma_3)} with
# m_i = gamma_1 + gamma_3 / 2, so the correction RELOCATES the marginal by m_i
# standard errors and a band on |gamma_3| alone bounds only the reshaping. Past
# the cutoff a coefficient reports the Gaussian quantiles and records
# `centre_unreliable`.
#
# IT IS `Inf`: THE BAND IS OFF (gcol33/tulpa#376). It shipped at 1.20, chosen as
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
# intercept of gcol33/tulpa#346, the small-group Bernoulli RE fit with a real
# outer grid of gcol33/tulpa#341, two rare-event binomial-logit designs carrying
# a SLOPE, and three small-group POISSON RE designs (gcol33/tulpa#364) which are
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
# bands at (gcol33/tulpa#304): a probed coordinate whose combined inner band is
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
# per-axis hyperparameter intervals (gcol33/tulpa#357), and `grid_resolved` is
# the cell-width-to-posterior-SD ratio below which the choice stops mattering.
#
# The outer grid's cell weights say how much mass each cell holds; they do not
# say how it is spread INSIDE the cell, and the reported quantile needs both.
# The shipped `chord` read places the cumulative mid-mass at each cell
# COORDINATE and interpolates between coordinates; `box_uniform` places the
# cumulative full mass at each cell EDGE and interpolates between edges, which
# is the same masses over the same boxes with the knots moved half a cell.
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
# WHY THE DEFAULT IS STILL `chord`. A within-cell reconstruction resolves an
# endpoint to within one cell, so the realized coverage of a reported interval
# depends on where in its cell the unknown truth fell. Swept directly, 12
# positions x 200 seeds at fixed resolution, box-uniform's conditional 95%
# coverage runs 0.585 to 1.000 across one cell -- box-averaged 0.9033 against
# nominal 0.95, which is the right average and better than the chord read's
# vacuous 1.0000, but a user has one fixed unknown truth and not an average. The
# chord read is not position-insensitive either (0.655 to 0.950 at nominal 0.50
# on the same sweep); it is wide enough to hide it at 0.95. And gcol33/tulpa#337
# named fixed-truth coverage as its verdict instrument IN ADVANCE, and that
# instrument still fails at a fixed truth. Both are on the issue; the default is
# the maintainer's to move, and the construction ships selectable and reported
# in the meantime.
#
# `grid_resolved = 1` is `h / sd`, both in the axis's own coordinate. It is not
# a tuning cutoff: at `h / sd` below 1 the cell is narrower than the posterior
# it discretizes, the two constructions converge to each other
# (`dev_notes/issue353/RESULTS.md` 2.3) and the position sensitivity is bounded
# by a fraction of an SD. Above it they part, and the 34-configuration census of
# the engine's own default axes puts every one of them above it -- minimum 1.01,
# median 4.25, maximum 18.06 -- so an unresolved axis is the ordinary case and
# is worth reporting rather than warning about.
.NL_DIAG <- list(
    within_cell          = "chord",
    grid_resolved        = 1,
    k_usable             = 0.7,
    k_samples            = 200L,
    gamma3_ok            = 0.5,
    gamma3_unreliable    = 1.0,
    centre_unreliable    = Inf,
    inner_k_material_ess = 0.995,
    skew_correct         = FALSE,
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
