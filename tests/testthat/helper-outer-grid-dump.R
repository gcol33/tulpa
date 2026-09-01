# Outer-grid dump / rebuild harness for `tulpa_nested_laplace_joint()`
# (gcol33/tulpa#322).
#
# A candidate construction for the OUTER integration weights is pure
# post-processing of a fit that already ran. Two reads are built off the grid,
# and both are post-processing:
#
#   the HYPERPARAMETER read -- the grid coordinates, each cell's inner
#   log-marginal and each cell's outer design weight are the whole of what the
#   reported per-axis summary is built from;
#
#   the FIXED-EFFECT read -- each cell's fixed-effect mode and marginal
#   precision (`$grid_modes` / `$grid_hessians`, retained per cell under
#   `control$keep_grid_hessians`, gcol33/tulpa#305) are the whole of what the
#   reported coefficient mean, covariance and standard error are built from.
#
# So a weight rule can be scored against the shipped one at zero fitting minutes,
# on either read -- dump the grid state once, then re-read under any weights
# offline. A coverage experiment on a SLOPE is a fixed-effect question, so it is
# the second read that carries it.
#
# What makes such a score attributable rather than a reimplementation is that the
# offline read goes through the ENGINE's own summary code -- the axis half
# through `.nl_axis_quantiles()` -> `.nl_summary_quantile()`, the fixed half
# through `.nested_fixed_moments()`, the one grid marginalizer behind
# `summary()` / `confint()` / `vcov()` on every nested tier -- never through a
# second copy of it. Nothing here forms a quantile, a mixture moment, a weight or
# a support rule of its own; every function below assembles the arguments the fit
# itself passed and hands them to the same routine. The round-trip assertions in
# test-outer-grid-dump.R are what hold that: rebuilding a dump with the fit's own
# weights has to return the numbers the fit shipped, to floating tolerance. If it
# ever stops holding, an offline weight experiment is no longer measuring what it
# claims to and the difference it reports is no longer attributable to the
# weights.
#
# The pieces:
#
#   outer_grid_dump()          fit  -> the grid state + the summary the fit
#                                      reported, optionally written to an RDS
#   outer_grid_rebuild()       dump + weights + coordinates -> the same per-axis
#                                      read
#   outer_grid_rebuild_fixed() dump + weights -> the same fixed-effect read
#   outer_grid_one_cell()      dump + a whole-grid candidate + one cell -> the
#                                      same candidate restricted to that cell
#   outer_grid_read_diff()     two reads -> mean absolute endpoint / width /
#                                      median difference
#   outer_grid_noise_floor()   dump -> the scale below which a difference is not
#                                      resolved by this grid

# The probabilities the joint fitters summarise every axis at. Fixed rather than
# a caller's choice: `.nl_axis_quantiles()` reads exactly three, in this order.
OGD_PROBS <- c(0.025, 0.5, 0.975)

# The parts a read is compared in, each named once and given as the numbers that
# part is compared on. Every layer that splits a read walks THIS set --
# `outer_grid_read_diff()` forms one difference per entry, the noise floor
# aggregates one per entry, and the verdict scores one per entry -- so a part
# named here is a part all three report.
#
# The alternative shape, one `<part>_above_floor` field per part written out by
# hand beside the others, is what let the median go unscored: a part gained a
# difference and a floor and no verdict, and nothing in the harness could notice
# because the verdict set was not derived from anything (gcol33/tulpa#330).
OGD_PARTS <- list(
  endpoints = function(r) c(r$ci_lo, r$ci_hi),
  widths    = function(r) r$ci_hi - r$ci_lo,
  median    = function(r) r$median)

# What KIND of node set the fit's read was taken off, and the per-axis domains a
# moment-rule read needs. Both come off the fit's own recorded provenance, so a
# dump replays the support the fit used rather than a guess at it: the multi-block
# driver stamps `theta_interval_read`, and a fit predating that (or the
# single-block path, which is a plain tensor grid throughout) falls back to the
# same `.nl_node_support()` the driver calls.
.ogd_support <- function(fit) {
  fit$theta_interval_read %||%
    tulpa:::.nl_node_support(fit$integration, fit$weight_kind)
}

# The WITHIN-CELL construction the fit's own intervals were read with
# (gcol33/tulpa#357). A dump replays the fit's read, so it carries this the same
# way it carries the support: a fit run with `control$within_cell` other than
# the default and re-read here has to come back with the numbers it shipped, and
# the round-trip assertion in test-outer-grid-dump.R is what holds that.
.ogd_within <- function(fit) {
  tulpa:::.nl_within_cell_mode(fit$within_cell_requested)
}

# `domains` is consumed only by a `moment_rule` support (a central-composite
# design's interval comes from its moments on the axis's own coordinate), and the
# driver passes NULL otherwise. Mirrored here so the dump carries the argument the
# fit passed, not a superset of it.
.ogd_domains <- function(fit, support) {
  if (!identical(support, "moment_rule")) return(NULL)
  tulpa:::.joint_axis_domains(fit)
}

# A per-cell list off the fit, held to describing the SAME grid the rest of the
# dump does. NULL passes through rather than erroring: the per-cell fixed-effect
# blocks are the one part of the state a fit can decline to retain, and a fit
# that declined still has an axis read worth dumping. What must not pass is a
# list of the wrong length -- marginalizing cell k's block against cell k's
# weight only means anything while the two index the same cell.
.ogd_check_cells <- function(x, what, n) {
  if (is.null(x)) return(NULL)
  if (length(x) != n) {
    stop("this fit's `$", what, "` (", length(x), " entries) does not align ",
         "with its outer grid (", n, " cell(s)); the grid state is not ",
         "dumpable.", call. = FALSE)
  }
  x
}

# Why a dump has no fixed-effect blocks, in the driver's own words:
# `.joint_finalize_grid_fixed()` records "not_requested" when the retention was
# switched off, and `.joint_attach_grid_fixed()` records "no_fixed_effects",
# "block_not_extracted", "grid_misaligned" or "cell_block_unavailable" when it
# was on and could not be honoured. A fit that retained them carries NA here,
# which is not a reason.
.ogd_declined_reason <- function(dump) {
  r <- dump$grid_fixed_declined
  if (is.null(r) || length(r) != 1L || is.na(r)) "reason not recorded"
  else as.character(r)
}

# Per-fit outer-grid state, everything a summary read needs and nothing that
# needs an inner solve to reproduce.
#
# `file` writes the dump to an RDS; the returned object is the same either way,
# so a caller can dump-and-use in one session or dump-and-reload across sessions.
outer_grid_dump <- function(fit, file = NULL) {
  if (!is.list(fit)) {
    stop("`fit` must be a tulpa fit (a list), got ", class(fit)[1L], ".",
         call. = FALSE)
  }
  tg <- fit$theta_grid
  if (is.null(tg) || !is.matrix(tg) || ncol(tg) == 0L) {
    stop("this fit carries no outer hyperparameter grid, so there is no ",
         "integration to dump. `outer_grid_dump()` takes a ",
         "`tulpa_nested_laplace_joint()` result whose `$theta_grid` is a ",
         "non-empty matrix.", call. = FALSE)
  }
  lm <- fit$log_marginal
  if (is.null(lm) || length(lm) != nrow(tg)) {
    stop("this fit's `$log_marginal` (", length(lm), " value(s)) does not ",
         "align with its outer grid (", nrow(tg), " cell(s)); the grid state ",
         "is not dumpable.", call. = FALSE)
  }
  w <- fit$weights
  if (is.null(w) || length(w) != nrow(tg)) {
    stop("this fit's `$weights` (", length(w), " value(s)) does not align ",
         "with its outer grid (", nrow(tg), " cell(s)); the grid state is ",
         "not dumpable.", call. = FALSE)
  }
  support <- .ogd_support(fit)
  # `refining_axis` tags the cells a mode-tracked refinement pass pinned to one
  # axis; the per-axis read drops the foreign ones, so it is part of the state.
  refining <- fit$refining_axis %||% rep("", nrow(tg))
  # The per-cell fixed-effect mode and marginal precision the coefficient read is
  # marginalized from. Both are O(n_fixed^2) per cell and flat in the latent
  # dimension, so carrying them is what a coefficient re-read offline costs:
  # measured at 9.1 KB over 27 cells and 34.1 KB over 105 at n_fixed = 2, against
  # a whole dump of 17.7 KB and 52.6 KB.
  grid_modes    <- .ogd_check_cells(fit$grid_modes, "grid_modes", nrow(tg))
  grid_hessians <- .ogd_check_cells(fit$grid_hessians, "grid_hessians", nrow(tg))
  dump <- list(
    joint_grid   = tg,
    log_marginal = as.numeric(lm),
    dnode        = fit$dnode,
    weight_kind  = fit$weight_kind,
    weights      = as.numeric(w),
    refining_axis = as.character(refining),
    axis_names   = colnames(tg),
    axis_tags    = tulpa:::.joint_axis_tags_raw(fit),
    axis_domains = .ogd_domains(fit, support),
    support      = support,
    within       = .ogd_within(fit),
    probs        = OGD_PROBS,
    integration            = fit$integration,
    integration_requested  = fit$integration_requested,
    integration_declined   = fit$integration_declined,
    local_ccd_info         = fit$local_ccd_info,
    theta_interval_read        = fit$theta_interval_read,
    theta_interval_design_mass = fit$theta_interval_design_mass,
    theta_within_cell          = fit$theta_within_cell,
    theta_within_cell_declined = fit$theta_within_cell_declined,
    theta_mean   = fit$theta_mean,
    theta_sd     = fit$theta_sd,
    grid_modes    = grid_modes,
    grid_hessians = grid_hessians,
    # NA on a fit that retained the blocks; the driver's reason string on one
    # that did not, so the fixed read can say why it has nothing.
    grid_fixed_declined = fit$grid_fixed_declined,
    n_fixed      = fit$n_fixed,
    fixed_names  = fit$fixed_names,
    # The read the fit itself shipped: the target of the round trip.
    reported     = list(median = fit$theta_median,
                        ci_lo  = fit$theta_ci_lo,
                        ci_hi  = fit$theta_ci_hi)
  )
  class(dump) <- c("tulpa_outer_grid_dump", "list")
  if (!is.null(file)) saveRDS(dump, file)
  dump
}

outer_grid_load <- function(file) {
  d <- readRDS(file)
  if (!inherits(d, "tulpa_outer_grid_dump")) {
    stop("`", file, "` does not hold an outer-grid dump.", call. = FALSE)
  }
  d
}

# The coordinates a read is taken at. `NULL` is the dump's own, which is the
# round trip; a candidate LOCATION rule (gcol33/tulpa#327) supplies a perturbed
# matrix of the same shape, so the atoms move within the grid the fit left while
# the weights they carry stay as they were. The two arguments are independent,
# which is what lets a mass rule and a place rule be scored apart and together.
#
# Held to the dump's own dimensions, for the reason `.ogd_check_cells()` holds
# the per-cell blocks to them: a read is attributable to the candidate rule only
# while cell k of the coordinates is the same cell as cell k of the weights.
.ogd_coords <- function(dump, joint_grid) {
  if (is.null(joint_grid)) return(dump$joint_grid)
  jg <- as.matrix(joint_grid)
  if (!identical(dim(jg), dim(dump$joint_grid))) {
    stop("`joint_grid` is ", nrow(jg), " x ", ncol(jg), " but the dumped grid ",
         "has ", nrow(dump$joint_grid), " cell(s) on ",
         ncol(dump$joint_grid), " axis(es).", call. = FALSE)
  }
  colnames(jg) <- colnames(dump$joint_grid)
  jg
}

# The per-axis median + 95% interval a dump's grid gives under `weights` at
# `joint_grid`, through the engine's own summary path. Both NULL uses the fit's
# own, which is the round trip.
#
# `.nl_axis_quantiles()` takes the weights explicitly on every call here. The
# driver passes NULL on a density grid and lets the helper form the per-axis
# softmax of `log_marginal` itself; that softmax restricted to an axis's kept
# cells and renormalised IS the fit's own weight vector restricted the same way,
# so the two agree to floating point and the harness has one entry point instead
# of two.
outer_grid_rebuild <- function(dump, weights = NULL, joint_grid = NULL) {
  w <- weights %||% dump$weights
  n <- nrow(dump$joint_grid)
  if (length(w) != n) {
    stop("`weights` has length ", length(w), " but the dumped grid has ", n,
         " cell(s).", call. = FALSE)
  }
  tulpa:::.nl_axis_quantiles(
    .ogd_coords(dump, joint_grid), dump$log_marginal, dump$refining_axis,
    probs = dump$probs, weights = as.numeric(w),
    support = dump$support, domains = dump$axis_domains,
    within = dump$within %||% tulpa:::.nl_within_cell_mode(NULL))
}

# The grid-marginalized fixed-effect mean and covariance a dump's cells give
# under `weights`, through the engine's own `.nested_fixed_moments()`. As with
# the axis half, `weights = NULL` uses the fit's own, which is the round trip.
#
# `.nested_fixed_moments()` reads exactly three fields off whatever list it is
# handed -- `grid_modes`, `grid_hessians`, `weights` -- and forms the law of
# total covariance over the cells,
#
#   mean = sum_k w_k mu_k
#   cov  = sum_k w_k (V_k + mu_k mu_k') - mean mean'
#
# with V_k = solve(grid_hessians[[k]]) and w_k the normalized weights. Nothing in
# it re-solves, and nothing in it reads a field a dump cannot carry. So the whole
# of the offline fixed-effect read is that one call on a list holding the dumped
# cells and the candidate weights; the mixture algebra is not restated here, and
# a cell that carries no usable block is skipped by the engine's own rule rather
# than by one of this harness's.
#
# `se` is the coefficient table's own derivation from the returned covariance:
# the square root of its diagonal, floored at zero because the two terms of the
# law of total covariance can cancel to a slightly negative variance.
outer_grid_rebuild_fixed <- function(dump, weights = NULL) {
  if (is.null(dump$grid_modes) || is.null(dump$grid_hessians)) {
    stop("this dump carries no per-cell fixed-effect blocks, so the ",
         "fixed-effect read cannot be rebuilt from it: the fit declined the ",
         "retention (", .ogd_declined_reason(dump), "). Refit with ",
         "`control$keep_grid_hessians = TRUE`.", call. = FALSE)
  }
  w <- weights %||% dump$weights
  n <- nrow(dump$joint_grid)
  if (length(w) != n) {
    stop("`weights` has length ", length(w), " but the dumped grid has ", n,
         " cell(s).", call. = FALSE)
  }
  mom <- tulpa:::.nested_fixed_moments(
    list(grid_modes = dump$grid_modes, grid_hessians = dump$grid_hessians,
         weights = as.numeric(w)))
  if (is.null(mom)) {
    stop("no dumped cell carries both a fixed-effect mode and a positive ",
         "weight, so there is nothing for the mixture to marginalize over.",
         call. = FALSE)
  }
  est <- mom$mean
  V   <- mom$cov
  nm  <- dump$fixed_names
  if (length(nm) == length(est)) {
    names(est) <- nm
    dimnames(V) <- list(nm, nm)
  }
  list(mean = est, cov = V, se = sqrt(pmax(diag(V), 0)))
}

# Integration weights for a candidate per-cell design weight `dnode`, through the
# engine's own `.joint_integration_weights()`. A weight rule that redistributes
# the outer design weight (which is what a subdivision rule does) enters the
# harness here, so the softmax and the non-finite-cell handling are the fit's own.
outer_grid_weights <- function(dump, dnode = NULL, log_marginal = NULL) {
  tulpa:::.joint_integration_weights(log_marginal %||% dump$log_marginal, dnode)
}

# A whole-grid candidate rule restricted to ONE cell, in the two arguments
# `outer_grid_rebuild()` already takes.
#
# What is scored off this grid is a weighted QUANTILE, and a quantile is not a
# sum over its atoms: moving one atom changes which OTHER atom sits at the
# crossing. So a per-cell property of a rule -- whether this cell is better off
# with its mass corrected, its place corrected, or both -- is not readable by
# comparing that cell's own contribution against a reference. It is readable by
# intervening on that cell alone and re-reading the whole grid, which is what
# this assembles (gcol33/tulpa#333).
#
# `dnode` / `joint_grid` are the candidate rule's OWN output over every cell,
# computed by the caller from the engine's rules; the restriction is the only
# thing done here. Cell `cell` takes the candidate's value and every other cell
# keeps the dump's, so the returned pair differs from the fit's own state in one
# row of one matrix and one entry of one weight vector. Nothing here forms a
# multiplier, a barycentre or a weight: `outer_grid_weights()` normalises the
# restricted design weights through the engine's own `.joint_integration_weights()`,
# the same call the whole-grid candidate goes through.
#
# Returns `weights` (NULL when no mass candidate was given) and `joint_grid`
# (NULL when no location candidate was given), which is exactly the argument
# pair `outer_grid_rebuild()` and `outer_grid_weight_report()` read -- so the
# one-cell candidate and the whole-grid one are scored by the same code.
outer_grid_one_cell <- function(dump, cell, dnode = NULL, joint_grid = NULL) {
  n <- nrow(dump$joint_grid)
  if (length(cell) != 1L || !is.finite(cell) || cell < 1L || cell > n) {
    stop("`cell` must be one cell index in 1:", n, ".", call. = FALSE)
  }
  cell <- as.integer(cell)
  w <- NULL
  if (!is.null(dnode)) {
    if (length(dnode) != n) {
      stop("`dnode` has length ", length(dnode), " but the dumped grid has ", n,
           " cell(s).", call. = FALSE)
    }
    dn <- dump$dnode %||% rep(1, n)
    dn[cell] <- as.numeric(dnode)[cell]
    w <- outer_grid_weights(dump, dnode = dn)
  }
  jg <- NULL
  if (!is.null(joint_grid)) {
    jg <- .ogd_coords(dump, joint_grid)
    out <- dump$joint_grid
    out[cell, ] <- jg[cell, ]
    jg <- out
  }
  list(weights = w, joint_grid = jg, cell = cell)
}

# Mean absolute difference between two reads, one part of `OGD_PARTS` at a time:
# the 2 x n_axes interval endpoints, the n_axes widths, and the n_axes medians.
# Axes whose read is NA on either side are dropped from the mean and counted
# (`n_<part>`), so a difference is never averaged against a missing number.
outer_grid_read_diff <- function(a, b) {
  out <- list()
  for (nm in names(OGD_PARTS)) {
    f  <- OGD_PARTS[[nm]]
    d  <- abs(as.numeric(f(b)) - as.numeric(f(a)))
    ok <- is.finite(d)
    out[[nm]] <- if (any(ok)) mean(d[ok]) else NA_real_
    out[[paste0("n_", nm)]] <- sum(ok)
  }
  out
}

# Which cells the per-axis read of axis `ax` uses. Mirrors the mask in
# `.nl_axis_quantiles()`: a cell a refinement pass pinned to a FOREIGN axis holds
# this axis at one non-varying value, so including it oversamples that value.
# The mirror is not trusted -- `outer_grid_noise_floor()` reads every axis
# uncoarsened as well, and the round-trip test asserts that read equals
# `outer_grid_rebuild()`, which fails the moment the two masks drift apart.
.ogd_axis_use <- function(dump, j) {
  ax <- dump$axis_names[j]
  keep <- dump$refining_axis == "" | dump$refining_axis == ax |
    dump$refining_axis == paste0("consistency_", ax)
  keep & is.finite(dump$joint_grid[, j])
}

# One axis's read off a coarsened version of its own atom set.
#
# `stride = 1` is the axis read as it stands. Above that, the atoms are sorted
# along the axis and consecutive runs of `stride` are merged into a single atom
# at their weighted mean carrying their summed weight -- a weight-preserving
# halving (or thirding) of the axis resolution. `offset` shifts where the runs
# start, so which atoms end up merged together is not one fixed choice.
.ogd_axis_read <- function(dump, w, j, stride = 1L, offset = 0L) {
  use <- .ogd_axis_use(dump, j)
  if (!any(use)) return(rep(NA_real_, length(dump$probs)))
  v  <- as.numeric(dump$joint_grid[use, j])
  ws <- as.numeric(w)[use]
  if (!any(is.finite(ws)) || sum(ws, na.rm = TRUE) <= 0) {
    return(rep(NA_real_, length(dump$probs)))
  }
  ws[!is.finite(ws)] <- 0
  ws <- ws / sum(ws)
  if (stride > 1L) {
    ord <- order(v)
    v <- v[ord]; ws <- ws[ord]
    grp <- ((seq_along(v) + as.integer(offset) - 1L) %/% as.integer(stride)) + 1L
    tot <- as.numeric(tapply(ws, grp, sum))
    v   <- as.numeric(tapply(ws * v, grp, sum)) / tot
    ws  <- tot
    keep <- is.finite(v) & is.finite(ws) & ws > 0
    v <- v[keep]; ws <- ws[keep]
    if (!length(v) || sum(ws) <= 0) return(rep(NA_real_, length(dump$probs)))
    ws <- ws / sum(ws)
  }
  dm <- if (length(dump$axis_domains) < j) NA_character_ else dump$axis_domains[[j]]
  tulpa:::.nl_summary_quantile(v, ws, dump$probs, dm, dump$support,
                               dump$within %||% tulpa:::.nl_within_cell_mode(NULL))
}

# Every axis at one coarsening, in the shape `outer_grid_rebuild()` returns.
.ogd_read_at <- function(dump, w, stride = 1L, offset = 0L) {
  nms <- dump$axis_names %||% paste0("V", seq_len(ncol(dump$joint_grid)))
  q <- vapply(seq_along(nms), function(j) .ogd_axis_read(dump, w, j, stride, offset),
              numeric(length(dump$probs)))
  q <- matrix(q, nrow = length(dump$probs))
  nm <- function(x) stats::setNames(x, nms)
  list(median = nm(q[2L, ]), ci_lo = nm(q[1L, ]), ci_hi = nm(q[3L, ]))
}

# The scale below which a difference between two reads is not resolved by this
# grid.
#
# What it estimates: the spread of the reported read under a change to the outer
# grid that a CONVERGED grid would be insensitive to. The change used is a
# weight-preserving coarsening of each axis's own atom set -- consecutive atoms
# merged into one at their weighted mean carrying their summed weight, at
# strides 2 and 3 and at every starting offset. Total integrated mass, and each
# merged group's first moment along the axis, are exactly preserved; what the
# merge removes is resolution. So on a grid fine enough for the read to be a
# property of the posterior the ensemble sits on top of the shipped read, and on
# a grid too coarse for that it does not, and the size of the gap is how much of
# the reported number is the grid's own discretisation rather than the posterior's
# shape.
#
# That is the comparison a candidate weight rule has to clear: a rule that
# redistributes weight among the SAME cells and moves the read by less than one
# step of coarsening moves it has not been shown to change anything this grid can
# resolve. Reported in the same three parts, and with the same mean-absolute
# aggregation, as `outer_grid_read_diff()`, so the two numbers are directly
# comparable.
#
# The estimate is one-sided by construction: merging atoms can only lose spread,
# never add it, so the coarsened interval tends to be the narrower one. It is a
# floor on the resolution of the read, not a symmetric error bar on it.
outer_grid_noise_floor <- function(dump, weights = NULL, strides = c(2L, 3L)) {
  w <- weights %||% dump$weights
  base <- .ogd_read_at(dump, w, stride = 1L)
  parts <- list()
  for (s in as.integer(strides)) {
    for (off in seq_len(s) - 1L) {
      parts[[length(parts) + 1L]] <-
        outer_grid_read_diff(base, .ogd_read_at(dump, w, stride = s, offset = off))
    }
  }
  agg <- function(nm) {
    v <- vapply(parts, function(p) p[[nm]], numeric(1))
    if (!any(is.finite(v))) NA_real_ else mean(v[is.finite(v)])
  }
  c(lapply(stats::setNames(nm = names(OGD_PARTS)), agg),
    list(n_perturbations = length(parts), strides = as.integer(strides),
         base = base))
}

# A candidate rule's difference from the shipped read, against the floor. The
# verdict is the whole point of the harness: a difference is reported relative to
# what this grid can resolve, never on its own.
#
# The candidate is a weight vector, a perturbed coordinate matrix, or both, so a
# mass rule and a place rule are scored on the same footing and their combination
# is a third candidate rather than a special case. The floor is the resolution of
# the grid the fit left, so it is read at the dumped coordinates whichever
# candidate is being scored.
#
# `above_floor` is one logical per part of `OGD_PARTS`, in that order, so the
# verdict covers exactly the parts the difference and the floor carry.
outer_grid_weight_report <- function(dump, weights = NULL, floor = NULL,
                                     joint_grid = NULL) {
  fl <- floor %||% outer_grid_noise_floor(dump)
  d  <- outer_grid_read_diff(outer_grid_rebuild(dump),
                             outer_grid_rebuild(dump, weights, joint_grid))
  list(diff = d, floor = fl,
       above_floor = vapply(stats::setNames(nm = names(OGD_PARTS)),
                            function(nm) isTRUE(d[[nm]] > fl[[nm]]),
                            logical(1)))
}

# --------------------------------------------------------------------------- #
# The shared measurement fixture                                              #
# --------------------------------------------------------------------------- #

# Three crossed iid blocks on one gaussian arm, each with its own pinned
# geometric `sigma_grid`, so the outer grid is a tensor of a size the caller
# chooses and is identical from run to run. Every file that scores a candidate
# read against this harness -- the box-mass rule, the barycentre place rule, the
# descriptor plane -- fits the same model, so it is built once here rather than
# copied into each.
ogd_fixture_sim <- function(sd_true, seed = 4242L, G = 30L, N = 600L) {
  set.seed(seed)
  grp <- lapply(seq_along(sd_true), function(k) sample.int(G, N, replace = TRUE))
  X <- cbind(1, stats::rnorm(N))
  eta <- as.numeric(X %*% c(0.2, 0.6))
  for (k in seq_along(sd_true)) {
    eta <- eta + stats::rnorm(G, 0, sd_true[k])[grp[[k]]]
  }
  list(y = eta + stats::rnorm(N, 0, 0.5), X = X, grp = grp, N = N, G = G,
       sd_true = sd_true)
}

# `within_cell` is STATED, not inherited (gcol33/tulpa#599). Every other input
# to these measurements is pinned -- the seed, the grid, the integration rule --
# and the within-cell construction is the one that was not, so a change to
# `.NL_DIAG$within_cell` silently re-targeted what the recorded numbers measure.
# The default here is the engine's own shipped read, so the rules are scored
# against what a user gets; a file wanting the other read passes it and says so.
ogd_fixture_fit <- function(sim, levels, spread = 3,
                            within_cell = "box_uniform") {
  prior <- lapply(seq_along(sim$grp), function(k) {
    s <- sim$sd_true[k]
    list(type = "iid", obs_idx = list(sim$grp[[k]]), n_units = sim$G,
         sigma_grid = exp(seq(log(s / spread), log(s * spread),
                              length.out = levels)))
  })
  suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = sim$y, n_trials = rep(1L, sim$N), X = sim$X,
                              family = "gaussian", phi = 0.0625)),
    prior = prior,
    control = list(n_threads = 1L, diagnose_k = FALSE, max_iter = 100L,
                   tol = 1e-8, integration = "grid",
                   within_cell = within_cell)))
}

for (.nm in c("OGD_PROBS", "OGD_PARTS", "outer_grid_dump", "outer_grid_load",
              "outer_grid_rebuild", "outer_grid_rebuild_fixed",
              "outer_grid_weights", "outer_grid_one_cell",
              "outer_grid_read_diff", "outer_grid_noise_floor",
              "outer_grid_weight_report", "ogd_fixture_sim",
              "ogd_fixture_fit")) {
  assign(.nm, get(.nm), envir = globalenv())
}
rm(.nm)
