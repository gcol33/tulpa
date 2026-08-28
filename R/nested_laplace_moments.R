# Posterior-moment machinery for the nested-Laplace fits: weighted mean / SD /
# median / CI over the outer hyperparameter grid, per-axis weighted quantiles,
# the axis-marginal Laplace-refined SD, and the multi-block joint / per-block
# marginal moments. Consumed by the nested-Laplace drivers in nested_laplace.R.

# Compute weighted theta_mean / theta_sd / theta_median / theta_ci_lo /
# theta_ci_hi from grid + weights. The mean and SD are produced via the
# usual `sum(w * x)` / `sum(w * x^2)` moments; the median and 2.5/97.5
# CI are produced via `.nl_axis_quantiles` so that asymmetric scale-like
# hyperparameters (sigma, alpha, range, phi) have a calibrated headline
# summary alongside mean +/- SD. Median is the recommended summary for
# right-skewed posteriors: `mean` of a weakly-identified positive ratio
# is pulled up by the right tail; `median` matches truth at small n.
.nl_posterior_moments <- function(res, type, within = .NL_WITHIN_CELL) {
  within <- match.arg(within)
  w <- res$weights
  tg <- res$theta_grid
  if (is.matrix(tg)) {
    res$theta_mean <- as.numeric(crossprod(w, tg))
    names(res$theta_mean) <- colnames(tg)
    res$theta_sd <- sqrt(pmax(0, as.numeric(crossprod(w, tg^2)) -
                                  res$theta_mean^2))
    names(res$theta_sd) <- colnames(tg)
  } else {
    ms <- .nl_wtd_mean_sd(tg, w)
    res$theta_mean <- ms$mean
    res$theta_sd   <- ms$sd
  }
  doms <- .nl_axis_domains(res, type)
  qs <- .nl_axis_quantiles(tg, res$log_marginal, res$refining_axis,
                           domains = doms, within = within)
  res$theta_median <- qs$median
  res$theta_ci_lo  <- qs$ci_lo
  res$theta_ci_hi  <- qs$ci_hi
  res$within_cell_requested <- within
  .nl_attach_interval_provenance(res, qs, tg, doms)
}

# The `.NL_DOMAIN_TRANSFORM` domain of every axis of a fit, from
# `.joint_axis_domains()` -- the SAME per-axis registry the outer Pareto-k
# unconstrains with, so "what support does this axis live on" keeps one
# definition (`R/nested_laplace_joint_pareto_k.R`). Two shims and nothing else:
# a single-axis grid is stored as a bare vector and is named by `theta_names`,
# and the registry path knows its family as an ARGUMENT while `res$prior` is
# only attached after the moments are taken.
.nl_axis_domains <- function(res, type = NULL) {
  tg <- res$theta_grid
  if (is.null(tg)) return(NULL)
  if (!is.matrix(tg)) {
    nm <- (res$theta_names %||% "theta")[1L]
    tg <- matrix(as.numeric(tg), ncol = 1L, dimnames = list(NULL, nm))
  }
  .joint_axis_domains(list(
    theta_grid   = tg,
    axis_offsets = res$axis_offsets,
    blocks       = res$blocks,
    prior        = if (is.null(type)) res$prior else list(type = type)))
}

# Marginal log-density along a single hyperparameter axis (logsumexp over
# the other-axis cells at each unique value). `vals` and `log_marg` are
# length n_cells; `keep` is an optional logical mask (cartesian + same-
# axis slice cells). Returns sorted unique axis values and the matching
# marginal log-density.
# Weighted quantile on a discrete (value, weight) distribution. Uses
# midpoint-of-mass cumulative probability (Type 7-like) plus linear
# interpolation, so quantiles vary smoothly with weights rather than
# snapping to grid cells.
#
#  * Aggregates duplicate values: weights at equal `values` are summed
#    before interpolation. Idempotent on already-unique axes; needed when
#    the joint grid carries repeated values (e.g. slice-cell refinement
#    re-uses the modal axis value across multiple Newton-Laplace cells).
#  * Filters non-finite values and non-positive weights.
#  * Returns `NA` per requested `probs` when the support is empty;
#    returns the unique support value when only one survives.
#
# Used by `.alpha_grid_moments` to surface the posterior median and 95%
# interval of `alpha` (a direct hyperparameter axis after the
# (sigma, alpha) reparameterization) from the joint nested-Laplace
# posterior. Weighted quantiles handle skewed/heavy-tailed marginals that
# `mean +/- 1.96 sd` summaries misrepresent.
#
# `outside` is the policy for a probability beyond the support's own cumulative
# range, `[p[1], p[n]]`, which the mid-mass convention leaves the outer half-cell
# of mass sitting in.
#
# `"clamp"` returns the extreme value, the convention a SAMPLE takes: beyond the
# extreme order statistic nothing is known about the tail, so the order statistic
# is the answer.
#
# `"extend"` is the convention a CELL PARTITION takes. A tensor grid's values are
# not order statistics, they are cell representatives: each carries the mass of
# an interval it sits at the centre of, so the extreme cell's mass reaches half a
# spacing past its coordinate and the support's edge is the design's own
# geometry rather than an unknown tail. Clamping there reports an interval the
# read cannot place its own outer half-cell of mass inside -- measured, on a
# prior-predictive experiment whose grid tiles the prior, as a PIT atom at 0 and
# 1 of exactly the outer half-cell mass. The half-spacing is
# mirrored in `log` when every value is positive, which is where a scale grid is
# equally spaced and where the edge cannot cross zero, and in the value itself
# otherwise -- the same `all(vals > 0)` test `.nl_laplace_at_mode_sd_axis()`
# picks its own coordinate with. Nothing inside `[values[1], values[n]]` moves:
# the interpolant, its knots and every probability the clamp did not bind at are
# unchanged.
#
# On weights that are a QUADRATURE DESIGN (`ccd_weights()`) the cumulative sum is
# not a CDF at all, and clamping reports the design's own extent as a posterior
# interval; `"na"` withholds the number instead. Such a
# support is summarized by `.nl_moment_quantile()`, which uses the moments the
# design does deliver.
.nl_wtd_quantile <- function(values, weights, probs,
                             outside = c("clamp", "extend", "na"),
                             domain = NA_character_) {
  outside <- match.arg(outside)
  a <- .nl_axis_atoms(values, weights)
  if (is.null(a)) return(rep(NA_real_, length(probs)))
  v <- a$v; w <- a$w
  if (length(v) == 1L) return(rep(v[1L], length(probs)))
  p <- cumsum(w) - w / 2
  # The outer half-cells are two more knots: mass 0 at the lower edge and the
  # whole mass at the upper one, so the same interpolant carries them and the
  # interior knots keep their positions.
  if (identical(outside, "extend")) {
    e <- .nl_cell_edges(v, domain)
    p <- c(0, p, 1)
    v <- c(e[1L], v, e[2L])
  }
  n <- length(v)
  # `approx` emits "collapsing to unique 'x' values" when tiny floor
  # weights against a dominant prefix sum produce numerically equal
  # midpoints. The collapse (tied p -> mean of v) is the right behavior
  # at that resolution -- the underlying mass between the tied cells is
  # zero up to floating-point error. Suppress the noise.
  na_outside <- identical(outside, "na")
  vapply(probs, function(q) {
    if (q <= p[1L]) return(if (na_outside) NA_real_ else v[1L])
    if (q >= p[n])  return(if (na_outside) NA_real_ else v[n])
    y <- suppressWarnings(approx(p, v, xout = q, method = "linear")$y)
    if (is.finite(y)) return(y)
    .nl_interp_repair(p, v, q, y)
  }, numeric(1L))
}

# The interior read's overflow guard.
#
# `stats::approx`'s linear interpolant is `y0 + (y1 - y0) * t`, which forms the
# DIFFERENCE before scaling it. Two adjacent knots more than the double range
# apart take that difference to `Inf`, and the reported bound comes back `Inf`
# at a probability sitting strictly between two FINITE coordinates -- on the
# issue's node set, a 50% bound of `Inf` between `-9.37e307` and `9.46e307`.
# The convex form `(1 - t) y0 + t y1` cannot overflow there: at `t` in [0, 1]
# each product is bounded by its own knot and the sum lies in `[y0, y1]`.
#
# THE CONVEX FORM IS NOT SUBSTITUTED WHOLESALE. The two are not the same double
# -- the first rounds twice -- and measured over 840000 randomized quantile
# reads (40000 node sets x five declarations x three outside policies x seven
# probabilities, `dev_notes/issue381/measure381.out`) 21.49% of them move, by up
# to 4.07e-10 relatively where the difference cancels. Three fixes on this exact
# path are pinned on
# `identical()` against the read this function returns, so the convex form is
# reached ONLY where the straight one already failed to return a double. Every
# read that returned one keeps the one it returned, byte for byte.
#
# Overflow of the difference is the ONLY way the straight form leaves the double
# range between finite knots, so a bracket whose knots are not both finite is a
# different defect -- an unrepresentable mirrored cell edge, handled at its
# source -- and is reported unchanged rather than repaired
# here.
.nl_interp_repair <- function(p, v, q, y) {
  # `approx` collapses tied `x` to the mean of their `y` before interpolating,
  # so the bracket has to be located on the grid it actually used. `p` is
  # non-decreasing by construction, so the tie groups are adjacent runs.
  if (anyDuplicated(p)) {
    is_first <- c(TRUE, p[-1L] != p[-length(p)])
    grp <- cumsum(is_first)
    v <- as.numeric(tapply(v, grp, mean))
    p <- p[is_first]
  }
  if (length(p) < 2L) return(y)
  i  <- findInterval(q, p, all.inside = TRUE)
  y0 <- v[i]; y1 <- v[i + 1L]
  if (!is.finite(y0) || !is.finite(y1)) return(y)
  dp <- p[i + 1L] - p[i]
  if (!is.finite(dp) || dp <= 0) return(y)
  t <- (q - p[i]) / dp
  (1 - t) * y0 + t * y1
}

# The sorted, de-duplicated, weight-normalized atoms of one axis's node set --
# the ATOMS of the chord read, whose knots are the positive-weight coordinates
# themselves. The box read does NOT go through here: its knots are the cell
# edges, so its partition has to keep a coordinate whose weight underflowed
# (see `.nl_box_quantile()`), and only its masses are filtered.
#
#  * Filters non-finite values and non-positive weights.
#  * Aggregates runs of strictly-equal adjacent values. Cannot use factor(v)
#    here: distinct doubles that share an `as.character` print form (e.g.
#    0.4/0.7 and a near-equal ratio off by ~1e-16) trigger "factor level [k] is
#    duplicated". Group by integer run-IDs derived from numeric equality on the
#    sorted vector.
#  * Returns NULL when nothing usable survives or the total weight is not
#    positive and finite, which every caller reports as NA.
.nl_axis_atoms <- function(values, weights) {
  ord <- order(values)
  v <- as.numeric(values)[ord]
  w <- as.numeric(weights)[ord]
  keep <- is.finite(v) & is.finite(w) & w > 0
  if (!any(keep)) return(NULL)
  v <- v[keep]; w <- w[keep]
  if (length(v) > 1L) {
    is_first <- c(TRUE, v[-1L] != v[-length(v)])
    if (!all(is_first)) {
      grp <- cumsum(is_first)
      w   <- as.numeric(tapply(w, grp, sum))
      v   <- v[is_first]
    }
  }
  w_tot <- sum(w)
  if (!is.finite(w_tot) || w_tot <= 0) return(NULL)
  list(v = v, w = w / w_tot)
}

# The outer edges of the cell partition a sorted vector of cell coordinates
# represents: the extreme cell mirrors the half-spacing it has, in the
# QUANTITY'S OWN COORDINATE -- `.NL_DOMAIN_TRANSFORM[[domain]]`, the registry
# right below, whose `to` / `from` are the monotone map onto the unbounded line
# the mirroring is well defined on. A `positive` quantity mirrors in log and a
# `unbounded` one in the value itself, which is what this did before it was
# given the domain; `unit` mirrors in logit and `correlation` in atanh, and
# those two are the reason for the argument.
#
# Without the domain the coordinate was guessed from the values -- log whenever
# they were all positive -- and a proportion axis is all positive, so a BYM2
# mixing weight whose top node is 0.95 got an upper edge of 1.0353 and the fit
# reported a 97.5% bound ABOVE 1 for a quantity that lives in (0, 1) and is
# singular at both ends. The guess is kept as the fallback for an axis whose
# support the registry will not name (`car_proper`'s `rho_car` on the adjacency
# eigenvalue interval): a guessed edge is what that case had, and inventing a
# support for it would be worse than the edge it has.
#
# A DECLARED SUPPORT IS NOT OVERRULED BY THE GUESS. The guess
# is for an axis whose support nothing named; where a caller DID name one and
# the named coordinate's mirrored edge is unusable, the guess is not filling a
# gap, it is contradicting a declaration -- and it is precisely what produces the
# out-of-support edge, because the two branches compute the same number on a
# `positive` axis and the second one checks only `is.finite()`. Measured on the
# issue's own node set, `c(1e-320, 1e-310, 1)` declared `positive`, the mirrored
# lower edge underflows to exactly 0, the guess reproduces it, and the fit
# reported a lower bound of 5.9e-323 for a quantity whose declared support is
# `x > 0`; from the other end of the double range, `c(1, 1e300, double.xmax)`
# overflowed to `Inf`, fell through the guess to the LINEAR mirror, and reported
# a lower bound of -4.97e+299 for the same declared support.
#
# So the precedence is:
#
#  1. A declared domain that CONTAINS every coordinate is authoritative. Its
#     mirrored edge is taken when finite and in-domain; otherwise the partition
#     DECLINES to the extreme coordinates themselves -- which are inside the
#     support by that same containment test, and are the conservative answer --
#     and records `mirrored_edge_outside_domain`.
#  2. The guess runs only where there is no declaration to contradict: no domain
#     named, a name the registry does not carry, or a node set the declaration
#     does not contain. That last one cannot be honoured by any edge at all: the
#     edges bracket the coordinates, so a coordinate outside the support puts the
#     edge outside it too, and the declaration is the thing that is wrong.
#
# THE GUESS'S OWN MIRROR IS CHECKED TOO. Restricting the guess
# to undeclared axes leaves the surviving branch still not looking at what it
# produced. The linear mirror needs the extreme
# coordinate plus half its own spacing to stay in the double range, and on an
# undeclared axis carrying the top of that range it does not: `c(1, 1e300,
# double.xmax)` mirrored to `Inf` and the DEFAULT chord read reported `Inf` as a
# 97.5% bound, from a partition recording no reason; `c(-double.xmax, 0,
# double.xmax)` mirrored to `-Inf` at the other end and the interpolation between
# a non-finite knot and a finite one reported `NaN`. So the same fallback the
# declared branch takes is taken here -- the extreme coordinates, which are
# representable by construction and are inside whatever support the axis has,
# since they ARE coordinates of it -- and it records
# `mirrored_edge_not_representable`.
#
# That name is a second entry rather than `mirrored_edge_outside_domain`
# because there is no domain here to be outside of: the edge is not a double.
# It TAKES PRECEDENCE over a `nodes_outside_declared_domain` / `unknown_domain`
# already in hand, which are why the GUESS ran; this one is why the guess's edge
# was not taken, and it is the one that describes the edges the reader has. A
# fit reporting either of the first two names has mirrored edges, and one
# reporting a fallback name has the coordinates.
#
# What is NOT guarded is a finite in-order edge the guess placed somewhere the
# axis's real support would not have -- an undeclared all-positive axis whose log
# mirror underflows to exactly 0 keeps that 0. That is the same boundary from
# the other side: with no declaration there is no support to measure the edge
# against, and inventing one is what the engine refuses to do for
# `car_proper`'s `rho_car`.
#
# The invariant that leaves is the one worth asserting: whenever every
# coordinate is inside a declared domain, both edges are finite and inside it
# too, for every entry of `.NL_DOMAIN_TRANSFORM`; and on ANY axis, declared or
# not, both edges are finite and bracket the coordinates.
.nl_cell_edges <- function(v, domain = NA_character_) {
  .nl_cell_partition(v, domain)$edges
}

# The same construction, returning the COORDINATE it settled on alongside the
# edges it produced. The coordinate is not recoverable from the edges -- the
# guard falls back when a mapped edge is non-finite or leaves the support -- and
# the box-uniform read has to bisect the interior spacings in
# the SAME coordinate the outer half-cells were mirrored in, or the partition it
# tiles the axis with is not one partition. So the choice is made once, here,
# and both readers take it from the same return.
#
# `coord` names that coordinate and `declined` is why a mirror did not produce
# the edges, from the closed vocabulary `.NL_EDGE_DECLINED` -- NA when the
# coordinate's own mirror stood, which on an undeclared axis includes the guess,
# since a guess where nothing was declared is the design and not a decline. A
# guess whose mirror is not a representable edge IS one, and says so. That pair
# is the reason field a silent-disable path requires, and
# it travels out to the fit through `.nl_summary_quantile_read()`.
.nl_cell_partition <- function(v, domain = NA_character_) {
  n <- length(v)
  lin <- .NL_DOMAIN_TRANSFORM$unbounded
  part <- function(tr, coord, edges, declined = NA_character_) {
    list(tr = tr, coord = coord, declined = declined, edges = edges)
  }
  if (n < 2L) return(part(lin, "unbounded", c(v[1L], v[n])))
  mirror <- function(u) c(u[1L] - 0.5 * (u[2L] - u[1L]),
                          u[n] + 0.5 * (u[n] - u[n - 1L]))
  named <- length(domain) == 1L && !is.na(domain)
  tr <- if (named) .NL_DOMAIN_TRANSFORM[[domain]] else NULL
  if (!is.null(tr)) {
    if (all(tr$in_domain(v))) {
      e <- tr$from(mirror(tr$to(v)))
      if (all(is.finite(e)) && all(tr$in_domain(e))) {
        return(part(tr, domain, e))
      }
      # The declaration stands and the mirror does not: the extreme coordinates
      # are inside the declared support, so they are the edge.
      return(part(tr, domain, c(v[1L], v[n]), "mirrored_edge_outside_domain"))
    }
    declined <- "nodes_outside_declared_domain"
  } else {
    declined <- if (named) "unknown_domain" else NA_character_
  }
  # Reached only where there is no declaration to contradict. `brackets` is what
  # the guess's own mirror has to satisfy to be an edge at all: finite, and on
  # the outside of the coordinates it brackets. The second half cannot fail for
  # the linear mirror in exact arithmetic and is a rounding guard, so it is
  # cheap and never fires on a partition anything reports.
  brackets <- function(e) {
    all(is.finite(e)) && e[1L] <= v[1L] && e[2L] >= v[n]
  }
  pos <- .NL_DOMAIN_TRANSFORM$positive
  if (all(v > 0)) {
    e <- pos$from(mirror(pos$to(v)))
    if (all(is.finite(e))) return(part(pos, "positive", e, declined))
  }
  e <- mirror(v)
  if (brackets(e)) return(part(lin, "unbounded", e, declined))
  part(lin, "unbounded", c(v[1L], v[n]), "mirrored_edge_not_representable")
}

# Why a mirrored edge did not produce the outer edges of a cell partition. A
# closed vocabulary, held to this list by `test-nl-interval-support.R` the same
# way `outside` is held to `.nl_wtd_quantile()`'s and `within` to
# `.NL_WITHIN_CELL`.
.NL_EDGE_DECLINED <- c("mirrored_edge_outside_domain",
                       "nodes_outside_declared_domain",
                       "unknown_domain",
                       "mirrored_edge_not_representable")

# The subset naming a FALLBACK: the edges are the extreme coordinates, so the
# reported bound is conservative on that side. The complement names a
# declaration that was set aside, after which the guess ran and its mirror
# stood, so the bound is a guessed edge. `.tulpa_interval_read_note()` reads the
# split because the two say opposite things about the bound.
.NL_EDGE_FALLBACK <- c("mirrored_edge_outside_domain",
                       "mirrored_edge_not_representable")

# The cell partition the CHORD read's outer half-cells are mirrored with, on the
# atoms that read uses. Two pure steps -- `.nl_axis_atoms()` then
# `.nl_cell_partition()` -- and they are the same two `.nl_wtd_quantile()` takes
# on the same inputs, so the reported coordinate cannot describe a partition
# other than the one the numbers came out of. NULL when there is no partition to
# report: no usable atom, or one, where no edge is formed at all.
.nl_extend_partition <- function(values, weights, domain = NA_character_) {
  a <- .nl_axis_atoms(values, weights)
  if (is.null(a) || length(a$v) < 2L) return(NULL)
  .nl_cell_partition(a$v, domain)
}

# The full tiling of an axis by the cell partition its coordinates represent:
# `length(v) + 1` edges, cell c owning `[e_c, e_{c+1}]`. The two outer edges are
# `.nl_cell_partition()`'s -- the extreme cell mirroring the half-spacing it has
# -- and the interior ones are the midpoints between adjacent coordinates in
# that same coordinate, which is the cell's own Voronoi interval and is exactly
# the `half_lo` / `half_hi` pair `.joint_local_ccd_diff3()` reports.
#
# The boxes TILE by construction: edge k is both the upper edge of cell k and
# the lower edge of cell k + 1, one number serving both, so there is no gap for
# mass to leave through and no overlap for it to be counted in twice. That is
# the property separating a within-cell reconstruction from barycentring
# (barycentring moves the atoms and contracts the support; nothing moves
# here). What is NOT automatic is that the edges come out finite and strictly
# increasing -- a degenerate coordinate, or a node set the declared support does
# not contain, can produce a zero-width or inverted box, and a box of zero width
# would put a cell's whole mass on a point. So that is checked, and a partition
# that fails it returns NULL for the caller to DECLINE on rather than erroring
# (a silent-disable path needs a reason, and an error is not a
# behaviour a reported interval can take).
#
# A partition whose outer edges are the extreme COORDINATES -- the decline
# taken where the declared domain's mirror was unusable --
# still tiles: edge k serves both neighbours exactly as before, and the two outer
# boxes are half-width with their coordinate on the boundary rather than inside.
# That is the same conservatism the chord read's own clamp takes, so both reads
# agree on the outer support instead of one of them reporting mass past a
# coordinate the other will not.
.nl_box_edges <- function(v, domain = NA_character_) {
  .nl_box_edges_from(.nl_cell_partition(v, domain), v)
}

# The same builder, driven by a partition the caller already has, so
# `.nl_box_quantile()` can report that partition's own coordinate and decline
# without computing it twice.
#
# The interior midpoint is `a / 2 + b / 2` and not `(a + b) / 2`: the sum is
# formed before the halving, so two coordinates near the top of the double
# range take it to `Inf` and the whole partition is declined for a midpoint
# that is perfectly representable. Only the LINEAR coordinate reaches that --
# `log` / `qlogis` / `atanh` land inside about
# +/- 745 and cannot overflow when added -- so it is the `unbounded`
# declaration and the undeclared axis whose values are not all positive.
#
# The three properties, measured in `dev_notes/issue378/midpoint378.R` over 4e6
# randomized pairs spanning the whole double range in both signs:
#
#   * `a / 2 + b / 2` never overflows. Halving a finite double is finite, and
#     the two halves are each at most `double.xmax / 2`, so their sum is in
#     range by construction. 0 spurious non-finite results, against 1086 for
#     the sum form.
#   * It is BYTE-identical to `(a + b) / 2` wherever that one is finite and
#     both halvings are exact -- which is every normal double, so every
#     coordinate any axis actually carries. Halving a normal decrements the
#     exponent and leaves the significand alone, so `a / 2` and `b / 2` are
#     exact and their sum carries ONE rounding of the same real the sum form
#     rounds once; rounding to nearest commutes with scaling by a power of two.
#     0 differences on 3621143 such pairs. It differs only where a halving is
#     subnormal (|x| below about 2.2e-308), by at most 1.0e-320.
#   * It stays inside `[a, b]`: 0 violations on 2e7 subnormal-heavy ordered
#     pairs (`dev_notes/issue378/bracket378.R`).
#
# `a + (b - a) / 2`, the usual overflow-safe form, is neither: `b - a`
# overflows for opposite-sign extremes -- reachable here, since the linear
# coordinate is signed -- and it rounds twice on ORDINARY operands, moving
# 52536 of the same 4e6 pairs off the shipped number.
#
# The decline is kept, not removed. A midpoint that no arithmetic can produce
# still exists -- the outer edges come from `.nl_cell_partition()` and can be
# non-finite where the mirrored half-spacing genuinely leaves the range, and a
# spacing below the coordinate's own resolution still collapses a box to zero
# width -- so a partition that is not finite and strictly increasing returns
# NULL for the caller to DECLINE on.
.nl_box_edges_from <- function(part, v) {
  n <- length(v)
  if (n < 2L) return(NULL)
  u  <- part$tr$to(v)
  ue <- part$tr$to(part$edges)
  e  <- part$tr$from(c(ue[1L], u[-1L] / 2 + u[-n] / 2, ue[2L]))
  if (!all(is.finite(e)) || is.unsorted(e, strictly = TRUE)) return(NULL)
  e
}

# The BOX-UNIFORM within-cell read: each cell's shipped mass spread uniformly
# across its own box instead of
# placed at its coordinate.
#
# The shipped `chord` read puts the cumulative MID-mass `cumsum(w) - w / 2` at
# each cell COORDINATE and interpolates between coordinates; this puts the
# cumulative FULL mass at each cell EDGE and interpolates between edges. Same
# family, same masses, different knots -- and that one difference is a whole
# order of convergence: measured against the closed-form posterior of a
# gaussian-LMM fixture the shipped read converges at order 1.04 and this at 2.00
# (`dev_notes/issue353/RESULTS.md` section 2.3).
#
# THE PARTITION COMES FROM THE GRID, THE MASSES FROM THE WEIGHTS -- which is
# what "keep the masses, tile the axis" says, and the two
# have to be taken from different places. A cell whose integration weight
# underflows to exactly 0 still SITS on the axis, and its coordinate is what
# fixes its neighbour's box edge; dropping it from the partition shrinks that
# neighbour's box to nothing. On the coarsest rung of the ladder
# (`dev_notes/issue357/coarse357b.R`, 2 cells at 400 groups) the softmax
# underflows one of two cells on 43 of 150 seeds, so filtering the coordinates
# by weight collapsed the read onto the chord one there and the rung stopped
# measuring the construction at all. The chord read filters both together
# because its knots ARE the positive-weight coordinates; this one does not,
# because its knots are edges.
#
# A zero-mass box is a FLAT segment of the CDF, not a merged one: the quantile
# is located on the cumulative mass and evaluated inside the box it lands in, so
# an empty box is stepped over rather than interpolated across. Reading the
# inverse off `approx` after dropping tied cumulative values would spread the
# next box's mass over both.
#
# `declined` is why the box read did not run, from a closed vocabulary, and is
# NA when it did. The caller falls back to the chord read on any decline, so an
# axis the partition could not be built for still reports an interval.
.nl_box_quantile <- function(values, weights, probs, domain = NA_character_) {
  v <- as.numeric(values)
  w <- as.numeric(weights)
  fin  <- is.finite(v)
  wpos <- fin & is.finite(w) & w > 0
  if (!any(wpos)) {
    return(list(q = rep(NA_real_, length(probs)), declined = "no_usable_node"))
  }
  uv <- sort(unique(v[fin]))
  if (length(uv) < 2L) {
    return(list(q = rep(uv[1L], length(probs)), declined = "single_node"))
  }
  pt <- .nl_cell_partition(uv, domain)
  e <- .nl_box_edges_from(pt, uv)
  if (is.null(e)) return(list(q = NULL, declined = "boxes_do_not_tile"))
  m <- as.numeric(tapply(w[wpos], factor(match(v[wpos], uv),
                                         levels = seq_along(uv)), sum))
  m[is.na(m)] <- 0
  tot <- sum(m)
  if (!is.finite(tot) || tot <= 0) {
    return(list(q = rep(NA_real_, length(probs)), declined = "no_usable_node"))
  }
  m <- m / tot
  # `cumsum(m)` can round ABOVE 1, and a grid whose trailing cells hold no mass
  # carries that same value on every entry from the last positive cell onward.
  # Forcing only the LAST entry to 1 then makes `cf` DECREASE there, and
  # `findInterval()` refuses a `vec` that is not non-decreasing -- so the read
  # errored on exactly the node sets this construction exists to handle, a grid
  # whose outermost cells' softmax weight underflowed to zero. `pmin` with a
  # constant preserves the order `cumsum` already has, so the tail is clamped
  # rather than one entry contradicted.
  cf <- c(0, pmin(cumsum(m), 1))
  cf[length(cf)] <- 1
  n_box <- length(m)
  q <- vapply(probs, function(p) {
    if (p <= 0) return(e[1L])
    if (p >= 1) return(e[n_box + 1L])
    k <- findInterval(p, cf, rightmost.closed = TRUE, all.inside = TRUE)
    while (k < n_box && m[k] <= 0) k <- k + 1L
    if (m[k] <= 0) return(e[k])
    e[k] + (p - cf[k]) / m[k] * (e[k + 1L] - e[k])
  }, numeric(1))
  list(q = q, declined = NA_character_,
       edge_coord = pt$coord, edge_declined = pt$declined)
}

# The `.NL_DOMAIN_TRANSFORM` entry a DECLARED `c(lower, upper)` support is, or
# NA. The second reader of that vocabulary, alongside `.joint_axis_domains()`:
# where an engine axis's support comes from the per-axis registry, a
# `hyper_axis_spec()` axis carries the user's own `bounds`, and ignoring a
# support the caller declared is worse than having none. Only bounds that ARE
# one of the four domains are recognised -- an arbitrary finite interval
# (`c(0.3, 30)`) is not one of them and reports NA rather than inventing a
# transform for it.
.nl_domain_of_bounds <- function(bounds, log_scale = FALSE) {
  if (is.null(bounds) || length(bounds) != 2L || anyNA(bounds)) {
    return(if (isTRUE(log_scale)) "positive" else NA_character_)
  }
  b <- as.numeric(bounds)
  if (b[1L] == 0 && b[2L] == 1)    return("unit")
  if (b[1L] == -1 && b[2L] == 1)   return("correlation")
  if (b[1L] == 0 && is.infinite(b[2L]) && b[2L] > 0) return("positive")
  if (all(is.infinite(b)))         return("unbounded")
  NA_character_
}

# Monotone maps between a derived quantity's own domain and the unbounded
# coordinate a moment-matched interval is formed on. `positive` covers standard
# deviations, variances, precisions and ranges, `correlation` the (-1, 1)
# interval of a correlation, `unit` the (0, 1) interval of a mixing weight
# (BYM2's rho, a probability), and `unbounded` a covariance entry or a copy
# coefficient, whose sign is free. A registry rather than a branch, so a new
# derived quantity names its domain and inherits the interval.
#
# `in_domain` is the map's own support, checked before it is applied so a
# degenerate node (a collapsed scale, a correlation on the boundary) is reported
# as unsummarizable rather than reaching `log` / `atanh` as a NaN.
.NL_DOMAIN_TRANSFORM <- list(
  positive    = list(to = log,      from = exp,
                     in_domain = function(x) x > 0),
  correlation = list(to = atanh,    from = tanh,
                     in_domain = function(x) abs(x) < 1),
  unit        = list(to = stats::qlogis, from = stats::plogis,
                     in_domain = function(x) x > 0 & x < 1),
  unbounded   = list(to = identity, from = identity,
                     in_domain = function(x) rep(TRUE, length(x)))
)

# Median and interval of a quantity evaluated on a QUADRATURE DESIGN rather than
# sampled from the posterior.
#
# A central-composite design (`ccd_grid()` + `ccd_weights()`) is a moment rule:
# its nodes sit where they reproduce the first two moments of the integrand, and
# their positions carry no probability mass of their own. So the cumulative
# design weight across them is not a CDF, and a discrete weighted quantile over
# them returns an interval bounded by the design's own extent -- at k
# parameters, `theta_hat +/- 1.1 sqrt(k) sd`, whose Gaussian coverage is
# `2 Phi(1.1 sqrt(k)) - 1` no matter how much data there is.
#
# The moments ARE delivered, so the interval comes from them: the first two
# weighted moments on the domain's unbounded coordinate define a Gaussian there
# and its quantiles are mapped back. On a positive quantity that is a lognormal
# interval -- positive, asymmetric, and free to leave the node range -- and on a
# correlation it stays inside (-1, 1). At `probs = 0.5` the Gaussian quantile is
# its mean, so the median is the back-transformed first moment.
#
# Returns NA for every requested probability when no node carries usable weight,
# or when any weighted node falls outside the domain: dropping such a node would
# evaluate the moment rule on a design other than the one that was integrated.
.nl_moment_quantile <- function(values, weights, probs, domain = "unbounded") {
  tr <- .NL_DOMAIN_TRANSFORM[[domain]]
  if (is.null(tr)) {
    stop("unknown derived-quantity domain '", domain, "'.", call. = FALSE)
  }
  use <- is.finite(weights) & weights > 0 & is.finite(values)
  if (!any(use)) return(rep(NA_real_, length(probs)))
  v <- as.numeric(values)[use]
  if (!all(tr$in_domain(v))) return(rep(NA_real_, length(probs)))
  u <- tr$to(v)
  if (!all(is.finite(u))) return(rep(NA_real_, length(probs)))
  w <- as.numeric(weights)[use]
  w <- w / sum(w)
  m <- sum(w * u)
  s <- sqrt(max(0, sum(w * u^2) - m^2))
  tr$from(m + stats::qnorm(probs) * s)
}

# The node-set KINDS a summary can be taken off, and the outer-edge policy each
# one's geometry implies. One table rather than a chain of branches, so a kind
# is named in exactly one place and a caller reading the tag can tell which
# geometry it has.
#
# `density` -- a tensor grid. The weights are proportional to posterior mass and
# the values are CELL REPRESENTATIVES of a partition with known spacing, so the
# cumulative sum is a CDF and the extreme cell's mass reaches half a spacing past
# its coordinate: the read runs to the outer cells' own edges
# (`outside = "extend"`).
#
# `sample` -- equal-weight posterior DRAWS, what `tulpa_re_cov_gibbs()`'s sweep
# produces. The cumulative sum is a CDF here too, so the interior read is the
# same weighted quantile, but the values are ORDER STATISTICS rather than cell
# representatives: beyond the largest draw nothing is known about the tail, and
# half the gap between the two extreme draws is not a cell width. So the outer
# edge CLAMPS, the convention a sample takes. That distinction
# binds only a probability outside
# `[1 / (2 n), 1 - 1 / (2 n)]`, which is why nothing measured moves at the
# 0.025 / 0.5 / 0.975 the backends report.
#
# `moment_rule` -- a central-composite design. Its weights reproduce the
# integrand's moments and the node positions carry no mass, so a cumulative sum
# is not a CDF at all and the interval comes from the moments on the quantity's
# own `domain`. It carries no `outside` policy because it
# never reaches the quantile read. A `moment_rule` quantity whose domain is NA
# has a support the engine will not guess -- a proper-CAR correlation on the
# adjacency eigenvalue interval -- and reports NA rather than the design's
# extent.
#
# `mixed` -- see below; it takes `density`'s policy on purpose.
#
# THE SECOND FIELD, `within`, is the set of WITHIN-CELL constructions the kind
# ADMITS. Order carries no meaning: the engine's default is
# `.nl_diag("within_cell")` and lives in one place, and a kind that does not
# admit it falls back to `chord`, which every kind admits -- so what this field
# has to guarantee is membership, not an ordering. It is a second field
# rather than a fifth kind on purpose: `outside` is a fact about what the
# producer left behind and is derived from the geometry, while `within` is a
# CHOICE the caller makes about how to read it, and the two are orthogonal --
# a density grid can be read either way and still be a density grid. Making
# box-uniform a KIND would have said a fit that asked for it produced a
# different node set, which it did not.
#
# What each kind admits follows from whether its nodes are cell representatives
# of a partition that tiles:
#
#  * `density` -- a tensor grid, and the only kind that admits `box_uniform`:
#    its cells ARE such a partition, with the interior edges at the midpoints
#    and the outer ones at the mirrored half-spacing (`.nl_box_edges()`).
#  * `mixed` -- a locally CCD-refined grid. Its refined cells' replacement
#    clouds sit INSIDE one base cell, so a Voronoi partition of the node set is
#    not the integration design's own boxes and each cloud node's mass would be
#    spread over a box it was not integrated for. That is the tiling property
#    the construction rests on, so the kind declines rather than approximating
#    it.
#  * `sample` -- order statistics. Half the gap between two draws is not a cell
#    width, which is the same reason its `outside` policy clamps.
#  * `moment_rule` -- never reaches the quantile read at all.
.NL_SUPPORT <- list(
  density     = list(outside = "extend",      within = c("box_uniform", "chord")),
  moment_rule = list(outside = NA_character_, within = "chord"),
  mixed       = list(outside = "extend",      within = "chord"),
  sample      = list(outside = "clamp",       within = "chord")
)

.NL_SUPPORT_KINDS <- names(.NL_SUPPORT)

# Every within-cell construction the engine knows, first entry the DEFAULT, so
# an internal caller that names no construction gets the same one the front
# doors resolve through `.nl_within_cell_mode()`. `box_uniform` places the
# cumulative full mass at each cell EDGE and interpolates between edges;
# `chord` places the cumulative mid-mass at each cell COORDINATE and
# interpolates between coordinates, and is both the selectable alternative and
# the fallback every support admits. Which one leads is decided in
# `.NL_DIAG$within_cell` (`R/settings.R`) on the measurement recorded there;
# this vector and `.NL_SUPPORT`'s `within` field have to agree with it, which
# `test-settings.R` and `test-support-sample.R` pin. A kind's own `within` entry
# is the subset it admits, and is held to this vocabulary by
# `test-within-cell-box-uniform.R` the same way `outside` is held to
# `.nl_wtd_quantile()`'s.
.NL_WITHIN_CELL <- c("box_uniform", "chord")

# Median and interval of one quantity, given what KIND of node set carries it.
#
# `support = "mixed"` is the locally CCD-refined grid, the one node set that is
# part cell masses and part quadrature design: the carried-over
# base cells hold their own mass, the refined cells' replacement clouds hold a
# partition-of-unity share of theirs placed at the design's radius. On the design
# part a cumulative sum is not a CDF, so the quantile there reads closer to the
# design's own per-axis extent than to a posterior property.
# It still takes the weighted quantile, because that is what measured best:
# scored against the converged m = 13 tensor reference on the four-axis
# multi-block fixture (noise floor 0.01716 on the endpoints, 0.03853 on the
# widths), summed absolute endpoint error over seven base grids is 0.63446 for
# the quantile against 0.73159 for collapsing each design block to its mean,
# 0.74464 for splitting the read into a mass CDF plus a per-cell moment-matched
# Gaussian, and 1.20402 for the moment read; on analytic outer targets whose
# axis quantiles are known in closed form the same ordering holds in the
# design-dominated regime. So the value of naming the support is that the fit can
# SAY it is mixed and how much of the weight is design (`theta_interval_read` /
# `theta_interval_design_mass`), not that a different formula replaces it. That
# is why `"mixed"` takes the SAME `outside` policy as `"density"` and not the
# conservative one its design nodes would argue for: the tag records provenance,
# and a refined fit reading its interval off a different construction from the
# unrefined fit of the same model is the defect this avoids.
#
# The single dispatcher for every consumer of the summaries, so a caller names
# its support and its domain and inherits the rest.
#
# `within` names the WITHIN-CELL construction. `"box_uniform"`
# is the shipped read and the default, taken where the support admits it and the
# partition builds; anything else falls back to the chord read and the reason
# travels out of `.nl_summary_quantile_read()`. `"mixed"` is one such fallback:
# a refined grid's replacement clouds sit inside one base cell, so a Voronoi
# partition of its node set is not the design's own boxes.
.nl_summary_quantile <- function(values, weights, probs,
                                 domain = NA_character_,
                                 support = .NL_SUPPORT_KINDS,
                                 within = .NL_WITHIN_CELL) {
  .nl_summary_quantile_read(values, weights, probs, domain, support, within)$q
}

# The same dispatch, returning what actually RAN alongside the numbers: the
# construction that produced them and, when the requested one did not, the
# reason. `.nl_summary_quantile()` is this function's `$q` -- there is one
# dispatcher, not two, so a caller wanting only the vector and a caller wanting
# the provenance cannot drift apart.
.nl_summary_quantile_read <- function(values, weights, probs,
                                      domain = NA_character_,
                                      support = .NL_SUPPORT_KINDS,
                                      within = .NL_WITHIN_CELL) {
  support <- match.arg(support)
  within  <- match.arg(within)
  chord <- function(declined = NA_character_) {
    outside <- .NL_SUPPORT[[support]]$outside
    ep <- NULL
    q <- if (!is.na(outside)) {
      # The `domain` reaches the CDF read too, not only the moment rule: the
      # outer half-cell an `extend` support adds is mirrored in the quantity's
      # own coordinate, so a bounded quantity's interval cannot leave its
      # support. A `clamp` support never forms an edge and
      # ignores it.
      if (identical(outside, "extend")) {
        ep <- .nl_extend_partition(values, weights, domain)
      }
      .nl_wtd_quantile(values, weights, probs, outside = outside,
                       domain = domain)
    } else if (length(domain) != 1L || is.na(domain)) {
      .nl_wtd_quantile(values, weights, probs, outside = "na")
    } else {
      .nl_moment_quantile(values, weights, probs, domain)
    }
    list(q = q, within = "chord", declined = declined,
         edge_coord    = ep$coord    %||% NA_character_,
         edge_declined = ep$declined %||% NA_character_)
  }
  if (identical(within, "chord")) return(chord())
  if (!within %in% .NL_SUPPORT[[support]]$within) {
    return(chord(paste0("support_", support)))
  }
  bx <- .nl_box_quantile(values, weights, probs, domain)
  if (is.na(bx$declined)) {
    return(list(q = bx$q, within = within, declined = NA_character_,
                edge_coord    = bx$edge_coord    %||% NA_character_,
                edge_declined = bx$edge_declined %||% NA_character_))
  }
  chord(bx$declined)
}

# What kind of node set the producer left behind. `integration` names what RAN,
# which describes a HOMOGENEOUS support: the central-composite design is a moment
# rule, a Gibbs sweep leaves draws, and the tensor grid and its adaptive subset
# discretize the density. A locally CCD-refined grid carries both kinds at once
# and reports `"grid"`, so the per-cell `weight_kind` tag decides ahead of the
# producer name -- otherwise a mixed support is read as a
# homogeneous density one and nothing downstream can tell.
#
# `"sample"` is the same distinction one layer up from `.NL_SUPPORT`: a sampler
# and a grid both leave a (value, weight) set whose cumulative sum is a CDF, and
# only the producer knows whether the values are order statistics or cell
# representatives, so the producer names itself and the read follows.
.nl_node_support <- function(integration, weight_kind = NULL) {
  if (length(weight_kind) > 1L) {
    kinds <- unique(weight_kind[!is.na(weight_kind)])
    if (length(kinds) > 1L) return("mixed")
  }
  switch(integration %||% "grid",
         ccd    = "moment_rule",
         sample = "sample",
         "density")
}

# What the reported per-axis intervals were read off, and the share of the
# integration weight sitting on nodes whose cumulative sum is not a CDF.
#
# `design_mass` is 0 on a pure density grid and on a sample, 1 on a global CCD,
# and the refined cells' share on a mixed one. It is the regime variable for the
# mixed read: the larger it is, the more of the reported interval comes from a
# moment rule being read as a CDF. Returned together so every consumer surfaces
# the same pair.
.nl_interval_provenance <- function(integration, weight_kind = NULL,
                                    weights = NULL) {
  read <- .nl_node_support(integration, weight_kind)
  dm <- if (identical(read, "moment_rule")) {
    1
  } else if (identical(read, "mixed") && !is.null(weights) &&
             length(weights) == length(weight_kind)) {
    sum(weights[weight_kind == "design"], na.rm = TRUE)
  } else if (identical(read, "mixed")) {
    NA_real_
  } else {
    0
  }
  list(read = read, design_mass = dm)
}

# Summary of a weighted mixture of Gaussians, one mixture per parameter.
#
# The continuous counterpart of `.nl_wtd_quantile`: where that summarizes a
# discrete (value, weight) set -- one number per grid cell -- this summarizes a
# posterior whose cells each contribute a whole Gaussian,
#   p(x_j) = sum_i w_i N(x_j; mu_ij, var_ij),
# which is what a nested-Laplace grid actually produces for a latent coefficient
# (per-node conditional mean and curvature, mixed by the node weights). Reporting
# only the spread BETWEEN nodes omits the within-node curvature and understates
# the posterior SD; reporting only one node's Gaussian omits the hyperparameter
# uncertainty the grid exists to carry. Both terms enter here.
#
# `mu` and `var` are n_node x n_par (a column is one parameter, a row one node);
# `w` is the node weight vector. Mean and SD are the exact mixture moments (law
# of total variance), and the quantiles invert the exact mixture CDF by bisection
# rather than assuming normality -- a mixture over a skewed hyperparameter
# posterior is itself skewed, so `mean +/- 1.96 sd` is not its interval. The
# bracket spans +/- 10 component SDs, which contains every quantile of the
# mixture for any `probs` in practice.
#
# Returns list(mean, sd, quantiles = n_par x length(probs)), or NULL when no node
# is usable. `var = NULL`, or variances that are missing / negative / non-finite,
# give the mixture mean with NA for SD and the quantiles: the between-node spread
# alone is not the posterior SD, so it is withheld rather than reported small.
.nl_gauss_mixture_summary <- function(mu, var, w, probs = c(0.025, 0.5, 0.975),
                                      n_bisect = 80L) {
  mu <- as.matrix(mu)
  w  <- as.numeric(w)
  np <- ncol(mu)
  ok <- is.finite(w) & w > 0 & is.finite(rowSums(mu))
  if (!any(ok) || np == 0L) return(NULL)
  mu <- mu[ok, , drop = FALSE]
  w  <- w[ok]; w <- w / sum(w)
  mean_p <- as.numeric(crossprod(w, mu))

  sdm <- NULL
  if (!is.null(var)) {
    v <- as.matrix(var)[ok, , drop = FALSE]
    if (all(is.finite(v)) && all(v >= 0)) sdm <- sqrt(v)
  }
  na_q <- matrix(NA_real_, np, length(probs))
  if (is.null(sdm)) {
    return(list(mean = mean_p, sd = rep(NA_real_, np), quantiles = na_q))
  }

  sd_p <- sqrt(pmax(as.numeric(crossprod(w, mu^2 + sdm^2)) - mean_p^2, 0))

  # A degenerate component (zero curvature-implied SD) is a step function in the
  # CDF; the floor keeps it one without branching the vectorized evaluation.
  s <- pmax(sdm, 1e-300)
  cdf <- function(q) {
    as.numeric(crossprod(w, stats::pnorm(
      matrix(q, nrow(mu), np, byrow = TRUE), mu, s)))
  }
  lo0 <- apply(mu - 10 * sdm, 2L, min)
  hi0 <- apply(mu + 10 * sdm, 2L, max)
  qs <- vapply(probs, function(p) {
    lo <- lo0; hi <- hi0
    for (it in seq_len(n_bisect)) {
      mid <- (lo + hi) / 2
      below <- cdf(mid) < p
      lo <- ifelse(below, mid, lo)
      hi <- ifelse(below, hi, mid)
    }
    (lo + hi) / 2
  }, numeric(np))
  list(mean = mean_p, sd = sd_p,
       quantiles = matrix(qs, np, length(probs)))
}

# Fixed-effect credible bounds on a nested-Laplace fit, and the provenance of
# whichever read produced them.
#
# The grid defines a Gaussian mixture per coefficient (see
# `.nested_fixed_moments()`). Its mean and variance are linear functionals and
# survive the collapse to one Gaussian; a quantile is nonlinear and does not, so
# the bounds invert the mixture CDF
#   F_j(b) = sum_k w_k Phi((b - mu_kj) / sqrt(V_kjj))
# through `.nl_gauss_mixture_summary()` -- the same construction `ranef()`
# already reports the per-group posterior with. `estimate`, `std.error` and
# `vcov()` are untouched: they are the moments either way.
#
# WHY THE SKEW CORRECTION IS NOT COMPOSED WITH THIS. Its gamma_3 is
# computed by re-dispatching the kernel at the single fitted MAP cell, so the
# fit retains one gamma_3 per coefficient and not one per cell. The composed
# marginal sum_k w_k F^CF_kj is therefore not identified by retained state: it
# would need gamma_3(k, j). The three available substitutes are each an
# unbacked assertion -- a scalar Cornish-Fisher shift of the mixture quantile
# double-counts, since the mixture already carries the across-cell asymmetry;
# applying the MAP cell's gamma_3 to every component asserts the conditional
# skew is constant over the grid; applying it to the dominant component alone
# privileges one component with no approximation theorem behind it. So a
# CORRECTED coefficient keeps the read it was measured on, and `declined` says
# why the mixture read did not run there. The two corrections address different
# non-Gaussianities: this one across cells, the other within the MAP cell.
#
# A DECLINED coefficient is a different case and keeps the mixture read. There
# is no corrected read to preserve at a coefficient the correction refuses, so
# falling back to the collapsed Gaussian would give up the across-cell shape
# for nothing -- and on a fit where every coefficient declines (a coupled one,
# whose gamma_1 is not reachable) it would move every bound while correcting
# none. Enabling the correction moves exactly the coefficients it applies to,
# which `skew_applied` names per row and `interval_source` names as
# `"skew_map_cell/mixture_cdf"` when both reads are in play.
#
# A GRID THAT DROPPED A POSITIVE-WEIGHT CELL still gets the mixture read. The
# moments renormalize over the cells that retained a block, so the components
# here carry the same weighting the reported `estimate` and `std.error` were
# formed under and the two reads describe one posterior -- the posterior
# CONDITIONAL ON THE RETAINED CELLS, not the full grid's, which the dropped mass
# is gone from either way. `mass` travels with the result and is the original
# retained share of the grid weight, so a caller reading the bounds can always
# tell a complete grid from a repaired one.
#
# `mom` is the `.nested_fixed_moments()` list, `idx` the reported coefficient
# columns, `est` / `se` their moments, `probs` the requested levels, `sc` the
# `.nl_skew_correction()` state. Returns list(q = length(est) x length(probs),
# applied, source, declined, mass).
.nl_fixed_interval <- function(mom, idx, est, se, probs, sc) {
  mass <- if (is.numeric(mom$mass) && length(mom$mass) == 1L) {
    as.numeric(mom$mass)
  } else NA_real_
  gaussian <- function(source, declined) {
    list(q = est + outer(se, stats::qnorm(probs)),
         applied = rep(FALSE, length(est)),
         source = source, declined = declined, mass = mass)
  }
  # The read every coefficient takes before the correction is consulted: the
  # mixture quantiles where the grid retained usable components, the collapsed
  # Gaussian where it did not.
  base <- if (is.null(mom$mu) || is.null(mom$var) || is.null(mom$w)) {
    gaussian("gaussian_moment", "no retained mixture components")
  } else if (ncol(mom$mu) < max(idx)) {
    gaussian("gaussian_moment",
             "retained component block is narrower than the reported coefficients")
  } else {
    mx <- .nl_gauss_mixture_summary(mom$mu[, idx, drop = FALSE],
                                    mom$var[, idx, drop = FALSE],
                                    mom$w, probs = probs)
    if (is.null(mx) || anyNA(mx$quantiles)) {
      gaussian("gaussian_moment",
               "component means or variances are not all usable")
    } else {
      list(q = mx$quantiles, applied = rep(FALSE, length(est)),
           source = "mixture_cdf", declined = NA_character_, mass = mass)
    }
  }
  if (!isTRUE(sc$enabled)) return(base)

  # A CORRECTED coefficient takes the skew-corrected read, which cannot be
  # composed with the mixture (above). A DECLINED one has none at all, so it
  # keeps
  # whatever the fit reports with the correction off -- the base read, not the
  # collapsed Gaussian. Enabling the correction therefore moves exactly the
  # coefficients it applies to, which is what lets it be a default: a fit whose
  # every coefficient declines (a coupled one, where gamma_1 is not reachable)
  # reports the same bounds either way.
  mg <- .nl_skew_marginal(est, se, .nl_skew_gamma3_eligible(sc)[idx],
                          .nl_skew_gamma1_eligible(sc)[idx], probs,
                          enabled = TRUE)
  if (!any(mg$applied)) return(base)
  q <- base$q
  q[mg$applied, ] <- mg$q[mg$applied, , drop = FALSE]
  list(q = q, applied = mg$applied,
       source = if (all(mg$applied)) "skew_map_cell"
                else paste0("skew_map_cell/", base$source),
       declined = paste("skew_correct: gamma_3 is retained at the MAP cell",
                        "only, so the mixture components carry no per-cell",
                        "skew to compose; the coefficients it declines keep",
                        "the read they would have had"),
       mass = mass)
}

# Weighted mean and SD of `values` under pre-normalized `weights` (which
# sum to 1). SD uses the E[x^2] - E[x]^2 form, floored at 0 to absorb the
# floating-point negatives that form can produce near a degenerate axis.
.nl_wtd_mean_sd <- function(values, weights) {
  mu <- sum(weights * values)
  list(mean = mu, sd = sqrt(max(0, sum(weights * values^2) - mu^2)))
}

# Weighted-quantile median + 2.5/97.5 empirical CI for every axis of a
# (scalar or matrix) theta_grid. Returns named numeric vectors so the
# joint nested-Laplace surface exposes `theta_median / theta_ci_lo /
# theta_ci_hi` alongside `theta_mean / theta_sd` for every hyperparameter.
#
# `tg` is a vector or matrix; `log_marginal` aligns with `tg` rows;
# `refining` is the per-cell refining-axis tag from mode-tracked
# refinement (NULL or all-"" outside the joint path). For each axis,
# slice cells from OTHER axes are dropped before computing the quantile
# -- those cells pin the current axis at a single non-varying value, so
# including them oversamples that value. Cartesian cells, same-axis
# slice cells, and same-axis consistency cells are kept. This is the
# same per-axis mask used by `.joint_recalibrate_axis_moments` for the
# mean/SD path.
#
# Returns list(median = named_vec, ci_lo = named_vec, ci_hi = named_vec).
# For scalar tg the returned vectors are length-1 with names = "value".
#
# `support` says what the supplied `weights` are and is passed through to
# `.nl_summary_quantile`; `domains` gives one `.NL_DOMAIN_TRANSFORM` name per
# axis. A `"moment_rule"` support needs it to form its interval at all (an axis
# whose support the engine will not guess carries NA and reports NA); a
# `"density"` / `"mixed"` one needs it to place its outer cell edges inside the
# quantity's support, so it is supplied WHATEVER the support
# rather than only for the design read.
#
# `within` is the requested within-cell construction, and the returned `within`
# / `within_declined` say what each axis actually got: an
# axis whose partition could not be built falls back to the chord read on its
# own rather than taking the whole fit with it, so a fit can carry a mixture of
# constructions and still say which produced each interval.
#
# `edge_coord` / `edge_declined` are the same statement one layer down: the
# COORDINATE that axis's outer half-cells were mirrored in, and -- when the
# axis declared a support whose own mirror was not usable -- why the declared
# one did not produce them. They come from whichever read ran, since the chord
# and box reads take their partitions off different atom sets.
.nl_axis_quantiles <- function(tg, log_marginal, refining = NULL,
                                probs = c(0.025, 0.5, 0.975),
                                weights = NULL,
                                support = .NL_SUPPORT_KINDS,
                                domains = NULL,
                                within = .NL_WITHIN_CELL) {
  support <- match.arg(support)
  within  <- match.arg(within)
  if (is.null(dim(tg))) {
    tg <- matrix(as.numeric(tg), ncol = 1L,
                 dimnames = list(NULL, "value"))
  }
  # Empty grid (no outer-grid axes -- e.g. an lf-only fit): nothing to
  # quantilize; return empty named vectors.
  if (ncol(tg) == 0L) {
    empty <- setNames(numeric(0), character(0))
    empty_c <- setNames(character(0), character(0))
    return(list(median = empty, ci_lo = empty, ci_hi = empty,
                within = empty_c, within_declined = empty_c,
                edge_coord = empty_c, edge_declined = empty_c,
                outside_nodes = empty_c))
  }
  nms <- .nl_axis_names(tg)
  n_ax <- length(nms)
  lo  <- setNames(rep(NA_real_, n_ax), nms)
  med <- setNames(rep(NA_real_, n_ax), nms)
  hi  <- setNames(rep(NA_real_, n_ax), nms)
  wc  <- setNames(rep(NA_character_, n_ax), nms)
  wcd <- setNames(rep(NA_character_, n_ax), nms)
  ec  <- setNames(rep(NA_character_, n_ax), nms)
  ecd <- setNames(rep(NA_character_, n_ax), nms)
  # Did the reported bound leave the axis the grid was actually laid on?
  # An endpoint past the outermost NODE is produced by the `outside` rule
  # (`extend` mirrors a half-cell beyond it), so it is an extrapolation rather
  # than a bound the design supports -- and on a diffuse axis that is the common
  # case, not the exception: measured over 48 (cap, span, node-count, policy)
  # rungs on two fixtures the 95% bound sits outside the nodes on 56% to 90% of
  # fits at EVERY setting, while the 50% bound never does.
  # Recorded rather than corrected, and per axis, because which it is changes
  # what the number means: the same rule that makes a declined placement
  # say so.
  onn <- setNames(rep(NA_character_, n_ax), nms)
  if (is.null(refining)) refining <- rep("", nrow(tg))
  for (j in seq_len(n_ax)) {
    ax    <- nms[j]
    keep  <- refining == "" | refining == ax |
             refining == paste0("consistency_", ax)
    use   <- keep & is.finite(tg[, j])
    if (sum(use) == 0L) next
    # Precomputed integration weights (CCD design weights * exp(log-marginal),
    # passed for scattered node sets where the per-axis softmax of the raw
    # log-marginal is not the integration weight); otherwise the regular-grid
    # softmax of the log-marginal.
    if (!is.null(weights)) {
      ws  <- weights[use]
      if (!any(is.finite(ws)) || sum(ws, na.rm = TRUE) <= 0) next
      ws[!is.finite(ws)] <- 0
      ws  <- ws / sum(ws)
    } else {
      lm_u  <- log_marginal[use]
      m     <- max(lm_u)
      if (!is.finite(m)) next
      ws    <- exp(lm_u - m); ws <- ws / sum(ws)
    }
    dm <- if (length(domains) < j) NA_character_ else domains[[j]]
    rd <- .nl_summary_quantile_read(as.numeric(tg[use, j]), ws, probs, dm,
                                    support, within)
    qs <- rd$q
    lo[j]  <- qs[1L]
    med[j] <- qs[2L]
    hi[j]  <- qs[3L]
    wc[j]  <- rd$within
    wcd[j] <- rd$declined
    ec[j]  <- rd$edge_coord    %||% NA_character_
    ecd[j] <- rd$edge_declined %||% NA_character_
    # Absolute tolerance scaled by the axis's own extent, not a multiplicative
    # one: an axis whose support straddles zero (a correlation) has no scale to
    # multiply by.
    rng <- range(as.numeric(tg[use, j]))
    tol <- 1e-9 * max(abs(rng), diff(rng), 1)
    below <- is.finite(lo[j]) && lo[j] < rng[1L] - tol
    above <- is.finite(hi[j]) && hi[j] > rng[2L] + tol
    onn[j] <- if (below && above) "both" else if (below) "lower" else
              if (above) "upper" else NA_character_
  }
  list(median = med, ci_lo = lo, ci_hi = hi,
       within = wc, within_declined = wcd,
       edge_coord = ec, edge_declined = ecd,
       outside_nodes = onn)
}

# The RESOLUTION of each outer-grid axis: its cell width `h`, the posterior SD
# on that axis, and their ratio -- both in the axis's OWN coordinate, the one
# `.nl_cell_partition()` lays the boxes out in, so the ratio is a pure number
# and a geometric sigma axis is measured in log where its spacing is constant.
#
# WHY A FIT REPORTS THIS. Every within-cell construction
# resolves an interval endpoint to within one cell, so the realized coverage of
# a reported interval depends on WHERE inside its cell the unknown truth fell,
# and how much it depends on that is governed by `h / sd`. Measured on the
# gaussian-LMM fixture, conditional 95% coverage swings 0.415 across one cell at
# `h / sd = 3.41` and 0.100 at 2.43; the shipped chord read has the same
# dependence and hides it behind width (its own conditional coverage at nominal
# 0.50 runs 0.655 to 0.950 across the same cell). `h / sd` is the regime
# variable for both, it costs one 3-point parabola per axis, and a fit that
# does not report it leaves the reader with no way to tell a resolved axis from
# an unresolved one. Below `.nl_diag("grid_resolved")` the two constructions
# converge to each other and the question stops mattering.
#
# `h` is the MEDIAN spacing rather than the first one: an adaptive or
# consistency-refined axis is no longer equally spaced, and one representative
# width is what the ratio is about. `sd` is the Laplace-at-mode SD of the axis
# marginal in the same coordinate.
#
# An axis that could not be scored says WHY, in `declined`, from the closed
# vocabulary `.NL_AXIS_SD_REASONS`. The reasons are not
# interchangeable: `mode_at_edge` is the grid failing to contain that axis's own
# posterior mode, a stronger statement about the fit than any ratio, while
# `too_few_nodes` is an axis too short to fit a parabola on. Reported per axis
# rather than folded into the NA, because the reader's next move differs.
.nl_axis_resolution <- function(tg, log_marginal, refining = NULL,
                                domains = NULL) {
  if (is.null(dim(tg))) {
    tg <- matrix(as.numeric(tg), ncol = 1L, dimnames = list(NULL, "value"))
  }
  nms <- .nl_axis_names(tg)
  n_ax <- length(nms)
  h  <- setNames(rep(NA_real_, n_ax), nms)
  sd <- setNames(rep(NA_real_, n_ax), nms)
  dec <- setNames(rep(NA_character_, n_ax), nms)
  if (n_ax == 0L) {
    return(list(h = h, sd = sd, h_over_sd = h, declined = dec))
  }
  if (is.null(refining)) refining <- rep("", nrow(tg))
  for (j in seq_len(n_ax)) {
    ax <- nms[j]
    keep <- refining == "" | refining == ax |
            refining == paste0("consistency_", ax)
    marg <- .nl_axis_marginal_logdensity(tg[, j], log_marginal, keep)
    v <- marg$vals
    if (length(v) < 3L) {
      dec[j] <- "too_few_nodes"
      next
    }
    dm <- if (length(domains) < j) NA_character_ else domains[[j]]
    part <- .nl_cell_partition(v, dm)
    u <- part$tr$to(v)
    if (!all(is.finite(u))) {
      dec[j] <- "coord_not_finite"
      next
    }
    du <- diff(u)
    if (!length(du) || !all(is.finite(du))) {
      dec[j] <- "spacing_not_finite"
      next
    }
    h[j]  <- stats::median(du)
    # The reason rides on the NA as an attribute, which assignment into `sd`
    # would drop, so it is read off the return before that.
    s <- .nl_laplace_at_mode_sd_axis(v, marg$log_marg, coord = part$tr,
                                     return_u_sd = TRUE)
    sd[j]  <- as.numeric(s)
    dec[j] <- .nl_axis_sd_reason(s)
  }
  list(h = h, sd = sd, h_over_sd = h / sd, declined = dec)
}

# Stamp onto a fit what its reported per-axis intervals were read off: the
# node-set KIND, the design share underneath it, the within-cell CONSTRUCTION
# per axis and why a requested one declined, the COORDINATE each axis's outer
# cell edges were mirrored in and why a declared support's own mirror did not
# produce them, and the per-axis resolution the
# construction's position sensitivity is governed by.
#
# The kind is filled only when the producer has not already named it: the
# multi-block joint driver stamps `theta_interval_read` before the moments run
# (it is the one path whose support is not homogeneous), and every other path --
# single-block, joint single-block, registry, ST -- leaves a plain tensor grid
# and had nothing stamped at all, so a reader could not
# distinguish "density" from "this fit does not say". `.nl_node_support()` is
# the same call `.ogd_support()` was falling back to for exactly that reason.
# The resolution is reported only for a `density` support. It is a statement
# about a CELL PARTITION's spacing, and the other kinds do not have one: a
# central-composite design's nodes carry no cells, a locally refined grid's
# clouds sit inside one base cell, and a sample's values are draws. The SD side
# is the same 3-point lattice profile `.nl_refit_axis_sd_laplace()` is skipped
# for on a design-weighted grid, for the same reason.
.nl_attach_interval_provenance <- function(res, qs, tg, domains = NULL) {
  prov <- .nl_interval_provenance(res$integration, res$weight_kind,
                                  res$weights)
  res$theta_interval_read <- res$theta_interval_read %||% prov$read
  res$theta_interval_design_mass <-
    res$theta_interval_design_mass %||% prov$design_mass
  res$theta_within_cell          <- qs$within
  res$theta_within_cell_declined <- qs$within_declined
  res$theta_ci_outside_nodes     <- qs$outside_nodes
  res$theta_cell_edge_coord      <- qs$edge_coord
  res$theta_cell_edge_declined   <- qs$edge_declined
  if (identical(res$theta_interval_read, "density")) {
    rs <- .nl_axis_resolution(tg, res$log_marginal, res$refining_axis, domains)
    res$outer_grid_cell_width <- rs$h
    res$outer_grid_axis_sd    <- rs$sd
    res$outer_grid_h_over_sd  <- rs$h_over_sd
    res$outer_grid_resolution_declined <- rs$declined
  }
  # An axis whose grid does not contain its own mode is a placement fact, and
  # `.nl_axis_rail()` reads it off the stored weights alone. Attached here so a
  # CALLER-PINNED grid reports it too: the rescue that used to
  # be its only attach point never runs on one, which is the placement that
  # most needs its provenance said out loud.
  res$outer_grid_railed_axes <- res$outer_grid_railed_axes %||%
    .nl_railed_axes(res)
  res
}

# The within-cell construction a fit was asked for, from its own control list.
# One resolver, so every front door spells the knob the same way and an unknown
# value is refused at the door rather than silently read as the default.
.nl_within_cell_mode <- function(x) {
  if (is.null(x)) return(.nl_diag("within_cell"))
  match.arg(as.character(x), .NL_WITHIN_CELL)
}

# The per-axis names of an outer grid. `paste0` recycles a scalar prefix past a
# zero-length integer, so `paste0("V", seq_len(0))` is the length-1 "V" and not
# `character(0)`: an unnamed 0-column grid -- what a fit whose only block
# carries no outer axis produces, e.g. `type = "lf"` -- would otherwise report
# one phantom axis and index past the grid. Reading the count off `ncol()`
# rather than off the generated names is what makes an empty grid empty here.
.nl_axis_names <- function(tg) {
  d <- ncol(tg)
  if (d == 0L) return(character(0))
  colnames(tg) %||% paste0("V", seq_len(d))
}

.nl_axis_marginal_logdensity <- function(vals, log_marg, keep = NULL) {
  if (is.null(keep)) keep <- rep(TRUE, length(vals))
  v <- vals[keep]; l <- log_marg[keep]
  uv <- sort(unique(v))
  if (length(uv) == 0L) return(list(vals = uv, log_marg = numeric(0)))
  lm <- vapply(uv, function(u) {
    li <- l[v == u]
    if (length(li) == 0L) return(-Inf)
    m <- max(li)
    if (!is.finite(m)) return(-Inf)
    m + log(sum(exp(li - m)))
  }, numeric(1))
  list(vals = uv, log_marg = lm)
}

# Laplace-at-mode SD on a single axis. Fits a 3-point parabola to the
# marginal log-density at the modal cell and one neighbour on each side,
# returns sqrt(-1 / (2 a)) from the quadratic coefficient. When the
# axis values are positive (sigma / phi / tau) the parabola is fit on
# log(theta), and the SD is mapped back to the linear axis via the
# delta method (sigma_theta = theta_mode * sigma_log_theta). Returns NA
# when the mode sits at an axis edge or the parabola is concave up.
#
# `coord` overrides that guess with a `.NL_DOMAIN_TRANSFORM` entry, so the
# parabola is fit in the coordinate the caller's own cell partition lives in --
# `unit` for a mixing weight, `correlation` for a correlation -- rather than in
# whichever one `all(vals > 0)` happens to pick.
#
# `return_u_sd = TRUE` skips the delta back-map and returns the SD in that
# coordinate directly, which is what a spacing-relative report needs
# (`.nl_axis_resolution()`): `h` is measured there too, so the ratio is a pure
# number. The back-map branch is the log delta method (`theta * sd_u`) and is
# taken only on the default coordinate, where it is the one that applies.
#
# The five conditions that withhold the number are DISTINGUISHABLE, and a bare
# `NA_real_` conflated them. `mode_at_edge` in particular is
# not a missing measurement: it says the grid does not contain the axis's own
# posterior mode, which is a stronger statement about the fit than any ratio
# this function could return. The reason rides on the NA as an attribute so
# every existing `is.finite()` caller is unaffected and only a reader that wants
# it pays attention to it.
.NL_AXIS_SD_REASONS <- c("too_few_nodes", "mode_at_edge", "coord_not_finite",
                         "stencil_degenerate", "curvature_not_negative")

# Everything an axis's resolution read can decline on: the SD estimator's own
# five, plus the one `.nl_axis_resolution()` reaches before it calls the
# estimator at all.
.NL_AXIS_RESOLUTION_REASONS <- c(.NL_AXIS_SD_REASONS, "spacing_not_finite")

.nl_axis_sd_declined <- function(reason) {
  structure(NA_real_, tulpa_reason = match.arg(reason, .NL_AXIS_SD_REASONS))
}

# The reason carried by a declined axis SD, `NA_character_` for a finite one.
.nl_axis_sd_reason <- function(x) {
  r <- attr(x, "tulpa_reason", exact = TRUE)
  if (is.null(r)) NA_character_ else as.character(r)
}

.nl_laplace_at_mode_sd_axis <- function(vals, log_marg, log_axis = NULL,
                                        return_u_sd = FALSE, coord = NULL) {
  if (length(vals) < 3L) return(.nl_axis_sd_declined("too_few_nodes"))
  ix <- which.max(log_marg)
  if (ix == 1L || ix == length(vals)) {
    return(.nl_axis_sd_declined("mode_at_edge"))
  }
  if (is.null(log_axis)) log_axis <- all(is.finite(vals)) && all(vals > 0)
  u <- if (!is.null(coord)) coord$to(vals) else if (log_axis) log(vals) else vals
  if (!all(is.finite(u))) return(.nl_axis_sd_declined("coord_not_finite"))
  dm <- u[ix - 1L] - u[ix]
  dp <- u[ix + 1L] - u[ix]
  det <- dm * dp * (dm - dp)
  if (!is.finite(det) || abs(det) < .Machine$double.eps) {
    return(.nl_axis_sd_declined("stencil_degenerate"))
  }
  lm_m <- log_marg[ix - 1L] - log_marg[ix]
  lm_p <- log_marg[ix + 1L] - log_marg[ix]
  a <- (lm_m * dp - lm_p * dm) / det
  if (!is.finite(a) || a >= 0) {
    return(.nl_axis_sd_declined("curvature_not_negative"))
  }
  sd_u <- sqrt(-1 / (2 * a))
  if (return_u_sd) {
    sd_u
  } else {
    if (log_axis) vals[ix] * sd_u else sd_u
  }
}

# Replace `theta_sd` (and `block_moments[[b]]$sd` when present) entries
# with the Laplace-at-mode SD wherever the 3-point fit succeeds. Axes
# with the mode at an edge or wrong-signed curvature keep their var-of-
# means SD. Grid-spacing-independent: fixes the symptom where a sharply
# peaked marginal log-likelihood on a coarse grid collapses var-of-
# means to ~0 and undercovers.
.nl_refit_axis_sd_laplace <- function(res, refining = NULL) {
  if (is.null(res$theta_grid) || is.null(res$log_marginal)) return(res)
  # The per-axis marginal is an integral against the outer prior measure, so the
  # node weights belong in it. Without them the curvature at the mode is read
  # off the node counts instead, and moves when refinement changes the spacing.
  lm_eff <- res$log_marginal
  if (!is.null(res$log_quad) && length(res$log_quad) == length(lm_eff)) {
    lm_eff <- lm_eff + res$log_quad
    lm_eff[is.na(lm_eff)] <- -Inf
  }
  tg <- res$theta_grid
  if (!is.matrix(tg)) {
    marg <- .nl_axis_marginal_logdensity(as.numeric(tg), lm_eff)
    sd_lam <- .nl_laplace_at_mode_sd_axis(marg$vals, marg$log_marg)
    if (is.finite(sd_lam)) res$theta_sd <- sd_lam
    return(res)
  }
  if (is.null(refining)) refining <- res$refining_axis
  if (is.null(refining)) refining <- rep("", nrow(tg))
  col_names <- colnames(tg)
  if (!is.null(col_names) && !is.null(res$theta_sd)) {
    for (col in col_names) {
      if (!col %in% names(res$theta_sd)) next
      keep <- refining == "" | refining == col |
              refining == paste0("consistency_", col)
      marg <- .nl_axis_marginal_logdensity(tg[, col], lm_eff, keep)
      sd_lam <- .nl_laplace_at_mode_sd_axis(marg$vals, marg$log_marg)
      if (is.finite(sd_lam)) res$theta_sd[[col]] <- sd_lam
    }
  }
  if (!is.null(res$block_moments)) {
    for (b in seq_along(res$block_moments)) {
      bm <- res$block_moments[[b]]
      axis_cols <- bm$axis_cols
      if (is.null(axis_cols) || length(axis_cols) == 0L) next
      bare <- names(bm$sd)
      for (j in seq_along(axis_cols)) {
        col_ix <- axis_cols[j]
        col_name <- if (!is.null(col_names)) col_names[col_ix] else ""
        keep <- refining == "" | refining == col_name |
                refining == paste0("consistency_", col_name)
        marg <- .nl_axis_marginal_logdensity(tg[, col_ix], lm_eff,
                                              keep)
        sd_lam <- .nl_laplace_at_mode_sd_axis(marg$vals, marg$log_marg)
        if (is.finite(sd_lam)) res$block_moments[[b]]$sd[[j]] <- sd_lam
      }
    }
  }
  res
}

# Posterior moments for multi-block grids. Two flavours:
#  * joint moments: across all axes (same as single-block 2D scatter).
#  * per-block marginal moments: integrate out the other blocks' axes.
.nl_posterior_moments_multi <- function(out, prepared, axis_offsets, joint_grid,
                                        within = .NL_WITHIN_CELL) {
  within <- match.arg(within)
  w  <- out$weights
  tg <- joint_grid
  out$theta_mean <- as.numeric(crossprod(w, tg))
  names(out$theta_mean) <- colnames(tg)
  out$theta_sd <- sqrt(pmax(0, as.numeric(crossprod(w, tg^2)) -
                              out$theta_mean^2))
  names(out$theta_sd) <- colnames(tg)

  # Per-block marginals: for each block, sum weights over rows that share the
  # same per-block axis values, then take weighted mean/sd within those rows.
  # Because the joint grid is a Cartesian product, the "rows that share this
  # block's values" form n_block_rows groups indexed by idx[, b]. The marginal
  # weight for group g is sum of joint weights over its rows.
  B <- length(prepared)
  per_block_moments <- vector("list", B)
  for (b in seq_len(B)) {
    cols <- (axis_offsets[b] + 1L):axis_offsets[b + 1L]
    sub <- joint_grid[, cols, drop = FALSE]
    block_mean <- as.numeric(crossprod(w, sub))
    block_sd   <- sqrt(pmax(0, as.numeric(crossprod(w, sub^2)) - block_mean^2))
    # Use bare per-block axis names (e.g. "tau", "rho") instead of the
    # joint-grid-prefixed "b<N>.tau" -- the block index is already implicit
    # in the list position.
    bare_names <- .nl_block_axis_grid(prepared[[b]])$names
    names(block_mean) <- bare_names
    names(block_sd)   <- bare_names
    per_block_moments[[b]] <- list(
      type      = tolower(prepared[[b]]$type),
      mean      = block_mean,
      sd        = block_sd,
      axis_cols = cols
    )
  }
  out$block_moments <- per_block_moments

  # Weighted-quantile median + 2.5/97.5 CI per axis (calibrated summary
  # for right-skewed scale-like hyperparameters; see `.nl_axis_quantiles`).
  doms <- .joint_axis_domains(list(theta_grid = joint_grid,
                                   axis_offsets = axis_offsets,
                                   blocks = prepared))
  qs <- .nl_axis_quantiles(
    joint_grid, out$log_marginal, out$refining_axis,
    domains = doms, within = within)
  out$theta_median <- qs$median
  out$theta_ci_lo  <- qs$ci_lo
  out$theta_ci_hi  <- qs$ci_hi
  out$within_cell_requested <- within
  .nl_attach_interval_provenance(out, qs, joint_grid, doms)
}
