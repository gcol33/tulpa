# Generic spec-driven outer-grid refinement (Step 2).
#
# Port of the joint nested-Laplace refinement machinery in
# `R/nested_laplace_joint_refine.R`, generalised so the axis metadata (log-
# scale, bounds, refinable, refinement priority) lives in
# `hyper_axis_spec` objects instead of hardcoded by-axis-name lookups. Driven
# only through a user-supplied `kernel_fn(new_cells, warm_start, store_extras)`
# callback, so the SAME refinement engine serves:
#   * `tulpa_hyper_grid()` (kernel_fn wraps the user's inner_fit; no per-cell
#     warm-start chain), and
#   * `tulpa_nested_laplace_joint()` (kernel_fn wraps the backend's joint C++
#     kernel call; warm_start is the anchor cell's mode vector).
#
# Math ground (porting note: the joint driver's file-level math note in
# `R/nested_laplace_joint_refine.R` applies verbatim here -- one source of
# truth, the formulas didn't move).
#
# The on-disk canonical representation throughout is:
#   * `theta_grid`     -- numeric matrix `[n_cells x n_axes]` of outer-grid
#                         hyperparameter values; columns named after axes.
#   * `log_marginal`   -- numeric `[n_cells]`; per-cell log integrand
#                         (kernel log marginal + log prior already baked in).
#   * `extras`         -- list of length `n_cells` or `NULL`; opaque per-cell
#                         side data the kernel returned (e.g. the joint
#                         driver's mode + Q_csc; hyper_grid passes NULL).
#                         Refinement carries it through by concatenation.
#   * `refining_axis`  -- character `[n_cells]`; per-cell tag, `""` for a
#                         cell of the grid as declared, the axis name for a
#                         cell a pass added. Both passes add LEVELS, laid in
#                         every row of the other axes that holds the
#                         posterior (`.hyper_tensor_level_cells()`), so the
#                         grid stays a tensor where it carries mass; the tag keeps the declared levels apart,
#                         which fix the span and any prior read off the nodes.
#
# kernel_fn signature:
#   function(new_cells, warm_start = NULL, store_extras = FALSE) -> list(
#     log_marginal = numeric[nrow(new_cells)],
#     extras       = list[nrow(new_cells)] or NULL
#   )
# `warm_start` is opaque to refinement: refinement passes whatever the
# anchor cell's extras carry, and the kernel decides how to use it (e.g.
# read $mode from joint extras to warm-start the kernel call).

# ============================================================================
# Spec lookups -- the single point where axis metadata is read.
# ============================================================================

# Axis names that opt in to refinement passes.
.hyper_refinable_axes <- function(specs) {
  vapply(specs, function(s)
    if (isTRUE(s$refinable)) s$name else NA_character_,
    character(1))
}
.hyper_refinable_names <- function(specs) {
  out <- .hyper_refinable_axes(specs)
  out[!is.na(out)]
}

# Per-spec helpers.
.hyper_spec_by_name <- function(specs, name) {
  for (s in specs) if (identical(s$name, name)) return(s)
  stop("Unknown axis '", name, "'.", call. = FALSE)
}
.hyper_axis_is_log_scale <- function(spec) isTRUE(spec$log_scale)
.hyper_axis_bounds <- function(spec) spec$bounds

# May refinement place a node PAST the outermost declared node on this axis?
# A spec that predates the field says nothing about it and keeps the historical
# answer.
.hyper_axis_may_extend <- function(spec) !identical(spec$extend, FALSE)

# The interval new nodes are confined to on an axis that may not be extended:
# the span the axis's DECLARED nodes cover, on the natural scale. `spec$grid`
# holds those nodes -- refinement appends to `theta_grid`, never to the spec --
# so this is the range the caller wrote down however many passes have run.
# A log-scale axis's zero level is its point mass rather than a point of the
# continuum, so the span is taken over the positive nodes.
# NULL where the axis may be extended, and where fewer than two continuum nodes
# leave no interior to place anything in.
.hyper_axis_node_limit <- function(spec) {
  if (.hyper_axis_may_extend(spec)) return(NULL)
  g <- as.numeric(spec$grid)
  g <- g[is.finite(g)]
  if (.hyper_axis_is_log_scale(spec)) g <- g[g > 0]
  if (length(g) < 2L) return(c(NA_real_, NA_real_))
  range(g)
}

# Drop proposed nodes that fall outside the declared span. A no-op on an axis
# that may be extended; on one that may not, this is the whole of what makes a
# stated axis a bound, so every proposal path goes through it rather than each
# restating the comparison.
.hyper_clip_to_node_limit <- function(pts, spec) {
  lim <- .hyper_axis_node_limit(spec)
  if (is.null(lim)) return(pts)
  if (anyNA(lim)) return(pts[0])
  pts[pts >= lim[1L] & pts <= lim[2L]]
}

# Refinement-order priority. Defaults to declaration order; callers may
# attach `refine_priority` (integer, smaller = earlier) to a spec to override.
.hyper_axis_refinement_order <- function(axes, specs) {
  prio <- vapply(axes, function(a) {
    s <- .hyper_spec_by_name(specs, a)
    as.integer(s$refine_priority %||% 100L)
  }, integer(1))
  axes[order(prio, seq_along(axes))]
}

# ============================================================================
# Edge scores per axis (boundary mass + integrand-density-at-boundary).
# ============================================================================
.hyper_axis_edge_scores <- function(theta_grid, log_marginal, specs, axes,
                                    refining_axis = NULL) {
  if (length(log_marginal) == 0L) return(list())
  # Non-finite cells (inner Newton non-convergent on consumer-package
  # joint fitters, e.g. occu_cover_joint_coupled at degenerate sigma+alpha
  # hyperpoints) must not poison the edge-score statistics. Treat them as
  # zero-mass: drop from the global max-shift, use the safe weight
  # normaliser, and skip them inside per-level max/sum reductions.
  finite_lm <- log_marginal[is.finite(log_marginal)]
  if (length(finite_lm) == 0L) return(list())
  lm_max_total <- max(finite_lm)
  weights      <- .nl_normalise_weights_safe(log_marginal,
                                              what = "adaptive_grid edge scores",
                                              log_quad = .hyper_log_quad_weights(
                                                  theta_grid, specs,
                                                  refining = refining_axis))
  if (all(is.na(weights))) return(list())
  out <- list()
  per_level_max <- function(lm_at_lev) {
    finite_lev <- lm_at_lev[is.finite(lm_at_lev)]
    if (length(finite_lev) == 0L) -Inf else max(finite_lev)
  }
  for (a in axes) {
    v   <- as.numeric(theta_grid[, a])
    lev <- sort(unique(v))
    if (length(lev) < 2L) next
    w_lo <- sum(weights[v == lev[1L]],          na.rm = TRUE)
    w_hi <- sum(weights[v == lev[length(lev)]], na.rm = TRUE)
    d_lo <- exp(per_level_max(log_marginal[v == lev[1L]])          - lm_max_total)
    d_hi <- exp(per_level_max(log_marginal[v == lev[length(lev)]]) - lm_max_total)
    out[[a]] <- list(
      levels    = lev,
      min_frac  = w_lo, max_frac = w_hi,
      min_dens  = d_lo, max_dens = d_hi,
      min_score = max(w_lo, d_lo),
      max_score = max(w_hi, d_hi)
    )
  }
  out
}

# ============================================================================
# Per-axis point proposals (boundary extension + interior densification).
# Identical to the joint driver's `.propose_axis_extension` /
# `.propose_interior_densification` but reading log_scale / bounds from `spec`.
# ============================================================================
.hyper_propose_axis_extension <- function(spec, lev, side,
                                          extend_ok = .hyper_axis_may_extend(spec)) {
  log_scale <- .hyper_axis_is_log_scale(spec)
  bounds    <- .hyper_axis_bounds(spec)
  mid <- if (log_scale)
    function(a, b) exp(0.5 * (log(a) + log(b))) else
    function(a, b) 0.5 * (a + b)
  if (side == "max") {
    edge      <- lev[length(lev)]
    neighbour <- lev[length(lev) - 1L]
    densify   <- mid(neighbour, edge)
    extend1   <- if (log_scale) edge * (edge / neighbour)
                 else            edge + (edge - neighbour)
    extend2   <- if (log_scale) extend1 * (edge / neighbour)
                 else            extend1 + (edge - neighbour)
  } else {
    edge      <- lev[1L]
    neighbour <- lev[2L]
    densify   <- mid(edge, neighbour)
    extend1   <- if (log_scale) edge * (edge / neighbour)
                 else            edge - (neighbour - edge)
    extend2   <- if (log_scale) extend1 * (edge / neighbour)
                 else            extend1 - (neighbour - edge)
  }
  pts <- if (isTRUE(extend_ok)) c(densify, extend1, extend2) else densify
  pts <- .hyper_clip_to_node_limit(pts, spec)
  if (!is.null(bounds)) {
    pts <- pts[pts > bounds[1L] & pts < bounds[2L]]
  }
  # The declared prior support is fixed, so a node outside it would carry zero
  # weight and only cost an inner solve.
  slab <- spec$slab_bounds
  if (!is.null(slab)) {
    pts <- pts[pts >= slab[1L] & pts <= slab[2L]]
  }
  keep <- vapply(pts, function(p) {
    all(abs(lev - p) > 1e-8 * max(1, abs(p)))
  }, logical(1))
  pts[keep]
}

# Every point here is a midpoint between two existing levels, so it lies inside
# the span whatever the axis's extension setting; no clip is needed.
.hyper_propose_interior_densification <- function(spec, lev, mode_idx,
                                                  do_left = FALSE,
                                                  do_right = FALSE) {
  log_scale <- .hyper_axis_is_log_scale(spec)
  mid <- if (log_scale)
    function(a, b) exp(0.5 * (log(a) + log(b))) else
    function(a, b) 0.5 * (a + b)
  pts <- numeric(0)
  if (do_left  && mode_idx > 1L)              pts <- c(pts, mid(lev[mode_idx - 1L], lev[mode_idx]))
  if (do_right && mode_idx < length(lev))     pts <- c(pts, mid(lev[mode_idx], lev[mode_idx + 1L]))
  if (length(pts) == 0L) return(pts)
  bounds <- .hyper_axis_bounds(spec)
  if (!is.null(bounds)) {
    pts <- pts[pts > bounds[1L] & pts < bounds[2L]]
  }
  keep <- vapply(pts, function(p) {
    all(abs(lev - p) > 1e-8 * max(1, abs(p)))
  }, logical(1))
  pts[keep]
}

# Points bisecting the gaps an axis marginal's mass sits across.
#
# `vals` / `log_mass` are the axis's levels and their log masses, the quadrature
# weight already folded in. Only the continuum is bisected: a declared point
# mass is a level of the model rather than a node of the rule, so it neither
# bounds a gap nor enters the shares. A gap between adjacent continuum levels is
# bisected on the axis's integration coordinate when the two levels bounding it
# together carry at least `1 / min_ess` of the continuum's mass, the share one
# level holds on an axis spread evenly at the ESS floor. The heaviest gap is
# always bisected, so a round under the floor never proposes nothing.
#
# The points are placed by where the mass IS rather than at a multiple of a
# modal SD: a marginal whose mass sits on two adjacent nodes has its modal
# parabola read the grid's spacing, and points placed at a fraction of that
# around the mean land inside the gap and leave both heavy nodes' outer sides
# as coarse as before (gcol33/tulpa#858).
#
# Nothing is proposed where the density peaks on an outermost continuum level.
#
# Every point is a midpoint between two existing levels, so it lies inside the
# span, the bounds and the slab whatever the axis declares; no clip is needed.
# Points come back heaviest gap first, so a caller spending a node budget
# spends it where the mass is.
.hyper_propose_mass_bisection <- function(spec, vals, log_mass,
                                          min_ess = .nl_diag("axis_sd_ess")) {
  vals <- as.numeric(vals)
  keep <- is.finite(vals) & !.hyper_is_atom_level(vals, spec)
  if (.hyper_axis_is_log_scale(spec)) keep <- keep & vals > 0
  v  <- vals[keep]
  lm <- as.numeric(log_mass)[keep]
  o  <- order(v); v <- v[o]; lm <- lm[o]
  if (length(v) < 2L) return(numeric(0))
  m <- max(lm)
  if (!is.finite(m)) return(numeric(0))
  p <- exp(lm - m)
  p[!is.finite(p)] <- 0
  p <- p / sum(p)
  u <- .hyper_axis_coord(v, spec)
  # A marginal whose density peaks on an outermost level is truncated by the
  # span, not under-resolved inside it: bisecting towards that level only
  # shrinks its box, and the extension that would serve it belongs to the
  # adaptive pass. The density is the mass less the level's own box width, so
  # a level made heavy by the width it owns does not read as the mode.
  lw <- .nl_level_log_width(u)
  ld <- if (is.null(lw)) lm else lm - lw
  if (which.max(ld) %in% c(1L, length(ld))) return(numeric(0))
  gap <- p[-1L] + p[-length(p)]
  width <- diff(u)
  ok <- is.finite(width) & width > 1e-8 * pmax(1, abs(u[-1L]))
  if (!any(ok)) return(numeric(0))
  pick <- ok & gap >= 1 / min_ess
  pick[which(ok)[which.max(gap[ok])]] <- TRUE
  idx <- which(pick)
  idx <- idx[order(-gap[idx])]
  u_mid <- 0.5 * (u[idx] + u[idx + 1L])
  if (.hyper_axis_is_log_scale(spec)) exp(u_mid) else u_mid
}

# Points laid at an axis's outer MODE, when the fit has one: the mode and the
# SD the placement mode-find measured there (`.nl_placement_mode()`), a ladder
# of nodes one SD apart in the coordinate the mode was found in. A Gaussian
# marginal read at that spacing has a quadrature ESS of 3.48, clear of the
# `axis_sd_ess` floor of 3, so an axis collapsed onto one node is resolved in
# the round that lays them rather than after a chain of bisections, each a
# kernel call of its own.
#
# This is what bisection could not do without a mode: its only spread was the
# grid's own, and a parabola read off coarse nodes places its vertex tens of
# posterior SDs from the mode (gcol33/tulpa#919). The mode-find's is a
# converged Newton step's, measured on the posterior itself.
#
# How far the ladder runs is set by the box rule. A node's measure is the box
# to the midpoints with its neighbours in its row, so the outermost node before
# a gap owns half of it, and a read that spreads a node's mass over its box --
# as the reported interval and SD do -- carries that mass across the gap. Five
# points laid into the Calluna fit's 36-SD gap on its pinned dispersion axis
# left the one at 2 SDs holding 62% of the axis's weight and the mean 1.2 SDs
# above the mode; bounding the MASS each end node reads into its gap was not
# enough either, since 0.7% of the posterior spread across a 100-SD box
# quadrupled the reported SD. On each side the ladder therefore runs out to the
# first step `K >= 2` whose reading into the gap beyond it adds at most
# `at_mode_gap_var` to the axis's variance (`.hyper_at_mode_reach()`,
# `.hyper_gap_read()`): 5 and 6 SDs on that axis.
#
# `vals` are the nodes of the row the points will be laid in (the fibre), whose
# gaps are the ones the new points' boxes take. `mode` is
# `list(mode_u, sd_u, tag)`. Points leave out a declared point mass, stay inside
# the axis's bounds and, on an axis the caller stated (`extend = FALSE`), inside
# its declared span. A point with a node of the axis within half an SD of it is
# dropped: that node already reads the density there, so on an axis a placement
# laid at 1.25 SDs the proposal comes back empty and the pass bisects as it
# would without a mode. They come back nearest the mode first.
.hyper_propose_at_mode <- function(spec, vals, mode,
                                   gap_var = .nl_diag("at_mode_gap_var")) {
  if (is.null(mode) || length(mode$mode_u) != 1L || length(mode$sd_u) != 1L ||
      !is.finite(mode$mode_u) || !is.finite(mode$sd_u) || mode$sd_u <= 0) {
    return(numeric(0))
  }
  vals <- as.numeric(vals)
  cont <- vals[is.finite(vals) & !.hyper_is_atom_level(vals, spec)]
  u_nodes <- .joint_pareto_fwd(mode$tag, cont)
  u_nodes <- u_nodes[is.finite(u_nodes)]
  d <- (u_nodes - mode$mode_u) / mode$sd_u
  k <- c(0, -seq_len(.hyper_at_mode_reach(-d[d < 0], gap_var)),
         seq_len(.hyper_at_mode_reach(d[d > 0], gap_var)))
  .hyper_mode_points(spec, cont, u_nodes, mode, k[order(abs(k))])
}

# What a node reads into a gap under the box rule, as a share of the axis's
# posterior variance about the mode: the part of its box past half an SD,
# `[a, b]` in the mode's SDs, holding density `dens` spread uniformly, has
# second moment `dens |b - a| (a^2 + a b + b^2) / 3` about the mode.
.hyper_gap_read <- function(dens, a, b) dens * abs(b - a) * (a^2 + a * b + b^2) / 3

# How many SDs an at-mode ladder runs out on one side: the first step `K >= 2`
# whose box reads at most `gap_var` of the axis's variance into the gap to the
# next node beyond it, at the Gaussian density the mode-find measured. `d`
# holds the row's existing nodes' distances on that side (positive). A side
# with no node beyond the ladder stops at 2, where the pass has no gap to read.
.hyper_at_mode_reach <- function(d, gap_var) {
  K <- 2
  repeat {
    beyond <- d[d > K + 0.5]
    if (!length(beyond)) return(K)
    h <- (min(beyond) - K) / 2
    if (h <= 0.5 ||
        .hyper_gap_read(stats::dnorm(K), K + 0.5, K + h) <= gap_var) return(K)
    K <- K + 1
  }
}

# Points `k` SDs from a mode, on the natural scale, that an axis admits: inside
# its bounds and, on an axis the caller stated (`extend = FALSE`), inside its
# declared span `cont`, with none within half an SD of a node `u_nodes` already
# holds. Kept in the order `k` gives them.
.hyper_mode_points <- function(spec, cont, u_nodes, mode, k) {
  pts <- .joint_pareto_inv(mode$tag, mode$mode_u + k * mode$sd_u)$theta
  pts <- pts[is.finite(pts)]
  if (.hyper_axis_is_log_scale(spec)) pts <- pts[pts > 0]
  bounds <- .hyper_axis_bounds(spec)
  if (!is.null(bounds)) pts <- pts[pts > bounds[1L] & pts < bounds[2L]]
  if (!isTRUE(spec$extend) && length(cont)) {
    pts <- pts[pts >= min(cont) & pts <= max(cont)]
  }
  u_pts <- .joint_pareto_fwd(mode$tag, pts)
  keep <- vapply(u_pts, function(u)
    all(abs(u_nodes - u) > (0.5 + 1e-6) * mode$sd_u), logical(1))
  pts[keep]
}

# The points that close a solved row where its nodes still read across a gap.
# The at-mode points are laid from where the mode-find stopped, and a mode a
# fraction of an SD off moves the density against them: on the Calluna fit the
# dispersion axis peaked 1.1 SDs above the found mode, and a ladder sized for a
# density centred on the mode left its end node reading a quarter of the axis
# into the gap. Read off the solved row instead: in each gap, the half a node
# owns is misread by the difference between its density held flat and the
# log-linear run to its neighbour's (`.hyper_gap_misread()`), and a node
# misreading more than `gap_var` of the axis's variance into a gap more than
# 1.5 SDs wide gets a point one SD out into it; so does the outermost node on
# a side while it carries more than `gap_var` of the row at all. `marg` is the
# row's marginal along the axis (`vals`, `log_marg`).
.hyper_propose_edge_close <- function(spec, marg, mode,
                                      gap_var = .nl_diag("at_mode_gap_var")) {
  if (is.null(mode) || !is.finite(mode$sd_u) || mode$sd_u <= 0) return(numeric(0))
  vals <- as.numeric(marg$vals)
  cont <- is.finite(vals) & !.hyper_is_atom_level(vals, spec)
  u  <- .joint_pareto_fwd(mode$tag, vals[cont])
  lm <- as.numeric(marg$log_marg)[cont]
  ok <- is.finite(u)
  u <- u[ok]; lm <- lm[ok]
  top <- if (length(lm)) max(lm) else -Inf
  if (!is.finite(top)) return(numeric(0))
  o <- order(u); u <- u[o]; p <- exp(lm[o] - top); p <- p / sum(p)
  x <- (u - mode$mode_u) / mode$sd_u
  n <- length(x)
  # A node's density is its share over the box it owns, both in the mode's SDs;
  # an outermost box is mirrored by its own half-spacing.
  edges <- if (n >= 2L) {
    c(x[1L] - (x[2L] - x[1L]) / 2, (x[-1L] + x[-n]) / 2,
      x[n] + (x[n] - x[n - 1L]) / 2)
  } else c(x - 0.5, x + 0.5)
  dens <- p / diff(edges)
  k <- numeric(0)
  if (n >= 1L && p[1L] > gap_var) k <- c(k, x[1L] - 1)
  if (n >= 2L && p[n] > gap_var)  k <- c(k, x[n] + 1)
  for (j in seq_len(n - 1L)) {
    L <- x[j + 1L] - x[j]
    if (L <= 1.5) next
    if (is.finite(dens[j]) &&
        .hyper_gap_misread(dens[j], dens[j + 1L], x[j], L, 1) > gap_var) {
      k <- c(k, x[j] + 1)
    }
    if (is.finite(dens[j + 1L]) &&
        .hyper_gap_misread(dens[j + 1L], dens[j], x[j + 1L], L, -1) > gap_var) {
      k <- c(k, x[j + 1L] - 1)
    }
  }
  if (!length(k)) return(numeric(0))
  .hyper_mode_points(spec, vals[cont], u, mode, unique(k[order(abs(k))]))
}

# What the box rule misreads in the half of a gap of length `L` a node at `x0`
# owns, on side `side`: its density `d0` held flat over the half, against the
# log-linear run to the neighbour's density `d1` at the gap's far end, weighted
# by the half's second moment about the mode. In the mode's SDs. A neighbour
# with no density leaves the whole flat half misread.
.hyper_gap_misread <- function(d0, d1, x0, L, side) {
  h <- L / 2
  flat <- d0 * h
  beta <- if (is.finite(d1) && d1 > 0) (log(d1) - log(d0)) / L else -Inf
  run <- if (is.infinite(beta)) 0
         else if (abs(beta) < 1e-12) flat
         else d0 * (exp(beta * h) - 1) / beta
  b <- x0 + side * h
  (flat - run) * (x0^2 + x0 * b + b^2) / 3
}

# New levels `pts` on `axis` laid in the rows of the grid (the combinations of
# the other axes) that hold a solved cell, so the grid stays a tensor: every
# row that carries the posterior integrates the axis at the same nodes,
# including the rows through levels an earlier pass added to another axis. The
# warm start is the heaviest solved cell.
#
# `log_weight` is each cell's log posterior mass under the grid's measure
# (log-marginal plus log quadrature weight). Given it, a level is laid only in
# the fewest rows, heaviest first, that hold all but `row_tail` of the mass. A
# row left out keeps the cells it has and lacks only the new level, which is a
# cell no row's measure counts as present: what it can misplace is bounded by
# its own mass. Without it every row is refined, and on a grid of four or more
# axes each pass's levels become rows of the next pass's axis, so the cells grow
# with the product of the levels every pass added.
.hyper_tensor_level_cells <- function(theta_grid, log_marginal, axis, pts,
                                      log_weight = NULL,
                                      row_tail = .nl_diag("level_row_tail")) {
  if (!length(pts)) return(NULL)
  base <- is.finite(log_marginal)
  if (!any(base)) return(NULL)
  others <- setdiff(colnames(theta_grid), axis)
  rows <- unique(theta_grid[base, others, drop = FALSE])
  if (length(others) && nrow(rows) > 1L && !is.null(log_weight) &&
      length(log_weight) == nrow(theta_grid) && row_tail > 0) {
    lw <- log_weight[base]
    lw[!is.finite(lw)] <- -Inf
    if (any(is.finite(lw))) {
      key <- function(m) do.call(paste, c(lapply(seq_len(ncol(m)), function(k)
        sprintf("%.10g", m[, k])), sep = ":"))
      w <- exp(lw - max(lw))
      mass <- as.numeric(rowsum(w, key(theta_grid[base, others, drop = FALSE]),
                                reorder = FALSE)[key(rows), 1L])
      ord <- order(mass, decreasing = TRUE)
      cum <- cumsum(mass[ord]) / sum(mass)
      rows <- rows[ord[seq_len(which(cum >= 1 - row_tail)[1L])], , drop = FALSE]
    }
  }
  n_rows <- if (length(others)) nrow(rows) else 1L
  cells <- matrix(NA_real_, n_rows * length(pts), ncol(theta_grid),
                  dimnames = list(NULL, colnames(theta_grid)))
  for (b in others) cells[, b] <- rep(rows[, b], each = length(pts))
  cells[, axis] <- rep(as.numeric(pts), n_rows)
  list(new_cells = cells, warm_start_idx = which(base)[which.max(log_marginal[base])])
}

# The cells of `new_cells` not already on the grid, at the precision the cells
# are keyed with, or NULL when none is new.
.hyper_new_cells_only <- function(new_cells, theta_grid) {
  if (is.null(new_cells) || nrow(new_cells) == 0L) return(NULL)
  axis_names <- colnames(theta_grid)
  fmt <- function(m) {
    cols <- lapply(axis_names, function(a) sprintf("%.10g", m[, a]))
    do.call(paste, c(cols, sep = ":"))
  }
  new_keys <- fmt(new_cells)
  keep <- !new_keys %in% fmt(theta_grid) & !duplicated(new_keys)
  if (!any(keep)) return(NULL)
  new_cells[keep, , drop = FALSE]
}

# ============================================================================
# Detect refinement triggers on one axis.
# Boundary (peak-at-edge or tail-mass-at-edge) and interior (peak between
# levels with wide spacing) checks. Returns the new levels' cells
# (`.hyper_tensor_level_cells()`), NULL if no trigger fires.
# ============================================================================
.hyper_detect_axis_refinement <- function(theta_grid, log_marginal, edge_info,
                                          axis_name, spec, edge_thresh,
                                          log_weight = NULL) {
  ei  <- edge_info
  lev <- ei$levels
  v   <- as.numeric(theta_grid[, axis_name])
  lm_max_at_lev <- vapply(lev, function(lv) {
    lm_lv <- log_marginal[v == lv]
    finite_lv <- lm_lv[is.finite(lm_lv)]
    if (length(finite_lv) == 0L) -Inf else max(finite_lv)
  }, numeric(1))
  mode_idx <- which.max(lm_max_at_lev)
  n_lev    <- length(lev)

  tr_min <- ei$min_score >= edge_thresh &&
            (mode_idx == 1L || ei$min_dens >= edge_thresh)
  tr_max <- ei$max_score >= edge_thresh &&
            (mode_idx == n_lev || ei$max_dens >= edge_thresh)

  interior_log_step <- 0.55
  interior_lin_step <- 0.25
  wide_left <- wide_right <- FALSE
  if (mode_idx > 1L && mode_idx < n_lev) {
    if (.hyper_axis_is_log_scale(spec)) {
      wide_left  <- (log(lev[mode_idx])     - log(lev[mode_idx - 1L])) >= interior_log_step
      wide_right <- (log(lev[mode_idx + 1L]) - log(lev[mode_idx]))     >= interior_log_step
    } else {
      span <- diff(range(lev))
      if (span > 0) {
        wide_left  <- (lev[mode_idx]     - lev[mode_idx - 1L]) / span >= interior_lin_step
        wide_right <- (lev[mode_idx + 1L] - lev[mode_idx])     / span >= interior_lin_step
      }
    }
  }

  if (!tr_min && !tr_max && !wide_left && !wide_right) return(NULL)
  pts <- c(if (tr_max) .hyper_propose_axis_extension(spec, lev, "max"),
           if (tr_min) .hyper_propose_axis_extension(spec, lev, "min"),
           if (wide_left || wide_right)
             .hyper_propose_interior_densification(spec, lev, mode_idx,
                                                   wide_left, wide_right))
  .hyper_tensor_level_cells(theta_grid, log_marginal, axis_name, pts,
                            log_weight = log_weight)
}

# ============================================================================
# Apply refinement for one axis: solve the new cells with kernel_fn from the
# anchor's warm-start material and merge them into the existing grid /
# log_marginal / extras / refining_axis, each tagged with the axis it refines.
#
# The rows a new level was laid in were chosen by the mass the grid held BEFORE
# the level was solved (`.hyper_tensor_level_cells()`). Where the refined axis
# correlates with another, the solved level can carry its mass in rows that
# were light at the old levels, so the level then grows into the neighbouring
# rows of every row it moved mass towards (`.hyper_level_frontier()`), until
# no such row has an unrefined neighbour.
# ============================================================================
.hyper_apply_axis_refinement <- function(theta_grid, log_marginal, extras,
                                          refining_axis, pack, axis_name,
                                          specs, kernel_fn, hp_fn = NULL) {
  new_cells <- .hyper_new_cells_only(pack$new_cells, theta_grid)
  if (is.null(new_cells)) {
    return(list(theta_grid = theta_grid, log_marginal = log_marginal,
                extras = extras, refining_axis = refining_axis, n_new = 0L,
                n_levels = 0L))
  }
  warm_start <- NULL
  if (!is.null(extras)) {
    idx0 <- pack$warm_start_idx
    if (length(idx0) == 1L && idx0 >= 1L && idx0 <= length(extras)) {
      warm_start <- extras[[idx0]]
    }
  }
  solve_merge <- function(cells) {
    n <- nrow(cells)
    fit_out <- kernel_fn(cells, warm_start = warm_start,
                         store_extras = !is.null(extras))
    new_lm <- fit_out$log_marginal
    if (!is.null(hp_fn)) {
      hp_new <- hp_fn(cells)
      if (!is.null(hp_new) && length(hp_new) == n) new_lm <- new_lm + hp_new
    }
    theta_grid    <<- rbind(theta_grid, cells)
    log_marginal  <<- c(log_marginal, new_lm)
    refining_axis <<- c(refining_axis, rep(axis_name, n))
    if (!is.null(extras)) {
      extras <<- c(extras, fit_out$extras %||% vector("list", n))
    }
    n
  }
  levels <- unique(new_cells[, axis_name])
  n_new <- solve_merge(new_cells)
  repeat {
    grow <- .hyper_new_cells_only(
      .hyper_level_frontier(theta_grid, log_marginal, specs, refining_axis,
                            axis_name, levels), theta_grid)
    if (is.null(grow)) break
    n_new <- n_new + solve_merge(grow)
  }
  list(theta_grid = theta_grid, log_marginal = log_marginal,
       extras = extras, refining_axis = refining_axis, n_new = n_new,
       n_levels = length(levels))
}

# The cells that extend `levels` of `axis` into the neighbours of the rows
# holding them. A row's share of the levels' solved mass, against its share of
# the mass at the other levels of `axis`, says whether the levels moved the
# posterior towards it. Every row holding more than `row_tail` of the levels'
# mass whose share GREW passes them to the rows one level away along each other
# axis that hold a solved cell and not the levels yet. On axes that do not
# correlate the shares stay put and the rows `.hyper_tensor_level_cells()` left
# out stay out; where the levels pull the mass towards the edge of the rows they
# were laid in, the levels follow it until the edge rows hold none of it. NULL
# when no such row is left.
.hyper_level_frontier <- function(theta_grid, log_marginal, specs, refining,
                                  axis, levels,
                                  row_tail = .nl_diag("level_row_tail")) {
  others <- setdiff(colnames(theta_grid), axis)
  if (!length(others) || row_tail <= 0) return(NULL)
  lq <- .hyper_log_quad_weights(theta_grid, specs, refining = refining)
  lw <- if (length(lq) == length(log_marginal)) log_marginal + lq else log_marginal
  at <- is.finite(lw) & theta_grid[, axis] %in% levels
  if (!any(at)) return(NULL)
  key <- function(m) do.call(paste, c(lapply(seq_len(ncol(m)), function(k)
    sprintf("%.10g", m[, k])), sep = ":"))
  share <- function(sel) {
    k <- key(theta_grid[sel, others, drop = FALSE])
    m <- tapply(exp(lw[sel] - max(lw[sel])), k, sum)
    m / sum(m)
  }
  rest <- is.finite(lw) & !theta_grid[, axis] %in% levels
  if (!any(rest)) return(NULL)
  held <- theta_grid[at, others, drop = FALSE]
  hk <- key(held)
  s_new <- share(at)
  # The old shares over the same rows, so a row's ratio compares like with
  # like rather than growing by the mass of the rows the levels were not laid in.
  s_old <- share(rest & key(theta_grid[, others, drop = FALSE]) %in% hk)
  grew <- s_new / s_old[names(s_new)]
  grew[is.na(grew)] <- Inf
  hot <- names(s_new)[s_new > row_tail & grew > 1 + 1e-8]
  if (!length(hot)) return(NULL)
  rows <- unique(theta_grid[is.finite(log_marginal), others, drop = FALSE])
  lev_of <- lapply(others, function(b) sort(unique(rows[, b])))
  src <- held[match(hot, hk), , drop = FALSE]
  nb <- list()
  for (j in seq_along(others)) {
    p <- match(src[, j], lev_of[[j]])
    for (step in c(-1L, 1L)) {
      q <- p + step
      ok <- q >= 1L & q <= length(lev_of[[j]])
      if (!any(ok)) next
      m <- src[ok, , drop = FALSE]
      m[, j] <- lev_of[[j]][q[ok]]
      nb[[length(nb) + 1L]] <- m
    }
  }
  if (!length(nb)) return(NULL)
  nb <- unique(do.call(rbind, nb))
  nk <- key(nb)
  nb <- nb[nk %in% key(rows) & !nk %in% hk, , drop = FALSE]
  if (!nrow(nb)) return(NULL)
  cells <- matrix(NA_real_, nrow(nb) * length(levels), ncol(theta_grid),
                  dimnames = list(NULL, colnames(theta_grid)))
  for (b in others) cells[, b] <- rep(nb[, b], each = length(levels))
  cells[, axis] <- rep(as.numeric(levels), nrow(nb))
  cells
}

# ============================================================================
# Main entries: adaptive boundary/interior pass and var-of-means consistency.
# ============================================================================

.hyper_adaptive_refine_pass <- function(theta_grid, log_marginal, extras,
                                        refining_axis, specs, kernel_fn,
                                        edge_thresh = 0.02, max_passes = 1L,
                                        hp_fn = NULL) {
  info <- list(triggered_axes = character(0),
               n_points_added = integer(0))
  if (max_passes < 1L) {
    return(list(theta_grid = theta_grid, log_marginal = log_marginal,
                extras = extras, refining_axis = refining_axis, info = NULL))
  }
  for (pass in seq_len(max_passes)) {
    axes <- .hyper_axis_refinement_order(.hyper_refinable_names(specs), specs)
    if (length(axes) == 0L) break
    any_triggered <- FALSE
    triggered_this_pass <- character(0)
    for (a in axes) {
      edge_info <- .hyper_axis_edge_scores(theta_grid, log_marginal, specs, a,
                                           refining_axis = refining_axis)
      ei <- edge_info[[a]]
      if (is.null(ei)) next
      spec <- .hyper_spec_by_name(specs, a)
      lw <- log_marginal + .hyper_log_quad_weights(theta_grid, specs,
                                                   refining = refining_axis)
      pack <- .hyper_detect_axis_refinement(theta_grid, log_marginal, ei, a,
                                             spec, edge_thresh, log_weight = lw)
      if (is.null(pack)) next
      step <- .hyper_apply_axis_refinement(theta_grid, log_marginal, extras,
                                            refining_axis, pack, a, specs,
                                            kernel_fn, hp_fn = hp_fn)
      if (step$n_new == 0L) next
      theta_grid   <- step$theta_grid
      log_marginal <- step$log_marginal
      extras       <- step$extras
      refining_axis <- step$refining_axis
      triggered_this_pass <- c(triggered_this_pass, a)
      info$n_points_added <- c(info$n_points_added, step$n_levels)
      any_triggered <- TRUE
    }
    if (!any_triggered) break
    info$triggered_axes <- c(info$triggered_axes,
                              paste(triggered_this_pass, collapse = ","))
  }
  if (length(info$triggered_axes) == 0L) info <- NULL
  list(theta_grid = theta_grid, log_marginal = log_marginal,
       extras = extras, refining_axis = refining_axis, info = info)
}

# Repopulate an axis whose marginal has collapsed onto too few nodes to carry a
# spread, by bisecting the gaps its mass sits across
# (`.hyper_propose_mass_bisection()`).
#
# The trigger is the axis's own quadrature effective sample size, read off the
# weights the fit integrates with, over the axis's continuum: a declared point
# mass is part of the model, and no node placed in the continuum changes the
# share it holds. It used to be the weighted SD compared against the parabola at
# the modal node, which is one SD estimator judging the other: with the reported
# SD now the weighted one wherever the axis is resolved (gcol33/tulpa#621), that
# comparison would have been the estimator against itself and the pass would
# never fire. The ESS answers the question the pass is actually asking -- how
# many nodes the marginal spreads over.
#
# The pass re-reads the ESS after every round and bisects again until the axis
# reaches `min_ess` or has taken `max_nodes` new levels. A single round cannot
# certify what it produced: bisecting two heavy nodes can leave the mass on the
# new midpoint and one of them, and a marginal is only resolved once the ESS it
# ends on says so (gcol33/tulpa#858).
#
# `axis_modes` names, per axis, the outer mode a placement mode-find found for
# it (`list(mode_u, sd_u, tag)`, `.nl_outer_mode_axes()`). Such an axis's first
# round lays its points AT the mode (`.hyper_propose_at_mode()`); each later
# round first closes any gap the solved points still read across
# (`.hyper_propose_edge_close()`), which runs until none is left even once the
# ESS is met, and bisects as above while the ESS is short. An axis whose at-mode
# proposal comes back empty already has a node within half a mode SD of every
# point of the ladder -- a placement laid it at 1.25 SDs -- so it resolves the
# posterior it was laid from and is left as it is: its ESS (2.8 for a 5-node
# ladder at 1.25 SDs) is under `min_ess` by construction, and bisecting it
# would only spend cells.
#
# A new point is a new LEVEL of the base tensor, laid in every row of the other
# axes that holds the posterior (`.hyper_tensor_level_cells()`,
# `.hyper_level_frontier()`), so every such row integrates the axis at the same
# nodes. Laying the
# points in the modal row alone (a slice) resolved the axis's own marginal and
# misread every other one: the modal row then integrates the axis finely and
# the rest at the coarse levels, so the other axes' levels carry quadrature
# errors that differ from row to row, and the slice cells are rows of one node
# along every other axis. On the two-arm ICAR fixture of gcol33/tulpa#932 that
# left the field SD's 95% interval 7% (16x16) and 8% (24x24) narrow against a
# dense reference; laid as levels, mean |F_ref(q) - p| fell from 0.013 / 0.032 to
# 0.004 / 0.001.
.hyper_consistency_pass <- function(theta_grid, log_marginal, extras,
                                    refining_axis, specs, kernel_fn,
                                    min_ess = .nl_diag("axis_sd_ess"),
                                    max_nodes = .nl_diag("axis_refine_nodes"),
                                    hp_fn = NULL, axis_modes = NULL) {
  refinable <- .hyper_refinable_names(specs)
  info <- list(axes = character(0), n_added = integer(0),
               ess_before = numeric(0), ess_after = numeric(0))
  n_added_total <- 0L
  if (length(refinable) == 0L) {
    return(list(theta_grid = theta_grid, log_marginal = log_marginal,
                extras = extras, refining_axis = refining_axis,
                info = NULL, n_added = 0L))
  }
  axis_ess <- function(axis, spec) {
    # Refinement grows theta_grid / log_marginal, so the log quadrature weights
    # are recomputed on every read to stay aligned with the current grid rows.
    log_quad <- .hyper_log_quad_weights(theta_grid, specs,
                                        refining = refining_axis)
    lm_eff <- log_marginal
    if (!is.null(log_quad) && length(log_quad) == length(lm_eff)) {
      lm_eff <- lm_eff + log_quad
      lm_eff[is.na(lm_eff)] <- -Inf
    }
    marg <- .nl_axis_marginal_logdensity(as.numeric(theta_grid[, axis]), lm_eff)
    cont <- !.hyper_is_atom_level(marg$vals, spec)
    list(marg = marg, ess = .nl_axis_quad_ess(marg$log_marg[cont]),
         lm_eff = lm_eff)
  }
  for (axis in refinable) {
    spec <- .hyper_spec_by_name(specs, axis)
    rd <- axis_ess(axis, spec)
    ess_before <- rd$ess
    if (!is.finite(ess_before) || ess_before >= min_ess) next
    at_mode <- axis_modes[[axis]]
    added <- 0L
    ladder <- 0L
    first <- TRUE
    while (added - ladder < max_nodes && is.finite(rd$ess)) {
      collapsed <- rd$ess < min_ess
      new_pts <- if (is.null(at_mode)) numeric(0)
        else if (first) .hyper_propose_at_mode(spec, rd$marg$vals, at_mode)
        else .hyper_propose_edge_close(spec, rd$marg, at_mode)
      if (first && !is.null(at_mode) && !length(new_pts)) break
      is_ladder <- first && length(new_pts) > 0L
      first <- FALSE
      if (length(new_pts) == 0L && collapsed) {
        new_pts <- .hyper_propose_mass_bisection(spec, rd$marg$vals,
                                                 rd$marg$log_marg, min_ess)
      }
      if (length(new_pts) == 0L) break
      if (!is_ladder) new_pts <- utils::head(new_pts, max_nodes - (added - ladder))
      pack <- .hyper_tensor_level_cells(theta_grid, log_marginal, axis,
                                        new_pts, log_weight = rd$lm_eff)
      if (is.null(pack)) break
      step <- .hyper_apply_axis_refinement(theta_grid, log_marginal, extras,
                                            refining_axis, pack, axis,
                                            specs, kernel_fn, hp_fn = hp_fn)
      if (step$n_new == 0L) break
      theta_grid    <- step$theta_grid
      log_marginal  <- step$log_marginal
      extras        <- step$extras
      refining_axis <- step$refining_axis
      added <- added + step$n_levels
      if (is_ladder) ladder <- step$n_levels
      rd <- axis_ess(axis, spec)
    }
    if (added == 0L) next
    info$axes       <- c(info$axes, axis)
    info$n_added    <- c(info$n_added, added)
    info$ess_before <- c(info$ess_before, ess_before)
    info$ess_after  <- c(info$ess_after, rd$ess)
    n_added_total   <- n_added_total + added
  }
  if (length(info$axes) == 0L) info <- NULL
  list(theta_grid = theta_grid, log_marginal = log_marginal,
       extras = extras, refining_axis = refining_axis, info = info,
       n_added = n_added_total)
}
