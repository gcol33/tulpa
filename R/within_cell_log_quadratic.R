# The LOG-QUADRATIC within-cell read: the outer grid's node densities joined by
# a smooth log density instead of each cell's mass spread flat over its box.
#
# A quadratic through three adjacent log densities is exact for a Gaussian at
# ANY spacing, where the box read's error is O(h^2) with a sign set by where the
# nodes fall. Between two nodes the read takes the average of the two quadratics
# through that pair (each pair sits in two triples; an end pair in one), and past
# the outer node it continues the end triple's quadratic when that quadratic is
# concave. That tail is what removes the truncation a node set laid over a FIXED
# extent imposes: the recentre ladder spans the mode +/- 2.5 SDs whatever its
# node count, so as nodes are added the box read's outer edge closes in on
# 2.5 SDs and its interval converges to a truncated Gaussian's, 0.952 of the
# true width (gcol33/tulpa#932). Where the end quadratic is not concave there is
# no tail to continue, and the end node's density is held flat out to its box
# edge, which is the box read's own outer half-cell. The tail is unbounded on
# the coordinate, which is every declared support's whole range
# (`.NL_DOMAIN_TRANSFORM` maps each onto the real line), so it stops at the
# declared support and nowhere else.
#
# ROWS. Each row of the grid along the axis (`.nl_axis_row_id()`: the cells
# whose other coordinates agree) is reconstructed on its own nodes and
# normalized to the mass its cells carry, so the read moves mass along a row and
# never between rows, and the axis's density is the sum of the rows'. A row is a
# conditional of the posterior, which is nearer a Gaussian than the mixture of
# rows the axis's marginal is.
#
# DENSITY TIMES SLAB. Along a row, a cell's mass is the posterior density at its
# node times its box on this axis times its box on the OTHER axes -- its slab.
# So the row's density along the axis is `exp(q(u)) t(u)`: `q` interpolates the
# cells' node log densities on the coordinate `u` the row is read on
# (`.nl_read_log_density()`), and `t` is each cell's slab,
# `w / (exp(log_density) width)`, held over that cell's box. The refinement
# passes add levels to the tensor, so `t` is one number along a row where the
# cell measure is taken on `u` itself, and differs from it only by the ratio of
# a cell's measured width to the change of coordinate at its node where the
# measure is taken on another coordinate (a correlation measured on its natural
# scale and read on atanh); a row whose cells carry a measure the product rule
# did not build (a caller's own weights) is still read on its own slabs. The
# mass over width alone would fold the slab into the density.
#
# `q` HAS TO BE THE DENSITY ON `u`. A quadratic through three log densities is
# exact for a Gaussian on the coordinate the densities are taken per unit of,
# and the row's resolution (`sd` below), its gate and the reach of its tails are
# all read off `q`. A density per unit of another coordinate leaves the change
# of coordinate in the slabs, a staircase `q` never sees: a correlation whose
# likelihood is nearly flat over its grid has a nearly flat density per unit of
# the correlation, so on an AR(1) rho grid running to tanh(+/- 4.5) `q` read the
# rows' sds on atanh at a median of 22 and up to 141667, where their densities
# on atanh, `1 - rho^2` times the likelihood, have sds of 0.70 to 0.76. The
# tails ran to atanh -424924, the table spaced its points 7 apart, wider than
# every row's nodes together, and the median read -0.918 against the cells'
# 0.110 (gcol33/tulpa#1274).
#
# THE GATE, PER ROW. A quadratic through nodes several posterior SDs apart reads
# the location of a skewed conditional off its tails, and on such a grid the box
# read's cell-wide spread is what keeps the interval covering: on the coarse
# rungs of the gcol33/tulpa#357 ladder (h / sd 3 to 27) the log-quadratic read's
# summed |coverage - 0.95| was 2.20 against the box read's 0.34, the failure
# `cell_normal_read.patch` showed in gcol33/tulpa#925. So a row is interpolated
# only where it resolves its own conditional, `h / sd <=
# .nl_diag("lq_max_h_over_sd")`, with `h` the row's node spacing where its mass
# sits (the median of its node gaps, each weighted by the mass of the two nodes
# bounding it: the spacing itself on an evenly spaced row) and `sd` the
# three-point parabola at its modal node.
# Read as the median spacing over every node, the levels a refinement pass laid
# down a slowly decaying tail -- light, and spaced at the tail's own scale --
# took the gate past 2 on a row whose core is laid at 1 SD, and the whole axis
# fell back to boxes that under-read the exponential tail by 4% a cell. A row
# above the gate, one whose
# mode is an end node, and one with fewer than three positive-mass nodes are
# read as their cells' boxes. An axis on which no row qualifies declines to the
# box read as `"unresolved"`.
#
# A quantity that is not a grid axis -- an entry of a covariance assembled from
# several axes -- has no rows to read, and declines as `"not_a_grid_axis"`; a
# caller with no `log_marginal` to hand (a summary taken off weights alone) has
# no node density, and declines as `"no_node_density"`. A continuum whose lowest
# cell reaches down to a point mass at zero declines as `"floor_cell"`.
#
# THE RECONSTRUCTION IS TABULATED. Each row's density is evaluated on one grid
# over the axis's coordinate and integrated by the trapezoid rule; the axis's
# CDF is the mass-weighted sum of the rows' CDFs, the quantile is read off it by
# linear interpolation, and `tulpa_hyper_draws()` inverts the same tables
# (`.nl_lq_axis_draw()`), so the interval and the draws describe one
# distribution. A tail runs to `lq_tail_nats` below the row's peak, and no
# farther than `lq_tail_sd` of the row's SDs past its outer node.
#
# A TAIL IS CONTINUED ONLY WHERE THE ROW'S NODES SAW ITS DENSITY FALL. Past the
# outer node the read extrapolates the end triple's quadratic, and the mass it
# puts there is set by that quadratic's curvature alone. For a Gaussian whose
# outer node lies `d` SDs from the mode, the tail beyond carries
# `(1 - Phi(d)) / (Phi(d) - 1/2)` times the mass between the mode and that node:
# unbounded as `d` goes to zero, so a row whose nodes barely see its density
# fall hands nearly all its mass to a curvature at the level of the log
# densities' own noise. On a two-row axis whose second row falls 2e-8 nats
# across its nodes, that row's tails reached +/- 1.9e5, 40000 times its span
# (gcol33/tulpa#1278). So a side is continued only where the extrapolated tail
# carries no more mass than the stretch the nodes measured, `d >= qnorm(3/4)`,
# i.e. where the row's top node stands at least `lq_tail_min_fall =
# qnorm(3/4)^2 / 2` nats (0.227) above that end node, read off the node log
# densities rather than off the curvature. A side short of it ends at its outer
# box edge, the box read's own half-cell.
#
# THE TABLE RESOLVES EVERY ROW WHEREVER ANOTHER ROW'S TAIL REACHES
# (`.nl_lq_table()`). Its spacing is at most an eighth of the closest node gap
# over the nodes and their boxes, where every row's body lies; a row whose
# density is nearly flat on its own coordinate has an sd far beyond its nodes,
# and its tails with it, and one uniform spacing over that extent would put the
# other rows' bodies between two points. Past the nodes the spacing grows
# geometrically, by 1/64 of the distance already covered, so a tail an sd `s`
# past the outer node is spaced about `s / 16` there.

# The node log density every axis's read interpolates (`log_density` below):
# at each cell, the posterior density per unit of the coordinate each axis is
# read on (`.nl_cell_partition()` over the axis's continuum levels, the
# partition `.nl_level_edges()` takes). `log_marginal` is the density on each
# axis's measure coordinate (`.hyper_axis_coord()` of its spec: log on a
# log-scale axis, the natural scale otherwise) with the folded hyperprior; each
# axis `specs` describes adds its declared unit density
# (`.hyper_axis_log_unit()`) and the change from its measure coordinate onto
# its read coordinate, the same carry `.hyper_outer_log_density_u()` makes onto
# a proposal coordinate. An axis's term depends on its own coordinate alone, so
# along a row of any other axis it is one constant, which the row's
# normalization absorbs: one vector serves every axis. A log-scale axis read on
# log with no declared unit density adds exactly zero. An axis no spec
# describes is taken as measured on its read coordinate. A cell its axis's read
# coordinate does not reach (the declared point mass, split off before the
# read) takes a zero term there, since that read drops it and every other
# axis's row holds the term constant.
.nl_read_log_density <- function(tg, log_marginal, specs = NULL,
                                 domains = NULL, atoms = NULL) {
  if (is.null(log_marginal)) return(NULL)
  if (is.null(dim(tg))) tg <- matrix(as.numeric(tg), ncol = 1L)
  out <- as.numeric(log_marginal)
  cn <- colnames(tg)
  if (!length(specs) || is.null(cn) || length(out) != nrow(tg)) return(out)
  names(specs) <- vapply(specs, function(s) s$name %||% "", character(1))
  for (j in which(cn %in% names(specs))) {
    sp <- specs[[cn[j]]]
    v <- as.numeric(tg[, j])
    at <- if (length(atoms) < j) NA_real_ else atoms[[j]]
    cont <- is.finite(v) & !(length(at) == 1L && is.finite(at) & v == at)
    uv <- sort(unique(v[cont]))
    if (length(uv) < 2L) next
    dm <- if (length(domains) < j) NA_character_ else domains[[j]]
    tr <- .nl_cell_partition(uv, dm)$tr
    x <- v[cont]
    log_dc <- if (isTRUE(sp$log_scale)) -log(x) else 0
    term <- rep(0, length(v))
    term[cont] <- .hyper_axis_log_unit(x, sp) + log_dc +
                  tr$log_dfrom(tr$to(x))
    term[!is.finite(term)] <- 0
    out <- out + term
  }
  out
}

# The reconstruction of ONE axis's continuum. `v` / `w` are the cells'
# coordinates and masses, `declared` the levels the axis was declared on
# (`.nl_level_edges()`), `row_id` each cell's row (`.nl_axis_row_id()`) and
# `log_density` each cell's node density on the read coordinate
# (`.nl_read_log_density()`). Returns either
# `list(declined = <reason>)` or the tables: `x` the grid on the coordinate
# `tr`, `F` the axis CDF on it, `pieces` the rows' density parameters,
# `mass` the rows' masses, `lq_mass` the share of the mass on
# rows that were interpolated, `h_over_sd` the heaviest such row's resolution,
# and per cell its row and the CDF interval `[c0, c1]` it owns inside that row
# (NA for a cell that holds no mass).
.nl_lq_reconstruct <- function(v, w, domain = NA_character_, declared = NULL,
                               row_id = NULL, log_density = NULL) {
  if (is.null(row_id) || length(row_id) != length(v)) {
    return(list(declined = "not_a_grid_axis"))
  }
  if (is.null(log_density) || length(log_density) != length(v)) {
    return(list(declined = "no_node_density"))
  }
  fin <- is.finite(v)
  wpos <- fin & is.finite(w) & w > 0
  if (!any(wpos)) return(list(declined = "no_usable_node"))
  uv <- sort(unique(v[fin]))
  if (length(uv) < 2L) return(list(declined = "single_node"))
  le <- .nl_level_edges(uv, domain, declared)
  if (is.null(le)) return(list(declined = "boxes_do_not_tile"))
  # A lowest cell reaching down to a point mass at zero has no finite width on
  # the log coordinate the rows are interpolated on, so the axis is read by
  # boxes on the natural scale instead.
  if (!is.finite(le$pt$tr$to(le$e[1L]))) return(list(declined = "floor_cell"))
  k <- match(v, uv)
  bx <- list(lo = le$e[k], hi = le$e[k + 1L], tr = le$pt$tr,
             coord = le$pt$coord, declined = le$pt$declined)
  tr <- bx$tr
  u  <- tr$to(v)
  ulo <- tr$to(bx$lo)
  uhi <- tr$to(bx$hi)
  ld_c <- as.numeric(log_density)
  ok <- wpos & is.finite(u) & is.finite(ulo) & is.finite(uhi) & uhi > ulo &
        is.finite(ld_c)
  if (!any(ok)) return(list(declined = "no_usable_node"))

  laid <- range(u[fin & is.finite(u)])
  row_cells <- .nl_lq_group_cells(row_id, which(ok))
  pieces <- lapply(row_cells, function(cells) {
    .nl_lq_group(cells, u, ulo, uhi, ld_c, w, laid = laid)
  })
  mass <- vapply(pieces, `[[`, numeric(1), "mass")
  is_lq <- vapply(pieces, function(p) identical(p$kind, "lq"), logical(1))
  hsd <- vapply(pieces, function(p) p$h_sd %||% NA_real_, numeric(1))
  heavy <- if (any(is.finite(hsd))) {
    hsd[is.finite(hsd)][which.max(mass[is.finite(hsd)])]
  } else NA_real_
  if (!any(is_lq)) return(list(declined = "unresolved", h_over_sd = heavy))

  tab <- .nl_lq_table(
    c(min(vapply(pieces, `[[`, numeric(1), "lo")),
      max(vapply(pieces, `[[`, numeric(1), "hi"))),
    c(min(vapply(pieces, function(p) min(p$lo_b), numeric(1))),
      max(vapply(pieces, function(p) max(p$hi_b), numeric(1)))),
    min(vapply(pieces, `[[`, numeric(1), "min_gap")))
  x <- tab$x

  # Each row's density on `x`, unnormalized and zero outside its extent: the
  # interpolated log density plus the slab of the box `x` falls in. Past the
  # outer boxes the outer cells' slabs carry the tails; a stretch between two of
  # the row's boxes that no cell of the row covers is held by other rows and is
  # zero here, as it is in the box read. Tabulated and integrated to the row's
  # normalized CDF in C++ (src/nl_lq_density.cpp), accumulated one row at a
  # time into the axis's CDF slopes.
  mixture <- cpp_nl_lq_mixture_cdf(pieces, mass, x, tab$r)
  bad <- !mixture$usable
  if (all(bad)) return(list(declined = "no_usable_node"))
  mass[bad] <- 0
  Fax <- mixture$F

  c0 <- c1 <- rep(NA_real_, length(v))
  cg <- rep(NA_integer_, length(v))
  for (i in seq_along(pieces)) {
    if (bad[i]) next
    p <- pieces[[i]]
    cg[p$cells] <- i
    c0[p$cells] <- p$c0
    c1[p$cells] <- p$c1
  }
  list(declined = NA_character_, tr = tr, x = x, r = tab$r, F = Fax,
       pieces = pieces, mass = mass,
       lq_mass = sum(mass[is_lq & !bad]) / sum(mass[!bad]),
       group = cg, c0 = c0, c1 = c1, lo = bx$lo, hi = bx$hi,
       h_over_sd = heavy, coord = bx$coord, edge_declined = bx$declined)
}

# The table the rows are tabulated on: `x` over `extent`, the rows' reach, and
# `r`, each interval's width in units of the first. `core` is the span of the
# nodes' boxes and `min_gap` the closest node gap. The spacing is at most
# `min_gap / 8`, with at least `lq_grid_points` points; when that spacing
# reaches the whole extent within `8 lq_grid_points + 1` points the table is
# one uniform grid over it. Otherwise the core takes that rule on its own span
# (within the same budget) and the stretches of the extent past it are laid by
# `.nl_lq_tail_points()`. The core is widened by one spacing on each side, so a
# box row, which lives inside the core, meets zeros on every interval of
# another width.
.nl_lq_table <- function(extent, core, min_gap) {
  m0 <- as.integer(.nl_diag("lq_grid_points"))
  cap <- 8L * m0 + 1L
  need <- ceiling(diff(extent) / (min_gap / 8)) + 1L
  if (need <= cap) {
    m <- max(m0, need)
    return(list(x = seq(extent[1L], extent[2L], length.out = m),
                r = rep(1, m - 1L)))
  }
  h <- max(min(min_gap / 8, diff(core) / (m0 - 1L)), diff(core) / (cap - 3L))
  lo <- max(extent[1L], core[1L] - h)
  hi <- min(extent[2L], core[2L] + h)
  nc <- max(2L, ceiling((hi - lo) / h) + 1L)
  xc <- seq(lo, hi, length.out = nc)
  hc <- xc[2L] - xc[1L]
  x <- c(rev(lo - .nl_lq_tail_points(lo - extent[1L], hc)), xc,
         hi + .nl_lq_tail_points(extent[2L] - hi, hc))
  dx <- diff(x)
  list(x = x, r = dx / dx[1L])
}

# Offsets past a core edge out to `reach`, starting at spacing `h` and growing
# each step by 1/64 of the offset already covered; the last one is `reach`.
.nl_lq_tail_points <- function(reach, h) {
  if (!(reach > 0)) return(numeric(0))
  g <- 1 + 1 / 64
  k <- ceiling(log1p(reach * (g - 1) / h) / log(g))
  d <- h * (g^seq_len(k) - 1) / (g - 1)
  d <- d[d < reach]
  c(d, reach)
}

.nl_lq_group_cells <- function(group, cells = which(!is.na(group))) {
  unname(split(cells, factor(group[cells], levels = sort(unique(group[cells])))))
}

# One row's nodes, its slabs, the quadratics through its log densities and the
# extent its density reaches. Cells repeating a coordinate within a row are one
# node: the level takes their mean log density and their summed mass, and its
# CDF interval is theirs jointly. A row that does not qualify (fewer than three
# nodes, its mode on an end node, a non-concave modal triple, or `h / sd` above
# the gate) is its cells' boxes, each holding its own mass -- the box read of
# that row.
#
# `laid` is the range of the axis's levels over every cell, those carrying no
# mass included. A tail continues a row past its outer node only where the axis
# has no level beyond that node: the tail is what removes the truncation of the
# grid's own extent, and inside that extent the levels the row carries no mass
# at -- a cell pruned by the screen, one solved to a negligible weight, or a
# level a refinement pass laid only in the rows holding the posterior -- say
# that the row's density there is negligible. A tail running across them puts
# there the mass the grid measured it not to hold, so such an end stops at its
# node's box edge.
.nl_lq_group <- function(cells, u, ulo, uhi, ld, w, laid = range(u)) {
  uv <- sort(unique(u[cells]))
  k <- match(u[cells], uv)
  f <- factor(k, levels = seq_along(uv))
  lgk <- as.numeric(tapply(ld[cells], f, mean))
  mk <- as.numeric(tapply(w[cells], f, sum))
  first <- match(seq_along(uv), k)
  lo_b <- ulo[cells][first]
  hi_b <- uhi[cells][first]
  cm <- c(0, cumsum(mk)) / sum(mk)
  n <- length(uv)
  p <- list(cells = cells, u = uv, lo_b = lo_b, hi_b = hi_b, mass = sum(mk),
            c0 = cm[k], c1 = cm[k + 1L],
            min_gap = if (n > 1L) min(diff(uv)) else hi_b - lo_b)
  as_box <- function(p) {
    p$kind <- "box"
    p$lg <- log(mk / (hi_b - lo_b))
    p$lo <- min(lo_b)
    p$hi <- max(hi_b)
    p
  }
  if (n < 3L) return(as_box(p))
  p$lg <- lgk - max(lgk)
  # Newton form of the quadratic through nodes (i, i + 1, i + 2):
  # lg_i + d1_i (x - u_i) + d2_i (x - u_i) (x - u_{i+1}), curvature d2_i.
  s <- diff(p$lg) / diff(uv)
  p$d1 <- s[-(n - 1L)]
  p$d2 <- diff(s) / (uv[-(1:2)] - uv[-((n - 1L):n)])
  im <- which.max(p$lg)
  if (im == 1L || im == n || p$d2[im - 1L] >= 0) return(as_box(p))
  sd_r <- sqrt(-1 / (2 * p$d2[im - 1L]))
  # The row's spacing where its mass sits: the median of its node gaps, each
  # weighted by the mass of the two nodes bounding it. On an evenly spaced row
  # it is the spacing; the levels a refinement pass lays out along a tail carry
  # little mass and leave it where the posterior is.
  gap <- diff(uv)
  gw <- mk[-1L] + mk[-n]
  o <- order(gap)
  p$h_sd <- gap[o][which(cumsum(gw[o]) >= sum(gw) / 2)[1L]] / sd_r
  if (!is.finite(p$h_sd) || p$h_sd > .nl_diag("lq_max_h_over_sd")) {
    return(as_box(p))
  }
  p$kind <- "lq"
  # The slab each cell's box carries, on the log scale.
  p$lt <- log(mk) - p$lg - log(hi_b - lo_b)
  tail_nats <- .nl_diag("lq_tail_nats")
  tail_cap <- .nl_diag("lq_tail_sd") * sd_r
  top <- max(p$lg)
  reach <- function(i, side) {
    a <- p$d2[i]
    end <- if (side < 0) uv[1L] else uv[n]
    walled <- if (side < 0) laid[1L] < end else laid[2L] > end
    unseen <- top - p$lg[if (side < 0) 1L else n] < .nl_diag("lq_tail_min_fall")
    if (a >= 0 || walled || unseen) return(if (side < 0) lo_b[1L] else hi_b[n])
    # The quadratic as a(x - x0)^2 + c: its peak and where it falls
    # `tail_nats` below the larger of that peak and the row's top node.
    b <- p$d1[i] - a * (uv[i] + uv[i + 1L])
    x0 <- -b / (2 * a)
    pk <- .nl_lq_quad(p, i, x0)
    ref <- if ((x0 - end) * side > 0) max(top, pk) else top
    r <- sqrt(max(0, (ref - tail_nats - pk) / a))
    cut <- x0 + side * r
    if ((cut - end) * side < 0) cut <- end
    if (abs(cut - end) > tail_cap) cut <- end + side * tail_cap
    if (side < 0) min(cut, lo_b[1L]) else max(cut, hi_b[n])
  }
  p$lo <- reach(1L, -1)
  p$hi <- reach(n - 2L, 1)
  p
}

.nl_lq_quad <- function(p, i, x) {
  p$lg[i] + p$d1[i] * (x - p$u[i]) + p$d2[i] * (x - p$u[i]) * (x - p$u[i + 1L])
}

# Quantiles of the reconstruction at `probs`, back on the axis's own scale.
.nl_lq_cdf_quantile <- function(rc, probs) {
  rc$tr$from(.nl_lq_inverse(rc$x, rc$F, probs))
}

.nl_lq_inverse <- function(x, F, probs) {
  p <- pmin(pmax(probs, 0), 1)
  out <- rep(NA_real_, length(p))
  out[which(p == 0)] <- x[max(which(F == 0))]
  out[which(p == 1)] <- x[which(F >= 1)[1L]]
  use <- which(p > 0 & p < 1)
  k <- findInterval(p[use], F)
  out[use] <- x[k] + (p[use] - F[k]) / (F[k + 1L] - F[k]) * (x[k + 1L] - x[k])
  out
}

# The read `.nl_summary_quantile_read()` dispatches to, with the box read's
# return shape: `q` NULL and `declined` the reason when it did not run. A
# declared point mass is split off and the continuum read on the probabilities
# above it, as the box read does.
.nl_lq_quantile <- function(values, weights, probs, domain = NA_character_,
                            atom = NA_real_, declared = NULL, row_id = NULL,
                            log_density = NULL) {
  v <- as.numeric(values)
  w <- as.numeric(weights)
  fin <- is.finite(v)
  wpos <- fin & is.finite(w) & w > 0
  if (!any(wpos)) {
    return(list(q = rep(NA_real_, length(probs)), declined = "no_usable_node"))
  }
  if (length(atom) == 1L && is.finite(atom) && any(v[fin] == atom) &&
      !any(v[fin] < atom)) {
    ia <- fin & v == atom
    mass <- sum(w[wpos & ia]) / sum(w[wpos])
    if (mass >= 1) {
      return(list(q = rep(atom, length(probs)), declined = "single_node"))
    }
    r <- .nl_lq_quantile(v[!ia], w[!ia], .nl_atom_rescale(probs, mass), domain,
                         NA_real_, declared,
                         if (is.null(row_id)) NULL else row_id[!ia],
                         if (is.null(log_density)) NULL else log_density[!ia])
    if (!is.null(r$q)) r$q[probs <= mass] <- atom
    return(r)
  }
  rc <- .nl_lq_reconstruct(v, w, domain, declared, row_id, log_density)
  if (!is.na(rc$declined)) return(list(q = NULL, declined = rc$declined))
  list(q = .nl_lq_cdf_quantile(rc, probs), declined = NA_character_,
       edge_coord = rc$coord, edge_declined = rc$edge_declined)
}

# The draw-side geometry of a log-quadratic axis (`.nl_hyper_axis_geometry()`):
# per cell, its row's CDF table and the interval `[c0, c1]` of that CDF its
# level owns. `lo` / `hi` stay the cell's box, which is what the copula reads a
# node density off.
.nl_lq_axis_geometry <- function(v, w, domain, atom = NA_real_, declared = NULL,
                                 row_id = NULL, log_density = NULL) {
  ia <- length(atom) == 1L && is.finite(atom) && any(v == atom, na.rm = TRUE) &&
        !any(v < atom, na.rm = TRUE)
  cont <- if (ia) is.na(v) | v != atom else rep(TRUE, length(v))
  rc <- .nl_lq_reconstruct(v[cont], w[cont], domain, declared,
                           if (is.null(row_id)) NULL else row_id[cont],
                           if (is.null(log_density)) NULL else
                             log_density[cont])
  if (!is.na(rc$declined)) return(list(declined = rc$declined))
  n <- length(v)
  full <- function(x, fill) { o <- rep(fill, n); o[cont] <- x; o }
  lo <- full(rc$lo, NA_real_)
  hi <- full(rc$hi, NA_real_)
  if (ia) { lo[!cont] <- atom; hi[!cont] <- atom }
  list(kind = "log_quadratic", per_cell = TRUE, values = v, lo = lo, hi = hi,
       x = rc$x, r = rc$r, pieces = rc$pieces, from = rc$tr$from,
       group = full(rc$group, NA_integer_), c0 = full(rc$c0, NA_real_),
       c1 = full(rc$c1, NA_real_), declined = NA_character_)
}

# Cell `k`'s draw at uniform `u`: the point its row's CDF reaches at
# `c0 + u (c1 - c0)`. Summed over a row's cells by mass, the levels' CDF
# intervals tile `[0, 1]`, so the draws reproduce the row's reconstruction
# exactly, and the axis's whatever the cell allocation. NA for a cell outside
# every row (a declared point mass, a massless cell), which the caller keeps at
# its node. `u` is one uniform per cell, or a matrix with one row per cell and
# one column per uniform; each row's CDF is built once and inverted at all of
# its cells' uniforms.
.nl_lq_axis_draw <- function(geom, k, u) {
  U <- matrix(u, length(k))
  out <- matrix(NA_real_, length(k), ncol(U))
  g <- geom$group[k]
  for (s in .nl_lq_group_cells(g)) {
    gi <- g[s[1L]]
    c0 <- geom$c0[k[s]]
    p <- c0 + U[s, , drop = FALSE] * (geom$c1[k[s]] - c0)
    curve <- cpp_nl_lq_group_cdf(geom$pieces[[gi]], geom$x, geom$r)
    out[s, ] <- geom$from(.nl_lq_inverse(geom$x[curve$index], curve$F, p))
  }
  if (is.matrix(u)) out else out[, 1L]
}
