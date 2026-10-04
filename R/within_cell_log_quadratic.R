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
# cells' own `log_marginal` (the node density on the coordinate the cell measure
# is taken in, the axis's own), and `t` is each cell's slab,
# `w / (exp(log_marginal) width)`, held over that cell's box. The refinement
# passes add levels to the tensor, so `t` is one number along a row and the read
# is the plain interpolation; a row whose cells carry a measure the product rule
# did not build (a caller's own weights) is still read on its own slabs. The
# mass over width alone would fold the slab into the density.
#
# THE GATE, PER ROW. A quadratic through nodes several posterior SDs apart reads
# the location of a skewed conditional off its tails, and on such a grid the box
# read's cell-wide spread is what keeps the interval covering: on the coarse
# rungs of the gcol33/tulpa#357 ladder (h / sd 3 to 27) the log-quadratic read's
# summed |coverage - 0.95| was 2.20 against the box read's 0.34, the failure
# `cell_normal_read.patch` showed in gcol33/tulpa#925. So a row is interpolated
# only where it resolves its own conditional, `h / sd <=
# .nl_diag("lq_max_h_over_sd")`, with `h` the median node spacing of the row and
# `sd` the three-point parabola at its modal node; a row above it, one whose
# mode is an end node, and one with fewer than three positive-mass nodes are
# read as their cells' boxes. An axis on which no row qualifies declines to the
# box read as `"unresolved"`.
#
# A quantity that is not a grid axis -- an entry of a covariance assembled from
# several axes -- has no rows to read, and declines as `"not_a_grid_axis"`; a
# caller with no `log_marginal` to hand (a summary taken off weights alone) has
# no node density, and declines as `"no_node_density"`.
#
# THE RECONSTRUCTION IS TABULATED. Each row's density is evaluated on one grid
# over the axis's coordinate and integrated by the trapezoid rule; the axis's
# CDF is the mass-weighted sum of the rows' CDFs, the quantile is read off it by
# linear interpolation, and `tulpa_hyper_draws()` inverts the same tables
# (`.nl_lq_axis_draw()`), so the interval and the draws describe one
# distribution. A tail runs to `lq_tail_nats` below the row's peak, and no
# farther than `lq_tail_sd` of the row's SDs past its outer node: a quadratic
# whose curvature is far flatter than the row's own would otherwise stretch the
# table over a range where its spacing no longer resolves the nodes.

# The reconstruction of ONE axis's continuum. `v` / `w` are the cells'
# coordinates and masses, `declared` the levels the axis was declared on
# (`.nl_level_edges()`), `row_id` each cell's row (`.nl_axis_row_id()`) and
# `log_density` each cell's `log_marginal`. Returns either
# `list(declined = <reason>)` or the tables: `x` the grid on the coordinate
# `tr`, `F` the axis CDF on it, `G` the per-row CDFs (one column per row, each
# running 0 to 1), `mass` the rows' masses, `lq_mass` the share of the mass on
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

  groups <- sort(unique(row_id[ok]))
  pieces <- lapply(groups, function(g) {
    .nl_lq_group(which(ok & row_id == g), u, ulo, uhi, ld_c, w)
  })
  mass <- vapply(pieces, `[[`, numeric(1), "mass")
  is_lq <- vapply(pieces, function(p) identical(p$kind, "lq"), logical(1))
  hsd <- vapply(pieces, function(p) p$h_sd %||% NA_real_, numeric(1))
  heavy <- if (any(is.finite(hsd))) {
    hsd[is.finite(hsd)][which.max(mass[is.finite(hsd)])]
  } else NA_real_
  if (!any(is_lq)) return(list(declined = "unresolved", h_over_sd = heavy))

  lo_x <- min(vapply(pieces, `[[`, numeric(1), "lo"))
  hi_x <- max(vapply(pieces, `[[`, numeric(1), "hi"))
  min_gap <- min(vapply(pieces, `[[`, numeric(1), "min_gap"))
  m <- as.integer(.nl_diag("lq_grid_points"))
  need <- ceiling((hi_x - lo_x) / (min_gap / 8)) + 1L
  m <- max(m, min(need, 8L * m + 1L))
  x <- seq(lo_x, hi_x, length.out = m)

  # Each row's density on `x`, unnormalized and zero outside its extent: the
  # interpolated log density plus the slab of the box `x` falls in. Past the
  # outer boxes the outer cells' slabs carry the tails; a stretch between two of
  # the row's boxes that no cell of the row covers is held by other rows and is
  # zero here, as it is in the box read. Tabulated and integrated to the row's
  # normalized CDF in C++ (src/nl_lq_density.cpp); NA where the row carries no
  # mass on the grid.
  G <- vapply(pieces, function(p) {
    is_lq <- identical(p$kind, "lq")
    cpp_nl_lq_group_cdf(is_lq, p$u, p$lg,
                        if (is_lq) p$d1 else numeric(0),
                        if (is_lq) p$d2 else numeric(0),
                        if (is_lq) p$lt else numeric(0),
                        p$lo_b, p$hi_b, p$lo, p$hi, x)
  }, numeric(m))
  G <- matrix(G, m)
  bad <- is.na(G[m, ])
  if (all(bad)) return(list(declined = "no_usable_node"))
  mass[bad] <- 0
  Fax <- drop(G[, !bad, drop = FALSE] %*% (mass[!bad] / sum(mass[!bad])))

  c0 <- c1 <- rep(NA_real_, length(v))
  cg <- rep(NA_integer_, length(v))
  for (i in seq_along(pieces)) {
    if (bad[i]) next
    p <- pieces[[i]]
    cg[p$cells] <- i
    c0[p$cells] <- p$c0
    c1[p$cells] <- p$c1
  }
  list(declined = NA_character_, tr = tr, x = x, F = Fax, G = G, mass = mass,
       lq_mass = sum(mass[is_lq & !bad]) / sum(mass[!bad]),
       group = cg, c0 = c0, c1 = c1, lo = bx$lo, hi = bx$hi,
       h_over_sd = heavy, coord = bx$coord, edge_declined = bx$declined)
}

# One row's nodes, its slabs, the quadratics through its log densities and the
# extent its density reaches. Cells repeating a coordinate within a row are one
# node: the level takes their mean log density and their summed mass, and its
# CDF interval is theirs jointly. A row that does not qualify (fewer than three
# nodes, its mode on an end node, a non-concave modal triple, or `h / sd` above
# the gate) is its cells' boxes, each holding its own mass -- the box read of
# that row.
.nl_lq_group <- function(cells, u, ulo, uhi, ld, w) {
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
  p$h_sd <- stats::median(diff(uv)) / sd_r
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
    if (a >= 0) return(if (side < 0) lo_b[1L] else hi_b[n])
    end <- if (side < 0) uv[1L] else uv[n]
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
  keep <- c(TRUE, diff(rc$F) > 0)
  xq <- stats::approx(rc$F[keep], rc$x[keep], xout = pmin(pmax(probs, 0), 1),
                      ties = "ordered", rule = 2)$y
  rc$tr$from(xq)
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
       x = rc$x, G = rc$G, from = rc$tr$from,
       group = full(rc$group, NA_integer_), c0 = full(rc$c0, NA_real_),
       c1 = full(rc$c1, NA_real_), declined = NA_character_)
}

# Cell `k`'s draw at uniform `u`: the point its row's CDF reaches at
# `c0 + u (c1 - c0)`. Summed over a row's cells by mass, the levels' CDF
# intervals tile `[0, 1]`, so the draws reproduce the row's reconstruction
# exactly, and the axis's whatever the cell allocation. NA for a cell outside
# every row (a declared point mass, a massless cell), which the caller keeps at
# its node.
.nl_lq_axis_draw <- function(geom, k, u) {
  out <- rep(NA_real_, length(k))
  g <- geom$group[k]
  for (gi in unique(stats::na.omit(g))) {
    s <- which(!is.na(g) & g == gi)
    p <- geom$c0[k[s]] + u[s] * (geom$c1[k[s]] - geom$c0[k[s]])
    Fg <- geom$G[, gi]
    keep <- c(TRUE, diff(Fg) > 0)
    out[s] <- geom$from(stats::approx(Fg[keep], geom$x[keep], xout = p,
                                      ties = "ordered", rule = 2)$y)
  }
  out
}
