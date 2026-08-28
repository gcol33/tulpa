# The local-quadratic mass barycentre of an outer tensor cell
# (gcol33/tulpa#327).
#
# The per-axis read places a cell's whole mass at the cell's own coordinate, so a
# cell whose log-marginal carries a gradient is read at a point its mass is not
# centred on. `.joint_local_ccd_barycentre()` moves each cell to the first moment
# of its own local quadratic over its own box, off the same three-point stencil
# `.joint_local_ccd_box_mass()` (gcol33/tulpa#326) reads the mass from, so it
# costs no inner solve. What is checked here is the closed form against
# quadrature, the per-axis handling of a convex axis, that a barycentre stays
# inside its own cell, and what the rule does to the reported read through the
# #322 harness at zero fitting minutes.

# The barycentre by composite Simpson on the max-shifted exponent. Independent of
# `stats::integrate()`, so the estimator's numeric route is not being compared
# against itself.
.bar_simpson <- function(g, a, h_lo, h_hi, m_pts = 200001L) {
  expo <- function(u) g * u - 0.5 * a * u^2
  cand <- c(-h_lo, h_hi)
  if (a > 0) cand <- c(cand, min(max(g / a, -h_lo), h_hi))
  mx <- max(expo(cand))
  u <- seq(-h_lo, h_hi, length.out = m_pts)
  w <- c(1, rep(c(4, 2), length.out = m_pts - 2L), 1)
  e <- exp(expo(u) - mx)
  sum(w * u * e) / sum(w * e)
}

# The same ratio through `stats::integrate()`, the arbiter the issue names.
.bar_integrate <- function(g, a, h_lo, h_hi) {
  expo <- function(u) g * u - 0.5 * a * u^2
  cand <- c(-h_lo, h_hi)
  if (a > 0) cand <- c(cand, min(max(g / a, -h_lo), h_hi))
  mx <- max(expo(cand))
  q <- function(f) stats::integrate(f, -h_lo, h_hi,
                                    rel.tol = .Machine$double.eps^0.75)$value
  q(function(u) u * exp(expo(u) - mx)) / q(function(u) exp(expo(u) - mx))
}

# --------------------------------------------------------------------------- #
# The closed form                                                             #
# --------------------------------------------------------------------------- #

test_that("the barycentre is the box's own first moment, on either sign of a", {
  # Unequal half-widths on both sides, since a cell's Voronoi box need not be
  # symmetric, and curvatures spanning concave / flat / convex, since the first
  # moment over a BOUNDED box is finite at either sign.
  gr <- expand.grid(g = c(0, 0.25, 1, 2, 4.44, -3, 10, -8),
                    a = c(-2, -0.5, -0.05, 0, 0.05, 0.5, 1, 3, 20),
                    h_lo = c(0.25, 1, 2), h_hi = c(0.25, 1, 2))
  e_int <- numeric(nrow(gr)); e_sim <- numeric(nrow(gr))
  route <- character(nrow(gr)); inbox <- logical(nrow(gr))
  for (i in seq_len(nrow(gr))) {
    f <- tulpa:::.joint_local_ccd_axis_bary(gr$g[i], gr$a[i], gr$h_lo[i],
                                            gr$h_hi[i])
    # Every combination has an answer: the axis is never declined for the sign
    # of its curvature.
    expect_false(is.null(f$u_bar))
    expect_true(is.na(f$reason))
    route[i] <- f$route
    inbox[i] <- f$u_bar >= -gr$h_lo[i] && f$u_bar <= gr$h_hi[i]
    e_int[i] <- abs(f$u_bar - .bar_integrate(gr$g[i], gr$a[i], gr$h_lo[i],
                                             gr$h_hi[i]))
    e_sim[i] <- abs(f$u_bar - .bar_simpson(gr$g[i], gr$a[i], gr$h_lo[i],
                                           gr$h_hi[i]))
  }
  expect_setequal(unique(route), c("closed", "numeric"))
  # Measured over the 648 combinations, in u-space units: 2.36e-11 against both
  # arbiters, which agree with each other to the same figure. The worst case is
  # the far-tail closed form (g = -8, a = 0.05, so a local peak 640 nats up and
  # 320 cell widths away), where `eps` times the cancellation product is 4.7e-11
  # of the cell width -- the budget `.LCCD_BAR_CANCEL` is set to. The numeric
  # route measures 1.55e-15. The tolerances are floating slack over those.
  expect_lt(max(e_int), 1e-9)
  expect_lt(max(e_sim), 1e-9)
  # The numeric route asks stats::integrate() for rel.tol = eps^0.75 on each of
  # the two integrals it forms the ratio from, so what it guarantees is about
  # twice that relative, and the box half-widths here are O(1). The bound is
  # that contract with slack. What any one platform measures inside it is the
  # arithmetic it happened to do: where the adaptive subdivision lands moves
  # with the libm, and the same grid measures 1.55e-15 on one and 2e-12 on
  # another.
  expect_lt(max(e_sim[route == "numeric"]), 8 * .Machine$double.eps^0.75)

  # A barycentre is the first moment of a positive density over the box, so it
  # is a convex combination of points in the box and cannot leave it. The
  # assertion is on the arithmetic, not on the integral.
  expect_true(all(inbox))
})

test_that("a symmetric cell with no gradient keeps its own coordinate", {
  # `g = 0` on a symmetric box is the whole content of the current placement,
  # and it is recovered exactly rather than to within a tolerance on the closed
  # route: the two standardized edges are equidistant from the local peak, so
  # the numerator difference is exactly zero and the barycentre is `mu = 0`. A
  # rule that jitters the cells it has nothing to say about is not a placement
  # rule.
  for (a in c(0.05, 3, 20)) {
    f <- tulpa:::.joint_local_ccd_axis_bary(0, a, 1, 1)
    expect_identical(f$route, "closed")
    expect_identical(f$u_bar, 0)
  }
  for (a in c(-2, -0.05, 0)) {
    f <- tulpa:::.joint_local_ccd_axis_bary(0, a, 1, 1)
    expect_identical(f$route, "numeric")
    expect_lt(abs(f$u_bar), 1e-14)
  }

  # An UNEQUAL box is the case the grid actually presents, and there `g = 0`
  # does not mean unmoved: the cell's coordinate is not the centre of mass of a
  # box that reaches further one way than the other. The barycentre is that
  # centre of mass, which is the box midpoint on a flat cell and is pulled back
  # toward the coordinate as the curvature concentrates the mass on it.
  h <- c(0.3, 1.7)
  flat <- tulpa:::.joint_local_ccd_axis_bary(0, 0, h[1L], h[2L])
  expect_equal(flat$u_bar, 0.5 * (h[2L] - h[1L]), tolerance = 1e-12)
  prev <- flat$u_bar
  for (a in c(0.05, 3, 20)) {
    f <- tulpa:::.joint_local_ccd_axis_bary(0, a, h[1L], h[2L])
    expect_equal(f$u_bar, .bar_simpson(0, a, h[1L], h[2L]), tolerance = 1e-11)
    expect_lt(f$u_bar, prev)
    expect_gt(f$u_bar, 0)
    prev <- f$u_bar
  }
})

test_that("the closed form covers the concave axis and quadrature the rest", {
  # A concave axis with the peak within reach of the box is the Gaussian case
  # and needs no quadrature.
  for (p in list(c(0.5, 1), c(2, 3), c(4.44, 20), c(-3, 0.5), c(10, 0.02))) {
    f <- tulpa:::.joint_local_ccd_axis_bary(p[1L], p[2L], 1, 1)
    expect_identical(f$route, "closed")
    expect_equal(f$u_bar, .bar_integrate(p[1L], p[2L], 1, 1), tolerance = 1e-9)
  }
  # A convex axis has no `pnorm` difference to form at all, and a near-flat
  # concave one sends `mu = g/a` out of double range against a vanishing
  # correction. Both take the bounded quadrature, which never sees `mu`.
  for (p in list(c(0.5, -0.5), c(-2, -2), c(0, 1e-14), c(5, 1e-8),
                 c(30, 0.01), c(1e3, 1e-3))) {
    f <- tulpa:::.joint_local_ccd_axis_bary(p[1L], p[2L], 1, 1)
    expect_identical(f$route, "numeric")
    expect_equal(f$u_bar, .bar_simpson(p[1L], p[2L], 1, 1), tolerance = 1e-11)
  }
  # A degenerate box carries no barycentre rather than a guessed one, and says
  # so: the quadratic never formed, which is a different refusal from the
  # conditioning ones below.
  for (p in list(c(1, 1, 0, 0), c(NA_real_, 1, 1, 1), c(1, 1, -1, 1))) {
    f <- tulpa:::.joint_local_ccd_axis_bary(p[1L], p[2L], p[3L], p[4L])
    expect_null(f$u_bar)
    expect_identical(f$reason, "no_factor")
  }
})

# --------------------------------------------------------------------------- #
# The cell                                                                    #
# --------------------------------------------------------------------------- #

test_that("a cell convex on one axis keeps the exact barycentre on the others", {
  # The same separability the box mass rests on: with `A` diagonal the local
  # quadratic factorizes, so the cell's barycentre is the vector of its axes'
  # and a fourth axis the finite difference came back convex on costs the other
  # three nothing.
  st <- list(g = c(1.0, -0.5, 2.0, 0.3), d2 = c(-2, -1, -0.5, 0.8),
             half_lo = rep(0.5, 4L), half_hi = rep(0.5, 4L))
  bc <- tulpa:::.joint_local_ccd_cell_bary(st)
  expect_identical(bc$n_axes_declined, 0L)
  expect_identical(bc$n_axes_closed, 3L)
  expect_identical(bc$n_axes_numeric, 1L)
  expect_equal(bc$u_bar,
               vapply(seq_len(4L), function(j)
                 .bar_simpson(st$g[j], -st$d2[j], st$half_lo[j], st$half_hi[j]),
                 numeric(1)), tolerance = 1e-11)

  # `bary_shift` is the largest share of its own half-cell any axis's atom
  # moves, measured against the half-width on the side it moves toward, so the
  # in-box property bounds it in [0, 1].
  share <- ifelse(bc$u_bar >= 0, bc$u_bar / st$half_hi, -bc$u_bar / st$half_lo)
  expect_equal(bc$bary_shift, max(share), tolerance = 1e-12)
  expect_gte(bc$bary_shift, 0)
  expect_lte(bc$bary_shift, 1)

  # A flat cell does not move on any axis.
  flat <- tulpa:::.joint_local_ccd_cell_bary(
    list(g = rep(0, 3L), d2 = rep(0, 3L), half_lo = rep(1, 3L),
         half_hi = rep(1, 3L)))
  expect_identical(flat$u_bar, rep(0, 3L))
  expect_identical(flat$bary_shift, 0)
})

# --------------------------------------------------------------------------- #
# The grid                                                                    #
# --------------------------------------------------------------------------- #

test_that("only interior cells move, and they move within their own box", {
  mu <- c(0.3, -0.2); s <- c(0.8, 1.3)
  lv <- list(a = c(-1, 0, 1), b = c(-1, 0, 1))
  gg <- as.matrix(expand.grid(lv, KEEP.OUT.ATTRS = FALSE))
  colnames(gg) <- names(lv)
  lm <- -0.5 * rowSums(sweep(sweep(gg, 2L, mu, "-")^2, 2L, s^2, "/"))
  tags <- c("identity", "identity")

  bc <- tulpa:::.joint_local_ccd_barycentre(gg, lm, colnames(gg), tags)
  interior <- gg[, 1L] == 0 & gg[, 2L] == 0
  expect_identical(bc$computed, unname(interior))
  # A boundary cell has no centred stencil, so it keeps the coordinate it had --
  # bit for bit, not to a tolerance.
  expect_identical(bc$joint_grid[!interior, ], gg[!interior, ])
  expect_true(all(bc$u_offset[!interior, ] == 0))
  expect_identical(bc$n_axes_declined, 0L)
  # The eight boundary cells account for their own axes under `boundary`, two
  # apiece, the stencil being a cell-level object; no axis that reached a
  # per-axis gate fell at one (gcol33/tulpa#334).
  expect_identical(bc$declined_reasons,
                   c(boundary = 16L, no_factor = 0L, cancellation = 0L,
                     out_of_box = 0L))
  expect_identical(bc$n_axes_declined,
                   sum(bc$declined_reasons[c("no_factor", "cancellation",
                                             "out_of_box")]))

  # The interior cell moves toward the peak on both axes and stays inside its
  # own Voronoi half-box, which on this grid is half a level in each direction.
  nb <- tulpa:::.joint_local_ccd_neighbors(gg, gg, seq_len(2L))
  st <- tulpa:::.joint_local_ccd_cell_stencil(which(interior), gg, lm,
                                              nb$up, nb$dn)
  expect_equal(bc$u_offset[interior, ],
               vapply(1:2, function(j)
                 .bar_simpson(st$g[j], -st$d2[j], st$half_lo[j], st$half_hi[j]),
                 numeric(1)), tolerance = 1e-11, ignore_attr = TRUE)
  expect_true(all(abs(bc$u_offset[interior, ]) < 0.5))
  expect_equal(sign(bc$u_offset[interior, ]), sign(mu), ignore_attr = TRUE)

  # An unguessable-support grid declines whole rather than guessing a transform,
  # the same decline the box mass makes.
  none <- tulpa:::.joint_local_ccd_barycentre(gg, lm, colnames(gg), NULL)
  expect_false(any(none$computed))
  expect_identical(none$joint_grid, gg)
  expect_true(all(none$declined_reasons == 0L))
  expect_identical(names(none$declined_reasons), tulpa:::.LCCD_BARY_REASONS)
})

# --------------------------------------------------------------------------- #
# What a declined axis declined at (gcol33/tulpa#334)                         #
# --------------------------------------------------------------------------- #

test_that("a declined barycentre axis names the gate it fell at", {
  # `cancellation`: the local peak sits 5e11 nats above a box it is half a
  # million widths away from, so the closed form's `mu + corr` is refused before
  # it is formed (`.LCCD_BAR_CANCEL`) and the quadrature that takes over is
  # handed a needle 1e-6 wide in a box of width 2 and returns nothing either.
  expect_gt(0.5 * 1e6^2 / 1 * max(1, 1e6 / 2), tulpa:::.LCCD_BAR_CANCEL)
  f <- tulpa:::.joint_local_ccd_axis_bary(1e6, 1, 1, 1)
  expect_null(f$u_bar)
  expect_identical(f$reason, "cancellation")

  # `out_of_box`: 700 nats of gradient across a box 1e-6 wide, whose whole
  # integral is then of order 1e-9. `stats::integrate()` defaults its absolute
  # tolerance to its relative one, so on integrals that small the absolute
  # criterion is met long before the needle at the lower edge is resolved, and
  # the two quadratures the ratio is formed from stop in different places. The
  # first moment of a positive density over a box is a convex combination of
  # points in the box and cannot leave it, so a value outside is the arithmetic
  # failing and the axis keeps the cell's own coordinate rather than shipping an
  # atom into the neighbouring cell's territory.
  f <- tulpa:::.joint_local_ccd_axis_bary(-7e8, 0, 1e-6, 1e-8)
  expect_null(f$u_bar)
  expect_identical(f$reason, "out_of_box")
  # Exactly the same shape of cell one decade wider is answered, and answered
  # correctly: the refusal is a property of the quadrature's own scale, not of
  # the cell's geometry.
  ok <- tulpa:::.joint_local_ccd_axis_bary(-7e3, 0, 1e-1, 1e-3)
  expect_identical(ok$route, "numeric")
  expect_true(is.na(ok$reason))
  expect_equal(ok$u_bar, .bar_simpson(-7e3, 0, 1e-1, 1e-3), tolerance = 1e-9)
})

test_that("a cell's decline tally is per axis, alongside the count", {
  # Four axes: one concave and answered in closed form, one whose closed form is
  # refused for conditioning and whose quadrature returns nothing, one with no
  # extent at all, and one whose quadrature puts the atom outside its own box.
  # Three declines, three different remedies, one count.
  st <- list(g = c(1.0, 1e6, 0.3, -7e8), d2 = c(-2, -1, 0, 0),
             half_lo = c(0.5, 1, 0, 1e-6), half_hi = c(0.5, 1, 0, 1e-8))
  bc <- tulpa:::.joint_local_ccd_cell_bary(st)
  expect_identical(bc$n_axes_declined, 3L)
  expect_identical(bc$declined_reasons,
                   c(boundary = 0L, no_factor = 1L, cancellation = 1L,
                     out_of_box = 1L))
  expect_identical(names(bc$declined_reasons), tulpa:::.LCCD_BARY_REASONS)

  # A declined axis keeps the cell's own coordinate, and the one axis that was
  # read still moves: the decline is per axis, so three unreadable axes do not
  # cost the fourth its barycentre.
  expect_identical(bc$n_axes_closed, 1L)
  expect_identical(bc$n_axes_numeric, 0L)
  expect_equal(bc$u_bar,
               c(.bar_simpson(1.0, 2, 0.5, 0.5), 0, 0, 0), tolerance = 1e-11)
})

test_that("the moved coordinate is the u-space barycentre through the axis map", {
  # The grid stores physical theta while the quadratic is fitted in
  # `u = log theta`, so the shipped coordinate is `exp` of the u-space
  # barycentre: a mean in `u` mapped through the axis's own monotone transform,
  # which is what the quantile read wants (a representative point of the cell)
  # and is NOT what the moment read would want.
  lv <- exp(seq(log(0.2), log(2), length.out = 5L))
  gg <- as.matrix(expand.grid(a = lv, b = lv, KEEP.OUT.ATTRS = FALSE))
  U <- log(gg)
  lm <- -0.5 * ((U[, 1L] - 0.15)^2 / 0.4^2 + (U[, 2L] + 0.3)^2 / 0.6^2)
  bc <- tulpa:::.joint_local_ccd_barycentre(gg, lm, colnames(gg),
                                            c(a = "log", b = "log"))
  expect_gt(sum(bc$computed), 0L)
  expect_equal(bc$joint_grid, gg * exp(bc$u_offset), tolerance = 1e-14)
  # And the moved atom is still inside its own cell on the physical axis, which
  # `exp` being monotone is what guarantees: it sits strictly between the
  # midpoints to its two neighbours.
  for (j in 1:2) {
    for (c in which(bc$computed)) {
      lo <- max(lv[lv < gg[c, j]]); hi <- min(lv[lv > gg[c, j]])
      expect_gte(bc$joint_grid[c, j], sqrt(lo * gg[c, j]))
      expect_lte(bc$joint_grid[c, j], sqrt(hi * gg[c, j]))
    }
  }
})

test_that("a refined cell records its barycentre shift beside its box mass", {
  mu <- c(0.5, 0.5); s <- c(0.3, 0.3)
  lv <- list(a = c(-1, 0.5, 2), b = c(-1, 0.5, 2))
  gg <- as.matrix(expand.grid(lv, KEEP.OUT.ATTRS = FALSE))
  colnames(gg) <- names(lv)
  lmf <- function(th) -0.5 * rowSums(sweep(sweep(th, 2L, mu, "-")^2, 2L, s^2, "/"))
  out <- tulpa:::.joint_local_ccd_refine(
    gg, lmf(gg), NULL, NULL, colnames(gg), c(a = "identity", b = "identity"),
    eval_nodes = function(theta_mat, warm)
      list(log_marginal = lmf(theta_mat), modes = NULL),
    max_cells = 4L)
  nc <- out$info$n_cells_refined
  expect_gt(nc, 0L)
  expect_length(out$info$bary_shift, nc)
  expect_true(all(is.finite(out$info$bary_shift)))
  expect_true(all(out$info$bary_shift >= 0 & out$info$bary_shift <= 1))

  # It is the grid-wide rule's reading of the same cells, so the two entry
  # points cannot drift apart.
  bc <- tulpa:::.joint_local_ccd_barycentre(gg, lmf(gg), colnames(gg),
                                            c("identity", "identity"))
  expect_equal(out$info$bary_shift, bc$bary_shift[out$info$cells],
               tolerance = 1e-12)
})

# --------------------------------------------------------------------------- #
# The measurement (gcol33/tulpa#322's harness, no fitting minutes)            #
# --------------------------------------------------------------------------- #

# The two candidates, each entering the harness the way that kind of rule does:
# the mass multiplier as a per-cell design weight, the barycentre as a perturbed
# coordinate matrix. Their combination is the pair.
.bar_mass_w <- function(d) {
  bm <- tulpa:::.joint_local_ccd_box_mass(d$joint_grid, d$log_marginal,
                                          d$axis_names, d$axis_tags)
  base <- if (is.null(d$dnode)) rep(1, nrow(d$joint_grid)) else d$dnode
  outer_grid_weights(d, dnode = base * exp(bm$log_box_ratio))
}
.bar_place_g <- function(d)
  tulpa:::.joint_local_ccd_barycentre(d$joint_grid, d$log_marginal,
                                      d$axis_names, d$axis_tags)

test_that("a rebuild at its own coordinates returns the read the fit shipped", {
  skip_on_cran()
  # The round trip the coordinate argument has to satisfy before any candidate
  # placement measured through it means anything: own coordinates plus own
  # weights is the shipped read. Measured at 0 on every reported number.
  d <- outer_grid_dump(ogd_fixture_fit(ogd_fixture_sim(c(0.8, 0.5, 0.3)), 5L))
  rb <- outer_grid_rebuild(d, d$weights, d$joint_grid)
  expect_equal(rb$median, d$reported$median, tolerance = 1e-12)
  expect_equal(rb$ci_lo, d$reported$ci_lo, tolerance = 1e-12)
  expect_equal(rb$ci_hi, d$reported$ci_hi, tolerance = 1e-12)
  expect_equal(unlist(outer_grid_rebuild(d, NULL, d$joint_grid)),
               unlist(outer_grid_rebuild(d)), tolerance = 1e-14)
  # A coordinate matrix describing a different grid is refused: a read is
  # attributable to the placement only while cell k of the coordinates is the
  # same cell as cell k of the weights.
  expect_error(outer_grid_rebuild(d, joint_grid = d$joint_grid[-1L, ]),
               "cell")
  expect_error(outer_grid_rebuild(d, joint_grid = d$joint_grid[, -1L]),
               "axis")
})

test_that("the barycentre's read is reported against what this grid can resolve", {
  skip_on_cran()
  sim <- ogd_fixture_sim(c(0.8, 0.5, 0.3))

  # Four levels, where a cell carries whole nats of gradient and the atom moves
  # most of the way to its own edge (largest shift 0.887 of a half-cell).
  d4 <- outer_grid_dump(ogd_fixture_fit(sim, 4L))
  bc4 <- .bar_place_g(d4)
  expect_identical(sum(bc4$computed), 8L)
  expect_identical(bc4$n_axes_declined, 0L)
  expect_gt(max(bc4$bary_shift, na.rm = TRUE), 0.5)
  # Every scored cell's atom is inside its own cell, which is what lets the
  # moved grid be read by the same weighted quantile as the unmoved one.
  expect_true(all(bc4$bary_shift[bc4$computed] <= 1))

  r4 <- outer_grid_weight_report(d4, joint_grid = bc4$joint_grid)
  # Measured under the read the engine ships (see `ogd_fixture_fit()`, which
  # states it rather than inheriting it -- gcol33/tulpa#599). Endpoints 0.2679
  # against a floor of 0.1994 and widths 0.5357 against 0.3629: the placement
  # moves the interval by more than one step of coarsening moves it. The median
  # moves 0.0910 against a floor of 0.0905, the same size, so this grid does not
  # separate the two and no verdict on the location is read off it. What it does
  # show is the ordering: the placement reaches the interval and not the centre.
  expect_true(r4$above_floor[["endpoints"]])
  expect_true(r4$above_floor[["widths"]])
  expect_lt(r4$diff[["median"]], 1.2 * r4$floor[["median"]])
  expect_gt(r4$diff[["widths"]] / r4$floor[["widths"]],
            r4$diff[["median"]] / r4$floor[["median"]])

  # Five levels, where the same grid already places its atoms close enough that
  # the interval barely notices.
  d5 <- outer_grid_dump(ogd_fixture_fit(sim, 5L))
  bc5 <- .bar_place_g(d5)
  expect_identical(sum(bc5$computed), 27L)
  r5 <- outer_grid_weight_report(d5, joint_grid = bc5$joint_grid)
  # Measured: all three parts above the floor, and the margins say which one the
  # placement reaches -- widths 0.2746 against 0.0798, three times the
  # resolution, against 1.5x on the median and 1.07x on the endpoints.
  expect_true(all(r5$above_floor))
  expect_gt(r5$diff$widths, 2.5 * r5$floor$widths)
  # Which part carries the margin is a property of the within-cell read, not of
  # the placement. Under `chord` the same fit answers the other way round: the
  # median clears its floor 22x (0.0521 against 0.0024) while the endpoints and
  # widths clear theirs 2.4x. Both reads put the same mass in the same cells and
  # differ by half a cell in where inside one they place it, which on a grid
  # this coarse is the scale the median is resolved at.
})

test_that("the pair is what moves both parts of the read toward the dense answer", {
  skip_on_cran()
  # The floor says whether a difference is resolved; it does not say which
  # direction is right. The arbiter is the same model on a grid fine enough for
  # the read to be a property of the posterior -- twelve levels per axis, 1728
  # cells -- against which the shipped read and all three candidate rules are
  # scored on the coarse grid.
  sim <- ogd_fixture_sim(c(0.8, 0.5, 0.3))
  ref <- outer_grid_rebuild(outer_grid_dump(ogd_fixture_fit(sim, 12L)))
  err <- function(d, w, g) outer_grid_read_diff(ref, outer_grid_rebuild(d, w, g))

  # Four levels. Measured (endpoints / widths / median):
  #   shipped   0.4067  0.8135  0.1158
  #   mass      0.1516  0.1656  0.2028
  #   location  0.1224  0.2449  0.0689
  #   pair      0.1868  0.3736  0.0730
  # The mass rule buys the interval and loses the location, which is the split
  # the barycentre exists to close: it improves all three at once, and the pair
  # improves all three while sitting between them on each. Over five seeds of
  # the same fixture the direction is unanimous -- against the shipped read the
  # mass rule wins the endpoints and widths 5 of 5 and the median 0 of 5, while
  # the location rule and the pair win all three parts 5 of 5.
  d4 <- outer_grid_dump(ogd_fixture_fit(sim, 4L))
  w4 <- .bar_mass_w(d4); g4 <- .bar_place_g(d4)$joint_grid
  e0 <- err(d4, NULL, NULL)
  em <- err(d4, w4, NULL)
  el <- err(d4, NULL, g4)
  eb <- err(d4, w4, g4)
  for (p in names(OGD_PARTS)) {
    expect_lt(el[[p]], e0[[p]])
    expect_lt(eb[[p]], e0[[p]])
  }
  expect_gt(em$median, e0$median)
  expect_lt(el$median, em$median)
  expect_lt(eb$median, em$median)
  # And the width gain the mass rule buys is partly given back by moving the
  # cells: the pair is better than the shipped read on the widths and not as
  # good as the mass rule alone.
  expect_lt(em$widths, eb$widths)

  # Five levels, where the shipped read is already close. Measured:
  #   shipped   0.0952  0.1904  0.0469
  #   mass      0.1281  0.2561  0.0459
  #   location  0.1632  0.3264  0.0145
  #   pair      0.1215  0.2430  0.0145
  # Each rule alone costs the interval here and only the location half reaches
  # the median, which it does by a factor of three. Over five seeds the pair is
  # the only candidate whose mean error falls on all three parts at this
  # resolution (0.1217 / 0.2379 / 0.0298 against 0.1322 / 0.2644 / 0.0421).
  d5 <- outer_grid_dump(ogd_fixture_fit(sim, 5L))
  e0 <- err(d5, NULL, NULL)
  el <- err(d5, NULL, .bar_place_g(d5)$joint_grid)
  expect_lt(el$median, e0$median / 2)
  expect_gt(el$endpoints, e0$endpoints)
})
