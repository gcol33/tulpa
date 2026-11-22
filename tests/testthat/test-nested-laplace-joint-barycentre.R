# The local-quadratic mass barycentre of an outer tensor cell
# (gcol33/tulpa#327).
#
# The per-axis read places a cell's whole mass at the cell's own coordinate, so a
# cell whose log-marginal carries a gradient is read at a point its mass is not
# centred on. `.joint_local_ccd_cell_bary()` reads the first moment of the
# cell's own local quadratic over its own box, off the same three-point stencil
# `.joint_local_ccd_cell_box_mass()` (gcol33/tulpa#326) reads the mass from, so
# it costs no inner solve. What is checked here is the closed form against
# quadrature, the per-axis handling of a convex axis, that a barycentre stays
# inside its own cell, and the reading the local-CCD refinement records for the
# cells it refines.

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

# The reading a grid of (gradient, curvature, box) combinations is held to. The
# CRAN-visible case and the full sweep below both come through here, so the two
# are one set of assertions over two grids rather than two copies.
.bary_check_grid <- function(gr) {
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
  # arbiters, which agree with each other to 1.55e-15. The worst case is the
  # far-tail closed form (g = -8, a = 0.05 on a box of width 0.5, so a local
  # peak 640 nats up and 320 cell widths away), where the error is 4.7e-11 of
  # the cell width against `eps` times the cancellation product of 4.5e-11 --
  # the budget `.LCCD_BAR_CANCEL` is set to. The numeric route measures 1.55e-15.
  # Measured on x86_64-w64-mingw32 by `dev_notes/issue327/bary327.R`. The
  # tolerances are floating slack over those.
  expect_lt(max(e_int), 1e-9)
  expect_lt(max(e_sim), 1e-9)
  # The numeric route asks stats::integrate() for rel.tol = eps^0.75 on each of
  # the two integrals it forms the ratio from, so what it guarantees is about
  # twice that relative, and the box half-widths here are O(1). The bound is
  # that contract with slack. What any one platform measures inside it is the
  # arithmetic it happened to do: where the adaptive subdivision lands moves
  # with the libm, and the same grid measures 1.55e-15 on x86_64-w64-mingw32
  # and 2e-12 on aarch64-apple-darwin23 (GitHub Actions run 32475678231, image
  # macos-26-arm64, R 4.6.1).
  expect_lt(max(e_sim[route == "numeric"]), 8 * .Machine$double.eps^0.75)

  # A barycentre is the first moment of a positive density over the box, so it
  # is a convex combination of points in the box and cannot leave it. The
  # assertion is on the arithmetic, not on the integral.
  expect_true(all(inbox))
  invisible(gr)
}

skip_on_cran_budget()

# --------------------------------------------------------------------------- #
# What a declined axis declined at (gcol33/tulpa#334)                         #
# --------------------------------------------------------------------------- #

test_that("a declined barycentre axis names the gate it fell at", {
  skip_on_cran_budget()
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
  skip_on_cran_budget()
  # Four axes: one concave and answered in closed form, one whose closed form is
  # refused for conditioning and whose quadrature returns nothing, one with no
  # extent at all, and one whose quadrature puts the atom outside its own box.
  # Three declines, three different remedies, one count.
  st <- list(g = c(1.0, 1e6, 0.3, -7e8), d2 = c(-2, -1, 0, 0),
             half_lo = c(0.5, 1, 0, 1e-6), half_hi = c(0.5, 1, 0, 1e-8))
  bc <- tulpa:::.joint_local_ccd_cell_bary(st)
  expect_identical(bc$n_axes_declined, 3L)
  expect_identical(bc$declined_reasons,
                   c(no_factor = 1L, cancellation = 1L, out_of_box = 1L))
  expect_identical(names(bc$declined_reasons), tulpa:::.LCCD_BARY_REASONS)

  # A declined axis keeps the cell's own coordinate, and the one axis that was
  # read still moves: the decline is per axis, so three unreadable axes do not
  # cost the fourth its barycentre.
  expect_identical(bc$n_axes_closed, 1L)
  expect_identical(bc$n_axes_numeric, 0L)
  expect_equal(bc$u_bar,
               c(.bar_simpson(1.0, 2, 0.5, 0.5), 0, 0, 0), tolerance = 1e-11)
})

test_that("a refined cell records its barycentre shift beside its box mass", {
  skip_on_cran_budget()
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

  # It is the cell rule's reading of each refined cell's own stencil.
  geo <- tulpa:::.joint_local_ccd_geometry(gg, colnames(gg),
                                           c("identity", "identity"))
  shift <- vapply(out$info$cells, function(c)
    tulpa:::.joint_local_ccd_cell_bary(tulpa:::.joint_local_ccd_cell_stencil(
      c, geo$U, lmf(gg), geo$up, geo$dn))$bary_shift, numeric(1))
  expect_equal(out$info$bary_shift, shift, tolerance = 1e-12)
})

skip_on_cran()

test_that("the barycentre is the box's own first moment, on either sign of a", {
  skip_on_cran()
  # Both routes, both signs of the curvature, and half-widths unequal on the two
  # sides, since a cell's Voronoi box need not be symmetric. `g = -8, a = 0.05`
  # on the narrow box is the far-tail case the closed form hands to quadrature,
  # so the route split is exercised here and not only in the sweep below.
  .bary_check_grid(expand.grid(g = c(0, 4.44, -8), a = c(-2, 0, 0.05, 3),
                               h_lo = c(0.25, 2), h_hi = c(0.25, 2)))
})

test_that("the barycentre holds over the full curvature and box grid", {
  skip_on_cran()
  # The same assertions over 648 combinations: curvatures from concave through
  # flat to strongly convex, gradients out to the far tail, and every pairing of
  # the three half-widths.
  .bary_check_grid(expand.grid(g = c(0, 0.25, 1, 2, 4.44, -3, 10, -8),
                               a = c(-2, -0.5, -0.05, 0, 0.05, 0.5, 1, 3, 20),
                               h_lo = c(0.25, 1, 2), h_hi = c(0.25, 1, 2)))
})

test_that("a symmetric cell with no gradient keeps its own coordinate", {
  skip_on_cran()
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
  skip_on_cran()
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
  skip_on_cran()
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

test_that("an interior cell's barycentre moves toward the peak within its own box", {
  skip_on_cran()
  mu <- c(0.3, -0.2); s <- c(0.8, 1.3)
  lv <- list(a = c(-1, 0, 1), b = c(-1, 0, 1))
  gg <- as.matrix(expand.grid(lv, KEEP.OUT.ATTRS = FALSE))
  colnames(gg) <- names(lv)
  lm <- -0.5 * rowSums(sweep(sweep(gg, 2L, mu, "-")^2, 2L, s^2, "/"))
  geo <- tulpa:::.joint_local_ccd_geometry(gg, colnames(gg),
                                           c("identity", "identity"))
  interior <- which(gg[, 1L] == 0 & gg[, 2L] == 0)
  st <- tulpa:::.joint_local_ccd_cell_stencil(interior, geo$U, lm, geo$up, geo$dn)
  bc <- tulpa:::.joint_local_ccd_cell_bary(st)
  expect_identical(bc$n_axes_declined, 0L)

  # The interior cell moves toward the peak on both axes and stays inside its
  # own Voronoi half-box, which on this grid is half a level in each direction.
  expect_equal(bc$u_bar,
               vapply(1:2, function(j)
                 .bar_simpson(st$g[j], -st$d2[j], st$half_lo[j], st$half_hi[j]),
                 numeric(1)), tolerance = 1e-11, ignore_attr = TRUE)
  expect_true(all(abs(bc$u_bar) < 0.5))
  expect_equal(sign(bc$u_bar), sign(mu), ignore_attr = TRUE)
})

test_that("a rebuild at its own coordinates returns the read the fit shipped", {
  skip_on_cran()
  # The round trip the coordinate argument has to satisfy before any candidate
  # placement measured through it means anything: own coordinates plus own
  # weights is the shipped read. Measured at 0 on every reported number.
  d <- outer_grid_dump(ogd_fixture_fit(ogd_fixture_sim(c(0.8, 0.5, 0.3)), 5L,
                                       hyperprior = "flat"))
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
