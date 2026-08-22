# The local-quadratic box-integral mass estimator for an outer tensor cell
# (gcol33/tulpa#326).
#
# A cell's shipped mass is the midpoint atom `Delta exp(ell_c)`, which cannot see
# the log-marginal's gradient across the cell. `.joint_local_ccd_box_mass()`
# integrates the cell's own local quadratic over the cell's own box instead, off
# the same three-point stencil the design scale already differences, so it costs
# no inner solve. What is checked here is the closed form against quadrature, the
# per-axis handling of a convex axis, and what the rule does to the reported read
# when it is run through the #322 harness at zero fitting minutes.

# The axis integral by composite Simpson on the max-shifted exponent. Independent
# of `stats::integrate()`, so the estimator's numeric route is not being compared
# against itself.
.bxm_simpson <- function(g, a, h_lo, h_hi, m_pts = 20001L) {
  expo <- function(u) g * u - 0.5 * a * u^2
  cand <- c(-h_lo, h_hi)
  if (a > 0) cand <- c(cand, min(max(g / a, -h_lo), h_hi))
  mx <- max(expo(cand))
  u <- seq(-h_lo, h_hi, length.out = m_pts)
  step <- (h_hi + h_lo) / (m_pts - 1L)
  w <- c(1, rep(c(4, 2), length.out = m_pts - 2L), 1)
  mx + log(step / 3 * sum(w * exp(expo(u) - mx))) - log(h_lo + h_hi)
}

# The same integral through `stats::integrate()`, on the max shift the closed
# form's own algebra supplies.
.bxm_integrate <- function(g, a, h_lo, h_hi) {
  mu <- g / a
  mx <- 0.5 * g^2 / a - 0.5 * a * (min(max(mu, -h_lo), h_hi) - mu)^2
  v <- stats::integrate(function(u) exp(g * u - 0.5 * a * u^2 - mx),
                        -h_lo, h_hi, rel.tol = .Machine$double.eps^0.75)$value
  mx + log(v) - log(h_lo + h_hi)
}

# --------------------------------------------------------------------------- #
# The closed form                                                             #
# --------------------------------------------------------------------------- #

test_that("the concave axis factor is the box integral, to machine precision", {
  gr <- expand.grid(g = c(0, 0.25, 1, 2, 4.44, -3), a = c(0.05, 0.5, 1, 3, 20),
                    h = c(0.25, 1, 2))
  err_int <- numeric(nrow(gr))
  err_sim <- numeric(nrow(gr))
  route   <- character(nrow(gr))
  for (i in seq_len(nrow(gr))) {
    f <- tulpa:::.joint_local_ccd_axis_box(gr$g[i], gr$a[i], gr$h[i], gr$h[i])
    route[i]   <- f$route
    err_int[i] <- abs(f$log_factor - .bxm_integrate(gr$g[i], gr$a[i],
                                                    gr$h[i], gr$h[i]))
    err_sim[i] <- abs(f$log_factor - .bxm_simpson(gr$g[i], gr$a[i],
                                                  gr$h[i], gr$h[i]))
  }
  # Every one of these takes the closed form: the whole point is that a concave
  # axis needs no quadrature.
  expect_identical(unique(route), "closed")
  # Measured over the 90 combinations, on the log scale: 6.13e-14 against
  # integrate() and 6.08e-14 against Simpson, the two arbiters agreeing with
  # each other to the same figure. The tolerances are floating slack.
  expect_lt(max(err_int), 1e-12)
  expect_lt(max(err_sim), 1e-12)

  # An asymmetric box -- the cell's Voronoi half-widths need not match, and the
  # factor divides by the axis's own extent so the product stays a multiplier on
  # the cell's own volume.
  expect_equal(tulpa:::.joint_local_ccd_axis_box(1.3, 0.7, 0.4, 1.1)$log_factor,
               .bxm_integrate(1.3, 0.7, 0.4, 1.1), tolerance = 1e-12)

  # A cell the log-marginal is flat across is its own midpoint atom exactly.
  expect_equal(tulpa:::.joint_local_ccd_axis_box(0, 0, 1, 1)$log_factor, 0,
               tolerance = 1e-14)
})

test_that("the factor is the issue's pnorm form where that form can be evaluated", {
  # exp(0.5 g^2/a) sqrt(2 pi/a) (Phi(h; mu, s) - Phi(-h; mu, s)) / (2h), written
  # out directly. Its two factors diverge against each other as the local peak
  # leaves the box -- exp(0.5 g^2/a) up, the pnorm difference down at the
  # matching rate -- so the product stays O(1) while each side runs out of double
  # range on its own: at g = 10, a = 0.02 the leading term is exp(2500) = Inf
  # against a pnorm difference of 0, and the answer comes back NaN. Only the logs
  # stay in range, which is why the implementation carries them.
  direct <- function(g, a, h) {
    exp(0.5 * g^2 / a) * sqrt(2 * pi / a) *
      (stats::pnorm(h, g / a, 1 / sqrt(a)) -
         stats::pnorm(-h, g / a, 1 / sqrt(a))) / (2 * h)
  }
  gr <- expand.grid(g = c(0, 0.5, 1, 2), a = c(0.5, 1, 3, 20), h = c(0.25, 1, 2))
  rel <- numeric(nrow(gr))
  for (i in seq_len(nrow(gr))) {
    m <- exp(tulpa:::.joint_local_ccd_axis_box(gr$g[i], gr$a[i],
                                               gr$h[i], gr$h[i])$log_factor)
    rel[i] <- abs(m / direct(gr$g[i], gr$a[i], gr$h[i]) - 1)
  }
  expect_lt(max(rel), 1e-12)

  expect_true(is.nan(direct(10, 0.02, 1)))
  for (p in list(c(4.44, 0.05), c(10, 0.02))) {
    f <- tulpa:::.joint_local_ccd_axis_box(p[1L], p[2L], 1, 1)
    expect_identical(f$route, "closed")
    expect_equal(f$log_factor, .bxm_integrate(p[1L], p[2L], 1, 1),
                 tolerance = 1e-10)
  }
})

# --------------------------------------------------------------------------- #
# The convex axis                                                             #
# --------------------------------------------------------------------------- #

test_that("a convex axis is integrated rather than declined", {
  gr <- expand.grid(g = c(0, 0.5, 2, -2), a = c(0, -0.05, -0.5, -2),
                    h = c(0.5, 1, 2))
  err <- numeric(nrow(gr))
  route <- character(nrow(gr))
  for (i in seq_len(nrow(gr))) {
    f <- tulpa:::.joint_local_ccd_axis_box(gr$g[i], gr$a[i], gr$h[i], gr$h[i])
    expect_false(is.null(f$log_factor))
    expect_true(is.na(f$reason))
    route[i] <- f$route
    err[i] <- abs(f$log_factor - .bxm_simpson(gr$g[i], gr$a[i],
                                              gr$h[i], gr$h[i]))
  }
  expect_identical(unique(route), "numeric")
  # Measured: 1.3e-14 over the 48 combinations. A bounded interval and a smooth
  # integrand is where quadrature is exact, which is what buys the convex axis a
  # factor instead of a decline.
  expect_lt(max(err), 1e-12)

  # And a near-flat concave axis, where the closed form's own two pieces cancel:
  # 0.5 g^2/a diverges while the pnorm difference vanishes at the matching rate.
  # The reading is refused there and taken by quadrature, not returned wrong.
  f <- tulpa:::.joint_local_ccd_axis_box(0, 1e-14, 1, 1)
  expect_identical(f$route, "numeric")
  expect_equal(f$log_factor, .bxm_simpson(0, 1e-14, 1, 1), tolerance = 1e-12)
  expect_true(is.na(tulpa:::.joint_local_ccd_log_pnorm_diff(-5e-8, 5e-8)))
})

test_that("a cell convex on one axis keeps the exact factor on the others", {
  # The separability is the whole argument for the diagonal rule: the decline is
  # per axis, so three good axes are not thrown away for a fourth the local
  # quadratic came back convex on.
  st <- list(g = c(1.0, -0.5, 2.0, 0.3), d2 = c(-2, -1, -0.5, 0.8),
             half_lo = rep(0.5, 4L), half_hi = rep(0.5, 4L))
  bm <- tulpa:::.joint_local_ccd_cell_box_mass(st)
  expect_identical(bm$n_axes_declined, 0L)
  expect_identical(bm$n_axes_closed, 3L)
  expect_identical(bm$n_axes_numeric, 1L)
  exact <- sum(vapply(seq_len(4L), function(j)
    .bxm_simpson(st$g[j], -st$d2[j], st$half_lo[j], st$half_hi[j]),
    numeric(1)))
  expect_equal(bm$log_box_ratio, exact, tolerance = 1e-12)

  # A flat cell is its own atom on every axis, so the multiplier is exactly 1.
  flat <- tulpa:::.joint_local_ccd_cell_box_mass(
    list(g = rep(0, 3L), d2 = rep(0, 3L), half_lo = rep(1, 3L),
         half_hi = rep(1, 3L)))
  expect_equal(flat$log_box_ratio, 0, tolerance = 1e-14)
})

# --------------------------------------------------------------------------- #
# The stencil the gradient comes off                                          #
# --------------------------------------------------------------------------- #

test_that("the gradient is the same three points the curvature is", {
  # An exactly quadratic target in u, so the three-point stencil reproduces both
  # derivatives exactly and the two readings can be checked against the truth
  # rather than against each other.
  mu <- c(0.3, -0.2, 0.7); s <- c(0.8, 1.3, 0.5)
  lv <- list(a = c(-1, 0, 1.5), b = c(-1.2, 0.1, 1), c = c(-0.8, 0.2, 1.4))
  gg <- as.matrix(expand.grid(lv, KEEP.OUT.ATTRS = FALSE))
  colnames(gg) <- names(lv)
  lm <- -0.5 * rowSums(sweep(sweep(gg, 2L, mu, "-")^2, 2L, s^2, "/"))
  nb <- tulpa:::.joint_local_ccd_neighbors(gg, gg, seq_len(3L))
  ctr <- which(gg[, 1L] == 0 & gg[, 2L] == 0.1 & gg[, 3L] == 0.2)

  st <- tulpa:::.joint_local_ccd_cell_stencil(ctr, gg, lm, nb$up, nb$dn)
  expect_equal(st$g, -(gg[ctr, ] - mu) / s^2, tolerance = 1e-12,
               ignore_attr = TRUE)
  expect_equal(st$d2, -1 / s^2, tolerance = 1e-12, ignore_attr = TRUE)

  # The curvature reading is unchanged by the gradient having been extracted
  # from the same stencil, and carries it.
  cc <- tulpa:::.joint_local_ccd_cell_curv(ctr, gg, lm, nb$up, nb$dn)
  expect_equal(cc$sd, s, tolerance = 1e-12, ignore_attr = TRUE)
  expect_equal(cc$g, st$g, tolerance = 1e-14)
  expect_equal(cc$d2, st$d2, tolerance = 1e-14)

  # Unequal spacing: the divided differences are the interpolating quadratic's,
  # so an unevenly spaced stencil is exact on a quadratic too.
  q <- function(u) 1.4 + 2.1 * u - 0.65 * u^2
  fd <- tulpa:::.joint_local_ccd_diff3(-0.3, 0, 1.7, q(-0.3), q(0), q(1.7))
  expect_equal(fd$d1, 2.1, tolerance = 1e-12)
  expect_equal(fd$d2, -1.3, tolerance = 1e-12)
  expect_equal(fd$half_lo, 0.15, tolerance = 1e-14)
  expect_equal(fd$half_hi, 0.85, tolerance = 1e-14)
  expect_null(tulpa:::.joint_local_ccd_diff3(0, 0, 1, 1, 1, 1))
  expect_null(tulpa:::.joint_local_ccd_diff3(-1, 0, 1, 1, NA_real_, 1))
})

test_that("a boundary cell has no centred stencil and keeps its midpoint atom", {
  mu <- c(0.3, -0.2); s <- c(0.8, 1.3)
  lv <- list(a = c(-1, 0, 1), b = c(-1, 0, 1))
  gg <- as.matrix(expand.grid(lv, KEEP.OUT.ATTRS = FALSE))
  colnames(gg) <- names(lv)
  lm <- -0.5 * rowSums(sweep(sweep(gg, 2L, mu, "-")^2, 2L, s^2, "/"))
  tags <- c("identity", "identity")

  bm <- tulpa:::.joint_local_ccd_box_mass(gg, lm, colnames(gg), tags)
  interior <- gg[, 1L] == 0 & gg[, 2L] == 0
  expect_identical(bm$computed, unname(interior))
  # Exactly one interior cell on a 3 x 3 grid, and every other cell is the atom
  # it already was: multiplier 1, log 0.
  expect_equal(sum(bm$computed), 1L)
  expect_true(all(bm$log_box_ratio[!bm$computed] == 0))
  expect_true(bm$log_box_ratio[interior] != 0)
  expect_identical(bm$n_axes_declined, 0L)
  # No axis that reached a per-axis gate fell at one, and the eight boundary
  # cells account for their own axes: the stencil is a cell-level object, so a
  # cell missing a neighbour on one axis has no local quadratic on either. The
  # count and the tally are read together rather than one summing to the other
  # (gcol33/tulpa#334).
  expect_identical(bm$declined_reasons,
                   c(boundary = 16L, no_factor = 0L, cancellation = 0L))
  expect_identical(bm$n_axes_declined,
                   sum(bm$declined_reasons[c("no_factor", "cancellation")]))

  # The interior cell's multiplier is the product of its own axis factors.
  nb <- tulpa:::.joint_local_ccd_neighbors(gg, gg, seq_len(2L))
  st <- tulpa:::.joint_local_ccd_cell_stencil(which(interior), gg, lm,
                                              nb$up, nb$dn)
  expect_equal(bm$log_box_ratio[interior],
               sum(vapply(1:2, function(j)
                 .bxm_simpson(st$g[j], -st$d2[j], st$half_lo[j], st$half_hi[j]),
                 numeric(1))), tolerance = 1e-10, ignore_attr = TRUE)

  # An unguessable-support grid declines whole rather than guessing a transform.
  # No cell was scored at all there, so no axis declined at a gate and the tally
  # is empty: the whole-grid decline is what `computed` reports, not a per-axis
  # refusal.
  none <- tulpa:::.joint_local_ccd_box_mass(gg, lm, colnames(gg), NULL)
  expect_false(any(none$computed))
  expect_true(all(none$log_box_ratio == 0))
  expect_true(all(none$declined_reasons == 0L))
  expect_identical(names(none$declined_reasons), tulpa:::.LCCD_BOX_REASONS)
})

# --------------------------------------------------------------------------- #
# What a declined axis declined at (gcol33/tulpa#334)                         #
# --------------------------------------------------------------------------- #

test_that("a declined box axis names the gate it fell at", {
  # `no_factor`: the quadratic never formed. A box of zero extent has no
  # integral to divide by its own width, and a non-finite coefficient has no
  # integrand at all.
  for (p in list(c(1, 1, 0, 0), c(NA_real_, 1, 1, 1), c(1, NA_real_, 1, 1))) {
    f <- tulpa:::.joint_local_ccd_axis_box(p[1L], p[2L], p[3L], p[4L])
    expect_null(f$log_factor)
    expect_identical(f$reason, "no_factor")
  }

  # `cancellation`: the local peak sits 5e11 nats above the box, past
  # `.LCCD_BOX_CANCEL`, so the closed form is refused before it is formed and
  # the quadrature that takes over is handed a needle 1e-6 wide in a box of
  # width 2 and returns nothing either. The quadratic here is perfectly good and
  # only its arithmetic is out of reach, which is the distinction the reason
  # carries and the count cannot.
  expect_gt(0.5 * 1e6^2 / 1, tulpa:::.LCCD_BOX_CANCEL)
  f <- tulpa:::.joint_local_ccd_axis_box(1e6, 1, 1, 1)
  expect_null(f$log_factor)
  expect_identical(f$reason, "cancellation")
  # One decade of gradient lower is inside the budget and is answered.
  ok <- tulpa:::.joint_local_ccd_axis_box(1e5, 1, 1, 1)
  expect_identical(ok$route, "numeric")
  expect_true(is.na(ok$reason))
  expect_true(is.finite(ok$log_factor))
})

test_that("a cell's decline tally is per axis, alongside the count", {
  # Four axes: one concave and answered in closed form, one whose closed form is
  # refused for conditioning and whose quadrature then returns nothing, one with
  # no extent at all, and one convex axis that is integrated rather than
  # declined. The count says two axes went unread; the tally says they went
  # unread for structurally different reasons, and only the second of the two
  # would be fixed by a finer grid.
  st <- list(g = c(1.0, 1e6, 0.3, 2.0), d2 = c(-2, -1, 0, 0.8),
             half_lo = c(0.5, 1, 0, 0.5), half_hi = c(0.5, 1, 0, 0.5))
  bm <- tulpa:::.joint_local_ccd_cell_box_mass(st)
  expect_identical(bm$n_axes_declined, 2L)
  expect_identical(bm$declined_reasons,
                   c(boundary = 0L, no_factor = 1L, cancellation = 1L))
  # `boundary` cannot fire on a cell that already holds a stencil, and the slot
  # is carried anyway so the tally is the same object the grid-wide entry point
  # returns.
  expect_identical(names(bm$declined_reasons), tulpa:::.LCCD_BOX_REASONS)

  # The two axes that were read still contribute their exact factors, which is
  # the separability the diagonal form is chosen for.
  expect_identical(bm$n_axes_closed, 1L)
  expect_identical(bm$n_axes_numeric, 1L)
  expect_equal(bm$log_box_ratio,
               .bxm_simpson(1.0, 2, 0.5, 0.5) +
                 .bxm_simpson(2.0, -0.8, 0.5, 0.5),
               tolerance = 1e-12)
})

test_that("a refined cell records the box mass beside the cloud's own", {
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
  expect_length(out$info$log_box_ratio, nc)
  expect_true(all(is.finite(out$info$log_box_ratio)))

  # It is the grid-wide rule's reading of the same cells, so the two entry
  # points cannot drift apart.
  bm <- tulpa:::.joint_local_ccd_box_mass(gg, lmf(gg), colnames(gg),
                                          c("identity", "identity"))
  expect_equal(out$info$log_box_ratio, bm$log_box_ratio[out$info$cells],
               tolerance = 1e-12)
})

# --------------------------------------------------------------------------- #
# The measurement (gcol33/tulpa#322's harness, no fitting minutes)            #
# --------------------------------------------------------------------------- #

# The candidate weight vector: the box multiplier entering as a per-cell design
# weight, exactly the way the harness takes one.
.bxm_weights <- function(d) {
  bm <- tulpa:::.joint_local_ccd_box_mass(d$joint_grid, d$log_marginal,
                                          d$axis_names, d$axis_tags)
  base <- if (is.null(d$dnode)) rep(1, nrow(d$joint_grid)) else d$dnode
  list(bm = bm, w = outer_grid_weights(d, dnode = base * exp(bm$log_box_ratio)))
}

# Every axis's own u-space level spacing. The multiplier form is only a rule at
# all because the common cell volume `Delta` cancels in the softmax, and that is
# a property of the grid rather than of the rule.
.bxm_spacings <- function(d) {
  lapply(seq_len(ncol(d$joint_grid)), function(j)
    diff(sort(unique(tulpa:::.joint_pareto_fwd(d$axis_tags[j],
                                               d$joint_grid[, j])))))
}

test_that("the base grid's cells have equal volume, so Delta cancels", {
  skip_on_cran()
  d <- outer_grid_dump(ogd_fixture_fit(ogd_fixture_sim(c(0.8, 0.5, 0.3)), 5L))
  # A geometric sigma grid is uniform in u = log sigma, so every interior cell
  # is congruent and the multiplier can enter without carrying Delta_c. Asserted
  # rather than assumed: on a grid whose cells differ in volume the same
  # multiplier would silently drop a per-cell factor.
  for (sp in .bxm_spacings(d)) {
    expect_gt(length(sp), 1L)
    expect_lt(stats::sd(sp) / mean(sp), 1e-12)
  }
  # And the cell volume is the same on every axis-interior cell, which is the
  # statement the softmax needs.
  vol <- prod(vapply(.bxm_spacings(d), function(sp) sp[1L], numeric(1)))
  expect_true(is.finite(vol) && vol > 0)
})

test_that("the box rule's read is reported against what this grid can resolve", {
  skip_on_cran()
  sim <- ogd_fixture_sim(c(0.8, 0.5, 0.3))

  # A five-level tensor base: 125 cells, 27 of them interior on all three axes,
  # and those 27 carry 0.99999 of the integration weight, so the rule is scored
  # where the posterior actually is.
  d5 <- outer_grid_dump(ogd_fixture_fit(sim, 5L))
  expect_null(d5$dnode)
  expect_identical(d5$support, "density")
  b5 <- .bxm_weights(d5)
  expect_identical(sum(b5$bm$computed), 27L)
  expect_identical(b5$bm$n_axes_declined, 0L)
  expect_gt(sum(d5$weights[b5$bm$computed]), 0.999)

  r5 <- outer_grid_weight_report(d5, b5$w)
  # Measured under the read the engine ships. endpoints 0.0745 against a floor
  # of 0.1287: the endpoints do not move by as much as one step of coarsening
  # moves them. The WIDTHS do -- 0.1491 against a floor of 0.0882 -- and the
  # median does not, 0.0124 against 0.0184, so on this grid the correction is
  # real and it is the spread, not the location, that it reaches.
  #
  # Which part it reaches is a property of the within-cell read, not of the rule
  # (gcol33/tulpa#599). Under `chord` the same fit answers the other way round:
  # widths 0.0657 against a floor of 0.2158 and median 0.0175 against 0.0024, so
  # the location moves and the spread does not. Both reads place the same mass
  # in the same cells and differ in where inside a cell they place it, and on a
  # five-level grid that is a half-cell -- which is the resolution these two
  # parts are being read at. The floor itself moves with the read for the same
  # reason (widths 0.2158 -> 0.0882, median 0.0024 -> 0.0184), so this is the
  # resolution of the reported read changing rather than the rule's effect
  # growing against a fixed one.
  expect_gt(r5$diff$widths, r5$floor$widths)
  expect_lt(r5$diff$median, r5$floor$median)
  expect_lt(r5$diff$endpoints, r5$floor$endpoints)
  expect_false(r5$above_floor[["endpoints"]])
  expect_true(r5$above_floor[["widths"]])
  expect_false(r5$above_floor[["median"]])

  # Coarser, which is where a cell carries a real gradient: at four levels the
  # same rule moves the endpoints 0.3423 against a floor of 0.1712, above what
  # the grid resolves. The multiplier is not a small correction there -- the
  # steepest cell's log multiplier is 25.3 nats -- because a four-level grid puts
  # whole nats of log-marginal across a single cell. This arm reads the same
  # under either within-cell construction; it is the five-level arm above, where
  # the effect sits at the grid's own resolution, that the read decides.
  d4 <- outer_grid_dump(ogd_fixture_fit(sim, 4L))
  b4 <- .bxm_weights(d4)
  expect_identical(sum(b4$bm$computed), 8L)
  r4 <- outer_grid_weight_report(d4, b4$w)
  expect_gt(r4$diff$endpoints, r4$floor$endpoints)
  expect_true(r4$above_floor[["endpoints"]])
  expect_gt(max(b4$bm$log_box_ratio), 10)
})

test_that("where the read moves above the floor it moves toward the dense answer", {
  skip_on_cran()
  # The floor says whether a difference is resolved; it does not say which
  # direction is right. The arbiter for that is the same model on a grid fine
  # enough for the read to be a property of the posterior -- twelve levels per
  # axis, 1728 cells -- against which both the midpoint atom and the box rule
  # are scored on the coarse grid.
  sim <- ogd_fixture_sim(c(0.8, 0.5, 0.3))
  ref <- outer_grid_rebuild(outer_grid_dump(ogd_fixture_fit(sim, 12L)))

  d4 <- outer_grid_dump(ogd_fixture_fit(sim, 4L))
  e0 <- outer_grid_read_diff(ref, outer_grid_rebuild(d4))
  e1 <- outer_grid_read_diff(ref, outer_grid_rebuild(d4, .bxm_weights(d4)$w))
  # Measured on the four-level grid: endpoint error 0.2014 -> 0.1838 and width
  # error 0.3153 -> 0.0955, so the interval the coarse grid reports moves toward
  # the converged one, by a third on the endpoints and threefold on the widths.
  # The median goes the other way, 0.1540 -> 0.1761: the rule redistributes mass
  # toward the cell edges nearest the peak, which sharpens the spread and
  # overshoots the location.
  #
  # The box-uniform read already recovers part of what the rule recovers -- the
  # same three errors read under `chord` start at 0.4067 / 0.8135 / 0.1158 -- so
  # the rule has a smaller remaining gain here than it does against that read,
  # and the endpoint margin is the thin one of the three.
  expect_lt(e1$endpoints, e0$endpoints)
  expect_lt(e1$widths, e0$widths)
  expect_gt(e1$median, e0$median)
})
