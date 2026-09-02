# Boundary MASS on an outer axis (gcol33/tulpa#622).
#
# The rail label asks whether the span contains the axis's own MODE. An axis can
# hold a third of its marginal on a boundary node with the mode one node in, and
# that is the same truncation -- what lies past the outer node is unrepresented
# either way. These tests pin the second, weaker label that names it, that it is
# read in every regime, and that the rail keeps its own stronger statement.
#
# The weights below are gcol33/tulpa#622's own cases, on the 9-node log-spaced
# field-SD axis of the fit it was found on.

NODES <- exp(seq(log(0.15), log(2.0), length.out = 9))

fake_fit <- function(w, nodes = NODES) {
  w <- w / sum(w)
  list(theta_grid = matrix(nodes, ncol = 1L, dimnames = list(NULL, "sigma")),
       log_marginal = log(w), weights = w)
}

W_OBSERVED <- c(0.0181, 0.0444, 0.1534, 0.0655, 0.1242, 0.1186, 0.1160,
                0.2907, 0.0691)
W_20       <- c(0.010, 0.020, 0.050, 0.060, 0.080, 0.100, 0.120, 0.360, 0.200)
W_34       <- c(0.005, 0.010, 0.020, 0.030, 0.050, 0.070, 0.130, 0.345, 0.340)
W_MODE     <- c(0.005, 0.010, 0.020, 0.030, 0.050, 0.070, 0.130, 0.245, 0.440)
W_COLLAPSE <- c(0.001, 0.002, 0.004, 0.008, 0.015, 0.020, 0.030, 0.720, 0.200)

test_that("a boundary node carrying mass is named without being the mode", {
  # A third of the marginal on the last node, mode one node in: the rail is
  # silent by construction (it reads the argmax) and the mass label fires.
  f <- fake_fit(W_34)
  expect_null(.nl_axis_rail(f, "sigma"))
  expect_identical(.nl_railed_axes(f), character(0))
  expect_identical(.nl_edge_mass_axes(f), "sigma:upper")

  em <- .nl_axis_edge_mass(f, "sigma")
  expect_length(em, 1L)
  expect_identical(em[[1L]]$side, "upper")
  expect_equal(em[[1L]]$mass, 0.34 / sum(W_34), tolerance = 1e-12)
  # The currency is the rail's: the boundary weight against what a flat
  # marginal puts there, so the threshold means the same at any node count.
  expect_equal(em[[1L]]$lift, 9 * 0.34 / sum(W_34), tolerance = 1e-12)
  expect_identical(em[[1L]]$node, NODES[9L])
})

test_that("a rail is also boundary mass, and keeps its own stronger label", {
  f <- fake_fit(W_MODE)
  expect_identical(.nl_axis_rail(f, "sigma")$side, "upper")
  expect_identical(.nl_railed_axes(f), "sigma:upper")
  expect_identical(.nl_edge_mass_axes(f), "sigma:upper")
})

test_that("a marginal that decayed by its boundary is not named", {
  # Resolved and centred: the outer nodes carry a thousandth of the marginal.
  f <- fake_fit(exp(-0.5 * ((seq_len(9) - 5) / 1.2)^2))
  expect_identical(.nl_edge_mass_axes(f), character(0))

  # And the fit #622 was found on: 6.9 % on the ceiling is BELOW what a flat
  # marginal puts on one node of nine, so it is not named either. What that
  # grid fails is resolution, which the axis-resolution read reports.
  expect_identical(.nl_edge_mass_axes(fake_fit(W_OBSERVED)), character(0))
})

test_that("both ends are tested", {
  w <- rev(W_34)
  expect_identical(.nl_edge_mass_axes(fake_fit(w)), "sigma:lower")
  # A marginal pressing on both ends names both.
  u <- c(0.30, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.30)
  expect_identical(.nl_edge_mass_axes(fake_fit(u)),
                   c("sigma:lower", "sigma:upper"))
})

test_that("the threshold is a lift, so it means the same at any node count", {
  # One marginal read at 5 and at 15 nodes, the same shape either way: a fixed
  # SHARE would make the longer axis the weaker detector, the lift does not.
  shape <- function(m) exp(seq(-3, 0, length.out = m))
  for (m in c(5L, 9L, 15L)) {
    f <- fake_fit(shape(m), nodes = exp(seq(log(0.15), log(2), length.out = m)))
    expect_identical(.nl_edge_mass_axes(f), "sigma:upper",
                     info = sprintf("m = %d", m))
  }
})

test_that("the grid regime reads boundary mass in every regime", {
  # Spread: `ess_grid >= 2` used to return before any axis was inspected.
  rg <- .joint_pareto_grid_regime(fake_fit(W_34))
  expect_identical(rg$regime, "spread")
  expect_gte(rg$ess_grid, 2)
  expect_identical(rg$edge_mass_axes, "sigma")
  expect_identical(rg$edge_mass_sides, "upper")
  # The collapse label keeps its own meaning: the dominant CELL is interior.
  rg2 <- .joint_pareto_grid_regime(fake_fit(W_COLLAPSE))
  expect_identical(rg2$regime, "collapsed_interior")
  expect_identical(rg2$edge_axes, character(0))
  expect_identical(rg2$edge_mass_axes, "sigma")

  # A clean grid names nothing, so the label is not vacuous.
  rg3 <- .joint_pareto_grid_regime(fake_fit(exp(-0.5 * ((seq_len(9) - 5) / 1.2)^2)))
  expect_identical(rg3$edge_mass_axes, character(0))
  expect_identical(rg3$edge_mass_sides, character(0))
})

test_that("a fit reports its boundary mass beside its railed axes", {
  f <- .nl_attach_interval_provenance(
    c(fake_fit(W_34), list(integration = "grid")),
    list(within = NA_character_, within_declined = NA_character_,
         outside_nodes = NA_character_, edge_coord = NA_character_,
         edge_declined = NA_character_),
    matrix(NODES, ncol = 1L, dimnames = list(NULL, "sigma")))
  expect_identical(f$outer_grid_railed_axes, character(0))
  expect_identical(f$outer_grid_edge_mass_axes, "sigma:upper")
})

test_that("the regime note says what a boundary-mass axis means", {
  note <- .tulpa_outer_regime_note(
    list(regime = "spread", edge_axes = character(0), edge_sides = character(0),
         edge_mass_axes = "sigma", edge_mass_sides = "upper",
         outer_skew_max = NA_real_))
  expect_true(grepl("boundary node", note))
  expect_true(grepl("sigma \\(upper\\)", note))
  # A spread grid with nothing on its boundaries still needs no sentence.
  expect_null(.tulpa_outer_regime_note(
    list(regime = "spread", edge_axes = character(0), edge_sides = character(0),
         edge_mass_axes = character(0), edge_mass_sides = character(0),
         outer_skew_max = NA_real_)))
})

test_that("a flat marginal is boundary mass on both ends", {
  # Every node carries exactly what a flat marginal puts there, so the lift is 1
  # on each end: the data say nothing about the axis and the span truncates a
  # density that has not decayed anywhere. Named, not exempt.
  f <- fake_fit(rep(1, 9))
  expect_identical(.nl_edge_mass_axes(f), c("sigma:lower", "sigma:upper"))
})

test_that("the axis marginal the labels read carries the grid's own measure", {
  # Four tight nodes and one far one: the outer node owns a log cell an order
  # of magnitude wider, so it holds prior mass a node count does not see. The
  # label reads the measure the fit integrates with.
  nodes <- c(0.15, 0.2, 0.25, 0.3, 2.0)
  f <- list(theta_grid = matrix(nodes, ncol = 1L,
                                dimnames = list(NULL, "sigma")),
            log_marginal = c(-2, -1, 0, -1, -1.6))
  # Read off the node counts alone, no end carries what a flat marginal would.
  expect_identical(.nl_edge_mass_axes(f), character(0))

  f$log_quad <- .nl_grid_log_quad(f$theta_grid)
  expect_identical(.nl_edge_mass_axes(f), "sigma:upper")
})

test_that("a fit whose axes all contain their mode still reports boundary mass", {
  # The placement read, which is the one a single-block nested fit takes: no
  # axis is railed, so the note used to be silent about a span that truncates.
  pl <- list(placement = NA_character_, railed = character(0),
             edge_mass = "sigma:upper", moved = character(0),
             clamped = character(0), declined = NA_character_)
  note <- .tulpa_grid_placement_note(pl)
  expect_true(grepl("boundary node", note))
  expect_true(grepl("sigma:upper", note))
  pl$edge_mass <- character(0)
  expect_null(.tulpa_grid_placement_note(pl))
})

# --- which measure the label reads (gcol33/tulpa#660) -------------------------

# A bounded axis whose naive half-step mirror reaches past its own support. The
# outermost cell is the one `.hyper_domain_clamp()` shortens, and it is the one
# the label is asking about.
.em_rho_fit <- function(lm, rho = seq(0.2, 0.95, length.out = 4L)) {
  tg <- matrix(rho, ncol = 1L, dimnames = list(NULL, "rho"))
  list(theta_grid = tg, log_marginal = lm,
       log_quad = .nl_grid_log_quad(tg))
}

test_that("the boundary label does not weaken where the axis's support closes", {
  # Rising toward the ceiling, and enough of the marginal on the last node to
  # clear the lift threshold on the axis's own spacing.
  f <- .em_rho_fit(log(c(0.15, 0.20, 0.32, 0.33)))

  span <- .nl_axis_marginal_w(f, "rho", measure = "span")
  post <- .nl_axis_marginal_w(f, "rho", measure = "posterior")
  m <- length(span$w)
  lift_span <- m * span$w[m]
  lift_post <- m * post$w[m]

  # The clamp shortens the top cell, so the same marginal reads lower against
  # the measure the fit integrates -- and it reads lower across the threshold.
  expect_gt(lift_span, lift_post)
  expect_gte(lift_span, .nl_diag("edge_mass_lift"))
  expect_lt(lift_post, .nl_diag("edge_mass_lift"))

  # The label is the span read, so the axis is named.
  expect_identical(.nl_edge_mass_axes(f), "rho:upper")
})

test_that("the span measure drops the closure and nothing else", {
  f <- .em_rho_fit(log(c(0.15, 0.20, 0.32, 0.33)))
  expect_equal(.nl_axis_measure_quad(f, "span"),
               .nl_grid_log_quad(f$theta_grid, close_domain = FALSE))
  # Exactly one cell's WIDTH differs. The level weights are normalized, so
  # every other cell moves by the one shared renormalization constant -- which
  # a softmax over the axis cancels -- and the clamped end moves by more.
  d <- .nl_axis_measure_quad(f, "span") - .nl_axis_measure_quad(f, "posterior")
  m <- length(d)
  expect_true(all(abs(d[-m] - d[1L]) < 1e-12))
  expect_gt(d[m], d[1L])

  # An axis whose mirror stays inside its own domain has nothing to drop, so
  # the two measures are identical to the bit -- which is what keeps the
  # wide-outer-cell case above reading the spacing it is about.
  g <- list(theta_grid = matrix(c(0.15, 0.2, 0.25, 0.3, 2.0), ncol = 1L,
                                dimnames = list(NULL, "sigma")),
            log_marginal = c(-2, -1, 0, -1, -1.6))
  g$log_quad <- .nl_grid_log_quad(g$theta_grid)
  expect_identical(.nl_axis_measure_quad(g, "span"),
                   .nl_axis_measure_quad(g, "posterior"))
})

test_that("the three measures are the three questions", {
  f <- .em_rho_fit(log(c(0.15, 0.20, 0.32, 0.33)))
  # A fit carrying no measure reads the same under all three.
  bare <- f; bare$log_quad <- NULL
  for (msr in .NL_AXIS_MEASURE) {
    expect_identical(.nl_axis_marginal_w(bare, "rho", measure = msr)$w,
                     .nl_axis_marginal_w(bare, "rho", measure = "inner")$w)
  }
  # And the rail reads the density, so no measure enters it at all.
  expect_identical(.nl_axis_rail(f, "rho"), .nl_axis_rail(bare, "rho"))
})
