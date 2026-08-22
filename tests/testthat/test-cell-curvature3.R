# The cell third-derivative tensor contraction (gcol33/tulpa#301), at cell
# granularity, away from any solve.
#
# `cpp_cell_coupling_curvature3()` returns sum_{a,b,c} T^{abc} u^a u^b u^c for one
# cell of a registered CellCouplingSpec, which by definition is the third
# derivative of that cell's log-density along the direction u. So the arbiter is
# a direct numerical third derivative of the spec's own `value`: a five-point
# central stencil on t -> log p_cell(eta + t u), computed from
# `cpp_cell_coupling_evaluate()` and touching none of the contraction's own
# machinery (no Hessian, no cross blocks, no step policy).
#
# The two-arm occupancy mixture is the fixture because its cells come in both
# shapes: a cell with a detection FACTORISES (every cross derivative is zero, so
# the tensor must reduce to the separable per-eta sum), while an all-undetected
# cell does not (the occupancy state sits inside the same logarithm as every
# visit).

.c3_register <- function() {
  cpp_register_test_occupancy_mixture_coupling()
  testthat::skip_if_not(
    cpp_cell_coupling_registry_has("test_occupancy_mixture"),
    "test_occupancy_mixture coupling spec not registered")
}

.c3_value <- function(a, v, y) {
  cpp_cell_coupling_evaluate("test_occupancy_mixture",
    eta = list(a, v), y = list(0, y),
    family = c("binomial", "binomial"), phi = c(1, 1))$value
}

# d^3/dt^3 log p_cell(eta + t u) at t = 0, five-point central stencil.
.c3_brute <- function(a, v, u_a, u_v, y, h = 1e-3) {
  f <- function(t) .c3_value(a + t * u_a, v + t * u_v, y)
  (f(2 * h) - 2 * f(h) + 2 * f(-h) - f(-2 * h)) / (2 * h^3)
}

.c3_tensor <- function(a, v, u_a, u_v, y, per_arm_step = TRUE) {
  cpp_cell_coupling_curvature3("test_occupancy_mixture",
    eta = list(a, v), u = list(u_a, u_v), y = list(0, y),
    family = c("binomial", "binomial"), phi = c(1, 1),
    per_arm_step = per_arm_step)
}

test_that("the contraction reproduces a direct third derivative of the cell log-density", {
  skip_on_cran()
  .c3_register()
  a   <- 0.35
  v   <- c(-0.6, 0.2, -1.1, 0.4)
  u_a <- 0.17
  u_v <- c(0.05, -0.09, 0.13, 0.02)

  # The coupled branch: nothing factorises, so every cross-arm and cross-visit
  # entry of T is nonzero and the contraction has something to get wrong.
  y_dark <- rep(0, 4)
  b <- .c3_brute(a, v, u_a, u_v, y_dark)
  expect_gt(abs(b), 1e-6)                       # the arbiter is not itself zero
  expect_equal(.c3_tensor(a, v, u_a, u_v, y_dark), b, tolerance = 1e-2)

  # The factorising branch: log psi + a per-visit Bernoulli sum.
  y_seen <- c(1, 0, 0, 0)
  b2 <- .c3_brute(a, v, u_a, u_v, y_seen)
  expect_gt(abs(b2), 1e-8)
  expect_equal(.c3_tensor(a, v, u_a, u_v, y_seen), b2, tolerance = 1e-2)
})

test_that("on a factorising cell the tensor collapses to the separable per-eta sum", {
  skip_on_cran()
  .c3_register()
  # A cell with a detection is log psi(eta_a) + sum_v Bern(y_v | p_v), so
  # T^{abc} is diagonal and the contraction must equal sum_j l_j'''(eta_j) u_j^3
  # -- the K = 1 formula the separable path uses. For a logit-Bernoulli,
  # l'''(eta) = -p q (q - p) and it does not depend on y.
  a   <- -0.4
  v   <- c(0.8, -0.25, 1.3, 0.05)
  u_a <- 0.11
  u_v <- c(-0.07, 0.16, 0.03, -0.12)
  l3  <- function(e) { p <- stats::plogis(e); q <- 1 - p; -p * q * (q - p) }
  separable <- l3(a) * u_a^3 + sum(l3(v) * u_v^3)
  expect_equal(.c3_tensor(a, v, u_a, u_v, c(1, 0, 0, 0)), separable,
               tolerance = 1e-3)
})

test_that("a direction that does not reach the cell contributes exactly zero", {
  skip_on_cran()
  .c3_register()
  # Not NaN (the cell is perfectly scorable) and not a small number: the cubic
  # form of a zero direction is identically zero.
  expect_identical(
    .c3_tensor(0.2, c(-0.5, 0.3, 0.1, -0.9), 0, rep(0, 4), rep(0, 4)), 0)
})

test_that("the per-arm step is at least as accurate as one global step", {
  skip_on_cran()
  .c3_register()
  # With the arms on the same eta scale the two policies pick the same step and
  # agree to the last bits; the per-arm policy exists for the case they do not.
  a <- 0.35; v <- c(-0.6, 0.2, -1.1, 0.4)
  u_a <- 0.17; u_v <- c(0.05, -0.09, 0.13, 0.02)
  y <- rep(0, 4)
  expect_equal(.c3_tensor(a, v, u_a, u_v, y, per_arm_step = TRUE),
               .c3_tensor(a, v, u_a, u_v, y, per_arm_step = FALSE),
               tolerance = 1e-8)

  # Occupancy logit 67x the detection logit, comparable directions: one global
  # step is sized off the larger arm and is then far too coarse for the smaller.
  a2 <- 20.0; v2 <- rep(-0.3, 4)
  u_a2 <- 0.4; u_v2 <- rep(0.4, 4)
  b <- .c3_brute(a2, v2, u_a2, u_v2, y)
  err_per_arm <- abs(.c3_tensor(a2, v2, u_a2, u_v2, y, TRUE)  - b) / abs(b)
  err_global  <- abs(.c3_tensor(a2, v2, u_a2, u_v2, y, FALSE) - b) / abs(b)
  expect_lt(err_per_arm, err_global)
  expect_lt(err_per_arm, 1e-5)
})

test_that("the tensor builder refuses an unregistered spec by name", {
  expect_error(
    cpp_cell_coupling_curvature3("no_such_spec", eta = list(0), u = list(0),
                                 y = list(0), family = "binomial", phi = 1),
    "not registered")
})

# --------------------------------------------------------------------------- #
# (gcol33/tulpa#448) An unreadable difference is not a smaller skew            #
#                                                                              #
# gamma_3 feeds the skew correction, and understating it moves the correction  #
# toward zero -- the direction that reads as "the Gaussian approximation was   #
# fine". A cell whose arm the difference quotient could not be formed on is    #
# precisely a cell where it was not, so the term most likely to drop is the    #
# term that mattered most. The contraction reports NaN there.                  #
#                                                                              #
# The per-arm drop and the whole-cell drop are not separable on this fixture:  #
# a CellCouplingSpec is evaluated cell-globally, so a non-finite eta or        #
# direction anywhere in the cell takes every arm's quotient down together.     #
# What is pinned here is the contract at the door -- no non-finite input ever  #
# comes back as a number -- and, below it, that the fix did not turn the one   #
# legitimate skip into a decline.                                              #
# --------------------------------------------------------------------------- #

test_that("no non-finite input is scored as a finite cubic term", {
  skip_on_cran()
  .c3_register()
  a <- 0.35; v <- c(-0.6, 0.2, -1.1, 0.4)
  u_a <- 0.17; u_v <- c(0.05, -0.09, 0.13, 0.02)
  y <- rep(0, 4)
  expect_true(is.finite(.c3_tensor(a, v, u_a, u_v, y)))

  for (bad in c(NaN, Inf, -Inf)) {
    for (step in c(TRUE, FALSE)) {
      # non-finite eta, on either arm
      expect_true(is.na(.c3_tensor(bad, v, u_a, u_v, y, step)))
      expect_true(is.na(.c3_tensor(a, replace(v, 2L, bad), u_a, u_v, y, step)))
      # non-finite direction, on either arm
      expect_true(is.na(.c3_tensor(a, v, bad, u_v, y, step)))
      expect_true(is.na(.c3_tensor(a, v, u_a, replace(u_v, 3L, bad), y, step)))
    }
  }
})

test_that("an arm the direction leaves alone is a zero, not a decline", {
  skip_on_cran()
  .c3_register()
  # The one skip inside the per-arm loop that is a VALUE: an arm with no
  # displacement contributes exactly zero to the cubic form, so the cell is
  # still scorable through the arm that did move. Reading it as unreadable
  # would decline every probe direction confined to one arm.
  a <- 0.35; v <- c(-0.6, 0.2, -1.1, 0.4); y <- rep(0, 4)
  occ_only <- .c3_tensor(a, v, 0.17, rep(0, 4), y)
  det_only <- .c3_tensor(a, v, 0.0,  c(0.05, -0.09, 0.13, 0.02), y)
  expect_true(is.finite(occ_only))
  expect_true(is.finite(det_only))
  expect_false(occ_only == 0)
  expect_false(det_only == 0)
  # And each reproduces a direct third derivative of the cell log-density. The
  # detection-only direction lands near 4e-06, where the five-point stencil is
  # itself dominated by cancellation, so it is read on the looser tolerance.
  expect_equal(occ_only, .c3_brute(a, v, 0.17, rep(0, 4), y), tolerance = 1e-3)
  expect_equal(det_only, .c3_brute(a, v, 0.0, c(0.05, -0.09, 0.13, 0.02), y),
               tolerance = 0.1)
})
