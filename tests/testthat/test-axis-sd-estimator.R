# Which estimator a reported per-axis `theta_sd` comes from (gcol33/tulpa#621).
#
# Two estimators of one quantity: the weighted spread of the axis marginal, and
# the 3-point parabola at the modal node. The parabola reads the curvature at
# the mode, so on a skewed axis it targets a different number and it moves with
# the spacing of the three nodes it reads -- a factor of two between two grids on
# one data set, which is what #621 reports. The weighted read is consistent as
# the grid refines but is a floor at zero once the marginal has collapsed onto
# one node, which is the case the parabola was added for.
#
# The choice is decided on the axis's own quadrature ESS, read off the weights,
# so it is not one estimator judging the other.

# A right-skewed axis marginal with an interior mode: Gamma(1.5, 2) as a log
# density, the copy-axis shape #621 measures the two estimators a factor of two
# apart on. Its own SD is `sqrt(1.5) * 2 = 2.449`.
SKEW_SD <- sqrt(1.5) * 2
skew_marginal <- function(v) {
  ifelse(v > 0, 0.5 * log(v) - v / 2, -Inf)
}

test_that("the axis ESS counts the nodes a marginal spreads over", {
  expect_equal(.nl_axis_quad_ess(log(rep(1, 7))), 7)
  expect_equal(.nl_axis_quad_ess(log(c(1, rep(1e-12, 6)))), 1, tolerance = 1e-6)
  expect_equal(.nl_axis_quad_ess(log(c(0.5, 0.5, 1e-15))), 2, tolerance = 1e-6)
  expect_true(is.na(.nl_axis_quad_ess(rep(-Inf, 4))))
  expect_true(is.na(.nl_axis_quad_ess(numeric(0))))
})

test_that("a resolved axis reports the weighted spread of its marginal", {
  v  <- seq(-4, 4, length.out = 17L)
  lm <- -0.5 * v^2
  ch <- .nl_axis_sd_choice(v, lm, log_axis = FALSE)
  expect_identical(ch$source, "weighted")
  expect_gte(ch$ess, .nl_diag("axis_sd_ess"))
  expect_equal(ch$sd, 1, tolerance = 1e-3)
  expect_true(is.na(ch$declined))
})

test_that("a collapsed axis reports the parabola instead of a floor", {
  # All but one node at negligible weight: the weighted spread is ~0 and says
  # nothing about the marginal, which is the regime the parabola serves.
  v  <- seq(-4, 4, length.out = 9L)
  lm <- -0.5 * (v / 0.15)^2
  ch <- .nl_axis_sd_choice(v, lm, log_axis = FALSE)
  expect_identical(ch$source, "stencil")
  expect_lt(ch$ess, .nl_diag("axis_sd_ess"))
  expect_equal(ch$sd, 0.15, tolerance = 1e-6)

  wtd <- .nl_axis_sd_choice(v, lm, log_axis = FALSE, min_ess = 1)$sd
  expect_lt(wtd, 0.02)
})

test_that("a collapsed axis whose parabola cannot be formed says so", {
  # Mode at an endpoint: the parabola declines and the weighted read stands,
  # with the reason recorded rather than the floor reported as a spread.
  v  <- seq(1, 5, length.out = 5L)
  lm <- c(0, -20, -40, -60, -80)
  ch <- .nl_axis_sd_choice(v, lm, log_axis = FALSE)
  expect_identical(ch$source, "weighted")
  expect_identical(ch$declined, "mode_at_edge")
})

test_that("the reported SD holds across grids on a skewed axis", {
  # #621's own comparison: one density, two grids. The weighted read agrees
  # between them; the parabola does not, which is why it is not the report.
  coarse  <- seq(0.2, 12, length.out = 15L)
  fine    <- seq(0.2, 12, length.out = 45L)
  shifted <- coarse + 0.5 * diff(coarse[1:2])

  sd_of <- function(v) .nl_axis_sd_choice(v, skew_marginal(v),
                                          log_axis = FALSE)$sd
  sten_of <- function(v) as.numeric(.nl_laplace_at_mode_sd_axis(
    v, skew_marginal(v), log_axis = FALSE))

  for (v in list(coarse, fine, shifted)) {
    expect_identical(.nl_axis_sd_choice(v, skew_marginal(v),
                                        log_axis = FALSE)$source, "weighted")
  }
  # Quadrature tolerance between grids of one density, and near the density's
  # own SD on all three.
  expect_equal(sd_of(coarse), sd_of(fine), tolerance = 0.03)
  expect_equal(sd_of(coarse), sd_of(shifted), tolerance = 0.03)
  expect_lt(abs(sd_of(fine) - SKEW_SD) / SKEW_SD, 0.10)

  # The parabola targets a different number on this axis -- half the spread --
  # and moves across the same grids by more than the weighted read does, which
  # is #621's factor of two on one data set.
  sten <- vapply(list(coarse, fine, shifted), sten_of, numeric(1))
  wtd  <- vapply(list(coarse, fine, shifted), sd_of, numeric(1))
  expect_lt(max(sten), min(wtd))
  expect_lt(sten[1L], 0.6 * wtd[1L])
  expect_gt(max(sten) / min(sten), 1.5)
  expect_lt(max(wtd) / min(wtd), 1.05)
})

test_that("a fit records which estimator produced each axis SD", {
  tg <- as.matrix(expand.grid(a = seq(-4, 4, length.out = 9L),
                              b = seq(-4, 4, length.out = 9L)))
  # `a` spreads, `b` collapses onto one node.
  lm <- -0.5 * tg[, "a"]^2 - 0.5 * (tg[, "b"] / 0.15)^2
  res <- .nl_attach_axis_sd(list(
    theta_grid = tg, log_marginal = lm,
    theta_sd = stats::setNames(c(NA_real_, NA_real_), c("a", "b"))))

  expect_identical(unname(res$theta_sd_source), c("weighted", "stencil"))
  expect_gte(res$theta_sd_ess[["a"]], .nl_diag("axis_sd_ess"))
  expect_lt(res$theta_sd_ess[["b"]], .nl_diag("axis_sd_ess"))
  expect_equal(unname(res$theta_sd[["a"]]), 1, tolerance = 1e-3)
  expect_equal(unname(res$theta_sd[["b"]]), 0.15, tolerance = 1e-6)
  expect_true(all(is.na(res$theta_sd_stencil_declined)))
})

test_that("the axis SD is read against the grid's own measure", {
  # Unequal spacing: the wide outer cells carry prior mass a node count does not
  # see, so the spread read with the quadrature weights differs from the one
  # read without them.
  v  <- c(0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 1.0, 2.0)
  tg <- matrix(v, ncol = 1L, dimnames = list(NULL, "sigma"))
  lm <- rep(0, length(v))
  base <- .nl_attach_axis_sd(list(theta_grid = tg, log_marginal = lm,
                                  theta_sd = c(sigma = NA_real_)))
  with_q <- .nl_attach_axis_sd(list(theta_grid = tg, log_marginal = lm,
                                    log_quad = .nl_grid_log_quad(tg),
                                    theta_sd = c(sigma = NA_real_)))
  expect_identical(with_q$theta_sd_source[["sigma"]], "weighted")
  expect_false(isTRUE(all.equal(base$theta_sd[["sigma"]],
                                with_q$theta_sd[["sigma"]])))
})

test_that("the consistency pass fires on the ESS, not on an SD comparison", {
  # A sharply peaked marginal on a coarse log grid: the axis marginal sits on
  # one node, so the pass adds points around the mode. The trigger is the ESS,
  # so it does not compare the reported SD against the estimator that placed
  # the points.
  lev <- exp(seq(log(0.1), log(10), length.out = 7L))
  specs <- list(hyper_axis_spec("sigma", grid = lev, log_scale = TRUE,
                                bounds = c(0, Inf), refinable = TRUE))
  tg <- matrix(lev, ncol = 1L, dimnames = list(NULL, "sigma"))
  lm <- -0.5 * ((log(lev) - log(1)) / 0.05)^2
  called <- 0L
  kernel_fn <- function(new_cells, warm_start = NULL, store_extras = FALSE) {
    called <<- called + 1L
    list(log_marginal = -0.5 * ((log(new_cells[, "sigma"]) - log(1)) / 0.05)^2,
         extras = NULL)
  }
  out <- .hyper_consistency_pass(
    theta_grid = tg, log_marginal = lm, extras = NULL,
    refining_axis = rep("", nrow(tg)), specs = specs,
    theta_mean = c(sigma = 1), kernel_fn = kernel_fn)
  expect_gt(out$n_added, 0L)
  expect_identical(out$info$axes, "sigma")
  expect_lt(out$info$ess_before, .nl_diag("axis_sd_ess"))

  # A resolved axis is left alone, and costs no kernel call.
  called <- 0L
  lm_wide <- -0.5 * ((log(lev) - log(1)) / 1.2)^2
  out2 <- .hyper_consistency_pass(
    theta_grid = tg, log_marginal = lm_wide, extras = NULL,
    refining_axis = rep("", nrow(tg)), specs = specs,
    theta_mean = c(sigma = 1), kernel_fn = kernel_fn)
  expect_identical(out2$n_added, 0L)
  expect_null(out2$info)
  expect_identical(called, 0L)
})

test_that("a design-weighted grid keeps the weighted read at any ESS", {
  # A central-composite design's nodes are not a per-axis lattice, so a 3-point
  # profile across them is not the curvature of anything. The corrected design
  # weights reproduce the Gaussian moments, so the weighted read is the
  # calibrated SD there whatever the ESS -- recorded, not left implicit.
  v  <- seq(-4, 4, length.out = 9L)
  lm <- -0.5 * (v / 0.15)^2
  tg <- matrix(v, ncol = 1L, dimnames = list(NULL, "a"))
  design <- .nl_attach_axis_sd(list(
    theta_grid = tg, log_marginal = lm, weight_kind = rep("design", 9L),
    theta_sd = c(a = NA_real_)))
  mass <- .nl_attach_axis_sd(list(
    theta_grid = tg, log_marginal = lm, weight_kind = rep("mass", 9L),
    theta_sd = c(a = NA_real_)))

  expect_identical(design$theta_sd_source[["a"]], "weighted")
  expect_identical(design$theta_sd_stencil_declined[["a"]], "design_weighted")
  expect_identical(mass$theta_sd_source[["a"]], "stencil")
  expect_lt(design$theta_sd[["a"]], mass$theta_sd[["a"]])
})

test_that("every nested path reports the estimator it used", {
  # The choice is made inside `.nl_posterior_moments()`, which every nested
  # driver goes through, rather than at each driver.
  res <- .nl_posterior_moments(
    list(theta_grid = seq(0.2, 4, length.out = 9L),
         log_marginal = -0.5 * ((seq(0.2, 4, length.out = 9L) - 2) / 0.6)^2,
         weights = rep(1 / 9, 9L)),
    "icar")
  expect_true(res$theta_sd_source %in% .NL_AXIS_SD_SOURCE)
  expect_true(is.finite(res$theta_sd_ess))
  expect_true(is.finite(res$theta_sd))
})
