# The outer-cell support of a density-support weighted quantile
# (gcol33/tulpa#353).
#
# `.nl_wtd_quantile()` places the cumulative mid-mass `cumsum(w) - w/2` at each
# value. That leaves `w_1 / 2` of the mass below the first value and `w_n / 2`
# above the last, and the `"clamp"` policy has nowhere to put it: every
# probability there reports the extreme value, so the read cannot place its own
# outer half-cell of mass inside the interval it reports. On a grid the values
# are cell representatives with known spacing, so the missing support is the
# outer cells' own half-width -- that is `"extend"`, and it is what a `density`
# support dispatches to.
#
# What is pinned here: that nothing INSIDE the coordinate range moves, that the
# edge is the design's own half-spacing in the right coordinate, that the result
# stays a monotone quantile function, and that the other two policies and the
# other two supports are untouched.

test_that("extend leaves every interior probability byte-identical", {
  v <- exp(seq(log(0.2), log(1.5), length.out = 5))
  w <- c(0.30, 0.25, 0.20, 0.15, 0.10)
  p <- cumsum(w / sum(w)) - (w / sum(w)) / 2
  pr <- seq(p[1], p[5], length.out = 41L)
  expect_identical(.nl_wtd_quantile(v, w, pr, outside = "extend"),
                   .nl_wtd_quantile(v, w, pr, outside = "clamp"))
})

test_that("the edge is the outer cell's own half-spacing", {
  # A geometric axis is equally spaced in log, so its half-cell is mirrored
  # there and the edge stays positive.
  v <- exp(seq(log(0.2), log(1.5), length.out = 5))
  w <- rep(0.2, 5)
  h <- diff(log(v))[1]
  q <- .nl_wtd_quantile(v, w, c(0, 1), outside = "extend")
  expect_equal(q[1], exp(log(v[1]) - h / 2))
  expect_equal(q[2], exp(log(v[5]) + h / 2))
  expect_identical(.nl_cell_edges(v), unname(q))

  # A signed axis has no log coordinate, so the mirror is in the value.
  s <- c(-2, 0, 2)
  expect_equal(.nl_cell_edges(s), c(-3, 3))
  expect_equal(.nl_wtd_quantile(s, rep(1, 3), c(0, 1), outside = "extend"),
               c(-3, 3))
})

test_that("a grid's outer half-cell of mass lands inside the extended support", {
  # The whole content of the defect: with equal weights over m cells, w_1 / 2 of
  # the mass sits below the first coordinate. Under the clamp every probability
  # there reads back as the first coordinate; under the extension it maps onto
  # the outer cell's own interval, one-to-one.
  v <- exp(seq(log(0.2), log(1.5), length.out = 4))
  w <- rep(0.25, 4)
  pr <- c(0.01, 0.05, 0.10)                       # all below w_1 / 2 = 0.125
  cl <- .nl_wtd_quantile(v, w, pr, outside = "clamp")
  ex <- .nl_wtd_quantile(v, w, pr, outside = "extend")
  expect_true(all(cl == v[1]))
  expect_false(any(duplicated(ex)))
  expect_true(all(ex < v[1]))
  expect_true(all(ex > .nl_cell_edges(v)[1]))
})

test_that("the extended read is still a monotone quantile function", {
  set.seed(4L)
  v <- sort(exp(rnorm(9, 0, 0.5)))
  w <- runif(9)
  q <- .nl_wtd_quantile(v, w, seq(0, 1, by = 0.005), outside = "extend")
  expect_false(is.unsorted(q))
  expect_true(all(is.finite(q)))
  # A degenerate support has no spacing to mirror, so it stays the one value.
  expect_identical(.nl_wtd_quantile(3, 1, c(0, 0.5, 1), outside = "extend"),
                   rep(3, 3))
  # Duplicated coordinates are aggregated before the edges are taken, so a
  # repeated extreme value does not give the outer cell zero width.
  expect_identical(.nl_wtd_quantile(c(v, v[1]), c(w, w[1]), 0.5,
                                    outside = "extend"),
                   .nl_wtd_quantile(v, c(w[1] * 2, w[-1]), 0.5,
                                    outside = "extend"))
})

test_that("the axis read carries the extension through to a reported interval", {
  # `.nl_axis_quantiles()` is the entry point every nested fit's
  # `theta_ci_lo` / `theta_median` / `theta_ci_hi` come out of. A two-axis grid
  # whose marginal for axis 1 puts more than 0.025 of the weight on the first
  # value is the regime the clamp binds in.
  v1 <- exp(seq(log(0.2), log(1.5), length.out = 4))
  v2 <- c(0.5, 1.5)
  tg <- as.matrix(expand.grid(sigma = v1, tau = v2))
  lm <- c(0, -0.4, -3, -6, -0.2, -0.6, -3.2, -6.2)
  q <- .nl_axis_quantiles(tg, lm)
  w <- exp(lm - max(lm)); w <- w / sum(w)
  w1 <- tapply(w, tg[, "sigma"], sum)
  expect_gt(w1[[1]] / 2, 0.025)
  expect_lt(q$ci_lo[["sigma"]], min(v1))
  expect_gt(q$ci_lo[["sigma"]], .nl_cell_edges(v1)[1])
  expect_identical(unname(q$median[["sigma"]]),
                   .nl_wtd_quantile(v1, w1, 0.5, outside = "clamp"))
})

test_that("every CDF support dispatches to the extension, the moment rule does not", {
  v <- exp(seq(log(0.2), log(1.5), length.out = 5))
  w <- c(0.30, 0.25, 0.20, 0.15, 0.10)
  pr <- c(0.01, 0.5, 0.99)
  expect_identical(.nl_summary_quantile(v, w, pr, NA_character_, "density"),
                   .nl_wtd_quantile(v, w, pr, outside = "extend"))
  # A locally CCD-refined grid takes the SAME construction as the density read.
  # Its `mixed` tag records provenance and does not switch the formula
  # (gcol33/tulpa#317), so a refined fit and the unrefined fit of the same model
  # cannot report intervals built two different ways.
  expect_identical(.nl_summary_quantile(v, w, pr, NA_character_, "mixed"),
                   .nl_summary_quantile(v, w, pr, NA_character_, "density"))
  # A moment rule with no known domain still withholds the number.
  expect_true(all(is.na(
    .nl_summary_quantile(v, w, c(0.01, 0.99), NA_character_, "moment_rule"))))
})
