# A periodic temporal GP's `period` is a LAG, and it makes the same trip the
# time axis does (gcol33/tulpa#687).
#
# validate_temporal_gp() standardizes the time values when scale_coords = TRUE
# (the default) and the kernel evaluates sin(pi * d / period) against those
# scaled lags. The declared period travelled untouched, so a monthly series with
# period = 12 met a scaled lag range of about 3.3: sin(pi d / 12) is monotone
# and small over that, and the periodic structure collapsed into a slowly
# decaying kernel with nothing periodic in it.

test_that("the period is expressed in the units the kernel sees", {
  months <- 0:59
  d <- data.frame(t = months, y = rnorm(length(months)))
  spec <- temporal_gp("t", cov = "periodic", period = 12)
  v <- tulpa:::validate_temporal_gp(spec, d)

  expect_equal(v$time_scale, sd(months))
  expect_equal(v$period_scaled, 12 / sd(months))

  # The declared number is left alone, so the spec still prints and validates as
  # the user wrote it.
  expect_equal(v$period, 12)

  # The point of the conversion: one period must span the same fraction of the
  # series in either coordinate.
  raw_cycles    <- diff(range(months)) / 12
  scaled_cycles <- diff(range(v$time_values)) / v$period_scaled
  expect_equal(scaled_cycles, raw_cycles, tolerance = 1e-10)
  expect_gt(scaled_cycles, 4)          # it was 0.28 before, i.e. no cycle at all
})

test_that("scale_coords = FALSE leaves the period alone", {
  months <- 0:59
  d <- data.frame(t = months, y = rnorm(length(months)))
  v <- tulpa:::validate_temporal_gp(
    temporal_gp("t", cov = "periodic", period = 12, scale_coords = FALSE), d)
  expect_equal(v$time_scale, 1)
  expect_equal(v$period_scaled, 12)
})

test_that("a non-periodic temporal GP carries no period either way", {
  d <- data.frame(t = 0:29, y = rnorm(30))
  v <- tulpa:::validate_temporal_gp(temporal_gp("t", cov = "matern", nu = 1.5), d)
  expect_null(v$period)
  expect_null(v$period_scaled)
})
