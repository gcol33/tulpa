test_that("axis log densities group by their numeric keys", {
  values <- c(3, 1, 3, 2, 1, 2)
  log_mass <- log(c(2, 1, 5, 3, 4, 7))
  out <- tulpa:::.nl_axis_marginal_logdensity(values, log_mass)
  expect_identical(out$vals, c(1, 2, 3))
  expect_equal(out$log_marg, log(c(5, 10, 7)), tolerance = 1e-15)
  keep <- c(TRUE, FALSE, TRUE, FALSE, TRUE, TRUE)
  out <- tulpa:::.nl_axis_marginal_logdensity(values, log_mass, keep)
  expect_equal(out$log_marg, log(c(4, 7, 7)), tolerance = 1e-15)
})

test_that("grouped log densities retain small finite masses and non-finite conventions", {
  out <- tulpa:::.nl_axis_marginal_logdensity(c(1, 1, 2, 2), c(-1000, -1001, 1000, 1001))
  expect_equal(out$log_marg, c(-1000, 1001) + log1p(exp(-1)), tolerance = 1e-15)
  out <- tulpa:::.nl_axis_marginal_logdensity(c(1, 1, 2), c(NA, 0, 3))
  expect_identical(out$log_marg, c(-Inf, 3))
  out <- tulpa:::.nl_axis_marginal_logdensity(c(1, NA, 2), c(0, 1, 3))
  expect_identical(out$log_marg, c(-Inf, -Inf))
})

test_that("an importance-sized axis preserves every singleton density", {
  skip_on_cran()
  set.seed(351)
  values <- sample(seq_len(65536L))
  log_mass <- sin(values) - values / 100
  out <- tulpa:::.nl_axis_marginal_logdensity(values, log_mass)
  expect_identical(out$vals, seq_len(65536L))
  expect_identical(out$log_marg, log_mass[order(values)])
})
