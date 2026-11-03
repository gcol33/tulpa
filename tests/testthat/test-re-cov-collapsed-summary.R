# A slope SD the data hold weakly puts the log-Cholesky mode down a funnel
# (log L22 near -10, curvature along it near zero), and a Gaussian laid there
# reaches design nodes of zero weight and scales that overflow when squared. A
# zero-weight node carries no mass, so it enters no moment; and the nested door
# lays such a block's nodes in the marginal-SD / partial-correlation
# coordinates, where the mode is interior.

test_that("a zero-weight node with an overflowing value leaves the moments finite", {
  m <- tulpa:::.nl_wtd_mean_sd(c(1, 3, 1e200), c(0.5, 0.5, 0))
  expect_equal(m$mean, 2)
  expect_equal(m$sd, 1)
  expect_true(is.na(tulpa:::.nl_wtd_mean_sd(c(1, 2), c(0.5, NA))$mean))
})

skip_on_cran()

test_that("a weakly held slope SD is placed where the posterior mass is", {
  # Binary (1 + x | g), 50 groups of 6, slope SD 0.7. The log-Cholesky mode is
  # at log L22 = -9.96; a grid laid from it spanned log L22 -552..532 with 162
  # of 729 nodes unsolvable and read sigma_2 at 0.0019 / [4.7e-5, 0.067].
  # Reference: a dense tensor over (log sigma_1, log sigma_2, atanh rho),
  # 0.05 / 0.15 / 0.15 apart over the box holding all but 1e-9 of each
  # marginal, with the same inner Laplace marginal and PC + LKJ prior; sigma_2
  # read off its marginal's CDF at 0.0103 / 0.238 / 0.737.
  set.seed(2)
  gg <- rep(1:50, each = 6)
  xx <- rnorm(300)
  b0 <- rnorm(50, 0, 1)
  b1 <- rnorm(50, 0, 0.7)
  yy <- rbinom(300, 1, plogis(-0.3 + 0.8 * xx + b0[gg] + b1[gg] * xx))
  db <- data.frame(y = yy, x = xx, g = factor(gg))
  fn <- tulpa(y ~ x + (1 + x | g), data = db, family = "binomial",
              mode = "laplace", control = list(diagnose_k = FALSE))
  expect_identical(fn$placement$coordinate, "sd_partial_correlation")
  expect_true(all(is.finite(fn$log_marginal)))
  expect_lt(diff(range(fn$theta_grid[, 3])), 20)
  h <- summary(fn)$hyperparameters
  expect_true(all(is.finite(h$mean)))
  expect_true(all(is.finite(h$sd)))
  s2 <- h[h$parameter == "sigma_2", ]
  expect_gt(s2$median, 0.15)
  expect_lt(s2$median, 0.30)
  expect_lt(s2$ci_lo, 0.05)
  expect_gt(s2$ci_hi, 0.55)
  expect_lt(s2$ci_hi, 1.0)
})
