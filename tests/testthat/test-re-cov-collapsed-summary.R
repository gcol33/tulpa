# A slope SD collapsed toward zero puts the outer Gaussian's scale along log
# L22 at many posterior SDs, so far design nodes carry zero weight and scales
# that overflow when squared. A zero-weight node carries no mass, so it enters
# no moment.

test_that("a zero-weight node with an overflowing value leaves the moments finite", {
  m <- tulpa:::.nl_wtd_mean_sd(c(1, 3, 1e200), c(0.5, 0.5, 0))
  expect_equal(m$mean, 2)
  expect_equal(m$sd, 1)
  expect_true(is.na(tulpa:::.nl_wtd_mean_sd(c(1, 2), c(0.5, NA))$mean))
})

skip_on_cran()

test_that("a collapsed slope SD reports finite moments for every Sigma entry", {
  set.seed(2)
  gg <- rep(1:50, each = 6)
  xx <- rnorm(300)
  b0 <- rnorm(50, 0, 1)
  b1 <- rnorm(50, 0, 0.7)
  yy <- rbinom(300, 1, plogis(-0.3 + 0.8 * xx + b0[gg] + b1[gg] * xx))
  db <- data.frame(y = yy, x = xx, g = factor(gg))
  fn <- tulpa(y ~ x + (1 + x | g), data = db, family = "binomial",
              mode = "laplace", control = list(diagnose_k = FALSE))
  h <- summary(fn)$hyperparameters
  expect_lt(h$median[h$parameter == "sigma_2"], 0.05)
  expect_true(all(is.finite(h$mean)))
  expect_true(all(is.finite(h$sd)))
})
