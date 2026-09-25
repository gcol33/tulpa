test_that("tulpa_gaussian fits a simple linear model", {
  skip_on_cran()
  set.seed(123)
  n <- 200
  x <- rnorm(n)
  y <- 2 + 3 * x + rnorm(n, sd = 0.5)
  df <- data.frame(y = y, x = x)

  fit <- tulpa_gaussian(y ~ x, data = df,
                        control = list(iter = 2000, warmup = 1000,
                                       max_treedepth = 8,
                                       seed = 42))

  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$n_samples, 1000)
  expect_equal(fit$p, 2)  # intercept + slope

  # Posterior means recover the truth. At n = 200 with sd = 0.5 the slope's
  # sampling SE is ~0.035, so 0.15 is ~4 SE -- a real recovery bound, not a
  # sanity bound.
  expect_true(abs(fit$means["beta[1]"] - 2) < 0.15,
              label = "Intercept should be near 2")
  expect_true(abs(fit$means["beta[2]"] - 3) < 0.15,
              label = "Slope should be near 3")
  expect_true(abs(fit$sigma - 0.5) < 0.1,
              label = "Sigma should be near 0.5")
})

test_that("tulpa_gaussian handles intercept-only model", {
  skip_on_cran()
  set.seed(456)
  n <- 100
  y <- rnorm(n, mean = 5, sd = 1)
  df <- data.frame(y = y)

  fit <- tulpa_gaussian(y ~ 1, data = df,
                        control = list(iter = 1500, warmup = 500,
                                       adapt_delta = 0.85,
                                       seed = 99))

  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$p, 1)  # intercept only
  expect_true(abs(fit$means["beta[1]"] - 5) < 1.0,
              label = "Intercept should be near 5")
})

test_that("print.tulpa_fit works", {
  skip_on_cran()
  set.seed(789)
  df <- data.frame(y = rnorm(50), x = rnorm(50))
  fit <- tulpa_gaussian(y ~ x, data = df,
                        control = list(iter = 500, warmup = 250,
                                       seed = 3))
  expect_output(print(fit), "tulpa fit")
})

test_that("tulpa_gaussian runs the AD-gradient NUTS path and reports its acceptance (#897)", {
  # The fixed-step HMC it ran took a finite-difference gradient of the whole
  # log-posterior: ~1 s per iteration at n = 40, and accept_rate = -1.
  skip_on_cran()
  set.seed(1)
  d <- data.frame(x = rnorm(40))
  d$y <- 1 + 2 * d$x + rnorm(40, 0, 0.6)
  tm <- system.time(
    fit <- tulpa_gaussian(y ~ x, d, control = list(seed = 1, iter = 2000,
                                                   warmup = 1000)))
  expect_lt(tm[["elapsed"]], 30)             # was tens of minutes
  expect_equal(fit$sampler, "nuts")
  expect_gt(fit$accept_rate, 0.5)
  expect_lte(fit$accept_rate, 1)
  expect_equal(fit$accept_rate, mean(fit$accept_prob))
  expect_equal(fit$n_samples, 1000L)
  ref <- stats::lm(y ~ x, d)
  expect_equal(unname(fit$means[1:2]), unname(stats::coef(ref)),
               tolerance = 0.05)
  expect_equal(unname(fit$sigma), summary(ref)$sigma, tolerance = 0.1)
})

test_that("tulpa_gaussian refuses the retired fixed-step knobs", {
  d <- data.frame(x = 1:5, y = c(1.2, 1.9, 3.1, 4.2, 4.8))
  expect_error(tulpa_gaussian(y ~ x, d, control = list(step_size = 0.05)),
               "Unknown control knob")
  expect_error(tulpa_gaussian(y ~ x, d, control = list(adapt_delta = 1.2)),
               "adapt_delta")
})
