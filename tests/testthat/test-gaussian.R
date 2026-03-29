test_that("tulpa_gaussian fits a simple linear model", {
  set.seed(123)
  n <- 200
  x <- rnorm(n)
  y <- 2 + 3 * x + rnorm(n, sd = 0.5)
  df <- data.frame(y = y, x = x)

  fit <- tulpa_gaussian(y ~ x, data = df, iter = 2000, warmup = 1000,
                        step_size = 0.02, n_leapfrog = 20, seed = 42)

  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$n_samples, 1000)
  expect_equal(fit$p, 2)  # intercept + slope

  # Check posterior means recover true values (within reasonable tolerance)
  # True: intercept = 2, slope = 3, sigma = 0.5
  expect_true(abs(fit$means["beta[1]"] - 2) < 0.5,
              label = "Intercept should be near 2")
  expect_true(abs(fit$means["beta[2]"] - 3) < 0.5,
              label = "Slope should be near 3")
  expect_true(abs(fit$sigma - 0.5) < 0.3,
              label = "Sigma should be near 0.5")
})

test_that("tulpa_gaussian handles intercept-only model", {
  set.seed(456)
  n <- 100
  y <- rnorm(n, mean = 5, sd = 1)
  df <- data.frame(y = y)

  fit <- tulpa_gaussian(y ~ 1, data = df, iter = 1500, warmup = 500,
                        step_size = 0.05, n_leapfrog = 10, seed = 99)

  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$p, 1)  # intercept only
  expect_true(abs(fit$means["beta[1]"] - 5) < 1.0,
              label = "Intercept should be near 5")
})

test_that("print.tulpa_fit works", {
  set.seed(789)
  df <- data.frame(y = rnorm(50), x = rnorm(50))
  fit <- tulpa_gaussian(y ~ x, data = df, iter = 500, warmup = 250,
                        step_size = 0.05, n_leapfrog = 10)
  expect_output(print(fit), "tulpa fit")
})
