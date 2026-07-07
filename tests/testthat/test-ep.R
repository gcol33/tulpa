# Expectation Propagation for a GLM (gcol33/tulpa C6). The definitive correctness
# anchor is that EP is EXACT for a Gaussian likelihood: the EP posterior must
# equal the closed-form Gaussian posterior. Also checked: Gauss-Hermite accuracy
# and logistic-GLM coefficient recovery.

test_that("Gauss-Hermite quadrature is accurate", {
  gh <- tulpa:::.gauss_hermite(20L)
  expect_equal(sum(gh$w), sqrt(pi), tolerance = 1e-10)
  # integral x^2 exp(-x^2) / integral exp(-x^2) = 1/2
  expect_equal(sum(gh$w * gh$x^2) / sum(gh$w), 0.5, tolerance = 1e-10)
  # odd moment is zero
  expect_equal(sum(gh$w * gh$x^3), 0, tolerance = 1e-8)
})

test_that("EP is exact for a Gaussian likelihood (closed-form tilted moments)", {
  set.seed(1)
  n <- 60L; X <- cbind(1, rnorm(n))
  beta <- c(0.5, -0.8); sig2 <- 0.7
  y <- as.numeric(X %*% beta) + rnorm(n, 0, sqrt(sig2))
  d <- data.frame(y = y, x = X[, 2])

  fit <- tulpa_ep(y ~ x, data = d, family = "gaussian", phi = sig2,
                  beta_prior_sd = 10)

  # Closed-form Gaussian posterior with prior N(0, 100 I).
  P0  <- diag(1 / 100, 2)
  Vex <- solve(P0 + crossprod(X) / sig2)
  mex <- as.numeric(Vex %*% (crossprod(X, y) / sig2))

  expect_equal(unname(coef(fit)), mex, tolerance = 1e-5)
  expect_equal(unname(vcov(fit)), unname(Vex), tolerance = 1e-5)
  expect_true(fit$converged)
})

test_that("EP recovers logistic-GLM coefficients", {
  skip_on_cran()
  set.seed(2)
  n <- 700L; x <- rnorm(n)
  b <- c(-0.3, 0.9)
  y <- rbinom(n, 1, plogis(b[1] + b[2] * x))
  d <- data.frame(y = y, x = x)

  fit <- tulpa_ep(y ~ x, data = d, family = "binomial")
  mle <- unname(coef(stats::glm(y ~ x, family = "binomial", data = d)))
  expect_lt(max(abs(unname(coef(fit)) - mle)), 0.1)
  expect_true(fit$converged)
  # PD covariance and draws.
  expect_true(all(eigen(vcov(fit), symmetric = TRUE, only.values = TRUE)$values > 0))
  expect_equal(ncol(fit$draws), 2L)
})
