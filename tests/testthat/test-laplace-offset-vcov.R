# The fixed-effect curvature H_beta must be evaluated at the FULL linear
# predictor, including the observation offset: for non-gaussian families the
# GLM weight depends on eta, so an offset omitted from the Hessian eta gives
# vcov/SEs for a different model even though the mode itself is correct.

test_that("Laplace H_beta includes the offset (glm oracle, poisson)", {
  skip_on_cran()
  set.seed(404)
  n   <- 400L
  x   <- rnorm(n)
  off <- log(runif(n, 0.5, 10))            # varying exposure
  X   <- cbind(1, x)
  eta <- 0.2 + 0.5 * x + off
  y   <- rpois(n, exp(eta))

  fit <- tulpa_laplace(y = y, n_trials = rep(1L, n), X = X,
                       family = "poisson", offset = off)
  expect_true(fit$converged)

  ref <- stats::glm(y ~ x, family = stats::poisson(), offset = off)

  # Mode == MLE (weak built-in prior), and H_beta^{-1} == vcov(glm).
  expect_equal(unname(fit$mode[1:2]), unname(coef(ref)), tolerance = 1e-3)
  V <- solve(fit$H_beta)
  expect_equal(unname(V), unname(stats::vcov(ref)), tolerance = 0.02)
})

test_that("Laplace H_beta is invariant to folding a constant offset into the intercept", {
  skip_on_cran()
  set.seed(405)
  n   <- 360L
  g   <- rep(1:12, each = 30L)
  x   <- rnorm(n)
  b_g <- rnorm(12, 0, 0.5)
  eta <- -0.5 + 0.6 * x + b_g[g] + 1.5
  y   <- rbinom(n, 1L, plogis(eta))
  X   <- cbind(1, x)
  re  <- list(list(idx = g, n_groups = 12L, sigma = 0.5))

  # Same y: the no-offset fit absorbs the constant into the intercept, so eta
  # at the mode -- and therefore W and H_beta -- must match the offset fit.
  fit_off <- tulpa_laplace(y = y, n_trials = rep(1L, n), X = X, re_list = re,
                           family = "binomial", offset = rep(1.5, n))
  fit_fold <- tulpa_laplace(y = y, n_trials = rep(1L, n), X = X, re_list = re,
                            family = "binomial")
  expect_true(fit_off$converged && fit_fold$converged)

  expect_equal(fit_off$mode[1] + 1.5, fit_fold$mode[1], tolerance = 1e-4)
  expect_equal(fit_off$mode[2], fit_fold$mode[2], tolerance = 1e-4)
  expect_equal(fit_off$H_beta, fit_fold$H_beta, tolerance = 1e-4)
})
