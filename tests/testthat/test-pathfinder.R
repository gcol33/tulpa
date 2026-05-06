# Pathfinder (single-path): L-BFGS + Gaussian fit + ELBO scoring.

test_that("pathfinder recovers conjugate-normal mode and covariance", {
  set.seed(201L)
  y <- c(0.6, -0.4)
  prior_var <- 10
  post_var <- 1 / (1 / prior_var + 1)   # one obs per dim
  post_mean <- y * post_var

  log_post <- function(theta) {
    sum(dnorm(y, theta, 1, log = TRUE)) +
      sum(dnorm(theta, 0, sqrt(prior_var), log = TRUE))
  }

  pf <- pathfinder(log_post, init = c(0, 0), n_draws = 2000L)

  expect_s3_class(pf, "tulpa_fit")
  expect_equal(pf$inference_tier, 2L)
  expect_equal(pf$backend, "pathfinder")
  expect_true(pf$converged)
  expect_true(all(abs(pf$mode - post_mean) < 0.01))
  # Diagonal of cov should match post_var on each dim.
  expect_true(all(abs(diag(pf$cov) - post_var) < 0.05))
})


test_that("pathfinder ELBO is well-defined and finite", {
  set.seed(202L)
  log_post <- function(theta) {
    sum(dnorm(theta, mean = c(1, 2), sd = 1, log = TRUE))
  }
  pf <- pathfinder(log_post, init = c(0, 0), n_draws = 1000L)
  expect_true(is.finite(pf$elbo))
  # For a Gaussian target with Gaussian fit at the right mode/cov,
  # ELBO ≈ 0 on average (the variational gap is small modulo MC noise).
  expect_lt(abs(pf$elbo), 0.5)
})


test_that("pathfinder accepts user-supplied gradient", {
  set.seed(203L)
  log_post <- function(theta) -0.5 * sum(theta^2)
  grad_log_post <- function(theta) -theta
  pf <- pathfinder(log_post, init = c(2, -1),
                   grad_log_posterior = grad_log_post,
                   n_draws = 500L)
  expect_true(pf$converged)
  expect_true(all(abs(pf$mode) < 0.01))
})


test_that("pathfinder registers in Tier 2 (Structured)", {
  expect_true("pathfinder" %in% INFERENCE_TIERS$structured$backends)
  ti <- get_backend_tier("pathfinder")
  expect_equal(ti$tier, 2L)
  expect_equal(ti$name, "Structured")
})


test_that("pathfinder errors on non-finite init", {
  log_post <- function(theta) {
    if (theta[1] > 100) -Inf else sum(dnorm(theta, log = TRUE))
  }
  expect_error(
    pathfinder(log_post, init = c(200), n_draws = 100L),
    "not finite"
  )
})


test_that("pathfinder fit works with generic methods", {
  set.seed(204L)
  log_post <- function(theta) sum(dnorm(theta, log = TRUE))
  pf <- pathfinder(log_post, init = c(0, 0), n_draws = 500L)
  expect_no_error(coef(pf))
  expect_no_error(summary(pf))
  expect_no_error(vcov(pf))
})
