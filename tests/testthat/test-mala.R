# MALA: Metropolis-Adjusted Langevin Algorithm.

test_that("mala recovers standard normal", {
  set.seed(301L)
  log_post <- function(t) -0.5 * sum(t^2)
  grad <- function(t) -t

  fit <- mala(log_post, grad, init = c(0, 0),
              n_iter = 4000L, warmup = 1000L)

  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$inference_tier, 1L)
  expect_equal(fit$backend, "mala")
  # Posterior means near 0.
  expect_true(all(abs(fit$means) < 0.1))
  # Empirical SDs near 1.
  emp_sd <- apply(fit$draws, 2L, sd)
  expect_true(all(abs(emp_sd - 1) < 0.15))
})


test_that("mala dual-averaging adapts toward target acceptance", {
  set.seed(302L)
  log_post <- function(t) -0.5 * sum(t^2)
  grad <- function(t) -t

  # Start with a deliberately bad epsilon; warmup should fix it.
  fit <- mala(log_post, grad, init = c(0, 0, 0),
              n_iter = 4000L, warmup = 2000L,
              epsilon = 5.0, target_accept = 0.574)

  # Adapted epsilon should land within a reasonable band.
  expect_gt(fit$epsilon, 0.05)
  expect_lt(fit$epsilon, 5.0)
  # Post-warmup acceptance roughly tracks target.
  expect_gt(fit$mean_accept, 0.30)
  expect_lt(fit$mean_accept, 0.85)
})


test_that("mala recovers shifted Gaussian", {
  set.seed(303L)
  mu_true <- c(2, -1, 0.5)
  log_post <- function(t) -0.5 * sum((t - mu_true)^2)
  grad <- function(t) -(t - mu_true)
  fit <- mala(log_post, grad, init = c(0, 0, 0), n_iter = 4000L)
  expect_true(all(abs(fit$means - mu_true) < 0.15))
})


test_that("mala registers in Tier 1 (Exact)", {
  expect_true("mala" %in% INFERENCE_TIERS$exact$backends)
  ti <- get_backend_tier("mala")
  expect_equal(ti$tier, 1L)
})


test_that("mala mass_diag preconditioner improves mixing on scaled target", {
  # Target with very different per-dimension scales: SD = (1, 100).
  # With mass_diag = c(1, 100^2), the proposal is rescaled and mixes.
  set.seed(304L)
  scales <- c(1, 100)
  log_post <- function(t) -0.5 * sum((t / scales)^2)
  grad <- function(t) -t / scales^2

  fit_pre <- mala(log_post, grad, init = c(0, 0),
                  n_iter = 3000L, warmup = 1000L,
                  mass_diag = scales^2)
  # With proper preconditioning, dim-2 SD should be in the right
  # ballpark (within factor of 2 of true SD = 100).
  emp_sd2 <- sd(fit_pre$draws[, 2])
  expect_gt(emp_sd2, 30)
  expect_lt(emp_sd2, 300)
})


test_that("mala errors on non-finite init", {
  log_post <- function(t) {
    if (t[1] > 100) -Inf else sum(dnorm(t, log = TRUE))
  }
  grad <- function(t) -t
  expect_error(
    mala(log_post, grad, init = c(200), n_iter = 100L),
    "not finite"
  )
})


test_that("mala fit works with generic methods", {
  set.seed(305L)
  log_post <- function(t) -0.5 * sum(t^2)
  grad <- function(t) -t
  fit <- mala(log_post, grad, init = c(0, 0), n_iter = 1000L,
              warmup = 200L)
  expect_no_error(coef(fit))
  expect_no_error(summary(fit))
  expect_no_error(vcov(fit))
  expect_no_error(glance(fit))
})
