# MALA: Metropolis-Adjusted Langevin Algorithm.

test_that("mala recovers standard normal", {
  skip_on_cran()
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
  skip_on_cran()
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
  skip_on_cran()
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
  skip_on_cran()
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


test_that("mala's dense metric samples a strongly correlated target exactly", {
  # A dense inverse mass changes the proposal, never the target: the MH step
  # keeps the draws exact, so the moments come back whatever the metric.
  set.seed(306L)
  S <- matrix(c(1, 0.95, 0.95, 1), 2)
  P <- solve(S)
  log_post <- function(t) -0.5 * drop(t %*% P %*% t)
  grad <- function(t) -drop(P %*% t)
  fit <- mala(log_post, grad, init = c(0, 0), n_iter = 4000L, warmup = 1000L,
              mass_matrix = S, seed = 1L)
  expect_equal(unname(cov(fit$draws)), S, tolerance = 0.15)
  expect_equal(unname(colMeans(fit$draws)), c(0, 0), tolerance = 0.15)
  expect_error(mala(log_post, grad, init = c(0, 0), mass_diag = c(1, 1),
                    mass_matrix = S), "at most one")
  expect_error(mala(log_post, grad, init = c(0, 0),
                    mass_matrix = matrix(c(1, 2, 2, 1), 2)),
               "positive definite")
})


# gcol33/tulpa#878: the front door ran MALA from the builder's zero start on an
# identity metric. On a GLMM the step dual-averaged down to the narrowest
# direction and the intercept chain crawled along the intercept / group-effect
# ridge: bulk ESS 1 to 9 of 1000, point estimates 0.31 to 0.48 across seeds
# against imh_laplace's 0.41. It now starts at the mode with the Laplace covariance
# as a dense metric.
test_that("front-door MALA mixes on a poisson random intercept at default length", {
  skip_on_cran()
  set.seed(12); J <- 15; n <- 200; gi <- sample(J, n, TRUE); x <- rnorm(n)
  d <- data.frame(y = rpois(n, exp(.5 + .4 * x + rnorm(J, 0, .6)[gi])), x,
                  g = factor(gi))
  lap <- tulpa(y ~ x + (1 | g), d, family = "poisson", mode = "laplace",
               sigma_re = .6)
  for (s in 1:2) {
    f <- expect_no_warning(tulpa(y ~ x + (1 | g), d, family = "poisson",
                                 mode = "mala", sigma_re = .6,
                                 control = list(seed = s)))
    expect_true(isTRUE(f$convergence$ok))
    expect_gt(f$convergence$ess_bulk_min, 100)
    expect_equal(coef(f), coef(lap), tolerance = 0.15)
  }
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
  skip_on_cran()
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
