

test_that("imh_laplace errors on non-finite init", {
  log_post <- function(theta) {
    if (any(theta > 0)) -Inf else sum(dnorm(theta, log = TRUE))
  }
  expect_error(
    imh_laplace(log_post, mode = c(0), hessian = matrix(1, 1L, 1L),
                init = c(5), n_iter = 100L),
    "not finite"
  )
})

skip_on_cran_budget()


test_that("imh_laplace registers in Tier 1 (Exact)", {
  skip_on_cran_budget()
  expect_true("imh_laplace" %in% INFERENCE_TIERS$exact$backends)
  ti <- get_backend_tier("imh_laplace")
  expect_equal(ti$tier, 1L)
  expect_equal(ti$name, "Exact")
})


test_that("imh_laplace errors on non-PD Hessian", {
  skip_on_cran_budget()
  log_post <- function(theta) sum(dnorm(theta, log = TRUE))
  expect_error(
    imh_laplace(log_post, mode = c(0, 0),
                hessian = matrix(c(1, 2, 2, 1), 2L, 2L),
                n_iter = 100L),
    "positive-definite|singular"
  )
})

skip_on_cran()
# Independence Metropolis-Hastings with Laplace proposal.
# Test coverage:
#   1. Recovers posterior mean / sd on conjugate normal (1-D and 2-D)
#   2. Wired into INFERENCE_TIERS as Tier 1 (Exact)
#   3. Returns a tulpa_fit-shaped object usable with generic methods
#   4. Sensible errors on bad inputs

test_that("imh_laplace recovers conjugate-normal posterior (1-D)", {
  skip_on_cran()
  set.seed(101L)
  y <- 0.7
  prior_var <- 100
  lik_var <- 1
  post_var <- 1 / (1 / prior_var + 1 / lik_var)
  post_mean <- y * post_var / lik_var

  log_post <- function(theta) {
    dnorm(y, theta, sqrt(lik_var), log = TRUE) +
      dnorm(theta, 0, sqrt(prior_var), log = TRUE)
  }

  fit <- imh_laplace(
    log_post,
    mode = post_mean,
    hessian = matrix(1 / post_var, 1L, 1L),
    n_iter = 4000L
  )

  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$inference_tier, 1L)
  expect_equal(fit$backend, "imh_laplace")
  # The proposal is a t with the target's own location and curvature, so the
  # acceptance is set by the t / Gaussian density ratio alone: 0.92 here.
  expect_gt(fit$mean_accept, 0.85)
  expect_lt(abs(fit$means[1] - post_mean), 0.05)
  expect_lt(abs(stats::sd(fit$draws[, 1]) - sqrt(post_var)), 0.1)
})


test_that("imh_laplace recovers 2-D conjugate posterior", {
  skip_on_cran()
  set.seed(102L)
  y <- c(0.5, -0.8)
  post_var <- 10 / 11
  post_mean <- y * 10 / 11

  log_post <- function(theta) {
    sum(dnorm(y, theta, 1, log = TRUE)) +
      sum(dnorm(theta, 0, sqrt(10), log = TRUE))
  }

  H <- diag(1 / post_var, 2L)
  fit <- imh_laplace(log_post, mode = post_mean, hessian = H,
                     n_iter = 4000L)

  # 0.87 for the 2-D t / Gaussian density ratio.
  expect_gt(fit$mean_accept, 0.8)
  expect_true(all(abs(fit$means - post_mean) < 0.05))
  emp_sd <- apply(fit$draws, 2L, sd)
  expect_true(all(abs(emp_sd - sqrt(post_var)) < 0.1))
})


test_that("imh_laplace fit works with generic S3 methods", {
  skip_on_cran()
  set.seed(103L)
  log_post <- function(theta) {
    sum(dnorm(theta, mean = c(1, 2), sd = 1, log = TRUE))
  }
  H <- diag(1, 2L)
  fit <- imh_laplace(log_post, mode = c(1, 2), hessian = H,
                     n_iter = 1000L, warmup = 200L)

  # Generic methods from R/methods_generic.R.
  expect_s3_class(fit, "tulpa_fit")
  expect_no_error(coef(fit))
  expect_no_error(vcov(fit))
  expect_no_error(summary(fit))
  expect_no_error(glance(fit))
  ll <- logLik(fit)
  expect_s3_class(ll, "logLik")
})


test_that("imh_laplace flags low acceptance for non-Gaussian targets", {
  # Target = standard Cauchy. Laplace at the mode (0, info=1) is a poor
  # match. Acceptance should be noticeably below 1 — this is the
  # diagnostic value of imh_laplace: low accept = Laplace is biased.
  skip_on_cran()
  set.seed(104L)
  log_post <- function(theta) dcauchy(theta, log = TRUE)
  fit <- imh_laplace(log_post, mode = 0, hessian = matrix(1, 1L, 1L),
                     n_iter = 4000L, scale = 1.0)
  # Acceptance well below 1 (the Cauchy's tails are heavier than the
  # proposal's t tails).
  expect_lt(fit$mean_accept, 0.85)
  # But it still draws *something* — never gets stuck.
  expect_gt(length(unique(fit$draws[, 1])), 100L)
})


test_that("imh_laplace's t proposal recovers a target with heavier than Gaussian tails", {
  # A t_7 target: its tails decay polynomially, so a Gaussian proposal at the
  # mode's curvature (SD 0.94 against the target's 1.18) is not uniformly
  # ergodic for it, and a chain that wanders into the tail sticks there.
  # Measured over these 20 seeds, the Gaussian's SD error ranges -0.10 to
  # +0.51; the t proposal stays within 0.04 on every seed.
  skip_on_cran()
  log_post <- function(theta) stats::dt(theta, df = 7, log = TRUE)
  true_sd <- sqrt(7 / 5)
  err <- vapply(1:20, function(s) {
    fit <- imh_laplace(log_post, mode = 0, hessian = matrix(8 / 7, 1L, 1L),
                       n_iter = 6000L, warmup = 1000L, seed = s)
    stats::sd(fit$draws[, 1]) / true_sd - 1
  }, numeric(1))
  expect_lt(max(abs(err)), 0.06)
  fit <- imh_laplace(log_post, mode = 0, hessian = matrix(8 / 7, 1L, 1L),
                     n_iter = 200L, seed = 1L)
  expect_identical(fit$proposal$df, .imh_proposal("df"))
})
