test_that("full-latent IS integrates a conjugate Gaussian marginal without bias", {
  set.seed(1174)
  y <- c(-.4, 1.2, .6)
  X <- matrix(c(1, 1, 1), 3, 1)
  obs <- c(1L, 2L, 1L)
  Q <- Matrix::Matrix(matrix(c(2, -.4, -.4, 1.5), 2), sparse = TRUE)
  blk <- tgmrf(Q = function(theta) Q,
    prior = function(theta) dnorm(theta, 0, 1, log = TRUE),
    mu = function(theta) c(.3, -.2), init = c(theta = 0), obs_idx = obs)
  A <- cbind(X, as.matrix(diag(2)[obs, ]))
  prior_cov <- as.matrix(Matrix::bdiag(tulpa:::.tulpa_prior_sd("laplace")^2,
                                      solve(Q)))
  prior_mean <- c(0, .3, -.2)
  phi <- 1.7
  precision <- solve(prior_cov) + crossprod(A) / phi
  mean <- solve(precision, solve(prior_cov, prior_mean) + crossprod(A, y) / phi)
  log_joint <- tulpa:::.tgmrf_log_joint(0, y, rep(1L, 3), X, blk, obs,
                                       rep(0L, 3), 0L, 1, "gaussian", phi)
  proposal <- list(mean = drop(mean), precision = Matrix::Matrix(precision, sparse = TRUE))
  estimates <- replicate(600L, exp(tulpa:::.tgmrf_is_estimate(
    proposal, 8L, log_joint)$log_prob))
  V <- phi * diag(3) + A %*% prior_cov %*% t(A)
  R <- chol(V)
  u <- forwardsolve(t(R), y - drop(A %*% prior_mean))
  truth <- exp(-3 / 2 * log(2 * pi) - sum(log(diag(R))) - sum(u^2) / 2)
  expect_lt(abs(mean(estimates) - truth), 5 * sd(estimates) / sqrt(length(estimates)))
  expect_lt(abs(mean(estimates) / truth - 1), .04)
})

test_that("pseudo-marginal MH retains its auxiliary state on rejection", {
  set.seed(1174)
  calls <- 0L
  target <- function(theta) {
    calls <<- calls + 1L
    list(log_prob = if (calls == 1L) 0 else -1e6, draw = calls)
  }
  fit <- tulpa:::.imh_laplace_chain(target, 0, matrix(1), 12L, 2L, thin = 3L)
  expect_equal(calls, 13L)
  expect_true(all(vapply(fit$states, `[[`, integer(1), "draw") == 1L))
  expect_equal(fit$draws, matrix(0, 4L, 1L))
  expect_equal(fit$log_prob, rep(0, 4L))
})

test_that("zero likelihood estimates preserve particle-row alignment", {
  set.seed(1174)
  proposal <- list(mean = c(0, 0), precision = Matrix::Diagonal(2))
  joint <- function(W) {
    colSums(matrix(dnorm(W, log = TRUE), nrow = 2)) +
      ifelse(W[1, ] > 0, 0, -Inf)
  }
  estimates <- replicate(600L, tulpa:::.tgmrf_is_estimate(proposal, 1L, joint),
                          simplify = FALSE)
  likelihood <- exp(vapply(estimates, `[[`, numeric(1), "log_prob"))
  draws <- do.call(rbind, lapply(estimates, `[[`, "draw"))
  expect_identical(dim(draws), c(600L, 2L))
  expect_true(any(likelihood == 0))
  expect_true(all(is.na(draws[likelihood == 0, , drop = FALSE])))
  expect_true(all(draws[likelihood > 0, 1] > 0))
  expect_lt(abs(mean(likelihood) - .5), 5 * sd(likelihood) / sqrt(length(likelihood)))
})

test_that("the measured log-estimate SD calibrates the shared particle budget", {
  skip_on_cran()
  set.seed(1174)
  settings <- utils::modifyList(tulpa:::.TGMRF_IS, list(
    initial_particles = 2L, pilot_batches = 24L, target_log_sd = .03,
    max_particles = 4096L))
  testthat::local_mocked_bindings(.TGMRF_IS = settings, .package = "tulpa")
  blk <- tgmrf(Q = function(theta) Matrix::Diagonal(3, exp(-2 * theta)),
    prior = function(theta) dnorm(theta, 0, .4, log = TRUE),
    init = c(log_sigma = 0))
  y <- c(-.2, 0, .5); X <- matrix(1, 3, 1)
  lm <- tulpa:::.tgmrf_make_log_marginal(y, rep(1L, 3), X, blk, 1:3,
    NULL, 0L, 1, "gaussian", 1.7, 50L, 1e-7, 1L)
  estimator <- tulpa:::.tgmrf_make_is(lm, 0, y, rep(1L, 3), X, blk, 1:3,
    NULL, 0L, 1, "gaussian", 1.7)
  expect_gt(estimator$n_particles, settings$initial_particles)
  expect_lte(estimator$log_estimate_sd, settings$target_log_sd)
  expect_gt(estimator$likelihood_evaluations(), estimator$n_particles)
})

test_that("uncorrected and gated NUTS fits report their Laplace target", {
  skip_on_cran()
  blk <- tgmrf(Q = function(theta) Matrix::Diagonal(3, exp(-2 * theta)),
    prior = function(theta) dnorm(theta, 0, .4, log = TRUE),
    init = c(log_sigma = 0), bounds = list(lower = -.8, upper = .8))
  args <- list(y = c(-.2, 0, .5), n_trials = rep(1L, 3), X = matrix(1, 3, 1),
               block = blk, family = "gaussian", mode = "nuts")
  fit <- do.call(tulpa_tgmrf, c(args, list(control = list(
    n_iter = 20L, warmup = 10L, max_depth = 2L, is_correction = FALSE))))
  expect_identical(fit$inference_tier, 2L)
  expect_identical(fit$inference_mode, "structured")
  expect_identical(fit$correction_declined, "not_requested")
  expect_false(fit$correction_applied)
  expect_match(fit$target, "Laplace-approximate", fixed = TRUE)
  fit <- do.call(tulpa_tgmrf, c(args, list(control = list(
    n_iter = 20L, warmup = 10L, max_depth = 2L))))
  expect_identical(fit$inference_tier, 2L)
  expect_identical(fit$correction_declined, "draws_too_few")
  expect_false(fit$correction_applied)
  expect_match(fit$target, "Laplace-approximate", fixed = TRUE)
})

test_that("a high PSIS k withholds the posterior correction", {
  skip_on_cran()
  testthat::local_mocked_bindings(tulpa_psis = function(log_ratios, ...) {
    list(pareto_k = .95, is_ess = 3, log_weights = rep(-log(length(log_ratios)),
                                                     length(log_ratios)))
  }, .package = "tulpa")
  blk <- tgmrf(Q = function(theta) Matrix::Diagonal(3, exp(-2 * theta)),
    prior = function(theta) dnorm(theta, 0, .4, log = TRUE),
    init = c(log_sigma = 0))
  fit <- tulpa_tgmrf(c(-.2, 0, .5), rep(1L, 3), matrix(1, 3, 1), blk,
    family = "gaussian", mode = "nuts", control = list(
      n_iter = 70L, warmup = 20L, max_depth = 2L))
  expect_false(fit$correction_applied)
  expect_identical(fit$correction_declined, "pareto_k_above_threshold")
  expect_identical(fit$inference_mode, "structured")
  expect_identical(fit$inference_tier, 2L)
  expect_equal(fit$pareto_k, .95)
  expect_match(fit$selection_reason, "not applied", fixed = TRUE)
})
