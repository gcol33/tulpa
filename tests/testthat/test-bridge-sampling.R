# Bridge sampling: marginal likelihood estimator.
#
# Test against closed-form marginal likelihood for the conjugate
# normal-normal model:
#
#   y | theta ~ N(theta, 1),   theta ~ N(0, 10)
#
# Posterior: theta | y ~ N(y * 100/101, 100/101)
# Marginal: p(y) = N(y | 0, sqrt(101))

test_that("bridge_sampling matches closed-form on conjugate normal", {
  skip_on_cran()
  set.seed(42L)
  y <- 1.5
  prior_var <- 100
  lik_var <- 1
  post_var <- 1 / (1 / prior_var + 1 / lik_var)
  post_mean <- y * post_var / lik_var
  marg_var <- prior_var + lik_var
  log_marg_true <- dnorm(y, 0, sqrt(marg_var), log = TRUE)

  log_post <- function(theta) {
    dnorm(y, theta, sqrt(lik_var), log = TRUE) +
      dnorm(theta, 0, sqrt(prior_var), log = TRUE)
  }

  draws <- matrix(rnorm(4000, mean = post_mean, sd = sqrt(post_var)),
                  ncol = 1L)

  bs <- bridge_sampling(draws, log_post, n_proposal = 4000L,
                        verbose = FALSE)

  expect_true(bs$converged)
  expect_lt(bs$n_iter, 50L)
  expect_lt(abs(bs$log_marginal - log_marg_true), 0.01)
  # Relative-MSE diagnostic should be small for this easy case.
  expect_lt(bs$re_sd, 0.05)
})


test_that("bridge_sampling handles 2-D conjugate Gaussian", {
  skip_on_cran()
  set.seed(43L)
  # y | theta ~ N(theta, I_2),   theta ~ N(0, 10 * I_2)
  # Marginal: p(y) = product over dims.
  y <- c(0.7, -1.1)
  prior_sd <- sqrt(10)
  post_sd <- sqrt(10 / 11)
  post_mean <- y * 10 / 11
  log_marg_true <- sum(dnorm(y, 0, sqrt(11), log = TRUE))

  log_post <- function(theta) {
    sum(dnorm(y, theta, 1, log = TRUE)) +
      sum(dnorm(theta, 0, prior_sd, log = TRUE))
  }

  draws <- cbind(
    rnorm(4000, post_mean[1], post_sd),
    rnorm(4000, post_mean[2], post_sd)
  )

  bs <- bridge_sampling(draws, log_post, n_proposal = 4000L)
  expect_true(bs$converged)
  expect_lt(abs(bs$log_marginal - log_marg_true), 0.05)
})


test_that("bridge_sampling errors on too-few draws", {
  expect_error(
    bridge_sampling(matrix(rnorm(6), ncol = 2L), function(t) sum(t)),
    "at least 4 posterior draws"
  )
})


test_that("bridge_sampling errors on non-function log_posterior", {
  expect_error(
    bridge_sampling(matrix(rnorm(20), ncol = 1L), 42),
    "must be a function"
  )
})


test_that("bridge_sampling tolerates non-finite log-posterior on some draws", {
  skip_on_cran()
  set.seed(44L)
  # Same toy as above but log_post returns -Inf on a small fraction
  # of draws (simulating out-of-support proposals).
  y <- 0.3
  log_post <- function(theta) {
    if (abs(theta) > 5) return(-Inf)  # crude support cap
    dnorm(y, theta, 1, log = TRUE) + dnorm(theta, 0, sqrt(10), log = TRUE)
  }
  draws <- matrix(rnorm(2000, mean = y * 10 / 11, sd = sqrt(10 / 11)),
                  ncol = 1L)
  bs <- bridge_sampling(draws, log_post)
  expect_true(bs$converged)
  log_marg_true <- dnorm(y, 0, sqrt(11), log = TRUE)
  expect_lt(abs(bs$log_marginal - log_marg_true), 0.05)
})


test_that("logmeanexp / logsumexp_pair are numerically stable", {
  # Internal helpers — overflow safety check.
  expect_equal(tulpa:::logmeanexp(c(1000, 1001, 1002)),
               1001 + log(mean(exp(c(-1, 0, 1)))),
               tolerance = 1e-12)
  expect_equal(tulpa:::logsumexp_pair(1000, 1001),
               1001 + log(1 + exp(-1)),
               tolerance = 1e-12)
})
