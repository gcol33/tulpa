# Tests for src/smc_sampler.h + src/smc_sampler.cpp
#
# The Rcpp entry cpp_smc_test runs SMC on an unnormalised Gaussian-Gaussian
# conjugate pair, where prior, posterior moments, and log marginal likelihood
# all have closed forms. SMC is stochastic, so tolerances are intentionally
# loose.

# Closed-form posterior for x_i ~ N(0, sigma_prior^2) with single observation
# y_i ~ N(x_i, sigma_t_i^2).
#
# SMC starts by sampling from the *normalised* prior, then tempers an
# *unnormalised* likelihood. So the marginal likelihood it estimates is
#
#   Z_smc = integral [ p_prior(x) * exp(log_lik_unnorm(x)) ] dx
#         = (normalised prior) * (unnormalised likelihood kernel)
#
# Equivalently: standard log p(y) plus the missing likelihood normalisation:
#   log Z_smc_i = log p(y_i) + 0.5 * log(2 pi sigma_t_i^2)
#              = 0.5 * log(sigma_t_i^2 / (sigma_p^2 + sigma_t_i^2))
#                - 0.5 * mu_t_i^2 / (sigma_p^2 + sigma_t_i^2)
gauss_gauss_truth <- function(mu_t, sigma_t, sigma_prior = 10) {
  v_post  <- (sigma_prior^2 * sigma_t^2) / (sigma_prior^2 + sigma_t^2)
  m_post  <- v_post * (mu_t / sigma_t^2)
  log_Z_per_dim <- 0.5 * log(sigma_t^2 / (sigma_prior^2 + sigma_t^2)) -
                   0.5 * mu_t^2 / (sigma_prior^2 + sigma_t^2)
  list(mean = m_post, sd = sqrt(v_post), log_Z = sum(log_Z_per_dim))
}

skip_on_cran()

test_that("SMC recovers Gaussian-Gaussian posterior moments and log Z (1-D)", {
  mu_t <- 2.5
  sigma_t <- 1.0
  truth <- gauss_gauss_truth(mu_t, sigma_t)

  res <- cpp_smc_test(mu_target = mu_t, sigma_target = sigma_t,
                      n_particles = 2000, n_mcmc_steps = 8, seed = 1L)

  expect_equal(res$means[1], truth$mean, tolerance = 0.1)
  expect_equal(res$sds[1],   truth$sd,   tolerance = 0.1)
  # log Z: SMC is biased downwards in finite N; allow ~0.3 nats slack
  expect_equal(res$log_marginal_likelihood, truth$log_Z, tolerance = 0.5)
})

test_that("SMC recovers a 3-D Gaussian-Gaussian posterior", {
  mu_t    <- c(-1, 0, 2)
  sigma_t <- c(0.8, 1.2, 0.6)
  truth <- gauss_gauss_truth(mu_t, sigma_t)

  res <- cpp_smc_test(mu_target = mu_t, sigma_target = sigma_t,
                      n_particles = 2000, n_mcmc_steps = 8, seed = 2L)

  expect_equal(as.numeric(res$means), truth$mean, tolerance = 0.15)
  expect_equal(as.numeric(res$sds),   truth$sd,   tolerance = 0.15)
  expect_equal(res$log_marginal_likelihood, truth$log_Z, tolerance = 1.0)
})

test_that("SMC diagnostic invariants hold", {
  res <- cpp_smc_test(mu_target = c(1, -1), sigma_target = c(1, 1),
                      n_particles = 500, n_mcmc_steps = 4, seed = 3L)

  # Temperatures start at 0, end at exactly 1, and are non-decreasing
  temps <- res$temperatures
  expect_gt(length(temps), 1)
  expect_equal(temps[1], 0)
  expect_equal(temps[length(temps)], 1)
  expect_true(all(diff(temps) >= 0))

  # n_temperatures excludes beta=0 (the initial state)
  expect_equal(res$n_temperatures, length(temps) - 1L)

  # ESS stays in (0, N] and is recorded once per tempering step
  expect_equal(length(res$ess_history), res$n_temperatures)
  expect_true(all(res$ess_history > 0))
  expect_true(all(res$ess_history <= res$n_particles + 1e-9))

  # Mutation count is N * n_mcmc_steps * n_temperatures
  expect_equal(res$n_mutations, res$n_particles * 4L * res$n_temperatures)
})

test_that("Different seeds give different particle realisations", {
  a <- cpp_smc_test(mu_target = 1.0, sigma_target = 1.0,
                    n_particles = 200, n_mcmc_steps = 3, seed = 11L)
  b <- cpp_smc_test(mu_target = 1.0, sigma_target = 1.0,
                    n_particles = 200, n_mcmc_steps = 3, seed = 22L)
  # Means differ at least a little
  expect_false(isTRUE(all.equal(a$means, b$means, tolerance = 1e-6)))
})

test_that("Identical seeds give identical results (deterministic)", {
  a <- cpp_smc_test(mu_target = c(0, 1), sigma_target = c(1, 1),
                    n_particles = 200, n_mcmc_steps = 3, seed = 7L)
  b <- cpp_smc_test(mu_target = c(0, 1), sigma_target = c(1, 1),
                    n_particles = 200, n_mcmc_steps = 3, seed = 7L)
  expect_equal(a$means, b$means)
  expect_equal(a$sds,   b$sds)
  expect_equal(a$log_marginal_likelihood, b$log_marginal_likelihood)
  expect_equal(a$temperatures, b$temperatures)
})
