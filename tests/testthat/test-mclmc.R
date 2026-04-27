# Tests for src/mclmc.h + src/mclmc.cpp
#
# Two samplers: unadjusted MCLMC (always accepts; eps adapted via energy-error
# cube-root rule) and MAMCLMC (Metropolis-adjusted; eps adapted via dual
# averaging targeting acceptance ~0.9). Both run on a diagonal Gaussian where
# posterior moments are exactly known.

skip_if_not_installed("testthat")

# Verify both samplers recover means/SDs of a standard 1-D Gaussian.
test_that("unadjusted MCLMC recovers a 1-D Gaussian", {
  # Long chain to keep stochastic noise below the asserted tolerance
  res <- cpp_mclmc_test(mu_target = 0.0, sigma_target = 1.0,
                        init = 0.0, n_iter = 10000, n_warmup = 2000,
                        seed = 1L, adjusted = FALSE)
  expect_equal(res$means[1], 0.0, tolerance = 0.2)
  expect_equal(res$sds[1],   1.0, tolerance = 0.25)
  expect_equal(res$n_draws, 10000)
  expect_true(res$step_size > 1e-4 && res$step_size < 5)
  expect_true(res$accept_rate > 0.5)  # mostly non-divergent
})

test_that("MAMCLMC recovers a 1-D Gaussian and acceptance is reasonable", {
  res <- cpp_mclmc_test(mu_target = 0.0, sigma_target = 1.0,
                        init = 0.0, n_iter = 10000, n_warmup = 2000,
                        seed = 1L, adjusted = TRUE)
  expect_equal(res$means[1], 0.0, tolerance = 0.2)
  expect_equal(res$sds[1],   1.0, tolerance = 0.3)
  # Dual-averaging targets 0.9; warmup may not fully converge — be permissive.
  expect_true(res$accept_rate > 0.3 && res$accept_rate <= 1.0,
              info = paste("accept_rate =", res$accept_rate))
})

test_that("unadjusted MCLMC recovers a 3-D anisotropic Gaussian", {
  mu <- c(-1, 0, 2)
  sd <- c(0.5, 1.5, 1.0)
  res <- cpp_mclmc_test(mu_target = mu, sigma_target = sd,
                        init = c(0, 0, 0), n_iter = 6000, n_warmup = 2000,
                        seed = 2L, adjusted = FALSE)
  expect_equal(as.numeric(res$means), mu, tolerance = 0.25)
  expect_equal(as.numeric(res$sds),   sd, tolerance = 0.25)
})

test_that("MAMCLMC recovers a 3-D anisotropic Gaussian", {
  mu <- c(-1, 0, 2)
  sd <- c(0.5, 1.5, 1.0)
  res <- cpp_mclmc_test(mu_target = mu, sigma_target = sd,
                        init = c(0, 0, 0), n_iter = 6000, n_warmup = 2000,
                        seed = 2L, adjusted = TRUE)
  expect_equal(as.numeric(res$means), mu, tolerance = 0.3)
  expect_equal(as.numeric(res$sds),   sd, tolerance = 0.3)
})

test_that("MCLMC reports gradient evaluations and structural fields sanely", {
  res <- cpp_mclmc_test(mu_target = 0.0, sigma_target = 1.0,
                        init = 0.0, n_iter = 500, n_warmup = 200,
                        seed = 3L, adjusted = FALSE)
  # Each iteration runs L leapfrog steps, each calling the gradient once,
  # plus the initial gradient at q0. So n_grad_evals >= total_iter * L_final.
  expect_gte(res$n_grad_evals, (500 + 200) * res$L)
  expect_equal(nrow(res$draws), 500)
  expect_gte(res$n_divergences, 0)
})

test_that("Identical seeds give identical results (deterministic)", {
  a <- cpp_mclmc_test(mu_target = 0.0, sigma_target = 1.0, init = 0.0,
                      n_iter = 200, n_warmup = 100, seed = 7L, adjusted = FALSE)
  b <- cpp_mclmc_test(mu_target = 0.0, sigma_target = 1.0, init = 0.0,
                      n_iter = 200, n_warmup = 100, seed = 7L, adjusted = FALSE)
  expect_equal(a$means, b$means)
  expect_equal(a$sds,   b$sds)
  expect_equal(a$accept_rate, b$accept_rate)

  c <- cpp_mclmc_test(mu_target = 0.0, sigma_target = 1.0, init = 0.0,
                      n_iter = 200, n_warmup = 100, seed = 7L, adjusted = TRUE)
  d <- cpp_mclmc_test(mu_target = 0.0, sigma_target = 1.0, init = 0.0,
                      n_iter = 200, n_warmup = 100, seed = 7L, adjusted = TRUE)
  expect_equal(c$means, d$means)
  expect_equal(c$sds,   d$sds)
  expect_equal(c$accept_rate, d$accept_rate)
})

test_that("Different seeds give different particle realisations", {
  a <- cpp_mclmc_test(mu_target = 0.0, sigma_target = 1.0, init = 0.0,
                      n_iter = 500, n_warmup = 200, seed = 11L, adjusted = TRUE)
  b <- cpp_mclmc_test(mu_target = 0.0, sigma_target = 1.0, init = 0.0,
                      n_iter = 500, n_warmup = 200, seed = 22L, adjusted = TRUE)
  expect_false(isTRUE(all.equal(a$means, b$means, tolerance = 1e-6)))
})
