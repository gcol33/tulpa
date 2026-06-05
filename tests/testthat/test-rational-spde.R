# test-rational-spde.R
# Tests for rational SPDE (fractional Matérn smoothness)

test_that("rational_spde_coefficients returns integer flag for integer nu", {
  rat <- rational_spde_coefficients(1.0)
  expect_true(rat$is_integer)
  expect_equal(length(rat$poles), 0)

  rat2 <- rational_spde_coefficients(2.0)
  expect_true(rat2$is_integer)
})

test_that("rational_spde_coefficients rejects fractional nu (gcol33/tulpa#71)", {
  # The wired rational assembly collapses to an integer alpha = 2 field, so
  # fractional smoothness cannot be represented. The coefficient function errors
  # rather than feed a mis-specified field.
  expect_error(rational_spde_coefficients(0.5, m = 2), "Fractional SPDE")
  expect_error(rational_spde_coefficients(1.5, m = 3), "Fractional SPDE")
})

test_that("fit_spde with nu = 0 runs without error", {
  # nu=0 (alpha=1) is the first-order operator — ill-conditioned, rarely used.
  # We verify it runs without crashing, not that it converges.
  set.seed(42)
  n_obs <- 80
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec <- spatial_spde(coords, nu = 0)

  y <- rbinom(n_obs, 1, 0.4)
  X <- matrix(1, nrow = n_obs, ncol = 1)

  # Should not error
  result <- fit_spde(y, X, spec, family = "binomial",
                     range = 0.5, sigma = 0.3)
  expect_true(result$n_iter > 0)
})

test_that("fit_spde works with nu = 1 (alpha = 2, standard)", {
  set.seed(42)
  n_obs <- 80
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec <- spatial_spde(coords, nu = 1)

  y <- rbinom(n_obs, 1, 0.4)
  X <- matrix(1, nrow = n_obs, ncol = 1)

  result <- fit_spde(y, X, spec, family = "binomial",
                     range = 0.3, sigma = 0.5)

  expect_true(result$converged)
  expect_true(is.finite(result$log_marginal))
})

test_that("fractional nu is rejected at spec construction (gcol33/tulpa#71)", {
  coords <- cbind(runif(20), runif(20))
  expect_error(spatial_spde(coords, nu = 0.5), "Fractional SPDE")
  expect_error(spatial_spde(coords, nu = 1.5), "Fractional SPDE")
})

test_that("different integer nu values produce different log-marginals", {
  set.seed(42)
  n_obs <- 100
  coords <- cbind(runif(n_obs), runif(n_obs))
  y <- rbinom(n_obs, 1, 0.4)
  X <- matrix(1, nrow = n_obs, ncol = 1)

  nu_vals <- c(0, 1, 2)
  lmls <- numeric(length(nu_vals))
  for (i in seq_along(nu_vals)) {
    spec <- spatial_spde(coords, nu = nu_vals[i])
    result <- fit_spde(y, X, spec, family = "binomial",
                       range = 0.3, sigma = 0.5)
    lmls[i] <- result$log_marginal
  }

  # Different nu should give different log-marginals
  expect_true(length(unique(round(lmls, 2))) > 1)
})
