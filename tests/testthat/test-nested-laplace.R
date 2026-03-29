# test-nested-laplace.R
# Tests for nested Laplace approximation (Feature 3a) and hot-start (Feature 5)

# Reuse helpers from sparse-cholesky tests
source(test_path("test-sparse-cholesky.R"), local = TRUE)

# =====================================================================
# Test: Hot-start reduces Newton iterations
# =====================================================================

test_that("hot-start reduces Newton iterations vs cold start", {
  adj <- make_grid_adjacency(5, 5)  # 25 sites (dense path)
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = -1.0, tau_spatial = 2.0, adj = adj
  )

  args <- list(
    y = dat$y, n = dat$n_trials, X = dat$X,
    re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat$spatial_idx, n_spatial_units = 25L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors, tau_spatial = 2.0,
    family = "binomial", phi = 1.0, max_iter = 100L, tol = 1e-6, n_threads = 1L
  )

  # Cold start
  res_cold <- do.call(cpp_laplace_fit_spatial, args)
  expect_true(res_cold$converged)

  # Warm start from the cold-start mode (same tau)
  res_warm <- do.call(cpp_laplace_fit_spatial, c(args, list(x_init = res_cold$mode)))
  expect_true(res_warm$converged)

  # Warm start should converge in fewer iterations (usually 1-2 vs 5-15)
  expect_true(res_warm$n_iter <= res_cold$n_iter)

  # Results should be nearly identical
  expect_equal(res_warm$log_marginal, res_cold$log_marginal, tolerance = 1e-6)
})

test_that("hot-start works across nearby tau values", {
  adj <- make_grid_adjacency(5, 5)
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = -0.5, tau_spatial = 3.0, adj = adj
  )

  args_base <- list(
    y = dat$y, n = dat$n_trials, X = dat$X,
    re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat$spatial_idx, n_spatial_units = 25L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors, family = "binomial", phi = 1.0,
    max_iter = 100L, tol = 1e-6, n_threads = 1L
  )

  # Fit at tau = 3.0
  res1 <- do.call(cpp_laplace_fit_spatial, c(args_base, list(tau_spatial = 3.0)))
  expect_true(res1$converged)

  # Fit at nearby tau = 3.5, cold vs warm
  res2_cold <- do.call(cpp_laplace_fit_spatial, c(args_base, list(tau_spatial = 3.5)))
  res2_warm <- do.call(cpp_laplace_fit_spatial,
                        c(args_base, list(tau_spatial = 3.5, x_init = res1$mode)))

  expect_true(res2_cold$converged)
  expect_true(res2_warm$converged)

  # Warm start should need fewer iterations
  expect_true(res2_warm$n_iter <= res2_cold$n_iter)

  # Both should give same answer
  expect_equal(res2_warm$log_marginal, res2_cold$log_marginal, tolerance = 1e-4)
})

# =====================================================================
# Test: Nested Laplace runs and produces reasonable output
# =====================================================================

test_that("nested_laplace_icar runs on small problem", {
  adj <- make_grid_adjacency(5, 5)  # 25 sites
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = -0.5, tau_spatial = 3.0, adj = adj
  )

  result <- nested_laplace_icar(
    y = dat$y, n_trials = dat$n_trials, X = dat$X,
    spatial_idx = dat$spatial_idx, n_spatial_units = 25L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    family = "binomial", tau_mode = 3.0,
    n_grid = 5L, grid_width = 3.0, verbose = FALSE
  )

  expect_true(is.list(result))
  expect_equal(length(result$tau_grid), 5L)
  expect_equal(length(result$log_marginal), 5L)
  expect_true(all(is.finite(result$log_marginal)))
  expect_true(result$tau_mean > 0)
  expect_true(result$tau_sd > 0)
  expect_true(is.finite(result$tau_mean))
  expect_true(is.finite(result$tau_sd))

  # Weights should sum to ~1
  expect_equal(sum(result$weights), 1.0, tolerance = 1e-6)

  # Mode vector should have correct length
  expect_equal(length(result$mode_at_tau_mode), 1L + 25L)
})

test_that("nested Laplace with auto tau_mode search", {
  adj <- make_grid_adjacency(5, 5)
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = 0.0, tau_spatial = 5.0, adj = adj
  )

  result <- nested_laplace_icar(
    y = dat$y, n_trials = dat$n_trials, X = dat$X,
    spatial_idx = dat$spatial_idx, n_spatial_units = 25L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    family = "binomial", tau_mode = NULL,
    n_grid = 7L, verbose = FALSE
  )

  expect_true(result$tau_mode > 0)
  expect_true(result$tau_mean > 0)
  expect_true(all(is.finite(result$log_marginal)))
})

test_that("hot-start reduces iterations in nested Laplace grid", {
  adj <- make_grid_adjacency(5, 5)
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = -0.5, tau_spatial = 3.0, adj = adj
  )

  result <- nested_laplace_icar(
    y = dat$y, n_trials = dat$n_trials, X = dat$X,
    spatial_idx = dat$spatial_idx, n_spatial_units = 25L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    family = "binomial", tau_mode = 3.0,
    n_grid = 9L, grid_width = 3.0, verbose = FALSE
  )

  # The first grid point starts cold (or close to mode),
  # subsequent points should need fewer iterations due to warm-starting
  # Check that later grid points typically converge faster
  mid <- ceiling(length(result$n_iter) / 2)
  # At least some warm-started points should need fewer iterations than the first
  expect_true(any(result$n_iter[2:length(result$n_iter)] <= result$n_iter[1]))
})
