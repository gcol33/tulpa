# test-nested-laplace-cpp.R
# Tests for C++ nested Laplace grid loop (Feature 3b)

source(test_path("test-sparse-cholesky.R"), local = TRUE)

# =====================================================================
# Test: C++ grid loop produces valid results
# =====================================================================

test_that("cpp_nested_laplace_icar runs and returns correct structure", {
  adj <- make_grid_adjacency(5, 5)  # 25 sites
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = -0.5, tau_spatial = 3.0, adj = adj
  )

  tau_grid <- exp(seq(log(0.5), log(10), length.out = 7))

  result <- cpp_nested_laplace_icar(
    y = dat$y, n = dat$n_trials, X = dat$X,
    re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat$spatial_idx, n_spatial_units = 25L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    tau_grid = tau_grid,
    family = "binomial", phi = 1.0, max_iter = 50L, tol = 1e-6, n_threads = 1L
  )

  expect_equal(result$n_grid, 7L)
  expect_equal(length(result$log_marginal), 7L)
  expect_equal(length(result$n_iter), 7L)
  expect_equal(nrow(result$modes), 7L)
  expect_equal(ncol(result$modes), 1L + 25L)
  expect_true(all(is.finite(result$log_marginal)))
  expect_true(all(result$n_iter > 0))
})

# =====================================================================
# Test: C++ matches R-level loop results
# =====================================================================

test_that("C++ grid loop matches R single-point calls", {
  adj <- make_grid_adjacency(5, 5)
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = -0.5, tau_spatial = 3.0, adj = adj
  )

  tau_grid <- c(1.0, 3.0, 8.0)
  n_x <- 1L + 25L

  # C++ grid loop
  cpp_result <- cpp_nested_laplace_icar(
    y = dat$y, n = dat$n_trials, X = dat$X,
    re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat$spatial_idx, n_spatial_units = 25L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    tau_grid = tau_grid,
    family = "binomial", phi = 1.0, max_iter = 50L, tol = 1e-6, n_threads = 1L
  )

  # Both the nested grid kernel and cpp_laplace_fit_spatial now route their inner
  # solve through the one unified spec path (B2-live), so each folds the same
  # beta-prior log-density (tau_beta = 1e-4 = DEFAULT_TAU_BETA) into the
  # log-marginal. They therefore agree exactly -- a test of grid-vs-single-point
  # agreement of the shared mode + likelihood + Hessian + prior.
  for (k in seq_along(tau_grid)) {
    r_result <- cpp_laplace_fit_spatial(
      y = dat$y, n = dat$n_trials, X = dat$X,
      re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
      spatial_idx = dat$spatial_idx, n_spatial_units = 25L,
      adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
      n_neighbors = adj$n_neighbors, tau_spatial = tau_grid[k],
      family = "binomial", phi = 1.0, max_iter = 50L, tol = 1e-6, n_threads = 1L
    )

    expect_equal(cpp_result$log_marginal[k], r_result$log_marginal,
                 tolerance = 1e-3,
                 label = paste("tau =", tau_grid[k]))
  }
})

# =====================================================================
# Test: C++ grid loop at large scale (sparse path)
# =====================================================================

test_that("cpp_nested_laplace_icar works with sparse Cholesky (300 sites)", {
  adj <- make_grid_adjacency(15, 20)  # 300 sites
  dat <- simulate_spatial_data(
    n_sites = 300, n_obs_per_site = 3,
    beta0 = -0.5, tau_spatial = 2.0, adj = adj
  )

  tau_grid <- exp(seq(log(0.5), log(8), length.out = 9))

  result <- cpp_nested_laplace_icar(
    y = dat$y, n = dat$n_trials, X = dat$X,
    re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat$spatial_idx, n_spatial_units = 300L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    tau_grid = tau_grid,
    family = "binomial", phi = 1.0, max_iter = 50L, tol = 1e-6, n_threads = 1L
  )

  expect_equal(result$n_grid, 9L)
  expect_true(all(is.finite(result$log_marginal)))
  expect_equal(ncol(result$modes), 1L + 300L)

  # Log-marginals should vary across the grid (not constant)
  lml_range <- max(result$log_marginal) - min(result$log_marginal)
  expect_true(lml_range > 0.1)  # non-trivial variation
})

# =====================================================================
# Test: Warm-start chain reduces total iterations
# =====================================================================

test_that("warm-start chain in C++ grid uses fewer total iterations", {
  adj <- make_grid_adjacency(5, 5)
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = -0.5, tau_spatial = 3.0, adj = adj
  )

  tau_grid <- exp(seq(log(1), log(10), length.out = 9))

  # C++ grid (warm-started chain)
  result_warm <- cpp_nested_laplace_icar(
    y = dat$y, n = dat$n_trials, X = dat$X,
    re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat$spatial_idx, n_spatial_units = 25L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    tau_grid = tau_grid,
    family = "binomial", phi = 1.0, max_iter = 50L, tol = 1e-6, n_threads = 1L
  )

  # Cold-start each point separately
  cold_iters <- integer(length(tau_grid))
  for (k in seq_along(tau_grid)) {
    r <- cpp_laplace_fit_spatial(
      y = dat$y, n = dat$n_trials, X = dat$X,
      re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
      spatial_idx = dat$spatial_idx, n_spatial_units = 25L,
      adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
      n_neighbors = adj$n_neighbors, tau_spatial = tau_grid[k],
      family = "binomial", phi = 1.0, max_iter = 50L, tol = 1e-6, n_threads = 1L
    )
    cold_iters[k] <- r$n_iter
  }

  # Warm-start chain should use fewer total iterations
  total_warm <- sum(result_warm$n_iter)
  total_cold <- sum(cold_iters)
  expect_true(total_warm <= total_cold)
})

# =====================================================================
# Benchmark: R vs C++ nested Laplace (informational, not a pass/fail test)
# =====================================================================

test_that("C++ grid is faster than R loop (benchmark)", {
  skip_on_cran()
  skip_if_fast()

  adj <- make_grid_adjacency(10, 10)  # 100 sites
  dat <- simulate_spatial_data(
    n_sites = 100, n_obs_per_site = 5,
    beta0 = -0.5, tau_spatial = 3.0, adj = adj
  )

  tau_grid <- exp(seq(log(0.5), log(10), length.out = 9))

  # Time C++ version
  t_cpp <- system.time({
    cpp_result <- cpp_nested_laplace_icar(
      y = dat$y, n = dat$n_trials, X = dat$X,
      re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
      spatial_idx = dat$spatial_idx, n_spatial_units = 100L,
      adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
      n_neighbors = adj$n_neighbors,
      tau_grid = tau_grid,
      family = "binomial", phi = 1.0, max_iter = 50L, tol = 1e-6, n_threads = 1L
    )
  })

  # Time R-level loop
  t_r <- system.time({
    prev_mode <- NULL
    for (k in seq_along(tau_grid)) {
      r_result <- cpp_laplace_fit_spatial(
        y = dat$y, n = dat$n_trials, X = dat$X,
        re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
        spatial_idx = dat$spatial_idx, n_spatial_units = 100L,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, tau_spatial = tau_grid[k],
        family = "binomial", phi = 1.0, max_iter = 50L, tol = 1e-6, n_threads = 1L,
        x_init = prev_mode
      )
      prev_mode <- r_result$mode
    }
  })

  message("\n  Benchmark (100 sites, 9 grid points):")
  message("    C++ grid:  ", round(t_cpp["elapsed"], 3), "s")
  message("    R loop:    ", round(t_r["elapsed"], 3), "s")
  message("    Speedup:   ", round(t_r["elapsed"] / max(t_cpp["elapsed"], 0.001), 1), "x")

  # Both should produce valid results
  expect_true(all(is.finite(cpp_result$log_marginal)))
})
