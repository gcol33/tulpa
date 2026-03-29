# test-nested-laplace-bym2.R
# Tests for BYM2 nested Laplace (2D grid over sigma_spatial, rho)

source(test_path("test-sparse-cholesky.R"), local = TRUE)

test_that("BYM2 nested Laplace runs with 2D grid", {
  adj <- make_grid_adjacency(5, 5)  # 25 sites
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = 0.0, tau_spatial = 2.0, adj = adj
  )

  # 3x3 grid over (sigma_spatial, rho)
  sigma_vals <- c(0.3, 0.5, 1.0)
  rho_vals <- c(0.3, 0.5, 0.8)
  grid <- expand.grid(sigma = sigma_vals, rho = rho_vals)

  result <- cpp_nested_laplace_bym2(
    y = dat$y, n = dat$n_trials, X = dat$X,
    re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat$spatial_idx, n_spatial_units = 25L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors, scale_factor = 1.0,
    sigma_spatial_grid = grid$sigma, rho_grid = grid$rho,
    family = "binomial", phi = 1.0,
    max_iter = 50L, tol = 1e-6, n_threads = 1L
  )

  expect_equal(result$n_grid, 9L)
  expect_true(all(is.finite(result$log_marginal)))
  expect_true(all(result$n_iter > 0))

  # Log-marginals should vary
  lml_range <- max(result$log_marginal) - min(result$log_marginal)
  expect_true(lml_range > 0.01)
})

test_that("BYM2 nested Laplace warm-start works", {
  adj <- make_grid_adjacency(5, 5)
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = -0.5, tau_spatial = 3.0, adj = adj
  )

  # 5 points along sigma axis (fixed rho)
  sigma_vals <- exp(seq(log(0.2), log(1.5), length.out = 5))
  rho_vals <- rep(0.5, 5)

  result <- cpp_nested_laplace_bym2(
    y = dat$y, n = dat$n_trials, X = dat$X,
    re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat$spatial_idx, n_spatial_units = 25L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors, scale_factor = 1.0,
    sigma_spatial_grid = sigma_vals, rho_grid = rho_vals,
    family = "binomial", phi = 1.0,
    max_iter = 50L, tol = 1e-6, n_threads = 1L
  )

  expect_true(all(is.finite(result$log_marginal)))
  # Warm-start should help later points converge faster
  expect_true(any(result$n_iter[2:5] <= result$n_iter[1]))
})
