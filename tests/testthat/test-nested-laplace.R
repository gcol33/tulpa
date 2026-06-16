# test-nested-laplace.R
# Tests for nested Laplace approximation (Feature 3a) and hot-start (Feature 5)

# Reuse helpers from sparse-cholesky tests
source(test_path("test-sparse-cholesky.R"), local = TRUE)

# =====================================================================
# Test: Hot-start reduces Newton iterations
# =====================================================================

test_that("hot-start reduces Newton iterations vs cold start", {
  skip_on_cran()
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
  skip_on_cran()
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

# R-level dispatcher (post-refactor): tulpa_nested_laplace(prior = list(type=...))

icar_prior <- function(adj, n_sites, tau_grid = NULL) {
  list(
    type = "icar",
    spatial_idx = NULL,  # filled per call
    n_spatial_units = n_sites,
    adj_row_ptr = adj$adj_row_ptr,
    adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    tau_grid = tau_grid
  )
}

test_that("nested_laplace dispatches ICAR with explicit grid", {
  skip_on_cran()
  adj <- make_grid_adjacency(5, 5)
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = -0.5, tau_spatial = 3.0, adj = adj
  )
  prior <- icar_prior(adj, 25L,
                      tau_grid = exp(seq(log(0.5), log(20), length.out = 5)))
  prior$spatial_idx <- dat$spatial_idx

  result <- tulpa_nested_laplace(
    y = dat$y, n_trials = dat$n_trials, X = dat$X,
    prior = prior, family = "binomial"
  )

  expect_s3_class(result, "tulpa_nested_laplace")
  expect_equal(length(result$theta_grid), 5L)
  expect_equal(length(result$log_marginal), 5L)
  expect_true(all(is.finite(result$log_marginal)))
  expect_true(result$theta_mean > 0)
  expect_true(result$theta_sd > 0)
  expect_equal(sum(result$weights), 1.0, tolerance = 1e-6)
  # modes matrix: n_grid x (p + n_spatial_units)
  expect_equal(nrow(result$modes), 5L)
  expect_equal(ncol(result$modes), 1L + 25L)
})

test_that("nested_laplace ICAR uses default grid when none supplied", {
  skip_on_cran()
  adj <- make_grid_adjacency(5, 5)
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = 0.0, tau_spatial = 5.0, adj = adj
  )
  prior <- icar_prior(adj, 25L)        # tau_grid = NULL
  prior$spatial_idx <- dat$spatial_idx

  result <- tulpa_nested_laplace(
    y = dat$y, n_trials = dat$n_trials, X = dat$X,
    prior = prior, family = "binomial"
  )

  expect_true(result$theta_mean > 0)
  expect_true(all(is.finite(result$log_marginal)))
  expect_true(length(result$theta_grid) >= 5L)
})

test_that("warm-start chain reduces inner iterations across grid", {
  skip_on_cran()
  adj <- make_grid_adjacency(5, 5)
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = -0.5, tau_spatial = 3.0, adj = adj
  )
  prior <- icar_prior(adj, 25L,
                      tau_grid = exp(seq(log(1), log(10), length.out = 9)))
  prior$spatial_idx <- dat$spatial_idx

  result <- tulpa_nested_laplace(
    y = dat$y, n_trials = dat$n_trials, X = dat$X,
    prior = prior, family = "binomial"
  )

  # Subsequent points should typically need <= as many iters as the first
  expect_true(any(result$n_iter[2:length(result$n_iter)] <= result$n_iter[1]))
})
