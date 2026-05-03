# Nested Laplace dispatch through the proper-CAR backend (2D grid over (tau, rho)).

source(test_path("test-sparse-cholesky.R"), local = TRUE)

test_that("proper-CAR nested Laplace runs on a 2D grid", {
  adj <- make_grid_adjacency(5, 5)  # 25 sites
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = 0.0, tau_spatial = 2.0, adj = adj
  )

  tau_vals <- c(0.5, 2, 8)
  rho_vals <- c(0.2, 0.6, 0.9)
  gr <- expand.grid(tau = tau_vals, rho = rho_vals)

  result <- cpp_nested_laplace_car_proper(
    y = dat$y, n = dat$n_trials, X = dat$X,
    re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat$spatial_idx, n_spatial_units = 25L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    tau_grid = gr$tau, rho_grid = gr$rho,
    family = "binomial", phi = 1.0,
    max_iter = 50L, tol = 1e-6, n_threads = 1L
  )

  expect_equal(result$n_grid, 9L)
  expect_true(all(is.finite(result$log_marginal)))
  expect_true(all(result$n_iter > 0))
  # Log-marginal varies across the (tau, rho) grid
  expect_gt(max(result$log_marginal) - min(result$log_marginal), 0.01)
})

test_that("proper-CAR collapses to ICAR as rho -> 1", {
  adj <- make_grid_adjacency(4, 4)
  dat <- simulate_spatial_data(
    n_sites = 16, n_obs_per_site = 8,
    beta0 = -0.3, tau_spatial = 4.0, adj = adj
  )

  tau_grid <- c(1, 4)
  res_proper <- cpp_nested_laplace_car_proper(
    y = dat$y, n = dat$n_trials, X = dat$X,
    re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat$spatial_idx, n_spatial_units = 16L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    # rho slightly below 1 — Q nearly singular but still PD
    tau_grid = tau_grid, rho_grid = rep(0.999, 2),
    family = "binomial", phi = 1.0,
    max_iter = 80L, tol = 1e-6, n_threads = 1L
  )

  res_icar <- cpp_nested_laplace_icar(
    y = dat$y, n = dat$n_trials, X = dat$X,
    re_idx = rep(0L, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat$spatial_idx, n_spatial_units = 16L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    tau_grid = tau_grid,
    family = "binomial", phi = 1.0,
    max_iter = 80L, tol = 1e-6, n_threads = 1L
  )

  # At rho≈1, proper-CAR Q is full rank (smallest eigenvalue ≈ 1-rho) while
  # ICAR's Q has rank n-1. Proper-CAR therefore lightly shrinks its mode
  # toward zero compared to ICAR, but the *shape* of the spatial pattern
  # must agree — check correlation of the spatial coefficients.
  p <- ncol(dat$X)
  spatial_cols <- (p + 1):(p + 16)
  mode_proper <- res_proper$modes[1, spatial_cols]
  mode_icar <- res_icar$modes[1, spatial_cols]
  expect_gt(stats::cor(mode_proper, mode_icar), 0.999)
})

test_that("tulpa_nested_laplace() routes a car_proper prior to the new backend", {
  adj <- make_grid_adjacency(4, 5)
  dat <- simulate_spatial_data(
    n_sites = 20, n_obs_per_site = 8,
    beta0 = 0.1, tau_spatial = 3.0, adj = adj
  )

  prior <- list(
    type = "car_proper",
    spatial_idx = dat$spatial_idx,
    n_spatial_units = 20L,
    adj_row_ptr = adj$adj_row_ptr,
    adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    rho_bounds = c(0, 1)
  )

  res <- tulpa_nested_laplace(dat$y, dat$n_trials, dat$X,
                        prior = prior, family = "binomial")

  expect_s3_class(res, "tulpa_nested_laplace")
  expect_named(res$theta_mean, c("tau", "rho"))
  expect_equal(sum(res$weights), 1.0, tolerance = 1e-6)
  # Posterior rho lives strictly inside the eigenvalue interval
  expect_gt(res$theta_mean[["rho"]], 0)
  expect_lt(res$theta_mean[["rho"]], 1)
})

test_that("tulpa_nested_laplace() accepts spatial_car_proper spec", {
  adj <- make_grid_adjacency(4, 4)
  W <- matrix(0, 16, 16)
  # Reconstruct dense adjacency from CSR to feed spatial_car_proper
  for (i in seq_len(16)) {
    s <- adj$adj_row_ptr[i] + 1L
    e <- adj$adj_row_ptr[i + 1]
    if (e >= s) {
      nbs <- adj$adj_col_idx[s:e] + 1L
      W[i, nbs] <- 1
    }
  }

  dat <- simulate_spatial_data(
    n_sites = 16, n_obs_per_site = 8,
    beta0 = 0.0, tau_spatial = 3.0, adj = adj
  )
  df <- data.frame(site = factor(dat$spatial_idx, levels = seq_len(16)))

  spec <- spatial_car_proper(W, level = "group", group_var = "site")
  res <- tulpa_nested_laplace(dat$y, dat$n_trials, dat$X,
                        spec = spec, data = df, family = "binomial")

  expect_s3_class(res, "tulpa_nested_laplace")
  expect_named(res$theta_mean, c("tau", "rho"))
  expect_true(all(is.finite(res$log_marginal)))
})
