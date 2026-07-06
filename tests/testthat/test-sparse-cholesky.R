# test-sparse-cholesky.R
# Validates CHOLMOD sparse solver path (Feature 1)
# These tests use n_spatial > 200 to trigger the sparse Cholesky backend.
# Results are compared against a small-grid dense-path baseline to verify
# numerical agreement of the shared Newton loop.

# Helper: create a regular grid adjacency structure (4-connected)
make_grid_adjacency <- function(nrow, ncol) {
  n <- nrow * ncol
  neighbors <- vector("list", n)
  for (i in seq_len(n)) {
    r <- (i - 1) %/% ncol + 1
    c <- (i - 1) %% ncol + 1
    nb <- integer(0)
    if (r > 1)    nb <- c(nb, i - ncol)
    if (r < nrow) nb <- c(nb, i + ncol)
    if (c > 1)    nb <- c(nb, i - 1)
    if (c < ncol) nb <- c(nb, i + 1)
    neighbors[[i]] <- nb
  }

  # Convert to CSR (row_ptr, col_idx)
  n_neighbors <- vapply(neighbors, length, integer(1))
  col_idx <- unlist(neighbors) - 1L  # 0-based
  row_ptr <- c(0L, cumsum(n_neighbors))

  list(
    n = n,
    adj_row_ptr = as.integer(row_ptr),
    adj_col_idx = as.integer(col_idx),
    n_neighbors = as.integer(n_neighbors)
  )
}

# Helper: simulate binomial spatial data on a grid
simulate_spatial_data <- function(n_sites, n_obs_per_site, beta0, tau_spatial, adj) {
  set.seed(42)
  n_obs <- n_sites * n_obs_per_site

  # Simulate ICAR spatial effects (approximate: draw from N(0, 1/tau) then smooth)
  spatial_raw <- rnorm(n_sites, 0, 1 / sqrt(tau_spatial))

  # Simple spatial smoothing via neighbor averaging (2 passes)
  for (pass in 1:2) {
    smoothed <- numeric(n_sites)
    for (s in seq_len(n_sites)) {
      start <- adj$adj_row_ptr[s] + 1L
      end <- adj$adj_row_ptr[s + 1]
      if (end >= start) {
        nb_idx <- adj$adj_col_idx[start:end] + 1L  # back to 1-based
        smoothed[s] <- mean(spatial_raw[nb_idx])
      }
    }
    spatial_raw <- smoothed
  }
  spatial_raw <- spatial_raw - mean(spatial_raw)

  # Design: intercept only
  X <- matrix(1, nrow = n_obs, ncol = 1)

  # Site assignments (1-based)
  spatial_idx <- rep(seq_len(n_sites), each = n_obs_per_site)

  # Linear predictor
  eta <- beta0 + spatial_raw[spatial_idx]
  p <- plogis(eta)

  # Binomial response
  n_trials <- rep(1L, n_obs)
  y <- rbinom(n_obs, size = n_trials, prob = p)

  list(
    y = as.integer(y),
    n_trials = as.integer(n_trials),
    X = X,
    spatial_idx = as.integer(spatial_idx),
    eta_true = eta,
    spatial_true = spatial_raw
  )
}


# =====================================================================
# Test: ICAR spatial with n=300 sites (triggers sparse solver)
# =====================================================================

test_that("ICAR Laplace works with sparse Cholesky (300 sites)", {
  adj <- make_grid_adjacency(15, 20)  # 300 sites
  expect_equal(adj$n, 300L)

  dat <- simulate_spatial_data(
    n_sites = 300, n_obs_per_site = 3,
    beta0 = -0.5, tau_spatial = 2.0, adj = adj
  )

  result <- cpp_laplace_fit_spatial(
    y = dat$y,
    n = dat$n_trials,
    X = dat$X,
    re_idx = rep(0, length(dat$y)),
    n_re_groups = 0L,
    sigma_re = 1.0,
    spatial_idx = dat$spatial_idx,
    n_spatial_units = 300L,
    adj_row_ptr = adj$adj_row_ptr,
    adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    tau_spatial = 2.0,
    family = "binomial",
    phi = 1.0,
    max_iter = 100L,
    tol = 1e-6,
    n_threads = 1L
  )

  expect_true(result$converged)
  expect_true(result$n_iter > 0)
  expect_true(result$n_iter < 100)
  expect_true(is.finite(result$log_det_Q))
  expect_true(is.finite(result$log_marginal))
  expect_equal(length(result$mode), 1L + 300L)  # 1 beta + 300 spatial

  # Mode should be reasonable: intercept near -0.5, spatial effects centered
  beta_hat <- result$mode[1]
  spatial_hat <- result$mode[2:301]
  expect_true(abs(beta_hat - (-0.5)) < 1.5)  # rough check
  expect_true(abs(mean(spatial_hat)) < 0.1)   # centered
})


# =====================================================================
# Test: BYM2 spatial with n=225 sites (triggers sparse solver)
# =====================================================================

test_that("BYM2 Laplace works with sparse Cholesky (225 sites)", {
  adj <- make_grid_adjacency(15, 15)  # 225 sites
  expect_equal(adj$n, 225L)

  dat <- simulate_spatial_data(
    n_sites = 225, n_obs_per_site = 4,
    beta0 = 0.0, tau_spatial = 3.0, adj = adj
  )

  result <- cpp_laplace_fit_bym2(
    y = dat$y,
    n = dat$n_trials,
    X = dat$X,
    re_idx = rep(0, length(dat$y)),
    n_re_groups = 0L,
    sigma_re = 1.0,
    spatial_idx = dat$spatial_idx,
    n_spatial_units = 225L,
    adj_row_ptr = adj$adj_row_ptr,
    adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    sigma_spatial = 0.5,
    rho = 0.7,
    scale_factor = 1.0,
    family = "binomial",
    phi = 1.0,
    max_iter = 100L,
    tol = 1e-6,
    n_threads = 1L
  )

  expect_true(result$converged)
  expect_true(result$n_iter > 0)
  expect_true(result$n_iter < 100)
  expect_true(is.finite(result$log_det_Q))
  expect_true(is.finite(result$log_marginal))
  # BYM2: 1 beta + 225 phi_scaled + 225 theta = 451
  expect_equal(length(result$mode), 1L + 2L * 225L)
})


# =====================================================================
# Test: Dense vs sparse numerical agreement
# =====================================================================
# Run the same ICAR problem at two sizes: one below threshold (dense)
# and one above (sparse). Both should produce consistent results
# (not bitwise identical due to different algorithms, but same structure).

test_that("sparse and dense paths produce consistent ICAR results", {
  # Small problem (dense path, n_x = 1 + 25 = 26)
  adj_small <- make_grid_adjacency(5, 5)
  dat_small <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = -1.0, tau_spatial = 2.0, adj = adj_small
  )

  res_small <- cpp_laplace_fit_spatial(
    y = dat_small$y, n = dat_small$n_trials, X = dat_small$X,
    re_idx = rep(0, length(dat_small$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat_small$spatial_idx, n_spatial_units = 25L,
    adj_row_ptr = adj_small$adj_row_ptr, adj_col_idx = adj_small$adj_col_idx,
    n_neighbors = adj_small$n_neighbors, tau_spatial = 2.0,
    family = "binomial", phi = 1.0, max_iter = 100L, tol = 1e-6, n_threads = 1L
  )

  # Large problem (sparse path, n_x = 1 + 400 = 401)
  adj_large <- make_grid_adjacency(20, 20)
  dat_large <- simulate_spatial_data(
    n_sites = 400, n_obs_per_site = 10,
    beta0 = -1.0, tau_spatial = 2.0, adj = adj_large
  )

  res_large <- cpp_laplace_fit_spatial(
    y = dat_large$y, n = dat_large$n_trials, X = dat_large$X,
    re_idx = rep(0, length(dat_large$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat_large$spatial_idx, n_spatial_units = 400L,
    adj_row_ptr = adj_large$adj_row_ptr, adj_col_idx = adj_large$adj_col_idx,
    n_neighbors = adj_large$n_neighbors, tau_spatial = 2.0,
    family = "binomial", phi = 1.0, max_iter = 100L, tol = 1e-6, n_threads = 1L
  )

  # Both should converge

  expect_true(res_small$converged)
  expect_true(res_large$converged)

  # Both should have finite, negative log-marginals
  expect_true(is.finite(res_small$log_marginal))
  expect_true(is.finite(res_large$log_marginal))
  expect_true(res_small$log_marginal < 0)
  expect_true(res_large$log_marginal < 0)

  # Log-det should be positive (positive definite Hessian)
  expect_true(res_small$log_det_Q > 0)
  expect_true(res_large$log_det_Q > 0)

  # Intercept estimates should be in the right ballpark
  expect_true(abs(res_small$mode[1] - (-1.0)) < 1.5)
  expect_true(abs(res_large$mode[1] - (-1.0)) < 1.5)
})


# =====================================================================
# Test: Poisson ICAR at large scale
# =====================================================================

test_that("Poisson ICAR works with sparse Cholesky (400 sites)", {
  adj <- make_grid_adjacency(20, 20)  # 400 sites
  set.seed(123)
  n_sites <- 400L
  n_obs <- n_sites * 5L

  X <- matrix(1, nrow = n_obs, ncol = 1)
  spatial_idx <- as.integer(rep(seq_len(n_sites), each = 5L))

  # Simulate Poisson counts
  eta <- 1.5 + rnorm(n_sites, 0, 0.3)[spatial_idx]
  y <- as.integer(rpois(n_obs, lambda = exp(eta)))
  n_trials <- rep(1L, n_obs)  # unused for Poisson but required by API

  result <- cpp_laplace_fit_spatial(
    y = y, n = n_trials, X = X,
    re_idx = rep(0, n_obs), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = spatial_idx, n_spatial_units = n_sites,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors, tau_spatial = 5.0,
    family = "poisson", phi = 1.0, max_iter = 100L, tol = 1e-6, n_threads = 1L
  )

  expect_true(result$converged)
  expect_true(is.finite(result$log_det_Q))
  expect_true(is.finite(result$log_marginal))
  expect_equal(length(result$mode), 1L + 400L)
})


# =====================================================================
# Test: byte-level dense == sparse on IDENTICAL data (single-response path)
# =====================================================================
# The dense and sparse factorizations solve the same Newton system, so on the
# same problem they must agree to Cholesky quantization. The size threshold
# picks one path per size, so the two must be forced to run on one data set:
# force_sparse = -1 forces dense, +1 forces sparse. This is the single-response
# analogue of the joint path's force_sparse equivalence gate
# (test-nested-laplace-joint-sparse-equivalence.R); it is the check the #57
# column-major footgun history most wants on this path.

test_that("ICAR single-response: dense and sparse agree byte-level on identical data", {
  adj <- make_grid_adjacency(15, 15)  # 225 sites -> n_x = 226 (auto-sparse)
  dat <- simulate_spatial_data(
    n_sites = 225, n_obs_per_site = 4,
    beta0 = -0.3, tau_spatial = 2.5, adj = adj
  )

  fit_one <- function(fs) cpp_laplace_fit_spatial(
    y = dat$y, n = dat$n_trials, X = dat$X,
    re_idx = rep(0, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat$spatial_idx, n_spatial_units = 225L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors, tau_spatial = 2.5,
    family = "binomial", phi = 1.0, max_iter = 100L, tol = 1e-8,
    n_threads = 1L, force_sparse = fs
  )
  dense  <- fit_one(-1L)
  sparse <- fit_one(1L)

  expect_true(dense$converged)
  expect_true(sparse$converged)
  expect_equal(sparse$log_marginal, dense$log_marginal, tolerance = 1e-8)
  expect_equal(sparse$log_det_Q,    dense$log_det_Q,    tolerance = 1e-8)
  expect_equal(sparse$mode,         dense$mode,         tolerance = 1e-7)
})

test_that("BYM2 single-response: dense and sparse agree byte-level on identical data", {
  adj <- make_grid_adjacency(15, 15)  # 225 sites -> n_x = 451 (auto-sparse)
  dat <- simulate_spatial_data(
    n_sites = 225, n_obs_per_site = 4,
    beta0 = 0.1, tau_spatial = 3.0, adj = adj
  )

  fit_one <- function(fs) cpp_laplace_fit_bym2(
    y = dat$y, n = dat$n_trials, X = dat$X,
    re_idx = rep(0, length(dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = dat$spatial_idx, n_spatial_units = 225L,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    sigma_spatial = 0.5, rho = 0.7, scale_factor = 1.0,
    family = "binomial", phi = 1.0, max_iter = 100L, tol = 1e-8,
    n_threads = 1L, force_sparse = fs
  )
  dense  <- fit_one(-1L)
  sparse <- fit_one(1L)

  expect_true(dense$converged)
  expect_true(sparse$converged)
  expect_equal(sparse$log_marginal, dense$log_marginal, tolerance = 1e-8)
  expect_equal(sparse$log_det_Q,    dense$log_det_Q,    tolerance = 1e-8)
  expect_equal(sparse$mode,         dense$mode,         tolerance = 1e-7)
})
