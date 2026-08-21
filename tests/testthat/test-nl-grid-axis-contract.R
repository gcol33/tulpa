# The outer-grid axis contract shared by every kernel in nested_laplace.cpp
# (gcol33/tulpa#421).
#
# Each kernel takes one NumericVector per hyperparameter axis, all indexed by
# the same cell number, and hands them to block factories that index them
# unchecked. cpp_nested_laplace_bym2 was the one entry that never compared its
# second axis against its first: a short rho_grid was read past the end of the
# R vector for every cell beyond its length, and a rho outside [0, 1] took
# sqrt(1 - rho) to NaN, which reached the inner Newton through eta and returned
# a NaN cell rather than an error.

source(test_path("test-sparse-cholesky.R"), local = TRUE)

.axis_fixture <- function(n_side = 4L, seed = 7L) {
  adj <- make_grid_adjacency(n_side, n_side)
  set.seed(seed)
  dat <- simulate_spatial_data(
    n_sites = n_side * n_side, n_obs_per_site = 4,
    beta0 = 0.0, tau_spatial = 2.0, adj = adj
  )
  list(adj = adj, dat = dat, n_units = as.integer(n_side * n_side))
}

.fit_bym2 <- function(fx, sigma_grid, rho_grid) {
  tulpa:::cpp_nested_laplace_bym2(
    y = fx$dat$y, n = fx$dat$n_trials, X = fx$dat$X,
    re_idx = rep(0L, length(fx$dat$y)), n_re_groups = 0L, sigma_re = 1.0,
    spatial_idx = fx$dat$spatial_idx, n_spatial_units = fx$n_units,
    adj_row_ptr = fx$adj$adj_row_ptr, adj_col_idx = fx$adj$adj_col_idx,
    n_neighbors = fx$adj$n_neighbors, scale_factor = 1.0,
    sigma_spatial_grid = sigma_grid, rho_grid = rho_grid,
    family = "binomial", phi = 1.0,
    max_iter = 50L, tol = 1e-6, n_threads = 1L
  )
}

test_that("BYM2 rejects a rho grid that does not pair with the sigma grid", {
  fx <- .axis_fixture()
  expect_error(
    .fit_bym2(fx, c(0.4, 0.7, 1.0), 0.5),
    "rho_grid.*length 1.*sigma_spatial_grid.*length 3"
  )
  expect_error(
    .fit_bym2(fx, 0.7, c(0.2, 0.5)),
    "rho_grid.*length 2.*sigma_spatial_grid.*length 1"
  )
})

test_that("BYM2 rejects a mixing weight outside [0, 1]", {
  fx <- .axis_fixture()
  for (bad in c(-1e-6, 1.0000001, 2, NA_real_, NaN, Inf)) {
    expect_error(.fit_bym2(fx, 0.7, bad), "must lie in [[]0, 1[]]",
                 info = paste("rho =", bad))
  }
  # The offending cell is named, not just the axis.
  expect_error(.fit_bym2(fx, c(0.7, 0.7, 0.7), c(0.2, 1.5, 0.5)),
               "rho_grid[[]2[]]")
})

test_that("both mixing-weight endpoints stay in domain and finite", {
  fx <- .axis_fixture()
  res <- .fit_bym2(fx, c(0.7, 0.7), c(0, 1))
  expect_equal(res$n_grid, 2L)
  expect_true(all(is.finite(res$log_marginal)))
})

test_that("the paired-length rule is one rule across the grid kernels", {
  fx <- .axis_fixture()
  # Same message shape from a sibling entry, which is what says the check is
  # stated once rather than restated per kernel.
  expect_error(
    tulpa:::cpp_nested_laplace_car_proper(
      y = fx$dat$y, n = fx$dat$n_trials, X = fx$dat$X,
      re_idx = rep(0L, length(fx$dat$y)), n_re_groups = 0L, sigma_re = 1.0,
      spatial_idx = fx$dat$spatial_idx, n_spatial_units = fx$n_units,
      adj_row_ptr = fx$adj$adj_row_ptr, adj_col_idx = fx$adj$adj_col_idx,
      n_neighbors = fx$adj$n_neighbors,
      tau_grid = c(1, 2, 3), rho_grid = 0.5,
      family = "binomial", phi = 1.0,
      max_iter = 20L, tol = 1e-6, n_threads = 1L
    ),
    "rho_grid.*length 1.*tau_grid.*length 3"
  )
})
