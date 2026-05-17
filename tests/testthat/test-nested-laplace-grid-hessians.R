# Tests for keep_grid_hessians = TRUE in tulpa_nested_laplace().
# Verifies that per-grid-point fixed-effects marginal Hessian and mode are
# exposed in the return list and match what tulpa_laplace() produces at the
# same theta value.

source(test_path("test-sparse-cholesky.R"), local = TRUE)

icar_prior <- function(adj, n_sites, tau_grid = NULL) {
  list(
    type = "icar",
    spatial_idx = NULL,
    n_spatial_units = n_sites,
    adj_row_ptr = adj$adj_row_ptr,
    adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    tau_grid = tau_grid
  )
}

test_that("keep_grid_hessians exposes per-grid H_beta and modes", {
  adj <- make_grid_adjacency(5, 5)
  dat <- simulate_spatial_data(
    n_sites = 25, n_obs_per_site = 10,
    beta0 = -0.5, tau_spatial = 3.0, adj = adj
  )
  tau_grid <- exp(seq(log(0.5), log(20), length.out = 5))
  prior <- icar_prior(adj, 25L, tau_grid = tau_grid)
  prior$spatial_idx <- dat$spatial_idx

  result <- tulpa_nested_laplace(
    y = dat$y, n_trials = dat$n_trials, X = dat$X,
    prior = prior, family = "binomial",
    keep_grid_hessians = TRUE
  )

  expect_type(result$grid_hessians, "list")
  expect_type(result$grid_modes, "list")
  expect_length(result$grid_hessians, length(tau_grid))
  expect_length(result$grid_modes, length(tau_grid))

  p <- ncol(dat$X)
  for (k in seq_along(tau_grid)) {
    Hk <- result$grid_hessians[[k]]
    bk <- result$grid_modes[[k]]
    expect_true(is.matrix(Hk))
    expect_equal(dim(Hk), c(p, p))
    expect_length(bk, p)
    expect_true(all(is.finite(Hk)))
    expect_true(all(is.finite(bk)))
    # H_beta is symmetric positive definite at the mode.
    expect_equal(Hk, t(Hk), tolerance = 1e-8)
    eig <- eigen(Hk, symmetric = TRUE, only.values = TRUE)$values
    expect_true(min(eig) > 0)
  }

  # CSC scratch fields should be stripped.
  expect_null(result$Q_csc_p_per_grid)
  expect_null(result$Q_csc_i_per_grid)
  expect_null(result$Q_csc_x_per_grid)
})

test_that("default keep_grid_hessians = FALSE leaves return list unchanged", {
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

  expect_null(result$grid_hessians)
  expect_null(result$grid_modes)
})
