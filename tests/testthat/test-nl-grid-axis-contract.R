# The outer-grid axis contract of a registry block's paired grids
# (gcol33/tulpa#421).
#
# A block's per-axis grids are indexed by the same cell number and handed to
# block factories that index them unchecked. The BYM2 entry once never compared
# its second axis against its first: a short rho_grid was read past the end of the
# R vector for every cell beyond its length, and a rho outside [0, 1] took
# sqrt(1 - rho) to NaN, which reached the inner Newton through eta and returned
# a NaN cell rather than an error.

.axis_fixture <- function(n_side = 4L, seed = 7L) {
  adj <- make_grid_adjacency(n_side, n_side)
  set.seed(seed)
  dat <- simulate_spatial_data(
    n_sites = n_side * n_side, n_obs_per_site = 4,
    beta0 = 0.0, tau_spatial = 2.0, adj = adj
  )
  list(adj = adj, dat = dat, n_units = as.integer(n_side * n_side))
}

.axis_ctl <- list(auto_recenter = FALSE, axis_refine = "none", progress = FALSE,
                  diagnose_k = FALSE, diagnose_skew = FALSE)

.areal_prior <- function(fx, type, ...) {
  list(type = type, spatial_idx = fx$dat$spatial_idx,
       n_spatial_units = fx$n_units, adj_row_ptr = fx$adj$adj_row_ptr,
       adj_col_idx = fx$adj$adj_col_idx, n_neighbors = fx$adj$n_neighbors, ...)
}

.fit_bym2 <- function(fx, sigma_grid, rho_grid) {
  tulpa_nested_laplace(fx$dat$y, fx$dat$n_trials, fx$dat$X,
    prior = .areal_prior(fx, "bym2", scale_factor = 1.0,
                         sigma_grid = sigma_grid, rho_grid = rho_grid),
    family = "binomial", control = .axis_ctl)
}

test_that("BYM2 rejects a rho grid that does not pair with the sigma grid", {
  fx <- .axis_fixture()
  expect_error(.fit_bym2(fx, c(0.4, 0.7, 1.0), c(0.2, 0.5)), "PAIRED.*3 / 2")
  # A length-1 axis is a value held fixed across every cell.
  res <- .fit_bym2(fx, c(0.4, 0.7, 1.0), 0.5)
  expect_equal(as.numeric(res$theta_grid[, "b1.rho"]), rep(0.5, 3))
})

test_that("the paired-length rule is one rule across the grid kernels", {
  fx <- .axis_fixture()
  # Same message shape from a sibling family, which is what says the check is
  # stated once rather than restated per family.
  expect_error(
    tulpa_nested_laplace(fx$dat$y, fx$dat$n_trials, fx$dat$X,
      prior = .areal_prior(fx, "car_proper", tau_grid = c(1, 2, 3),
                           rho_grid = c(0.3, 0.5)),
      family = "binomial", control = .axis_ctl),
    "PAIRED.*3 / 2"
  )
})

skip_on_cran_budget()

test_that("BYM2 rejects a mixing weight outside [0, 1]", {
  skip_on_cran_budget()
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
  skip_on_cran_budget()
  fx <- .axis_fixture()
  res <- .fit_bym2(fx, c(0.7, 0.7), c(0, 1))
  expect_equal(nrow(res$theta_grid), 2L)
  expect_cells_solved(res)
})
