# test-spatial-spde-api.R
# Tests for spatial_spde() R API

test_that("spatial_spde creates valid spec from coordinate matrix", {
  set.seed(42)
  coords <- cbind(runif(50), runif(50))
  spec <- spatial_spde(coords)

  expect_s3_class(spec, "tulpa_spatial")
  expect_equal(spec$type, "spde")
  expect_true(spec$n_mesh >= 50)
  expect_equal(spec$nu, 1)
  expect_true(length(spec$A_x) > 0)
  expect_true(length(spec$G1_x) > 0)
  expect_true(length(spec$C0_diag) == spec$n_mesh)
})

test_that("spatial_spde creates valid spec from formula", {
  set.seed(42)
  df <- data.frame(lon = runif(30), lat = runif(30), y = rbinom(30, 1, 0.5))
  spec <- spatial_spde(~ lon + lat, data = df)

  expect_s3_class(spec, "tulpa_spatial")
  expect_equal(spec$type, "spde")
  expect_equal(nrow(spec$obs_coords), 30)
})

test_that("spatial_spde_custom works with fmesher matrices", {
  skip_if_not_installed("fmesher")

  set.seed(42)
  coords <- cbind(runif(50), runif(50))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.2, 0.5), cutoff = 0.05)
  fem <- fmesher::fm_fem(mesh)
  A <- fmesher::fm_basis(mesh, loc = coords)

  spec <- spatial_spde_custom(C = fem$c0, G = fem$g1, A = A)

  expect_s3_class(spec, "tulpa_spatial")
  expect_equal(spec$type, "spde")
  expect_equal(spec$n_mesh, mesh$n)
})

test_that("print.tulpa_spatial works for SPDE", {
  set.seed(42)
  spec <- spatial_spde(cbind(runif(30), runif(30)))
  expect_output(print(spec), "SPDE")
  expect_output(print(spec), "Mesh nodes")
})

test_that("fit_spde works with fixed hyperparameters", {
  set.seed(42)
  n_obs <- 100
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec <- spatial_spde(coords)

  y <- rbinom(n_obs, 1, 0.4)
  X <- matrix(1, nrow = n_obs, ncol = 1)

  result <- fit_spde(y, X, spec,
                     family = "binomial", n_trials = rep(1L, n_obs),
                     range = 0.3, sigma = 0.5)

  expect_true(result$converged)
  expect_true(is.finite(result$log_marginal))
  expect_equal(length(result$beta), 1)
  expect_equal(length(result$spatial_effects), spec$n_mesh)
})

test_that("ParamLayout: legacy fixed-hyper mode leaves hyper slots at -1", {
  layout <- tulpa:::cpp_spde_layout_probe(
    n_mesh = 50, p = 2, joint_hypers = FALSE, n_extra_params = 1L
  )

  expect_true(layout$is_spde)
  expect_equal(layout$spde_w_start, 2)            # after 2 beta slots
  expect_equal(layout$spde_w_end,   52)           # half-open: 2 + 50
  expect_equal(layout$log_kappa_spde_idx, -1)     # sentinel: hypers off
  expect_equal(layout$log_tau_spde_idx,   -1)
  expect_equal(layout$extra_offset, 52)
  expect_equal(layout$total_params, 53)           # 2 + 50 + 1
})

test_that("ParamLayout: joint-NUTS mode allocates hyper slots after z block", {
  layout <- tulpa:::cpp_spde_layout_probe(
    n_mesh = 50, p = 2, joint_hypers = TRUE, n_extra_params = 1L
  )

  expect_true(layout$is_spde)
  expect_equal(layout$spde_w_start, 2)
  expect_equal(layout$spde_w_end,   52)
  expect_true(layout$log_kappa_spde_idx >= 0)
  expect_true(layout$log_tau_spde_idx   >= 0)
  expect_equal(layout$log_kappa_spde_idx, layout$spde_w_end)            # hyper slots immediately after z
  expect_equal(layout$log_tau_spde_idx,   layout$log_kappa_spde_idx + 1)
  expect_equal(layout$extra_offset, layout$log_tau_spde_idx + 1)
  expect_equal(layout$total_params, 55)           # 2 + 50 + 2 + 1
})

test_that("ParamLayout: hyper slot ordering is mesh-size invariant", {
  for (n_mesh in c(10, 100, 500)) {
    layout <- tulpa:::cpp_spde_layout_probe(
      n_mesh = n_mesh, p = 1, joint_hypers = TRUE, n_extra_params = 0L
    )
    expect_equal(layout$spde_w_end - layout$spde_w_start, n_mesh,
                 info = paste("n_mesh =", n_mesh))
    expect_equal(layout$log_kappa_spde_idx, layout$spde_w_end,
                 info = paste("n_mesh =", n_mesh))
    expect_equal(layout$log_tau_spde_idx,   layout$log_kappa_spde_idx + 1,
                 info = paste("n_mesh =", n_mesh))
  }
})

test_that("fit_spde works with nested Laplace", {
  set.seed(42)
  n_obs <- 100
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec <- spatial_spde(coords)

  y <- rbinom(n_obs, 1, 0.4)
  X <- matrix(1, nrow = n_obs, ncol = 1)

  result <- fit_spde(y, X, spec,
                     family = "binomial", n_trials = rep(1L, n_obs),
                     method = "grid", n_grid = 3L)

  expect_true(!is.null(result$nested))
  expect_true(result$nested$range_mean > 0)
  expect_true(result$nested$sigma_mean > 0)
  expect_true(all(is.finite(result$log_marginal)))
})
