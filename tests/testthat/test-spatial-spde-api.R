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

test_that("compute_spde_prior: joint mode returns -0.5 * sum(z^2) and leaves spde_w empty", {
  set.seed(101)
  z <- rnorm(7)
  res <- tulpa:::cpp_spde_prior_probe(
    vals = z, joint_hypers = TRUE,
    C0_diag = numeric(0), G1_x = numeric(0),
    G1_i = integer(0), G1_p = integer(0)
  )
  expect_equal(res$prior_val, -0.5 * sum(z^2), tolerance = 1e-12)
  expect_false(res$spde_w_filled)             # NC transform owns w in this mode
})

test_that("compute_spde_prior: joint mode is invariant to (kappa, tau_spde)", {
  # The unit-Gaussian-on-z prior must NOT depend on hyperparameters; the
  # change-of-variable Jacobian is absorbed into the NC adjoint downstream.
  set.seed(102)
  z <- rnorm(10)
  res_a <- tulpa:::cpp_spde_prior_probe(
    vals = z, joint_hypers = TRUE,
    C0_diag = numeric(0), G1_x = numeric(0),
    G1_i = integer(0), G1_p = integer(0),
    kappa = 1.0, tau_spde = 1.0
  )
  res_b <- tulpa:::cpp_spde_prior_probe(
    vals = z, joint_hypers = TRUE,
    C0_diag = numeric(0), G1_x = numeric(0),
    G1_i = integer(0), G1_p = integer(0),
    kappa = 5.7, tau_spde = 0.13
  )
  expect_equal(res_a$prior_val, res_b$prior_val, tolerance = 1e-12)
})

test_that("compute_spde_prior: legacy mode is quadratic in w and fills spde_w", {
  skip_if_not_installed("fmesher")
  set.seed(103)
  coords <- cbind(runif(40), runif(40))
  spec <- spatial_spde(coords)

  probe <- function(w) tulpa:::cpp_spde_prior_probe(
    vals = w, joint_hypers = FALSE,
    C0_diag = spec$C0_diag, G1_x = spec$G1_x,
    G1_i = spec$G1_i,       G1_p = spec$G1_p,
    kappa = 2.0, tau_spde = 1.5, alpha = 2L
  )

  zero <- probe(rep(0.0, spec$n_mesh))
  expect_equal(zero$prior_val, 0.0, tolerance = 1e-12)
  expect_true(zero$spde_w_filled)

  set.seed(104)
  w <- rnorm(spec$n_mesh, sd = 0.3)
  res <- probe(w)
  expect_true(is.finite(res$prior_val))
  expect_true(res$prior_val < 0)               # -0.5 w' Q w with Q SPD

  # Quadratic homogeneity: prior(c * w) = c^2 * prior(w).
  c <- 1.7
  res_c <- probe(c * w)
  expect_equal(res_c$prior_val, c^2 * res$prior_val, tolerance = 1e-9)
})

test_that("NC transform: z=0 yields w=0 and lazily builds the cache", {
  skip_if_not_installed("fmesher")
  set.seed(201)
  coords <- cbind(runif(40), runif(40))
  spec <- spatial_spde(coords)

  z0 <- rep(0.0, spec$n_mesh)
  res <- tulpa:::cpp_spde_nc_apply_probe(
    z = z0, log_kappa = log(2.0), log_tau = log(1.5),
    C0_diag = spec$C0_diag, G1_x = spec$G1_x,
    G1_i = spec$G1_i, G1_p = spec$G1_p
  )

  expect_equal(length(res$w), spec$n_mesh)
  expect_true(res$nc_transform_built)
  expect_equal(max(abs(res$w)), 0.0, tolerance = 1e-12)
})

test_that("NC transform: z^T z == w^T Q(theta) w (consistency with legacy prior)", {
  # The NC transform sets w = L^{-T}(theta) z so L^T w = z, hence
  # z^T z = w^T L L^T w = w^T Q(theta) w. compute_spde_prior in joint
  # mode returns -0.5 z^T z, and in legacy mode (with Q built at the
  # same kappa, tau) returns -0.5 w^T Q w; the two must match.
  skip_if_not_installed("fmesher")
  set.seed(202)
  coords <- cbind(runif(45), runif(45))
  spec <- spatial_spde(coords)

  z <- rnorm(spec$n_mesh, sd = 0.7)
  log_kappa <- log(2.3)
  log_tau   <- log(1.4)
  kappa <- exp(log_kappa)
  tau   <- exp(log_tau)

  w_res <- tulpa:::cpp_spde_nc_apply_probe(
    z = z, log_kappa = log_kappa, log_tau = log_tau,
    C0_diag = spec$C0_diag, G1_x = spec$G1_x,
    G1_i = spec$G1_i, G1_p = spec$G1_p
  )
  w <- w_res$w

  joint_prior <- tulpa:::cpp_spde_prior_probe(
    vals = z, joint_hypers = TRUE,
    C0_diag = numeric(0), G1_x = numeric(0),
    G1_i = integer(0), G1_p = integer(0)
  )$prior_val

  legacy_prior <- tulpa:::cpp_spde_prior_probe(
    vals = w, joint_hypers = FALSE,
    C0_diag = spec$C0_diag, G1_x = spec$G1_x,
    G1_i = spec$G1_i, G1_p = spec$G1_p,
    kappa = kappa, tau_spde = tau, alpha = 2L
  )$prior_val

  expect_equal(joint_prior, legacy_prior, tolerance = 1e-7)
})

test_that("NC transform: w is linear in z", {
  # If w = L^{-T}(theta) z then w(c z) = c w(z) at the same theta.
  skip_if_not_installed("fmesher")
  set.seed(203)
  coords <- cbind(runif(35), runif(35))
  spec <- spatial_spde(coords)

  z <- rnorm(spec$n_mesh, sd = 0.5)
  log_kappa <- log(1.7)
  log_tau   <- log(0.9)
  c_scale   <- -2.4

  w_z <- tulpa:::cpp_spde_nc_apply_probe(
    z = z, log_kappa = log_kappa, log_tau = log_tau,
    C0_diag = spec$C0_diag, G1_x = spec$G1_x,
    G1_i = spec$G1_i, G1_p = spec$G1_p
  )$w
  w_cz <- tulpa:::cpp_spde_nc_apply_probe(
    z = c_scale * z, log_kappa = log_kappa, log_tau = log_tau,
    C0_diag = spec$C0_diag, G1_x = spec$G1_x,
    G1_i = spec$G1_i, G1_p = spec$G1_p
  )$w

  expect_equal(w_cz, c_scale * w_z, tolerance = 1e-9)
})

test_that("PC hyper prior: disabled when any anchor is non-positive or >= 1", {
  # Disabled means improper flat hyper-prior — compute_spde_hyper_prior
  # must return 0 verbatim so the joint log-post still converges in floor-
  # check runs and the gradient-verification harness.
  expect_equal(
    tulpa:::cpp_spde_hyper_prior_probe(
      log_kappa = log(2.0), log_tau = log(1.5)
    )$prior_val, 0.0, tolerance = 1e-12
  )
  # range anchor present, sigma anchor missing -> still disabled
  expect_equal(
    tulpa:::cpp_spde_hyper_prior_probe(
      log_kappa = 0.5, log_tau = -0.3,
      prior_range_0 = 0.3, prior_range_alpha = 0.05
    )$prior_val, 0.0, tolerance = 1e-12
  )
  # alpha == 1 is a knife-edge degeneracy: -log(1) = 0 -> lambda_r/lambda_s = 0
  # which kills the PC density.  The guard rejects alpha >= 1 explicitly.
  expect_equal(
    tulpa:::cpp_spde_hyper_prior_probe(
      log_kappa = 0.0, log_tau = 0.0,
      prior_range_0 = 0.3, prior_range_alpha = 1.0,
      prior_sigma_0 = 1.0, prior_sigma_alpha = 0.05
    )$prior_val, 0.0, tolerance = 1e-12
  )
})

test_that("PC hyper prior: matches Fuglstad et al. (2019) reference at multiple points", {
  # Hand-coded reference of the PC density in (log_kappa, log_tau).
  pc_ref <- function(log_kappa, log_tau, nu,
                     range_0, alpha_r, sigma_0, alpha_s) {
    range  <- sqrt(8 * nu) * exp(-log_kappa)
    sigma  <- exp(-log_kappa - log_tau) / sqrt(4 * pi)
    lambda_r <- -log(alpha_r) * sqrt(range_0)
    lambda_s <- -log(alpha_s) / sigma_0

    log_pi_range <- log(lambda_r / 2) - 1.5 * log(range) -
                    lambda_r / sqrt(range)
    log_pi_sigma <- log(lambda_s) - lambda_s * sigma

    log_pi_range + log_pi_sigma + log(range) + log(sigma)
  }

  grid <- expand.grid(
    log_kappa = c(-1.0, 0.0, 0.7, 1.5),
    log_tau   = c(-1.2, 0.0, 0.4)
  )
  for (i in seq_len(nrow(grid))) {
    res <- tulpa:::cpp_spde_hyper_prior_probe(
      log_kappa = grid$log_kappa[i], log_tau = grid$log_tau[i],
      nu = 1.0,
      prior_range_0 = 0.2, prior_range_alpha = 0.05,
      prior_sigma_0 = 1.0, prior_sigma_alpha = 0.05
    )
    expected <- pc_ref(grid$log_kappa[i], grid$log_tau[i], nu = 1.0,
                       range_0 = 0.2, alpha_r = 0.05,
                       sigma_0 = 1.0, alpha_s = 0.05)
    expect_equal(res$prior_val, expected, tolerance = 1e-10,
                 info = sprintf("log_kappa = %g, log_tau = %g",
                                grid$log_kappa[i], grid$log_tau[i]))
  }
})

test_that("PC hyper prior: anchor scaling moves density in the right direction", {
  # The PC prior on range uses 'large range' as the base model and
  # penalises deviation toward small range. The penalty strength is
  # lambda_r = -log(alpha_r) * sqrt(range_0). Smaller range_0 (with alpha
  # held fixed) yields smaller lambda_r, weaker penalty, and therefore
  # *higher* log-density at a large fixed range. Verify both signs.
  range_at <- log(2.0)   # range = sqrt(8)/2 = sqrt(2) ~ 1.41 -- a 'large' range
  tau_at   <- log(1.0)

  small_r0 <- tulpa:::cpp_spde_hyper_prior_probe(
    log_kappa = range_at, log_tau = tau_at,
    prior_range_0 = 0.1, prior_range_alpha = 0.05,
    prior_sigma_0 = 1.0, prior_sigma_alpha = 0.05
  )$prior_val
  large_r0 <- tulpa:::cpp_spde_hyper_prior_probe(
    log_kappa = range_at, log_tau = tau_at,
    prior_range_0 = 0.5, prior_range_alpha = 0.05,
    prior_sigma_0 = 1.0, prior_sigma_alpha = 0.05
  )$prior_val
  expect_gt(small_r0, large_r0)

  # PC prior on sigma uses sigma = 0 as the base model and penalises
  # deviation toward large sigma. lambda_s = -log(alpha_s) / sigma_0.
  # At fixed (log_kappa, log_tau) with sigma ~ 1 / (sqrt(4 pi) * 2 * 1)
  # ~ 0.14, smaller sigma_0 -> larger lambda_s -> stronger penalty ->
  # lower log-density.
  small_s0 <- tulpa:::cpp_spde_hyper_prior_probe(
    log_kappa = range_at, log_tau = tau_at,
    prior_range_0 = 0.3, prior_range_alpha = 0.05,
    prior_sigma_0 = 0.5, prior_sigma_alpha = 0.05
  )$prior_val
  large_s0 <- tulpa:::cpp_spde_hyper_prior_probe(
    log_kappa = range_at, log_tau = tau_at,
    prior_range_0 = 0.3, prior_range_alpha = 0.05,
    prior_sigma_0 = 2.0, prior_sigma_alpha = 0.05
  )$prior_val
  expect_gt(small_s0, large_s0)
})

test_that("PC hyper prior: dependence on log_kappa matches analytical decomposition", {
  # At fixed log_tau and fixed PC anchors, the part of log p that varies
  # with log_kappa is
  #     -0.5 * log_kappa
  #     - lambda_r * (8 nu)^{-1/4} * exp(0.5 log_kappa)
  #     - lambda_s * exp(-log_kappa - log_tau - 0.5 log(4 pi))
  # Sign check: log_pi_range contributes +1.5 log_kappa via
  # (-1.5 * log_range), log_range itself contributes -log_kappa, and
  # log_sigma contributes another -log_kappa, summing to -0.5 log_kappa.
  # log_pi_sigma contributes the lambda_s exp term (since sigma =
  # exp(log_sigma) and log_sigma is linear in -log_kappa). Compute the
  # difference between two log_kappa points analytically and compare.
  nu       <- 1.0
  range_0  <- 0.3; alpha_r <- 0.05
  sigma_0  <- 0.7; alpha_s <- 0.05
  log_tau  <- 0.2
  lambda_r <- -log(alpha_r) * sqrt(range_0)
  lambda_s <- -log(alpha_s) / sigma_0

  log_kappa_a <- 0.4
  log_kappa_b <- 1.1

  pa <- tulpa:::cpp_spde_hyper_prior_probe(
    log_kappa = log_kappa_a, log_tau = log_tau, nu = nu,
    prior_range_0 = range_0, prior_range_alpha = alpha_r,
    prior_sigma_0 = sigma_0, prior_sigma_alpha = alpha_s
  )$prior_val
  pb <- tulpa:::cpp_spde_hyper_prior_probe(
    log_kappa = log_kappa_b, log_tau = log_tau, nu = nu,
    prior_range_0 = range_0, prior_range_alpha = alpha_r,
    prior_sigma_0 = sigma_0, prior_sigma_alpha = alpha_s
  )$prior_val

  varies_in_lk <- function(lk) {
     -0.5 * lk -
     lambda_r * (8 * nu)^(-0.25) * exp(0.5 * lk) -
     lambda_s * exp(-lk - log_tau - 0.5 * log(4 * pi))
  }
  delta_expected <- varies_in_lk(log_kappa_b) - varies_in_lk(log_kappa_a)

  expect_equal(pb - pa, delta_expected, tolerance = 1e-10)
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
