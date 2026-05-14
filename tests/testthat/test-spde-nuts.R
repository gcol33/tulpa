# test-spde-nuts.R
# Tests for the SPDE-NUTS sampler (cpp_tulpa_fit_spde_nuts /
# tulpa_nuts_spde). Uses fmesher for mesh generation and FEM
# matrices, matching test-spde.R, then compares against the SPDE
# Laplace mode for the same fixed (range, sigma).

helper_make_spde_spec <- function(coords, max_edge = c(0.15, 0.5),
                                  cutoff = 0.05, nu = 1) {
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = max_edge,
                              cutoff = cutoff)
  fem  <- fmesher::fm_fem(mesh)
  A    <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  spatial_spde_custom(
    C = fem$c0, G = fem$g1, A = A, nu = nu,
    prior_range = c(0.5, 0.5), prior_sigma = c(1, 0.5)
  )
}

test_that("tulpa_nuts_spde rejects non-SPDE spatial spec", {
  expect_error(
    tulpa_nuts_spde(y = 1:5, X = cbind(1, 1:5),
                    spatial = list(type = "bym2"),
                    n_iter = 50L, n_warmup = 25L),
    "SPDE tulpa_spatial"
  )
})

test_that("tulpa_nuts_spde recovers gaussian SPDE on small mesh", {
  skip_if_not_installed("fmesher")

  set.seed(2026)
  n_obs <- 200
  coords <- cbind(runif(n_obs), runif(n_obs))

  spec <- helper_make_spde_spec(coords, max_edge = c(0.2, 0.5), cutoff = 0.08)
  expect_true(spec$n_mesh < 200,
              info = paste("mesh size for the test:", spec$n_mesh))

  # Simulate a small spatial field on the mesh; gaussian observation noise.
  beta0      <- 0.5
  beta1      <- -0.8
  sigma_obs  <- 0.3
  range_true <- 0.4
  sigma_w    <- 0.5

  x_cov <- runif(n_obs, -1, 1)
  X     <- cbind(1, x_cov)

  w_true <- rnorm(spec$n_mesh, 0, 0.4)
  w_true <- w_true - mean(w_true)
  eta    <- beta0 + beta1 * x_cov + as.numeric(spec$A %*% w_true)
  y      <- eta + rnorm(n_obs, 0, sigma_obs)

  fit <- tulpa_nuts_spde(
    y = y, X = X, spatial = spec,
    family = "gaussian",
    range = range_true, sigma = sigma_w,
    log_phi_init = log(sigma_obs),
    n_iter = 800L, n_warmup = 400L, seed = 2026L
  )

  beta_post <- colMeans(fit$draws[, c("beta[1]", "beta[2]"), drop = FALSE])
  sigma_post <- fit$phi_summary[["mean"]]

  expect_lt(abs(beta_post[1] - beta0), 0.25)
  expect_lt(abs(beta_post[2] - beta1), 0.20)
  expect_lt(abs(sigma_post - sigma_obs) / sigma_obs, 0.30)
  expect_true(mean(fit$accept_prob) > 0.4)
  expect_true(sum(fit$divergent) < 0.10 * length(fit$divergent))
})

test_that("tulpa_nuts_spde poisson recovers intercept", {
  skip_if_not_installed("fmesher")

  set.seed(7)
  n_obs <- 150
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec <- helper_make_spde_spec(coords, max_edge = c(0.25, 0.6), cutoff = 0.10)

  beta0  <- 1.0
  w_true <- rnorm(spec$n_mesh, 0, 0.25)
  w_true <- w_true - mean(w_true)
  eta    <- beta0 + as.numeric(spec$A %*% w_true)
  y      <- rpois(n_obs, lambda = exp(eta))

  X <- matrix(1.0, nrow = n_obs, ncol = 1)

  fit <- tulpa_nuts_spde(
    y = y, X = X, spatial = spec,
    family = "poisson",
    range = 0.4, sigma = 0.4,
    n_iter = 600L, n_warmup = 300L, seed = 7L
  )

  beta_post <- mean(fit$draws[, "beta[1]"])
  expect_lt(abs(beta_post - beta0), 0.40)
  expect_true(mean(fit$accept_prob) > 0.4)
})

test_that("tulpa_nuts_spde gaussian beta posterior matches Laplace mode", {
  skip_if_not_installed("fmesher")

  set.seed(3)
  n_obs <- 250
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec <- helper_make_spde_spec(coords, max_edge = c(0.20, 0.5), cutoff = 0.08)

  beta0      <- 0.2
  beta1      <- 0.6
  sigma_obs  <- 0.4
  range_true <- 0.35
  sigma_w    <- 0.45

  x_cov <- runif(n_obs, -2, 2)
  X     <- cbind(1, x_cov)

  w_true <- rnorm(spec$n_mesh, 0, 0.3)
  w_true <- w_true - mean(w_true)
  eta    <- beta0 + beta1 * x_cov + as.numeric(spec$A %*% w_true)
  y      <- eta + rnorm(n_obs, 0, sigma_obs)

  # NUTS posterior at the same fixed hypers
  fit_n <- tulpa_nuts_spde(
    y = y, X = X, spatial = spec,
    family = "gaussian",
    range = range_true, sigma = sigma_w,
    log_phi_init = log(sigma_obs),
    n_iter = 800L, n_warmup = 400L, seed = 3L
  )
  beta_n <- colMeans(fit_n$draws[, c("beta[1]", "beta[2]"), drop = FALSE])

  # Gaussian Laplace fit at the same hypers (note: fit_spde supports
  # binomial/poisson/negbin in its docstring, but the underlying
  # cpp_laplace_fit_spde + family-link path also handles gaussian — phi
  # is the residual variance).
  kappa    <- sqrt(8 * spec$nu) / range_true
  tau_spde <- 1.0 / (sqrt(4 * pi) * kappa * sigma_w)
  res_l <- cpp_laplace_fit_spde(
    y = y, n_trials = rep(1L, n_obs), X = X,
    A_x = spec$A_x, A_i = spec$A_i, A_p = spec$A_p,
    n_obs = n_obs, n_mesh = spec$n_mesh,
    C0_diag = spec$C0_diag,
    G1_x = spec$G1_x, G1_i = spec$G1_i, G1_p = spec$G1_p,
    kappa = kappa, tau_spde = tau_spde,
    family = "gaussian", phi = sigma_obs * sigma_obs,
    alpha = 2L, max_iter = 100L, tol = 1e-6, n_threads = 1L
  )
  beta_l <- res_l$mode[1:2]

  # Posteriors and modes should align well (Gaussian likelihood is
  # log-concave, mode and mean coincide for the conditional latent).
  expect_lt(abs(beta_n[1] - beta_l[1]), 0.10)
  expect_lt(abs(beta_n[2] - beta_l[2]), 0.10)
})
