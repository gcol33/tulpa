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

test_that("tulpa_nuts_spde gamma recovers intercept and shape", {
  skip_if_not_installed("fmesher")

  set.seed(11)
  n_obs <- 200
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec <- helper_make_spde_spec(coords, max_edge = c(0.25, 0.6), cutoff = 0.10)

  beta0      <- 0.6
  shape_true <- 4.0
  w_true     <- rnorm(spec$n_mesh, 0, 0.2)
  w_true     <- w_true - mean(w_true)
  eta        <- beta0 + as.numeric(spec$A %*% w_true)
  mu         <- exp(eta)
  y          <- rgamma(n_obs, shape = shape_true, rate = shape_true / mu)
  X          <- matrix(1.0, nrow = n_obs, ncol = 1)

  fit <- tulpa_nuts_spde(
    y = y, X = X, spatial = spec,
    family = "gamma",
    range = 0.4, sigma = 0.3,
    log_phi_init = log(shape_true),
    n_iter = 600L, n_warmup = 300L, seed = 11L
  )

  beta_post  <- mean(fit$draws[, "beta[1]"])
  shape_post <- fit$phi_summary[["mean"]]

  expect_lt(abs(beta_post - beta0), 0.30)
  expect_lt(abs(shape_post - shape_true) / shape_true, 0.40)
  expect_true(mean(fit$accept_prob) > 0.4)
})

test_that("tulpa_nuts_spde neg_binomial_2 recovers intercept and size", {
  skip_if_not_installed("fmesher")

  set.seed(13)
  n_obs <- 200
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec <- helper_make_spde_spec(coords, max_edge = c(0.25, 0.6), cutoff = 0.10)

  beta0     <- 1.2
  size_true <- 3.0  # NegBin size r (phi in the type-2 parametrisation)
  w_true    <- rnorm(spec$n_mesh, 0, 0.2)
  w_true    <- w_true - mean(w_true)
  eta       <- beta0 + as.numeric(spec$A %*% w_true)
  mu        <- exp(eta)
  y         <- rnbinom(n_obs, size = size_true, mu = mu)
  X         <- matrix(1.0, nrow = n_obs, ncol = 1)

  fit <- tulpa_nuts_spde(
    y = y, X = X, spatial = spec,
    family = "neg_binomial_2",
    range = 0.4, sigma = 0.3,
    log_phi_init = log(size_true),
    n_iter = 600L, n_warmup = 300L, seed = 13L
  )

  beta_post <- mean(fit$draws[, "beta[1]"])
  size_post <- fit$phi_summary[["mean"]]

  expect_lt(abs(beta_post - beta0), 0.40)
  # NegBin size is notoriously weakly identified; tolerate a wide envelope.
  expect_lt(abs(log(size_post) - log(size_true)), 0.8)
  expect_true(mean(fit$accept_prob) > 0.4)
})

test_that("tulpa_nuts_spde beta recovers intercept and precision", {
  skip_if_not_installed("fmesher")

  set.seed(17)
  n_obs <- 200
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec <- helper_make_spde_spec(coords, max_edge = c(0.25, 0.6), cutoff = 0.10)

  beta0    <- 0.2  # logit(mu0) ~ 0.55
  phi_true <- 12.0
  w_true   <- rnorm(spec$n_mesh, 0, 0.15)
  w_true   <- w_true - mean(w_true)
  eta      <- beta0 + as.numeric(spec$A %*% w_true)
  mu       <- 1.0 / (1.0 + exp(-eta))
  y        <- rbeta(n_obs, shape1 = mu * phi_true, shape2 = (1 - mu) * phi_true)
  X        <- matrix(1.0, nrow = n_obs, ncol = 1)

  fit <- tulpa_nuts_spde(
    y = y, X = X, spatial = spec,
    family = "beta",
    range = 0.4, sigma = 0.3,
    log_phi_init = log(phi_true),
    n_iter = 600L, n_warmup = 300L, seed = 17L
  )

  beta_post <- mean(fit$draws[, "beta[1]"])
  phi_post  <- fit$phi_summary[["mean"]]

  expect_lt(abs(beta_post - beta0), 0.30)
  expect_lt(abs(log(phi_post) - log(phi_true)), 0.50)
  expect_true(mean(fit$accept_prob) > 0.4)
})

test_that("tulpa_nuts_spde rejects unsupported family", {
  skip_if_not_installed("fmesher")
  set.seed(2)
  coords <- cbind(runif(20), runif(20))
  spec <- helper_make_spde_spec(coords, max_edge = c(0.3, 0.6), cutoff = 0.15)
  y <- rnorm(20)
  X <- matrix(1.0, nrow = 20, ncol = 1)
  expect_error(
    tulpa_nuts_spde(y = y, X = X, spatial = spec, family = "inverse_gaussian",
                    range = 0.3, sigma = 0.3,
                    n_iter = 50L, n_warmup = 25L),
    "should be one of"  # match.arg() rejects before the C++ shim sees it
  )
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
    re_idx = rep(0, n_obs), n_re_groups = 0L, sigma_re = 1.0,
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
  # Asymmetric tolerances are intentional: the intercept absorbs the
  # spatial field's sum-to-zero residual and is the noisier of the two
  # under finite-sample MCSE, while the slope is pinned by the
  # covariate signal. A 20-seed sweep at this 800/400 budget puts
  # |beta_n[1] - beta_l[1]| at median 0.04 / max 0.11 / q95 0.10, and
  # |beta_n[2] - beta_l[2]| at median 0.005 / max 0.015 — so 0.15 / 0.05
  # gives a 3-4 MCSE margin without changing the test's semantic claim.
  expect_lt(abs(beta_n[1] - beta_l[1]), 0.15)
  expect_lt(abs(beta_n[2] - beta_l[2]), 0.05)
})

test_that("non-centered fixed-hyper NUTS calibrates beta SD to the Laplace SE (#87)", {
  skip_if_not_installed("fmesher")
  skip_on_cran()

  set.seed(2026)
  n_obs <- 200
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec <- helper_make_spde_spec(coords, max_edge = c(0.2, 0.5), cutoff = 0.08)

  beta0 <- 0.5; beta1 <- -0.8; sigma_obs <- 0.3
  range_true <- 0.4; sigma_w <- 0.5
  x_cov <- runif(n_obs, -1, 1); X <- cbind(1, x_cov)
  w_true <- rnorm(spec$n_mesh, 0, 0.4); w_true <- w_true - mean(w_true)
  eta <- beta0 + beta1 * x_cov + as.numeric(spec$A %*% w_true)
  y   <- eta + rnorm(n_obs, 0, sigma_obs)

  # Non-centered (the default) decorrelates beta from the mesh field, so the
  # sampler traverses the beta/field ridge #87 identified. The marginal beta SD
  # should now track the exact Gaussian Laplace marginal-SE (Schur complement),
  # not the under-dispersed value the direct-field path produced at a short
  # budget.
  fit <- tulpa_nuts_spde(
    y = y, X = X, spatial = spec, family = "gaussian",
    range = range_true, sigma = sigma_w, log_phi_init = log(sigma_obs),
    noncenter = TRUE, n_iter = 1500L, n_warmup = 800L, seed = 7L)
  sd_n <- apply(fit$draws[, c("beta[1]", "beta[2]"), drop = FALSE], 2L, sd)
  expect_lt(sum(fit$divergent), 0.05 * length(fit$divergent))

  lap <- laplace_spde_at(
    y = y, n_trials = rep(1L, n_obs), X = X, spatial = spec,
    family = "gaussian", phi = sigma_obs^2,
    range = range_true, sigma = sigma_w, max_iter = 100L, tol = 1e-6)
  Hb <- tulpa:::.marginal_H_beta_spde(
    mode = lap$mode, X = X, spatial = spec, family = "gaussian",
    phi = sigma_obs^2, n_trials = rep(1L, n_obs),
    range_val = range_true, sigma_val = sigma_w)
  se_lap <- sqrt(diag(solve(as.matrix(Hb))))

  # Calibrated, not under-dispersed: the NUTS SD is within 30% of the exact SE
  # on both coefficients (a 5-8% gap at this budget, with MCSE margin).
  expect_lt(abs(sd_n[1] - se_lap[1]) / se_lap[1], 0.30)
  expect_lt(abs(sd_n[2] - se_lap[2]) / se_lap[2], 0.30)
})

test_that("tulpa_nuts_spde fixed-hyper fractional nu recovers + matches Laplace (#85)", {
  skip_if_not_installed("fmesher")

  set.seed(404)
  n_obs  <- 200
  coords <- cbind(runif(n_obs), runif(n_obs))
  # nu = 0.5 -> alpha = 1.5: the operator-based rational SPDE field.
  spec   <- helper_make_spde_spec(coords, max_edge = c(0.2, 0.5),
                                  cutoff = 0.08, nu = 0.5)
  expect_true(tulpa:::.spde_nu_is_fractional(spec$nu))

  beta0 <- 0.4; beta1 <- -0.7; sigma_obs <- 0.3
  range_true <- 0.4; sigma_w <- 0.5
  x_cov <- runif(n_obs, -1, 1); X <- cbind(1, x_cov)
  w_true <- rnorm(spec$n_mesh, 0, 0.4); w_true <- w_true - mean(w_true)
  eta <- beta0 + beta1 * x_cov + as.numeric(spec$A %*% w_true)
  y   <- eta + rnorm(n_obs, 0, sigma_obs)

  # Joint-hyper fractional NUTS stays gated.
  expect_error(
    tulpa_nuts_spde(y = y, X = X, spatial = spec, family = "gaussian",
                    joint = TRUE, n_iter = 10L, n_warmup = 5L),
    "fractional")

  fit <- tulpa_nuts_spde(
    y = y, X = X, spatial = spec, family = "gaussian",
    range = range_true, sigma = sigma_w, log_phi_init = log(sigma_obs),
    n_iter = 800L, n_warmup = 400L, seed = 404L)

  expect_true(isTRUE(fit$rational))
  beta_post <- colMeans(fit$draws[, c("beta[1]", "beta[2]"), drop = FALSE])
  expect_lt(abs(beta_post[1] - beta0), 0.25)
  expect_lt(abs(beta_post[2] - beta1), 0.20)
  expect_true(mean(fit$accept_prob) > 0.4)
  expect_true(sum(fit$divergent) < 0.10 * length(fit$divergent))

  # Field draws reconstructed to the full mesh (u = Pr x).
  expect_equal(ncol(fit$field_draws), spec$n_mesh)
  expect_true(all(is.finite(colMeans(fit$field_draws))))

  # Cross-check the beta MODE against the fractional Laplace fit at the SAME
  # fixed (range, sigma) -- the same convention as the integer NUTS-vs-Laplace
  # test above. The marginal beta SD is calibrated separately on the integer
  # path by the non-centered #87 test (the non-centered reparameterization
  # decorrelates beta from the field; the rational path shares that transform
  # via init_fixed). The Laplace rational SE is validated in its own right by
  # the Schur identity in test-spde.R.
  lap <- tulpa:::laplace_spde_at(
    y = y, n_trials = rep(1L, n_obs), X = X, spatial = spec,
    family = "gaussian", phi = sigma_obs^2,
    range = range_true, sigma = sigma_w, max_iter = 100L, tol = 1e-6)
  lap_beta <- lap$mode[seq_len(ncol(X))]
  expect_lt(max(abs(beta_post - lap_beta)), 0.12)
})
