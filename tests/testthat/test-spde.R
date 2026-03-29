# test-spde.R
# Tests for SPDE spatial Laplace (Matérn via FEM mesh)
# Uses fmesher for mesh generation and FEM matrices

test_that("SPDE Laplace fits binomial model on fmesher mesh", {
  skip_if_not_installed("fmesher")

  set.seed(42)
  n_obs <- 200
  coords <- cbind(runif(n_obs), runif(n_obs))

  # Build mesh
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.15, 0.5), cutoff = 0.05)
  n_mesh <- mesh$n

  # FEM matrices
  fem <- fmesher::fm_fem(mesh)
  C0 <- fem$c0  # lumped mass (diagonal)
  G1 <- fem$g1  # stiffness

  # Projection matrix
  A <- fmesher::fm_basis(mesh, loc = coords)

  # Convert to CSC components for C++ (Matrix uses 0-based indices internally)
  # A is dgCMatrix: slots @i (row indices, 0-based), @p (col pointers), @x (values)
  A_csc <- as(A, "CsparseMatrix")
  G1_csc <- as(G1, "CsparseMatrix")

  # C0 diagonal
  C0_diag <- Matrix::diag(C0)

  # Simulate spatial field on mesh
  # Simple: just use intercept + noise
  beta0 <- -0.5
  w_true <- rnorm(n_mesh, 0, 0.3)
  w_true <- w_true - mean(w_true)

  eta <- beta0 + as.numeric(A %*% w_true)
  prob <- plogis(eta)
  y <- rbinom(n_obs, size = 1, prob = prob)

  # Design matrix: intercept only
  X <- matrix(1.0, nrow = n_obs, ncol = 1)

  # SPDE parameters: kappa and tau
  # For range ~ 0.3, nu = 1: kappa = sqrt(8*1) / 0.3 ~ 9.4
  # tau = 1 / (sqrt(4*pi) * kappa * sigma) where sigma ~ 0.5
  range_true <- 0.3
  sigma_true <- 0.5
  kappa <- sqrt(8) / range_true
  tau_spde <- 1.0 / (sqrt(4 * pi) * kappa * sigma_true)

  result <- cpp_laplace_fit_spde(
    y = as.integer(y),
    n_trials = as.integer(rep(1L, n_obs)),
    X = X,
    A_x = A_csc@x,
    A_i = A_csc@i,
    A_p = A_csc@p,
    n_obs = n_obs,
    n_mesh = n_mesh,
    C0_diag = C0_diag,
    G1_x = G1_csc@x,
    G1_i = G1_csc@i,
    G1_p = G1_csc@p,
    kappa = kappa,
    tau_spde = tau_spde,
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

  # Mode should have correct length: p + n_mesh
  expect_equal(length(result$mode), 1L + n_mesh)

  # Intercept should be in reasonable range
  beta_hat <- result$mode[1]
  expect_true(abs(beta_hat - beta0) < 2.0)

  # Spatial effects should be centered
  w_hat <- result$mode[2:(n_mesh + 1)]
  expect_true(abs(mean(w_hat)) < 0.1)
})

test_that("SPDE Laplace works with Poisson family", {
  skip_if_not_installed("fmesher")

  set.seed(123)
  n_obs <- 150
  coords <- cbind(runif(n_obs), runif(n_obs))

  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.2, 0.5), cutoff = 0.05)
  fem <- fmesher::fm_fem(mesh)
  A <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  G1 <- as(fem$g1, "CsparseMatrix")
  C0_diag <- Matrix::diag(fem$c0)

  # Simulate Poisson
  w_true <- rnorm(mesh$n, 0, 0.2)
  w_true <- w_true - mean(w_true)
  eta <- 1.0 + as.numeric(A %*% w_true)
  y <- rpois(n_obs, lambda = exp(eta))

  X <- matrix(1.0, nrow = n_obs, ncol = 1)
  kappa <- sqrt(8) / 0.4
  tau_spde <- 1.0 / (sqrt(4 * pi) * kappa * 0.3)

  result <- cpp_laplace_fit_spde(
    y = as.integer(y),
    n_trials = as.integer(rep(1L, n_obs)),
    X = X,
    A_x = A@x, A_i = A@i, A_p = A@p,
    n_obs = n_obs, n_mesh = mesh$n,
    C0_diag = C0_diag,
    G1_x = G1@x, G1_i = G1@i, G1_p = G1@p,
    kappa = kappa, tau_spde = tau_spde,
    family = "poisson", phi = 1.0,
    max_iter = 100L, tol = 1e-6, n_threads = 1L
  )

  expect_true(result$converged)
  expect_true(is.finite(result$log_marginal))
  expect_equal(length(result$mode), 1L + mesh$n)
})

test_that("SPDE scales to 500+ mesh nodes with sparse Q", {
  skip_if_not_installed("fmesher")

  set.seed(99)
  n_obs <- 500
  coords <- cbind(runif(n_obs), runif(n_obs))

  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.08, 0.3), cutoff = 0.03)
  n_mesh <- mesh$n

  fem <- fmesher::fm_fem(mesh)
  A <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  G1 <- as(fem$g1, "CsparseMatrix")
  C0_diag <- Matrix::diag(fem$c0)

  # Simulate
  w_true <- rnorm(n_mesh, 0, 0.3)
  w_true <- w_true - mean(w_true)
  eta <- -0.5 + as.numeric(A %*% w_true)
  y <- rbinom(n_obs, 1, plogis(eta))

  X <- matrix(1.0, nrow = n_obs, ncol = 1)
  kappa <- sqrt(8) / 0.2
  tau_spde <- 1.0 / (sqrt(4 * pi) * kappa * 0.4)

  t_elapsed <- system.time({
    result <- cpp_laplace_fit_spde(
      y = as.integer(y), n_trials = as.integer(rep(1L, n_obs)),
      X = X,
      A_x = A@x, A_i = A@i, A_p = A@p,
      n_obs = n_obs, n_mesh = n_mesh,
      C0_diag = C0_diag,
      G1_x = G1@x, G1_i = G1@i, G1_p = G1@p,
      kappa = kappa, tau_spde = tau_spde,
      family = "binomial", phi = 1.0,
      max_iter = 100L, tol = 1e-6, n_threads = 1L
    )
  })["elapsed"]

  expect_true(result$converged)
  expect_true(is.finite(result$log_marginal))
  expect_equal(length(result$mode), 1L + n_mesh)

  # Q should be sparse: nnz << n_mesh^2
  q_density <- result$Q_nnz / (as.numeric(n_mesh) * n_mesh)
  message("\n  SPDE scale test: n_mesh=", n_mesh,
          " Q_nnz=", result$Q_nnz,
          " density=", round(q_density, 4),
          " iters=", result$n_iter,
          " time=", round(t_elapsed, 2), "s")

  expect_true(q_density < 0.1)  # Q should be < 10% dense
  expect_true(n_mesh >= 500)     # verify mesh is large enough
})

# =====================================================================
# Nested Laplace for SPDE: 2D grid over (range, sigma)
# =====================================================================

test_that("nested Laplace SPDE runs with 2D hyperparameter grid", {
  skip_if_not_installed("fmesher")

  set.seed(42)
  n_obs <- 200
  coords <- cbind(runif(n_obs), runif(n_obs))

  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.15, 0.5), cutoff = 0.05)
  fem <- fmesher::fm_fem(mesh)
  A <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  G1 <- as(fem$g1, "CsparseMatrix")
  C0_diag <- Matrix::diag(fem$c0)

  # Simulate
  w_true <- rnorm(mesh$n, 0, 0.3)
  w_true <- w_true - mean(w_true)
  eta <- -0.5 + as.numeric(A %*% w_true)
  y <- rbinom(n_obs, 1, plogis(eta))
  X <- matrix(1.0, nrow = n_obs, ncol = 1)

  # 2D grid: range × sigma (3 × 3 = 9 points)
  range_vals <- exp(seq(log(0.1), log(0.6), length.out = 3))
  sigma_vals <- exp(seq(log(0.2), log(1.0), length.out = 3))
  grid <- expand.grid(range = range_vals, sigma = sigma_vals)

  result <- cpp_nested_laplace_spde(
    y = as.integer(y), n_trials = as.integer(rep(1L, n_obs)),
    X = X,
    A_x = A@x, A_i = A@i, A_p = A@p,
    n_obs = n_obs, n_mesh = mesh$n,
    C0_diag = C0_diag,
    G1_x = G1@x, G1_i = G1@i, G1_p = G1@p,
    range_grid = grid$range,
    sigma_grid = grid$sigma,
    nu = 1.0,
    family = "binomial", phi = 1.0,
    max_iter = 50L, tol = 1e-6, n_threads = 1L
  )

  expect_equal(result$n_grid, 9L)
  expect_true(all(is.finite(result$log_marginal)))
  expect_true(all(result$n_iter > 0))

  # Log-marginals should vary across grid
  lml_range <- max(result$log_marginal) - min(result$log_marginal)
  expect_true(lml_range > 0.1)
})

test_that("nested Laplace SPDE warm-start reduces iterations", {
  skip_if_not_installed("fmesher")

  set.seed(42)
  n_obs <- 150
  coords <- cbind(runif(n_obs), runif(n_obs))

  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.2, 0.5), cutoff = 0.05)
  fem <- fmesher::fm_fem(mesh)
  A <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  G1 <- as(fem$g1, "CsparseMatrix")
  C0_diag <- Matrix::diag(fem$c0)

  w_true <- rnorm(mesh$n, 0, 0.2)
  w_true <- w_true - mean(w_true)
  eta <- 0 + as.numeric(A %*% w_true)
  y <- rbinom(n_obs, 1, plogis(eta))
  X <- matrix(1.0, nrow = n_obs, ncol = 1)

  # Grid of nearby points (should benefit from warm-starting)
  range_grid <- exp(seq(log(0.2), log(0.5), length.out = 5))
  sigma_grid <- rep(0.5, 5)

  result <- cpp_nested_laplace_spde(
    y = as.integer(y), n_trials = as.integer(rep(1L, n_obs)),
    X = X,
    A_x = A@x, A_i = A@i, A_p = A@p,
    n_obs = n_obs, n_mesh = mesh$n,
    C0_diag = C0_diag,
    G1_x = G1@x, G1_i = G1@i, G1_p = G1@p,
    range_grid = range_grid, sigma_grid = sigma_grid,
    nu = 1.0, family = "binomial", phi = 1.0,
    max_iter = 50L, tol = 1e-6, n_threads = 1L
  )

  expect_true(all(is.finite(result$log_marginal)))

  # Later grid points should converge faster due to warm-start
  expect_true(any(result$n_iter[2:5] <= result$n_iter[1]))
})
