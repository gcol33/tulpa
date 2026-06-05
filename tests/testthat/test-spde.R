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
# Marginal H_beta for SPDE Laplace (issue #16)
# =====================================================================

test_that("tulpa_laplace returns finite PD H_beta for SPDE (#16)", {
  skip_if_not_installed("tulpaMesh")

  set.seed(42)
  n      <- 200
  coords <- cbind(runif(n), runif(n))
  X      <- cbind(1, rnorm(n), rnorm(n))
  beta_true <- c(0.3, -0.6, 1.0)

  # Smooth spatial field
  d <- as.matrix(dist(coords))
  Sigma_w <- exp(-d / 0.25)
  w_true <- as.numeric(t(chol(Sigma_w + diag(1e-6, n))) %*% rnorm(n))
  eta <- as.numeric(X %*% beta_true) + 0.7 * w_true
  y   <- rbinom(n, 1, plogis(eta))

  mesh    <- tulpaMesh::tulpa_mesh(coords, max_edge = c(0.1, 0.3), cutoff = 0.03)
  spatial <- spatial_spde(coords, mesh = mesh, nu = 1,
                          prior_range = c(0.25, 0.5),
                          prior_sigma = c(1.0, 0.5))

  fit <- tulpa_laplace(
    y = y, n_trials = rep(1L, n), X = X,
    family = "binomial", spatial = spatial,
    max_iter = 100L, tol = 1e-6,
    return_hessian = TRUE
  )

  expect_true(fit$converged)
  expect_false(is.null(fit$H_beta))
  expect_equal(dim(fit$H_beta), c(ncol(X), ncol(X)))
  expect_true(all(is.finite(fit$H_beta)))
  expect_true(isSymmetric(fit$H_beta, tol = 1e-8))

  # Positive definite
  ev <- eigen(fit$H_beta, symmetric = TRUE, only.values = TRUE)$values
  expect_true(all(ev > 0))

  # SEs finite and in a sensible range
  sd <- sqrt(diag(solve(fit$H_beta)))
  expect_true(all(is.finite(sd)))
  expect_true(all(sd > 0 & sd < 5))

  # return_hessian = FALSE should not populate H_beta
  fit_noH <- tulpa_laplace(
    y = y, n_trials = rep(1L, n), X = X,
    family = "binomial", spatial = spatial,
    max_iter = 100L, tol = 1e-6,
    return_hessian = FALSE
  )
  expect_null(fit_noH$H_beta)
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
    re_idx = rep(0L, n_obs), n_re_groups = 0L, sigma_re = 1.0,
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

  expect_equal(length(result$log_marginal), 9L)
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
    re_idx = rep(0L, n_obs), n_re_groups = 0L, sigma_re = 1.0,
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

# ============================================================================
# Dispatch tests: tulpa_laplace(spatial = spatial_spde(...)) wires through
# ============================================================================

# Build a minimal SPDE spec without needing fmesher: a tiny synthetic mesh
# (4 nodes on a unit square) is enough to exercise the dispatch path.
make_synthetic_spde_spec <- function(n_obs = 30, seed = 7) {
  set.seed(seed)
  # 4 mesh nodes at the unit square corners
  n_mesh <- 4L
  C <- Matrix::Diagonal(n_mesh, x = 0.25)
  # Stiffness for two right triangles spanning the unit square
  G <- Matrix::Matrix(0, n_mesh, n_mesh, sparse = TRUE)
  G[1, 1] <- 2; G[2, 2] <- 1; G[3, 3] <- 1; G[4, 4] <- 2
  G[1, 2] <- G[2, 1] <- -1
  G[1, 3] <- G[3, 1] <- -1
  G[2, 4] <- G[4, 2] <- -0.5
  G[3, 4] <- G[4, 3] <- -0.5
  G <- as(G, "CsparseMatrix")
  # Bilinear interpolation A: each obs to a single mesh node (nearest corner).
  obs_xy <- matrix(runif(2 * n_obs), n_obs, 2)
  corners <- rbind(c(0, 0), c(1, 0), c(0, 1), c(1, 1))
  nearest <- apply(obs_xy, 1, function(p) which.min(rowSums((corners - p)^2)))
  A <- Matrix::sparseMatrix(
    i = seq_len(n_obs), j = nearest, x = 1.0,
    dims = c(n_obs, n_mesh)
  )
  list(
    spec = spatial_spde_custom(C = C, G = G, A = A,
                               nu = 1, prior_range = c(0.3, 0.5),
                               prior_sigma = c(1, 0.5)),
    n_obs = n_obs
  )
}

test_that("dispatch_laplace_spatial routes SPDE specs (no longer errors)", {
  ss <- make_synthetic_spde_spec()
  X <- matrix(1.0, nrow = ss$n_obs, ncol = 1)
  y <- rbinom(ss$n_obs, 1, 0.5)

  # Dispatcher previously threw "Spatial type 'spde' not yet supported in
  # Laplace". The new branch should reach the C++ kernel and either return
  # a result or surface a kernel error — but never the legacy dispatch gap.
  msg <- tryCatch(
    {
      dispatch_laplace_spatial(
        y = y, n_trials = rep(1L, ss$n_obs), X = X,
        re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
        spatial = ss$spec, family = "binomial", phi = 1.0,
        max_iter = 5L, tol = 1e-3, n_threads = 1L
      )
      NA_character_
    },
    error = function(e) conditionMessage(e)
  )
  if (!is.na(msg)) {
    expect_false(grepl("not yet supported", msg, fixed = TRUE))
  }
})

test_that("SPDE Laplace rejects an additional iid RE block with a clear error", {
  ss <- make_synthetic_spde_spec()
  X <- matrix(1.0, nrow = ss$n_obs, ncol = 1)
  y <- rbinom(ss$n_obs, 1, 0.5)

  expect_error(
    dispatch_laplace_spatial(
      y = y, n_trials = rep(1L, ss$n_obs), X = X,
      re_idx = sample.int(3, ss$n_obs, replace = TRUE),
      n_re_groups = 3L, sigma_re = 1.0,
      spatial = ss$spec, family = "binomial", phi = 1.0,
      max_iter = 5L, tol = 1e-3, n_threads = 1L
    ),
    "does not yet support an additional iid RE block"
  )
})

test_that("tulpa_laplace(spatial = spatial_spde_custom(...)) runs end-to-end", {
  ss <- make_synthetic_spde_spec(n_obs = 50)
  X <- cbind(1, rnorm(ss$n_obs))
  y <- rbinom(ss$n_obs, 1, 0.5)

  fit <- tulpa_laplace(
    y = y, n_trials = rep(1L, ss$n_obs), X = X,
    re_list = list(),
    family = "binomial",
    spatial = ss$spec,
    max_iter = 50L, tol = 1e-6, n_threads = 1L,
    return_hessian = TRUE
  )

  # mode = c(beta, spatial_effects)
  expect_length(fit$mode, ncol(X) + ss$spec$n_mesh)
  expect_true(is.finite(fit$log_marginal))
  # Since issue #16: H_beta is the Schur'd marginal precision for SPDE.
  expect_false(is.null(fit$H_beta))
  expect_equal(dim(fit$H_beta), c(ncol(X), ncol(X)))
  expect_true(all(is.finite(fit$H_beta)))
})

test_that("fit_spde and dispatch_laplace_spatial agree on the same problem", {
  ss <- make_synthetic_spde_spec(n_obs = 40)
  X <- matrix(1.0, nrow = ss$n_obs, ncol = 1)
  y <- rbinom(ss$n_obs, 1, 0.5)

  via_dispatch <- dispatch_laplace_spatial(
    y = y, n_trials = rep(1L, ss$n_obs), X = X,
    re_idx = NULL, n_re_groups = 0L, sigma_re = 1.0,
    spatial = ss$spec, family = "binomial", phi = 1.0,
    max_iter = 50L, tol = 1e-6, n_threads = 1L
  )
  via_fit_spde <- fit_spde(
    y = y, X = X, spatial = ss$spec,
    family = "binomial", n_trials = rep(1L, ss$n_obs),
    range = ss$spec$prior_range[1], sigma = ss$spec$prior_sigma[1],
    nested_laplace = FALSE,
    max_iter = 50L, tol = 1e-6, n_threads = 1L
  )

  # Same kernel, same hyperparameters → identical mode and log_marginal.
  expect_equal(via_dispatch$mode, via_fit_spde$mode, tolerance = 1e-10)
  expect_equal(via_dispatch$log_marginal, via_fit_spde$log_marginal,
               tolerance = 1e-10)
})

# --- offset() threading through the SPDE Laplace paths (gcol33/tulpa#72) -------

test_that("fit_spde threads offset() into the field linear predictor (gcol33/tulpa#72)", {
  ss  <- make_synthetic_spde_spec(n_obs = 60, seed = 11)
  X   <- matrix(1.0, nrow = ss$n_obs, ncol = 1)
  set.seed(11)
  # Poisson counts with a real per-observation log-exposure offset.
  off <- log(runif(ss$n_obs, 0.5, 2))
  y   <- rpois(ss$n_obs, exp(0.3 + off))
  r   <- ss$spec$prior_range[1]; s <- ss$spec$prior_sigma[1]
  fit <- function(o) fit_spde(
    y = y, X = X, spatial = ss$spec, family = "poisson",
    n_trials = rep(1L, ss$n_obs), range = r, sigma = s,
    nested_laplace = FALSE, max_iter = 200L, tol = 1e-12, offset = o)

  base <- fit(NULL)
  zero <- fit(rep(0, ss$n_obs))
  # offset = 0 reproduces the no-offset fit exactly (no corruption of eta).
  expect_equal(zero$mode, base$mode, tolerance = 1e-10)
  expect_equal(zero$log_marginal, base$log_marginal, tolerance = 1e-10)
  # A non-trivial offset moves the latent mode -- it is not silently dropped.
  shifted <- fit(off)
  expect_gt(max(abs(shifted$mode - base$mode)), 1e-3)
})

test_that("fit_spde absorbs a constant offset into the intercept (gcol33/tulpa#72)", {
  ss <- make_synthetic_spde_spec(n_obs = 60, seed = 12)
  X  <- matrix(1.0, nrow = ss$n_obs, ncol = 1)
  set.seed(12)
  y  <- rpois(ss$n_obs, exp(0.4))
  r  <- ss$spec$prior_range[1]; s <- ss$spec$prior_sigma[1]
  cc <- 0.5
  fit <- function(o) fit_spde(
    y = y, X = X, spatial = ss$spec, family = "poisson",
    n_trials = rep(1L, ss$n_obs), range = r, sigma = s,
    nested_laplace = FALSE, max_iter = 300L, tol = 1e-12, offset = o)

  base <- fit(NULL)
  con  <- fit(rep(cc, ss$n_obs))
  # eta = c + b0 + field is invariant under (b0 -> b0 - c): the fitted intercept
  # shifts by -c, the centered field and log-marginal unchanged to the weak
  # intercept-prior tolerance.
  expect_equal(con$beta[1], base$beta[1] - cc, tolerance = 1e-3)
  expect_equal(con$spatial_effects, base$spatial_effects, tolerance = 1e-3)
  expect_equal(con$log_marginal, base$log_marginal, tolerance = 1e-2)
})

test_that("fit_spde nested integrates offset (offset = 0 == no offset) (gcol33/tulpa#72)", {
  ss <- make_synthetic_spde_spec(n_obs = 50, seed = 13)
  X  <- matrix(1.0, nrow = ss$n_obs, ncol = 1)
  set.seed(13)
  y  <- rpois(ss$n_obs, exp(0.3))
  fit <- function(o) fit_spde(
    y = y, X = X, spatial = ss$spec, family = "poisson",
    n_trials = rep(1L, ss$n_obs), method = "grid", n_grid = 3L,
    diagnose_k = FALSE, offset = o)

  base <- fit(NULL)
  zero <- fit(rep(0, ss$n_obs))
  expect_equal(zero$log_marginal, base$log_marginal, tolerance = 1e-8)
})
