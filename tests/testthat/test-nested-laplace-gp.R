# Nested Laplace dispatch through the continuous-spatial GP backends:
#   - NNGP: 2D grid over (sigma2_gp, phi_gp)
#   - HSGP: 2D grid over (sigma2, lengthscale)

simulate_gp_binomial <- function(n_spatial = 50L, n_obs_per_site = 6L,
                                  beta0 = 0.0,
                                  sigma2 = 0.5, phi_gp = 0.4,
                                  seed = 2026L) {
  set.seed(seed)
  coords <- cbind(runif(n_spatial), runif(n_spatial))
  D <- as.matrix(dist(coords))
  K <- sigma2 * exp(-D / phi_gp) + diag(1e-6, n_spatial)
  L <- chol(K)
  w_true <- as.numeric(crossprod(L, rnorm(n_spatial)))
  w_true <- w_true - mean(w_true)

  spatial_idx <- rep(seq_len(n_spatial), each = n_obs_per_site)
  N <- length(spatial_idx)
  X <- matrix(1, N, 1)
  eta <- beta0 + w_true[spatial_idx]
  p <- 1 / (1 + exp(-eta))
  y <- rbinom(N, 1L, p)
  list(y = as.integer(y), n_trials = rep(1L, N), X = X,
       coords = coords, spatial_idx = spatial_idx,
       n_spatial = n_spatial, w_true = w_true)
}

# Build a simple NNGP neighbor structure (k nearest among earlier nodes
# in the supplied coord ordering). Returns nn_idx as 1-based indices into
# the NNGP-ordered list, nn_dist as the corresponding distances, and a
# 0-based nn_order ready to feed cpp_nested_laplace_nngp directly.
make_nngp_neighbors <- function(coords, n_neighbors = 10L) {
  n <- nrow(coords)
  order_idx <- order(coords[, 1], coords[, 2])
  coords_ord <- coords[order_idx, ]
  nn_idx <- matrix(0L, n, n_neighbors)
  nn_dist <- matrix(0, n, n_neighbors)
  for (i in 2:n) {
    dists <- sqrt((coords_ord[1:(i - 1), 1] - coords_ord[i, 1])^2 +
                    (coords_ord[1:(i - 1), 2] - coords_ord[i, 2])^2)
    nc <- min(length(dists), n_neighbors)
    ord <- order(dists)[1:nc]
    nn_idx[i, seq_len(nc)] <- ord
    nn_dist[i, seq_len(nc)] <- dists[ord]
  }
  list(coords_ord = coords_ord,
       nn_idx = nn_idx, nn_dist = nn_dist,
       nn_order_0 = as.integer(order_idx - 1L),  # 0-based perm for C++
       n_neighbors = n_neighbors)
}

# =====================================================================
# NNGP
# =====================================================================

# NNGP nested-Laplace requires N == n_spatial (one obs per spatial unit),
# matching laplace_mode_gp in laplace_core.cpp. Build the data accordingly.

make_nngp_dataset <- function(n_spatial = 60L, beta0 = -0.3,
                                sigma2 = 0.6, phi_gp = 0.3,
                                seed = 2026L) {
  set.seed(seed)
  coords <- cbind(runif(n_spatial), runif(n_spatial))
  D <- as.matrix(dist(coords))
  K <- sigma2 * exp(-D / phi_gp) + diag(1e-6, n_spatial)
  L <- chol(K)
  w_true <- as.numeric(crossprod(L, rnorm(n_spatial)))
  w_true <- w_true - mean(w_true)
  eta <- beta0 + w_true
  y <- rbinom(n_spatial, 1L, plogis(eta))
  X <- matrix(1, n_spatial, 1)
  list(y = as.integer(y), n_trials = rep(1L, n_spatial), X = X,
       coords = coords, n_spatial = n_spatial, w_true = w_true)
}

test_that("nested_laplace NNGP runs on a 2D (sigma2, phi_gp) grid", {
  d <- make_nngp_dataset(n_spatial = 60L, sigma2 = 0.6, phi_gp = 0.3)
  nb <- make_nngp_neighbors(d$coords, n_neighbors = 8L)

  s2 <- c(0.3, 0.6, 1.0)
  pg <- c(0.2, 0.4, 0.8)
  gr <- expand.grid(sigma2 = s2, phi_gp = pg)

  res <- cpp_nested_laplace_nngp(
    y = d$y, n = d$n_trials, X = d$X,
    re_idx = rep(0, d$n_spatial), n_re_groups = 0L, sigma_re = 1.0,
    coords = nb$coords_ord, nn_idx = nb$nn_idx, nn_dist = nb$nn_dist,
    nn_order = nb$nn_order_0,
    n_spatial = d$n_spatial, nn = nb$n_neighbors,
    sigma2_grid = gr$sigma2, phi_gp_grid = gr$phi_gp,
    cov_type = 0L,    # exponential — matches simulation
    family = "binomial", phi = 1.0,
    max_iter = 50L, tol = 1e-6, n_threads = 1L
  )

  expect_equal(res$n_grid, 9L)
  expect_true(all(is.finite(res$log_marginal)))
  expect_true(all(res$n_iter > 0))
  expect_gt(max(res$log_marginal) - min(res$log_marginal), 0.01)
})

test_that("tulpa_nested_laplace() routes a nngp prior through the dispatch", {
  d <- make_nngp_dataset(n_spatial = 50L)
  nb <- make_nngp_neighbors(d$coords, n_neighbors = 6L)

  prior <- list(
    type        = "nngp",
    coords      = nb$coords_ord,
    nn_idx      = nb$nn_idx,
    nn_dist     = nb$nn_dist,
    nn_order    = nb$nn_order_0 + 1L,  # dispatcher subtracts 1 to make 0-based
    n_spatial   = d$n_spatial,
    nn          = nb$n_neighbors,
    cov_type    = 0L,
    sigma2_grid = c(0.3, 0.8, 1.2),
    phi_gp_grid = c(0.2, 0.5, 0.9)
  )

  res <- tulpa_nested_laplace(d$y, d$n_trials, d$X, prior = prior,
                         family = "binomial")
  expect_s3_class(res, "tulpa_nested_laplace")
  expect_named(res$theta_mean, c("sigma2", "phi_gp"))
  expect_equal(sum(res$weights), 1.0, tolerance = 1e-6)
})

# =====================================================================
# HSGP
# =====================================================================

# Build a 2D HSGP basis (m functions per dimension) and the matching
# Laplacian eigenvalues. Mirrors setup_hsgp_2d in src/hmc_hsgp.h.
make_hsgp_basis_2d <- function(coords, m_per_dim = 5L, c = 1.5) {
  x <- coords[, 1]; y <- coords[, 2]
  L1 <- max(c * (max(x) - min(x)) / 2, 0.1)
  L2 <- max(c * (max(y) - min(y)) / 2, 0.1)
  xs <- x - (max(x) + min(x)) / 2
  ys <- y - (max(y) + min(y)) / 2

  M <- m_per_dim * m_per_dim
  N <- nrow(coords)
  phi_basis <- matrix(0, N, M)
  lambda_eig <- numeric(M)
  for (j1 in seq_len(m_per_dim)) {
    lam_j1 <- (pi * j1 / (2 * L1))^2
    for (j2 in seq_len(m_per_dim)) {
      lam_j2 <- (pi * j2 / (2 * L2))^2
      idx <- (j1 - 1) * m_per_dim + j2
      lambda_eig[idx] <- lam_j1 + lam_j2
      phi1 <- (1 / sqrt(L1)) * sin(pi * j1 * (xs + L1) / (2 * L1))
      phi2 <- (1 / sqrt(L2)) * sin(pi * j2 * (ys + L2) / (2 * L2))
      phi_basis[, idx] <- phi1 * phi2
    }
  }
  list(phi_basis = phi_basis, lambda_eig = lambda_eig,
       L1 = L1, L2 = L2, m = m_per_dim, M = M)
}

test_that("nested_laplace HSGP runs on a 2D (sigma2, lengthscale) grid", {
  set.seed(7)
  N <- 80
  obs_coords <- cbind(runif(N), runif(N))
  basis <- make_hsgp_basis_2d(obs_coords, m_per_dim = 4L, c = 1.5)
  beta_true <- rnorm(basis$M, 0, 0.3)
  f <- as.numeric(basis$phi_basis %*% beta_true)
  y <- rbinom(N, 1L, plogis(0.2 + f))

  s2 <- c(0.2, 0.7, 1.5)
  ls <- c(0.1, 0.3, 0.7)
  gr <- expand.grid(sigma2 = s2, ell = ls)

  res <- cpp_nested_laplace_hsgp(
    y = as.integer(y), n = rep(1L, N), X = matrix(1, N, 1),
    re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
    phi_basis = basis$phi_basis, lambda_eig = basis$lambda_eig,
    sigma2_grid = gr$sigma2, lengthscale_grid = gr$ell,
    family = "binomial", phi = 1.0,
    max_iter = 50L, tol = 1e-6, n_threads = 1L
  )

  expect_equal(res$n_grid, 9L)
  expect_true(all(is.finite(res$log_marginal)))
  expect_true(all(res$n_iter > 0))
  expect_gt(max(res$log_marginal) - min(res$log_marginal), 0.01)
})

test_that("tulpa_nested_laplace() routes a hsgp prior through the dispatch", {
  set.seed(11)
  N <- 60
  obs_coords <- cbind(runif(N), runif(N))
  basis <- make_hsgp_basis_2d(obs_coords, m_per_dim = 4L, c = 1.5)
  y <- rbinom(N, 1L, 0.5)

  prior <- list(
    type             = "hsgp",
    phi_basis        = basis$phi_basis,
    lambda_eig       = basis$lambda_eig,
    sigma2_grid      = c(0.3, 0.8),
    lengthscale_grid = c(0.2, 0.6)
  )
  res <- tulpa_nested_laplace(as.integer(y), rep(1L, N),
                         matrix(1, N, 1), prior = prior,
                         family = "binomial")
  expect_s3_class(res, "tulpa_nested_laplace")
  expect_named(res$theta_mean, c("sigma2", "lengthscale"))
  expect_equal(sum(res$weights), 1.0, tolerance = 1e-6)
})

test_that("HSGP rejects mismatched basis dimensions", {
  set.seed(3)
  N <- 25
  y <- rbinom(N, 1L, 0.5)
  expect_error(
    cpp_nested_laplace_hsgp(
      y = as.integer(y), n = rep(1L, N), X = matrix(1, N, 1),
      re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
      phi_basis = matrix(0, N - 1L, 4L),  # wrong N
      lambda_eig = c(1, 2, 3, 4),
      sigma2_grid = c(1), lengthscale_grid = c(1),
      family = "binomial"
    ),
    "phi_basis must have N rows"
  )

  expect_error(
    cpp_nested_laplace_hsgp(
      y = as.integer(y), n = rep(1L, N), X = matrix(1, N, 1),
      re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
      phi_basis = matrix(0, N, 4L),
      lambda_eig = c(1, 2, 3),  # wrong M
      sigma2_grid = c(1), lengthscale_grid = c(1),
      family = "binomial"
    ),
    "lambda_eig must have length"
  )
})
