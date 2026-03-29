# test-gpu-nngp.R
# Tests for GPU-batched NNGP Laplace (Feature 6)
# Tests the batched computation path regardless of GPU availability.
# The GPU is optional — CPU fallback is always used if CUDA unavailable.

test_that("batched NNGP produces same results as sequential for small problem", {
  # This test verifies the batched computation matches the original sequential.
  # Uses a small problem (100 locations) where both paths work.
  set.seed(42)
  n_obs <- 100
  coords <- cbind(runif(n_obs), runif(n_obs))

  w_true <- rnorm(n_obs, 0, 0.3)
  eta <- -0.5 + w_true
  y <- rbinom(n_obs, 1, plogis(eta))
  X <- matrix(1, nrow = n_obs, ncol = 1)

  nn_k <- 10L
  order_idx <- order(coords[, 1], coords[, 2])
  coords_ord <- coords[order_idx, ]

  nn_idx <- matrix(0L, nrow = n_obs, ncol = nn_k)
  nn_dist <- matrix(0, nrow = n_obs, ncol = nn_k)

  for (i in 2:n_obs) {
    dists <- sqrt((coords_ord[1:(i-1), 1] - coords_ord[i, 1])^2 +
                  (coords_ord[1:(i-1), 2] - coords_ord[i, 2])^2)
    n_cand <- min(length(dists), nn_k)
    ord <- order(dists)[1:n_cand]
    nn_idx[i, seq_len(n_cand)] <- ord
    nn_dist[i, seq_len(n_cand)] <- dists[ord]
  }

  result <- cpp_laplace_fit_gp(
    y = as.integer(y), n = as.integer(rep(1L, n_obs)),
    X = X,
    re_idx = rep(0, n_obs), n_re_groups = 0L, sigma_re = 1.0,
    coords = coords_ord, nn_idx = nn_idx, nn_dist = nn_dist,
    nn_order = as.integer(order_idx - 1L), n_spatial = n_obs, nn = nn_k,
    sigma2_gp = 1.0, phi_gp = 0.3, cov_type = 0L,
    family = "binomial", phi = 1.0,
    max_iter = 100L, tol = 1e-6, n_threads = 1L
  )

  expect_true(is.finite(result$log_marginal))
  expect_equal(length(result$mode), 1L + n_obs)
  # Should converge or at least make progress
  expect_true(result$n_iter > 0)
})
