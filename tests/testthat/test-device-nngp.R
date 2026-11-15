skip_on_cran()

test_that("CPU and CUDA factors reconstruct each SPD matrix", {
  skip_on_cran()
  skip_if_no_cuda("cusolver")
  set.seed(1179)
  for (nn in c(10L, 15L, 30L)) {
    batch <- 64L
    matrices <- lapply(seq_len(batch), function(i) {
      x <- matrix(rnorm(nn * nn), nn)
      crossprod(x) + diag(nn) * (0.5 + i / batch)
    })
    flat <- t(vapply(matrices, as.numeric, numeric(nn * nn)))
    cpu <- tulpa:::cpp_device_cholesky(flat, nn, "cpu")
    cuda <- tulpa:::cpp_device_cholesky(flat, nn, "cuda")
    expect_true(cuda$used_cuda)
    expect_device_equal(cuda$factors, cpu$factors, 1e-12, "Cholesky factors")
    for (i in c(1L, 31L, 64L)) {
      L <- matrix(cuda$factors[i, ], nn, byrow = TRUE)
      expect_device_equal(tcrossprod(L), matrices[[i]], 1e-12, "L L'")
    }
    flat[batch, ] <- 0
    expect_error(tulpa:::cpp_device_cholesky(flat, nn, "cuda"), "batch index 63")
  }
})

test_that("NNGP moments agree across the measured size gates", {
  skip_on_cran()
  skip_if_no_cuda("cusolver")
  table <- utils::read.csv(system.file("device", "benchmarks.csv", package = "tulpa"))
  gates <- tulpa:::.device_thresholds(table)
  for (nn in c(10L, 15L, 30L)) {
    threshold <- gates$threshold[gates$operation == "nngp_cholesky" & gates$variant == nn]
    sizes <- if (length(threshold) && is.finite(threshold))
      unique(as.integer(c(max(nn + 1L, threshold - 1L), threshold, threshold + 1L))) else c(49L, 51L, 256L)
    for (batch in sizes) {
      args <- device_nngp_fixture(batch, nn)
      cpu <- do.call(tulpa:::cpp_device_nngp, c(args, list(choice = "cpu")))
      cuda <- do.call(tulpa:::cpp_device_nngp, c(args, list(choice = "cuda")))
      auto <- do.call(tulpa:::cpp_device_nngp, c(args, list(choice = "auto")))
      expect_false(cpu$used_cuda)
      expect_true(cuda$used_cuda)
      want <- length(threshold) == 1L && is.finite(threshold) && batch >= threshold
      expect_identical(auto$used_cuda, want)
      for (quantity in c("mean", "variance", "alpha")) {
        expect_device_equal(cuda[[quantity]], cpu[[quantity]], 1e-8, quantity)
        expect_device_equal(auto[[quantity]], cpu[[quantity]], 1e-8, quantity)
      }
    }
  }
})
test_that("NNGP GP Laplace kernel fits a small binary spatial problem", {
  skip_on_cran()
  # Exercises the Vecchia (nearest-neighbour) GP Laplace path end to end on a
  # 100-location binary problem: the marginal is finite, the mode has the
  # fixed-effect + per-location field length, and the inner Newton makes progress.
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

  result <- tulpa:::cpp_laplace_fit_gp(
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
  expect_equal(length(result$mode), 1L + n_obs)   # intercept + one field value / obs
  expect_true(result$n_iter > 0)
})
