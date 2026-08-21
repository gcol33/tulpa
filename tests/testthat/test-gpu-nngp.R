# test-gpu-nngp.R
# Tests for GPU-batched NNGP Laplace (Feature 6)
# Tests the batched computation path regardless of GPU availability.
# The GPU is optional — CPU fallback is always used if CUDA unavailable.

test_that("NNGP GP Laplace kernel fits a small binary spatial problem", {
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
  expect_equal(length(result$mode), 1L + n_obs)   # intercept + one field value / obs
  expect_true(result$n_iter > 0)
})

test_that("the batched-CUDA backend has one definition and says which one it is", {
  # gcol33/tulpa#396. `cuda_batched_cholesky` and its siblings were defined
  # TWICE, differently: `gpu_backend.h` compiled stubs returning FALSE in the
  # `#else` of `#ifdef TULPA_ENABLE_CUDA` -- which neither Makevars ever defined
  # -- while `gpu_nngp_laplace.h` included `gpu_cuda.h` directly and compiled the
  # real ones. Two `inline` definitions of the same entity across translation
  # units is an ODR violation: the linker keeps one COMDAT and discards the rest,
  # so whether CUDA ran at all was decided by link order, and nothing in the
  # package could report which had been built.
  #
  # `gpu_cuda.h` is now included from exactly one place and the choice is
  # reported, so this asserts BOTH halves of the fix.
  kind <- cpp_gpu_backend_kind()
  expect_type(kind, "character")
  expect_length(kind, 1L)
  expect_true(kind %in% c("cuda", "stub"))
  # The shipped default is to compile CUDA in and resolve it dynamically, so it
  # is used when a device is present and degrades to FALSE when it is not. That
  # needs no CUDA SDK at build time, which is what makes it expressible as a
  # default. Only an explicit TULPA_DISABLE_CUDA build reports "stub".
  expect_identical(kind, "cuda")

  # Compiled-in is a different question from usable-at-runtime, and the two are
  # deliberately separate calls: this one must answer without a GPU present.
  expect_type(cpp_gpu_available(), "logical")
  info <- cpp_gpu_info()
  expect_true(is.list(info))
  expect_true(all(c("available", "backend", "device_count") %in% names(info)))
  # A machine with no device must still report cleanly rather than error.
  expect_true(is.numeric(info$device_count) && info$device_count >= 0)
  if (!cpp_gpu_available()) expect_equal(info$device_count, 0)
})

test_that("the batched GPU Cholesky solve reproduces an independent solve", {
  # gpu_batched_cholesky_solve chains a batched Cholesky into a forward and a
  # back substitution. The factor passes between the three steps in one buffer,
  # and each step reads it under a different library's layout convention:
  # cuSOLVER writes it column-major, the CPU consumers read it row-major, cuBLAS
  # reads it column-major again. A mismatch there returns TRUE and gives finite,
  # plausible numbers -- reading the buffer's zero triangle solves against
  # diag(L) alone -- so scoring alpha against solve(C, c) is the only thing that
  # sees it.
  set.seed(9)
  k <- 6L
  batch <- 64L   # above the 50-matrix threshold the NNGP path dispatches at

  C_list <- lapply(seq_len(batch), function(b) {
    A <- matrix(rnorm(k * k), k, k)
    crossprod(A) + diag(k) * (0.5 + b / batch)
  })
  # Each row is one matrix flattened; the matrices are symmetric, so the
  # flattening order is immaterial and what is under test is the internal
  # handoff, not the input layout.
  C_flat <- t(vapply(C_list, as.numeric, numeric(k * k)))
  c_rhs <- matrix(rnorm(batch * k), batch, k)

  res <- cpp_gpu_batched_cholesky_solve(C_flat, c_rhs, k)
  expect_type(res$used_gpu, "logical")

  if (!cpp_gpu_available()) {
    expect_false(res$used_gpu)
    expect_true(all(is.na(res$alpha)))
  }
  skip_if_not(res$used_gpu, "no CUDA device: the batched GPU solve did not run")

  expected <- t(vapply(seq_len(batch),
                       function(b) solve(C_list[[b]], c_rhs[b, ]),
                       numeric(k)))
  scale <- max(abs(expected))
  expect_lt(max(abs(res$alpha - expected)), 1e-8 * scale)
})
