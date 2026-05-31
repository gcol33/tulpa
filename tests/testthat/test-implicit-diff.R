# test-implicit-diff.R
# Tests for implicit differentiation of SPDE Laplace log-marginal

test_that("cpp_spde_laplace_gradient returns finite gradients", {
  skip_if_not_installed("fmesher")

  set.seed(42)
  n_obs <- 100
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.2, 0.5), cutoff = 0.05)
  fem <- fmesher::fm_fem(mesh)
  A <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  G1 <- as(fem$g1, "CsparseMatrix")
  C0_diag <- Matrix::diag(fem$c0)

  w <- rnorm(mesh$n, 0, 0.3); w <- w - mean(w)
  eta <- -0.5 + as.numeric(A %*% w)
  y <- rbinom(n_obs, 1, plogis(eta))
  X <- matrix(1, nrow = n_obs, ncol = 1)

  result <- cpp_spde_laplace_gradient(
    y = as.integer(y), n_trials = as.integer(rep(1L, n_obs)),
    X = X,
    A_x = A@x, A_i = A@i, A_p = A@p,
    n_obs = n_obs, n_mesh = mesh$n,
    C0_diag = C0_diag,
    G1_x = G1@x, G1_i = G1@i, G1_p = G1@p,
    log_range = log(0.3), log_sigma = log(0.5),
    nu = 1.0, family = "binomial"
  )

  expect_true(is.finite(result$log_marginal))
  expect_true(is.finite(result$grad_log_range))
  expect_true(is.finite(result$grad_log_sigma))
  expect_true(result$converged)
})

test_that("gradient points in correct direction (finite difference check)", {
  skip_if_not_installed("fmesher")

  set.seed(42)
  n_obs <- 100
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.2, 0.5), cutoff = 0.05)
  fem <- fmesher::fm_fem(mesh)
  A <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  G1 <- as(fem$g1, "CsparseMatrix")
  C0_diag <- Matrix::diag(fem$c0)

  w <- rnorm(mesh$n, 0, 0.3); w <- w - mean(w)
  eta <- -0.5 + as.numeric(A %*% w)
  y <- rbinom(n_obs, 1, plogis(eta))
  X <- matrix(1, nrow = n_obs, ncol = 1)

  common_args <- list(
    y = as.integer(y), n_trials = as.integer(rep(1L, n_obs)),
    X = X, A_x = A@x, A_i = A@i, A_p = A@p,
    n_obs = n_obs, n_mesh = mesh$n,
    C0_diag = C0_diag,
    G1_x = G1@x, G1_i = G1@i, G1_p = G1@p,
    nu = 1.0, family = "binomial"
  )

  # Evaluate at center
  lr0 <- log(0.3); ls0 <- log(0.5)
  res0 <- do.call(cpp_spde_laplace_gradient,
                   c(common_args, list(log_range = lr0, log_sigma = ls0)))

  # Finite difference for log_range
  eps <- 0.01
  res_r_plus <- do.call(cpp_spde_laplace_gradient,
                         c(common_args, list(log_range = lr0 + eps, log_sigma = ls0)))
  res_r_minus <- do.call(cpp_spde_laplace_gradient,
                          c(common_args, list(log_range = lr0 - eps, log_sigma = ls0)))
  fd_range <- (res_r_plus$log_marginal - res_r_minus$log_marginal) / (2 * eps)

  # Finite difference for log_sigma
  res_s_plus <- do.call(cpp_spde_laplace_gradient,
                         c(common_args, list(log_range = lr0, log_sigma = ls0 + eps)))
  res_s_minus <- do.call(cpp_spde_laplace_gradient,
                          c(common_args, list(log_range = lr0, log_sigma = ls0 - eps)))
  fd_sigma <- (res_s_plus$log_marginal - res_s_minus$log_marginal) / (2 * eps)

  cat("\n  Implicit diff:  range=", round(res0$grad_log_range, 3),
      " sigma=", round(res0$grad_log_sigma, 3), "\n")
  cat("  Finite diff:    range=", round(fd_range, 3),
      " sigma=", round(fd_sigma, 3), "\n")

  expect_true(is.finite(fd_range))
  expect_true(is.finite(fd_sigma))

  # The full Takahashi selected inversion makes the trace term exact, so the
  # implicit gradient agrees with the central finite difference of the
  # log-marginal. The diagonal-only trace term this replaced missed the
  # off-diagonal smoothing contributions and would fail this tolerance.
  expect_equal(res0$grad_log_range, fd_range, tolerance = 0.05)
  expect_equal(res0$grad_log_sigma, fd_sigma, tolerance = 0.05)
})
