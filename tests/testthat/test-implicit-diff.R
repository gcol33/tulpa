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

# --- gcol33/tulpa#435: the value and the gradient are the Laplace marginal ----
#
# The finite-difference test above compares this entry's gradient against a
# finite difference of this entry's own value, so it certifies internal
# consistency and nothing more: with the GMRF prior normalizer 0.5 log|Q(theta)|
# dropped from BOTH, it passes while neither quantity is the Laplace
# log-marginal. The tests below bind each of them to something outside the
# function.

.impl_fixture <- function(seed = 42L, n_obs = 100L) {
  set.seed(seed)
  coords <- cbind(runif(n_obs), runif(n_obs))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.2, 0.5), cutoff = 0.05)
  fem  <- fmesher::fm_fem(mesh)
  A    <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  G1   <- as(fem$g1, "CsparseMatrix")
  w    <- rnorm(mesh$n, 0, 0.3); w <- w - mean(w)
  list(n_obs = n_obs, n_mesh = mesh$n, A = A, G1 = G1,
       C0_diag = Matrix::diag(fem$c0),
       y = rbinom(n_obs, 1, plogis(-0.5 + as.numeric(A %*% w))),
       X = matrix(1, nrow = n_obs, ncol = 1))
}

.impl_grad <- function(f, log_range, log_sigma) {
  cpp_spde_laplace_gradient(
    y = as.integer(f$y), n_trials = as.integer(rep(1L, f$n_obs)), X = f$X,
    A_x = f$A@x, A_i = f$A@i, A_p = f$A@p,
    n_obs = f$n_obs, n_mesh = f$n_mesh, C0_diag = f$C0_diag,
    G1_x = f$G1@x, G1_i = f$G1@i, G1_p = f$G1@p,
    log_range = log_range, log_sigma = log_sigma, nu = 1.0, family = "binomial")
}

.IMPL_CELLS <- list(c(log(0.2), log(0.5)), c(log(0.2), log(1.1)),
                    c(log(0.3), log(0.5)), c(log(0.3), log(1.1)),
                    c(log(0.6), log(0.5)), c(log(0.6), log(1.1)))

test_that("the implicit-diff marginal is the same number the nested SPDE grid reports", {
  skip_if_not_installed("fmesher")
  f <- .impl_fixture()
  for (cell in .IMPL_CELLS) {
    got <- .impl_grad(f, cell[1], cell[2])$log_marginal
    nl  <- cpp_nested_laplace_spde(
      y = as.numeric(f$y), n_trials = as.integer(rep(1L, f$n_obs)), X = f$X,
      re_idx = rep(0, f$n_obs), n_re_groups = 0L, sigma_re = 1.0,
      A_x = f$A@x, A_i = f$A@i, A_p = f$A@p,
      n_obs = f$n_obs, n_mesh = f$n_mesh, C0_diag = f$C0_diag,
      G1_x = f$G1@x, G1_i = f$G1@i, G1_p = f$G1@p,
      range_grid = exp(cell[1]), sigma_grid = exp(cell[2]),
      nu = 1.0, family = "binomial")
    expect_equal(got, nl$log_marginal[1], tolerance = 1e-9,
                 info = sprintf("log_range = %g, log_sigma = %g",
                                cell[1], cell[2]))
  }
})

test_that("the normalizer it carries is 0.5 log|Q| of the R-assembled precision", {
  skip_if_not_installed("fmesher")
  # .spde_precision_Q is the R mirror of the FEM assembly, so this reads the
  # term off a matrix built outside the C++ operator chain.
  f <- .impl_fixture()
  spatial <- list(n_mesh = f$n_mesh, C0_diag = f$C0_diag, G = f$G1, nu = 1)
  for (cell in .IMPL_CELLS) {
    res <- .impl_grad(f, cell[1], cell[2])
    kt  <- tulpa:::.spde_kappa_tau(exp(cell[1]), exp(cell[2]), 1)
    Q   <- tulpa:::.spde_precision_Q(spatial, kt$kappa, kt$tau_spde)
    expect_equal(res$half_log_det_Q,
                 0.5 * as.numeric(Matrix::determinant(Q, logarithm = TRUE)$modulus),
                 tolerance = 1e-9,
                 info = sprintf("log_range = %g, log_sigma = %g",
                                cell[1], cell[2]))
  }
})

test_that("the dropped term is large and moves with the hyperparameters", {
  skip_if_not_installed("fmesher")
  # Without this the two tests above would be vacuous. Over the grid they run
  # on, 0.5 log|Q| spans more than 200 nats and is not constant along either
  # axis -- which is what makes the marginal monotone in sigma when it is
  # missing (src/spde_logdet.h), and what a NUTS chain driven by the gradient
  # alone is biased by.
  f  <- .impl_fixture()
  hl <- vapply(.IMPL_CELLS,
               function(cell) .impl_grad(f, cell[1], cell[2])$half_log_det_Q,
               numeric(1))
  expect_gt(diff(range(hl)), 100)

  # Along log_sigma the closed form is exact: Q is proportional to tau^2 and
  # tau to 1 / sigma, so d(0.5 log|Q|) / d(log_sigma) = -n_mesh at every cell.
  # That constant IS the term the gradient's sigma channel restores.
  for (lr in c(log(0.2), log(0.3), log(0.6))) {
    a <- .impl_grad(f, lr, log(0.5))$half_log_det_Q
    b <- .impl_grad(f, lr, log(1.1))$half_log_det_Q
    expect_equal((b - a) / (log(1.1) - log(0.5)), -f$n_mesh,
                 tolerance = 1e-9, info = sprintf("log_range = %g", lr))
  }
})

test_that("value and gradient still agree under a tighter finite difference", {
  skip_if_not_installed("fmesher")
  # Kept as before: this certifies that the value and the gradient are of ONE
  # function. It is the tests above that say WHICH function.
  f <- .impl_fixture()
  lr <- log(0.3); ls <- log(0.5); eps <- 2e-3
  res <- .impl_grad(f, lr, ls)
  fd_r <- (.impl_grad(f, lr + eps, ls)$log_marginal -
           .impl_grad(f, lr - eps, ls)$log_marginal) / (2 * eps)
  fd_s <- (.impl_grad(f, lr, ls + eps)$log_marginal -
           .impl_grad(f, lr, ls - eps)$log_marginal) / (2 * eps)
  expect_equal(res$grad_log_range, fd_r, tolerance = 1e-3)
  expect_equal(res$grad_log_sigma, fd_s, tolerance = 1e-3)
})

test_that("the entry refuses a nu its analytic dQ/dtheta is not written for", {
  skip_if_not_installed("fmesher")
  f <- .impl_fixture()
  expect_error(
    cpp_spde_laplace_gradient(
      y = as.integer(f$y), n_trials = as.integer(rep(1L, f$n_obs)), X = f$X,
      A_x = f$A@x, A_i = f$A@i, A_p = f$A@p,
      n_obs = f$n_obs, n_mesh = f$n_mesh, C0_diag = f$C0_diag,
      G1_x = f$G1@x, G1_i = f$G1@i, G1_p = f$G1@p,
      log_range = log(0.3), log_sigma = log(0.5), nu = 2.0,
      family = "binomial"),
    "nu = 1", fixed = TRUE)
})
