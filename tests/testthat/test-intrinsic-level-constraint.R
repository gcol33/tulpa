# The level of an intrinsic field is a HARD sum-to-zero constraint, and the
# fixed-effect covariance has to be read conditional on it (gcol33/tulpa#901).
#
# The inner solve augments the field's constant direction with a finite
# precision (tau / J per component, sum_to_zero.h) so its Hessian inverts, and
# centres the level into the intercept. The unconditioned inverse therefore
# carried the augmentation's 1 / (tau J) variance on the level into the
# intercept's standard error: on a 50-node ICAR with a gaussian response it read
# 0.145 (laplace) and 0.089 (nested_laplace) against an exact posterior SD of
# 0.033, which HMC reproduced. Conditioning by kriging on the constraint -- the
# correction a joint fit's retained block already used -- is exact for a
# gaussian likelihood, so the assertions below are against the closed form.

.lattice_icar <- function(nr = 5L, nc = 10L) {
  n <- nr * nc
  A <- matrix(0, n, n)
  id <- function(i, j) (j - 1L) * nr + i
  for (i in seq_len(nr)) for (j in seq_len(nc)) {
    if (i < nr) A[id(i, j), id(i + 1L, j)] <- A[id(i + 1L, j), id(i, j)] <- 1
    if (j < nc) A[id(i, j), id(i, j + 1L)] <- A[id(i, j + 1L), id(i, j)] <- 1
  }
  A
}

.icar_gaussian_data <- function(A, sd_field = 0.7, seed = 3L) {
  n <- nrow(A)
  R <- diag(rowSums(A)) - A
  e <- eigen(R, TRUE)
  set.seed(1)
  phi <- sd_field * e$vectors[, 1:(n - 1)] %*%
    (rnorm(n - 1) / sqrt(e$values[1:(n - 1)]))
  set.seed(seed)
  d <- data.frame(region = rep(seq_len(n), each = 3L), x = rnorm(3L * n))
  d$y <- 0.3 + 0.8 * d$x + phi[d$region] + rnorm(nrow(d), sd = 0.4)
  d
}

# Exact posterior SDs of (intercept, slope) under a hard sum-to-zero ICAR of
# variance sigma^2 and a gaussian likelihood of variance s2, with the engine's
# N(0, 100^2) fixed-effect prior.
.icar_exact_se <- function(A, d, sigma, s2) {
  n <- nrow(A)
  R <- diag(rowSums(A)) - A
  Z <- model.matrix(~ factor(region) - 1, d)
  B <- qr.Q(qr(cbind(1, diag(n))))[, 2:n]      # basis of the sum-zero subspace
  W <- cbind(1, d$x, Z %*% B)
  P <- matrix(0, ncol(W), ncol(W))
  P[-(1:2), -(1:2)] <- t(B) %*% R %*% B / sigma^2
  diag(P)[1:2] <- 1 / 100^2
  sqrt(diag(solve(crossprod(W) / s2 + P))[1:2])
}

test_that("the constrained Schur complement is the sum-zero subspace's", {
  # .schur_H_beta(constr =) against the brute-force integral of the latent over
  # the constrained subspace, on a small dense problem.
  set.seed(4)
  n_u <- 6L
  A <- .lattice_icar(2L, 3L)
  X <- cbind(1, rnorm(24))
  D <- Matrix::Matrix(model.matrix(~ factor(rep(1:n_u, 4)) - 1), sparse = TRUE)
  Q <- tulpa:::.icar_precision_Q(list(adjacency = A))
  Wt <- runif(24, 0.5, 2)
  H_c <- tulpa:::.schur_H_beta(X, D, Q, Wt, constr = matrix(1, n_u, 1L))

  Dm <- as.matrix(D)
  B <- qr.Q(qr(cbind(1, diag(n_u))))[, 2:n_u]
  DB <- Dm %*% B
  ref <- crossprod(X, Wt * X) - crossprod(X, Wt * DB) %*%
    solve(crossprod(DB, Wt * DB) + t(B) %*% as.matrix(Q) %*% B,
          crossprod(DB, Wt * X))
  expect_equal(unname(H_c), unname(ref), tolerance = 1e-10)
  # Without the constraint the intercept is looser: the augmentation's level
  # variance was in it.
  H_u <- tulpa:::.schur_H_beta(X, D, Q, Wt)
  expect_gt(solve(H_u)[1, 1], solve(H_c)[1, 1])
})

test_that("an intrinsic block's driver names its constraint group", {
  skip_on_cran()
  A <- .lattice_icar(3L, 4L)
  d <- .icar_gaussian_data(A)
  f <- suppressWarnings(tulpa(y ~ x + spatial(region), data = d, phi = 0.16,
                              mode = "nested_laplace",
                              spatial = spatial_car(A, group_var = "region"),
                              control = list(n_threads = 1L)))
  # p = 2 fixed effects, then the 12-node field.
  expect_identical(f$constraint_cols, list(3:14))
})

test_that("laplace and nested_laplace intercept SEs match the exact ICAR posterior (#901)", {
  skip_on_cran()
  A <- .lattice_icar()
  d <- .icar_gaussian_data(A)
  sp <- spatial_car(A, group_var = "region")

  # mode = "laplace" conditions on the field at tau = 1 (sigma = 1).
  fl <- suppressWarnings(tulpa(y ~ x + spatial(region), data = d, phi = 0.16,
                               mode = "laplace", spatial = sp))
  ex1 <- .icar_exact_se(A, d, sigma = 1, s2 = 0.16)
  expect_equal(unname(summary(fl)[1:2, 2]), ex1, tolerance = 0.02)

  # nested_laplace integrates tau. The exact intercept SD barely moves with the
  # field scale here (0.0327 at the generating sigma = 0.7; the issue measured
  # it flat over sigma in [0.3, 3]), so the integrated SE sits on it. Before the
  # fix it read 0.089.
  fn <- suppressWarnings(tulpa(y ~ x + spatial(region), data = d, phi = 0.16,
                               mode = "nested_laplace", spatial = sp,
                               control = list(n_threads = 1L)))
  ex07 <- .icar_exact_se(A, d, sigma = 0.7, s2 = 0.16)
  expect_equal(summary(fn)[1, 2], ex07[1], tolerance = 0.1)
  expect_equal(summary(fn)[2, 2], ex07[2], tolerance = 0.1)
})

test_that("a BYM2 laplace fit conditions only the structured component", {
  skip_on_cran()
  A <- .lattice_icar(4L, 5L)
  d <- .icar_gaussian_data(A)
  f <- suppressWarnings(tulpa(y ~ x + spatial(region), data = d, phi = 0.16,
                              mode = "laplace",
                              spatial = spatial_bym2(A, group_var = "region")))
  # Exact at the conditional kernel's sigma = 1, rho = 0.5: phi carries the
  # hard constraint, theta is iid and unconstrained.
  n <- nrow(A)
  R <- diag(rowSums(A)) - A
  sf <- spatial_bym2(A, group_var = "region")$scale_factor
  Z <- model.matrix(~ factor(region) - 1, d)
  B <- qr.Q(qr(cbind(1, diag(n))))[, 2:n]
  W <- cbind(1, d$x, sqrt(0.5) * sf * Z %*% B, sqrt(0.5) * Z)
  k <- ncol(W)
  P <- diag(c(1e-4, 1e-4, rep(0, n - 1), rep(1, n)))
  P[2 + seq_len(n - 1), 2 + seq_len(n - 1)] <- t(B) %*% R %*% B
  ex <- unname(sqrt(diag(solve(crossprod(W) / 0.16 + P))[1:2]))
  expect_equal(unname(summary(f)[1:2, 2]), ex, tolerance = 0.02)
})

test_that("the RW1 intercept SE is conditioned the same way", {
  skip_on_cran()
  # The temporal intrinsic blocks carry the same flag: an RW1 over 20 times
  # with a gaussian response against its closed form at the fitted tau.
  set.seed(9)
  T_ <- 20L
  tt <- rep(seq_len(T_), each = 4L)
  x <- rnorm(length(tt))
  f_t <- cumsum(rnorm(T_, 0, 0.3)); f_t <- f_t - mean(f_t)
  d <- data.frame(tt = tt, x = x, y = 0.5 + 0.4 * x + f_t[tt] + rnorm(length(tt), 0, 0.3))
  fit <- suppressWarnings(tulpa(y ~ x, data = d, phi = 0.09,
                                temporal = temporal_rw1(time_var = "tt"),
                                control = list(n_threads = 1L)))
  expect_identical(fit$constraint_cols, list(2L + seq_len(T_)))
  # Unconditioned, the level's augmentation variance 1 / (tau T) sat in the
  # intercept; conditioned, the intercept SE is the data's: close to
  # sqrt(0.09 / n) for a balanced design.
  expect_lt(summary(fit)[1, 2], 1.5 * sqrt(0.09 / nrow(d)))
})
