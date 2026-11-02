# An intrinsic field's level is a HARD constraint on the Laplace path
# (gcol33/tulpa#1053).
#
# The constant direction of an ICAR / BYM2-structured / RW field aliases the
# intercept. The model puts the level in the intercept, where the intercept's
# own prior acts, so the inner mode is the maximiser of the penalized objective
# on the set a'u = 0 (a the level weights) and the Laplace expansion runs over
# that set. Solving the augmented problem (the pin (tau / J) (1'u)^2 left as a
# soft prior on the level) and folding the field's mean into the intercept
# afterwards is a different model: one whose intercept prior is wider by the
# pin's variance 1 / (tau J). Under a flat intercept prior the two coincide;
# under N(0, 2.5^2) the folded point sat 2.4e-06 off this mode with a
# subspace score of 2.9e-04, and its log-marginal 0.0031 off.
#
# The arbiter below is written from the definition and owes nothing to the
# engine's augmentation: the prior precision is assembled whole, the mode is a
# Newton solve on null(a), and the log-marginal is
#   log p(y | x) + log p(x | a'x = 0) - 0.5 log|B' H B| + (n_x - 1)/2 log(2 pi)
# with B an orthonormal basis of null(a) and the conditioned prior density
#   log N(x; 0, P^-1) - log N(0; 0, a' P^-1 a) - 0.5 log(a'a)
# taken with respect to Lebesgue measure on the subspace.

skip_on_cran()

.rook <- function(nr, nc) {
  n <- nr * nc; W <- matrix(0, n, n)
  id <- function(r, c) (r - 1) * nc + c
  for (r in seq_len(nr)) for (c in seq_len(nc)) {
    if (r < nr) W[id(r, c), id(r + 1, c)] <- W[id(r + 1, c), id(r, c)] <- 1
    if (c < nc) W[id(r, c), id(r, c + 1)] <- W[id(r, c + 1), id(r, c)] <- 1
  }
  W
}

# Exact constrained Laplace reference for a binomial model with latent
# x = [beta | field (| theta)], eta = M x, prior precision P (full rank), and
# the one constraint row `a` over x.
.constrained_ref <- function(y, ntr, M, P, a) {
  n_x <- ncol(M)
  B <- qr.Q(qr(cbind(a, diag(n_x))))[, -1]
  score <- function(th) {
    e <- as.numeric(M %*% th)
    as.numeric(crossprod(M, y - ntr * plogis(e))) - as.numeric(P %*% th)
  }
  hess <- function(th) {
    pr <- plogis(as.numeric(M %*% th))
    crossprod(M, (ntr * pr * (1 - pr)) * M) + P
  }
  th <- rep(0, n_x)
  for (it in seq_len(200)) {
    st <- as.numeric(B %*% solve(crossprod(B, hess(th) %*% B),
                                 crossprod(B, score(th))))
    th <- th + st
    if (max(abs(st)) < 1e-13) break
  }
  ll <- sum(dbinom(y, ntr, plogis(as.numeric(M %*% th)), log = TRUE))
  ldP <- as.numeric(determinant(P)$modulus)
  s_a <- as.numeric(crossprod(a, solve(P, a)))
  lp <- 0.5 * ldP - 0.5 * n_x * log(2 * pi) - 0.5 * sum(th * (P %*% th)) -
    dnorm(0, 0, sqrt(s_a), log = TRUE) - 0.5 * log(sum(a^2))
  lm <- ll + lp - 0.5 * as.numeric(determinant(crossprod(B, hess(th) %*% B))$modulus) +
    0.5 * (n_x - 1) * log(2 * pi)
  list(theta = th, log_marginal = lm,
       tscore = function(x) as.numeric(crossprod(B, score(x))))
}

# The augmented ICAR precision of a (possibly disconnected) adjacency: per
# component c, w_c (Q_c + 1 1' / J_c) times the field precision.
.icar_aug_precision <- function(W, tau, w = rep(1, nrow(W))) {
  g <- .components(W)
  Q <- diag(rowSums(W)) - W
  for (cm in unique(g)) {
    i <- which(g == cm)
    Q[i, i] <- Q[i, i] + 1 / length(i)
  }
  tau * (sqrt(w) %o% sqrt(w)) * Q
}

# Connected components of an adjacency matrix by breadth-first search.
.components <- function(W) {
  n <- nrow(W); comp <- rep(0L, n); k <- 0L
  for (s in seq_len(n)) {
    if (comp[s]) next
    k <- k + 1L; comp[s] <- k; queue <- s
    while (length(queue)) {
      v <- queue[1]; queue <- queue[-1]
      nb <- which(W[v, ] != 0 & comp == 0L)
      comp[nb] <- k; queue <- c(queue, nb)
    }
  }
  comp
}

.tau_b <- function() 1 / default_sigma_beta()^2

test_that("the ICAR single fit is the constrained mode and evidence (dense)", {
  set.seed(1053)
  W <- .rook(5, 5); J <- nrow(W); N <- 300L
  unit <- sample.int(J, N, replace = TRUE)
  X <- cbind(1, rnorm(N), rnorm(N))
  u <- rnorm(J, 0, 0.7); u <- u - mean(u)
  y <- as.numeric(rbinom(N, 1, plogis(as.numeric(X %*% c(-0.4, 0.8, -0.5)) + u[unit])))
  adj <- tulpa:::adjacency_to_csr_tulpa(W)
  fit <- tulpa:::cpp_laplace_fit_spatial(
    y = y, n = rep(1L, N), X = X, re_idx = rep(0, N), n_re_groups = 0L,
    sigma_re = 1, spatial_idx = as.integer(unit), n_spatial_units = J,
    adj_row_ptr = adj$row_ptr, adj_col_idx = adj$col_idx,
    n_neighbors = adj$n_neighbors, tau_spatial = 1.3, family = "binomial",
    max_iter = 300L, tol = 1e-12)

  Z <- matrix(0, N, J); Z[cbind(seq_len(N), unit)] <- 1
  P <- as.matrix(Matrix::bdiag(diag(.tau_b(), 3), .icar_aug_precision(W, 1.3)))
  ref <- .constrained_ref(y, rep(1, N), cbind(X, Z), P, c(0, 0, 0, rep(1, J)))

  expect_true(fit$converged)
  expect_lt(max(abs(fit$mode - ref$theta)), 1e-9)
  expect_lt(max(abs(ref$tscore(fit$mode))), 1e-8)
  expect_equal(fit$log_marginal, ref$log_marginal, tolerance = 1e-9)
})

test_that("the ICAR fit on the sparse joint route is the same constrained model", {
  # 15 x 15 lattice: 228 latents, past SPARSE_THRESHOLD, so the registry fit
  # runs on the sparse joint Newton, and the 225-node component is past the
  # densify cutoff, so its pin is folded at solve time (block-Schur) and the
  # constraint's columns are solved through the same fold.
  set.seed(10531)
  W <- .rook(15, 15); J <- nrow(W); N <- 600L
  unit <- sample.int(J, N, replace = TRUE)
  X <- cbind(1, rnorm(N))
  u <- rnorm(J, 0, 0.6); u <- u - mean(u)
  y <- as.numeric(rbinom(N, 1, plogis(0.3 + 0.7 * X[, 2] + u[unit])))
  adj <- tulpa:::adjacency_to_csr_tulpa(W)
  tau <- 0.8
  ns <- tulpa:::cpp_nested_laplace_icar(
    y = y, n = rep(1L, N), X = X, re_idx = rep(0, N), n_re_groups = 0L,
    sigma_re = 1, spatial_idx = as.integer(unit), n_spatial_units = J,
    adj_row_ptr = adj$row_ptr, adj_col_idx = adj$col_idx,
    n_neighbors = adj$n_neighbors, tau_grid = tau, family = "binomial",
    max_iter = 300L, tol = 1e-12)

  Z <- matrix(0, N, J); Z[cbind(seq_len(N), unit)] <- 1
  P <- as.matrix(Matrix::bdiag(diag(.tau_b(), 2), .icar_aug_precision(W, tau)))
  ref <- .constrained_ref(y, rep(1, N), cbind(X, Z), P, c(0, 0, rep(1, J)))

  mode <- as.numeric(ns$modes[1, ])
  expect_lt(max(abs(mode - ref$theta)), 1e-8)
  expect_lt(max(abs(ref$tscore(mode))), 1e-7)
  expect_equal(as.numeric(ns$log_marginal)[1], ref$log_marginal, tolerance = 1e-9)
})

test_that("a weighted BYM2 level on a disconnected map is the constrained model", {
  # Two components scaled by different node precisions: the level row is the
  # node_prec-weighted sum, and the second component's contrast stays carried
  # by its pin. The evidence reference conditions the whole augmented prior on
  # that row numerically, so it checks the closed form the engine uses.
  set.seed(10532)
  W1 <- .rook(3, 3); W2 <- .rook(4, 4)
  W <- as.matrix(Matrix::bdiag(W1, W2)); J <- nrow(W); N <- 260L
  w <- c(rep(0.7, 9), rep(1.6, 16))
  unit <- sample.int(J, N, replace = TRUE)
  X <- cbind(1, rnorm(N))
  y <- as.numeric(rbinom(N, 1, plogis(-0.2 + 0.5 * X[, 2] +
                                        rnorm(J, 0, 0.5)[unit])))
  adj <- tulpa:::adjacency_to_csr_tulpa(W)
  sigma <- 0.9; rho <- 0.6; sf <- 1.2
  fit <- tulpa:::cpp_laplace_fit_bym2(
    y = y, n = rep(1L, N), X = X, re_idx = rep(0, N), n_re_groups = 0L,
    sigma_re = 1, spatial_idx = as.integer(unit), n_spatial_units = J,
    adj_row_ptr = adj$row_ptr, adj_col_idx = adj$col_idx,
    n_neighbors = adj$n_neighbors, sigma_spatial = sigma, rho = rho,
    scale_factor = sf, family = "binomial", max_iter = 300L, tol = 1e-12,
    node_prec_nullable = w)

  Z <- matrix(0, N, J); Z[cbind(seq_len(N), unit)] <- 1
  M <- cbind(X, sigma * sqrt(rho) * sf * Z, sigma * sqrt(1 - rho) * Z)
  P <- as.matrix(Matrix::bdiag(diag(.tau_b(), 2), .icar_aug_precision(W, 1, w),
                               diag(J)))
  ref <- .constrained_ref(y, rep(1, N), M, P, c(0, 0, w, rep(0, J)))

  expect_true(fit$converged)
  expect_lt(max(abs(fit$mode - ref$theta)), 1e-9)
  expect_lt(max(abs(ref$tscore(fit$mode))), 1e-8)
  expect_equal(fit$log_marginal, ref$log_marginal, tolerance = 1e-9)
})

test_that("the subspace debias walks the Gaussian conditioned on the level", {
  # On a Gaussian response the debias target along its surface is exactly the
  # inner Gaussian's marginal of x_S, so the draws' covariance is that of the
  # inner Gaussian conditioned on 1'u = 0. Unconditioned, the intercept and a
  # field node share the level's ridge: variance 2.18 here against 0.0178.
  set.seed(3)
  W <- .rook(1, 6); J <- nrow(W); N <- 60L
  unit <- sample.int(J, N, TRUE); X <- cbind(1, rnorm(N))
  y <- 0.3 + 0.5 * X[, 2] + rnorm(J)[unit] + rnorm(N)
  adj <- tulpa:::adjacency_to_csr_tulpa(W)
  tau <- 0.05
  set.seed(4)
  ns <- tulpa:::cpp_nested_laplace_icar(
    y = y, n = rep(1L, N), X = X, re_idx = rep(0, N), n_re_groups = 0L,
    sigma_re = 1, spatial_idx = as.integer(unit), n_spatial_units = J,
    adj_row_ptr = adj$row_ptr, adj_col_idx = adj$col_idx,
    n_neighbors = adj$n_neighbors, tau_grid = tau, family = "gaussian",
    phi = 1, max_iter = 300L, tol = 1e-12,
    debias = list(idx = c(1L, 3L), n_iter = 20000L, warmup = 2000L))
  expect_true(is.na(ns$debias_declined))
  Z <- outer(unit, seq_len(J), "==") * 1
  P <- as.matrix(Matrix::bdiag(diag(.tau_b(), 2), .icar_aug_precision(W, tau)))
  Sig <- solve(crossprod(cbind(X, Z)) + P)
  a <- c(0, 0, rep(1, J)); w <- Sig %*% a
  Sc <- (Sig - tcrossprod(w) / sum(a * w))[c(1, 3), c(1, 3)]
  expect_equal(diag(cov(ns$debias_draws_per_grid[[1]])), diag(Sc),
               tolerance = 0.1)
})
