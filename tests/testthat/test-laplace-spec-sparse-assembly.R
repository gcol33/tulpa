# A sparse spec Laplace solve over a [beta | RE] latent layout assembles its
# Hessian straight into a SparseHessianBuilder on the structural pattern
# (build_spec_hessian_pattern); a dense one assembles into the row-major scratch
# matrix. The two are one Newton solve on two containers, so a problem driven
# through both (sparse_override = +1 / -1) has to give the same answer.

sparse_asm_fixture <- function(seed, G1, G2, per = 3L, empty = 2L) {
  set.seed(seed)
  g1 <- rep(seq_len(G1), each = per)
  N <- length(g1)
  g2 <- sample.int(G2, N, replace = TRUE)
  x <- rnorm(N)
  X <- cbind(1, x, rnorm(N))
  b <- matrix(rnorm(2L * G1, 0, c(0.6, 0.3)), G1, 2L, byrow = TRUE)
  y <- as.numeric(X %*% c(0.4, -0.3, 0.6)) + b[g1, 1L] + b[g1, 2L] * x +
    rnorm(G2, 0, 0.5)[g2] + rnorm(N, 0, 0.8)
  # The correlated term declares `empty` groups no observation reaches: their
  # block carries the prior alone.
  terms <- list(
    list(group_idx = g1, n_groups = G1 + empty, n_coefs = 2L,
         sigma = c(0.6, 0.3), correlated = TRUE,
         slope_mat = matrix(x, ncol = 1L), chol_raw = 0.4),
    list(group_idx = g2, n_groups = G2, n_coefs = 1L, sigma = 0.5,
         correlated = FALSE))
  list(y = y, X = X, terms = terms,
       n_x = ncol(X) + 2L * (G1 + empty) + G2)
}

sparse_asm_fit <- function(fx, override) {
  cpp_laplace_spec_test_multi_re(
    y = fx$y, X = fx$X, re_terms = fx$terms, sigma_beta = 10, phi = 0.8,
    max_iter = 200L, tol = 1e-12, n_threads = 1L,
    sparse_override = override, store_Q = TRUE, return_re_cov = TRUE)
}

csc_lower_to_dense <- function(f, n) {
  M <- matrix(0, n, n)
  for (j in seq_len(n)) {
    k <- seq.int(f$Q_p[j] + 1L, length.out = f$Q_p[j + 1L] - f$Q_p[j])
    M[f$Q_i[k] + 1L, j] <- f$Q_x[k]
  }
  M
}

expect_assemblies_agree <- function(dn, sp, n_x) {
  expect_true(dn$converged)
  expect_true(sp$converged)
  expect_equal(sp$mode, dn$mode, tolerance = 1e-10)
  expect_equal(sp$log_marginal, dn$log_marginal, tolerance = 1e-10)
  expect_equal(sp$log_det_Q, dn$log_det_Q, tolerance = 1e-10)
  expect_equal(sp$re_cov, dn$re_cov, tolerance = 1e-10)
  expect_equal(csc_lower_to_dense(sp, n_x), csc_lower_to_dense(dn, n_x),
               tolerance = 1e-12)
}

test_that("sparse and dense assembly agree at a sparse size", {
  fx <- sparse_asm_fixture(11L, G1 = 110L, G2 = 9L)
  expect_gte(fx$n_x, 200L)
  dn <- sparse_asm_fit(fx, -1L)
  sp <- sparse_asm_fit(fx, 1L)
  expect_assemblies_agree(dn, sp, fx$n_x)
  # At this size the automatic choice is the sparse assembly.
  expect_identical(sparse_asm_fit(fx, 0L), sp)
})

test_that("sparse assembly forced at a dense size agrees with the dense one", {
  fx <- sparse_asm_fixture(12L, G1 = 14L, G2 = 5L)
  expect_lt(fx$n_x, 200L)
  expect_assemblies_agree(sparse_asm_fit(fx, -1L), sparse_asm_fit(fx, 1L),
                          fx$n_x)
})

test_that("two processes sharing a random effect assemble identically", {
  set.seed(13L)
  N <- 240L; G <- 30L
  X1 <- cbind(1, rnorm(N)); X2 <- cbind(1, rnorm(N), rnorm(N))
  re_idx <- sample.int(G, N, replace = TRUE)
  u <- rnorm(G, 0, 0.5)
  y1 <- as.numeric(X1 %*% c(0.5, -0.7)) + u[re_idx] + rnorm(N, 0, 0.6)
  y2 <- as.numeric(X2 %*% c(-0.2, 0.9, 0.4)) + u[re_idx] + rnorm(N, 0, 0.9)
  for (into1 in c(TRUE, FALSE)) {
    fit <- function(override) cpp_laplace_spec_test_gaussian2p(
      y1 = y1, y2 = y2, X1 = X1, X2 = X2,
      offset1 = numeric(0), offset2 = numeric(0),
      re_idx = re_idx, n_re_groups = G, sigma_re = 0.5, sigma_beta = 10,
      phi1 = 0.6, phi2 = 0.9, re_into_proc0 = TRUE, re_into_proc1 = into1,
      max_iter = 200L, tol = 1e-12, n_threads = 1L,
      sparse_override = override)
    dn <- fit(-1L); sp <- fit(1L)
    expect_true(sp$converged)
    expect_equal(sp$mode, dn$mode, tolerance = 1e-10)
    expect_equal(sp$log_marginal, dn$log_marginal, tolerance = 1e-10)
    expect_equal(sp$log_det_Q, dn$log_det_Q, tolerance = 1e-10)
  }
})

test_that("a crossed, correlated batch at a sparse size writes inside its pattern", {
  skip_on_cran()
  set.seed(14L)
  G <- 120L; H <- 7L
  grp <- rep(seq_len(G), each = 3L); n <- length(grp)
  h <- sample.int(H, n, replace = TRUE)
  x <- rnorm(n)
  y <- rbinom(n, 1L, plogis(-0.5 + 0.6 * x + rnorm(G, 0, 0.7)[grp] +
                              rnorm(H, 0, 0.4)[h]))
  X <- cbind(1, x)
  re_of <- function(s) list(
    list(idx = grp, n_groups = G, n_coefs = 2L, Z = cbind(1, x),
         L = matrix(c(s, 0.1, 0, 0.3), 2L)),
    list(idx = h, n_groups = H, n_coefs = 1L, sigma = 0.4))
  sigs <- c(0.4, 0.7, 1.1)
  one <- vapply(sigs, function(s) tulpa_laplace(
    y = y, n_trials = rep(1L, n), X = X, re_list = re_of(s),
    family = "binomial", return_hessian = FALSE, tol = 1e-8)$log_marginal,
    numeric(1))
  args <- .laplace_multi_re_model_args(y, rep(1L, n), X, re_of(1),
                                       "binomial", 1)
  # The batch runs under a HessianPatternGuard, which raises on any nonzero
  # write that misses the structural pattern.
  bat <- do.call(cpp_laplace_log_marginal_multi_re_batch, c(args, list(
    re_sigma_batch = lapply(sigs, function(s) .laplace_multi_re_sigma_list(
      re_of(s))), tol = 1e-8)))
  expect_identical(bat, one)
})
