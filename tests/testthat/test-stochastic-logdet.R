# test-stochastic-logdet.R
# Tests for stochastic Lanczos quadrature log-determinant

test_that("SLQ Gauss-quadrature log-det matches exact dense log-det", {
  n <- 50
  Q <- Matrix::bandSparse(n, k = c(0, 1), diag = list(
    rep(4, n), rep(-1, n - 1)
  ), symmetric = TRUE)
  Q_lower <- Matrix::tril(Q)

  # Exact log-det
  exact <- determinant(as.matrix(Q), logarithm = TRUE)$modulus[1]

  # Stochastic estimate (Gauss-quadrature SLQ weights)
  stoch <- cpp_stochastic_log_determinant(
    Q_x = Q_lower@x, Q_i = Q_lower@i, Q_p = Q_lower@p,
    n = n, n_probes = 80, n_lanczos = 45, seed = 42
  )

  rel_err <- abs(stoch - exact) / abs(exact)
  cat("\n  Exact log-det:", round(exact, 4), "\n")
  cat("  Stochastic:   ", round(stoch, 4), "\n")
  cat("  Relative error:", round(rel_err, 5), "\n")

  expect_true(is.finite(stoch))
  expect_true(stoch > 0)  # positive definite -> positive log-det
  # Gauss-quadrature SLQ is unbiased up to Monte-Carlo noise over probes; the
  # (n/m)log|T| average it replaced was biased and would miss this tolerance.
  expect_lt(rel_err, 0.03)
})

test_that("SLQ log-det is accurate for a dense SPD matrix", {
  set.seed(7)
  n <- 40
  A <- matrix(rnorm(n * n), n, n)
  S <- crossprod(A) + diag(n) * n  # SPD, well-conditioned
  S_sp <- methods::as(Matrix::Matrix(S, sparse = TRUE), "CsparseMatrix")
  S_lower <- Matrix::tril(S_sp)

  exact <- determinant(S, logarithm = TRUE)$modulus[1]
  stoch <- cpp_stochastic_log_determinant(
    Q_x = S_lower@x, Q_i = S_lower@i, Q_p = S_lower@p,
    n = n, n_probes = 120, n_lanczos = 38, seed = 11
  )

  rel_err <- abs(stoch - exact) / abs(exact)
  cat("\n  dense SPD exact:", round(exact, 4), "stoch:", round(stoch, 4),
      "rel_err:", round(rel_err, 5), "\n")
  expect_lt(rel_err, 0.05)
})

test_that("stochastic log-det works for larger matrix", {
  n <- 200
  Q <- Matrix::bandSparse(n, k = c(0, 1, 2), diag = list(
    rep(6, n), rep(-2, n - 1), rep(0.5, n - 2)
  ), symmetric = TRUE)
  Q_lower <- Matrix::tril(Q)

  exact <- determinant(as.matrix(Q), logarithm = TRUE)$modulus[1]

  stoch <- cpp_stochastic_log_determinant(
    Q_x = Q_lower@x, Q_i = Q_lower@i, Q_p = Q_lower@p,
    n = n, n_probes = 50, n_lanczos = 50, seed = 42
  )

  cat("\n  N=200 exact:", round(exact, 2), "stoch:", round(stoch, 2),
      "rel_err:", round(abs(stoch - exact) / abs(exact), 4), "\n")

  expect_true(is.finite(stoch))
})
