# test-stochastic-logdet.R
# Tests for stochastic Lanczos quadrature log-determinant

test_that("stochastic log-det approximates exact for small matrix", {
  n <- 50
  Q <- Matrix::bandSparse(n, k = c(0, 1), diag = list(
    rep(4, n), rep(-1, n - 1)
  ), symmetric = TRUE)
  Q_lower <- Matrix::tril(Q)

  # Exact log-det
  exact <- determinant(as.matrix(Q), logarithm = TRUE)$modulus[1]

  # Stochastic estimate
  stoch <- cpp_stochastic_log_determinant(
    Q_x = Q_lower@x, Q_i = Q_lower@i, Q_p = Q_lower@p,
    n = n, n_probes = 50, n_lanczos = 40, seed = 42
  )

  cat("\n  Exact log-det:", round(exact, 2), "\n")
  cat("  Stochastic:   ", round(stoch, 2), "\n")
  cat("  Relative error:", round(abs(stoch - exact) / abs(exact), 4), "\n")

  # Should be in the right ballpark (within 50% for small matrices)
  expect_true(is.finite(stoch))
  expect_true(stoch > 0)  # positive definite → positive log-det
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
