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

# --- gcol33/tulpa#444 -------------------------------------------------------

.csc_lower <- function(M) {
  n <- nrow(M)
  Q_p <- 0L; Q_i <- integer(0); Q_x <- numeric(0)
  for (j in seq_len(n)) {
    rows <- which(M[, j] != 0 & seq_len(n) >= j)
    Q_i <- c(Q_i, as.integer(rows - 1L))
    Q_x <- c(Q_x, M[rows, j])
    Q_p <- c(Q_p, length(Q_i))
  }
  list(x = Q_x, i = Q_i, p = as.integer(Q_p), n = n)
}

test_that("a Lanczos breakdown is the exact answer, not a definiteness error", {
  # H = c * I. Step 0 gives w = H q - alpha q = 0, so beta_0 = 0 and the
  # recurrence stops with the Krylov space exhausted -- the one case where the
  # quadrature is exact. Padding out to n_lanczos with zeros put m - 1 Ritz
  # values at 0, and the definiteness guard fired on them.
  for (n in c(5L, 40L)) {
    for (cval in c(2, 0.25)) {
      cs <- .csc_lower(diag(cval, n))
      got <- cpp_stochastic_log_determinant(cs$x, cs$i, cs$p, n,
                                            n_probes = 4L, n_lanczos = 10L,
                                            seed = 1L)
      expect_equal(got, n * log(cval), tolerance = 1e-10,
                   info = sprintf("n=%d c=%g", n, cval))
    }
  }
})

test_that("a repeated spectrum converges rather than erroring", {
  # Two distinct eigenvalues: the Krylov space is 2-dimensional whatever
  # n_lanczos says, so this breaks down at step 2 on every probe.
  n <- 30L
  d <- rep(c(1.5, 4.0), length.out = n)
  cs <- .csc_lower(diag(d, n))
  got <- cpp_stochastic_log_determinant(cs$x, cs$i, cs$p, n,
                                        n_probes = 60L, n_lanczos = 12L,
                                        seed = 7L)
  expect_equal(got, sum(log(d)), tolerance = 0.05 * abs(sum(log(d))))
})

test_that("the estimate stops moving once the Krylov space is exhausted", {
  # A spectrum with k distinct eigenvalues spans a k-dimensional Krylov space,
  # so every n_lanczos >= k must give the SAME number at the same seed: the
  # recurrence breaks down at step k and the quadrature there is exact. Two
  # things have to hold for that. The breakdown has to end the recurrence at
  # m_eff rather than pad alpha/beta out to m with zeros, and the basis has to
  # stay orthogonal, or ghost Ritz values appear at larger m and move the
  # answer. This is what the estimator itself can say about either; the
  # per-probe quadrature being exact does not make the probe AVERAGE exact, so
  # a fixed-tolerance comparison against the true log-determinant measures the
  # Hutchinson variance instead.
  n <- 30L
  d <- rep(c(1.5, 4.0), length.out = n)
  cs <- .csc_lower(diag(d, n))
  got <- vapply(c(3L, 6L, 15L, n), function(m)
    cpp_stochastic_log_determinant(cs$x, cs$i, cs$p, n,
                                   n_probes = 20L, n_lanczos = m, seed = 5L),
    numeric(1))
  expect_equal(got[-1], got[-length(got)], tolerance = 1e-12)
  # and it is the right number, within the estimator's own noise
  expect_equal(got[1], sum(log(d)), tolerance = 0.05 * abs(sum(log(d))))
})

test_that("a malformed CSC triple is an R error rather than an out-of-bounds write", {
  n <- 6L
  cs <- .csc_lower(diag(2, n))
  expect_error(cpp_stochastic_log_determinant(cs$x, cs$i, cs$p, n + 3L),
               "Q_p")
  bad_i <- cs$i; bad_i[1] <- n            # 0-based, so n is one past the end
  expect_error(cpp_stochastic_log_determinant(cs$x, bad_i, cs$p, n), "Q_i")
  bad_i2 <- cs$i; bad_i2[2] <- -1L
  expect_error(cpp_stochastic_log_determinant(cs$x, bad_i2, cs$p, n), "Q_i")
  expect_error(cpp_stochastic_log_determinant(cs$x[-1], cs$i, cs$p, n), "Q_")
})
