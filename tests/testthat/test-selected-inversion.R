# test-selected-inversion.R
# Tests for Takahashi selected inversion (diagonal of Q^{-1})

test_that("selected inversion matches dense solve for small matrix", {
  # Build a small positive-definite sparse matrix (tridiagonal + diagonal)
  n <- 20
  Q <- Matrix::bandSparse(n, k = c(0, 1), diag = list(
    rep(4, n),
    rep(-1, n - 1)
  ), symmetric = TRUE)

  # Ground truth: diagonal of Q^{-1} via dense inversion
  Q_dense <- as.matrix(Q)
  Q_inv_diag_true <- diag(solve(Q_dense))

  # Selected inversion via CHOLMOD
  Q_csc <- as(Q, "CsparseMatrix")

  # Extract lower triangle for CHOLMOD (stype = -1)
  Q_lower <- Matrix::tril(Q_csc)

  diag_inv <- cpp_selected_inversion_diagonal(
    Q_x = Q_lower@x,
    Q_i = Q_lower@i,
    Q_p = Q_lower@p,
    n = n
  )

  expect_equal(length(diag_inv), n)
  expect_true(all(is.finite(diag_inv)))
  expect_true(all(diag_inv > 0))  # variances must be positive

  # Should match dense solve to high precision
  expect_equal(diag_inv, Q_inv_diag_true, tolerance = 1e-8)
})

test_that("selected inversion works for ICAR-like precision", {
  # Build ICAR precision on a 5x5 grid
  source(test_path("test-sparse-cholesky.R"), local = TRUE)
  adj <- make_grid_adjacency(5, 5)
  n <- 25

  # Build Q = tau * (D - W) + epsilon * I (regularized for pos-def)
  tau <- 2.0
  Q_dense <- matrix(0, n, n)
  for (i in seq_len(n)) {
    Q_dense[i, i] <- tau * adj$n_neighbors[i] + 0.01  # regularize
    start <- adj$adj_row_ptr[i] + 1L
    end <- adj$adj_row_ptr[i + 1]
    if (end >= start) {
      for (idx in start:end) {
        j <- adj$adj_col_idx[idx] + 1L
        Q_dense[i, j] <- -tau
      }
    }
  }

  Q_inv_diag_true <- diag(solve(Q_dense))

  Q_sparse <- as(Matrix::Matrix(Q_dense, sparse = TRUE), "CsparseMatrix")
  Q_lower <- Matrix::tril(Q_sparse)

  diag_inv <- cpp_selected_inversion_diagonal(
    Q_x = Q_lower@x, Q_i = Q_lower@i, Q_p = Q_lower@p, n = n
  )

  expect_true(all(is.finite(diag_inv)))
  expect_true(all(diag_inv > 0))
  expect_equal(diag_inv, Q_inv_diag_true, tolerance = 1e-6)
})

test_that("selected inversion rejects a CSC triple that does not match n", {
  n <- 6
  Q <- Matrix::Diagonal(n, x = 2)
  Q_lower <- Matrix::tril(as(Q, "CsparseMatrix"))
  x <- Q_lower@x
  i <- Q_lower@i
  p <- Q_lower@p

  # n unrelated to the vectors: Q_p is read to n + 1 and Q_i / Q_x to nnz, so
  # every mismatch below is an out-of-bounds read without the check.
  expect_error(
    cpp_selected_inversion_diagonal(numeric(0), integer(0), integer(0), 1000L),
    "Q_p"
  )
  expect_error(cpp_selected_inversion_diagonal(x, i, p, 0L), "positive")
  expect_error(cpp_selected_inversion_diagonal(x, i[-1], p, n), "Q_i")
  expect_error(cpp_selected_inversion_diagonal(x, i, p[-1], n), "Q_p")
  expect_error(
    cpp_selected_inversion_diagonal(x, replace(i, 1L, n), p, n),
    "outside"
  )
})
