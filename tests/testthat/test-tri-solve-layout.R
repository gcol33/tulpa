# The shared small-dense Cholesky core reads the layout its call site declares
# (gcol33/tulpa#285).
#
# linalg_fast.h carried two triangular-solve pairs on opposite storage
# conventions, named so that neither said which. A column-major lower factor is
# the same bytes as a row-major upper one, so a factor handed to the wrong pair
# solves against the transpose: finite, plausible, and undetectable downstream.
# gcol33/tulpa#283 was that mistake on a cuSOLVER factor, and it corrupted every
# NNGP fit with 51+ locations.
#
# The layout is now a required template argument. These tests state the contract
# once -- the routine reads the lower triangle of the matrix its declared layout
# spells out of the buffer -- and check it on matched and mismatched buffers, so
# the two conventions can never quietly become the same thing.

ROW_MAJOR <- 0L
COL_MAJOR <- 1L

# Serialize a matrix into a flat buffer under each convention. R matrices are
# already column-major, so as_col_major is the identity on storage.
as_row_major <- function(M) as.numeric(t(M))
as_col_major <- function(M) as.numeric(M)

# The matrix a given layout spells out of a flat buffer -- the reference for
# what the C++ side is looking at.
read_as <- function(buf, n, layout) {
  if (layout == ROW_MAJOR) matrix(buf, n, n, byrow = TRUE) else matrix(buf, n, n)
}

# Only the lower triangle is read by either solve.
tri_lower <- function(M) {
  M[upper.tri(M)] <- 0
  M
}

# A lower factor with a distinctly non-symmetric off-diagonal, so a transposed
# read gives a different answer rather than the same one.
test_L <- function() {
  matrix(c(2.0, 0.0, 0.0,
           -1.5, 3.0, 0.0,
           0.5, -2.5, 1.25),
         nrow = 3, byrow = TRUE)
}

# The buffer a batched cuSOLVER call actually returns: the factor written into
# the column-major lower triangle, with the opposite triangle still holding the
# input covariance it was never asked to touch. This is the #283 buffer.
cusolver_style_buffer <- function(C) {
  L <- t(chol(C))
  X <- L
  X[upper.tri(X)] <- C[upper.tri(C)]
  as_col_major(X)
}

test_that("tri_solve_lower solves against the matrix its layout spells out", {
  L <- test_L()
  b <- c(1.0, -2.0, 0.75)

  cases <- list(
    list(buf = as_row_major(L), layout = ROW_MAJOR, what = "row-major buffer, row-major read"),
    list(buf = as_col_major(L), layout = COL_MAJOR, what = "col-major buffer, col-major read"),
    list(buf = as_row_major(L), layout = COL_MAJOR, what = "row-major buffer, col-major read"),
    list(buf = as_col_major(L), layout = ROW_MAJOR, what = "col-major buffer, row-major read")
  )
  for (cs in cases) {
    seen <- tri_lower(read_as(cs$buf, 3L, cs$layout))
    expect_equal(cpp_test_tri_solve(cs$buf, 3L, b, cs$layout, FALSE),
                 as.numeric(forwardsolve(seen, b)),
                 tolerance = 1e-12, info = cs$what)
  }
})

test_that("tri_solve_lower_transpose solves L' x = y for the same matrix", {
  L <- test_L()
  y <- c(0.4, 1.1, -0.6)

  for (layout in c(ROW_MAJOR, COL_MAJOR)) {
    buf <- if (layout == ROW_MAJOR) as_row_major(L) else as_col_major(L)
    seen <- tri_lower(read_as(buf, 3L, layout))
    expect_equal(cpp_test_tri_solve(buf, 3L, y, layout, TRUE),
                 as.numeric(backsolve(t(seen), y)), tolerance = 1e-12)
  }
})

test_that("a densely-stored factor read under the wrong layout loses its off-diagonal", {
  # A lower factor with a zeroed opposite triangle transposes into an UPPER
  # triangular matrix, whose lower triangle is the diagonal alone. So the wrong
  # read degenerates to a diagonal solve: every off-diagonal entry of the factor
  # is silently dropped, and the answer is finite and ordinary-looking.
  L <- test_L()
  b <- c(1.0, -2.0, 0.75)

  wrong <- cpp_test_tri_solve(as_row_major(L), 3L, b, COL_MAJOR, FALSE)
  right <- cpp_test_tri_solve(as_row_major(L), 3L, b, ROW_MAJOR, FALSE)

  expect_equal(wrong, b / diag(L), tolerance = 1e-12)
  expect_equal(right, as.numeric(forwardsolve(L, b)), tolerance = 1e-12)
  expect_true(all(is.finite(wrong)))
  expect_false(isTRUE(all.equal(wrong, right)))
})

test_that("a cuSOLVER-style buffer read row-major returns plausible nonsense", {
  # The #283 buffer: the opposite triangle still holds input covariances, so the
  # wrong read picks up real nonzero numbers rather than zeros. Nothing about
  # the result announces itself as wrong.
  C <- matrix(c(1.0, 0.6, 0.3,
                0.6, 1.0, 0.5,
                0.3, 0.5, 1.0),
              nrow = 3, byrow = TRUE)
  buf <- cusolver_style_buffer(C)
  b <- c(0.9, -0.4, 1.2)

  right <- cpp_test_tri_solve(buf, 3L, b, COL_MAJOR, FALSE)
  wrong <- cpp_test_tri_solve(buf, 3L, b, ROW_MAJOR, FALSE)

  expect_equal(right, as.numeric(forwardsolve(t(chol(C)), b)), tolerance = 1e-12)
  expect_equal(wrong,
               as.numeric(forwardsolve(tri_lower(read_as(buf, 3L, ROW_MAJOR)), b)),
               tolerance = 1e-12)
  expect_true(all(is.finite(wrong)))
  expect_false(isTRUE(all.equal(wrong, right)))
})

test_that("chol_factor_lower writes the layout it was asked for", {
  A <- matrix(c(4.0, -1.0, 0.5,
                -1.0, 3.0, -1.5,
                0.5, -1.5, 2.0),
              nrow = 3, byrow = TRUE)
  ref <- t(chol(A))

  expect_equal(cpp_test_chol_factor(as_row_major(A), 3L, ROW_MAJOR),
               as_row_major(ref), tolerance = 1e-10)
  expect_equal(cpp_test_chol_factor(as_col_major(A), 3L, COL_MAJOR),
               as_col_major(ref), tolerance = 1e-10)
})

test_that("NNGP kriging moments read the layout of the factor they are given", {
  # The #283 call site: cond_var = sigma2 - c' C^-1 c off an already-built
  # factor. Under the wrong layout it stayed finite and landed on the variance
  # floor, which reads as near-determinism rather than as a broken factor.
  C <- matrix(c(1.0, 0.6, 0.3,
                0.6, 1.0, 0.5,
                0.3, 0.5, 1.0),
              nrow = 3, byrow = TRUE)
  L <- t(chol(C))
  c_vec <- c(0.7, 0.4, 0.2)
  w_nb <- c(0.5, -0.3, 1.1)
  sigma2 <- 1.0
  var_floor <- 1e-10

  alpha_ref <- solve(C, c_vec)
  mean_ref <- sum(alpha_ref * w_nb)
  var_ref <- sigma2 - sum(c_vec * alpha_ref)

  for (layout in c(ROW_MAJOR, COL_MAJOR)) {
    buf <- if (layout == ROW_MAJOR) as_row_major(L) else as_col_major(L)
    got <- cpp_test_nngp_moments(buf, 3L, c_vec, w_nb, sigma2, var_floor, layout)
    expect_equal(got$cond_mean, mean_ref, tolerance = 1e-10)
    expect_equal(got$cond_var, var_ref, tolerance = 1e-10)
    expect_equal(got$alpha, as.numeric(alpha_ref), tolerance = 1e-10)
  }

  # Wrong layout: the routine solves M M' alpha = c against the diagonal-only
  # matrix the transposed read leaves it, so the moments move while staying
  # finite -- the signature of the #283 failure.
  wrong <- cpp_test_nngp_moments(as_row_major(L), 3L, c_vec, w_nb, sigma2,
                                 var_floor, COL_MAJOR)
  seen <- tri_lower(read_as(as_row_major(L), 3L, COL_MAJOR))
  alpha_wrong <- solve(seen %*% t(seen), c_vec)
  expect_equal(wrong$alpha, as.numeric(alpha_wrong), tolerance = 1e-10)
  expect_true(is.finite(wrong$cond_var))
  expect_false(isTRUE(all.equal(wrong$cond_var, var_ref)))
})
