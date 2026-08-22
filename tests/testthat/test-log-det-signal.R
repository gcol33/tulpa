# A failed factorization is reported, on both backends (gcol33/tulpa#436).
#
# dense_cholesky_log_det_raw deliberately applies no in-pivot clamp: every call
# site adds LAPLACE_UNIFORM_RIDGE upstream, so a non-positive pivot means the
# Hessian at the returned point is not PD -- a point the solve stopped at
# without reaching a mode. An indefinite H leaves a NaN there and an exactly
# singular one a -Inf.
#
# "Propagate" is only true if a caller checks. Carried into a Laplace
# log-marginal, a NaN turns one outer-grid cell's weight into NaN, and a -Inf
# log-determinant is a +Inf log-marginal, which does not merely lose the cell
# but makes it take the whole grid's weight. Both the dispatch and the dense
# core now return a bool, and the three Newton call sites read it.
#
# A 1e-10 uniform ridge is a floor on the pivots of a PSD H, not a guarantee: it
# does not make an indefinite H positive definite, which is why the ridged entry
# refuses the same matrices.

.PD_H    <- diag(c(2, 3, 4))
.IND_H   <- diag(c(2, -1, 4))              # one negative eigenvalue
.SING_H  <- matrix(c(1, 1, 0,
                     1, 1, 0,
                     0, 0, 2), 3, 3)        # exactly singular, PSD

test_that("a positive definite H factors and reports its log-determinant", {
  for (sparse in c(FALSE, TRUE)) {
    r <- cpp_test_log_det_signal(.PD_H, prefer_sparse = sparse)
    expect_true(r$ok, info = paste("sparse =", sparse))
    expect_equal(r$log_det, log(prod(diag(.PD_H))), tolerance = 1e-10,
                 info = paste("sparse =", sparse))
    expect_true(r$ok_dense)
    expect_equal(r$log_det_dense, log(prod(diag(.PD_H))), tolerance = 1e-10)
  }
})

test_that("an indefinite H is refused rather than returned as a number", {
  expect_lt(min(eigen(.IND_H, symmetric = TRUE, only.values = TRUE)$values), 0)
  for (sparse in c(FALSE, TRUE)) {
    r <- cpp_test_log_det_signal(.IND_H, prefer_sparse = sparse)
    expect_false(r$ok, info = paste("sparse =", sparse))
    expect_false(is.finite(r$log_det), info = paste("sparse =", sparse))
    expect_false(r$ok_dense)
    expect_false(is.finite(r$log_det_dense))
  }
})

test_that("an exactly singular H is refused too", {
  expect_lt(min(eigen(.SING_H, symmetric = TRUE, only.values = TRUE)$values),
            1e-12)
  for (sparse in c(FALSE, TRUE)) {
    r <- cpp_test_log_det_signal(.SING_H, prefer_sparse = sparse)
    expect_false(r$ok, info = paste("sparse =", sparse))
    expect_false(is.finite(r$log_det), info = paste("sparse =", sparse))
  }
})

test_that("the base ridge does not rescue an indefinite H", {
  # dispatch_factor_log_det adds LAPLACE_UNIFORM_RIDGE before factoring. On the
  # PD matrix that shifts the log-determinant by a hair and still succeeds; on
  # the indefinite one it changes nothing about the verdict.
  ok_pd <- cpp_test_log_det_signal(.PD_H, add_ridge = TRUE)
  expect_true(ok_pd$ok)
  expect_equal(ok_pd$log_det, log(prod(diag(.PD_H) + 1e-10)), tolerance = 1e-9)

  bad <- cpp_test_log_det_signal(.IND_H, add_ridge = TRUE)
  expect_false(bad$ok)
  expect_false(is.finite(bad$log_det))
})

test_that("the two backends agree on which matrices they refuse", {
  # The joint Newton loop and the plain one pick different arms of the same
  # dispatch, so a verdict that depended on the arm would put two paths on the
  # same model into disagreement.
  for (H in list(.PD_H, .IND_H, .SING_H)) {
    a <- cpp_test_log_det_signal(H, prefer_sparse = FALSE)
    b <- cpp_test_log_det_signal(H, prefer_sparse = TRUE)
    expect_identical(a$ok, b$ok)
    if (a$ok) expect_equal(a$log_det, b$log_det, tolerance = 1e-9)
  }
})
