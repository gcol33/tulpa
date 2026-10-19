
test_that(".spd_inverse() errors on an indefinite matrix, dense or sparse", {
  A <- matrix(c(1, 2, 2, 1), 2)
  expect_error(tulpa:::.spd_inverse(A))
  expect_error(suppressWarnings(
    tulpa:::.spd_inverse(Matrix::Matrix(A, sparse = TRUE))))
})

skip_on_cran_budget()
test_that(".spd_inverse() inverts an SPD matrix, dense or sparse, keeping names", {
  skip_on_cran_budget()
  set.seed(1)
  B <- matrix(rnorm(25), 5)
  A <- crossprod(B) + diag(5)
  dimnames(A) <- list(letters[1:5], letters[1:5])
  V <- tulpa:::.spd_inverse(A)
  expect_equal(V %*% A, diag(5), tolerance = 1e-10, ignore_attr = TRUE)
  expect_identical(dimnames(V), dimnames(A))
  expect_true(isSymmetric(V))

  As <- Matrix::Matrix(A, sparse = TRUE)
  expect_equal(tulpa:::.spd_inverse(As), V, tolerance = 1e-10)
})

test_that(".spd_inverse() symmetrizes a Hessian that is symmetric only to rounding", {
  skip_on_cran_budget()
  A <- matrix(c(2, 1 + 1e-12, 1, 2), 2)
  expect_true(isSymmetric(tulpa:::.spd_inverse(A)))
})
