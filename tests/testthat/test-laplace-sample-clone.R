# cpp_laplace_sample must not mutate the precision it is handed
# (gcol33/tulpa#451).
#
# Rcpp binds a REALSXP argument without duplicating it, so ridging `H` in place
# writes into the R matrix the caller still holds. Two consequences: the
# caller's Hessian silently gains the ridge, and a second call on the same
# matrix samples from a precision carrying it twice.

test_that("the caller's precision matrix is unchanged by sampling", {
  set.seed(3L)
  n <- 4L
  A <- matrix(rnorm(n * n), n, n)
  H <- crossprod(A) + diag(n)
  H_before <- H
  mode <- rnorm(n)

  s <- tulpa:::cpp_laplace_sample(mode, H, 5L)
  expect_equal(dim(s), c(5L, n))
  expect_true(all(is.finite(s)))
  expect_identical(H, H_before)
})

test_that("repeated calls sample from the same precision", {
  set.seed(4L)
  n <- 3L
  A <- matrix(rnorm(n * n), n, n)
  H <- crossprod(A) + diag(n)
  mode <- rep(0, n)

  set.seed(99L); s1 <- tulpa:::cpp_laplace_sample(mode, H, 400L)
  set.seed(99L); s2 <- tulpa:::cpp_laplace_sample(mode, H, 400L)
  # An accumulating in-place ridge makes the second draw set a different one
  # even at the same seed.
  expect_equal(s1, s2, tolerance = 0)
})
