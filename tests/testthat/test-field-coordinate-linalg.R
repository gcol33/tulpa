test_that("coordinate matrix products have exact rectangular pullbacks", {
  load_cpp_fixture('coordinate_linalg')
  a <- matrix(sin(seq_len(12)), 3, 4)
  b <- matrix(cos(seq_len(8)), 4, 2)
  w <- matrix(seq(-.5, .5, length.out = 6), 3, 2)
  for (transpose in c(FALSE, TRUE)) {
    left <- if (transpose) t(a) else a
    out <- cpp_coordinate_product(left, b, w, transpose)
    expect_equal(out$value, a %*% b, tolerance = 1e-14)
    da <- w %*% t(b)
    expect_equal(out$left_gradient, if (transpose) t(da) else da, tolerance = 1e-14)
    expect_equal(out$right_gradient, crossprod(a, w), tolerance = 1e-14)
  }
})

test_that("coordinate Cholesky pullbacks agree with symmetric dense perturbations", {
  load_cpp_fixture('coordinate_linalg')
  for (n in c(1L, 3L, 7L)) {
    x <- matrix(sin(seq_len(n * n)), n)
    h <- crossprod(x) + diag(n)
    w <- matrix(cos(seq_len(n * n)), n)
    out <- cpp_coordinate_cholesky(h, w)
    expect_true(out$valid)
    expect_equal(out$value, t(chol(h)), tolerance = 1e-13)
    expect_equal(out$gradient, t(out$gradient), tolerance = 1e-13)
    for (j in seq_len(4L)) {
      d <- matrix(sin(seq_len(n * n) + j), n)
      d <- (d + t(d)) / 2
      step <- 1e-5
      numerical <- sum((t(chol(h + step * d)) - t(chol(h - step * d))) * w) / (2 * step)
      expect_equal(sum(out$gradient * d), numerical, tolerance = 1e-8)
    }
  }
  expect_false(cpp_coordinate_cholesky(matrix(c(1, 2, 2, 1), 2), matrix(1, 2, 2))$valid)
})
